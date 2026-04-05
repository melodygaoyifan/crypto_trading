#!/usr/bin/env python3
"""
Enhanced MarketRegimeNavigator - Vol-of-Vol Awareness
=====================================================
SOTA v3.0 Component for Advanced Regime Detection

Enhancements over v2:
1. Correlation & Vol-of-Vol Tensor
2. Cost-aware Hysteresis (covers 26bps + slippage)
3. SOL Network Stress Integration
4. Beta-decoupling detection with persistence
5. Regime transition probability forecasting

Hardware: RTX 5090 (32GB VRAM)
Frequency: 4H/1D strategy decisions

Author: Senior Quant Team
Version: SOTA v3.0
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Conditional imports
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False

try:
    from scipy import stats
    from scipy.special import softmax
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EnhancedRegimeNavigator')


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class MarketRegime(IntEnum):
    """6-state market regime classification."""
    STRONG_BULL = 0
    MODERATE_BULL = 1
    NEUTRAL = 2
    VOLATILE_CHOP = 3
    MODERATE_BEAR = 4
    STRONG_BEAR = 5


class LiquidityRegime(IntEnum):
    """5-state liquidity regime for SOL."""
    DEEP_LIQUID = 0
    MODERATE_LIQUID = 1
    THIN_LIQUID = 2
    STRESSED = 3
    CRISIS = 4


class VolatilityRegime(IntEnum):
    """Volatility-of-volatility regime."""
    LOW_VOL = 0      # Stable, low vol-of-vol
    NORMAL_VOL = 1   # Normal conditions
    HIGH_VOL = 2     # Elevated but stable
    EXPLOSIVE = 3    # Vol-of-vol spike, unstable


@dataclass
class VolOfVolTensor:
    """
    Correlation and Volatility-of-Volatility Tensor
    
    Structure: [assets × timeframes × metrics]
    - Assets: BTC, ETH, SOL
    - Timeframes: 1H, 4H, 1D
    - Metrics: volatility, vol_of_vol, correlation
    """
    timestamp: float
    
    # Volatility matrix [3 assets × 3 timeframes]
    volatility: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    
    # Vol-of-vol matrix [3 assets × 3 timeframes]
    vol_of_vol: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    
    # Cross-asset correlation tensor [3 assets × 3 assets × 3 timeframes]
    correlation: np.ndarray = field(default_factory=lambda: np.zeros((3, 3, 3)))
    
    # Correlation stability (rolling std of correlation)
    correlation_stability: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    
    # Vol-of-vol regime
    vov_regime: VolatilityRegime = VolatilityRegime.NORMAL_VOL
    
    # Decoupling indicators [BTC-ETH, BTC-SOL, ETH-SOL]
    decoupling_scores: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def get_asset_idx(self, asset: str) -> int:
        """Get index for asset."""
        return {'BTC': 0, 'ETH': 1, 'SOL': 2}.get(asset, 0)
    
    def get_timeframe_idx(self, tf: str) -> int:
        """Get index for timeframe."""
        return {'1H': 0, '4H': 1, '1D': 2}.get(tf, 1)


@dataclass
class RegimeTensorV3:
    """
    Enhanced Regime Tensor for SOTA v3.0
    
    Combines:
    - Market regime (6 states)
    - Liquidity regime per asset (5 states)
    - Vol-of-vol tensor
    - Hysteresis state
    - Cost-adjusted transition probabilities
    """
    timestamp: float
    
    # Core regime
    market_regime: MarketRegime
    regime_confidence: float  # [0, 1]
    regime_probabilities: np.ndarray  # [6] probabilities
    
    # Liquidity per asset
    liquidity_regimes: Dict[str, LiquidityRegime] = field(default_factory=dict)
    
    # Vol-of-vol tensor
    vov_tensor: Optional[VolOfVolTensor] = None
    
    # Hysteresis state
    hysteresis_active: bool = False
    hysteresis_counter: int = 0
    hysteresis_threshold: int = 3  # 3 × 4H = 12H persistence
    
    # Cost-aware metrics
    expected_regime_duration: float = 24.0  # hours
    regime_switch_cost_bps: float = 0.0  # Est. cost to switch
    switch_recommended: bool = True
    
    # Beta metrics
    beta_btc_eth: float = 1.0
    beta_btc_sol: float = 1.5
    beta_decoupling: bool = False
    
    # Correlation breakdown
    correlation_btc_eth: float = 0.8
    correlation_btc_sol: float = 0.6
    correlation_eth_sol: float = 0.7
    
    # SOL network stress
    sol_network_stress: str = 'LOW'
    sol_cu_utilization: float = 0.0
    
    def to_drl_features(self) -> np.ndarray:
        """Convert to DRL feature vector."""
        features = [
            # Regime probabilities (6)
            *self.regime_probabilities,
            # Confidence
            self.regime_confidence,
            # Liquidity (3)
            self.liquidity_regimes.get('BTC', LiquidityRegime.MODERATE_LIQUID).value / 4,
            self.liquidity_regimes.get('ETH', LiquidityRegime.MODERATE_LIQUID).value / 4,
            self.liquidity_regimes.get('SOL', LiquidityRegime.MODERATE_LIQUID).value / 4,
            # Vol-of-vol regime
            self.vov_tensor.vov_regime.value / 3 if self.vov_tensor else 0.5,
            # Betas
            self.beta_btc_eth,
            self.beta_btc_sol,
            # Decoupling flag
            1.0 if self.beta_decoupling else 0.0,
            # Correlations
            self.correlation_btc_eth,
            self.correlation_btc_sol,
            self.correlation_eth_sol,
            # Hysteresis
            1.0 if self.hysteresis_active else 0.0,
            self.hysteresis_counter / self.hysteresis_threshold,
            # Cost metrics
            self.regime_switch_cost_bps / 100,  # Normalize
            1.0 if self.switch_recommended else 0.0,
        ]
        return np.array(features, dtype=np.float32)


@dataclass
class HysteresisConfig:
    """Configuration for cost-aware hysteresis."""
    # Base transaction cost (Kraken taker)
    base_cost_bps: float = 26.0
    
    # Estimated slippage by asset
    slippage_bps: Dict[str, float] = field(default_factory=lambda: {
        'BTC': 3.0,
        'ETH': 5.0,
        'SOL': 10.0,
    })
    
    # Minimum probability margin for regime switch
    min_probability_margin: float = 0.20
    
    # Persistence requirement (number of periods)
    persistence_periods: int = 3
    
    # Minimum expected holding period to justify switch (hours)
    min_holding_period: float = 8.0
    
    # Emergency override sigma (sentiment)
    emergency_sigma: float = 5.0
    
    def get_total_switch_cost(self, asset_weights: Dict[str, float]) -> float:
        """Calculate total cost to switch positions."""
        # Round-trip cost = 2 × (base + slippage) × weight
        total = 0.0
        for asset, weight in asset_weights.items():
            slip = self.slippage_bps.get(asset, 5.0)
            total += 2 * (self.base_cost_bps + slip) * weight
        return total


# =============================================================================
# VOL-OF-VOL CALCULATOR
# =============================================================================

class VolOfVolCalculator:
    """
    Calculate Volatility-of-Volatility and Correlation Tensor
    
    Uses GPU acceleration for parallel computation across
    multiple assets and timeframes.
    """
    
    ASSETS = ['BTC', 'ETH', 'SOL']
    TIMEFRAMES = ['1H', '4H', '1D']
    TIMEFRAME_PERIODS = {'1H': 1, '4H': 4, '1D': 24}
    
    def __init__(
        self,
        vol_window: int = 24,      # Base volatility window
        vov_window: int = 12,      # Vol-of-vol window
        corr_window: int = 48,     # Correlation window
        use_gpu: bool = True,
    ):
        self.vol_window = vol_window
        self.vov_window = vov_window
        self.corr_window = corr_window
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        # Return history per asset [asset][returns]
        self.returns_history: Dict[str, deque] = {
            asset: deque(maxlen=500) for asset in self.ASSETS
        }
        
        # Volatility history for vol-of-vol
        self.vol_history: Dict[str, Dict[str, deque]] = {
            asset: {tf: deque(maxlen=100) for tf in self.TIMEFRAMES}
            for asset in self.ASSETS
        }
        
        # Correlation history for stability
        self.corr_history: Dict[Tuple[str, str], deque] = {
            (a1, a2): deque(maxlen=50)
            for i, a1 in enumerate(self.ASSETS)
            for a2 in self.ASSETS[i+1:]
        }
        
        logger.info(f"Vol-of-Vol Calculator initialized | GPU: {self.use_gpu}")
    
    def update_returns(self, asset: str, ret: float):
        """Update return history for an asset."""
        if asset in self.returns_history:
            self.returns_history[asset].append(ret)
    
    def _calculate_volatility_gpu(
        self,
        returns: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """GPU-accelerated rolling volatility."""
        if len(returns) < window:
            return np.array([np.std(returns) if len(returns) > 0 else 0.0])
        
        returns_gpu = self.xp.asarray(returns)
        
        # Rolling std using convolution
        rolling_mean = self.xp.convolve(
            returns_gpu, self.xp.ones(window) / window, mode='valid'
        )
        
        # Calculate rolling variance
        returns_centered = returns_gpu[window-1:] - rolling_mean[:len(returns_gpu)-window+1]
        
        # Simple rolling std approximation
        rolling_std = self.xp.sqrt(
            self.xp.convolve(
                returns_gpu ** 2, self.xp.ones(window) / window, mode='valid'
            ) - rolling_mean ** 2
        )
        
        if self.use_gpu:
            rolling_std = cp.asnumpy(rolling_std)
        
        return rolling_std
    
    def _calculate_correlation_gpu(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        window: int,
    ) -> float:
        """Calculate rolling correlation."""
        min_len = min(len(returns1), len(returns2))
        if min_len < window:
            if min_len < 2:
                return 0.0
            return float(np.corrcoef(returns1[-min_len:], returns2[-min_len:])[0, 1])
        
        r1 = self.xp.asarray(returns1[-window:])
        r2 = self.xp.asarray(returns2[-window:])
        
        # Correlation calculation
        r1_centered = r1 - self.xp.mean(r1)
        r2_centered = r2 - self.xp.mean(r2)
        
        cov = self.xp.mean(r1_centered * r2_centered)
        std1 = self.xp.std(r1)
        std2 = self.xp.std(r2)
        
        if std1 * std2 == 0:
            return 0.0
        
        corr = cov / (std1 * std2)
        
        if self.use_gpu:
            corr = float(cp.asnumpy(corr))
        
        return float(corr)
    
    def calculate_tensor(self) -> VolOfVolTensor:
        """
        Calculate full Vol-of-Vol tensor.
        
        Returns:
            VolOfVolTensor with all metrics
        """
        tensor = VolOfVolTensor(timestamp=time.time())
        
        # Calculate per-asset, per-timeframe metrics
        for i, asset in enumerate(self.ASSETS):
            returns = np.array(self.returns_history[asset])
            if len(returns) < 5:
                continue
            
            for j, tf in enumerate(self.TIMEFRAMES):
                periods = self.TIMEFRAME_PERIODS[tf]
                
                # Aggregate returns for timeframe
                if len(returns) >= periods:
                    tf_returns = np.array([
                        np.sum(returns[k:k+periods])
                        for k in range(0, len(returns) - periods + 1, periods)
                    ])
                else:
                    tf_returns = returns
                
                # Volatility
                vol = np.std(tf_returns) * np.sqrt(24 / periods) if len(tf_returns) > 1 else 0.0
                tensor.volatility[i, j] = vol
                
                # Update vol history
                self.vol_history[asset][tf].append(vol)
                
                # Vol-of-vol
                vol_hist = np.array(self.vol_history[asset][tf])
                if len(vol_hist) >= self.vov_window:
                    vov = np.std(vol_hist[-self.vov_window:])
                    tensor.vol_of_vol[i, j] = vov
        
        # Calculate cross-asset correlations
        for i, a1 in enumerate(self.ASSETS):
            for j, a2 in enumerate(self.ASSETS):
                if i == j:
                    tensor.correlation[i, j, :] = 1.0
                    continue
                
                r1 = np.array(self.returns_history[a1])
                r2 = np.array(self.returns_history[a2])
                
                for k, tf in enumerate(self.TIMEFRAMES):
                    periods = self.TIMEFRAME_PERIODS[tf]
                    window = self.corr_window // periods
                    corr = self._calculate_correlation_gpu(r1, r2, window)
                    tensor.correlation[i, j, k] = corr
                
                # Update correlation history
                if i < j:
                    avg_corr = np.mean(tensor.correlation[i, j, :])
                    self.corr_history[(a1, a2)].append(avg_corr)
                    
                    # Correlation stability
                    corr_hist = np.array(self.corr_history[(a1, a2)])
                    if len(corr_hist) > 5:
                        tensor.correlation_stability[i, j] = np.std(corr_hist[-20:])
                        tensor.correlation_stability[j, i] = tensor.correlation_stability[i, j]
        
        # Calculate decoupling scores
        # BTC-ETH decoupling
        btc_eth_corr = tensor.correlation[0, 1, 1]  # 4H timeframe
        btc_eth_baseline = 0.85
        tensor.decoupling_scores[0] = max(0, btc_eth_baseline - btc_eth_corr)
        
        # BTC-SOL decoupling
        btc_sol_corr = tensor.correlation[0, 2, 1]
        btc_sol_baseline = 0.70
        tensor.decoupling_scores[1] = max(0, btc_sol_baseline - btc_sol_corr)
        
        # ETH-SOL decoupling
        eth_sol_corr = tensor.correlation[1, 2, 1]
        eth_sol_baseline = 0.75
        tensor.decoupling_scores[2] = max(0, eth_sol_baseline - eth_sol_corr)
        
        # Determine vol-of-vol regime
        avg_vov = np.mean(tensor.vol_of_vol)
        if avg_vov < 0.005:
            tensor.vov_regime = VolatilityRegime.LOW_VOL
        elif avg_vov < 0.015:
            tensor.vov_regime = VolatilityRegime.NORMAL_VOL
        elif avg_vov < 0.03:
            tensor.vov_regime = VolatilityRegime.HIGH_VOL
        else:
            tensor.vov_regime = VolatilityRegime.EXPLOSIVE
        
        return tensor


# =============================================================================
# COST-AWARE HYSTERESIS FILTER
# =============================================================================

class CostAwareHysteresisFilter:
    """
    Hysteresis filter that accounts for trading costs.
    
    A regime switch must be:
    1. Statistically significant (probability margin > 20%)
    2. Persistent (3 × 4H periods)
    3. Cost-justified (expected profit > switch cost)
    
    Unless emergency override (5σ sentiment shock).
    """
    
    def __init__(self, config: HysteresisConfig):
        self.config = config
        
        # State
        self.current_regime = MarketRegime.NEUTRAL
        self.pending_regime: Optional[MarketRegime] = None
        self.persistence_counter = 0
        
        # History
        self.regime_history: deque = deque(maxlen=100)
        self.switch_history: List[Dict] = []
        
        logger.info(f"Hysteresis Filter initialized | "
                   f"Margin: {config.min_probability_margin:.0%} | "
                   f"Persistence: {config.persistence_periods}")
    
    def _estimate_switch_cost(
        self,
        current: MarketRegime,
        proposed: MarketRegime,
        positions: Dict[str, float],
    ) -> float:
        """
        Estimate cost to switch from current to proposed regime.
        
        Considers:
        - Current positions that would be unwound
        - New positions that would be opened
        - Slippage based on regime (higher in volatile regimes)
        """
        # Base cost
        base_cost = self.config.get_total_switch_cost(
            {a: abs(p) for a, p in positions.items()}
        )
        
        # Regime-specific multiplier
        volatility_multiplier = {
            MarketRegime.STRONG_BULL: 1.0,
            MarketRegime.MODERATE_BULL: 1.0,
            MarketRegime.NEUTRAL: 1.2,
            MarketRegime.VOLATILE_CHOP: 2.0,
            MarketRegime.MODERATE_BEAR: 1.5,
            MarketRegime.STRONG_BEAR: 2.5,
        }
        
        current_mult = volatility_multiplier.get(current, 1.0)
        proposed_mult = volatility_multiplier.get(proposed, 1.0)
        
        return base_cost * (current_mult + proposed_mult) / 2
    
    def _estimate_expected_profit(
        self,
        proposed: MarketRegime,
        regime_probs: np.ndarray,
        expected_duration: float,
    ) -> float:
        """
        Estimate expected profit from regime switch.
        
        Based on:
        - Regime confidence
        - Expected holding period
        - Historical regime returns
        """
        # Historical daily returns by regime (calibrated)
        regime_daily_returns = {
            MarketRegime.STRONG_BULL: 0.02,    # +2% daily
            MarketRegime.MODERATE_BULL: 0.008,  # +0.8% daily
            MarketRegime.NEUTRAL: 0.0,          # 0%
            MarketRegime.VOLATILE_CHOP: -0.002, # -0.2% (due to chop)
            MarketRegime.MODERATE_BEAR: -0.008, # -0.8%
            MarketRegime.STRONG_BEAR: -0.02,    # -2%
        }
        
        confidence = regime_probs[proposed.value]
        daily_return = regime_daily_returns.get(proposed, 0.0)
        
        # Expected profit = confidence × return × days
        days = expected_duration / 24
        expected_profit_pct = confidence * daily_return * days * 100  # In bps × 100
        
        return expected_profit_pct * 100  # Convert to bps
    
    def should_switch(
        self,
        regime_probs: np.ndarray,
        sentiment_sigma: float = 0.0,
        positions: Optional[Dict[str, float]] = None,
        expected_duration: float = 24.0,
    ) -> Tuple[bool, MarketRegime, str]:
        """
        Determine if regime should switch.
        
        Returns:
            (should_switch, proposed_regime, reason)
        """
        positions = positions or {'BTC': 0.25, 'ETH': 0.25, 'SOL': 0.50}
        
        # Emergency override
        if abs(sentiment_sigma) >= self.config.emergency_sigma:
            emergency_regime = (MarketRegime.VOLATILE_CHOP 
                              if sentiment_sigma > 0 
                              else MarketRegime.STRONG_BEAR)
            return True, emergency_regime, f"Emergency override: {sentiment_sigma:.1f}σ sentiment"
        
        # Find most probable regime
        proposed_idx = np.argmax(regime_probs)
        proposed = MarketRegime(proposed_idx)
        proposed_prob = regime_probs[proposed_idx]
        current_prob = regime_probs[self.current_regime.value]
        
        # Check probability margin
        margin = proposed_prob - current_prob
        if margin < self.config.min_probability_margin:
            self.pending_regime = None
            self.persistence_counter = 0
            return False, self.current_regime, "Insufficient probability margin"
        
        # Check if this is a new pending regime
        if proposed != self.pending_regime:
            self.pending_regime = proposed
            self.persistence_counter = 1
            return False, self.current_regime, f"New pending regime: {proposed.name} (1/{self.config.persistence_periods})"
        
        # Increment persistence
        self.persistence_counter += 1
        
        if self.persistence_counter < self.config.persistence_periods:
            return False, self.current_regime, f"Building persistence: {self.persistence_counter}/{self.config.persistence_periods}"
        
        # Check cost justification
        switch_cost = self._estimate_switch_cost(self.current_regime, proposed, positions)
        expected_profit = self._estimate_expected_profit(proposed, regime_probs, expected_duration)
        
        if expected_profit < switch_cost:
            return False, self.current_regime, f"Cost not justified: profit {expected_profit:.1f}bps < cost {switch_cost:.1f}bps"
        
        # Switch approved
        old_regime = self.current_regime
        self.current_regime = proposed
        self.pending_regime = None
        self.persistence_counter = 0
        
        self.regime_history.append({
            'timestamp': time.time(),
            'from': old_regime,
            'to': proposed,
            'margin': margin,
            'cost': switch_cost,
            'expected_profit': expected_profit,
        })
        
        return True, proposed, f"Switch approved: {old_regime.name} -> {proposed.name} (profit {expected_profit:.1f}bps > cost {switch_cost:.1f}bps)"
    
    def get_hysteresis_state(self) -> Dict[str, Any]:
        """Get current hysteresis state."""
        return {
            'current_regime': self.current_regime,
            'pending_regime': self.pending_regime,
            'persistence_counter': self.persistence_counter,
            'persistence_required': self.config.persistence_periods,
            'recent_switches': len(self.regime_history),
        }


# =============================================================================
# SOL NETWORK STRESS INTEGRATOR
# =============================================================================

class SOLNetworkStressIntegrator:
    """
    Integrate Solana network metrics into regime detection.
    
    Adjusts SOL-specific regime based on:
    1. Compute Unit utilization
    2. Priority fees
    3. TPS
    4. Slot skip rate
    """
    
    def __init__(self):
        self.cu_history = deque(maxlen=100)
        self.fee_history = deque(maxlen=100)
        self.tps_history = deque(maxlen=100)
        
        self.current_stress_level = 'LOW'
        self.last_update = 0.0
    
    def update(
        self,
        cu_utilization: float,
        priority_fee: float,
        tps: float,
    ) -> str:
        """
        Update network metrics and return stress level.
        """
        self.last_update = time.time()
        
        self.cu_history.append(cu_utilization)
        self.fee_history.append(priority_fee)
        self.tps_history.append(tps)
        
        # Calculate composite stress score
        cu_score = min(cu_utilization, 1.0)
        fee_score = min(np.log10(priority_fee + 1) / 6, 1.0)
        tps_degradation = max(0, 1 - (tps / 4000))
        
        stress = 0.5 * cu_score + 0.3 * fee_score + 0.2 * tps_degradation
        
        # Map to level
        if stress < 0.25:
            self.current_stress_level = 'LOW'
        elif stress < 0.50:
            self.current_stress_level = 'MODERATE'
        elif stress < 0.75:
            self.current_stress_level = 'HIGH'
        else:
            self.current_stress_level = 'CRITICAL'
        
        return self.current_stress_level
    
    def adjust_sol_liquidity_regime(
        self,
        base_regime: LiquidityRegime,
    ) -> LiquidityRegime:
        """
        Adjust SOL liquidity regime based on network stress.
        """
        stress_adjustment = {
            'LOW': 0,
            'MODERATE': 1,
            'HIGH': 2,
            'CRITICAL': 3,
        }
        
        adjustment = stress_adjustment.get(self.current_stress_level, 0)
        adjusted_value = min(base_regime.value + adjustment, 4)
        
        return LiquidityRegime(adjusted_value)


# =============================================================================
# GMM REGIME CLASSIFIER
# =============================================================================

class EnhancedGMMClassifier:
    """
    Enhanced Gaussian Mixture Model for regime classification.
    
    Features:
    - 6-component GMM
    - GPU-accelerated feature extraction
    - Confidence calibration
    """
    
    def __init__(
        self,
        n_regimes: int = 6,
        use_gpu: bool = True,
    ):
        self.n_regimes = n_regimes
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        # GMM model
        self.gmm = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.fitted = False
        
        # Feature history
        self.feature_history = deque(maxlen=1000)
        
        logger.info(f"Enhanced GMM Classifier | Regimes: {n_regimes}")
    
    def extract_features(
        self,
        returns: np.ndarray,
        funding: np.ndarray,
        obi: np.ndarray,
        volume: np.ndarray,
    ) -> np.ndarray:
        """
        Extract regime features.
        
        Features:
        1. Price velocity (momentum)
        2. Return volatility
        3. Funding rate skew
        4. OBI momentum
        5. Volume profile
        6. Return skewness
        """
        features = []
        
        # Price velocity (20-period momentum)
        if len(returns) >= 20:
            velocity = np.sum(returns[-20:])
        else:
            velocity = np.sum(returns)
        features.append(velocity)
        
        # Volatility (20-period)
        if len(returns) >= 20:
            vol = np.std(returns[-20:])
        else:
            vol = np.std(returns) if len(returns) > 0 else 0.0
        features.append(vol)
        
        # Funding skew
        if len(funding) >= 10:
            funding_skew = np.mean(funding[-10:]) - np.mean(funding[-50:] if len(funding) >= 50 else funding)
        else:
            funding_skew = 0.0
        features.append(funding_skew)
        
        # OBI momentum
        if len(obi) >= 10:
            obi_momentum = np.mean(obi[-5:]) - np.mean(obi[-10:-5] if len(obi) >= 10 else obi)
        else:
            obi_momentum = 0.0
        features.append(obi_momentum)
        
        # Volume profile (current vs average)
        if len(volume) >= 20:
            vol_ratio = np.mean(volume[-5:]) / (np.mean(volume[-20:]) + 1e-10)
        else:
            vol_ratio = 1.0
        features.append(vol_ratio)
        
        # Return skewness
        if len(returns) >= 20 and SCIPY_AVAILABLE:
            skew = stats.skew(returns[-20:])
        else:
            skew = 0.0
        features.append(skew)
        
        return np.array(features)
    
    def fit(self, features: np.ndarray):
        """Fit GMM to historical features."""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available - using mock classifier")
            self.fitted = True
            return
        
        if len(features) < 100:
            logger.warning(f"Insufficient data for GMM fitting: {len(features)}")
            return
        
        # Scale features
        scaled = self.scaler.fit_transform(features)
        
        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            random_state=42,
            n_init=5,
        )
        self.gmm.fit(scaled)
        self.fitted = True
        
        logger.info(f"GMM fitted with {len(features)} samples")
    
    def predict(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict regime and probabilities.
        
        Returns:
            (regime_idx, probabilities)
        """
        if not self.fitted or self.gmm is None:
            # Return neutral with uniform probabilities
            probs = np.ones(self.n_regimes) / self.n_regimes
            return 2, probs  # NEUTRAL
        
        features = features.reshape(1, -1)
        scaled = self.scaler.transform(features)
        
        probs = self.gmm.predict_proba(scaled)[0]
        regime_idx = np.argmax(probs)
        
        return regime_idx, probs


# =============================================================================
# ENHANCED MARKET REGIME NAVIGATOR
# =============================================================================

class EnhancedMarketRegimeNavigator:
    """
    Enhanced Market Regime Navigator - SOTA v3.0
    
    Integrates:
    1. GMM/HMM regime detection
    2. Vol-of-Vol tensor
    3. Cost-aware hysteresis
    4. SOL network stress
    5. Beta-decoupling detection
    """
    
    def __init__(
        self,
        hysteresis_config: Optional[HysteresisConfig] = None,
        use_gpu: bool = True,
    ):
        # Configuration
        self.hysteresis_config = hysteresis_config or HysteresisConfig()
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        # Components
        self.vov_calculator = VolOfVolCalculator(use_gpu=use_gpu)
        self.hysteresis_filter = CostAwareHysteresisFilter(self.hysteresis_config)
        self.sol_stress_integrator = SOLNetworkStressIntegrator()
        self.gmm_classifier = EnhancedGMMClassifier(use_gpu=use_gpu)
        
        # State
        self.current_tensor: Optional[RegimeTensorV3] = None
        self.regime_history = deque(maxlen=500)
        
        # Feature history for GMM
        self.returns_history: Dict[str, deque] = {
            asset: deque(maxlen=500) for asset in ['BTC', 'ETH', 'SOL']
        }
        self.funding_history = deque(maxlen=500)
        self.obi_history = deque(maxlen=500)
        self.volume_history = deque(maxlen=500)
        
        logger.info("Enhanced Regime Navigator initialized | "
                   f"GPU: {self.use_gpu}")
    
    def update(
        self,
        # Price data
        returns: Dict[str, float],  # {asset: return}
        prices: Optional[Dict[str, float]] = None,
        
        # Derivatives data
        funding_rates: Optional[Dict[str, float]] = None,
        
        # Orderbook data
        obi: Optional[Dict[str, float]] = None,
        
        # Volume data
        volumes: Optional[Dict[str, float]] = None,
        
        # SOL network data
        sol_cu_utilization: float = 0.0,
        sol_priority_fee: float = 0.0,
        sol_tps: float = 4000.0,
        
        # Sentiment
        sentiment_sigma: float = 0.0,
        
        # Positions (for cost calculation)
        positions: Optional[Dict[str, float]] = None,
    ) -> RegimeTensorV3:
        """
        Update regime navigator with new market data.
        
        Returns:
            RegimeTensorV3 with full regime state
        """
        timestamp = time.time()
        
        # Update return histories
        for asset, ret in returns.items():
            if asset in self.returns_history:
                self.returns_history[asset].append(ret)
                self.vov_calculator.update_returns(asset, ret)
        
        # Update other histories
        if funding_rates:
            avg_funding = np.mean(list(funding_rates.values()))
            self.funding_history.append(avg_funding)
        
        if obi:
            avg_obi = np.mean(list(obi.values()))
            self.obi_history.append(avg_obi)
        
        if volumes:
            total_vol = sum(volumes.values())
            self.volume_history.append(total_vol)
        
        # Calculate Vol-of-Vol tensor
        vov_tensor = self.vov_calculator.calculate_tensor()
        
        # Update SOL network stress
        sol_stress = self.sol_stress_integrator.update(
            sol_cu_utilization, sol_priority_fee, sol_tps
        )
        
        # Extract GMM features
        btc_returns = np.array(self.returns_history['BTC'])
        funding_arr = np.array(self.funding_history)
        obi_arr = np.array(self.obi_history)
        volume_arr = np.array(self.volume_history)
        
        features = self.gmm_classifier.extract_features(
            btc_returns, funding_arr, obi_arr, volume_arr
        )
        
        # Get regime probabilities
        regime_idx, regime_probs = self.gmm_classifier.predict(features)
        
        # Apply hysteresis filter
        should_switch, final_regime, reason = self.hysteresis_filter.should_switch(
            regime_probs=regime_probs,
            sentiment_sigma=sentiment_sigma,
            positions=positions,
            expected_duration=24.0,
        )
        
        if not should_switch:
            final_regime = self.hysteresis_filter.current_regime
        
        # Calculate liquidity regimes
        liquidity_regimes = self._calculate_liquidity_regimes(obi, volumes)
        
        # Adjust SOL liquidity based on network stress
        if 'SOL' in liquidity_regimes:
            liquidity_regimes['SOL'] = self.sol_stress_integrator.adjust_sol_liquidity_regime(
                liquidity_regimes['SOL']
            )
        
        # Calculate betas
        beta_btc_eth = self._calculate_beta('BTC', 'ETH')
        beta_btc_sol = self._calculate_beta('BTC', 'SOL')
        
        # Check for decoupling
        beta_decoupling = (
            abs(beta_btc_sol - 1.5) > 0.5 or  # SOL beta deviation
            np.max(vov_tensor.decoupling_scores) > 0.15  # Correlation breakdown
        )
        
        # Calculate switch cost
        positions = positions or {'BTC': 0.25, 'ETH': 0.25, 'SOL': 0.50}
        switch_cost = self.hysteresis_filter._estimate_switch_cost(
            self.hysteresis_filter.current_regime, final_regime, positions
        )
        
        # Build tensor
        tensor = RegimeTensorV3(
            timestamp=timestamp,
            market_regime=final_regime,
            regime_confidence=regime_probs[final_regime.value],
            regime_probabilities=regime_probs,
            liquidity_regimes=liquidity_regimes,
            vov_tensor=vov_tensor,
            hysteresis_active=not should_switch,
            hysteresis_counter=self.hysteresis_filter.persistence_counter,
            expected_regime_duration=self._estimate_duration(final_regime, regime_probs),
            regime_switch_cost_bps=switch_cost,
            switch_recommended=should_switch,
            beta_btc_eth=beta_btc_eth,
            beta_btc_sol=beta_btc_sol,
            beta_decoupling=beta_decoupling,
            correlation_btc_eth=float(vov_tensor.correlation[0, 1, 1]),
            correlation_btc_sol=float(vov_tensor.correlation[0, 2, 1]),
            correlation_eth_sol=float(vov_tensor.correlation[1, 2, 1]),
            sol_network_stress=sol_stress,
            sol_cu_utilization=sol_cu_utilization,
        )
        
        self.current_tensor = tensor
        self.regime_history.append({
            'timestamp': timestamp,
            'regime': final_regime,
            'confidence': tensor.regime_confidence,
        })
        
        return tensor
    
    def _calculate_liquidity_regimes(
        self,
        obi: Optional[Dict[str, float]],
        volumes: Optional[Dict[str, float]],
    ) -> Dict[str, LiquidityRegime]:
        """Calculate liquidity regime per asset."""
        regimes = {}
        
        for asset in ['BTC', 'ETH', 'SOL']:
            asset_obi = obi.get(asset, 0.0) if obi else 0.0
            asset_vol = volumes.get(asset, 1e6) if volumes else 1e6
            
            # Simple liquidity score
            # High volume + balanced OBI = liquid
            vol_score = min(asset_vol / 1e7, 1.0)  # Normalize to $10M baseline
            obi_penalty = abs(asset_obi)  # High imbalance = less liquid
            
            liquidity_score = vol_score * (1 - obi_penalty)
            
            if liquidity_score > 0.7:
                regimes[asset] = LiquidityRegime.DEEP_LIQUID
            elif liquidity_score > 0.4:
                regimes[asset] = LiquidityRegime.MODERATE_LIQUID
            elif liquidity_score > 0.2:
                regimes[asset] = LiquidityRegime.THIN_LIQUID
            elif liquidity_score > 0.1:
                regimes[asset] = LiquidityRegime.STRESSED
            else:
                regimes[asset] = LiquidityRegime.CRISIS
        
        return regimes
    
    def _calculate_beta(self, asset: str, benchmark: str) -> float:
        """Calculate rolling beta."""
        if asset not in self.returns_history or benchmark not in self.returns_history:
            return 1.0
        
        asset_rets = np.array(self.returns_history[asset])
        bench_rets = np.array(self.returns_history[benchmark])
        
        min_len = min(len(asset_rets), len(bench_rets))
        if min_len < 20:
            return 1.0
        
        asset_rets = asset_rets[-min_len:]
        bench_rets = bench_rets[-min_len:]
        
        cov = np.cov(asset_rets, bench_rets)[0, 1]
        var = np.var(bench_rets)
        
        if var < 1e-10:
            return 1.0
        
        return cov / var
    
    def _estimate_duration(
        self,
        regime: MarketRegime,
        probs: np.ndarray,
    ) -> float:
        """Estimate expected regime duration in hours."""
        # Base durations (calibrated)
        base_durations = {
            MarketRegime.STRONG_BULL: 48.0,
            MarketRegime.MODERATE_BULL: 72.0,
            MarketRegime.NEUTRAL: 96.0,
            MarketRegime.VOLATILE_CHOP: 24.0,
            MarketRegime.MODERATE_BEAR: 72.0,
            MarketRegime.STRONG_BEAR: 36.0,
        }
        
        base = base_durations.get(regime, 48.0)
        confidence = probs[regime.value]
        
        # Higher confidence = longer expected duration
        return base * (0.5 + confidence)
    
    def get_execution_guidance(self) -> Dict[str, Any]:
        """
        Get execution guidance based on current regime.
        """
        if self.current_tensor is None:
            return {
                'mode': 'PASSIVE',
                'urgency': 'LOW',
                'size_multiplier': 0.5,
                'sol_special': {},
            }
        
        tensor = self.current_tensor
        
        # Base guidance by regime
        regime_guidance = {
            MarketRegime.STRONG_BULL: {'mode': 'AGGRESSIVE', 'urgency': 'HIGH', 'size_mult': 1.0},
            MarketRegime.MODERATE_BULL: {'mode': 'NORMAL', 'urgency': 'MEDIUM', 'size_mult': 0.8},
            MarketRegime.NEUTRAL: {'mode': 'PASSIVE', 'urgency': 'LOW', 'size_mult': 0.5},
            MarketRegime.VOLATILE_CHOP: {'mode': 'HIDDEN', 'urgency': 'VERY_LOW', 'size_mult': 0.3},
            MarketRegime.MODERATE_BEAR: {'mode': 'PASSIVE', 'urgency': 'LOW', 'size_mult': 0.5},
            MarketRegime.STRONG_BEAR: {'mode': 'DEFENSIVE', 'urgency': 'MEDIUM', 'size_mult': 0.4},
        }
        
        guidance = regime_guidance.get(tensor.market_regime, regime_guidance[MarketRegime.NEUTRAL]).copy()
        
        # Adjust for vol-of-vol
        if tensor.vov_tensor and tensor.vov_tensor.vov_regime == VolatilityRegime.EXPLOSIVE:
            guidance['size_mult'] *= 0.5
            guidance['mode'] = 'HIDDEN'
        
        # SOL-specific guidance
        sol_guidance = {
            'network_stress': tensor.sol_network_stress,
            'cu_utilization': tensor.sol_cu_utilization,
            'liquidity': tensor.liquidity_regimes.get('SOL', LiquidityRegime.MODERATE_LIQUID).name,
            'use_priority_fee': tensor.sol_network_stress in ['HIGH', 'CRITICAL'],
            'abort_execution': tensor.sol_network_stress == 'CRITICAL',
        }
        
        return {
            'mode': guidance['mode'],
            'urgency': guidance['urgency'],
            'size_multiplier': guidance['size_mult'],
            'regime': tensor.market_regime.name,
            'confidence': tensor.regime_confidence,
            'hysteresis_active': tensor.hysteresis_active,
            'switch_cost_bps': tensor.regime_switch_cost_bps,
            'sol_special': sol_guidance,
        }
    
    def fit_gmm(self):
        """Fit GMM to accumulated history."""
        # Build feature matrix from history
        btc_rets = np.array(self.returns_history['BTC'])
        funding = np.array(self.funding_history)
        obi = np.array(self.obi_history)
        volume = np.array(self.volume_history)
        
        if len(btc_rets) < 100:
            logger.warning("Insufficient history for GMM fitting")
            return
        
        features = []
        for i in range(100, len(btc_rets)):
            feat = self.gmm_classifier.extract_features(
                btc_rets[:i], funding[:i], obi[:i], volume[:i]
            )
            features.append(feat)
        
        self.gmm_classifier.fit(np.array(features))


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Demonstrate Enhanced Regime Navigator."""
    
    # Initialize
    navigator = EnhancedMarketRegimeNavigator(use_gpu=True)
    
    # Simulate data updates
    np.random.seed(42)
    
    for i in range(200):
        # Simulate returns
        returns = {
            'BTC': np.random.normal(0.001, 0.02),
            'ETH': np.random.normal(0.0012, 0.025),
            'SOL': np.random.normal(0.0015, 0.035),
        }
        
        # Simulate other data
        funding = {'BTC': 0.0001, 'ETH': 0.00012, 'SOL': 0.00015}
        obi = {'BTC': np.random.uniform(-0.1, 0.1), 
               'ETH': np.random.uniform(-0.15, 0.15),
               'SOL': np.random.uniform(-0.2, 0.2)}
        volumes = {'BTC': 50e6, 'ETH': 20e6, 'SOL': 5e6}
        
        # SOL network
        sol_cu = np.random.uniform(0.3, 0.7)
        sol_fee = np.random.uniform(1000, 10000)
        sol_tps = np.random.uniform(2500, 4000)
        
        # Update
        tensor = navigator.update(
            returns=returns,
            funding_rates=funding,
            obi=obi,
            volumes=volumes,
            sol_cu_utilization=sol_cu,
            sol_priority_fee=sol_fee,
            sol_tps=sol_tps,
        )
        
        if i % 50 == 0:
            print(f"\n=== Update {i} ===")
            print(f"Regime: {tensor.market_regime.name} "
                  f"(conf: {tensor.regime_confidence:.2f})")
            print(f"Hysteresis: {tensor.hysteresis_active} "
                  f"(counter: {tensor.hysteresis_counter})")
            print(f"Vol-of-Vol: {tensor.vov_tensor.vov_regime.name}")
            print(f"SOL Network: {tensor.sol_network_stress}")
            print(f"Beta BTC-SOL: {tensor.beta_btc_sol:.2f}")
            print(f"Decoupling: {tensor.beta_decoupling}")
    
    # Fit GMM after warmup
    navigator.fit_gmm()
    
    # Get execution guidance
    guidance = navigator.get_execution_guidance()
    print("\n=== Execution Guidance ===")
    for k, v in guidance.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    example_usage()

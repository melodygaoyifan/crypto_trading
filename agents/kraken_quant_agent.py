"""
HMATS Institutional Quant Agent - Kraken Edition (V6)
=====================================================
STATUS: HMATS v6 compatible. Supplementary ADVISE-authority agent.
Primary live quant path: main.py Best-of-N runtime + execution gating.
This agent provides additional alpha signals via 12 research-grade strategies.

Public interface:
  - KrakenQuantAgentV6.generate_signal(asset, market_data, regime, **kw) -> KrakenQuantPayload
  - KrakenQuantPayload.to_agent_signal_dict() -> Dict[str, Any]
  - get_kraken_quant_agent() / reset_kraken_quant_agent() (singleton)

3x4 Strategy Matrix: 3 Regimes × 4 Strategies = 12 Total Strategies

Mathematical foundations:
- Stochastic Calculus (OU Process, GBM)
- Information Theory (Shannon Entropy, Mutual Information)
- Derivatives Microstructure (Liquidations, Funding, OI Dynamics)

Exchange: Kraken (Spot, Margin, Futures)
Assets: BTC, ETH, SOL
Authority: ADVISE (supplementary to main.py Best-of-N live quant core)
"""

import json
import numpy as np
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from abc import ABC, abstractmethod
import logging
import os
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Optional imports (scipy used by some strategies internally)
try:
    from scipy import stats, optimize  # noqa: F401
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================

SQRT_2 = np.sqrt(2)
LOG_2 = np.log(2)
EULER_GAMMA = 0.5772156649


# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class Regime(Enum):
    """Market regime classification."""
    BEAR = "bear"      # Crash & Decay
    BULL = "bull"      # Institutional Flow
    SIDEWAYS = "chop"  # Neutral Extraction


class SignalType(Enum):
    """Trading signal types."""
    NO_SIGNAL = 0
    AGGRESSIVE_LONG = 1
    LONG = 2
    REDUCE_LONG = 3
    FLATTEN = 4
    SHORT = 5
    AGGRESSIVE_SHORT = 6
    REDUCE_SHORT = 7
    PAIR_LONG_A_SHORT_B = 8  # For stat-arb
    PAIR_SHORT_A_LONG_B = 9


class Signal(NamedTuple):
    """Trading signal with full metadata."""
    timestamp: float
    strategy_id: int
    strategy_name: str
    regime: Regime
    asset: str
    signal_type: SignalType
    size_pct: float          # 0.0 to 1.0 of allocated capital
    confidence: float        # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    metadata: Dict


@dataclass
class StrategyState:
    """Internal state for each strategy."""
    strategy_id: int
    name: str
    regime: Regime
    enabled: bool = True
    in_position: bool = False
    position_side: Optional[str] = None
    entry_price: float = 0.0
    entry_time: float = 0.0
    pnl_history: deque = field(default_factory=lambda: deque(maxlen=100))
    signal_count: int = 0
    win_count: int = 0
    
    @property
    def win_rate(self) -> float:
        return self.win_count / max(1, self.signal_count)
    
    @property
    def information_ratio(self) -> float:
        """Calculate Information Ratio from recent PnL."""
        if len(self.pnl_history) < 10:
            return 0.0
        pnl = np.array(self.pnl_history)
        mean_ret = np.mean(pnl)
        std_ret = np.std(pnl) + 1e-10
        # Annualize (assuming hourly data)
        return (mean_ret * np.sqrt(365 * 24)) / std_ret


# =============================================================================
# BASE STRATEGY CLASS
# =============================================================================

class BaseStrategy(ABC):
    """Abstract base class for all strategies."""
    
    def __init__(
        self,
        strategy_id: int,
        name: str,
        regime: Regime,
        lookback: int = 100,
    ):
        self.strategy_id = strategy_id
        self.name = name
        self.regime = regime
        self.lookback = lookback
        self.state = StrategyState(strategy_id, name, regime)
        
        # Data buffers
        self.price_buffer: Dict[str, deque] = {
            'BTC': deque(maxlen=lookback),
            'ETH': deque(maxlen=lookback),
            'SOL': deque(maxlen=lookback),
        }
        
    @abstractmethod
    def update(self, market_data: Dict) -> Optional[Signal]:
        """Process new market data and generate signal."""
        pass

    def _dynamic_stop(self, price: float, direction: str, asset: str = "SOL") -> float:
        """
        [UPG6] ATR-adaptive stop loss from price_buffer history.
        direction: 'short' or 'long'
        """
        buf = self.price_buffer.get(asset, deque())
        if len(buf) < 14:
            # Fallback: fixed 3%
            return price * 1.03 if direction == "short" else price * 0.97

        prices = list(buf)[-14:]
        high_low_range = max(prices) - min(prices)
        mean_price = np.mean(prices) if np.mean(prices) > 0 else price
        atr_proxy = high_low_range / len(prices)  # Simple avg range per bar

        stop_distance = atr_proxy * 2.0  # 2× ATR multiplier
        # Clamp: 1% to 5% of price
        min_stop = price * 0.01
        max_stop = price * 0.05
        stop_distance = max(min_stop, min(stop_distance, max_stop))

        if direction == "short":
            return price + stop_distance
        else:
            return price - stop_distance

    def record_trade_result(self, pnl: float, won: bool):
        """Record trade result for IR calculation."""
        self.state.pnl_history.append(pnl)
        self.state.signal_count += 1
        if won:
            self.state.win_count += 1


# =============================================================================
# REGIME A: BEAR MARKET STRATEGIES (Crash & Decay)
# =============================================================================

class LiquidationCascadeHunter(BaseStrategy):
    """
    Strategy 1: Liquidation Cascade Hunter
    ======================================
    
    MATHEMATICAL FRAMEWORK:
    -----------------------
    
    Open Interest Decay Velocity:
        dOI/dt = (OI_t - OI_{t-1}) / Δt
    
    Liquidation Cluster Detection:
        L_intensity = Σ L_i / Δt  (liquidation volume per unit time)
        
    Cascade Condition:
        dOI/dt < -θ_oi  AND  dP/dt < -θ_price  AND  L_intensity > θ_liq
    
    Signal Strength:
        S = α₁|dOI/dt| + α₂|dP/dt| + α₃·L_intensity
    
    Exit Condition:
        Taker_Buy_Sell_Ratio = Taker_Buys / Taker_Sells > 1.0  (Absorption)
    """
    
    def __init__(self):
        super().__init__(
            strategy_id=1,
            name="LiquidationCascadeHunter",
            regime=Regime.BEAR,
            lookback=200
        )
        
        # Thresholds
        self.oi_velocity_threshold = -0.02  # -2% OI drop per period
        self.price_velocity_threshold = -0.005  # -0.5% price drop
        self.liquidation_threshold_sigma = 2.5
        self.absorption_threshold = 1.0  # Taker ratio for exit
        # [PORT 2026-04-22] CVD z-score threshold ported from
        # agents/quant_agent.py:LiquidationCascadeHunter. Previously missing —
        # adds "aggressive selling" signal dimension to cascade detection.
        self.cvd_zscore_threshold = -1.5

        # Buffers
        self.oi_buffer: Dict[str, deque] = {
            'BTC': deque(maxlen=100),
            'ETH': deque(maxlen=100),
            'SOL': deque(maxlen=100),
        }
        self.liquidation_buffer: Dict[str, deque] = {
            'BTC': deque(maxlen=500),
            'ETH': deque(maxlen=500),
            'SOL': deque(maxlen=500),
        }
        self.taker_ratio_buffer: Dict[str, deque] = {
            'BTC': deque(maxlen=50),
            'ETH': deque(maxlen=50),
            'SOL': deque(maxlen=50),
        }
        # [PORT 2026-04-22] CVD buffer for z-score-based cascade detection
        self.cvd_buffer: Dict[str, deque] = {
            'BTC': deque(maxlen=100),
            'ETH': deque(maxlen=100),
            'SOL': deque(maxlen=100),
        }

        # Cascade state
        self.cascade_active = False
        self.cascade_start_time = 0.0
        
    def _calculate_oi_velocity(self, asset: str) -> float:
        """Calculate dOI/dt (normalized)."""
        if len(self.oi_buffer[asset]) < 5:
            return 0.0
        
        oi = np.array(self.oi_buffer[asset])
        # Linear regression slope
        x = np.arange(len(oi))
        slope = np.polyfit(x, oi, 1)[0]
        
        # Normalize by mean OI
        mean_oi = np.mean(oi) + 1e-10
        return slope / mean_oi
    
    def _calculate_liquidation_intensity(self, asset: str) -> Tuple[float, float]:
        """Calculate liquidation intensity and z-score."""
        if len(self.liquidation_buffer[asset]) < 20:
            return 0.0, 0.0
        
        liq = np.array(self.liquidation_buffer[asset])
        current = liq[-1] if len(liq) > 0 else 0
        
        mean_liq = np.mean(liq)
        std_liq = np.std(liq) + 1e-10
        
        zscore = (current - mean_liq) / std_liq
        return current, zscore
    
    def _check_absorption(self, asset: str) -> bool:
        """Check if buying absorption is occurring."""
        if len(self.taker_ratio_buffer[asset]) < 5:
            return False

        recent = list(self.taker_ratio_buffer[asset])[-5:]
        avg_ratio = np.mean(recent)

        return avg_ratio > self.absorption_threshold

    def _calculate_cvd_zscore(self, asset: str) -> float:
        """[PORT 2026-04-22] Rolling z-score of CVD (Cumulative Volume Delta).
        Ported from agents/quant_agent.py:LiquidationCascadeHunter.
        Strongly negative z-score (< -1.5) = aggressive selling, cascade fuel.
        """
        buf = self.cvd_buffer[asset]
        if len(buf) < 10:
            return 0.0
        cvd_array = np.array(buf)
        current = cvd_array[-1]
        mean_cvd = np.mean(cvd_array)
        std_cvd = np.std(cvd_array) + 1e-10
        return (current - mean_cvd) / std_cvd
    
    def update(self, market_data: Dict) -> Optional[Signal]:
        """
        Process market data and detect liquidation cascades.
        
        Expected market_data keys:
        - prices: {BTC: float, ETH: float, SOL: float}
        - open_interest: {BTC: float, ETH: float, SOL: float}
        - liquidations: {BTC: float, ETH: float, SOL: float}  # Recent liq volume
        - taker_ratio: {BTC: float, ETH: float, SOL: float}
        - timestamp: float
        """
        timestamp = market_data.get('timestamp', time.time())
        
        # Update buffers
        for asset in ['BTC', 'ETH', 'SOL']:
            if asset in market_data.get('prices', {}):
                self.price_buffer[asset].append(market_data['prices'][asset])
            if asset in market_data.get('open_interest', {}):
                self.oi_buffer[asset].append(market_data['open_interest'][asset])
            if asset in market_data.get('liquidations', {}):
                self.liquidation_buffer[asset].append(market_data['liquidations'][asset])
            if asset in market_data.get('taker_ratio', {}):
                self.taker_ratio_buffer[asset].append(market_data['taker_ratio'][asset])
            # [PORT 2026-04-22] CVD buffer — optional, graceful fallback if key missing
            if asset in market_data.get('cvd', {}):
                self.cvd_buffer[asset].append(market_data['cvd'][asset])

        # Need enough data
        if len(self.price_buffer['BTC']) < 20:
            return None

        # Check for cascade on BTC (leading indicator)
        btc_oi_velocity = self._calculate_oi_velocity('BTC')
        btc_liq_intensity, btc_liq_zscore = self._calculate_liquidation_intensity('BTC')
        btc_cvd_zscore = self._calculate_cvd_zscore('BTC')

        # Calculate BTC price velocity
        btc_prices = np.array(self.price_buffer['BTC'])
        btc_price_velocity = (btc_prices[-1] - btc_prices[-5]) / btc_prices[-5] if len(btc_prices) >= 5 else 0

        # [PORT 2026-04-22] Cascade detection — 4 conditions now (CVD z-score added).
        # CVD check is tolerant: counts as "unknown" (neutral) when buffer is cold,
        # counts as "confirming" when cvd_zscore < -1.5 (aggressive selling).
        cascade_conditions = {
            'oi_dropping': btc_oi_velocity < self.oi_velocity_threshold,
            'price_dropping': btc_price_velocity < self.price_velocity_threshold,
            'liquidation_spike': btc_liq_zscore > self.liquidation_threshold_sigma,
            'cvd_aggressive_selling': btc_cvd_zscore < self.cvd_zscore_threshold,
        }

        conditions_met = sum(cascade_conditions.values())
        # [PORT 2026-04-22] Keep threshold at 3 (not 4) so cold-CVD buffers don't
        # block signals. When CVD is present and confirming, hits 4/4 and
        # signal gets a stronger confidence bump (below).
        if conditions_met >= 3 and not self.cascade_active:
            self.cascade_active = True
            self.cascade_start_time = timestamp
            self.state.in_position = True
            self.state.position_side = 'short'
            
            # Short SOL (highest beta)
            sol_price = market_data['prices'].get('SOL', 0)
            
            # [PORT 2026-04-22] Signal strength — CVD component added (matches
            # quant_agent weight scheme: 0.35 liq + 0.25 vel + 0.25 cvd + 0.15 oi).
            # Default weights preserved; CVD replaces old OI weight when active.
            cvd_component = min(1.0, abs(btc_cvd_zscore) / 3.0) if btc_cvd_zscore < 0 else 0.0
            strength = (
                0.30 * min(1.0, abs(btc_oi_velocity) / 0.05) +
                0.30 * min(1.0, abs(btc_price_velocity) / 0.02) +
                0.25 * min(1.0, btc_liq_zscore / 5) +
                0.15 * cvd_component
            )

            return Signal(
                timestamp=timestamp,
                strategy_id=self.strategy_id,
                strategy_name=self.name,
                regime=self.regime,
                asset="SOL",
                signal_type=SignalType.AGGRESSIVE_SHORT,
                size_pct=min(1.0, 0.5 + strength * 0.5),
                confidence=min(0.95, 0.6 + strength * 0.3),
                entry_price=sol_price,
                stop_loss=self._dynamic_stop(sol_price, "short", "SOL"),  # [UPG6]
                take_profit=sol_price * 0.94,
                metadata={
                    'btc_oi_velocity': btc_oi_velocity,
                    'btc_price_velocity': btc_price_velocity,
                    'btc_liq_zscore': btc_liq_zscore,
                    'btc_cvd_zscore': btc_cvd_zscore,  # [PORT 2026-04-22]
                    'conditions': cascade_conditions,
                }
            )
        
        # Exit logic (absorption detected)
        if self.cascade_active:
            sol_absorbed = self._check_absorption('SOL')
            
            if sol_absorbed:
                self.cascade_active = False
                self.state.in_position = False
                
                sol_price = market_data['prices'].get('SOL', 0)
                
                return Signal(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    strategy_name=self.name,
                    regime=self.regime,
                    asset="SOL",
                    signal_type=SignalType.FLATTEN,
                    size_pct=1.0,
                    confidence=0.8,
                    entry_price=sol_price,
                    stop_loss=sol_price,
                    take_profit=sol_price,
                    metadata={'reason': 'absorption_detected'}
                )
        
        return None


class HurstExponentStrategy(BaseStrategy):
    """
    Strategy 2: Fractal Trend Persistence (Hurst Exponent)
    ======================================================
    
    MATHEMATICAL FRAMEWORK:
    -----------------------
    
    The Hurst Exponent H measures long-term memory in time series:
        - H = 0.5: Random walk (Brownian motion)
        - H > 0.5: Trending (persistent)
        - H < 0.5: Mean-reverting (anti-persistent)
    
    R/S Analysis (Rescaled Range):
        1. Divide series into m sub-periods of length n
        2. For each sub-period:
           - Calculate mean: μ = (1/n) Σ x_i
           - Calculate deviations: y_i = x_i - μ
           - Cumulative deviations: Z_i = Σ_{j=1}^{i} y_j
           - Range: R = max(Z) - min(Z)
           - Standard deviation: S = √[(1/n) Σ (x_i - μ)²]
           - R/S statistic: (R/S)_n
        3. Fit: log(R/S) = H × log(n) + c
    
    Alternative: DFA (Detrended Fluctuation Analysis)
        F(n) ~ n^H
    
    Signal:
        SHORT when H > 0.65 AND Price < VWAP (trend persistence in decline)
        Filters out "dead cat bounces"
    """
    
    def __init__(self):
        super().__init__(
            strategy_id=2,
            name="HurstExponentStrategy",
            regime=Regime.BEAR,
            lookback=500
        )
        
        # Thresholds
        self.hurst_trending_threshold = 0.65
        self.min_data_points = 100
        
        # State
        self.hurst_history: Dict[str, deque] = {
            'BTC': deque(maxlen=50),
            'ETH': deque(maxlen=50),
            'SOL': deque(maxlen=50),
        }
        
        # VWAP tracking
        self.vwap_buffer: Dict[str, deque] = {
            'BTC': deque(maxlen=100),
            'ETH': deque(maxlen=100),
            'SOL': deque(maxlen=100),
        }
        
    def calculate_hurst_rs(self, prices: np.ndarray) -> float:
        """
        Calculate Hurst exponent using R/S (Rescaled Range) analysis.
        
        Implementation of the R/S algorithm:
        1. Calculate returns
        2. For different time scales n, calculate R/S
        3. Fit log(R/S) vs log(n) to get H
        """
        if len(prices) < self.min_data_points:
            return 0.5
        
        # Convert to log returns
        returns = np.diff(np.log(prices))
        N = len(returns)
        
        # Range of time scales
        max_k = int(np.log2(N)) - 1
        if max_k < 2:
            return 0.5
        
        ns = [2**i for i in range(2, max_k + 1)]
        rs_values = []
        
        for n in ns:
            # Number of sub-periods
            m = N // n
            if m < 1:
                continue
            
            rs_n = []
            for i in range(m):
                subseries = returns[i*n:(i+1)*n]
                
                # Mean and deviations
                mean = np.mean(subseries)
                deviations = subseries - mean
                
                # Cumulative deviations
                cumsum = np.cumsum(deviations)
                
                # Range
                R = np.max(cumsum) - np.min(cumsum)
                
                # Standard deviation
                S = np.std(subseries, ddof=1) + 1e-10
                
                rs_n.append(R / S)
            
            rs_values.append((n, np.mean(rs_n)))
        
        if len(rs_values) < 2:
            return 0.5
        
        # Fit log-log regression
        log_n = np.log([x[0] for x in rs_values])
        log_rs = np.log([x[1] for x in rs_values])
        
        # OLS fit: log(R/S) = H * log(n) + c
        H = np.polyfit(log_n, log_rs, 1)[0]
        
        # Bound to valid range
        return np.clip(H, 0.0, 1.0)
    
    def calculate_hurst_dfa(self, prices: np.ndarray) -> float:
        """
        Alternative: Detrended Fluctuation Analysis (DFA).
        
        More robust for non-stationary series.
        
        Algorithm:
        1. Integrate the series: Y(i) = Σ_{k=1}^{i} [x_k - <x>]
        2. Divide into boxes of size n
        3. Fit polynomial trend in each box, calculate residuals
        4. F(n) = √[mean(residuals²)]
        5. Fit: log(F) = H * log(n) + c
        """
        if len(prices) < self.min_data_points:
            return 0.5
        
        returns = np.diff(np.log(prices))
        N = len(returns)
        
        # Integrate (cumulative sum of mean-centered returns)
        Y = np.cumsum(returns - np.mean(returns))
        
        # Range of scales
        scales = [2**i for i in range(4, int(np.log2(N/4)) + 1)]
        fluct = []
        
        for n in scales:
            # Number of boxes
            n_boxes = N // n
            if n_boxes < 1:
                continue
            
            F_n = []
            for b in range(n_boxes):
                idx_start = b * n
                idx_end = idx_start + n
                segment = Y[idx_start:idx_end]
                
                # Fit linear trend
                x = np.arange(n)
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                
                # Variance of residuals
                F_n.append(np.mean((segment - trend)**2))
            
            fluct.append((n, np.sqrt(np.mean(F_n))))
        
        if len(fluct) < 2:
            return 0.5
        
        # Log-log fit
        log_n = np.log([x[0] for x in fluct])
        log_F = np.log([x[1] for x in fluct])
        
        H = np.polyfit(log_n, log_F, 1)[0]
        
        return np.clip(H, 0.0, 1.0)
    
    def _calculate_vwap(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate VWAP."""
        if len(prices) == 0 or len(volumes) == 0:
            return prices[-1] if len(prices) > 0 else 0
        
        return np.sum(prices * volumes) / (np.sum(volumes) + 1e-10)
    
    def update(self, market_data: Dict) -> Optional[Signal]:
        """
        Calculate Hurst exponent and generate trend-following signals.
        """
        timestamp = market_data.get('timestamp', time.time())
        
        # Update price buffers
        for asset in ['BTC', 'ETH', 'SOL']:
            if asset in market_data.get('prices', {}):
                self.price_buffer[asset].append(market_data['prices'][asset])
        
        # Need enough data
        if len(self.price_buffer['SOL']) < self.min_data_points:
            return None
        
        # Calculate Hurst for SOL
        sol_prices = np.array(self.price_buffer['SOL'])
        H = self.calculate_hurst_rs(sol_prices)
        
        self.hurst_history['SOL'].append(H)
        
        # Also calculate for BTC as confirmation
        btc_prices = np.array(self.price_buffer['BTC'])
        H_btc = self.calculate_hurst_rs(btc_prices) if len(btc_prices) >= self.min_data_points else 0.5
        
        # Calculate simple VWAP proxy (equal volume assumption)
        sol_vwap = np.mean(sol_prices[-20:])
        current_price = sol_prices[-1]
        
        # Check trend direction
        price_below_vwap = current_price < sol_vwap
        price_trend = (sol_prices[-1] - sol_prices[-20]) / sol_prices[-20] if len(sol_prices) >= 20 else 0
        
        # Signal: Trending (H > 0.65) + Bearish (price < VWAP) + Downtrend
        if (H > self.hurst_trending_threshold and 
            price_below_vwap and 
            price_trend < 0 and
            not self.state.in_position):
            
            self.state.in_position = True
            self.state.position_side = 'short'
            
            # Confidence based on Hurst strength
            confidence = 0.5 + (H - 0.5) * 0.8
            
            return Signal(
                timestamp=timestamp,
                strategy_id=self.strategy_id,
                strategy_name=self.name,
                regime=self.regime,
                asset="SOL",
                signal_type=SignalType.SHORT,
                size_pct=min(1.0, 0.3 + (H - 0.5) * 1.4),
                confidence=min(0.9, confidence),
                entry_price=current_price,
                stop_loss=self._dynamic_stop(current_price, "short", "SOL"),  # [UPG6]
                take_profit=current_price * 0.95,
                metadata={
                    'hurst_sol': H,
                    'hurst_btc': H_btc,
                    'vwap': sol_vwap,
                    'price_trend': price_trend,
                }
            )
        
        # Exit: H dropping below 0.5 (no longer trending) or price above VWAP
        if self.state.in_position:
            if H < 0.5 or current_price > sol_vwap * 1.01:
                self.state.in_position = False
                
                return Signal(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    strategy_name=self.name,
                    regime=self.regime,
                    asset="SOL",
                    signal_type=SignalType.FLATTEN,
                    size_pct=1.0,
                    confidence=0.7,
                    entry_price=current_price,
                    stop_loss=current_price,
                    take_profit=current_price,
                    metadata={'reason': 'trend_exhaustion', 'hurst': H}
                )
        
        return None


class ShannonEntropyStrategy(BaseStrategy):
    """
    Strategy 3: Shannon Entropy Breakdown
    =====================================
    
    MATHEMATICAL FRAMEWORK:
    -----------------------
    
    Shannon Entropy measures the "disorder" or unpredictability of returns:
    
        H(X) = -Σ p(x) × log₂(p(x))
    
    Where p(x) is the probability of return falling in bin x.
    
    Implementation:
    1. Discretize returns into N bins
    2. Calculate probability distribution
    3. Compute entropy
    
    Normalized Entropy (0 to 1):
        H_norm = H(X) / log₂(N)
    
    Interpretation:
        - High entropy (->1): Chaotic, unpredictable (breakdown starting)
        - Low entropy (->0): Orderly, consolidation (quiet market)
    
    Signal:
        SHORT when entropy SPIKES from low to high (transition to chaos)
        This often precedes violent breakdowns.
    
    Also: Conditional Entropy for regime transition detection
        H(X|Y) = -Σ p(x,y) × log₂(p(x|y))
    """
    
    def __init__(self):
        super().__init__(
            strategy_id=3,
            name="ShannonEntropyStrategy",
            regime=Regime.BEAR,
            lookback=300
        )
        
        self.n_bins = 20
        self.entropy_lookback = 50
        self.spike_threshold = 1.5  # Entropy spike ratio
        self.low_entropy_threshold = 0.4  # Consolidation
        self.high_entropy_threshold = 0.75  # Chaos
        
        # Entropy history
        self.entropy_buffer: Dict[str, deque] = {
            'BTC': deque(maxlen=100),
            'ETH': deque(maxlen=100),
            'SOL': deque(maxlen=100),
        }
        
        # Track consolidation state
        self.in_consolidation = {'BTC': False, 'ETH': False, 'SOL': False}
        
    def calculate_shannon_entropy(self, returns: np.ndarray) -> float:
        """
        Calculate normalized Shannon entropy of returns.
        
        H_norm = H(X) / H_max = -Σ p(x)log₂(p(x)) / log₂(N)
        """
        if len(returns) < 20:
            return 0.5
        
        # Remove NaN and Inf
        returns = returns[np.isfinite(returns)]
        
        # Create histogram (probability distribution)
        hist, _ = np.histogram(returns, bins=self.n_bins, density=True)
        
        # Normalize to probabilities
        hist = hist / (np.sum(hist) + 1e-10)
        
        # Remove zeros for log calculation
        hist = hist[hist > 0]
        
        # Shannon entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log2(self.n_bins)
        
        return entropy / max_entropy
    
    def calculate_entropy_rate(self, asset: str) -> Tuple[float, float]:
        """Calculate current entropy and rate of change."""
        if len(self.entropy_buffer[asset]) < 10:
            return 0.5, 0.0
        
        entropy_array = np.array(self.entropy_buffer[asset])
        current = entropy_array[-1]
        
        # Rate of change
        mean_recent = np.mean(entropy_array[-5:])
        mean_past = np.mean(entropy_array[-20:-5]) if len(entropy_array) >= 20 else mean_recent
        
        rate = (mean_recent - mean_past) / (mean_past + 1e-10)
        
        return current, rate
    
    def update(self, market_data: Dict) -> Optional[Signal]:
        """
        Detect entropy regime transitions.
        """
        timestamp = market_data.get('timestamp', time.time())
        
        # Update price buffers
        for asset in ['BTC', 'ETH', 'SOL']:
            if asset in market_data.get('prices', {}):
                self.price_buffer[asset].append(market_data['prices'][asset])
        
        # Need enough data
        if len(self.price_buffer['SOL']) < self.entropy_lookback:
            return None
        
        # Calculate returns and entropy for each asset
        for asset in ['BTC', 'ETH', 'SOL']:
            prices = np.array(self.price_buffer[asset])
            returns = np.diff(np.log(prices))
            
            if len(returns) >= self.entropy_lookback:
                H = self.calculate_shannon_entropy(returns[-self.entropy_lookback:])
                self.entropy_buffer[asset].append(H)
                
                # Track consolidation state
                if H < self.low_entropy_threshold:
                    self.in_consolidation[asset] = True
                elif H > self.high_entropy_threshold:
                    self.in_consolidation[asset] = False
        
        # Get SOL entropy metrics
        sol_entropy, sol_entropy_rate = self.calculate_entropy_rate('SOL')
        btc_entropy, btc_entropy_rate = self.calculate_entropy_rate('BTC')
        
        # Signal: Was in consolidation + Entropy spike + Negative price action
        sol_prices = np.array(self.price_buffer['SOL'])
        price_trend = (sol_prices[-1] - sol_prices[-10]) / sol_prices[-10] if len(sol_prices) >= 10 else 0
        
        # Detect breakdown: entropy spiking from consolidation
        breakdown_signal = (
            self.in_consolidation.get('SOL', False) and
            sol_entropy > self.high_entropy_threshold and
            sol_entropy_rate > 0.2 and
            price_trend < 0
        )
        
        if breakdown_signal and not self.state.in_position:
            self.state.in_position = True
            self.state.position_side = 'short'
            
            current_price = sol_prices[-1]
            
            return Signal(
                timestamp=timestamp,
                strategy_id=self.strategy_id,
                strategy_name=self.name,
                regime=self.regime,
                asset="SOL",
                signal_type=SignalType.AGGRESSIVE_SHORT,
                size_pct=min(1.0, 0.5 + sol_entropy_rate),
                confidence=min(0.85, 0.5 + sol_entropy * 0.4),
                entry_price=current_price,
                stop_loss=self._dynamic_stop(current_price, "short", "SOL"),  # [UPG6]
                take_profit=current_price * 0.93,
                metadata={
                    'sol_entropy': sol_entropy,
                    'sol_entropy_rate': sol_entropy_rate,
                    'btc_entropy': btc_entropy,
                    'was_consolidating': True,
                }
            )
        
        # Exit: Entropy normalizing
        if self.state.in_position:
            if sol_entropy < 0.5 or sol_entropy_rate < -0.1:
                self.state.in_position = False
                
                return Signal(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    strategy_name=self.name,
                    regime=self.regime,
                    asset="SOL",
                    signal_type=SignalType.FLATTEN,
                    size_pct=1.0,
                    confidence=0.7,
                    entry_price=sol_prices[-1],
                    stop_loss=sol_prices[-1],
                    take_profit=sol_prices[-1],
                    metadata={'reason': 'entropy_normalizing'}
                )
        
        return None


class VarianceRiskPremiumStrategy(BaseStrategy):
    """
    Strategy 4: Variance Risk Premium (Skew Harvesting)
    ===================================================
    
    MATHEMATICAL FRAMEWORK:
    -----------------------
    
    Variance Risk Premium (VRP):
        VRP = E[RV] - IV
        
    Where:
        RV = Realized Volatility (actual historical vol)
        IV = Implied Volatility (from options)
    
    In crypto (no liquid options), we proxy with:
        Pseudo_IV = ATR-based volatility × Fear_Multiplier
        
    Fear_Multiplier sources:
        - Funding rate extremes
        - Liquidation intensity
        - Put-call proxy from perp-spot basis
    
    Strategy:
        When VRP is very NEGATIVE (IV >> RV), fear is overpriced.
        Sell the fear by:
        1. Shorting volatility (short straddle proxy)
        2. Or: Fade extreme moves (mean reversion bet)
    
    Signal:
        When Pseudo_VRP < -2σ, fear is overpriced -> position for reversion
    """
    
    def __init__(self):
        super().__init__(
            strategy_id=4,
            name="VarianceRiskPremiumStrategy",
            regime=Regime.BEAR,
            lookback=200
        )
        
        self.vrp_threshold = -2.0  # Standard deviations
        self.atr_period = 14
        
        # Buffers
        self.rv_buffer: Dict[str, deque] = {
            'BTC': deque(maxlen=100),
            'ETH': deque(maxlen=100),
            'SOL': deque(maxlen=100),
        }
        self.pseudo_iv_buffer: Dict[str, deque] = {
            'BTC': deque(maxlen=100),
            'ETH': deque(maxlen=100),
            'SOL': deque(maxlen=100),
        }
        self.vrp_buffer: Dict[str, deque] = {
            'BTC': deque(maxlen=100),
            'ETH': deque(maxlen=100),
            'SOL': deque(maxlen=100),
        }
        
    def calculate_realized_volatility(self, prices: np.ndarray, period: int = 20) -> float:
        """
        Calculate realized volatility (annualized).
        
        RV = σ(log returns) × √(365 × 24)  # For hourly data
        """
        if len(prices) < period:
            return 0.0
        
        returns = np.diff(np.log(prices[-period:]))
        rv = np.std(returns) * np.sqrt(365 * 24)  # Annualize
        
        return rv
    
    def calculate_pseudo_implied_vol(
        self,
        prices: np.ndarray,
        funding_rate: float,
        liquidation_intensity: float
    ) -> float:
        """
        Calculate pseudo-implied volatility using fear proxies.
        
        Since Kraken doesn't have liquid options, we estimate IV from:
        1. ATR-based volatility (base)
        2. Funding rate extremes (fear premium)
        3. Liquidation intensity (panic premium)
        """
        if len(prices) < self.atr_period:
            return 0.0
        
        # Base: ATR-like volatility
        high_low_range = np.max(prices[-self.atr_period:]) - np.min(prices[-self.atr_period:])
        base_vol = (high_low_range / np.mean(prices[-self.atr_period:])) * np.sqrt(365 * 24 / self.atr_period)
        
        # Fear multiplier from funding rate
        # Negative funding = bearish, implies puts are expensive
        funding_fear = 1.0 + max(0, -funding_rate * 100)  # Scale funding rate
        
        # Panic multiplier from liquidations
        liq_fear = 1.0 + min(1.0, liquidation_intensity / 3)  # Cap at 2x
        
        # Pseudo IV
        pseudo_iv = base_vol * funding_fear * liq_fear
        
        return pseudo_iv
    
    def update(self, market_data: Dict) -> Optional[Signal]:
        """
        Calculate VRP and generate signals on extreme fear.
        """
        timestamp = market_data.get('timestamp', time.time())
        
        # Update price buffers
        for asset in ['BTC', 'ETH', 'SOL']:
            if asset in market_data.get('prices', {}):
                self.price_buffer[asset].append(market_data['prices'][asset])
        
        # Need enough data
        if len(self.price_buffer['SOL']) < 50:
            return None
        
        # Calculate metrics for SOL
        sol_prices = np.array(self.price_buffer['SOL'])
        
        # Realized vol
        rv = self.calculate_realized_volatility(sol_prices)
        self.rv_buffer['SOL'].append(rv)
        
        # Pseudo IV
        funding_rate = market_data.get('funding_rate', {}).get('SOL', 0)
        liq_intensity = market_data.get('liquidation_intensity', {}).get('SOL', 0)
        
        pseudo_iv = self.calculate_pseudo_implied_vol(sol_prices, funding_rate, liq_intensity)
        self.pseudo_iv_buffer['SOL'].append(pseudo_iv)
        
        # VRP = RV - IV (negative = fear overpriced)
        vrp = rv - pseudo_iv
        self.vrp_buffer['SOL'].append(vrp)
        
        # Z-score of VRP
        if len(self.vrp_buffer['SOL']) < 20:
            return None
        
        vrp_array = np.array(self.vrp_buffer['SOL'])
        vrp_mean = np.mean(vrp_array)
        vrp_std = np.std(vrp_array) + 1e-10
        vrp_zscore = (vrp - vrp_mean) / vrp_std
        
        # Signal: Extreme negative VRP (fear overpriced)
        # In bear market, we don't fade the fear, we JOIN it but with better timing
        # Wait for VRP to become LESS negative (fear subsiding) for short entry
        
        if vrp_zscore < self.vrp_threshold and not self.state.in_position:
            # Fear is maximum - wait for it to subside slightly then short
            # This is a "fade the bounce" strategy
            pass
        
        # Alternative: When VRP starts normalizing from extreme, enter short
        # because the "dead cat bounce" is ending
        vrp_rate = (vrp_array[-1] - vrp_array[-5]) / (abs(vrp_array[-5]) + 1e-10) if len(vrp_array) >= 5 else 0
        
        if (vrp_zscore < -1.5 and  # Still fearful
            vrp_rate > 0.1 and     # Fear subsiding
            not self.state.in_position):
            
            self.state.in_position = True
            self.state.position_side = 'short'
            
            current_price = sol_prices[-1]
            
            return Signal(
                timestamp=timestamp,
                strategy_id=self.strategy_id,
                strategy_name=self.name,
                regime=self.regime,
                asset="SOL",
                signal_type=SignalType.SHORT,
                size_pct=0.5,
                confidence=0.6,
                entry_price=current_price,
                stop_loss=self._dynamic_stop(current_price, "short", "SOL"),  # [UPG6]
                take_profit=current_price * 0.94,
                metadata={
                    'realized_vol': rv,
                    'pseudo_iv': pseudo_iv,
                    'vrp': vrp,
                    'vrp_zscore': vrp_zscore,
                    'vrp_rate': vrp_rate,
                }
            )
        
        return None


# =============================================================================
# REGIME B: BULL MARKET STRATEGIES (Institutional Flow)
# =============================================================================

class FundingDivergenceStrategy(BaseStrategy):
    """
    Strategy 5: Funding Rate Divergence
    ====================================

    Uses Kraken Futures Funding Rate Index (KFRI).

    Logic (two-sided detection after [PORT 2026-04-22]):
      BULLISH (original): rising funding + flat price + 4H high breakout
                          → aggressive longs entering → momentum squeeze
      BEARISH (ported from quant_agent.py:FundingDivergence):
          price_momentum > -1.0 (flat/rising)
          AND funding_momentum < -1.5σ (funding collapsing)
          AND oi_growth > 5% (smart money building shorts)
          AND predicted_funding < funding_rate (forward curve bearish)
          → smart-money short build-up → spot eventually dumps
    """

    def __init__(self):
        super().__init__(
            strategy_id=5,
            name="FundingDivergenceStrategy",
            regime=Regime.BULL,
            lookback=200
        )

        self.funding_buffer: Dict[str, deque] = {
            'BTC': deque(maxlen=100),
            'ETH': deque(maxlen=100),
            'SOL': deque(maxlen=100),
        }
        self.high_4h: Dict[str, float] = {'BTC': 0, 'ETH': 0, 'SOL': 0}
        # [PORT 2026-04-22] Buffers for bearish-divergence detection
        self.oi_buffer_fd: Dict[str, deque] = {
            'BTC': deque(maxlen=48),
            'ETH': deque(maxlen=48),
            'SOL': deque(maxlen=48),
        }
        # Bearish divergence thresholds (from quant_agent.py)
        self.bearish_funding_mom_thresh = -1.5       # σ
        self.bearish_oi_growth_thresh = 0.05         # +5% OI
        self.bearish_price_mom_min = -1.0            # σ (price NOT crashing yet)

    def update(self, market_data: Dict) -> Optional[Signal]:
        """Detect funding divergence for momentum entries (bullish OR bearish)."""
        timestamp = market_data.get('timestamp', time.time())

        # Update buffers
        for asset in ['BTC', 'ETH', 'SOL']:
            if asset in market_data.get('prices', {}):
                self.price_buffer[asset].append(market_data['prices'][asset])
            if asset in market_data.get('funding_rate', {}):
                self.funding_buffer[asset].append(market_data['funding_rate'][asset])
            # [PORT 2026-04-22] OI for bearish-divergence branch
            if asset in market_data.get('open_interest', {}):
                self.oi_buffer_fd[asset].append(market_data['open_interest'][asset])

        # Track 4H high (simplified: rolling 240-minute high)
        for asset in ['BTC', 'ETH', 'SOL']:
            if len(self.price_buffer[asset]) >= 240:
                self.high_4h[asset] = max(list(self.price_buffer[asset])[-240:])

        if len(self.funding_buffer['SOL']) < 20:
            return None

        # Calculate funding trend (used by both branches)
        sol_funding = np.array(self.funding_buffer['SOL'])
        funding_trend = (sol_funding[-1] - sol_funding[-10]) if len(sol_funding) >= 10 else 0

        # Price trend
        sol_prices = np.array(self.price_buffer['SOL'])
        price_trend = (sol_prices[-1] - sol_prices[-10]) / sol_prices[-10] if len(sol_prices) >= 10 else 0

        current_price = sol_prices[-1]

        # =====================================================================
        # [PORT 2026-04-22] BEARISH divergence branch — checked FIRST because
        # smart-money short-build signal is higher conviction than breakout chase.
        # =====================================================================
        predicted_funding = market_data.get('predicted_funding', {}).get('SOL')
        sol_oi = list(self.oi_buffer_fd['SOL'])
        if (
            predicted_funding is not None
            and len(sol_oi) >= 24
            and not self.state.in_position
        ):
            # Funding momentum in σ units (vs full-buffer std)
            funding_std = np.std(sol_funding) + 1e-10
            funding_mom_sigma = (sol_funding[-1] - sol_funding[0]) / funding_std

            # Price momentum in σ units (vs return std × √N)
            price_returns = np.diff(np.log(sol_prices))
            price_vol = np.std(price_returns) + 1e-10
            price_mom_sigma = (
                (sol_prices[-1] - sol_prices[0])
                / (price_vol * np.sqrt(len(sol_prices)))
            )

            # OI growth
            oi_growth = (sol_oi[-1] - sol_oi[0]) / (sol_oi[0] + 1e-10)

            bearish_conditions = [
                price_mom_sigma > self.bearish_price_mom_min,
                funding_mom_sigma < self.bearish_funding_mom_thresh,
                oi_growth > self.bearish_oi_growth_thresh,
                predicted_funding < sol_funding[-1],   # forward curve lower
            ]
            if sum(bearish_conditions) >= 4:
                self.state.in_position = True
                self.state.position_side = 'short'
                return Signal(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    strategy_name=self.name,
                    regime=self.regime,
                    asset="SOL",
                    signal_type=SignalType.AGGRESSIVE_SHORT,
                    size_pct=0.8,
                    confidence=0.78,
                    entry_price=current_price,
                    stop_loss=current_price * 1.03,
                    take_profit=current_price * 0.94,
                    metadata={
                        'branch': 'bearish_divergence_ported',
                        'funding_mom_sigma': funding_mom_sigma,
                        'price_mom_sigma': price_mom_sigma,
                        'oi_growth': oi_growth,
                        'predicted_funding': predicted_funding,
                        'current_funding': sol_funding[-1],
                    }
                )

        # =====================================================================
        # BULLISH divergence branch (original kraken_quant logic)
        # =====================================================================
        bullish_divergence = (
            funding_trend > 0.00002 and  # [FIX-AG2] Kraken 8h typical delta
            abs(price_trend) < 0.02 and
            current_price > self.high_4h.get('SOL', 0) * 0.99
        )
        breakout = current_price > self.high_4h.get('SOL', 0)

        if bullish_divergence and breakout and not self.state.in_position:
            self.state.in_position = True
            self.state.position_side = 'long'

            return Signal(
                timestamp=timestamp,
                strategy_id=self.strategy_id,
                strategy_name=self.name,
                regime=self.regime,
                asset="SOL",
                signal_type=SignalType.AGGRESSIVE_LONG,
                size_pct=0.8,
                confidence=0.75,
                entry_price=current_price,
                stop_loss=current_price * 0.97,
                take_profit=current_price * 1.06,
                metadata={
                    'branch': 'bullish_divergence_original',
                    'funding_trend': funding_trend,
                    'price_trend': price_trend,
                    'high_4h': self.high_4h.get('SOL', 0),
                }
            )

        return None


class ETFSpotCointegrationStrategy(BaseStrategy):
    """
    Strategy 6: ETF-Spot Cointegration
    ==================================
    
    Price discovery from US Spot ETF flows.
    Requires external ETF flow data.
    """
    
    def __init__(self):
        super().__init__(
            strategy_id=6,
            name="ETFSpotCointegration",
            regime=Regime.BULL,
            lookback=100
        )
        
        self.etf_flow_buffer = deque(maxlen=50)  # Daily ETF flows
        
    def update(self, market_data: Dict) -> Optional[Signal]:
        """
        Track ETF flows for price discovery.
        
        Uses price momentum as proxy when real ETF data unavailable.
        Real implementation would integrate with ETF flow APIs (e.g., Bitwise, Grayscale).
        """
        try:
            # [FIX-AG3] Was market_data['BTC']['close'] (nested) but market_data is flat per-asset dict
            btc_close = market_data.get('close', market_data.get('current_price', 0))
            btc_open = market_data.get('open', btc_close)
            btc_volume = market_data.get('volume_24h', market_data.get('volume', 0))
            
            if btc_close == 0:
                return None
            
            # Store price change as ETF flow proxy
            price_change_pct = (btc_close - btc_open) / btc_open if btc_open > 0 else 0
            volume_normalized = btc_volume / 1e9  # Normalize to billions
            
            # Simulate ETF flow based on price action (positive price = inflows)
            estimated_flow = price_change_pct * volume_normalized * 100  # Millions USD
            self.etf_flow_buffer.append(estimated_flow)
            
            if len(self.etf_flow_buffer) < 5:
                return None
            
            # Calculate cumulative flow over recent period
            recent_flows = list(self.etf_flow_buffer)[-5:]
            cumulative_flow = sum(recent_flows)
            flow_momentum = recent_flows[-1] - recent_flows[0] if len(recent_flows) >= 2 else 0
            
            # Generate signal based on flow patterns
            # Strong inflows (> $100M cumulative) = bullish
            # Strong outflows (< -$100M) = bearish
            if cumulative_flow > 100:
                direction = 1.0
                confidence = min(0.7, cumulative_flow / 500)
            elif cumulative_flow < -100:
                direction = -1.0
                confidence = min(0.7, abs(cumulative_flow) / 500)
            else:
                # No strong signal
                return None
            
            # Boost confidence if flow momentum confirms
            if (direction > 0 and flow_momentum > 0) or (direction < 0 and flow_momentum < 0):
                confidence = min(0.85, confidence + 0.15)
            
            signal_type = SignalType.LONG if direction > 0 else SignalType.SHORT
            return Signal(
                timestamp=time.time(),
                strategy_id=self.strategy_id,
                strategy_name=self.name,
                regime=self.regime,
                asset="BTC",
                signal_type=signal_type,
                size_pct=0.5,
                confidence=confidence,
                entry_price=btc_close,
                stop_loss=btc_close * (1 - 0.02 * abs(direction)),
                take_profit=btc_close * (1 + 0.04 * abs(direction)),
                metadata={
                    'direction': direction,
                    'cumulative_flow': cumulative_flow,
                    'flow_momentum': flow_momentum,
                }
            )
            
        except Exception as e:
            logger.debug(f"ETF strategy error: {e}")
            return None


class RelativeStrengthStrategy(BaseStrategy):
    """
    Strategy 7: SOL/BTC Relative Strength
    =====================================
    
    Captures SOL outperformance during high-liquidity bull phases.
    
    Math:
        RS = (SOL_return / BTC_return) over rolling window
        
    Signal: Long SOL when RS > 1.2 and trending up
    """
    
    def __init__(self):
        super().__init__(
            strategy_id=7,
            name="RelativeStrengthStrategy",
            regime=Regime.BULL,
            lookback=200
        )
        
        self.rs_buffer = deque(maxlen=100)
        self.rs_threshold = 1.2
        
    def update(self, market_data: Dict) -> Optional[Signal]:
        """Calculate relative strength and generate signals."""
        timestamp = market_data.get('timestamp', time.time())
        
        for asset in ['BTC', 'SOL']:
            if asset in market_data.get('prices', {}):
                self.price_buffer[asset].append(market_data['prices'][asset])
        
        if len(self.price_buffer['SOL']) < 50 or len(self.price_buffer['BTC']) < 50:
            return None
        
        sol_prices = np.array(self.price_buffer['SOL'])
        btc_prices = np.array(self.price_buffer['BTC'])
        
        # Calculate returns
        sol_return = (sol_prices[-1] - sol_prices[-20]) / sol_prices[-20]
        btc_return = (btc_prices[-1] - btc_prices[-20]) / btc_prices[-20]
        
        rs = sol_return / (btc_return + 1e-10) if btc_return != 0 else 1.0
        self.rs_buffer.append(rs)
        
        # RS trend
        rs_array = np.array(self.rs_buffer)
        rs_trend = (rs_array[-1] - rs_array[-5]) if len(rs_array) >= 5 else 0
        
        if rs > self.rs_threshold and rs_trend > 0 and not self.state.in_position:
            self.state.in_position = True
            self.state.position_side = 'long'
            
            return Signal(
                timestamp=timestamp,
                strategy_id=self.strategy_id,
                strategy_name=self.name,
                regime=self.regime,
                asset="SOL",
                signal_type=SignalType.LONG,
                size_pct=0.6,
                confidence=min(0.8, 0.5 + rs * 0.2),
                entry_price=sol_prices[-1],
                stop_loss=sol_prices[-1] * 0.96,
                take_profit=sol_prices[-1] * 1.08,
                metadata={'relative_strength': rs, 'rs_trend': rs_trend}
            )
        
        return None


class OrderBookImbalanceStrategy(BaseStrategy):
    """
    Strategy 8: Order Book Imbalance (OBM)
    ======================================
    
    High-frequency analysis of Kraken L2 book.
    
    Math:
        OBM = (Bid_Volume - Ask_Volume) / (Bid_Volume + Ask_Volume)
        
    Predicts short-term price direction.
    """
    
    def __init__(self):
        super().__init__(
            strategy_id=8,
            name="OrderBookImbalance",
            regime=Regime.BULL,
            lookback=100
        )
        
        self.obm_buffer: Dict[str, deque] = {
            'BTC': deque(maxlen=100),
            'ETH': deque(maxlen=100),
            'SOL': deque(maxlen=100),
        }
        self.obm_threshold = 0.3
        
    def update(self, market_data: Dict) -> Optional[Signal]:
        """Calculate OBM and generate short-term signals."""
        timestamp = market_data.get('timestamp', time.time())
        
        for asset in ['BTC', 'ETH', 'SOL']:
            if asset in market_data.get('prices', {}):
                self.price_buffer[asset].append(market_data['prices'][asset])
            
            bid_depth = market_data.get('bid_depth', {}).get(asset, 0)
            ask_depth = market_data.get('ask_depth', {}).get(asset, 0)
            
            if bid_depth + ask_depth > 0:
                obm = (bid_depth - ask_depth) / (bid_depth + ask_depth)
                self.obm_buffer[asset].append(obm)
        
        if len(self.obm_buffer['SOL']) < 10:
            return None
        
        sol_obm = np.mean(list(self.obm_buffer['SOL'])[-5:])
        
        if sol_obm > self.obm_threshold and not self.state.in_position:
            self.state.in_position = True
            
            sol_price = list(self.price_buffer['SOL'])[-1]
            
            return Signal(
                timestamp=timestamp,
                strategy_id=self.strategy_id,
                strategy_name=self.name,
                regime=self.regime,
                asset="SOL",
                signal_type=SignalType.LONG,
                size_pct=0.3,  # Small size for HF
                confidence=0.6,
                entry_price=sol_price,
                stop_loss=sol_price * 0.995,
                take_profit=sol_price * 1.005,
                metadata={'obm': sol_obm}
            )
        
        return None


# =============================================================================
# REGIME C: SIDEWAYS STRATEGIES (Neutral Extraction)
# =============================================================================

class KalmanCointegrationStrategy(BaseStrategy):
    """
    Strategy 9: Kalman Filter Cointegration (Pairs Arb)
    ===================================================
    
    MATHEMATICAL FRAMEWORK:
    -----------------------
    
    State-Space Model:
        Observation: y_t = β_t × x_t + α_t + ε_t
        State:       θ_t = [β_t, α_t]' = θ_{t-1} + w_t
    
    Where:
        y_t = log(SOL) or log(ETH)
        x_t = log(BTC)
        β_t = Time-varying hedge ratio
        α_t = Time-varying intercept
    
    KALMAN FILTER EQUATIONS:
    ------------------------
    
    PREDICT:
        θ̂_{t|t-1} = θ̂_{t-1|t-1}
        P_{t|t-1} = P_{t-1|t-1} + Q
    
    UPDATE:
        H_t = [x_t, 1]
        v_t = y_t - H_t × θ̂_{t|t-1}           (Innovation)
        S_t = H_t × P_{t|t-1} × H_t' + R      (Innovation covariance)
        K_t = P_{t|t-1} × H_t' × S_t^{-1}     (Kalman gain)
        θ̂_{t|t} = θ̂_{t|t-1} + K_t × v_t     (State update)
        P_{t|t} = (I - K_t × H_t) × P_{t|t-1} (Covariance update)
    
    SPREAD Z-SCORE:
        spread_t = y_t - β̂_t × x_t - α̂_t
        z_t = (spread_t - μ_spread) / σ_spread
    
    TRADING RULES:
        LONG A / SHORT B when z < -2σ
        SHORT A / LONG B when z > +2σ
        EXIT when |z| < 0.5σ
    """
    
    def __init__(self, pair: Tuple[str, str] = ('SOL', 'ETH')):
        super().__init__(
            strategy_id=9,
            name=f"KalmanCointegration_{pair[0]}_{pair[1]}",
            regime=Regime.SIDEWAYS,
            lookback=500
        )
        
        self.asset_a, self.asset_b = pair
        
        # Kalman parameters
        self.delta = 1e-5   # Process noise (Q)
        self.R = 1e-3       # Observation noise
        
        # State initialization
        self.theta = np.array([1.0, 0.0])  # [beta, alpha]
        self.P = np.eye(2) * 1.0           # State covariance
        self.Q = np.eye(2) * self.delta    # Process noise covariance
        
        # Spread tracking
        self.spread_buffer = deque(maxlen=200)
        self.entry_zscore = 2.0
        self.exit_zscore = 0.5
        
    def kalman_update(self, y: float, x: float) -> Tuple[np.ndarray, float, float]:
        """
        Single Kalman filter update step.
        
        Args:
            y: Dependent variable (log price of asset A)
            x: Independent variable (log price of asset B)
            
        Returns:
            (updated_state, spread, innovation_variance)
        """
        # PREDICT STEP
        theta_pred = self.theta.copy()
        P_pred = self.P + self.Q
        
        # Observation matrix H = [x, 1]
        H = np.array([x, 1.0])
        
        # Predicted observation
        y_pred = H @ theta_pred
        
        # UPDATE STEP
        # Innovation (prediction error)
        innovation = y - y_pred
        
        # Innovation covariance: S = H @ P @ H' + R
        S = H @ P_pred @ H.T + self.R
        
        # Kalman gain: K = P @ H' @ S^{-1}
        K = (P_pred @ H.T) / S
        
        # State update: θ = θ_pred + K * innovation
        self.theta = theta_pred + K * innovation
        
        # Covariance update: P = (I - K @ H) @ P_pred
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred
        
        # Ensure positive definiteness
        self.P = (self.P + self.P.T) / 2
        self.P += np.eye(2) * 1e-8
        
        # Spread = actual - predicted
        spread = innovation
        
        return self.theta, spread, S
    
    def update(self, market_data: Dict) -> Optional[Signal]:
        """
        Update Kalman filter and generate pair trading signals.
        """
        timestamp = market_data.get('timestamp', time.time())
        
        # Update price buffers
        for asset in [self.asset_a, self.asset_b]:
            if asset in market_data.get('prices', {}):
                self.price_buffer[asset].append(market_data['prices'][asset])
        
        if (len(self.price_buffer[self.asset_a]) < 50 or 
            len(self.price_buffer[self.asset_b]) < 50):
            return None
        
        # Get log prices
        # [P39 2026-04-24] Guard against feed glitches sending price=0 or
        # negative — np.log(0)=-inf, np.log(-1)=NaN, both poison the Kalman
        # spread estimator and propagate NaN through downstream signals.
        _pa = self.price_buffer[self.asset_a][-1]
        _pb = self.price_buffer[self.asset_b][-1]
        if _pa <= 0 or _pb <= 0:
            return None
        y = np.log(_pa)
        x = np.log(_pb)
        
        # Kalman update
        theta, spread, S = self.kalman_update(y, x)
        beta, alpha = theta
        
        # Store spread
        self.spread_buffer.append(spread)
        
        if len(self.spread_buffer) < 30:
            return None
        
        # Calculate z-score
        spread_array = np.array(self.spread_buffer)
        spread_mean = np.mean(spread_array)
        spread_std = np.std(spread_array) + 1e-10
        zscore = (spread - spread_mean) / spread_std
        
        # Get current prices
        price_a = self.price_buffer[self.asset_a][-1]
        price_b = self.price_buffer[self.asset_b][-1]
        
        # Entry signals
        if not self.state.in_position:
            if zscore > self.entry_zscore:
                # A is overvalued relative to B: SHORT A, LONG B
                self.state.in_position = True
                self.state.position_side = 'short_a'
                
                return Signal(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    strategy_name=self.name,
                    regime=self.regime,
                    asset=f"{self.asset_a}-{self.asset_b}",
                    signal_type=SignalType.PAIR_SHORT_A_LONG_B,
                    size_pct=min(1.0, 0.3 + abs(zscore) * 0.2),
                    confidence=min(0.85, 0.5 + abs(zscore) * 0.15),
                    entry_price=price_a,
                    stop_loss=price_a * 1.03,
                    take_profit=price_a * 0.97,
                    metadata={
                        'beta': beta,
                        'alpha': alpha,
                        'zscore': zscore,
                        'spread': spread,
                        'hedge_ratio': beta,
                        'price_b': price_b,
                    }
                )
            
            elif zscore < -self.entry_zscore:
                # A is undervalued: LONG A, SHORT B
                self.state.in_position = True
                self.state.position_side = 'long_a'
                
                return Signal(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    strategy_name=self.name,
                    regime=self.regime,
                    asset=f"{self.asset_a}-{self.asset_b}",
                    signal_type=SignalType.PAIR_LONG_A_SHORT_B,
                    size_pct=min(1.0, 0.3 + abs(zscore) * 0.2),
                    confidence=min(0.85, 0.5 + abs(zscore) * 0.15),
                    entry_price=price_a,
                    stop_loss=price_a * 0.97,
                    take_profit=price_a * 1.03,
                    metadata={
                        'beta': beta,
                        'alpha': alpha,
                        'zscore': zscore,
                        'spread': spread,
                        'hedge_ratio': beta,
                        'price_b': price_b,
                    }
                )
        
        # Exit signals
        elif self.state.in_position:
            if abs(zscore) < self.exit_zscore:
                self.state.in_position = False
                
                return Signal(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    strategy_name=self.name,
                    regime=self.regime,
                    asset=f"{self.asset_a}-{self.asset_b}",
                    signal_type=SignalType.FLATTEN,
                    size_pct=1.0,
                    confidence=0.8,
                    entry_price=price_a,
                    stop_loss=price_a,
                    take_profit=price_a,
                    metadata={'exit_zscore': zscore, 'reason': 'mean_reversion'}
                )
        
        return None
    
    def get_hedge_ratio(self) -> float:
        """Return current Kalman-estimated hedge ratio."""
        return self.theta[0]


class OrnsteinUhlenbeckStrategy(BaseStrategy):
    """
    Strategy 10: Ornstein-Uhlenbeck Mean Reversion
    ==============================================
    
    MATHEMATICAL FRAMEWORK:
    -----------------------
    
    The OU Process:
        dX_t = θ(μ - X_t)dt + σdW_t
    
    Where:
        θ = Mean reversion speed
        μ = Long-term mean (equilibrium)
        σ = Volatility
        W_t = Wiener process
    
    Discretized form:
        X_{t+1} - X_t = θ(μ - X_t)Δt + σ√Δt × ε
        
    Or equivalently:
        X_{t+1} = a + b × X_t + ε
        
        where: a = θμΔt, b = 1 - θΔt
    
    MLE ESTIMATION:
    ---------------
    Given observations {x_0, x_1, ..., x_n}:
    
    θ̂ = -ln(b̂) / Δt
    μ̂ = â / (1 - b̂)
    σ̂² = 2θ̂ × Var(residuals) / (1 - b̂²)
    
    Half-life = ln(2) / θ
    
    TRADING SIGNAL:
    ---------------
    Z-score: z = (X_t - μ̂) / σ̂
    
    LONG when z < -2
    SHORT when z > +2
    EXIT when |z| < 0.5
    """
    
    def __init__(self):
        super().__init__(
            strategy_id=10,
            name="OrnsteinUhlenbeckStrategy",
            regime=Regime.SIDEWAYS,
            lookback=500
        )
        
        # OU parameters (estimated)
        self.theta = 0.0  # Mean reversion speed
        self.mu = 0.0     # Long-term mean
        self.sigma = 0.0  # Volatility
        
        # Trading thresholds
        self.entry_z = 2.0
        self.exit_z = 0.5
        
        # Buffer for spread/ratio
        self.ratio_buffer = deque(maxlen=500)
        
    def estimate_ou_parameters(self, x: np.ndarray, dt: float = 1.0) -> Tuple[float, float, float, float]:
        """
        Estimate OU parameters using MLE.
        
        Returns: (theta, mu, sigma, half_life)
        """
        if len(x) < 50:
            return 0.0, np.mean(x), np.std(x), np.inf
        
        # AR(1) regression: x_{t+1} = a + b × x_t + ε
        x_lag = x[:-1]
        x_curr = x[1:]
        
        # OLS estimation
        n = len(x_lag)
        sum_x = np.sum(x_lag)
        sum_y = np.sum(x_curr)
        sum_xy = np.sum(x_lag * x_curr)
        sum_x2 = np.sum(x_lag ** 2)
        
        b_hat = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2 + 1e-10)
        a_hat = (sum_y - b_hat * sum_x) / n
        
        # Bound b to prevent numerical issues
        b_hat = np.clip(b_hat, 0.001, 0.999)
        
        # Convert to OU parameters
        theta_hat = -np.log(b_hat) / dt
        mu_hat = a_hat / (1 - b_hat)
        
        # Estimate sigma from residuals
        residuals = x_curr - (a_hat + b_hat * x_lag)
        var_residuals = np.var(residuals)
        sigma_hat = np.sqrt(2 * theta_hat * var_residuals / (1 - b_hat ** 2 + 1e-10))
        
        # Half-life
        half_life = np.log(2) / (theta_hat + 1e-10)
        
        return theta_hat, mu_hat, sigma_hat, half_life
    
    def update(self, market_data: Dict) -> Optional[Signal]:
        """
        Estimate OU parameters and trade mean reversion.
        """
        timestamp = market_data.get('timestamp', time.time())
        
        # Update price buffers
        for asset in ['BTC', 'ETH', 'SOL']:
            if asset in market_data.get('prices', {}):
                self.price_buffer[asset].append(market_data['prices'][asset])
        
        if len(self.price_buffer['SOL']) < 100 or len(self.price_buffer['ETH']) < 100:
            return None
        
        # Calculate log ratio (spread proxy)
        sol_price = self.price_buffer['SOL'][-1]
        eth_price = self.price_buffer['ETH'][-1]
        # P75: guard against feed glitch (price=0 or negative) — same shape
        # as the P39 Kalman fix at line 1659. Without this guard,
        # np.log(sol/eth) can produce inf or nan and poison ratio_buffer →
        # estimate_ou_parameters() → z-score → downstream signals until
        # restart. eth_price=0 also raises ZeroDivisionError on the inner
        # division before np.log even runs.
        if sol_price <= 0 or eth_price <= 0:
            return None

        ratio = np.log(sol_price / eth_price)
        self.ratio_buffer.append(ratio)
        
        if len(self.ratio_buffer) < 100:
            return None
        
        # Estimate OU parameters
        ratio_array = np.array(self.ratio_buffer)
        self.theta, self.mu, self.sigma, half_life = self.estimate_ou_parameters(ratio_array)
        
        # Z-score
        z = (ratio - self.mu) / (self.sigma + 1e-10)
        
        # Only trade if mean reversion is fast enough (half-life < 50 periods)
        if half_life > 50:
            return None
        
        # Entry signals
        if not self.state.in_position:
            if z > self.entry_z:
                # Ratio too high: SHORT SOL, LONG ETH
                self.state.in_position = True
                self.state.position_side = 'short_sol'
                
                return Signal(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    strategy_name=self.name,
                    regime=self.regime,
                    asset="SOL-ETH",
                    signal_type=SignalType.PAIR_SHORT_A_LONG_B,
                    size_pct=min(1.0, 0.3 + abs(z) * 0.15),
                    confidence=min(0.8, 0.5 + (1 / half_life) * 10),
                    entry_price=sol_price,
                    stop_loss=self._dynamic_stop(sol_price, "short", "SOL"),  # [UPG6]
                    take_profit=sol_price * 0.975,
                    metadata={
                        'theta': self.theta,
                        'mu': self.mu,
                        'sigma': self.sigma,
                        'half_life': half_life,
                        'zscore': z,
                        'ratio': ratio,
                    }
                )
            
            elif z < -self.entry_z:
                # Ratio too low: LONG SOL, SHORT ETH
                self.state.in_position = True
                self.state.position_side = 'long_sol'
                
                return Signal(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    strategy_name=self.name,
                    regime=self.regime,
                    asset="SOL-ETH",
                    signal_type=SignalType.PAIR_LONG_A_SHORT_B,
                    size_pct=min(1.0, 0.3 + abs(z) * 0.15),
                    confidence=min(0.8, 0.5 + (1 / half_life) * 10),
                    entry_price=sol_price,
                    stop_loss=sol_price * 0.975,
                    take_profit=sol_price * 1.025,
                    metadata={
                        'theta': self.theta,
                        'mu': self.mu,
                        'sigma': self.sigma,
                        'half_life': half_life,
                        'zscore': z,
                        'ratio': ratio,
                    }
                )
        
        # Exit
        elif self.state.in_position:
            if abs(z) < self.exit_z:
                self.state.in_position = False
                
                return Signal(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    strategy_name=self.name,
                    regime=self.regime,
                    asset="SOL-ETH",
                    signal_type=SignalType.FLATTEN,
                    size_pct=1.0,
                    confidence=0.75,
                    entry_price=sol_price,
                    stop_loss=sol_price,
                    take_profit=sol_price,
                    metadata={'exit_z': z, 'reason': 'mean_reversion_complete'}
                )
        
        return None


class DarkPoolVolumeStrategy(BaseStrategy):
    """
    Strategy 11: Kraken Dark Pool Volume Tracking
    =============================================
    
    Identifies institutional trades on Kraken Pro.
    Large block trades often precede directional moves.
    """
    
    def __init__(self):
        super().__init__(
            strategy_id=11,
            name="DarkPoolVolumeStrategy",
            regime=Regime.SIDEWAYS,
            lookback=200
        )
        
        self.large_trade_buffer: Dict[str, deque] = {
            'BTC': deque(maxlen=100),
            'ETH': deque(maxlen=100),
            'SOL': deque(maxlen=100),
        }
        self.large_trade_threshold = 2.0  # Multiplier of avg trade size
        
    def update(self, market_data: Dict) -> Optional[Signal]:
        """
        Track large trades for institutional flow detection.
        
        Uses volume spike analysis as proxy for dark pool activity.
        Real implementation would use WebSocket trade feed for large trade detection.
        """
        try:
            signals = []
            
            for asset in ['BTC', 'ETH', 'SOL']:
                asset_data = market_data.get(asset, {})
                volume = asset_data.get('volume', 0)
                close = asset_data.get('close', 0)
                
                if volume == 0 or close == 0:
                    continue
                
                # Store volume data
                self.large_trade_buffer[asset].append(volume)
                
                if len(self.large_trade_buffer[asset]) < 20:
                    continue
                
                # Calculate volume statistics
                volumes = list(self.large_trade_buffer[asset])
                avg_volume = np.mean(volumes[:-1])  # Exclude current
                std_volume = np.std(volumes[:-1]) + 1e-10
                current_volume = volumes[-1]
                
                # Volume z-score
                vol_zscore = (current_volume - avg_volume) / std_volume
                
                # Detect large volume spikes (potential institutional activity)
                if vol_zscore > self.large_trade_threshold:
                    # Large volume spike - check price action for direction
                    price_change = (close - asset_data.get('open', close)) / close if close > 0 else 0
                    
                    # Positive price + high volume = bullish accumulation
                    # Negative price + high volume = distribution
                    if price_change > 0.001:  # 0.1% threshold
                        direction = 1.0
                        signal_type = "institutional_buy"
                    elif price_change < -0.001:
                        direction = -1.0
                        signal_type = "institutional_sell"
                    else:
                        continue  # Ambiguous
                    
                    confidence = min(0.75, 0.4 + vol_zscore * 0.1)
                    
                    st = SignalType.LONG if direction > 0 else SignalType.SHORT
                    signals.append(Signal(
                        timestamp=time.time(),
                        strategy_id=self.strategy_id,
                        strategy_name=self.name,
                        regime=self.regime,
                        asset=asset,
                        signal_type=st,
                        size_pct=0.3,
                        confidence=confidence,
                        entry_price=close,
                        stop_loss=close * (1 - 0.015 * abs(direction)),
                        take_profit=close * (1 + 0.03 * abs(direction)),
                        metadata={
                            'asset': asset,
                            'vol_zscore': vol_zscore,
                            'signal_type': signal_type,
                            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1.0
                        }
                    ))
            
            # Return strongest signal if any
            if signals:
                return max(signals, key=lambda s: s.confidence)
            
            return None
            
        except Exception as e:
            logger.debug(f"DarkPool strategy error: {e}")
            return None


class DeltaNeutralFundingStrategy(BaseStrategy):
    """
    Strategy 12: Delta-Neutral Funding Harvest
    ==========================================
    
    Collect funding rate risk-free during low-volatility periods.
    
    Position:
        LONG Spot + SHORT Perp (equal notional)
        
    PnL = Funding Rate × Notional (when funding > 0)
    
    Best during sideways markets with positive funding.
    """
    
    def __init__(self):
        super().__init__(
            strategy_id=12,
            name="DeltaNeutralFundingStrategy",
            regime=Regime.SIDEWAYS,
            lookback=100
        )
        
        self.min_funding_threshold = 0.0001  # 0.01% per 8h
        self.max_volatility = 0.02  # 2% daily vol
        
        self.volatility_buffer: Dict[str, deque] = {
            'BTC': deque(maxlen=50),
            'ETH': deque(maxlen=50),
            'SOL': deque(maxlen=50),
        }
        
    def update(self, market_data: Dict) -> Optional[Signal]:
        """Enter delta-neutral position when conditions are favorable."""
        timestamp = market_data.get('timestamp', time.time())
        
        # Update price buffers
        for asset in ['BTC', 'ETH', 'SOL']:
            if asset in market_data.get('prices', {}):
                self.price_buffer[asset].append(market_data['prices'][asset])
        
        if len(self.price_buffer['SOL']) < 30:
            return None
        
        # Calculate realized volatility
        sol_prices = np.array(self.price_buffer['SOL'])
        returns = np.diff(np.log(sol_prices[-30:]))
        realized_vol = np.std(returns) * np.sqrt(24)  # Daily vol
        
        self.volatility_buffer['SOL'].append(realized_vol)
        
        # Get funding rate
        funding = market_data.get('funding_rate', {}).get('SOL', 0)
        
        # Conditions for entry
        low_vol = realized_vol < self.max_volatility
        positive_funding = funding > self.min_funding_threshold
        
        if low_vol and positive_funding and not self.state.in_position:
            self.state.in_position = True
            
            sol_price = sol_prices[-1]
            
            # Annualized funding yield
            annual_yield = funding * 3 * 365  # 3 funding periods per day
            
            return Signal(
                timestamp=timestamp,
                strategy_id=self.strategy_id,
                strategy_name=self.name,
                regime=self.regime,
                asset="SOL",
                signal_type=SignalType.LONG,  # Delta-neutral: Long Spot + Short Perp
                size_pct=0.5,
                confidence=0.7,
                entry_price=sol_price,
                stop_loss=sol_price * 0.95,  # Wide stop for low-risk strategy
                take_profit=sol_price * 1.05,
                metadata={
                    'strategy_type': 'delta_neutral',
                    'funding_rate': funding,
                    'annualized_yield': annual_yield,
                    'realized_vol': realized_vol,
                }
            )
        
        # Exit if volatility increases
        if self.state.in_position:
            if realized_vol > self.max_volatility * 1.5:
                self.state.in_position = False
                
                return Signal(
                    timestamp=timestamp,
                    strategy_id=self.strategy_id,
                    strategy_name=self.name,
                    regime=self.regime,
                    asset="SOL",
                    signal_type=SignalType.FLATTEN,
                    size_pct=1.0,
                    confidence=0.8,
                    entry_price=sol_prices[-1],
                    stop_loss=sol_prices[-1],
                    take_profit=sol_prices[-1],
                    metadata={'reason': 'volatility_spike', 'vol': realized_vol}
                )
        
        return None


# =============================================================================
# STRATEGY ALLOCATOR (The Brain)
# =============================================================================

class StrategyAllocator:
    """
    The Allocator: Orchestrates all strategies based on regime.
    
    ALLOCATION LOGIC:
    -----------------
    
    1. Regime Selection: Upstream RegimeAgent provides current regime
    
    2. Information Ratio (IR) Softmax Weighting:
       
       w_i = exp(IR_i / τ) / Σ exp(IR_j / τ)
       
       where:
           IR_i = Information Ratio of strategy i (48h rolling)
           τ = Temperature (lower = more aggressive toward best)
    
    3. Volatility Target:
       If 1H realized vol > 150% of normal -> reduce all sizes by 50%
    
    4. Global Drawdown Kill-Switch:
       If portfolio drawdown > threshold -> flatten all
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        vol_reduction_threshold: float = 1.5,  # 150% of normal vol
        max_drawdown: float = 0.15,  # 15% kill switch
    ):
        self.temperature = temperature
        self.vol_reduction_threshold = vol_reduction_threshold
        self.max_drawdown = max_drawdown
        
        # Initialize all strategies
        self.strategies: Dict[Regime, List[BaseStrategy]] = {
            Regime.BEAR: [
                LiquidationCascadeHunter(),
                HurstExponentStrategy(),
                ShannonEntropyStrategy(),
                VarianceRiskPremiumStrategy(),
            ],
            Regime.BULL: [
                FundingDivergenceStrategy(),
                ETFSpotCointegrationStrategy(),
                RelativeStrengthStrategy(),
                OrderBookImbalanceStrategy(),
            ],
            Regime.SIDEWAYS: [
                KalmanCointegrationStrategy(pair=('SOL', 'ETH')),
                OrnsteinUhlenbeckStrategy(),
                DarkPoolVolumeStrategy(),
                DeltaNeutralFundingStrategy(),
            ],
        }
        
        # [UPG1] Portfolio state - read from env, not hardcoded
        import os
        _init_cap = float(os.environ.get('HMATS_INITIAL_CAPITAL', 10000))
        self.portfolio_value = _init_cap
        self.peak_value = _init_cap
        self.current_drawdown = 0.0
        
        # Volatility tracking
        self.vol_buffer = deque(maxlen=100)
        self.normal_vol = 0.02  # 2% daily vol baseline
        
        # Kill switch state
        self.killed = False

        # [KQ-DIAG] Per-strategy firing telemetry: name -> {attempts, fires}
        # attempts = how many times strategy.update() was invoked (regime-gated)
        # fires    = how many times it returned a non-None Signal
        from collections import Counter as _Counter
        self._strategy_attempts: _Counter = _Counter()
        self._strategy_fires: _Counter = _Counter()
        self._strategy_archived_skips: _Counter = _Counter()
        self._regime_ticks: _Counter = _Counter()  # per-regime tick count
        self._diag_started_at = time.time()

        # [v5.1 Phase 1] Load archive decisions. Fail-open: missing file = empty
        # set = all strategies active (current behavior preserved). Iron Law 6
        # enforced: count active strategies after archive set is applied.
        self._archived_strategies: Set[str] = self._load_archive_decisions()
        self._enforce_iron_law_6()

    def _load_archive_decisions(self) -> Set[str]:
        """[v5.1 Phase 1] Load archived-strategy names from
        configs/strategy_v5_1_decisions.json. Fail-open: any error → empty set
        (all strategies active, current behavior). Logged at WARNING per
        Iron Law 4.
        """
        path = Path(__file__).resolve().parents[1] / "configs" / "strategy_v5_1_decisions.json"
        archived: Set[str] = set()
        if not path.exists():
            logger.info(f"[KQ_ARCHIVE] decisions file absent ({path}) — all strategies active")
            return archived
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for name, cfg in (data.get("strategies") or {}).items():
                if cfg.get("archived", False):
                    archived.add(name)
            logger.info(f"[KQ_ARCHIVE] loaded {len(archived)} archived strategies: {sorted(archived)}")
        except Exception as e:
            logger.warning(
                f"[KQ_ARCHIVE] failed to parse {path.name}: "
                f"{type(e).__name__}: {e} — fail-open, all strategies active"
            )
        return archived

    def _enforce_iron_law_6(self) -> None:
        """[v5.1 Iron Law 6] After archive set is applied, ≥3 active strategies
        must remain. If violated, log CRITICAL and reset _archived_strategies
        to empty (fail-open: better to run all 12 than violate the law silently).
        """
        active_count = 0
        for regime, strategies in self.strategies.items():
            for s in strategies:
                if s.name not in self._archived_strategies:
                    active_count += 1
        if active_count < 3:
            logger.critical(
                f"[KQ_ARCHIVE] Iron Law 6 violation: only {active_count} active "
                f"strategies after archiving {sorted(self._archived_strategies)}. "
                f"Resetting to all-active (fail-open). Operator: review "
                f"configs/strategy_v5_1_decisions.json."
            )
            self._archived_strategies = set()
        else:
            logger.info(
                f"[KQ_ARCHIVE] Iron Law 6 OK: {active_count} active strategies "
                f"(min=3). Archived: {len(self._archived_strategies)}."
            )

    def calculate_ir_weights(self, regime: Regime) -> np.ndarray:
        """
        Calculate Information Ratio Softmax weights.
        
        w_i = exp(IR_i / τ) / Σ exp(IR_j / τ)
        """
        strategies = self.strategies[regime]
        irs = np.array([s.state.information_ratio for s in strategies])
        
        # Softmax with temperature
        exp_irs = np.exp(irs / self.temperature)
        weights = exp_irs / (np.sum(exp_irs) + 1e-10)
        
        return weights
    
    def calculate_vol_adjustment(self, current_vol: float) -> float:
        """
        Calculate position size adjustment based on volatility.
        
        Returns multiplier [0.5, 1.0]
        """
        vol_ratio = current_vol / (self.normal_vol + 1e-10)
        
        if vol_ratio > self.vol_reduction_threshold:
            return 0.5
        else:
            return 1.0
    
    def update_portfolio_state(self, pnl: float):
        """Update portfolio value and check drawdown."""
        self.portfolio_value += pnl
        
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        
        self.current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        
        # Check kill switch
        if self.current_drawdown > self.max_drawdown:
            self.killed = True
            # P75: was print() — wouldn't reach the operator's log aggregator
            # in containerized live mode (Docker stdout vs hmats logger
            # pipeline). Promoted to logger.warning so the kill-switch
            # event is auditable in the same place as every other risk
            # warning. Kill switch is rare so log rate isn't a concern.
            logger.warning(
                f"[KQ_KILL_SWITCH] activated: drawdown={self.current_drawdown:.2%} "
                f"> max={self.max_drawdown:.2%}"
            )
    
    def process_tick(
        self,
        regime: Regime,
        market_data: Dict,
    ) -> List[Signal]:
        """
        Process market tick and generate weighted signals.
        """
        # Check kill switch
        if self.killed:
            return [Signal(
                timestamp=time.time(),
                strategy_id=0,
                strategy_name="KILL_SWITCH",
                regime=regime,
                asset="ALL",
                signal_type=SignalType.FLATTEN,
                size_pct=1.0,
                confidence=1.0,
                entry_price=0,
                stop_loss=0,
                take_profit=0,
                metadata={'reason': 'drawdown_limit'}
            )]
        
        # Get active strategies for regime
        strategies = self.strategies[regime]

        # [KQ-DIAG] record which regime bucket ran this tick
        self._regime_ticks[regime.value] += 1

        # Calculate IR weights
        weights = self.calculate_ir_weights(regime)
        
        # Calculate vol adjustment
        prices = market_data.get('prices', {})
        if 'BTC' in prices:
            self.vol_buffer.append(prices['BTC'])
        
        if len(self.vol_buffer) > 20:
            returns = np.diff(np.log(np.array(self.vol_buffer)))
            current_vol = np.std(returns) * np.sqrt(24)
            vol_mult = self.calculate_vol_adjustment(current_vol)
        else:
            vol_mult = 1.0
        
        # Collect signals from all strategies
        signals = []
        
        for i, strategy in enumerate(strategies):
            # [v5.1 Phase 1] Archive gate: skip strategies marked archived
            # in configs/strategy_v5_1_decisions.json. Equivalent to returning
            # Signal.NEUTRAL — Iron Law 4 fail-closed (no signal = no vote).
            if strategy.name in self._archived_strategies:
                self._strategy_archived_skips[strategy.name] += 1
                continue

            # [KQ-DIAG] count attempt (every regime-gated update call)
            self._strategy_attempts[strategy.name] += 1
            signal = strategy.update(market_data)

            if signal:
                # [KQ-DIAG] count fire (strategy returned a real Signal)
                self._strategy_fires[strategy.name] += 1
                # Apply weight and vol adjustment
                adjusted_size = signal.size_pct * weights[i] * vol_mult
                
                adjusted_signal = Signal(
                    timestamp=signal.timestamp,
                    strategy_id=signal.strategy_id,
                    strategy_name=signal.strategy_name,
                    regime=signal.regime,
                    asset=signal.asset,
                    signal_type=signal.signal_type,
                    size_pct=adjusted_size,
                    confidence=signal.confidence,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    metadata={
                        **signal.metadata,
                        'ir_weight': weights[i],
                        'vol_multiplier': vol_mult,
                        'original_size': signal.size_pct,
                    }
                )
                
                signals.append(adjusted_signal)
        
        return signals
    
    def get_strategy_summary(self) -> Dict:
        """Get summary of all strategy states."""
        summary = {}

        for regime, strategies in self.strategies.items():
            weights = self.calculate_ir_weights(regime)

            summary[regime.value] = [
                {
                    'id': s.strategy_id,
                    'name': s.name,
                    'ir': s.state.information_ratio,
                    'weight': float(weights[i]),
                    'win_rate': s.state.win_rate,
                    'in_position': s.state.in_position,
                }
                for i, s in enumerate(strategies)
            ]

        return summary

    def get_firing_stats(self) -> Dict:
        """[KQ-DIAG] Return per-strategy firing statistics.

        Structure:
          {
            'uptime_sec': float,
            'regime_ticks': {'BEAR': int, 'BULL': int, 'SIDEWAYS': int},
            'by_regime': {
                'BEAR': [{'name', 'attempts', 'fires', 'fire_rate'}, ...],
                ...
            },
            'never_fired': [strategy_name, ...]
          }
        """
        uptime = time.time() - self._diag_started_at
        by_regime: Dict = {}
        never_fired = []
        for regime, strategies in self.strategies.items():
            rows = []
            for s in strategies:
                att = int(self._strategy_attempts.get(s.name, 0))
                fires = int(self._strategy_fires.get(s.name, 0))
                rate = (fires / att) if att > 0 else 0.0
                rows.append({
                    'name': s.name,
                    'id': s.strategy_id,
                    'attempts': att,
                    'fires': fires,
                    'fire_rate': round(rate, 4),
                })
                if fires == 0:
                    never_fired.append(s.name)
            by_regime[regime.value] = rows
        return {
            'uptime_sec': round(uptime, 1),
            'regime_ticks': dict(self._regime_ticks),
            'by_regime': by_regime,
            'never_fired': never_fired,
        }


# =============================================================================
# SIGNAL TYPE -> DIRECTION MAPPING
# =============================================================================

_SIGNAL_DIRECTION = {
    SignalType.NO_SIGNAL: 0.0,
    SignalType.AGGRESSIVE_LONG: 1.0,
    SignalType.LONG: 0.6,
    SignalType.REDUCE_LONG: -0.3,
    SignalType.FLATTEN: 0.0,
    SignalType.SHORT: -0.6,
    SignalType.AGGRESSIVE_SHORT: -1.0,
    SignalType.REDUCE_SHORT: 0.3,
    SignalType.PAIR_LONG_A_SHORT_B: 0.5,
    SignalType.PAIR_SHORT_A_LONG_B: -0.5,
}

# HMATS regime name -> local Regime enum
# Covers all 6 canonical GMM regimes + common synonyms + REGIME_N fallthrough
_REGIME_MAP = {
    # BULL family
    "MOMENTUM_RALLY": Regime.BULL,
    "MOMENTUM": Regime.BULL,
    "BULL": Regime.BULL,
    # BEAR family
    "PANIC_SELLOFF": Regime.BEAR,
    "EXTREME_VOLATILITY": Regime.BEAR,
    "BEAR": Regime.BEAR,
    "CRASH": Regime.BEAR,
    "DISTRIBUTION": Regime.BEAR,
    # SIDEWAYS family
    "VOLATILE_CHOP": Regime.SIDEWAYS,
    "QUIET_ACCUMULATION": Regime.SIDEWAYS,
    "WEAK_CONSOLIDATION": Regime.SIDEWAYS,
    "SIDEWAYS": Regime.SIDEWAYS,
    "CHOP": Regime.SIDEWAYS,
    "ACCUMULATION": Regime.SIDEWAYS,
    "RANGE_BOUND": Regime.SIDEWAYS,
}

# Expected data fields per regime (for coverage scoring)
_EXPECTED_FIELDS = {
    Regime.BEAR: ["price", "open_interest", "liquidations", "taker_ratio",
                   "funding_rate", "liquidation_intensity"],
    Regime.BULL: ["price", "funding_rate", "bid_depth", "ask_depth"],
    Regime.SIDEWAYS: ["price", "funding_rate"],
}


# =============================================================================
# V6 PAYLOAD
# =============================================================================

@dataclass
class KrakenQuantPayload:
    """V6-compatible signal payload for the Kraken Quant strategy matrix."""
    asset: str = ""
    direction: float = 0.0
    confidence: float = 0.0
    primary_strategy: str = "none"
    signal_edge_bps: float = 0.0
    data_quality: float = 0.0
    data_age_seconds: float = 0.0
    asof_timestamp: float = 0.0
    missing_inputs: List[str] = field(default_factory=list)
    diagnostics: Dict = field(default_factory=dict)
    is_valid: bool = True

    def to_agent_signal_dict(self) -> Dict[str, Any]:
        """Flat dict for Authority Fusion consumption."""
        return {
            "kq_direction": self.direction,
            "kq_confidence": self.confidence,
            "kq_primary_strategy": self.primary_strategy,
            "kq_signal_edge_bps": self.signal_edge_bps,
            "kq_data_quality": self.data_quality,
            "kq_is_valid": self.is_valid,
            "missing_inputs": self.missing_inputs,
            "asof_timestamp": self.asof_timestamp,
            "data_age_seconds": self.data_age_seconds,
            "asset": self.asset,
            "diagnostics": self.diagnostics,
        }
        


def _neutral_payload(asset: str = "") -> KrakenQuantPayload:
    """Fail-closed neutral payload: direction=0, confidence=0."""
    return KrakenQuantPayload(asset=asset, asof_timestamp=time.time())


# =============================================================================
# V6 AGENT
# =============================================================================

class KrakenQuantAgentV6:
    """
    HMATS v6 adapter for the 12-strategy Kraken Quant matrix.

    Wraps the research-grade strategies (liquidation cascade, Hurst, entropy,
    VRP, funding divergence, cointegration, OU, etc.) behind the canonical
    HMATS agent interface.

    Authority: ADVISE (supplementary to main.py Best-of-N live quant core)
    """

    ASSETS = ("BTC", "ETH", "SOL")

    def __init__(self):
        self.allocator = StrategyAllocator(
            temperature=0.5,
            vol_reduction_threshold=1.5,
            max_drawdown=0.15,
        )
        self._last_payloads: Dict[str, KrakenQuantPayload] = {}
        self._tick_count = 0
        logger.info("[KrakenQuantV6] Initialized with 12 strategies (3 regimes x 4)")

    # ---- public V6 interface ------------------------------------------------

    def generate_signal(
        self,
        asset: str = "BTC",
        market_data: Optional[Dict] = None,
        regime: Optional[str] = None,
        price: Optional[float] = None,
        **kwargs,
    ) -> KrakenQuantPayload:
        """
        Generate signal for a single asset.  Never raises, never returns None.
        Returns neutral payload (direction=0, confidence=0) on any error.
        """
        try:
            now = time.time()
            self._tick_count += 1

            if asset not in self.ASSETS:
                p = _neutral_payload(asset)
                p.diagnostics = {"reason": f"unsupported_asset_{asset}"}
                self._last_payloads[asset] = p
                return p

            # Resolve HMATS regime name -> local Regime enum
            local_regime = self._map_regime(regime)

            # Convert flat HMATS market_data -> nested strategy format
            flat = dict(market_data) if market_data else {}
            if price is not None:
                flat.setdefault("price", price)
            nested = self._convert_market_data(asset, flat)

            # Assess data coverage
            expected = _EXPECTED_FIELDS.get(local_regime, ["price"])
            present = [k for k in expected if self._has_field(flat, k, asset)]
            missing = [k for k in expected if k not in present]
            data_quality = len(present) / max(len(expected), 1)

            # Run strategy allocator
            try:
                signals = self.allocator.process_tick(local_regime, nested)
            except Exception as exc:
                logger.warning("[KrakenQuantV6] allocator.process_tick failed: %s", exc)
                signals = []

            # Filter to signals that target the requested asset
            asset_signals = [s for s in signals if self._signal_matches_asset(s, asset)]

            if not asset_signals:
                p = _neutral_payload(asset)
                p.data_quality = data_quality
                p.missing_inputs = missing
                p.asof_timestamp = now
                p.diagnostics = {
                    "regime": local_regime.value,
                    "tick": self._tick_count,
                    "strategies_fired": len(signals),
                    "asset_signals": 0,
                }
                self._last_payloads[asset] = p
                return p

            # Pick strongest signal by confidence
            best = max(asset_signals, key=lambda s: s.confidence)
            direction = self._extract_direction(best, asset)

            # Signal edge in bps (entry -> take_profit distance)
            edge_bps = 0.0
            if best.entry_price > 0 and best.take_profit > 0:
                edge_bps = abs(best.take_profit - best.entry_price) / best.entry_price * 10_000

            p = KrakenQuantPayload(
                asset=asset,
                direction=round(direction, 4),
                confidence=round(min(best.confidence, 1.0), 4),
                primary_strategy=best.strategy_name,
                signal_edge_bps=round(edge_bps, 1),
                data_quality=round(data_quality, 3),
                asof_timestamp=now,
                data_age_seconds=0.0,
                missing_inputs=missing,
                diagnostics={
                    "regime": local_regime.value,
                    "tick": self._tick_count,
                    "signal_type": best.signal_type.name,
                    "strategy_id": best.strategy_id,
                    "strategies_fired": len(signals),
                    "asset_signals": len(asset_signals),
                    "ir_weight": best.metadata.get("ir_weight", 0.0),
                },
                is_valid=True,
            )
            self._last_payloads[asset] = p
            return p

        except Exception as exc:
            logger.error("[KrakenQuantV6] generate_signal(%s) failed: %s",
                         asset, exc, exc_info=True)
            p = _neutral_payload(asset)
            p.diagnostics = {"error": str(exc)}
            self._last_payloads[asset] = p
            return p

    def get_last_payload(self, asset: str) -> KrakenQuantPayload:
        """Return the most recently cached payload (or neutral)."""
        return self._last_payloads.get(asset, _neutral_payload(asset))

    def get_firing_stats(self) -> Dict:
        """[KQ-DIAG] Proxy to allocator per-strategy firing telemetry."""
        try:
            return self.allocator.get_firing_stats()
        except Exception as exc:
            return {"error": str(exc)}

    def reset_state(self, asset: Optional[str] = None):
        """Clear cached payloads (single asset or all)."""
        if asset:
            self._last_payloads.pop(asset, None)
        else:
            self._last_payloads.clear()

    # ---- internal helpers ---------------------------------------------------

    @staticmethod
    def _map_regime(regime_str: Optional[str]) -> Regime:
        """Map HMATS regime name to local Regime enum, default SIDEWAYS."""
        if regime_str is None:
            return Regime.SIDEWAYS
        key = str(regime_str).upper().replace("-", "_")
        mapped = _REGIME_MAP.get(key)
        if mapped:
            return mapped
        # REGIME_0..7 from per-asset GMM -> conservative default
        if key.startswith("REGIME_"):
            return Regime.SIDEWAYS
        return Regime.SIDEWAYS

    @staticmethod
    def _convert_market_data(asset: str, flat: Dict) -> Dict:
        """Convert flat HMATS market_data dict to nested strategy format."""
        now = flat.get("timestamp", time.time())

        prices: Dict[str, float] = {}
        for a in ("BTC", "ETH", "SOL"):
            al = a.lower()
            p = flat.get(f"price_{al}", flat.get(f"{al}_price", 0))
            if p == 0 and a == asset:
                p = flat.get("price", flat.get("close", 0))
            if p > 0:
                prices[a] = float(p)

        def _per_asset(field_name: str) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for a in ("BTC", "ETH", "SOL"):
                al = a.lower()
                v = flat.get(f"{field_name}_{al}",
                             flat.get(field_name, 0) if a == asset else 0)
                if v:
                    out[a] = float(v)
            return out

        return {
            "timestamp": now,
            "prices": prices,
            "open_interest": _per_asset("open_interest"),
            "funding_rate": _per_asset("funding_rate"),
            "liquidations": _per_asset("liquidation_volume"),
            "taker_ratio": _per_asset("taker_ratio"),
            "liquidation_intensity": _per_asset("liquidation_intensity"),
            "bid_depth": _per_asset("bid_depth"),
            "ask_depth": _per_asset("ask_depth"),
        }

    @staticmethod
    def _has_field(flat: Dict, field_name: str, asset: str) -> bool:
        """Check if a data field is present (any truthy value)."""
        al = asset.lower()
        return bool(
            flat.get(f"{field_name}_{al}")
            or flat.get(field_name)
            or flat.get(f"{al}_{field_name}")
        )

    @staticmethod
    def _signal_matches_asset(signal: Signal, asset: str) -> bool:
        """Check if a strategy signal targets the requested asset."""
        sig_asset = signal.asset.upper()
        if sig_asset == asset:
            return True
        if asset in sig_asset:          # e.g. "SOL-ETH" contains "SOL"
            return True
        if sig_asset == "ALL":
            return True
        return False

    @staticmethod
    def _extract_direction(signal: Signal, asset: str) -> float:
        """Convert signal type to [-1, +1] direction, pair-aware."""
        parts = signal.asset.upper().split("-")
        if len(parts) == 2:
            asset_a, asset_b = parts
            if signal.signal_type == SignalType.PAIR_LONG_A_SHORT_B:
                return 0.6 if asset == asset_a else -0.6
            if signal.signal_type == SignalType.PAIR_SHORT_A_LONG_B:
                return -0.6 if asset == asset_a else 0.6
        return _SIGNAL_DIRECTION.get(signal.signal_type, 0.0)


# =============================================================================
# MODULE SINGLETON
# =============================================================================

_kq_instance: Optional[KrakenQuantAgentV6] = None


def get_kraken_quant_agent() -> KrakenQuantAgentV6:
    """Return (or create) the module-level KrakenQuantAgentV6 singleton."""
    global _kq_instance
    if _kq_instance is None:
        _kq_instance = KrakenQuantAgentV6()
    return _kq_instance


def reset_kraken_quant_agent():
    """Destroy the singleton so the next call to get_* creates a fresh one."""
    global _kq_instance
    _kq_instance = None


# ---------------------------------------------------------------------------
# PUBLIC ADAPTER FUNCTIONS (dependency-free - no ccxt/websocket/scipy needed)
# ---------------------------------------------------------------------------

def map_v6_regime_to_research_regime(v6_regime: str) -> str:
    """
    Map an HMATS v6 regime label to the research-strategy regime.

    Returns one of "BEAR", "BULL", "SIDEWAYS".  Unknown regimes (including
    per-asset REGIME_N labels) default to "SIDEWAYS".

    >>> map_v6_regime_to_research_regime("MOMENTUM_RALLY")
    'BULL'
    >>> map_v6_regime_to_research_regime("REGIME_5")
    'SIDEWAYS'
    """
    key = str(v6_regime).upper().replace("-", "_")
    local = _REGIME_MAP.get(key)
    if local is None and key.startswith("REGIME_"):
        local = Regime.SIDEWAYS
    return (local or Regime.SIDEWAYS).name


def signals_to_v6_quant_packet(
    signals: List,
    primary_asset: str = "BTC",
) -> Dict[str, Any]:
    """
    Aggregate a list of research-strategy Signal namedtuples into a flat
    v6-compatible quant packet dict.

    No heavy deps required (no ccxt, websocket, scipy).

    Returns dict with keys: kq_direction, kq_confidence, kq_primary_strategy,
    kq_signal_edge_bps, kq_is_valid, asset, signals_count.
    """
    if not signals:
        return {
            "kq_direction": 0.0,
            "kq_confidence": 0.0,
            "kq_primary_strategy": "none",
            "kq_signal_edge_bps": 0.0,
            "kq_is_valid": True,
            "asset": primary_asset,
            "signals_count": 0,
        }

    # Pick strongest by confidence
    best = max(signals, key=lambda s: getattr(s, "confidence", 0.0))
    direction = _SIGNAL_DIRECTION.get(
        getattr(best, "signal_type", None), 0.0
    )

    edge_bps = 0.0
    entry = getattr(best, "entry_price", 0.0) or 0.0
    tp = getattr(best, "take_profit", 0.0) or 0.0
    if entry > 0 and tp > 0:
        edge_bps = abs(tp - entry) / entry * 10_000

    return {
        "kq_direction": round(direction, 4),
        "kq_confidence": round(min(getattr(best, "confidence", 0.0), 1.0), 4),
        "kq_primary_strategy": getattr(best, "strategy_name", "unknown"),
        "kq_signal_edge_bps": round(edge_bps, 1),
        "kq_is_valid": True,
        "asset": primary_asset,
        "signals_count": len(signals),
    }


# ---------------------------------------------------------------------------
# Backward compatibility alias - gated behind env flag for research isolation
# ---------------------------------------------------------------------------

if os.getenv("HMATS_ENABLE_RESEARCH_AGENTS") == "1":
    KrakenQuantAgent = KrakenQuantAgentV6
else:
    # Default: alias exists but logs a deprecation on first use
    class _DeprecatedKrakenQuantAgent(KrakenQuantAgentV6):
        """Wrapper that emits a one-time deprecation warning."""
        _warned = False

        def __init__(self):
            if not _DeprecatedKrakenQuantAgent._warned:
                _DeprecatedKrakenQuantAgent._warned = True
                logger.warning(
                    "KrakenQuantAgent alias is deprecated. "
                    "Use KrakenQuantAgentV6 directly, or set "
                    "HMATS_ENABLE_RESEARCH_AGENTS=1 to suppress."
                )
            super().__init__()

    KrakenQuantAgent = _DeprecatedKrakenQuantAgent


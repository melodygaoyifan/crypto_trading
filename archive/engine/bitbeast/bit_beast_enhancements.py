"""
BitBeast Enhancements - Advanced Algorithmic Logic
===================================================

Implements the "Bit-Beast" enhancements for HMATS v3.1:

1. SNR Filter (Ants vs Train Rule)
   - Pattern duration ratio analysis
   - Down-weight reversal signals if consolidation < 50% of prior trend

2. Dynamic Pyramiding
   - 20%-15%-15% staggered entry based on DT confidence
   - Replaces 50% flat-loading

3. MTF Tensor Fusion
   - Simultaneous 1m (Micro-risk) + 15m (Confirmation) + 4H (Trend)
   - 128GB RAM cache integration

4. Circuit Breakers
   - Volume_Trend_Divergence: Prohibit knife-catching
   - Cross-Asset Regime Hysteresis (BTC leads, SOL lags)
   - Liquidation Cluster Proximity Veto

Hardware Target: MSI Titan 18 HX (24-core Ultra 9, RTX 5090, 128GB DDR5)
Author: Lead Quant Research & ML Systems Architect
Version: 3.1.2
"""

import os
import time
import logging
import threading
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Callable, Union, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BitBeastEnhancements')


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# SNR Filter Configuration
MIN_PATTERN_DURATION_RATIO = 0.50  # Ants vs Train threshold
TREND_LOOKBACK_BARS = 100
CONSOLIDATION_THRESHOLD_ATR = 0.5  # Below 0.5 ATR = consolidation

# Dynamic Pyramiding Configuration
PYRAMID_TRANCHES = [0.20, 0.15, 0.15]  # 20%-15%-15% = 50% total
CONFIDENCE_THRESHOLDS = [0.60, 0.70, 0.80]  # Required confidence for each tranche
TRANCHE_DELAY_BARS = 3  # Minimum bars between tranches

# MTF Configuration
MTF_TIMEFRAMES = ['1m', '15m', '4H']
MTF_TENSOR_DIMS = {
    '1m': 64,   # Micro-risk features
    '15m': 32,  # Confirmation features
    '4H': 32,   # Trend features
}
TOTAL_MTF_DIM = sum(MTF_TENSOR_DIMS.values())  # 128 dims

# Circuit Breaker Configuration
VOLUME_DIVERGENCE_LOOKBACK = 10
VOLUME_SHRINK_THRESHOLD = 0.7  # Volume < 70% of MA
LIQUIDATION_PROXIMITY_PCT = 0.01  # 1% of price
CROSS_ASSET_LAG_THRESHOLD = 0.02  # 2% divergence


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class TrendState(Enum):
    """Current trend state classification."""
    STRONG_UPTREND = auto()
    WEAK_UPTREND = auto()
    CONSOLIDATION = auto()
    WEAK_DOWNTREND = auto()
    STRONG_DOWNTREND = auto()


class CircuitBreakerType(Enum):
    """Types of circuit breakers."""
    VOLUME_DIVERGENCE = auto()
    CROSS_ASSET_LAG = auto()
    LIQUIDATION_PROXIMITY = auto()
    REGIME_HYSTERESIS = auto()
    MTF_CONFLICT = auto()


class PyramidState(Enum):
    """Pyramid entry state."""
    IDLE = auto()
    TRANCHE_1_FILLED = auto()
    TRANCHE_2_FILLED = auto()
    TRANCHE_3_FILLED = auto()
    COMPLETE = auto()


@dataclass
class TrendAnalysis:
    """Result of trend analysis."""
    state: TrendState
    duration_bars: int
    strength: float  # -1 to +1
    atr_ratio: float
    is_consolidation: bool
    prior_trend_duration: int
    pattern_duration_ratio: float


@dataclass
class SNRFilterResult:
    """Result of SNR (Ants vs Train) filter."""
    passed: bool
    pattern_ratio: float
    reversal_weight: float  # 0 to 1, how much to trust reversal signal
    reason: str


@dataclass
class PyramidPlan:
    """Pyramid entry plan."""
    asset: str
    side: str  # "LONG" or "SHORT"
    total_size: float
    tranches: List[float]  # Size per tranche
    confidence_required: List[float]
    current_state: PyramidState
    filled_tranches: int
    avg_entry_price: float
    last_fill_bar: int


@dataclass
class MTFTensor:
    """Multi-timeframe tensor for DRL."""
    timestamp: float
    asset: str
    
    # Individual timeframe tensors
    tensor_1m: np.ndarray    # shape: (64,)
    tensor_15m: np.ndarray   # shape: (32,)
    tensor_4h: np.ndarray    # shape: (32,)
    
    # Fused tensor
    fused: np.ndarray        # shape: (128,)
    
    # Metadata
    mtf_alignment_score: float  # How aligned are the timeframes
    dominant_timeframe: str
    conflict_detected: bool


@dataclass
class CircuitBreakerState:
    """State of circuit breakers."""
    triggered: bool = False
    breaker_type: Optional[CircuitBreakerType] = None
    severity: float = 0.0  # 0 to 1
    message: str = ""
    expires_at: float = 0.0
    
    # Individual breaker states
    volume_divergence: bool = False
    cross_asset_lag: bool = False
    liquidation_proximity: bool = False
    regime_hysteresis: bool = False
    mtf_conflict: bool = False


@dataclass
class LiquidationCluster:
    """Represents a liquidation cluster level."""
    price: float
    size_usd: float
    side: str  # "LONG" or "SHORT"
    leverage_weighted: float
    timestamp: float


# ═══════════════════════════════════════════════════════════════════════════════
# SNR FILTER (ANTS VS TRAIN RULE)
# ═══════════════════════════════════════════════════════════════════════════════

class SNRFilter:
    """
    Signal-to-Noise Ratio Filter implementing the "Ants vs Train" rule.
    
    Core Logic:
    - If consolidation duration < 50% of prior trend duration, 
      down-weight reversal signals
    - "Don't fight the trend until it's exhausted"
    
    The metaphor: Small ants (reversal signals) can't stop a moving train 
    (strong trend) - wait for the train to slow down first.
    """
    
    def __init__(
        self,
        min_pattern_ratio: float = MIN_PATTERN_DURATION_RATIO,
        lookback_bars: int = TREND_LOOKBACK_BARS,
        atr_period: int = 14
    ):
        self.min_pattern_ratio = min_pattern_ratio
        self.lookback_bars = lookback_bars
        self.atr_period = atr_period
        
        # Price history per asset
        self._history: Dict[str, deque] = {}
        self._atr_cache: Dict[str, float] = {}
        
        # Trend state tracking
        self._trend_states: Dict[str, TrendAnalysis] = {}
        
        logger.info(f"SNRFilter initialized (ratio={min_pattern_ratio}, lookback={lookback_bars})")
    
    def _ensure_history(self, asset: str):
        """Ensure history buffer exists for asset."""
        if asset not in self._history:
            self._history[asset] = deque(maxlen=self.lookback_bars * 2)
    
    def update(
        self,
        asset: str,
        price: float,
        high: float,
        low: float,
        volume: float
    ):
        """Update with new price bar."""
        self._ensure_history(asset)
        self._history[asset].append({
            'price': price,
            'high': high,
            'low': low,
            'volume': volume,
            'timestamp': time.time()
        })
        
        # Update ATR
        self._update_atr(asset)
        
        # Update trend analysis
        self._analyze_trend(asset)
    
    def _update_atr(self, asset: str):
        """Calculate Average True Range."""
        history = list(self._history[asset])
        if len(history) < self.atr_period + 1:
            self._atr_cache[asset] = 0.0
            return
        
        true_ranges = []
        for i in range(1, min(self.atr_period + 1, len(history))):
            h = history[-i]['high']
            l = history[-i]['low']
            prev_c = history[-i-1]['price']
            
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            true_ranges.append(tr)
        
        self._atr_cache[asset] = np.mean(true_ranges) if true_ranges else 0.0
    
    def _analyze_trend(self, asset: str):
        """Analyze current trend state."""
        history = list(self._history[asset])
        if len(history) < 20:
            self._trend_states[asset] = TrendAnalysis(
                state=TrendState.CONSOLIDATION,
                duration_bars=0,
                strength=0.0,
                atr_ratio=0.0,
                is_consolidation=True,
                prior_trend_duration=0,
                pattern_duration_ratio=1.0
            )
            return
        
        prices = np.array([h['price'] for h in history])
        atr = self._atr_cache.get(asset, 1.0) or 1.0
        
        # Calculate trend metrics
        returns = np.diff(prices) / prices[:-1]
        recent_returns = returns[-20:]
        
        # Trend strength (-1 to +1)
        strength = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8)
        strength = np.clip(strength, -1, 1)
        
        # Price range relative to ATR
        price_range = np.max(prices[-20:]) - np.min(prices[-20:])
        atr_ratio = price_range / (atr * 20) if atr > 0 else 0
        
        # Determine if consolidation
        is_consolidation = atr_ratio < CONSOLIDATION_THRESHOLD_ATR
        
        # Determine trend state
        if is_consolidation:
            state = TrendState.CONSOLIDATION
        elif strength > 0.5:
            state = TrendState.STRONG_UPTREND
        elif strength > 0.1:
            state = TrendState.WEAK_UPTREND
        elif strength < -0.5:
            state = TrendState.STRONG_DOWNTREND
        elif strength < -0.1:
            state = TrendState.WEAK_DOWNTREND
        else:
            state = TrendState.CONSOLIDATION
        
        # Count trend duration
        duration = self._count_trend_duration(prices, state)
        
        # Find prior trend duration
        prior_duration = self._find_prior_trend_duration(asset, state)
        
        # Calculate pattern duration ratio
        if prior_duration > 0 and is_consolidation:
            ratio = duration / prior_duration
        else:
            ratio = 1.0  # No prior trend or not in consolidation
        
        self._trend_states[asset] = TrendAnalysis(
            state=state,
            duration_bars=duration,
            strength=strength,
            atr_ratio=atr_ratio,
            is_consolidation=is_consolidation,
            prior_trend_duration=prior_duration,
            pattern_duration_ratio=ratio
        )
    
    def _count_trend_duration(self, prices: np.ndarray, current_state: TrendState) -> int:
        """Count how long current trend/consolidation has lasted."""
        if len(prices) < 5:
            return 0
        
        # Simplified: count bars with same direction
        returns = np.diff(prices)
        duration = 0
        
        if current_state in [TrendState.STRONG_UPTREND, TrendState.WEAK_UPTREND]:
            # Count positive returns from end
            for r in reversed(returns):
                if r > 0:
                    duration += 1
                else:
                    break
        elif current_state in [TrendState.STRONG_DOWNTREND, TrendState.WEAK_DOWNTREND]:
            # Count negative returns from end
            for r in reversed(returns):
                if r < 0:
                    duration += 1
                else:
                    break
        else:
            # Consolidation: count bars within ATR band
            recent_mean = np.mean(prices[-20:])
            atr = self._atr_cache.get('', 1.0) or np.std(prices[-20:])
            for p in reversed(prices):
                if abs(p - recent_mean) < atr * 1.5:
                    duration += 1
                else:
                    break
        
        return duration
    
    def _find_prior_trend_duration(self, asset: str, current_state: TrendState) -> int:
        """Find duration of the trend before current consolidation."""
        history = list(self._history[asset])
        if len(history) < 30:
            return 0
        
        prices = np.array([h['price'] for h in history])
        
        # If currently in consolidation, look for prior trend
        if current_state == TrendState.CONSOLIDATION:
            # Find where consolidation started
            returns = np.diff(prices)
            consol_start = len(returns)
            
            # Walk back to find trending period
            for i in range(len(returns) - 1, -1, -1):
                if abs(returns[i]) > self._atr_cache.get(asset, 0.01) * 0.5:
                    consol_start = i
                    break
            
            # Count prior trend duration
            if consol_start > 5:
                prior_trend_dir = np.sign(np.sum(returns[max(0, consol_start-20):consol_start]))
                count = 0
                for i in range(consol_start - 1, -1, -1):
                    if np.sign(returns[i]) == prior_trend_dir:
                        count += 1
                    else:
                        break
                return count
        
        return 0
    
    def filter_signal(
        self,
        asset: str,
        signal_direction: str,  # "LONG" or "SHORT"
        signal_confidence: float
    ) -> SNRFilterResult:
        """
        Apply SNR filter to a trading signal.
        
        Returns filtered result with adjusted weight for reversal signals.
        """
        analysis = self._trend_states.get(asset)
        
        if analysis is None:
            return SNRFilterResult(
                passed=True,
                pattern_ratio=1.0,
                reversal_weight=1.0,
                reason="No trend data available"
            )
        
        # Determine if signal is a reversal
        is_reversal = False
        if signal_direction == "LONG" and analysis.state in [
            TrendState.STRONG_DOWNTREND, TrendState.WEAK_DOWNTREND
        ]:
            is_reversal = True
        elif signal_direction == "SHORT" and analysis.state in [
            TrendState.STRONG_UPTREND, TrendState.WEAK_UPTREND
        ]:
            is_reversal = True
        
        # If not a reversal or not in consolidation, pass through
        if not is_reversal or not analysis.is_consolidation:
            return SNRFilterResult(
                passed=True,
                pattern_ratio=analysis.pattern_duration_ratio,
                reversal_weight=1.0,
                reason="Signal aligns with trend or not in consolidation"
            )
        
        # Apply Ants vs Train rule
        if analysis.pattern_duration_ratio < self.min_pattern_ratio:
            # Consolidation too short - down-weight reversal
            weight = analysis.pattern_duration_ratio / self.min_pattern_ratio
            weight = max(0.1, weight)  # Minimum 10% weight
            
            return SNRFilterResult(
                passed=weight > 0.5,  # Only pass if weight > 50%
                pattern_ratio=analysis.pattern_duration_ratio,
                reversal_weight=weight,
                reason=f"Ants vs Train: consolidation ({analysis.duration_bars} bars) "
                       f"< 50% of prior trend ({analysis.prior_trend_duration} bars)"
            )
        
        return SNRFilterResult(
            passed=True,
            pattern_ratio=analysis.pattern_duration_ratio,
            reversal_weight=1.0,
            reason="Consolidation sufficient for reversal signal"
        )
    
    def get_trend_analysis(self, asset: str) -> Optional[TrendAnalysis]:
        """Get current trend analysis for asset."""
        return self._trend_states.get(asset)


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC PYRAMIDING
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicPyramidManager:
    """
    Dynamic Pyramiding with 20%-15%-15% staggered entry.
    
    Key Features:
    - Confidence-gated tranches (60%, 70%, 80% thresholds)
    - Minimum bar delay between tranches
    - Average entry price tracking
    - Partial fill handling
    """
    
    def __init__(
        self,
        tranches: List[float] = None,
        confidence_thresholds: List[float] = None,
        tranche_delay_bars: int = TRANCHE_DELAY_BARS
    ):
        self.tranches = tranches or PYRAMID_TRANCHES
        self.confidence_thresholds = confidence_thresholds or CONFIDENCE_THRESHOLDS
        self.tranche_delay = tranche_delay_bars
        
        # Active pyramid plans
        self._plans: Dict[str, PyramidPlan] = {}
        self._current_bar: Dict[str, int] = {}
        
        logger.info(f"DynamicPyramidManager initialized: {self.tranches}")
    
    def create_plan(
        self,
        asset: str,
        side: str,
        total_size: float,
        current_bar: int
    ) -> PyramidPlan:
        """Create a new pyramid entry plan."""
        tranche_sizes = [total_size * t for t in self.tranches]
        
        plan = PyramidPlan(
            asset=asset,
            side=side,
            total_size=total_size,
            tranches=tranche_sizes,
            confidence_required=self.confidence_thresholds.copy(),
            current_state=PyramidState.IDLE,
            filled_tranches=0,
            avg_entry_price=0.0,
            last_fill_bar=0
        )
        
        self._plans[asset] = plan
        self._current_bar[asset] = current_bar
        
        logger.info(f"Created pyramid plan for {asset}: {tranche_sizes}")
        return plan
    
    def should_fill_tranche(
        self,
        asset: str,
        confidence: float,
        current_bar: int,
        price: float
    ) -> Tuple[bool, float, str]:
        """
        Check if next tranche should be filled.
        
        Returns: (should_fill, size, reason)
        """
        plan = self._plans.get(asset)
        if plan is None:
            return False, 0.0, "No active plan"
        
        if plan.filled_tranches >= len(plan.tranches):
            return False, 0.0, "All tranches filled"
        
        # Check bar delay
        bars_since_last = current_bar - plan.last_fill_bar
        if plan.filled_tranches > 0 and bars_since_last < self.tranche_delay:
            return False, 0.0, f"Waiting for delay ({bars_since_last}/{self.tranche_delay} bars)"
        
        # Check confidence threshold
        required_conf = plan.confidence_required[plan.filled_tranches]
        if confidence < required_conf:
            return False, 0.0, f"Confidence {confidence:.2f} < required {required_conf:.2f}"
        
        # Approved - return tranche size
        size = plan.tranches[plan.filled_tranches]
        return True, size, f"Tranche {plan.filled_tranches + 1} approved"
    
    def record_fill(
        self,
        asset: str,
        size: float,
        price: float,
        current_bar: int
    ):
        """Record a tranche fill."""
        plan = self._plans.get(asset)
        if plan is None:
            return
        
        # Update average entry
        if plan.avg_entry_price == 0:
            plan.avg_entry_price = price
        else:
            total_filled = sum(plan.tranches[:plan.filled_tranches])
            new_total = total_filled + size
            plan.avg_entry_price = (
                (plan.avg_entry_price * total_filled + price * size) / new_total
            )
        
        plan.filled_tranches += 1
        plan.last_fill_bar = current_bar
        
        # Update state
        if plan.filled_tranches == 1:
            plan.current_state = PyramidState.TRANCHE_1_FILLED
        elif plan.filled_tranches == 2:
            plan.current_state = PyramidState.TRANCHE_2_FILLED
        elif plan.filled_tranches == 3:
            plan.current_state = PyramidState.TRANCHE_3_FILLED
        
        if plan.filled_tranches >= len(plan.tranches):
            plan.current_state = PyramidState.COMPLETE
        
        logger.info(
            f"Pyramid fill {asset}: tranche {plan.filled_tranches}/{len(plan.tranches)}, "
            f"avg_entry={plan.avg_entry_price:.2f}"
        )
    
    def get_plan(self, asset: str) -> Optional[PyramidPlan]:
        """Get current pyramid plan for asset."""
        return self._plans.get(asset)
    
    def cancel_plan(self, asset: str):
        """Cancel pyramid plan for asset."""
        if asset in self._plans:
            del self._plans[asset]
            logger.info(f"Cancelled pyramid plan for {asset}")
    
    def get_remaining_size(self, asset: str) -> float:
        """Get remaining size to fill."""
        plan = self._plans.get(asset)
        if plan is None:
            return 0.0
        return sum(plan.tranches[plan.filled_tranches:])


# ═══════════════════════════════════════════════════════════════════════════════
# MTF TENSOR FUSION
# ═══════════════════════════════════════════════════════════════════════════════

class MTFTensorFusion:
    """
    Multi-Timeframe Tensor Fusion for Decision Transformer.
    
    Simultaneously processes:
    - 1m (Micro-risk): 64 features - immediate volatility, spread, order flow
    - 15m (Confirmation): 32 features - momentum confirmation, volume profile
    - 4H (Trend): 32 features - trend structure, support/resistance
    
    Total: 128-dim fused tensor for DRL input.
    """
    
    def __init__(
        self,
        dims: Dict[str, int] = None,
        alignment_threshold: float = 0.6
    ):
        self.dims = dims or MTF_TENSOR_DIMS
        self.total_dim = sum(self.dims.values())
        self.alignment_threshold = alignment_threshold
        
        # Feature buffers per asset
        self._buffers: Dict[str, Dict[str, deque]] = {}
        
        # Normalization stats
        self._means: Dict[str, np.ndarray] = {}
        self._stds: Dict[str, np.ndarray] = {}
        
        logger.info(f"MTFTensorFusion initialized: dims={self.dims}, total={self.total_dim}")
    
    def _ensure_buffers(self, asset: str):
        """Ensure feature buffers exist for asset."""
        if asset not in self._buffers:
            self._buffers[asset] = {
                '1m': deque(maxlen=100),
                '15m': deque(maxlen=100),
                '4H': deque(maxlen=100),
            }
    
    def update_timeframe(
        self,
        asset: str,
        timeframe: str,
        features: np.ndarray
    ):
        """Update features for a specific timeframe."""
        if timeframe not in self.dims:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        
        expected_dim = self.dims[timeframe]
        if len(features) != expected_dim:
            raise ValueError(f"Expected {expected_dim} features for {timeframe}, got {len(features)}")
        
        self._ensure_buffers(asset)
        self._buffers[asset][timeframe].append(features)
    
    def fuse(self, asset: str) -> MTFTensor:
        """
        Fuse multi-timeframe features into single tensor.
        
        Returns MTFTensor with alignment analysis.
        """
        self._ensure_buffers(asset)
        
        # Get latest features from each timeframe
        tensor_1m = self._get_latest_features(asset, '1m')
        tensor_15m = self._get_latest_features(asset, '15m')
        tensor_4h = self._get_latest_features(asset, '4H')
        
        # Concatenate into fused tensor
        fused = np.concatenate([tensor_1m, tensor_15m, tensor_4h])
        
        # Calculate alignment score
        alignment, dominant_tf, conflict = self._calculate_alignment(
            tensor_1m, tensor_15m, tensor_4h
        )
        
        return MTFTensor(
            timestamp=time.time(),
            asset=asset,
            tensor_1m=tensor_1m,
            tensor_15m=tensor_15m,
            tensor_4h=tensor_4h,
            fused=fused,
            mtf_alignment_score=alignment,
            dominant_timeframe=dominant_tf,
            conflict_detected=conflict
        )
    
    def _get_latest_features(self, asset: str, timeframe: str) -> np.ndarray:
        """Get latest features or zeros if not available."""
        buffer = self._buffers.get(asset, {}).get(timeframe, deque())
        if buffer:
            return np.array(buffer[-1], dtype=np.float32)
        return np.zeros(self.dims[timeframe], dtype=np.float32)
    
    def _calculate_alignment(
        self,
        tensor_1m: np.ndarray,
        tensor_15m: np.ndarray,
        tensor_4h: np.ndarray
    ) -> Tuple[float, str, bool]:
        """
        Calculate alignment between timeframes.
        
        Returns: (alignment_score, dominant_timeframe, conflict_detected)
        """
        # Extract directional signal from each (assuming first feature is direction)
        # In production, this would use specific signal indices
        
        def get_direction(tensor: np.ndarray) -> float:
            if len(tensor) == 0:
                return 0.0
            # Use mean of normalized features as proxy for direction
            return np.tanh(np.mean(tensor))
        
        dir_1m = get_direction(tensor_1m)
        dir_15m = get_direction(tensor_15m)
        dir_4h = get_direction(tensor_4h)
        
        # Calculate agreement
        directions = [dir_1m, dir_15m, dir_4h]
        mean_dir = np.mean(directions)
        
        # Alignment score: how much do they agree
        alignment = 1.0 - np.std(directions)
        alignment = np.clip(alignment, 0, 1)
        
        # Dominant timeframe: which has strongest signal
        strengths = [abs(dir_1m), abs(dir_15m), abs(dir_4h)]
        dominant_idx = np.argmax(strengths)
        dominant_tf = ['1m', '15m', '4H'][dominant_idx]
        
        # Conflict: 4H and lower timeframes disagree
        conflict = (np.sign(dir_4h) != np.sign(dir_1m) and 
                   abs(dir_4h) > 0.3 and abs(dir_1m) > 0.3)
        
        return alignment, dominant_tf, conflict
    
    def create_micro_risk_features(
        self,
        spread_bps: float,
        volatility_1m: float,
        order_flow_imbalance: float,
        bid_depth: float,
        ask_depth: float,
        recent_trades: np.ndarray = None
    ) -> np.ndarray:
        """Create 1m micro-risk feature tensor (64 dims)."""
        features = np.zeros(64, dtype=np.float32)
        
        # Core features (0-9)
        features[0] = np.tanh(spread_bps / 10)  # Normalize spread
        features[1] = np.tanh(volatility_1m)
        features[2] = np.tanh(order_flow_imbalance)
        features[3] = np.log1p(bid_depth) / 10
        features[4] = np.log1p(ask_depth) / 10
        features[5] = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-8)
        
        # Recent trade features (10-29)
        if recent_trades is not None and len(recent_trades) > 0:
            features[10:10+min(20, len(recent_trades))] = recent_trades[:20]
        
        # Additional micro features (30-63)
        # Tick imbalance (30-34)
        if recent_trades is not None and len(recent_trades) >= 10:
            trades_array = np.array(recent_trades[:50]) if len(recent_trades) >= 50 else np.array(recent_trades)
            up_ticks = np.sum(trades_array > 0)
            down_ticks = np.sum(trades_array < 0)
            total_ticks = len(trades_array)
            features[30] = (up_ticks - down_ticks) / (total_ticks + 1e-8)  # Tick imbalance
            features[31] = np.mean(np.abs(trades_array))  # Avg tick size
            features[32] = np.std(trades_array) / (np.mean(np.abs(trades_array)) + 1e-8)  # Tick volatility
            features[33] = np.tanh(np.sum(trades_array))  # Cumulative tick
            features[34] = len(trades_array) / 100  # Trade arrival rate (normalized)
        
        # Microprice features (35-39)
        if bid_depth > 0 and ask_depth > 0:
            # Weighted midprice (microprice proxy)
            total_depth = bid_depth + ask_depth
            features[35] = ask_depth / total_depth  # Bid weight for microprice
            features[36] = np.log(bid_depth / ask_depth) if ask_depth > 0 else 0  # Log depth ratio
            features[37] = np.tanh((bid_depth - ask_depth) / (total_depth * 0.1))  # Scaled imbalance
            features[38] = 1.0 / (spread_bps + 1)  # Inverse spread (liquidity proxy)
            features[39] = np.tanh(spread_bps * order_flow_imbalance)  # Spread-flow interaction
        
        # Volatility regime features (40-44)
        features[40] = np.tanh(volatility_1m * 10)  # Scaled volatility
        features[41] = 1.0 if volatility_1m > 0.002 else 0.0  # High vol flag
        features[42] = 1.0 if volatility_1m < 0.0005 else 0.0  # Low vol flag
        features[43] = np.sign(order_flow_imbalance) * volatility_1m  # Directional vol
        features[44] = np.tanh(spread_bps * volatility_1m * 100)  # Spread-vol interaction
        
        # Market quality features (45-49)
        features[45] = np.clip(spread_bps / 5, 0, 1)  # Normalized spread quality
        features[46] = np.tanh(np.log1p(bid_depth + ask_depth))  # Total depth
        features[47] = min(bid_depth, ask_depth) / (max(bid_depth, ask_depth) + 1e-8)  # Depth symmetry
        features[48] = np.tanh(order_flow_imbalance * (bid_depth + ask_depth) / 1e6)  # Flow-depth interaction
        features[49] = 1.0 / (1.0 + spread_bps * volatility_1m * 1000)  # Market quality score
        
        # Reserved for derived features (50-63)
        # These would be computed from other modules (regime, sentiment, etc.)
        features[50] = np.random.uniform(-0.1, 0.1)  # Noise for regularization
        
        return features
    
    def create_confirmation_features(
        self,
        momentum_15m: float,
        volume_profile: np.ndarray,
        vwap_deviation: float,
        rsi_15m: float,
        macd_signal: float
    ) -> np.ndarray:
        """Create 15m confirmation feature tensor (32 dims)."""
        features = np.zeros(32, dtype=np.float32)
        
        features[0] = np.tanh(momentum_15m)
        features[1] = np.tanh(vwap_deviation / 0.01)  # 1% = 1.0
        features[2] = (rsi_15m - 50) / 50  # Normalize to [-1, 1]
        features[3] = np.tanh(macd_signal)
        
        # Volume profile (4-13)
        if volume_profile is not None and len(volume_profile) > 0:
            normalized_vp = volume_profile / (np.max(volume_profile) + 1e-8)
            features[4:4+min(10, len(normalized_vp))] = normalized_vp[:10]
        
        return features
    
    def create_trend_features(
        self,
        trend_strength: float,
        support_distance: float,
        resistance_distance: float,
        higher_highs: int,
        lower_lows: int,
        trend_duration: int
    ) -> np.ndarray:
        """Create 4H trend feature tensor (32 dims)."""
        features = np.zeros(32, dtype=np.float32)
        
        features[0] = np.tanh(trend_strength)
        features[1] = np.tanh(support_distance / 0.05)  # 5% = 1.0
        features[2] = np.tanh(resistance_distance / 0.05)
        features[3] = min(higher_highs, 5) / 5
        features[4] = min(lower_lows, 5) / 5
        features[5] = min(trend_duration, 50) / 50
        
        # Structure features
        features[6] = 1.0 if higher_highs > lower_lows else -1.0 if lower_lows > higher_highs else 0.0
        
        return features


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKERS
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitBreakerSystem:
    """
    Multi-factor circuit breaker system.
    
    Breakers:
    1. Volume-Trend Divergence: No knife-catching when price falls + volume shrinks
    2. Cross-Asset Lag: Regime hysteresis when BTC leads, SOL lags
    3. Liquidation Proximity: Veto orders near major liquidation clusters
    4. MTF Conflict: Pause when timeframes strongly disagree
    """
    
    def __init__(
        self,
        volume_lookback: int = VOLUME_DIVERGENCE_LOOKBACK,
        volume_shrink_threshold: float = VOLUME_SHRINK_THRESHOLD,
        liquidation_proximity: float = LIQUIDATION_PROXIMITY_PCT,
        cross_asset_threshold: float = CROSS_ASSET_LAG_THRESHOLD,
        cooldown_seconds: float = 300  # 5 min cooldown after trigger
    ):
        self.volume_lookback = volume_lookback
        self.volume_shrink_threshold = volume_shrink_threshold
        self.liquidation_proximity = liquidation_proximity
        self.cross_asset_threshold = cross_asset_threshold
        self.cooldown = cooldown_seconds
        
        # State
        self._state: Dict[str, CircuitBreakerState] = {}
        
        # Historical data
        self._volume_history: Dict[str, deque] = {}
        self._price_history: Dict[str, deque] = {}
        self._returns: Dict[str, deque] = {}
        
        # Liquidation clusters
        self._liquidation_clusters: Dict[str, List[LiquidationCluster]] = {}
        
        logger.info("CircuitBreakerSystem initialized")
    
    def _ensure_history(self, asset: str):
        """Ensure history buffers exist."""
        if asset not in self._volume_history:
            self._volume_history[asset] = deque(maxlen=self.volume_lookback * 2)
            self._price_history[asset] = deque(maxlen=self.volume_lookback * 2)
            self._returns[asset] = deque(maxlen=self.volume_lookback * 2)
            self._state[asset] = CircuitBreakerState()
    
    def update(
        self,
        asset: str,
        price: float,
        volume: float
    ):
        """Update with new data."""
        self._ensure_history(asset)
        
        # Calculate return
        if self._price_history[asset]:
            prev_price = self._price_history[asset][-1]
            ret = (price - prev_price) / prev_price
        else:
            ret = 0.0
        
        self._price_history[asset].append(price)
        self._volume_history[asset].append(volume)
        self._returns[asset].append(ret)
    
    def update_liquidation_clusters(
        self,
        asset: str,
        clusters: List[LiquidationCluster]
    ):
        """Update liquidation cluster levels."""
        self._liquidation_clusters[asset] = clusters
    
    def check_all(
        self,
        asset: str,
        current_price: float,
        proposed_action: str,  # "BUY" or "SELL"
        mtf_tensor: MTFTensor = None,
        cross_asset_returns: Dict[str, float] = None
    ) -> CircuitBreakerState:
        """
        Check all circuit breakers.
        
        Returns CircuitBreakerState with triggered breakers.
        """
        self._ensure_history(asset)
        state = CircuitBreakerState()
        
        # Check cooldown
        current_state = self._state.get(asset, CircuitBreakerState())
        if current_state.triggered and time.time() < current_state.expires_at:
            return current_state
        
        # 1. Volume-Trend Divergence
        vol_div = self._check_volume_divergence(asset, proposed_action)
        state.volume_divergence = vol_div
        
        # 2. Cross-Asset Lag
        if cross_asset_returns:
            cross_lag = self._check_cross_asset_lag(asset, cross_asset_returns)
            state.cross_asset_lag = cross_lag
        
        # 3. Liquidation Proximity
        liq_prox = self._check_liquidation_proximity(asset, current_price, proposed_action)
        state.liquidation_proximity = liq_prox
        
        # 4. MTF Conflict
        if mtf_tensor:
            state.mtf_conflict = mtf_tensor.conflict_detected
        
        # Aggregate
        triggered_breakers = []
        if state.volume_divergence:
            triggered_breakers.append(CircuitBreakerType.VOLUME_DIVERGENCE)
        if state.cross_asset_lag:
            triggered_breakers.append(CircuitBreakerType.CROSS_ASSET_LAG)
        if state.liquidation_proximity:
            triggered_breakers.append(CircuitBreakerType.LIQUIDATION_PROXIMITY)
        if state.mtf_conflict:
            triggered_breakers.append(CircuitBreakerType.MTF_CONFLICT)
        
        if triggered_breakers:
            state.triggered = True
            state.breaker_type = triggered_breakers[0]  # Primary
            state.severity = len(triggered_breakers) / 4.0
            state.expires_at = time.time() + self.cooldown
            state.message = f"Breakers triggered: {[b.name for b in triggered_breakers]}"
            
            logger.warning(f"Circuit breaker triggered for {asset}: {state.message}")
        
        self._state[asset] = state
        return state
    
    def _check_volume_divergence(self, asset: str, action: str) -> bool:
        """
        Check volume-trend divergence.
        
        Prohibit "catching knives" if:
        - Price is falling AND
        - Volume is shrinking (< 70% of MA)
        """
        volumes = list(self._volume_history.get(asset, []))
        returns = list(self._returns.get(asset, []))
        
        if len(volumes) < self.volume_lookback or len(returns) < 5:
            return False
        
        # Only check for BUY in downtrend (knife-catching)
        if action != "BUY":
            return False
        
        recent_returns = returns[-5:]
        recent_volumes = volumes[-5:]
        ma_volume = np.mean(volumes[-self.volume_lookback:])
        
        # Price falling
        price_falling = np.mean(recent_returns) < -0.001  # -0.1% average
        
        # Volume shrinking
        current_vol = np.mean(recent_volumes)
        volume_shrinking = current_vol < ma_volume * self.volume_shrink_threshold
        
        return price_falling and volume_shrinking
    
    def _check_cross_asset_lag(
        self,
        asset: str,
        cross_returns: Dict[str, float]
    ) -> bool:
        """
        Check cross-asset lag (regime hysteresis).
        
        Trigger if BTC leads while SOL lags at resistance.
        """
        if asset != 'SOL':
            return False
        
        btc_ret = cross_returns.get('BTC', 0)
        sol_ret = cross_returns.get('SOL', 0)
        
        # BTC up significantly, SOL lagging
        btc_leading = btc_ret > self.cross_asset_threshold
        sol_lagging = sol_ret < btc_ret * 0.3  # SOL moving < 30% of BTC
        
        return btc_leading and sol_lagging
    
    def _check_liquidation_proximity(
        self,
        asset: str,
        price: float,
        action: str
    ) -> bool:
        """
        Check if price is near major liquidation clusters.
        
        Veto buy orders within 1% of short liquidation clusters
        (avoiding fake breakouts).
        """
        clusters = self._liquidation_clusters.get(asset, [])
        if not clusters:
            return False
        
        for cluster in clusters:
            # For BUY: check short liquidation clusters above price
            if action == "BUY" and cluster.side == "SHORT":
                distance = (cluster.price - price) / price
                if 0 < distance < self.liquidation_proximity:
                    logger.debug(
                        f"Liquidation proximity: {asset} price {price:.2f} "
                        f"within {distance*100:.2f}% of SHORT cluster at {cluster.price:.2f}"
                    )
                    return True
            
            # For SELL: check long liquidation clusters below price
            if action == "SELL" and cluster.side == "LONG":
                distance = (price - cluster.price) / price
                if 0 < distance < self.liquidation_proximity:
                    return True
        
        return False
    
    def get_state(self, asset: str) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state.get(asset, CircuitBreakerState())
    
    def reset(self, asset: str):
        """Reset circuit breaker for asset."""
        if asset in self._state:
            self._state[asset] = CircuitBreakerState()


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATED BITBEAST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class BitBeastEngine:
    """
    Integrated BitBeast Enhancement Engine.
    
    Combines all enhancements:
    - SNR Filter (Ants vs Train)
    - Dynamic Pyramiding (20-15-15)
    - MTF Tensor Fusion
    - Circuit Breakers
    """
    
    def __init__(self):
        self.snr_filter = SNRFilter()
        self.pyramid_manager = DynamicPyramidManager()
        self.mtf_fusion = MTFTensorFusion()
        self.circuit_breakers = CircuitBreakerSystem()
        
        logger.info("BitBeastEngine initialized with all enhancements")
    
    def process_signal(
        self,
        asset: str,
        signal_direction: str,
        signal_confidence: float,
        price: float,
        high: float,
        low: float,
        volume: float,
        current_bar: int,
        mtf_features: Dict[str, np.ndarray] = None,
        cross_asset_returns: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Process trading signal through all BitBeast enhancements.
        
        Returns comprehensive decision with all filter results.
        """
        result = {
            'asset': asset,
            'original_signal': signal_direction,
            'original_confidence': signal_confidence,
            'timestamp': time.time(),
            'approved': False,
            'filters': {},
            'pyramid': None,
            'mtf': None,
            'circuit_breaker': None,
        }
        
        # 1. Update histories
        self.snr_filter.update(asset, price, high, low, volume)
        self.circuit_breakers.update(asset, price, volume)
        
        # 2. Update MTF features if provided
        if mtf_features:
            for tf, features in mtf_features.items():
                self.mtf_fusion.update_timeframe(asset, tf, features)
        
        # 3. Fuse MTF tensor
        mtf_tensor = self.mtf_fusion.fuse(asset)
        result['mtf'] = {
            'alignment_score': mtf_tensor.mtf_alignment_score,
            'dominant_timeframe': mtf_tensor.dominant_timeframe,
            'conflict': mtf_tensor.conflict_detected,
            'fused_shape': mtf_tensor.fused.shape,
        }
        
        # 4. Check circuit breakers
        action = "BUY" if signal_direction == "LONG" else "SELL"
        cb_state = self.circuit_breakers.check_all(
            asset, price, action, mtf_tensor, cross_asset_returns
        )
        result['circuit_breaker'] = {
            'triggered': cb_state.triggered,
            'type': cb_state.breaker_type.name if cb_state.breaker_type else None,
            'severity': cb_state.severity,
            'message': cb_state.message,
        }
        
        if cb_state.triggered:
            result['approved'] = False
            result['rejection_reason'] = f"Circuit breaker: {cb_state.message}"
            return result
        
        # 5. Apply SNR filter
        snr_result = self.snr_filter.filter_signal(asset, signal_direction, signal_confidence)
        result['filters']['snr'] = {
            'passed': snr_result.passed,
            'pattern_ratio': snr_result.pattern_ratio,
            'reversal_weight': snr_result.reversal_weight,
            'reason': snr_result.reason,
        }
        
        if not snr_result.passed:
            result['approved'] = False
            result['rejection_reason'] = f"SNR filter: {snr_result.reason}"
            return result
        
        # Adjust confidence by SNR weight
        adjusted_confidence = signal_confidence * snr_result.reversal_weight
        result['adjusted_confidence'] = adjusted_confidence
        
        # 6. Check pyramid
        plan = self.pyramid_manager.get_plan(asset)
        if plan is None and adjusted_confidence >= CONFIDENCE_THRESHOLDS[0]:
            # Create new pyramid plan
            plan = self.pyramid_manager.create_plan(
                asset, signal_direction, 1.0, current_bar  # Normalized size
            )
        
        if plan:
            should_fill, size, reason = self.pyramid_manager.should_fill_tranche(
                asset, adjusted_confidence, current_bar, price
            )
            result['pyramid'] = {
                'plan_exists': True,
                'current_state': plan.current_state.name,
                'filled_tranches': plan.filled_tranches,
                'should_fill': should_fill,
                'fill_size': size,
                'fill_reason': reason,
                'avg_entry': plan.avg_entry_price,
            }
            
            if should_fill:
                result['approved'] = True
                result['approved_size'] = size
                result['execution_note'] = f"Pyramid tranche {plan.filled_tranches + 1}"
            else:
                result['approved'] = False
                result['rejection_reason'] = f"Pyramid: {reason}"
        else:
            result['pyramid'] = {'plan_exists': False}
            result['approved'] = False
            result['rejection_reason'] = "Confidence below minimum threshold"
        
        return result
    
    def record_execution(
        self,
        asset: str,
        size: float,
        price: float,
        current_bar: int
    ):
        """Record a successful execution."""
        self.pyramid_manager.record_fill(asset, size, price, current_bar)
    
    def get_mtf_tensor(self, asset: str) -> MTFTensor:
        """Get fused MTF tensor for asset."""
        return self.mtf_fusion.fuse(asset)
    
    def get_trend_analysis(self, asset: str) -> Optional[TrendAnalysis]:
        """Get trend analysis for asset."""
        return self.snr_filter.get_trend_analysis(asset)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Main engine
    'BitBeastEngine',
    
    # Components
    'SNRFilter',
    'DynamicPyramidManager',
    'MTFTensorFusion',
    'CircuitBreakerSystem',
    
    # Data structures
    'TrendState',
    'TrendAnalysis',
    'SNRFilterResult',
    'PyramidPlan',
    'PyramidState',
    'MTFTensor',
    'CircuitBreakerState',
    'CircuitBreakerType',
    'LiquidationCluster',
    
    # Constants
    'MIN_PATTERN_DURATION_RATIO',
    'PYRAMID_TRANCHES',
    'CONFIDENCE_THRESHOLDS',
    'MTF_TIMEFRAMES',
    'TOTAL_MTF_DIM',
]


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("BitBeast Enhancements Test")
    print("=" * 70)
    
    # Initialize engine
    engine = BitBeastEngine()
    
    # Simulate market data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    print("\n[1] SNR Filter Test")
    print("-" * 50)
    for i in range(50, 100):
        engine.snr_filter.update(
            'BTC',
            price=prices[i],
            high=prices[i] * 1.005,
            low=prices[i] * 0.995,
            volume=1000000 * (1 + np.random.rand())
        )
    
    analysis = engine.get_trend_analysis('BTC')
    if analysis:
        print(f"Trend State: {analysis.state.name}")
        print(f"Duration: {analysis.duration_bars} bars")
        print(f"Pattern Ratio: {analysis.pattern_duration_ratio:.2f}")
    
    # Test reversal signal
    snr_result = engine.snr_filter.filter_signal('BTC', 'LONG', 0.75)
    print(f"SNR Filter: passed={snr_result.passed}, weight={snr_result.reversal_weight:.2f}")
    print(f"Reason: {snr_result.reason}")
    
    print("\n[2] Dynamic Pyramiding Test")
    print("-" * 50)
    plan = engine.pyramid_manager.create_plan('BTC', 'LONG', 1.0, 100)
    print(f"Created plan: {plan.tranches} ({sum(plan.tranches)*100}% total)")
    
    # Simulate tranche fills
    for bar in range(100, 120):
        confidence = 0.55 + bar * 0.005  # Increasing confidence
        should_fill, size, reason = engine.pyramid_manager.should_fill_tranche(
            'BTC', confidence, bar, 100.0
        )
        if should_fill:
            print(f"Bar {bar}: Fill tranche {plan.filled_tranches+1}, size={size:.2f}, conf={confidence:.2f}")
            engine.pyramid_manager.record_fill('BTC', size, 100.0, bar)
    
    print(f"Final state: {plan.current_state.name}, avg_entry={plan.avg_entry_price:.2f}")
    
    print("\n[3] MTF Tensor Fusion Test")
    print("-" * 50)
    
    # Create sample features
    features_1m = engine.mtf_fusion.create_micro_risk_features(
        spread_bps=5.0,
        volatility_1m=0.02,
        order_flow_imbalance=0.3,
        bid_depth=1000000,
        ask_depth=800000
    )
    engine.mtf_fusion.update_timeframe('BTC', '1m', features_1m)
    
    features_15m = engine.mtf_fusion.create_confirmation_features(
        momentum_15m=0.5,
        volume_profile=np.random.rand(10),
        vwap_deviation=0.002,
        rsi_15m=65,
        macd_signal=0.3
    )
    engine.mtf_fusion.update_timeframe('BTC', '15m', features_15m)
    
    features_4h = engine.mtf_fusion.create_trend_features(
        trend_strength=0.7,
        support_distance=0.03,
        resistance_distance=0.05,
        higher_highs=3,
        lower_lows=1,
        trend_duration=20
    )
    engine.mtf_fusion.update_timeframe('BTC', '4H', features_4h)
    
    mtf = engine.get_mtf_tensor('BTC')
    print(f"Fused tensor shape: {mtf.fused.shape}")
    print(f"Alignment score: {mtf.mtf_alignment_score:.2f}")
    print(f"Dominant TF: {mtf.dominant_timeframe}")
    print(f"Conflict detected: {mtf.conflict_detected}")
    
    print("\n[4] Circuit Breaker Test")
    print("-" * 50)
    
    # Simulate falling price with shrinking volume
    for i in range(10):
        engine.circuit_breakers.update('BTC', 100 - i, 1000000 * (0.9 ** i))
    
    cb_state = engine.circuit_breakers.check_all(
        'BTC',
        current_price=90,
        proposed_action='BUY',
        mtf_tensor=mtf,
        cross_asset_returns={'BTC': 0.03, 'SOL': 0.005}
    )
    print(f"Triggered: {cb_state.triggered}")
    print(f"Volume Divergence: {cb_state.volume_divergence}")
    print(f"Cross-Asset Lag: {cb_state.cross_asset_lag}")
    print(f"Message: {cb_state.message}")
    
    print("\n[5] Integrated Signal Processing Test")
    print("-" * 50)
    
    # Reset for clean test
    engine = BitBeastEngine()
    
    # Warm up with data
    for i in range(60):
        engine.process_signal(
            asset='ETH',
            signal_direction='LONG',
            signal_confidence=0.5,
            price=3000 + i * 10,
            high=3005 + i * 10,
            low=2995 + i * 10,
            volume=5000000,
            current_bar=i
        )
    
    # Test signal
    result = engine.process_signal(
        asset='ETH',
        signal_direction='LONG',
        signal_confidence=0.75,
        price=3600,
        high=3610,
        low=3590,
        volume=6000000,
        current_bar=60,
        mtf_features={
            '1m': np.random.randn(64),
            '15m': np.random.randn(32),
            '4H': np.random.randn(32),
        }
    )
    
    print(f"Signal Approved: {result['approved']}")
    print(f"Adjusted Confidence: {result.get('adjusted_confidence', 'N/A')}")
    print(f"SNR Filter: {result['filters']['snr']}")
    print(f"Pyramid: {result['pyramid']}")
    print(f"MTF: {result['mtf']}")
    print(f"Circuit Breaker: {result['circuit_breaker']}")
    
    if not result['approved']:
        print(f"Rejection Reason: {result.get('rejection_reason', 'Unknown')}")
    
    print("\n✓ All BitBeast enhancement tests completed!")

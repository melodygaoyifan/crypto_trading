"""
================================================================================
HMATS v5.0.5 - Enhanced Adaptive Stop-Loss Module
================================================================================

v5.0.5 Enhancements:
1. Multiple rolling ATR windows (14d, 30d, 60d)
2. Calmar ratio based stop adjustment
3. Volatility percentile ranking for dynamic scaling
4. Enhanced breakeven and trailing stop logic
5. Full backward compatibility with v4.5/v5.0

Stop Distance Logic (v5.0.5 Enhanced):
    base_atr = weighted_average(ATR_14d, ATR_30d, ATR_60d)
    vol_scale = volatility_percentile_scale()
    calmar_adj = calmar_ratio_adjustment()
    
    initial_stop = max(
        atr_multiplier x base_atr x vol_scale x calmar_adj,
        entry_price x min_stop_pct
    )

Breakeven Enhancement:
    - Standard: Move to breakeven at 2R
    - Calmar-adjusted: Earlier breakeven in high-vol regimes
    - Volatility percentile: Adjust R-trigger based on vol ranking

Trailing Enhancement:
    - Dynamic trail distance based on current ATR vs entry ATR
    - Vol expansion = wider trail, vol contraction = tighter trail

Author: HMATS v5.0.5 Patch
Version: 5.0.5
"""

import logging
import threading
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class StopType(Enum):
    """Stop-loss type classification"""
    INITIAL = "initial"
    BREAKEVEN = "breakeven"
    TRAILING = "trailing"
    MANUAL = "manual"
    CALMAR_ADJUSTED = "calmar_adjusted"  # v5.0.5


class PositionDirection(Enum):
    """Trade direction"""
    LONG = "long"
    SHORT = "short"


class StopTriggerReason(Enum):
    """Why the stop was triggered"""
    PRICE_HIT = "price_hit"
    TIME_DECAY = "time_decay"
    VOLATILITY_SPIKE = "vol_spike"
    MANUAL_EXIT = "manual_exit"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CALMAR_BREACH = "calmar_breach"  # v5.0.5


class VolatilityRegime(Enum):
    """v5.0.5: Volatility regime classification"""
    LOW = "low"           # < 25th percentile
    NORMAL = "normal"     # 25-75th percentile
    HIGH = "high"         # 75-90th percentile
    EXTREME = "extreme"   # > 90th percentile


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MultiWindowATRState:
    """
    v5.0.5: Multi-window ATR state for an asset.
    
    Tracks ATR across multiple lookback periods for robust volatility estimation.
    """
    symbol: str
    
    # Multi-window ATR values
    atr_14d: float = 0.0
    atr_30d: float = 0.0
    atr_60d: float = 0.0
    
    # Weighted composite ATR
    composite_atr: float = 0.0
    composite_atr_pct: float = 0.0
    
    # Volatility percentile (vs 252-day history)
    vol_percentile: float = 50.0
    vol_regime: VolatilityRegime = VolatilityRegime.NORMAL
    
    # Rolling windows
    true_ranges_14d: deque = field(default_factory=lambda: deque(maxlen=14))
    true_ranges_30d: deque = field(default_factory=lambda: deque(maxlen=30))
    true_ranges_60d: deque = field(default_factory=lambda: deque(maxlen=60))
    historical_atr: deque = field(default_factory=lambda: deque(maxlen=252))
    
    # Timestamps
    last_update: Optional[datetime] = None
    last_close: Optional[float] = None
    last_high: Optional[float] = None
    last_low: Optional[float] = None
    
    def __post_init__(self):
        if not isinstance(self.true_ranges_14d, deque):
            self.true_ranges_14d = deque(maxlen=14)
        if not isinstance(self.true_ranges_30d, deque):
            self.true_ranges_30d = deque(maxlen=30)
        if not isinstance(self.true_ranges_60d, deque):
            self.true_ranges_60d = deque(maxlen=60)
        if not isinstance(self.historical_atr, deque):
            self.historical_atr = deque(maxlen=252)


@dataclass
class CalmarMetrics:
    """
    v5.0.5: Calmar ratio metrics for stop adjustment.
    """
    calmar_ratio: float = 0.0          # Return / Max Drawdown
    max_drawdown_pct: float = 0.0
    rolling_return_pct: float = 0.0
    drawdown_duration_days: int = 0
    recovery_factor: float = 1.0       # > 1 = recovering, < 1 = declining
    
    @property
    def is_high_risk(self) -> bool:
        """Check if Calmar indicates high risk."""
        return self.calmar_ratio < 0.5 or self.max_drawdown_pct > 20


@dataclass
class StopState:
    """Complete stop-loss state for a position (v5.0.5 enhanced)."""
    symbol: str
    direction: PositionDirection
    entry_price: float
    entry_time: datetime
    position_size: float
    
    # Initial risk calculation
    initial_stop: float
    initial_risk_pct: float
    initial_risk_usd: float
    
    # Current stop state
    current_stop: float
    stop_type: StopType = StopType.INITIAL
    
    # R-multiple tracking
    current_r_multiple: float = 0.0
    highest_r_multiple: float = 0.0
    
    # Breakeven tracking
    breakeven_triggered: bool = False
    breakeven_trigger_time: Optional[datetime] = None
    
    # Trailing state
    trailing_active: bool = False
    trailing_distance: float = 0.0
    
    # ATR reference
    entry_atr: float = 0.0
    entry_atr_pct: float = 0.0
    
    # v5.0.5 New fields
    entry_vol_percentile: float = 50.0
    entry_calmar: float = 0.0
    dynamic_r_trigger: float = 2.0      # Adjusted breakeven trigger
    dynamic_trail_multiplier: float = 1.0
    vol_regime_at_entry: VolatilityRegime = VolatilityRegime.NORMAL
    
    # Timestamps
    last_update: Optional[datetime] = None
    stop_adjustments: List[Dict] = field(default_factory=list)


@dataclass
class StopConfiguration:
    """
    v5.0.5: Configurable stop parameters.
    
    Allows customization of stop behavior per-trade or globally.
    """
    # ATR settings
    atr_multiplier: float = 2.0
    min_stop_pct: float = 3.0
    
    # Window weights for composite ATR [14d, 30d, 60d]
    atr_window_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    
    # Breakeven settings
    breakeven_r_trigger: float = 2.0
    use_calmar_adjustment: bool = True
    use_vol_percentile_adjustment: bool = True
    
    # Trailing settings
    trailing_r_offset: float = 0.5
    dynamic_trail_enabled: bool = True
    
    # Vol scaling
    vol_scale_low: float = 0.8         # Scale for low vol
    vol_scale_high: float = 1.2        # Scale for high vol
    vol_scale_extreme: float = 1.5     # Scale for extreme vol
    
    # Calmar thresholds
    calmar_tighten_threshold: float = 0.5   # Below this, tighten stops
    calmar_loosen_threshold: float = 2.0    # Above this, can loosen
    
    # Time
    max_holding_hours: int = 168


# ============================================================================
# Multi-Window ATR Calculator (v5.0.5)
# ============================================================================

class MultiWindowATRCalculator:
    """
    v5.0.5: ATR calculator with multiple rolling windows.
    
    Provides:
    - 14-day ATR (short-term volatility)
    - 30-day ATR (medium-term, standard)
    - 60-day ATR (longer-term baseline)
    - Composite weighted ATR
    - Volatility percentile ranking
    """
    
    def __init__(
        self,
        weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    ):
        """
        Initialize multi-window ATR calculator.
        
        Args:
            weights: Weights for [14d, 30d, 60d] in composite calculation
        """
        self.weights = weights
        self._states: Dict[str, MultiWindowATRState] = {}
        self._lock = threading.RLock()
        
        logger.info(f"MultiWindowATRCalculator initialized with weights {weights}")
    
    def update(
        self,
        symbol: str,
        high: float,
        low: float,
        close: float,
        timestamp: Optional[datetime] = None
    ) -> MultiWindowATRState:
        """
        Update ATR with new OHLC data.
        
        Returns:
            Updated MultiWindowATRState
        """
        with self._lock:
            if symbol not in self._states:
                self._states[symbol] = MultiWindowATRState(symbol=symbol)
            
            state = self._states[symbol]
            ts = timestamp or datetime.now()
            
            # Calculate True Range
            if state.last_close is not None:
                tr = max(
                    high - low,
                    abs(high - state.last_close),
                    abs(low - state.last_close)
                )
            else:
                tr = high - low
            
            # Update all windows
            state.true_ranges_14d.append(tr)
            state.true_ranges_30d.append(tr)
            state.true_ranges_60d.append(tr)
            
            # Calculate window ATRs
            if len(state.true_ranges_14d) > 0:
                state.atr_14d = sum(state.true_ranges_14d) / len(state.true_ranges_14d)
            
            if len(state.true_ranges_30d) > 0:
                state.atr_30d = sum(state.true_ranges_30d) / len(state.true_ranges_30d)
            
            if len(state.true_ranges_60d) > 0:
                state.atr_60d = sum(state.true_ranges_60d) / len(state.true_ranges_60d)
            
            # Calculate composite ATR
            atrs = [state.atr_14d, state.atr_30d, state.atr_60d]
            valid_atrs = [(a, w) for a, w in zip(atrs, self.weights) if a > 0]
            
            if valid_atrs:
                total_weight = sum(w for _, w in valid_atrs)
                state.composite_atr = sum(a * w for a, w in valid_atrs) / total_weight
                state.composite_atr_pct = (state.composite_atr / close) * 100 if close > 0 else 0
            
            # Track historical ATR for percentile
            if state.atr_30d > 0:
                state.historical_atr.append(state.atr_30d)
            
            # Calculate volatility percentile
            if len(state.historical_atr) >= 20:
                state.vol_percentile = self._calculate_percentile(
                    state.atr_30d, list(state.historical_atr)
                )
                state.vol_regime = self._classify_regime(state.vol_percentile)
            
            # Update state
            state.last_close = close
            state.last_high = high
            state.last_low = low
            state.last_update = ts
            
            return state
    
    def _calculate_percentile(self, value: float, history: List[float]) -> float:
        """Calculate percentile rank of value in history."""
        if not history:
            return 50.0
        
        sorted_hist = sorted(history)
        n = len(sorted_hist)
        
        # Count values less than current
        rank = sum(1 for v in sorted_hist if v < value)
        percentile = (rank / n) * 100
        
        return min(99.9, max(0.1, percentile))
    
    def _classify_regime(self, percentile: float) -> VolatilityRegime:
        """Classify volatility regime based on percentile."""
        if percentile < 25:
            return VolatilityRegime.LOW
        elif percentile < 75:
            return VolatilityRegime.NORMAL
        elif percentile < 90:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def get_state(self, symbol: str) -> Optional[MultiWindowATRState]:
        """Get current ATR state for symbol."""
        with self._lock:
            return self._states.get(symbol)
    
    def get_composite_atr(self, symbol: str) -> float:
        """Get composite ATR value."""
        state = self.get_state(symbol)
        return state.composite_atr if state else 0.0
    
    def get_vol_percentile(self, symbol: str) -> float:
        """Get volatility percentile."""
        state = self.get_state(symbol)
        return state.vol_percentile if state else 50.0
    
    def get_vol_regime(self, symbol: str) -> VolatilityRegime:
        """Get current volatility regime."""
        state = self.get_state(symbol)
        return state.vol_regime if state else VolatilityRegime.NORMAL
    
    def bulk_update(self, symbol: str, ohlc_data: List[Dict]) -> MultiWindowATRState:
        """Bulk update with historical data."""
        state = None
        for bar in ohlc_data:
            state = self.update(
                symbol=symbol,
                high=bar['high'],
                low=bar['low'],
                close=bar['close'],
                timestamp=bar.get('timestamp')
            )
        return state


# ============================================================================
# Calmar Ratio Calculator (v5.0.5)
# ============================================================================

class CalmarCalculator:
    """
    v5.0.5: Calculate Calmar ratio for stop adjustment.
    
    Calmar = Annualized Return / Max Drawdown
    
    Used to:
    - Tighten stops when performance deteriorates
    - Adjust breakeven triggers based on risk/reward history
    """
    
    def __init__(self, lookback_days: int = 90):
        self.lookback_days = lookback_days
        self._equity_curves: Dict[str, deque] = {}
        self._lock = threading.Lock()
    
    def update(
        self,
        symbol: str,
        equity_value: float,
        timestamp: Optional[datetime] = None
    ) -> CalmarMetrics:
        """
        Update equity curve and calculate Calmar metrics.
        
        Args:
            symbol: Trading symbol or strategy ID
            equity_value: Current equity value
            timestamp: Observation timestamp
            
        Returns:
            CalmarMetrics with ratio and drawdown info
        """
        with self._lock:
            if symbol not in self._equity_curves:
                self._equity_curves[symbol] = deque(maxlen=self.lookback_days)
            
            curve = self._equity_curves[symbol]
            ts = timestamp or datetime.now()
            curve.append({"value": equity_value, "timestamp": ts})
            
            return self._calculate_metrics(list(curve))
    
    def _calculate_metrics(self, curve: List[Dict]) -> CalmarMetrics:
        """Calculate Calmar metrics from equity curve."""
        if len(curve) < 5:
            return CalmarMetrics()
        
        values = [p["value"] for p in curve]
        
        # Calculate return
        start_val = values[0]
        end_val = values[-1]
        total_return = (end_val - start_val) / start_val if start_val > 0 else 0
        
        # Annualize (assuming daily data)
        n_days = len(values)
        annual_return = total_return * (252 / n_days)
        
        # Calculate max drawdown
        peak = values[0]
        max_dd = 0.0
        dd_start = 0
        current_dd_duration = 0
        max_dd_duration = 0
        
        for i, val in enumerate(values):
            if val > peak:
                peak = val
                current_dd_duration = 0
            else:
                dd = (peak - val) / peak
                if dd > max_dd:
                    max_dd = dd
                current_dd_duration += 1
                if current_dd_duration > max_dd_duration:
                    max_dd_duration = current_dd_duration
        
        # Calmar ratio
        calmar = annual_return / max_dd if max_dd > 0.01 else 10.0  # Cap at 10
        
        # Recovery factor (recent trend)
        recent_n = min(10, len(values) - 1)
        if recent_n > 0:
            recent_return = (values[-1] - values[-recent_n-1]) / values[-recent_n-1]
            recovery_factor = 1.0 + recent_return * 10  # Amplify for sensitivity
        else:
            recovery_factor = 1.0
        
        return CalmarMetrics(
            calmar_ratio=calmar,
            max_drawdown_pct=max_dd * 100,
            rolling_return_pct=total_return * 100,
            drawdown_duration_days=max_dd_duration,
            recovery_factor=recovery_factor
        )
    
    def get_metrics(self, symbol: str) -> CalmarMetrics:
        """Get current Calmar metrics for symbol."""
        with self._lock:
            curve = self._equity_curves.get(symbol)
            if not curve:
                return CalmarMetrics()
            return self._calculate_metrics(list(curve))


# ============================================================================
# Enhanced Adaptive Stop Manager (v5.0.5)
# ============================================================================

class EnhancedAdaptiveStopManager:
    """
    v5.0.5: Enhanced adaptive stop-loss manager.
    
    Features:
    - Multi-window ATR for robust volatility estimation
    - Calmar ratio based stop adjustment
    - Volatility percentile scaling
    - Dynamic breakeven and trailing logic
    """
    
    def __init__(
        self,
        config: Optional[StopConfiguration] = None,
        atr_calculator: Optional[MultiWindowATRCalculator] = None,
        calmar_calculator: Optional[CalmarCalculator] = None
    ):
        """
        Initialize enhanced stop manager.
        
        Args:
            config: Stop configuration (uses defaults if None)
            atr_calculator: Multi-window ATR calculator
            calmar_calculator: Calmar ratio calculator
        """
        self.config = config or StopConfiguration()
        self.atr_calculator = atr_calculator or MultiWindowATRCalculator(
            weights=self.config.atr_window_weights
        )
        self.calmar_calculator = calmar_calculator or CalmarCalculator()
        
        self._positions: Dict[str, StopState] = {}
        self._lock = threading.RLock()
        
        logger.info(
            f"EnhancedAdaptiveStopManager initialized: "
            f"ATRx{self.config.atr_multiplier}, min={self.config.min_stop_pct}%, "
            f"breakeven@{self.config.breakeven_r_trigger}R"
        )
    
    def calculate_initial_stop(
        self,
        symbol: str,
        entry_price: float,
        direction: PositionDirection,
        calmar_metrics: Optional[CalmarMetrics] = None
    ) -> Tuple[float, float, str, float]:
        """
        Calculate initial stop price with v5.0.5 enhancements.
        
        Returns:
            (stop_price, risk_pct, method_used, dynamic_r_trigger)
        """
        # Get multi-window ATR state
        atr_state = self.atr_calculator.get_state(symbol)
        
        if atr_state is None or atr_state.composite_atr == 0:
            # Fallback to minimum percentage
            stop_distance = entry_price * (self.config.min_stop_pct / 100)
            risk_pct = self.config.min_stop_pct
            method = f"MIN_PCT ({self.config.min_stop_pct}%)"
            dynamic_r = self.config.breakeven_r_trigger
        else:
            # v5.0.5: Calculate base ATR stop with volatility scaling
            base_atr = atr_state.composite_atr
            
            # Volatility percentile scaling
            vol_scale = self._get_vol_scale(atr_state.vol_regime)
            
            # Calmar adjustment
            calmar_adj = 1.0
            if self.config.use_calmar_adjustment and calmar_metrics:
                calmar_adj = self._get_calmar_adjustment(calmar_metrics)
            
            # Final stop distance
            atr_stop_distance = (
                self.config.atr_multiplier * 
                base_atr * 
                vol_scale * 
                calmar_adj
            )
            atr_stop_pct = (atr_stop_distance / entry_price) * 100
            
            # Minimum percentage check
            min_stop_distance = entry_price * (self.config.min_stop_pct / 100)
            
            if atr_stop_distance >= min_stop_distance:
                stop_distance = atr_stop_distance
                risk_pct = atr_stop_pct
                method = (
                    f"ATR_COMPOSITEx{self.config.atr_multiplier:.1f} "
                    f"vol_scale={vol_scale:.2f} calmar_adj={calmar_adj:.2f} "
                    f"({atr_stop_pct:.2f}%)"
                )
            else:
                stop_distance = min_stop_distance
                risk_pct = self.config.min_stop_pct
                method = f"MIN_PCT ({self.config.min_stop_pct}%)"
            
            # v5.0.5: Dynamic R trigger based on volatility
            dynamic_r = self._calculate_dynamic_r_trigger(
                atr_state.vol_regime, calmar_metrics
            )
        
        # Calculate stop price
        if direction == PositionDirection.LONG:
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance
        
        return stop_price, risk_pct, method, dynamic_r
    
    def _get_vol_scale(self, regime: VolatilityRegime) -> float:
        """Get volatility scaling factor based on regime."""
        scales = {
            VolatilityRegime.LOW: self.config.vol_scale_low,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.HIGH: self.config.vol_scale_high,
            VolatilityRegime.EXTREME: self.config.vol_scale_extreme
        }
        return scales.get(regime, 1.0)
    
    def _get_calmar_adjustment(self, metrics: CalmarMetrics) -> float:
        """
        Get Calmar-based stop adjustment.
        
        Low Calmar (poor performance) = tighter stops (lower multiplier)
        High Calmar (good performance) = can maintain wider stops
        """
        if metrics.calmar_ratio < self.config.calmar_tighten_threshold:
            # Poor performance: tighten stops
            return 0.8
        elif metrics.calmar_ratio > self.config.calmar_loosen_threshold:
            # Good performance: can be slightly wider
            return 1.1
        else:
            return 1.0
    
    def _calculate_dynamic_r_trigger(
        self,
        vol_regime: VolatilityRegime,
        calmar_metrics: Optional[CalmarMetrics]
    ) -> float:
        """
        v5.0.5: Calculate dynamic R-multiple trigger for breakeven.
        
        High volatility / poor Calmar = earlier breakeven
        Low volatility / good Calmar = standard breakeven
        """
        base_r = self.config.breakeven_r_trigger
        
        # Vol regime adjustment
        vol_adj = {
            VolatilityRegime.LOW: 0.0,
            VolatilityRegime.NORMAL: 0.0,
            VolatilityRegime.HIGH: -0.25,
            VolatilityRegime.EXTREME: -0.5
        }
        
        r_trigger = base_r + vol_adj.get(vol_regime, 0.0)
        
        # Calmar adjustment
        if calmar_metrics and calmar_metrics.calmar_ratio < 0.5:
            r_trigger -= 0.25  # Earlier breakeven on poor performance
        
        return max(1.0, r_trigger)  # Minimum 1R
    
    def register_position(
        self,
        symbol: str,
        entry_price: float,
        position_size: float,
        direction: PositionDirection,
        entry_time: Optional[datetime] = None,
        calmar_metrics: Optional[CalmarMetrics] = None
    ) -> StopState:
        """
        Register a new position with v5.0.5 enhanced stop calculation.
        """
        with self._lock:
            ts = entry_time or datetime.now()
            
            # Get ATR state for recording
            atr_state = self.atr_calculator.get_state(symbol)
            
            # Calculate initial stop
            initial_stop, risk_pct, method, dynamic_r = self.calculate_initial_stop(
                symbol, entry_price, direction, calmar_metrics
            )
            
            # Risk in USD
            risk_usd = position_size * entry_price * (risk_pct / 100)
            
            # Create enhanced stop state
            state = StopState(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                entry_time=ts,
                position_size=position_size,
                initial_stop=initial_stop,
                initial_risk_pct=risk_pct,
                initial_risk_usd=risk_usd,
                current_stop=initial_stop,
                stop_type=StopType.INITIAL,
                entry_atr=atr_state.composite_atr if atr_state else 0.0,
                entry_atr_pct=atr_state.composite_atr_pct if atr_state else 0.0,
                entry_vol_percentile=atr_state.vol_percentile if atr_state else 50.0,
                entry_calmar=calmar_metrics.calmar_ratio if calmar_metrics else 0.0,
                dynamic_r_trigger=dynamic_r,
                vol_regime_at_entry=atr_state.vol_regime if atr_state else VolatilityRegime.NORMAL,
                last_update=ts
            )
            
            # Record adjustment
            state.stop_adjustments.append({
                'timestamp': ts.isoformat(),
                'old_stop': None,
                'new_stop': initial_stop,
                'reason': f"Initial stop: {method}",
                'stop_type': StopType.INITIAL.value,
                'r_multiple': 0.0,
                'price': entry_price
            })
            
            self._positions[symbol] = state
            
            logger.info(
                f"Position registered: {symbol} {direction.value} "
                f"entry={entry_price:.4f}, stop={initial_stop:.4f}, "
                f"risk={risk_pct:.2f}%, dynamic_R_trigger={dynamic_r:.2f}"
            )
            
            return state
    
    def update_stop(
        self,
        symbol: str,
        current_price: float,
        timestamp: Optional[datetime] = None
    ) -> Optional[StopState]:
        """
        Update stop-loss with v5.0.5 enhanced logic.
        
        Includes:
        - Dynamic R-trigger for breakeven
        - ATR-adjusted trailing distance
        - Volatility expansion/contraction handling
        """
        with self._lock:
            state = self._positions.get(symbol)
            if not state:
                return None
            
            ts = timestamp or datetime.now()
            old_stop = state.current_stop
            
            # Calculate R-multiple
            if state.direction == PositionDirection.LONG:
                profit = current_price - state.entry_price
            else:
                profit = state.entry_price - current_price
            
            initial_risk = abs(state.entry_price - state.initial_stop)
            r_multiple = profit / initial_risk if initial_risk > 0 else 0
            
            state.current_r_multiple = r_multiple
            state.highest_r_multiple = max(state.highest_r_multiple, r_multiple)
            
            # v5.0.5: Check dynamic breakeven trigger
            if not state.breakeven_triggered and r_multiple >= state.dynamic_r_trigger:
                self._trigger_breakeven(state, current_price, ts)
            
            # v5.0.5: Enhanced trailing with dynamic distance
            if state.breakeven_triggered and state.trailing_active:
                self._update_trailing_stop(state, current_price, ts)
            elif state.breakeven_triggered and not state.trailing_active:
                # Activate trailing
                self._activate_trailing(state, current_price, ts)
            
            state.last_update = ts
            return state
    
    def _trigger_breakeven(
        self,
        state: StopState,
        current_price: float,
        timestamp: datetime
    ) -> None:
        """Trigger breakeven stop."""
        old_stop = state.current_stop
        
        # Move to entry (or slightly better)
        if state.direction == PositionDirection.LONG:
            # For long, stop just below entry
            state.current_stop = state.entry_price * 0.999
        else:
            state.current_stop = state.entry_price * 1.001
        
        state.breakeven_triggered = True
        state.breakeven_trigger_time = timestamp
        state.stop_type = StopType.BREAKEVEN
        
        state.stop_adjustments.append({
            'timestamp': timestamp.isoformat(),
            'old_stop': old_stop,
            'new_stop': state.current_stop,
            'reason': f"Breakeven at {state.current_r_multiple:.2f}R (trigger={state.dynamic_r_trigger:.2f}R)",
            'stop_type': StopType.BREAKEVEN.value,
            'r_multiple': state.current_r_multiple,
            'price': current_price
        })
        
        logger.info(f"{state.symbol}: Breakeven triggered at {state.current_r_multiple:.2f}R")
    
    def _activate_trailing(
        self,
        state: StopState,
        current_price: float,
        timestamp: datetime
    ) -> None:
        """Activate trailing stop with dynamic distance."""
        state.trailing_active = True
        
        # v5.0.5: Calculate dynamic trail distance
        current_atr_state = self.atr_calculator.get_state(state.symbol)
        
        if current_atr_state and state.entry_atr > 0:
            # Compare current ATR to entry ATR
            atr_ratio = current_atr_state.composite_atr / state.entry_atr
            
            # If vol expanded, trail wider; if contracted, trail tighter
            state.dynamic_trail_multiplier = min(1.5, max(0.7, atr_ratio))
        else:
            state.dynamic_trail_multiplier = 1.0
        
        # Calculate trail distance
        base_trail = abs(state.entry_price - state.initial_stop) * self.config.trailing_r_offset
        state.trailing_distance = base_trail * state.dynamic_trail_multiplier
        
        # Set initial trailing stop
        if state.direction == PositionDirection.LONG:
            state.current_stop = current_price - state.trailing_distance
        else:
            state.current_stop = current_price + state.trailing_distance
        
        state.stop_type = StopType.TRAILING
        
        logger.debug(
            f"{state.symbol}: Trailing activated, distance={state.trailing_distance:.4f}, "
            f"multiplier={state.dynamic_trail_multiplier:.2f}"
        )
    
    def _update_trailing_stop(
        self,
        state: StopState,
        current_price: float,
        timestamp: datetime
    ) -> None:
        """Update trailing stop position."""
        old_stop = state.current_stop
        
        # v5.0.5: Optionally update trail distance based on current vol
        if self.config.dynamic_trail_enabled:
            current_atr_state = self.atr_calculator.get_state(state.symbol)
            if current_atr_state and state.entry_atr > 0:
                atr_ratio = current_atr_state.composite_atr / state.entry_atr
                state.dynamic_trail_multiplier = min(1.5, max(0.7, atr_ratio))
                base_trail = abs(state.entry_price - state.initial_stop) * self.config.trailing_r_offset
                state.trailing_distance = base_trail * state.dynamic_trail_multiplier
        
        # Calculate new stop
        if state.direction == PositionDirection.LONG:
            new_stop = current_price - state.trailing_distance
            if new_stop > state.current_stop:
                state.current_stop = new_stop
        else:
            new_stop = current_price + state.trailing_distance
            if new_stop < state.current_stop:
                state.current_stop = new_stop
        
        # Log if stop moved
        if state.current_stop != old_stop:
            state.stop_adjustments.append({
                'timestamp': timestamp.isoformat(),
                'old_stop': old_stop,
                'new_stop': state.current_stop,
                'reason': f"Trailing update (R={state.current_r_multiple:.2f})",
                'stop_type': StopType.TRAILING.value,
                'r_multiple': state.current_r_multiple,
                'price': current_price
            })
    
    def check_stop_hit(
        self,
        symbol: str,
        current_price: float
    ) -> Tuple[bool, Optional[StopTriggerReason], Optional[StopState]]:
        """Check if stop has been hit."""
        with self._lock:
            state = self._positions.get(symbol)
            if not state:
                return False, None, None
            
            hit = False
            reason = None
            
            if state.direction == PositionDirection.LONG:
                if current_price <= state.current_stop:
                    hit = True
                    reason = StopTriggerReason.PRICE_HIT
            else:
                if current_price >= state.current_stop:
                    hit = True
                    reason = StopTriggerReason.PRICE_HIT
            
            # Check time decay
            if not hit and self.config.max_holding_hours > 0:
                holding_hours = (datetime.now() - state.entry_time).total_seconds() / 3600
                if holding_hours > self.config.max_holding_hours:
                    hit = True
                    reason = StopTriggerReason.TIME_DECAY
            
            return hit, reason, state
    
    def get_stop_state(self, symbol: str) -> Optional[StopState]:
        """Get current stop state."""
        with self._lock:
            return self._positions.get(symbol)
    
    def remove_position(self, symbol: str) -> Optional[StopState]:
        """Remove position after exit."""
        with self._lock:
            return self._positions.pop(symbol, None)
    
    def get_position_summary(self, symbol: str) -> Dict:
        """Get summary of position stop state."""
        state = self.get_stop_state(symbol)
        if not state:
            return {}
        
        return {
            'symbol': state.symbol,
            'direction': state.direction.value,
            'entry_price': state.entry_price,
            'current_stop': state.current_stop,
            'stop_type': state.stop_type.value,
            'current_r': state.current_r_multiple,
            'highest_r': state.highest_r_multiple,
            'breakeven': state.breakeven_triggered,
            'trailing': state.trailing_active,
            'dynamic_r_trigger': state.dynamic_r_trigger,
            'trail_multiplier': state.dynamic_trail_multiplier,
            'vol_regime': state.vol_regime_at_entry.value,
            'adjustments': len(state.stop_adjustments)
        }


# ============================================================================
# Backward Compatibility Layer
# ============================================================================

class ATRCalculator(MultiWindowATRCalculator):
    """
    Backward compatible ATR calculator.
    
    Provides v4.5 interface while using v5.0.5 implementation.
    """
    
    def __init__(self, default_period: int = 30):
        # Use v5.0.5 implementation
        super().__init__(weights=(0.3, 0.5, 0.2))
        self.default_period = default_period
    
    def get_atr(self, symbol: str):
        """v4.5 compatible get_atr."""
        state = self.get_state(symbol)
        if not state:
            return None
        
        # Return object with atr_value and atr_pct
        class LegacyATRState:
            def __init__(self, s):
                self.atr_value = s.atr_30d  # Use 30d for compatibility
                self.atr_pct = (s.atr_30d / s.last_close * 100) if s.last_close else 0
                self.symbol = s.symbol
                self.period = 30
                self.true_ranges = s.true_ranges_30d
                self.last_update = s.last_update
        
        return LegacyATRState(state)
    
    def get_atr_value(self, symbol: str) -> float:
        """v4.5 compatible get_atr_value."""
        state = self.get_state(symbol)
        return state.atr_30d if state else 0.0
    
    def get_atr_pct(self, symbol: str) -> float:
        """v4.5 compatible get_atr_pct."""
        state = self.get_state(symbol)
        if state and state.last_close:
            return (state.atr_30d / state.last_close) * 100
        return 0.0


class AdaptiveStopManager(EnhancedAdaptiveStopManager):
    """
    Backward compatible adaptive stop manager.
    
    Provides v4.5 interface while using v5.0.5 implementation.
    """
    
    def __init__(
        self,
        atr_calculator: Optional[ATRCalculator] = None,
        atr_multiplier: float = 2.0,
        min_stop_pct: float = 3.0,
        breakeven_r_trigger: float = 2.0,
        trailing_r_offset: float = 0.5,
        max_holding_hours: int = 168,
        enable_volatility_adjustment: bool = True
    ):
        # Build config from v4.5 parameters
        config = StopConfiguration(
            atr_multiplier=atr_multiplier,
            min_stop_pct=min_stop_pct,
            breakeven_r_trigger=breakeven_r_trigger,
            trailing_r_offset=trailing_r_offset,
            max_holding_hours=max_holding_hours,
            use_calmar_adjustment=enable_volatility_adjustment,
            use_vol_percentile_adjustment=enable_volatility_adjustment
        )
        
        # Wrap legacy ATR calculator
        if atr_calculator and not isinstance(atr_calculator, MultiWindowATRCalculator):
            # Create new multi-window calculator
            multi_atr = MultiWindowATRCalculator()
            super().__init__(config=config, atr_calculator=multi_atr)
        else:
            super().__init__(config=config, atr_calculator=atr_calculator)


# ============================================================================
# Global Singletons
# ============================================================================

_atr_calculator: Optional[MultiWindowATRCalculator] = None
_stop_manager: Optional[EnhancedAdaptiveStopManager] = None
_calmar_calculator: Optional[CalmarCalculator] = None


def get_atr_calculator() -> MultiWindowATRCalculator:
    """Get global ATR calculator instance."""
    global _atr_calculator
    if _atr_calculator is None:
        _atr_calculator = MultiWindowATRCalculator()
    return _atr_calculator


def get_calmar_calculator() -> CalmarCalculator:
    """Get global Calmar calculator instance."""
    global _calmar_calculator
    if _calmar_calculator is None:
        _calmar_calculator = CalmarCalculator()
    return _calmar_calculator


def get_stop_manager() -> EnhancedAdaptiveStopManager:
    """Get global adaptive stop manager instance."""
    global _stop_manager
    if _stop_manager is None:
        _stop_manager = EnhancedAdaptiveStopManager(
            atr_calculator=get_atr_calculator(),
            calmar_calculator=get_calmar_calculator()
        )
    return _stop_manager


# ============================================================================
# Convenience Functions (v4.5 compatible)
# ============================================================================

def register_trade(
    symbol: str,
    entry_price: float,
    position_size: float,
    direction: str,
    entry_time: Optional[datetime] = None
) -> StopState:
    """Register a new trade (v4.5 compatible)."""
    dir_enum = PositionDirection.LONG if direction.lower() == "long" else PositionDirection.SHORT
    return get_stop_manager().register_position(
        symbol=symbol,
        entry_price=entry_price,
        position_size=position_size,
        direction=dir_enum,
        entry_time=entry_time
    )


def update_atr(
    symbol: str,
    high: float,
    low: float,
    close: float,
    timestamp: Optional[datetime] = None
) -> float:
    """Update ATR (v4.5 compatible, returns 30d ATR)."""
    state = get_atr_calculator().update(symbol, high, low, close, timestamp)
    return state.atr_30d


def check_stop(
    symbol: str,
    current_price: float
) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """Check stop status (v4.5 compatible)."""
    hit, reason, state = get_stop_manager().check_stop_hit(symbol, current_price)
    reason_str = reason.value if reason else None
    details = get_stop_manager().get_position_summary(symbol) if state else None
    return hit, reason_str, details


def get_current_stop(symbol: str) -> Optional[float]:
    """Get current stop price."""
    state = get_stop_manager().get_stop_state(symbol)
    return state.current_stop if state else None


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import random
    
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("HMATS v5.0.5 - Enhanced Adaptive Stop-Loss Module Test")
    print("=" * 70)
    
    # Create instances
    atr_calc = MultiWindowATRCalculator()
    calmar_calc = CalmarCalculator()
    stop_mgr = EnhancedAdaptiveStopManager(
        atr_calculator=atr_calc,
        calmar_calculator=calmar_calc
    )
    
    # Simulate 60 days of OHLC data
    print("\n1. Building multi-window ATR with 60 days of data...")
    base_price = 100.0
    equity = 10000.0
    
    for day in range(60):
        daily_return = random.gauss(0, 0.04)  # 4% daily vol
        close = base_price * (1 + daily_return)
        high = close * (1 + abs(random.gauss(0, 0.015)))
        low = close * (1 - abs(random.gauss(0, 0.015)))
        
        atr_calc.update("SOL", high, low, close)
        
        # Update equity for Calmar
        equity *= (1 + daily_return * 0.5)
        calmar_calc.update("SOL", equity)
        
        base_price = close
    
    atr_state = atr_calc.get_state("SOL")
    calmar_metrics = calmar_calc.get_metrics("SOL")
    
    print(f"   ATR 14d: {atr_state.atr_14d:.4f}")
    print(f"   ATR 30d: {atr_state.atr_30d:.4f}")
    print(f"   ATR 60d: {atr_state.atr_60d:.4f}")
    print(f"   Composite ATR: {atr_state.composite_atr:.4f}")
    print(f"   Vol Percentile: {atr_state.vol_percentile:.1f}%")
    print(f"   Vol Regime: {atr_state.vol_regime.value}")
    print(f"   Calmar Ratio: {calmar_metrics.calmar_ratio:.2f}")
    print(f"   Max Drawdown: {calmar_metrics.max_drawdown_pct:.1f}%")
    
    # Register position
    print("\n2. Registering LONG position...")
    entry_price = base_price
    state = stop_mgr.register_position(
        symbol="SOL",
        entry_price=entry_price,
        position_size=10.0,
        direction=PositionDirection.LONG,
        calmar_metrics=calmar_metrics
    )
    
    print(f"   Entry: ${entry_price:.2f}")
    print(f"   Initial Stop: ${state.initial_stop:.4f}")
    print(f"   Risk: {state.initial_risk_pct:.2f}%")
    print(f"   Dynamic R Trigger: {state.dynamic_r_trigger:.2f}")
    print(f"   Vol Regime at Entry: {state.vol_regime_at_entry.value}")
    
    # Simulate price movement
    print("\n3. Simulating price movement...")
    
    prices = [
        (entry_price * 1.02, "Small profit"),
        (entry_price * 1.04, "1R profit"),
        (entry_price * (1 + state.initial_risk_pct/100 * state.dynamic_r_trigger), "Dynamic R trigger"),
        (entry_price * 1.08, "Further profit"),
        (entry_price * 1.10, "Strong profit"),
        (entry_price * 1.07, "Pullback"),
    ]
    
    for price, desc in prices:
        state = stop_mgr.update_stop("SOL", price)
        print(f"\n   Price: ${price:.2f} - {desc}")
        print(f"   R-Multiple: {state.current_r_multiple:.2f}")
        print(f"   Stop: ${state.current_stop:.4f}")
        print(f"   Type: {state.stop_type.value}")
        print(f"   Breakeven: {state.breakeven_triggered}")
        print(f"   Trailing: {state.trailing_active}")
        if state.trailing_active:
            print(f"   Trail Multiplier: {state.dynamic_trail_multiplier:.2f}")
    
    # Test backward compatibility
    print("\n4. Testing backward compatibility...")
    legacy_atr = ATRCalculator()
    for day in range(30):
        legacy_atr.update("BTC", 50000 + random.uniform(-500, 500), 
                         49000 + random.uniform(-500, 500), 
                         49500 + random.uniform(-500, 500))
    
    legacy_state = legacy_atr.get_atr("BTC")
    print(f"   Legacy ATR value: {legacy_state.atr_value:.2f}")
    print(f"   Legacy ATR %: {legacy_state.atr_pct:.2f}%")
    
    print("\n" + "=" * 70)
    print("v5.0.5 Adaptive Stop Module Test Complete!")
    print("=" * 70)

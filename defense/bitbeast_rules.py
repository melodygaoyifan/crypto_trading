"""
================================================================================
BITBEAST BEHAVIOR RULES - BitBeast行为约束 (3-in-1 Entry Quality Guard)
================================================================================
Version: 3.3.0
Purpose: 强制执行BitBeast交易规则 - 三合一入场质量守卫

规则:
- SNRFilter (Ants vs Train): 减少强趋势中的反转信号
- VolumeFilter: 禁止缩量下跌抄底
- StructureBreakoutFilter: 结构突破才顺势

三票否决制: 任何一个 filter reject -> veto intent
================================================================================
"""

import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import threading
import numpy as np

logger = logging.getLogger("HMATS.BitBeastRules")


# =============================================================================
# SNR FILTER CONSTANTS
# =============================================================================

MIN_PATTERN_DURATION_RATIO = 0.50  # Ants vs Train threshold
TREND_LOOKBACK_BARS = 100
CONSOLIDATION_THRESHOLD_ATR = 0.5  # Below 0.5 ATR = consolidation


# =============================================================================
# SNR FILTER DATA STRUCTURES
# =============================================================================

class TrendState(Enum):
    """Current trend state classification."""
    STRONG_UPTREND = auto()
    WEAK_UPTREND = auto()
    CONSOLIDATION = auto()
    WEAK_DOWNTREND = auto()
    STRONG_DOWNTREND = auto()


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


# =============================================================================
# SNR FILTER (ANTS VS TRAIN RULE)
# =============================================================================

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

        # Price history (keyed by asset for multi-asset support)
        self._history: Dict[str, deque] = {}
        self._atr_cache: Dict[str, float] = {}
        self._trend_states: Dict[str, TrendAnalysis] = {}

        logger.info(f"SNRFilter initialized (ratio={min_pattern_ratio}, lookback={lookback_bars})")

    def _ensure_history(self, asset: str):
        if asset not in self._history:
            self._history[asset] = deque(maxlen=self.lookback_bars * 2)

    def update(self, asset: str, price: float, high: float, low: float, volume: float):
        """Update with new price bar."""
        self._ensure_history(asset)
        self._history[asset].append({
            'price': price, 'high': high, 'low': low,
            'volume': volume, 'timestamp': time.time()
        })
        self._update_atr(asset)
        self._analyze_trend(asset)

    def _update_atr(self, asset: str):
        history = list(self._history[asset])
        if len(history) < self.atr_period + 1:
            self._atr_cache[asset] = 0.0
            return
        true_ranges = []
        for i in range(1, min(self.atr_period + 1, len(history))):
            h = history[-i]['high']
            l = history[-i]['low']
            prev_c = history[-i - 1]['price']
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            true_ranges.append(tr)
        self._atr_cache[asset] = np.mean(true_ranges) if true_ranges else 0.0

    def _analyze_trend(self, asset: str):
        history = list(self._history[asset])
        if len(history) < 20:
            self._trend_states[asset] = TrendAnalysis(
                state=TrendState.CONSOLIDATION, duration_bars=0,
                strength=0.0, atr_ratio=0.0, is_consolidation=True,
                prior_trend_duration=0, pattern_duration_ratio=1.0
            )
            return
        prices = np.array([h['price'] for h in history])
        atr = self._atr_cache.get(asset, 1.0) or 1.0

        returns = np.diff(prices) / prices[:-1]
        recent_returns = returns[-20:]
        strength = np.mean(recent_returns) / (np.std(recent_returns) + 1e-8)
        strength = np.clip(strength, -1, 1)
        price_range = np.max(prices[-20:]) - np.min(prices[-20:])
        atr_ratio = price_range / (atr * 20) if atr > 0 else 0
        is_consolidation = atr_ratio < CONSOLIDATION_THRESHOLD_ATR

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

        duration = self._count_trend_duration(prices, state)
        prior_duration = self._find_prior_trend_duration(asset, state)
        if prior_duration > 0 and is_consolidation:
            ratio = duration / prior_duration
        else:
            ratio = 1.0

        self._trend_states[asset] = TrendAnalysis(
            state=state, duration_bars=duration, strength=strength,
            atr_ratio=atr_ratio, is_consolidation=is_consolidation,
            prior_trend_duration=prior_duration, pattern_duration_ratio=ratio
        )

    def _count_trend_duration(self, prices: np.ndarray, current_state: TrendState) -> int:
        if len(prices) < 5:
            return 0
        returns = np.diff(prices)
        duration = 0
        if current_state in [TrendState.STRONG_UPTREND, TrendState.WEAK_UPTREND]:
            for r in reversed(returns):
                if r > 0:
                    duration += 1
                else:
                    break
        elif current_state in [TrendState.STRONG_DOWNTREND, TrendState.WEAK_DOWNTREND]:
            for r in reversed(returns):
                if r < 0:
                    duration += 1
                else:
                    break
        else:
            recent_mean = np.mean(prices[-20:])
            atr = np.std(prices[-20:]) or 1.0
            for p in reversed(prices):
                if abs(p - recent_mean) < atr * 1.5:
                    duration += 1
                else:
                    break
        return duration

    def _find_prior_trend_duration(self, asset: str, current_state: TrendState) -> int:
        history = list(self._history[asset])
        if len(history) < 30:
            return 0
        prices = np.array([h['price'] for h in history])
        if current_state == TrendState.CONSOLIDATION:
            returns = np.diff(prices)
            consol_start = len(returns)
            for i in range(len(returns) - 1, -1, -1):
                if abs(returns[i]) > self._atr_cache.get(asset, 0.01) * 0.5:
                    consol_start = i
                    break
            if consol_start > 5:
                prior_trend_dir = np.sign(np.sum(returns[max(0, consol_start - 20):consol_start]))
                count = 0
                for i in range(consol_start - 1, -1, -1):
                    if np.sign(returns[i]) == prior_trend_dir:
                        count += 1
                    else:
                        break
                return count
        return 0

    def filter_signal(self, asset: str, signal_direction: str, signal_confidence: float) -> SNRFilterResult:
        """
        Apply SNR filter to a trading signal.

        Args:
            asset: Asset name (BTC/ETH/SOL)
            signal_direction: "LONG" or "SHORT"
            signal_confidence: 0 to 1

        Returns:
            SNRFilterResult with passed/weight/reason
        """
        analysis = self._trend_states.get(asset)
        if analysis is None:
            return SNRFilterResult(passed=True, pattern_ratio=1.0, reversal_weight=1.0,
                                   reason="No trend data available")

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

        if not is_reversal or not analysis.is_consolidation:
            return SNRFilterResult(
                passed=True, pattern_ratio=analysis.pattern_duration_ratio,
                reversal_weight=1.0,
                reason="Signal aligns with trend or not in consolidation"
            )

        # Apply Ants vs Train rule
        if analysis.pattern_duration_ratio < self.min_pattern_ratio:
            weight = analysis.pattern_duration_ratio / self.min_pattern_ratio
            weight = max(0.1, weight)
            return SNRFilterResult(
                passed=weight > 0.5,
                pattern_ratio=analysis.pattern_duration_ratio,
                reversal_weight=weight,
                reason=f"Ants vs Train: consolidation ({analysis.duration_bars} bars) "
                       f"< 50% of prior trend ({analysis.prior_trend_duration} bars)"
            )

        return SNRFilterResult(
            passed=True, pattern_ratio=analysis.pattern_duration_ratio,
            reversal_weight=1.0, reason="Consolidation sufficient for reversal signal"
        )

    def get_trend_analysis(self, asset: str) -> Optional[TrendAnalysis]:
        return self._trend_states.get(asset)


# =============================================================================
# PART 11.1: VOLUME FILTER - 禁止缩量下跌抄底
# =============================================================================

class VolumeCondition(Enum):
    STRONG_VOLUME = auto()      # > 1.5x avg
    NORMAL_VOLUME = auto()      # 0.8x - 1.5x avg
    WEAK_VOLUME = auto()        # 0.5x - 0.8x avg
    DECLINING_VOLUME = auto()   # < 0.5x avg


class PriceTrend(Enum):
    STRONG_UP = auto()
    MODERATE_UP = auto()
    NEUTRAL = auto()
    MODERATE_DOWN = auto()
    STRONG_DOWN = auto()


@dataclass
class VolumeAnalysis:
    current_volume: float = 0.0
    avg_volume_20: float = 0.0
    volume_ratio: float = 1.0
    volume_condition: VolumeCondition = VolumeCondition.NORMAL_VOLUME
    price_trend: PriceTrend = PriceTrend.NEUTRAL
    price_change_pct: float = 0.0
    is_declining_on_weak_volume: bool = False
    can_buy_dip: bool = True
    reject_reason: str = ""
    timestamp: float = field(default_factory=time.time)


class VolumeFilter:
    """
    Volume Filter - 禁止缩量下跌抄底

    条件:
    1. 价格下跌 > 3%
    2. 成交量 < 0.8x 20期均量
    3. 连续3根K线缩量下跌
    -> 禁止做多，等待放量反转
    """

    WEAK_VOLUME_RATIO = 0.8
    DECLINING_VOLUME_RATIO = 0.5
    STRONG_VOLUME_RATIO = 1.5
    PRICE_DOWN_THRESHOLD = -0.03
    PRICE_UP_THRESHOLD = 0.03
    CONSECUTIVE_DECLINING_BARS = 3

    def __init__(self):
        self.volume_history: deque = deque(maxlen=100)
        self.price_history: deque = deque(maxlen=100)
        self.current_analysis = VolumeAnalysis()
        self._lock = threading.Lock()
        logger.info("VolumeFilter initialized")

    def update(self, price: float, volume: float) -> VolumeAnalysis:
        with self._lock:
            now = time.time()
            self.volume_history.append({'volume': volume, 'timestamp': now})
            self.price_history.append({'price': price, 'timestamp': now})
            self._compute_analysis()
            return self.current_analysis

    def _compute_analysis(self):
        if len(self.volume_history) < 20:
            return
        volumes = [v['volume'] for v in self.volume_history]
        prices = [p['price'] for p in self.price_history]
        current_volume = volumes[-1]
        current_price = prices[-1]
        avg_volume_20 = np.mean(volumes[-20:])
        volume_ratio = current_volume / max(avg_volume_20, 1)

        if volume_ratio > self.STRONG_VOLUME_RATIO:
            volume_condition = VolumeCondition.STRONG_VOLUME
        elif volume_ratio > self.WEAK_VOLUME_RATIO:
            volume_condition = VolumeCondition.NORMAL_VOLUME
        elif volume_ratio > self.DECLINING_VOLUME_RATIO:
            volume_condition = VolumeCondition.WEAK_VOLUME
        else:
            volume_condition = VolumeCondition.DECLINING_VOLUME

        if len(prices) >= 20:
            price_change = (current_price - prices[-20]) / prices[-20]
        else:
            price_change = 0.0

        if price_change > 0.10:
            price_trend = PriceTrend.STRONG_UP
        elif price_change > self.PRICE_UP_THRESHOLD:
            price_trend = PriceTrend.MODERATE_UP
        elif price_change < -0.10:
            price_trend = PriceTrend.STRONG_DOWN
        elif price_change < self.PRICE_DOWN_THRESHOLD:
            price_trend = PriceTrend.MODERATE_DOWN
        else:
            price_trend = PriceTrend.NEUTRAL

        is_declining_on_weak_volume = False
        can_buy_dip = True
        reject_reason = ""

        if price_trend in [PriceTrend.MODERATE_DOWN, PriceTrend.STRONG_DOWN]:
            if volume_condition in [VolumeCondition.WEAK_VOLUME, VolumeCondition.DECLINING_VOLUME]:
                if len(volumes) >= self.CONSECUTIVE_DECLINING_BARS:
                    recent_volumes = volumes[-self.CONSECUTIVE_DECLINING_BARS:]
                    is_consecutive_decline = all(
                        recent_volumes[i] < recent_volumes[i - 1] * 1.1
                        for i in range(1, len(recent_volumes))
                    )
                    if is_consecutive_decline:
                        is_declining_on_weak_volume = True
                        can_buy_dip = False
                        reject_reason = (
                            f"Declining volume dip: {self.CONSECUTIVE_DECLINING_BARS} "
                            f"consecutive bars shrinking, price down {abs(price_change) * 100:.1f}%"
                        )

        self.current_analysis = VolumeAnalysis(
            current_volume=current_volume, avg_volume_20=avg_volume_20,
            volume_ratio=volume_ratio, volume_condition=volume_condition,
            price_trend=price_trend, price_change_pct=price_change * 100,
            is_declining_on_weak_volume=is_declining_on_weak_volume,
            can_buy_dip=can_buy_dip, reject_reason=reject_reason
        )

    def can_open_long(self) -> Tuple[bool, str]:
        with self._lock:
            analysis = self.current_analysis
            if not analysis.can_buy_dip:
                return False, analysis.reject_reason
            if analysis.volume_condition == VolumeCondition.DECLINING_VOLUME:
                return False, "Extreme volume decline, wait for recovery"
            return True, "OK"

    def get_status(self) -> Dict:
        with self._lock:
            a = self.current_analysis
            return {
                'volume_ratio': round(a.volume_ratio, 2),
                'volume_condition': a.volume_condition.name,
                'price_trend': a.price_trend.name,
                'price_change_pct': round(a.price_change_pct, 2),
                'is_declining_weak_volume': a.is_declining_on_weak_volume,
                'can_buy_dip': a.can_buy_dip,
            }


# =============================================================================
# PART 11.2: STRUCTURE BREAKOUT FILTER - 结构突破才顺势
# =============================================================================

class StructureType(Enum):
    HIGHER_HIGH = auto()
    HIGHER_LOW = auto()
    LOWER_HIGH = auto()
    LOWER_LOW = auto()
    CONSOLIDATION = auto()
    BREAKOUT_UP = auto()
    BREAKOUT_DOWN = auto()


class TrendBias(Enum):
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()


@dataclass
class SwingPoint:
    price: float
    timestamp: float
    is_high: bool


@dataclass
class StructureAnalysis:
    recent_highs: List[float] = field(default_factory=list)
    recent_lows: List[float] = field(default_factory=list)
    structure_type: StructureType = StructureType.CONSOLIDATION
    trend_bias: TrendBias = TrendBias.NEUTRAL
    resistance: float = 0.0
    support: float = 0.0
    is_breakout: bool = False
    breakout_direction: str = ""
    breakout_confirmed: bool = False
    can_follow_trend_long: bool = False
    can_follow_trend_short: bool = False
    reject_reason: str = ""
    timestamp: float = field(default_factory=time.time)


class StructureBreakoutFilter:
    """
    Structure Breakout Filter - 结构突破才顺势

    做多条件: HH confirmed or HL bounce
    做空条件: LL confirmed or LH drop
    禁止: 下跌趋势抄底(无结构反转), 上涨趋势摸顶(无结构反转)
    """

    SWING_LOOKBACK = 5
    BREAKOUT_BUFFER_PCT = 0.002

    def __init__(self):
        self.price_history: deque = deque(maxlen=200)
        self.swing_highs: deque = deque(maxlen=20)
        self.swing_lows: deque = deque(maxlen=20)
        self.current_analysis = StructureAnalysis()
        self._lock = threading.Lock()
        logger.info("StructureBreakoutFilter initialized")

    def update(self, high: float, low: float, close: float) -> StructureAnalysis:
        with self._lock:
            now = time.time()
            self.price_history.append({
                'high': high, 'low': low, 'close': close, 'timestamp': now
            })
            self._detect_swing_points()
            self._analyze_structure(close)
            return self.current_analysis

    def _detect_swing_points(self):
        if len(self.price_history) < self.SWING_LOOKBACK * 2 + 1:
            return
        prices = list(self.price_history)
        idx = len(prices) - self.SWING_LOOKBACK - 1
        if idx < self.SWING_LOOKBACK:
            return
        center = prices[idx]
        left_bars = prices[idx - self.SWING_LOOKBACK:idx]
        right_bars = prices[idx + 1:idx + self.SWING_LOOKBACK + 1]

        center_high = center['high']
        if all(center_high >= p['high'] for p in left_bars) and \
           all(center_high >= p['high'] for p in right_bars):
            self.swing_highs.append(SwingPoint(
                price=center_high, timestamp=center['timestamp'], is_high=True
            ))

        center_low = center['low']
        if all(center_low <= p['low'] for p in left_bars) and \
           all(center_low <= p['low'] for p in right_bars):
            self.swing_lows.append(SwingPoint(
                price=center_low, timestamp=center['timestamp'], is_high=False
            ))

    def _analyze_structure(self, current_close: float):
        recent_highs = [s.price for s in list(self.swing_highs)[-4:]]
        recent_lows = [s.price for s in list(self.swing_lows)[-4:]]

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            self.current_analysis = StructureAnalysis(
                recent_highs=recent_highs, recent_lows=recent_lows
            )
            return

        last_high, prev_high = recent_highs[-1], recent_highs[-2]
        last_low, prev_low = recent_lows[-1], recent_lows[-2]

        higher_high = last_high > prev_high
        higher_low = last_low > prev_low
        lower_high = last_high < prev_high
        lower_low = last_low < prev_low

        if higher_high and higher_low:
            structure_type = StructureType.HIGHER_HIGH
            trend_bias = TrendBias.BULLISH
        elif lower_high and lower_low:
            structure_type = StructureType.LOWER_LOW
            trend_bias = TrendBias.BEARISH
        elif higher_low and not higher_high:
            structure_type = StructureType.HIGHER_LOW
            trend_bias = TrendBias.BULLISH
        elif lower_high and not lower_low:
            structure_type = StructureType.LOWER_HIGH
            trend_bias = TrendBias.BEARISH
        else:
            structure_type = StructureType.CONSOLIDATION
            trend_bias = TrendBias.NEUTRAL

        resistance = max(recent_highs) if recent_highs else current_close * 1.05
        support = min(recent_lows) if recent_lows else current_close * 0.95

        breakout_buffer = current_close * self.BREAKOUT_BUFFER_PCT
        is_breakout = False
        breakout_direction = ""
        breakout_confirmed = False

        if current_close > resistance + breakout_buffer:
            is_breakout = True
            breakout_direction = "up"
            breakout_confirmed = True
            structure_type = StructureType.BREAKOUT_UP
        elif current_close < support - breakout_buffer:
            is_breakout = True
            breakout_direction = "down"
            breakout_confirmed = True
            structure_type = StructureType.BREAKOUT_DOWN

        can_follow_trend_long = False
        can_follow_trend_short = False
        reject_reason = ""

        if trend_bias == TrendBias.BULLISH:
            if structure_type in [StructureType.BREAKOUT_UP, StructureType.HIGHER_HIGH,
                                  StructureType.HIGHER_LOW]:
                can_follow_trend_long = True
            else:
                reject_reason = "Bullish structure but no breakout confirmation"
        elif trend_bias == TrendBias.BEARISH:
            reject_reason = "Bearish structure, no long dip-buying"
        else:
            reject_reason = "Consolidation, wait for breakout direction"

        if trend_bias == TrendBias.BEARISH:
            if structure_type in [StructureType.BREAKOUT_DOWN, StructureType.LOWER_LOW,
                                  StructureType.LOWER_HIGH]:
                can_follow_trend_short = True

        self.current_analysis = StructureAnalysis(
            recent_highs=recent_highs, recent_lows=recent_lows,
            structure_type=structure_type, trend_bias=trend_bias,
            resistance=resistance, support=support,
            is_breakout=is_breakout, breakout_direction=breakout_direction,
            breakout_confirmed=breakout_confirmed,
            can_follow_trend_long=can_follow_trend_long,
            can_follow_trend_short=can_follow_trend_short,
            reject_reason=reject_reason
        )

    def can_open_long(self) -> Tuple[bool, str]:
        with self._lock:
            a = self.current_analysis
            if a.can_follow_trend_long:
                return True, f"Structure confirmed: {a.structure_type.name}"
            return False, a.reject_reason or "Structure does not support long"

    def can_open_short(self) -> Tuple[bool, str]:
        with self._lock:
            a = self.current_analysis
            if a.can_follow_trend_short:
                return True, f"Structure confirmed: {a.structure_type.name}"
            return False, "Structure does not support short"

    def get_status(self) -> Dict:
        with self._lock:
            a = self.current_analysis
            return {
                'structure': a.structure_type.name,
                'trend_bias': a.trend_bias.name,
                'resistance': round(a.resistance, 2),
                'support': round(a.support, 2),
                'is_breakout': a.is_breakout,
                'can_long': a.can_follow_trend_long,
                'can_short': a.can_follow_trend_short,
            }


# =============================================================================
# BITBEAST BEHAVIOR GUARD - 三合一入场质量守卫
# =============================================================================

class BitBeastBehaviorGuard:
    """
    BitBeast 3-in-1 Entry Quality Guard.

    Integrates:
    1. SNRFilter (Ants vs Train) - reduce reversal signals in strong trends
    2. VolumeFilter - block longs during volume-declining drops
    3. StructureBreakoutFilter - require structural break before opening

    三票否决制: Any single filter reject -> veto intent.
    """

    def __init__(self, asset: str):
        self.asset = asset
        self.snr_filter = SNRFilter()
        self.volume_filter = VolumeFilter()
        self.structure_filter = StructureBreakoutFilter()
        logger.info(f"BitBeastBehaviorGuard initialized for {asset}")

    def update(self, high: float, low: float, close: float, volume: float):
        """Update all filters with new bar data."""
        self.snr_filter.update(self.asset, close, high, low, volume)
        self.volume_filter.update(close, volume)
        self.structure_filter.update(high, low, close)

    def check_entry(self, direction: int, confidence: float = 0.5) -> Tuple[bool, List[str]]:
        """
        Check if entry is allowed by all three filters.

        Args:
            direction: +1 for long, -1 for short
            confidence: signal confidence (0 to 1)

        Returns:
            (allowed, list of veto reasons)
        """
        reasons = []

        # 1. SNR Filter (Ants vs Train)
        sig_dir = "LONG" if direction > 0 else "SHORT"
        snr_result = self.snr_filter.filter_signal(self.asset, sig_dir, confidence)
        if not snr_result.passed:
            reasons.append(f"SNR: {snr_result.reason}")

        # 2. Volume Filter (long-only check)
        if direction > 0:
            vol_ok, vol_reason = self.volume_filter.can_open_long()
            if not vol_ok:
                reasons.append(f"Volume: {vol_reason}")

        # 3. Structure Breakout Filter
        if direction > 0:
            struct_ok, struct_reason = self.structure_filter.can_open_long()
        else:
            struct_ok, struct_reason = self.structure_filter.can_open_short()
        if not struct_ok:
            reasons.append(f"Structure: {struct_reason}")

        allowed = len(reasons) == 0
        return allowed, reasons

    def get_status(self) -> Dict:
        """Get combined status of all filters."""
        snr_trend = self.snr_filter.get_trend_analysis(self.asset)
        return {
            'asset': self.asset,
            'snr': {
                'trend': snr_trend.state.name if snr_trend else 'N/A',
                'strength': round(snr_trend.strength, 3) if snr_trend else 0.0,
                'pattern_ratio': round(snr_trend.pattern_duration_ratio, 2) if snr_trend else 1.0,
            },
            'volume': self.volume_filter.get_status(),
            'structure': self.structure_filter.get_status(),
        }
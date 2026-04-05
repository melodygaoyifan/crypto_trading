"""
Structure Break Analyzer - Price Action Fractal Analysis
=========================================================

Implements Structure Break detection for HMATS v3.1:

- DRL Agent can only trigger "Long" if 4H fractal high is breached
- DRL Agent can only trigger "Short" if 4H fractal low is breached

Fractal Definition:
- Fractal High: Bar where High is higher than N bars before AND after
- Fractal Low: Bar where Low is lower than N bars before AND after

Hardware Target: MSI Titan 18 HX
Author: Lead Quant Research & ML Systems Architect
Version: 3.1.2
"""

import time
import logging
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('StructureBreakAnalyzer')


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Fractal lookback (N bars each side)
DEFAULT_FRACTAL_PERIOD = 2  # 2 bars each side = Williams Fractal

# Structure break confirmation
BREAK_CONFIRMATION_CLOSE = True  # Require close above/below fractal
BREAK_BUFFER_PCT = 0.001  # 0.1% buffer to avoid false breaks


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class FractalType(Enum):
    """Type of fractal."""
    HIGH = auto()
    LOW = auto()


class BreakDirection(Enum):
    """Direction of structure break."""
    BULLISH = auto()  # Broke fractal high
    BEARISH = auto()  # Broke fractal low
    NONE = auto()


@dataclass
class Fractal:
    """Represents a fractal point."""
    type: FractalType
    price: float
    bar_index: int
    timestamp: float
    confirmed: bool = False
    broken: bool = False
    broken_at: float = 0.0


@dataclass
class StructureBreakResult:
    """Result of structure break analysis."""
    break_detected: bool
    direction: BreakDirection
    fractal_price: float
    current_price: float
    break_distance_pct: float
    can_long: bool  # True if fractal high is broken
    can_short: bool  # True if fractal low is broken
    recent_fractals: List[Fractal]
    message: str


@dataclass
class PriceBar:
    """OHLCV price bar."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURE BREAK ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class StructureBreakAnalyzer:
    """
    Analyzes price structure for fractal breaks.
    
    Rules:
    - Long entries ONLY allowed when price breaks above recent fractal high
    - Short entries ONLY allowed when price breaks below recent fractal low
    - Uses Williams Fractal (2-bar lookback by default)
    """
    
    def __init__(
        self,
        fractal_period: int = DEFAULT_FRACTAL_PERIOD,
        require_close_break: bool = BREAK_CONFIRMATION_CLOSE,
        break_buffer_pct: float = BREAK_BUFFER_PCT,
        max_fractals: int = 20
    ):
        self.period = fractal_period
        self.require_close = require_close_break
        self.buffer_pct = break_buffer_pct
        self.max_fractals = max_fractals
        
        # Price history per asset
        self._bars: Dict[str, deque] = {}
        
        # Detected fractals
        self._fractal_highs: Dict[str, deque] = {}
        self._fractal_lows: Dict[str, deque] = {}
        
        # Break state
        self._last_break: Dict[str, StructureBreakResult] = {}
        
        logger.info(f"StructureBreakAnalyzer initialized (period={fractal_period})")
    
    def _ensure_buffers(self, asset: str):
        """Ensure buffers exist for asset."""
        if asset not in self._bars:
            self._bars[asset] = deque(maxlen=200)
            self._fractal_highs[asset] = deque(maxlen=self.max_fractals)
            self._fractal_lows[asset] = deque(maxlen=self.max_fractals)
    
    def update(
        self,
        asset: str,
        bar: PriceBar = None,
        timestamp: float = None,
        open_: float = None,
        high: float = None,
        low: float = None,
        close: float = None,
        volume: float = None
    ):
        """
        Update with new price bar.
        
        Can pass either a PriceBar object or individual components.
        """
        self._ensure_buffers(asset)
        
        if bar is None:
            bar = PriceBar(
                timestamp=timestamp or time.time(),
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume or 0.0
            )
        
        self._bars[asset].append(bar)
        
        # Check for new fractals (need enough bars)
        if len(self._bars[asset]) >= 2 * self.period + 1:
            self._detect_fractals(asset)
    
    def _detect_fractals(self, asset: str):
        """Detect new fractal highs/lows."""
        bars = list(self._bars[asset])
        
        # Check the middle bar (N bars ago)
        middle_idx = len(bars) - self.period - 1
        if middle_idx < self.period:
            return
        
        middle_bar = bars[middle_idx]
        bar_index = middle_idx
        
        # Check fractal high
        is_fractal_high = True
        for i in range(1, self.period + 1):
            if (bars[middle_idx - i].high >= middle_bar.high or
                bars[middle_idx + i].high >= middle_bar.high):
                is_fractal_high = False
                break
        
        if is_fractal_high:
            fractal = Fractal(
                type=FractalType.HIGH,
                price=middle_bar.high,
                bar_index=bar_index,
                timestamp=middle_bar.timestamp,
                confirmed=True
            )
            
            # Check if not duplicate
            if not self._is_duplicate_fractal(asset, fractal, FractalType.HIGH):
                self._fractal_highs[asset].append(fractal)
                logger.debug(f"{asset}: New fractal HIGH at {fractal.price:.2f}")
        
        # Check fractal low
        is_fractal_low = True
        for i in range(1, self.period + 1):
            if (bars[middle_idx - i].low <= middle_bar.low or
                bars[middle_idx + i].low <= middle_bar.low):
                is_fractal_low = False
                break
        
        if is_fractal_low:
            fractal = Fractal(
                type=FractalType.LOW,
                price=middle_bar.low,
                bar_index=bar_index,
                timestamp=middle_bar.timestamp,
                confirmed=True
            )
            
            if not self._is_duplicate_fractal(asset, fractal, FractalType.LOW):
                self._fractal_lows[asset].append(fractal)
                logger.debug(f"{asset}: New fractal LOW at {fractal.price:.2f}")
    
    def _is_duplicate_fractal(
        self,
        asset: str,
        fractal: Fractal,
        ftype: FractalType
    ) -> bool:
        """Check if fractal is duplicate of recent one."""
        fractals = (self._fractal_highs[asset] if ftype == FractalType.HIGH 
                   else self._fractal_lows[asset])
        
        for f in fractals:
            if abs(f.price - fractal.price) / fractal.price < 0.001:  # Within 0.1%
                return True
        return False
    
    def check_break(
        self,
        asset: str,
        current_price: float,
        current_close: float = None
    ) -> StructureBreakResult:
        """
        Check if price has broken structure.
        
        Returns StructureBreakResult with trading permissions.
        """
        self._ensure_buffers(asset)
        
        if current_close is None:
            current_close = current_price
        
        # Get recent fractals
        recent_highs = list(self._fractal_highs[asset])
        recent_lows = list(self._fractal_lows[asset])
        
        if not recent_highs and not recent_lows:
            return StructureBreakResult(
                break_detected=False,
                direction=BreakDirection.NONE,
                fractal_price=0.0,
                current_price=current_price,
                break_distance_pct=0.0,
                can_long=False,
                can_short=False,
                recent_fractals=[],
                message="No fractals detected yet"
            )
        
        # Find most recent unbroken fractals
        nearest_high = None
        nearest_low = None
        
        for f in reversed(recent_highs):
            if not f.broken:
                nearest_high = f
                break
        
        for f in reversed(recent_lows):
            if not f.broken:
                nearest_low = f
                break
        
        # Check for bullish break (above fractal high)
        can_long = False
        bullish_break = False
        bullish_distance = 0.0
        
        if nearest_high:
            break_level = nearest_high.price * (1 + self.buffer_pct)
            
            if self.require_close:
                bullish_break = current_close > break_level
            else:
                bullish_break = current_price > break_level
            
            if bullish_break:
                can_long = True
                nearest_high.broken = True
                nearest_high.broken_at = time.time()
                bullish_distance = (current_price - nearest_high.price) / nearest_high.price
        
        # Check for bearish break (below fractal low)
        can_short = False
        bearish_break = False
        bearish_distance = 0.0
        
        if nearest_low:
            break_level = nearest_low.price * (1 - self.buffer_pct)
            
            if self.require_close:
                bearish_break = current_close < break_level
            else:
                bearish_break = current_price < break_level
            
            if bearish_break:
                can_short = True
                nearest_low.broken = True
                nearest_low.broken_at = time.time()
                bearish_distance = (nearest_low.price - current_price) / nearest_low.price
        
        # Determine direction and result
        if bullish_break and not bearish_break:
            direction = BreakDirection.BULLISH
            fractal_price = nearest_high.price if nearest_high else 0.0
            distance = bullish_distance
            message = f"BULLISH break: price {current_price:.2f} > fractal high {fractal_price:.2f}"
        elif bearish_break and not bullish_break:
            direction = BreakDirection.BEARISH
            fractal_price = nearest_low.price if nearest_low else 0.0
            distance = bearish_distance
            message = f"BEARISH break: price {current_price:.2f} < fractal low {fractal_price:.2f}"
        elif bullish_break and bearish_break:
            direction = BreakDirection.BULLISH if bullish_distance > bearish_distance else BreakDirection.BEARISH
            fractal_price = (nearest_high.price if direction == BreakDirection.BULLISH 
                           else nearest_low.price)
            distance = max(bullish_distance, bearish_distance)
            message = "Both breaks detected - using stronger break"
        else:
            direction = BreakDirection.NONE
            fractal_price = nearest_high.price if nearest_high else (nearest_low.price if nearest_low else 0.0)
            distance = 0.0
            
            if nearest_high and nearest_low:
                message = f"No break: price between fractal low {nearest_low.price:.2f} and high {nearest_high.price:.2f}"
            elif nearest_high:
                message = f"No break: price below fractal high {nearest_high.price:.2f}"
            else:
                message = f"No break: price above fractal low {nearest_low.price:.2f}"
        
        result = StructureBreakResult(
            break_detected=bullish_break or bearish_break,
            direction=direction,
            fractal_price=fractal_price,
            current_price=current_price,
            break_distance_pct=distance * 100,
            can_long=can_long,
            can_short=can_short,
            recent_fractals=recent_highs[-5:] + recent_lows[-5:],
            message=message
        )
        
        self._last_break[asset] = result
        return result
    
    def get_fractal_levels(self, asset: str) -> Dict[str, List[float]]:
        """Get all fractal levels for an asset."""
        self._ensure_buffers(asset)
        
        return {
            'highs': [f.price for f in self._fractal_highs[asset] if not f.broken],
            'lows': [f.price for f in self._fractal_lows[asset] if not f.broken],
            'broken_highs': [f.price for f in self._fractal_highs[asset] if f.broken],
            'broken_lows': [f.price for f in self._fractal_lows[asset] if f.broken],
        }
    
    def get_nearest_levels(self, asset: str, current_price: float) -> Dict[str, Optional[float]]:
        """Get nearest fractal levels above and below current price."""
        self._ensure_buffers(asset)
        
        highs = [f.price for f in self._fractal_highs[asset] if not f.broken and f.price > current_price]
        lows = [f.price for f in self._fractal_lows[asset] if not f.broken and f.price < current_price]
        
        return {
            'nearest_resistance': min(highs) if highs else None,
            'nearest_support': max(lows) if lows else None,
        }
    
    def validate_entry(
        self,
        asset: str,
        side: str,  # "LONG" or "SHORT"
        current_price: float,
        current_close: float = None
    ) -> Tuple[bool, str]:
        """
        Validate if entry is allowed based on structure break.
        
        DRL Rule:
        - LONG only if fractal high is broken
        - SHORT only if fractal low is broken
        
        Returns: (allowed, reason)
        """
        result = self.check_break(asset, current_price, current_close)
        
        if side == "LONG":
            if result.can_long:
                return True, f"LONG allowed: {result.message}"
            else:
                return False, f"LONG blocked: fractal high not broken. {result.message}"
        
        elif side == "SHORT":
            if result.can_short:
                return True, f"SHORT allowed: {result.message}"
            else:
                return False, f"SHORT blocked: fractal low not broken. {result.message}"
        
        return False, f"Unknown side: {side}"


# ═══════════════════════════════════════════════════════════════════════════════
# 4H TIMEFRAME STRUCTURE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class FourHourStructureTracker:
    """
    Specialized tracker for 4H timeframe structure breaks.
    
    Aggregates 1m bars into 4H bars for structure analysis.
    """
    
    def __init__(self, fractal_period: int = 2):
        self.analyzer = StructureBreakAnalyzer(fractal_period=fractal_period)
        
        # 1m to 4H aggregation
        self._minute_bars: Dict[str, List[PriceBar]] = {}
        self._current_4h_bar: Dict[str, Dict] = {}
        self._4h_bar_count: Dict[str, int] = {}
        
        self.MINUTES_PER_4H = 240
        
        logger.info("FourHourStructureTracker initialized")
    
    def _ensure_buffers(self, asset: str):
        """Ensure buffers exist."""
        if asset not in self._minute_bars:
            self._minute_bars[asset] = []
            self._current_4h_bar[asset] = None
            self._4h_bar_count[asset] = 0
    
    def update_1m(
        self,
        asset: str,
        timestamp: float,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float
    ):
        """Update with 1-minute bar, aggregate to 4H."""
        self._ensure_buffers(asset)
        
        # Start new 4H bar if needed
        if self._current_4h_bar[asset] is None:
            self._current_4h_bar[asset] = {
                'timestamp': timestamp,
                'open': open_,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'minute_count': 1
            }
        else:
            bar = self._current_4h_bar[asset]
            bar['high'] = max(bar['high'], high)
            bar['low'] = min(bar['low'], low)
            bar['close'] = close
            bar['volume'] += volume
            bar['minute_count'] += 1
            
            # Check if 4H bar complete
            if bar['minute_count'] >= self.MINUTES_PER_4H:
                # Finalize 4H bar
                completed_bar = PriceBar(
                    timestamp=bar['timestamp'],
                    open=bar['open'],
                    high=bar['high'],
                    low=bar['low'],
                    close=bar['close'],
                    volume=bar['volume']
                )
                
                # Update analyzer with 4H bar
                self.analyzer.update(asset, completed_bar)
                self._4h_bar_count[asset] += 1
                
                # Reset for next 4H bar
                self._current_4h_bar[asset] = None
                
                logger.debug(f"{asset}: 4H bar completed #{self._4h_bar_count[asset]}")
    
    def check_4h_break(
        self,
        asset: str,
        current_price: float
    ) -> StructureBreakResult:
        """Check structure break on 4H timeframe."""
        return self.analyzer.check_break(asset, current_price)
    
    def validate_4h_entry(
        self,
        asset: str,
        side: str,
        current_price: float
    ) -> Tuple[bool, str]:
        """Validate entry based on 4H structure break."""
        return self.analyzer.validate_entry(asset, side, current_price)
    
    def get_4h_levels(self, asset: str) -> Dict[str, List[float]]:
        """Get 4H fractal levels."""
        return self.analyzer.get_fractal_levels(asset)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'StructureBreakAnalyzer',
    'FourHourStructureTracker',
    'StructureBreakResult',
    'Fractal',
    'FractalType',
    'BreakDirection',
    'PriceBar',
]


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Structure Break Analyzer Test")
    print("=" * 70)
    
    analyzer = StructureBreakAnalyzer(fractal_period=2)
    
    # Generate test data with clear fractals
    np.random.seed(42)
    
    # Create uptrend with pullbacks
    base_prices = [100]
    for i in range(50):
        if i % 10 < 7:  # Uptrend
            base_prices.append(base_prices[-1] * 1.01)
        else:  # Pullback
            base_prices.append(base_prices[-1] * 0.995)
    
    print("\n[1] Updating with price bars...")
    for i, price in enumerate(base_prices):
        high = price * 1.005
        low = price * 0.995
        
        analyzer.update(
            'BTC',
            timestamp=time.time() + i * 60,
            open_=price * 0.999,
            high=high,
            low=low,
            close=price,
            volume=1000000
        )
    
    # Check fractal levels
    levels = analyzer.get_fractal_levels('BTC')
    print(f"\n[2] Detected Fractals:")
    print(f"  Unbroken Highs: {[f'{p:.2f}' for p in levels['highs'][-5:]]}")
    print(f"  Unbroken Lows: {[f'{p:.2f}' for p in levels['lows'][-5:]]}")
    
    # Test structure break
    current_price = base_prices[-1]
    result = analyzer.check_break('BTC', current_price)
    
    print(f"\n[3] Structure Break Check (price={current_price:.2f}):")
    print(f"  Break Detected: {result.break_detected}")
    print(f"  Direction: {result.direction.name}")
    print(f"  Can Long: {result.can_long}")
    print(f"  Can Short: {result.can_short}")
    print(f"  Message: {result.message}")
    
    # Test entry validation
    print("\n[4] Entry Validation:")
    allowed, reason = analyzer.validate_entry('BTC', 'LONG', current_price)
    print(f"  LONG: allowed={allowed}")
    print(f"  Reason: {reason}")
    
    allowed, reason = analyzer.validate_entry('BTC', 'SHORT', current_price)
    print(f"  SHORT: allowed={allowed}")
    print(f"  Reason: {reason}")
    
    # Test with price above fractal high
    print("\n[5] Test with price above highest fractal:")
    if levels['highs']:
        test_price = max(levels['highs']) * 1.01
        result = analyzer.check_break('BTC', test_price)
        print(f"  Price: {test_price:.2f}")
        print(f"  Can Long: {result.can_long}")
        print(f"  Message: {result.message}")
    
    # Test 4H tracker
    print("\n[6] 4H Structure Tracker Test:")
    tracker = FourHourStructureTracker()
    
    # Simulate 10 4H bars
    for bar_idx in range(10):
        for minute in range(240):
            price = 100 + bar_idx * 2 + np.random.randn() * 0.5
            tracker.update_1m(
                'ETH',
                timestamp=time.time() + (bar_idx * 240 + minute) * 60,
                open_=price - 0.5,
                high=price + 1,
                low=price - 1,
                close=price + 0.5,
                volume=100000
            )
    
    levels_4h = tracker.get_4h_levels('ETH')
    print(f"  4H Bars completed: {tracker._4h_bar_count['ETH']}")
    print(f"  4H Fractal Highs: {len(levels_4h['highs'])}")
    print(f"  4H Fractal Lows: {len(levels_4h['lows'])}")
    
    allowed, reason = tracker.validate_4h_entry('ETH', 'LONG', 120)
    print(f"  4H LONG validation: {allowed}")
    print(f"  Reason: {reason}")
    
    print("\n✓ Structure Break Analyzer tests completed!")

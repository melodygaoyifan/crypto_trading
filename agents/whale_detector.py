"""
WhaleDetector + WhalePatternAnalyzer - Exchange-level large order detection (§20).

3-dimensional whale detection:
  1. Absolute notional (> $100K)
  2. Relative trade size (> 10× average)
  3. Relative depth impact (> 5% of 1% orderbook depth)

4 behavioral patterns:
  - ACCUMULATION: Repeated buying, price stable (whales absorbing supply)
  - DISTRIBUTION: Repeated selling, price stable (whales distributing)
  - ICEBERG: Cluster of same-side small orders (whale splitting large order)
  - REVERSAL: Whale direction flips (whale changing stance)

Signals feed into MicrostructureAgent for VETO decisions.
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================================
# Data types
# =========================================================================

@dataclass
class WhaleSignal:
    """Single whale detection event."""
    is_whale: bool = False
    timestamp: float = 0.0
    notional_usd: float = 0.0
    side: str = ""                # "BUY" or "SELL"
    dimensions_triggered: int = 0
    details: List[Tuple[str, float]] = field(default_factory=list)
    # (dimension_name, ratio/value)


@dataclass
class WhalePressure:
    """Aggregate whale pressure over a time window."""
    buy_volume_usd: float = 0.0
    sell_volume_usd: float = 0.0
    net_pressure: float = 0.0     # [-1, +1]: positive=buy, negative=sell
    whale_count: int = 0
    window_seconds: int = 3600


@dataclass
class WhalePatternResult:
    """Identified whale behavioral pattern."""
    pattern: str = "UNKNOWN"      # ACCUMULATION, DISTRIBUTION, ICEBERG, REVERSAL, UNKNOWN
    confidence: float = 0.0       # 0-1
    whale_count: int = 0
    dominant_side: str = ""       # BUY or SELL
    should_veto_long: bool = False
    should_veto_short: bool = False


# =========================================================================
# WhaleDetector
# =========================================================================

class WhaleDetector:
    """Detect large orders on exchange using 3 dimensions.

    Dimensions:
      1. ABSOLUTE: notional_usd > ABSOLUTE_NOTIONAL_USD
      2. RELATIVE_SIZE: notional > RELATIVE_TRADE_SIZE × avg_trade_size
      3. DEPTH_IMPACT: notional > RELATIVE_DEPTH_PCT × orderbook_depth_usd

    A detection requires at least 2 dimensions to trigger.
    """

    ABSOLUTE_NOTIONAL_USD = 100_000.0   # $100K
    RELATIVE_TRADE_SIZE = 10.0           # 10× average trade
    RELATIVE_DEPTH_PCT = 0.05            # 5% of 1% orderbook depth
    MIN_DIMENSIONS = 2                    # At least 2 dims to confirm

    TRADE_HISTORY_SIZE = 10_000
    WHALE_HISTORY_SIZE = 500
    PRESSURE_WINDOW_S = 3600              # 1h window for pressure calc

    DEDUP_HISTORY_SIZE = 5_000

    def __init__(self):
        # [P265] PER-ASSET baselines. A single global deque blended BTC/ETH/
        # SOL notionals, so SOL's "10x average trade" was 10x an average
        # dominated by BTC/ETH tickets — detection systematically suppressed
        # on the small-ticket asset and inflated on the large one, flipping
        # real detections at MIN_DIMENSIONS=2.
        self._trade_sizes: Dict[str, deque] = {}
        self._whale_signals: Dict[str, deque] = {}  # per-asset
        self._avg_trade_size: Dict[str, float] = {}
        # [P265] Bounded per-asset dedup. The pipeline feeds the most recent
        # 1000 trades every 4H tick with NO overlap filter: when fewer than
        # 1000 trades occurred in 4h (routine on SOL), the same whale was
        # re-detected every tick with a fresh detection-time stamp — a trade
        # from DAYS ago registered as "pressure in the last hour" until it
        # rolled out of the top-1000, and the duplicates skewed the size
        # baseline too.
        self._seen_ids: Dict[str, set] = {}
        self._seen_order: Dict[str, deque] = {}

    def _is_duplicate(self, asset: str, trade_id: Optional[str]) -> bool:
        if not trade_id:
            return False
        seen = self._seen_ids.setdefault(asset, set())
        if trade_id in seen:
            return True
        order = self._seen_order.setdefault(
            asset, deque(maxlen=self.DEDUP_HISTORY_SIZE))
        if len(order) == order.maxlen:
            seen.discard(order[0])
        order.append(trade_id)
        seen.add(trade_id)
        return False

    def detect(
        self,
        asset: str,
        trade_notional_usd: float,
        side: str,
        orderbook_depth_usd: float = 0.0,
        trade_ts: Optional[float] = None,
        trade_id: Optional[str] = None,
    ) -> WhaleSignal:
        """Check if a trade qualifies as a whale order.

        Args:
            asset: Asset symbol.
            trade_notional_usd: Trade value in USD.
            side: "BUY" or "SELL".
            orderbook_depth_usd: Total 1% depth on both sides in USD.
            trade_ts: The TRADE's own epoch timestamp. [P265] Signals used to
                be stamped at detection time, so re-fed old trades read as
                fresh flow; pass the venue's trade time whenever known.
            trade_id: Venue trade id for dedup across overlapping fetches.

        Returns:
            WhaleSignal with detection result.
        """
        if self._is_duplicate(asset, trade_id):
            return WhaleSignal(
                is_whale=False,
                timestamp=float(trade_ts) if trade_ts else time.time(),
                notional_usd=trade_notional_usd,
                side=side,
                dimensions_triggered=0,
                details=[],
            )

        signals: List[Tuple[str, float]] = []

        # Dimension 1: Absolute size
        if trade_notional_usd > self.ABSOLUTE_NOTIONAL_USD:
            signals.append(("ABSOLUTE", trade_notional_usd))

        # Dimension 2: Relative to THIS asset's average trade size
        _avg = self._avg_trade_size.get(asset, 0.0)
        if (
            _avg > 0
            and trade_notional_usd > self.RELATIVE_TRADE_SIZE * _avg
        ):
            ratio = trade_notional_usd / _avg
            signals.append(("RELATIVE_SIZE", ratio))

        # Dimension 3: Relative to orderbook depth
        if (
            orderbook_depth_usd > 0
            and trade_notional_usd > self.RELATIVE_DEPTH_PCT * orderbook_depth_usd
        ):
            depth_pct = trade_notional_usd / orderbook_depth_usd
            signals.append(("DEPTH_IMPACT", depth_pct))

        # Update this asset's rolling average
        sizes = self._trade_sizes.setdefault(
            asset, deque(maxlen=self.TRADE_HISTORY_SIZE))
        sizes.append(trade_notional_usd)
        if sizes:
            self._avg_trade_size[asset] = float(np.mean(sizes))

        is_whale = len(signals) >= self.MIN_DIMENSIONS

        result = WhaleSignal(
            is_whale=is_whale,
            # [P265] The trade's own time when known — never detection time.
            timestamp=float(trade_ts) if trade_ts else time.time(),
            notional_usd=trade_notional_usd,
            side=side,
            dimensions_triggered=len(signals),
            details=signals,
        )

        if is_whale:
            if asset not in self._whale_signals:
                self._whale_signals[asset] = deque(maxlen=self.WHALE_HISTORY_SIZE)
            self._whale_signals[asset].append(result)
            logger.info(
                f"[WHALE] {asset}: {side} ${trade_notional_usd:,.0f} "
                f"({len(signals)} dims: {[d[0] for d in signals]})"
            )

        return result

    def get_pressure(self, asset: str) -> WhalePressure:
        """Compute aggregate whale pressure over the last PRESSURE_WINDOW_S.

        Returns:
            WhalePressure with net buy/sell imbalance.
        """
        whales = self._whale_signals.get(asset, deque())
        if not whales:
            return WhalePressure()

        now = time.time()
        cutoff = now - self.PRESSURE_WINDOW_S

        buy_vol = 0.0
        sell_vol = 0.0
        count = 0

        for w in whales:
            if w.timestamp >= cutoff and w.is_whale:
                if w.side == "BUY":
                    buy_vol += w.notional_usd
                else:
                    sell_vol += w.notional_usd
                count += 1

        total = buy_vol + sell_vol
        net = (buy_vol - sell_vol) / total if total > 0 else 0.0

        return WhalePressure(
            buy_volume_usd=buy_vol,
            sell_volume_usd=sell_vol,
            net_pressure=net,
            whale_count=count,
            window_seconds=self.PRESSURE_WINDOW_S,
        )

    def get_recent_whales(self, asset: str, n: int = 20) -> List[WhaleSignal]:
        """Get the N most recent whale signals for an asset."""
        whales = self._whale_signals.get(asset, deque())
        return list(whales)[-n:]


# =========================================================================
# WhalePatternAnalyzer
# =========================================================================

class WhalePatternAnalyzer:
    """Identify 4 whale behavioral patterns from recent signals.

    Patterns:
      ACCUMULATION: >70% buy whales + price stable -> whales absorbing supply
      DISTRIBUTION: >70% sell whales + price stable -> whales distributing
      ICEBERG: Many small-ish orders in same direction in short time
      REVERSAL: Whale direction suddenly flips (>60% reversal in last 5 orders)
    """

    MIN_WHALES = 3                  # Minimum whale events for pattern detection
    DOMINANCE_THRESHOLD = 0.70      # 70% one-sided for accumulation/distribution
    PRICE_STABLE_PCT = 0.01         # <1% move = "price stable"
    REVERSAL_LOOKBACK = 5           # Check last 5 whale events for reversal
    REVERSAL_THRESHOLD = 0.60       # 60% flip to detect reversal

    def analyze(
        self,
        whale_signals: List[WhaleSignal],
        price_start: float,
        price_end: float,
    ) -> WhalePatternResult:
        """Analyze recent whale signals for behavioral patterns.

        Args:
            whale_signals: Recent whale signals (from WhaleDetector).
            price_start: Price at start of analysis window.
            price_end: Price at end of analysis window.

        Returns:
            WhalePatternResult with identified pattern and VETO flags.
        """
        active_whales = [w for w in whale_signals if w.is_whale]

        if len(active_whales) < self.MIN_WHALES:
            return WhalePatternResult(
                pattern="INSUFFICIENT_DATA",
                whale_count=len(active_whales),
            )

        # Count buy vs sell
        buy_count = sum(1 for w in active_whales if w.side == "BUY")
        sell_count = sum(1 for w in active_whales if w.side == "SELL")
        total = buy_count + sell_count
        buy_ratio = buy_count / total if total > 0 else 0.5
        sell_ratio = sell_count / total if total > 0 else 0.5

        # Price stability check
        if price_start > 0:
            price_change = abs(price_end - price_start) / price_start
        else:
            price_change = 0.0
        price_stable = price_change < self.PRICE_STABLE_PCT

        # Dominant side
        dominant_side = "BUY" if buy_count >= sell_count else "SELL"

        # Pattern 1: ACCUMULATION (bulk buying + stable price)
        if buy_ratio >= self.DOMINANCE_THRESHOLD and price_stable:
            confidence = min(1.0, buy_ratio * (1.0 - price_change / self.PRICE_STABLE_PCT))
            logger.info(
                f"[WHALE_PATTERN] ACCUMULATION: buy_ratio={buy_ratio:.2f} "
                f"price_change={price_change:.4f} conf={confidence:.2f}"
            )
            return WhalePatternResult(
                pattern="ACCUMULATION",
                confidence=confidence,
                whale_count=total,
                dominant_side="BUY",
                should_veto_short=True,  # Don't short during accumulation
            )

        # Pattern 2: DISTRIBUTION (bulk selling + stable price)
        if sell_ratio >= self.DOMINANCE_THRESHOLD and price_stable:
            confidence = min(1.0, sell_ratio * (1.0 - price_change / self.PRICE_STABLE_PCT))
            logger.info(
                f"[WHALE_PATTERN] DISTRIBUTION: sell_ratio={sell_ratio:.2f} "
                f"price_change={price_change:.4f} conf={confidence:.2f}"
            )
            return WhalePatternResult(
                pattern="DISTRIBUTION",
                confidence=confidence,
                whale_count=total,
                dominant_side="SELL",
                should_veto_long=True,  # Don't buy during distribution
            )

        # Pattern 3: REVERSAL (direction flip in recent whales)
        if len(active_whales) >= self.REVERSAL_LOOKBACK:
            recent = active_whales[-self.REVERSAL_LOOKBACK:]
            older = active_whales[:-self.REVERSAL_LOOKBACK]

            if older:
                old_buy_ratio = sum(1 for w in older if w.side == "BUY") / len(older)
                new_buy_ratio = sum(1 for w in recent if w.side == "BUY") / len(recent)

                # Significant flip: was mostly buy, now mostly sell (or vice versa)
                flip_magnitude = abs(new_buy_ratio - old_buy_ratio)
                if flip_magnitude >= self.REVERSAL_THRESHOLD:
                    new_dominant = "BUY" if new_buy_ratio > 0.5 else "SELL"
                    confidence = min(1.0, flip_magnitude)
                    logger.info(
                        f"[WHALE_PATTERN] REVERSAL: old_buy={old_buy_ratio:.2f}->"
                        f"new_buy={new_buy_ratio:.2f} (flip={flip_magnitude:.2f})"
                    )
                    return WhalePatternResult(
                        pattern="REVERSAL",
                        confidence=confidence,
                        whale_count=total,
                        dominant_side=new_dominant,
                        # Veto the OLD direction (whales are leaving)
                        should_veto_long=(new_dominant == "SELL"),
                        should_veto_short=(new_dominant == "BUY"),
                    )

        return WhalePatternResult(
            pattern="UNKNOWN",
            whale_count=total,
            dominant_side=dominant_side,
        )


# =========================================================================
# Module-level singletons
# =========================================================================

_detector: Optional[WhaleDetector] = None
_analyzer: Optional[WhalePatternAnalyzer] = None


def get_whale_detector() -> WhaleDetector:
    """Get or create the singleton WhaleDetector."""
    global _detector
    if _detector is None:
        _detector = WhaleDetector()
    return _detector


def get_whale_pattern_analyzer() -> WhalePatternAnalyzer:
    """Get or create the singleton WhalePatternAnalyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = WhalePatternAnalyzer()
    return _analyzer

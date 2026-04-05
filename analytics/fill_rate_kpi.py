"""
SOTA G8: Fill Rate KPI - per-strategy fill rate tracking with feedback.

Tracks each strategy's order fill rate over a 7-day rolling window:
- rate > 70%: OK
- rate 50-70%: WIDEN_PRICE (suggest wider limit price)
- rate < 50%: REDUCE_SIZE (reduce order size by 30%)

Usage:
    kpi = get_fill_rate_kpi()
    kpi.record_order(strategy="momentum", placed=True, filled=True, fill_pct=1.0, asset="BTC")
    advice = kpi.get_advice("momentum")  # FillAdvice(action="OK", fill_rate=1.0, ...)
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 7 * 24 * 3600  # 7-day rolling window


@dataclass
class OrderRecord:
    timestamp: float
    strategy: str
    placed: bool
    filled: bool
    fill_pct: float  # 0.0 - 1.0 (partial fills count proportionally)
    asset: str = ""


@dataclass
class FillAdvice:
    strategy: str
    fill_rate: float       # 0.0 - 1.0
    total_placed: int
    total_filled: int
    action: str            # "OK" | "WIDEN_PRICE" | "REDUCE_SIZE"
    detail: str = ""


class FillRateKPI:
    """Per-strategy fill rate tracker with adaptive feedback."""

    RATE_OK = 0.70         # > 70% = normal
    RATE_WARN = 0.50       # 50-70% = widen price
    MIN_SAMPLES = 5        # need at least 5 orders to produce advice

    def __init__(self, window_seconds: int = WINDOW_SECONDS):
        self._window = window_seconds
        self._records: List[OrderRecord] = []

    def record_order(
        self,
        strategy: str,
        placed: bool = True,
        filled: bool = False,
        fill_pct: float = 0.0,
        asset: str = "",
    ) -> None:
        """Record one order's fill outcome."""
        self._records.append(OrderRecord(
            timestamp=time.time(),
            strategy=strategy,
            placed=placed,
            filled=filled,
            fill_pct=fill_pct,
            asset=asset,
        ))
        if len(self._records) > 2000:
            self._prune()

    def get_advice(self, strategy: str) -> FillAdvice:
        """Return fill-rate advice for a strategy."""
        self._prune()

        records = [r for r in self._records if r.strategy == strategy and r.placed]

        if len(records) < self.MIN_SAMPLES:
            return FillAdvice(
                strategy=strategy,
                fill_rate=1.0,
                total_placed=len(records),
                total_filled=sum(1 for r in records if r.filled),
                action="OK",
                detail=f"insufficient samples ({len(records)}<{self.MIN_SAMPLES})",
            )

        total_placed = len(records)
        total_filled = sum(1 for r in records if r.filled)
        weighted_filled = sum(r.fill_pct for r in records)
        fill_rate = weighted_filled / total_placed

        if fill_rate >= self.RATE_OK:
            action = "OK"
        elif fill_rate >= self.RATE_WARN:
            action = "WIDEN_PRICE"
        else:
            action = "REDUCE_SIZE"

        advice = FillAdvice(
            strategy=strategy,
            fill_rate=fill_rate,
            total_placed=total_placed,
            total_filled=total_filled,
            action=action,
            detail=f"7d rate={fill_rate:.1%} ({total_filled}/{total_placed})",
        )

        if action != "OK":
            logger.info(f"[SOTA] FillRateKPI {strategy}: {action} - {advice.detail}")

        return advice

    def get_all_advice(self) -> Dict[str, FillAdvice]:
        """Return advice for every strategy that has records."""
        strategies = set(r.strategy for r in self._records if r.placed)
        return {s: self.get_advice(s) for s in strategies}

    def _prune(self) -> None:
        """Drop records older than the rolling window."""
        cutoff = time.time() - self._window
        self._records = [r for r in self._records if r.timestamp > cutoff]


_singleton: Optional[FillRateKPI] = None


def get_fill_rate_kpi() -> FillRateKPI:
    """Module-level singleton accessor."""
    global _singleton
    if _singleton is None:
        _singleton = FillRateKPI()
    return _singleton
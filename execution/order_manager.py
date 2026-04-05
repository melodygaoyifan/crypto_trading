"""
execution/order_manager.py - Post-Submission Order Management (S12)

Monitors limit orders after submission and recommends cancel/replace actions
based on LOB drift, time elapsed, and partial fill progress.

Shadow mode by default - logs recommendations only, does NOT auto-cancel.

References:
    - Hafsi & Vittori (2024, arXiv:2411.06389): RL execution with cancel action
    - Schnaubelt (2022, EJOR): PPO limit/cancel preference with imbalance
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class OrderManagerConfig:
    """Configuration for post-submission order management."""
    enabled: bool = False           # shadow mode: log only, no API calls
    max_wait_seconds: float = 30.0  # limit order max wait before escalation
    price_drift_threshold: float = 1.5  # mid-price drift > 1.5× spread -> replace
    imbalance_threshold: float = 0.6    # LOB imbalance against us -> convert
    replace_step_pct: float = 0.3       # replace price moves 30% of spread toward mid
    max_replaces: int = 3               # max replace attempts per order


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OrderState:
    """Tracked state of a monitored limit order."""
    order_id: str
    side: str              # "buy" or "sell"
    price: float
    size: float
    submitted_at: float    # time.time()
    max_wait_seconds: float = 30.0
    replace_count: int = 0
    partial_fill_pct: float = 0.0


@dataclass
class OrderAction:
    """Recommended action for a monitored order."""
    action: str            # HOLD / CANCEL_REPLACE / CANCEL_ONLY / CONVERT_TO_MARKET
    reason: str
    new_price: Optional[float] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "new_price": self.new_price,
            "confidence": round(self.confidence, 3),
        }


# ---------------------------------------------------------------------------
# PostSubmissionOrderManager
# ---------------------------------------------------------------------------

class PostSubmissionOrderManager:
    """
    Monitors submitted limit orders and recommends cancel/replace actions.

    Lifecycle:
        1. register_order() - called after limit order submission
        2. update_fill() - called when partial fill is reported
        3. evaluate_order() - called in 200ms refinement loop
        4. unregister_order() - called after full fill or manual cancel

    All recommendations are logged. In shadow mode (default), no API calls
    are made.
    """

    def __init__(self, config: Optional[OrderManagerConfig] = None):
        self.config = config or OrderManagerConfig()
        self._orders: Dict[str, OrderState] = {}

    def register_order(
        self,
        order_id: str,
        side: str,
        price: float,
        size: float,
        submitted_at: Optional[float] = None,
        max_wait_seconds: Optional[float] = None,
    ) -> None:
        """Register a new limit order for monitoring."""
        self._orders[order_id] = OrderState(
            order_id=order_id,
            side=side,
            price=price,
            size=size,
            submitted_at=submitted_at or time.time(),
            max_wait_seconds=max_wait_seconds or self.config.max_wait_seconds,
        )
        logger.debug(f"[ORDER_MGR] Registered {order_id} {side} {size}@{price}")

    def update_fill(self, order_id: str, partial_fill_pct: float) -> None:
        """Update fill progress for a monitored order."""
        order = self._orders.get(order_id)
        if order:
            order.partial_fill_pct = min(1.0, max(0.0, partial_fill_pct))

    def unregister_order(self, order_id: str) -> None:
        """Remove order from monitoring (filled or cancelled)."""
        self._orders.pop(order_id, None)

    def evaluate_order(self, order_id: str, current_market: Dict) -> OrderAction:
        """
        Evaluate a monitored order and return recommended action.

        Args:
            order_id: The order to evaluate.
            current_market: Dict with keys:
                best_bid, best_ask, mid_price, order_imbalance (-1..1),
                partial_fill_pct (optional override).
        """
        order = self._orders.get(order_id)
        if order is None:
            return OrderAction(action="HOLD", reason="unknown_order", confidence=0.0)

        now = time.time()
        elapsed = now - order.submitted_at
        fill_pct = current_market.get("partial_fill_pct", order.partial_fill_pct)
        order.partial_fill_pct = fill_pct

        best_bid = current_market.get("best_bid", 0.0)
        best_ask = current_market.get("best_ask", 0.0)
        mid_price = current_market.get("mid_price", 0.0)
        imbalance = current_market.get("order_imbalance", 0.0)
        spread = best_ask - best_bid if best_ask > best_bid else 1e-8

        # Rule 1: Mostly filled -> HOLD (wait for remainder)
        if fill_pct >= 0.7:
            return OrderAction(action="HOLD", reason="mostly_filled", confidence=0.9)

        # Rule 2: Deadline approaching + low fill -> CONVERT_TO_MARKET
        if elapsed > order.max_wait_seconds * 0.8 and fill_pct < 0.5:
            return OrderAction(
                action="CONVERT_TO_MARKET",
                reason=f"deadline({elapsed:.0f}s/{order.max_wait_seconds:.0f}s)_low_fill({fill_pct:.0%})",
                confidence=0.85,
            )

        # Rule 3: Price drift -> CANCEL_REPLACE
        if mid_price > 0:
            drift = abs(mid_price - order.price) / spread
            if drift > self.config.price_drift_threshold:
                if order.replace_count >= self.config.max_replaces:
                    return OrderAction(
                        action="CONVERT_TO_MARKET",
                        reason=f"max_replaces({self.config.max_replaces})_exceeded",
                        confidence=0.8,
                    )
                new_price = self._compute_replace_price(order, mid_price, spread)
                return OrderAction(
                    action="CANCEL_REPLACE",
                    reason=f"price_drift({drift:.1f}x_spread)",
                    new_price=new_price,
                    confidence=0.7,
                )

        # Rule 4: LOB imbalance shifted against us -> CONVERT_TO_MARKET
        if abs(imbalance) > self.config.imbalance_threshold:
            against_us = (
                (order.side == "buy" and imbalance < -self.config.imbalance_threshold)
                or (order.side == "sell" and imbalance > self.config.imbalance_threshold)
            )
            if against_us:
                return OrderAction(
                    action="CONVERT_TO_MARKET",
                    reason=f"imbalance_against({imbalance:+.2f})",
                    confidence=0.75,
                )

        # Default: HOLD
        return OrderAction(action="HOLD", reason="within_parameters", confidence=0.5)

    def _compute_replace_price(
        self, order: OrderState, mid_price: float, spread: float
    ) -> float:
        """Compute a more aggressive limit price closer to mid."""
        step = spread * self.config.replace_step_pct
        if order.side == "buy":
            # Move bid up (closer to mid)
            new_price = order.price + step
            return min(new_price, mid_price - spread * 0.05)  # stay below mid
        else:
            # Move ask down (closer to mid)
            new_price = order.price - step
            return max(new_price, mid_price + spread * 0.05)  # stay above mid

    def evaluate_all(self, current_market: Dict) -> Dict[str, OrderAction]:
        """Evaluate all monitored orders. Returns dict of order_id -> OrderAction."""
        results = {}
        for oid in list(self._orders.keys()):
            action = self.evaluate_order(oid, current_market)
            results[oid] = action
            if action.action != "HOLD":
                logger.info(
                    f"[ORDER_MGR] {oid}: {action.action} - {action.reason}"
                    + (f" new_price={action.new_price:.4f}" if action.new_price else "")
                )
        return results

    def apply_replace(self, order_id: str, new_price: float) -> None:
        """Record that a replace was applied (increments counter, updates price)."""
        order = self._orders.get(order_id)
        if order:
            order.replace_count += 1
            order.price = new_price
            order.submitted_at = time.time()  # reset timer

    @property
    def monitored_count(self) -> int:
        return len(self._orders)

    def get_status(self) -> Dict:
        """Return status summary for diagnostics."""
        return {
            "enabled": self.config.enabled,
            "monitored_orders": self.monitored_count,
            "orders": {
                oid: {
                    "side": o.side,
                    "price": o.price,
                    "fill_pct": o.partial_fill_pct,
                    "elapsed_s": round(time.time() - o.submitted_at, 1),
                    "replaces": o.replace_count,
                }
                for oid, o in self._orders.items()
            },
        }
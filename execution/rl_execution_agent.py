"""
================================================================================
HMATS v6 - RL Execution Agent (5-Level Orderbook)
================================================================================
Reinforcement-learning-based execution agent that reads 5-level orderbook
snapshots and selects execution actions (TWAP / VWAP / aggressive / etc.).

Pipeline position: execution layer - receives trade intent from fusion,
selects *how* to execute (slice size, order type, timing).
Does NOT generate alpha - only optimises execution cost.

RULE: All network I/O happens outside this module.  ``decide()`` is pure
      computation on a pre-fetched ``OrderbookSnapshot``.
================================================================================
"""

import logging
import math
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ExecutionAction(Enum):
    """Available execution actions."""
    MARKET = 1
    LIMIT = 2
    TWAP = 3
    VWAP = 4
    ADAPTIVE_SLICE = 5
    ICEBERG = 6
    PASSIVE = 7


# =============================================================================
# ORDERBOOK DATA
# =============================================================================

@dataclass
class OrderbookSnapshot:
    """5-level orderbook snapshot with derived metrics."""

    timestamp: datetime
    symbol: str
    bid_prices: List[float] = field(default_factory=list)
    bid_quantities: List[float] = field(default_factory=list)
    ask_prices: List[float] = field(default_factory=list)
    ask_quantities: List[float] = field(default_factory=list)

    # Derived metrics (populated by calculate_metrics)
    level_spreads_bps: List[float] = field(default_factory=list)
    level_imbalances: List[float] = field(default_factory=list)
    depth_pressure: float = 0.0
    mid_price: float = 0.0

    def calculate_metrics(self) -> None:
        """Compute derived orderbook features in-place."""
        n = min(len(self.bid_prices), len(self.ask_prices), 5)
        if n == 0:
            return

        self.mid_price = (self.bid_prices[0] + self.ask_prices[0]) / 2.0

        self.level_spreads_bps = []
        self.level_imbalances = []

        for i in range(n):
            # Spread at level i
            spread = (self.ask_prices[i] - self.bid_prices[i]) / self.mid_price * 10000.0
            self.level_spreads_bps.append(round(spread, 2))

            # Imbalance at level i
            bq = self.bid_quantities[i] if i < len(self.bid_quantities) else 0
            aq = self.ask_quantities[i] if i < len(self.ask_quantities) else 0
            total = bq + aq
            imb = (bq - aq) / total if total > 0 else 0.0
            self.level_imbalances.append(round(imb, 4))

        # Depth pressure: total bid qty vs total ask qty
        total_bid = sum(self.bid_quantities[:n])
        total_ask = sum(self.ask_quantities[:n])
        total = total_bid + total_ask
        self.depth_pressure = round((total_bid - total_ask) / total, 4) if total > 0 else 0.0

        # Pad to 5 if needed
        while len(self.level_spreads_bps) < 5:
            self.level_spreads_bps.append(0.0)
        while len(self.level_imbalances) < 5:
            self.level_imbalances.append(0.0)

    def estimate_fill_price(
        self, quantity: float, side: str
    ) -> Tuple[float, int]:
        """
        Estimate average fill price for a given order quantity.

        Returns:
            (average_fill_price, levels_consumed)
        """
        if side == "buy":
            prices = self.ask_prices
            quantities = self.ask_quantities
        else:
            prices = self.bid_prices
            quantities = self.bid_quantities

        remaining = quantity
        total_cost = 0.0
        levels = 0

        for i in range(len(prices)):
            if remaining <= 0:
                break
            fill = min(remaining, quantities[i])
            total_cost += fill * prices[i]
            remaining -= fill
            levels += 1

        filled = quantity - remaining
        avg_price = total_cost / filled if filled > 0 else (prices[0] if prices else 0.0)
        return round(avg_price, 6), levels


# =============================================================================
# EXECUTION DECISION
# =============================================================================

@dataclass
class ExecutionDecision:
    """Output of the RL execution agent."""
    action: ExecutionAction
    slice_pct: float  # fraction of order to execute now (0-1)
    limit_offset_bps: float = 0.0
    reason: str = ""


# =============================================================================
# RL EXECUTION AGENT
# =============================================================================

class RLExecutionAgent:
    """
    RL-based execution agent with 5-level orderbook features.

    Two modes:
    - ``use_extended_state=True``  -> 25-dim state (v5.0.5+)
    - ``use_extended_state=False`` -> 15-dim state (v4.5 compat)

    Currently uses a rule-based policy (no trained RL model loaded).
    When a trained model is available, ``decide()`` delegates to it.
    """

    def __init__(self, use_extended_state: bool = True):
        self._extended = use_extended_state
        self._model = None  # placeholder for future RL model

    @property
    def state_dim(self) -> int:
        return 25 if self._extended else 15

    def decide(
        self,
        orderbook: OrderbookSnapshot,
        order_size: float,
        side: str,
        urgency: float = 0.5,
    ) -> ExecutionDecision:
        """
        Select execution action given current orderbook and order params.

        Args:
            orderbook: Pre-fetched 5-level snapshot.
            order_size: Target order size in base units.
            side: "buy" or "sell".
            urgency: 0 (patient) to 1 (immediate).

        Returns:
            ExecutionDecision with action, slice_pct, and limit offset.
        """
        if not orderbook.level_spreads_bps:
            orderbook.calculate_metrics()

        # Estimate fill depth
        _, levels_needed = orderbook.estimate_fill_price(order_size, side)

        # Rule-based policy (placeholder for trained RL)
        if urgency > 0.7:
            if levels_needed <= 1:
                action = ExecutionAction.MARKET
                slice_pct = 1.0
            else:
                action = ExecutionAction.ADAPTIVE_SLICE
                slice_pct = 0.5
        elif urgency < 0.3:
            action = ExecutionAction.PASSIVE
            slice_pct = 0.2
        else:
            if order_size > 3000:
                action = ExecutionAction.ICEBERG
                slice_pct = 0.25
            elif levels_needed > 2:
                action = ExecutionAction.TWAP
                slice_pct = 0.33
            else:
                action = ExecutionAction.LIMIT
                slice_pct = 0.5

        spread = orderbook.level_spreads_bps[0] if orderbook.level_spreads_bps else 5.0
        offset = spread * 0.5

        return ExecutionDecision(
            action=action,
            slice_pct=round(slice_pct, 2),
            limit_offset_bps=round(offset, 1),
            reason=f"urgency={urgency:.1f}, levels={levels_needed}, spread={spread:.1f}bps",
        )

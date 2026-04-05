"""
HMATS v3.2 - FastRiskTick
Purpose: 30-second risk check between 4H decision intervals
Mode: SHADOW (log only, no action) until promoted

[v3.2-A7] Addresses the 4H blind spot: 200ms loop cannot modify
exposure/direction, so extreme moves go unchecked for up to 4 hours.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class FastRiskAction(Enum):
    HOLD = "HOLD"           # No action needed
    REDUCE_50 = "REDUCE_50" # Cut exposure to 50%
    EXIT_ONLY = "EXIT_ONLY" # Flatten entirely


@dataclass
class FastRiskResult:
    action: FastRiskAction
    reason: str
    price_move_pct: float
    triggered_at: float  # timestamp


class FastRiskTick:
    """
    30-second risk evaluator. Runs between 4H ticks.

    Authority: Can only REDUCE or EXIT. Never opens or adds.
    Mode: Starts in SHADOW (log only). Promote via ComponentLifecycle.
    """

    PRICE_MOVE_THRESHOLD = 0.03    # 3%
    VOLATILITY_SPIKE_MULT = 2.0    # 2x normal
    DEPTH_DROP_THRESHOLD = 0.65    # 65% drop (was 50%; BTC orderbook varies ±50% intraday)
    DEPTH_DROP_CONFIRM_STREAK = 3  # 3 consecutive checks ~90s (was 2/60s)
    MIN_VALID_DEPTH_USD = 100_000.0
    REDUCE_COOLDOWN_SEC = 3600.0   # 1h cooldown (was 300s; prevents cascade halving within 4H tick)

    def __init__(self, shadow_mode: bool = True):
        self.shadow_mode = shadow_mode
        self._last_4h_prices: Dict[str, float] = {}
        self._baseline_volatility: Dict[str, float] = {}
        self._baseline_depth: Dict[str, float] = {}
        self._depth_drop_streak: Dict[str, int] = {}
        self._last_reduce_time: Dict[str, float] = {}  # cooldown tracking
        self._trigger_count = 0
        self._shadow_log: list = []
        logger.info(f"[FastRiskTick] Initialized (shadow={shadow_mode})")

    def set_4h_anchor(self, asset: str, price: float,
                      volatility: float = 0.0, depth: float = 0.0):
        """Called after each 4H decision to set reference values."""
        self._last_4h_prices[asset] = price
        if volatility > 0:
            self._baseline_volatility[asset] = volatility
        if depth > 0:
            self._baseline_depth[asset] = depth
        self._depth_drop_streak[asset] = 0

    def on_reduce_executed(self, asset: str, new_depth: float = 0.0):
        """Called after a REDUCE/EXIT action is executed. Refreshes baseline and applies cooldown."""
        now = time.time()
        self._last_reduce_time[asset] = now
        self._depth_drop_streak[asset] = 0
        # Refresh baseline depth to current level so we don't re-trigger on the same drop
        if new_depth > 0:
            self._baseline_depth[asset] = new_depth
            logger.info(f"[FastRiskTick] {asset}: baseline depth refreshed to ${new_depth:,.0f} after REDUCE")

    def evaluate(self, asset: str, market_data: Dict[str, Any]) -> FastRiskResult:
        """Evaluate whether emergency action is needed."""
        now = time.time()
        current_price = market_data.get('current_price', 0)
        anchor_price = self._last_4h_prices.get(asset, current_price)

        if anchor_price <= 0:
            return FastRiskResult(FastRiskAction.HOLD, "no_anchor", 0.0, now)

        price_move_pct = abs(current_price - anchor_price) / anchor_price
        data_valid = bool(market_data.get("data_valid", True))
        if not data_valid:
            self._depth_drop_streak[asset] = 0
            return FastRiskResult(FastRiskAction.HOLD, "data_invalid", price_move_pct, now)

        # Cooldown: skip REDUCE/EXIT if recently triggered (except EXIT_ONLY which always fires)
        _in_cooldown = False
        _last_reduce = self._last_reduce_time.get(asset, 0.0)
        if _last_reduce > 0 and (now - _last_reduce) < self.REDUCE_COOLDOWN_SEC:
            _in_cooldown = True

        # Check triggers - any one fires the highest-severity action
        reason = None
        action = FastRiskAction.HOLD

        # Trigger 1: Price move > 3% (EXIT_ONLY bypasses cooldown)
        if price_move_pct > self.PRICE_MOVE_THRESHOLD:
            action = FastRiskAction.EXIT_ONLY
            reason = f"price_move={price_move_pct:.1%}"

        # Trigger 2: Volatility spike > 2x baseline
        current_vol = market_data.get('volatility_30m', 0.0)
        baseline_vol = self._baseline_volatility.get(asset, 0.0)
        if baseline_vol > 0 and current_vol > baseline_vol * self.VOLATILITY_SPIKE_MULT:
            vol_ratio = current_vol / baseline_vol
            if action == FastRiskAction.HOLD:
                action = FastRiskAction.REDUCE_50
            reason = reason or f"vol_spike={vol_ratio:.1f}x"

        # Trigger 3: Orderbook depth drop > 50% (with stale-data suppression + confirm streak)
        current_depth = market_data.get('orderbook_depth_1pct_usd', 0.0)
        baseline_depth = self._baseline_depth.get(asset, 0.0)
        orderbook_stale = bool(market_data.get("orderbook_stale", False))
        depth_drop = (
            baseline_depth >= self.MIN_VALID_DEPTH_USD
            and current_depth >= self.MIN_VALID_DEPTH_USD
            and not orderbook_stale
            and current_depth < baseline_depth * (1 - self.DEPTH_DROP_THRESHOLD)
        )
        if depth_drop:
            drop_pct = 1 - current_depth / baseline_depth
            streak = self._depth_drop_streak.get(asset, 0) + 1
            self._depth_drop_streak[asset] = streak
            if streak >= self.DEPTH_DROP_CONFIRM_STREAK:
                if action == FastRiskAction.HOLD:
                    action = FastRiskAction.REDUCE_50
                reason = reason or f"depth_drop={drop_pct:.0%}({streak}x)"
        else:
            self._depth_drop_streak[asset] = 0

        # Enforce cooldown for REDUCE_50 (EXIT_ONLY always allowed)
        if _in_cooldown and action == FastRiskAction.REDUCE_50:
            remaining = self.REDUCE_COOLDOWN_SEC - (now - _last_reduce)
            logger.debug(
                f"[FastRiskTick] {asset}: REDUCE_50 suppressed (cooldown {remaining:.0f}s remaining)"
            )
            action = FastRiskAction.HOLD
            reason = None

        result = FastRiskResult(
            action=action,
            reason=reason or "ok",
            price_move_pct=price_move_pct,
            triggered_at=now
        )

        if action != FastRiskAction.HOLD:
            self._trigger_count += 1
            if self.shadow_mode:
                self._shadow_log.append(result)
                logger.warning(
                    f"[FastRiskTick][SHADOW] {asset}: WOULD {action.value} - {reason} "
                    f"(anchor=${anchor_price:,.0f} -> ${current_price:,.0f})"
                )
            else:
                logger.critical(
                    f"[FastRiskTick][LIVE] {asset}: {action.value} - {reason}"
                )

        return result

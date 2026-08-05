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
    # [P110] After a REJECTED emergency exit (e.g. spot locked by stop-loss
    # that can't be cancelled), suppress further EXIT_ONLY firings for
    # EXIT_FAILED_BACKOFF_SEC. set_4h_anchor() clears the backoff so the
    # next 4H tick gets a fresh attempt.
    EXIT_FAILED_BACKOFF_SEC = 1800.0  # 30 min
    # [P156] Every trigger below compares "now" against a reference captured by
    # set_4h_anchor(). That call sits at the END of the 4H decision path
    # (main.py:10166) and is skipped by every early return before it — notably
    # the P0 ABORT at main.py:7998. Nothing bounded how old the reference could
    # get, so an anchor frozen at a healthy value made every later reading look
    # like a catastrophic drop, forever. 1.5 × the 4H tick: long enough that a
    # normally-late tick does not trip it, short enough that two consecutive
    # missed anchors do.
    ANCHOR_MAX_AGE_SEC = 21600.0  # 6h

    def __init__(self, shadow_mode: bool = True):
        self.shadow_mode = shadow_mode
        self._anchor_set_at: Dict[str, float] = {}  # [P156] anchor freshness
        self._anchor_stale_log_at: Dict[str, float] = {}
        self._last_4h_prices: Dict[str, float] = {}
        self._baseline_volatility: Dict[str, float] = {}
        self._baseline_depth: Dict[str, float] = {}
        self._depth_drop_streak: Dict[str, int] = {}
        self._last_reduce_time: Dict[str, float] = {}  # cooldown tracking
        self._exit_failed_at: Dict[str, float] = {}    # [P110] last REJECTED exit ts
        self._exit_failed_reason: Dict[str, str] = {}  # [P110] last REJECTED exit msg
        self._exit_suppress_log_at: Dict[str, float] = {}  # [P110] rate-limit suppression log
        self._trigger_count = 0
        self._shadow_log: list = []
        logger.info(f"[FastRiskTick] Initialized (shadow={shadow_mode})")

    def set_4h_anchor(self, asset: str, price: float,
                      volatility: float = 0.0, depth: float = 0.0):
        """Called after each 4H decision to set reference values."""
        self._last_4h_prices[asset] = price
        self._anchor_set_at[asset] = time.time()  # [P156]
        self._anchor_stale_log_at.pop(asset, None)
        if volatility > 0:
            self._baseline_volatility[asset] = volatility
        if depth > 0:
            self._baseline_depth[asset] = depth
        self._depth_drop_streak[asset] = 0
        # [P110] 4H rebalance clears any failed-exit backoff so the next
        # decision boundary gets a fresh attempt with a new anchor.
        if asset in self._exit_failed_at:
            self._exit_failed_at.pop(asset, None)
            self._exit_failed_reason.pop(asset, None)
            self._exit_suppress_log_at.pop(asset, None)
            logger.info(
                f"[FastRiskTick] {asset}: cleared EXIT_ONLY suppression on "
                f"4H anchor refresh (P110)"
            )

    def on_reduce_executed(self, asset: str, new_depth: float = 0.0):
        """Called after a REDUCE/EXIT action is executed. Refreshes baseline and applies cooldown."""
        now = time.time()
        self._last_reduce_time[asset] = now
        self._depth_drop_streak[asset] = 0
        # Refresh baseline depth to current level so we don't re-trigger on the same drop
        if new_depth > 0:
            self._baseline_depth[asset] = new_depth
            logger.info(f"[FastRiskTick] {asset}: baseline depth refreshed to ${new_depth:,.0f} after REDUCE")

    def on_exit_failed(self, asset: str, reason: str = ""):
        """[P110] Called when an emergency EXIT/REDUCE returns REJECTED.

        Records a backoff timestamp so subsequent evaluate() calls suppress
        EXIT_ONLY (the only action that bypasses the normal cooldown). The
        backoff is cleared by set_4h_anchor() OR after EXIT_FAILED_BACKOFF_SEC,
        whichever comes first. Other triggers (vol spike, depth drop) still
        compose REDUCE_50 normally — only the price_move EXIT_ONLY trigger
        is suppressed.
        """
        now = time.time()
        self._exit_failed_at[asset] = now
        self._exit_failed_reason[asset] = reason
        logger.critical(
            f"[FastRiskTick] {asset}: EXIT_ONLY backoff engaged for "
            f"{self.EXIT_FAILED_BACKOFF_SEC/60:.0f} min after REJECTED "
            f"emergency exit. Reason: {reason or '(no message)'}"
        )

    def evaluate(self, asset: str, market_data: Dict[str, Any]) -> FastRiskResult:
        """Evaluate whether emergency action is needed."""
        now = time.time()
        current_price = market_data.get('current_price', 0)
        anchor_price = self._last_4h_prices.get(asset, current_price)

        if anchor_price <= 0:
            return FastRiskResult(FastRiskAction.HOLD, "no_anchor", 0.0, now)

        # [2026-04-14] Skip during warmup: if no anchor was explicitly set via set_4h_anchor(),
        # the default current_price may be stale from cached pipeline data after network recovery.
        # Only evaluate once at least one set_4h_anchor() call has occurred for this asset.
        if asset not in self._last_4h_prices:
            return FastRiskResult(FastRiskAction.HOLD, "warmup_no_anchor", 0.0, now)

        # [P156] Refuse to act on a stale reference. set_4h_anchor() is the last
        # statement of the 4H decision path, so ANY early return before it (P0
        # ABORT at main.py:7998, and any future one) silently leaves every
        # baseline below frozen — while this evaluator keeps comparing live
        # readings against them every 30s. A depth baseline anchored during a
        # healthy book then makes normal depth look like an 80%+ collapse
        # indefinitely, firing REDUCE_50 forever and ratcheting exposure toward
        # zero. Same failure shape as P155's `_last_quant_directions`
        # high-water mark: state that reads as live but stopped updating.
        # Fail-SAFE: HOLD (this evaluator can only reduce, so refusing to act is
        # always the conservative side).
        _anchor_age = now - self._anchor_set_at.get(asset, 0.0)
        if _anchor_age > self.ANCHOR_MAX_AGE_SEC:
            self._depth_drop_streak[asset] = 0
            _last_log = self._anchor_stale_log_at.get(asset, 0.0)
            if (now - _last_log) > 600.0:
                self._anchor_stale_log_at[asset] = now
                logger.warning(
                    f"[FastRiskTick] {asset}: anchor is {_anchor_age/3600:.1f}h old "
                    f"(max {self.ANCHOR_MAX_AGE_SEC/3600:.1f}h) — HOLDING. The 4H "
                    f"path has not reached set_4h_anchor(); check for an aborted "
                    f"tick. All triggers are suppressed until the anchor refreshes."
                )
            return FastRiskResult(FastRiskAction.HOLD, "anchor_stale", 0.0, now)

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
        # [P110] Suppress EXIT_ONLY if a prior emergency exit was REJECTED
        # within EXIT_FAILED_BACKOFF_SEC. Cleared by set_4h_anchor().
        _price_move_triggered = price_move_pct > self.PRICE_MOVE_THRESHOLD
        _exit_suppressed = False
        if _price_move_triggered:
            _failed_ts = self._exit_failed_at.get(asset, 0.0)
            if _failed_ts > 0 and (now - _failed_ts) < self.EXIT_FAILED_BACKOFF_SEC:
                _exit_suppressed = True
                _last_log = self._exit_suppress_log_at.get(asset, 0.0)
                if (now - _last_log) > 60.0:
                    self._exit_suppress_log_at[asset] = now
                    _remaining = self.EXIT_FAILED_BACKOFF_SEC - (now - _failed_ts)
                    logger.warning(
                        f"[FastRiskTick] {asset}: EXIT_ONLY suppressed "
                        f"(price_move={price_move_pct:.1%}, prior exit "
                        f"REJECTED, backoff {_remaining/60:.1f}min remaining: "
                        f"{self._exit_failed_reason.get(asset, '')})"
                    )
        if _price_move_triggered and not _exit_suppressed:
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

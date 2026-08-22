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
    # [P329] ...but ONLY for a STRUCTURAL rejection. A transient failure to
    # READ the venue is the opposite condition and must not suppress anything:
    # it self-corrects on the next 30s cycle, and retrying costs exactly one
    # reconcile call because the staleness guard in sleeve_fast_risk_action
    # returns BEFORE execute_target — so a retry cannot place an order.
    # Measured incident 2026-08-19 23:39:53: ONE Coinbase 502 disarmed ETH's
    # inter-tick watchdog for 23m09s during a real 7% move.
    TRANSIENT_ESCALATE_AFTER = 6   # ~3 min of an unreadable venue at 30s cadence
    # [P156] Every trigger below compares "now" against a reference captured by
    # set_4h_anchor(). That call sits at the END of the 4H decision path
    # (main.py:10166) and is skipped by every early return before it — notably
    # the P0 ABORT at main.py:7998. Nothing bounded how old the reference could
    # get, so an anchor frozen at a healthy value made every later reading look
    # like a catastrophic drop, forever. 1.5 × the 4H tick: long enough that a
    # normally-late tick does not trip it, short enough that two consecutive
    # missed anchors do.
    ANCHOR_MAX_AGE_SEC = 21600.0  # 6h

    def __init__(self, shadow_mode: bool = True,
                 velocity_trigger: bool = False,
                 price_move_threshold: Optional[float] = None,
                 vol_spike_mult: Optional[float] = None):
        self.shadow_mode = shadow_mode
        # [P367] DEFAULT OFF. Arming it changes a live emergency exit.
        self.velocity_trigger = bool(velocity_trigger)
        # [P370] The two per-instance thresholds, each with a DISABLE value.
        # Backtested six years x three assets (training/risk_control_audit_lab.py,
        # P369): the 3% price-move EXIT_ONLY costs 10-93%/yr of notional, buys
        # no tail protection, is era-unstable on BTC and SOL, and measures a
        # quantity (drift from a RESETTING 4H anchor) with no relationship to
        # the position. The 2x vol-spike REDUCE_50 fires 166-194x/yr for a
        # 29-40%/yr tax and ~zero tail effect. The 10% venue-resting stop
        # (P197) is the control that actually protects, anchored to ENTRY and
        # surviving process death. Published evidence agrees: stops layered on
        # a trend strategy leave the Sharpe the same or lower (York 12/11).
        #
        # None -> the class constant (byte-identical to before). A value
        # <= 0 or >= 1.0 for price_move_threshold, or <= 0 for
        # vol_spike_mult, DISABLES that trigger — a threshold nothing can
        # reach is the honest way to retire a control without deleting the
        # code path the shadow counters still need (P367 keeps measuring
        # both quantities regardless, so the evidence keeps accruing).
        self.price_move_threshold = (
            float(self.PRICE_MOVE_THRESHOLD) if price_move_threshold is None
            else float(price_move_threshold))
        self.vol_spike_mult = (
            float(self.VOLATILITY_SPIKE_MULT) if vol_spike_mult is None
            else float(vol_spike_mult))
        self.price_trigger_enabled = 0.0 < self.price_move_threshold < 1.0
        self.vol_trigger_enabled = self.vol_spike_mult > 0.0
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
        self._venue_unreadable_streak: Dict[str, int] = {}  # [P329] transient reads
        self._venue_unreadable_log_at: Dict[str, float] = {}
        # [P367] velocity trigger: price at the PREVIOUS evaluation, so a
        # move can be measured between ticks instead of against a 4H-old
        # anchor. Shadow counters accrue the evidence for arming it.
        self._last_eval_price: Dict[str, float] = {}
        self._shadow_drift_fires: Dict[str, int] = {}
        self._shadow_velocity_fires: Dict[str, int] = {}
        self._shadow_evals: Dict[str, int] = {}
        self._trigger_count = 0
        self._shadow_log: list = []
        # [P370] the boot line names the state of both triggers, so a
        # retired control is visible as RETIRED rather than silently absent.
        logger.info(
            f"[FastRiskTick] Initialized (shadow={shadow_mode}, "
            f"price_trigger={'ON @' + format(self.price_move_threshold, '.0%') if self.price_trigger_enabled else 'RETIRED'}, "
            f"vol_trigger={'ON @' + format(self.vol_spike_mult, '.1f') + 'x' if self.vol_trigger_enabled else 'RETIRED'})")

    def set_4h_anchor(self, asset: str, price: float,
                      volatility: float = 0.0, depth: float = 0.0):
        """Called after each 4H decision to set reference values."""
        # [P367] Report the anchor period that is ENDING before resetting it:
        # how often each quantity would have fired the emergency exit. One
        # line per asset per 4H bar, so the evidence accrues without becoming
        # wallpaper (P202) — ~19% of evaluations is far too many to log per
        # occurrence, which is the whole finding.
        _n = self._shadow_evals.pop(asset, 0)
        if _n:
            _d = self._shadow_drift_fires.pop(asset, 0)
            _v = self._shadow_velocity_fires.pop(asset, 0)
            logger.info(
                "[FastRiskTick][P367-SHADOW] %s: over %d evaluations, "
                "drift-from-anchor would fire %d (%.1f%%), inter-tick "
                "velocity %d (%.1f%%) — active=%s. Drift measures up to 4h of "
                "cumulative move; velocity measures the gap this control "
                "exists for.",
                asset, _n, _d, 100.0 * _d / _n, _v, 100.0 * _v / _n,
                "velocity" if self.velocity_trigger else "drift")
        self._shadow_drift_fires.pop(asset, None)
        self._shadow_velocity_fires.pop(asset, None)
        # The velocity reference is deliberately NOT cleared here: it is the
        # previous EVALUATION's price, not an anchor, and the 30s loop keeps
        # running across the 4H boundary.

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

    def on_venue_readable(self, asset: str) -> None:
        """[P329] The venue answered — clear any transient-unreadable streak.

        Without this a long-lived process drifts into permanent escalation
        after enough isolated blips (the P303/P265f lesson: a streak counter
        that only ever counts up stops describing the present).
        """
        if self._venue_unreadable_streak.pop(asset, 0):
            self._venue_unreadable_log_at.pop(asset, None)

    def on_exit_failed(self, asset: str, reason: str = "",
                       transient: bool = False):
        """[P110] Called when an emergency EXIT/REDUCE could not be carried out.

        [P329] TWO DIFFERENT CONDITIONS, and conflating them disarmed a live
        safety control.

        STRUCTURAL (`transient=False`, the default and P110's original case):
        the venue REJECTED the order — spot locked by a stop-loss that cannot
        be cancelled, insufficient funds, a rejected reduce. Retrying every 30s
        will fail identically, so a backoff is right: it suppresses the
        price-move EXIT_ONLY trigger (the only action that bypasses the normal
        cooldown) until set_4h_anchor() or EXIT_FAILED_BACKOFF_SEC, whichever
        comes first. Other triggers still compose REDUCE_50 normally.

        TRANSIENT (`transient=True`): we could not READ the venue, so no exit
        was ever attempted. This must NOT suppress anything:

          - it self-corrects — the next 30s cycle usually reconciles fine;
          - retrying is free of execution risk, because the staleness guard in
            `sleeve_fast_risk_action` returns before `execute_target`, so a
            retry cannot place an order;
          - and suppressing it is precisely backwards: an unreadable venue
            during a fast move is when the watchdog matters most.

        Measured incident (2026-08-19 23:39:53): one Coinbase 502 lasting a
        single 30s cycle engaged the 30-minute backoff and disarmed ETH's
        watchdog for 23m09s while ETH moved 7% from its 4H anchor. Nothing was
        lost only because the book happened to be flat.

        The default stays False so the legacy Kraken order-rejection caller is
        unchanged; the sleeve caller passes the flag explicitly.
        """
        now = time.time()
        if transient:
            # No `_exit_failed_at` write: the ACTION is never suppressed.
            streak = self._venue_unreadable_streak.get(asset, 0) + 1
            self._venue_unreadable_streak[asset] = streak
            last = self._venue_unreadable_log_at.get(asset, 0.0)
            if streak == 1 or streak == self.TRANSIENT_ESCALATE_AFTER or (now - last) > 300.0:
                self._venue_unreadable_log_at[asset] = now
                msg = (
                    f"[FastRiskTick] {asset}: could not READ the venue, so the "
                    f"emergency exit was NOT attempted (streak={streak}). "
                    f"The watchdog stays ARMED and retries next cycle; a retry "
                    f"cannot place an order. Reason: {reason or '(no message)'}"
                )
                # Sustained unreadability IS actionable — we may be holding
                # risk we cannot see. An isolated blip is not (P202/P240).
                if streak >= self.TRANSIENT_ESCALATE_AFTER:
                    logger.critical(msg + " — SUSTAINED; check the venue API.")
                else:
                    logger.warning(msg)
            return

        self._exit_failed_at[asset] = now
        self._exit_failed_reason[asset] = reason
        logger.critical(
            f"[FastRiskTick] {asset}: EXIT_ONLY backoff engaged for "
            f"{self.EXIT_FAILED_BACKOFF_SEC/60:.0f} min after REJECTED "
            f"emergency exit. Reason: {reason or '(no message)'}"
        )

    def evaluate(self, asset: str, market_data: Dict[str, Any],
                 has_position: Optional[bool] = None) -> FastRiskResult:
        """Evaluate whether emergency action is needed.

        [P240] `has_position` affects the ALERT SEVERITY ONLY — never the action.
        A REDUCE_50 on a flat asset is unactionable by construction (the P227
        sleeve handler already returns FLAT / "no sleeve position" and does
        nothing), so escalating it to CRITICAL and forwarding it to Discord
        teaches everyone to ignore the channel. Live evidence: 74 CRITICALs for
        `SOL: REDUCE_50 - depth_drop` on 2026-08-08 while SOL was flat and zero
        reduces executed. That is the P202 pattern — an alert nobody can act on.

        FAIL-SAFE, and this is the load-bearing part: only an EXPLICIT False
        downgrades. `None` means "the caller does not know", and an unknown
        position must still alert at full severity — a downgrade must never be
        the DEFAULT, or a caller that simply forgets to pass it silences a real
        emergency. The returned action is byte-identical in every case.
        """
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

        drift_pct = abs(current_price - anchor_price) / anchor_price

        # [P367] VELOCITY: the move since the PREVIOUS evaluation, which is
        # what an inter-tick watchdog is actually for. Measured over ~13,800
        # live samples per asset (P366): a one-step move >= 3% happened 4
        # times, while drift from the 4H anchor was >= 3% on ~19% of samples
        # — a ~650x gap, because drift accumulates over up to four hours
        # while the loop runs every ~34 seconds.
        _prev = self._last_eval_price.get(asset)
        velocity_pct = (abs(current_price - _prev) / _prev
                        if _prev and _prev > 0 else 0.0)
        self._last_eval_price[asset] = current_price

        # Shadow counters — both are always measured so the arming decision
        # rests on forward evidence rather than on one 24h sample (P287's
        # shadow-first pattern). Reported once per anchor refresh.
        self._shadow_evals[asset] = self._shadow_evals.get(asset, 0) + 1
        if drift_pct > self.PRICE_MOVE_THRESHOLD:
            self._shadow_drift_fires[asset] = \
                self._shadow_drift_fires.get(asset, 0) + 1
        if velocity_pct > self.PRICE_MOVE_THRESHOLD:
            self._shadow_velocity_fires[asset] = \
                self._shadow_velocity_fires.get(asset, 0) + 1

        # The quantity the trigger acts on. DEFAULT is the historical drift,
        # so this ships changing nothing (P201 trio; the flag is absent from
        # the live profile and pinned absent).
        price_move_pct = velocity_pct if self.velocity_trigger else drift_pct
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
        #
        # [P364] DIRECTION-BLIND, and that is not obvious from the code above:
        # `price_move_pct` is an ABSOLUTE move from the 4H anchor, so a rally
        # fires the emergency exit exactly as hard as a crash. Measured live
        # over 12h on 2026-08-21 (ETH +6.7%, SOL +4.7%): 32 real flattens of
        # a LONG book, 31 re-fires on an already-flat asset, and the sleeve
        # still ENDED THE WINDOW UP $43.66 at a new equity high — so this is
        # a fee cost in a window it won, not a losing control.
        #
        # Left symmetric ON PURPOSE, pending an operator decision (P141):
        # making it direction-aware would make an emergency exit fire LESS,
        # which is a loosening of a live risk control; and "reduce risk when
        # the market moves 3% inside one 4H bar, whichever way it went" is a
        # defensible design for a 30-second watchdog rather than obviously a
        # bug. What was wrong was that this comment did not SAY so, leaving a
        # reader unable to tell a decision from an oversight (P177/P202).
        # Pinned both directions in tests/test_p364_fastrisk_direction_blind.py.
        # [P110] Suppress EXIT_ONLY if a prior emergency exit was REJECTED
        # within EXIT_FAILED_BACKOFF_SEC. Cleared by set_4h_anchor().
        # [P370] a RETIRED price trigger can never fire; the quantity is still
        # computed and shadow-counted above so the evidence keeps accruing.
        _price_move_triggered = (self.price_trigger_enabled
                                 and price_move_pct > self.price_move_threshold)
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
        # [P370] per-instance multiplier (live 4x) and a retire switch.
        if (self.vol_trigger_enabled and baseline_vol > 0
                and current_vol > baseline_vol * self.vol_spike_mult):
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
                # [P240] Carry the RAW values. "depth_drop=69%" alone cannot
                # distinguish a genuine liquidity collapse from a degraded feed
                # reading, which is exactly the question the 2026-08-08 burst
                # left unanswered and nobody could settle from the log.
                reason = reason or (
                    f"depth_drop={drop_pct:.0%}({streak}x) "
                    f"[depth=${current_depth:,.0f} vs baseline=${baseline_depth:,.0f}]"
                )
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
            elif has_position is False:
                # [P240] Nothing to reduce — the action is returned unchanged,
                # but this is not an emergency anyone can act on.
                logger.info(
                    f"[FastRiskTick][LIVE] {asset}: {action.value} - {reason} "
                    f"(asset is FLAT — no position to act on; action still "
                    f"returned, alert downgraded from CRITICAL)"
                )
            else:
                logger.critical(
                    f"[FastRiskTick][LIVE] {asset}: {action.value} - {reason}"
                )

        return result

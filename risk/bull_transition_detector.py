"""
BullTransitionDetector - Mild Bull Market Regime Detector.

Audit S23: System blind spot. In a mild bull market:
  - Regime != STRONG_BULL -> existing protections don't trigger
  - Short strategies bleed slowly without detection
  - 8 weeks can lose 20-25%

Detection: 4 conditions, 4-stage state machine:
  INACTIVE -> POTENTIAL (1 cond) -> ACTIVE (2+ cond) -> CONFIRMED (5+ days)

Actions:
  ACTIVE:    Halve new short exposure, flag in proof log
  CONFIRMED: Block naked shorts (hedged shorts still allowed)

Fail-open: All exceptions caught, returns INACTIVE on error.

[P383] PER-ASSET INSTANCES. main.py calls `evaluate()` once per ASSET per
tick with that asset's inputs (`relative_alpha_vs_btc`, its funding streak,
its OI/liquidation readings). Before P383 every call hit ONE module-level
singleton, so one state machine was driven by three different input vectors
in sequence: a 2-condition SOL tick set ACTIVE with a fresh entry time, the
following 1-condition BTC tick downgraded it to POTENTIAL (resetting the
entry time again), and the next SOL tick re-entered ACTIVE from scratch.
`days_in_state` could never accumulate and CONFIRMED (5 continuous days) was
a rung that could not fire — while P227b persisted that thrash across
deploys as if it were a state. Same shape as the P306 cascade governor
(one singleton fed per-asset) and the P225 `_last_phase_result` leak.

The registry below mirrors `risk/cascade_exhaustion_governor.py` exactly:
`get_bull_transition_detector(asset=...)` returns one instance per asset,
the no-arg call keeps returning the legacy shared instance (byte-identical
for callers with no asset in scope), `all_bull_transition_states()` /
`restore_bull_transition_states()` persist the whole registry, and the
restore accepts the PRE-P383 flat shape (a single instance's `to_dict`)
into the shared instance — a state file written by the previous build must
not read as "no state", nor be mistaken for a per-asset map.
"""
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """The clock every instance reads. Module-level (not inlined) so tests
    can monkeypatch it and drive `days_in_state` across simulated days —
    the CONFIRMED rung is otherwise untestable without a 5-day wait."""
    return datetime.now(timezone.utc)


def _days_since(entry: Optional[datetime], now: datetime) -> int:
    """Whole days between `entry` and `now`. A NAIVE entry (only ever seen
    from a hand-built payload — the live writer stamps tz-aware UTC) is read
    as UTC rather than raising into the fail-open path (P40/P97)."""
    if entry is None:
        return 0
    if entry.tzinfo is None:
        entry = entry.replace(tzinfo=timezone.utc)
    return (now - entry).days


class BullTransitionState(Enum):
    INACTIVE = "INACTIVE"
    POTENTIAL = "POTENTIAL"
    ACTIVE = "ACTIVE"
    CONFIRMED = "CONFIRMED"


@dataclass
class BullTransitionSignal:
    state: BullTransitionState = BullTransitionState.INACTIVE
    conditions_met: int = 0
    conditions_detail: Dict[str, bool] = field(default_factory=dict)
    days_in_state: int = 0
    action: str = "NONE"  # NONE | REDUCE_SHORT | BLOCK_NAKED_SHORT


class BullTransitionDetector:
    """Detect mild bull market transitions that bleed short strategies.

    4 detection conditions:
      1. BTC price above 50-bar MA (weekly approx via 4H bars)
      2. Relative strength of SOL vs BTC positive over 14 days
      3. Funding rate positive for 7+ consecutive days
      4. OI rising while liquidations declining (supply/demand imbalance)
    """

    ACTIVE_THRESHOLD = 2       # 2+ conditions -> ACTIVE
    CONFIRMED_DAYS = 5         # 5 consecutive days -> CONFIRMED
    DEACTIVATE_BELOW = 0       # 0 conditions -> back to INACTIVE

    def __init__(self, asset: str = ""):
        # [P383] `asset` is a LABEL for the log lines only (which machine
        # transitioned); it never enters the decision. Empty = the legacy
        # shared instance. Registry lookups use the canonical key, not this.
        self.asset = str(asset).upper() if asset else ""
        self._state = BullTransitionState.INACTIVE
        self._state_entry_time: Optional[datetime] = None

    def evaluate(
        self,
        btc_price: float = 0.0,
        btc_ma50: float = 0.0,
        sol_btc_relative_strength: float = 0.0,
        funding_positive_streak_days: int = 0,
        oi_rising: bool = False,
        liquidations_declining: bool = False,
    ) -> BullTransitionSignal:
        """Evaluate bull transition conditions.

        Args:
            btc_price: Current BTC price.
            btc_ma50: 50-bar moving average of BTC (from 4H bars).
            sol_btc_relative_strength: SOL/BTC 14-day relative strength (>0 = SOL outperforming).
            funding_positive_streak_days: Days of consecutive positive funding.
            oi_rising: Whether open interest is increasing.
            liquidations_declining: Whether long liquidations are declining.

        Returns:
            BullTransitionSignal with state, conditions, and recommended action.
        """
        try:
            conditions = {
                "btc_above_ma50": btc_price > btc_ma50 > 0,
                "sol_relative_strength": sol_btc_relative_strength > 0,
                "funding_streak_7d": funding_positive_streak_days >= 7,
                "oi_liq_divergence": oi_rising and liquidations_declining,
            }
            conditions_met = sum(conditions.values())

            prev_state = self._state
            now = _utcnow()

            # State transitions
            if conditions_met >= self.ACTIVE_THRESHOLD:
                if self._state in (BullTransitionState.INACTIVE, BullTransitionState.POTENTIAL):
                    self._state = BullTransitionState.ACTIVE
                    self._state_entry_time = now
                elif self._state == BullTransitionState.ACTIVE:
                    days = _days_since(self._state_entry_time, now)
                    if days >= self.CONFIRMED_DAYS:
                        self._state = BullTransitionState.CONFIRMED
                # CONFIRMED stays CONFIRMED while conditions hold

            elif conditions_met >= 1:
                if self._state == BullTransitionState.INACTIVE:
                    self._state = BullTransitionState.POTENTIAL
                    self._state_entry_time = now
                elif self._state in (BullTransitionState.ACTIVE, BullTransitionState.CONFIRMED):
                    # Downgrade
                    self._state = BullTransitionState.POTENTIAL
                    self._state_entry_time = now

            else:
                self._state = BullTransitionState.INACTIVE
                self._state_entry_time = None

            days_in_state = _days_since(self._state_entry_time, now)

            # Log transitions
            if self._state != prev_state:
                level = logging.WARNING if self._state in (BullTransitionState.ACTIVE, BullTransitionState.CONFIRMED) else logging.INFO
                logger.log(
                    level,
                    f"[BULL-TRANSITION] {self.asset or '<shared>'}: "
                    f"{prev_state.value} -> {self._state.value} "
                    f"({conditions_met} conditions: {conditions})"
                )

            action = self._determine_action()

            return BullTransitionSignal(
                state=self._state,
                conditions_met=conditions_met,
                conditions_detail=conditions,
                days_in_state=days_in_state,
                action=action,
            )

        except Exception as e:
            logger.error(f"[BULL-TRANSITION] Error: {e} -> fail-open INACTIVE")
            return BullTransitionSignal()

    def _determine_action(self) -> str:
        if self._state == BullTransitionState.CONFIRMED:
            return "BLOCK_NAKED_SHORT"
        elif self._state == BullTransitionState.ACTIVE:
            return "REDUCE_SHORT"
        elif self._state == BullTransitionState.POTENTIAL:
            return "REDUCE_SHORT_LIGHT"  # [FIX-L2-02] Early warning: mild reduction
        return "NONE"

    @property
    def is_bull_transition(self) -> bool:
        return self._state in (BullTransitionState.ACTIVE, BullTransitionState.CONFIRMED)

    @property
    def naked_short_blocked(self) -> bool:
        return self._state == BullTransitionState.CONFIRMED

    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            "state": self._state.value,
            "state_entry_time": self._state_entry_time.isoformat() if self._state_entry_time else None,
        }

    def from_dict(self, data: Dict) -> None:
        """Restore from persisted state."""
        try:
            self._state = BullTransitionState(data.get("state", "INACTIVE"))
            entry = data.get("state_entry_time")
            self._state_entry_time = datetime.fromisoformat(entry) if entry else None
        except Exception as e:
            logger.warning(f"[BULL-TRANSITION] Failed to restore state: {e}")
            self._state = BullTransitionState.INACTIVE
            self._state_entry_time = None


# =============================================================================
# [P383] Registry: the legacy shared instance + one instance per asset.
# Mirrors risk/cascade_exhaustion_governor.py (P306) exactly — same key
# canonicalisation, same "" = shared convention, same persist/restore shapes.
# =============================================================================
_detector: Optional[BullTransitionDetector] = None          # shared (legacy)
_detectors: Dict[str, BullTransitionDetector] = {}          # per-asset


def _registry_key(asset: Optional[str]) -> str:
    return str(asset).upper() if asset else ""


def get_bull_transition_detector(
    asset: Optional[str] = None,
) -> BullTransitionDetector:
    """Get or create the detector for `asset` (None/"" = the shared instance).

    [P383] The no-arg call is byte-identical to the pre-P383 singleton
    accessor, so callers that legitimately have no asset in scope (and the
    main.py construction site that only needs a truthy handle) are
    unchanged. The evaluate call site MUST pass `asset=` — three assets fed
    through one machine is the defect this registry exists to end.
    """
    global _detector
    key = _registry_key(asset)
    if not key:
        if _detector is None:
            _detector = BullTransitionDetector()
        return _detector
    inst = _detectors.get(key)
    if inst is None:
        inst = BullTransitionDetector(asset=key)
        _detectors[key] = inst
    return inst


def all_bull_transition_states() -> Dict[str, Dict]:
    """[P383] Every instance's state, keyed by asset ("" = the shared one).

    Without this the per-asset machines would reset on every deploy while
    only the shared instance's (now unfed) state kept being persisted —
    the RAM-only-control class P227b closed for ONE machine, re-opened for
    three (P148/P150/P209 family).
    """
    out: Dict[str, Dict] = {}
    try:
        if _detector is not None:
            out[""] = _detector.to_dict()
        for k, inst in _detectors.items():
            out[k] = inst.to_dict()
    except Exception as e:  # noqa: silent-swallow — logged; persistence only
        logger.warning("[BULL-TRANSITION] state capture failed (%s: %s)",
                       type(e).__name__, e)
    return out


def restore_bull_transition_states(data: Optional[Dict]) -> int:
    """Restore states written by `all_bull_transition_states`. Returns count.

    Accepts the PRE-P383 flat shape too (a single detector's own to_dict:
    {"state": ..., "state_entry_time": ...}), which restores into the shared
    instance — a state file written by the previous build must not be read
    as "no state", nor mistaken for a per-asset map (its keys are field
    names, not asset names). Shape rule, identical to the cascade governor:
    per-asset iff EVERY value is a dict.

    Fail direction (P227b's rule, unchanged): a malformed payload — at the
    top level or inside any one asset's entry — falls back to INACTIVE for
    the instance it could not read; it can only DELAY the shorts-block,
    never falsely CONFIRM it.
    """
    if not isinstance(data, dict) or not data:
        return 0
    per_asset = all(isinstance(v, dict) for v in data.values())
    n = 0
    try:
        if not per_asset:
            get_bull_transition_detector().from_dict(data)
            return 1
        for key, state in data.items():
            if not isinstance(state, dict):
                continue
            inst = get_bull_transition_detector(asset=key or None)
            inst.from_dict(state)
            n += 1
    except Exception as e:  # noqa: silent-swallow — logged; cold start is safe
        logger.warning("[BULL-TRANSITION] state restore failed (%s: %s) — "
                       "starting from INACTIVE", type(e).__name__, e)
    return n


def reset_bull_transition_detectors() -> None:
    """Drop every instance (tests / teardown only)."""
    global _detector
    _detector = None
    _detectors.clear()
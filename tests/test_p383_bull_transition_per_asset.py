"""[P383] BullTransitionDetector — one machine per ASSET, not one singleton
fed three assets' inputs in sequence.

main.py calls `evaluate()` once per asset per tick with THAT asset's inputs.
Pre-P383 every call hit the same module-level singleton, so a 2-condition
SOL tick set ACTIVE (entry time T), the following 1-condition BTC tick
downgraded it to POTENTIAL (entry time reset), and the next SOL tick
re-entered ACTIVE from scratch. `days_in_state` never accumulated and the
CONFIRMED rung (5 continuous days — the only one that blocks naked shorts)
was a check that could not fire (P174 class), while P227b persisted that
thrash as if it were a state.

The registry mirrors risk/cascade_exhaustion_governor.py (P306):
  get_bull_transition_detector(asset=None) -> per-asset, ""/None = shared
  all_bull_transition_states()             -> {asset_or_"": to_dict()}
  restore_bull_transition_states(data)     -> accepts per-asset map AND the
                                              pre-P383 flat to_dict shape

Falsification (2026-08-23): routing every key to the shared instance
(`_detectors.get(key)` -> `_detector`) turned the isolation, CONFIRMED and
persistence-roundtrip tests red; restored byte-identically.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import inspect

import pytest

import risk.bull_transition_detector as bt
from risk.bull_transition_detector import (
    BullTransitionDetector,
    BullTransitionState,
    all_bull_transition_states,
    get_bull_transition_detector,
    reset_bull_transition_detectors,
    restore_bull_transition_states,
)


# Inputs that satisfy exactly N of the detector's four conditions.
TWO_COND = dict(btc_price=65_000.0, btc_ma50=60_000.0,
                sol_btc_relative_strength=0.0,
                funding_positive_streak_days=7,
                oi_rising=False, liquidations_declining=False)
ONE_COND = dict(btc_price=65_000.0, btc_ma50=60_000.0,
                sol_btc_relative_strength=0.0,
                funding_positive_streak_days=0,
                oi_rising=False, liquidations_declining=False)
ZERO_COND = dict(btc_price=0.0, btc_ma50=0.0,
                 sol_btc_relative_strength=-1.0,
                 funding_positive_streak_days=0,
                 oi_rising=False, liquidations_declining=False)


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_bull_transition_detectors()
    yield
    reset_bull_transition_detectors()


class FakeClock:
    """Drives `bt._utcnow` — the ONE clock every instance reads — so days
    in state can be advanced without waiting for them."""

    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw):
        self.now = self.now + timedelta(**kw)


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock(datetime(2026, 8, 23, 0, 0, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(bt, "_utcnow", c)
    return c


# ---------------------------------------------------------------------------
# 1. Registry semantics
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_legacy_noarg_accessor_is_unchanged(self):
        """The pre-P383 call shape still works, still returns ONE object,
        and the signature still admits zero arguments (main.py's ctor site
        and the import at :837 must not break before the parent rewires)."""
        sig = inspect.signature(get_bull_transition_detector)
        assert all(p.default is not inspect.Parameter.empty
                   for p in sig.parameters.values())
        a = get_bull_transition_detector()
        b = get_bull_transition_detector()
        assert a is b
        assert isinstance(a, BullTransitionDetector)

    def test_none_and_empty_string_are_the_shared_instance(self):
        shared = get_bull_transition_detector()
        assert get_bull_transition_detector(None) is shared
        assert get_bull_transition_detector("") is shared
        assert get_bull_transition_detector(asset="") is shared

    def test_per_asset_instances_are_distinct_from_each_other_and_shared(self):
        shared = get_bull_transition_detector()
        btc = get_bull_transition_detector(asset="BTC")
        eth = get_bull_transition_detector(asset="ETH")
        sol = get_bull_transition_detector(asset="SOL")
        ids = {id(shared), id(btc), id(eth), id(sol)}
        assert len(ids) == 4, "per-asset calls must not collapse onto one machine"

    def test_same_asset_returns_the_same_instance_case_insensitively(self):
        assert (get_bull_transition_detector(asset="sol")
                is get_bull_transition_detector(asset="SOL"))

    def test_instance_carries_its_asset_label(self):
        assert get_bull_transition_detector(asset="btc").asset == "BTC"
        assert get_bull_transition_detector().asset == ""

    def test_reset_drops_every_instance(self):
        a = get_bull_transition_detector(asset="BTC")
        s = get_bull_transition_detector()
        reset_bull_transition_detectors()
        assert get_bull_transition_detector(asset="BTC") is not a
        assert get_bull_transition_detector() is not s


# ---------------------------------------------------------------------------
# 2. Per-asset isolation — the defect
# ---------------------------------------------------------------------------

class TestIsolation:
    def test_one_assets_inputs_do_not_reset_anothers_entry_time(self, clock):
        """The live sequence: SOL at 2 conditions, then BTC at 1 condition,
        same tick. Pre-P383 the BTC call downgraded the SOL machine to
        POTENTIAL and reset its entry time every tick."""
        sol = get_bull_transition_detector(asset="SOL")
        btc = get_bull_transition_detector(asset="BTC")

        s0 = sol.evaluate(**TWO_COND)
        assert s0.state is BullTransitionState.ACTIVE
        sol_entry = sol._state_entry_time

        b0 = btc.evaluate(**ONE_COND)
        assert b0.state is BullTransitionState.POTENTIAL

        # The BTC call must not have touched the SOL machine.
        assert sol._state is BullTransitionState.ACTIVE
        assert sol._state_entry_time == sol_entry

        clock.advance(days=1)
        s1 = sol.evaluate(**TWO_COND)
        assert s1.state is BullTransitionState.ACTIVE
        assert sol._state_entry_time == sol_entry, \
            "entry time must survive another asset's tick"
        assert s1.days_in_state == 1

    def test_confirmed_is_reachable_for_one_asset_while_another_stays_potential(self, clock):
        """5 simulated days of SOL=2 conditions interleaved with BTC=1
        condition every tick (the live interleaving). SOL must reach
        CONFIRMED; BTC must sit at POTENTIAL throughout. Under the old
        singleton neither is possible — the machine thrashes ACTIVE <->
        POTENTIAL and days never accumulate."""
        sol = get_bull_transition_detector(asset="SOL")
        btc = get_bull_transition_detector(asset="BTC")
        last_sol = last_btc = None
        for _ in range(6 * 6):          # 6 days of 4H ticks
            last_sol = sol.evaluate(**TWO_COND)
            last_btc = btc.evaluate(**ONE_COND)
            assert last_btc.state is BullTransitionState.POTENTIAL
            clock.advance(hours=4)
        assert last_sol.state is BullTransitionState.CONFIRMED
        assert last_sol.action == "BLOCK_NAKED_SHORT"
        assert last_sol.days_in_state >= BullTransitionDetector.CONFIRMED_DAYS
        assert sol.naked_short_blocked is True
        assert btc.naked_short_blocked is False
        assert get_bull_transition_detector().naked_short_blocked is False, \
            "the shared instance was never fed and must not have moved"

    def test_the_old_singleton_pattern_could_not_confirm(self, clock):
        """Characterisation of the defect: ONE machine fed SOL(2)/BTC(1)
        in sequence never reaches CONFIRMED — pinned so the reason for the
        registry cannot be argued away later."""
        one = BullTransitionDetector()
        seen = set()
        for _ in range(6 * 6):
            seen.add(one.evaluate(**TWO_COND).state)
            seen.add(one.evaluate(**ONE_COND).state)
            clock.advance(hours=4)
        assert BullTransitionState.CONFIRMED not in seen

    def test_zero_conditions_on_one_asset_does_not_deactivate_another(self, clock):
        sol = get_bull_transition_detector(asset="SOL")
        eth = get_bull_transition_detector(asset="ETH")
        sol.evaluate(**TWO_COND)
        eth.evaluate(**ZERO_COND)
        assert eth._state is BullTransitionState.INACTIVE
        assert sol._state is BullTransitionState.ACTIVE


# ---------------------------------------------------------------------------
# 3. Persistence — both shapes
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_all_states_keys_shared_as_empty_string_and_assets_by_name(self):
        get_bull_transition_detector()
        get_bull_transition_detector(asset="BTC").evaluate(**TWO_COND)
        get_bull_transition_detector(asset="SOL")
        states = all_bull_transition_states()
        assert set(states) == {"", "BTC", "SOL"}
        assert states["BTC"]["state"] == "ACTIVE"
        assert states["SOL"]["state"] == "INACTIVE"
        assert states[""]["state"] == "INACTIVE"
        # Each value is a detector's own to_dict shape.
        for v in states.values():
            assert set(v) == {"state", "state_entry_time"}

    def test_all_states_is_empty_when_nothing_was_constructed(self):
        assert all_bull_transition_states() == {}

    def test_per_asset_map_round_trips(self, clock):
        sol = get_bull_transition_detector(asset="SOL")
        btc = get_bull_transition_detector(asset="BTC")
        sol.evaluate(**TWO_COND)
        btc.evaluate(**ONE_COND)
        sol_entry = sol._state_entry_time
        payload = all_bull_transition_states()

        reset_bull_transition_detectors()
        assert restore_bull_transition_states(payload) == 2

        sol2 = get_bull_transition_detector(asset="SOL")
        btc2 = get_bull_transition_detector(asset="BTC")
        assert sol2._state is BullTransitionState.ACTIVE
        assert sol2._state_entry_time == sol_entry
        assert btc2._state is BullTransitionState.POTENTIAL
        # The restored entry time keeps counting — the whole point of P227b.
        clock.advance(days=2)
        assert sol2.evaluate(**TWO_COND).days_in_state == 2

    def test_per_asset_map_with_shared_entry_round_trips(self):
        get_bull_transition_detector().evaluate(**TWO_COND)
        get_bull_transition_detector(asset="ETH").evaluate(**ONE_COND)
        payload = all_bull_transition_states()
        reset_bull_transition_detectors()
        assert restore_bull_transition_states(payload) == 2
        assert get_bull_transition_detector()._state is BullTransitionState.ACTIVE
        assert get_bull_transition_detector(asset="ETH")._state is BullTransitionState.POTENTIAL

    def test_pre_p383_flat_shape_restores_into_the_shared_instance(self):
        """A state file written by the previous build holds ONE detector's
        to_dict. Its keys are field names, not asset names — it must land
        in the shared instance, not be read as a per-asset map (which would
        create detectors called "STATE" and "STATE_ENTRY_TIME")."""
        entry = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
        flat = {"state": "ACTIVE", "state_entry_time": entry.isoformat()}
        assert restore_bull_transition_states(flat) == 1
        shared = get_bull_transition_detector()
        assert shared._state is BullTransitionState.ACTIVE
        assert shared._state_entry_time == entry
        assert set(all_bull_transition_states()) == {""}, \
            "flat shape must not spawn per-asset instances named after fields"

    def test_shape_rule_matches_the_cascade_governor(self):
        """Per-asset iff EVERY value is a dict — the exact predicate
        `restore_governor_states` uses (P306). Pin it by comparing the two
        sources so the two registries cannot drift."""
        import risk.cascade_exhaustion_governor as cg
        a = inspect.getsource(restore_bull_transition_states)
        b = inspect.getsource(cg.restore_governor_states)
        rule = "per_asset = all(isinstance(v, dict) for v in data.values())"
        assert rule in a and rule in b

    @pytest.mark.parametrize("payload", [
        None, {}, [], "garbage", 42,
        {"state": "NOT_A_STATE", "state_entry_time": "garbage"},       # flat, malformed
        {"state": "CONFIRMED", "state_entry_time": "not-a-date"},      # flat, bad date
        {"BTC": {"state": "NOT_A_STATE"}},                             # per-asset, malformed
        {"BTC": {"state": "CONFIRMED", "state_entry_time": "junk"}},   # per-asset, bad date
        {"BTC": ["CONFIRMED"]},                                         # not a dict
    ])
    def test_malformed_payloads_never_confirm(self, payload):
        """P227b's rule: a bad restore may DELAY the shorts-block, never
        falsely CONFIRM it — for the shared instance and every asset."""
        restore_bull_transition_states(payload)
        for inst in [get_bull_transition_detector(),
                     get_bull_transition_detector(asset="BTC")]:
            assert inst._state is BullTransitionState.INACTIVE
            assert inst.naked_short_blocked is False

    def test_a_malformed_asset_entry_does_not_poison_the_others(self):
        entry = datetime(2026, 8, 20, tzinfo=timezone.utc).isoformat()
        payload = {
            "SOL": {"state": "ACTIVE", "state_entry_time": entry},
            "BTC": {"state": "BOGUS", "state_entry_time": "junk"},
        }
        assert restore_bull_transition_states(payload) == 2
        assert get_bull_transition_detector(asset="SOL")._state is BullTransitionState.ACTIVE
        assert get_bull_transition_detector(asset="BTC")._state is BullTransitionState.INACTIVE

    def test_naive_entry_time_from_a_restored_payload_does_not_break_evaluate(self, clock):
        """P40/P97: a naive entry (hand-built payload) must be read as UTC,
        not raise into the fail-open path and report INACTIVE forever."""
        d = get_bull_transition_detector(asset="SOL")
        d.from_dict({"state": "ACTIVE",
                     "state_entry_time": "2026-08-17T00:00:00"})  # naive, 6d ago
        sig = d.evaluate(**TWO_COND)
        assert sig.state is BullTransitionState.CONFIRMED
        assert sig.days_in_state == 6


# ---------------------------------------------------------------------------
# 4. Behaviour of a single machine is unchanged
# ---------------------------------------------------------------------------

class TestSingleMachineUnchanged:
    def test_thresholds_and_actions(self):
        assert BullTransitionDetector.ACTIVE_THRESHOLD == 2
        assert BullTransitionDetector.CONFIRMED_DAYS == 5
        d = BullTransitionDetector()
        assert d.evaluate(**ONE_COND).action == "REDUCE_SHORT_LIGHT"
        assert d.evaluate(**TWO_COND).action == "REDUCE_SHORT"
        assert d.evaluate(**ZERO_COND).action == "NONE"

    def test_evaluate_reads_the_module_clock(self):
        """The clock hook is the ONLY thing that makes the CONFIRMED rung
        testable — pin that evaluate goes through it."""
        src = inspect.getsource(BullTransitionDetector.evaluate)
        assert "_utcnow()" in src
        assert "datetime.now(" not in src

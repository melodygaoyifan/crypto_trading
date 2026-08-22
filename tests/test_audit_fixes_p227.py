"""[P227] System-audit fixes: the P0 equity feed, the un-nested sleeve cycle,
FastRiskTick's sleeve branch, and config honesty.

From the 2026-08-08 docs-vs-code audit. Four defects, one shape each:

1. `pre_tick_update` fed SOTARiskController — the ONLY 35% kill-switch — from
   Kraken-only equity, frozen since 2026-06-13 (P201 §2b's bug on the second
   feed P201 never touched, and this controller CAN veto the sleeve).
2. The sleeve driver + P209 fuse feed + run_live's only _save_paper_positions
   were nested under `if self.audit_manager:` inside the heartbeat try — a
   logging object was load-bearing for order flow.
3. FastRiskTick's 30s watchdog early-returned on the empty Kraken book and
   exited via Kraken orders — structurally unable to touch the sleeve.
   New branch is DEFAULT OFF (`fast_risk_sleeve_enabled`), P141 discipline.
4. `daily_loss_limit` reached P0 as a hardcoded 0.08 through a getattr on a
   nonexistent field; RiskManager ran on dataclass defaults while the JSON
   documented different numbers; four dead keys read as controls.

Source guards are used where the behavior lives inside the 20k-line runner
(the P152 lesson); the sleeve watchdog helper is module-level precisely so it
can be tested functionally here (the P206 pattern).
"""

import asyncio
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAIN = (REPO / "main.py").read_text(encoding="utf-8-sig", errors="replace")


# ---------------------------------------------------------------------------
# Functional: sleeve_fast_risk_action
# ---------------------------------------------------------------------------

class _FakeSleeve:
    def __init__(self, contracts=0, reconcile_ok=True, boom=False):
        self._contracts = contracts
        self._reconcile_ok = reconcile_ok
        self._boom = boom
        self.calls = []

    def reconcile_positions(self):
        return {}

    def signed_contracts(self, asset):
        return self._contracts

    async def execute_target(self, asset, target, order_type="LIMIT",
                             urgent=False):
        # [P270] the watchdog passes urgent=True (emergency exits must never
        # maker-wait); the fake accepts and records it like the real ctor
        self.last_urgent = urgent
        if self._boom:
            raise RuntimeError("venue exploded")
        self.calls.append((asset, target))
        # [P366] was "FILLED" — a status `execute_target` NEVER returns. Its
        # vocabulary is OK / BLOCKED / FAILED / ERROR / NOOP / NOT_READY /
        # SKIPPED_STALE. The placeholder went unnoticed for as long as the
        # helper ignored the status entirely and reported "EXITED" whatever
        # came back; the moment the helper started classifying, this fixture
        # was modelling a venue that does not exist. Fixed at the fixture
        # rather than by widening the success set to admit it (P248).
        return {"status": "OK"}


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def helper():
    import main as m
    return m.sleeve_fast_risk_action


class TestSleeveFastRiskAction:
    def test_disabled_is_reported_not_silent(self, helper):
        sl = _FakeSleeve(contracts=1)
        st, why = _run(helper(sl, "BTC", "EXIT_ONLY", enabled=False))
        assert st == "DISABLED"
        assert sl.calls == []

    def test_no_sleeve(self, helper):
        st, _ = _run(helper(None, "BTC", "EXIT_ONLY", enabled=True))
        assert st == "NO_SLEEVE"

    def test_stale_snapshot_refuses_to_act(self, helper):
        """P141: never trade on last-known state."""
        sl = _FakeSleeve(contracts=1, reconcile_ok=False)
        st, _ = _run(helper(sl, "BTC", "EXIT_ONLY", enabled=True))
        assert st == "SKIPPED_STALE"
        assert sl.calls == []

    def test_flat_is_a_noop(self, helper):
        sl = _FakeSleeve(contracts=0)
        st, _ = _run(helper(sl, "BTC", "EXIT_ONLY", enabled=True))
        assert st == "FLAT"
        assert sl.calls == []

    @pytest.mark.parametrize("contracts", [1, -1])
    def test_exit_only_flattens(self, helper, contracts):
        sl = _FakeSleeve(contracts=contracts)
        st, why = _run(helper(sl, "SOL", "EXIT_ONLY", enabled=True))
        assert st == "EXITED"
        assert sl.calls == [("SOL", 0)]

    @pytest.mark.parametrize("contracts", [1, -1])
    def test_reduce_50_at_one_contract_is_a_logged_noop(self, helper, contracts):
        """Half of one nano contract is not expressible; escalating a reduce
        into a full exit would make the watchdog MORE aggressive than its 4H
        counterpart."""
        sl = _FakeSleeve(contracts=contracts)
        st, why = _run(helper(sl, "ETH", "REDUCE_50", enabled=True))
        assert st == "REDUCE_NOOP"
        assert sl.calls == []

    @pytest.mark.parametrize("contracts,expected", [(2, 1), (-2, -1), (4, 2), (-5, -2)])
    def test_reduce_50_halves_toward_zero_when_expressible(
            self, helper, contracts, expected):
        sl = _FakeSleeve(contracts=contracts)
        st, _ = _run(helper(sl, "ETH", "REDUCE_50", enabled=True))
        assert st == "REDUCED"
        assert sl.calls == [("ETH", expected)]

    def test_venue_error_fails_soft(self, helper):
        """The watchdog must never kill the 30s loop."""
        sl = _FakeSleeve(contracts=1, boom=True)
        st, why = _run(helper(sl, "BTC", "EXIT_ONLY", enabled=True))
        assert st == "ERROR"
        assert "RuntimeError" in why

    def test_unknown_action_ignored(self, helper):
        sl = _FakeSleeve(contracts=1)
        st, _ = _run(helper(sl, "BTC", "SOMETHING_NEW", enabled=True))
        assert st == "IGNORED"
        assert sl.calls == []


# ---------------------------------------------------------------------------
# Wiring guards (P152 lesson: a helper that exists but is not called is
# invisible to unit tests of the helper alone)
# ---------------------------------------------------------------------------

class TestWatchdogWiring:
    def test_handler_routes_sleeve_positions_through_the_helper(self):
        body = MAIN[MAIN.find("async def _handle_fast_risk_action"):]
        body = body[:body.find("\n    async def ", 10)]
        assert "sleeve_fast_risk_action(" in body, (
            "P227 regression: _handle_fast_risk_action no longer consults the "
            "sleeve — the 30s watchdog is structurally inert for the only "
            "positions that exist."
        )
        # The sleeve check must come BEFORE the legacy _paper_positions read.
        assert body.find("sleeve_fast_risk_action(") < body.find(
            "self._paper_positions.get(asset")

    def test_flag_off_logs_instead_of_silence(self):
        assert "fast_risk_sleeve_enabled=false" in MAIN or \
               "fast_risk_sleeve_enabled=False" in MAIN

    def test_config_trio(self):
        """Declared on ProductionConfig AND parsed in from_file (the P201
        trap: getattr-read but never parsed)."""
        assert re.search(r"^\s+fast_risk_sleeve_enabled: bool = False", MAIN, re.M)
        assert 'data.get("fast_risk_sleeve_enabled"' in MAIN


class TestP0EquityFeed:
    def test_pre_tick_update_folds_in_the_sleeve(self):
        blk = MAIN[MAIN.find("P0 SAFETY PRE-TICK UPDATE"):]
        blk = blk[:blk.find("pre_tick_update(") + 400]
        assert '_coinbase_sleeve' in blk, (
            "P227 regression: pre_tick_update is back to Kraken-only equity — "
            "the 35% kill-switch measures a book frozen since 2026-06-13."
        )
        assert "_last_equity_usd" in blk

    def test_partial_book_falls_back_to_last_combined_not_kraken_only(self):
        blk = MAIN[MAIN.find("P0 SAFETY PRE-TICK UPDATE"):]
        blk = blk[:blk.find("pre_tick_update(") + 400]
        assert "_p0_last_combined_equity" in blk, (
            "An unreadable sleeve must feed the last KNOWN combined equity — "
            "a partial book reads as a spurious drawdown exactly when the "
            "venue API is unhealthy."
        )


class TestSleeveCycleUnnested:
    def test_coinbase_block_runs_after_the_heartbeat_handler(self):
        """The order path must be OUTSIDE the heartbeat try and OUTSIDE the
        audit_manager gate."""
        moved = MAIN.find("[P227] The Coinbase block below was moved OUT")
        assert moved > 0, "the P227 move marker is gone"
        hb_handler = MAIN.rfind("except Exception as _hb_err:", 0, moved)
        assert hb_handler > 0, (
            "the heartbeat exception handler must PRECEDE the Coinbase block"
        )
        # Between the heartbeat handler and the moved block there must be no
        # new `try:` that swallows both (i.e. the block is not re-nested).
        gap = MAIN[hb_handler:moved]
        assert "if self.audit_manager" not in gap

    def test_the_block_is_not_gated_on_audit_manager(self):
        moved = MAIN.find("[P227] The Coinbase block below was moved OUT")
        blk = MAIN[moved:moved + 3000]
        first_if = blk.find("if getattr(self.config, \"coinbase_routing_enabled\"")
        assert first_if > 0
        # 16-space indent = while-loop body level, NOT nested under
        # audit_manager (24) or deeper.
        line_start = blk.rfind("\n", 0, first_if) + 1
        indent = first_if - line_start
        assert indent == 16, f"block re-nested (indent={indent}, expected 16)"

    def test_heartbeat_handler_no_longer_claims_the_order_path(self):
        assert "Discord/heartbeat only" in MAIN


class TestConfigHonesty:
    @pytest.fixture(scope="class")
    def live(self):
        return json.loads(
            (REPO / "configs" / "live_high_risk.json").read_text(
                encoding="utf-8-sig"))

    def test_dead_keys_are_gone(self, live):
        for k in ("single_exchange_mode", "allowed_venues", "logging"):
            assert k not in live, (
                f"{k} is back in the live profile — it has zero consumers and "
                f"makes the config overstate what it enforces (P227)."
            )

    def test_the_removal_is_recorded_in_place(self, live):
        assert "_p227_dead_keys_removed" in live

    def test_daily_loss_limit_is_declared_parsed_consumed(self):
        assert re.search(r"^\s+daily_loss_limit: float = 0\.08", MAIN, re.M)
        assert 'risk.get("daily_loss_limit"' in MAIN
        assert '"daily_loss_limit": self.config.daily_loss_limit' in MAIN, (
            "P227 regression: P0 is back on the hardcoded getattr default."
        )

    def test_risk_manager_receives_the_parsed_config(self):
        assert "RiskManager(\n" in MAIN or "RiskManager(" in MAIN
        blk = MAIN[MAIN.find("self.risk_manager = RiskManager("):]
        blk = blk[:600]
        assert "max_position_pct=self.config.max_position_pct" in blk, (
            "P227 regression: RiskManager constructed with config=None again — "
            "the JSON's risk.* numbers and the enforced numbers diverge."
        )

    def test_max_position_pct_default_matches_old_enforcement(self):
        """Absent key must reproduce the OLD ENFORCED value (0.40), not the
        old configured-but-ignored one."""
        assert re.search(r"^\s+max_position_pct: float = 0\.40", MAIN, re.M)


class TestFusionObservability:
    def test_zero_weighted_advise_agents_are_logged(self):
        src = (REPO / "signals" / "authority_fusion.py").read_text(
            encoding="utf-8-sig", errors="replace")
        assert "ADVISE-WEIGHTS]" in src, (
            "The zero-weighting of 12/18 ADVISE agents is invisible again — "
            "restore the one-shot roster log."
        )
        # [P228] and the decision itself must be recorded AT the table.
        assert "DELIBERATELY OFF" in src and "P166" in src, (
            "The P228 decision block above ADVISE_WEIGHTS_BY_REGIME is gone — "
            "the zero-weighting is back to being an undecided accident."
        )


# ---------------------------------------------------------------------------
# P227b cleanup batch
# ---------------------------------------------------------------------------

class TestPromotionGateHonorsPersistedLevel:
    """[P227b] get_authority_level() used to silently restore ACTIVE whenever
    `_demoted_at` was set — a self-reversing demotion mechanism (the exact
    shape P198 removed from main.py). Any demotion through the gate's own
    `_demote` would have been undone by the very next read."""

    def _gate(self, tmp_path, payload):
        import json as _json
        from drl.promotion_gate import DRLPromotionGate
        f = tmp_path / "state.json"
        f.write_text(_json.dumps(payload), encoding="utf-8")
        return DRLPromotionGate(state_file=str(f))

    def test_persisted_shadow_with_demoted_at_stays_shadow(self, tmp_path):
        g = self._gate(tmp_path, {
            "authority_level": "SHADOW",
            "demoted_at": "2026-08-07T01:45:00",
            "peak_equity": 0.0, "current_equity": 0.0,
            "demotion_history": [],
        })
        # Pre-P227b this returned "ACTIVE" and SAVED it — a read with a
        # promotion side-effect.
        assert g.get_authority_level() == "SHADOW"
        # And repeated reads must not drift.
        assert g.get_authority_level() == "SHADOW"

    def test_a_demotion_through_the_gates_own_api_sticks(self, tmp_path):
        g = self._gate(tmp_path, {
            "authority_level": "ACTIVE", "demoted_at": None,
            "peak_equity": 0.0, "current_equity": 0.0,
            "demotion_history": [],
        })
        g._demote("test demotion")  # sets _demoted_at at the end
        level_after = g.get_authority_level()
        assert level_after != "ACTIVE", (
            "P227b regression: the gate un-demoted itself on the next read."
        )

    def test_fresh_volume_boots_disabled_not_active(self, tmp_path):
        """The audit's 'repo-tracked state says ACTIVE' scare was a false
        positive — pin the actual fresh-boot behavior."""
        from drl.promotion_gate import DRLPromotionGate
        g = DRLPromotionGate(state_file=str(tmp_path / "nonexistent.json"))
        assert g.get_authority_level() == "DISABLED"


class TestBullTransitionPersistence:
    """[P227b] to_dict/from_dict had ZERO callers — CONFIRMED (5 continuous
    days) could never arm across the deploy cadence (P148/P150/P209 class)."""

    def test_state_round_trips(self):
        from datetime import datetime, timedelta
        from risk.bull_transition_detector import (
            BullTransitionDetector, BullTransitionState)
        d = BullTransitionDetector()
        d._state = BullTransitionState.ACTIVE
        d._state_entry_time = datetime(2026, 8, 1, 12, 0, 0)
        d2 = BullTransitionDetector()
        d2.from_dict(d.to_dict())
        assert d2._state == BullTransitionState.ACTIVE
        assert d2._state_entry_time == d._state_entry_time

    def test_malformed_payload_falls_back_to_inactive(self):
        """Conservative direction: a bad restore may DELAY the shorts-block,
        never falsely CONFIRM it."""
        from risk.bull_transition_detector import (
            BullTransitionDetector, BullTransitionState)
        d = BullTransitionDetector()
        d.from_dict({"state": "NOT_A_STATE", "state_entry_time": "garbage"})
        assert d._state == BullTransitionState.INACTIVE

    def test_save_payload_carries_it(self):
        assert '"bull_transition_state"' in MAIN

    def test_restore_calls_from_dict(self):
        assert "self._bull_detector.from_dict(bt_data)" in MAIN

    def test_restore_is_in_the_governor_section_not_positions_gated(self):
        """P211: run_live restores with restore_positions=False; the bull
        restore must sit with the governors (unconditional), so find it AFTER
        the positions gate but verify it does not reference restore_positions."""
        idx = MAIN.find("bt_data = data.get(\"bull_transition_state\"")
        assert idx > 0
        surrounding = MAIN[idx - 400:idx]
        assert "if restore_positions" not in surrounding


class TestV51AttributionEntry:
    """[P227b] The one P8 3-file-rule violation closed: extractor existed,
    _attr_collected entry did not."""

    def test_collected_entry_exists(self):
        assert '"v5_1_strats": {k: agent_signals.get(k, 0.0) for k in' in MAIN

    def test_collected_keys_match_what_the_extractor_reads(self):
        """The rule breaks at KEY level, so pin the keys, not just presence."""
        env = (REPO / "agents" / "signal_envelope.py").read_text(
            encoding="utf-8-sig", errors="replace")
        for key in ("v5_1_strats_direction", "v5_1_strats_confidence"):
            assert key in env, f"extractor no longer reads {key}"
            assert f'"{key}"' in MAIN, f"_attr_collected no longer passes {key}"


class TestLoopControllerRetired:
    def test_main_no_longer_imports_the_200ms_loop(self):
        assert "from execution.loop_controller import" not in MAIN, (
            "P227b regression: the dead 200ms loop import is back in the "
            "EXECUTION_AVAILABLE try-block — an ImportError in dead code "
            "would silently disable the REAL ExecutionManager."
        )

    def test_the_module_itself_survives_for_tests(self):
        assert (REPO / "execution" / "loop_controller.py").exists()


class TestRegimeSmootherParity:
    """[P227b] Runtime re-implements the smoothing inline in the pipeline.
    Full object-level parity needs a constructed pipeline; what is pinned
    here: (a) the core class matches an INDEPENDENTLY-written oracle (not a
    copy of either implementation), (b) the pipeline block still carries the
    same structural machine, so a semantic edit to either side trips a test."""

    @staticmethod
    def _oracle(seq, n):
        """Hold current until the newcomer appears n times consecutively."""
        out, cur, cand, streak = [], None, None, 0
        for r in seq:
            if cur is None:
                cur = r
            elif r == cur:
                cand, streak = None, 0
            elif r == cand:
                streak += 1
                if streak >= n:
                    cur, cand, streak = r, None, 0
            else:
                cand, streak = r, 1
            out.append(cur)
        return out

    def test_core_class_matches_oracle_on_random_sequences(self):
        import random
        import pandas as pd
        from core.regime_smoother import RegimeSmoother
        rng = random.Random(42)
        for trial in range(20):
            seq = [rng.choice("ABC") for _ in range(60)]
            df = pd.DataFrame({"regime": seq})
            got = RegimeSmoother(min_persistence=2).smooth_column(
                df, "regime")["regime"].tolist()
            assert got == self._oracle(seq, 2), f"trial {trial}: {seq}"

    def test_pipeline_inline_machine_still_has_the_same_structure(self):
        src = (REPO / "data_mgmt" / "market_data_pipeline.py").read_text(
            encoding="utf-8-sig", errors="replace")
        for token in (
            '_state["count"] >= self._regime_smoother_persistence',
            '{"current": gmm_regime_name, "pending": None, "count": 0}',
        ):
            assert token in src, (
                f"pipeline smoother structure changed ({token!r} gone) — "
                f"re-verify parity with core/regime_smoother.py"
            )


class TestHeartbeatDiagnosticsP229:
    """[P229] Three heartbeat readability defects found by inspecting the
    first post-deploy tick: (1) the VETOED branch read
    _dashboard_asset_runtime["veto_reason"], a key with NO writer in that
    dict (P170 orphan-read shape) — a live alpha-gate veto displayed as bare
    "NOT_CALLED"; (2) the P152 routing skip displayed as bare "SKIPPED",
    reading as a fault (P162 shape); (3) "NO RESULT YET — driver has never
    run" fired on every first tick after a restart while the driver was
    seconds from running."""

    def test_veto_is_read_from_the_intent_not_the_orphan_key(self):
        hb = MAIN[MAIN.find("_hb_intent = _live_intents.get(_hb_asset)"):]
        hb = hb[:600]
        assert "veto_active" in hb and "veto_reason" in hb
        # the orphan read must NOT come back
        assert '_hb_veto = self._dashboard_asset_runtime' not in MAIN, (
            "P229 regression: the heartbeat veto is back to reading a key "
            "nothing writes — the VETOED branch will never fire again."
        )

    def test_benign_routing_skip_is_labelled_not_bare(self):
        assert "KR-entry-skip" in MAIN

    def test_first_tick_sleeve_text_no_longer_claims_never_ran(self):
        """[P229, updated P263] The durable contract: the first-tick sleeve
        text must never claim a fault (the old all-caps never-ran text), and
        must state that the driver runs after the message. P263 replaced the
        P229 wording with a reconciled-book lead + 'manage pending'; the old
        assertion pinned the exact P229 phrase and broke on a line-wrap in
        the new string literal — pin the contract, not the phrasing (P171)."""
        assert "NO RESULT YET — driver has never run" not in MAIN
        assert "no result yet this process" not in MAIN  # P263: retired too
        assert "manage pending (driver" in MAIN, (
            "the first-tick sleeve text no longer says the driver runs "
            "after the message — the next reader will misread idle as fault"
        )


class TestGrowOnlyRosterLog:
    """[P229] The one-shot bool latch under-reported the zero-weight roster
    (fired on tick 1 with a barely-populated signals dict: named 2 of 12).
    Now a set-based latch converges to the full roster."""

    def test_set_based_latch_replaced_the_bool(self):
        src = (REPO / "signals" / "authority_fusion.py").read_text(
            encoding="utf-8-sig", errors="replace")
        assert "_p228_zero_weight_named" in src
        assert "_new_names = _zero_weighted - _already" in src
        assert "_p227_zero_weight_logged" not in src, (
            "the tick-1 bool latch is back — it under-reports the roster"
        )


class TestBetaAuditRefusesLoudly:
    def test_missing_input_exits_2_before_reporting(self, tmp_path):
        """P199 pattern: 'no data source' must be a refusal, not an empty
        report. Run from a cwd with no data/ so the file is absent."""
        import os
        import subprocess
        import sys as _sys
        env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
        r = subprocess.run(
            [_sys.executable, "-X", "utf8",
             str(REPO / "scripts" / "run_beta_audit.py")],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
        encoding="utf-8")
        assert r.returncode == 2, (r.returncode, r.stderr[-300:])
        assert "REFUSING TO REPORT" in r.stderr

"""[P232] The four research stages implemented: gate hysteresis (shadow +
default-OFF enforce), sleeve re-entry cooldown (default OFF), RegimeICFusion
shadow wiring + persistence, shadow slope calibrator — plus the consensus
roster fix. All from the P231 research; nothing here changes live behavior
at default config.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAIN = (REPO / "main.py").read_text(encoding="utf-8-sig", errors="replace")
V36 = (REPO / "integration" / "integration_v36.py").read_text(
    encoding="utf-8-sig", errors="replace")


class TestGateHysteresis:
    """[P234] Behavioral tests of gate_hysteresis_decision() — the block was
    extracted to a pure function after the inline version shipped dead: it
    read the intent's direction field before fusion assigned it, so
    agreement was always False, the shadow line always said WOULD-EXIT, and
    enforcement could never fire. The old tests here asserted source
    substrings and passed on that dead code."""

    def test_shadow_line_always_logs_the_counterfactual(self):
        assert "[GATE-HYST]" in V36
        assert "WOULD-HOLD" in V36 and "WOULD-EXIT" in V36

    def test_p234_agreement_input_is_the_pre_fusion_signal(self):
        """The exact regression: the hysteresis block must feed the signal
        the gate judged (_alpha_input_direction), and must not read the
        intent's not-yet-assigned direction field."""
        blk = V36[V36.find("[P232] Gate hysteresis"):]
        blk = blk[:blk.find("pure new entry")]
        assert "gate_hysteresis_decision(" in blk
        assert "_alpha_input_direction" in blk, (
            "P234 regression: the hold band no longer receives the "
            "pre-fusion signal direction"
        )
        assert 'getattr(intent, "direction"' not in blk, (
            "P234 regression: intent.direction is 0.0 until STEP 8 fusion — "
            "reading it here makes the hold band dead code again"
        )

    def test_a_held_agreeing_position_actually_holds(self):
        """The case that was impossible before P234: long 1 contract, long
        signal, alpha inside the band -> enforce holds."""
        from integration.integration_v36 import gate_hysteresis_decision
        agrees, shadow, enforce = gate_hysteresis_decision(
            sleeve_position_contracts=1, hold_ratio=0.65,
            signal_direction=0.4, alpha_bps=30.0, threshold_bps=40.0)
        assert agrees and shadow and enforce

    def test_zero_signal_never_holds(self):
        """The dead-code state the old block was permanently stuck in
        (direction read as 0.0) must itself never grant a hold."""
        from integration.integration_v36 import gate_hysteresis_decision
        agrees, shadow, enforce = gate_hysteresis_decision(
            1, 0.65, 0.0, 1000.0, 40.0)
        assert not agrees and not shadow and not enforce

    def test_a_flip_is_never_hold_banded(self):
        """Reversals keep the full enter threshold + P198 persistence —
        even with alpha far above the bar."""
        from integration.integration_v36 import gate_hysteresis_decision
        agrees, shadow, enforce = gate_hysteresis_decision(
            1, 0.65, -0.9, 1000.0, 40.0)
        assert not agrees and not shadow and not enforce
        agrees, shadow, enforce = gate_hysteresis_decision(
            -1, 0.65, 0.9, 1000.0, 40.0)
        assert not agrees and not shadow and not enforce

    def test_a_real_exit_still_exits(self):
        """Alpha below ratio x threshold does not hold — the band widens
        the exit side only down to the ratio, never further."""
        from integration.integration_v36 import gate_hysteresis_decision
        agrees, shadow, enforce = gate_hysteresis_decision(
            1, 0.65, 0.4, 20.0, 40.0)
        assert agrees and not shadow and not enforce

    def test_enforcement_requires_positive_ratio(self):
        """Default-OFF contract: ratio 0 accumulates shadow evidence at
        0.65 but never enforces."""
        from integration.integration_v36 import (
            gate_hysteresis_decision, GATE_HYST_SHADOW_RATIO)
        assert GATE_HYST_SHADOW_RATIO == 0.65
        agrees, shadow, enforce = gate_hysteresis_decision(
            1, 0.0, 0.4, 30.0, 40.0)
        assert agrees and shadow and not enforce

    def test_short_side_holds_symmetrically(self):
        from integration.integration_v36 import gate_hysteresis_decision
        agrees, shadow, enforce = gate_hysteresis_decision(
            -1, 0.65, -0.4, 30.0, 40.0)
        assert agrees and shadow and enforce

    def test_degenerate_threshold_never_holds(self):
        """threshold <= 0 must not grant a hold — a broken gate must not
        become a standing exemption."""
        from integration.integration_v36 import gate_hysteresis_decision
        for thresh in (0.0, -5.0):
            agrees, shadow, enforce = gate_hysteresis_decision(
                1, 0.65, 0.4, 30.0, thresh)
            assert not shadow and not enforce

    def test_hold_marker_precedes_the_pure_entry_veto(self):
        held = V36.find('getattr(intent, "alpha_gate_hold", False)')
        veto = V36.find("pure new entry, block it")
        assert 0 < held < veto

    def test_reads_the_sleeve_key_not_current_exposure(self):
        """Overloading current_exposure would re-arm every Kraken-shaped
        consumer with sleeve semantics (P139/P140 class)."""
        assert 'market_data.get("sleeve_position_contracts"' in V36
        assert 'market_data["sleeve_position_contracts"]' in MAIN

    def test_secondary_veto_respects_the_hold(self):
        blk = MAIN[MAIN.find("FRICTION_EXCEEDS_EDGE") - 2500:
                   MAIN.find("FRICTION_EXCEEDS_EDGE") + 200]
        assert 'getattr(intent, "alpha_gate_hold", False)' in blk, (
            "P232 regression: the v9-PATCH-2 friction veto silently "
            "overrides the hold band again"
        )

    def test_config_trio_and_decided_value(self):
        """[P237] The operator decision was made (delegated, 2026-08-08):
        hold band ON at the researched 0.65. The dataclass default stays 0
        (absent key = off); the LIVE value is pinned so a silent revert or a
        silent widening both fail loudly."""
        assert re.search(r"^\s+alpha_gate_hold_ratio: float = 0\.0", MAIN, re.M)
        assert 'data.get("alpha_gate_hold_ratio"' in MAIN
        live = json.loads((REPO / "configs" / "live_high_risk.json"
                           ).read_text(encoding="utf-8-sig"))
        assert live.get("alpha_gate_hold_ratio") == 0.65, (
            "P237 decision drifted: hold ratio is not the decided 0.65 — "
            "changing it needs its own recorded decision"
        )


class TestReentryCooldown:
    def test_branch_exists_and_only_blocks_entry_from_flat(self):
        blk = MAIN[MAIN.find("[P232] Re-entry cooldown"):]
        blk = blk[:2200]
        assert "_cd_pre == 0" in blk, (
            "the cooldown must require a FLAT book — anything else can "
            "defer an exit (P195 violation)"
        )
        assert "[COINBASE-COOLDOWN]" in blk

    def test_stop_still_reconciled_on_the_skip_path(self):
        blk = MAIN[MAIN.find("[P232] Re-entry cooldown"):]
        blk = blk[:blk.find("manage_to_signal")]
        assert "ensure_protective_stop" in blk

    def test_flatten_events_are_recorded(self):
        assert "_sleeve_flatten_tick[_m_a]" in MAIN

    def test_config_trio_and_decided_value(self):
        """[P237] Cooldown ON at 2 ticks (8h) by the delegated operator
        decision — tightening-only, P168 evidence."""
        assert re.search(
            r"^\s+coinbase_reentry_cooldown_ticks: int = 0", MAIN, re.M)
        assert 'data.get("coinbase_reentry_cooldown_ticks"' in MAIN
        live = json.loads((REPO / "configs" / "live_high_risk.json"
                           ).read_text(encoding="utf-8-sig"))
        assert live.get("coinbase_reentry_cooldown_ticks") == 2


class TestRegimeICFusionWiring:
    """[P231]: the module had ZERO importers while being the direct
    implementation of the P228 promotion path's evidence accumulation."""

    def test_main_now_imports_and_drives_it(self):
        assert "from signals.regime_ic_fusion import RegimeICFusion" in MAIN
        assert "record_outcome" in MAIN and "shadow_fuse" in MAIN
        assert "[RIC-SHADOW]" in MAIN

    def test_state_is_persisted_and_restored(self):
        """RAM-only evidence restarts the clock on every deploy (P150)."""
        assert '"regime_ic_state"' in MAIN
        assert "RegimeICFusion restore failed" in MAIN or \
               "Restored RegimeICFusion" in MAIN

    def test_module_round_trips(self):
        from signals.regime_ic_fusion import RegimeICFusion
        r = RegimeICFusion()
        for _ in range(10):
            r.record_outcome("CHOP", {"quant": 0.5, "whale": -0.2}, 0.001)
        r2 = RegimeICFusion()
        r2.from_dict(r.to_dict())
        assert r2.ic("quant", "CHOP") == r.ic("quant", "CHOP")

    def test_shadow_fuse_is_log_only_by_construction(self):
        """The wiring must never touch the intent — pin that no intent
        attribute is written inside the RIC block."""
        blk = MAIN[MAIN.find("[P232] RegimeICFusion SHADOW wiring"):]
        blk = blk[:blk.find("Call v3.6 engine")]
        assert "intent." not in blk


class TestSlopeCalibrator:
    def test_refuses_on_missing_logs(self, tmp_path):
        env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
        r = subprocess.run(
            [sys.executable, "-X", "utf8",
             str(REPO / "analytics" / "calibration" / "slope_calibrator.py"),
             "--log-dir", str(tmp_path / "nope")],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 2
        assert "REFUSING TO REPORT" in r.stderr

    def test_negative_slope_floors_to_zero_never_inverts(self):
        from analytics.calibration.slope_calibrator import shrunk_slope
        pairs = [(d / 10.0, -8.0 * d / 10.0) for d in range(-20, 21) if d] * 3
        r = shrunk_slope(pairs, overlap=1)
        assert r["slope_ols"] < 0
        assert r["slope_published"] == 0.0 and r["floored"]

    def test_huge_slope_caps_at_todays_effective_ceiling(self):
        from analytics.calibration.slope_calibrator import (
            shrunk_slope, SLOPE_CAP)
        pairs = [(d / 10.0, 500.0 * d / 10.0) for d in range(-20, 21) if d] * 20
        r = shrunk_slope(pairs, overlap=1)
        assert r["slope_published"] == SLOPE_CAP and r["capped"], (
            "the calibrator may never claim MORE edge than the current "
            "constants — anything above the cap is a gate-loosening"
        )

    def test_insufficient_n_refuses_a_number(self):
        from analytics.calibration.slope_calibrator import shrunk_slope
        assert shrunk_slope([(0.5, 10.0)] * 10, 1)["verdict"] == "INSUFFICIENT"

    def test_shrinkage_downweights_small_samples(self):
        from analytics.calibration.slope_calibrator import shrunk_slope
        small = shrunk_slope([(d / 10.0, 30.0 * d / 10.0)
                              for d in range(-20, 21) if d], overlap=1)
        big = shrunk_slope([(d / 10.0, 30.0 * d / 10.0)
                            for d in range(-20, 21) if d] * 20, overlap=1)
        assert small["slope_published"] < big["slope_published"]


class TestConsensusRosterFix:
    def test_sentiment_is_in_the_boost_roster(self):
        src = (REPO / "signals" / "authority_fusion.py").read_text(
            encoding="utf-8-sig", errors="replace")
        i = src.find("_advise_names = [")
        roster = src[i:i + 200]
        assert '"sentiment"' in roster, (
            "P232 regression: the consensus boost roster omits the only "
            "weighted agent that fires — the boost is structurally dead again"
        )


class TestTripwireActuatorAndChecker:
    """[P237] The calibration tripwire made executable: a per-asset actuator
    (trend_assets) and a Monday checker that states FIRED loudly but NEVER
    edits config (deactivating live behavior stays a recorded human step)."""

    def test_actuator_config_trio_default_all_three(self):
        assert 'data.get("trend_assets", ["BTC", "ETH", "SOL"])' in MAIN
        live = json.loads((REPO / "configs" / "live_high_risk.json"
                           ).read_text(encoding="utf-8-sig"))
        assert "trend_assets" not in live, (
            "trend_assets appeared in the live profile — firing the tripwire "
            "needs its own recorded decision referencing the report evidence"
        )

    def test_call_site_skips_excluded_assets(self):
        blk = MAIN[MAIN.find("[P237] Tripwire ACTUATOR"):]
        blk = blk[:blk.find("get_trend_decision_layer")]
        assert "asset not in _trend_assets" in blk
        assert "EXCLUDED by trend_assets" in blk

    def _write_report(self, d, day, closed_assets):
        rep = {"generated": f"{day}T06:20:00+00:00", "assets": {}}
        for a in ("BTC", "ETH", "SOL"):
            v = ("GATE-CLOSED under honest calibration"
                 if a in closed_assets else "TRADEABLE")
            rep["assets"][a] = {"4h": {"vs_threshold": f"max alpha x -> {v}"},
                                "16h": {"vs_threshold": f"max alpha y -> {v}"}}
        (d / f"slope_{day.replace('-', '')}_062000.json").write_text(
            json.dumps(rep), encoding="utf-8")

    def _run(self, d, today):
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        return subprocess.run(
            [sys.executable, "-X", "utf8",
             str(REPO / "analytics" / "calibration" / "tripwire_check.py"),
             "--reports-dir", str(d), "--today", today],
            capture_output=True, text=True, timeout=60, env=env)

    def test_no_reports_is_a_refusal_not_a_not_fired(self, tmp_path):
        r = self._run(tmp_path, "2026-09-01")
        assert r.returncode == 2

    def test_four_closed_reports_past_the_date_fires(self, tmp_path):
        for day in ("2026-08-11", "2026-08-18", "2026-08-25", "2026-09-01"):
            self._write_report(tmp_path, day, closed_assets={"SOL"})
        r = self._run(tmp_path, "2026-09-01")
        assert r.returncode == 3
        assert "TRIPWIRE FIRED for SOL" in r.stdout
        assert "BTC: armed 0/4" in r.stdout  # tradeable asset never fires

    def test_before_the_date_only_arms(self, tmp_path):
        for day in ("2026-08-04", "2026-08-11", "2026-08-18", "2026-08-25"):
            self._write_report(tmp_path, day, closed_assets={"SOL"})
        r = self._run(tmp_path, "2026-08-25")
        assert r.returncode == 0
        assert "SOL: armed 4/4" in r.stdout

    def test_fewer_than_four_reports_never_fires(self, tmp_path):
        for day in ("2026-08-25", "2026-09-01"):
            self._write_report(tmp_path, day, closed_assets={"SOL"})
        r = self._run(tmp_path, "2026-09-08")
        assert r.returncode == 0

    def test_checker_never_edits_config(self):
        src = (REPO / "analytics" / "calibration" / "tripwire_check.py"
               ).read_text(encoding="utf-8-sig")
        assert "live_high_risk" not in src.replace(
            "configs/live_high_risk.json", "")  # named only in the MESSAGE
        assert "write_text" not in src.split('def main')[1].replace(
            'read_text', '')  # no writes in main


class TestP251StaleSnapshotGuard:
    """[P251] The hold band's position feed is one reconcile stale on the
    tick after a flatten (reconcile runs in the heartbeat AFTER decide).
    Observed live 2026-08-10 00:02: [GATE-HYST] pos=+1 on a venue-flat
    book. A phantom position + alpha inside the band would clear the veto
    and admit an ENTRY FROM FLAT at below-enter alpha. Intent beats
    snapshot (P207): the feed reads 0 inside the post-flatten window."""

    def test_truth_table(self):
        from main import sleeve_snapshot_is_post_flatten_stale as stale
        # the live incident: flatten at round R, decide at R+1, snapshot +1
        assert stale(1, flatten_tick=10, round_count=11)
        # same-round (flatten later this round hasn't happened yet at decide,
        # but a record from a restart-replay same round is still the window)
        assert stale(-1, flatten_tick=10, round_count=10)
        # two rounds later the reconcile has run — trust the venue again,
        # even if it still shows a position (the flatten genuinely failed)
        assert not stale(1, flatten_tick=10, round_count=12)
        # a flat snapshot needs no guard
        assert not stale(0, flatten_tick=10, round_count=11)
        # no flatten recorded / not in the live loop -> inert
        assert not stale(1, flatten_tick=None, round_count=11)
        assert not stale(1, flatten_tick=10, round_count=None)

    def test_feed_function_is_the_load_bearing_path(self):
        """Behavioral: the value the band sees comes from
        sleeve_position_feed, which zeroes inside the window. (A pin on the
        surrounding if-statement was falsified by a `False and` probe — the
        P234 lesson applied to its own fix — so the ASSIGNMENT goes through
        the pure function instead.)"""
        from main import sleeve_position_feed as feed
        assert feed(1, flatten_tick=10, round_count=11) == 0   # the incident
        assert feed(-1, flatten_tick=10, round_count=10) == 0
        assert feed(1, flatten_tick=10, round_count=12) == 1   # trust again
        assert feed(-2, flatten_tick=None, round_count=11) == -2
        assert feed(0, flatten_tick=10, round_count=11) == 0

    def test_market_data_assignment_goes_through_the_feed_function(self):
        assert ('market_data["sleeve_position_contracts"] = '
                'sleeve_position_feed(') in MAIN, (
            "the assignment bypasses sleeve_position_feed — the P251 guard "
            "is decorative again"
        )

    def test_cooldown_records_on_flatten_sent_not_only_on_fill(self):
        """The P207 window also skipped the cooldown's flatten record on any
        slow-fill flatten — the cooldown then silently never armed. Record
        at SEND (conservative: only ever starts the cooldown earlier)."""
        i = MAIN.find("_cd_sent_flat = (")
        assert i > 0, "sent-based flatten record is gone"
        blk = MAIN[i:i + 400]
        assert "target_for_signal(_m_dir) == 0" in blk
        assert '== "OK"' in blk
        assert "_cd_now_flat" in MAIN[i:i + 600], (
            "keep the observed-fill record as the belt to the sent strap"
        )

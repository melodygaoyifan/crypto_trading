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
    def test_shadow_line_always_logs_the_counterfactual(self):
        assert "[GATE-HYST]" in V36
        assert "WOULD-HOLD" in V36 and "WOULD-EXIT" in V36

    def test_enforcement_requires_positive_ratio(self):
        blk = V36[V36.find("[P232] Gate hysteresis"):]
        blk = blk[:blk.find("pure new entry")]
        assert "_hy_ratio > 0" in blk, (
            "hold enforcement no longer gated on alpha_gate_hold_ratio > 0 — "
            "the default-OFF contract is broken"
        )

    def test_hold_requires_direction_agreement(self):
        """A flip is never hold-banded — reversals keep the full enter
        threshold + P198 persistence."""
        blk = V36[V36.find("[P232] Gate hysteresis"):]
        blk = blk[:blk.find("pure new entry")]
        assert "_hy_agrees" in blk
        assert "(_hy_dir * _hy_pos) > 0" in blk

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

    def test_config_trio_and_default_off(self):
        assert re.search(r"^\s+alpha_gate_hold_ratio: float = 0\.0", MAIN, re.M)
        assert 'data.get("alpha_gate_hold_ratio"' in MAIN
        live = json.loads((REPO / "configs" / "live_high_risk.json"
                           ).read_text(encoding="utf-8-sig"))
        assert "alpha_gate_hold_ratio" not in live, (
            "enforcement flipped on in the live profile — that is an "
            "operator decision needing WOULD-HOLD shadow evidence first"
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

    def test_config_trio_and_default_off(self):
        assert re.search(
            r"^\s+coinbase_reentry_cooldown_ticks: int = 0", MAIN, re.M)
        assert 'data.get("coinbase_reentry_cooldown_ticks"' in MAIN
        live = json.loads((REPO / "configs" / "live_high_risk.json"
                           ).read_text(encoding="utf-8-sig"))
        assert "coinbase_reentry_cooldown_ticks" not in live


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

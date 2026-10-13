"""[P425] The three profitability builds (operator: "any improvement on
profitability perspective" -> "can you do all"), each shipped the evidence-
gated way:

  1. ETH skew seat behind the ETF AGREE-GATE -- built, config-gated, NOT armed
     (skew_seat_agree_assets default [] / absent from the live profile), with
     a pre-committed 2026-12-12 read (scripts/skewetf_agree_check.py, cron,
     image allowlist both halves, registry entry).
  2. calbasis SLOW-EXPRESSION variant ledger (calbasis_slow12): a taken fast
     direction held 12 bars by wall-clock, persisted; pooled like its parent.
  3. ETF de-risk RELEASE rule: built into the live function (default None =
     today's behaviour), judged by the P404 lab through that function --
     RECORD_NOT_ARM on the pre-committed rule, so NO config knob exists and
     the live construction passes no release band (pinned).

Every pin here is a DECIDED-VALUE pin (P237): a silent arming and a silent
widening both fail.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
LIVE = json.loads((REPO / "configs" / "live_high_risk.json").read_text(encoding="utf-8-sig"))


# =============================================================== 2. calbasis_slow12
class TestSlowHoldStep:
    def test_taken_then_held_then_retaken_on_expiry(self):
        from defense.enhancement_shadows import SLOW_HOLD_SEC, slow_hold_step
        st, d, r = slow_hold_step(None, 1.0, 1000.0)
        assert (d, r) == (1.0, "taken") and st["until"] == 1000.0 + SLOW_HOLD_SEC
        # an OPPOSITE fast signal inside the hold does not move the claim
        st2, d2, r2 = slow_hold_step(st, -1.0, 1000.0 + SLOW_HOLD_SEC - 1)
        assert (d2, r2) == (1.0, "held") and st2 == st
        # a neutral fast signal inside the hold does not move it either
        _, d3, r3 = slow_hold_step(st, 0.0, 1000.0 + 10)
        assert (d3, r3) == (1.0, "held")
        # on expiry the fast direction is re-taken for another full hold
        st4, d4, r4 = slow_hold_step(st, -1.0, 1000.0 + SLOW_HOLD_SEC)
        assert (d4, r4) == (-1.0, "taken") and st4["until"] == 1000.0 + 2 * SLOW_HOLD_SEC

    def test_expired_hold_with_flat_fast_signal_is_flat_and_clears_state(self):
        from defense.enhancement_shadows import SLOW_HOLD_SEC, slow_hold_step
        st = {"dir": 1.0, "until": 500.0}
        assert slow_hold_step(st, 0.0, 500.0 + SLOW_HOLD_SEC) == (None, 0.0, "flat")

    def test_unusable_state_reads_as_no_hold_never_a_position(self):
        """Absence is never a position (P2): a corrupt or partial state dict
        falls through to the fast signal, it does not fabricate a held claim."""
        from defense.enhancement_shadows import slow_hold_step
        assert slow_hold_step({"dir": "x", "until": "y"}, 0.0, 10.0) == (None, 0.0, "flat")
        assert slow_hold_step({"until": 10 ** 9}, 0.0, 10.0) == (None, 0.0, "flat")
        assert slow_hold_step({"dir": 1.0}, 0.0, 10.0) == (None, 0.0, "flat")

    def test_hold_is_twelve_4h_bars(self):
        from defense.enhancement_shadows import SLOW_HOLD_BARS, SLOW_HOLD_SEC
        assert SLOW_HOLD_BARS == 12 and SLOW_HOLD_SEC == 12 * 4 * 3600


class TestSlowLedger:
    def _enh(self, tmp_path):
        from defense.enhancement_shadows import EnhancementShadows
        return EnhancementShadows(data_dir=str(tmp_path))

    def _rows(self, tmp_path, strategy, asset="BTC"):
        p = tmp_path / "strategy_shadow" / f"{strategy}_{asset}.jsonl"
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    def test_slow_row_is_written_beside_the_fast_row_and_holds(self, tmp_path):
        enh = self._enh(tmp_path)
        enh._basis_hist["BTC"] = [0.1, 0.2] * 12          # mean 0.15, sd 0.05
        # slope 0.0 -> z = -3 -> fast +1 "tightening"
        q = {"BTC": {"near_px": 100.0, "near_days": 10.0, "far_px": 100.0, "far_days": 375.0}}
        enh.tick(cde_quotes=q)
        fast = self._rows(tmp_path, "calbasis"); slow = self._rows(tmp_path, "calbasis_slow12")
        assert fast[-1]["direction"] == 1.0 and fast[-1]["reason"] == "tightening"
        assert slow[-1]["direction"] == 1.0 and slow[-1]["reason"] == "taken"
        assert slow[-1]["strategy"] == "calbasis_slow12" and slow[-1]["hold_bars"] == 12
        # next tick: fast goes NEUTRAL (slope at the mean) -> slow still HELD +1
        q2 = {"BTC": {"near_px": 100.0, "near_days": 10.0, "far_px": 115.0, "far_days": 375.0}}
        enh.tick(cde_quotes=q2)
        fast = self._rows(tmp_path, "calbasis"); slow = self._rows(tmp_path, "calbasis_slow12")
        assert fast[-1]["direction"] == 0.0
        assert slow[-1]["direction"] == 1.0 and slow[-1]["reason"] == "held"
        assert slow[-1]["fast_dir"] == 0.0
        # the scorer multiplies direction x confidence: a held claim carries 1.0
        assert slow[-1]["confidence"] == 1.0

    def test_hold_survives_a_restart(self, tmp_path):
        """RAM-only hold state is the P154 class: a deploy would shorten every
        hold to the deploy cadence."""
        enh = self._enh(tmp_path)
        enh._slow_state["ETH"] = {"dir": -1.0, "until": time.time() + 3600.0}
        enh._persist_slow()
        enh2 = self._enh(tmp_path)
        assert enh2._slow_state["ETH"]["dir"] == -1.0
        assert (tmp_path / "calbasis_slow_state.json").exists()

    def test_corrupt_state_file_restarts_flat_not_crash(self, tmp_path):
        (tmp_path / "calbasis_slow_state.json").write_text("{not json", encoding="utf-8")
        enh = self._enh(tmp_path)
        assert enh._slow_state == {}

    def test_slow_state_is_its_own_file_not_the_basis_history(self, tmp_path):
        """The basis file holds LISTS and its loader filters on that shape;
        overloading it would silently drop the hold on restore."""
        enh = self._enh(tmp_path)
        enh._basis_hist["BTC"] = [0.1] * 5
        enh._persist_basis()
        enh._slow_state["BTC"] = {"dir": 1.0, "until": 1.0}
        enh._persist_slow()
        basis = json.loads((tmp_path / "calbasis_state.json").read_text(encoding="utf-8"))
        assert "_slow12" not in basis and isinstance(basis["BTC"], list)


class TestSlowScorerWiring:
    def test_declared_by_the_producer_and_pooled_by_the_scorer(self):
        from analytics.shadow_ic.compute_shadow_ic import POOLABLE_FAMILIES
        from defense.enhancement_shadows import SHADOW_STRATEGY_NAMES
        assert "calbasis_slow12" in SHADOW_STRATEGY_NAMES
        assert "calbasis_slow12" in POOLABLE_FAMILIES, "pooled like its parent (P293g power)"

    def test_the_calbasis_prefix_glob_picks_up_the_variant_file(self, tmp_path):
        """The cron row scores --prefixes calbasis; the variant's file name must
        fall under that glob or the Monday read never sees it (P264 class)."""
        import glob
        (tmp_path / "calbasis_slow12_BTC.jsonl").write_text("", encoding="utf-8")
        assert glob.glob(str(tmp_path / "calbasis_*.jsonl"))

    def test_calbasis_check_still_reads_only_the_parent_record(self):
        """The slow variant must NOT satisfy the 11-10 calbasis criterion."""
        from scripts.calbasis_check import find_record
        rep = {"per_strategy": [{"strategy": "calbasis_slow12", "asset": "POOLED",
                                 "verdict": "PROMOTE"}]}
        assert find_record(rep) is None


# =============================================================== 1. agree gate
class TestAgreeGate:
    @pytest.fixture(scope="class")
    def ok(self):
        import main
        return main.skew_seat_agree_ok

    def test_ungated_asset_is_inert(self, ok):
        assert ok("BTC", -1.0, None, ["ETH"]) == (True, "ungated")
        assert ok("BTC", -1.0, (1.0, True), []) == (True, "ungated")
        assert ok("BTC", -1.0, (1.0, True), None) == (True, "ungated")

    @pytest.mark.parametrize("etf,expect", [
        (None, "etf_absent"),
        ((-1.0, False), "etf_not_fresh"),
        ((1.0, True), "disagree"),
        ((0.0, True), "disagree"),
        (("x", True), "etf_unreadable"),
    ])
    def test_gated_asset_refuses_without_fresh_agreement(self, ok, etf, expect):
        good, why = ok("ETH", -1.0, etf, ["ETH"])
        assert good is False and why == expect

    def test_gated_asset_passes_on_fresh_agreement(self, ok):
        assert ok("ETH", -1.0, (-1.0, True), ["ETH"]) == (True, "agree")
        assert ok("ETH", 1.0, (1.0, True), ["ETH"]) == (True, "agree")

    def test_the_gate_is_the_ledgers_own_combiner(self, ok):
        """One implementation (P172): the live gate must agree with the
        skewetf_agree record for every (skew, etf) pair."""
        from defense.skew_etf_combo_shadow import combo_directions
        for s in (-1.0, 1.0):
            for e in (-1.0, 0.0, 1.0):
                rec = combo_directions(s, True, e, True)["skewetf_agree"]
                assert ok("ETH", s, (e, True), ["ETH"])[0] is (rec != 0.0)


class TestAgreeGateWiring:
    SRC = (REPO / "main.py").read_text(encoding="utf-8-sig")

    def test_config_trio_declared_parsed_consumed(self):
        assert "skew_seat_agree_assets: Optional[List[str]] = None" in self.SRC
        assert 'list(data["skew_seat_agree_assets"])' in self.SRC
        assert 'getattr(self.config, "skew_seat_agree_assets", None) or []' in self.SRC

    def test_gate_runs_between_the_seat_decision_and_the_seat_write(self):
        i = self.SRC.index("_took = skew_seat_decision(")
        j = self.SRC.index("skew_seat_agree_ok(asset, float(_took[1]), _efr, _agree_assets)")
        k = self.SRC.index('market_data["primary_strategy"] = _lbl', i)
        assert i < j < k, "the gate must be able to clear _took before the seat is written"
        blk = self.SRC[j:k]
        assert "_took = None" in blk, "a refused gate must leave NO seat (incumbent stands)"

    def test_live_profile_does_not_arm_the_gate_and_eth_stays_parked(self):
        """Arming ETH is the skewetf_eth_agree_read decision (2026-12-12),
        never a side effect of building the mechanism."""
        assert "skew_seat_agree_assets" not in LIVE
        assert LIVE["skew_seat_assets"] == ["BTC"]

    def test_default_config_is_no_gating(self):
        from main import ProductionConfig
        cfg = ProductionConfig.from_file(REPO / "configs" / "live_high_risk.json")
        assert cfg.skew_seat_agree_assets is None


class TestAgreeChecker:
    AFTER = "2026-12-13"
    BEFORE = "2026-11-01"

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-X", "utf8", str(REPO / "scripts" / "skewetf_agree_check.py"),
             *args], capture_output=True, text=True, cwd=str(REPO),
            timeout=120, encoding="utf-8")

    def _h(self, ic=0.2, t=2.5, edge=100.0, need=58.0):
        return {"ic": ic, "t_stat": t, "edge_bps": edge, "required_bps": need,
                "n": 300, "n_eff": 40}

    def _report(self, tmp_path, recs, window_days=75):
        p = tmp_path / "rep.json"
        p.write_text(json.dumps({"window_days": window_days, "per_strategy": recs}),
                     encoding="utf-8")
        return p

    def _rec(self, strategy="skewetf_agree", asset="ETH", verdict="HOLD"):
        ph = {"4": self._h(), "12": self._h(), "24": self._h()}
        return {"strategy": strategy, "asset": asset, "verdict": verdict,
                "per_horizon": ph, "n_per_horizon": {k: 300 for k in ph}}

    def test_not_due_before_the_read_date(self):
        r = self._run("--today", self.BEFORE)
        assert r.returncode == 0 and "not due" in r.stdout

    def test_promote_fires_exit_3_naming_the_human_config_change(self, tmp_path):
        p = self._report(tmp_path, [self._rec(verdict="PROMOTE")])
        r = self._run("--today", self.AFTER, "--report", str(p))
        assert r.returncode == 3, r.stdout + r.stderr
        assert "skew_seat_agree_assets" in r.stdout and "P141" in r.stdout

    def test_hold_keeps_eth_parked_exit_0(self, tmp_path):
        p = self._report(tmp_path, [self._rec(verdict="HOLD")])
        r = self._run("--today", self.AFTER, "--report", str(p))
        assert r.returncode == 0 and "PARKED" in r.stdout

    def test_a_skew_solo_promote_is_context_not_a_criterion(self, tmp_path):
        p = self._report(tmp_path, [self._rec(verdict="HOLD"),
                                    self._rec(strategy="skewetf_skew", verdict="PROMOTE")])
        r = self._run("--today", self.AFTER, "--report", str(p))
        assert r.returncode == 0 and "context" in r.stdout

    def test_a_btc_record_cannot_satisfy_the_eth_read(self, tmp_path):
        p = self._report(tmp_path, [self._rec(asset="BTC", verdict="PROMOTE")])
        r = self._run("--today", self.AFTER, "--report", str(p))
        assert r.returncode == 2 and "no scoreable" in r.stderr

    def test_a_short_window_is_refused_not_scored(self, tmp_path):
        p = self._report(tmp_path, [self._rec(verdict="PROMOTE")], window_days=30)
        r = self._run("--today", self.AFTER, "--report", str(p))
        assert r.returncode == 2 and "does not span" in r.stderr

    def test_missing_report_refuses(self, tmp_path):
        r = self._run("--today", self.AFTER, "--report", str(tmp_path / "nope.json"))
        assert r.returncode == 2

    def test_ledger_start_is_the_replica_feed_deploy(self):
        import datetime as dt
        from scripts.skewetf_agree_check import LEDGER_START, READ_DATE
        assert LEDGER_START == dt.date(2026, 10, 13)
        assert READ_DATE == dt.date(2026, 12, 12)


class TestAgreeOps:
    def test_cronned_and_in_the_image_both_halves(self):
        cron = (REPO / "scripts/ops/hmats.crontab").read_text(encoding="utf-8")
        row = [l for l in cron.splitlines() if "skewetf_agree_check.py" in l and not l.startswith("#")]
        assert len(row) == 1
        assert "--prefixes skewetf" in row[0] and "--allow-kraken-ohlcv" in row[0]
        assert "--pool-assets" not in row[0], "the question is ETH per-asset, not the pooled pair"
        assert "--window-days 75" in row[0]
        assert "+%Y" not in row[0] and "\\%Y\\%m\\%d" in row[0]
        docker = (REPO / "Dockerfile.engine").read_text(encoding="utf-8")
        assert "scripts/skewetf_agree_check.py" in docker
        ign = (REPO / ".dockerignore").read_text(encoding="utf-8")
        assert "!scripts/skewetf_agree_check.py" in ign

    def test_registry_carries_the_pre_committed_read(self):
        from analytics.decisions import registry as r
        d = {x["id"]: x for x in r.load_registry()}["skewetf_eth_agree_read"]
        assert d["status"] == "pending" and d["deadline"] == "2026-12-12"
        assert "skewetf_agree_check.py" in d["decision_rule"]
        assert "keep the current live state" in d["default"]


# =============================================================== 3. ETF release rule
class TestEtfRelease:
    TRAIL = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
             11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]  # mean 10.5, sd ~5.77

    def test_default_none_is_todays_rule_exactly(self):
        from defense.etf_flow_shadow import etf_flow_zscore_direction as f
        # z ~ -0.09 inside the band from a held -1: today's rule HOLDS
        assert f(10.0, 0.0, self.TRAIL, -1.0) == (-1.0, pytest.approx(-0.0866, abs=1e-3), "deadband_hold")
        assert f(10.0, 0.0, self.TRAIL, -1.0, release_band=None)[2] == "deadband_hold"

    def test_release_frees_a_held_minus_one_once_z_crosses_the_level(self):
        from defense.etf_flow_shadow import etf_flow_zscore_direction as f
        d, z, r = f(11.0, 0.0, self.TRAIL, -1.0, release_band=0.0)   # z ~ +0.09 > 0
        assert (d, r) == (0.0, "release_to_flat")
        d, z, r = f(10.0, 0.0, self.TRAIL, -1.0, release_band=0.0)   # z ~ -0.09 < 0
        assert (d, r) == (-1.0, "deadband_hold")

    def test_release_is_symmetric_and_never_fires_from_flat(self):
        from defense.etf_flow_shadow import etf_flow_zscore_direction as f
        assert f(10.0, 0.0, self.TRAIL, 1.0, release_band=0.0)[2] == "release_to_flat"
        assert f(11.0, 0.0, self.TRAIL, 1.0, release_band=0.0)[2] == "deadband_hold"
        assert f(11.0, 0.0, self.TRAIL, 0.0, release_band=0.0) == (0.0, pytest.approx(0.0866, abs=1e-3), "deadband_hold")

    def test_a_band_breach_still_wins_over_the_release(self):
        from defense.etf_flow_shadow import etf_flow_zscore_direction as f
        assert f(30.0, 0.0, self.TRAIL, -1.0, release_band=0.0)[2] == "inflow_z"
        assert f(-10.0, 0.0, self.TRAIL, 1.0, release_band=0.0)[2] == "outflow_z"

    def test_replay_hold_threads_the_release(self):
        from defense.etf_flow_shadow import replay_hold
        flows = [1.0, 2.0] * 8 + [-50.0] + [1.6] * 3   # an outflow spike then quiet, z back above 0
        assert replay_hold(flows) == -1.0                       # today's rule holds the -1
        assert replay_hold(flows, release_band=0.0) == 0.0     # the release lets it go

    def test_release_is_a_fresh_flat_claim_for_the_seat(self):
        from defense.etf_flow_shadow import EtfFlowShadow
        assert "release_to_flat" in EtfFlowShadow._SEAT_FRESH_REASONS

    def test_live_construction_passes_no_release_band_and_no_knob_exists(self):
        """The lab verdict was RECORD_NOT_ARM: the mechanism exists for the
        lab, the live book keeps the P400/P402 hold, and there is deliberately
        NO config knob to turn (a knob nobody should turn invites turning)."""
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        assert 'EtfFlowShadow(data_dir="data")' in src
        assert "release_band" not in src and "etf_derisk_release_z" not in src
        assert "etf_derisk_release_z" not in LIVE
        from defense.etf_flow_shadow import EtfFlowShadow
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            assert EtfFlowShadow(data_dir=d)._release_band is None

    def test_lab_report_records_the_verdict_against_its_rule(self):
        rep = json.loads((REPO / "training/reports/etf_release_lab_p425.json").read_text(encoding="utf-8"))
        assert rep["verdict"] == "RECORD_NOT_ARM"
        assert "release_z=0.00" in rep["rule"]
        for a in ("BTC", "ETH"):
            assert rep["assets"][a]["candidate_earns"] is False

    def test_lab_steps_the_live_function(self):
        src = (REPO / "training/scripts/etf_sma_overlay_check.py").read_text(encoding="utf-8")
        assert "from defense.etf_flow_shadow import" in src
        assert "etf_flow_zscore_direction(" in src
        assert "PARITY FAIL" in src, "the no-release step must be pinned to the lab's own deadband"

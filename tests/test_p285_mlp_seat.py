"""[P285] The mlp early-seat criterion + actuator, pinned.

The operator adopted the early-swap criterion 2026-08-16 ("adopt the
criterion and wire up"): the P283b-certified BTC mlp_small may take the BTC
direction seat before the full 30d P166 read, at UNCHANGED size/caps/stops,
iff scripts/mlp_seat_check.py fires. These tests pin: the harness accessor's
freshness + coverage contract (a coverage-gap emit must NEVER feed the
seat), the config trio (P201), the live profile NOT setting the mode
(adding it IS the firing action), the seat block's load-bearing writes, and
the checker's pure criterion incl. missing-data-never-passes (P199).
"""

import inspect
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _harness(tmp_path):
    from defense.mlp_shadow import MlpShadow
    s = object.__new__(MlpShadow)
    s._dir = tmp_path
    s._models = {"BTC": {
        "asset": "BTC", "feature_names": ["f0", "f1"],
        "scaler_mean": [0.0, 0.0], "scaler_scale": [1.0, 1.0],
        "w1": [[1.0], [1.0]], "b1": [0.0], "w2": [1.0], "b2": 0.0,
        "sig": 1.0, "deadband": 0.25, "decision_interval": 1,
    }}
    s._state = {}
    s._warned = {}
    s._last_records = {}
    return s


# ---------------------------------------------------------------------------
# last_direction accessor — the seat feed contract
# ---------------------------------------------------------------------------

class TestLastDirection:
    def test_no_record_is_none_not_flat(self, tmp_path):
        s = _harness(tmp_path)
        assert s.last_direction("BTC") is None, (
            "absent must be None — collapsing it to 0.0 turns a harness "
            "outage into a flatten (P2 missing-vs-neutral)")

    def test_full_coverage_tick_feeds_the_seat(self, tmp_path):
        s = _harness(tmp_path)
        s.tick({"BTC": {"f0": 3.0, "f1": 3.0}},
               {"BTC": {"f0": True, "f1": True}})
        got = s.last_direction("BTC")
        assert got is not None
        d, z, age = got
        assert d in (-1.0, 0.0, 1.0) and age < 5

    def test_coverage_gap_does_not_feed_the_seat(self, tmp_path):
        # THE load-bearing pin: a coverage-gap tick records flat in the
        # LEDGER (honest) but must NOT refresh the seat feed — the seat
        # treating "cannot compute" as a fresh flat would flatten the book
        # on a feed outage (the P2 trap the whole design exists to avoid).
        s = _harness(tmp_path)
        s.tick({"BTC": {"f0": 3.0, "f1": 3.0}},
               {"BTC": {"f0": True, "f1": False}})
        assert s.last_direction("BTC") is None, (
            "a coverage-gap emit fed the seat — 'cannot compute' became a "
            "tradeable flat")

    def test_stale_record_is_refused(self, tmp_path):
        s = _harness(tmp_path)
        s._last_records["BTC"] = {"ts": time.time() - 7 * 3600,
                                  "direction": 1.0, "z": 2.0}
        assert s.last_direction("BTC") is None, (
            "a stale emit must not hold the seat (P156 staleness bound)")

    def test_deadband_flat_is_served_not_hidden(self, tmp_path):
        # the model saying FLAT is a position (gate -> flatten); it must be
        # distinguishable from absence
        s = _harness(tmp_path)
        s._last_records["BTC"] = {"ts": time.time(), "direction": 0.0,
                                  "z": 0.05}
        got = s.last_direction("BTC")
        assert got is not None and got[0] == 0.0


# ---------------------------------------------------------------------------
# config trio (P201) + live profile
# ---------------------------------------------------------------------------

class TestConfigContract:
    def test_field_defaults(self):
        import dataclasses

        import main as hm
        f = {f.name: f for f in dataclasses.fields(hm.ProductionConfig)}
        assert f["mlpshadow_mode"].default == "off"
        assert f["mlpshadow_seat_assets"].default_factory() == ["BTC"], (
            "the default seat roster is BTC only — the one certified asset")

    def test_from_file_parses_both_keys(self, tmp_path):
        import main as hm
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"mlpshadow_mode": "enforce",
                                 "mlpshadow_seat_assets": ["BTC"]}),
                     encoding="utf-8")
        cfg = hm.ProductionConfig.from_file(p)
        assert cfg.mlpshadow_mode == "enforce", (
            "declared but not parsed — the P201 inert-flag shape")
        assert cfg.mlpshadow_seat_assets == ["BTC"]

    def test_live_profile_does_not_set_the_mode(self):
        live = json.loads((REPO / "configs" / "live_high_risk.json"
                           ).read_text(encoding="utf-8-sig"))
        assert "mlpshadow_mode" not in live, (
            "the live profile set mlpshadow_mode — that is the P285 firing "
            "action and requires mlp_seat_check.py exit 3 + its own "
            "recorded decision (P141/P237 actuator pattern)")


# ---------------------------------------------------------------------------
# seat block wiring pins
# ---------------------------------------------------------------------------

class TestSeatBlockWiring:
    @pytest.fixture(scope="class")
    def src(self):
        import main as hm
        return inspect.getsource(
            hm.HMATSProductionRunner._process_4h_tick_inner)

    def test_seat_runs_after_the_regimebook_seat(self, src):
        assert src.index("[P285] MLP SEAT") > src.index(
            "[P256] REGIMEBOOK SEAT"), (
            "the mlp seat must run AFTER the regimebook seat so the "
            "certified model deterministically wins if both are enforced")

    def test_enforce_only_and_roster_gated(self, src):
        i = src.index("[P285] MLP SEAT")
        blk = src[i:i + 5000]
        assert '_ms_mode == "enforce"' in blk
        assert "asset in _ms_assets" in blk, (
            "the seat must be gated per-asset — enforce for BTC must not "
            "claim ETH/SOL")

    def test_both_dicts_dq_and_the_sleeve_bridge_are_written(self, src):
        i = src.index("[P285] MLP SEAT")
        blk = src[i:i + 6000]
        for needle in ('market_data["quant_direction"]',
                       'agent_signals["quant_direction"]',
                       'market_data["signal_edge_bps"]',
                       'agent_signals["signal_edge_bps"]',
                       'agent_signals["quant_data_quality"]',
                       "_last_quant_directions[asset]"):
            assert needle in blk, (
                f"seat missing write: {needle} — one-dict writes are "
                f"shadowed (P170), a missing dq stamp gets the seated "
                f"signal excluded on degraded ticks (P265), and skipping "
                f"the P149 bridge leaves the sleeve on the stale snapshot")

    def test_alpha_constant_is_not_looser_than_the_other_seats(self, src):
        i = src.index("[P285] MLP SEAT")
        blk = src[i:i + 6000]
        assert "30.0 * abs(" in blk, (
            "the seat swap changes the DIRECTION source, never the alpha "
            "bar — a looser constant would loosen the gate as a side effect")

    def test_absence_takes_no_seat(self, src):
        i = src.index("[P285] MLP SEAT")
        blk = src[i:i + 6000]
        assert "if _ms is None" in blk and "seat NOT taken" in blk

    def test_uses_the_coverage_gated_accessor(self, src):
        i = src.index("[P285] MLP SEAT")
        blk = src[i:i + 6000]
        assert "._mlp_shadow.last_direction(" in blk, (
            "the seat must read the coverage-gated accessor, never the "
            "ledger or raw state")


# ---------------------------------------------------------------------------
# the checker's pure criterion
# ---------------------------------------------------------------------------

class TestCriterion:
    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mlp_seat_check", REPO / "scripts" / "mlp_seat_check.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_all_conditions_met_fires(self):
        m = self._mod()
        fire, _ = m.decide(date(2026, 8, 28), 15.0, 60, 0.02, True)
        assert fire

    def test_each_single_failure_blocks(self):
        m = self._mod()
        base = dict(today=date(2026, 8, 28), span_days=15.0,
                    n_directional=60, ic16=0.02, trend_closed_latest=True)
        for k, bad in [("today", date(2026, 8, 27)), ("span_days", 13.0),
                       ("n_directional", 39), ("ic16", -0.001),
                       ("trend_closed_latest", False)]:
            args = dict(base)
            args[k] = bad
            fire, _ = m.decide(**args)
            assert not fire, f"criterion fired despite failing {k}={bad}"

    def test_zero_ic_passes_the_kill_screen(self):
        # the screen is "not negative", not "positive" — pre-committed
        m = self._mod()
        fire, _ = m.decide(date(2026, 9, 1), 20.0, 80, 0.0, True)
        assert fire

    def test_missing_data_never_passes(self):
        # P199: an unevaluated input must never satisfy a condition
        m = self._mod()
        for ic, tc in [(None, True), (0.02, None), (None, None)]:
            fire, conds = m.decide(date(2026, 9, 1), 20.0, 80, ic, tc)
            assert not fire, (
                f"criterion fired with unevaluated inputs ic={ic} "
                f"trend_closed={tc} — missing data read as passing")

    def test_missing_ledger_is_a_refusal_not_a_verdict(self, tmp_path):
        r = subprocess.run(
            [sys.executable, "-X", "utf8",
             str(REPO / "scripts" / "mlp_seat_check.py"),
             "--ledger-dir", str(tmp_path / "nowhere"),
             "--today", "2026-09-01"],
            capture_output=True, text=True, cwd=str(REPO))
        assert r.returncode == 2, (
            f"missing ledger must exit 2 (refusal), got {r.returncode}: "
            f"{r.stdout} {r.stderr}")

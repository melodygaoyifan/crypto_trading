"""[P1-2] Realized all-in friction model + the main.py shadow/enforce hook."""
import json
import math
import types
from pathlib import Path

import pytest

from execution.cost.realized_friction import (EMA_DOWN, EMA_UP,
                                              RealizedFrictionModel,
                                              shadow_threshold)

REPO = Path(__file__).resolve().parents[1]


def _model(tmp_path, per=None, min_hold=None):
    st = tmp_path / "static.json"
    st.write_text(json.dumps({"per_asset": per if per is not None else {
        "BTC": {"rt_fee_bps": 20.0, "rt_spread_bps": 1.0, "median_hold_bars": 2.5}},
        "MIN_HOLD_BARS": min_hold}), encoding="utf-8")
    return RealizedFrictionModel(static_path=str(st), state_path=str(tmp_path / "state.json"))


class TestModel:
    def test_sum_of_terms(self, tmp_path):
        m = _model(tmp_path)
        f, br = m.friction_rt_bps("BTC", 1.0, latency_rt_bps=4.0)
        assert f == pytest.approx(20.0 + 1.0 + 4.0)
        assert br["funding"] == 0.0

    def test_missing_static_is_inf_fail_closed(self, tmp_path):
        m = _model(tmp_path)
        f, br = m.friction_rt_bps("ETH", 1.0, latency_rt_bps=4.0)
        assert math.isinf(f) and br["reason"] == "no_static"

    def test_unreadable_static_file_is_inf(self, tmp_path):
        m = RealizedFrictionModel(static_path=str(tmp_path / "nope.json"),
                                  state_path=str(tmp_path / "s.json"))
        assert math.isinf(m.friction_rt_bps("BTC", 1.0, latency_rt_bps=4.0)[0])

    def test_funding_cost_for_a_long_paying(self, tmp_path):
        m = _model(tmp_path)
        f, br = m.friction_rt_bps("BTC", 1.0, latency_rt_bps=0.0,
                                  funding_rate_hourly=1e-5, hold_horizon_bars=3)
        assert br["funding"] == pytest.approx(1e-5 * 4 * 3 * 1e4)

    def test_funding_income_is_never_a_credit(self, tmp_path):
        """A short collecting positive funding must not lower friction."""
        m = _model(tmp_path)
        base, _ = m.friction_rt_bps("BTC", -1.0, latency_rt_bps=4.0)
        f, br = m.friction_rt_bps("BTC", -1.0, latency_rt_bps=4.0,
                                  funding_rate_hourly=5e-5, hold_horizon_bars=3)
        assert br["funding_raw"] < 0 and br["funding"] == 0.0 and f == base

    def test_fee_term_floored_at_venue_schedule(self, tmp_path):
        m = _model(tmp_path)  # static rt_fee 20.0 < 2 x current schedule taker (~22.7)
        f, br = m.friction_rt_bps("BTC", 1.0, latency_rt_bps=0.0, price=81315.0)
        assert br["fee_floor"] == pytest.approx(2 * (10.0 + 0.11 / 813.15 * 1e4), abs=1e-6)
        assert br["fee"] == br["fee_floor"] > 20.0

    def test_negative_spread_never_reduces_friction(self, tmp_path):
        m = _model(tmp_path, per={"BTC": {"rt_fee_bps": 20.0, "rt_spread_bps": -3.0,
                                          "median_hold_bars": 2}})
        assert m.friction_rt_bps("BTC", 1.0, latency_rt_bps=0.0)[1]["spread"] == 0.0

    def test_ema_is_asymmetric(self, tmp_path):
        m = _model(tmp_path)
        m.update_from_fill("BTC", 15.0, None)       # obs 30 > 20 -> up alpha
        assert m.ema["BTC"]["rt_fee_bps"] == pytest.approx(20 + EMA_UP * 10)
        before = m.ema["BTC"]["rt_fee_bps"]
        m.update_from_fill("BTC", 5.0, None)        # obs 10 < -> down alpha
        assert m.ema["BTC"]["rt_fee_bps"] == pytest.approx(before + EMA_DOWN * (10 - before))

    def test_hold_horizon_uses_larger_of_min_hold_and_median(self, tmp_path):
        assert _model(tmp_path).hold_horizon_bars("BTC") == 2.5
        assert _model(tmp_path, min_hold=4).hold_horizon_bars("BTC") == 4

    def test_fill_ledger_ingest_updates_and_persists(self, tmp_path):
        m = _model(tmp_path)
        led = tmp_path / "fq.jsonl"
        led.write_text(json.dumps({"ts": 10.0, "asset": "BTC", "status": "filled",
                                   "contracts": 1, "fill_avg_price": 80000.0,
                                   "fees_usd": 0.91, "realized_slippage_bps": 1.0}) + "\n"
                       + "not json\n", encoding="utf-8")
        assert m.ingest_fill_ledger(str(led)) == 1
        assert m.last_ingest_ts == 10.0
        assert m.ingest_fill_ledger(str(led)) == 0           # already ingested
        m2 = RealizedFrictionModel(static_path=m.static_path, state_path=m.state_path)
        assert m2.ema["BTC"] == m.ema["BTC"]


class TestShadowThreshold:
    def test_only_friction_changes(self):
        assert shadow_threshold(30.8, 28.0, 3.0, 35.0) == pytest.approx(30.8 * 35 / 28)

    def test_min_alpha_floor_kept(self):
        assert shadow_threshold(10.0, 28.0, 25.0, 14.0) == 25.0

    def test_no_priced_friction_returns_none(self):
        assert shadow_threshold(0.0, 0.0, 0.0, 30.0) is None

    def test_inf_friction_is_inf_threshold(self):
        assert math.isinf(shadow_threshold(30.0, 28.0, 0.0, math.inf))


# ---- the main.py hook, on a minimal fake runner --------------------------

def _runner(mode, ar, model):
    import main
    r = object.__new__(main.HMATSProductionRunner)
    r.config = types.SimpleNamespace(realized_friction_gate_mode=mode)
    fr = types.SimpleNamespace(latency_cost_bps=2.0)
    r.engine = types.SimpleNamespace(_last_alpha_result=ar,
                                     guarantees=types.SimpleNamespace(
                                         alpha_calculator=types.SimpleNamespace(FRICTION=fr)))
    r._rf_last_ingest = 1e18  # skip ingest
    import execution.cost.realized_friction as rfm
    rfm._MODEL = model
    return r


def _ar(passes, est, ev, fric):
    return types.SimpleNamespace(passes_threshold=passes, estimated_alpha_bps=est,
                                 ev_threshold_bps=ev, friction_total_bps=fric,
                                 min_alpha_bps_used=0.0, threshold_bps=ev)


def _intent():
    return types.SimpleNamespace(direction=1.0, veto_active=False, veto_reason="")


class TestHook:
    @pytest.fixture(autouse=True)
    def _data(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        self.tmp = tmp_path
        yield
        import execution.cost.realized_friction as rfm
        rfm._MODEL = None

    def test_shadow_never_vetoes(self):
        r = _runner("shadow", _ar(True, 30.0, 29.0, 26.0), _model(self.tmp))
        it = _intent()
        rec = r._apply_realized_friction_gate("BTC", it, {}, {"current_price": 81000.0})
        assert rec["verdict"] == "WOULD_SUPPRESS" and it.veto_active is False
        assert (self.tmp / "realized_friction_shadow.jsonl").exists()

    def test_enforce_is_tighten_only(self):
        r = _runner("enforce", _ar(True, 30.0, 29.0, 26.0), _model(self.tmp))
        it = _intent()
        r._apply_realized_friction_gate("BTC", it, {}, {"current_price": 81000.0})
        assert it.veto_active and it.veto_reason.startswith("[P1-2-FRICTION]")

    def test_enforce_never_unvetoes(self):
        r = _runner("enforce", _ar(False, 30.0, 60.0, 60.0), _model(self.tmp))
        it = _intent(); it.veto_active = True; it.veto_reason = "[v3.6.1] Alpha gate: x"
        rec = r._apply_realized_friction_gate("BTC", it, {}, {"current_price": 81000.0})
        assert rec["verdict"] == "WOULD_ALLOW"
        assert it.veto_active and it.veto_reason == "[v3.6.1] Alpha gate: x"

    def test_enforce_error_fails_closed(self):
        r = _runner("enforce", _ar(True, 30.0, 29.0, 26.0), _model(self.tmp))
        r.engine.guarantees = None                     # raises inside
        it = _intent()
        r._apply_realized_friction_gate("BTC", it, {}, {"current_price": 81000.0})
        assert it.veto_active and "fail-closed" in it.veto_reason

    def test_shadow_error_does_not_veto(self):
        r = _runner("shadow", _ar(True, 30.0, 29.0, 26.0), _model(self.tmp))
        r.engine.guarantees = None
        it = _intent()
        r._apply_realized_friction_gate("BTC", it, {}, {"current_price": 81000.0})
        assert not it.veto_active

    def test_off_does_nothing(self):
        r = _runner("off", _ar(True, 30.0, 29.0, 26.0), _model(self.tmp))
        assert r._apply_realized_friction_gate("BTC", _intent(), {}, {}) is None


class TestWiring:
    def test_default_is_shadow_and_parsed(self):
        import main
        assert main.ProductionConfig().realized_friction_gate_mode == "shadow"
        src = (REPO / "main.py").read_text(encoding="utf-8")
        assert 'data.get("realized_friction_gate_mode", "shadow")' in src
        assert "self._apply_realized_friction_gate(asset, intent, agent_signals, market_data)" in src

    def test_live_profile_does_not_arm_enforce(self):
        live = json.loads((REPO / "configs" / "live_high_risk.json").read_text(encoding="utf-8"))
        assert live.get("realized_friction_gate_mode", "shadow") != "enforce"

    def test_veto_reason_is_entry_quality_for_the_sleeve(self):
        import main
        assert "[P1-2-FRICTION]" in main._SLEEVE_ENTRY_QUALITY_VETOES

    def test_constitution_untouched_by_phase1(self):
        import subprocess
        out = subprocess.run(["git", "diff", "--stat", "HEAD", "--", "defense/constitution.py"],
                             cwd=REPO, capture_output=True, text=True, encoding="utf-8")
        assert out.stdout.strip() == ""

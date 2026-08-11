"""[P256] The regimebook SEAT: default-off wiring that lets a P166-passing
roster take the direction seat through the existing gated chain.

Built AHEAD of the ~2026-09-09 forward-gate read (P230: the instrument in the
same breath as the rule). These tests pin: the accessor's freshness contract,
the config trio (P201), the live profile's explicit "off", and the seat
block's load-bearing properties (both-dict writes, enforce-only, ordering
after the trend inject, the no-looser alpha constant).
"""

import inspect
import json
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _harness(tmp_path):
    from defense.regime_book_shadow import RegimeBookShadow
    return RegimeBookShadow(data_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# last_direction accessor
# ---------------------------------------------------------------------------

class TestLastDirection:
    def test_no_record_is_none_not_flat(self, tmp_path):
        h = _harness(tmp_path)
        assert h.last_direction("BTC") is None, (
            "absent must be None — collapsing it to 0.0 turns a data outage "
            "into a flatten (P2 missing-vs-neutral)")

    def test_fresh_record_serves_direction_and_leg(self, tmp_path):
        h = _harness(tmp_path)
        h._last_records["BTC"] = {"ts": time.time(), "direction": 1.0,
                                  "leg": "hold"}
        d, leg, age = h.last_direction("BTC")
        assert d == 1.0 and leg == "hold" and age < 5

    def test_stale_record_is_refused(self, tmp_path):
        h = _harness(tmp_path)
        h._last_records["BTC"] = {"ts": time.time() - 7 * 3600,
                                  "direction": 1.0, "leg": "hold"}
        assert h.last_direction("BTC") is None, (
            "a stale target must not hold the seat (P156 staleness bound)")

    def test_flat_zero_is_served_not_hidden(self, tmp_path):
        # the book saying FLAT is a position (gate -> flatten); it must be
        # distinguishable from absence
        h = _harness(tmp_path)
        h._last_records["ETH"] = {"ts": time.time(), "direction": 0.0,
                                  "leg": "trend_flat"}
        d, leg, _ = h.last_direction("ETH")
        assert d == 0.0 and leg == "trend_flat"

    def test_record_tick_stashes_on_success(self, tmp_path):
        h = _harness(tmp_path)
        closes = list(range(1, 400)) * 2  # enough bars, rising
        closes = [float(c) for c in closes[:720]]
        rec = h.record_tick("ETH", closes, price=closes[-1])
        if rec is not None:  # warmup/regime label independent of this pin
            assert h._last_records["ETH"] is rec


# ---------------------------------------------------------------------------
# config trio (P201) + live profile
# ---------------------------------------------------------------------------

class TestConfigContract:
    def test_field_default_is_off(self):
        import main as hm
        import dataclasses
        f = {f.name: f for f in dataclasses.fields(hm.ProductionConfig)}
        assert f["regimebook_mode"].default == "off"

    def test_from_file_parses_the_key(self, tmp_path):
        import main as hm
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"regimebook_mode": "enforce"}),
                     encoding="utf-8")
        cfg = hm.ProductionConfig.from_file(p)
        assert cfg.regimebook_mode == "enforce", (
            "the key is declared but not parsed — the P201 inert-flag shape")

    def test_live_profile_pins_off(self):
        live = json.loads((REPO / "configs" / "live_high_risk.json"
                           ).read_text(encoding="utf-8-sig"))
        assert live.get("regimebook_mode") == "off", (
            "the live profile flipped the seat on — that requires the P166 "
            "forward-gate read + its own P-entry (P141)")


# ---------------------------------------------------------------------------
# seat block wiring pins
# ---------------------------------------------------------------------------

class TestSeatBlockWiring:
    @pytest.fixture(scope="class")
    def src(self):
        import main as hm
        return inspect.getsource(
            hm.HMATSProductionRunner._process_4h_tick_inner)

    def test_seat_runs_after_the_trend_inject(self, src):
        assert src.index("[REGIMEBOOK-SEAT]") > src.index("[TREND-LAYER"), (
            "the seat must run AFTER the trend inject so enforce "
            "deterministically wins the seat")

    def test_enforce_only(self, src):
        i = src.index("REGIMEBOOK SEAT")
        blk = src[i:i + 4000]
        assert '_rb_mode == "enforce"' in blk, (
            "the seat must be reachable ONLY in enforce — shadow/off must be "
            "byte-neutral")

    def test_both_dicts_and_the_sleeve_bridge_are_written(self, src):
        i = src.index("REGIMEBOOK SEAT")
        blk = src[i:i + 5000]
        for needle in ('market_data["quant_direction"]',
                       'agent_signals["quant_direction"]',
                       'market_data["signal_edge_bps"]',
                       'agent_signals["signal_edge_bps"]',
                       "_last_quant_directions[asset]"):
            assert needle in blk, (
                f"seat missing write: {needle} — writing one dict alone is "
                f"shadowed by the earlier copy (the P170 two-dict trap), and "
                f"skipping the P149 bridge leaves the sleeve on the stale "
                f"pre-inject snapshot")

    def test_alpha_constant_is_not_looser_than_the_trend_seat(self, src):
        i = src.index("REGIMEBOOK SEAT")
        blk = src[i:i + 5000]
        assert "30.0 * abs(" in blk, (
            "the seat's asserted edge must stay at the trend seat's "
            "effective 30bps (40 x 0.75) — a looser constant would loosen "
            "the alpha gate as a side effect of a direction-source swap")

    def test_absence_takes_no_seat(self, src):
        i = src.index("REGIMEBOOK SEAT")
        blk = src[i:i + 5000]
        assert "seat NOT taken" in blk and "if _rb is None" in blk


# ---------------------------------------------------------------------------
# [P256] adjusted-book ledger: lab/runtime parity
# ---------------------------------------------------------------------------

class TestAdjustedLedger:
    def test_step_function_matches_the_lab_exactly(self):
        """The runtime adjust_step must reproduce mechanism_lab.apply_adjust
        bar-for-bar — otherwise the forward ledger measures a DIFFERENT
        mechanism than the one the lab selected (the P164/P214 lab-vs-live
        skew class, in the harness whose whole job is parity)."""
        import numpy as np
        from defense.regime_book_shadow import adjust_step
        from training.mechanism_lab import apply_adjust
        rng = np.random.default_rng(7)
        raw = rng.choice([-1.0, 0.0, 1.0], size=500,
                         p=[0.2, 0.35, 0.45])
        for (ke, kf, mh) in [(1, 2, 0), (3, 1, 0), (1, 1, 6), (2, 2, 3)]:
            lab_out = apply_adjust(raw, ke, kf, mh)
            st: dict = {}
            run_out = [adjust_step(st, float(w), ke, kf, mh) for w in raw]
            assert list(lab_out) == run_out, (
                f"lab/runtime divergence at params ke={ke} kf={kf} mh={mh}")

    def test_record_tick_writes_both_ledgers(self, tmp_path):
        h = _harness(tmp_path)
        closes = [100.0 + i * 0.1 for i in range(720)]  # rising -> bull
        rec = h.record_tick("ETH", closes, price=closes[-1])
        assert rec is not None
        assert (tmp_path / "strategy_shadow" / "regimebook_ETH.jsonl").exists()
        apath = tmp_path / "strategy_shadow" / "regimebook_adj_ETH.jsonl"
        assert apath.exists(), "adjusted ledger not written"
        arec = json.loads(apath.read_text(encoding="utf-8").strip())
        assert arec["strategy"] == "regimebook_adj", (
            "the adjusted rows must carry a DISTINCT strategy name — the "
            "scorer groups by that field, and sharing 'regimebook' would "
            "pollute the raw book's forward IC")

    def test_banded_lab_uses_the_shared_step_single_source(self):
        """[P259] The lab must run the band through defense/regime_book_shadow
        .banded_step — one source of truth beats parity tests (P172). A
        re-inlined copy in the lab is how lab/live drift starts."""
        src = (REPO / "training" / "banded_forecast_lab.py").read_text(
            encoding="utf-8")
        assert "from defense.regime_book_shadow import banded_step" in src

    def test_banded_features_lab_vs_live_parity(self):
        """[P259] The harness's stdlib feature builder must reproduce the
        lab's pandas builder on the same series — these are the model's
        INPUTS; divergence here silently retargets the forward ledger."""
        import numpy as np
        from training.banded_forecast_lab import close_features
        from defense.regime_book_shadow import banded_features_from_closes
        rng = np.random.default_rng(11)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 800)))
        fz = np.full(800, 0.7)
        X, names = close_features(close, fz)
        live = banded_features_from_closes(list(close), 0.7)
        assert live is not None and len(live) == len(names)
        lab_last = X[-1]
        for i, name in enumerate(names):
            assert abs(live[i] - lab_last[i]) < 1e-6, (
                f"feature {name}: live={live[i]} lab={lab_last[i]}")

    def test_banded_overlay_semantics(self, tmp_path):
        """Book non-flat wins the overlay; the banded model only fills the
        cells the book leaves flat."""
        h = _harness(tmp_path)
        h._banded_models["BTC"] = {
            "features": [], "mean": [], "scale": [], "coef": [],
            "intercept": 0.5, "forecast_sigma": 0.1,
            "band": {"t_enter": 1.0, "t_exit": 0.25, "regime_gated": False}}
        # monkeypatch features to a valid empty vector path: use real closes
        closes = [100.0] * 720
        r = h._banded_overlay_target("BTC", closes, 1.0, "bull", None)
        assert r is not None and r[0] == 1.0, "book target must win when non-flat"
        r2 = h._banded_overlay_target("BTC", closes, 0.0, "peace", None)
        assert r2 is not None and r2[0] == r2[1], (
            "flat book cell must take the banded target")

    def test_all_banded_exports_absent_by_decision(self):
        """[P259b] The banded overlay was WITHDRAWN the same day it was
        built: its single ledgered validation-era read (the one window the
        selection never touched) showed the increment NEGATIVE on both
        earners (BTC -0.212, ETH -0.618) — era-fragility that survived the
        entire same-day kill battery and died only on unread data. SOL never
        earned an export in the first place. An export reappearing without a
        new P-entry is a withdrawn candidate resurrecting silently (the
        exact failure the SOL-bear-ridge refusal gate exists for)."""
        for a in ("BTC", "ETH", "SOL"):
            assert not (REPO / "configs" / "regimebook" / f"{a}_banded.json"
                        ).exists(), (
                f"{a}_banded.json exists — the banded overlay was withdrawn "
                f"on its validation read (P259b); re-export requires "
                f"--force-withdrawn + new era-stable evidence + a P-entry")

    def test_export_script_refuses_without_force(self):
        src = (REPO / "training" / "scripts" /
               "export_banded_models.py").read_text(encoding="utf-8")
        assert "--force-withdrawn" in src and "REFUSING" in src


class TestAdjustSemantics:
    """[P260] Pins min_hold's TRUE behavior (fresh-mind review finding #1):
    under CONTINUOUS exit demand, `held` never accrues and a plain exit is
    blocked indefinitely — NOT "hold N bars then exit freely". The lab
    measured and selected exactly this, so the mechanism stays; the pin
    exists so nobody flips an enforce decision believing the docstring's
    old, wrong description."""

    def test_continuous_exit_demand_never_releases_under_min_hold(self):
        from defense.regime_book_shadow import adjust_step
        st: dict = {}
        assert adjust_step(st, 1.0, 1, 1, 6) == 1.0     # instant entry
        for _ in range(50):                              # 50 bars of exit demand
            out = adjust_step(st, 0.0, 1, 1, 6)
        assert out == 1.0, (
            "expected: continuous exit demand is blocked indefinitely by "
            "min_hold (the MEASURED semantics) — if this now exits, the "
            "mechanism changed and the forward ledger no longer tests what "
            "the lab selected")

    def test_flip_demand_escapes_min_hold(self):
        from defense.regime_book_shadow import adjust_step
        st: dict = {}
        adjust_step(st, 1.0, 1, 1, 6)
        assert adjust_step(st, -1.0, 1, 1, 6) == -1.0, (
            "a flip (k_flip=1) must escape min_hold — flips honor k_flip, "
            "not the hold")

    def test_relented_then_renewed_exit_can_release(self):
        from defense.regime_book_shadow import adjust_step
        st: dict = {}
        adjust_step(st, 1.0, 1, 1, 3)
        adjust_step(st, 0.0, 1, 1, 3)        # exit demand (blocked, held=0)
        for _ in range(4):                   # demand relents -> held accrues
            adjust_step(st, 1.0, 1, 1, 3)
        assert adjust_step(st, 0.0, 1, 1, 3) == 0.0, (
            "after the demand relents long enough for held >= min_hold, a "
            "plain exit must succeed")

    def test_adj_params_match_the_lab_report(self):
        """The deployed params must be the mechanism-lab winners — if the lab
        re-runs and selects differently, this fails until both move
        together."""
        from defense.regime_book_shadow import ADJ_PARAMS
        rpt = REPO / "training" / "reports" / "mechanism_lab_p256.json"
        if not rpt.exists():
            pytest.skip("lab report absent in this checkout")
        adj = json.loads(rpt.read_text(encoding="utf-8")).get("adjust", {})
        for a, (ke, kf, mh) in ADJ_PARAMS.items():
            best = adj.get(a, {}).get("grid", [{}])[0]
            if best:
                assert (best["k_exit"], best["k_flip"],
                        best["min_hold"]) == (ke, kf, mh), (
                    f"{a}: deployed ADJ_PARAMS != lab winner {best}")

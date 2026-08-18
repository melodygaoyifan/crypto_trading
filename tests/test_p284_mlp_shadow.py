"""P284 — the pooled-certified BTC mlp_small Rung-3 shadow, pinned.

The parity pins here ARE the deliverable: the SOL ridge died because its
live feature path silently diverged from the lab's; this harness's whole
design is that the forward pass, the features, and the cadence are the
certified ones or the tick records flat-with-names.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tests._source_scan import read_source
from defense.mlp_shadow import MlpShadow, BAR_SEC

REPO = Path(__file__).resolve().parent.parent


def _tiny_export(seed=0):
    """A REAL sklearn fit exported through the same shape the exporter
    writes — the parity oracle."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, 5))
    y = X[:, 0] * 0.1 + rng.normal(scale=0.05, size=400)
    sc = StandardScaler().fit(X)
    m = MLPRegressor(hidden_layer_sizes=(7,), alpha=1e-2, max_iter=200,
                     early_stopping=True, random_state=7).fit(
        sc.transform(X), y)
    sig = float(np.std(m.predict(sc.transform(X)))) or 1e-9
    return {
        "asset": "BTC", "feature_names": [f"f{i}" for i in range(5)],
        "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
        "w1": m.coefs_[0].tolist(), "b1": m.intercepts_[0].tolist(),
        "w2": np.asarray(m.coefs_[1]).reshape(-1).tolist(),
        "b2": float(np.asarray(m.intercepts_[1]).reshape(-1)[0]),
        "hidden_activation": "relu", "sig": sig,
        "deadband": 0.25, "decision_interval": 4,
    }, m, sc, sig


class TestForwardParity:
    def test_stdlib_forward_matches_sklearn_exactly(self):
        # THE load-bearing pin: if the serve-side math drifts from the
        # certified sklearn model, the ledger forward-tests a different
        # model than was certified (the P164/P214 class).
        export, m, sc, sig = _tiny_export()
        rng = np.random.default_rng(1)
        for _ in range(50):
            x = rng.normal(size=5)
            want = float(m.predict(sc.transform(x.reshape(1, -1)))[0]) / sig
            got = MlpShadow.forward(export, x.tolist())
            assert abs(want - got) < 1e-9, (
                f"forward-pass parity broken: sklearn {want} vs stdlib "
                f"{got}")


class TestDecisionCadence:
    def _shadow(self, export, tmp_path):
        s = object.__new__(MlpShadow)
        s._dir = tmp_path
        s._models = {"BTC": export}
        s._state = {}
        s._warned = {}
        return s

    def test_holds_between_decision_bins(self):
        export, *_ = _tiny_export()
        s = self._shadow(export, Path("."))
        t0 = (1_800_000_000 // (BAR_SEC * 4)) * (BAR_SEC * 4)  # bin%4==0
        d0 = s.decide("BTC", 2.0, t0)
        assert d0 == 1.0
        # next bar (bin%4==1): a violent opposite z must NOT change it
        d1 = s.decide("BTC", -3.0, t0 + BAR_SEC)
        assert d1 == 1.0, ("the DI=4 hold broke — the certified candidate "
                          "holds 16h between decisions (positions_from_z "
                          "semantics)")
        # the NEXT decision bin honors the new signal
        d4 = s.decide("BTC", -3.0, t0 + 4 * BAR_SEC)
        assert d4 == -1.0

    def test_deadband_flattens(self):
        export, *_ = _tiny_export()
        s = self._shadow(export, Path("."))
        t0 = (1_800_000_000 // (BAR_SEC * 4)) * (BAR_SEC * 4)
        assert s.decide("BTC", 0.1, t0) == 0.0


class TestCoverageRefusal:
    def test_missing_feature_records_flat_with_names(self, tmp_path):
        export, *_ = _tiny_export()
        s = object.__new__(MlpShadow)
        s._dir = tmp_path
        s._models = {"BTC": export}
        s._state = {}
        s._warned = {}
        feats = {f"f{i}": 1.0 for i in range(5)}
        pres = {f"f{i}": True for i in range(5)}
        pres["f3"] = False   # one uncovered feature
        s.tick({"BTC": feats}, {"BTC": pres})
        rec = json.loads((tmp_path / "mlpshadow_BTC.jsonl")
                         .read_text(encoding="utf-8").splitlines()[-1])
        assert rec["direction"] == 0.0 and rec["confidence"] == 0.0
        assert "f3" in str(rec["coverage_note"]), (
            "a coverage gap must record flat WITH the missing names — a "
            "partial vector is a different model (the SOL-ridge death)")

    def test_full_coverage_emits_the_sign_expression(self, tmp_path):
        export, *_ = _tiny_export()
        s = object.__new__(MlpShadow)
        s._dir = tmp_path
        s._models = {"BTC": export}
        s._state = {}
        s._warned = {}
        feats = {f"f{i}": 3.0 for i in range(5)}
        pres = {f"f{i}": True for i in range(5)}
        s.tick({"BTC": feats}, {"BTC": pres})
        rec = json.loads((tmp_path / "mlpshadow_BTC.jsonl")
                         .read_text(encoding="utf-8").splitlines()[-1])
        assert rec["direction"] in (-1.0, 0.0, 1.0)
        assert rec["confidence"] == abs(rec["direction"])  # P236/P224


class TestExportAndWiring:
    def test_export_exists_and_carries_the_contract(self):
        p = REPO / "configs" / "mlpshadow" / "BTC.json"
        assert p.exists(), "the certified export is missing"
        m = json.loads(p.read_text(encoding="utf-8"))
        assert len(m["feature_names"]) == 24
        assert m["deadband"] == 0.25 and m["decision_interval"] == 4
        assert m["provenance"]["fit_policy"] == "split_aware_verified"
        # the two fv2 members that force the flow-feed wiring
        assert "fv2_rel_strength_24h" in m["feature_names"]

    def test_main_wiring_and_both_scorer_sites(self):
        src = read_source(REPO / "main.py")
        assert "MlpShadow" in src and "_mlp_shadow.tick(" in src
        assert "get_flow_feed" in src, (
            "the fv2 members need the live flow feed — without it the "
            "harness records flat(cov) forever")
        scorer = read_source(REPO / "analytics" / "shadow_ic" /
                             "compute_shadow_ic.py")
        assert scorer.count("mlpshadow") >= 2  # P192 two-site rule

    def test_obs_builder_stashes_raw_presence(self):
        src = read_source(REPO / "drl" / "runtime_obs_builder.py")
        assert "last_raw_features" in src and "last_raw_presence" in src
        # the stash must be PRE-scaling: it appears before _scale_features
        assert src.find("last_raw_features[asset]") < src.find(
            "self._scale_features(asset, features_122)"), (
            "the raw stash moved after DRL scaling — the harness would "
            "feed fold-scaled values into its OWN scaler (double scaling)")

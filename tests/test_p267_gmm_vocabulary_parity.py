"""[P267] The deployed GMM artifacts and the regime-keyed maps must agree.

The regime vocabulary is DATA (cluster names come out of a fit) while a dozen
maps hardcode name-keyed entries — the drift between them is how
STEADY_UPTREND spent 4 months routed to the mean-reversion bucket (P217) and
how three vocabularies came to coexist (P215). This file pins the deploy-side
artifacts (models/regime_classifier — the fits the RUNTIME serves) against:

  * kraken_quant's `_REGIME_MAP`: every emitted name EXPLICITLY mapped (the
    P217 pin, generalized to read the artifact instead of a typed list);
  * a known-name pool: a refit that invents a novel name fails HERE, loudly,
    instead of silently falling into every table's default hole;
  * artifact-set integrity: k == len(names) == len(weights), scaler arrays
    match the 12 feature columns, and fit_policy is split-aware — a
    full-sample fit reaching the deploy directory is the P164/P200 leak
    arriving at the runtime, whatever the training side did.

The artifacts are operator-local (models/ is gitignored) — in CI this file
skips loudly with its reason (the P252b pattern) rather than failing on an
absent file or, worse, passing vacuously while claiming coverage.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GMM_DIR = REPO / "models" / "regime_classifier"
ASSETS = ("BTC", "ETH", "SOL")

# Every name any HMATS GMM naming pass may legitimately emit. A refit that
# produces a name outside this pool must be a deliberate vocabulary change
# (update this pool AND every regime-keyed map in the same commit), never a
# silent addition.
KNOWN_REGIME_POOL = {
    "EXTREME_VOLATILITY", "QUIET_ACCUMULATION", "MOMENTUM_RALLY",
    "WEAK_CONSOLIDATION", "VOLATILE_CHOP", "PANIC_SELLOFF",
    "STEADY_UPTREND", "NEUTRAL_DRIFT",
}


def _configs():
    out = {}
    for a in ASSETS:
        p = GMM_DIR / a / "gmm_config.json"
        if p.exists():
            out[a] = json.loads(p.read_text(encoding="utf-8"))
    return out


needs_artifacts = pytest.mark.skipif(
    not any((GMM_DIR / a / "gmm_config.json").exists() for a in ASSETS),
    reason="models/regime_classifier is operator-local (gitignored) — the "
           "artifact pins run only where the deploy-side fits live (P252b "
           "pattern; CI verifies the code half below)")


@needs_artifacts
class TestDeployedArtifacts:
    def test_every_emitted_name_is_explicitly_mapped_in_kraken_quant(self):
        from agents.kraken_quant_agent import _REGIME_MAP
        for a, cfg in _configs().items():
            for name in cfg.get("regime_names", []):
                assert name in _REGIME_MAP, (
                    f"{a} emits {name!r} but kraken_quant._REGIME_MAP has no "
                    f"explicit entry — it would silently take the SIDEWAYS "
                    f"default, the exact P217 failure (STEADY_UPTREND spent "
                    f"4 months in the mean-reversion bucket)")

    def test_no_novel_names_outside_the_known_pool(self):
        for a, cfg in _configs().items():
            novel = set(cfg.get("regime_names", [])) - KNOWN_REGIME_POOL
            assert not novel, (
                f"{a} emits {novel} — names unknown to every regime-keyed "
                f"table (ADVISE weights, price-reversal, dd multipliers, "
                f"gambler, short_bias, smart beta). A vocabulary change must "
                f"update the pool and the tables in the same commit.")

    def test_artifact_internal_consistency(self):
        for a, cfg in _configs().items():
            k = cfg["n_components"]
            assert len(cfg["regime_names"]) == k, (
                f"{a}: {len(cfg['regime_names'])} names for k={k}")
            assert len(cfg["weights"]) == k
            n_feat = len(cfg["feature_cols"])
            # [P307] read the expected count from the training list rather
            # than a literal — the literal went stale the moment the
            # feature set legitimately changed (12 -> 9), which is a
            # test failing on a correct change rather than a wrong one.
            from tests.test_rebuild_pipeline_gmm_split import (
                _load_rebuild_module)
            GMM_FEATURE_COLS = _load_rebuild_module().GMM_FEATURE_COLS
            assert n_feat == len(GMM_FEATURE_COLS), (
                f"{a}: deployed GMM has {n_feat} features, the builder "
                f"declares {len(GMM_FEATURE_COLS)} — redeploy the artifacts")
            assert len(cfg["scaler_mean"]) == n_feat
            assert len(cfg["scaler_scale"]) == n_feat

    def test_deployed_fits_are_split_aware(self):
        # A full-sample fit in the DEPLOY directory is the P164/P200 leak
        # arriving at the runtime regardless of what the training side did.
        for a, cfg in _configs().items():
            assert cfg.get("fit_policy") == "split_aware", (
                f"{a}: deploy-side fit_policy={cfg.get('fit_policy')!r} — "
                f"pre-P200 full-sample fits must never ship again")

    def test_model_and_scaler_files_travel_with_the_config(self):
        for a in _configs():
            for f in ("gmm_model.pkl", "scaler.pkl"):
                assert (GMM_DIR / a / f).exists(), (
                    f"{a}/{f} missing — a config without its model is half "
                    f"an artifact set (P215: the set moves as ONE)")


class TestCodeSideAlwaysRuns:
    """The half CI can check without the artifacts."""

    def test_the_regime_map_covers_the_whole_known_pool(self):
        from agents.kraken_quant_agent import _REGIME_MAP
        missing = KNOWN_REGIME_POOL - set(_REGIME_MAP)
        assert not missing, (
            f"kraken_quant._REGIME_MAP lacks explicit entries for {missing} "
            f"— any fit emitting them falls into the silent SIDEWAYS default")

    def test_the_runtime_loader_is_k_agnostic(self):
        # The loader must read k/names/scaler from the config, never assume
        # the old k=8 shape (the clean fits are k=6/7/7).
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        i = src.index("Try per-asset models first")
        seg = src[i:i + 3000]
        assert '_cfg.get("regime_names"' in seg or "_cfg[\"regime_names\"]" in seg.replace("'", '"'), (
            "the per-asset GMM loader no longer reads regime_names from the "
            "artifact config")
        assert "range(8)" not in seg, "the loader hardcodes k=8"

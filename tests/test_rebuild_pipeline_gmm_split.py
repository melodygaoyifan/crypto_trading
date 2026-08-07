"""[P199] rebuild_pipeline's GMM must fit train-only — the leak P164 missed.

P164 fixed the full-history GMM fit in train_per_asset_gmm.py, but
rebuild_pipeline.py Step 4 — the script that generated the deployed training
parquets — kept fitting scaler + GMM + BIC-k + cluster names on 100% of
history (and deploying the result). These tests pin the fix:

  * the fit boundary is the STRICTEST fold's train_end (fold_3), never
    fold_1's — fitting to fold_1's boundary still leaks folds 2/3's val
    windows, which sit inside fold_1's train range;
  * the boundary arithmetic never exceeds the trainer's own fold_3 train_end
    for any data length (conservative under int truncation);
  * full-sample fitting survives only behind the explicit --gmm-no-split
    opt-in, and the artifact records which policy produced it (a leaky GMM
    and a clean GMM are indistinguishable by value — P179: record the source).
"""

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_rebuild_module():
    """Exec rebuild_pipeline.py in isolation from the repo-root `scripts`
    package. rebuild_pipeline does `from scripts.wavelet_denoise import ...`
    expecting training/ on sys.path (training/scripts), but if an earlier
    test imported the repo-root scripts/ package (e.g. test_health_validator
    -> scripts.live_watchdog), sys.modules['scripts'] is already bound to the
    wrong package and the exec fails. Temporarily clear the cached entries;
    restore afterwards so this test cannot break other tests' imports."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == "scripts" or k.startswith("scripts.")}
    training_dir = str(REPO / "training")
    sys.path.insert(0, training_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            "rebuild_pipeline_under_test",
            REPO / "training" / "scripts" / "rebuild_pipeline.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        with_prefix = [k for k in list(sys.modules)
                       if k == "scripts" or k.startswith("scripts.")]
        for k in with_prefix:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
        if training_dir in sys.path:
            sys.path.remove(training_dir)


@pytest.fixture(scope="module")
def rp():
    return _load_rebuild_module()


# ---------------------------------------------------------------------------
# Boundary arithmetic
# ---------------------------------------------------------------------------

def test_boundary_never_exceeds_the_trainers_strictest_fold(rp):
    """train_drl_full._get_fold_splits: val_size = int(n*0.15), gap=42,
    fold_3 train_end = n - 3*val_size - gap. The rebuild boundary must be
    <= that for every n, or some fold's val window enters the GMM fit."""
    for n in range(2000, 30000, 137):
        trainer_fold3_end = n - 3 * int(n * 0.15) - 42
        assert rp.gmm_fit_boundary(n) <= trainer_fold3_end, (
            f"n={n}: rebuild boundary {rp.gmm_fit_boundary(n)} > trainer "
            f"fold_3 train_end {trainer_fold3_end} — the GMM fit would see "
            f"validation bars"
        )


def test_trainer_fold_arithmetic_still_matches_what_this_test_assumes():
    """Drift guard: the parity test above replicates the trainer's fold
    arithmetic. If train_drl_full changes val_ratio/gap, this must fail so
    the boundary constant is reconsidered rather than silently diverging."""
    src = io.open(REPO / "training" / "train_drl_full.py", encoding="utf-8").read()
    assert "val_size = int(n * 0.15)" in src.replace("*0.15", "* 0.15"), (
        "train_drl_full no longer computes val_size = int(n * 0.15) — update "
        "GMM_FIT_VAL_RATIO in rebuild_pipeline.py and this test together"
    )
    assert "gap = 42" in src, (
        "train_drl_full no longer uses gap=42 — update GMM_FIT_GAP in "
        "rebuild_pipeline.py and this test together"
    )


def test_boundary_refuses_too_little_data(rp):
    with pytest.raises(ValueError, match="too little data"):
        rp.gmm_fit_boundary(1500)


# ---------------------------------------------------------------------------
# Behavioral: the fit actually uses only pre-boundary rows
# ---------------------------------------------------------------------------

def _synthetic_features(rp, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    d = len(rp.GMM_FEATURE_COLS)
    # two mild clusters so BIC search converges quickly
    X = rng.normal(0, 1, size=(n, d))
    X[n // 2:] += 0.8
    return X


def test_split_aware_fit_uses_only_pre_boundary_rows(rp, tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "GMM_BUILD_DIR", tmp_path)
    X = _synthetic_features(rp)
    rp.retrain_gmm_per_asset("BTC", X, no_split=False)
    import json
    cfg = json.load(open(tmp_path / "BTC" / "gmm_config.json", encoding="utf-8"))
    expected = rp.gmm_fit_boundary(len(X))
    assert cfg["training_samples"] == expected, (
        f"GMM fit on {cfg['training_samples']} rows, expected the boundary "
        f"{expected} — validation bars entered the fit"
    )
    assert cfg["fit_policy"] == "split_aware"


def test_no_split_fits_everything_and_is_labelled_leaky(rp, tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "GMM_BUILD_DIR", tmp_path)
    X = _synthetic_features(rp)
    rp.retrain_gmm_per_asset("BTC", X, no_split=True)
    import json
    cfg = json.load(open(tmp_path / "BTC" / "gmm_config.json", encoding="utf-8"))
    assert cfg["training_samples"] == len(X)
    assert cfg["fit_policy"] == "full_sample_LEAKY", (
        "a full-sample GMM artifact must be self-describing — it is "
        "indistinguishable from a clean one by value"
    )


def test_nan_rows_do_not_shift_the_boundary_past_val_bars(rp, tmp_path, monkeypatch):
    """The boundary counts VALID rows; leading-NaN warmup rows must not let
    later (val-window) bars slip into the fit."""
    monkeypatch.setattr(rp, "GMM_BUILD_DIR", tmp_path)
    X = _synthetic_features(rp)
    X[:200] = np.nan  # indicator warmup
    rp.retrain_gmm_per_asset("BTC", X, no_split=False)
    import json
    cfg = json.load(open(tmp_path / "BTC" / "gmm_config.json", encoding="utf-8"))
    n_valid = len(X) - 200
    assert cfg["training_samples"] == rp.gmm_fit_boundary(n_valid)


# ---------------------------------------------------------------------------
# CLI wiring (P152 shape: an unwired knob is not a knob)
# ---------------------------------------------------------------------------

def test_cli_passes_no_split_through():
    src = io.open(REPO / "training" / "scripts" / "rebuild_pipeline.py",
                  encoding="utf-8").read()
    assert "--gmm-no-split" in src
    assert "no_split=args.gmm_no_split" in src, (
        "the --gmm-no-split flag is no longer wired into "
        "retrain_gmm_per_asset — default-path behavior is undefined"
    )


def test_orchestrator_passes_the_extractor():
    """[P199 blocker 3] run_training.run_drl must pass --extractor
    lstm_film_a or it produces a 126-dim ULTIMATE model the runtime's
    hardcoded 1008-dim input cannot consume."""
    src = io.open(REPO / "training" / "run_training.py", encoding="utf-8").read()
    assert "'--extractor', 'lstm_film_a'" in src

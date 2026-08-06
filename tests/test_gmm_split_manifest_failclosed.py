"""[P164] The GMM must not silently fit on the validation and test windows.

`train_per_asset_gmm.py` exists to enforce Iron Rule #12 — fit the regime model
on training data only. It read the fold boundaries from

    PROJECT_ROOT / "config" / "split_manifest.json"

while `generate_split_manifest.py` writes

    PROJECT_ROOT / "configs" / "split_manifest.json"

`config/` exists (it holds optuna_winner.json), so the path resolved without
error and simply never matched. `load_split_manifest` then returned `{}`,
`train_end` arrived as None, and `train_gmm_for_asset` logged "Using ALL data
for GMM fit" — fitting the scaler, the GaussianMixture, the BIC k-selection and
the cluster naming on 100% of history, then emitting `regime_proba_0..7` for
every bar. Eight contaminated features, on every run this script ever had.

Two failures compounded: a one-character path typo, and a fallback that treated
"I could not find the boundaries" as "proceed without boundaries." The second
is the dangerous one — it turned a missing file into a silent leak. `--no-split`
already exists as the explicit way to ask for a full-sample fit, so the load
path has no reason to degrade quietly.
"""

import json

import pytest

from training.scripts import train_per_asset_gmm as gmm


VALID_MANIFEST = {
    "assets": {
        "BTC": {"folds": [{"fold": 1, "train_end": 12000},
                          {"fold": 3, "train_end": 15000}]},
        "ETH": {"folds": [{"fold": 1, "train_end": 11800}]},
    }
}


def _write_manifest(root, payload, dirname="configs"):
    target = root / dirname
    target.mkdir(parents=True, exist_ok=True)
    (target / "split_manifest.json").write_text(json.dumps(payload))
    return target


def test_reads_from_configs_not_config(tmp_path, monkeypatch):
    """The regression: the manifest lives in configs/, plural."""
    _write_manifest(tmp_path, VALID_MANIFEST, dirname="configs")
    # a decoy in the singular directory, as on disk today
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "optuna_winner.json").write_text("{}")

    monkeypatch.setattr(gmm, "PROJECT_ROOT", tmp_path)
    train_ends = gmm.load_split_manifest(fold=1)

    assert train_ends == {"BTC": 12000, "ETH": 11800}


def test_selects_the_requested_fold(tmp_path, monkeypatch):
    _write_manifest(tmp_path, VALID_MANIFEST)
    monkeypatch.setattr(gmm, "PROJECT_ROOT", tmp_path)

    assert gmm.load_split_manifest(fold=3) == {"BTC": 15000}


def test_missing_manifest_raises_instead_of_leaking(tmp_path, monkeypatch):
    """Fail closed. Returning {} here is what disabled Iron Rule #12."""
    monkeypatch.setattr(gmm, "PROJECT_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError, match="Refusing to fall back"):
        gmm.load_split_manifest(fold=1)


def test_manifest_without_requested_fold_raises(tmp_path, monkeypatch):
    """A present-but-unusable manifest must not degrade to a full-sample fit."""
    _write_manifest(tmp_path, VALID_MANIFEST)
    monkeypatch.setattr(gmm, "PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="no fold_9 boundaries"):
        gmm.load_split_manifest(fold=9)


def test_empty_manifest_raises(tmp_path, monkeypatch):
    _write_manifest(tmp_path, {"assets": {}})
    monkeypatch.setattr(gmm, "PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError):
        gmm.load_split_manifest(fold=1)


def test_real_repo_manifest_is_where_the_loader_looks():
    """Belt and braces: the shipped path must resolve in this checkout."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert (root / "configs" / "split_manifest.json").exists(), (
        "configs/split_manifest.json missing — regenerate it with "
        "training/scripts/generate_split_manifest.py"
    )

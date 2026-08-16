"""P280 — the clean-GMM gate: no trainer or lab may consume regime features
whose GMM provenance is leaked or unverifiable.

Was a HUMAN step (Guide V2 §3 'MUST print: split_aware'; the P257 launch
verified it by hand). P279's research found nothing enforced it — a run
that skipped the manual check trained on whatever was on disk. Now
`training/splits.py::assert_clean_gmm` refuses at startup in BOTH
train_drl_full (before any GPU spend) and train_supervised_full.load_asset
(every lab flows through it).
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

from tests._source_scan import read_source

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
import splits  # noqa: E402


def _with_cfg(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(splits, "REPO", tmp_path)
    d = tmp_path / "training" / "training_data" / "gmm_models" / "BTC"
    d.mkdir(parents=True)
    if cfg is not None:
        (d / "gmm_config.json").write_text(json.dumps(cfg), encoding="utf-8")


class TestAssertCleanGmm:
    def test_split_aware_passes_and_returns_config(self, tmp_path, monkeypatch):
        _with_cfg(tmp_path, monkeypatch,
                  {"fit_policy": "split_aware", "n_components": 6})
        cfg = splits.assert_clean_gmm("BTC")
        assert cfg["n_components"] == 6

    def test_full_sample_fit_refuses(self, tmp_path, monkeypatch):
        # the P164/P200 leak arriving at a training run
        _with_cfg(tmp_path, monkeypatch, {"fit_policy": "full_sample"})
        with pytest.raises(SystemExit, match="REFUSING"):
            splits.assert_clean_gmm("BTC")

    def test_missing_fit_policy_refuses(self, tmp_path, monkeypatch):
        # a pre-P200 fit carries no policy — leaky by construction, and a
        # check that cannot run must never read as passed (P159)
        _with_cfg(tmp_path, monkeypatch, {"n_components": 8})
        with pytest.raises(SystemExit, match="REFUSING"):
            splits.assert_clean_gmm("BTC")

    def test_missing_config_refuses(self, tmp_path, monkeypatch):
        _with_cfg(tmp_path, monkeypatch, None)
        with pytest.raises(SystemExit, match="does not exist"):
            splits.assert_clean_gmm("BTC")

    def test_no_override_flag_exists(self):
        # a leaked GMM has no legitimate training use; the only sanctioned
        # leaky fit is rebuild_pipeline's explicit --gmm-no-split
        # visualization path, which never reaches this gate
        src = read_source(REPO / "training" / "splits.py")
        import re
        m = re.search(r"def assert_clean_gmm.*?(?=\ndef |\Z)", src, re.S)
        assert m and "allow" not in m.group(0).lower().replace(
            "disallow", ""), "an override crept into the clean-GMM gate"


class TestGateIsWired:
    def test_drl_trainer_calls_it_before_data_load(self):
        src = read_source(REPO / "training" / "train_drl_full.py")
        gate = src.find("assert_clean_gmm(args.asset)")
        load = src.find("Loading data:")
        assert 0 < gate < load, (
            "train_drl_full lost the clean-GMM gate (or it moved after the "
            "data load) — a leaked fit would burn GPU before refusing")

    def test_supervised_loader_calls_it(self):
        src = read_source(REPO / "training" / "train_supervised_full.py")
        assert "assert_clean_gmm(asset)" in src, (
            "load_asset lost the clean-GMM gate — every lab flows through "
            "this loader (regime_model_lab, mechanism_lab, the filter labs)")

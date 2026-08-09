"""[P1b] Runtime obs-139 support: the checkpoint declares its width, the
builder honors it, and absence of fv2 is a refusal — never a fabrication.

Rung-3 (P200-LADDER) deploys fv2-era checkpoints (139x8=1112) next to legacy
126x8=1008 ones. These tests pin the three load-bearing behaviors:
  * the loader derives single_obs_dim from the checkpoint and hard-refuses
    unknown widths (no squeezing a 139 model into a 126 pipeline);
  * a persisted 126-frame buffer is discarded, not stacked, under a 139
    model (the P148 persistence must not poison a Rung-3 deploy);
  * the builder's fv2 insertion mirrors the trainer exactly (before the
    trailing regime_proba block — the FiLM end-relative slice contract),
    and a missing fv2 dict returns None, never 13 zeros.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Loader: dimension derivation + refusal (source pins; behavior needs sb3)
# ---------------------------------------------------------------------------

def _ens_src():
    return io.open(REPO / "drl" / "ensemble.py", encoding="utf-8").read()


def test_loader_derives_dim_from_checkpoint():
    src = _ens_src()
    assert "self._tqc_model.observation_space.shape" in src
    assert "self.single_obs_dim = _total // N_STACK" in src


def test_loader_refuses_unknown_widths():
    src = _ens_src()
    assert "(126, 139)" in src, (
        "the width whitelist is gone — an arbitrary-width checkpoint would "
        "be served instead of refused"
    )


def test_buffer_restore_validates_dim_after_load():
    src = _ens_src()
    assert src.count("discarding buffer, warming up fresh (P1b)") >= 1
    assert "warming up fresh (P1b)" in src.split("def _load_tqc")[1], (
        "the POST-LOAD buffer revalidation is gone — _restore_buffer runs in "
        "__init__ before the dim is known, so the load-time check is the one "
        "that actually protects a Rung-3 deploy"
    )


# ---------------------------------------------------------------------------
# Builder: fv2 insertion + refusal
# ---------------------------------------------------------------------------

@pytest.fixture()
def builder_cls(monkeypatch):
    import drl.runtime_obs_builder as rob
    return rob


def _mk_builder(rob, include_fv2):
    b = rob.RuntimeObsBuilder.__new__(rob.RuntimeObsBuilder)
    # minimal manifest-shaped state (mirrors _load_manifest output)
    base = [f"f{i}" for i in range(102)]
    den = [f"d{i}" for i in range(5)]
    ext = [f"e{i}" for i in range(7)]
    reg = [f"regime_proba_{i}" for i in range(8)]
    b._base_features = base
    b._denoised_features = den
    b._external_features = ext
    b._regime_proba_features = reg
    b._all_features = base + den + ext + reg
    b._feature_engineer = None
    b._scalers = {}
    b._include_fv2 = include_fv2
    if include_fv2:
        from data_mgmt.flow_features import FV2_COLUMNS
        b._fv2_features = sorted(FV2_COLUMNS)
        b._all_features = b._all_features[:-8] + b._fv2_features + reg
    else:
        b._fv2_features = []
    return b


def test_legacy_builder_still_produces_126(builder_cls):
    b = _mk_builder(builder_cls, include_fv2=False)
    obs = b.build_obs("BTC", None, {}, [0.125] * 8, [0, 0, 0, 0])
    assert obs is not None and obs.shape == (126,)


def test_fv2_builder_produces_139_with_regime_at_film_slice(builder_cls):
    from data_mgmt.flow_features import FV2_COLUMNS
    b = _mk_builder(builder_cls, include_fv2=True)
    fv2 = {c: 0.5 for c in FV2_COLUMNS}
    probs = [1.0] + [0.0] * 7
    obs = b.build_obs("BTC", None, {}, probs, [1, 2, 3, 4], fv2=fv2)
    assert obs is not None and obs.shape == (139,)
    # FiLM contract: regime_proba at obs[-12:-4], env state last 4 —
    # exactly what the extractor slices, END-relative.
    assert obs[-12] == pytest.approx(1.0) and all(obs[-11:-4] == 0.0), \
        "regime_proba is not at obs[-12:-4] — the FiLM slice would read fv2"
    assert list(obs[-4:]) == [1.0, 2.0, 3.0, 4.0]
    # fv2 landed immediately before the regime block
    assert all(obs[-25:-12] == 0.5)


def test_fv2_builder_refuses_without_fv2(builder_cls):
    b = _mk_builder(builder_cls, include_fv2=True)
    obs = b.build_obs("BTC", None, {}, [0.125] * 8, [0, 0, 0, 0])
    assert obs is None, (
        "a 139 builder without fv2 values must return None — 13 fabricated "
        "zeros is a fake observation (P160/P170)"
    )


def test_fv2_builder_wrong_obs_call_passes_fv2_kwarg(builder_cls):
    from data_mgmt.flow_features import FV2_COLUMNS
    b = _mk_builder(builder_cls, include_fv2=True)
    obs = b.build_obs("BTC", None, {}, [0.125] * 8, [0, 0, 0, 0],
                      fv2={c: 0.1 for c in FV2_COLUMNS})
    assert obs is not None and obs.shape == (139,)

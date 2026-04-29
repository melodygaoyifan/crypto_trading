"""Phase 6 ML factor extraction + MLFactorFusionAgent shadow tests.

Verifies:
  - autoencoder builder returns the correct architecture (or None if torch absent)
  - MLFactorFusionAgent fail-closed when model missing (no crash)
  - MLFactorDispatcher routes per-asset
  - shadow harness writes ml_factor_*.jsonl ledger
  - sleeve allocator includes ml_factor sleeve after Phase 6
  - constitutional override is documented in __init__.py
  - Iron Law 6 confidence cap = 0.50 (ADVISE only)
"""

import json
from pathlib import Path

import pytest

from agents.ml_factor_fusion_agent import (
    MLFactorDispatcher,
    MLFactorFusionAgent,
    build_ml_factor_agents,
)
from defense.strategy_shadow_v5_1 import build_ml_factor_shadow_harness
from risk.sleeve_allocator_v5_1 import build_phase7_sleeve_allocator
from training.factor_extraction.autoencoder_factor import (
    DEFAULT_HIDDEN_DIMS,
    DEFAULT_LATENT_DIM,
    DEFAULT_PROMOTE_IC,
    build_autoencoder,
    compute_per_latent_ic,
    load_training_data,
    main as autoencoder_main,
)

# Skip torch-dependent tests if torch not installed
try:
    import torch  # noqa
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ---------- Constitutional override documentation -------------------------

def test_constitutional_override_documented():
    """Iron Law 3 explicit override must be in training/factor_extraction/__init__.py."""
    init_path = Path("training/factor_extraction/__init__.py")
    assert init_path.exists()
    src = init_path.read_text(encoding="utf-8")
    assert "CONSTITUTIONAL OVERRIDE" in src
    assert "Iron Law 3" in src
    assert "approved by operator" in src
    assert "Reversible" in src
    # Must explicitly preserve other Iron Laws
    assert "DRL ACTIVE" in src   # Iron Law 5
    assert "ADVISE" in src       # Iron Law 6


# ---------- build_autoencoder ---------------------------------------------

@pytest.mark.skipif(not TORCH_OK, reason="torch not installed")
def test_build_autoencoder_default_arch():
    enc, dec = build_autoencoder()
    assert enc is not None and dec is not None
    # Smoke shape
    import torch
    x = torch.randn(2, 122)
    z = enc(x)
    assert z.shape == (2, DEFAULT_LATENT_DIM)
    x_recon = dec(z)
    assert x_recon.shape == (2, 122)


@pytest.mark.skipif(not TORCH_OK, reason="torch not installed")
def test_build_autoencoder_custom_dims():
    enc, dec = build_autoencoder(input_dim=50, latent_dim=3, hidden_dims=(20, 8))
    import torch
    x = torch.randn(4, 50)
    z = enc(x)
    assert z.shape == (4, 3)


def test_build_autoencoder_returns_none_when_torch_absent(monkeypatch):
    """Mock _try_import_torch to return False; builder returns (None, None)."""
    from training.factor_extraction import autoencoder_factor as ae
    monkeypatch.setattr(ae, "_try_import_torch", lambda: False)
    enc, dec = ae.build_autoencoder()
    assert enc is None and dec is None


# ---------- MLFactorFusionAgent fail-closed -------------------------------

def test_agent_neutral_when_model_missing(tmp_path: Path):
    agent = MLFactorFusionAgent(asset="BTC", model_path=tmp_path / "missing.pt")
    out = agent.evaluate("BTC", {})
    assert out.direction == 0.0 and out.confidence == 0.0
    if TORCH_OK:
        assert "model_missing" in out.reason
    else:
        assert "torch_unavailable" in out.reason


def test_agent_neutral_on_asset_mismatch(tmp_path: Path):
    agent = MLFactorFusionAgent(asset="BTC", model_path=tmp_path / "missing.pt")
    out = agent.evaluate("ETH", {"close": 3000})
    assert out.direction == 0.0
    # Could be model_missing or asset_bound depending on torch
    if TORCH_OK:
        assert ("model_missing" in out.reason or "agent_bound_to_BTC" in out.reason)


def test_agent_iron_law_6_confidence_cap_is_advise_tier():
    """Iron Law 6: ML factor enters as ADVISE; confidence cap MUST stay <= 0.5."""
    agent = MLFactorFusionAgent(asset="BTC")
    assert agent.confidence_cap <= 0.50


# ---------- MLFactorDispatcher --------------------------------------------

def test_dispatcher_routes_per_asset():
    d = MLFactorDispatcher(assets=("BTC", "ETH", "SOL"))
    out = d.evaluate("BTC", {"close": 60000})
    # Will be NEUTRAL because no model — but routing should hit the BTC agent
    assert out.strategy_name == "ml_factor"
    assert out.asset == "BTC"


def test_dispatcher_unknown_asset_neutral():
    d = MLFactorDispatcher(assets=("BTC", "ETH"))
    out = d.evaluate("DOGE", {})
    assert out.direction == 0.0
    assert "no_agent_for_DOGE" in out.reason


# ---------- Shadow harness wiring -----------------------------------------

def test_ml_factor_harness_writes_jsonl(tmp_path: Path):
    h = build_ml_factor_shadow_harness(log_dir=tmp_path)
    h.observe("BTC", {})
    h.observe("ETH", {})
    files = list(tmp_path.glob("ml_factor_*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().split("\n")
    # Single dispatcher × 2 assets = 2 records
    assert len(lines) == 2


def test_ml_factor_harness_does_not_mutate_market_data(tmp_path: Path):
    h = build_ml_factor_shadow_harness(log_dir=tmp_path)
    md = {"close": 60000.0, "ret_4h": 0.01}
    snapshot = dict(md)
    h.observe("BTC", md)
    assert dict(md) == snapshot


# ---------- Sleeve allocator includes ml_factor ---------------------------

def test_phase7_allocator_includes_ml_factor_sleeve():
    a = build_phase7_sleeve_allocator()
    assert "ml_factor" in a._sleeves
    assert a._sleeves["ml_factor"].is_live is False


def test_phase7_allocator_now_5_sleeves():
    a = build_phase7_sleeve_allocator()
    # 5: directional_short + microstructure + cascade + funding + ml_factor
    assert len(a._sleeves) == 5


# ---------- compute_per_latent_ic -----------------------------------------

@pytest.mark.skipif(not TORCH_OK, reason="torch not installed")
def test_compute_per_latent_ic_shape():
    """Smoke: returns one IC value per latent dim."""
    import torch
    import pandas as pd
    enc, _ = build_autoencoder(input_dim=10, latent_dim=3, hidden_dims=(8,))
    features = pd.DataFrame({f"f{i}": [0.1 * j for j in range(20)] for i in range(10)})
    forward = pd.Series([0.001 * j for j in range(20)])
    ics = compute_per_latent_ic(enc, features, forward)
    assert len(ics) == 3
    # All values in [-1, 1]
    for ic in ics:
        assert -1.0 <= ic <= 1.0


def test_compute_per_latent_ic_returns_empty_when_torch_absent(monkeypatch):
    from training.factor_extraction import autoencoder_factor as ae
    monkeypatch.setattr(ae, "_try_import_torch", lambda: False)
    ics = ae.compute_per_latent_ic(None, None, None)
    assert ics == []


# ---------- load_training_data fail-closed --------------------------------

def test_load_training_data_missing_parquet(monkeypatch, tmp_path: Path):
    from training.factor_extraction import autoencoder_factor as ae
    monkeypatch.setattr(ae, "PARQUET_DIR", tmp_path)
    out = ae.load_training_data("NONEXISTENT")
    assert out is None


# ---------- builder + Iron Law check --------------------------------------

def test_build_ml_factor_agents_returns_3_per_default():
    agents = build_ml_factor_agents()
    assert len(agents) == 3
    assets = {a.asset for a in agents}
    assert assets == {"BTC", "ETH", "SOL"}


def test_agent_does_not_import_runtime_state():
    """Iron Law 10: agent reads model file + market_data; no runtime sizer/fusion."""
    src = Path("agents/ml_factor_fusion_agent.py").read_text(encoding="utf-8")
    assert "from risk.unified_position_sizer" not in src
    assert "from signals.authority_fusion" not in src


def test_autoencoder_factor_does_not_import_runtime_state():
    src = Path("training/factor_extraction/autoencoder_factor.py").read_text(encoding="utf-8")
    assert "from risk.unified_position_sizer" not in src
    assert "from signals.authority_fusion" not in src
    assert "from agents.drl_agent" not in src   # Iron Law 5: don't touch DRL training

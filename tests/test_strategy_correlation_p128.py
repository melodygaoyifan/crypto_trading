"""
test_strategy_correlation_p128.py — correlation matrix script (P131, v3 1.4)
================================================================================

v3 Track A item 1.4 (P1-6). Verifies the correlation script's structure
+ that core math primitives behave correctly. End-to-end execution
requires synced IC data (P130 sync_audit_data.sh output).
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "analytics" / "ic" / "compute_strategy_correlation.py"


class TestP131ScriptStructure:
    def test_script_exists(self):
        assert SCRIPT_PATH.exists(), f"P131 regression: {SCRIPT_PATH.name} missing"

    def test_script_imports_pandas_numpy(self):
        src = SCRIPT_PATH.read_text(encoding="utf-8-sig")
        assert "import pandas" in src
        assert "import numpy" in src

    def test_script_uses_pearson(self):
        """Pearson is the documented choice (linear correlation, robust to
        scale). Spearman would be a different design decision — flag if
        anyone changes it without updating the doc."""
        src = SCRIPT_PATH.read_text(encoding="utf-8-sig")
        assert 'method="pearson"' in src or "method='pearson'" in src, (
            "P131 regression: correlation method changed from Pearson. "
            "Operator review required."
        )

    def test_script_uses_kaiser_eigenvalue_threshold(self):
        """Effective independent sources = count of eigenvalues > 1.0
        (Kaiser criterion / unit-variance noise floor). Different
        threshold = different decision rule."""
        src = SCRIPT_PATH.read_text(encoding="utf-8-sig")
        assert "eigvals > 1.0" in src, (
            "P131 regression: eigenvalue threshold changed. "
            "v3 prompt decision rule assumes threshold=1.0."
        )


class TestCorrelationMath:
    """Sanity-check the math primitives the script relies on."""

    def test_correlation_matrix_symmetric(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.normal(size=(100, 5)),
                          columns=["a", "b", "c", "d", "e"])
        corr = df.corr(method="pearson")
        # Symmetric within float tolerance
        assert np.allclose(corr.values, corr.values.T)

    def test_correlation_matrix_diagonal_one(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.normal(size=(100, 5)),
                          columns=list("abcde"))
        corr = df.corr(method="pearson")
        assert np.allclose(np.diag(corr.values), 1.0)

    def test_correlation_matrix_in_unit_range(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.normal(size=(100, 5)),
                          columns=list("abcde"))
        corr = df.corr(method="pearson")
        # All entries in [-1, 1]
        assert (corr.values >= -1.0 - 1e-9).all()
        assert (corr.values <= 1.0 + 1e-9).all()

    def test_eigenvalue_threshold_counts_independent(self):
        """Identity matrix: 5 dims, all eigenvalues = 1. Threshold > 1
        gives 0 'independent' sources, threshold >= 1 gives 5. v3 rule
        uses > 1.0 strictly, so identity → 0 (which is the conservative
        case — it says 'this is just noise')."""
        identity = np.eye(5)
        eigvals = np.linalg.eigvalsh(identity)
        n_above = int(np.sum(eigvals > 1.0))
        # Identity eigvals are exactly 1.0 — strictly > 1.0 = 0
        assert n_above == 0

    def test_eigenvalue_threshold_finds_real_factors(self):
        """Build a matrix with 2 strong factors: rank-2 approximation
        should produce 2 large eigenvalues > 1."""
        rng = np.random.default_rng(42)
        n_obs, n_dims = 200, 6
        # 2 underlying factors driving 6 observed series
        factors = rng.normal(size=(n_obs, 2))
        loadings = rng.normal(size=(2, n_dims))
        observed = factors @ loadings + rng.normal(scale=0.1, size=(n_obs, n_dims))
        df = pd.DataFrame(observed, columns=list("abcdef"))
        corr = df.corr().fillna(0).values
        eigvals = np.linalg.eigvalsh(corr)
        # At least 2 eigvals should be > 1 (the 2 real factors); the rest
        # close to 0 (noise)
        assert int(np.sum(eigvals > 1.0)) >= 2


class TestP131OutputContract:
    """If a sync directory exists, the script's report JSON must have
    the expected schema."""

    def test_report_has_per_asset_blocks(self):
        from datetime import datetime
        report_path = Path("analytics/ic/reports") / f"agent_correlation_matrix_{datetime.now().strftime('%Y-%m-%d')}.json"
        if not report_path.exists():
            pytest.skip(f"No report at {report_path} — run script first")
        with open(report_path) as f:
            report = json.load(f)
        assert "per_asset" in report
        assert "version" in report
        assert "scope" in report
        for asset, info in report["per_asset"].items():
            if "note" in info:
                continue
            assert "n_agents" in info
            assert "effective_independent_sources" in info
            assert "high_corr_pairs" in info
            assert "eigenvalue_spectrum" in info
            # Effective sources sanity: 0 <= eff <= n_agents
            assert 0 <= info["effective_independent_sources"] <= info["n_agents"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

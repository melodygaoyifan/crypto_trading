"""[P393] Tests for training/scripts/macro_factor_ext_p393.py — fast,
synthetic, NO network, NO parquet reads. Pins:
  - the three new series (DGS2, T10Y2Y, BAMLH0A0HYM2) are in the run roster,
  - every convention is IMPORTED from macro_factor_lab, never re-typed
    (function identity + an AST scan that the ext module redefines none of
    the convention-bearing names),
  - the ext cache is a NEW file — the P392 cache path is never the write
    target,
  - the planted-lead control on the NEW alignment path (lag_bdays=3,
    horizon 5d) recovers IC ~ 1 (P174),
  - the DGS2 literature-lag alignment never reads a value before its
    publication lag (lag 3 >= the 2-bday publication lag, per sample),
  - the causal level-z: perturbing FUTURE levels leaves the past z values
    and the past aligned rows bit-identical (P164 construction test),
  - level-z fabricates nothing on absent days (P2),
  - the verdict truth table through the ext's beta-context application.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.scripts import macro_factor_lab as lab  # noqa: E402
from training.scripts import macro_factor_ext_p393 as ext  # noqa: E402

EXT_SRC = Path(ext.__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------- fixtures --

def _synthetic_dclose(n_days: int = 500, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    r = rng.normal(0, 0.02, n_days)
    return pd.Series(100.0 * np.exp(np.cumsum(r)), index=idx)


def _synthetic_level(n_bd: int = 400, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_bd, tz="UTC")
    return pd.Series(np.cumsum(rng.normal(0, 0.05, n_bd)) + 4.0, index=idx)


# ------------------------------------------------------------------ roster --

class TestRoster:
    def test_the_three_series_are_in_the_run_roster(self):
        assert set(ext.EXT_SERIES) == {"DGS2", "T10Y2Y", "BAMLH0A0HYM2"}

    def test_variants_cover_change_and_level_z(self):
        levels = {
            "DGS2": _synthetic_level(seed=1),
            "T10Y2Y": _synthetic_level(seed=2),
            "BAMLH0A0HYM2": _synthetic_level(seed=3) + 2.0,
        }
        variants, beta_ctx = ext.build_variants(levels)
        assert set(variants) == {"DGS2_chg", "T10Y2Y_chg",
                                 "HY_OAS_chg", "HY_OAS_levelz"}
        # the level-z variant's beta context is the change variant's (same
        # underlying series) — pre-committed in the docstring.
        assert beta_ctx["HY_OAS_levelz"] == "HY_OAS_chg"


# --------------------------------------------- conventions imported (P172) --

_CONVENTION_NAMES = (
    "known_from", "align_lead", "bd_changes", "lead_cell", "required_ic",
    "implied_edge_bps", "overlap_t", "decide_verdict", "contemporaneous_cell",
    "planted_factor", "shuffled_control", "spearman_ic", "obs_to_series",
    "daily_closes", "fetch_fred_series", "load_fred_key",
)


class TestConventionsImportedNotRetyped:
    def test_function_identity_with_the_lab(self):
        # the ext module's convention functions must BE the lab's objects.
        for name in ("known_from", "align_lead", "bd_changes", "lead_cell",
                     "required_ic", "decide_verdict", "contemporaneous_cell",
                     "planted_factor", "shuffled_control", "spearman_ic",
                     "obs_to_series", "daily_closes", "fetch_fred_series",
                     "load_fred_key"):
            assert getattr(ext, name) is getattr(lab, name), (
                f"{name} is not the lab's object — convention forked (P172)")

    def test_constants_are_the_labs(self):
        assert ext.PUBLICATION_LAG_BDAYS is lab.PUBLICATION_LAG_BDAYS
        assert ext.RT_COST_BPS is lab.RT_COST_BPS
        assert ext.EDGE_MARGIN == lab.EDGE_MARGIN
        assert ext.T_BAR == lab.T_BAR
        assert ext.MIN_N_EFF == lab.MIN_N_EFF

    def test_ext_module_redefines_no_convention_name(self):
        # AST scan: the ext file may not define (def/assign) any of the
        # convention-bearing names — an import is the only legal source.
        tree = ast.parse(EXT_SRC)
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in _CONVENTION_NAMES:
                    offenders.append(node.name)
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in _CONVENTION_NAMES:
                        offenders.append(tgt.id)
        assert not offenders, (
            f"ext module redefines convention names {offenders} — "
            f"conventions must be IMPORTED from macro_factor_lab (P172)")


# ---------------------------------------------------------- cache separation --

class TestCacheSeparation:
    def test_ext_cache_is_a_new_file_not_the_p392_cache(self):
        assert ext.EXT_CACHE_PATH.name == "macro_factor_series_p393_ext.json"
        assert ext.EXT_CACHE_PATH != ext.P392_CACHE_PATH
        assert ext.EXT_REPORT_PATH.name == "macro_factor_ext_p393.json"

    def test_ext_loader_never_writes_the_p392_cache(self):
        # the only write_text target in load_or_fetch_ext is EXT_CACHE_PATH.
        tree = ast.parse(EXT_SRC)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "load_or_fetch_ext")
        writes = [n for n in ast.walk(fn)
                  if isinstance(n, ast.Attribute) and n.attr == "write_text"]
        assert writes, "loader writes no cache at all?"
        for w in writes:
            assert (isinstance(w.value, ast.Name)
                    and w.value.id == "EXT_CACHE_PATH"), (
                "load_or_fetch_ext writes a path other than EXT_CACHE_PATH")


# ------------------------------------------------- controls on the new path --

class TestPlantedLeadOnNewPath:
    def test_planted_lead_at_lag3_horizon5_is_detected(self):
        # [P174] a factor that IS the 5d-forward return, planted at
        # lag_bdays=3 (the DGS2 literature-lag path), must come out IC ~ 1.
        dc = _synthetic_dclose()
        pf = lab.planted_factor(dc, horizon_days=ext.DGS2_LIT_HORIZON_D,
                                lag_bdays=3)
        al = lab.align_lead(pf, dc, ext.DGS2_LIT_HORIZON_D, lag_bdays=3)
        ic = lab.spearman_ic(al["x"].to_numpy(), al["fwd"].to_numpy())
        assert len(al) > 100
        assert ic > 0.99, (
            f"planted lead not recovered on the lag-3 path (IC={ic}) — "
            f"alignment lags the signal away (P164 family)")

    def test_lag3_alignment_never_reads_before_publication(self):
        # every sample date must be >= obs + 3 business days, and the lag
        # itself must be >= the publication lag (else the path would leak).
        assert min(ext.DGS2_LIT_LAGS_BDAYS) >= lab.PUBLICATION_LAG_BDAYS
        dc = _synthetic_dclose()
        ch = lab.bd_changes(_synthetic_level(), "diff")
        al = lab.align_lead(ch, dc, ext.DGS2_LIT_HORIZON_D, lag_bdays=3)
        assert len(al) > 100
        for sample_date, obs_date in zip(al.index, al["obs_date"]):
            usable = obs_date + pd.offsets.BusinessDay(3)
            assert sample_date >= usable, (
                f"sample {sample_date} reads obs {obs_date} before its lag")


# --------------------------------------------------- level-z causality (P164) --

class TestLevelZCausal:
    def test_perturbing_future_levels_leaves_past_z_bit_identical(self):
        level = _synthetic_level(400)
        cutoff = level.index[250]
        z_orig = ext.level_z(level)
        level2 = level.copy()
        level2.loc[level2.index > cutoff] += 999.0  # violent future perturbation
        z_pert = ext.level_z(level2)
        past_o = z_orig[z_orig.index <= cutoff]
        past_p = z_pert[z_pert.index <= cutoff]
        pd.testing.assert_series_equal(past_o, past_p, check_exact=True)

    def test_perturbing_future_levels_leaves_past_aligned_rows_bit_identical(self):
        # [P164] full construction test THROUGH the imported alignment.
        dc = _synthetic_dclose()
        level = _synthetic_level(300)
        cutoff = level.index[200]
        level2 = level.copy()
        level2.loc[level2.index > cutoff] += 999.0
        al_o = lab.align_lead(ext.level_z(level), dc, 1)
        al_p = lab.align_lead(ext.level_z(level2), dc, 1)
        past_o = al_o[al_o["obs_date"] <= cutoff]
        past_p = al_p[al_p["obs_date"] <= cutoff]
        assert len(past_o) > 20
        pd.testing.assert_frame_equal(past_o, past_p, check_exact=True)

    def test_level_z_fabricates_nothing_on_absent_days(self):
        # absent prints stay absent: z exists only on observation days, and
        # never before min_periods observations (P2).
        level = _synthetic_level(200)
        z = ext.level_z(level, window=100, min_periods=50)
        assert set(z.index).issubset(set(level.index))
        assert len(z) == len(level) - 49  # first 49 obs have no z
        assert not z.isna().any()


# ------------------------------------------------------------------ verdicts --

def _cell(passes, ic):
    return {"passes": passes, "ic": ic, "t": 3.0 if passes else 0.5}


def _beta(t, beta):
    return {"t": t, "beta": beta, "n": 100}


class TestVerdictTruthTableViaImportedRule:
    def test_tradeable_needs_both_eras_same_sign(self):
        v = ext.decide_verdict  # the imported (identical) object
        assert v(_cell(True, 0.2), _cell(True, 0.15),
                 _beta(0.1, 1.0), _beta(0.1, 1.0)) == "TRADEABLE-LEAD"
        assert v(_cell(True, 0.2), _cell(False, 0.15),
                 _beta(0.1, 1.0), _beta(0.1, 1.0)) == "NOISE"
        assert v(_cell(True, 0.2), _cell(True, -0.15),
                 _beta(0.1, 1.0), _beta(0.1, 1.0)) == "NOISE"

    def test_beta_only_and_noise(self):
        v = ext.decide_verdict
        assert v(_cell(False, 0.0), _cell(False, 0.0),
                 _beta(3.0, 1.2), _beta(2.5, 0.8)) == "BETA-ONLY"
        assert v(_cell(False, 0.0), _cell(False, 0.0),
                 _beta(3.0, 1.2), _beta(2.5, -0.8)) == "NOISE"
        # missing beta context (level-z variant with an absent change cell)
        # must degrade to NOISE, never crash or pass.
        assert v(_cell(False, 0.0), _cell(False, 0.0), {}, {}) == "NOISE"

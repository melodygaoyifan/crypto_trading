"""[P392] Tests for training/scripts/macro_factor_lab.py — fast, synthetic,
NO network, NO parquet reads. Pins:
  - the publication-lag convention (a value stamped business day B is never
    readable before B + 2 BUSINESS days at 00:00 UTC; Friday -> Tuesday),
  - the P164 construction test (perturb FUTURE factor values, the past
    alignment must be bit-identical),
  - weekend/absent handling (absent is ABSENT, never a fabricated 0.0),
  - the planted-lead control (a correct alignment recovers IC ~ 1),
  - the shuffled control (|IC| ~ 0),
  - the pre-committed verdict truth table,
  - the required-IC arithmetic round-trip against the P166 formula,
  - the P231 overlap correction (n_eff = n / h).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.scripts import macro_factor_lab as lab  # noqa: E402


# ---------------------------------------------------------------- fixtures --

def _synthetic_dclose(n_days: int = 400, seed: int = 7) -> pd.Series:
    """Calendar-daily crypto closes (24/7), random walk."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    r = rng.normal(0, 0.02, n_days)
    return pd.Series(100.0 * np.exp(np.cumsum(r)), index=idx)


def _synthetic_factor_level(n_days: int = 400, seed: int = 11) -> pd.Series:
    """Business-day factor level (macro does not print on weekends)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=int(n_days * 5 / 7), tz="UTC")
    return pd.Series(np.cumsum(rng.normal(0, 0.05, len(idx))) + 4.0, index=idx)


# ------------------------------------------------------ causality convention --

class TestPublicationLagConvention:
    def test_friday_value_not_usable_before_tuesday(self):
        # 2024-01-05 is a Friday. +2 BUSINESS days = Tuesday 2024-01-09.
        kf = lab.known_from(pd.DatetimeIndex([pd.Timestamp("2024-01-05", tz="UTC")]))
        assert kf[0] == pd.Timestamp("2024-01-09", tz="UTC")
        # A calendar shift(2) would have said Sunday 2024-01-07 — the leak
        # the business-day convention exists to prevent.
        assert kf[0] != pd.Timestamp("2024-01-07", tz="UTC")

    def test_monday_value_usable_wednesday(self):
        kf = lab.known_from(pd.DatetimeIndex([pd.Timestamp("2024-01-08", tz="UTC")]))
        assert kf[0] == pd.Timestamp("2024-01-10", tz="UTC")

    def test_no_sample_reads_an_unpublished_value(self):
        dc = _synthetic_dclose()
        ch = lab.bd_changes(_synthetic_factor_level(), "diff")
        al = lab.align_lead(ch, dc, 1)
        assert len(al) > 100
        for sample_date, obs_date in zip(al.index, al["obs_date"]):
            usable_from = obs_date + pd.offsets.BusinessDay(
                lab.PUBLICATION_LAG_BDAYS)
            assert sample_date >= usable_from, (
                f"sample {sample_date} reads obs {obs_date} before publication")

    def test_perturbing_future_factor_values_leaves_past_alignment_bit_identical(self):
        # [P164] construction test: mutate every factor value after a cutoff;
        # aligned rows sampled before the cutoff must not move at all.
        dc = _synthetic_dclose()
        level = _synthetic_factor_level()
        cutoff = level.index[len(level) // 2]
        ch_orig = lab.bd_changes(level, "diff")
        level2 = level.copy()
        level2.loc[level2.index > cutoff] += 999.0  # violent future perturbation
        ch_pert = lab.bd_changes(level2, "diff")
        al_o = lab.align_lead(ch_orig, dc, 1)
        al_p = lab.align_lead(ch_pert, dc, 1)
        past_o = al_o[al_o["obs_date"] <= cutoff]
        past_p = al_p[al_p["obs_date"] <= cutoff]
        pd.testing.assert_frame_equal(past_o, past_p, check_exact=True)


# ------------------------------------------------------------- absent != 0 --

class TestWeekendAbsence:
    def test_lead_sample_has_no_fabricated_zero_days(self):
        # Every sample row must trace to a REAL factor observation; days with
        # no fresh macro are absent from the sample, not zero-filled.
        dc = _synthetic_dclose()
        ch = lab.bd_changes(_synthetic_factor_level(), "diff")
        al = lab.align_lead(ch, dc, 1)
        obs_set = set(ch.index)
        assert all(o in obs_set for o in al["obs_date"])
        # sample count bounded by the number of factor observations — a
        # calendar-day zero-fill would produce ~7/5 as many rows.
        assert len(al) <= len(ch)

    def test_bd_changes_span_weekend_once_never_zero_fill(self):
        idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in
                                ("2024-01-04", "2024-01-05", "2024-01-08")])
        level = pd.Series([1.0, 2.0, 5.0], index=idx)  # Thu, Fri, Mon
        ch = lab.bd_changes(level, "diff")
        # exactly two changes: Thu->Fri and Fri->Mon (ONE weekend-spanning
        # change), no Saturday/Sunday zeros.
        assert list(ch.to_numpy()) == [1.0, 3.0]
        assert all(ts.dayofweek < 5 for ts in ch.index)


# ------------------------------------------------------------------ controls --

class TestPlantedAndShuffled:
    def test_planted_lead_is_detected_huge(self):
        # A factor that IS tomorrow's return, injected upstream of the full
        # publication-lag alignment, must come out with IC ~ 1 — proving the
        # alignment does not lag a real signal away.
        dc = _synthetic_dclose()
        pf = lab.planted_factor(dc, horizon_days=1)
        al = lab.align_lead(pf, dc, 1)
        ic = lab.spearman_ic(al["x"].to_numpy(), al["fwd"].to_numpy())
        assert len(al) > 100
        assert ic > 0.99, f"planted lead not recovered (IC={ic}) — alignment lags the signal away"

    def test_shuffled_factor_ic_near_zero(self):
        dc = _synthetic_dclose()
        ch = lab.bd_changes(_synthetic_factor_level(), "diff")
        out = lab.shuffled_control(ch, dc, 1, seed=3)
        assert out["n"] > 100
        assert abs(out["ic"]) < 0.15


# ------------------------------------------------------------- arithmetic --

class TestCostBarArithmetic:
    def test_required_ic_roundtrip_p166(self):
        for cost, sigma in ((19.7, 350.0), (29.0, 500.0), (29.0, 900.0)):
            ic_req = lab.required_ic(cost, sigma)
            edge = lab.implied_edge_bps(ic_req, sigma)
            assert math.isclose(edge, lab.EDGE_MARGIN * cost, rel_tol=1e-9), (
                f"edge({ic_req}) = {edge} != {lab.EDGE_MARGIN}*{cost}")

    def test_required_ic_unclearable_cell_is_inf(self):
        # cost so large vs sigma that no IC in [0,1] clears -> inf, so the
        # cell can never PASS (a check that cannot fail must not read as one
        # that passed, P174 — here inverted: a cell that cannot clear must
        # not clear).
        assert lab.required_ic(100.0, 50.0) == float("inf")

    def test_overlap_t_uses_n_eff(self):
        # [P231] t = IC * sqrt(n/h - 1)
        assert math.isclose(lab.overlap_t(0.1, 500, 5.0),
                            0.1 * math.sqrt(100 - 1))
        assert math.isclose(lab.overlap_t(0.1, 500, 1.0),
                            0.1 * math.sqrt(499))
        # h below 1 must never INFLATE n_eff
        assert math.isclose(lab.overlap_t(0.1, 500, 0.5),
                            0.1 * math.sqrt(499))


# ------------------------------------------------------------------ verdicts --

def _cell(passes, ic):
    return {"passes": passes, "ic": ic, "t": 3.0 if passes else 0.5}


def _beta(t, beta):
    return {"t": t, "beta": beta, "n": 100}


class TestVerdictTruthTable:
    def test_tradeable_needs_both_eras(self):
        assert lab.decide_verdict(_cell(True, 0.2), _cell(True, 0.15),
                                  _beta(0.1, 1.0), _beta(0.1, 1.0)) == "TRADEABLE-LEAD"
        # one era only -> not tradeable
        assert lab.decide_verdict(_cell(True, 0.2), _cell(False, 0.15),
                                  _beta(0.1, 1.0), _beta(0.1, 1.0)) == "NOISE"
        assert lab.decide_verdict(_cell(False, 0.2), _cell(True, 0.15),
                                  _beta(0.1, 1.0), _beta(0.1, 1.0)) == "NOISE"

    def test_sign_flip_across_eras_is_not_tradeable(self):
        assert lab.decide_verdict(_cell(True, 0.2), _cell(True, -0.15),
                                  _beta(0.1, 1.0), _beta(0.1, 1.0)) == "NOISE"

    def test_beta_only_requires_both_eras_same_sign(self):
        assert lab.decide_verdict(_cell(False, 0.0), _cell(False, 0.0),
                                  _beta(3.0, 1.2), _beta(2.5, 0.8)) == "BETA-ONLY"
        assert lab.decide_verdict(_cell(False, 0.0), _cell(False, 0.0),
                                  _beta(3.0, 1.2), _beta(2.5, -0.8)) == "NOISE"
        assert lab.decide_verdict(_cell(False, 0.0), _cell(False, 0.0),
                                  _beta(3.0, 1.2), _beta(0.5, 0.8)) == "NOISE"

    def test_tradeable_outranks_beta_only(self):
        assert lab.decide_verdict(_cell(True, 0.2), _cell(True, 0.15),
                                  _beta(3.0, 1.2), _beta(2.5, 0.8)) == "TRADEABLE-LEAD"


# -------------------------------------------------------- contemporaneous --

class TestContemporaneous:
    def test_recovers_a_known_beta(self):
        rng = np.random.default_rng(0)
        level = _synthetic_factor_level(600, seed=5)
        ch = lab.bd_changes(level, "diff")
        # build a crypto close whose business-interval return = 2*change + eps
        idx = pd.date_range(level.index[0], level.index[-1], freq="D", tz="UTC")
        dclose = pd.Series(index=idx, dtype=float)
        dclose.iloc[0] = 100.0
        ch_by_day = ch.reindex(idx)
        px = 100.0
        vals = []
        for d in idx:
            c = ch_by_day.get(d)
            r = (2.0 * float(c) if c is not None and np.isfinite(c) else 0.0)
            r += rng.normal(0, 0.001)
            px *= (1.0 + r)
            vals.append(px)
        dclose = pd.Series(vals, index=idx)
        out = lab.contemporaneous_cell(ch, dclose, idx[0], None)
        assert out["n"] > 100
        assert abs(out["beta"] - 2.0) < 0.2
        assert abs(out["t"]) >= 2.0
        assert out["r2"] > 0.5

    def test_insufficient_sample_is_named_not_zeroed(self):
        dc = _synthetic_dclose(60)
        ch = lab.bd_changes(_synthetic_factor_level(30), "diff")
        out = lab.contemporaneous_cell(
            ch, dc, pd.Timestamp("2030-01-01", tz="UTC"), None)
        assert out.get("note") == "insufficient"
        assert "beta" not in out


# ------------------------------------------------------------------ events --

class TestEventBars:
    def test_event_bar_selection_and_vol_multiple(self):
        idx = pd.date_range("2026-01-20", periods=6 * 30, freq="4h", tz="UTC")
        rng = np.random.default_rng(1)
        r = rng.normal(0, 0.001, len(idx))
        # blow up the 16:00/20:00 bars on one known event day
        for i, ts in enumerate(idx):
            if ts.normalize() == pd.Timestamp("2026-01-28", tz="UTC") and ts.hour in (16, 20):
                r[i] = 0.05
        closes = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
        out = lab.event_bars(closes, ("2026-01-28",))
        assert out["n_event_bars"] == 2
        assert out["vol_multiple"] > 5.0

    def test_no_direction_claim_below_t2(self):
        idx = pd.date_range("2026-01-20", periods=6 * 30, freq="4h", tz="UTC")
        rng = np.random.default_rng(2)
        closes = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.002, len(idx)))),
                           index=idx)
        out = lab.event_bars(closes, ("2026-01-28",))
        # noise events: direction must not be claimed
        assert out["direction_claim"] is False


# ---------------------------------------------------------------- plumbing --

class TestObsParsing:
    def test_missing_fred_prints_are_dropped_not_zero(self):
        obs = [{"date": "2024-01-04", "value": "4.0"},
               {"date": "2024-01-05", "value": "."},
               {"date": "2024-01-08", "value": "4.5"}]
        s = lab.obs_to_series(obs)
        assert len(s) == 2  # '.' is ABSENT, never 0.0 (P2)
        assert 0.0 not in set(s.to_numpy())
        ch = lab.bd_changes(s, "diff")
        # one change spanning the missing print: 4.0 -> 4.5
        assert list(ch.to_numpy()) == [0.5]


class TestRtCostIsSingleSourced:
    def test_rt_cost_equals_twice_the_measured_cde_fee(self):
        """[P392b] RT_COST_BPS is registered in the P382 cost-dict roster on
        the claim that it equals 2 x the measured per-leg CDE fee — pin the
        claim so the roster entry cannot rot (P361)."""
        from core.cde_fees import CDE_FEE_BPS
        for a, rt in lab.RT_COST_BPS.items():
            leg = CDE_FEE_BPS[a]["taker"]   # the sleeve pays taker on a cross
            assert abs(rt - 2.0 * leg) < 0.1, (a, rt, leg)

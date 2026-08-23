"""[P384] Pins for training/scripts/corr_collapse_lab.py — the lab that decides,
on six years of history, whether the sleeve should FLATTEN or HOLD when the
now-conjuncted CORRELATION_COLLAPSE NO_TRADE trigger fires.

Fast: every test here runs on SYNTHETIC frames (no parquet reads). What is
pinned, and why each pin is load-bearing:

  * the lab's corr window is the pipeline's 20 and its threshold / direction
    band are IMPORTED from the live `NoTradeTriggerChecker` — the lab cannot
    drift from the thing it measures (P172/P228);
  * the mask's `all_same` agrees with the REAL checker's `_correlation_conjuncts`
    on a grid of direction triples (strict `>` / `<` on every asset, NaN ->
    not aligned) — a parity test, not a restatement;
  * the FLATTEN counterfactual forces zero on fired bars and is charged the
    exit AND the re-entry through the real `funding_legs_lab.pnl`;
  * the random-mask control is turnover-matched (same fired-bar count, same
    episode-length multiset, non-overlapping);
  * the verdict rule truth table, and that the committed report (if present)
    carries the pre-committed rule string and a verdict consistent with its
    own numbers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from defense.constitution import NoTradeTriggerChecker
from training.scripts import corr_collapse_lab as lab

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "training" / "reports" / "corr_collapse_lab_p384.json"


def _closes(n: int, seed: int = 0, corr_all: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    if corr_all:
        r = rng.normal(0, 0.01, size=n)
        base = 100 * np.exp(np.cumsum(r))
        return pd.DataFrame({"BTC": base, "ETH": base * 0.1, "SOL": base * 0.01}, index=idx)
    r = rng.normal(0, 0.01, size=(n, 3))
    px = 100 * np.exp(np.cumsum(r, axis=0))
    return pd.DataFrame(px, index=idx, columns=["BTC", "ETH", "SOL"])


# ---------------------------------------------------------------------------
# constants: imported from the live checker, window pinned to the pipeline
# ---------------------------------------------------------------------------

class TestConstantsComeFromTheLiveChecker:
    def test_threshold_and_band_are_the_checkers(self):
        assert lab.CORR_THRESHOLD == NoTradeTriggerChecker.CORRELATION_COLLAPSE_THRESHOLD
        assert lab.DIR_MIN == NoTradeTriggerChecker.CORRELATION_ALIGNMENT_DIRECTION_MIN
        assert lab.ASSETS == NoTradeTriggerChecker.CORRELATION_ALIGNMENT_ASSETS

    def test_window_is_the_pipelines_twenty(self):
        assert lab.CORR_WINDOW == 20
        src = (REPO / "data_mgmt" / "market_data_pipeline.py").read_text(encoding="utf-8")
        # the live read: `_btc_c["rets"][-20:]` — if the pipeline's window moves,
        # this lab's CORR_WINDOW must move with it
        assert re.search(r'\["rets"\]\[-20:\]', src), (
            "pipeline no longer takes the last 20 returns — re-derive CORR_WINDOW")

    def test_corr_fallbacks_match_the_pipeline(self):
        src = (REPO / "data_mgmt" / "market_data_pipeline.py").read_text(encoding="utf-8")
        assert "else 0.85" in src and "else 0.80" in src
        assert lab.CORR_FALLBACK[("BTC", "ETH")] == 0.85
        assert lab.CORR_FALLBACK[("BTC", "SOL")] == 0.80
        assert lab.CORR_FALLBACK[("ETH", "SOL")] == 0.80


# ---------------------------------------------------------------------------
# correlation replication
# ---------------------------------------------------------------------------

class TestPipelineCorr:
    def test_nan_before_window_then_defined(self):
        c = _closes(60)
        s = lab.pipeline_corr(c)
        assert s.iloc[:lab.CORR_WINDOW].isna().all()
        assert s.iloc[lab.CORR_WINDOW:].notna().all()

    def test_identical_walks_give_one(self):
        c = _closes(60, corr_all=True)
        s = lab.pipeline_corr(c)
        assert np.allclose(s.iloc[lab.CORR_WINDOW:], 1.0)

    def test_matches_a_hand_computation_on_pct_returns(self):
        c = _closes(80, seed=3)
        s = lab.pipeline_corr(c)
        t = 55
        seg = c.iloc[t - 20: t + 1].to_numpy()
        rets = np.diff(seg, axis=0) / seg[:-1]
        exp = np.mean([np.corrcoef(rets[:, 0], rets[:, 1])[0, 1],
                       np.corrcoef(rets[:, 0], rets[:, 2])[0, 1],
                       np.corrcoef(rets[:, 1], rets[:, 2])[0, 1]])
        assert abs(s.iloc[t] - exp) < 1e-12

    def test_a_window_argument_of_ten_uses_ten_returns(self):
        c = _closes(80, seed=4)
        s = lab.pipeline_corr(c, window=10)
        t = 30
        seg = c.iloc[t - 10: t + 1].to_numpy()
        rets = np.diff(seg, axis=0) / seg[:-1]
        exp = np.mean([np.corrcoef(rets[:, 0], rets[:, 1])[0, 1],
                       np.corrcoef(rets[:, 0], rets[:, 2])[0, 1],
                       np.corrcoef(rets[:, 1], rets[:, 2])[0, 1]])
        assert abs(s.iloc[t] - exp) < 1e-12

    def test_zero_std_leg_takes_the_pipeline_fallbacks(self):
        c = _closes(60, seed=5)
        c["SOL"] = 50.0                      # constant -> zero std on both SOL pairs
        s = lab.pipeline_corr(c)
        t = 40
        seg = c.iloc[t - 20: t + 1].to_numpy()
        rets = np.diff(seg, axis=0) / seg[:-1]
        be = np.corrcoef(rets[:, 0], rets[:, 1])[0, 1]
        assert abs(s.iloc[t] - (be + 0.80 + 0.80) / 3.0) < 1e-12


# ---------------------------------------------------------------------------
# the mask — parity with the live checker's conjunct semantics
# ---------------------------------------------------------------------------

def _checker_all_same(b, e, s):
    chk = NoTradeTriggerChecker()
    all_same, _edge, reason = chk._correlation_conjuncts(
        {"cross_asset_directions": {"BTC": b, "ETH": e, "SOL": s}}, {})
    return all_same, reason


class TestAllSameDirectionParity:
    GRID = [0.0, 0.1, 0.2, 0.21, 0.5, 1.0, -0.2, -0.21, -1.0]

    def test_parity_with_the_real_checker_on_a_grid(self):
        rows, expect = [], []
        for b in self.GRID:
            for e in self.GRID:
                for s in self.GRID:
                    rows.append({"BTC": b, "ETH": e, "SOL": s})
                    got, reason = _checker_all_same(b, e, s)
                    assert reason is None
                    expect.append(bool(got))
        df = pd.DataFrame(rows)
        ours = lab.all_same_direction(df).to_numpy().tolist()
        assert ours == expect

    def test_exact_band_edge_is_not_aligned_strict(self):
        m = lab.DIR_MIN
        df = pd.DataFrame([{"BTC": m, "ETH": 1.0, "SOL": 1.0},
                           {"BTC": -m, "ETH": -1.0, "SOL": -1.0}])
        assert not lab.all_same_direction(df).any()
        got, _ = _checker_all_same(m, 1.0, 1.0)
        assert got is False

    def test_nan_direction_is_not_aligned(self):
        df = pd.DataFrame([{"BTC": 1.0, "ETH": np.nan, "SOL": 1.0}])
        assert not lab.all_same_direction(df).any()


class TestFireMask:
    def _dirs(self, n, val=1.0):
        return pd.DataFrame({a: [val] * n for a in lab.ASSETS})

    def test_fires_at_and_above_threshold_only_when_aligned(self):
        thr = lab.CORR_THRESHOLD
        corr = pd.Series([thr - 0.001, thr, thr + 0.05, np.nan])
        d = self._dirs(4)
        m = lab.fire_mask(corr, d)
        assert m.tolist() == [False, True, True, False]

    def test_no_alignment_no_fire(self):
        thr = lab.CORR_THRESHOLD
        corr = pd.Series([thr + 0.05] * 3)
        d = pd.DataFrame({"BTC": [1.0, 1.0, 0.0], "ETH": [1.0, -1.0, 1.0], "SOL": [1.0, 1.0, 1.0]})
        assert lab.fire_mask(corr, d).tolist() == [True, False, False]

    def test_all_short_also_fires(self):
        corr = pd.Series([lab.CORR_THRESHOLD + 0.01])
        assert lab.fire_mask(corr, self._dirs(1, -1.0)).tolist() == [True]


# ---------------------------------------------------------------------------
# episodes / flatten / control / bootstrap
# ---------------------------------------------------------------------------

class TestEpisodesAndFlatten:
    def test_episodes(self):
        assert lab.episodes([False, True, True, False, True]) == [(1, 2), (4, 1)]
        assert lab.episodes([]) == []
        assert lab.episodes([True, True]) == [(0, 2)]

    def test_flatten_forces_zero_on_fired_bars_only(self):
        idx = pd.RangeIndex(6)
        book = pd.Series([1.0, 1.0, -1.0, 1.0, 0.0, 1.0], index=idx)
        mask = pd.Series([False, True, True, False, False, False], index=idx)
        out = lab.flatten_positions(book, mask)
        assert out.tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]

    def test_flatten_counterfactual_pays_exit_and_reentry_through_real_pnl(self):
        """The real funding_legs_lab.pnl charges turnover per transition, so a
        one-episode flatten on a held long costs exactly two legs more than
        HOLD, and its gross differs by exactly the book's return on the fired
        bars."""
        from training.funding_legs_lab import per_leg_cost_series, pnl
        n = 40
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        rng = np.random.default_rng(11)
        closes = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
        funding = pd.Series(0.0, index=pd.date_range("2023-12-01", periods=60, freq="D", tz="UTC"))
        book = pd.Series(1.0, index=idx)
        mask = pd.Series(False, index=idx)
        mask.iloc[10:13] = True
        hold = pnl("BTC", book, closes, funding)
        flat = pnl("BTC", lab.flatten_positions(book, mask), closes, funding)
        leg = float(per_leg_cost_series("BTC", closes).iloc[0])
        extra_cost = float(flat["cost"].sum() - hold["cost"].sum())
        assert abs(extra_cost - 2 * leg) < 1e-12, "exit + re-entry must be charged"
        ret = closes.pct_change().shift(-1)
        gross_diff = float(flat["gross"].sum() - hold["gross"].sum())
        assert abs(gross_diff + ret.iloc[10:13].sum()) < 1e-12
        # and net diff = -(fired-bar return) - 2 legs (carry is zero here)
        net_diff = float(flat["net"].sum() - hold["net"].sum())
        assert abs(net_diff - (gross_diff - 2 * leg)) < 1e-12


class TestRandomMask:
    def test_turnover_matched_same_count_same_lengths_non_overlapping(self):
        rng = np.random.default_rng(1)
        lengths = [5, 2, 1, 3]
        for _ in range(20):
            m = lab.random_mask(200, lengths, rng)
            assert int(m.sum()) == sum(lengths)
            got = sorted(L for _, L in lab.episodes(m))
            assert got == sorted(lengths), "episodes merged/overlapped — control is not turnover-matched"

    def test_different_seeds_move(self):
        a = lab.random_mask(300, [4, 4, 2], np.random.default_rng(1))
        b = lab.random_mask(300, [4, 4, 2], np.random.default_rng(2))
        assert a.tolist() != b.tolist()

    def test_empty(self):
        assert not lab.random_mask(50, [], np.random.default_rng(0)).any()

    def test_too_long_refuses(self):
        with pytest.raises(ValueError):
            lab.random_mask(5, [3, 3], np.random.default_rng(0))


class TestBootstrap:
    def test_constant_series_gives_its_sum(self):
        d = np.full(900, 0.001)
        lo, hi = lab.block_bootstrap_sum_ci(d, block=90, n=50, seed=1)
        assert abs(lo - 0.9) < 1e-9 and abs(hi - 0.9) < 1e-9

    def test_short_series_is_nan(self):
        lo, hi = lab.block_bootstrap_sum_ci(np.zeros(100), block=90)
        assert lo != lo and hi != hi

    def test_noisy_series_ci_brackets_zero_mean(self):
        d = np.random.default_rng(3).normal(0, 1e-3, 3000)
        lo, hi = lab.block_bootstrap_sum_ci(d, block=90, n=200, seed=2)
        assert lo < 0 < hi


# ---------------------------------------------------------------------------
# verdict rule — truth table
# ---------------------------------------------------------------------------

class TestVerdictRule:
    def test_earns_only_when_both_eras_positive_and_beat_p90(self):
        v, b = lab.decide_verdict({"design": 0.02, "pre_design": 0.01},
                                  {"design": 0.005, "pre_design": 0.002}, fired_bars=10)
        assert v == "FLATTEN_EARNS" and b == []

    @pytest.mark.parametrize("diff,p90", [
        ({"design": -0.01, "pre_design": 0.01}, {"design": -0.02, "pre_design": 0.0}),   # design <= 0
        ({"design": 0.01, "pre_design": 0.0}, {"design": 0.0, "pre_design": -0.01}),     # pre_design == 0
        ({"design": 0.01, "pre_design": 0.01}, {"design": 0.02, "pre_design": 0.0}),     # design below p90
        ({"design": 0.01, "pre_design": 0.01}, {"design": 0.0, "pre_design": 0.01}),     # pre at p90 (must EXCEED)
        ({"design": 0.01}, {"design": 0.0}),                                              # pre_design missing
    ])
    def test_hold_stands_otherwise(self, diff, p90):
        v, b = lab.decide_verdict(diff, p90, fired_bars=10)
        assert v == "HOLD_STANDS" and b

    def test_never_fires_is_unreachable(self):
        v, b = lab.decide_verdict({}, {}, fired_bars=0)
        assert v == "TRIGGER_UNREACHABLE"

    def test_validation_era_does_not_decide(self):
        v, _ = lab.decide_verdict({"design": 0.02, "pre_design": 0.01, "validation": -5.0},
                                  {"design": 0.0, "pre_design": 0.0, "validation": 0.0}, 10)
        assert v == "FLATTEN_EARNS"

    def test_rule_string_names_its_clauses(self):
        r = lab.VERDICT_RULE
        for w in ("DESIGN", "PRE-DESIGN", "90th percentile", "HOLD STANDS", "UNREACHABLE"):
            assert w in r


class TestEraMasks:
    def test_bands_are_the_labs(self):
        from training.funding_legs_lab import ERAS
        m = lab.era_masks(10000)
        assert set(m) == set(ERAS)
        assert m["pre_design"][799] is np.bool_(False) and m["pre_design"][800] and not m["pre_design"][3000]
        assert m["design"][3000] and m["design"][9099] and not m["design"][9100]
        assert m["validation"][9100] and m["validation"][9999]


# ---------------------------------------------------------------------------
# the committed report
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REPORT.exists(), reason="report is operator-local (built by the lab)")
class TestReport:
    def test_carries_the_precommitted_rule_and_a_consistent_verdict(self):
        d = json.loads(REPORT.read_text(encoding="utf-8"))
        assert d["verdict_rule"] == lab.VERDICT_RULE
        assert d["verdict"] in ("FLATTEN_EARNS", "HOLD_STANDS", "TRIGGER_UNREACHABLE")
        if d["verdict"] == "TRIGGER_UNREACHABLE":
            assert d["fired_bars"] == 0
            return
        # anti-vacuity: the mask fired, and the control moved
        assert d["fired_bars"] >= 1
        c = d["control"]["overall"]
        assert c["p90_pct"] != c["p10_pct"]
        # the verdict must be re-derivable from the report's own numbers
        diff = {e: v["flatten_minus_hold_pct"] / 100.0 for e, v in d["sleeve"]["by_era"].items()}
        p90 = {e: v["p90_pct"] / 100.0 for e, v in d["control"]["by_era"].items()}
        v, _ = lab.decide_verdict(diff, p90, d["fired_bars"])
        assert v == d["verdict"]
        assert d["provenance"]["corr_threshold"] == lab.CORR_THRESHOLD
        assert d["provenance"]["direction_min"] == lab.DIR_MIN
        assert d["provenance"]["corr_window"] == lab.CORR_WINDOW

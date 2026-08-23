"""[P384] `liq_imbalance` is a causal TRAILING-24H series built from the 4h
CoinGlass liquidation archive — the same quantity the live feed serves.

Background: P382 closed the same-day lookahead in `liq_imbalance` by
shifting the 1d archive one day, which left a recorded train/serve skew:
the TRAINED feature was day D-1's COMPLETED (long-short)/total while the
LIVE feature (`data_mgmt/feeds/coinglass_feed.py` -> market_data
["liquidation_imbalance"] -> main.py -> DRL obs builder) is CoinGlass's
trailing-24h (long-short)/total at fetch time (P214 class). The 4h archive
(rows stamped at bucket OPEN, hours 0/4/8/12/16/20; P382 verified
sum-of-six-4h-rows == the 1d row to corr 1.0000) lets the TRAINING side
compute the trailing-24h imbalance at every 4H bar causally:

  * a 4H parquet bar stamped t (kline OPEN) has its close at t+4h, and every
    other feature on that row is as-of that close;
  * the 4h row stamped t is bucket [t, t+4h), complete at t+4h;
  * so the trailing-24h window at the row's close is the six buckets stamped
    t-20h .. t inclusive — a rolling six-bucket sum stamped at the bucket
    open t, merge_asof'd backward at bar t, picks the row stamped t and is
    causal as-of the row's close by construction;
  * live fetches at ~t+4h+10min, i.e. the same six buckets minus the first
    ~10 minutes of bucket t-20h plus ~10 minutes of the NEW bucket — the
    honest residual.

These tests pin that BEHAVIOURALLY (by calling the rebuild's pure seam and
its real merge function), never by grepping source:

  * the P164 construction test: perturb every row AFTER t violently and the
    series at <= t is bit-identical;
  * window semantics: exactly six buckets INCLUDING the same-stamp one;
    min_periods=6 -> NaN before; a gap inside the window -> NaN;
  * total == 0 -> NaN (absence), never 0.0 ("balanced");
  * merge precedence: 4h beats 1d when present, 1d-shifted when absent, and
    the funding/oi/futures columns are BYTE-IDENTICAL in both cases;
  * the merged bar t carries the 4h row stamped t (same stamp, not t-4h);
  * the arithmetic is the live feed's formula, (long-short)/total clipped
    [-1, 1], pinned by number (nothing is imported from the live feed);
  * the loader routes through the seam and a missing file yields an EMPTY
    frame (the fallback trigger), not a crash;
  * a PREMISE/RESULT pin on the real archive + rebuilt parquets (skips
    loudly when the gitignored files are absent, P252b): the parquet's
    liq_imbalance equals the 4h-derived series at every in-coverage bar.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
CG_DIR = REPO / "training" / "training_data" / "coinglass_history"
DRL_DIR = REPO / "training" / "training_data" / "drl_training"


def _load_rebuild_module():
    """Exec rebuild_pipeline.py in isolation from the repo-root `scripts`
    package (same trick as tests/test_p382_coinglass_day_stamp_causal.py —
    snapshot and restore the WHOLE sys.path, P382c)."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == "scripts" or k.startswith("scripts.")}
    training_dir = str(REPO / "training")
    saved_path = list(sys.path)
    sys.path.insert(0, training_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            "rebuild_pipeline_under_test_p384",
            REPO / "training" / "scripts" / "rebuild_pipeline.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k in [k for k in list(sys.modules)
                  if k == "scripts" or k.startswith("scripts.")]:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path


@pytest.fixture(scope="module")
def rp():
    return _load_rebuild_module()


# ---------------------------------------------------------------- fixtures
N_DAYS = 40
N_4H = N_DAYS * 6
T = 150          # the perturbation boundary (row index into the 4h archive)
START = "2026-01-01"


def _grid_4h(n=N_4H, start=START):
    # Bucket-OPEN stamps on the 4h grid, exactly the archive's convention.
    return pd.date_range(start, periods=n, freq="4h", tz="UTC")


def _synthetic_4h(seed=11, n=N_4H):
    rng = np.random.default_rng(seed)
    long_l = rng.uniform(1e5, 5e6, n)
    short_l = rng.uniform(1e5, 5e6, n)
    return pd.DataFrame({
        "timestamp": _grid_4h(n),
        "long_liq_usd": long_l,
        "short_liq_usd": short_l,
        "total_liq_usd": long_l + short_l,
        "liq_imbalance": (long_l - short_l) / (long_l + short_l),
        "asset": "BTC",
    })


def _daily_index(n=N_DAYS):
    return pd.date_range(START, periods=n, freq="D", tz="UTC")


def _synthetic_daily(seed=7):
    """The P382 fixture shape: day-OPEN stamped 1d frames for funding / oi /
    liquidation, plus a day-CLOSE stamped futures frame (P281)."""
    rng = np.random.default_rng(seed)
    ts = _daily_index()
    funding = pd.DataFrame({
        "timestamp": ts,
        "funding_open": rng.normal(0, 1e-4, N_DAYS),
        "funding_high": rng.normal(0, 1e-4, N_DAYS),
        "funding_low": rng.normal(0, 1e-4, N_DAYS),
        "funding_close": rng.normal(0, 1e-4, N_DAYS),
    })
    oi_close = 1e9 * np.cumprod(1.0 + rng.normal(0, 0.01, N_DAYS))
    oi = pd.DataFrame({
        "timestamp": ts, "oi_open": oi_close, "oi_high": oi_close,
        "oi_low": oi_close, "oi_close": oi_close,
    })
    long_l = rng.uniform(1e6, 5e7, N_DAYS)
    short_l = rng.uniform(1e6, 5e7, N_DAYS)
    liq1d = pd.DataFrame({
        "timestamp": ts, "long_liq_usd": long_l, "short_liq_usd": short_l,
        "total_liq_usd": long_l + short_l,
        "liq_imbalance": (long_l - short_l) / (long_l + short_l),
    })
    fut = pd.DataFrame({
        "timestamp": ts,
        "marketorder_volume": np.linspace(1.0, 2.0, N_DAYS),
        "marketorder_volume_from": np.linspace(1.0, 2.0, N_DAYS),
        "tradecount": np.linspace(100.0, 200.0, N_DAYS),
    })
    return funding, oi, liq1d, fut


def _bars_4h(n=N_4H):
    ts = _grid_4h(n)
    return pd.DataFrame({"timestamp": ts, "close": np.linspace(100, 200, n)})


def _write_sources(tmp_path, funding, oi, liq1d, fut, liq4h=None, asset="BTC"):
    cg = tmp_path / "cg"
    fu = tmp_path / "fut"
    cg.mkdir(parents=True, exist_ok=True)
    fu.mkdir(parents=True, exist_ok=True)
    funding.to_parquet(cg / f"{asset}_funding_1d.parquet", index=False)
    oi.to_parquet(cg / f"{asset}_oi_1d.parquet", index=False)
    liq1d.to_parquet(cg / f"{asset}_liquidation_1d.parquet", index=False)
    fut.to_parquet(fu / f"{asset}_futures_daily.parquet", index=False)
    if liq4h is not None:
        liq4h.to_parquet(cg / f"{asset}_liquidation_4h.parquet", index=False)
    return cg, fu


def _merge(rp, monkeypatch, tmp_path, liq4h, daily=None):
    funding, oi, liq1d, fut = daily if daily is not None else _synthetic_daily()
    cg, fu = _write_sources(tmp_path, funding, oi, liq1d, fut, liq4h=liq4h)
    monkeypatch.setattr(rp, "COINGLASS_DIR", cg)
    monkeypatch.setattr(rp, "FUTURES_DIR", fu)
    return rp.merge_external_data(_bars_4h(), "BTC")


def _manual_trailing(liq4h, i, window=6):
    """Independent oracle: (sum long - sum short)/sum total over rows
    i-window+1 .. i inclusive — the live feed's formula over six buckets."""
    w = liq4h.iloc[i - window + 1: i + 1]
    tot = w["total_liq_usd"].sum()
    if tot <= 0:
        return np.nan
    return float(np.clip((w["long_liq_usd"].sum() - w["short_liq_usd"].sum()) / tot, -1, 1))


# ---------------------------------------------------------------- tests
class TestConstructionCausality:
    """P164-style: perturb the FUTURE and require the past to be bit-identical."""

    def test_perturbing_every_row_after_T_leaves_rows_up_to_T_bit_identical(self, rp):
        arch = _synthetic_4h()
        base = rp._derive_liq_trailing_24h(arch)
        pert_in = arch.copy()
        n = len(pert_in)
        # Violent, in-range perturbations of EVERY row strictly after T.
        pert_in.loc[T + 1:, "long_liq_usd"] = 9e9
        pert_in.loc[T + 1:, "short_liq_usd"] = 1.0
        pert_in.loc[T + 1:, "total_liq_usd"] = 9e9 + 1.0
        pert_in.loc[T + 1:, "liq_imbalance"] = 0.999
        pert = rp._derive_liq_trailing_24h(pert_in)
        assert len(base) == len(pert) == n
        assert np.array_equal(base["timestamp"].to_numpy(), pert["timestamp"].to_numpy())
        b = base["liq_imbalance"].to_numpy()
        p = pert["liq_imbalance"].to_numpy()
        assert np.array_equal(b[: T + 1], p[: T + 1], equal_nan=True), \
            "a row at or before T moved when rows AFTER T were perturbed (look-ahead)"
        # and the future did react — row T+1's window now contains a
        # perturbed bucket, so the series is not simply constant
        assert not np.isclose(b[T + 1], p[T + 1])
        # from T+6 on every window is six perturbed buckets: (6*9e9-6)/(6*9e9+6) ~ 1.0
        assert np.allclose(p[T + 6:], 1.0, atol=1e-6)

    def test_perturbing_row_T_itself_moves_row_T(self, rp):
        """The SAME-stamp bucket is IN the window (it is complete at the
        row's close) — perturbing row T must move the series at T, and at
        nothing before T."""
        arch = _synthetic_4h()
        base = rp._derive_liq_trailing_24h(arch)["liq_imbalance"].to_numpy()
        pert_in = arch.copy()
        pert_in.loc[T, ["long_liq_usd", "short_liq_usd", "total_liq_usd"]] = [9e9, 1.0, 9e9 + 1.0]
        pert = rp._derive_liq_trailing_24h(pert_in)["liq_imbalance"].to_numpy()
        assert np.array_equal(base[:T], pert[:T], equal_nan=True)
        assert not np.isclose(base[T], pert[T])
        # and exactly six rows (T .. T+5) see the perturbed bucket
        moved = ~np.isclose(base, pert, equal_nan=True)
        assert moved.nonzero()[0].tolist() == list(range(T, T + 6))


class TestWindowSemantics:
    def test_value_is_exactly_the_six_bucket_trailing_ratio(self, rp):
        arch = _synthetic_4h()
        out = rp._derive_liq_trailing_24h(arch)
        assert len(out) == len(arch)
        for i in (5, 6, 37, T, len(arch) - 1):
            assert out.loc[i, "liq_imbalance"] == pytest.approx(_manual_trailing(arch, i), abs=1e-12), i
        # NOT five buckets, NOT seven, and NOT the mean of six per-bucket ratios
        i = T
        five = float((arch.iloc[i-4:i+1]["long_liq_usd"].sum() - arch.iloc[i-4:i+1]["short_liq_usd"].sum())
                     / arch.iloc[i-4:i+1]["total_liq_usd"].sum())
        seven = float((arch.iloc[i-6:i+1]["long_liq_usd"].sum() - arch.iloc[i-6:i+1]["short_liq_usd"].sum())
                      / arch.iloc[i-6:i+1]["total_liq_usd"].sum())
        mean6 = float(arch.iloc[i-5:i+1]["liq_imbalance"].mean())
        for wrong in (five, seven, mean6):
            assert not np.isclose(out.loc[i, "liq_imbalance"], wrong, atol=1e-9)
        assert rp.LIQ_4H_WINDOW == 6

    def test_first_five_rows_are_nan_and_the_sixth_is_not(self, rp):
        arch = _synthetic_4h()
        out = rp._derive_liq_trailing_24h(arch)
        assert out["liq_imbalance"].iloc[:5].isna().all()
        assert not np.isnan(out["liq_imbalance"].iloc[5])
        assert out["liq_imbalance"].iloc[5:].notna().all()

    def test_a_gap_in_the_archive_yields_nan_not_a_bridged_window(self, rp):
        """Six non-adjacent buckets are not a trailing 24h. Drop one row;
        the six windows that would have contained it must be NaN, and the
        output must still be on the full 4h grid (the gap row is present,
        as NaN)."""
        arch = _synthetic_4h()
        gap = T
        arch_gap = arch.drop(index=gap).reset_index(drop=True)
        out = rp._derive_liq_trailing_24h(arch_gap)
        assert len(out) == len(arch)                      # grid restored
        stamps = out["timestamp"].to_numpy()
        assert np.array_equal(stamps, arch["timestamp"].to_numpy())
        nan_rows = out.index[out["liq_imbalance"].isna()].tolist()
        assert nan_rows == list(range(0, 5)) + list(range(gap, gap + 6))

    def test_zero_total_window_is_nan_not_balanced_zero(self, rp):
        arch = _synthetic_4h()
        # make eleven consecutive buckets zero -> windows fully inside are total==0
        arch.loc[T:T + 10, ["long_liq_usd", "short_liq_usd", "total_liq_usd"]] = 0.0
        out = rp._derive_liq_trailing_24h(arch)
        fully_zero = list(range(T + 5, T + 11))   # windows [T..T+5] .. [T+5..T+10]
        for i in fully_zero:
            assert np.isnan(out.loc[i, "liq_imbalance"]), \
                f"row {i}: total==0 must be NaN (absence), not 0.0 (balanced)"
        # a window straddling the zero run still has a value
        assert not np.isnan(out.loc[T + 2, "liq_imbalance"])
        assert not np.isnan(out.loc[T + 11, "liq_imbalance"])

    def test_output_is_clipped_and_stamped_at_bucket_open(self, rp):
        arch = _synthetic_4h()
        # total deliberately smaller than long - short so the raw ratio > 1
        arch.loc[T:T + 5, "long_liq_usd"] = 5e6
        arch.loc[T:T + 5, "short_liq_usd"] = 0.0
        arch.loc[T:T + 5, "total_liq_usd"] = 1e6
        out = rp._derive_liq_trailing_24h(arch)
        assert out.loc[T + 5, "liq_imbalance"] == 1.0
        assert out["liq_imbalance"].dropna().between(-1.0, 1.0).all()
        # stamps are the archive's bucket-OPEN stamps, unchanged
        assert np.array_equal(out["timestamp"].to_numpy(), arch["timestamp"].to_numpy())

    def test_empty_input_is_an_empty_frame(self, rp):
        out = rp._derive_liq_trailing_24h(pd.DataFrame())
        assert list(out.columns) == ["timestamp", "liq_imbalance"]
        assert len(out) == 0


class TestLiveFormulaParity:
    """The live feed computes (long_24h - short_24h) / total_24h, clipped
    [-1, 1] (data_mgmt/feeds/coinglass_feed.py, 'Positive = more longs
    liquidated'). Nothing is imported from the feed — the formula is pinned
    here BY NUMBER on a hand-built window, so a sign flip or a mean-of-
    ratios on either side fails."""

    def test_hand_computed_window_matches(self, rp):
        longs = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])   # sum 210
        shorts = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])         # sum 30
        arch = pd.DataFrame({
            "timestamp": _grid_4h(6),
            "long_liq_usd": longs, "short_liq_usd": shorts,
            "total_liq_usd": longs + shorts,
        })
        out = rp._derive_liq_trailing_24h(arch)
        # live formula: (210 - 30) / 240 = 0.75, positive = more longs liquidated
        assert out.loc[5, "liq_imbalance"] == pytest.approx(0.75)
        # and not the mean of per-bucket ratios (which would be ~0.54)
        per_bucket = ((longs - shorts) / (longs + shorts)).mean()
        assert not np.isclose(out.loc[5, "liq_imbalance"], per_bucket, atol=1e-3)


class TestMergePrecedence:
    FUND_OI_FUT = ("funding_rate_zscore", "oi_change_5d",
                   "taker_ratio_zscore", "tradecount_zscore", "taker_vol_momentum")

    def test_4h_series_wins_when_the_archive_exists(self, rp, monkeypatch, tmp_path):
        arch = _synthetic_4h()
        out = _merge(rp, monkeypatch, tmp_path, liq4h=arch)
        series = rp._derive_liq_trailing_24h(arch)
        expected = series["liq_imbalance"].fillna(0.0).to_numpy()
        # bars are on the same grid as the archive: bar t carries row t
        assert np.allclose(out["liq_imbalance"].to_numpy(), expected, atol=1e-12)
        # ... and NOT the 1d-shifted series (they differ on essentially every bar)
        funding, oi, liq1d, fut = _synthetic_daily()
        shifted = rp._derive_coinglass_daily(funding, oi, liq1d)
        bars = out[["timestamp"]].copy()
        m = pd.merge_asof(bars, shifted[["timestamp", "liq_imbalance"]],
                          on="timestamp", direction="backward")
        one_d = m["liq_imbalance"].fillna(0.0).to_numpy()
        assert not np.allclose(out["liq_imbalance"].to_numpy(), one_d)

    def test_1d_shifted_fallback_when_the_archive_is_absent(self, rp, monkeypatch, tmp_path):
        out = _merge(rp, monkeypatch, tmp_path, liq4h=None)
        funding, oi, liq1d, fut = _synthetic_daily()
        # the P382 semantics: every bar of day k carries source row k-1
        out = out.assign(day=out["timestamp"].dt.floor("D"))
        for k in (10, 20, N_DAYS - 1):
            vals = out[out["day"] == _daily_index()[k]]["liq_imbalance"].to_numpy()
            assert len(vals) == 6
            assert np.allclose(vals, float(np.clip(liq1d.loc[k - 1, "liq_imbalance"], -1, 1)))

    def test_funding_oi_futures_columns_are_byte_identical_either_way(self, rp, monkeypatch, tmp_path):
        arch = _synthetic_4h()
        with_4h = _merge(rp, monkeypatch, tmp_path / "a", liq4h=arch)
        without = _merge(rp, monkeypatch, tmp_path / "b", liq4h=None)
        assert list(with_4h.columns) == list(without.columns)     # same order, too
        for c in self.FUND_OI_FUT + ("timestamp", "close"):
            assert with_4h[c].equals(without[c]), f"{c} changed between the 4h and 1d paths"
        # and the two liq columns DO differ (the precedence is real)
        assert not with_4h["liq_imbalance"].equals(without["liq_imbalance"])

    def test_has_external_data_is_one_where_any_feature_is_present(self, rp, monkeypatch, tmp_path):
        arch = _synthetic_4h()
        out = _merge(rp, monkeypatch, tmp_path, liq4h=arch)
        ext = [c for c in rp.EXTERNAL_FEATURE_COLS if c != "has_external_data"]
        # recompute the flag from the merged frame's own NaN pattern: the
        # rebuild fills NaN with 0.0 AFTER setting the flag, so re-derive
        # from a fresh merge of the unfilled pieces instead: any bar with at
        # least one present feature is flagged, none without.
        assert set(out["has_external_data"].unique()) <= {0.0, 1.0}
        # day 0 bars: funding/oi/liq1d are shifted (NaN) and the futures
        # z-scores need 10 days / 5 days of history (NaN) — so the ONLY
        # present feature is the 4h liq from its sixth bucket: [0,0,0,0,0,1]
        assert out["has_external_data"].iloc[:6].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        # once funding/oi/futures exist the flag is 1 regardless of liq
        assert (out["has_external_data"].iloc[12 * 6:] == 1.0).all()
        # a frame with NOTHING but the 4h liq: flag follows the liq NaNs exactly
        funding, oi, liq1d, fut = _synthetic_daily()
        empty = funding.iloc[0:0]
        cg, fu = _write_sources(tmp_path / "only_liq", empty, oi.iloc[0:0], liq1d.iloc[0:0], fut, liq4h=arch)
        # futures is mandatory (P281 refusal) -> make it all-NaN instead of absent
        fut_nan = fut.copy()
        for c in ("marketorder_volume", "marketorder_volume_from", "tradecount"):
            fut_nan[c] = np.nan
        fut_nan.to_parquet(fu / "BTC_futures_daily.parquet", index=False)
        monkeypatch.setattr(rp, "COINGLASS_DIR", cg)
        monkeypatch.setattr(rp, "FUTURES_DIR", fu)
        out2 = rp.merge_external_data(_bars_4h(), "BTC")
        series = rp._derive_liq_trailing_24h(arch)["liq_imbalance"]
        assert np.array_equal(out2["has_external_data"].to_numpy(),
                              series.notna().astype(float).to_numpy())
        assert out2["has_external_data"].iloc[:5].sum() == 0.0
        assert out2["has_external_data"].iloc[5:].all()

    def test_bar_t_carries_the_row_stamped_t_not_t_minus_4h(self, rp, monkeypatch, tmp_path):
        """The alignment claim itself: the same-stamp row (bucket [t,t+4h),
        complete at the bar's close) is what the bar reads."""
        arch = _synthetic_4h()
        out = _merge(rp, monkeypatch, tmp_path, liq4h=arch)
        series = rp._derive_liq_trailing_24h(arch).set_index("timestamp")["liq_imbalance"]
        for i in (T, T + 1, N_4H - 1):
            t = out.loc[i, "timestamp"]
            assert out.loc[i, "liq_imbalance"] == pytest.approx(series.loc[t], abs=1e-12)
            assert not np.isclose(out.loc[i, "liq_imbalance"], series.loc[t - pd.Timedelta(hours=4)], atol=1e-9)

    def test_a_bar_more_than_one_bucket_past_the_archive_end_reads_absent(self, rp, monkeypatch, tmp_path):
        """tolerance = one bucket (inclusive): the bar ONE bucket past the
        archive's last row may still read it (stale by one completed bucket,
        still causal); a bar further past must NOT inherit it (P287's
        staleness lesson, at 4h scale)."""
        arch = _synthetic_4h(n=N_4H - 12)     # archive ends 48h before the bars do
        out = _merge(rp, monkeypatch, tmp_path, liq4h=arch)
        series = rp._derive_liq_trailing_24h(arch)["liq_imbalance"]
        last = N_4H - 13                      # index of the archive's last row / bar
        assert out.loc[last, "liq_imbalance"] == pytest.approx(series.iloc[-1])
        assert out.loc[last + 1, "liq_imbalance"] == pytest.approx(series.iloc[-1])   # within 4h
        assert (out["liq_imbalance"].iloc[last + 2:] == 0.0).all()                      # beyond: absent


class TestLoaderWiring:
    def test_loader_routes_through_the_pure_seam(self, rp, monkeypatch, tmp_path):
        arch = _synthetic_4h()
        funding, oi, liq1d, fut = _synthetic_daily()
        cg, _ = _write_sources(tmp_path, funding, oi, liq1d, fut, liq4h=arch)
        monkeypatch.setattr(rp, "COINGLASS_DIR", cg)
        via_loader = rp._load_coinglass_liq_4h("BTC")
        direct = rp._derive_liq_trailing_24h(pd.read_parquet(cg / "BTC_liquidation_4h.parquet"))
        pd.testing.assert_frame_equal(via_loader, direct)

    def test_missing_archive_yields_an_empty_frame_not_a_crash(self, rp, monkeypatch, tmp_path):
        funding, oi, liq1d, fut = _synthetic_daily()
        cg, _ = _write_sources(tmp_path, funding, oi, liq1d, fut, liq4h=None)
        monkeypatch.setattr(rp, "COINGLASS_DIR", cg)
        out = rp._load_coinglass_liq_4h("BTC")
        assert len(out) == 0
        assert list(out.columns) == ["timestamp", "liq_imbalance"]

    def test_liq_imbalance_is_not_a_gmm_input(self, rp):
        """P215/P253b: --skip-gmm is the correct rebuild mode for this change
        BECAUSE liq_imbalance is not among the GMM features. If it ever
        becomes one, the GMM must be refit in the same step."""
        assert "liq_imbalance" not in rp.GMM_FEATURE_COLS
        assert "liq_imbalance" in rp.EXTERNAL_FEATURE_COLS


class TestPremiseAndResultOnTheRealData:
    """Skips loudly when the gitignored parquets are absent (P252b)."""

    @pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
    def test_rebuilt_parquet_carries_the_4h_trailing_series(self, rp, asset):
        arch_p = CG_DIR / f"{asset}_liquidation_4h.parquet"
        pq_p = DRL_DIR / f"{asset}_4H_full.parquet"
        if not (arch_p.exists() and pq_p.exists()):
            pytest.skip(f"[P384] {asset} archive/parquet not on this machine "
                        f"(gitignored, operator-local) — result not re-measured here")
        arch = pd.read_parquet(arch_p)
        arch["timestamp"] = pd.to_datetime(arch["timestamp"], utc=True)
        # premise: bucket-open stamps on the 4h grid, contiguous
        assert set(arch["timestamp"].dt.hour.unique()) <= {0, 4, 8, 12, 16, 20}
        assert (arch["timestamp"].diff().dropna() == pd.Timedelta(hours=4)).all()
        series = rp._derive_liq_trailing_24h(arch).set_index("timestamp")["liq_imbalance"]
        pq = pd.read_parquet(pq_p, columns=["timestamp", "liq_imbalance"])
        ts = pd.to_datetime(pq["timestamp"])
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        pq = pq.assign(timestamp=ts).set_index("timestamp")
        common = pq.index.intersection(series.dropna().index)
        assert len(common) > 500, "archive/parquet overlap unexpectedly small"
        assert np.allclose(pq.loc[common, "liq_imbalance"].to_numpy(),
                           series.loc[common].to_numpy(), atol=1e-9), \
            f"{asset}: the parquet's liq_imbalance is not the 4h trailing-24h series — rebuild it"
        # and in-coverage values are genuinely not the day-shifted 1d ones:
        # within a UTC day the trailing series VARIES bar to bar
        day = pq.loc[common].groupby(pq.loc[common].index.floor("D"))["liq_imbalance"].nunique()
        assert (day[day.index.isin(common.floor("D"))] > 1).mean() > 0.9

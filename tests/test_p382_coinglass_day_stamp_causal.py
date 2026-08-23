"""[P382] CoinGlass daily features are shifted one day BEFORE the merge_asof —
for ALL three sources, not just funding.

CoinGlass `*_1d.parquet` rows are stamped at the day's OPEN (00:00 UTC) while
their content is the day's CLOSE: `oi_close` on row D equals the 4h file's
20:00 close of day D exactly, and `liq_*` on row D equals the SUM of day D's
six 4h rows (measured 2026-08-22: corr 1.0000 over n=181 days). P247/P253
fixed this for `funding_close` (shift(1) before the z-score) and left
`oi_change_5d` and `liq_imbalance` unshifted, so every 4H bar of day D
carried day D's realised liquidation imbalance and OI change — up to 24h of
same-day look-ahead (leaked liq_imbalance: IC +0.38 vs 16h fwd return,
decaying by bar-of-day h0 +0.74 -> h20 +0.02; causal −0.06..−0.09).

[P384] The 1d-shifted liq_imbalance is now the FALLBACK path: when the 4h
liquidation archive exists, merge_external_data replaces the column with the
causal trailing-24h series (see tests/test_p384_liq_4h_alignment.py). The
fixtures here write NO 4h archive, so every pin below exercises — and must
keep holding for — the 1d fallback; `TestP384Precedence` pins that the 4h
series wins when present while funding/oi stay byte-identical.

These tests pin the fix BEHAVIOURALLY (by calling the rebuild's own merge
function and its pure derivation seam), never by grepping for `.shift(1)`:

  * P164-style construction test: perturb day D's source values violently
    and assert the 4H bars of day D (and every earlier day) are bit-identical
    across all three features while day D+1's bars move;
  * a per-source pin that row D of each derived feature does not read row D
    of its own source (funding, oi, liq — parametrized);
  * the P253 funding semantics are byte-identical to before;
  * the loader actually routes through the pure seam (a seam nothing calls
    is decoration, P170) and a missing file still degrades to an absent
    feature;
  * the PREMISE on the real archive (skips loudly when the gitignored
    parquets are absent, P252b): 1d@D == 4h-of-day-D. If the fetcher ever
    changes to day-CLOSE stamping, the shift here becomes a 2-day lag and
    this test says so.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
CG_DIR = REPO / "training" / "training_data" / "coinglass_history"


def _load_rebuild_module():
    """Exec rebuild_pipeline.py in isolation from the repo-root `scripts`
    package (same trick as tests/test_rebuild_pipeline_gmm_split.py)."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == "scripts" or k.startswith("scripts.")}
    training_dir = str(REPO / "training")
    # [P382] rebuild_pipeline.py PREPENDS its own dirs to sys.path at import
    # time; restoring only our own insert left `training/` ahead of the repo
    # root, so a later bare `import scripts.seat_check` resolved `scripts`
    # to training/scripts (ModuleNotFoundError in an unrelated test file —
    # seen only in a clean worktree). Snapshot and restore the WHOLE path.
    saved_path = list(sys.path)
    sys.path.insert(0, training_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            "rebuild_pipeline_under_test_p382",
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
D = 30          # the perturbed day (row index into the daily frames)
FEATS = ("funding_rate_zscore", "oi_change_5d", "liq_imbalance")


def _daily_index(n=N_DAYS):
    # Day-OPEN stamps, exactly the archive's convention (00:00 UTC).
    return pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")


def _synthetic_sources(seed=7):
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
    liq = pd.DataFrame({
        "timestamp": ts, "long_liq_usd": long_l, "short_liq_usd": short_l,
        "total_liq_usd": long_l + short_l,
        "liq_imbalance": (long_l - short_l) / (long_l + short_l),
    })
    return funding, oi, liq


def _futures_daily():
    # Stamped at day CLOSE by its fetcher (P281) — not under test here, but
    # merge_external_data REFUSES without it, so provide a benign one.
    ts = _daily_index()
    return pd.DataFrame({
        "timestamp": ts,
        "marketorder_volume": np.linspace(1.0, 2.0, N_DAYS),
        "marketorder_volume_from": np.linspace(1.0, 2.0, N_DAYS),
        "tradecount": np.linspace(100.0, 200.0, N_DAYS),
    })


def _bars_4h():
    ts = pd.date_range("2026-01-01", periods=N_DAYS * 6, freq="4h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "close": np.linspace(100, 200, len(ts))})


def _write_sources(tmp_path, funding, oi, liq, asset="BTC"):
    cg = tmp_path / "cg"
    fu = tmp_path / "fut"
    cg.mkdir(parents=True, exist_ok=True)
    fu.mkdir(parents=True, exist_ok=True)
    funding.to_parquet(cg / f"{asset}_funding_1d.parquet", index=False)
    oi.to_parquet(cg / f"{asset}_oi_1d.parquet", index=False)
    liq.to_parquet(cg / f"{asset}_liquidation_1d.parquet", index=False)
    _futures_daily().to_parquet(fu / f"{asset}_futures_daily.parquet", index=False)
    return cg, fu


def _merge(rp, monkeypatch, tmp_path, funding, oi, liq):
    cg, fu = _write_sources(tmp_path, funding, oi, liq)
    monkeypatch.setattr(rp, "COINGLASS_DIR", cg)
    monkeypatch.setattr(rp, "FUTURES_DIR", fu)
    out = rp.merge_external_data(_bars_4h(), "BTC")
    out["day"] = out["timestamp"].dt.floor("D")
    return out


def _day_rows(out, k):
    return out[out["day"] == _daily_index()[k]]


# ---------------------------------------------------------------- tests
class TestConstructionCausality:
    """P164-style: perturb the FUTURE (day D) and require the past (every
    bar of day D and earlier) to be bit-identical — for all three sources."""

    def test_perturbing_day_D_leaves_day_D_bars_bit_identical_and_moves_day_D_plus_1(
            self, rp, monkeypatch, tmp_path):
        funding, oi, liq = _synthetic_sources()
        base = _merge(rp, monkeypatch, tmp_path, funding, oi, liq)

        f2, o2, l2 = funding.copy(), oi.copy(), liq.copy()
        # Violent, in-range perturbations of day D only.
        f2.loc[D, "funding_close"] = 0.05            # ~500x the noise scale
        o2.loc[D, "oi_close"] = o2.loc[D, "oi_close"] * 3.0
        l2.loc[D, "liq_imbalance"] = -0.99 if liq.loc[D, "liq_imbalance"] > 0 else 0.99
        pert = _merge(rp, monkeypatch, tmp_path / "pert", f2, o2, l2)

        assert len(base) == len(pert) == N_DAYS * 6
        # Bars on day D and every earlier day: bit-identical on ALL external
        # columns (the futures columns are untouched by construction, the
        # CoinGlass columns by the shift).
        upto_D = base["day"] <= _daily_index()[D]
        ext = [c for c in rp.EXTERNAL_FEATURE_COLS]
        for c in ext:
            assert np.array_equal(base.loc[upto_D, c].to_numpy(),
                                  pert.loc[upto_D, c].to_numpy(),
                                  equal_nan=True), \
                f"{c}: a bar on/before day D moved when day D's source was perturbed (look-ahead)"
        # Day D+1: every one of the three CoinGlass features must MOVE, on
        # every one of its six bars (a shift that merely delays the leak by
        # one bar would fail this).
        for c in FEATS:
            b1 = _day_rows(base, D + 1)[c].to_numpy()
            p1 = _day_rows(pert, D + 1)[c].to_numpy()
            assert len(b1) == 6
            assert not np.any(np.isclose(b1, p1)), \
                f"{c}: day D+1 bars did not react to day D's value — the feature is not being merged at all"

    def test_day_D_bars_carry_exactly_day_D_minus_1_source_value(
            self, rp, monkeypatch, tmp_path):
        """The merged liq_imbalance on every bar of day D equals the SOURCE
        row D-1 (clipped) — the precise statement of 'bars read yesterday'."""
        funding, oi, liq = _synthetic_sources()
        out = _merge(rp, monkeypatch, tmp_path, funding, oi, liq)
        for k in (D - 1, D, D + 1):
            vals = _day_rows(out, k)["liq_imbalance"].to_numpy()
            assert len(vals) == 6
            expected = float(np.clip(liq.loc[k - 1, "liq_imbalance"], -1, 1))
            assert np.allclose(vals, expected), (k, vals, expected)
            # and NOT the same-day value (the pre-P382 behaviour)
            assert not np.allclose(vals, float(liq.loc[k, "liq_imbalance"]))
        # Same for oi_change_5d: row k of the merged value equals
        # oi_close[k-1] / oi_close[k-6] - 1 (shift THEN 5d change).
        k = D
        vals = _day_rows(out, k)["oi_change_5d"].to_numpy()
        exp = float(np.clip(oi.loc[k - 1, "oi_close"] / oi.loc[k - 6, "oi_close"] - 1.0, -5, 5))
        assert np.allclose(vals, exp)
        leaked = float(np.clip(oi.loc[k, "oi_close"] / oi.loc[k - 5, "oi_close"] - 1.0, -5, 5))
        assert not np.allclose(vals, leaked)


class TestEachSourceIsShiftedBehaviourally:
    """Row D of each derived feature must not read row D of its own source.
    Written as a CALL into the pure seam, parametrized over the three
    sources — a substring pin on `.shift(1)` would pass on a shift applied
    to the wrong column or after the transform."""

    @pytest.mark.parametrize("source,col,feat,perturb", [
        ("funding", "funding_close", "funding_rate_zscore", lambda v: 0.05),
        ("oi", "oi_close", "oi_change_5d", lambda v: v * 3.0),
        ("liq", "liq_imbalance", "liq_imbalance", lambda v: -0.99 if v > 0 else 0.99),
    ])
    def test_row_D_does_not_read_source_row_D(self, rp, source, col, feat, perturb):
        funding, oi, liq = _synthetic_sources()
        base = rp._derive_coinglass_daily(funding, oi, liq)
        frames = {"funding": funding.copy(), "oi": oi.copy(), "liq": liq.copy()}
        frames[source].loc[D, col] = perturb(frames[source].loc[D, col])
        pert = rp._derive_coinglass_daily(frames["funding"], frames["oi"], frames["liq"])
        b, p = base[feat].to_numpy(), pert[feat].to_numpy()
        # rows <= D unchanged (bit-identical, NaNs included)
        assert np.array_equal(b[: D + 1], p[: D + 1], equal_nan=True), feat
        # row D+1 changed — the shifted source row D now lands there
        assert not np.isclose(b[D + 1], p[D + 1]), feat
        # and the other two features are untouched entirely
        for other in FEATS:
            if other != feat:
                assert np.array_equal(base[other].to_numpy(), pert[other].to_numpy(),
                                      equal_nan=True), other


class TestFundingSemanticsUnchanged:
    def test_funding_z_is_exactly_the_p253_formula(self, rp):
        """The P253 fix is byte-identical: z-score of funding_close.shift(1),
        window 30 — this change touched OI and liquidation only."""
        funding, oi, liq = _synthetic_sources()
        out = rp._derive_coinglass_daily(funding, oi, liq)
        expected = rp._rolling_zscore(funding["funding_close"].shift(1), 30).to_numpy()
        assert np.array_equal(out["funding_rate_zscore"].to_numpy(), expected, equal_nan=True)

    def test_first_row_of_every_feature_is_nan_not_a_fabricated_value(self, rp):
        """Row 0 has no yesterday: all three must be NaN (merge fills 0.0
        later, flagged by has_external_data), never a same-day reading."""
        funding, oi, liq = _synthetic_sources()
        out = rp._derive_coinglass_daily(funding, oi, liq)
        for c in FEATS:
            assert np.isnan(out.loc[0, c]), c
        # liq row 1 is exactly source row 0 (clipped) — a one-row shift, not two
        assert out.loc[1, "liq_imbalance"] == pytest.approx(
            float(np.clip(liq.loc[0, "liq_imbalance"], -1, 1)))


class TestLoaderWiring:
    def test_load_coinglass_daily_routes_through_the_pure_seam(self, rp, monkeypatch, tmp_path):
        funding, oi, liq = _synthetic_sources()
        cg, _ = _write_sources(tmp_path, funding, oi, liq)
        monkeypatch.setattr(rp, "COINGLASS_DIR", cg)
        via_loader = rp._load_coinglass_daily("BTC")
        direct = rp._derive_coinglass_daily(
            pd.read_parquet(cg / "BTC_funding_1d.parquet"),
            pd.read_parquet(cg / "BTC_oi_1d.parquet"),
            pd.read_parquet(cg / "BTC_liquidation_1d.parquet"))
        pd.testing.assert_frame_equal(via_loader, direct)

    def test_missing_files_still_yield_an_absent_feature_not_a_crash(self, rp, monkeypatch, tmp_path):
        funding, oi, liq = _synthetic_sources()
        cg, _ = _write_sources(tmp_path, funding, oi, liq)
        (cg / "BTC_liquidation_1d.parquet").unlink()
        monkeypatch.setattr(rp, "COINGLASS_DIR", cg)
        out = rp._load_coinglass_daily("BTC")
        assert "liq_imbalance" in out.columns
        assert out["liq_imbalance"].isna().all()
        assert out["oi_change_5d"].notna().sum() > 0


class TestPremiseOnTheRealArchive:
    """The whole fix rests on the archive being day-OPEN stamped with
    day-CLOSE content. Re-measure it on the real files when present; skip
    loudly otherwise (the parquets are gitignored, P252b)."""

    @pytest.mark.parametrize("asset", ["BTC", "ETH", "SOL"])
    def test_1d_row_D_is_day_D_content(self, asset):
        l1p = CG_DIR / f"{asset}_liquidation_1d.parquet"
        l4p = CG_DIR / f"{asset}_liquidation_4h.parquet"
        o1p = CG_DIR / f"{asset}_oi_1d.parquet"
        o4p = CG_DIR / f"{asset}_oi_4h.parquet"
        if not all(p.exists() for p in (l1p, l4p, o1p, o4p)):
            pytest.skip(f"[P382] CoinGlass archive for {asset} not on this machine "
                        f"(gitignored, operator-local) — premise not re-measured here")
        l1, l4 = pd.read_parquet(l1p), pd.read_parquet(l4p)
        o1, o4 = pd.read_parquet(o1p), pd.read_parquet(o4p)
        for df in (l1, l4, o1, o4):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # All 1d rows are stamped at 00:00 UTC (day OPEN).
        assert set(l1["timestamp"].dt.hour.unique()) == {0}
        assert set(o1["timestamp"].dt.hour.unique()) == {0}
        # liq 1d @ D == sum of the 4h rows of day D (not day D-1).
        s4 = l4.groupby(l4["timestamp"].dt.floor("D"))["total_liq_usd"].sum()
        s1 = l1.set_index(l1["timestamp"].dt.floor("D"))["total_liq_usd"]
        j = pd.concat([s1.rename("d1"), s4.rename("s4")], axis=1).dropna()
        assert len(j) >= 30
        corr_same = np.corrcoef(j["d1"], j["s4"])[0, 1]
        j2 = pd.concat([s1.rename("d1"), s4.shift(1, freq="D").rename("s4p")], axis=1).dropna()
        corr_prev = np.corrcoef(j2["d1"], j2["s4p"])[0, 1]
        assert corr_same > 0.999, (asset, corr_same)
        assert corr_prev < corr_same, (asset, corr_prev, corr_same)
        # oi_close 1d @ D == 4h close at D 20:00 (the day's last bar).
        last20 = o4[o4["timestamp"].dt.hour == 20].set_index(
            o4.loc[o4["timestamp"].dt.hour == 20, "timestamp"].dt.floor("D"))["oi_close"]
        s1o = o1.set_index(o1["timestamp"].dt.floor("D"))["oi_close"]
        jo = pd.concat([s1o.rename("d1"), last20.rename("l20")], axis=1).dropna()
        frac_exact = float((jo["d1"] == jo["l20"]).mean())
        assert frac_exact > 0.95, (asset, frac_exact)


class TestP384Precedence:
    """[P384] With a 4h liquidation archive present, the merged liq_imbalance
    is the trailing-24h series, NOT the 1d-shifted one — and the 1d-shifted
    fallback (every pin above) is untouched: funding/oi are byte-identical
    across the two paths."""

    def _archive_4h(self):
        rng = np.random.default_rng(3)
        ts = pd.date_range("2026-01-01", periods=N_DAYS * 6, freq="4h", tz="UTC")
        long_l = rng.uniform(1e5, 5e6, len(ts))
        short_l = rng.uniform(1e5, 5e6, len(ts))
        return pd.DataFrame({"timestamp": ts, "long_liq_usd": long_l,
                             "short_liq_usd": short_l, "total_liq_usd": long_l + short_l,
                             "liq_imbalance": (long_l - short_l) / (long_l + short_l)})

    def test_4h_archive_takes_precedence_and_funding_oi_are_byte_identical(
            self, rp, monkeypatch, tmp_path):
        funding, oi, liq = _synthetic_sources()
        fallback = _merge(rp, monkeypatch, tmp_path / "fb", funding, oi, liq)
        cg, fu = _write_sources(tmp_path / "p384", funding, oi, liq)
        arch = self._archive_4h()
        arch.to_parquet(cg / "BTC_liquidation_4h.parquet", index=False)
        monkeypatch.setattr(rp, "COINGLASS_DIR", cg)
        monkeypatch.setattr(rp, "FUTURES_DIR", fu)
        with_4h = rp.merge_external_data(_bars_4h(), "BTC")
        # funding / oi / futures: byte-identical in both paths
        for c in ("funding_rate_zscore", "oi_change_5d", "taker_ratio_zscore",
                  "tradecount_zscore", "taker_vol_momentum"):
            assert with_4h[c].equals(fallback[c]), c
        # liq: the 4h trailing series (bar t == row t), not day D-1's 1d value
        series = rp._derive_liq_trailing_24h(arch)["liq_imbalance"].fillna(0.0).to_numpy()
        assert np.allclose(with_4h["liq_imbalance"].to_numpy(), series, atol=1e-12)
        assert not with_4h["liq_imbalance"].equals(fallback["liq_imbalance"])
        # and the fallback is still exactly the P382 semantics (day D reads D-1)
        fb_day = fallback[fallback["day"] == _daily_index()[D]]["liq_imbalance"].to_numpy()
        assert np.allclose(fb_day, float(np.clip(liq.loc[D - 1, "liq_imbalance"], -1, 1)))


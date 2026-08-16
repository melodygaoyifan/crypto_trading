"""P281 — the missing-data layer for the derivatives retrain program.

The P279 audit's data gaps, closed by reusing APIs already in the tree:
the Binance Vision futures archives fill the three all-zero feature
columns at full depth; the breadth spot closes backfill; the loader
refuses instead of silently zeroing; calbasis state survives deploys.
"""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from tests._source_scan import read_source

REPO = Path(__file__).resolve().parent.parent


class TestFuturesFetcher:
    def test_causality_stamp_is_day_close(self):
        # the P247 lesson at BUILD time: day-D data must be stamped at the
        # day-CLOSE boundary so merge_asof never shows it to day-D bars
        src = read_source(
            REPO / "training" / "scripts" / "fetch_binance_futures_daily.py")
        assert 'raw["open_time"] + timedelta(days=1)' in src, (
            "the availability stamp lost its +1-day shift — day-D futures "
            "aggregates would be visible to day-D 4H bars (the P247 leak "
            "shape, at the source this time)")

    def test_in_progress_day_dropped(self):
        src = read_source(
            REPO / "training" / "scripts" / "fetch_binance_futures_daily.py")
        assert 'out["data_date"] < today' in src, (
            "today's in-progress day is no longer dropped (P253c class)")

    def test_column_mapping_documented_and_unit_free_rationale(self):
        src = read_source(
            REPO / "training" / "scripts" / "fetch_binance_futures_daily.py")
        for col in ("marketorder_volume", "marketorder_volume_from",
                    "tradecount"):
            assert col in src, f"loader-contract column {col} missing"
        assert "taker_buy_quote" in src and "taker_buy_base" in src

    def test_merge_not_overwrite(self):
        # P266: an archive fetcher must union, never overwrite
        src = read_source(
            REPO / "training" / "scripts" / "fetch_binance_futures_daily.py")
        assert 'drop_duplicates(subset="timestamp", keep="last")' in src

    def test_breadth_assets_mapped(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fbfd", REPO / "training" / "scripts" /
            "fetch_binance_futures_daily.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert set(mod.SYMBOLS) >= {"BTC", "ETH", "SOL", "XRP", "ADA",
                                    "LTC", "DOGE", "BNB"}


class TestLoaderRefusal:
    def test_missing_futures_file_refuses_never_zeros(self):
        # the original sin: this branch silently fed three all-zero columns
        # to every model ever trained (P279). A missing file is a broken
        # chain, not a neutral default (P2/P199).
        src = read_source(
            REPO / "training" / "scripts" / "rebuild_pipeline.py")
        import re
        m = re.search(r"def _load_futures_daily.*?(?=\ndef )", src, re.S)
        assert m and "REFUSING" in m.group(0), (
            "_load_futures_daily reverted to the silent empty-frame "
            "fallback — the dead-columns defect returns")
        assert "return pd.DataFrame(columns=[" not in m.group(0).split(
            "REFUSING")[0], "the silent fallback still precedes the refusal"

    def test_fetcher_in_refresh_chain(self):
        mk = read_source(REPO / "training" / "Makefile")
        import re
        m = re.search(r"^refresh-data:.*?(?=^\w)", mk, __import__("re").M
                      | __import__("re").S)
        assert m and "fetch_binance_futures_daily.py" in m.group(0), (
            "the futures fetch left the refresh-data chain — the loader's "
            "refusal would then fire on every routine rebuild")
        # dependency order: fetch BEFORE rebuild
        blk = m.group(0)
        assert blk.find("fetch_binance_futures_daily") < blk.find(
            "rebuild_pipeline")


class TestSpotFetcherBreadth:
    def test_assets_param_and_breadth_symbols(self):
        src = read_source(REPO / "training" / "fetch_binance_full.py")
        assert '"--assets"' in src
        for a in ("XRP", "ADA", "LTC", "DOGE", "BNB"):
            assert f'"{a}"' in src


class TestCostConventionUnified:
    def test_both_labs_charge_the_same_trade_identically(self):
        # [P279 finding / P281 fix] the two labs read ONE constant under
        # OPPOSITE conventions — supervised charged full RT per leg (2x).
        # Pin: one unit of turnover at RT=6bps costs 3bps in BOTH.
        import numpy as np
        import sys
        sys.path.insert(0, str(REPO / "training"))
        from train_supervised_full import evaluate_segment
        from mechanism_lab import pnl_after_cost
        close = np.array([100.0] * 10)
        pos = np.array([0.0] * 5 + [1.0] * 5)  # one entry leg
        ev = evaluate_segment(close, pos, 6.0, 0, 10)
        # flat prices -> pnl_pct is pure cost; entry leg = 3bps = -0.03%
        assert abs(ev["pnl_pct"] - (-0.03)) < 1e-6, (
            f"supervised charged {ev['pnl_pct']}% for one leg at RT=6 — "
            f"the 2x per-side overcharge is back (P279)")
        ml = pnl_after_cost(close, pos, 6.0, 0, 10)
        assert abs(ml["cost"] - 0.0003) < 1e-9


class TestCalbasisPersistence:
    def _enh(self, tmp_path):
        from defense.enhancement_shadows import EnhancementShadows
        return EnhancementShadows(data_dir=str(tmp_path))

    def test_state_round_trips_across_construction(self, tmp_path):
        # P154 class: the only CDE-native signal's warmup must survive a
        # deploy, or deploy-heavy weeks keep calbasis permanently flat
        e1 = self._enh(tmp_path)
        for i in range(25):
            e1.calbasis_direction("BTC", 63360.0 + i, 40.0, 63045.0, 1587.0)
        assert len(e1._basis_hist["BTC"]) >= 20
        e2 = self._enh(tmp_path)   # a fresh process
        assert len(e2._basis_hist.get("BTC", [])) >= 20, (
            "calbasis history did not survive re-construction — every "
            "deploy restarts the ~3.3-day warmup")
        # and the restored instance is immediately past warmup
        d, sl, r = e2.calbasis_direction("BTC", 63360.0, 40.0,
                                         63045.0, 1587.0)
        assert r != "warmup"

    def test_corrupt_state_degrades_to_cold_start(self, tmp_path):
        (tmp_path / "calbasis_state.json").write_text("{not json",
                                                      encoding="utf-8")
        e = self._enh(tmp_path)   # must not raise
        assert e._basis_hist == {}

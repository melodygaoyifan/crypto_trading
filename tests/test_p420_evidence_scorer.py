"""[P420] The pooled `regimebook` exam pooled 8 books running 3 rules, and
three analytics tools dropped XRP/BNB rows silently.

  * breadth books now write `strategy: "regimebook_breadth"`; the scorer
    pools `regimebook` over the HOME TRIO only (POOL_ASSET_FILTER) so the
    pre-P420 breadth rows carrying "regimebook" cannot re-contaminate it;
  * `load_ohlcv` also reads `{ASSET}_4H_ohlcv_kraken.parquet` (contract with
    scripts/september_check.py, which no longer overwrites the Binance-derived
    `_4H_ohlcv.parquet`);
  * agent_ic_review prices every asset PRESENT in the window and names any
    asset it cannot price; trade_attributor's per-asset waterfall covers
    every asset present in the trades.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import analytics.shadow_ic.compute_shadow_ic as sic  # noqa: E402
from analytics.shadow_ic.compute_shadow_ic import (  # noqa: E402
    POOLABLE_FAMILIES, POOL_ASSET_FILTER, POOLED_KEY, compute_per_strategy_ic,
    pool_key_for)
from defense.regime_book_shadow import (  # noqa: E402
    BREADTH_ASSETS, HOME_TRIO, RegimeBookShadow, SHADOW_STRATEGY_NAMES,
    strategy_name_for)


class TestBreadthFamily:
    def test_breadth_rows_are_their_own_family(self):
        for a in BREADTH_ASSETS:
            assert strategy_name_for(a) == "regimebook_breadth"
        for a in HOME_TRIO:
            assert strategy_name_for(a) == "regimebook"
        assert "regimebook_breadth" in SHADOW_STRATEGY_NAMES
        assert "regimebook_breadth" in POOLABLE_FAMILIES

    def test_the_writer_stamps_the_breadth_name(self, tmp_path):
        """Driven through the real record_tick, not a source pin."""
        h = RegimeBookShadow(data_dir=str(tmp_path))
        closes = [100.0 + 0.05 * i for i in range(600)]   # bull, > MIN_BARS
        rec = h.record_tick("XRP", closes, price=closes[-1])
        assert rec is not None and rec["strategy"] == "regimebook_breadth"
        assert rec["book_version"] == "v1_breadth_trend_only"
        rec_btc = h.record_tick("BTC", closes, price=closes[-1])
        assert rec_btc["strategy"] == "regimebook"
        # same ledger-file prefix: no scorer prefix change needed (P192)
        assert (tmp_path / "strategy_shadow" / "regimebook_XRP.jsonl").exists()

    def test_the_docstring_records_the_ledger_discontinuity(self):
        import defense.regime_book_shadow as m
        assert "LEDGER DISCONTINUITY" in m.__doc__
        assert "regimebook_breadth" in m.__doc__


class TestPoolAssetFilter:
    def test_pool_key_truth_table(self):
        assert pool_key_for("regimebook", "BTC", True) == POOLED_KEY
        assert pool_key_for("regimebook", "XRP", True) == "XRP"      # old row
        assert pool_key_for("regimebook_breadth", "XRP", True) == POOLED_KEY
        assert pool_key_for("regimebook_breadth", "BTC", True) == "BTC"
        assert pool_key_for("etfflow", "BTC", True) == POOLED_KEY   # unfiltered family
        assert pool_key_for("regimebook", "BTC", False) == "BTC"
        assert pool_key_for("mlpshadow", "BTC", True) == "BTC"      # per-asset family

    def test_every_regimebook_family_is_filtered_to_its_own_assets(self):
        for fam in ("regimebook", "regimebook_adj", "regimebook_volskip",
                    "regimebook_fgshort"):
            assert POOL_ASSET_FILTER[fam] == HOME_TRIO
        assert POOL_ASSET_FILTER["regimebook_breadth"] == BREADTH_ASSETS

    def test_a_mixed_ledger_pools_only_the_trio_under_regimebook(self, monkeypatch):
        """Trio rows -> (regimebook, POOLED); a pre-P420 breadth row still
        labelled "regimebook" -> per-asset (visible, never in the pool, never
        dropped); new breadth rows -> (regimebook_breadth, POOLED)."""
        monkeypatch.setattr(sic, "load_ohlcv", lambda a: None)
        now = time.time()
        recs = []
        for a in HOME_TRIO:
            recs.append({"strategy": "regimebook", "asset": a, "direction": 1.0,
                         "confidence": 1.0, "_parsed_ts": None})
        recs.append({"strategy": "regimebook", "asset": "XRP", "direction": 1.0,
                     "confidence": 1.0, "_parsed_ts": None})          # OLD breadth row
        for a in BREADTH_ASSETS:
            recs.append({"strategy": "regimebook_breadth", "asset": a,
                         "direction": 1.0, "confidence": 1.0, "_parsed_ts": None})
        out = compute_per_strategy_ic(recs, horizons_bars=(4,), pool_assets=True)
        keys = set(out)
        assert ("regimebook", POOLED_KEY) in keys
        assert ("regimebook", "XRP") in keys, "the old breadth row is scored per-asset"
        assert ("regimebook_breadth", POOLED_KEY) in keys
        assert ("regimebook_breadth", "XRP") not in keys
        assert not any(k[0] == "regimebook" and k[1] in BREADTH_ASSETS
                       and k[1] != "XRP" for k in keys)


class TestOhlcvKrakenSeries:
    def test_load_ohlcv_falls_back_to_the_kraken_series(self, tmp_path, monkeypatch):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        monkeypatch.setattr(sic, "OHLCV_DIR", tmp_path)
        df = pd.DataFrame({"timestamp": pd.date_range("2026-08-01", periods=6,
                                                       freq="4h", tz="UTC"),
                           "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        df.to_parquet(tmp_path / "XRP_4H_ohlcv_kraken.parquet")
        got = sic.load_ohlcv("XRP")
        assert len(got) == 6 and "close" in got.columns
        with pytest.raises(FileNotFoundError):
            sic.load_ohlcv("BNB")

    def test_the_binance_series_still_wins_when_present(self, tmp_path, monkeypatch):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        monkeypatch.setattr(sic, "OHLCV_DIR", tmp_path)
        pd.DataFrame({"timestamp": pd.date_range("2026-08-01", periods=3, freq="4h", tz="UTC"),
                      "close": [1.0, 1.0, 1.0]}).to_parquet(tmp_path / "BTC_4H_ohlcv.parquet")
        pd.DataFrame({"timestamp": pd.date_range("2026-08-01", periods=3, freq="4h", tz="UTC"),
                      "close": [9.0, 9.0, 9.0]}).to_parquet(tmp_path / "BTC_4H_ohlcv_kraken.parquet")
        assert sic.load_ohlcv("BTC")["close"].iloc[0] == 1.0


class TestAnalyticsNoLongerDropBreadthSilently:
    def test_agent_ic_review_keeps_xrp_and_bnb_rows(self, tmp_path, capsys):
        from analytics.ic import agent_ic_review as air
        assert "XRP" in air.KRAKEN_PAIRS and "BNB" in air.KRAKEN_PAIRS
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        with open(tmp_path / "signals_1.jsonl", "w", encoding="utf-8") as fh:
            for a in ("BTC", "XRP", "BNB", "ZZZ"):
                fh.write(json.dumps({"ts": ts, "asset": a, "agent": "quant",
                                     "direction": 1.0}) + "\n")
        recs = air.load_signal_records(tmp_path, 30)
        assert {r["asset"] for r in recs} == {"BTC", "XRP", "BNB"}
        err = capsys.readouterr().err
        assert "ZZZ" in err, "an asset dropped for lack of a pair must be SAID"

    def test_agent_ic_review_prices_only_assets_present(self):
        from tests._source_scan import code_only
        src = code_only(REPO / "analytics" / "ic" / "agent_ic_review.py")
        assert "for a in _present}" in src
        assert "bars = {a: fetch_closes(a) for a in KRAKEN_PAIRS}" not in src

    def test_trade_attributor_waterfall_covers_every_asset_present(self, tmp_path):
        from analytics.trade_attributor import TradeAttributor
        ta = TradeAttributor(persist_path=str(tmp_path / "ta.jsonl"))
        for a, pnl in (("BTC", 5.0), ("XRP", -2.0)):
            ta.record_entry(a, price=1.0, fee=0.1, notional=100.0, direction=1,
                            strategy="regimebook")
            ta.record_exit(a, price=1.0, fee=0.1, notional=100.0, gross_pnl=pnl)
        rep = ta.report()
        assert "XRP" in rep["per_asset"] and "BTC" in rep["per_asset"]
        assert "XRP" in ta.report_text()

    def test_deliberate_trio_sites_say_so(self):
        """Where the trio is by design the site must say so (P420 task 8)."""
        for rel in ("analytics/calibration/tripwire_check.py",
                    "analytics/drl_drift/_common.py"):
            src = (REPO / rel).read_text(encoding="utf-8")
            i = src.index('ASSETS = ')
            assert "BY DESIGN" in src[i - 400:i], rel
        src = (REPO / "scripts" / "seat_check.py").read_text(encoding="utf-8")
        i = src.index('for a in ("BTC", "ETH", "SOL")')
        assert "BY DESIGN" in src[i - 500:i]

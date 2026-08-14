"""[P266] Training data-collection: the archive must grow, the scorer must
see September, and a rebuild must be ONE step.

Four gaps closed:
  1. `fetch_coinglass_history.py` bare-overwrote its output while the API
     serves only ~180 days of liquidation/OI depth — the trainable history
     of `liq_imbalance` (the external group's strict-window carrier, P256)
     was capped at 180 rolling days forever, and a re-fetch after a gap
     longer than the API window would have permanently lost the middle.
     Now merges (union by timestamp, new wins), atomically.
  2. `refresh_ohlcv_4h.py` was bounded by Binance's MONTHLY archives, so the
     ~2026-09-07..09 P166 forward reads could only join prices through
     ~Aug-31 (P264 recorded this; never built). It now extends past the raw
     parquet via the DAILY vision archives (T+1), completed days only —
     validated live 2026-08-14: coverage 07-31 -> 08-13, overlap parity with
     the training parquet 13,095/13,095 bars at 0.0 diff.
  3. A parquet rebuild was TWO steps (rebuild_pipeline then
     build_flow_features — the P253c standing rule, written after the P253b
     rebuild silently dropped the 13 fv2 columns). The fv2 build now runs
     INSIDE rebuild_pipeline as STEP 5b and a failure fails the rebuild.
  4. `make collect` invoked the legacy multi-source collect_data.py, which
     wrote {ASSET}_60m.parquet into the SAME dir as the canonical 6-year
     Binance parquets — its Kraken source caps at ~720 bars, so one run
     would have truncated six years of training data to ~a month (the P186
     landmine class). Target retired with a refusal; script archived.
     `make refresh-data` is the canonical one-command sequence.
"""

import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
TRAINING = REPO / "training"
sys.path.insert(0, str(TRAINING / "scripts"))

MAKEFILE = (TRAINING / "Makefile").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. CoinGlass archive merge
# ---------------------------------------------------------------------------

class TestCoinglassMerge:
    def _mod(self):
        import importlib
        import fetch_coinglass_history as m
        return importlib.reload(m)

    def test_merge_grows_the_archive_past_the_api_window(self):
        m = self._mod()
        old = pd.DataFrame({
            "timestamp": pd.date_range("2026-02-08", periods=10, freq="4h",
                                       tz="UTC"),
            "v": range(10)})
        new = pd.DataFrame({
            "timestamp": pd.date_range("2026-02-09 16:00", periods=10,
                                       freq="4h", tz="UTC"),
            "v": range(100, 110)})
        merged = m.merge_history(old, new)
        assert merged["timestamp"].min() == old["timestamp"].min(), (
            "the pre-window history was dropped — the archive can never "
            "grow past the API's 180-day depth again (P266)")
        assert merged["timestamp"].max() == new["timestamp"].max()
        assert len(merged) == 20 - 0  # no overlap in this fixture

    def test_collisions_prefer_the_new_fetch(self):
        m = self._mod()
        ts = pd.date_range("2026-08-01", periods=5, freq="4h", tz="UTC")
        old = pd.DataFrame({"timestamp": ts, "v": [1] * 5})
        new = pd.DataFrame({"timestamp": ts[2:], "v": [9] * 3})
        merged = m.merge_history(old, new)
        assert len(merged) == 5
        assert merged["v"].tolist() == [1, 1, 9, 9, 9], (
            "the API restates recent bars; the newer values must win")

    def test_none_existing_is_a_plain_copy(self):
        m = self._mod()
        new = pd.DataFrame({
            "timestamp": pd.date_range("2026-08-01", periods=3, freq="4h",
                                       tz="UTC"),
            "v": [1, 2, 3]})
        merged = m.merge_history(None, new)
        assert len(merged) == 3

    def test_output_is_sorted_and_deduplicated(self):
        m = self._mod()
        ts = pd.date_range("2026-08-01", periods=4, freq="4h", tz="UTC")
        old = pd.DataFrame({"timestamp": [ts[3], ts[1]], "v": [4, 2]})
        new = pd.DataFrame({"timestamp": [ts[0], ts[1]], "v": [10, 20]})
        merged = m.merge_history(old, new)
        assert merged["timestamp"].is_monotonic_increasing
        assert merged["timestamp"].nunique() == len(merged)

    def test_the_writer_actually_merges_and_writes_atomically(self):
        src = (TRAINING / "scripts" / "fetch_coinglass_history.py").read_text(
            encoding="utf-8")
        assert "merge_history(existing" in src, (
            "the save path no longer merges — bare overwrite caps "
            "liq_imbalance's trainable history at 180 rolling days (P266)")
        assert "os.replace(" in src, "the write is no longer atomic"


# ---------------------------------------------------------------------------
# 2. daily-archive extension
# ---------------------------------------------------------------------------

class TestDailyExtension:
    def _mod(self):
        import importlib
        import refresh_ohlcv_4h as m
        return importlib.reload(m)

    def test_dates_run_from_boundary_day_through_yesterday(self):
        m = self._mod()
        dates = m.daily_dates_needed(date(2026, 7, 31), date(2026, 8, 14))
        assert dates[0] == "2026-07-31", (
            "the boundary day must be re-fetched (a partial boundary day "
            "would leave a seam; dedup makes the re-fetch harmless)")
        assert dates[-1] == "2026-08-13", (
            "TODAY was fetched — its day is incomplete, so the appended 4H "
            "bars would be partial (the P253c/P265 in-progress-bar class)")

    def test_current_data_needs_nothing(self):
        m = self._mod()
        assert m.daily_dates_needed(date(2026, 8, 14), date(2026, 8, 14)) == []

    def test_huge_gaps_refuse_rather_than_hammer(self):
        m = self._mod()
        out = m.daily_dates_needed(date(2026, 1, 1), date(2026, 8, 14))
        assert out is None, (
            "a 200+ day gap should route to the monthly fetcher, not ~200 "
            "daily zip downloads")

    def test_kline_parser_handles_both_timestamp_units(self):
        m = self._mod()
        ms = 1_755_100_800_000          # a 2025-08 instant in milliseconds
        us = ms * 1000                  # the SAME instant in microseconds
        parsed = []
        for unit_val in (ms, us):
            raw = pd.DataFrame([[unit_val, "1", "2", "0.5", "1.5", "10",
                                 0, 0, 0, 0, 0, 0]])
            out = m.parse_kline_frame(raw)
            parsed.append(out["timestamp"].iloc[0])
            assert out["close"].iloc[0] == 1.5
        assert parsed[0] == parsed[1], (
            f"unit detection failed: ms->{parsed[0]} vs us->{parsed[1]} — "
            "Binance switched to µs ~2025-01; a wrong unit puts every bar "
            "in 1970 or 55688 (fetch_binance_full's convention)")
        assert 2020 < parsed[0].year < 2100

    def test_build_wires_the_extension_in(self):
        src = (TRAINING / "scripts" / "refresh_ohlcv_4h.py").read_text(
            encoding="utf-8")
        i = src.index("raw, _ext_note = extend_with_daily(asset, raw)")
        # the extension must run BEFORE the resample
        assert i < src.index('.resample("4h", origin="start_day")'), (
            "the daily extension no longer feeds the resample — the scorer "
            "series is monthly-bounded again and the September P166 reads "
            "join only through the previous month-end (P264/P266)")

    @pytest.mark.skipif(
        not (TRAINING / "training_data" / "drl_training" /
             "BTC_4H_ohlcv.parquet").exists(),
        reason="operator-local parquet absent (gitignored, P252b class) — "
               "coverage assertion runs only where the data lives")
    def test_live_output_extends_past_the_monthly_boundary(self):
        df = pd.read_parquet(TRAINING / "training_data" / "drl_training" /
                             "BTC_4H_ohlcv.parquet")
        end = pd.Timestamp(df["timestamp"].max())
        # the raw parquet ends at a month boundary; the scorer series must
        # reach past it (daily archives are T+1)
        raw = pd.read_parquet(TRAINING / "training_data" / "raw" /
                              "BTC_60m.parquet")
        raw_end = pd.Timestamp(raw["timestamp"].max())
        assert end > raw_end, (
            f"scorer series ends {end}, raw ends {raw_end} — the daily "
            "extension did not extend anything")


# ---------------------------------------------------------------------------
# 3. one-step rebuild
# ---------------------------------------------------------------------------

class TestOneStepRebuild:
    def test_rebuild_invokes_the_fv2_build(self):
        src = (TRAINING / "scripts" / "rebuild_pipeline.py").read_text(
            encoding="utf-8")
        assert "from build_flow_features import main as _fv2_main" in src, (
            "the fv2 build is no longer inside rebuild_pipeline — the "
            "two-step footgun is re-armed (the P253b rebuild silently "
            "dropped all 13 fv2 columns exactly this way)")
        i = src.index("_fv2_rc = _fv2_main()")
        seg = src[i:i + 400]
        assert "sys.exit(2)" in seg, (
            "an fv2 failure no longer fails the rebuild — a parquet without "
            "fv2 columns looks complete (that is why the P253c rule existed)")

    def test_fv2_runs_after_all_parquets_are_written(self):
        src = (TRAINING / "scripts" / "rebuild_pipeline.py").read_text(
            encoding="utf-8")
        assert src.index("_fv2_rc = _fv2_main()") > src.index(
            "df[keep_cols].to_parquet(out_path, index=False)"), (
            "fv2 runs before the per-asset parquets finish — cross-asset "
            "features need all three on disk")


# ---------------------------------------------------------------------------
# 4. Makefile: landmine retired, refresh sequence exists
# ---------------------------------------------------------------------------

class TestMakefileTargets:
    def test_collect_is_a_refusal(self):
        m = re.search(r"^collect:\n((?:\t.*\n)+)", MAKEFILE, re.M)
        assert m, "the collect target vanished entirely — keep the refusal"
        body = m.group(1)
        assert "collect_data.py" not in body, (
            "make collect invokes the legacy collector again — its Kraken "
            "source caps at ~720 bars and writes into the canonical raw "
            "dir: one run truncates SIX YEARS of training data (P266)")
        assert "exit 1" in body

    def test_the_legacy_collector_is_archived(self):
        assert not (TRAINING / "scripts" / "collect_data.py").exists()
        assert (REPO / "archive" / "legacy_scripts" / "collect_data.py").exists()

    def test_refresh_data_chains_the_canonical_sequence_in_order(self):
        m = re.search(r"^refresh-data:\n((?:\t.*\n)+)", MAKEFILE, re.M)
        assert m, "make refresh-data is gone — the one-command refresh (P266)"
        body = m.group(1)
        order = ["fetch_binance_full.py", "fetch_binance_funding.py",
                 "fetch_coinglass_history.py", "rebuild_pipeline.py",
                 "generate_split_manifest.py", "refresh_ohlcv_4h.py"]
        idx = [body.index(s) for s in order]
        assert idx == sorted(idx), (
            f"refresh-data steps out of dependency order: {order}")
        assert "--skip-gmm" in body, (
            "the refresh refits the GMM by default — {GMM, parquets, "
            "checkpoints} move as ONE versioned set (P215/P253b); a refit "
            "is a deliberate recorded decision, not a side effect")

    def test_check_reports_freshness(self):
        assert "check_data_freshness.py" in MAKEFILE
        assert (TRAINING / "scripts" / "check_data_freshness.py").exists()

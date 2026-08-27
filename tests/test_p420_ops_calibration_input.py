"""[P420] The seat-alpha calibration INPUT was being overwritten by a reader.

`scripts/september_check.py::build_breadth_ohlcv` fetched a ~120d Kraken window
for the breadth assets and MERGED it (keep="last") into
`training/training_data/drl_training/{ASSET}_4H_ohlcv.parquet` — the exact file
`training/funding_legs_lab.load_closes` reads for
`training/seat_alpha_calibration.py --verify`, i.e. the producer behind the
LIVE XRP/BNB gate constants in core.seat_alpha. ~560 Binance bars per asset
were replaced with Kraken prints and the producer drifted from its own shipped
table the first time it ran after the P412c commit (BNB validation +36.6 ->
+51.1, XRP -21.9 -> -14.8; parquets rewritten 08-26 15:04, after the 12:52
commit). A reader of live evidence must never write the calibration input.

THE TWO-FILE CONVENTION, pinned here:
    {ASSET}_4H_ohlcv.parquet          PRIMARY  — refresh_ohlcv_4h.py ONLY
    {ASSET}_4H_ohlcv_kraken.parquet   EXTENSION — september_check.py ONLY
(the scorer unions them, primary wins — fork-4 contract).

And the producer's own --verify had three holes (P420 task 2): it compared
per-era cells only (never the MEDIAN the gate reads), it passed VACUOUSLY on an
asset absent from the shipped table (zero comparisons -> OK, P174), and its
report carried no input provenance. Falsification (recorded): drifting one
shipped median -> exit 3; an unlisted asset -> exit 2; the old primary path
reintroduced in build_breadth_ohlcv -> the source + behavioural pins go red.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from _source_scan import code_only  # noqa: E402

pd = pytest.importorskip("pandas")


# ── 1. september_check never writes the primary ───────────────────────────────

class TestSeptemberCheckNeverWritesThePrimary:

    def test_source_never_names_the_primary_path_in_the_builder(self):
        import inspect
        import scripts.september_check as m
        src = code_only(REPO / "scripts" / "september_check.py")
        i = src.index("def build_breadth_ohlcv")
        blk = src[i:i + 3000]
        assert 'f"{asset}_4H_ohlcv.parquet"' not in blk, (
            "build_breadth_ohlcv writes the PRIMARY series again — that is the "
            "seat-alpha calibration input (P420)")
        assert "kraken_ohlcv_path(asset)" in blk
        assert m.KRAKEN_OHLCV_SUFFIX == "_4H_ohlcv_kraken.parquet"
        assert m.PRIMARY_OHLCV_SUFFIX == "_4H_ohlcv.parquet"
        assert m.kraken_ohlcv_path("XRP").name == "XRP_4H_ohlcv_kraken.parquet"
        # the helper is what the builder uses — not a decoration (P170)
        assert "kraken_ohlcv_path" in inspect.getsource(m.build_breadth_ohlcv)

    def test_behavioural_primary_untouched_kraken_written(self, tmp_path, monkeypatch):
        """Run the REAL builder against a tmp dir with a canned Kraken
        response: the primary must be byte-identical afterwards and the
        extension must exist."""
        import scripts.september_check as m
        monkeypatch.setattr(m, "OHLCV_DIR", tmp_path)
        monkeypatch.setattr(m, "KRAKEN_PAIRS_BREADTH", {"XRP": "XRPUSD"})
        primary = tmp_path / "XRP_4H_ohlcv.parquet"
        pd.DataFrame({"timestamp": pd.to_datetime(["2026-08-01 00:00", "2026-08-01 04:00"]),
                      "open": [1.0, 1.1], "high": [1.2, 1.2], "low": [0.9, 1.0],
                      "close": [1.05, 1.15], "volume": [10.0, 11.0]}).to_parquet(primary, index=False)
        before = primary.read_bytes()

        rows = [[1_754_006_400 + 14400 * k, "1", "1.2", "0.9", str(1.0 + k / 100),
                 "1", "5", 3] for k in range(60)]
        payload = json.dumps({"error": [], "result": {"XRPUSD": rows, "last": 1}}).encode()

        class _Resp(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(m.urllib.request, "urlopen",
                            lambda url, timeout=20: _Resp(payload))
        failed = m.build_breadth_ohlcv()
        assert failed == 0
        assert primary.read_bytes() == before, "the PRIMARY was rewritten (P420)"
        ext = tmp_path / "XRP_4H_ohlcv_kraken.parquet"
        assert ext.exists()
        df = pd.read_parquet(ext)
        assert len(df) == 59, "the in-progress candle must be dropped (P253c)"
        assert df["timestamp"].dt.tz is None

    def test_second_run_merges_the_extension_not_the_primary(self, tmp_path, monkeypatch):
        import scripts.september_check as m
        monkeypatch.setattr(m, "OHLCV_DIR", tmp_path)
        monkeypatch.setattr(m, "KRAKEN_PAIRS_BREADTH", {"BNB": "BNBUSD"})
        ext = tmp_path / "BNB_4H_ohlcv_kraken.parquet"
        pd.DataFrame({"timestamp": pd.to_datetime(["2020-01-01 00:00"]),
                      "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                      "volume": [1.0]}).to_parquet(ext, index=False)
        rows = [[1_754_006_400 + 14400 * k, "1", "1", "1", "1", "1", "5", 3]
                for k in range(60)]
        payload = json.dumps({"error": [], "result": {"BNBUSD": rows, "last": 1}}).encode()

        class _Resp(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(m.urllib.request, "urlopen",
                            lambda url, timeout=20: _Resp(payload))
        assert m.build_breadth_ohlcv() == 0
        df = pd.read_parquet(ext)
        assert len(df) == 60, "history must GROW across runs (P266)"
        assert not (tmp_path / "BNB_4H_ohlcv.parquet").exists()


# ── 2. refresh_ohlcv_4h --assets builds the primary from raw 60m ─────────────

class TestRefreshOhlcvAssets:

    def _mod(self):
        import importlib
        sys.path.insert(0, str(REPO / "training" / "scripts"))
        import refresh_ohlcv_4h as m
        return importlib.reload(m)

    def test_symbols_match_the_fetcher_universe(self):
        """Two maps of one universe drift (P172): the refresh must accept
        exactly what fetch_binance_full can fetch."""
        m = self._mod()
        src = (REPO / "training" / "fetch_binance_full.py").read_text(encoding="utf-8")
        import re
        i = src.index("SYMBOLS = {")
        blk = src[i:src.index("}", i) + 1]
        fetch = dict(re.findall(r'"([A-Z]+)":\s*"([A-Z]+)"', blk))
        assert fetch == m.SYMBOLS

    def test_unknown_asset_refuses(self):
        m = self._mod()
        with pytest.raises(ValueError, match="unknown asset"):
            m.parse_assets("XRP,FOO")
        assert m.main(["--assets", "FOO"]) == 2

    def test_default_roster_is_the_trio(self):
        m = self._mod()
        assert m.parse_assets(None) == ["BTC", "ETH", "SOL"]
        assert m.parse_assets("xrp, bnb") == ["XRP", "BNB"]

    def test_builds_a_breadth_primary_from_raw_60m_same_convention(self, tmp_path, monkeypatch):
        m = self._mod()
        raw_dir = tmp_path / "raw"; raw_dir.mkdir()
        out_dir = tmp_path / "out"
        monkeypatch.setattr(m, "RAW_DIR", raw_dir)
        monkeypatch.setattr(m, "OUT_DIR", out_dir)
        ts = pd.date_range("2026-01-01 00:00", periods=48, freq="h")
        raw = pd.DataFrame({"timestamp": ts, "open": range(48), "high": range(48),
                            "low": range(48), "close": [float(x) for x in range(48)],
                            "volume": [1.0] * 48})
        raw.to_parquet(raw_dir / "XRP_60m.parquet", index=False)
        df = m.build("XRP", daily_extension=False)
        assert df is not None and len(df) == 12
        # origin="start_day": bins at 00/04/08/... and close = last hour of the bin
        assert df["timestamp"].iloc[1] == pd.Timestamp("2026-01-01 04:00")
        assert df["close"].iloc[0] == 3.0
        assert (out_dir / "XRP_4H_ohlcv.parquet").exists()
        assert not (out_dir / "XRP_4H_ohlcv_kraken.parquet").exists(), (
            "the refresh must never write the Kraken EXTENSION file")

    def test_the_resample_convention_is_unchanged(self):
        src = (REPO / "training" / "scripts" / "refresh_ohlcv_4h.py").read_text(encoding="utf-8")
        assert '.resample("4h", origin="start_day")' in src


# ── 3. seat_alpha_calibration --verify: median, refusal, provenance ───────────

class TestVerifyReadsTheMedianAndRefusesUnlisted:

    def _run(self, argv, tmp_path, cells_by_asset):
        """Drive main() with calibrate() stubbed so the tests need no
        operator-local parquets (P213) — the LOGIC under test is the
        comparison, not the lab."""
        import training.seat_alpha_calibration as m
        stub = lambda asset, series="book", ledger=True: dict(cells_by_asset[asset])  # noqa: E731
        m_calibrate = m.calibrate
        m.calibrate = stub
        m_stamp = m.input_stamp
        m.input_stamp = lambda asset: {"closes": {"sha256": "x", "rows": 1}}
        try:
            return m.main(argv + ["--no-ledger", "--report", str(tmp_path / "r.json")])
        finally:
            m.calibrate = m_calibrate
            m.input_stamp = m_stamp

    def test_exact_reproduction_passes(self, tmp_path):
        from core.seat_alpha import REGIMEBOOK_ALPHA_BY_ERA as T
        cells = {a: dict(T[a]) for a in ("XRP", "BNB")}
        assert self._run(["--verify", "--assets", "XRP,BNB"], tmp_path, cells) == 0

    def test_a_drifted_MEDIAN_is_named_even_when_the_cells_are_within_tolerance(self, tmp_path, monkeypatch):
        """The value the gate reads is the per-RT median; drift it in the
        shipped table while the cells reproduce -> exit 3 naming MEDIAN."""
        import core.seat_alpha as sa
        from core.seat_alpha import REGIMEBOOK_ALPHA_BY_ERA as T
        monkeypatch.setitem(sa.REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP, "XRP", 99.9)
        cells = {"XRP": dict(T["XRP"])}
        rc = self._run(["--verify", "--assets", "XRP"], tmp_path, cells)
        assert rc == 3
        rep = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
        assert any(d["cell"] == "MEDIAN" for d in rep["drift"])

    def test_a_drifted_cell_is_still_named(self, tmp_path):
        from core.seat_alpha import REGIMEBOOK_ALPHA_BY_ERA as T
        cells = {"BNB": dict(T["BNB"])}
        cells["BNB"]["validation"] += 14.5    # the live incident: 36.6 -> 51.1
        assert self._run(["--verify", "--assets", "BNB"], tmp_path, cells) == 3

    def test_unlisted_asset_refuses_instead_of_passing_vacuously(self, tmp_path):
        cells = {"DOGE": {"pre_design": 1.0, "design": 2.0, "validation": 3.0}}
        assert self._run(["--verify", "--assets", "DOGE"], tmp_path, cells) == 2
        assert not (tmp_path / "r.json").exists(), (
            "a refusal must not leave a report that reads as a run")

    def test_default_assets_are_the_shipped_tables_keys(self, tmp_path):
        from core.seat_alpha import (REGIMEBOOK_ALPHA_BY_ERA as T,
                                     REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP as RT)
        cells = {a: dict(T[a]) for a in RT}
        # NOTE: an asset whose shipped series is not "book" (ETH=donchian since
        # P419) is verified against ITS series; the stub returns the shipped
        # cells for every series, so the default roster passes end to end.
        assert self._run(["--verify"], tmp_path, cells) == 0
        rep = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
        assert set(rep["assets"]) == set(RT)

    def test_report_carries_input_sha256_and_rows(self, tmp_path):
        """The real stamp on a real file: a mutated input changes the hash."""
        import training.seat_alpha_calibration as m
        import training.funding_legs_lab as lab
        pdir = tmp_path / "p"; fdir = tmp_path / "f"
        pdir.mkdir(); fdir.mkdir()
        closes = pdir / "XRP_4H_ohlcv.parquet"
        pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01"]), "close": [1.0]}).to_parquet(closes, index=False)
        pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01"]), "funding_close": [0.0]}).to_parquet(fdir / "XRP_funding_1d.parquet", index=False)
        saved = lab.PRICE_DIR, lab.FUNDING_DIR
        lab.PRICE_DIR, lab.FUNDING_DIR = pdir, fdir
        try:
            s1 = m.input_stamp("XRP")
            assert s1["closes"]["rows"] == 1 and len(s1["closes"]["sha256"]) == 64
            assert s1["funding"]["rows"] == 1
            pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                          "close": [1.0, 2.0]}).to_parquet(closes, index=False)
            s2 = m.input_stamp("XRP")
            assert s2["closes"]["sha256"] != s1["closes"]["sha256"]
            assert s2["closes"]["rows"] == 2
        finally:
            lab.PRICE_DIR, lab.FUNDING_DIR = saved

    def test_the_producer_ledgers_its_validation_read(self):
        src = code_only(REPO / "training" / "seat_alpha_calibration.py")
        i = src.index("def calibrate(")
        blk = src[i:i + 2500]
        assert "record_window_usage(" in blk, (
            "the per-asset validation edge feeds a live gate constant; the read "
            "must be ledgered (P420)")


class TestLedgerBackfill:
    """[P420] The five unledgered validation reads were backfilled, one record
    per read that already happened (the P382 precedent), so the multiplicity
    discount every later caller is told about sees them."""

    @pytest.fixture(autouse=True)
    def _ledger(self):
        from training.splits import LEDGER_PATH
        if not LEDGER_PATH.exists():
            pytest.skip("window ledger absent")
        self.recs = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))["records"]

    @pytest.mark.parametrize("experiment,assets", [
        ("conviction_channel_lab:p417", {"BTC", "ETH", "SOL"}),
        ("conviction_sizing_lab:ws2", {"BTC", "ETH"}),
        ("gate_probes_lab:ab", {"BTC", "ETH", "SOL"}),
        ("seat_alpha_calibration:book", {"BTC", "ETH", "SOL", "XRP", "BNB"}),
        ("ridge_16h_pooled_check:p403", {"BTC", "ETH", "SOL"}),
    ])
    def test_each_read_is_ledgered_as_a_validation_purpose(self, experiment, assets):
        from training.splits import _is_validation_purpose
        got = [r for r in self.recs if r["experiment"] == experiment]
        assert assets <= {r["asset"] for r in got}, experiment
        assert all(_is_validation_purpose(r["purpose"]) for r in got), (
            f"{experiment}: a validation read that does not COUNT as one "
            f"dodges the multiplicity discount (P287)")

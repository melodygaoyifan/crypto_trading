"""
================================================================================
HMATS [P301] - breadth funding + persisted v5.1 warmups
================================================================================

Operator: "do both" - fetch the breadth funding so the derivatives exam can
price carry, and persist the v5.1 warmups so three "dead" agents become
measurable.
================================================================================
"""

import importlib
import os
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FUNDING_DIR = REPO / "training" / "training_data" / "coinglass_history"
BREADTH = ("XRP", "ADA", "LTC", "DOGE", "BNB")


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


class TestBreadthFundingIsFetchable:
    """The exam could not price carry for these five, and carry is the
    dominant drag on a long-biased perp book (P296: -59.7% over six years)."""

    def test_fetcher_knows_the_breadth_symbols(self):
        from training.scripts.fetch_binance_funding import SYMBOLS
        for a in BREADTH:
            assert a in SYMBOLS, f"{a} has no Binance symbol mapping"
            assert SYMBOLS[a].endswith("USDT")

    def test_fetcher_refuses_an_unknown_asset(self):
        """[P291] a fetcher that accepts any string writes a file for an asset
        that does not exist and reports success."""
        from training.scripts.fetch_binance_funding import main
        assert main(["--assets", "NOTACOIN"]) == 2

    def test_fetcher_merges_rather_than_overwrites(self):
        """[P266] a partial outage must not silently replace a complete
        series with a truncated one."""
        src = _src(REPO / "training" / "scripts" / "fetch_binance_funding.py")
        assert "drop_duplicates(\"timestamp\", keep=\"last\")" in src
        assert "if out.exists():" in src

    @pytest.mark.skipif(not FUNDING_DIR.exists(), reason="no local archives")
    def test_the_five_archives_landed_with_major_span(self):
        import pandas as pd
        spans = {}
        for a in BREADTH + ("BTC",):
            p = FUNDING_DIR / f"{a}_funding_1d.parquet"
            if not p.exists():
                pytest.skip(f"{a} archive not fetched on this machine")
            spans[a] = len(pd.read_parquet(p))
        # every breadth asset must reach the majors' depth, or the exam is
        # comparing a 6-year book against a partial one
        assert all(spans[a] >= spans["BTC"] * 0.95 for a in BREADTH), spans


class TestWarmupSurvivesRestart:
    """MEASURED on the server: funding reported history_warmup(1/12) and
    regime_warmup(1/30). Those counters read 1 - not 11, not 29 - because the
    deques start empty on every construction, so a warmup needing ~4 and ~10
    days of uninterrupted uptime never completed between deploys. P154/P148/
    P150 class: state that re-arms on restart is not state."""

    @pytest.fixture()
    def isolated(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        import strategies._warmup_state as ws
        importlib.reload(ws)
        import strategies.funding_rate_v5_1 as F
        importlib.reload(F)
        return F

    def test_history_survives_a_restart(self, isolated):
        F = isolated
        s = F.FundingRateMeanReversionStrategy()
        for i in range(8):
            s.evaluate("BTC", {"funding_rate_8h": 0.0001 * (i + 1)})
        assert len(s._history["BTC"]) == 8

        fresh = F.FundingRateMeanReversionStrategy()   # a restart
        assert len(fresh._history.get("BTC", [])) == 8, (
            "the warmup restarted at 0 — the counter will read 1/12 forever "
            "at this deploy cadence"
        )
        out = fresh.evaluate("BTC", {"funding_rate_8h": 0.0009})
        assert "9/12" in out.reason, out.reason

    def test_the_regime_strategy_persists_too(self, isolated):
        F = isolated
        s = F.FundingRatePostETFRegimeStrategy()
        for i in range(5):
            s.evaluate("ETH", {"funding_rate_8h": 0.0002 * (i + 1)})
        assert len(F.FundingRatePostETFRegimeStrategy()._history.get("ETH", [])) == 5

    def test_the_two_strategies_do_not_share_a_file(self, isolated):
        """Distinct maxlens and distinct thresholds — one file would let the
        60-day regime history feed the 14-day mean-reversion z-score."""
        F = isolated
        assert (F.FundingRateMeanReversionStrategy._WARMUP_NAME
                != F.FundingRatePostETFRegimeStrategy._WARMUP_NAME)

    def test_a_missing_state_file_is_a_cold_start_not_an_error(self, isolated):
        F = isolated
        s = F.FundingRateMeanReversionStrategy()
        assert s._history == {}

    def test_a_corrupt_state_file_degrades_to_cold_start(self, isolated, tmp_path):
        import strategies._warmup_state as ws
        d = tmp_path / "v5_1_warmup"
        d.mkdir(parents=True, exist_ok=True)
        (d / "funding_mean_reversion.json").write_text("{not json",
                                                       encoding="utf-8")
        assert ws.load("funding_mean_reversion") == {}

    def test_a_stale_history_is_dropped_rather_than_trusted(self, isolated,
                                                            tmp_path):
        """A funding distribution from two months ago is not the current
        regime; restoring it would give the z-score a scale nobody measured."""
        import json
        import strategies._warmup_state as ws
        d = tmp_path / "v5_1_warmup"
        d.mkdir(parents=True, exist_ok=True)
        (d / "stale.json").write_text(json.dumps({
            "version": "v5_1_warmup_v1",
            "saved_ts": 0.0,                      # 1970
            "series": {"BTC": [0.1, 0.2, 0.3]},
        }), encoding="utf-8")
        assert ws.load("stale") == {}

    def test_nan_values_never_enter_the_history(self, isolated, tmp_path):
        import json
        import strategies._warmup_state as ws
        import time
        d = tmp_path / "v5_1_warmup"
        d.mkdir(parents=True, exist_ok=True)
        (d / "n.json").write_text(json.dumps({
            "version": "v5_1_warmup_v1",
            "saved_ts": time.time(),
            "series": {"BTC": [0.1, float("nan"), 0.3]},
        }), encoding="utf-8")
        assert ws.load("n") == {"BTC": [0.1, 0.3]}

    def test_persisting_never_raises_into_the_tick(self, isolated, monkeypatch):
        """A warmup that cannot be saved must not take a tick down."""
        F = isolated
        import strategies.funding_rate_v5_1 as mod

        def boom(*a, **k):
            raise OSError("disk full")
        monkeypatch.setattr(mod, "_warmup_save", boom)
        s = F.FundingRateMeanReversionStrategy()
        out = s.evaluate("BTC", {"funding_rate_8h": 0.0001})
        assert out is not None


class TestNoSilentIndexRealignment:
    """[P301] `pd.Series(df["col"], index=other)` ALIGNS on the Series' own
    index instead of replacing it, so a non-overlapping index yields all-NaN.

    Measured: this turned the breadth exam's entire subject - the funding
    carry - into exactly 0.0% for all eight assets, and `fillna(0.0)` made it
    look like a measurement. It survived because a zero is a plausible-looking
    number; it was caught only because a 0.0 six-year funding bill is not.

    The dangerous shape is specifically a PANDAS expression (`df[col]`, or an
    `.astype()` chain on one) passed as `data` alongside `index=`. A plain
    list or a numpy array has no index and is safe, so the scan is written to
    ignore those rather than cry wolf.
    """

    @staticmethod
    def _is_pandas_expr(node):
        import ast
        # df["col"] / df.loc[...] — a subscript on a name or attribute
        if isinstance(node, ast.Subscript):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("to_numpy", "values", "tolist", "to_list"):
                return False        # explicitly de-indexed: the correct form
            if node.func.attr in ("astype", "fillna", "round", "abs", "copy"):
                return TestNoSilentIndexRealignment._is_pandas_expr(
                    node.func.value)
        return False

    def _scan(self, src):
        import ast
        out = []
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return out
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            name = (f.attr if isinstance(f, ast.Attribute)
                    else (f.id if isinstance(f, ast.Name) else None))
            if name not in ("Series", "DataFrame") or not n.args:
                continue
            if not any(k.arg == "index" for k in n.keywords):
                continue
            if self._is_pandas_expr(n.args[0]):
                out.append(n.lineno)
        return out

    def test_the_scan_bites_on_the_exact_bug_that_happened(self):
        """[P174] anti-vacuity, using the real line that produced the zero."""
        bad = 'pd.Series(df["funding_close"].astype(float), index=idx)\n'
        assert self._scan(bad) == [1]
        good = ('pd.Series(df["funding_close"].to_numpy(dtype=float), index=idx)\n'
                "pd.Series(vals, index=idx)\n"
                "pd.Series([1, 2, 3], index=idx)\n"
                "pd.Series(np.zeros(3), index=idx)\n")
        assert self._scan(good) == []

    def test_no_such_site_survives_anywhere(self):
        import io
        import os
        skip = {"venv", "archive", "__pycache__", ".git", "node_modules",
                "models", "training_data"}
        offenders = []
        for dp, dn, fn in os.walk(REPO):
            dn[:] = [d for d in dn if d not in skip]
            for f in fn:
                if not f.endswith(".py"):
                    continue
                p = Path(dp) / f
                try:
                    src = io.open(p, encoding="utf-8").read()
                except (UnicodeDecodeError, OSError):
                    continue
                for ln in self._scan(src):
                    offenders.append(f"{p.relative_to(REPO)}:{ln}")
        assert not offenders, (
            "pandas will ALIGN on the source index here, not replace it — "
            "a non-overlapping index yields silent all-NaN:\n  "
            + "\n  ".join(offenders)
        )


class TestMicrostructureStateSurvivesRestart:
    """[P302] Of 6,168 recorded microstructure rows exactly ONE was
    directional. Both of its silences came from RAM-only state, not from a
    quiet market:

        no_prev_price          `_last_price` starts empty every process, so
                               the return since the last tick cannot exist
        z_below_threshold(+0.00)  the lambda history needs >= max(8, 0.3*30)
                               samples and restarts at 0

    At one sample per 4H tick that is ~1.5 days of uninterrupted uptime.
    Same class as the P301 funding warmups and P154/P148/P150 before them.
    """

    @pytest.fixture()
    def isolated(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        import strategies._warmup_state as ws
        importlib.reload(ws)
        import strategies.microstructure_v5_1 as M
        importlib.reload(M)
        return M

    @staticmethod
    def _md(px):
        return {"current_price": px, "order_book_imbalance": 0.1,
                "spread_bps": 3.0}

    def test_previous_price_survives_a_restart(self, isolated):
        M = isolated
        s = M.KyleLambdaStrategy()
        assert s.evaluate("BTC", self._md(64000)).reason == "no_prev_price"
        fresh = M.KyleLambdaStrategy()
        assert fresh._last_price.get("BTC") == 64000, (
            "every restart would otherwise burn a tick per asset on "
            "no_prev_price"
        )
        assert fresh.evaluate("BTC", self._md(64100)).reason != "no_prev_price"

    def test_lambda_history_survives_a_restart(self, isolated):
        M = isolated
        s = M.KyleLambdaStrategy()
        for i in range(10):
            s.evaluate("BTC", self._md(64000 + i * 40))
        n = len(s._lambda_history["BTC"])
        assert n >= 8
        assert len(M.KyleLambdaStrategy()._lambda_history.get("BTC", [])) == n

    def test_the_price_key_cannot_collide_with_an_asset(self, isolated):
        """`_last_price` and the lambda history share one file; the price
        rows are suffixed so an asset literally named like the suffix cannot
        overwrite a history."""
        M = isolated
        s = M.KyleLambdaStrategy()
        for i in range(3):
            s.evaluate("BTC", self._md(64000 + i * 40))
        s._persist()
        import strategies._warmup_state as ws
        blob = ws.load("kyle_lambda")
        assert "BTC::price" in blob and "BTC" in blob
        assert blob["BTC::price"] != blob["BTC"]

    def test_a_failed_save_never_raises_into_the_tick(self, isolated,
                                                      monkeypatch):
        M = isolated
        import strategies.microstructure_v5_1 as mod

        def boom(*a, **k):
            raise OSError("disk full")
        monkeypatch.setattr(mod, "_warmup_save", boom)
        s = M.KyleLambdaStrategy()
        assert s.evaluate("BTC", self._md(64000)) is not None

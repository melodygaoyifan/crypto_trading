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

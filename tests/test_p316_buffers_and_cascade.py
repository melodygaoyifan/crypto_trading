"""[P316] Two RAM-only rolling buffers persisted, and the cascade window
disarmed on better data.

Both halves came out of one question — "why are these items waiting?" — and
in both cases the wait was not a data fact.
"""
from __future__ import annotations

import collections
import io
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return io.open(REPO / rel, encoding="utf-8").read()


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """P294: construct the state, never inherit it. Without this the tests
    read whatever the engine last wrote on this machine."""
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    yield


# ---------------------------------------------------------------------------
# 1. the buffers
# ---------------------------------------------------------------------------
class TestRollingBufferPersistence:
    def _restorer(self):
        from data_mgmt.market_data_pipeline import MarketDataPipeline as M

        class _Fake:
            _BUFFER_MAX_AGE_SEC = M._BUFFER_MAX_AGE_SEC
            _restore_rolling_buffer = M._restore_rolling_buffer
        return _Fake()

    def _bufs(self, maxlen=42):
        return {a: collections.deque(maxlen=maxlen)
                for a in ("BTC", "ETH", "SOL")}

    def test_a_restored_ofi_buffer_clears_the_five_sample_gate(self):
        """The gate is `len(_ofi_buf) >= 5` at ONE sample per 4H tick — 20
        hours of uptime. RAM-only, it restarted at zero on every deploy, so
        `ofi_zscore` was structurally 0.0 forever."""
        from strategies._warmup_state import save
        save("ofi_history", {"BTC": [0.1, -0.2, 0.3, 0.05, -0.15, 0.22]})
        b = self._bufs()
        self._restorer()._restore_rolling_buffer("ofi_history", b)
        assert len(b["BTC"]) >= 5
        assert len(b["ETH"]) == 0, "an absent asset must restore nothing"

    def test_restore_never_exceeds_the_deque_bound(self):
        from strategies._warmup_state import save
        save("ofi_history", {"BTC": [0.01 * i for i in range(200)]})
        b = self._bufs(maxlen=42)
        self._restorer()._restore_rolling_buffer("ofi_history", b)
        assert len(b["BTC"]) == 42

    def test_a_missing_file_is_a_cold_start_not_a_crash(self):
        b = self._bufs()
        self._restorer()._restore_rolling_buffer("no_such_buffer", b)
        assert all(len(d) == 0 for d in b.values())

    def test_corrupt_state_degrades_to_todays_behaviour(self, tmp_path):
        (tmp_path / "v5_1_warmup").mkdir(parents=True, exist_ok=True)
        (tmp_path / "v5_1_warmup" / "ofi_history.json").write_text(
            "{not json", encoding="utf-8")
        b = self._bufs()
        self._restorer()._restore_rolling_buffer("ofi_history", b)
        assert all(len(d) == 0 for d in b.values())

    def test_persistence_reuses_the_p301_helper_rather_than_a_second_copy(self):
        """P172: one atomic writer, one staleness rule, one set of fail
        directions. A second implementation is how two 'the same' stores
        start disagreeing."""
        src = _src("data_mgmt/market_data_pipeline.py")
        i = src.index("def _restore_rolling_buffer")
        blk = src[i:i + 2500]
        assert "from strategies._warmup_state import load" in blk
        assert "os.replace" not in blk, "a second atomic writer was rolled here"

    def test_both_buffers_are_restored_and_persisted(self):
        src = _src("data_mgmt/market_data_pipeline.py")
        for name in ("ofi_history", "depth_history"):
            assert f'self._restore_rolling_buffer("{name}"' in src, name
            assert f'("{name}", getattr(self, "_{name}", None))' in src, name

    def test_the_persist_call_runs_after_the_buffers_are_appended(self):
        """Persisting before the append writes yesterday's buffer forever —
        the P234 ordering class, which has already bitten this session."""
        src = _src("data_mgmt/market_data_pipeline.py")
        append = src.index("self._ofi_history[asset].append(order_book_imbalance)")
        persist = src.index("self._persist_rolling_buffers()")
        assert persist > append

    def test_a_failed_save_cannot_raise_into_a_tick(self):
        src = _src("data_mgmt/market_data_pipeline.py")
        i = src.index("def _persist_rolling_buffers")
        blk = src[i:i + 1200]
        assert "except Exception" in blk


# ---------------------------------------------------------------------------
# 2. the cascade correction
# ---------------------------------------------------------------------------
class TestCascadeRecalibration:
    def _live(self):
        return json.loads(
            (REPO / "configs" / "live_high_risk.json").read_text(
                encoding="utf-8"))

    def test_the_window_is_disarmed(self):
        """P311 armed it on 85 observations/asset; six months of per-4H-bar
        liquidation history (1,039 bars) says the same multiple fires on
        11-14% of bars, not 1-2%, and a 5x spike carries no forward
        information (|t| <= 0.93 on all three)."""
        assert "cascade_real_liquidation_window" not in self._live()

    def test_the_disarming_records_why(self):
        note = self._live().get("_p316_cascade_disarmed_note", "")
        for token in ("11-14%", "1,039", "PRECONDITION"):
            assert token in note, f"the note lost {token!r}"

    def test_the_multiple_is_calibrated_on_the_larger_sample(self):
        """20x is where the firing rate is 1-2% on the 6-month basis, and
        p99 is 20-27x on all three assets — so one multiple still fits."""
        from risk.cascade_exhaustion_governor import CascadeExhaustionConfig
        c = CascadeExhaustionConfig()
        assert c.cascade_detect_liq_multiple == 20.0
        assert c.cascade_accelerate_liq_multiple == 50.0

    def test_the_structural_fix_survives_the_disarming(self):
        """The single-dollar-threshold defect (routine on BTC, unreachable on
        SOL) is what the multiple exists to fix, and it is orthogonal to
        whether the window is armed."""
        from risk.cascade_exhaustion_governor import (
            get_cascade_exhaustion_governor, reset_cascade_exhaustion_governor)
        reset_cascade_exhaustion_governor()
        try:
            g = get_cascade_exhaustion_governor()
            btc, _ = g.effective_liq_thresholds(79_629_742 / 24)
            sol, _ = g.effective_liq_thresholds(6_366_840 / 24)
            assert btc > sol * 5, (
                "the thresholds are no longer scaled to each asset's own rate")
        finally:
            reset_cascade_exhaustion_governor()

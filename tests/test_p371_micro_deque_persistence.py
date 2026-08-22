"""[P371] The `micro` agent's sample deques survive a restart.

P316/P370 measured `micro` neutral (`insufficient_samples`, dq 0.7-0.8) after
every restart: its `min_samples=5` gate counts spread_history +
imbalance_history + the lag detector's per-exchange price_history, all of
which were per-process deques fed one sample per 4H tick. The P301/P316 class,
one more location. The fix reuses strategies/_warmup_state (P172) — restore on
construction, persist after each append, every failure path a logged cold
start.

Every test here CONSTRUCTS its state under HMATS_DATA_DIR=tmp_path (P294: never
inherit it from the machine) and drives the real agent.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests._guard_pins import assert_guard_live  # noqa: E402
from tests._source_scan import code_only  # noqa: E402

AGENT_SRC = REPO / "agents" / "microstructure_agent.py"


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """P294: construct the state, never inherit it."""
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    from agents import microstructure_agent as ma
    ma.reset_microstructure_agent()
    yield
    ma.reset_microstructure_agent()


def _agent(**kw):
    from agents.microstructure_agent import (MicrostructureArbitrageAgent,
                                             MicrostructureConfig)
    return MicrostructureArbitrageAgent(MicrostructureConfig(**kw))


def _feed(agent, asset="BTC", rounds=3, base=100.0):
    """One 'tick' = a follower and a leader snapshot, like _ingest_market_data."""
    now_ms = time.time() * 1000.0
    for i in range(rounds):
        agent.update_exchange(asset, "kraken", bid=base + i, ask=base + i + 0.2,
                              bid_size=5, ask_size=4, timestamp_ms=now_ms + i)
        agent.update_exchange(asset, "binance", bid=base + i - 0.1,
                              ask=base + i + 0.1, bid_size=6, ask_size=4,
                              timestamp_ms=now_ms + i)


def _state_file(tmp_path) -> Path:
    from strategies._warmup_state import state_path
    from agents.microstructure_agent import MicrostructureArbitrageAgent as A
    p = Path(state_path(A._WARMUP_STATE_NAME))
    assert str(p).startswith(str(tmp_path)), (
        "state must live under the fixture's HMATS_DATA_DIR, never the repo")
    return p


def _md(base=100.0):
    """A market_data dict that _ingest_market_data turns into two snapshots."""
    return {"bid": base, "ask": base + 0.2, "bid_size": 5, "ask_size": 4,
            "binance_bid": base - 0.1, "binance_ask": base + 0.1,
            "binance_bid_size": 6, "binance_ask_size": 4}


# ---------------------------------------------------------------------------
# 1. round trip through disk on a FRESH object
# ---------------------------------------------------------------------------
class TestRoundTrip:
    def test_samples_survive_into_a_fresh_object(self, tmp_path):
        a1 = _agent()
        _feed(a1, "BTC", rounds=3)
        st1 = a1._per_asset["BTC"]
        assert _state_file(tmp_path).exists()

        a2 = _agent()                       # a fresh process, in effect
        assert "BTC" in a2._per_asset, "restore must rebuild the asset state"
        st2 = a2._per_asset["BTC"]
        assert list(st2.spread_history) == list(st1.spread_history)
        assert list(st2.imbalance_history) == list(st1.imbalance_history)
        for ex in ("kraken", "binance"):
            assert [tuple(map(float, t)) for t in st2.lag_detector.price_history[ex]] == \
                   [tuple(map(float, t)) for t in st1.lag_detector.price_history[ex]]
            assert st2.lag_detector.price_history[ex].maxlen == \
                   st1.lag_detector.price_history[ex].maxlen

    def test_the_defect_a_restarted_agent_no_longer_reads_insufficient_samples(self):
        """THE BEHAVIOURAL PIN. A cold agent's first tick is `insufficient_samples`
        (4 < 5 — exactly the live dq 0.8). After a restart with persisted
        samples, the same first tick is past the gate. Decision logic untouched:
        the gate still says 5, it just sees the samples it had earned."""
        cold = _agent()
        p = cold.generate_signal("BTC", market_data=_md())
        assert p["diagnostics"].get("reason") == "insufficient_samples", p
        assert p["micro_data_quality"] == pytest.approx(0.8)

        # cold now holds 4 samples (and persisted them). A "restart":
        warm = _agent()
        p2 = warm.generate_signal("BTC", market_data=_md())
        assert p2["diagnostics"].get("reason") != "insufficient_samples", p2

    def test_persist_happens_after_each_append(self, tmp_path):
        a = _agent()
        lens = []
        for i in range(3):
            _feed(a, "ETH", rounds=1, base=50.0 + i)
            payload = json.loads(_state_file(tmp_path).read_text(encoding="utf-8"))
            lens.append(len(payload["series"]["spread::ETH"]))
        assert lens == sorted(lens) and lens[-1] > lens[0], lens

    def test_restore_does_not_exceed_the_deque_bounds(self, tmp_path):
        """A restored series longer than maxlen keeps the TAIL only."""
        a1 = _agent(imbalance_lookback=3)
        _feed(a1, "SOL", rounds=10)
        a2 = _agent(imbalance_lookback=3)
        assert len(a2._per_asset["SOL"].imbalance_history) == 3
        assert a2._per_asset["SOL"].imbalance_history.maxlen == 3
        assert len(a2._per_asset["SOL"].spread_history) <= 100

    def test_per_asset_isolation_survives_the_round_trip(self):
        a1 = _agent()
        _feed(a1, "BTC", rounds=2)
        _feed(a1, "SOL", rounds=4, base=80.0)
        a2 = _agent()
        assert len(a2._per_asset["BTC"].spread_history) == len(
            a1._per_asset["BTC"].spread_history)
        assert len(a2._per_asset["SOL"].spread_history) == len(
            a1._per_asset["SOL"].spread_history)
        assert "ETH" not in a2._per_asset


# ---------------------------------------------------------------------------
# 2. every failure path is a logged cold start, never a raise, never a
#    fabricated history
# ---------------------------------------------------------------------------
class TestColdStartPaths:
    def test_no_file_is_a_logged_cold_start(self, caplog):
        with caplog.at_level(logging.INFO, logger="agents.microstructure_agent"):
            a = _agent()
        assert a._per_asset == {}
        assert any("cold start" in r.getMessage() for r in caplog.records), (
            "a cold start must announce itself — a silent return is how a "
            "warmup becomes a permanent NEUTRAL (P316)")

    def test_a_stale_file_restores_nothing(self, tmp_path, caplog):
        from agents.microstructure_agent import MicrostructureArbitrageAgent as A
        a1 = _agent()
        _feed(a1, "BTC", rounds=3)
        p = _state_file(tmp_path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["saved_ts"] = time.time() - A._WARMUP_MAX_AGE_SEC - 3600.0
        p.write_text(json.dumps(payload), encoding="utf-8")
        with caplog.at_level(logging.INFO):
            a2 = _agent()
        assert a2._per_asset == {}, "a stale history is not the current regime"
        assert any("cold start" in r.getMessage() for r in caplog.records)

    def test_a_fresh_file_just_inside_the_bound_restores(self, tmp_path):
        from agents.microstructure_agent import MicrostructureArbitrageAgent as A
        a1 = _agent()
        _feed(a1, "BTC", rounds=3)
        p = _state_file(tmp_path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["saved_ts"] = time.time() - A._WARMUP_MAX_AGE_SEC + 3600.0
        p.write_text(json.dumps(payload), encoding="utf-8")
        a2 = _agent()
        assert len(a2._per_asset["BTC"].spread_history) > 0

    def test_a_version_mismatch_restores_nothing(self, tmp_path):
        a1 = _agent()
        _feed(a1, "BTC", rounds=3)
        p = _state_file(tmp_path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["version"] = "some_other_version"
        p.write_text(json.dumps(payload), encoding="utf-8")
        a2 = _agent()
        assert a2._per_asset == {}

    def test_a_corrupt_file_restores_nothing_and_does_not_raise(self, tmp_path):
        a1 = _agent()
        _feed(a1, "BTC", rounds=1)
        _state_file(tmp_path).write_text("{not json", encoding="utf-8")
        a2 = _agent()
        assert a2._per_asset == {}

    def test_a_lag_ts_px_length_mismatch_drops_that_exchange_only(self, tmp_path, caplog):
        """Pairing the wrong timestamp with a price would be a fabricated
        observation; the exchange is dropped, the float series still restore."""
        a1 = _agent()
        _feed(a1, "BTC", rounds=3)
        p = _state_file(tmp_path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["series"]["lag_px::BTC::kraken"].pop()
        p.write_text(json.dumps(payload), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            a2 = _agent()
        st = a2._per_asset["BTC"]
        assert "kraken" not in st.lag_detector.price_history
        assert len(st.lag_detector.price_history["binance"]) == 3
        assert len(st.spread_history) > 0
        assert any("mismatch" in r.getMessage() for r in caplog.records)

    def test_unknown_keys_are_ignored(self, tmp_path):
        a1 = _agent()
        _feed(a1, "BTC", rounds=1)
        p = _state_file(tmp_path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["series"]["taker::BTC"] = [1.0, 2.0]
        payload["series"]["garbage"] = [3.0]
        p.write_text(json.dumps(payload), encoding="utf-8")
        a2 = _agent()
        assert len(a2._per_asset["BTC"].spread_history) == len(
            a1._per_asset["BTC"].spread_history)

    def test_a_restore_that_raises_is_a_cold_start_not_a_crash(self, monkeypatch, caplog):
        import strategies._warmup_state as ws

        def boom(*a, **k):
            raise RuntimeError("disk on fire")
        monkeypatch.setattr(ws, "load", boom)
        with caplog.at_level(logging.WARNING):
            a = _agent()
        assert a._per_asset == {}
        assert any("restore failed" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 3. a save failure cannot raise into a tick
# ---------------------------------------------------------------------------
class TestSaveFailure:
    def test_a_raising_save_helper_cannot_break_update_exchange(self, monkeypatch):
        import strategies._warmup_state as ws

        def boom(*a, **k):
            raise RuntimeError("disk full")
        monkeypatch.setattr(ws, "save", boom)
        a = _agent()
        _feed(a, "BTC", rounds=2)          # must not raise
        assert len(a._per_asset["BTC"].spread_history) > 0

    def test_an_unwritable_state_dir_cannot_break_a_tick(self, monkeypatch):
        import strategies._warmup_state as ws

        def boom(*a, **k):
            raise PermissionError("read-only volume")
        monkeypatch.setattr(ws.os, "makedirs", boom)
        a = _agent()
        _feed(a, "BTC", rounds=1)
        p = a.generate_signal("BTC", market_data=_md())
        assert "error" not in p["diagnostics"], p

    def test_nothing_is_written_when_there_are_no_samples(self, tmp_path):
        _agent()
        assert not _state_file(tmp_path).exists()


# ---------------------------------------------------------------------------
# 4. what must NOT have moved
# ---------------------------------------------------------------------------
class TestDecisionLogicUntouched:
    def test_min_samples_default_and_gate_are_unchanged(self):
        from agents.microstructure_agent import MicrostructureConfig
        assert MicrostructureConfig().min_samples == 5
        src = code_only(AGENT_SRC)
        assert_guard_live(src, "samples < self.config.min_samples",
                          why="P371 moves WHERE samples live, not the gate")
        assert "len(st.spread_history) + len(st.imbalance_history)" in src

    def test_max_age_matches_the_pipeline_buffer_bound(self):
        from agents.microstructure_agent import MicrostructureArbitrageAgent as A
        from data_mgmt.market_data_pipeline import MarketDataPipeline as M
        assert A._WARMUP_MAX_AGE_SEC == M._BUFFER_MAX_AGE_SEC == 7 * 24 * 3600.0

    def test_the_lag_deque_bound_is_single_sourced(self):
        from agents.microstructure_agent import LagDetector
        assert LagDetector.PRICE_HISTORY_MAXLEN == 500
        src = code_only(AGENT_SRC)
        assert "deque(maxlen=500)" not in src, (
            "the restore path and update() must build the deque from ONE "
            "constant or they drift")

    def test_restore_runs_at_construction_before_the_first_tick(self):
        src = code_only(AGENT_SRC)
        i = src.index("class MicrostructureArbitrageAgent")
        j = src.index("def __init__", i)
        k = src.index("def _state", j)
        assert "self._restore_warmup_samples()" in src[j:k]

    def test_persist_is_wired_to_the_append_site(self):
        src = code_only(AGENT_SRC)
        i = src.index("def update_exchange(")
        j = src.index("def _update_cross_state", i)
        assert "self._persist_warmup_samples()" in src[i:j]

"""[P376] The whale seat is a live decider (whale_seat_mode:enforce, P293j) and
must name itself in primary_strategy, or the strategy-aging tracker + attribution
record its decisions under the stale 'trend_following' (Agent-3 wire-up finding).
Mirrors the P313 regimebook fix + P317 vocabulary add.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_whale_seat_branch_writes_primary_strategy_whale():
    # the deciding branch must stamp primary_strategy into BOTH dicts, like the
    # regimebook seat (P313). Pinned near the whale sleeve bridge so a future
    # edit that drops it is caught.
    src = (REPO / "main.py").read_text(encoding="utf-8")
    assert 'market_data["primary_strategy"] = "whale"' in src
    assert 'agent_signals["primary_strategy"] = "whale"' in src


def test_aging_tracker_knows_whale():
    # the tracker must record 'whale' rather than take its unknown-name branch
    from analytics.strategy_aging import ALL_STRATEGIES, STRATEGY_GROUPS
    assert "whale" in ALL_STRATEGIES
    assert "whale" in STRATEGY_GROUPS["universal"]
    # regimebook (P317) must remain — this fix adds, does not replace
    assert "regimebook" in ALL_STRATEGIES


def test_aging_tracker_records_whale_not_unknown():
    # behavioural: a whale signal is accepted, not the '' unknown-name branch
    from analytics.strategy_aging import StrategyAgingManager
    mgr = StrategyAgingManager()
    out = mgr.record_signal("whale", direction=1.0, confidence=0.9, regime="STEADY_UPTREND")
    assert out != "", "whale still hits the unknown-strategy branch"

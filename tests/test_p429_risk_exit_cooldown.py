"""[P429] Risk-exit rebuild cooldown on the Coinbase sleeve.

The live loop (2026-10-14/15): watchdog REDUCE_50 BTC -2 -> -1, then the 4H
driver re-added to -2 after the P416 resize streak. These tests pin:
  * the pure decision: only exposure-INCREASING targets are suppressible,
    and reduces/flattens never are;
  * shadow changes no order; enforce holds an add, demotes a flip to a
    flatten, keeps flat flat;
  * the watchdog arms it only on an EXECUTED reduce/exit;
  * the stamp survives a restart (P154);
  * config is declared, parsed and passed (P201), default shadow.
"""
import asyncio
import inspect
import json
import os
import time

import pytest

import main
from exchange import coinbase_sleeve as cs
from exchange.coinbase_sleeve import CoinbaseSleeve, risk_exit_cooldown_decision

W = 24 * 3600.0


@pytest.mark.parametrize("cur,tgt,expect", [
    (-1, -2, (True, -1.0, "add_held")),
    (1, 2, (True, 1.0, "add_held")),
    (0, 2, (True, 0.0, "entry_from_flat")),
    (0, -1, (True, 0.0, "entry_from_flat")),
    (-1, 2, (True, 0.0, "flip_demoted_to_flatten")),
    (2, -1, (True, 0.0, "flip_demoted_to_flatten")),
    (-2, -1, (False, -1.0, "not_a_rebuild")),   # reduce
    (-2, 0, (False, 0.0, "not_a_rebuild")),     # flatten
    (-1, -1, (False, -1.0, "not_a_rebuild")),   # hold
    (0, 0, (False, 0.0, "not_a_rebuild")),
])
def test_decision_truth_table(cur, tgt, expect):
    assert risk_exit_cooldown_decision(cur, tgt, 3600.0, W) == expect


@pytest.mark.parametrize("age", [None, W, W + 1, -5, float("nan")])
def test_no_recent_exit_never_suppresses(age):
    s, held, why = risk_exit_cooldown_decision(-1, -2, age, W)
    assert s is False and held == -2


def test_zero_window_disables():
    assert risk_exit_cooldown_decision(-1, -2, 10.0, 0.0)[0] is False


def test_never_returns_more_exposure_than_current_or_target():
    for cur in range(-3, 4):
        for tgt in range(-3, 4):
            s, held, _ = risk_exit_cooldown_decision(cur, tgt, 60.0, W)
            if s:
                assert abs(held) <= abs(cur)
                assert held == 0 or (held > 0) == (cur > 0)
            else:
                assert held == tgt


class _FakeAdapter:
    def __init__(self):
        self.orders = []


def _sleeve(tmp_path, monkeypatch, mode, cur):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    s = CoinbaseSleeve.__new__(CoinbaseSleeve)
    s._risk_cd_mode = mode
    s._risk_cd_window_sec = W
    s._risk_exit_at = {}
    s._flip_persist_ticks = 0
    s._resize_persist_ticks = 0
    s._flip_pending = {}
    s._resize_pending = {}
    s._reconcile_ok = True
    s._pos = {"BTC": cur}
    s.executed = []
    s.reconcile_positions = lambda: None
    s.signed_contracts = lambda a: s._pos.get(a, 0)
    s.target_for = lambda a, d, th=0.15, cv=1.0: int(d) * 2
    s._persist_state = lambda: None

    async def _exec(a, t, urgent=False):
        s.executed.append((a, t))
        return {"status": "OK"}

    async def _sweep(a):
        return 0
    s.execute_target = _exec
    s.sweep_stale_entries = _sweep
    return s


def _ledger(tmp_path):
    p = tmp_path / "risk_exit_cooldown_shadow.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def test_shadow_changes_no_order_but_ledgers(tmp_path, monkeypatch):
    s = _sleeve(tmp_path, monkeypatch, "shadow", -1)
    s.note_risk_exit("BTC", "risk_exit:fast_risk:REDUCE_50", -2, -1)
    r = asyncio.run(s.manage_to_signal("BTC", -1.0))
    assert s.executed == [("BTC", -2)]          # re-add happened (shadow)
    assert r["status"] == "OK"
    ev = [x["event"] for x in _ledger(tmp_path)]
    assert ev == ["risk_exit", "would_suppress"]


def test_enforce_holds_the_add(tmp_path, monkeypatch):
    s = _sleeve(tmp_path, monkeypatch, "enforce", -1)
    s._resize_pending = {"BTC": (-2, 1, time.time())}
    s.note_risk_exit("BTC", "x", -2, -1)
    r = asyncio.run(s.manage_to_signal("BTC", -1.0))
    assert s.executed == []
    assert r["status"] == "RISK_COOLDOWN" and r["target"] == -1
    assert "BTC" not in s._resize_pending       # streak cannot ride through


def test_enforce_flip_becomes_flatten(tmp_path, monkeypatch):
    s = _sleeve(tmp_path, monkeypatch, "enforce", -1)
    s.note_risk_exit("BTC", "x", -2, -1)
    asyncio.run(s.manage_to_signal("BTC", 1.0))
    assert s.executed == [("BTC", 0)]


def test_enforce_never_blocks_a_reduce_or_flatten(tmp_path, monkeypatch):
    s = _sleeve(tmp_path, monkeypatch, "enforce", -2)
    s.note_risk_exit("BTC", "x", -4, -2)
    asyncio.run(s.manage_to_signal("BTC", 0.0))   # target 0 = flatten
    assert s.executed == [("BTC", 0)]


def test_expired_window_allows_rebuild(tmp_path, monkeypatch):
    s = _sleeve(tmp_path, monkeypatch, "enforce", -1)
    s._risk_exit_at["BTC"] = {"ts": time.time() - W - 60, "cause": "x"}
    asyncio.run(s.manage_to_signal("BTC", -1.0))
    assert s.executed == [("BTC", -2)]


def test_object_new_sleeve_reads_off(tmp_path, monkeypatch):
    s = _sleeve(tmp_path, monkeypatch, "enforce", -1)
    del s._risk_cd_mode
    asyncio.run(s.manage_to_signal("BTC", -1.0))
    assert s.executed == [("BTC", -2)]


def test_stamp_persists_across_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    s = CoinbaseSleeve.__new__(CoinbaseSleeve)
    s._risk_exit_at = {}
    s._BASE_VERSION = CoinbaseSleeve._BASE_VERSION
    s.note_risk_exit("ETH", "risk_exit:fast_risk:REDUCE_50", 6, 3)
    st = json.loads((tmp_path / "coinbase_sleeve_state.json").read_text(encoding="utf-8"))
    s2 = CoinbaseSleeve.__new__(CoinbaseSleeve)
    s2._risk_exit_at = {}
    s2._restore_risk_exits(st)
    assert s2.risk_exit_age_sec("ETH") is not None
    assert s2.risk_exit_age_sec("ETH") < 60


def test_restore_state_wires_the_stamp_back(tmp_path, monkeypatch):
    """Through the REAL restore path, not the helper alone."""
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    (tmp_path / "coinbase_sleeve_state.json").write_text(json.dumps({
        "base_version": CoinbaseSleeve._BASE_VERSION,
        "risk_exit_at": {"BTC": {"ts": time.time() - 3600, "cause": "x"}}}),
        encoding="utf-8")
    s = CoinbaseSleeve.__new__(CoinbaseSleeve)
    s._risk_exit_at = {}
    s._flip_pending, s._resize_pending = {}, {}
    s._restore_state()
    assert 3500 < s.risk_exit_age_sec("BTC") < 3700


def test_malformed_stamp_is_no_stamp():
    s = CoinbaseSleeve.__new__(CoinbaseSleeve)
    s._risk_exit_at = {}
    s._restore_risk_exits({"risk_exit_at": {"BTC": {"cause": "no ts"}, "ETH": "junk"}})
    assert s.risk_exit_age_sec("BTC") is None and s.risk_exit_age_sec("ETH") is None


class _WatchdogSleeve:
    def __init__(self, cur, status="OK"):
        self.cur, self.status = cur, status
        self._reconcile_ok = True
        self.exits, self.triggers = [], []

    def reconcile_positions(self):
        pass

    def signed_contracts(self, a):
        return self.cur

    async def execute_target(self, a, t, urgent=False):
        return {"status": self.status}

    def note_risk_exit(self, a, cause, b, af):
        self.exits.append((a, cause, b, af))

    def note_risk_trigger(self, a, action, status, c):
        self.triggers.append((a, action, status))


def test_watchdog_arms_only_on_executed_reduce():
    sl = _WatchdogSleeve(-2)
    st, _ = asyncio.run(main.sleeve_fast_risk_action(sl, "BTC", "REDUCE_50", True))
    assert st == "REDUCED" and sl.exits == [("BTC", "risk_exit:fast_risk:REDUCE_50", -2, -1)]


def test_watchdog_failed_reduce_does_not_arm():
    sl = _WatchdogSleeve(-2, status="FAILED")
    asyncio.run(main.sleeve_fast_risk_action(sl, "BTC", "REDUCE_50", True))
    assert sl.exits == [] and sl.triggers


def test_watchdog_noop_is_ledgered_as_trigger():
    sl = _WatchdogSleeve(-1)
    st, _ = asyncio.run(main.sleeve_fast_risk_action(sl, "BTC", "REDUCE_50", True))
    assert st == "REDUCE_NOOP" and sl.exits == [] and sl.triggers == [("BTC", "REDUCE_50", "REDUCE_NOOP")]


def test_watchdog_exit_arms():
    sl = _WatchdogSleeve(-2)
    st, _ = asyncio.run(main.sleeve_fast_risk_action(sl, "BTC", "EXIT_ONLY", True))
    assert st == "EXITED" and sl.exits and sl.exits[0][3] == 0


def test_config_trio_default_shadow():
    cfg = main.ProductionConfig()
    assert cfg.coinbase_risk_exit_cooldown_mode == "shadow"
    assert cfg.coinbase_risk_exit_cooldown_hours == 24.0
    src = inspect.getsource(main)
    assert 'data.get("coinbase_risk_exit_cooldown_mode"' in src
    assert "risk_exit_cooldown_mode=str(getattr(" in src
    sig = inspect.signature(CoinbaseSleeve.__init__)
    assert sig.parameters["risk_exit_cooldown_mode"].default == "shadow"


def test_live_profile_does_not_arm_enforce():
    with open(os.path.join(os.path.dirname(__file__), "..", "configs",
                           "live_high_risk.json"), encoding="utf-8") as f:
        prof = json.load(f)
    assert prof.get("coinbase_risk_exit_cooldown_mode", "shadow") != "enforce"


def test_review_verdicts():
    import importlib.util
    p = os.path.join(os.path.dirname(__file__), "..", "scripts",
                     "risk_cooldown_shadow_review.py")
    spec = importlib.util.spec_from_file_location("rcsr", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    t0 = 1_800_000_000.0
    rows = [
        {"ts": t0, "event": "risk_exit", "asset": "BTC"},
        {"ts": t0 + 3600, "event": "would_suppress", "asset": "BTC", "age_h": 1.0,
         "window_h": 24.0, "current": -1, "target": -2, "held_target": -1},
        {"ts": t0 + 7200, "event": "would_suppress", "asset": "BTC", "age_h": 2.0,
         "window_h": 24.0, "current": -1, "target": -2, "held_target": -1},
        {"ts": t0 + 15 * 3600, "event": "risk_exit", "asset": "BTC"},
        {"ts": t0 + 50 * 3600, "event": "risk_trigger", "asset": "BTC"},
    ]
    flat = lambda a: {int(t0 - 3600): 100.0, int(t0 + 100 * 3600): 100.0}
    out = m.review(rows, fetch=flat)
    assert out["rebuild_cycles"] == 1 and out["avoided_round_trips"] == 1
    assert out["verdict"] == "ENFORCE_ELIGIBLE"
    assert m.review(rows[:3], fetch=flat)["verdict"] == "NOT_YET"
    nopx = m.review(rows, fetch=lambda a: {})
    assert nopx["verdict"] == "KEEP_SHADOW"   # unknown opportunity cost never passes

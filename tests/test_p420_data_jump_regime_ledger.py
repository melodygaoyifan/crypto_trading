"""[P420] Jump-regime shadow: counters persist, a ledger accrues, OOD skipped.

P414c persisted only the cost vector + last label; the churn counters
(`_nseen/_jsw/_gsw/_last_gmm`) were RAM-only and the only output was a log
line — so every restart (7 on 2026-08-27) zeroed the "jump X/N vs gmm Y/N"
comparison the P414c cutover is supposed to be judged on. Now: counters ride
the same atomic write, one JSONL row per asset per decision tick lands in
data/strategy_shadow/jumpregime_{ASSET}.jsonl, and a tick whose GMM label
is the OOD ADX proxy is skipped (observation-only, Iron Law 7)."""
from __future__ import annotations

import json
from pathlib import Path

from defense.jump_regime_shadow import JumpRegimeShadow, _STATE_VERSION


def _model():
    return {"asset": "BTC", "lambda": 20.0, "k": 2,
            "scaler_mean": [0.0], "scaler_std": [1.0],
            "centroids": [[-1.0], [1.0]],
            "state_to_name": {"0": "QUIET_ACCUMULATION",
                              "1": "MOMENTUM_RALLY"}}


def _shadow(tmp_path):
    s = JumpRegimeShadow.__new__(JumpRegimeShadow)
    s._models = {"BTC": _model()}
    s._cost = {}; s._last_label = {}; s._jsw = {}; s._gsw = {}
    s._nseen = {}; s._last_gmm = {}
    s._state_path = tmp_path / "jumpregime_state.json"
    s._ledger_dir = tmp_path / "strategy_shadow"
    return s


def _rows(tmp_path):
    p = tmp_path / "strategy_shadow" / "jumpregime_BTC.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _drive(s):
    """three ticks with one jump switch and one gmm switch"""
    s.tick({"BTC": [1.0]}, {"BTC": "A"})
    s.tick({"BTC": [1.0]}, {"BTC": "A"})
    for _ in range(3):                       # decisive move -> jump switch
        s.tick({"BTC": [-1.0]}, {"BTC": "B"})


def test_counters_survive_a_restart(tmp_path):
    s = _shadow(tmp_path)
    _drive(s)
    n, js, gs, lg = (s._nseen["BTC"], s._jsw["BTC"], s._gsw["BTC"],
                     s._last_gmm["BTC"])
    assert n == 5 and js >= 1 and gs == 1 and lg == "B"
    s2 = _shadow(tmp_path)
    s2._restore_state()
    assert (s2._nseen["BTC"], s2._jsw["BTC"], s2._gsw["BTC"],
            s2._last_gmm["BTC"]) == (n, js, gs, lg), (
        "the churn counters reset on restart — the comparison the cutover "
        "is judged on restarts from zero every deploy")
    # and the next tick CONTINUES the count rather than restarting it
    s2.tick({"BTC": [-1.0]}, {"BTC": "B"})
    assert s2._nseen["BTC"] == n + 1


def test_a_pre_p418_state_file_restores_cold_counters_never_fabricated(tmp_path):
    s = _shadow(tmp_path)
    s.step("BTC", [1.0])
    s._state_path.write_text(json.dumps({
        "v": _STATE_VERSION, "saved_ts": 0.0,
        "assets": {"BTC": {"cost": s._cost["BTC"], "last_label": 1}}}),
        encoding="utf-8")
    s2 = _shadow(tmp_path)
    s2._restore_state()
    assert s2._cost["BTC"] == s._cost["BTC"]
    assert s2._nseen.get("BTC", 0) == 0 and s2._jsw.get("BTC", 0) == 0


def test_one_ledger_row_per_asset_per_tick_with_the_declared_shape(tmp_path):
    s = _shadow(tmp_path)
    _drive(s)
    rows = _rows(tmp_path)
    assert len(rows) == 5
    need = {"ts", "iso", "asset", "jump_label", "gmm_label",
            "jump_switches", "gmm_switches", "n", "strategy"}
    for r in rows:
        assert need <= set(r), sorted(need - set(r))
        assert r["asset"] == "BTC" and r["strategy"] == "jumpregime"
    assert [r["n"] for r in rows] == [1, 2, 3, 4, 5]
    assert rows[-1]["jump_switches"] == s._jsw["BTC"]
    assert rows[-1]["gmm_label"] == "B"
    assert rows[2]["gmm_switches"] == 1          # the A->B switch


def test_an_ood_fallback_tick_is_skipped_entirely(tmp_path):
    s = _shadow(tmp_path)
    s.tick({"BTC": [1.0]}, {"BTC": "A"})
    out = s.tick({"BTC": [-1.0]}, {"BTC": "PANIC_SELLOFF"},
                 fallback_by_asset={"BTC": "distribution_shift"})
    assert any("SKIP" in x for x in out)
    assert s._nseen["BTC"] == 1, "a proxy label was counted as a GMM tick"
    assert s._gsw.get("BTC", 0) == 0, "a proxy label was counted as a switch"
    assert len(_rows(tmp_path)) == 1, "a skipped tick wrote a ledger row"


def test_no_stash_means_no_row_and_no_count(tmp_path):
    """The pipeline withholds `_gmm_raw_features` on the fallback path, so
    the caller's stash is empty for that asset — the same skip."""
    s = _shadow(tmp_path)
    s.tick({}, {"BTC": "A"})
    assert not s._nseen and not _rows(tmp_path)


def test_ledger_write_failure_cannot_break_the_tick(tmp_path, caplog):
    s = _shadow(tmp_path)
    (tmp_path / "blocked").write_text("x", encoding="utf-8")
    s._ledger_dir = tmp_path / "blocked"          # a FILE, not a dir
    out = s.tick({"BTC": [1.0]}, {"BTC": "A"})
    assert out and s._nseen["BTC"] == 1
    assert any("ledger append failed" in r.getMessage()
               for r in caplog.records)


def test_a_shadow_built_without_a_ledger_dir_still_ticks(tmp_path):
    """The P414c test helper builds via __new__ with no _ledger_dir (P85)."""
    s = _shadow(tmp_path)
    del s._ledger_dir
    assert s.tick({"BTC": [1.0]}, {"BTC": "A"})


def test_still_observation_only():
    """Iron Law 7: the shadow writes a log line and a ledger, never an
    order — no execution symbol may appear in the module."""
    import inspect
    from defense import jump_regime_shadow as m
    src = inspect.getsource(m)
    for bad in ("execute_target", "manage_to_signal", "place_order",
                "target_for"):
        assert bad not in src, bad

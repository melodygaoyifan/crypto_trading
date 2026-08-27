"""[P198] Sleeve flip-persistence: a sign-flip must persist N consecutive ticks.

The Coinbase sleeve is driven by `_last_quant_directions` raw, so it inherited
NONE of the P142 Layer-2 churn controls — measured live 2026-06-14 -> 08-06:
BTC flipped direction 29 times in 54 days while the sleeve lost 5.64%
(daily Sharpe -4.5). These tests pin the control's exact asymmetry:

  * only FLIPS of a live position are deferred;
  * entries from flat, flattens, and same-direction targets execute
    immediately (exits stay instant — the P195 principle);
  * a deferred flip HOLDS the position (P142 semantics: no close, no reverse);
  * the streak requires CONSECUTIVE opposing ticks — any same-sign or flatten
    tick resets it; a SKIPPED_STALE tick pauses it (neither confirms nor
    refutes).
"""

import asyncio

import pytest

from exchange.coinbase_sleeve import CoinbaseSleeve


@pytest.fixture(autouse=True)
def _private_data_dir(tmp_path, monkeypatch):
    """[P420] deferred ticks now PERSIST the streak to
    $HMATS_DATA_DIR/coinbase_sleeve_state.json; point it at a private dir
    so the suite never writes into the repo's data/ (P294 pattern)."""
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))


class _Recorder:
    def __init__(self):
        self.calls = []

    async def __call__(self, asset, target):
        self.calls.append((asset, target))
        return {"status": "EXECUTED", "asset": asset, "target": target}


def _sleeve(current_contracts: float, flip_persist_ticks: int = 2):
    """Sleeve with just enough wired up to exercise manage_to_signal's
    flip-persistence block: reconcile stubbed OK, execute_target recorded."""
    s = object.__new__(CoinbaseSleeve)
    s._flip_persist_ticks = flip_persist_ticks
    s._flip_pending = {}
    s._reconcile_ok = True
    s._cur = current_contracts
    s.reconcile_positions = lambda: {}  # type: ignore[assignment]
    s.signed_contracts = lambda asset: s._cur  # type: ignore[assignment]
    s.execute_target = _Recorder()  # type: ignore[assignment]
    return s


def _manage(s, direction):
    return asyncio.run(s.manage_to_signal("BTC", direction))


# ---------------------------------------------------------------------------
# The deferral itself
# ---------------------------------------------------------------------------

def test_first_opposing_tick_defers_and_holds():
    s = _sleeve(current_contracts=+1.0)
    res = _manage(s, -0.5)
    assert res["status"] == "FLIP_DEFERRED"
    assert res["streak"] == 1 and res["need"] == 2
    assert s.execute_target.calls == [], (
        "a deferred flip must HOLD the position — no order of any kind"
    )


def test_second_consecutive_opposing_tick_executes_the_flip():
    s = _sleeve(current_contracts=+1.0)
    assert _manage(s, -0.5)["status"] == "FLIP_DEFERRED"
    res = _manage(s, -0.5)
    assert res["status"] == "EXECUTED"
    assert s.execute_target.calls == [("BTC", -1)]


def test_alternating_single_tick_reversals_never_flip():
    """The measured live bleed: sign oscillation. With persistence, an
    opposing tick that is not CONSECUTIVE never executes a flip."""
    s = _sleeve(current_contracts=+1.0)
    for d in (-0.5, +0.5, -0.5, +0.5, -0.5, +0.5):
        _manage(s, d)
    flips = [c for c in s.execute_target.calls if c[1] < 0]
    assert flips == [], f"oscillating signal executed flip(s): {flips}"


def test_same_sign_tick_resets_the_streak():
    s = _sleeve(current_contracts=+1.0)
    _manage(s, -0.5)          # streak 1
    _manage(s, +0.5)          # same-sign -> clears pending
    res = _manage(s, -0.5)    # must start over at streak 1
    assert res["status"] == "FLIP_DEFERRED" and res["streak"] == 1


# ---------------------------------------------------------------------------
# What must NEVER be deferred
# ---------------------------------------------------------------------------

def test_flatten_is_never_deferred_even_mid_streak():
    s = _sleeve(current_contracts=+1.0)
    _manage(s, -0.5)                       # pending flip streak
    res = _manage(s, 0.0)                  # hold signal -> flatten
    assert res["status"] == "EXECUTED"
    assert ("BTC", 0) in s.execute_target.calls, (
        "flatten-on-hold is the sleeve's exit path (P141/P195) and must "
        "execute instantly regardless of any pending flip streak"
    )


def test_entry_from_flat_is_never_deferred():
    s = _sleeve(current_contracts=0.0)
    res = _manage(s, -0.5)
    assert res["status"] == "EXECUTED"
    assert s.execute_target.calls == [("BTC", -1)]


def test_same_direction_target_is_never_deferred():
    s = _sleeve(current_contracts=+1.0)
    res = _manage(s, +0.9)
    assert res["status"] == "EXECUTED"
    assert s.execute_target.calls == [("BTC", 1)]


# ---------------------------------------------------------------------------
# Configuration edges
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ticks", [0, 1])
def test_ticks_leq_1_disables_persistence(ticks):
    s = _sleeve(current_contracts=+1.0, flip_persist_ticks=ticks)
    res = _manage(s, -0.5)
    assert res["status"] == "EXECUTED"
    assert s.execute_target.calls == [("BTC", -1)]


def test_streak_is_cleared_after_an_executed_flip():
    s = _sleeve(current_contracts=+1.0)
    _manage(s, -0.5)
    _manage(s, -0.5)                       # flip executes
    s._cur = -1.0                          # position is now short
    res = _manage(s, +0.5)                 # flip back must need a fresh streak
    assert res["status"] == "FLIP_DEFERRED" and res["streak"] == 1


def test_stale_reconcile_pauses_but_does_not_reset_the_streak():
    s = _sleeve(current_contracts=+1.0)
    _manage(s, -0.5)                       # streak 1
    s._reconcile_ok = False
    assert _manage(s, -0.5)["status"] == "SKIPPED_STALE"
    s._reconcile_ok = True
    res = _manage(s, -0.5)                 # streak resumes -> 2 -> execute
    assert res["status"] == "EXECUTED"
    assert s.execute_target.calls == [("BTC", -1)]


# ---------------------------------------------------------------------------
# Wiring: main.py must actually pass the config through (P152 lesson — a
# guard defined and tested in isolation but never CALLED is invisible)
# ---------------------------------------------------------------------------

def test_main_passes_flip_persist_ticks_to_the_sleeve():
    import io
    src = io.open("main.py", encoding="utf-8").read()
    anchor = "self._coinbase_sleeve = CoinbaseSleeve("
    i = src.find(anchor)
    assert i != -1, "main.py no longer constructs the driver sleeve inline"
    window = src[i:i + 1500]
    assert "flip_persist_ticks=" in window, (
        "main.py's CoinbaseSleeve construction no longer passes "
        "flip_persist_ticks — the churn control would silently revert to the "
        "class default of 0 (disabled). P152 shape: wired-in-tests-only."
    )
    assert "coinbase_flip_persist_ticks" in window


def test_default_is_off_at_class_level():
    """The class default must stay 0 (inert) so read-only constructions —
    e.g. core/execution_service._coinbase_get_sleeve() — are unaffected;
    the live value comes from config at the main.py driver site."""
    import inspect
    sig = inspect.signature(CoinbaseSleeve.__init__)
    assert sig.parameters["flip_persist_ticks"].default == 0

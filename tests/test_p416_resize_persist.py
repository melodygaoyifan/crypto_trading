"""[P416] Same-direction RESIZE persistence -- the conviction-flap churn fix.

End-to-end validation against the operator's goal (hold the trend, pay fees
only on trend changes) measured the ONE live conflict: direction flipped ZERO
times in 14 days while the fill ledger showed same-direction resize round
trips (SOL 2->1->2ct) driven by fusion_conviction flapping 1.09<->0.58 within
minutes -- ~10-15bps of fees per leg for zero trend information. A resize now
needs the SAME proposed target on N consecutive manage calls. Flips keep
their own P198 streak; entries, exits and flattens are NEVER deferred (P195).
"""
from __future__ import annotations

import asyncio

import pytest
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from exchange.coinbase_sleeve import CoinbaseSleeve  # noqa: E402


@pytest.fixture(autouse=True)
def _private_data_dir(tmp_path, monkeypatch):
    """[P420] deferred ticks now PERSIST the streak to
    $HMATS_DATA_DIR/coinbase_sleeve_state.json; point it at a private dir
    so the suite never writes into the repo's data/ (P294 pattern)."""
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))


def _sleeve(resize_ticks=2, cur=2):
    """Minimal live-shaped sleeve: manage_to_signal's own logic, stubbed IO.

    target_for mimics the live sizing shape: |dir| below the 0.15 deadband ->
    flatten (0); else sign(dir) * round(2 * conviction) contracts.
    """
    s = CoinbaseSleeve.__new__(CoinbaseSleeve)
    s._resize_persist_ticks = resize_ticks
    s._resize_pending = {}
    s._flip_persist_ticks = 0
    s._flip_pending = {}
    s._reconcile_ok = True
    s._cur = float(cur)
    s._executed = []
    s.reconcile_positions = lambda: None
    s.signed_contracts = lambda asset: s._cur

    def _target_for(asset, d, threshold=0.15, conviction=1.0):
        if abs(d) < threshold:
            return 0
        sized = max(1, int(round(2 * conviction)))
        return sized if d > 0 else -sized
    s.target_for = _target_for

    async def _exec(asset, target):
        s._executed.append(target)
        return {"status": "OK", "asset": asset}
    s.execute_target = _exec
    return s


def _manage(s, direction, conviction=1.0):
    return asyncio.run(s.manage_to_signal("SOL", direction,
                                          conviction=conviction))


class TestResizeDeferral:
    def test_first_differing_resize_is_deferred_and_places_nothing(self):
        s = _sleeve(cur=2)
        r = _manage(s, +1.0, conviction=0.58)   # sized -> 1ct vs cur 2
        assert r["status"] == "RESIZE_DEFERRED"
        assert s._executed == []
        # the stop reconcile must size to the book actually HELD
        assert r["target"] == 2

    def test_second_consecutive_same_proposal_executes(self):
        s = _sleeve(cur=2)
        _manage(s, +1.0, conviction=0.58)
        r = _manage(s, +1.0, conviction=0.58)
        assert r["status"] == "OK"
        assert s._executed == [1]

    def test_the_live_oscillation_produces_zero_position_changes(self):
        """THE MEASURED INCIDENT: conviction 1.09 -> 0.58 -> 1.09 -> 0.58.
        Under the old code each boundary crossing was a fee-paying
        sell-then-buy round trip with no trend change; now nothing that
        moves the position may execute."""
        s = _sleeve(cur=2)
        _manage(s, +1.0, conviction=1.0)    # target 2 == cur -> executes NOOP
        _manage(s, +1.0, conviction=0.58)   # target 1 -> deferred (streak 1)
        _manage(s, +1.0, conviction=1.0)    # target 2 == cur -> streak broken
        _manage(s, +1.0, conviction=0.58)   # target 1 -> streak RESET to 1
        assert all(t == 2 for t in s._executed), s._executed

    def test_a_changed_proposal_resets_the_streak(self):
        s = _sleeve(cur=4)
        _manage(s, +1.0, conviction=0.9)    # propose 2 (round(1.8)) streak 1
        r = _manage(s, +1.0, conviction=0.58)  # propose 1 -> differs -> 1
        assert r["status"] == "RESIZE_DEFERRED" and r["streak"] == 1
        assert s._executed == []


class TestNeverDeferred:
    def test_entry_from_flat_is_instant(self):
        s = _sleeve(cur=0)
        r = _manage(s, +1.0)
        assert r["status"] == "OK" and s._executed == [2]

    def test_flatten_is_instant(self):
        s = _sleeve(cur=2)
        r = _manage(s, 0.0)
        assert r["status"] == "OK" and s._executed == [0]

    def test_a_flip_is_not_captured_by_the_resize_block(self):
        # flip persist 0 -> the flip must execute immediately; the resize
        # block must not treat an opposite-sign target as a resize
        s = _sleeve(cur=2)
        r = _manage(s, -1.0)
        assert r["status"] == "OK" and s._executed == [-2]

    def test_default_zero_is_byte_identical(self):
        s = _sleeve(resize_ticks=0, cur=2)
        r = _manage(s, +1.0, conviction=0.58)
        assert r["status"] == "OK" and s._executed == [1]


class TestWiringAndSafety:
    def test_config_trio_and_decided_value(self):
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        assert "coinbase_resize_persist_ticks: int = 0" in src
        assert 'data.get("coinbase_resize_persist_ticks", 0)' in src
        assert "resize_persist_ticks=int(getattr(" in src
        import json
        live = json.loads((REPO / "configs" / "live_high_risk.json"
                           ).read_text(encoding="utf-8"))
        assert live.get("coinbase_resize_persist_ticks") == 2, (
            "the DECIDED live value (P416, operator-authorized churn fix); "
            "a silent revert to 0 re-opens the conviction-flap fee leak")

    def test_deferred_status_lets_the_snapshot_govern_the_stop(self):
        import main
        assert main.stop_reconcile_intended_target(
            "RESIZE_DEFERRED", 1) is None

    def test_deferred_streak_survives_a_stale_tick_pause(self):
        # a stale reconcile SKIPs before the resize block runs -- the pending
        # streak is left untouched (pause, not reset), like flip-persist
        s = _sleeve(cur=2)
        _manage(s, +1.0, conviction=0.58)          # streak 1
        s._reconcile_ok = False
        r = _manage(s, +1.0, conviction=0.58)      # SKIPPED_STALE
        assert r["status"] == "SKIPPED_STALE"
        s._reconcile_ok = True
        r = _manage(s, +1.0, conviction=0.58)      # streak 2 -> executes
        assert r["status"] == "OK" and s._executed == [1]

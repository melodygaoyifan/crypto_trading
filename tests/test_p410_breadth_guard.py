"""[P410] The breadth Kraken-window guard: an asset in config.assets that is
NOT one of the home trio (BTC/ETH/SOL) and NOT Coinbase-routed must never open
a NEW Kraken entry — it stays inert until routed to the sleeve.

WHY: venue_for() returns "kraken" for anything not in coinbase_assets, so a
breadth perp (XRP/ADA/LTC/DOGE/BNB) added to config.assets as the sleeve's
tradeable universe would otherwise fall through to a Kraken order while
unrouted. This guard closes that window so breadth can join config.assets and
stay inert until the operator's routing flip. It is a strict NO-OP today
(config.assets is the home trio, all routed → P152 catches them first).

The load-bearing safety property is that the HOME TRIO is never skipped by this
guard — a bug here that skipped BTC/ETH/SOL would stop live trading."""
from __future__ import annotations

import pytest

from core.execution_service import (
    _COINBASE_HOME_ASSETS,
    _should_skip_breadth_kraken_entry as skip,
    BENIGN_EXEC_SKIP_REASONS,
    is_benign_exec_skip,
)


@pytest.mark.parametrize("home", ["BTC", "ETH", "SOL", "btc", "eth", "sol"])
def test_home_trio_is_never_skipped(home):
    """The live 3 must never be skipped by this guard, in ANY state — this is
    the property whose failure would stop live trading."""
    for has_pos in (True, False):
        for is_exit in (True, False):
            for routed in (True, False):
                assert not skip(home, has_pos, is_exit, True, routed), (
                    home, has_pos, is_exit, routed)


def test_breadth_new_entry_unrouted_is_skipped():
    for a in ("XRP", "ADA", "LTC", "DOGE", "BNB"):
        assert skip(a, False, False, True, False), a


def test_breadth_routed_is_not_skipped_here():
    # a routed breadth asset is caught by the P152 skip above, not this guard
    assert not skip("XRP", False, False, True, True)


def test_breadth_exit_or_reduce_still_executes():
    # an exit/reduce of a real holding must still run (mirrors P152)
    assert not skip("XRP", False, True, True, False)   # full exit
    assert not skip("XRP", True, False, True, False)   # has a position


def test_routing_disabled_changes_nothing():
    # pre-Phase-2 / flag off: behaviour is unchanged, never skipped by this guard
    assert not skip("XRP", False, False, False, False)


def test_home_set_is_exactly_the_trio():
    assert _COINBASE_HOME_ASSETS == frozenset({"BTC", "ETH", "SOL"})


def test_the_skip_reason_is_benign_not_a_veto():
    """P338: a SKIPPED reason the caller does not recognise as benign is
    stamped as a veto and FLATTENS the sleeve. This one must be benign."""
    assert "breadth_not_routed_no_kraken_entry" in BENIGN_EXEC_SKIP_REASONS
    assert is_benign_exec_skip(
        {"status": "SKIPPED", "reason": "breadth_not_routed_no_kraken_entry"})
    # and a non-skip status with that reason is NOT benign (guard against
    # over-broad matching, P338's own rule)
    assert not is_benign_exec_skip(
        {"status": "FILLED", "reason": "breadth_not_routed_no_kraken_entry"})

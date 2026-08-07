"""[P195] The Coinbase sleeve's drawdown halt must never block an exit.

`can_trade` used to test `_halted` as its first statement and return False for
EVERY order, so tripping the 15% drawdown halt froze the sleeve INTO the losing
position — the control meant to cap losses prevented the exit that realises the
cap. Two concrete consequences, both live:

  * manage_to_signal(asset, 0.0) -> execute_target(asset, 0) -> can_trade -> BLOCKED,
    so a halted sleeve could not flatten on a hold signal;
  * scripts/coinbase_flatten.py constructs a fresh CoinbaseSleeve, which restores
    `halted` from data/coinbase_sleeve_state.json (P150), so the documented
    EMERGENCY FLATTEN was blocked too until an operator called reset_halt().

P150 made the halt sticky across restarts. That is correct for a loss cap and
compounding for a trade block; the two were conflated. The halt now blocks
opening and never blocks exiting.

The predicate is `abs(resulting) < abs(cur)` — strictly reducing. A FLIP is
deliberately NOT a reduction (+1 -> -1 leaves abs at 1), because a halted sleeve
must not open fresh directional risk in the opposite direction.
"""

import types

import pytest

from exchange.coinbase_sleeve import CoinbaseSleeve


def _sleeve(halted: bool, current_contracts: float, max_contracts: int = 1):
    """A CoinbaseSleeve with just enough wired up to exercise can_trade.

    __init__ builds an adapter and restores persisted state, neither of which
    this gate touches — so construct without __init__ and set only the fields
    can_trade actually reads.
    """
    s = object.__new__(CoinbaseSleeve)
    s._halted = halted
    s._halt_reason = "sleeve drawdown 15.2% >= 15%" if halted else ""
    s._max_contracts_per_asset = max_contracts
    s.signed_contracts = lambda asset: current_contracts  # type: ignore[assignment]
    return s


# ---------------------------------------------------------------------------
# HALTED — the case that motivated P195
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cur,intended,label", [
    (1.0, -1.0, "long -> flat (the emergency flatten)"),
    (-1.0, +1.0, "short -> flat (the emergency flatten)"),
    (2.0, -1.0, "long partial reduce"),
    (-2.0, +1.0, "short partial reduce"),
])
def test_halted_allows_orders_that_reduce_exposure(cur, intended, label):
    allowed, reason = _sleeve(halted=True, current_contracts=cur).can_trade("BTC", intended)
    assert allowed, (
        f"halted sleeve refused a risk-REDUCING order ({label}): {reason}. This is "
        f"the P195 trap — the drawdown halt freezing the position it exists to exit."
    )
    assert reason == "halted_but_reducing"


@pytest.mark.parametrize("cur,intended,label", [
    (0.0, +1.0, "flat -> long (new entry)"),
    (0.0, -1.0, "flat -> short (new entry)"),
    (1.0, +1.0, "adding to a long"),
    (-1.0, -1.0, "adding to a short"),
])
def test_halted_still_blocks_orders_that_add_risk(cur, intended, label):
    allowed, reason = _sleeve(halted=True, current_contracts=cur).can_trade("BTC", intended)
    assert not allowed, f"halted sleeve allowed a risk-ADDING order ({label})"
    assert "coinbase_sleeve_halted" in reason


@pytest.mark.parametrize("cur,intended,label", [
    (1.0, -2.0, "long -> short flip"),
    (-1.0, +2.0, "short -> long flip"),
])
def test_halted_blocks_a_flip_because_a_flip_is_not_a_reduction(cur, intended, label):
    """The case `abs(resulting) < abs(cur)` is specifically chosen to catch.

    A flip ends at the same absolute size, so it opens fresh directional risk
    while superficially looking like an exit followed by an entry. A halted
    sleeve must not take it. A naive `resulting <= abs(cur)` would let it through.
    """
    allowed, reason = _sleeve(halted=True, current_contracts=cur).can_trade("BTC", intended)
    assert not allowed, f"halted sleeve allowed a FLIP ({label}) — that opens new risk"
    assert "coinbase_sleeve_halted" in reason


# ---------------------------------------------------------------------------
# NOT HALTED — pre-existing behaviour must be untouched
# ---------------------------------------------------------------------------

def test_not_halted_allows_a_normal_entry():
    allowed, reason = _sleeve(halted=False, current_contracts=0.0).can_trade("BTC", +1.0)
    assert allowed and reason == "ok"


def test_not_halted_still_enforces_the_contract_cap():
    allowed, reason = _sleeve(halted=False, current_contracts=1.0).can_trade("BTC", +1.0)
    assert not allowed
    assert "coinbase_contract_cap" in reason


def test_the_cap_does_not_block_an_exit_either():
    """Reducing from an over-cap position must stay possible."""
    allowed, _ = _sleeve(halted=False, current_contracts=3.0).can_trade("BTC", -1.0)
    assert allowed, "the contract cap blocked a reduce from an over-cap position"


def test_halt_is_checked_before_the_cap_for_adds_but_never_traps_a_reduce():
    """Both gates active at once: an over-cap halted position can still exit."""
    s = _sleeve(halted=True, current_contracts=3.0, max_contracts=1)
    assert s.can_trade("BTC", -1.0)[0], "halted + over-cap position could not reduce"
    assert not s.can_trade("BTC", +1.0)[0], "halted + over-cap position could still add"

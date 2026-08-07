"""[P208] The net-exposure cap, enforced on the book that actually holds risk.

P144 added `max_net_exposure` because the book ran +0.54 net-long into a -23%
market — about half the Apr-Jun loss — and gross caps do not constrain net
DIRECTION. Its only enforcement site is `core/execution_service.py`, which sits
past the P152 early return and reads Kraken-shaped `_paper_positions`, `{}`
since the June flatten. So on the only venue that trades, the cap has never once
been evaluated (P201).

The per-asset contract cap does not substitute: it is per-asset with no
aggregation, so all-three-long is ~+0.5x net while every asset is individually
"within cap" — precisely the shape P144 exists to prevent.

Deliberately NOT routed through GlobalExposureCapManager: that object carries
Kraken-shaped state, and feeding Coinbase positions into it is the cross-venue
contamination P139/P140 came from. Same policy number, enforced locally, on a
book read from the venue.

Two properties matter more than the threshold itself:
  * de-risking is ALWAYS free (the P144 rule, and the P195 lesson about a
    control that traps you in the position it was meant to limit);
  * a pricing failure fails OPEN — a risk control that fires on missing data is
    a data outage that halts trading.
"""

import types

import pytest

from exchange.coinbase_sleeve import CoinbaseSleeve

# contract sizes: BTC 0.01, ETH 0.1, SOL 5.0
_CS = {"BIP-20DEC30-CDE": 0.01, "ETP-20DEC30-CDE": 0.1, "SLP-20DEC30-CDE": 5.0}
_PID = {"BTC": "BIP-20DEC30-CDE", "ETH": "ETP-20DEC30-CDE", "SOL": "SLP-20DEC30-CDE"}
_PX = {"BTC": 64000.0, "ETH": 1900.0, "SOL": 72.0}


class _FakeAdapter:
    def __init__(self, priceable=True):
        self.priceable = priceable
        self._client = types.SimpleNamespace(
            get_product=lambda product_id: {"mid_market_price": str(
                _PX[[k for k, v in _PID.items() if v == product_id][0]])}
            if priceable else {})

    def to_venue_symbol(self, asset, market="perp"): return _PID[asset]
    def _contract_size(self, pid): return _CS.get(pid)


def _sleeve(positions, equity=4000.0, cap=0.50, priceable=True, max_contracts=3):
    """positions: {asset: signed_contracts}. Priced at _PX."""
    s = object.__new__(CoinbaseSleeve)
    s._adapter = _FakeAdapter(priceable)
    s._assets = ("BTC", "ETH", "SOL")
    s._halted = False
    s._halt_reason = ""
    s._max_contracts_per_asset = max_contracts
    s._max_net_exposure = cap
    s._last_equity_usd = equity
    s._last_positions = {
        a: {"signed_contracts": float(c),
            "current_price": _PX[a] if priceable else None}
        for a, c in positions.items()}
    return s


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

class TestExposureMeasurement:

    def test_net_and_gross_from_the_venue_book(self):
        # BTC +1 = 0.01*64000 = 640 ; SOL -1 = -5*72 = -360
        s = _sleeve({"BTC": 1, "SOL": -1}, equity=4000.0)
        e = s.sleeve_exposure()
        assert e["net_usd"] == pytest.approx(280.0)
        assert e["gross_usd"] == pytest.approx(1000.0)
        assert e["net_pct"] == pytest.approx(0.07)
        assert e["priced_ok"] is True

    def test_all_three_long_is_the_p144_shape(self):
        """Every asset individually within the 1-contract cap, ~+0.5x net."""
        s = _sleeve({"BTC": 1, "ETH": 1, "SOL": 1}, equity=4000.0)
        # 640 + 190 + 360 = 1190 on 4000
        assert s.sleeve_exposure()["net_pct"] == pytest.approx(0.2975)

    def test_overrides_let_a_proposed_order_be_evaluated(self):
        s = _sleeve({"BTC": 1}, equity=4000.0)
        assert s.sleeve_exposure(overrides={"BTC": 2})["net_usd"] == pytest.approx(1280.0)

    def test_an_unpriceable_position_is_flagged_not_counted_as_zero(self):
        s = _sleeve({"BTC": 1}, priceable=False)
        e = s.sleeve_exposure()
        assert e["priced_ok"] is False, (
            "a position we cannot price silently counted as zero exposure"
        )


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

class TestNetExposureGate:

    def test_an_order_that_breaches_the_budget_is_blocked(self):
        # equity 1000 -> BTC 1ct = 640 = 64% net, over the 50% budget
        s = _sleeve({}, equity=1000.0, cap=0.50)
        ok, reason = s.can_trade("BTC", +1)
        assert not ok
        assert "coinbase_net_exposure_cap" in reason

    def test_an_order_within_the_budget_passes(self):
        s = _sleeve({}, equity=4000.0, cap=0.50)
        assert s.can_trade("BTC", +1)[0] is True

    def test_reducing_is_always_free_even_when_already_over_budget(self):
        """The P144 rule and the P195 lesson: never trap a position."""
        s = _sleeve({"BTC": 2}, equity=1000.0, cap=0.50)   # ~128% net
        assert s.can_trade("BTC", -1)[0] is True, "blocked a de-risking order"

    def test_flattening_is_always_free_when_over_budget(self):
        s = _sleeve({"BTC": 2}, equity=1000.0, cap=0.50)
        assert s.can_trade("BTC", -2)[0] is True

    def test_a_hedging_order_that_reduces_net_is_free(self):
        """Opposite-side exposure lowers |net| — must not be blocked even though
        it is technically 'opening' a position."""
        s = _sleeve({"BTC": 1}, equity=1000.0, cap=0.50)   # +64% net
        ok, _ = s.can_trade("SOL", -1)                     # -360 -> +28% net
        assert ok is True, "blocked an order that REDUCES net exposure"

    def test_the_aggregate_is_what_binds_not_the_per_asset_cap(self):
        """Each asset stays within the 1-contract cap; the net budget still
        binds. This is the case the contract cap cannot see."""
        s = _sleeve({"BTC": 1, "ETH": 1}, equity=1800.0, cap=0.50, max_contracts=1)
        # 640 + 190 = 830 = 46% ; adding SOL +1 (360) -> 1190 = 66%
        assert s.can_trade("SOL", +1)[0] is False

    def test_disabled_when_no_budget_configured(self):
        s = _sleeve({}, equity=100.0, cap=None)
        assert s.can_trade("BTC", +1)[0] is True

    def test_a_pricing_failure_fails_OPEN(self):
        """A control that fires on missing data is a data outage that halts
        trading. Inconclusive must not mean blocked."""
        s = _sleeve({"BTC": 1}, equity=1000.0, cap=0.50, priceable=False)
        assert s.can_trade("ETH", +1)[0] is True

    def test_the_halt_still_takes_precedence(self):
        s = _sleeve({}, equity=4000.0, cap=0.50)
        s._halted = True
        s._halt_reason = "drawdown"
        ok, reason = s.can_trade("BTC", +1)
        assert not ok and "halted" in reason

    def test_a_halted_sleeve_can_still_reduce(self):
        """P195 composes with P208: neither control traps a position."""
        s = _sleeve({"BTC": 2}, equity=1000.0, cap=0.50)
        s._halted = True
        s._halt_reason = "drawdown"
        assert s.can_trade("BTC", -1)[0] is True

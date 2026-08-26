"""[P210] Per-asset gross exposure, denominated in the sleeve's own equity.

WHY NOT JUST RECONNECT `intent.target_exposure`. It looks like the obvious
missing link — the risk stack computes a per-asset size and the sleeve throws it
away, taking a flat +/-1 contract. But that number is converted at
`core/unit_system.py:232` as `target_exposure * account_equity`, and
`account_equity` comes from `account_sync`, built `exchange_name="kraken"`
(`main.py:2866`). So it is KRAKEN NAV (~$9.8k), not sleeve equity (~$3.8k).

Reconnecting it would therefore:
  * size ~$2,457 at target_exposure=0.25 against the ~$643 the sleeve holds —
    a ~4x increase, on a strategy the system measures at Sharpe -4.5; and
  * denominate one venue's risk in another venue's capital, which is the
    cross-venue contamination P139/P140 came from.

So the chain stays disconnected, and the same POLICY (config
`post_leverage_caps`: BTC/ETH 0.25, SOL 0.20) is enforced natively against the
equity that actually backs the positions.

THE REASON THIS IS NOT A CONTROL THAT CAN NEVER FIRE. Contract granularity is
FIXED while equity moves: one BTC nano is ~17% of a $3.8k sleeve but ~34% of a
$1.9k one. The contract cap counts contracts and cannot see it; P208's net
budget aggregates and so passes a single concentrated asset. This is the only
control that notices, and it says "one contract is now too large for this
account" instead of quietly holding an oversized position.
"""

import types

import pytest

from exchange.coinbase_sleeve import CoinbaseSleeve

_CS = {"BIP-20DEC30-CDE": 0.01, "ETP-20DEC30-CDE": 0.1, "SLP-20DEC30-CDE": 5.0}
_PID = {"BTC": "BIP-20DEC30-CDE", "ETH": "ETP-20DEC30-CDE", "SOL": "SLP-20DEC30-CDE"}
_PX = {"BTC": 64000.0, "ETH": 1900.0, "SOL": 72.0}

# the live policy numbers (configs/live_high_risk.json:post_leverage_caps)
_CAPS = {"BTC": 0.25, "ETH": 0.25, "SOL": 0.20, "XRP": 0.10, "BNB": 0.10}  # [P412] XRP+BNB breadth


class _FakeAdapter:
    def __init__(self, priceable=True):
        self._client = types.SimpleNamespace(
            get_product=lambda product_id: {"mid_market_price": str(
                _PX[[k for k, v in _PID.items() if v == product_id][0]])}
            if priceable else {})

    def to_venue_symbol(self, asset, market="perp"): return _PID[asset]
    def _contract_size(self, pid): return _CS.get(pid)


def _sleeve(positions, equity, caps=_CAPS, priceable=True, max_contracts=3,
            net_cap=None):
    s = object.__new__(CoinbaseSleeve)
    s._adapter = _FakeAdapter(priceable)
    s._assets = ("BTC", "ETH", "SOL")
    s._halted = False
    s._halt_reason = ""
    s._max_contracts_per_asset = max_contracts
    s._max_net_exposure = net_cap
    s._max_asset_exposure = dict(caps or {})
    s._last_equity_usd = equity
    s._last_positions = {
        a: {"signed_contracts": float(c),
            "current_price": _PX[a] if priceable else None}
        for a, c in positions.items()}
    return s


# ---------------------------------------------------------------------------
# the shrinking-account case this exists for
# ---------------------------------------------------------------------------

class TestGranularityVsShrinkingEquity:

    def test_one_btc_contract_is_within_cap_at_todays_equity(self):
        """~$643 on ~$3,772 = 17% < 25%. Must not block normal trading."""
        s = _sleeve({}, equity=3772.0)
        ok, reason = s.can_trade("BTC", +1)
        assert ok is True, f"blocked a normal entry: {reason}"

    def test_the_same_contract_is_blocked_once_equity_halves(self):
        """THE POINT. $643 on $1,900 = 34% > 25%. Contract count is unchanged,
        so neither the contract cap nor the aggregate net budget notices."""
        s = _sleeve({}, equity=1900.0)
        ok, reason = s.can_trade("BTC", +1)
        assert not ok
        assert "coinbase_asset_exposure_cap" in reason

    def test_the_contract_cap_alone_would_have_allowed_it(self):
        s = _sleeve({}, equity=1900.0, caps=None)
        assert s.can_trade("BTC", +1)[0] is True, (
            "control is redundant — the contract cap already blocked this"
        )

    def test_the_net_budget_alone_would_have_allowed_it(self):
        """P208 aggregates, so one concentrated asset passes: 34% < 50%."""
        s = _sleeve({}, equity=1900.0, caps=None, net_cap=0.50)
        assert s.can_trade("BTC", +1)[0] is True

    def test_sol_uses_its_own_tighter_cap(self):
        """SOL 0.20 in the live config, not 0.25."""
        # 1 SOL contract = 5 * 72 = $360; 20% cap binds below $1,800
        assert _sleeve({}, equity=1900.0).can_trade("SOL", +1)[0] is True
        ok, reason = _sleeve({}, equity=1700.0).can_trade("SOL", +1)
        assert not ok and "coinbase_asset_exposure_cap" in reason


# ---------------------------------------------------------------------------
# never trap a position (the P195/P208 invariant)
# ---------------------------------------------------------------------------

class TestDeRiskingIsAlwaysFree:

    def test_reducing_an_over_cap_position_is_allowed(self):
        s = _sleeve({"BTC": 2}, equity=1900.0)   # ~67%, far over cap
        assert s.can_trade("BTC", -1)[0] is True, "blocked a de-risking order"

    def test_flattening_an_over_cap_position_is_allowed(self):
        s = _sleeve({"BTC": 2}, equity=1900.0)
        assert s.can_trade("BTC", -2)[0] is True

    def test_holding_over_cap_is_not_retroactively_blocked(self):
        """The cap gates ORDERS, not existing positions — an asset that drifts
        over cap because equity fell must still be manageable."""
        s = _sleeve({"BTC": 1}, equity=1900.0)
        assert s.can_trade("BTC", 0)[0] is True

    def test_a_flip_that_does_not_increase_exposure_is_allowed(self):
        """+1 -> -1 keeps |exposure| identical, so it is not an increase."""
        s = _sleeve({"BTC": 1}, equity=1900.0)
        assert s.can_trade("BTC", -2)[0] is True


# ---------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------

class TestFailureModes:

    def test_a_pricing_failure_fails_OPEN(self):
        """Same rule as P208: a control that fires on missing data is a data
        outage that halts trading."""
        s = _sleeve({}, equity=1900.0, priceable=False)
        assert s.can_trade("BTC", +1)[0] is True

    def test_zero_equity_fails_OPEN(self):
        s = _sleeve({}, equity=0.0)
        assert s.can_trade("BTC", +1)[0] is True

    def test_disabled_when_no_caps_configured(self):
        s = _sleeve({}, equity=100.0, caps=None)
        assert s.can_trade("BTC", +1)[0] is True

    def test_an_asset_absent_from_the_cap_map_is_ungated(self):
        s = _sleeve({}, equity=1900.0, caps={"ETH": 0.25})
        assert s.can_trade("BTC", +1)[0] is True

    def test_missing_attribute_does_not_refuse_every_order(self):
        """P85, and the third time this session a hot-path attribute read had to
        defend itself: an AttributeError here blocks ALL trading."""
        s = _sleeve({}, equity=3772.0)
        del s._max_asset_exposure
        assert s.can_trade("BTC", +1)[0] is True

    def test_the_halt_still_takes_precedence(self):
        s = _sleeve({}, equity=3772.0)
        s._halted = True
        s._halt_reason = "drawdown"
        ok, reason = s.can_trade("BTC", +1)
        assert not ok and "halted" in reason


# ---------------------------------------------------------------------------
# the decision not to reconnect target_exposure
# ---------------------------------------------------------------------------

class TestTargetExposureStaysDisconnected:

    def test_the_kraken_denominator_is_still_what_unit_system_uses(self):
        """If this ever changes to a per-venue denominator, revisit the
        decision recorded here rather than discovering it by surprise."""
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "core" / "unit_system.py"
               ).read_text(encoding="utf-8", errors="replace")
        assert "usd_notional = abs(exposure_fraction) * account_equity" in src

    def test_the_sleeve_still_ignores_magnitude(self):
        """target_for_signal maps to +/-1 contract regardless of conviction —
        deliberate, and the reason the cap above is denominated independently."""
        assert CoinbaseSleeve.target_for_signal(+0.20) == 1
        assert CoinbaseSleeve.target_for_signal(+0.99) == 1
        assert CoinbaseSleeve.target_for_signal(-0.99) == -1

    def test_the_wired_policy_matches_the_kraken_one(self):
        import json
        from pathlib import Path
        cfg = json.loads(
            (Path(__file__).resolve().parents[1] / "configs" /
             "live_high_risk.json").read_text(encoding="utf-8"))
        assert cfg["post_leverage_caps"] == _CAPS, (
            "live per-asset caps moved; this suite's numbers are now fiction"
        )

    def test_the_sleeve_is_constructed_with_those_caps(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "main.py").read_text(
            encoding="utf-8", errors="replace")
        i = src.index("max_asset_exposure=getattr(")
        assert '"post_leverage_caps"' in src[i:i + 200]


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

class TestComposition:

    def test_net_and_per_asset_caps_compose(self):
        """Both armed: the tighter one binds, and neither traps a position."""
        s = _sleeve({}, equity=1900.0, net_cap=0.50)
        assert s.can_trade("BTC", +1)[0] is False      # per-asset 34% > 25%
        assert s.can_trade("ETH", +1)[0] is True       # 10% of equity, fine

    def test_reducing_passes_both(self):
        s = _sleeve({"BTC": 2}, equity=1900.0, net_cap=0.50)
        assert s.can_trade("BTC", -1)[0] is True

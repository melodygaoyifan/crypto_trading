"""
HMATS v5.1 Phase 2 - CoinbaseSleeve tests (separate-sleeve state tracking).

Verifies venue-authoritative reconciliation (anti-P139), signed-contract math,
buying-power read, and fail-soft behavior — all with a mock venue client.
"""
from exchange.coinbase_sleeve import CoinbaseSleeve


class FakeClient:
    def __init__(self, positions=None, bp="4000", fail=False):
        self._positions = positions if positions is not None else []
        self._bp = bp
        self._fail = fail

    def list_futures_positions(self):
        if self._fail:
            raise RuntimeError("venue boom")
        return {"positions": self._positions}

    def get_futures_balance_summary(self):
        return {"balance_summary": {"futures_buying_power": {"value": self._bp}}}

    # [P165] P153 moved `sleeve_equity_usd()` off the futures-balance summary
    # and onto the Default PORTFOLIO's `total_balance` (the cross-collateralized
    # equity). This mock only spoke the old API, so after P153 the sleeve fell
    # through to the degraded fallback, read total_usd_balance=0, and held
    # `_last_equity_usd` constant — meaning `test_drawdown_halts_*` moved `bp`
    # and the equity never budged. The fixture, not the guard, was stale.
    def get_portfolios(self):
        return {"portfolios": [{"uuid": "fake-default-uuid", "type": "DEFAULT"}]}

    def get_portfolio_breakdown(self, portfolio_uuid=None, **kw):
        return {
            "breakdown": {
                "portfolio_balances": {"total_balance": {"value": self._bp}}
            }
        }

    def get_product(self, product_id, **kw):
        return {"product_id": product_id, "mid_market_price": "100",
                "future_product_details": {"contract_size": "1"}}


class FakeAdapterFull:
    """Richer adapter mock that supports execute_target/manage_to_signal."""
    def __init__(self, positions=None):
        self._client = FakeClient(positions)
        self.placed = []

    def is_connected(self):
        return True

    # [P287] The real adapter has this; before P287 its ABSENCE here was
    # invisible because the sweep swallowed the AttributeError and
    # "proceeded" — the exact fail-open contract P287 retired. An empty
    # book is the honest fixture state.
    async def fetch_open_orders(self, symbol=None):
        return []

    async def cancel_order(self, order_id, symbol):
        return True

    def to_venue_symbol(self, asset, market="perp"):
        from exchange.symbol_mapping import to_venue_symbol
        return to_venue_symbol(asset, "coinbase", market)

    def _contract_size(self, pid):
        return {"SLP-20DEC30-CDE": 5.0, "ETP-20DEC30-CDE": 0.1,
                "BIP-20DEC30-CDE": 0.01}.get(pid, 1.0)

    async def place_order(self, req):
        from exchange.adapter import OrderResult
        self.placed.append(req)
        return OrderResult(success=True, venue="coinbase", order_id="X", status="SUBMITTED")


class FakeAdapter:
    def __init__(self, client, connected=True):
        self._client = client
        self._connected = connected

    def is_connected(self):
        return self._connected


def _sleeve(positions=None, bp="4000", fail=False, connected=True):
    return CoinbaseSleeve(FakeAdapter(FakeClient(positions, bp, fail), connected))


def test_reconcile_long_signed_contracts():
    s = _sleeve([{"product_id": "BIP-20DEC30-CDE", "side": "LONG",
                  "number_of_contracts": "3"}])
    pos = s.reconcile_positions()
    assert "BTC" in pos
    assert pos["BTC"]["signed_contracts"] == 3.0
    assert s.signed_contracts("BTC") == 3.0


def test_reconcile_short_is_negative():
    s = _sleeve([{"product_id": "SLP-20DEC30-CDE", "side": "SHORT",
                  "number_of_contracts": "2"}])
    s.reconcile_positions()
    assert s.signed_contracts("SOL") == -2.0


def test_unknown_product_skipped():
    s = _sleeve([{"product_id": "DOGE-PERP-INTX", "side": "LONG",
                  "number_of_contracts": "5"}])
    assert s.reconcile_positions() == {}


def test_buying_power_parsed():
    assert _sleeve(bp="4000").buying_power_usd() == 4000.0


def test_not_ready_returns_empty():
    s = _sleeve(connected=False)
    assert s.reconcile_positions() == {}
    assert s.is_ready() is False


def test_fail_soft_returns_last_snapshot():
    # first a good reconcile, then the venue errors -> keep last snapshot
    client = FakeClient([{"product_id": "ETP-20DEC30-CDE", "side": "LONG",
                          "number_of_contracts": "1"}])
    s = CoinbaseSleeve(FakeAdapter(client))
    assert s.reconcile_positions()["ETH"]["signed_contracts"] == 1.0
    client._fail = True
    # error path returns the last good snapshot, never raises
    assert s.reconcile_positions()["ETH"]["signed_contracts"] == 1.0


def test_can_trade_contract_cap():
    s = _sleeve()  # default max_contracts_per_asset=1
    s.reconcile_positions()
    assert s.can_trade("BTC", 1)[0] is True    # 0 + 1 = 1 OK
    assert s.can_trade("BTC", 2)[0] is False   # 0 + 2 = 2 > 1


def test_can_trade_with_existing_position_allows_close():
    s = _sleeve([{"product_id": "BIP-20DEC30-CDE", "side": "LONG",
                  "number_of_contracts": "1"}])
    s.reconcile_positions()
    assert s.can_trade("BTC", 1)[0] is False   # 1 + 1 = 2 > cap
    assert s.can_trade("BTC", -1)[0] is True   # 1 - 1 = 0 (closing) OK


def test_drawdown_halts_and_blocks_then_manual_reset():
    client = FakeClient(bp="4000")
    s = CoinbaseSleeve(FakeAdapter(client), max_sleeve_drawdown_pct=0.15)
    assert s.update_risk()["halted"] is False          # baseline 4000
    client._bp = "3000"                                 # -25% drawdown
    r = s.update_risk()
    assert r["halted"] is True and r["drawdown_pct"] == 0.25
    assert s.can_trade("BTC", 1)[0] is False            # halt blocks trades
    s.reset_halt()
    assert s.can_trade("BTC", 1)[0] is True             # manual recovery clears


def test_target_for_signal_flattens_below_threshold():
    f = CoinbaseSleeve.target_for_signal
    assert f(0.5) == 1
    assert f(-0.5) == -1
    assert f(0.15) == 1           # at threshold
    assert f(0.10) == 0           # below threshold -> FLATTEN (the exit fix)
    assert f(-0.05) == 0
    assert f(0.0) == 0


def test_manage_to_signal_flattens_held_position_on_hold():
    import asyncio
    a = FakeAdapterFull([{"product_id": "SLP-20DEC30-CDE", "side": "LONG",
                          "number_of_contracts": "1"}])
    s = CoinbaseSleeve(a)
    # held LONG 1, signal goes neutral (0.05 < 0.15) -> target 0 -> places a SELL
    asyncio.run(s.manage_to_signal("SOL", 0.05))
    assert len(a.placed) == 1
    assert a.placed[0].side == "SELL"


def test_manage_skips_on_stale_reconcile():
    import asyncio
    a = FakeAdapterFull()
    a._client._fail = True  # venue reconcile times out
    s = CoinbaseSleeve(a)
    r = asyncio.run(s.manage_to_signal("ETH", 0.5))
    assert r["status"] == "SKIPPED_STALE"
    assert len(a.placed) == 0  # never trades on a stale snapshot


def test_manage_to_signal_noop_when_already_aligned():
    import asyncio
    a = FakeAdapterFull([{"product_id": "SLP-20DEC30-CDE", "side": "LONG",
                          "number_of_contracts": "1"}])
    s = CoinbaseSleeve(a)
    # held LONG 1, strong long signal -> target +1 == current -> NOOP, no order
    r = asyncio.run(s.manage_to_signal("SOL", 0.6))
    assert r["status"] == "NOOP"
    assert len(a.placed) == 0


def test_snapshot_shape():
    s = _sleeve([{"product_id": "BIP-20DEC30-CDE", "side": "LONG",
                  "number_of_contracts": "1"}])
    snap = s.snapshot()
    assert snap["venue"] == "coinbase"
    assert snap["buying_power_usd"] == 4000.0
    assert "BTC" in snap["positions"]

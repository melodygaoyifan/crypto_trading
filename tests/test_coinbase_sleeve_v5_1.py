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


def test_snapshot_shape():
    s = _sleeve([{"product_id": "BIP-20DEC30-CDE", "side": "LONG",
                  "number_of_contracts": "1"}])
    snap = s.snapshot()
    assert snap["venue"] == "coinbase"
    assert snap["buying_power_usd"] == 4000.0
    assert "BTC" in snap["positions"]

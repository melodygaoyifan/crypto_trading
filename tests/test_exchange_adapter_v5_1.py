"""Phase 2 exchange adapter + symbol mapping + routing tests.

Verifies:
  - Symbol mapping bidirectional + supported_assets
  - KrakenAdapter wraps cleanly (fails closed when ccxt missing)
  - CoinbaseAdapter skeleton (fails closed; documented stub)
  - RoutingPolicy phase transitions with Iron Law 8 enforcement
  - DRL not ACTIVE → advance refused
  - ROLLBACK always permitted
"""

import pytest

from exchange.adapter import (
    ExchangeAdapter,
    FundingRateData,
    OrderRequest,
    OrderResult,
)
from exchange.coinbase_adapter import CoinbaseAdapter
from exchange.kraken_adapter import KrakenAdapter
from exchange.routing import CutoverPhase, RoutingPolicy
from exchange.symbol_mapping import (
    SYMBOL_MAP,
    from_venue_symbol,
    supported_assets,
    to_venue_symbol,
)


# ---------- symbol_mapping ------------------------------------------------

@pytest.mark.parametrize("asset,venue,market,expected", [
    ("BTC", "kraken", "perp", "PF_XBTUSD"),
    ("ETH", "kraken", "perp", "PF_ETHUSD"),
    ("SOL", "kraken", "perp", "PF_SOLUSD"),
    ("BTC", "coinbase", "perp", "BIP-20DEC30-CDE"),
    ("ETH", "coinbase", "perp", "ETP-20DEC30-CDE"),
    ("SOL", "coinbase", "perp", "SLP-20DEC30-CDE"),
    ("BTC", "coinbase", "spot", "BTC-USD"),
    # [P253d] BTC/USDT -> BTC/USD: the map was aligned to
    # core/execution_service._CANONICAL_SPOT_SYMBOL (BTC/ETH trade USD;
    # SOL stays USDT per P133/P135/P137 — its USD pair died on Kraken).
    # The full per-asset parity is pinned in test_p253_readthrough_fixes.
    ("BTC", "kraken", "spot", "BTC/USD"),
    ("SOL", "kraken", "spot", "SOL/USDT"),
    ("btc", "KRAKEN", "PERP", "PF_XBTUSD"),  # case-insensitive
])
def test_to_venue_symbol_known(asset, venue, market, expected):
    assert to_venue_symbol(asset, venue, market) == expected


def test_to_venue_symbol_unknown_asset_raises():
    with pytest.raises(KeyError):
        to_venue_symbol("DOGE", "kraken", "perp")


def test_to_venue_symbol_unknown_venue_raises():
    with pytest.raises(KeyError):
        to_venue_symbol("BTC", "binance", "perp")


def test_to_venue_symbol_unknown_market_raises():
    with pytest.raises(KeyError):
        to_venue_symbol("BTC", "kraken", "futures")


@pytest.mark.parametrize("venue,market,vsym,expected", [
    ("kraken", "perp", "PF_XBTUSD", "BTC"),
    ("coinbase", "perp", "ETP-20DEC30-CDE", "ETH"),
    ("coinbase", "spot", "SOL-USD", "SOL"),
])
def test_from_venue_symbol_inverse(venue, market, vsym, expected):
    assert from_venue_symbol(vsym, venue, market) == expected


def test_from_venue_symbol_unknown_raises():
    with pytest.raises(KeyError):
        from_venue_symbol("UNKNOWN", "kraken", "perp")


def test_supported_assets():
    assert set(supported_assets("kraken", "perp")) == {"BTC", "ETH", "SOL"}
    assert set(supported_assets("coinbase", "perp")) == {"BTC", "ETH", "SOL"}
    assert supported_assets("unknown", "perp") == []


# ---------- KrakenAdapter -------------------------------------------------

def test_kraken_adapter_venue():
    a = KrakenAdapter()
    assert a.venue == "kraken"


def test_kraken_adapter_to_venue_symbol():
    a = KrakenAdapter()
    assert a.to_venue_symbol("BTC", "perp") == "PF_XBTUSD"


def test_kraken_adapter_place_order_returns_delegated():
    """During dual-venue, KrakenAdapter.place_order returns DELEGATED;
    actual execution flows through execution_manager.execute_order."""
    import asyncio
    a = KrakenAdapter(ccxt_client=object())  # mock client to bypass init
    req = OrderRequest(symbol="PF_XBTUSD", side="BUY", size=0.001)
    res = asyncio.run(a.place_order(req))
    assert res.venue == "kraken"
    assert res.error_code == "DELEGATED"


def test_kraken_adapter_no_client_init_failure():
    """When ccxt unavailable / fails, adapter records init failure but
    doesn't crash. fetch_balance returns {} fail-closed."""
    import asyncio
    a = KrakenAdapter()
    # Force init to fail by mocking the import
    a._client_init_failed = "ccxt_test_error"
    assert a.is_connected() is False
    bal = asyncio.run(a.fetch_balance())
    assert bal == {}


# ---------- CoinbaseAdapter (skeleton) -------------------------------------

def test_coinbase_adapter_venue():
    a = CoinbaseAdapter()
    assert a.venue == "coinbase"


@pytest.fixture
def unconfigured_adapter(monkeypatch):
    """A CoinbaseAdapter guaranteed to have NO credentials, regardless of any
    ambient .coinbase_key.json or COINBASE_* env on the dev/CI machine. This
    keeps the fail-closed contract tests hermetic AND prevents them from ever
    issuing a live API call."""
    import exchange.coinbase_adapter as cba
    monkeypatch.delenv("COINBASE_API_KEY", raising=False)
    monkeypatch.delenv("COINBASE_API_SECRET", raising=False)
    monkeypatch.delenv("COINBASE_KEY_FILE", raising=False)
    monkeypatch.setattr(cba, "_DEFAULT_KEY_FILE", "/nonexistent/.coinbase_key.json")
    return cba.CoinbaseAdapter()


def test_coinbase_adapter_returns_not_configured(unconfigured_adapter):
    """place_order fails closed with NOT_CONFIGURED when no creds present."""
    import asyncio
    req = OrderRequest(symbol="BTC-PERP", side="BUY", size=0.001)
    res = asyncio.run(unconfigured_adapter.place_order(req))
    assert res.success is False
    assert res.error_code == "NOT_CONFIGURED"


def test_coinbase_adapter_balance_empty_when_unconfigured(unconfigured_adapter):
    import asyncio
    assert asyncio.run(unconfigured_adapter.fetch_balance()) == {}


def test_coinbase_adapter_fetch_funding_raises_when_unconfigured(unconfigured_adapter):
    import asyncio
    with pytest.raises(RuntimeError, match="not_configured"):
        asyncio.run(unconfigured_adapter.fetch_funding_rate("BTC-PERP"))


def test_coinbase_adapter_to_venue_symbol():
    a = CoinbaseAdapter()
    assert a.to_venue_symbol("BTC", "perp") == "BIP-20DEC30-CDE"
    assert a.to_venue_symbol("BTC", "spot") == "BTC-USD"


# ---------- RoutingPolicy + Iron Law 8 -----------------------------------

def test_routing_pre_phase_2_all_kraken():
    p = RoutingPolicy()
    assert p.venue_for("BTC") == "kraken"
    assert p.venue_for("SOL") == "kraken"


def test_routing_shadow_orders_still_kraken():
    """SHADOW = Coinbase read-only; orders still route to Kraken."""
    p = RoutingPolicy(phase=CutoverPhase.SHADOW)
    assert p.venue_for("BTC") == "kraken"
    assert p.venue_for("SOL") == "kraken"


def test_routing_dual_venue_split():
    """Default split: SOL → Coinbase, BTC+ETH → Kraken."""
    p = RoutingPolicy(phase=CutoverPhase.DUAL_VENUE)
    assert p.venue_for("BTC") == "kraken"
    assert p.venue_for("ETH") == "kraken"
    assert p.venue_for("SOL") == "coinbase"


def test_routing_coinbase_primary_all_coinbase():
    p = RoutingPolicy(phase=CutoverPhase.COINBASE_PRIMARY)
    for asset in ("BTC", "ETH", "SOL"):
        assert p.venue_for(asset) == "coinbase"


def test_routing_rollback_returns_to_kraken():
    p = RoutingPolicy(phase=CutoverPhase.ROLLBACK)
    for asset in ("BTC", "ETH", "SOL"):
        assert p.venue_for(asset) == "kraken"


def test_advance_phase_iron_law_8_blocks_when_drl_demoted():
    p = RoutingPolicy()
    ok, reason = p.advance_phase(CutoverPhase.SHADOW, drl_authority_level="SHADOW")
    assert ok is False
    # [P155] advance_phase now delegates to cutover.validate_drl_active instead
    # of re-implementing the comparison, so the reason is that function's
    # wording ("Iron Law 5/8 violation") rather than a second copy of it.
    assert "5/8" in reason and "SHADOW" in reason
    assert p.phase == CutoverPhase.PRE_PHASE_2


def test_advance_phase_fails_closed_on_missing_drl_level():
    """[P155] The old inline check compared `str(None).upper() != "ACTIVE"`, so
    it happened to block — but an empty string was never explicitly handled.
    Delegating to validate_drl_active makes fail-closed the documented
    behaviour."""
    for level in ("", None, "unknown"):
        p = RoutingPolicy()
        ok, _ = p.advance_phase(CutoverPhase.SHADOW, drl_authority_level=level)
        assert ok is False, f"level={level!r} must not advance the cutover"
        assert p.phase == CutoverPhase.PRE_PHASE_2


def test_advance_phase_drl_active_permits():
    p = RoutingPolicy()
    ok, _ = p.advance_phase(CutoverPhase.SHADOW, drl_authority_level="ACTIVE")
    assert ok is True
    assert p.phase == CutoverPhase.SHADOW
    assert p.phase_started_at is not None


def test_advance_phase_rollback_always_permitted():
    """ROLLBACK bypasses both Iron Law 8 + monotonic progression."""
    p = RoutingPolicy(phase=CutoverPhase.COINBASE_PRIMARY)
    ok, _ = p.advance_phase(CutoverPhase.ROLLBACK, drl_authority_level="SHADOW")
    assert ok is True
    assert p.phase == CutoverPhase.ROLLBACK


def test_advance_phase_invalid_transition_blocked():
    """Cannot skip from PRE_PHASE_2 → COINBASE_PRIMARY directly."""
    p = RoutingPolicy()
    ok, reason = p.advance_phase(CutoverPhase.COINBASE_PRIMARY, drl_authority_level="ACTIVE")
    assert ok is False
    assert "invalid transition" in reason


def test_advance_phase_full_progression():
    p = RoutingPolicy()
    assert p.phase == CutoverPhase.PRE_PHASE_2
    assert p.advance_phase(CutoverPhase.SHADOW, "ACTIVE")[0]
    assert p.advance_phase(CutoverPhase.DUAL_VENUE, "ACTIVE")[0]
    assert p.advance_phase(CutoverPhase.COINBASE_PRIMARY, "ACTIVE")[0]
    assert p.phase == CutoverPhase.COINBASE_PRIMARY


def test_routing_to_dict_serializable():
    p = RoutingPolicy()
    p.advance_phase(CutoverPhase.SHADOW, "ACTIVE")
    d = p.to_dict()
    assert d["phase"] == "SHADOW"
    assert d["phase_started_at"]


def test_is_dual_venue_active():
    p = RoutingPolicy(phase=CutoverPhase.PRE_PHASE_2)
    assert p.is_dual_venue_active() is False
    p.phase = CutoverPhase.SHADOW
    assert p.is_dual_venue_active() is True
    p.phase = CutoverPhase.DUAL_VENUE
    assert p.is_dual_venue_active() is True
    p.phase = CutoverPhase.COINBASE_PRIMARY
    assert p.is_dual_venue_active() is False


# ---------- ExchangeAdapter ABC ------------------------------------------

def test_adapter_abc_cant_instantiate():
    with pytest.raises(TypeError):
        ExchangeAdapter()

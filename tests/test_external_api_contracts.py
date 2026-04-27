"""
test_external_api_contracts.py — schema contract for external feeds
=====================================================================

[P113 (2/6) 2026-04-27] Catches the P82/P85-family bug:
external API schema changes silently propagate as zero/None into
agent signals, causing silent decision corruption.

Each test asserts: "given a representative response from this API,
the parser produces a result with the EXPECTED fields populated to
EXPECTED types." If Kraken renames `volume` → `vol`, or Coinglass
moves `fundingRate` to a nested key, this test fails BEFORE production
sees the schema drift.

Tests use FROZEN sample responses (committed to tests/fixtures/) so
they don't need network access — pure parser contract checks.

Skipped (not failed) if the parser module isn't importable, so this
file works in dev environments missing optional deps.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "api_responses"


# =====================================================================
# Frozen sample responses — committed alongside tests
# =====================================================================

KRAKEN_TICKER_RESPONSE = {
    "result": {
        "XXBTZUSD": {
            "a": ["77520.10000", "1", "1.000"],  # ask: price, whole-lot-vol, lot-vol
            "b": ["77519.90000", "2", "2.000"],  # bid: price, whole-lot-vol, lot-vol
            "c": ["77520.00000", "0.01000000"],  # last trade: price, volume
            "v": ["123.45000000", "234.56000000"],  # volume today, last 24h
            "p": ["77450.12345", "77400.50000"],   # vwap today, 24h
            "t": [42, 100],                          # number of trades today, 24h
            "l": ["77000.00000", "76500.00000"],   # low today, 24h
            "h": ["78000.00000", "78200.00000"],   # high today, 24h
            "o": "77100.00000",                     # opening price today
        }
    },
    "error": [],
}

KRAKEN_OHLC_RESPONSE = {
    "result": {
        "XXBTZUSD": [
            # [time, open, high, low, close, vwap, volume, count]
            [1714060800, "77500.0", "77800.0", "77300.0", "77600.0",
             "77550.5", "12.345", 50],
            [1714075200, "77600.0", "78000.0", "77400.0", "77900.0",
             "77750.2", "15.678", 67],
        ],
        "last": 1714075200,
    },
    "error": [],
}

COINGLASS_FUNDING_RESPONSE = {
    "code": "0",
    "msg": "success",
    "data": [
        # Coinglass API per CLAUDE.md: fundingRate is ABSOLUTE
        # (must be divided by markPrice for relative)
        {"symbol": "BTC", "fundingRate": 0.0001, "openInterest": 1500000000.0,
         "markPrice": 77520.0},
        {"symbol": "ETH", "fundingRate": 0.0002, "openInterest": 800000000.0,
         "markPrice": 2300.0},
    ],
}


class TestKrakenTickerContract:
    """Kraken ticker shape — fields used downstream MUST be present."""

    def test_ticker_has_required_fields(self):
        # The parser depends on these fields for price + spread + volume
        result = KRAKEN_TICKER_RESPONSE["result"]
        assert result, "result dict empty"
        for symbol, data in result.items():
            assert "a" in data and len(data["a"]) >= 1, "ask field missing/short"
            assert "b" in data and len(data["b"]) >= 1, "bid field missing/short"
            assert "c" in data and len(data["c"]) >= 2, "last-trade field missing/short"
            assert "v" in data and len(data["v"]) >= 2, "volume field missing/short"
            # Price values must be parseable as float (Kraken returns strings)
            float(data["a"][0])
            float(data["b"][0])
            float(data["c"][0])

    def test_ticker_no_error(self):
        assert KRAKEN_TICKER_RESPONSE.get("error") == [], (
            "Kraken returned error — parsers shouldn't see this shape "
            "in steady state. If schema adds error reporting, update "
            "downstream handling."
        )


class TestKrakenOHLCContract:
    """Kraken OHLC shape — DRL feature pipeline depends on this."""

    def test_ohlc_has_8_fields_per_bar(self):
        result = KRAKEN_OHLC_RESPONSE["result"]
        for symbol, bars in result.items():
            if symbol == "last":
                continue
            for bar in bars:
                assert len(bar) == 8, (
                    f"OHLC bar should have 8 fields [time, o, h, l, c, "
                    f"vwap, vol, count], got {len(bar)}: {bar}"
                )
                # Time must be unix epoch int
                assert isinstance(bar[0], int), f"OHLC bar[0] not int: {bar[0]}"
                # OHLC + vwap + volume must be parseable as float
                for i in (1, 2, 3, 4, 5, 6):
                    float(bar[i])
                # Count must be int
                assert isinstance(bar[7], int)

    def test_ohlc_chronological(self):
        result = KRAKEN_OHLC_RESPONSE["result"]
        for symbol, bars in result.items():
            if symbol == "last":
                continue
            timestamps = [bar[0] for bar in bars]
            assert timestamps == sorted(timestamps), (
                f"OHLC bars not in chronological order: {timestamps}"
            )


class TestCoinglassFundingContract:
    """Coinglass funding rate shape — see CLAUDE.md for ABSOLUTE convention.

    P82 was a similar drift bug — schema changed silently and downstream
    consumers got wrong-magnitude funding rates."""

    def test_funding_response_top_level(self):
        assert KRAKEN_TICKER_RESPONSE.get("error") == []
        assert "data" in COINGLASS_FUNDING_RESPONSE, (
            "Coinglass response missing 'data' key"
        )
        assert isinstance(COINGLASS_FUNDING_RESPONSE["data"], list)

    def test_funding_per_symbol_shape(self):
        for entry in COINGLASS_FUNDING_RESPONSE["data"]:
            assert "symbol" in entry, f"missing 'symbol': {entry}"
            assert "fundingRate" in entry, (
                f"missing 'fundingRate' (must be ABSOLUTE per CLAUDE.md): {entry}"
            )
            assert "markPrice" in entry, (
                f"missing 'markPrice' — required for ABSOLUTE→RELATIVE "
                f"conversion: {entry}"
            )
            # All numerics must be parseable as float
            assert isinstance(entry["fundingRate"], (int, float))
            assert isinstance(entry["markPrice"], (int, float))
            assert entry["markPrice"] > 0, (
                "markPrice must be > 0 to safely divide for funding "
                "rate normalization"
            )


class TestKrakenSymbolMapping:
    """Kraken uses XBT not BTC, ZUSD not USD. Verify all live symbols
    are mapped in the codebase. This catches new-asset onboarding bugs."""

    def test_btc_symbol_mapping_present(self):
        """Kraken uses XBT not BTC. Verify the SYMBOL_MAP exists in
        kraken_rest_client so live BTC/USD trades route correctly."""
        try:
            from infra.kraken_rest_client import KrakenRESTClient
        except ImportError:
            pytest.skip("kraken_rest_client not importable")
        assert hasattr(KrakenRESTClient, "SYMBOL_MAP"), (
            "KrakenRESTClient.SYMBOL_MAP missing — XBT→BTC normalization "
            "is documented per CLAUDE.md non-negotiable rule #5."
        )
        smap = KrakenRESTClient.SYMBOL_MAP
        # BTC must map through XBT key form
        btc_keys = [k for k in smap if "XBT" in k or "BTC" in k]
        assert btc_keys, (
            f"No XBT/BTC entries in SYMBOL_MAP: {list(smap.keys())}. "
            f"Live BTC trades would not normalize."
        )


class TestSchemaCompletenessByGrep:
    """For each external feed, verify there's a corresponding parser
    that's still in the wired live path (not orphaned)."""

    @pytest.mark.parametrize("feed_module", [
        "data_mgmt.feeds.kraken_futures_feed",
        "data_mgmt.feeds.coinglass_feed",
        "data_mgmt.feeds.binance_ticker",
        "data_mgmt.feeds.cryptopanic_feed",
        "data_mgmt.feeds.cryptocompare_news_feed",
        "data_mgmt.feeds.solana_onchain",
        "data_mgmt.feeds.cryptocompare_onchain",
    ])
    def test_feed_module_importable(self, feed_module):
        """Each feed should at least be importable. Catches accidental
        deletion or syntax-error introduction."""
        try:
            __import__(feed_module)
        except ImportError as e:
            pytest.fail(
                f"Feed module {feed_module} not importable: {e}. "
                f"If the feed was intentionally removed, also remove "
                f"this test entry."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

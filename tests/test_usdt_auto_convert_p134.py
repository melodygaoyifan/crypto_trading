"""
test_usdt_auto_convert_p134.py — USD->USDT auto-conversion path (P134)
================================================================================

P133 migrated SOL spot to SOL/USDT. SOL LONG entries blocked because operator
had only $30 USDT free vs $3941 USD. P134 adds auto-conversion via Kraken
USDT/USD pair before SOL/USDT BUY orders.

Safety design:
  - Env-gated (HMATS_USDT_AUTO_CONVERT_ENABLED, default OFF)
  - Per-order cap (HMATS_USDT_AUTO_CONVERT_MAX_USD, default $200)
  - Fail-CLOSED: any conversion failure -> SOL order REJECTED
  - PREFLIGHT_QUOTE_CONVERT_FAILED prefix maps to PERMANENT in P79 classifier
  - All conversions log CRITICAL for operator visibility

Tests verify:
  - Default OFF behavior (env unset -> no-op)
  - SELL orders skip conversion (no quote currency needed)
  - Non-USDT pairs skip conversion (BTC/USD untouched)
  - Sufficient USDT skips conversion (no waste)
  - Per-order cap enforces (no runaway conversions)
  - Insufficient USD fails-CLOSED (no silent error)
  - Error prefix maps to PERMANENT (no retry storm)
  - Source markers preserved
"""
from __future__ import annotations

import inspect
import os
from unittest.mock import MagicMock, patch

import pytest


def _make_em_with_mocks(usd_free=3941.0, usdt_free=30.0, usdt_ask=0.99978):
    """Build a minimal ExecutionManager with mocked exchange + balance."""
    from execution.execution_manager import ExecutionManager
    em = ExecutionManager.__new__(ExecutionManager)
    em.exchange = MagicMock()
    em.logger = MagicMock()
    em.exchange.fetch_balance.return_value = {
        "free": {"USD": usd_free, "USDT": usdt_free, "SOL": 8.14, "BTC": 0.018, "ETH": 0.59},
    }
    # Default ticker probes
    def _ticker(symbol):
        if symbol == "USDT/USD":
            return {"ask": usdt_ask, "bid": usdt_ask - 1e-5, "last": usdt_ask}
        if symbol == "SOL/USDT":
            return {"ask": 84.05, "bid": 84.04, "last": 84.05}
        return {"ask": 0, "bid": 0, "last": 0}
    em.exchange.fetch_ticker.side_effect = _ticker
    em.exchange.market.side_effect = lambda s: {
        "limits": {"amount": {"min": 5.0 if s == "USDT/USD" else 0.02}}
    }
    em.exchange.create_market_order.return_value = {
        "id": "TEST_ORDER_ID",
        "filled": 200.0,  # default fill
        "amount": 200.0,
    }
    return em


class _MockSide:
    def __init__(self, val):
        self.value = val


class TestP134Gating:
    """Verify the env-gate + side/quote/balance gates correctly skip
    when conversion shouldn't fire."""

    def test_env_disabled_skips_silently(self, monkeypatch):
        monkeypatch.delenv("HMATS_USDT_AUTO_CONVERT_ENABLED", raising=False)
        em = _make_em_with_mocks()
        ok, msg = em._ensure_quote_currency_available(
            "SOL/USDT", _MockSide("buy"), size=10.0, price=84.0
        )
        assert ok is True and msg == ""
        em.exchange.create_market_order.assert_not_called()

    def test_sell_order_skips(self, monkeypatch):
        monkeypatch.setenv("HMATS_USDT_AUTO_CONVERT_ENABLED", "true")
        em = _make_em_with_mocks()
        ok, msg = em._ensure_quote_currency_available(
            "SOL/USDT", _MockSide("sell"), size=10.0, price=84.0
        )
        assert ok is True and msg == ""
        em.exchange.create_market_order.assert_not_called()

    def test_non_usdt_pair_skips(self, monkeypatch):
        """BTC/USD doesn't need USDT conversion."""
        monkeypatch.setenv("HMATS_USDT_AUTO_CONVERT_ENABLED", "true")
        em = _make_em_with_mocks()
        ok, msg = em._ensure_quote_currency_available(
            "BTC/USD", _MockSide("buy"), size=0.01, price=76000
        )
        assert ok is True and msg == ""
        em.exchange.create_market_order.assert_not_called()

    def test_sufficient_usdt_skips(self, monkeypatch):
        """If we already have enough USDT, no conversion needed."""
        monkeypatch.setenv("HMATS_USDT_AUTO_CONVERT_ENABLED", "true")
        em = _make_em_with_mocks(usdt_free=2000.0)  # plenty of USDT
        # Need ~84 USDT for 1 SOL @ $84
        ok, msg = em._ensure_quote_currency_available(
            "SOL/USDT", _MockSide("buy"), size=1.0, price=84.0
        )
        assert ok is True
        em.exchange.create_market_order.assert_not_called()


class TestP134Conversion:
    """Verify the conversion path itself when triggered."""

    def test_conversion_fires_when_short_usdt(self, monkeypatch):
        monkeypatch.setenv("HMATS_USDT_AUTO_CONVERT_ENABLED", "true")
        monkeypatch.setenv("HMATS_USDT_AUTO_CONVERT_MAX_USD", "1000")
        em = _make_em_with_mocks(usd_free=3941.0, usdt_free=30.0)
        # Need ~84 USDT, have 30 -> shortfall ~54 + 5% buffer = ~57 USD
        ok, msg = em._ensure_quote_currency_available(
            "SOL/USDT", _MockSide("buy"), size=1.0, price=84.0
        )
        assert ok is True, f"Expected ok=True, got msg={msg}"
        em.exchange.create_market_order.assert_called_once()
        call = em.exchange.create_market_order.call_args
        assert call.kwargs["symbol"] == "USDT/USD"
        assert call.kwargs["side"] == "buy"
        # USDT amount should be ~57 USDT (the shortfall)
        assert 50 < call.kwargs["amount"] < 70

    def test_per_order_cap_enforced(self, monkeypatch):
        """Conversion >cap should fail-CLOSED, not silently truncate."""
        monkeypatch.setenv("HMATS_USDT_AUTO_CONVERT_ENABLED", "true")
        monkeypatch.setenv("HMATS_USDT_AUTO_CONVERT_MAX_USD", "200")
        em = _make_em_with_mocks(usdt_free=0.0)
        # Need ~840 USDT, cap is 200 -> exceeds
        ok, msg = em._ensure_quote_currency_available(
            "SOL/USDT", _MockSide("buy"), size=10.0, price=84.0
        )
        assert ok is False
        assert "USDT_CONVERT_EXCEEDS_CAP" in msg
        em.exchange.create_market_order.assert_not_called()

    def test_insufficient_usd_fails_closed(self, monkeypatch):
        monkeypatch.setenv("HMATS_USDT_AUTO_CONVERT_ENABLED", "true")
        monkeypatch.setenv("HMATS_USDT_AUTO_CONVERT_MAX_USD", "10000")
        em = _make_em_with_mocks(usd_free=10.0, usdt_free=0.0)
        # Need ~840 USDT, only have $10 USD -> fail
        ok, msg = em._ensure_quote_currency_available(
            "SOL/USDT", _MockSide("buy"), size=10.0, price=84.0
        )
        assert ok is False
        assert "INSUFFICIENT_USD_FOR_CONVERSION" in msg
        em.exchange.create_market_order.assert_not_called()

    def test_kraken_partial_fill_fails_closed(self, monkeypatch):
        monkeypatch.setenv("HMATS_USDT_AUTO_CONVERT_ENABLED", "true")
        monkeypatch.setenv("HMATS_USDT_AUTO_CONVERT_MAX_USD", "1000")
        em = _make_em_with_mocks(usdt_free=0.0)
        # Mock partial fill (50% of requested)
        em.exchange.create_market_order.return_value = {
            "id": "PARTIAL", "filled": 5.0, "amount": 100.0,
        }
        ok, msg = em._ensure_quote_currency_available(
            "SOL/USDT", _MockSide("buy"), size=1.0, price=84.0
        )
        assert ok is False
        assert "USDT_CONVERT_PARTIAL" in msg

    def test_kraken_exception_fails_closed(self, monkeypatch):
        monkeypatch.setenv("HMATS_USDT_AUTO_CONVERT_ENABLED", "true")
        em = _make_em_with_mocks(usdt_free=0.0)
        em.exchange.create_market_order.side_effect = RuntimeError("kraken down")
        ok, msg = em._ensure_quote_currency_available(
            "SOL/USDT", _MockSide("buy"), size=1.0, price=84.0
        )
        assert ok is False
        assert "USDT_CONVERT_FAILED" in msg


class TestP134ClassifierIntegration:
    """The PREFLIGHT_QUOTE_CONVERT_FAILED prefix must map to PERMANENT in
    P79 classifier so the order doesn't enter retry storm."""

    def test_prefix_classified_permanent(self):
        from execution.execution_manager import ExecutionManager
        cat, _ = ExecutionManager._classify_kraken_order_error(
            "PREFLIGHT_QUOTE_CONVERT_FAILED: insufficient USD"
        )
        assert cat == "PERMANENT", (
            f"P134 regression: PREFLIGHT_QUOTE_CONVERT_FAILED no longer in "
            f"P79 PERMANENT prefix list. Conversion failures will trigger "
            f"3-attempt retry storm."
        )


class TestP134SourceContract:

    def test_method_exists(self):
        from execution.execution_manager import ExecutionManager
        assert hasattr(ExecutionManager, "_ensure_quote_currency_available"), (
            "P134 regression: _ensure_quote_currency_available method removed."
        )

    def test_called_from_market_order(self):
        from execution.execution_manager import ExecutionManager
        src = inspect.getsource(ExecutionManager._execute_market_order)
        assert "_ensure_quote_currency_available" in src, (
            "P134 regression: market order no longer calls "
            "_ensure_quote_currency_available pre-clamp."
        )

    def test_called_from_limit_order(self):
        from execution.execution_manager import ExecutionManager
        src = inspect.getsource(ExecutionManager._execute_limit_order)
        assert "_ensure_quote_currency_available" in src, (
            "P134 regression: limit order no longer calls "
            "_ensure_quote_currency_available pre-clamp."
        )

    def test_default_off_documented(self):
        env_doc = open(".env.example", encoding="utf-8-sig").read()
        assert "HMATS_USDT_AUTO_CONVERT_ENABLED" in env_doc, (
            "P134 regression: env var docs removed from .env.example. "
            "Operators won't know the conversion is opt-in."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

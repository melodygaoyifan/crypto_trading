"""
test_post_only_fallback_p136.py — aggressive-limit fallback when pair in post_only mode (P136)
================================================================================

Production 2026-04-29 05:13:01 UTC — SOL/USDT market order rejected by
Kraken with `EService:Market in post_only mode`. The PAIR was in
post-only mode (Kraken does this during volatile/illiquid windows;
only LIMIT orders accepted). Pre-P136: order REJECTED, [BUGFIX H1]
aborted remaining 5 slices, position unchanged.

P136: detect this specific Kraken error in the market-order exception
handler + auto-convert to AGGRESSIVE LIMIT (limit at opposite-side
best price, NOT postOnly). Fills like a market order, allowed in
post-only-mode markets.

Tests:
  - Detection: error string match (case-insensitive, two formats)
  - Fallback: aggressive limit placed at correct side (BUY=ask, SELL=bid)
  - postOnly stripped from params (otherwise we'd hit the same error)
  - Successful fallback returns FILLED status with limit order_type
  - Non-post-only error (e.g. insufficient funds) does NOT trigger fallback
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest


def _make_em():
    """Minimal ExecutionManager with mocked exchange."""
    from execution.execution_manager import ExecutionManager
    em = ExecutionManager.__new__(ExecutionManager)
    em.exchange = MagicMock()
    em.logger = MagicMock()
    return em


class _MockSide:
    def __init__(self, val):
        self.value = val


class TestP136SourceContract:
    def test_market_order_handler_has_post_only_branch(self):
        from execution.execution_manager import ExecutionManager
        src = inspect.getsource(ExecutionManager._execute_market_order)
        assert "post_only mode" in src.lower(), (
            "P136 regression: post-only-mode detection removed from "
            "_execute_market_order exception handler. Kraken pair-state "
            "post_only rejections will fully fail again instead of "
            "auto-falling-back to aggressive limit."
        )

    def test_post_only_fallback_marker_present(self):
        from execution.execution_manager import ExecutionManager
        src = inspect.getsource(ExecutionManager._execute_market_order)
        assert "[POST_ONLY_FALLBACK]" in src, (
            "P136 marker comment removed; future operator won't see "
            "why aggressive-limit fallback exists."
        )

    def test_fallback_strips_postonly_from_params(self):
        """The aggressive limit params MUST drop postOnly/oflags or we'd
        hit the same Kraken error."""
        from execution.execution_manager import ExecutionManager
        src = inspect.getsource(ExecutionManager._execute_market_order)
        assert ".pop('postOnly'" in src and ".pop('oflags'" in src, (
            "P136: aggressive-limit fallback no longer strips postOnly + "
            "oflags. Order would re-trigger the same EService:Market in "
            "post_only mode error."
        )

    def test_fallback_uses_opposite_side_price(self):
        """BUY → ask, SELL → bid. Wrong side = no fill."""
        from execution.execution_manager import ExecutionManager
        src = inspect.getsource(ExecutionManager._execute_market_order)
        # Must reference both 'ask' (for buy) and 'bid' (for sell)
        assert ".get('ask')" in src and ".get('bid')" in src, (
            "P136: aggressive-limit fallback no longer uses opposite-side "
            "price. Wrong side picks would not fill immediately."
        )


class TestP136Behavior:
    """Mock-driven behavior tests."""

    def test_post_only_buy_falls_back_to_aggressive_ask(self):
        from execution.execution_manager import ExecutionManager, OrderSide, OrderStatus
        em = _make_em()

        # Market order raises with post_only error
        em.exchange.create_market_order.side_effect = Exception(
            'kraken {"error":["EService:Market in post_only mode"]}'
        )
        # Aggressive limit succeeds at ask
        em.exchange.fetch_ticker.return_value = {"ask": 84.10, "bid": 84.05, "last": 84.07}
        em.exchange.create_limit_order.return_value = {
            "id": "AGGRO_LIMIT_1", "average": 84.10, "filled": 1.0,
            "fee": {"cost": 0.22, "currency": "USDT"}
        }
        # Bypass other gates
        em._ensure_quote_currency_available = lambda *a, **kw: (True, "")
        em._clamp_size_to_balance = lambda *a, **kw: 1.0
        em.exchange.market.return_value = {"limits": {"amount": {"min": 0.02}}}

        # P135 added _normalize_kraken_pair logic; minimal stub
        result = em._execute_market_order("SOL/USDT", OrderSide.BUY, 1.0)
        assert result.success is True, f"Expected fallback success, got {result.error_message}"
        assert result.order_id == "AGGRO_LIMIT_1"
        # Aggressive limit must be at ASK for BUY
        call = em.exchange.create_limit_order.call_args
        assert call.kwargs["side"] == "buy"
        assert call.kwargs["price"] == 84.10  # ask
        # postOnly must be stripped
        assert "postOnly" not in call.kwargs.get("params", {})
        assert "oflags" not in call.kwargs.get("params", {})

    def test_post_only_sell_falls_back_to_aggressive_bid(self):
        from execution.execution_manager import ExecutionManager, OrderSide
        em = _make_em()
        em.exchange.create_market_order.side_effect = Exception(
            "EService:Market in post_only mode"
        )
        em.exchange.fetch_ticker.return_value = {"ask": 84.10, "bid": 84.05, "last": 84.07}
        em.exchange.create_limit_order.return_value = {
            "id": "AGGRO_LIMIT_SELL", "average": 84.05, "filled": 1.0,
            "fee": {"cost": 0.22, "currency": "USDT"}
        }
        em._ensure_quote_currency_available = lambda *a, **kw: (True, "")
        em._clamp_size_to_balance = lambda *a, **kw: 1.0
        em.exchange.market.return_value = {"limits": {"amount": {"min": 0.02}}}

        result = em._execute_market_order("SOL/USDT", OrderSide.SELL, 1.0)
        assert result.success is True
        call = em.exchange.create_limit_order.call_args
        assert call.kwargs["side"] == "sell"
        assert call.kwargs["price"] == 84.05  # bid (opposite side for SELL)

    def test_non_post_only_error_does_not_trigger_fallback(self):
        """Other errors (insufficient funds, etc.) must NOT trigger the
        aggressive-limit retry — they have different remediation."""
        from execution.execution_manager import ExecutionManager, OrderSide
        em = _make_em()
        em.exchange.create_market_order.side_effect = Exception(
            'kraken {"error":["EOrder:Insufficient funds"]}'
        )
        em._ensure_quote_currency_available = lambda *a, **kw: (True, "")
        em._clamp_size_to_balance = lambda *a, **kw: 1.0
        em.exchange.market.return_value = {"limits": {"amount": {"min": 0.02}}}

        result = em._execute_market_order("SOL/USDT", OrderSide.BUY, 1.0)
        assert result.success is False
        # create_limit_order must NOT have been called as fallback
        em.exchange.create_limit_order.assert_not_called()
        assert "Insufficient funds" in result.error_message

    def test_fallback_failure_still_returns_rejected(self):
        """If the aggressive limit ALSO fails, original error is preserved
        in REJECTED result (don't swallow)."""
        from execution.execution_manager import ExecutionManager, OrderSide
        em = _make_em()
        em.exchange.create_market_order.side_effect = Exception(
            "EService:Market in post_only mode"
        )
        em.exchange.fetch_ticker.return_value = {"ask": 84.10, "bid": 84.05}
        em.exchange.create_limit_order.side_effect = Exception("kraken down")
        em._ensure_quote_currency_available = lambda *a, **kw: (True, "")
        em._clamp_size_to_balance = lambda *a, **kw: 1.0
        em.exchange.market.return_value = {"limits": {"amount": {"min": 0.02}}}

        result = em._execute_market_order("SOL/USDT", OrderSide.BUY, 1.0)
        assert result.success is False
        # Original market error preserved
        assert "post_only" in result.error_message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

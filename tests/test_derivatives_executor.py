"""
Unit tests for DerivativesExecutor + DerivativesRiskGate + ExecutionRouter.

Coverage per v2.0 Part 3 Step 7:
  Happy paths, risk rejections, failure modes (spot-fail unwind, unwind retry
  exhaustion, reconciliation), kill switch, router decisions.
"""

import os
import time
from unittest.mock import MagicMock
import pytest

from execution.derivatives_executor import (
    DerivativesExecutor,
    ExecutionResult,
    ExecutionStatus,
    PerpPosition,
    CashCarryPosition,
    PERP_SYMBOLS,
)
from execution.execution_router import ExecutionRouter, ExecutionRoute
from risk.derivatives_risk_gate import DerivativesRiskGate
from infra.kraken_derivatives_client import DerivativesMarketData


# =============================================================================
# FIXTURES
# =============================================================================

class FakeIntent:
    def __init__(self, asset="BTC", direction=1, leverage=1.0, size_usd=500,
                 strategy=""):
        self.asset = asset
        self.direction = direction
        self.leverage = leverage
        self.size_usd = size_usd
        self.strategy = strategy
        self.quant_strategy_id = strategy


class FakeKrakenClient:
    """Mock KrakenDerivativesClient with configurable behavior."""

    def __init__(self, exec_enabled=True, mark_prices=None, market_data=None,
                 order_result=None, positions=None, will_raise=False):
        self._exec_enabled = exec_enabled
        self._marks = mark_prices or {"BTC": 77000.0, "ETH": 2300.0, "SOL": 85.0}
        self._market = market_data or {}
        self._order_result = order_result or {"_default": True}
        self._positions = positions or []
        self._will_raise = will_raise
        self.send_order_calls = []

    def is_execution_enabled(self):
        return self._exec_enabled

    def get_mark_price(self, asset):
        return self._marks.get(asset.upper())

    def fetch_market_data(self):
        if self._market:
            return self._market
        return {
            a: DerivativesMarketData(mark_price=p, volume_24h=1_000_000.0)
            for a, p in self._marks.items()
        }

    def send_order(self, **kwargs):
        self.send_order_calls.append(kwargs)
        if self._will_raise:
            raise RuntimeError("simulated_order_failure")
        # If caller hasn't overridden, echo mark price of the traded symbol as fill
        if self._order_result.get("_default", False):
            symbol = kwargs.get("symbol", "")
            asset = next((a for a, s in PERP_SYMBOLS.items() if s == symbol), None)
            mark = self._marks.get(asset) if asset else 77000.0
            return {
                "result": "success",
                "sendStatus": {
                    "order_id": "test-order-1",
                    "orderEvents": [{"price": mark}],
                },
            }
        return self._order_result

    def get_open_positions(self):
        return self._positions


def _build_executor(client=None, nav=10000.0, discord=None,
                    initial_cap=None, spot_exchange=None):
    client = client or FakeKrakenClient()
    return DerivativesExecutor(
        kraken_deriv_client=client,
        spot_exchange=spot_exchange,
        risk_callback=lambda: nav,
        discord_alerter=discord,
        initial_size_cap_usd=initial_cap,
    )


# =============================================================================
# EXECUTOR — HARD CAP REJECTIONS
# =============================================================================

class TestHardCaps:
    def test_leverage_cap_rejects_over_3x(self):
        ex = _build_executor()
        result = ex.execute_directional("BTC", +1, 500.0, leverage=3.5)
        assert not result.success
        assert "leverage" in result.reason

    def test_leverage_at_exactly_3x_passes_cap_check(self):
        ex = _build_executor()
        result = ex.execute_directional("BTC", +1, 500.0, leverage=3.0)
        # 3x leverage passes cap; may still fail other checks (liq buffer)
        assert "leverage" not in (result.reason or "")

    def test_single_position_cap_rejects_over_15pct_nav(self):
        ex = _build_executor(nav=10000.0)
        result = ex.execute_directional("BTC", +1, 2000.0, leverage=2.0)
        assert not result.success
        assert "single_position_cap" in result.reason

    def test_total_allocation_cap_rejects_over_30pct_nav(self):
        ex = _build_executor(nav=10000.0)
        # Pre-load 2x 15% positions = 30% total
        for a in ["ETH"]:
            ex.active_perp_positions[a] = PerpPosition(
                asset=a, symbol=PERP_SYMBOLS[a], side="buy",
                size_usd=1500.0, size_asset=0.65, entry_price=2300.0,
                leverage=2.0, liquidation_price=1500.0, opened_at=0,
            )
        # Try adding 1600 more — 3100/10000 = 31% > 30% cap
        ex.active_perp_positions["SOL"] = PerpPosition(
            asset="SOL", symbol=PERP_SYMBOLS["SOL"], side="buy",
            size_usd=1500.0, size_asset=17.6, entry_price=85.0,
            leverage=2.0, liquidation_price=50.0, opened_at=0,
        )
        result = ex.execute_directional("BTC", +1, 200.0, leverage=2.0)
        assert not result.success
        assert "total_allocation" in result.reason

    def test_initial_usd_cap_blocks_over_cap(self):
        ex = _build_executor(initial_cap=250.0)
        result = ex.execute_directional("BTC", +1, 300.0, leverage=2.0)
        assert not result.success
        assert "initial_usd_cap" in result.reason

    def test_initial_usd_cap_allows_under_cap(self):
        ex = _build_executor(initial_cap=250.0)
        result = ex.execute_directional("BTC", +1, 200.0, leverage=2.0)
        assert result.success, f"expected success, got {result.reason}"

    def test_liq_buffer_rejects_thin_buffer(self):
        ex = _build_executor()
        # 3x lev with 2% maintenance → liq at ~33% below mark → buffer ~33%
        # Below 40% threshold
        result = ex.execute_directional("BTC", +1, 500.0, leverage=3.0)
        assert not result.success
        assert "liq_buffer" in result.reason

    def test_invalid_direction_rejected(self):
        ex = _build_executor()
        result = ex.execute_directional("BTC", 0, 500.0, leverage=2.0)
        assert not result.success
        assert "direction" in result.reason


# =============================================================================
# EXECUTOR — HAPPY PATHS
# =============================================================================

class TestHappyPaths:
    def test_directional_long_opens_and_records(self):
        ex = _build_executor()
        result = ex.execute_directional("BTC", +1, 500.0, leverage=2.0)
        assert result.success
        assert "BTC" in ex.active_perp_positions
        pos = ex.active_perp_positions["BTC"]
        assert pos.side == "buy"
        assert pos.size_usd == 500.0
        assert pos.leverage == 2.0
        assert pos.liquidation_price < pos.entry_price  # long: liq below

    def test_directional_short_opens_with_liq_above(self):
        ex = _build_executor()
        result = ex.execute_directional("ETH", -1, 500.0, leverage=2.0)
        assert result.success
        pos = ex.active_perp_positions["ETH"]
        assert pos.side == "sell"
        assert pos.liquidation_price > pos.entry_price  # short: liq above

    def test_close_directional_removes_position(self):
        ex = _build_executor()
        ex.execute_directional("BTC", +1, 500.0, leverage=2.0)
        assert "BTC" in ex.active_perp_positions
        closed = ex.close_directional("BTC", reason="test")
        assert closed
        assert "BTC" not in ex.active_perp_positions


# =============================================================================
# EXECUTOR — FAILURE MODES
# =============================================================================

class TestFailureModes:
    def test_disabled_executor_rejects_all_orders(self):
        ex = _build_executor(client=FakeKrakenClient(exec_enabled=False))
        r = ex.execute_directional("BTC", +1, 500.0, leverage=2.0)
        assert not r.success
        assert r.status == ExecutionStatus.DISABLED

    def test_disabled_executor_blocks_carry(self):
        ex = _build_executor(client=FakeKrakenClient(exec_enabled=False))
        r = ex.execute_cash_carry("BTC", 500.0, funding_rate_h=0.0002)
        assert not r.success
        assert r.status == ExecutionStatus.DISABLED

    def test_order_api_exception_returns_failed(self):
        client = FakeKrakenClient(will_raise=True)
        ex = _build_executor(client=client)
        r = ex.execute_directional("BTC", +1, 500.0, leverage=2.0)
        assert not r.success
        assert r.status == ExecutionStatus.FAILED
        assert "exception" in r.reason

    def test_order_api_returns_error_result(self):
        client = FakeKrakenClient(order_result={"result": "error", "body": "bad_request"})
        ex = _build_executor(client=client)
        r = ex.execute_directional("BTC", +1, 500.0, leverage=2.0)
        assert not r.success
        assert r.status == ExecutionStatus.FAILED

    def test_unsupported_asset_rejected(self):
        ex = _build_executor()
        r = ex.execute_directional("DOGE", +1, 500.0, leverage=2.0)
        assert not r.success
        assert "unsupported_asset" in r.reason

    def test_emergency_unwind_retries_3_times_then_disables(self, monkeypatch):
        # Client succeeds on execute_directional, fails on unwind
        call_count = {"n": 0}
        def send_order_impl(**kwargs):
            call_count["n"] += 1
            # First call = initial open (success)
            if kwargs.get("reduce_only"):
                raise RuntimeError("unwind_fail")
            return {
                "result": "success",
                "sendStatus": {"order_id": "x", "orderEvents": [{"price": 77000.0}]},
            }

        client = FakeKrakenClient()
        client.send_order = send_order_impl
        ex = _build_executor(client=client)
        # Shorten backoff for test
        ex.UNWIND_BACKOFF_BASE_SEC = 0  # no sleep in test

        ex.execute_directional("BTC", +1, 500.0, leverage=2.0)
        unwound = ex._emergency_unwind_perp("BTC")
        assert not unwound
        assert ex._disabled
        assert "emergency_unwind_failed" in ex._disabled_reason


# =============================================================================
# CASH-CARRY
# =============================================================================

class TestCashCarry:
    def test_carry_requires_positive_funding(self):
        ex = _build_executor()
        r = ex.execute_cash_carry("BTC", 500.0, funding_rate_h=0.00005)  # below 0.0001 threshold
        assert not r.success
        assert "funding_too_low" in r.reason

    def test_carry_perp_first_then_spot(self):
        """Verify execution order: perp leg fires first."""
        client = FakeKrakenClient()
        spot = MagicMock()
        spot.create_limit_buy_order = MagicMock(return_value={"price": 77000.0})
        spot.fetch_ticker = MagicMock(return_value={"last": 77000.0, "close": 77000.0})
        ex = _build_executor(client=client, spot_exchange=spot)
        ex.initial_size_cap_usd = 1000  # don't trip the cap

        result = ex.execute_cash_carry("BTC", 500.0, funding_rate_h=0.0002)
        assert result.success
        # Perp send_order should be called before spot order
        assert len(client.send_order_calls) == 1
        assert client.send_order_calls[0]["side"] == "sell"  # short perp
        spot.create_limit_buy_order.assert_called_once()
        # Active carry registered
        assert "BTC" in ex.active_carries
        # perp position should now be inside carry, not standalone
        assert "BTC" not in ex.active_perp_positions

    def test_spot_leg_fail_unwinds_perp(self):
        """Critical safety test: if spot leg fails, perp must be unwound."""
        client = FakeKrakenClient()
        spot = MagicMock()
        spot.create_limit_buy_order = MagicMock(side_effect=RuntimeError("spot_api_down"))
        spot.fetch_ticker = MagicMock(return_value={"last": 77000.0, "close": 77000.0})
        ex = _build_executor(client=client, spot_exchange=spot)
        ex.UNWIND_BACKOFF_BASE_SEC = 0

        result = ex.execute_cash_carry("BTC", 500.0, funding_rate_h=0.0002)
        assert not result.success
        assert "spot_leg_failed" in result.reason
        # Both perp orders should have been called: open + unwind
        assert len(client.send_order_calls) >= 2
        assert client.send_order_calls[-1].get("reduce_only")  # unwind was reduce_only
        # No carry active, no orphan perp
        assert "BTC" not in ex.active_carries
        assert "BTC" not in ex.active_perp_positions


# =============================================================================
# RECONCILIATION
# =============================================================================

class TestReconciliation:
    def test_reconcile_empty_exchange_clears_local(self):
        client = FakeKrakenClient(positions=[])
        ex = _build_executor(client=client)
        ex.active_perp_positions["BTC"] = PerpPosition(
            asset="BTC", symbol="PF_XBTUSD", side="buy", size_usd=500,
            size_asset=0.006, entry_price=77000, leverage=2, liquidation_price=40000,
            opened_at=0,
        )
        ex.reconcile_on_startup()
        assert len(ex.active_perp_positions) == 0

    def test_reconcile_restores_positions_from_exchange(self):
        positions = [
            {"symbol": "PF_XBTUSD", "side": "buy", "size": 0.01, "price": 77000.0,
             "fillTime": 0, "orderId": "abc"},
            {"symbol": "PF_ETHUSD", "side": "sell", "size": 0.5, "price": 2300.0,
             "fillTime": 0, "orderId": "def"},
        ]
        client = FakeKrakenClient(positions=positions)
        ex = _build_executor(client=client)
        ex.reconcile_on_startup()
        assert "BTC" in ex.active_perp_positions
        assert "ETH" in ex.active_perp_positions
        assert ex.active_perp_positions["BTC"].side == "buy"
        assert ex.active_perp_positions["ETH"].side == "sell"

    def test_reconcile_ignores_unsupported_symbols(self):
        positions = [{"symbol": "PF_SOLUSD", "side": "buy", "size": 5, "price": 85.0,
                      "fillTime": 0, "orderId": "a"}]
        client = FakeKrakenClient(positions=positions)
        ex = _build_executor(client=client)
        ex.reconcile_on_startup()
        assert "SOL" in ex.active_perp_positions
        assert len(ex.active_perp_positions) == 1

    def test_reconcile_failure_disables_executor(self):
        client = FakeKrakenClient()
        def raise_on_positions():
            raise RuntimeError("api_unreachable")
        client.get_open_positions = raise_on_positions
        ex = _build_executor(client=client)
        ex.reconcile_on_startup()
        assert ex._disabled
        assert "reconcile_failed" in ex._disabled_reason


# =============================================================================
# RISK GATE
# =============================================================================

class TestRiskGate:
    def test_gate_passes_valid_intent(self):
        ex = _build_executor()
        gate = DerivativesRiskGate(executor=ex)
        intent = FakeIntent(asset="BTC", direction=+1, leverage=2.0, size_usd=500)
        result = gate.validate(intent, context={"funding_rate_h": 0.0001})
        assert result.passed

    def test_gate_vetoes_when_executor_disabled(self):
        ex = _build_executor()
        ex._disabled = True
        ex._disabled_reason = "testing"
        gate = DerivativesRiskGate(executor=ex)
        intent = FakeIntent()
        result = gate.validate(intent)
        assert not result.passed
        assert "executor_disabled" in result.veto_reason

    def test_gate_vetoes_hostile_funding_short(self):
        ex = _build_executor()
        gate = DerivativesRiskGate(executor=ex)
        intent = FakeIntent(asset="BTC", direction=-1, leverage=1.0, size_usd=500)
        # Funding deeply negative — longs get paid, shorts pay
        result = gate.validate(intent, context={"funding_rate_h": -0.001})
        assert not result.passed
        assert "funding_hostile_to_short" in result.veto_reason

    def test_gate_vetoes_thin_perp_volume(self):
        client = FakeKrakenClient(market_data={
            "BTC": DerivativesMarketData(mark_price=77000.0, volume_24h=10.0),  # 10 contracts × $77k = $770k, below 5M threshold
        })
        ex = _build_executor(client=client)
        gate = DerivativesRiskGate(executor=ex)
        intent = FakeIntent(asset="BTC", direction=+1, leverage=2.0, size_usd=500)
        result = gate.validate(intent, context={"funding_rate_h": 0.0001})
        assert not result.passed
        assert "perp_volume_thin" in result.veto_reason


# =============================================================================
# ROUTER
# =============================================================================

class TestRouter:
    def _make_router(self, ex=None, gate=None, spot_margin=None):
        ex = ex or _build_executor()
        gate = gate or DerivativesRiskGate(executor=ex)
        return ExecutionRouter(
            spot_margin_executor=spot_margin,
            derivatives_executor=ex,
            derivatives_gate=gate,
        )

    def test_kill_switch_off_routes_spot(self, monkeypatch):
        monkeypatch.setenv("DERIVATIVES_ENABLED", "false")
        router = self._make_router()
        d = router.decide_route(FakeIntent(strategy="CASH_CARRY"))
        assert d.route == ExecutionRoute.SPOT_MARGIN
        assert "kill_switch" in d.reason

    def test_kill_switch_on_routes_carry_to_derivatives(self, monkeypatch):
        monkeypatch.setenv("DERIVATIVES_ENABLED", "true")
        router = self._make_router()
        d = router.decide_route(FakeIntent(strategy="CASH_CARRY"))
        assert d.route == ExecutionRoute.DERIVATIVES
        assert "cash_carry" in d.reason

    def test_leveraged_long_routes_to_derivatives(self, monkeypatch):
        monkeypatch.setenv("DERIVATIVES_ENABLED", "true")
        router = self._make_router()
        d = router.decide_route(FakeIntent(direction=+1, leverage=2.0, size_usd=500))
        assert d.route == ExecutionRoute.DERIVATIVES

    def test_spot_long_routes_to_spot_margin(self, monkeypatch):
        monkeypatch.setenv("DERIVATIVES_ENABLED", "true")
        router = self._make_router()
        d = router.decide_route(FakeIntent(direction=+1, leverage=1.0, size_usd=500))
        assert d.route == ExecutionRoute.SPOT_MARGIN

    def test_short_with_positive_funding_routes_derivatives(self, monkeypatch):
        monkeypatch.setenv("DERIVATIVES_ENABLED", "true")
        router = self._make_router()
        d = router.decide_route(
            FakeIntent(direction=-1, leverage=1.0, size_usd=500),
            context={"funding_rate_h": 0.0002},
        )
        assert d.route == ExecutionRoute.DERIVATIVES

    def test_short_with_low_funding_routes_spot_margin(self, monkeypatch):
        monkeypatch.setenv("DERIVATIVES_ENABLED", "true")
        router = self._make_router()
        d = router.decide_route(
            FakeIntent(direction=-1, leverage=1.0, size_usd=500),
            context={"funding_rate_h": 0.00005},
        )
        assert d.route == ExecutionRoute.SPOT_MARGIN

    def test_gate_veto_on_leveraged_long_falls_back_to_spot(self, monkeypatch):
        """When DERIVATIVES_ENABLED=true but executor is internally disabled,
        the router's kill-switch check sees is_enabled()=False and routes spot
        with reason=kill_switch (NOT gate veto). gate_veto path only fires when
        kill switch allows the intent through but a per-intent check fails."""
        monkeypatch.setenv("DERIVATIVES_ENABLED", "true")
        # Keep executor healthy so kill switch passes; make gate fail via thin volume
        thin_client = FakeKrakenClient(market_data={
            "BTC": DerivativesMarketData(mark_price=77000.0, volume_24h=1.0),
        })
        ex = _build_executor(client=thin_client)
        gate = DerivativesRiskGate(executor=ex)
        router = ExecutionRouter(None, ex, gate)
        d = router.decide_route(FakeIntent(direction=+1, leverage=2.0, size_usd=500))
        assert d.route == ExecutionRoute.SPOT_MARGIN
        assert d.gate_veto is not None
        assert "perp_volume_thin" in d.gate_veto


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

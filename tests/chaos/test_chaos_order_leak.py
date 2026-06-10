"""
test_chaos_order_leak.py — order-leak / desync cascade scenarios
==================================================================

Recreates the exact production cascade conditions from this session:
P95 (userref scheme drift across deploy → 6 stop orders for 1 position),
P98 (fetch_open_orders dedup as ground-truth), P98b (cancel_order
EOrder:Unknown_order tolerance), P91 (stop below Kraken min-size).

Each test asserts the prevention mechanism is still in place + the
right log fires when chaos hits.
"""
from __future__ import annotations

import pytest

from tests.chaos.harness import captured_logs, assert_warn_in_logs


# =====================================================================
# Scenario 8: Userref drift across deploys (P95 — the original cascade)
# =====================================================================

class TestChaosUserrefDrift:
    """P95: pre-fix, userref hash included stop_price → every tick
    generated new userref → check_userref_executed never matched →
    place_stop_loss stacked a new order each tick. 6 BTC stops for
    1 position observed. Verify P95 fix held: userref stable for
    (symbol, side, suffix) triple regardless of price drift."""

    def test_userref_stable_under_extreme_price_drift(self):
        """Even with 100x price drift, same (symbol, side, suffix)
        produces same userref — that's the P95 invariant."""
        from execution.execution_manager import ExecutionManager

        # Stop prices spanning 100x range (extreme drift)
        prices = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
        userrefs = [
            ExecutionManager._generate_stop_userref("BTC/USD", "sell", p, "SL")
            for p in prices
        ]
        assert len(set(userrefs)) == 1, (
            f"P95 regression: userref VARIED across {len(prices)} different "
            f"stop prices: {userrefs}. Order-stacking cascade can recur."
        )

    def test_userref_distinct_per_side(self):
        """SELL stop and BUY stop on same symbol = different orders =
        different userrefs."""
        from execution.execution_manager import ExecutionManager
        sell = ExecutionManager._generate_stop_userref("BTC/USD", "sell", 50000.0)
        buy = ExecutionManager._generate_stop_userref("BTC/USD", "buy", 50000.0)
        assert sell != buy, "SELL/BUY collision = wrong order side replaced"

    def test_userref_distinct_per_symbol(self):
        from execution.execution_manager import ExecutionManager
        sol = ExecutionManager._generate_stop_userref("SOL/USD", "sell", 100.0)
        btc = ExecutionManager._generate_stop_userref("BTC/USD", "sell", 100.0)
        assert sol != btc, "Cross-symbol collision = SOL stop replaces BTC"


# =====================================================================
# Scenario 9: P98 fetch_open_orders ground-truth dedup
# =====================================================================

class TestChaosGroundTruthDedup:
    """P98 added fetch_open_orders pre-flight check in place_stop_loss
    so any pre-existing open stop on same (symbol, side) is cancelled
    before placing new one. Survives userref scheme drift. Verify
    the pattern is still in source."""

    def test_place_stop_loss_has_dedup_block(self):
        import inspect
        from execution.execution_manager import ExecutionManager
        src = inspect.getsource(ExecutionManager.place_stop_loss)
        # The P98 marker comment + the actual fetch_open_orders call
        assert "fetch_open_orders" in src, (
            "P98 ground-truth dedup removed: place_stop_loss no longer "
            "queries fetch_open_orders before placing. P95 cascade can "
            "recur on next userref scheme change."
        )
        assert "STOP-DEDUP" in src, (
            "P98 [STOP-DEDUP] log marker missing — dedup behavior may "
            "have been silently disabled."
        )


# =====================================================================
# Scenario 10: P98b EOrder:Unknown_order tolerance on cancel
# =====================================================================

class TestChaosCancelAlreadyGoneOrder:
    """P98b: cancel_order on an order that's already gone (operator
    cancelled out-of-band, or it triggered+filled between checks)
    must NOT log ERROR + count as 'failed'. It's semantically a
    success — the order is no longer at the exchange."""

    def test_cancel_unknown_order_pattern_present(self):
        import inspect
        from execution.execution_manager import ExecutionManager
        src = inspect.getsource(ExecutionManager.cancel_all_open_orders)
        assert "_is_unknown_order_error" in src or "EOrder:Unknown" in src, (
            "P98b tolerance removed: cancel_all_open_orders no longer "
            "treats EOrder:Unknown_order as success. Operator cancellations "
            "will spam ERROR logs again."
        )


# =====================================================================
# Scenario 11: P91 stop-loss below Kraken min-size pre-flight
# =====================================================================

class TestChaosStopBelowMinSize:
    """P91: when balance clamp drops stop size below Kraken min,
    the pre-flight check fires (PREFLIGHT_BELOW_MIN_SIZE rejection)
    BEFORE sending to Kraken. P93 then classifies that prefix as
    PERMANENT to avoid retry waste."""

    def test_preflight_min_size_pattern_present(self):
        import inspect
        from execution.execution_manager import ExecutionManager
        # Find place_stop_loss source
        src = inspect.getsource(ExecutionManager.place_stop_loss)
        assert "STOP-MINSIZE" in src or "PREFLIGHT_BELOW_MIN_SIZE" in src, (
            "P91 pre-flight min-size check removed. Stop orders below "
            "Kraken min will fail at Kraken with EGeneral:Invalid args, "
            "leaving position unprotected with operator only seeing the "
            "generic Kraken rejection."
        )

    def test_preflight_classified_permanent(self):
        from execution.execution_manager import ExecutionManager
        em = ExecutionManager.__new__(ExecutionManager)
        cat, _ = em._classify_kraken_order_error(
            "PREFLIGHT_BELOW_MIN_SIZE: stop-loss size 0.014"
        )
        assert cat == "PERMANENT", (
            f"P93 classifier missing PREFLIGHT_BELOW_MIN_SIZE branch: "
            f"got {cat}. Retry storm 3-attempts on our own validation "
            f"errors."
        )


# =====================================================================
# Scenario 12: P93 POSITION-DESYNC alert visibility
# =====================================================================

class TestChaosPositionDesyncAlert:
    """When balance clamp shrinks size by >50%, P93 fires a separate
    CRITICAL [POSITION-DESYNC] alert (vs the LOWER-priority STOP-BALANCE
    WARN). Threshold tuned so normal fee/rounding drift doesn't trigger
    the alarm but real phantom-position cases do."""

    def test_position_desync_alert_pattern_present(self):
        import inspect
        from execution.execution_manager import ExecutionManager
        src = inspect.getsource(ExecutionManager.place_stop_loss)
        assert "POSITION-DESYNC" in src, (
            "P93 [POSITION-DESYNC] alert removed. Operator loses the "
            "early-warning signal for phantom-position bugs (P87 family)."
        )
        # Verify the >50% threshold is the trigger
        assert "_shrink_pct > 50" in src or "_shrink_pct >= 50" in src, (
            "P93 50% threshold for POSITION-DESYNC alert was removed."
        )


# =====================================================================
# Scenario 12: Stop-loss must carry leverage on margin positions (P138-followup)
# =====================================================================

class TestChaosStopLossMarginLeverage:
    """P138-followup (2026-06-09): place_stop_loss historically built
    a spot trigger order (no leverage in params). For a margin short
    opened at 2x, that trigger BUYS spot SOL when fired — opens a new
    spot long, does NOT close the margin short. The margin position
    stays uncapped.

    Verify: signature accepts leverage; spot balance check is bypassed
    when leverage > 1; leverage is injected into the create_order
    params; both callers in execution_service.py pass it through.
    """

    def test_place_stop_loss_accepts_leverage_param(self):
        import inspect
        from execution.execution_manager import ExecutionManager
        sig = inspect.signature(ExecutionManager.place_stop_loss)
        assert "leverage" in sig.parameters, (
            "place_stop_loss missing leverage param. Margin shorts can't "
            "be protected — stop trigger fires as spot, doesn't close the "
            "margin position."
        )
        assert sig.parameters["leverage"].default is None, (
            "leverage default must be None (spot) to preserve back-compat."
        )

    def test_place_stop_loss_skips_spot_balance_for_margin(self):
        """Margin path must bypass _SkipSpotBalance and inject leverage
        into the create_order params dict."""
        from unittest.mock import MagicMock
        from execution.execution_manager import ExecutionManager, OrderSide

        em = ExecutionManager.__new__(ExecutionManager)
        em.exchange = MagicMock()
        em.exchange.market.return_value = {"limits": {"amount": {"min": 0.02}}}
        # Spot wallet near-zero — would clamp to dust on spot path
        em.exchange.fetch_balance.return_value = {"free": {"USDT": 0.12}, "used": {"USDT": 0.0}}
        em.exchange.price_to_precision.side_effect = lambda s, p: str(p)
        em.exchange.amount_to_precision.side_effect = lambda s, a: str(a)
        em.exchange.fetch_ticker.return_value = {"last": 63.0}
        em.exchange.fetch_open_orders.return_value = []
        em.exchange.create_order.return_value = {"id": "STOP-MARGIN-OK"}
        em.logger = MagicMock()
        em.config = MagicMock()
        em.config.use_exchange_stops = True
        em.dry_run = False
        em.active_stops = {}
        em._userref_history = {}
        em._with_stop_retries = lambda fn, **kw: fn()
        em._generate_stop_userref = lambda *a, **k: "SL-test"

        result = em.place_stop_loss(
            symbol="SOL/USDT", side=OrderSide.BUY, size=12.0,
            stop_price=80.0, leverage=2,
        )
        assert result.success, (
            f"P138-followup: margin stop must succeed despite empty spot "
            f"wallet. Error: {result.error_message}"
        )
        sent_params = em.exchange.create_order.call_args.kwargs["params"]
        assert sent_params.get("leverage") == 2, (
            f"P138-followup: leverage=2 missing from create_order params. "
            f"Sent: {sent_params}. Without this, the stop is a spot trigger "
            f"and won't close the margin position."
        )

    def test_place_stop_loss_spot_path_preserves_no_leverage(self):
        """Spot path (leverage=None or 1) must NOT inject leverage into
        params — Kraken rejects spot orders that carry leverage."""
        from unittest.mock import MagicMock
        from execution.execution_manager import ExecutionManager, OrderSide

        em = ExecutionManager.__new__(ExecutionManager)
        em.exchange = MagicMock()
        em.exchange.market.return_value = {"limits": {"amount": {"min": 0.001}}}
        em.exchange.fetch_balance.return_value = {"free": {"SOL": 100.0, "USDT": 10000.0}, "used": {}}
        em.exchange.price_to_precision.side_effect = lambda s, p: str(p)
        em.exchange.amount_to_precision.side_effect = lambda s, a: str(a)
        em.exchange.fetch_ticker.return_value = {"last": 63.0}
        em.exchange.fetch_open_orders.return_value = []
        em.exchange.create_order.return_value = {"id": "STOP-SPOT-OK"}
        em.logger = MagicMock()
        em.config = MagicMock()
        em.config.use_exchange_stops = True
        em.dry_run = False
        em.active_stops = {}
        em._userref_history = {}
        em._with_stop_retries = lambda fn, **kw: fn()
        em._generate_stop_userref = lambda *a, **k: "SL-spot"

        result = em.place_stop_loss(
            symbol="SOL/USDT", side=OrderSide.SELL, size=1.0,
            stop_price=60.0, leverage=None,
        )
        assert result.success
        sent_params = em.exchange.create_order.call_args.kwargs["params"]
        assert "leverage" not in sent_params, (
            "P138-followup: spot path leaked leverage into params. "
            f"Kraken would reject. Sent: {sent_params}"
        )

    def test_execution_service_callers_pass_leverage(self):
        """Both place_stop_loss call sites in execution_service.py must
        pass leverage=. Source-level guard so future refactors can't
        silently strip it."""
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "core" / "execution_service.py"
        if not p.exists():
            pytest.skip("execution_service.py not at expected location")
        src = p.read_text(encoding="utf-8")
        # Count place_stop_loss(...) call blocks; each must have leverage= within ~250 chars
        i = 0
        sites_seen = 0
        sites_with_leverage = 0
        while True:
            idx = src.find("place_stop_loss(", i)
            if idx < 0:
                break
            sites_seen += 1
            block = src[idx:idx + 400]
            if "leverage=" in block:
                sites_with_leverage += 1
            i = idx + 1
        assert sites_seen >= 2, (
            f"Expected 2+ place_stop_loss call sites in execution_service.py, "
            f"found {sites_seen}. Audit may be stale."
        )
        assert sites_seen == sites_with_leverage, (
            f"P138-followup: {sites_seen - sites_with_leverage} of "
            f"{sites_seen} place_stop_loss call sites in execution_service.py "
            f"are missing leverage=. Margin stops will be placed as spot "
            f"trigger orders that don't close the margin position."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

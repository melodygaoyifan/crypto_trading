"""
Tests for P0-03: Kraken-only execution hard isolation.

Covers:
  1. assert_not_dex_execution blocks DEX venues under SINGLE_EXCHANGE_MODE
  2. assert_not_dex_execution allows DEX when single_exchange_mode=False
  3. ExecutionManager.execute_order rejects DEX venue with CRITICAL log
  4. ExecutionManager.execute_order allows Kraken venue
  5. Kraken is never blocked by the DEX guard
  6. All known DEX venues are blocked
"""
import logging

import pytest

from core.exchange_guard import (
    assert_not_dex_execution,
    DEXExecutionBlockedError,
    DEX_VENUES,
    validate_kraken_only,
    NonKrakenExchangeError,
)
from execution.execution_manager import (
    ExecutionConfig,
    ExecutionManager,
    OrderResult,
    OrderSide,
    OrderStatus,
)


# ---------------------------------------------------------------------------
# Tests: assert_not_dex_execution
# ---------------------------------------------------------------------------

class TestAssertNotDexExecution:

    def test_blocks_jupiter(self):
        with pytest.raises(DEXExecutionBlockedError, match="DEX_EXECUTION_BLOCKED"):
            assert_not_dex_execution(
                "jupiter", action="swap", asset="SOL",
                single_exchange_mode=True,
            )

    def test_blocks_raydium(self):
        with pytest.raises(DEXExecutionBlockedError):
            assert_not_dex_execution(
                "raydium", action="place_order", asset="SOL",
                single_exchange_mode=True,
            )

    def test_blocks_orca(self):
        with pytest.raises(DEXExecutionBlockedError):
            assert_not_dex_execution(
                "orca", action="submit_tx", asset="SOL",
                single_exchange_mode=True,
            )

    def test_blocks_generic_dex(self):
        with pytest.raises(DEXExecutionBlockedError):
            assert_not_dex_execution(
                "dex", action="route_trade", asset="SOL",
                single_exchange_mode=True,
            )

    def test_all_dex_venues_blocked(self):
        """Every venue in DEX_VENUES must be blocked."""
        for venue in DEX_VENUES:
            with pytest.raises(DEXExecutionBlockedError):
                assert_not_dex_execution(
                    venue, action="test", asset="SOL",
                    single_exchange_mode=True,
                )

    def test_kraken_never_blocked(self):
        """Kraken must never be treated as DEX."""
        # Should not raise
        assert_not_dex_execution(
            "kraken", action="swap", asset="SOL",
            single_exchange_mode=True,
        )

    def test_allows_dex_when_single_exchange_off(self):
        """When single_exchange_mode=False, DEX execution is allowed."""
        # Should not raise
        assert_not_dex_execution(
            "jupiter", action="swap", asset="SOL",
            single_exchange_mode=False,
        )

    def test_critical_log_on_block(self, caplog):
        """Blocking must emit a CRITICAL log."""
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(DEXExecutionBlockedError):
                assert_not_dex_execution(
                    "jupiter", action="swap", asset="SOL",
                    single_exchange_mode=True,
                )
        critical = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert len(critical) >= 1
        assert "DEX_EXECUTION_BLOCKED" in critical[0].message

    def test_case_insensitive(self):
        """Venue matching should be case-insensitive."""
        with pytest.raises(DEXExecutionBlockedError):
            assert_not_dex_execution(
                "Jupiter", action="swap", asset="SOL",
                single_exchange_mode=True,
            )


# ---------------------------------------------------------------------------
# Tests: ExecutionManager DEX venue rejection
# ---------------------------------------------------------------------------

class TestExecutionManagerDexBlock:

    def _make_manager(self) -> ExecutionManager:
        return ExecutionManager(exchange=None, dry_run=True)

    def test_dex_venue_rejected(self, caplog):
        """execute_order with venue='jupiter' must return blocked result with CRITICAL."""
        mgr = self._make_manager()

        with caplog.at_level(logging.DEBUG):
            result = mgr.execute_order(
                symbol="SOL/USD",
                side=OrderSide.BUY,
                size=10.0,
                price=150.0,
                venue="jupiter",
            )

        assert result.success is False
        assert "DEX_EXECUTION_BLOCKED" in result.error_message
        assert result.status == OrderStatus.REJECTED

        critical = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert len(critical) >= 1

    def test_kraken_venue_allowed(self):
        """execute_order with venue='kraken' should not be blocked by venue gate."""
        mgr = self._make_manager()
        result = mgr.execute_order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.01,
            price=45000.0,
            venue="kraken",
        )
        # In dry_run mode, should succeed (simulated fill)
        assert result.success is True

    def test_non_dex_non_kraken_rejected(self, caplog):
        """execute_order with venue='binance' still blocked as SINGLE_EXCHANGE_VIOLATION."""
        mgr = self._make_manager()

        with caplog.at_level(logging.DEBUG):
            result = mgr.execute_order(
                symbol="BTC/USD",
                side=OrderSide.BUY,
                size=0.01,
                price=45000.0,
                venue="binance",
            )

        assert result.success is False
        assert "SINGLE_EXCHANGE_VIOLATION" in result.error_message

    def test_all_dex_venues_rejected(self):
        """Every known DEX venue must be rejected by execute_order."""
        mgr = self._make_manager()
        from execution.execution_manager import _DEX_VENUES

        for venue in _DEX_VENUES:
            result = mgr.execute_order(
                symbol="SOL/USD",
                side=OrderSide.BUY,
                size=1.0,
                price=150.0,
                venue=venue,
            )
            assert result.success is False, f"venue={venue} was not blocked"
            assert "DEX_EXECUTION_BLOCKED" in result.error_message, (
                f"venue={venue} missing DEX_EXECUTION_BLOCKED"
            )

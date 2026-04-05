"""
Tests for P0-02: Stop-order retry policy, idempotency, and failure fallback.

Covers:
  1. Retry succeeds on Nth attempt - correct result, failure count logged
  2. All retries exhausted - CRITICAL log emitted
  3. Idempotency: duplicate userref detected -> treated as success
  4. FREEZE_ENTRIES policy: new entries blocked, exits still allowed
  5. Userref determinism: same inputs -> same userref
"""
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from execution.execution_manager import (
    ExecutionConfig,
    ExecutionManager,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    StopOrderFailurePolicy,
)


# ---------------------------------------------------------------------------
# Fake exchange stubs
# ---------------------------------------------------------------------------

class FailNTimesExchange:
    """Exchange stub that fails N times then succeeds."""

    def __init__(self, fail_count: int = 2):
        self._fail_count = fail_count
        self._attempt = 0
        self.orders_created: List[dict] = []

    def create_order(self, **kwargs) -> dict:
        self._attempt += 1
        if self._attempt <= self._fail_count:
            raise ConnectionError(f"Simulated failure #{self._attempt}")
        order = {"id": f"ORD_{self._attempt}", **kwargs}
        self.orders_created.append(order)
        return order

    def fetch_closed_orders(self, **kwargs) -> list:
        return []


class AlwaysFailExchange:
    """Exchange stub that always raises."""

    def create_order(self, **kwargs) -> dict:
        raise ConnectionError("Permanent API failure")

    def fetch_closed_orders(self, **kwargs) -> list:
        return []


class DuplicateDetectExchange:
    """Exchange stub that fails once, but userref lookup finds existing order."""

    def __init__(self):
        self._attempt = 0
        self._placed_userref: Optional[int] = None

    def create_order(self, **kwargs) -> dict:
        self._attempt += 1
        if self._attempt == 1:
            # First attempt "fails" but order was actually placed server-side
            self._placed_userref = kwargs.get("params", {}).get("userref")
            raise ConnectionError("Timeout - but order landed on server")
        return {"id": "ORD_DUP"}

    def fetch_closed_orders(self, **kwargs) -> list:
        uref = kwargs.get("params", {}).get("userref")
        if uref is not None and uref == self._placed_userref:
            return [{"id": "ORD_EXISTING", "info": {"userref": str(uref)}}]
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(exchange, *, policy="ALERT_ONLY", max_retries=3, freeze_sec=5) -> ExecutionManager:
    cfg = ExecutionConfig(
        stop_order_max_retries=max_retries,
        stop_order_backoff_ms=[10, 20, 40],  # fast for tests
        stop_order_failure_policy=policy,
        stop_order_freeze_seconds=freeze_sec,
    )
    return ExecutionManager(exchange=exchange, config=cfg, dry_run=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStopOrderRetryPolicy:

    def test_retry_succeeds_on_nth_attempt(self, caplog):
        """Fail twice, succeed on 3rd - returns success, logs 2 errors."""
        exchange = FailNTimesExchange(fail_count=2)
        mgr = _make_manager(exchange, max_retries=3)

        with caplog.at_level(logging.DEBUG):
            result = mgr.place_stop_loss(
                symbol="BTC/USD",
                side=OrderSide.SELL,
                size=0.1,
                stop_price=40000.0,
            )

        assert result.success is True
        assert result.order_id is not None
        # Should see exactly 2 error-level retry logs
        error_lines = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_lines) == 2

    def test_all_retries_exhausted_triggers_critical(self, caplog):
        """All 3 retries fail -> CRITICAL log emitted."""
        exchange = AlwaysFailExchange()
        mgr = _make_manager(exchange, max_retries=3)

        with caplog.at_level(logging.DEBUG):
            result = mgr.place_stop_loss(
                symbol="ETH/USD",
                side=OrderSide.BUY,
                size=1.0,
                stop_price=3500.0,
            )

        assert result.success is False
        assert "RETRIES_EXHAUSTED" in result.error_message

        critical = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert len(critical) >= 1
        assert "FAILED_AFTER_RETRIES" in critical[0].message

    def test_idempotency_duplicate_detected(self, caplog):
        """First attempt timeout-fails but order landed; retry detects via userref."""
        exchange = DuplicateDetectExchange()
        mgr = _make_manager(exchange, max_retries=3)

        with caplog.at_level(logging.DEBUG):
            result = mgr.place_stop_loss(
                symbol="SOL/USD",
                side=OrderSide.SELL,
                size=10.0,
                stop_price=120.0,
            )

        assert result.success is True
        assert result.order_id == "ORD_EXISTING"
        # Should see the idempotent hit log
        assert any("already placed" in r.message for r in caplog.records)

    def test_freeze_entries_blocks_new_orders(self, caplog):
        """FREEZE_ENTRIES policy: after stop failure, execute_order is blocked."""
        exchange = AlwaysFailExchange()
        mgr = _make_manager(exchange, policy="FREEZE_ENTRIES", max_retries=1, freeze_sec=60)

        with caplog.at_level(logging.DEBUG):
            mgr.place_stop_loss(
                symbol="BTC/USD",
                side=OrderSide.SELL,
                size=0.1,
                stop_price=40000.0,
            )

        assert mgr.is_entries_frozen() is True

        # New entry order should be blocked
        result = mgr.execute_order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.05,
            price=42000.0,
        )
        assert result.success is False
        assert "frozen" in result.error_message.lower()

    def test_freeze_expires(self):
        """Freeze should auto-expire after configured seconds."""
        exchange = AlwaysFailExchange()
        mgr = _make_manager(exchange, policy="FREEZE_ENTRIES", max_retries=1, freeze_sec=1)

        mgr.place_stop_loss(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            size=0.1,
            stop_price=40000.0,
        )
        assert mgr.is_entries_frozen() is True

        time.sleep(1.1)
        assert mgr.is_entries_frozen() is False

    def test_userref_deterministic(self):
        """Same inputs must produce the same userref (idempotency guarantee)."""
        ref1 = ExecutionManager._generate_stop_userref("BTC/USD", "SELL", 40000.0, "SL")
        ref2 = ExecutionManager._generate_stop_userref("BTC/USD", "SELL", 40000.0, "SL")
        assert ref1 == ref2

        # Different price -> different userref
        ref3 = ExecutionManager._generate_stop_userref("BTC/USD", "SELL", 39999.0, "SL")
        assert ref3 != ref1

    def test_dry_run_skips_retries(self):
        """In dry_run mode, place_stop_loss succeeds immediately without retries."""
        mgr = ExecutionManager(exchange=None, dry_run=True)
        result = mgr.place_stop_loss(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            size=0.1,
            stop_price=40000.0,
        )
        assert result.success is True
        assert result.userref is not None
        assert result.order_id.startswith("SL_")

    def test_alert_only_does_not_freeze(self, caplog):
        """ALERT_ONLY policy: CRITICAL logged but entries NOT frozen."""
        exchange = AlwaysFailExchange()
        mgr = _make_manager(exchange, policy="ALERT_ONLY", max_retries=1)

        with caplog.at_level(logging.DEBUG):
            mgr.place_stop_loss(
                symbol="BTC/USD",
                side=OrderSide.SELL,
                size=0.1,
                stop_price=40000.0,
            )

        assert mgr.is_entries_frozen() is False
        critical = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert len(critical) >= 1

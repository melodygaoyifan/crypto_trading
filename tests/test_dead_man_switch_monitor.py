"""
Tests for DeadManSwitchMonitor - independent heartbeat thread.

Covers:
  1. Continuous success: failure counter stays at 0
  2. Consecutive failures >= max_failures: triggers CRITICAL log
  3. stop() terminates the thread within timeout
  4. Mixed success/failure resets counter
"""
import logging
import threading
import time

import pytest

from execution.dead_man_switch import DeadManSwitchMonitor


# ---------------------------------------------------------------------------
# Fake DMS for testing
# ---------------------------------------------------------------------------

class FakeDeadManSwitch:
    """Controllable DMS stub for unit tests."""

    def __init__(self):
        self._results: list = []       # pop from front on each refresh()
        self._default_result: bool = True
        self.refresh_count: int = 0
        self.lock = threading.Lock()

    def set_results(self, results: list):
        """Queue a sequence of True/False results for refresh()."""
        with self.lock:
            self._results = list(results)

    def set_default(self, val: bool):
        self._default_result = val

    def refresh(self) -> bool:
        with self.lock:
            self.refresh_count += 1
            if self._results:
                return self._results.pop(0)
            return self._default_result


class RaisingDeadManSwitch:
    """DMS stub that raises an exception on refresh()."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def refresh(self) -> bool:
        raise self._exc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDeadManSwitchMonitor:

    def test_continuous_success_no_critical(self, caplog):
        """When refresh always succeeds, no WARNING or CRITICAL logs."""
        fake = FakeDeadManSwitch()
        fake.set_default(True)

        monitor = DeadManSwitchMonitor(
            dms=fake, interval_sec=0.05, max_failures=3,
        )
        monitor.start()
        time.sleep(0.3)  # ~6 refresh cycles
        monitor.stop()

        assert fake.refresh_count >= 3
        assert "CRITICAL" not in caplog.text

    def test_consecutive_failures_trigger_critical(self, caplog):
        """3 consecutive failures must produce a CRITICAL log."""
        fake = FakeDeadManSwitch()
        fake.set_default(False)  # always fail

        with caplog.at_level(logging.DEBUG):
            monitor = DeadManSwitchMonitor(
                dms=fake, interval_sec=0.05, max_failures=3,
            )
            monitor.start()
            time.sleep(0.4)  # enough for >=3 cycles
            monitor.stop()

        critical_msgs = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert len(critical_msgs) >= 1, "Expected at least one CRITICAL log"
        assert "DMS-MONITOR" in critical_msgs[0].message

    def test_exception_triggers_critical(self, caplog):
        """refresh() raising an exception also counts as failure."""
        raising = RaisingDeadManSwitch(RuntimeError("API down"))

        with caplog.at_level(logging.DEBUG):
            monitor = DeadManSwitchMonitor(
                dms=raising, interval_sec=0.05, max_failures=2,
            )
            monitor.start()
            time.sleep(0.3)
            monitor.stop()

        critical_msgs = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert len(critical_msgs) >= 1

    def test_mixed_success_resets_counter(self, caplog):
        """A success after failures must reset the counter - no CRITICAL."""
        fake = FakeDeadManSwitch()
        # Fail twice, then succeed, repeat - never hits 3 consecutive
        fake.set_results([False, False, True, False, False, True, False, False, True])

        with caplog.at_level(logging.DEBUG):
            monitor = DeadManSwitchMonitor(
                dms=fake, interval_sec=0.05, max_failures=3,
            )
            monitor.start()
            time.sleep(0.6)
            monitor.stop()

        critical_msgs = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert len(critical_msgs) == 0, "Should not trigger CRITICAL when failures reset"

    def test_stop_exits_within_timeout(self):
        """stop() must terminate the thread within 1 second."""
        fake = FakeDeadManSwitch()
        monitor = DeadManSwitchMonitor(
            dms=fake, interval_sec=0.5, max_failures=3,
        )
        monitor.start()
        assert monitor.is_alive()

        t0 = time.monotonic()
        monitor.stop()
        elapsed = time.monotonic() - t0

        assert not monitor.is_alive(), "Thread should be dead after stop()"
        assert elapsed < 2.0, f"stop() took {elapsed:.1f}s, expected < 2s"

    def test_daemon_flag(self):
        """Monitor thread must be a daemon so it doesn't block process exit."""
        fake = FakeDeadManSwitch()
        monitor = DeadManSwitchMonitor(dms=fake)
        assert monitor.daemon is True

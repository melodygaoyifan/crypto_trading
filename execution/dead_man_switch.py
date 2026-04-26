"""
================================================================================
HMATS v6.5.1 - Server-Side Dead-Man's Switch
================================================================================

Uses Kraken's CancelAllOrdersAfter REST API to set a server-side timer
that auto-cancels ALL open orders if not refreshed within the timeout.

This protects against scenarios where the CLIENT-SIDE cancel-on-disconnect
cannot fire:
  - Process crash (SIGKILL, OOM)
  - Machine power loss
  - OS freeze / blue screen
  - Network partition where WS close frame is never received

Architecture:
  - On start: set server timer to TIMEOUT_SEC
  - On each tick: refresh the timer (reset countdown)
  - On shutdown: disable the timer (timeout=0)
  - If client dies: Kraken server cancels orders after TIMEOUT_SEC

Usage:
    switch = DeadManSwitch(rest_client=kraken_rest_client)
    switch.start()           # Set initial timer
    switch.refresh()         # Call every tick (< TIMEOUT_SEC)
    switch.stop()            # Graceful shutdown - disable timer

================================================================================
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Default timeout: 60 seconds
# Must be longer than tick interval (~30s for 4H candle checks)
# but short enough to limit exposure after crash
DEFAULT_TIMEOUT_SEC = 60

# After this many consecutive refresh failures, escalate to CRITICAL
MAX_CONSECUTIVE_FAILURES = 3


class DeadManSwitch:
    """
    Server-side dead-man's switch using Kraken CancelAllOrdersAfter API.

    The switch must be refreshed periodically (each tick). If the client
    stops refreshing (crash/freeze), Kraken's server cancels all orders
    after the timeout expires.
    """

    def __init__(
        self,
        rest_client=None,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        dry_run: bool = False,
    ):
        """
        Args:
            rest_client: KrakenRESTClient instance with cancel_all_orders_after()
            timeout_sec: Seconds until server-side auto-cancel. Default 60.
            dry_run: If True, log but don't call API (for paper trading).
        """
        self._rest_client = rest_client
        self._timeout_sec = timeout_sec
        self._dry_run = dry_run
        # [P50 2026-04-25] _active is read by DeadManSwitchMonitor in a
        # background thread + written by start()/stop() on main thread.
        # Without a lock, stop() may flip False mid-refresh, leading to
        # half-cancelled positions / hung shutdown.
        import threading as _threading
        self._lock = _threading.RLock()
        self._active = False
        self._last_refresh: float = 0.0
        self._refresh_count: int = 0
        self._error_count: int = 0
        self._consecutive_failures: int = 0

    def start(self) -> bool:
        """
        Activate the dead-man's switch.

        Sets the initial server-side timer. Must be refreshed before
        timeout_sec expires.

        Returns:
            True if successfully set (or dry_run), False on error.
        """
        if self._dry_run:
            self._active = True
            self._last_refresh = time.monotonic()
            logger.info(
                f"[DEAD_MAN] Started (DRY RUN, timeout={self._timeout_sec}s)"
            )
            return True

        if not self._rest_client:
            logger.warning("[DEAD_MAN] No REST client - cannot set server timer")
            return False

        success = self._rest_client.cancel_all_orders_after(self._timeout_sec)
        if success:
            self._active = True
            self._last_refresh = time.monotonic()
            logger.info(
                f"[DEAD_MAN] Server-side timer ACTIVE: {self._timeout_sec}s"
            )
        else:
            self._error_count += 1
            logger.error("[DEAD_MAN] Failed to set server-side timer")

        return success

    def refresh(self) -> bool:
        """
        Refresh the dead-man's switch timer.

        Call this every tick to prevent the server from cancelling orders.
        Must be called more frequently than timeout_sec.

        Returns:
            True if successfully refreshed (or dry_run/inactive), False on error.
        """
        if not self._active:
            return True  # Not active, nothing to do

        if self._dry_run:
            self._last_refresh = time.monotonic()
            self._refresh_count += 1
            return True

        if not self._rest_client:
            return False

        success = self._rest_client.cancel_all_orders_after(self._timeout_sec)
        if success:
            self._last_refresh = time.monotonic()
            self._refresh_count += 1
            self._consecutive_failures = 0
        else:
            self._error_count += 1
            self._consecutive_failures += 1
            # Pull the underlying cause from the rest client (set in
            # cancel_all_orders_after's except branch). Without this the
            # operator has to scroll back to find a [KrakenREST] line and
            # correlate timestamps. Now each failure message carries why.
            _cause = getattr(self._rest_client, "_last_dead_man_error", None) or "unknown"
            logger.warning(
                f"[DEAD_MAN] Refresh FAILED ({self._consecutive_failures}/{MAX_CONSECUTIVE_FAILURES} "
                f"consecutive). Timer may expire in {self._timeout_sec}s! cause={_cause}"
            )
            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.critical(
                    f"[DEAD_MAN] CRITICAL: {self._consecutive_failures} consecutive refresh "
                    f"failures! Server timer will expire - all orders will be cancelled. "
                    f"cause={_cause}. Check network/API connectivity immediately."
                )

        return success

    def stop(self) -> bool:
        """
        Disable the dead-man's switch (graceful shutdown).

        Sends timeout=0 to Kraken to cancel the server-side timer.

        Returns:
            True if successfully disabled, False on error.
        """
        # [P50 2026-04-25] Guard read+write under lock so DeadManSwitchMonitor
        # background thread can't race the flip mid-refresh.
        with self._lock:
            if not self._active:
                return True
            self._active = False

        if self._dry_run:
            logger.info("[DEAD_MAN] Stopped (DRY RUN)")
            return True

        if not self._rest_client:
            return False

        success = self._rest_client.cancel_all_orders_after(0)
        if success:
            logger.info(
                f"[DEAD_MAN] Server-side timer DISABLED "
                f"(refreshed {self._refresh_count} times, {self._error_count} errors)"
            )
        else:
            logger.error(
                "[DEAD_MAN] Failed to disable server-side timer! "
                "Orders may be auto-cancelled."
            )

        return success

    @property
    def is_active(self) -> bool:
        # [P50 2026-04-25] Read under lock to match start()/stop() writes.
        with self._lock:
            return self._active

    @property
    def seconds_since_refresh(self) -> float:
        if self._last_refresh == 0:
            return float('inf')
        return time.monotonic() - self._last_refresh

    def get_status(self) -> dict:
        """Get dead-man switch status for diagnostics."""
        return {
            "active": self._active,
            "timeout_sec": self._timeout_sec,
            "dry_run": self._dry_run,
            "seconds_since_refresh": round(self.seconds_since_refresh, 1),
            "refresh_count": self._refresh_count,
            "error_count": self._error_count,
            "consecutive_failures": self._consecutive_failures,
        }


class DeadManSwitchMonitor(threading.Thread):
    """
    Independent heartbeat thread for the Dead-Man's Switch.

    Calls dms.refresh() on a fixed interval, independent of the 4H tick loop.
    This prevents the Kraken server-side timer from expiring when the main
    loop is slow or blocked.

    Escalates to CRITICAL after max_failures consecutive refresh failures.
    """

    def __init__(
        self,
        dms: DeadManSwitch,
        interval_sec: int = 30,
        max_failures: int = 3,
        event_bus=None,
    ):
        super().__init__(name="DMS-Monitor", daemon=True)
        self._dms = dms
        self._interval_sec = interval_sec
        self._max_failures = max_failures
        self._event_bus = event_bus
        self._stop_event = threading.Event()
        self._consecutive_failures = 0

    def run(self) -> None:
        logger.info(
            f"[DMS-MONITOR] Heartbeat thread started "
            f"(interval={self._interval_sec}s, max_failures={self._max_failures})"
        )
        while not self._stop_event.is_set():
            try:
                success = self._dms.refresh()
                if success:
                    self._consecutive_failures = 0
                else:
                    self._consecutive_failures += 1
                    logger.warning(
                        f"[DMS-MONITOR] Refresh failed "
                        f"({self._consecutive_failures}/{self._max_failures} consecutive)"
                    )
                    if self._consecutive_failures >= self._max_failures:
                        logger.critical(
                            f"[DMS-MONITOR] CRITICAL: {self._consecutive_failures} consecutive "
                            f"heartbeat failures! Server timer may expire - all orders at risk. "
                            f"Check network/API connectivity immediately."
                        )
                        self._publish_alert()
            except Exception as exc:
                self._consecutive_failures += 1
                logger.warning(
                    f"[DMS-MONITOR] Refresh exception: {exc} "
                    f"({self._consecutive_failures}/{self._max_failures} consecutive)"
                )
                if self._consecutive_failures >= self._max_failures:
                    logger.critical(
                        f"[DMS-MONITOR] CRITICAL: {self._consecutive_failures} consecutive "
                        f"heartbeat failures (exception: {exc}). Server timer may expire."
                    )
                    self._publish_alert()

            self._stop_event.wait(self._interval_sec)

        logger.info("[DMS-MONITOR] Heartbeat thread stopped")

    def stop(self) -> None:
        """Signal the heartbeat thread to stop and wait for it to exit."""
        self._stop_event.set()
        self.join(timeout=5)
        if self.is_alive():
            logger.warning("[DMS-MONITOR] Thread did not exit within 5s timeout")

    def _publish_alert(self) -> None:
        """Publish alert event to event bus if available."""
        if self._event_bus is None:
            return
        try:
            # Import here to avoid circular deps - event_bus is optional
            from infra.event_bus import Event, EventType, EventPriority
            self._event_bus.publish(Event(
                event_type=EventType.ERROR,
                source="dead_man_switch_monitor",
                priority=EventPriority.CRITICAL,
                payload={
                    "message": "DMS heartbeat consecutive failures",
                    "consecutive_failures": self._consecutive_failures,
                    "max_failures": self._max_failures,
                },
            ))
        except Exception:
            pass  # Best-effort - CRITICAL log is the primary alert

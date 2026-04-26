"""
preserve_fresh_state.py — generic "fresh-state preservation across
transient updates" helper.

The pattern this codifies (P69 + the existing inline guard at
core/account_sync.py:359-367):

    Some piece of state has a TIMESTAMP and a STATUS (VALID / STALE /
    UNAVAILABLE). It's refreshed periodically by an update function that
    can fail transiently (network blip, brief auth issue, parse hiccup).
    The CORRECT behavior on failure is:

      - If prior state is FRESH (within max-age window): keep it VALID,
        log a WARNING, return False.
      - If prior state is STALE or never set: flip to UNAVAILABLE, log
        an ERROR, return False.

    The INCORRECT behavior (P69 root cause): unconditionally flip to
    UNAVAILABLE on any update failure. A single transient hiccup then
    cascades into a global P0 fail-closed even though the cached value
    was perfectly usable.

The pattern has appeared 3+ times in this codebase:
  - core/account_sync.py:359 (parse exception path — already correct)
  - core/account_sync.py:264 (fetch_balance failure path — fixed in P69)
  - core/account_sync.py:197 (no-exchange-client path — fixed in P69)

This helper exposes the contract as a single API so future state
holders don't reinvent it (or, more importantly, don't FAIL to invent
it and ship the P69 bug shape again).

Usage
-----

    from infra.preserve_fresh_state import PreserveFreshState

    # Inside your stateful object:
    self._equity_state = PreserveFreshState[float](
        max_age_seconds=60.0,
        invalid_default=0.0,
    )

    # When refreshing — wrap the update function:
    success, error = await self._equity_state.try_update_async(
        fn=self._fetch_equity_from_exchange,
        # Optional: log_label for the WARN/ERROR messages.
        log_label="ACCOUNT_SYNC.equity",
    )

    # When reading — fail-closed contract via get():
    equity = self._equity_state.get()  # raises StaleStateError if not VALID

    # When reading — graceful contract via get_safe():
    equity, is_valid = self._equity_state.get_safe()

The `try_update_*` methods accept either a sync or async fn. On failure
they decide whether to keep prior VALID state or flip to UNAVAILABLE
based on age. Operator-visible log lines include the failure type and
an explicit "keeping VALID state" or "flipping to UNAVAILABLE" verdict.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Awaitable, Callable, Generic, Optional, Tuple, TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StaleStateError(RuntimeError):
    """Raised by `get()` when the held state is not VALID. Use
    `get_safe()` for graceful (no-raise) access."""


class StateStatus(Enum):
    UNINITIALIZED = auto()
    VALID = auto()
    STALE = auto()        # past max_age_seconds since last successful update
    UNAVAILABLE = auto()  # update has failed AND no fresh-state preservation


@dataclass
class _Snapshot(Generic[T]):
    value: Optional[T] = None
    timestamp_monotonic: float = 0.0  # for age comparison; not wall-clock
    timestamp_wall: float = 0.0       # for human-readable display
    status: StateStatus = StateStatus.UNINITIALIZED
    last_error: Optional[str] = None
    failure_count: int = 0


class PreserveFreshState(Generic[T]):
    """Holds a value with a freshness window. Update can fail without
    invalidating the cached value if it's still within the freshness
    window.

    Thread-safe — reads + updates are guarded by an internal lock.
    Use one of the two `try_update_*` paths from a single coroutine /
    thread context to avoid pile-ups (the lock is held briefly across
    the user's update fn; if multiple callers race the second one
    waits).

    Type parameter T is the held value type (e.g. float, dict, your
    custom dataclass).
    """

    __slots__ = ("_max_age", "_invalid_default", "_lock", "_state", "_label")

    def __init__(
        self,
        max_age_seconds: float,
        invalid_default: Optional[T] = None,
        label: str = "state",
    ) -> None:
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        self._max_age = float(max_age_seconds)
        self._invalid_default = invalid_default
        self._lock = threading.Lock()
        self._state: _Snapshot[T] = _Snapshot()
        self._label = label

    # ----- read API ----------------------------------------------------------

    def is_valid(self) -> bool:
        """True iff cached value is VALID and within max-age window."""
        with self._lock:
            return self._is_fresh_locked()

    def get(self) -> T:
        """Fail-closed read. Raises StaleStateError if not VALID."""
        with self._lock:
            if not self._is_fresh_locked():
                raise StaleStateError(
                    f"{self._label}: state not VALID. "
                    f"status={self._state.status.name}, "
                    f"age={self._age_seconds_locked():.1f}s, "
                    f"last_error={self._state.last_error!r}"
                )
            assert self._state.value is not None or self._invalid_default is not None
            return self._state.value  # type: ignore[return-value]

    def get_safe(self) -> Tuple[Optional[T], bool]:
        """Graceful read. Returns (value, is_valid). On invalid, returns
        (invalid_default, False) — NEVER raises."""
        with self._lock:
            if self._is_fresh_locked():
                return self._state.value, True
            return self._invalid_default, False

    def get_status(self) -> StateStatus:
        with self._lock:
            self._refresh_status_locked()
            return self._state.status

    def age_seconds(self) -> float:
        """Age of the held value in seconds. inf if never set."""
        with self._lock:
            return self._age_seconds_locked()

    def failure_count(self) -> int:
        with self._lock:
            return self._state.failure_count

    # ----- update API --------------------------------------------------------

    def try_update(
        self,
        fn: Callable[[], T],
        log_label: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Synchronous update. Returns (success, error_message).

        On success: state set to VALID, error cleared.
        On failure within max-age: state KEPT VALID, last_error updated,
            failure_count incremented, WARNING logged.
        On failure outside max-age (or first call): state flipped to
            UNAVAILABLE, ERROR logged.
        """
        try:
            value = fn()
        except Exception as e:  # noqa: silent-swallow
            # _record_failure logs at WARN (preserve fresh) or ERROR
            # (flip to UNAVAILABLE) — visible per spec. Lint can't see
            # through the helper call so we annotate.
            return self._record_failure(e, log_label)
        return self._record_success(value)

    async def try_update_async(
        self,
        fn: Callable[[], Awaitable[T]],
        log_label: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Async update. Same contract as try_update."""
        try:
            value = await fn()
        except Exception as e:  # noqa: silent-swallow — see try_update above
            return self._record_failure(e, log_label)
        return self._record_success(value)

    def force_set(self, value: T) -> None:
        """Bypass — set value to VALID immediately. For tests / startup
        warm-up. Don't use on the hot path."""
        with self._lock:
            now_m = time.monotonic()
            now_w = time.time()
            self._state = _Snapshot(
                value=value,
                timestamp_monotonic=now_m,
                timestamp_wall=now_w,
                status=StateStatus.VALID,
                last_error=None,
                failure_count=0,
            )

    def force_invalidate(self, reason: str = "manual") -> None:
        """Bypass — flip to UNAVAILABLE regardless of age. For tests /
        kill-switch paths."""
        with self._lock:
            self._state.status = StateStatus.UNAVAILABLE
            self._state.last_error = f"force_invalidate: {reason}"

    # ----- internals ---------------------------------------------------------

    def _record_success(self, value: T) -> Tuple[bool, Optional[str]]:
        with self._lock:
            now_m = time.monotonic()
            now_w = time.time()
            self._state = _Snapshot(
                value=value,
                timestamp_monotonic=now_m,
                timestamp_wall=now_w,
                status=StateStatus.VALID,
                last_error=None,
                failure_count=0,  # reset on success
            )
        return True, None

    def _record_failure(
        self,
        exc: BaseException,
        log_label: Optional[str],
    ) -> Tuple[bool, Optional[str]]:
        err_str = f"{type(exc).__name__}: {exc!s}"
        label = log_label or self._label
        with self._lock:
            was_valid = self._state.status == StateStatus.VALID
            age = self._age_seconds_locked()
            self._state.failure_count += 1
            self._state.last_error = err_str
            if was_valid and age < self._max_age:
                # Preserve fresh state — this is the P69 invariant.
                logger.warning(
                    f"[{label}] update failed but prior value is fresh "
                    f"(age={age:.1f}s < {self._max_age:.0f}s); "
                    f"keeping VALID state. failure={err_str[:200]}"
                )
                return False, f"transient_keep_valid: {err_str}"
            # Genuinely invalid now.
            self._state.status = StateStatus.UNAVAILABLE
        logger.error(
            f"[{label}] update failed and prior state is "
            f"{'STALE' if was_valid else 'NOT_VALID'} "
            f"(age={age:.1f}s); flipping to UNAVAILABLE. failure={err_str}"
        )
        return False, err_str

    def _is_fresh_locked(self) -> bool:
        if self._state.status == StateStatus.UNAVAILABLE:
            return False
        if self._state.status == StateStatus.UNINITIALIZED:
            return False
        return self._age_seconds_locked() < self._max_age

    def _age_seconds_locked(self) -> float:
        if self._state.timestamp_monotonic == 0.0:
            return float("inf")
        return time.monotonic() - self._state.timestamp_monotonic

    def _refresh_status_locked(self) -> None:
        """Lazy STALE detection — flip VALID → STALE if past max_age."""
        if self._state.status == StateStatus.VALID:
            if self._age_seconds_locked() >= self._max_age:
                self._state.status = StateStatus.STALE

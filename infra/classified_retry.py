"""
classified_retry.py — generic "classify error, then decide retry policy
per class" helper.

The pattern this codifies (P68 dead-man classification + the parallel
P0_FAIL_CLOSED equity-fetch retry path):

    A function calls an external API. The API can fail in MULTIPLE
    ways:

      - PERMANENT (auth error, key permission missing, quota exhausted)
        — no point retrying inside the call. Surface a CRITICAL log
        with operator guidance. Return failure immediately so the
        endpoint isn't hammered.

      - TRANSIENT-NETWORK (connection reset, timeout, server unavailable)
        — short backoff (seconds), retry within the same call.

      - TRANSIENT-RATE-LIMIT (429, EAPI:Rate limit) — longer backoff,
        retry. Honor Retry-After if present.

      - DOMAIN-SPECIFIC (e.g. "Invalid nonce" for Kraken) — call a
        recovery callback (bump ratchet, reload time-difference) before
        retrying.

    The INCORRECT (P68 root-cause) pattern: retry loop that only
    handles ONE category and falls through to immediate failure for
    everything else, wasting attempts on permanent failures + losing
    transient-network recoveries.

This helper exposes the contract as a decorator + dispatch table so
future retry sites declare WHAT each error class maps to instead of
hand-rolling the if/elif tree.

Usage
-----

    from infra.classified_retry import (
        classified_retry, RetryAction, ErrorClassifier,
    )

    # Define how to classify your endpoint's errors:
    def classify_kraken_error(exc: BaseException) -> RetryAction:
        s = str(exc)
        if "Invalid nonce" in s:
            # Recovery before retry — caller registers handler below.
            return RetryAction.recover_then_retry("nonce_bump")
        if "Rate limit" in s or "429" in s:
            return RetryAction.backoff_and_retry(seconds=2.0, kind="rate_limit")
        if isinstance(exc, ccxt.AuthenticationError):
            return RetryAction.permanent(
                guidance="API key likely lacks 'Cancel & Close Orders' "
                         "permission on Kraken."
            )
        if isinstance(exc, (ccxt.NetworkError, ccxt.RequestTimeout)):
            return RetryAction.backoff_and_retry(seconds=0.5, kind="network")
        return RetryAction.unknown_fail()

    # Decorate the call:
    @classified_retry(
        classifier=classify_kraken_error,
        max_attempts=3,
        recovery_handlers={
            "nonce_bump": lambda self, attempt: self._bump_nonce_past_error(60),
        },
        log_label="KrakenREST.cancel_all_orders_after",
    )
    def cancel_all_orders_after(self, timeout_sec: int) -> bool:
        ...

    # Or use procedurally without decoration:
    success, errors = classified_retry_run(
        fn=lambda: self._exchange.cancel_all_orders_after(timeout_ms),
        classifier=classify_kraken_error,
        max_attempts=3,
        log_label="KrakenREST.cancel_all_orders_after",
    )

The procedural form is what fits the existing P68 site best — the
decorator form is for new code where you'd write the function fresh.
"""
from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class _ActionKind(Enum):
    PERMANENT = auto()        # don't retry — log CRITICAL with guidance
    RECOVER_RETRY = auto()    # call recovery handler, then retry
    BACKOFF_RETRY = auto()    # sleep, then retry
    UNKNOWN_FAIL = auto()     # treat as final attempt, log + return failure


@dataclass
class RetryAction:
    """What to do with a particular caught exception. Constructed via
    the classmethods so calling code reads as intent ("permanent",
    "backoff_and_retry", etc.)."""
    kind: _ActionKind
    backoff_seconds: float = 0.0
    recovery_handler_name: Optional[str] = None
    log_kind: str = ""           # "rate_limit" / "network" / etc.
    guidance: Optional[str] = None  # operator guidance for permanent failures

    @classmethod
    def permanent(cls, guidance: str) -> "RetryAction":
        return cls(kind=_ActionKind.PERMANENT, guidance=guidance)

    @classmethod
    def recover_then_retry(cls, handler_name: str) -> "RetryAction":
        return cls(
            kind=_ActionKind.RECOVER_RETRY,
            recovery_handler_name=handler_name,
        )

    @classmethod
    def backoff_and_retry(cls, seconds: float, kind: str = "transient") -> "RetryAction":
        return cls(
            kind=_ActionKind.BACKOFF_RETRY,
            backoff_seconds=float(seconds),
            log_kind=kind,
        )

    @classmethod
    def unknown_fail(cls) -> "RetryAction":
        return cls(kind=_ActionKind.UNKNOWN_FAIL)


# A classifier maps exception → RetryAction. Pure function.
ErrorClassifier = Callable[[BaseException], RetryAction]

# Recovery handler signature: (caller_self_or_None, attempt_index) -> Any
# Return value is ignored. Exceptions inside a handler are logged and
# treated as "recovery failed, fall through to UNKNOWN_FAIL".
RecoveryHandler = Callable[..., Any]


@dataclass
class _AttemptRecord:
    attempt: int
    error_type: str
    error_str: str
    action_kind: str

    def __str__(self) -> str:
        return f"attempt{self.attempt}={self.error_type}: {self.error_str[:200]} → {self.action_kind}"


def _format_verbose_error(exc: BaseException) -> str:
    """Drain `exc.args` so a ccxt-style URL-only `repr` doesn't hide the
    actual response body. Same shape as kraken_rest_client.py P67 verbose
    formatter."""
    arg_parts: List[str] = []
    for a in (getattr(exc, "args", ()) or ()):
        try:
            s = a if isinstance(a, str) else repr(a)
        except Exception:
            s = "<unrepr>"
        if s and s not in arg_parts:
            arg_parts.append(s)
    cause_chain = ""
    ce = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if ce is not None and ce is not exc:
        cause_chain = f" | caused_by={type(ce).__name__}: {ce}"
    return (
        f"{type(exc).__name__}: str={str(exc)!r} "
        f"args=[{' || '.join(arg_parts) if arg_parts else '<empty>'}]"
        f"{cause_chain}"
    )


def classified_retry_run(
    fn: Callable[[], T],
    classifier: ErrorClassifier,
    max_attempts: int = 3,
    recovery_handlers: Optional[Dict[str, RecoveryHandler]] = None,
    log_label: str = "classified_retry",
    caller_self: Any = None,
) -> Tuple[bool, Optional[T], List[_AttemptRecord]]:
    """Procedural form. Run `fn` up to max_attempts times, classifying
    each exception via `classifier` and dispatching the action.

    Returns (success, value_if_success, attempts_log).
      - success=True: fn returned without raising. value contains its
        return value.
      - success=False: fn exhausted retries OR hit a permanent action.
        value is None. attempts_log carries the per-attempt record so
        callers can stash it in their `_last_error` field for the
        downstream operator log line.

    Args:
        fn: zero-arg callable to attempt. For methods, pass a lambda
            that closes over self.
        classifier: error → action lookup.
        max_attempts: total attempts including the first. Default 3.
        recovery_handlers: name → handler dict. Names referenced from
            `RetryAction.recover_then_retry(name=...)`. Each handler is
            called as `handler(caller_self, attempt_idx)`.
        log_label: prefix for log lines (e.g. "KrakenREST.cancel_all").
        caller_self: passed as first positional to recovery handlers.
            Use to access self._bump_nonce_past_error etc.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    handlers = recovery_handlers or {}
    attempts: List[_AttemptRecord] = []

    for attempt in range(1, max_attempts + 1):
        try:
            result = fn()
            return True, result, attempts
        except Exception as e:
            verbose = _format_verbose_error(e)
            action = classifier(e)
            attempts.append(_AttemptRecord(
                attempt=attempt,
                error_type=type(e).__name__,
                error_str=str(e),
                action_kind=action.kind.name,
            ))
            logger.warning(
                f"[{log_label}] attempt {attempt}/{max_attempts} failed: {verbose} "
                f"→ {action.kind.name}"
            )

            if action.kind == _ActionKind.PERMANENT:
                # No point retrying. CRITICAL log with operator guidance.
                logger.critical(
                    f"[{log_label}] PERMANENT failure ({type(e).__name__}). "
                    f"NOT retrying. Guidance: {action.guidance or '(none provided)'}. "
                    f"Full error: {verbose}"
                )
                return False, None, attempts

            if attempt >= max_attempts:
                # Out of attempts. Log final + return.
                logger.error(
                    f"[{log_label}] failed after {max_attempts} attempts. "
                    f"Sequence: {' ; '.join(str(a) for a in attempts)}"
                )
                return False, None, attempts

            if action.kind == _ActionKind.UNKNOWN_FAIL:
                # Don't retry an unclassified failure. Same as PERMANENT
                # but no operator-guidance hook fires.
                logger.error(
                    f"[{log_label}] UNCLASSIFIED failure ({type(e).__name__}); "
                    f"not retrying. Add this case to the classifier if it's "
                    f"actually retryable. Full error: {verbose}"
                )
                return False, None, attempts

            if action.kind == _ActionKind.RECOVER_RETRY:
                handler = handlers.get(action.recovery_handler_name)
                if handler is None:
                    logger.error(
                        f"[{log_label}] classifier requested recovery "
                        f"'{action.recovery_handler_name}' but no handler "
                        f"registered. Treating as UNKNOWN_FAIL."
                    )
                    return False, None, attempts
                try:
                    handler(caller_self, attempt)
                    logger.info(
                        f"[{log_label}] recovery '{action.recovery_handler_name}' "
                        f"completed; retrying"
                    )
                except Exception as rec_exc:
                    logger.error(
                        f"[{log_label}] recovery '{action.recovery_handler_name}' "
                        f"failed: {type(rec_exc).__name__}: {rec_exc!s}; "
                        f"falling through to next attempt without recovery"
                    )
                continue

            if action.kind == _ActionKind.BACKOFF_RETRY:
                if action.backoff_seconds > 0:
                    logger.warning(
                        f"[{log_label}] transient {action.log_kind} on "
                        f"attempt {attempt}; backing off "
                        f"{action.backoff_seconds:.2f}s and retrying"
                    )
                    time.sleep(action.backoff_seconds)
                continue

    # Unreachable, but keeps mypy happy.
    return False, None, attempts


def classified_retry(
    classifier: ErrorClassifier,
    max_attempts: int = 3,
    recovery_handlers: Optional[Dict[str, RecoveryHandler]] = None,
    log_label: Optional[str] = None,
):
    """Decorator form. Wraps a method so each call goes through the
    retry/classify loop.

    The decorated function should return its success value normally;
    on failure (after all retries / permanent classification) the
    decorator returns whatever `default_on_failure` is set to (None).

    For more control over the failure return value, use the procedural
    form (`classified_retry_run`) directly.
    """
    def deco(fn: Callable[..., T]) -> Callable[..., Optional[T]]:
        label = log_label or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(self_or_arg=None, *args, **kwargs) -> Optional[T]:
            success, value, _ = classified_retry_run(
                fn=lambda: fn(self_or_arg, *args, **kwargs) if self_or_arg is not None else fn(*args, **kwargs),
                classifier=classifier,
                max_attempts=max_attempts,
                recovery_handlers=recovery_handlers,
                log_label=label,
                caller_self=self_or_arg,
            )
            return value if success else None

        return wrapper
    return deco

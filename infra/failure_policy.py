"""[P345] One home for "is this external failure transient or structural, and
until when should we stop calling?"

WHY. That question is re-derived at ~12 sites in this tree, and the recorded
history is that each site gets it wrong in a DIFFERENT direction:

  P293b  a MONTHLY quota was retried on a 900s transient backoff — ~2,900
         pointless requests, each logging a warning that read like a blip.
  P319   the backoff was persisted as a GUESS about the vendor's billing
         cycle, which fails ~19 days per month in the worst direction.
  P329   a transient READ failure (one Coinbase 502, one 30s cycle) engaged a
         30-minute structural backoff and disarmed an emergency watchdog for
         23 minutes during a real 7% move.
  P329b  transient timeouts on an advisory feed logged at ERROR forever — 9
         of 10 log lines, none actionable.
  P329c  a hard-dated account cap (400 "regain access on <date>") matched
         NEITHER the non-retryable list NOR the 429 branch, so it fell to a
         bare warning with NO backoff at all.

Five instances, five different wrong answers, one missing abstraction.

NOT the same axis as `infra/classified_retry.py`, which codifies INTRA-call
retry (sleep, retry, up to N attempts inside one call). Every bug above is
INTER-call suppression: "do not call again until X", consulted on a later
tick. Wiring classified_retry would not have prevented any of them. (It also
has zero callers — recorded here so the next reader does not mistake it for
the thing that was missing.)

THE CONTRACT, and each clause is one of the bugs above:

  TRANSIENT        never suppresses. A failure to READ is not evidence the
                   next attempt fails, and for a safety-relevant caller
                   suppression is the dangerous direction (P329).
  RATE_LIMITED     suppresses for the interval the SERVER stated.
  QUOTA_EXHAUSTED  suppresses until the reset the server stated; when no
                   date is given, a bounded re-probe cadence — never a
                   guessed calendar boundary (P319).
  PERMANENT        suppresses for the process. A 404 on a hardcoded path
                   cannot succeed by retrying (P218).

  An UNRECOGNISED failure classifies TRANSIENT, deliberately: going dark by
  mistake is worse than one wasted call, and that is the fail direction
  P293b recorded after the opposite choice cost a month of coverage.

  Severity tracks what an operator can DO (P202/P240): transient warns and
  escalates only when sustained; quota/permanent are ERROR once, because
  they are budget or configuration states, not market conditions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

__all__ = [
    "FailureClass",
    "FailurePolicy",
    "classify_external_failure",
    "DEFAULT_QUOTA_REPROBE_SEC",
]

# [P319] When a quota gives no reset instant, re-probe on a bounded cadence
# rather than encoding a guess about someone else's billing cycle.
DEFAULT_QUOTA_REPROBE_SEC = 24 * 3600.0

# Bounds a transient suppression can never exceed. TRANSIENT never suppresses
# at all, so this exists only so a malformed Retry-After cannot be laundered
# into a multi-day outage.
MAX_RATE_LIMIT_SEC = 6 * 3600.0


class FailureClass(Enum):
    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    PERMANENT = "permanent"


@dataclass(frozen=True)
class FailurePolicy:
    """What to do about one failure.

    `retry_after_sec is None` means DO NOT SUPPRESS — the caller should try
    again on its normal cadence. That is a distinct state from 0.0, which
    would read as "suppress for no time" and invites a caller to write a
    timestamp of `now`.
    """
    failure_class: FailureClass
    retry_after_sec: Optional[float]
    severity: str            # "warning" | "error"
    reason: str

    @property
    def suppresses(self) -> bool:
        return self.retry_after_sec is not None

    def retry_not_before(self, now: Optional[datetime] = None) -> Optional[datetime]:
        if self.retry_after_sec is None:
            return None
        base = now or datetime.now(timezone.utc)
        from datetime import timedelta
        return base + timedelta(seconds=self.retry_after_sec)


# "You will regain access on 2026-09-01 at 00:00 UTC."
_REGAIN = re.compile(
    r"regain access on (\d{4}-\d{2}-\d{2})(?:\s+at\s+(\d{2}:\d{2}))?",
    re.IGNORECASE)

_QUOTA_HINTS = (
    "usage limit", "quota exceeded", "monthly quota", "regain access",
    "upgrade your", "out of credits", "credit balance",
)
_PERMANENT_STATUSES = (401, 403, 404, 422)


def _parse_regain(message: str) -> Optional[float]:
    """Seconds until the reset the SERVER named, or None if it named none."""
    m = _REGAIN.search(message or "")
    if not m:
        return None
    try:
        stamp = f"{m.group(1)}T{m.group(2) or '00:00'}:00+00:00"
        dt = datetime.fromisoformat(stamp)
    except (ValueError, TypeError):  # noqa: silent-swallow — a reworded
        # message is not an error; returning None makes the caller fall
        # back to the bounded re-probe, which is the P319 fail direction.
        return None
    return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())


def _coerce_retry_after(value) -> Optional[float]:
    try:
        secs = float(value)
    except (TypeError, ValueError):  # noqa: silent-swallow — an unparseable
        # Retry-After is 'the server said nothing usable', and the caller
        # substitutes its default rather than trusting a garbage number.
        return None
    if secs != secs or secs in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return min(max(0.0, secs), MAX_RATE_LIMIT_SEC)


def classify_external_failure(
    *,
    status: Optional[int] = None,
    message: str = "",
    retry_after: Optional[float] = None,
) -> FailurePolicy:
    """Classify one external-API failure into an inter-call policy.

    Keyword-only so a caller cannot silently swap `status` and `retry_after`
    — they are both numbers and the resulting policy would be nonsense.
    """
    msg = (message or "").lower()

    # Quota FIRST: a dated cap can arrive on 400 or 429, and matching on the
    # status alone is exactly how P329c fell through every branch.
    if any(h in msg for h in _QUOTA_HINTS):
        stated = _parse_regain(message or "")
        return FailurePolicy(
            FailureClass.QUOTA_EXHAUSTED,
            stated if stated is not None else DEFAULT_QUOTA_REPROBE_SEC,
            "error",
            ("quota exhausted; reset stated by the server"
             if stated is not None else
             "quota exhausted with no stated reset — bounded re-probe, "
             "never a guessed billing boundary (P319)"),
        )

    if status == 429:
        secs = _coerce_retry_after(retry_after)
        return FailurePolicy(
            FailureClass.RATE_LIMITED,
            secs if secs is not None else 900.0,
            "warning",
            "rate limited; honouring the server's stated interval"
            if secs is not None else
            "rate limited with no usable Retry-After — default interval",
        )

    if status in _PERMANENT_STATUSES:
        return FailurePolicy(
            FailureClass.PERMANENT, float("inf"), "error",
            f"HTTP {status} cannot succeed by retrying — "
            f"credentials, permissions or a dead path",
        )

    # Everything else — timeouts, 5xx, connection resets, unknown shapes.
    # NOTE the None: a transient failure must NOT suppress the next attempt.
    return FailurePolicy(
        FailureClass.TRANSIENT, None, "warning",
        "transient; the next attempt is not suppressed",
    )

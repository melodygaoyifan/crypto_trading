"""[P345] The transient-vs-structural contract, and the five bugs it encodes.

Each test below is one recorded incident. The module exists because the same
question — "is this failure transient or structural, and until when do we
stop calling?" — is re-derived at ~12 sites and was answered wrongly five
times, each in a different direction.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from infra.failure_policy import (  # noqa: E402
    FailureClass, classify_external_failure, DEFAULT_QUOTA_REPROBE_SEC,
)


class TestTheFiveRecordedBugs:

    def test_p329_a_transient_read_failure_never_suppresses(self):
        """One Coinbase 502 lasting one 30s cycle engaged a 30-minute backoff
        and disarmed an emergency watchdog for 23 minutes."""
        p = classify_external_failure(status=502, message="Bad Gateway")
        assert p.failure_class is FailureClass.TRANSIENT
        assert p.retry_after_sec is None, (
            "None means DO NOT SUPPRESS — distinct from 0.0, which a caller "
            "would turn into a timestamp of now")
        assert p.suppresses is False

    def test_p293b_a_monthly_quota_is_not_a_transient_rate_limit(self):
        """A monthly quota retried on a 900s backoff = ~2,900 pointless
        requests, each logging like a blip."""
        p = classify_external_failure(
            status=429, message="API monthly quota exceeded - Upgrade your API plan")
        assert p.failure_class is FailureClass.QUOTA_EXHAUSTED
        assert p.retry_after_sec == DEFAULT_QUOTA_REPROBE_SEC

    def test_p329c_a_dated_cap_on_400_is_caught_by_MESSAGE_not_status(self):
        """It matched neither the non-retryable list nor the 429 branch, so it
        fell through to a bare warning with no backoff at all."""
        p = classify_external_failure(
            status=400,
            message=("You have reached your specified API usage limits. "
                     "You will regain access on 2099-09-01 at 00:00 UTC."))
        assert p.failure_class is FailureClass.QUOTA_EXHAUSTED
        assert p.suppresses and p.retry_after_sec > 0

    def test_p319_a_stated_reset_beats_a_guessed_one(self):
        """The backoff must come from the date the SERVER named, not from a
        guess about its billing cycle."""
        stated = classify_external_failure(
            status=400,
            message="usage limit; you will regain access on 2099-09-01 at 00:00 UTC")
        unstated = classify_external_failure(
            status=400, message="usage limit reached, try later")
        assert stated.retry_after_sec != DEFAULT_QUOTA_REPROBE_SEC
        assert unstated.retry_after_sec == DEFAULT_QUOTA_REPROBE_SEC, (
            "with no stated reset, fall back to a bounded re-probe")

    def test_p329b_transient_is_a_warning_not_an_error(self):
        """9 of 10 FRED log lines were ERROR-level timeouts on a feed that
        degrades to a documented neutral mock."""
        assert classify_external_failure(status=None, message="timed out").severity == "warning"

    def test_quota_and_permanent_are_errors_because_they_need_an_operator(self):
        for p in (classify_external_failure(status=429, message="monthly quota exceeded"),
                  classify_external_failure(status=403, message="forbidden")):
            assert p.severity == "error"


class TestTheFailDirection:

    def test_an_unrecognised_failure_classifies_TRANSIENT(self):
        """Going dark by mistake is worse than one wasted call — the
        direction P293b recorded after the opposite choice cost a month."""
        p = classify_external_failure(status=None, message="something novel")
        assert p.failure_class is FailureClass.TRANSIENT
        assert not p.suppresses

    def test_a_permanent_status_suppresses_for_the_process(self):
        for s in (401, 403, 404, 422):
            p = classify_external_failure(status=s, message="")
            assert p.failure_class is FailureClass.PERMANENT
            assert p.retry_after_sec == float("inf")

    @pytest.mark.parametrize("bad", [None, "", "abc", float("nan"), float("inf"), -5])
    def test_a_malformed_retry_after_cannot_produce_a_long_outage(self, bad):
        p = classify_external_failure(status=429, message="rate limited", retry_after=bad)
        assert p.failure_class is FailureClass.RATE_LIMITED
        assert 0.0 <= p.retry_after_sec <= 6 * 3600.0

    def test_a_server_stated_interval_is_honoured(self):
        p = classify_external_failure(status=429, message="rate limited", retry_after=42.0)
        assert p.retry_after_sec == 42.0

    def test_quota_is_checked_before_status(self):
        """A dated cap arrives on 400 AND on 429; matching status first is how
        P329c fell through every branch."""
        p = classify_external_failure(status=429, message="monthly quota exceeded")
        assert p.failure_class is FailureClass.QUOTA_EXHAUSTED

    def test_retry_not_before_is_none_exactly_when_not_suppressing(self):
        t = classify_external_failure(status=500, message="boom")
        assert t.retry_not_before() is None
        q = classify_external_failure(status=429, message="quota exceeded")
        assert q.retry_not_before() is not None


class TestItIsNotDecoration:
    """[P170] A seam nothing calls is decoration — and this tree already has
    `infra/classified_retry.py`, built for the neighbouring INTRA-call axis,
    with zero callers to this day. This module must not become the second."""

    def _production_callers(self):
        hits = []
        for p in REPO.rglob("*.py"):
            s = str(p)
            if any(x in s for x in ("venv", "archive", "site-packages",
                                    "tests", "failure_policy.py")):
                continue
            txt = p.read_text(encoding="utf-8-sig", errors="replace")
            # Match an IMPORT, not the substring. `stop_order_failure_policy`
            # is an unrelated config field in main.py and execution_manager.py,
            # and a substring scan counted both as callers — this guard was
            # vacuous on its first run, which is the P174 shape inside the
            # guard written to prevent decoration.
            if re.search(r"(from\s+infra\.failure_policy\s+import"
                         r"|import\s+infra\.failure_policy)", txt):
                hits.append(str(p.relative_to(REPO)))
        return hits

    def test_it_has_at_least_one_production_caller(self):
        callers = self._production_callers()
        assert callers, (
            "failure_policy has no production caller — it is decoration, "
            "exactly like classified_retry.py, which was written for this "
            "family and never called")

    def test_classified_retry_is_recorded_as_the_other_axis(self):
        """So the next reader does not wire the wrong helper and conclude the
        class is covered."""
        src = (REPO / "infra" / "failure_policy.py").read_text(encoding="utf-8-sig")
        assert "classified_retry" in src and "INTRA-call" in src

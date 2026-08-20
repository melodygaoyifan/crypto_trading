"""
[P329] A transient venue read failure must not disarm the inter-tick watchdog.

THE INCIDENT (live, 2026-08-19 23:39:53 UTC):

    [COINBASE_SLEEVE] reconcile failed: HTTPError: 502 Server Error: Bad Gateway
    [FastRiskTick][SLEEVE] ETH: EXIT_ONLY -> SKIPPED_STALE
                           (venue snapshot stale - refusing to act) - price_move=7.0%
    [FastRiskTick] ETH: EXIT_ONLY backoff engaged for 30 min after REJECTED
                        emergency exit. Reason: venue snapshot stale
    ... 23 minutes of "EXIT_ONLY suppressed" ...
    [FastRiskTick] ETH: cleared EXIT_ONLY suppression on 4H anchor refresh (P110)

ONE 502, lasting a single 30s cycle, disarmed ETH's emergency watchdog for
23m09s while ETH moved 7% from its 4H anchor. The move was real (measured over
the same log: ETH ranged 1916 -> 2321, SOL 77.24 -> 87.08). Nothing was lost
only because the book happened to be flat, which is luck, not design.

WHY THE CONFLATION WAS WRONG. P110's backoff exists for a STRUCTURAL rejection
- spot locked by a stop-loss that cannot be cancelled - which persists, so
retrying every 30s is pointless. `SKIPPED_STALE` is the opposite: we could not
READ the venue, no exit was ever attempted, and the condition self-corrects on
the next cycle. Retrying is also free of execution risk, because the staleness
guard in `sleeve_fast_risk_action` returns BEFORE `execute_target` - a retry
cannot place an order.

The fail direction that matters: for a safety watchdog, wrongly suppressing is
far worse than wrongly retrying.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from execution.fast_risk_tick import FastRiskTick, FastRiskAction  # noqa: E402


PRICE_MOVE_MD = {
    "current_price": 107.0,          # 7% above the anchor set below
    "volatility_30m": 0.0,
    "orderbook_depth_1pct_usd": 1_000_000.0,
}


def _armed_tick() -> FastRiskTick:
    t = FastRiskTick(shadow_mode=False)
    t.set_4h_anchor("ETH", price=100.0, volatility=0.01, depth=1_000_000.0)
    return t


class TestTheIncident:

    def test_a_transient_read_failure_leaves_the_watchdog_armed(self):
        """The regression. One 502 must not cost 30 minutes of coverage."""
        t = _armed_tick()
        before = t.evaluate("ETH", PRICE_MOVE_MD)
        assert before.action == FastRiskAction.EXIT_ONLY

        t.on_exit_failed("ETH", "venue snapshot stale - refusing to act",
                         transient=True)

        after = t.evaluate("ETH", PRICE_MOVE_MD)
        assert after.action == FastRiskAction.EXIT_ONLY, (
            "a failure to READ the venue must not suppress the exit trigger - "
            "the next cycle usually reconciles fine, and a retry cannot place "
            "an order because the staleness guard precedes execute_target")

    def test_a_structural_rejection_still_backs_off(self):
        """P110's real case is untouched: a venue that REJECTED the order will
        reject it again in 30s, so storming it is pointless."""
        t = _armed_tick()
        t.on_exit_failed("ETH", "spot locked by stop-loss", transient=False)
        after = t.evaluate("ETH", PRICE_MOVE_MD)
        assert after.action != FastRiskAction.EXIT_ONLY, (
            "P110's structural backoff must survive this change")

    def test_the_default_is_structural_so_the_legacy_caller_is_unchanged(self):
        """main.py's Kraken path reports a genuine order REJECTED and calls
        this without the flag. Its behaviour must not move."""
        t = _armed_tick()
        t.on_exit_failed("ETH", "EOrder:Insufficient funds")
        assert t.evaluate("ETH", PRICE_MOVE_MD).action != FastRiskAction.EXIT_ONLY


class TestSustainedUnreadabilityIsStillLoud:
    """Quieting the blip must not quiet the outage - the P303 shape."""

    def test_an_isolated_blip_warns_rather_than_criticals(self, caplog):
        t = _armed_tick()
        with caplog.at_level("WARNING"):
            t.on_exit_failed("ETH", "502", transient=True)
        assert not [r for r in caplog.records if r.levelname == "CRITICAL"]
        assert [r for r in caplog.records if r.levelname == "WARNING"]

    def test_a_sustained_outage_escalates_to_critical(self, caplog):
        t = _armed_tick()
        with caplog.at_level("WARNING"):
            for _ in range(t.TRANSIENT_ESCALATE_AFTER):
                t.on_exit_failed("ETH", "502", transient=True)
        crits = [r for r in caplog.records if r.levelname == "CRITICAL"]
        assert crits, (
            "a venue unreadable for minutes while we may be holding risk IS "
            "actionable, even though a single blip is not")
        assert "SUSTAINED" in crits[-1].getMessage()

    def test_the_streak_resets_when_the_venue_answers(self, caplog):
        """Without a reset, isolated blips accumulate over a long-lived process
        into a permanent CRITICAL that no longer describes the present."""
        t = _armed_tick()
        for _ in range(t.TRANSIENT_ESCALATE_AFTER - 1):
            t.on_exit_failed("ETH", "502", transient=True)
        t.on_venue_readable("ETH")
        with caplog.at_level("WARNING"):
            t.on_exit_failed("ETH", "502", transient=True)
        assert not [r for r in caplog.records if r.levelname == "CRITICAL"], (
            "one blip after a recovery must not inherit the old streak")

    def test_the_streak_is_per_asset(self):
        t = _armed_tick()
        for _ in range(3):
            t.on_exit_failed("ETH", "502", transient=True)
        assert t._venue_unreadable_streak.get("SOL", 0) == 0

    def test_transient_never_writes_the_structural_backoff_state(self):
        """The load-bearing mechanism: no `_exit_failed_at` write means no
        suppression can be computed from it."""
        t = _armed_tick()
        t.on_exit_failed("ETH", "502", transient=True)
        assert "ETH" not in t._exit_failed_at


class TestTheCallerWiresItCorrectly:
    """The class change is inert unless main.py passes the flag."""

    def _sleeve_block(self) -> str:
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        i = src.index("_frs_st, _frs_why = await sleeve_fast_risk_action")
        return src[i:i + 6000]

    def test_the_sleeve_caller_marks_skipped_stale_as_transient(self):
        blk = self._sleeve_block()
        assert 'transient=(_frs_st == "SKIPPED_STALE")' in blk, (
            "SKIPPED_STALE is the read-failure status; without the flag the "
            "incident recurs exactly as it happened")

    def test_error_is_left_structural(self):
        """An exception inside the helper is not evidence the next call
        succeeds, so it keeps the conservative backoff."""
        blk = self._sleeve_block()
        assert '"ERROR", "SKIPPED_STALE"' in blk
        assert 'transient=(_frs_st == "ERROR")' not in blk

    def test_the_caller_clears_the_streak_when_the_venue_answers(self):
        blk = self._sleeve_block()
        assert "on_venue_readable" in blk, (
            "a streak that only counts up stops describing the present")
        # The reset must NOT fire on the statuses that mean it failed.
        i = blk.index("on_venue_readable")
        guard = blk[max(0, i - 400):i]
        for bad in ("SKIPPED_STALE", "ERROR"):
            assert bad in guard, (
                f"{bad} must be excluded from the reset, or a permanently "
                f"unreadable venue would keep clearing its own streak")


class TestSleeveHelperOrderingIsWhatMakesRetrySafe:
    """The claim 'a retry cannot place an order' is load-bearing for the whole
    fix, so it is pinned rather than asserted in a comment."""

    def test_the_staleness_guard_precedes_execute_target(self):
        from tests._source_scan import code_only
        src = code_only(REPO / "main.py", strip_docstrings=True)
        i = src.index("async def sleeve_fast_risk_action")
        body = src[i:i + 3000]
        # Docstrings stripped: that helper's own docstring names
        # execute_target several lines ABOVE the code, so an unstripped scan
        # compares prose against code and fails on a correct implementation.
        stale = body.index("SKIPPED_STALE")
        exec_at = body.index("execute_target")
        assert stale < exec_at, (
            "if execute_target ever moves above the staleness guard, retrying "
            "a transient failure starts placing orders and P329's reasoning "
            "no longer holds")


# =============================================================================
# [P329b] The same shape, one layer out: FRED's advisory timeouts
# =============================================================================

class TestFredAdvisoryFailuresAreNotAllCriticals:
    """Measured live 2026-08-19: 9 of the 10 FRED log lines in a 16h window
    were ERROR-level timeouts forwarded to Discord, while the GCI's own log
    said the affected indicator "contributes nothing to Macro". An alert an
    operator cannot act on, every 4H, is the P202/P240 shape — and the
    `Releases` endpoint timed out on EVERY refresh, so the one genuinely
    diagnostic signal was buried in its own noise.
    """

    def _feed(self):
        from data_mgmt.feeds.fred_feed import FREDFeed
        return FREDFeed(api_key="x", mock_mode=True)

    def test_an_isolated_timeout_warns_rather_than_errors(self, caplog):
        f = self._feed()
        with caplog.at_level("WARNING"):
            f._report_endpoint_failure("series:DFF", "Series DFF timed out",
                                       "degrades to a NEUTRAL mock")
        assert not [r for r in caplog.records if r.levelname == "ERROR"]
        assert [r for r in caplog.records if r.levelname == "WARNING"]

    def test_a_sustained_failure_escalates_exactly_once(self, caplog):
        """Escalate so a dead endpoint is heard — but only once, or the ERROR
        becomes the wallpaper it replaced."""
        f = self._feed()
        with caplog.at_level("WARNING"):
            for _ in range(f.SUSTAINED_FAILURES + 4):
                f._report_endpoint_failure("releases", "Releases timed out",
                                           "event_window stays INACTIVE")
        errs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errs) == 1, f"expected exactly one ERROR, got {len(errs)}"
        assert "SUSTAINED" in errs[0].getMessage()

    def test_the_message_states_the_consequence(self, caplog):
        """'Series DGS10 timed out' tells an operator nothing about whether
        anything is now wrong."""
        f = self._feed()
        with caplog.at_level("WARNING"):
            f._report_endpoint_failure("series:DGS10", "Series DGS10 timed out",
                                       "contributes nothing to Macro")
        assert "contributes nothing to Macro" in caplog.records[-1].getMessage()

    def test_recovery_resets_the_streak(self, caplog):
        f = self._feed()
        for _ in range(f.SUSTAINED_FAILURES + 2):
            f._report_endpoint_failure("series:DFF", "x", "y")
        f._report_endpoint_ok("series:DFF")
        caplog.clear()   # the pre-recovery ERROR is captured by default
        with caplog.at_level("WARNING"):
            f._report_endpoint_failure("series:DFF", "x", "y")
        assert not [r for r in caplog.records if r.levelname == "ERROR"], (
            "a blip after recovery must not inherit the old streak")

    def test_streaks_are_per_endpoint(self):
        f = self._feed()
        for _ in range(f.SUSTAINED_FAILURES + 1):
            f._report_endpoint_failure("releases", "x", "y")
        assert f._endpoint_fail_streak.get("series:DFF", 0) == 0

    def test_the_success_paths_reset(self):
        """Without a reset in the code (not just the helper) the streak only
        ever counts up."""
        from tests._source_scan import code_only
        src = code_only(REPO / "data_mgmt" / "feeds" / "fred_feed.py",
                        strip_docstrings=True)
        assert src.count("_report_endpoint_ok(") >= 3, (
            "expected the definition plus a reset on BOTH success paths "
            "(series observation and releases)")


# =============================================================================
# [P329c] A hard-dated account cap is not a transient error
# =============================================================================

class TestHaikuUsageLimitIsNotRetried:
    """Live 2026-08-20 03:05:08:

        400 invalid_request_error: You have reached your specified API usage
        limits. You will regain access on 2026-09-01 at 00:00 UTC.

    400 was in neither the non-retryable list (401/403/404/422) nor the 429
    branch, so it fell to a bare warning with NO backoff and would have been
    retried once per asset per tick (~18/day) until September. P293b/P319, in
    a third feed.
    """

    def _classify(self, status: int, message: str):
        from agents.sentiment_llm_agent import LLMSentimentAgent

        class _Err(Exception):
            status_code = status

            def __str__(self):
                return message

        return LLMSentimentAgent._classify_haiku_error(_Err())

    def test_a_usage_limit_400_is_non_retryable(self):
        code, non_retryable, reason = self._classify(
            400, "You have reached your specified API usage limits. "
                 "You will regain access on 2026-09-01 at 00:00 UTC.")
        assert code == 400 and non_retryable is True
        assert "2026-09-01T00:00Z" in reason

    def test_an_ordinary_400_is_left_alone(self):
        """A genuinely malformed request is a BUG we want to keep seeing —
        silencing it behind a long cooldown would hide a real defect."""
        code, non_retryable, _ = self._classify(400, "messages.0: invalid field")
        assert not (code == 400 and non_retryable is True)

    def test_the_reset_instant_parses(self):
        from agents.sentiment_llm_agent import _parse_regain_utc
        assert _parse_regain_utc(
            "you will regain access on 2026-09-01 at 00:00 utc") == "2026-09-01T00:00Z"

    def test_unparseable_wording_returns_none_not_a_guess(self):
        """The fail direction: if Anthropic rewords the message we fall back to
        the standard hard-disable cooldown, never to 'retry immediately'."""
        from agents.sentiment_llm_agent import _parse_regain_utc
        assert _parse_regain_utc("you have reached your limits, try later") is None

    def test_the_caller_sizes_the_cooldown_from_the_stated_reset(self):
        from tests._source_scan import code_only
        src = code_only(REPO / "agents" / "sentiment_llm_agent.py",
                        strip_docstrings=True)
        # The CALLER's occurrence, not the classifier's return statement.
        i = src.rindex('usage_limit_400')
        blk = src[i:i + 1600]
        assert "_open_hard_disable" in blk, "a hard-dated cap must open a cooldown"
        assert "total_seconds()" in blk, "the cooldown must come from the stated reset"

    def test_the_message_says_it_is_a_billing_state(self):
        """P202: an alert an operator cannot fix by debugging must say so, and
        must say what still works."""
        from tests._source_scan import code_only
        src = code_only(REPO / "agents" / "sentiment_llm_agent.py",
                        strip_docstrings=True)
        i = src.index("ACCOUNT USAGE LIMIT")
        blk = src[i:i + 600]
        assert "BILLING" in blk and "heuristic" in blk


class TestTheFredHandlersActuallyCallTheHelper:
    """[P329b] The tests above exercise `_report_endpoint_failure` directly,
    which proves the helper works and says NOTHING about whether the except
    handlers reach it — the P234 gap. These drive a REAL timeout through
    `_fetch_series` / `_fetch_releases`.
    """

    def _feed_and_timing_out_session(self):
        import asyncio
        from data_mgmt.feeds.fred_feed import FREDFeed

        class _Session:
            def get(self, *a, **k):
                raise asyncio.TimeoutError()

        return FREDFeed(api_key="x", mock_mode=False), _Session()

    def test_a_real_series_timeout_routes_through_the_helper(self, caplog):
        import asyncio
        f, sess = self._feed_and_timing_out_session()
        with caplog.at_level("WARNING"):
            out = asyncio.run(f._fetch_series(sess, "VIXCLS"))
        assert out is None, "the caller contract (None on failure) must not move"
        msgs = [r.getMessage() for r in caplog.records]
        assert any("consecutive=" in m for m in msgs), (
            "the handler still logs the bare pre-P329b line — the helper is "
            "wired in the tests but not in the code path that runs")
        assert any("VIXCLS" in m for m in msgs)

    def test_a_real_releases_timeout_routes_through_the_helper(self, caplog):
        import asyncio
        f, sess = self._feed_and_timing_out_session()
        with caplog.at_level("WARNING"):
            out = asyncio.run(f._fetch_releases(sess))
        assert out == [], "the caller contract ([] on failure) must not move"
        assert any("consecutive=" in r.getMessage() for r in caplog.records)

    def test_a_real_timeout_states_the_consequence(self, caplog):
        import asyncio
        f, sess = self._feed_and_timing_out_session()
        with caplog.at_level("WARNING"):
            asyncio.run(f._fetch_series(sess, "DGS10"))
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "NEUTRAL mock" in joined, (
            "'Series DGS10 timed out' alone tells an operator nothing about "
            "whether anything is now wrong")

    def test_repeated_real_timeouts_escalate_once_then_go_quiet(self, caplog):
        import asyncio
        f, sess = self._feed_and_timing_out_session()
        with caplog.at_level("WARNING"):
            for _ in range(f.SUSTAINED_FAILURES + 5):
                asyncio.run(f._fetch_series(sess, "DFF"))
        errs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errs) == 1, (
            f"a dead endpoint must be heard once, not every refresh; got {len(errs)}")

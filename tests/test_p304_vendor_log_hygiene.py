"""
[P304] A transient Coinbase 502 sent 27 lines of the venue's HTML error page
to Discord — for a condition the system had already handled correctly.

The 502 itself was fine and the handling was right (P265a: refuse the
wrong-denomination FCM subset, serve last-known equity). What was wrong was
the REPORTING: the vendor SDK logs at ERROR with the whole error page
attached, and DiscordLogHandler sits on the ROOT logger at min_level=ERROR.

These tests pin the three properties that make the fix a fix rather than a
mute button:
  * a 4xx stays ERROR (that one is ours: key, permission, malformed request)
  * a SUSTAINED 5xx re-escalates (a venue that is down IS actionable)
  * no record is ever dropped
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from infra.vendor_log_hygiene import (  # noqa: E402
    MAX_MESSAGE_CHARS, SUSTAINED_5XX_COUNT, VendorHTTPLogFilter,
    http_status_in, install_vendor_log_filters, strip_html_body)

# The real thing, from the 2026-08-18 10:54:29 incident.
REAL_502 = (
    "HTTP Error: 502 Server Error: Bad Gateway <html>\n<head>\n"
    "<title>Coinbase</title>\n<meta name=\"robots\" content=\"noindex\">\n"
    "<style type=\"text/css\">html{line-height:1.15;"
    "-webkit-text-size-adjust:100%}main{display:block}h1{font-size:2em;"
    "margin:.67em 0}" + ("a{background-color:transparent}" * 40) + "</style>"
)


def _rec(msg, level=logging.ERROR, created=1_000_000.0):
    r = logging.LogRecord("coinbase.RESTClient", level, __file__, 1, msg,
                          None, None)
    r.created = created
    return r


# =============================================================================
# Truncation
# =============================================================================

class TestTheHtmlNeverSurvives:

    def test_the_real_incident_message_is_cut_to_one_short_line(self):
        out = strip_html_body(REAL_502)
        assert len(out) <= MAX_MESSAGE_CHARS
        assert "\n" not in out
        assert "webkit" not in out and "line-height" not in out
        assert "502" in out, "the status code is the part worth keeping"

    def test_a_normal_message_is_left_alone(self):
        msg = "HTTP Error: 401 Unauthorized"
        assert strip_html_body(msg) == msg

    @pytest.mark.parametrize("opener", [
        "<html>", "<HTML>", "<!DOCTYPE html>", "<head>", "<style>", "<body>",
    ])
    def test_every_markup_opener_is_recognised(self, opener):
        out = strip_html_body(f"HTTP Error: 502 Bad Gateway {opener} junk" +
                              "x" * 5000)
        assert len(out) <= MAX_MESSAGE_CHARS
        assert "junk" not in out

    def test_a_giant_message_with_no_markup_is_still_capped(self):
        """Length alone must bound it — an SDK could dump JSON, not HTML."""
        out = strip_html_body("HTTP Error: 500 " + "z" * 10_000)
        assert len(out) <= MAX_MESSAGE_CHARS

    def test_empty_and_none_are_safe(self):
        assert strip_html_body("") == ""
        assert strip_html_body(None) is None


# =============================================================================
# Severity — the part that must not become a mute button
# =============================================================================

class TestSeverityStaysHonest:

    def test_an_isolated_5xx_is_demoted_to_warning(self):
        f = VendorHTTPLogFilter()
        r = _rec(REAL_502)
        assert f.filter(r) is True
        assert r.levelno == logging.WARNING
        assert "transient" in r.getMessage()

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 429])
    def test_every_4xx_stays_at_error(self, status):
        """A 4xx is OUR bug — bad key, missing permission, malformed request.
        Quieting the venue's problems must not quiet ours."""
        f = VendorHTTPLogFilter()
        r = _rec(f"HTTP Error: {status} Client Error <html>junk</html>")
        f.filter(r)
        assert r.levelno == logging.ERROR, f"{status} must stay loud"

    def test_a_sustained_5xx_burst_re_escalates(self):
        """A venue that is genuinely down IS actionable — we cannot trade."""
        f = VendorHTTPLogFilter()
        last = None
        for i in range(SUSTAINED_5XX_COUNT):
            last = _rec(REAL_502, created=1_000_000.0 + i)
            f.filter(last)
        assert last.levelno == logging.ERROR
        assert "SUSTAINED" in last.getMessage()

    def test_5xx_spread_beyond_the_window_stay_blips(self):
        """Otherwise one blip a day eventually 'sustains' and the escalation
        becomes meaningless (the latch that never re-arms, P265f)."""
        f = VendorHTTPLogFilter()
        last = None
        for i in range(SUSTAINED_5XX_COUNT + 3):
            last = _rec(REAL_502, created=1_000_000.0 + i * 10_000.0)
            f.filter(last)
        assert last.levelno == logging.WARNING
        assert "SUSTAINED" not in last.getMessage()

    def test_a_warning_level_5xx_is_not_touched_upward(self):
        f = VendorHTTPLogFilter()
        r = _rec(REAL_502, level=logging.WARNING)
        f.filter(r)
        assert r.levelno == logging.WARNING

    def test_no_record_is_ever_dropped(self):
        """The filter may re-level and re-word; it may never silence."""
        f = VendorHTTPLogFilter()
        for msg, lvl in [(REAL_502, logging.ERROR),
                         ("HTTP Error: 404", logging.ERROR),
                         ("something else entirely", logging.INFO),
                         ("", logging.ERROR)]:
            assert f.filter(_rec(msg, level=lvl)) is True

    def test_args_are_cleared_so_nothing_can_re_expand_the_blob(self):
        """record.msg is rewritten; a surviving args tuple would let a
        downstream formatter rebuild the original message."""
        f = VendorHTTPLogFilter()
        r = logging.LogRecord("coinbase.RESTClient", logging.ERROR, __file__,
                              1, "HTTP Error: %s %s", (502, REAL_502), None)
        r.created = 1_000_000.0
        f.filter(r)
        assert r.args == ()
        assert len(r.getMessage()) <= MAX_MESSAGE_CHARS


class TestStatusParsing:

    @pytest.mark.parametrize("msg,expect", [
        ("HTTP Error: 502 Server Error", 502),
        ("HTTP Error: 401 Unauthorized", 401),
        ("no status here", None),
        ("", None),
        ("value 1234 is not a status", None),
    ])
    def test_status_extraction(self, msg, expect):
        assert http_status_in(msg) == expect


# =============================================================================
# Installation
# =============================================================================

class TestInstallation:

    def _clean(self):
        for n in ("coinbase", "coinbase.RESTClient"):
            lg = logging.getLogger(n)
            for flt in list(lg.filters):
                if isinstance(flt, VendorHTTPLogFilter):
                    lg.removeFilter(flt)

    def test_install_is_idempotent(self):
        self._clean()
        try:
            first = install_vendor_log_filters()
            assert first >= 1
            assert install_vendor_log_filters() == 0, (
                "a second call must add nothing — startup can run twice")
        finally:
            self._clean()

    def test_end_to_end_the_blob_does_not_reach_a_root_handler(self):
        """The actual failure path: SDK logs ERROR -> propagates to root ->
        DiscordLogHandler(min_level=ERROR) emits. Assert both halves change."""
        self._clean()
        seen = []

        class _Capture(logging.Handler):
            def emit(self, record):
                seen.append((record.levelno, record.getMessage()))

        root = logging.getLogger()
        h = _Capture(level=logging.WARNING)
        root.addHandler(h)
        try:
            install_vendor_log_filters()
            lg = logging.getLogger("coinbase.RESTClient")
            lg.error(REAL_502)
            assert seen, "the record must still be emitted"
            lvl, msg = seen[-1]
            assert lvl == logging.WARNING, "isolated 502 must not reach Discord"
            assert len(msg) <= MAX_MESSAGE_CHARS
            assert "webkit" not in msg
        finally:
            root.removeHandler(h)
            self._clean()

    def test_main_installs_it_before_attaching_discord(self):
        """Order is load-bearing: the Discord handler goes on the ROOT logger,
        so the filter must be on the vendor logger first."""
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        i_filter = src.index("install_vendor_log_filters")
        i_discord = src.index("DiscordLogHandler(self.audit_manager.discord)")
        assert i_filter < i_discord


class TestOurOwnSiteDoesNotEmbedTheBlobEither:

    def test_the_sleeve_equity_warning_is_truncated(self):
        src = (REPO / "exchange" / "coinbase_sleeve.py").read_text(
            encoding="utf-8-sig")
        i = src.index("portfolio equity fetch failed")
        block = src[i - 400:i + 400]
        assert "strip_html_body" in block or "_shb(" in block, (
            "our own WARNING carried the same HTML — it is the line the "
            "operator actually reads")

    def test_the_handled_path_still_refuses_the_wrong_denomination(self):
        """The 502 handling itself was CORRECT (P153/P265a). Truncating the
        log must not have touched it."""
        src = (REPO / "exchange" / "coinbase_sleeve.py").read_text(
            encoding="utf-8-sig")
        assert "WRONG-DENOMINATION" in src, (
            "the P265a refusal to substitute the FCM subset must survive")
        assert "NOT substituted" in src
        assert "FCM-ONLY SUBSET" in src

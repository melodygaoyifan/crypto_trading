"""
[P308] Three standing warnings, each of which was either spending money or
telling the operator something false.

  1. The options agent re-requested three permanently-404 CoinGlass paths on
     every tick. P218 latched the WARNING and left the CALL in place, so the
     noise stopped and the paid quota kept draining.
  2. `execution/__init__` imported `ExecutionPlan`, a name that has never
     existed in sota_scheduler — so the whole try-block raised and the two
     symbols that DO import were silently dropped, while every boot logged
     "SOTA scheduler unavailable" about a module that is present.
  3. The GCI mock warning told the operator to "Check Yahoo Finance API" for
     a series P293 deliberately left unmapped in FRED and which P294 verified
     is write-only. An instruction nobody can act on (P202).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# =============================================================================
# 1. A permanently-404 endpoint must stop being requested
# =============================================================================

class TestDead404EndpointsAreNotReRequested:

    def _agent(self):
        from agents.options_sentiment_agent import OptionsSentimentAgent
        a = OptionsSentimentAgent.__new__(OptionsSentimentAgent)
        # Class-level registries — isolate so tests cannot leak into each other.
        type(a)._DEAD_ENDPOINTS_404 = set()
        type(a)._DEAD_ENDPOINTS_WARNED = set()
        return a

    def test_a_404_marks_the_path_dead(self):
        a = self._agent()
        assert a._is_known_dead("/option/info/oi") is False
        a._report_http("/option/info/oi", 404)
        assert a._is_known_dead("/option/info/oi") is True

    def test_the_registry_fills_even_when_the_warning_is_already_latched(self):
        """THE BUG. The warn-latch returns early after the first report; if the
        registration sat below it, the very first 404 would warn and never mark
        the path dead — and the calls would continue forever."""
        a = self._agent()
        a._report_http("/option/info/oi", 404)          # warns + registers
        type(a)._DEAD_ENDPOINTS_404 = set()             # simulate losing it
        a._report_http("/option/info/oi", 404)          # latched: no warning
        assert a._is_known_dead("/option/info/oi") is True, (
            "registration must not be gated behind the warn-once latch")

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_transient_statuses_are_never_marked_dead(self, status):
        """Skipping a 5xx/429 would convert a venue blip into a permanently
        dark feed — the opposite failure, and a worse one (P303)."""
        a = self._agent()
        a._report_http("/option/info/oi", status)
        assert a._is_known_dead("/option/info/oi") is False

    def test_all_three_fetchers_consult_the_registry(self):
        """A guard on two of three endpoints still leaks paid calls."""
        src = (REPO / "agents" / "options_sentiment_agent.py").read_text(
            encoding="utf-8-sig")
        assert src.count("_is_known_dead(") >= 4, (
            "one helper definition + one guard per dead endpoint")
        for ep in ("max-pain", "oi", "volume"):
            i = src.index(f'/option/info/{ep}"')
            assert "_is_known_dead" in src[i:i + 260], f"{ep} unguarded"

    def test_the_guarded_fetch_makes_no_http_call(self):
        """Behavioural, not a source pin (P234): drive the real method with a
        session that fails the test if it is used."""
        from agents.options_sentiment_agent import OptionsSentimentAgent
        a = self._agent()
        a.api_key = "x"
        a._headers = lambda: {}
        type(a)._DEAD_ENDPOINTS_404 = {"/option/info/oi"}

        class _NoCallSession:
            def get(self, *a, **k):
                raise AssertionError("a known-dead endpoint was requested")

        out = asyncio.run(
            OptionsSentimentAgent._fetch_option_oi(a, _NoCallSession(), "BTC"))
        assert out is None, "a skipped fetch returns None, like a failed one"

    def test_a_live_endpoint_is_still_requested(self):
        """The guard must not become a blanket disable."""
        from agents.options_sentiment_agent import OptionsSentimentAgent
        a = self._agent()
        a.api_key = "x"
        a._headers = lambda: {}
        called = {"n": 0}

        class _Resp:
            status = 404

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def json(self):
                return {}

        class _Session:
            def get(self, *a, **k):
                called["n"] += 1
                return _Resp()

        asyncio.run(
            OptionsSentimentAgent._fetch_option_oi(a, _Session(), "BTC"))
        assert called["n"] == 1, "an unproven endpoint must still be tried once"
        # ...and that one 404 is what registers it.
        assert a._is_known_dead("/option/info/oi") is True


# =============================================================================
# 2. The export list named a symbol that does not exist
# =============================================================================

class TestExecutionExportsAreReal:

    def test_the_scheduler_is_actually_available(self):
        import execution
        for name in ("SOTAExecutionScheduler", "SchedulerConfig",
                     "ScheduledOrder", "ScheduleType"):
            assert name in execution.__all__, name
            assert hasattr(execution, name), name

    def test_execution_plan_is_gone(self):
        """It never existed; naming it disabled the whole block.

        Comment-stripped, because the fix's own comment names the retired
        symbol to explain the retirement — a bare substring scan fires on its
        own explanation (P177, and the P192 `_emergency_flatten` mistake).
        """
        from tests._source_scan import code_only
        src = code_only(REPO / "execution" / "__init__.py")
        assert "ExecutionPlan" not in src

    def test_every_name_in_the_scheduler_block_exists_in_the_module(self):
        """The durable guard: the import list is a contract with the module."""
        import ast
        import execution.sota_scheduler as m
        src = (REPO / "execution" / "__init__.py").read_text(encoding="utf-8-sig")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and \
                    (node.module or "").endswith("sota_scheduler"):
                for alias in node.names:
                    assert hasattr(m, alias.name), (
                        f"execution/__init__ imports {alias.name!r}, which "
                        f"sota_scheduler does not define — that disables every "
                        f"other name in the same statement")

    def test_the_warning_now_names_the_cause(self):
        from tests._source_scan import code_only
        src = code_only(REPO / "execution" / "__init__.py")
        i = src.index("SOTA scheduler unavailable")
        assert "%s" in src[i:i + 60], (
            "a bare 'unavailable' hid which symbol was missing for months")


# =============================================================================
# 3. The GCI mock warning must not send the operator somewhere useless
# =============================================================================

class TestGciMockWarningIsActionable:

    def _warn(self, ticker):
        from data_mgmt.global_context_informer import MacroDataFetcher
        g = MacroDataFetcher.__new__(MacroDataFetcher)
        import logging as _l
        recs = []

        class _H(_l.Handler):
            def emit(self, r):
                recs.append(r.getMessage())

        lg = _l.getLogger("GlobalContextInformer")
        h = _H(level=_l.WARNING)
        lg.addHandler(h)
        try:
            MacroDataFetcher._generate_mock_indicator(g, ticker)
        finally:
            lg.removeHandler(h)
        return " ".join(recs)

    def test_gold_is_reported_as_expected_not_as_a_fault(self):
        msg = self._warn("GOLD")
        assert "Yahoo" not in msg, (
            "P293 moved this to FRED; sending the operator to Yahoo is a "
            "false instruction")
        assert "EXPECTED" in msg and "needs no action" in msg

    def test_an_unexpected_ticker_still_points_somewhere_real(self):
        msg = self._warn("DXY")
        assert "fred_macro_series" in msg
        assert "EXPECTED" not in msg, (
            "only the deliberately-unmapped ticker may be called expected")

    def test_the_mock_is_still_neutral_by_construction(self):
        """MACRO-FIX5: the whole reason a mock is tolerable here."""
        from data_mgmt.global_context_informer import MacroDataFetcher
        g = MacroDataFetcher.__new__(MacroDataFetcher)
        ind = MacroDataFetcher._generate_mock_indicator(g, "GOLD")
        assert ind.change_pct == 0.0
        assert ind.zscore_30d == 0.0
        assert ind.value == ind.prev_value

"""
test_mock_fallback_signaling.py — assert all mock fallbacks self-flag
======================================================================

[P113 (3/6) 2026-04-27] When a production feed falls back to mock data
(API key missing, fetch failed, response malformed), it MUST signal:
  1. Set `is_mock=True` on the returned data structure (downstream
     consumers can opt-out)
  2. Log at WARNING level (operator visibility)

Silent mock fallback was the P101 bug class: onchain_feed and
sentiment_feed silently returned mock data for EVERY call because
self.config was undefined → AttributeError → caught by outer except
→ mock returned with no flag and no log. Took an audit pass to find.

This test enforces that EVERY feed module with a `_fetch_mock` method
also has the proper signaling so the bug class can't recur.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
FEEDS_DIR = REPO / "data_mgmt" / "feeds"


# Feeds that have a _fetch_mock method
EXPECTED_FEEDS_WITH_MOCK = [
    "coinglass_feed",
    "cryptopanic_feed",
    "fred_feed",
    "lob_feed",
    "lunarcrush_feed",
    "macro_feed",
    "onchain_feed",
    "sentiment_feed",
    "trading_economics_feed",
]


@pytest.mark.parametrize("feed_name", EXPECTED_FEEDS_WITH_MOCK)
class TestFeedMockSignaling:
    """For each feed that returns mock data on fallback, verify it
    signals properly so downstream + operator can detect the degradation."""

    def test_feed_has_fetch_mock(self, feed_name):
        """Every listed feed must actually have _fetch_mock — catches
        accidental removal that would change fallback behavior."""
        feed_path = FEEDS_DIR / f"{feed_name}.py"
        if not feed_path.exists():
            pytest.skip(f"{feed_name}.py not found in repo")
        src = feed_path.read_text(encoding="utf-8")
        assert "_fetch_mock" in src, (
            f"{feed_name}.py listed as having mock fallback but no "
            f"_fetch_mock method found. Either remove from "
            f"EXPECTED_FEEDS_WITH_MOCK or restore the method."
        )

    def test_mock_returns_have_is_mock_or_logging(self, feed_name):
        """Each return path that falls to mock should EITHER set is_mock=True
        on the returned data OR be wrapped in a logger.warning() that
        names the feed.

        This is a structural check (grep-style) — not perfect but catches
        the most common silent-mock pattern that hit P101."""
        feed_path = FEEDS_DIR / f"{feed_name}.py"
        if not feed_path.exists():
            pytest.skip(f"{feed_name}.py not found")
        src = feed_path.read_text(encoding="utf-8")

        # Find _fetch_mock definition and read its body (indented block)
        m = re.search(r"def\s+_fetch_mock\s*\([^)]*\)[^:]*:", src)
        assert m is not None, f"{feed_name}: _fetch_mock signature not findable"
        # Read 100 lines after the def to capture body
        start = m.end()
        body = src[start:start + 5000]

        has_is_mock = "is_mock=True" in body or 'is_mock": True' in body
        # Some feeds use a different "mock indicator" pattern (e.g., neutral
        # constants in lunarcrush_feed.py post-P101). Allow these explicit
        # opt-outs by checking for the documentation.
        has_documented_neutral = (
            "PATCH-7a" in src or
            "neutral constants" in src.lower() or
            "neutral defaults" in src.lower() or
            "fail-safe" in src.lower() or
            "fail-CLOSED" in src
        )
        # P113 (3/6): logger.warning at the top of _fetch_mock is also
        # valid signaling — operator gets a once-per-process WARN log
        # when production falls back to mock data.
        has_warn_signal = (
            "logger.warning" in body[:500]  # in the FIRST 500 chars
            or "_mock_warned" in body[:500]
        )
        assert has_is_mock or has_documented_neutral or has_warn_signal, (
            f"{feed_name}._fetch_mock returns data without is_mock=True "
            f"flag AND without documented neutral-defaults design. This "
            f"is the P101 silent-mock-fallback shape. Either:\n"
            f"  (a) set is_mock=True on the returned data structure, OR\n"
            f"  (b) add a docstring/comment explaining why the mock is "
            f"     safe to consume silently (e.g. neutral constants per "
            f"     PATCH-7a) and re-run this test."
        )


class TestProductionMockGuards:
    """Verify the GUARDED test simulator pattern (P110 onchain_graph_alpha
    fix) is also applied to other test-only data sources."""

    def test_onchain_graph_simulator_guard_present(self):
        """OnChainDataSimulator must require HMATS_ALLOW_TEST_SIMULATORS
        env var per P110 fix."""
        path = REPO / "agents" / "onchain_graph_alpha.py"
        if not path.exists():
            pytest.skip("onchain_graph_alpha.py not found")
        src = path.read_text(encoding="utf-8")
        assert "HMATS_ALLOW_TEST_SIMULATORS" in src, (
            "OnChainDataSimulator missing the P110 production guard "
            "(HMATS_ALLOW_TEST_SIMULATORS env-gate). Without it, "
            "simulator can be accidentally instantiated in production "
            "and inject random data into agent signals."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

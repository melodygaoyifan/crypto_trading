"""[P413] Breadth-asset headline coverage: XRP/BNB scoped RSS sources.

XRP and BNB joined the tradeable universe (P412/P412c) with `[LLM_SENTIMENT]
... headlines=0 tradeable=False` -- the RSS roster was scoped to the home
trio. Probed 2026-08-26 before adding (the P314 admission method): a bare
`q=BNB` query FAILS the bar (1 inside 4h, median ~105 days -- the
newsbtc/utoday rejection class), but Google's `when:1d` operator flips both
assets above it (XRP: 32 inside 4h; BNB: 5 inside 4h -- above bitcoincom's
admitted 4). This lights coverage, not edge: llm_sentiment's IC remains
insignificant and it reaches orders only through the de-risk conviction
channel.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_mgmt.feeds.rss_news_feed import (  # noqa: E402
    RSS_SOURCES, RSSNewsItem, SOURCE_ASSET_SCOPE, _ASSET_PATTERNS)


def _item(title, source="cointelegraph", age_h=1.0):
    return RSSNewsItem(
        title=title,
        published_at=datetime.now(timezone.utc) - timedelta(hours=age_h),
        source=source, link="")


class TestBreadthSources:
    def test_the_scoped_sources_exist_and_are_fetched(self):
        names = {n for n, _ in RSS_SOURCES}
        assert "gnews_xrp" in names and "gnews_bnb" in names
        assert SOURCE_ASSET_SCOPE.get("gnews_xrp") == "XRP"
        assert SOURCE_ASSET_SCOPE.get("gnews_bnb") == "BNB"

    def test_the_queries_carry_the_when_operator(self):
        """A bare `q=BNB` measured 1 inside 4h with a ~105-day median -- the
        when:1d restriction IS what makes these sources admissible. Dropping
        it silently re-creates the stale-corpus failure P314 rejected
        newsbtc/utoday for."""
        urls = dict(RSS_SOURCES)
        assert "when%3A1d" in urls["gnews_xrp"]
        assert "when%3A1d" in urls["gnews_bnb"]

    def test_a_scoped_item_serves_its_asset_without_the_word(self):
        it = _item("Payments giant expands cross-border pilot",
                   source="gnews_xrp")
        assert it.matches("XRP") is True
        assert it.matches("BTC") is False

    def test_scope_is_not_a_wildcard(self):
        it = _item("Exchange token rallies", source="gnews_bnb")
        assert it.matches("BNB") is True
        assert it.matches("XRP") is False


class TestBreadthWordBounding:
    def test_xrp_and_ripple_match_on_unscoped_sources(self):
        assert _item("Ripple wins appeal").matches("XRP") is True
        assert _item("XRP ETF filing advances").matches("XRP") is True

    def test_ripples_and_rippled_do_not_match(self):
        """Word-bounded, the P293c solution/sold/console rule."""
        assert _item("The decision ripples through markets").matches("XRP") is False
        assert _item("News rippled across the sector").matches("XRP") is False

    def test_airbnb_cannot_match_bnb(self):
        assert _item("Airbnb reports record quarter").matches("BNB") is False

    def test_bare_binance_does_not_match_bnb(self):
        """Exchange news is not coin news -- deliberately no `binance` term."""
        assert _item("Binance lists a new token").matches("BNB") is False
        assert _item("BNB burns accelerate").matches("BNB") is True

    def test_every_scoped_asset_has_a_word_pattern(self):
        """A scoped source covers its own feed; the PATTERN is what lets the
        eleven unscoped sources serve the asset too. One without the other
        is half the coverage claiming the whole."""
        for asset in set(SOURCE_ASSET_SCOPE.values()):
            assert asset in _ASSET_PATTERNS, asset

"""
[P314] `llm_sentiment` was dark on BTC and SOL, and I had recorded the cause
wrongly as "a data fact".

The measured state before this change: the four RSS sources produced 94 dated
items with a MEDIAN AGE of 20.5h, and only 9 fell inside the agent's 4h
freshness window — per asset BTC 0, ETH 1, SOL 0. So `[LLM_SENTIMENT]` read
`headlines=0 tradeable=False` on two of three assets while `[RSS] 94
headline(s) ... relevant {BTC: 23}` sat four lines above it in the same log.

I concluded that widening the window would fabricate recency (true) and that
nothing else could be done (FALSE). The window was never the problem: the
ARRIVAL RATE was. Public RSS from four publishers is a corpus, not a firehose,
and the fix is more/denser sources — which changes no definition of freshness
at all.

Probed 2026-08-19 before adding anything (P218), in-window yield per source:
    gnews_btc/eth/sol   100-102 items, 4-6 inside 4h   <- decisive
    bitcoincom          10 items, 4 inside 4h, median  5.1h
    cryptoslate         10 items, 3 inside 4h, median  7.1h
    ambcrypto           16 items, 2 inside 4h, median  7.2h
    coindesk            25 items, 2 inside 4h, median 12.7h
    newsbtc             REJECTED  median 174.9h, 0 inside 4h
    utoday              REJECTED  97 items, median 64.4h, 1 inside 4h

Measured after: BTC 0 -> 10 in-window, ETH 1 -> 4, SOL 0 -> 5.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data_mgmt.feeds.rss_news_feed import (  # noqa: E402
    REJECTED_SOURCES, RSS_SOURCES, RSSNewsItem, SOURCE_ASSET_SCOPE)


def _item(title, source="cointelegraph", age_h=1.0):
    return RSSNewsItem(
        title=title,
        published_at=datetime.now(timezone.utc) - timedelta(hours=age_h),
        source=source, link="")


# =============================================================================
# The source set
# =============================================================================

class TestSourceSet:

    def test_the_dense_sources_are_present(self):
        """Anti-vacuity: without these the arrival rate is unchanged and every
        other test here passes while the agent stays dark."""
        names = {n for n, _ in RSS_SOURCES}
        for required in ("gnews_btc", "gnews_eth", "gnews_sol"):
            assert required in names, (
                f"{required} carries 4-6 in-window items on its own; it is "
                f"what closes BTC 0 -> 10 and SOL 0 -> 5")
        assert len(RSS_SOURCES) >= 10, f"only {len(RSS_SOURCES)} sources"

    def test_every_asset_has_a_scoped_source(self):
        """A per-asset query is the only source type that guarantees an asset
        gets in-window items rather than hoping a general feed mentions it."""
        assert set(SOURCE_ASSET_SCOPE.values()) == {
            "BTC", "ETH", "SOL", "XRP", "BNB"}  # [P413] breadth joined
        names = {n for n, _ in RSS_SOURCES}
        for scoped in SOURCE_ASSET_SCOPE:
            assert scoped in names, f"{scoped} is scoped but not fetched"

    @pytest.mark.parametrize("rejected", ["newsbtc", "utoday"])
    def test_the_stale_sources_stay_out(self, rejected):
        """Both answer HTTP 200 with well-formed XML, so they LOOK healthy.
        A source whose items are days old adds corpus and no freshness — the
        exact failure this change exists to fix."""
        assert rejected not in {n for n, _ in RSS_SOURCES}
        assert rejected in REJECTED_SOURCES

    def test_each_rejection_carries_its_measurement(self):
        """A rejection without a number is an opinion someone will re-litigate."""
        for name, reason in REJECTED_SOURCES.items():
            assert any(c.isdigit() for c in reason), name
            assert "h" in reason, f"{name}: state the age that disqualified it"

    def test_no_duplicate_source_names_or_urls(self):
        names = [n for n, _ in RSS_SOURCES]
        urls = [u for _, u in RSS_SOURCES]
        assert len(names) == len(set(names))
        assert len(urls) == len(set(urls))


# =============================================================================
# Attribution — the query is a relevance claim, but not a licence
# =============================================================================

class TestAssetScoping:

    def test_a_scoped_item_serves_its_queried_asset_without_the_word(self):
        """THE POINT. A Google-News "bitcoin" result headlined "Crypto market
        rallies as ETF inflows resume" is about BTC even though the
        word-bounded matcher cannot see it. Attributing by title alone would
        discard most of what makes these feeds worth adding."""
        it = _item("Crypto market rallies as ETF inflows resume",
                   source="gnews_btc")
        assert it.matches("BTC") is True

    def test_a_scoped_item_does_not_serve_the_other_assets(self):
        """Scope is a claim about ONE asset, not a wildcard."""
        it = _item("Crypto market rallies", source="gnews_btc")
        assert it.matches("ETH") is False
        assert it.matches("SOL") is False

    def test_a_scoped_item_mentioning_another_asset_serves_both(self):
        """Title matching still runs on top — scope is additive."""
        it = _item("Bitcoin and Solana both rally", source="gnews_btc")
        assert it.matches("BTC") is True
        assert it.matches("SOL") is True

    def test_an_unscoped_source_is_unchanged(self):
        it = _item("Ethereum upgrade ships", source="cointelegraph")
        assert it.matches("ETH") is True
        assert it.matches("BTC") is False

    @pytest.mark.parametrize("title", [
        "A solution to scaling", "Shares were sold today",
        "Console makers report", "Solar power for miners"])
    def test_word_bounding_still_holds_on_unscoped_sources(self, title):
        """P293c's trap: a naive `"sol" in title` floods SOL with unrelated
        news. Adding scoped sources must not have relaxed this."""
        assert _item(title, source="cointelegraph").matches("SOL") is False

    def test_scoping_cannot_smuggle_a_word_bound_failure(self):
        """A SOL-scoped item is SOL by query — but an unscoped item is still
        judged on word boundaries, so the two mechanisms stay separable."""
        assert _item("A solution to scaling", source="gnews_sol").matches("SOL")
        assert not _item("A solution to scaling",
                         source="ambcrypto").matches("SOL")

    def test_the_asset_argument_tolerates_pair_form(self):
        assert _item("news", source="gnews_btc").matches("BTC/USD") is True


# =============================================================================
# What this does NOT claim
# =============================================================================

class TestHonestScope:

    def test_the_module_records_that_this_is_coverage_not_edge(self):
        """`llm_sentiment` measured 90d IC +0.040/+0.048, both insignificant.
        Making it speak is not evidence it should be listened to, and the file
        must not read as if it were."""
        src = (REPO / "data_mgmt" / "feeds"
               / "rss_news_feed.py").read_text(encoding="utf-8-sig")
        assert "ARRIVAL RATE" in src or "arrival rate" in src.lower(), (
            "the reason for the change must be stated where it was made")

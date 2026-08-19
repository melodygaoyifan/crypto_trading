"""
[P322] Making the CryptoPanic decision safe to act on — and the audit found a
fabrication hazard sitting on the exact path the decision would take.

THE COST PICTURE (from the operator's plan table + measured demand):
    Growth monthly  $199/mo for 3,000 requests
    Growth weekly   $50/wk (~$217/mo) for 600 req/week
    measured demand 3 requests x 6 four-hour ticks = ~18/day ~= 540/month
    => ~18% utilisation, i.e. ~$0.37 per request actually issued.
Unresolved and OPERATOR-SIDE: 540/month cannot exhaust 3,000, yet the API
answers 429. Either the plan is smaller than Growth-monthly or something
outside this engine spends the key; only the dashboard's usage counter says.

WHY LEAVING IS CHEAP (measured, not argued): with the quota exhausted and
CryptoPanic contributing ZERO fetches, `llm_sentiment` reads
`src=haiku status=live tradeable=True` on all three assets, on 20/8/6 RSS
headlines (P314). Its unique fields feed `_score_cryptopanic_metrics`, which
sits BELOW the Haiku return — a fallback reachable only when Haiku produces
nothing — and its `panic_score` never reaches market_data at all (that comes
from macro_crowd_context, P287).

THE HAZARD ON THE EXIT PATH: `mock_mode` is derived as
`not bool(CRYPTOPANIC_API_KEY)`, and `_fetch_mock` invents 3-10 items per
currency titled "Mock BTC News 0", stamped within the last 240 minutes — fresh
enough to pass the 4h window and satisfy the _c3_live gate. `is_mock` was
consulted only by the metrics path; the headline loop never looked at it. So
the obvious way to cancel — delete the key — would have sent fabricated
titles to Haiku and returned the reading labelled `status=live` (P2/P223).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# =============================================================================
# 1. Mock means NO news, never FAKE news
# =============================================================================

class TestMockHeadlinesNeverReachHaiku:

    def _meta(self, mock: bool):
        import asyncio
        from datetime import datetime, timedelta, timezone
        import agents.sentiment_llm_agent as mod

        now = datetime.now(timezone.utc)

        class _Item:
            def __init__(self, t):
                self.title = t
                self.currencies = ["BTC"]
                self.published_at = now - timedelta(minutes=30)

        class _Data:
            timestamp = now
            recent_news = [_Item("Mock BTC News 0"), _Item("Mock BTC News 1")]

        class _Feed:
            _mock_mode = mock

            def get_latest(self):
                return _Data()

            async def fetch(self):
                return _Data()

            async def fetch_if_stale(self):
                return _Data()

        mod_feed = _Feed()
        import data_mgmt.feeds.cryptopanic_feed as cp
        orig = cp.get_cryptopanic_feed
        cp.get_cryptopanic_feed = lambda *a, **k: mod_feed
        try:
            return asyncio.run(mod.fetch_headlines_with_meta("BTC"))
        finally:
            cp.get_cryptopanic_feed = orig

    def test_mock_mode_yields_no_headlines(self):
        """THE HAZARD. 'Mock BTC News 0' stamped 30 minutes ago passes the 4h
        window and would be scored by Haiku as live news."""
        meta = self._meta(mock=True)
        assert meta["headlines"] == [], (
            "fabricated titles must never reach the model — mock means NO "
            "news, not FAKE news")
        assert meta.get("reason") == "cryptopanic_mock_mode_no_headlines"

    def test_real_mode_still_returns_headlines(self):
        """The guard must not become a blanket disable — a real feed with real
        items has to keep working."""
        meta = self._meta(mock=False)
        assert meta["headlines"], "a non-mock feed must still deliver"

    def test_is_mock_is_still_reported(self):
        """Downstream gates (the metrics path) read this flag; suppressing the
        headlines must not also suppress the provenance."""
        assert self._meta(mock=True)["is_mock"] is True


# =============================================================================
# 2. The disable is a CONFIG flag, not a deleted key
# =============================================================================

class TestDisablingIsSafeAndDefaultsOff:

    def test_the_flag_is_declared_and_parsed(self):
        """P201's trio: declared on ProductionConfig AND read in from_file, or
        the key silently does nothing."""
        from main import ProductionConfig
        assert hasattr(ProductionConfig, "cryptopanic_enabled")
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        assert 'data.get("cryptopanic_enabled"' in src

    def test_the_default_is_enabled(self):
        """An absent key must be byte-identical to today — a config addition
        may not change live behaviour by itself (P201)."""
        from main import ProductionConfig
        assert ProductionConfig.cryptopanic_enabled is True

    def test_the_live_profile_does_not_set_it_yet(self):
        """Turning it off is an OPERATOR decision tied to a billing action
        (P141); adding the key IS the act."""
        import json
        cfg = json.loads((REPO / "configs" / "live_high_risk.json").read_text(
            encoding="utf-8-sig"))
        assert "cryptopanic_enabled" not in cfg

    def test_disabling_leaves_the_feed_unbuilt_rather_than_mocked(self):
        """The whole point: `mock_mode = not bool(key)`, so the intuitive exit
        (delete the key) ENABLES fabrication. The flag must skip construction
        entirely."""
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        i = src.index('getattr(self.config, "cryptopanic_enabled"')
        block = src[i:i + 500]
        assert "self.cryptopanic_feed = None" in block
        assert "get_cryptopanic_feed" in block, "the else-branch must remain"

    def test_the_reason_deleting_the_key_is_wrong_is_recorded_at_source(self):
        """A future reader will reach for the env var first; the code has to
        say why that is the wrong lever."""
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        i = src.index('getattr(self.config, "cryptopanic_enabled"')
        note = src[max(0, i - 1200):i]
        #  wraps across a comment line-break, so
        # match the contiguous half rather than a phrase the formatter split.
        assert "mock" in note.lower()
        assert "bool(cryptopanic_key)" in note


# =============================================================================
# 3. Independence, so a regression cannot make leaving expensive later
# =============================================================================

class TestSentimentDoesNotDependOnCryptoPanic:

    def test_the_metrics_path_is_a_fallback_below_haiku(self):
        """If this ever moves above the Haiku return, CryptoPanic stops being
        optional and the cost analysis is void."""
        src = (REPO / "agents" / "sentiment_llm_agent.py").read_text(
            encoding="utf-8-sig")
        i_haiku = src.index("self._cache[asset] = result")
        i_cp = src.index("_score_cryptopanic_metrics(asset, _cp_metrics")
        assert i_haiku < i_cp, (
            "the CryptoPanic metrics scorer must remain reachable only when "
            "Haiku produced nothing")

    def test_rss_is_blended_independently_of_cryptopanic(self):
        """RSS is what carries the agent today (P314). It must not be nested
        inside a CryptoPanic-success branch."""
        src = (REPO / "agents" / "sentiment_llm_agent.py").read_text(
            encoding="utf-8-sig")
        assert "RSS_BLEND" in src
        assert "rss_news_feed" in src

    def test_cryptopanic_panic_score_does_not_feed_market_data(self):
        """55 consumers read `panic_score`, but from market_data['crowd'],
        injected by macro_crowd_context (P287) — not from this feed. If that
        ever changed, leaving would cost something real."""
        import re
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        for m in re.finditer(r"cryptopanic", src, re.I):
            line_start = src.rfind("\n", 0, m.start()) + 1
            line = src[line_start:src.find("\n", m.start())]
            assert "panic_score" not in line, (
                f"CryptoPanic now writes panic_score: {line.strip()[:110]}")

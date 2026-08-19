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

    def test_the_live_profile_pins_the_DECIDED_value(self):
        """[P322b] Was "the key must be absent". The operator instructed the
        disable on 2026-08-19, so the pin flips to the DECIDED value rather
        than being deleted (the P237 pattern): now a silent REVERT to true and
        a silent removal of the key both fail, because either is a live
        behaviour change that should be argued for, not drifted into."""
        import json
        cfg = json.loads((REPO / "configs" / "live_high_risk.json").read_text(
            encoding="utf-8-sig"))
        assert cfg.get("cryptopanic_enabled") is False, (
            "the live profile must carry the decided value; re-enabling is a "
            "spend decision and needs its own record")
        assert "_cryptopanic_enabled_note" in cfg, (
            "the decision must travel with its reason and its revert")

    def test_the_default_still_protects_every_other_profile(self):
        """Disabling it LIVE must not change the dataclass default — paper and
        test profiles that never mention the key keep today's behaviour."""
        from main import ProductionConfig
        assert ProductionConfig.cryptopanic_enabled is True

    def test_disabling_leaves_the_feed_unbuilt_rather_than_mocked(self):
        """The whole point: `mock_mode = not bool(key)`, so the intuitive exit
        (delete the key) ENABLES fabrication. The flag must skip construction
        entirely."""
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        i = src.index('getattr(self.config, "cryptopanic_enabled"')
        block = src[i:i + 1400]   # widened: P322c inserted the module-switch call between the two
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


class TestTheDisableReachesTheSingletonsOtherCallers:
    """[P322c] P322 gated main.py's construction site and the feed kept
    initialising on the live box — because `agents/sentiment_llm_agent.py`
    calls `get_cryptopanic_feed()` DIRECTLY at two sites. Gating one handle
    left the actual consumers untouched: the P152/P2 shape, inside the fix
    written to close it. Caught by reading the live log after the deploy
    rather than trusting the config (P295b)."""

    def _reset(self):
        import data_mgmt.feeds.cryptopanic_feed as cp
        cp.set_feed_enabled(True)

    def test_the_switch_is_module_level_not_a_handle(self):
        import data_mgmt.feeds.cryptopanic_feed as cp
        assert hasattr(cp, "set_feed_enabled") and hasattr(cp, "is_feed_enabled")

    def test_default_is_enabled(self):
        self._reset()
        import data_mgmt.feeds.cryptopanic_feed as cp
        assert cp.is_feed_enabled() is True

    def test_a_disabled_feed_makes_no_request_and_no_mock(self, tmp_path,
                                                          monkeypatch):
        """Silent AND empty. Checked before the mock branch, or disabling
        would turn the fabricator ON for a keyless instance."""
        import asyncio
        import data_mgmt.feeds.cryptopanic_feed as cp
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        try:
            cp.set_feed_enabled(False)
            feed = cp.CryptoPanicFeed(api_key="", mock_mode=True)
            assert asyncio.run(feed.fetch()) is None, (
                "a disabled feed must return nothing — not mock data")
        finally:
            self._reset()

    def test_the_agents_direct_caller_respects_it(self):
        """The site that bypassed main.py entirely."""
        import asyncio
        import agents.sentiment_llm_agent as mod
        import data_mgmt.feeds.cryptopanic_feed as cp
        try:
            cp.set_feed_enabled(False)
            meta = asyncio.run(mod.fetch_headlines_with_meta("BTC"))
            assert meta["headlines"] == []
            assert meta.get("reason") == "cryptopanic_disabled_by_config"
        finally:
            self._reset()

    def test_main_sets_the_switch_when_the_flag_is_false(self):
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        i = src.index('getattr(self.config, "cryptopanic_enabled"')
        block = src[i:i + 900]
        assert "set_feed_enabled(False)" in block, (
            "clearing the handle is not enough — the singleton's other "
            "callers only see the module switch")

    def test_an_unreadable_switch_defaults_to_ON(self):
        """Fail direction: a broken import must not silently disable a feed
        the operator is paying for (P2 — absence is not a decision)."""
        src = (REPO / "agents" / "sentiment_llm_agent.py").read_text(
            encoding="utf-8-sig")
        i = src.index("is_feed_enabled")
        assert "_cp_on = True" in src[i:i + 400]

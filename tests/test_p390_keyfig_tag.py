"""[P390] Key-figure headline tag — EVIDENCE ONLY.

The operator asked whether key-person speech (Trump, Powell, ...) is fetched.
It is — as ordinary headlines through the three-source blend (P322d) — but
nothing tagged it, so "do key-figure headline days behave differently" was
unmeasurable. `fetch_headlines_with_meta` now writes two meta keys
(`keyfig_hits`, `keyfig_total`) computed on the FINAL blended list.

What this file pins, and why each pin exists:

  * WORD-BOUNDARY TRAPS (P293c): "sol" in a bare substring test matched
    solution/sold/console; "sec" inside "second"/"insecure"/"consecutive"
    and "trump" inside "trumpet" are the same trap. Every trap test first
    asserts the naive substring WOULD match (anti-vacuity, P174) so the
    trap is proven to discriminate.
  * EVIDENCE-ONLY (the load-bearing pin): the returned headline list is
    identical in content and order whether tagging runs or not. No
    consumer may act on the tag; it is measurement plumbing.
  * P2 SEMANTICS: `keyfig_hits == {}` is a MEASURED ZERO over a real
    blend; the KEY BEING ABSENT means the blend did not complete. The two
    must never collapse.
  * ROSTER SHAPE: every pattern is word-bounded BY CONSTRUCTION, and the
    figure set is pinned so an addition is a deliberate, reviewed change
    (offices and stable surnames only — no guessed officeholders).
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import agents.sentiment_llm_agent as mod  # noqa: E402
from agents.sentiment_llm_agent import (  # noqa: E402
    KEY_FIGURE_TERMS,
    _KEYFIG_PATTERNS,
    keyfig_hits,
)


@pytest.fixture(autouse=True)
def _no_ambient_anthropic_key(monkeypatch):
    """[P165] `api_key=""` falls back to the ambient ANTHROPIC_API_KEY and
    BILLS REAL API CALLS. Nothing in this file constructs the agent today,
    but the fixture is cheap insurance against a future test that does."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# =============================================================================
# Stubbed three-source blend (the P322 pattern: isolate every source so the
# assertions cannot pass or fail on a real network feed's contents)
# =============================================================================

def _run_blend(rss_titles, cp_titles=(), patch_keyfig=None):
    """Drive the REAL fetch_headlines_with_meta over stubbed sources.

    CryptoPanic and CC News are stubbed EMPTY by default and RSS carries the
    synthetic corpus — the same isolation the P322d tests use, because RSS is
    a real network feed and the CryptoPanic singleton may hold real cache.
    """
    import data_mgmt.feeds.cryptopanic_feed as cp
    import data_mgmt.feeds.cryptocompare_news_feed as ccn
    import data_mgmt.feeds.rss_news_feed as rss

    now = datetime.now(timezone.utc)

    class _CpItem:
        def __init__(self, t):
            self.title = t
            self.currencies = ["BTC"]
            self.published_at = now - timedelta(minutes=30)

    class _CpData:
        timestamp = now
        recent_news = [_CpItem(t) for t in cp_titles]

    class _CpFeed:
        _mock_mode = False

        def get_latest(self):
            return _CpData()

        async def fetch(self):
            return _CpData()

    class _CcFeed:
        async def fetch_headlines(self, asset, limit=25):
            return []

    class _RssItem:
        def __init__(self, t):
            self.title = t
            self.published_at = now - timedelta(minutes=5)

    class _Rss:
        async def fetch_if_stale(self):
            return None

        def get_items(self, asset):
            return [_RssItem(t) for t in rss_titles]

    orig_cp = cp.get_cryptopanic_feed
    orig_cc = ccn.get_cc_news_feed
    orig_rss = rss.get_rss_news_feed
    orig_kf = mod.keyfig_hits
    cp.get_cryptopanic_feed = lambda *a, **k: _CpFeed()
    ccn.get_cc_news_feed = lambda *a, **k: _CcFeed()
    rss.get_rss_news_feed = lambda *a, **k: _Rss()
    if patch_keyfig is not None:
        mod.keyfig_hits = patch_keyfig
    try:
        return asyncio.run(mod.fetch_headlines_with_meta("BTC"))
    finally:
        cp.get_cryptopanic_feed = orig_cp
        ccn.get_cc_news_feed = orig_cc
        rss.get_rss_news_feed = orig_rss
        mod.keyfig_hits = orig_kf


# =============================================================================
# 1. Word-boundary traps (P293c) — each trap first proves the naive
#    substring WOULD match, so the test cannot pass vacuously (P174).
# =============================================================================

class TestWordBoundaryTraps:

    def test_consecutive_does_not_hit_sec(self):
        title = "Seventh consecutive green candle for bitcoin"
        assert "sec" in title.lower()  # the naive substring test WOULD hit
        assert keyfig_hits([title]) == {}

    def test_second_does_not_hit_sec(self):
        title = "Bitcoin rallies for the second day running"
        assert "sec" in title.lower()
        assert keyfig_hits([title]) == {}

    def test_insecure_does_not_hit_sec(self):
        title = "Insecure bridge contract exploited for $40M"
        assert "sec" in title.lower()
        assert keyfig_hits([title]) == {}

    def test_trumpet_does_not_hit_trump(self):
        title = "Analysts trumpet a new era for stablecoins"
        assert "trump" in title.lower()
        assert keyfig_hits([title]) == {}

    def test_muskmelon_does_not_hit_musk(self):
        title = "Muskmelon futures are not a real market"
        assert "musk" in title.lower()
        assert keyfig_hits([title]) == {}

    @pytest.mark.parametrize("title", [
        "Powell signals patience on rates",
        "POWELL SIGNALS PATIENCE ON RATES",
        "powell signals patience on rates",
    ])
    def test_powell_hits_case_insensitive(self, title):
        assert keyfig_hits([title]) == {"powell": 1}

    def test_standalone_sec_hits(self):
        assert keyfig_hits(["SEC sues another exchange"]) == {"sec": 1}

    def test_fomc_and_federal_reserve_hit_fed(self):
        assert keyfig_hits(["FOMC minutes released today"]) == {"fed": 1}
        assert keyfig_hits(["Federal Reserve holds rates"]) == {"fed": 1}

    def test_bare_fed_is_deliberately_not_a_term(self):
        """'fed' is a common English verb ('data fed to the model') — the
        roster carries fomc/federal reserve/rate-cut terms instead. This
        absence is a decision, not a gap."""
        assert keyfig_hits(["Data fed to the model improves accuracy"]) == {}
        assert "fed" not in KEY_FIGURE_TERMS["fed"]


# =============================================================================
# 2. Count semantics
# =============================================================================

class TestCounts:

    def test_counts_are_per_headline_not_per_occurrence(self):
        assert keyfig_hits(["Trump says Trump-era tariffs were Trump's idea"]) \
            == {"trump": 1, "tariff": 1}

    def test_one_headline_can_count_for_multiple_figures(self):
        hits = keyfig_hits(["Trump pressures Powell over rate cuts"])
        assert hits == {"trump": 1, "powell": 1, "fed": 1}

    def test_counts_over_a_synthetic_blend(self):
        blend = [
            "Trump announces sweeping tariffs on imports",
            "Powell holds rates steady",
            "Bitcoin ETF inflows resume",
            "SEC delays another ETF decision",
            "Trump criticises the SEC",
            "Solana upgrade ships on schedule",
        ]
        assert keyfig_hits(blend) == {
            "trump": 2, "tariff": 1, "powell": 1, "sec": 2,
        }

    def test_zero_hit_figures_are_omitted(self):
        hits = keyfig_hits(["Powell holds rates steady"])
        assert "musk" not in hits and "trump" not in hits

    def test_non_string_items_are_skipped_not_fatal(self):
        assert keyfig_hits(["Trump speaks", None, 42, b"bytes"]) == {"trump": 1}

    def test_empty_list_is_a_measured_zero(self):
        assert keyfig_hits([]) == {}


# =============================================================================
# 3. Meta keys, end to end through the REAL blend (anti-vacuity)
# =============================================================================

class TestMetaKeysEndToEnd:

    def test_a_real_hit_is_detected_end_to_end(self):
        """ANTI-VACUITY: a genuine key-figure headline entering through a
        stubbed source must surface in the meta keys of the real blend."""
        meta = _run_blend([
            "Trump imposes new tariffs on chips",
            "Powell holds rates steady",
            "Bitcoin ETF inflows resume",
        ])
        assert meta["keyfig_hits"] == {"trump": 1, "tariff": 1, "powell": 1}
        assert meta["keyfig_total"] == 3

    def test_measured_zero_is_an_empty_dict_with_the_key_present(self):
        """[P2] No hits is `{}` with the key PRESENT — never a missing key."""
        meta = _run_blend([
            "Bitcoin ETF inflows resume",
            "Solana upgrade ships on schedule",
        ])
        assert "keyfig_hits" in meta and "keyfig_total" in meta
        assert meta["keyfig_hits"] == {}
        assert meta["keyfig_total"] == 0

    def test_absence_of_the_key_means_the_blend_did_not_complete(self):
        """[P2] The other half of the distinction: when the blend dies before
        the tag is computed, the keys are ABSENT — a failed blend must never
        read as a measured zero."""
        def _boom(headlines):
            raise RuntimeError("deliberate")

        meta = _run_blend(["Trump speaks"], patch_keyfig=_boom)
        assert "keyfig_hits" not in meta
        assert "keyfig_total" not in meta

    def test_the_tag_counts_the_final_deduped_list(self):
        """A duplicate title supplied by two sources is deduped by the blend
        and must count ONCE — the tag reads the final list, not the feeds."""
        meta = _run_blend(
            rss_titles=["Trump imposes new tariffs on chips"],
            cp_titles=["Trump imposes new tariffs on chips"],
        )
        assert meta["headlines"].count("Trump imposes new tariffs on chips") == 1
        assert meta["keyfig_hits"] == {"trump": 1, "tariff": 1}


# =============================================================================
# 4. EVIDENCE-ONLY — the pin that the tag changes nothing it measures
# =============================================================================

class TestEvidenceOnly:

    def test_headlines_are_identical_with_tagging_neutralised(self):
        """THE LOAD-BEARING PIN. The returned headline list — content AND
        order — must be byte-identical whether the tagger runs for real or
        is replaced with a no-op. Tagging that filters, reorders or rewrites
        is a consumer acting on the tag, which is forbidden."""
        corpus = [
            "Trump imposes new tariffs on chips",
            "Bitcoin ETF inflows resume",
            "Powell holds rates steady",
            "Solana upgrade ships on schedule",
        ]
        real = _run_blend(corpus)
        neutered = _run_blend(corpus, patch_keyfig=lambda hs: {})
        assert real["headlines"] == neutered["headlines"]
        assert real["headlines"], "vacuity guard: the blend must return items"

    def test_the_helper_never_mutates_its_input(self):
        lst = ["Trump speaks", "Bitcoin rallies"]
        snapshot = list(lst)
        keyfig_hits(lst)
        assert lst == snapshot

    def test_no_reader_of_the_meta_keys_exists_in_the_agent(self):
        """The agent WRITES the keys and never reads them back — the
        source_status whitelist in analyze() must not quietly grow a
        keyfig consumer (that would be plumbing beyond the meta keys)."""
        src = (REPO / "agents" / "sentiment_llm_agent.py").read_text(
            encoding="utf-8")
        assert 'meta["keyfig_hits"]' in src, "the writer must exist"
        assert 'cryptopanic_meta.get("keyfig' not in src
        analyze_src = inspect.getsource(mod.LLMSentimentAgent.analyze)
        assert "keyfig" not in analyze_src


# =============================================================================
# 5. Roster construction — word-bounded BY CONSTRUCTION, pinned membership
# =============================================================================

class TestRosterConstruction:

    def test_every_pattern_is_word_bounded_by_construction(self):
        for fig, pat in _KEYFIG_PATTERNS.items():
            assert pat.pattern.startswith(r"\b(?:"), fig
            assert pat.pattern.endswith(r")\b"), fig

    def test_every_term_is_lowercase_stripped_and_nonempty(self):
        """Matching lowercases the title, so an uppercase term could never
        match anything — a term that cannot fire is a P174 check."""
        for fig, terms in KEY_FIGURE_TERMS.items():
            assert isinstance(terms, tuple) and terms, fig
            for t in terms:
                assert isinstance(t, str) and t, (fig, t)
                assert t == t.lower(), (fig, t)
                assert t == t.strip(), (fig, t)

    def test_roster_and_patterns_agree(self):
        assert set(KEY_FIGURE_TERMS) == set(_KEYFIG_PATTERNS)

    def test_the_figure_set_is_pinned(self):
        """Offices and stable surnames only — no guessed officeholders.
        Adding a figure is a deliberate change that updates this pin
        (P237 pattern), never a drive-by edit."""
        assert set(KEY_FIGURE_TERMS) == {
            "trump", "powell", "musk", "fed", "sec",
            "whitehouse", "treasury", "tariff",
        }

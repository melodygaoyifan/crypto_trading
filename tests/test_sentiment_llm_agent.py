"""
Tests for agents/sentiment_llm_agent.py - production-readiness checks.

No network calls.  Anthropic SDK mocked via _client injection.
"""

import asyncio
import json
import time
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from agents.sentiment_llm_agent import (
    LLMSentimentAgent,
    SentimentLLMAgent,
    SentimentResult,
    SentimentLevel,
    SentimentAgentConfig,
    SentimentSignal,
    get_llm_sentiment_agent,
    get_sentiment_agent,
    reset_sentiment_agent,
    reset_llm_sentiment_agent,
    _parse_llm_json,
    _compute_combined,
    fetch_headlines,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_haiku_json(
    fact_dir=0.4, subj_dir=0.2,
    fact_conf=0.7, subj_conf=0.5,
    summary="BTC looks bullish",
):
    return json.dumps({
        "factual_direction": fact_dir,
        "subjective_direction": subj_dir,
        "factual_confidence": fact_conf,
        "subjective_confidence": subj_conf,
        "summary": summary,
    })


def _mock_anthropic_response(text: str):
    """Create a mock response object matching anthropic SDK shape."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


@pytest.fixture(autouse=True)
def _no_ambient_anthropic_key(monkeypatch):
    """[P165] `LLMSentimentAgent(api_key="")` does NOT disable Haiku.

    The constructor is `api_key or os.environ.get("ANTHROPIC_API_KEY", "")`
    (`sentiment_llm_agent.py:466`), and `""` is falsy — so every test in this
    file that constructs with `api_key=""` to mean "no Haiku, exercise the
    fallback" silently picked up the developer's ambient key instead, took the
    Haiku branch, and **issued a real billed API call**. That is why
    `test_fg_zscore_dict` asserted `source == "f&g"` and got `"haiku"`: the test
    was correct, the environment was leaking into it.

    Clearing the variable for the whole module makes the intent real. Tests that
    want Haiku pass an explicit key (`_make_agent_with_mock_client`) and are
    unaffected — and they mock the client, so nothing here touches the network.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _make_agent_with_mock_client(response_text: str = None, side_effect=None):
    """Create an LLMSentimentAgent with a mocked anthropic client."""
    agent = LLMSentimentAgent(api_key="test-key-123")
    mock_client = MagicMock()
    if side_effect is not None:
        mock_client.messages.create = AsyncMock(side_effect=side_effect)
    elif response_text is not None:
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(response_text),
        )
    else:
        mock_client.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(_make_haiku_json()),
        )
    agent._client = mock_client
    return agent


# ---------------------------------------------------------------------------
# 1) Cache staleness regression
# ---------------------------------------------------------------------------

class TestCacheStaleness:
    @pytest.mark.asyncio
    async def test_btc_not_stale_when_eth_updates(self):
        """
        Put BTC in cache with old timestamp, then call ETH (which would
        update a global _cache_ts in the old code).  Verify BTC request
        does NOT return the stale cached result.
        """
        agent = _make_agent_with_mock_client(_make_haiku_json(
            fact_dir=0.8, subj_dir=0.3, fact_conf=0.9, subj_conf=0.6,
            summary="Fresh data",
        ))

        # Seed BTC cache with a timestamp past the TTL.
        # [P165] Was a hardcoded 7200s "beyond 1h TTL". `_cache_ttl` was raised
        # to 4h by [#11] (`sentiment_llm_agent.py:477`, "per Step 18 spec"), so
        # 2h is now well INSIDE the TTL and the stale entry was correctly
        # returned — the test was pinning a retired constant. Read the TTL off
        # the agent so this cannot rot again.
        old_ts = time.time() - (agent._cache_ttl + 60)
        agent._cache["BTC"] = SentimentResult(
            asset="BTC", direction=0.1, confidence=0.1,
            source="haiku", summary="Old BTC",
            timestamp=old_ts,
        )

        # Call for ETH to "update" (in old code this would set global _cache_ts)
        eth_result = await agent.analyze("ETH", headlines=["ETH upgrade coming"])
        assert eth_result.source == "haiku"

        # Now call BTC - must NOT return stale cache
        btc_result = await agent.analyze("BTC", headlines=["BTC rally continues"])
        assert btc_result.source == "haiku"
        assert btc_result.summary == "Fresh data"  # From mock, not old cache
        assert btc_result.cache_hit is False

    @pytest.mark.asyncio
    async def test_cache_hit_within_ttl(self):
        """Cached result within TTL should return with cache_hit=True."""
        agent = _make_agent_with_mock_client()

        # First call populates cache
        r1 = await agent.analyze("BTC", headlines=["BTC news"])
        assert r1.cache_hit is False

        # Second call should hit cache
        r2 = await agent.analyze("BTC", headlines=["BTC news"])
        assert r2.cache_hit is True
        assert r2.asset == "BTC"

    @pytest.mark.asyncio
    async def test_per_asset_cache_independent(self):
        """BTC and ETH caches are independent."""
        agent = _make_agent_with_mock_client()

        await agent.analyze("BTC", headlines=["BTC news"])
        await agent.analyze("ETH", headlines=["ETH news"])

        assert "BTC" in agent._cache
        assert "ETH" in agent._cache
        assert agent._cache["BTC"].asset == "BTC"
        assert agent._cache["ETH"].asset == "ETH"


# ---------------------------------------------------------------------------
# 2) LLM JSON parsing
# ---------------------------------------------------------------------------

class TestLLMParsing:
    def test_valid_json(self):
        text = _make_haiku_json(fact_dir=0.5, subj_dir=-0.3, fact_conf=0.8, subj_conf=0.4)
        result = _parse_llm_json(text)
        assert result is not None
        assert result["factual_direction"] == 0.5
        assert result["subjective_direction"] == -0.3

    def test_json_with_code_fences(self):
        text = '```json\n' + _make_haiku_json() + '\n```'
        result = _parse_llm_json(text)
        assert result is not None

    def test_plain_code_fences(self):
        text = '```\n' + _make_haiku_json() + '\n```'
        result = _parse_llm_json(text)
        assert result is not None

    def test_clipping_out_of_range(self):
        text = json.dumps({
            "factual_direction": 5.0,
            "subjective_direction": -3.0,
            "factual_confidence": 1.5,
            "subjective_confidence": -0.5,
            "summary": "test",
        })
        result = _parse_llm_json(text)
        assert result is not None
        assert result["factual_direction"] == 1.0
        assert result["subjective_direction"] == -1.0
        assert result["factual_confidence"] == 1.0
        assert result["subjective_confidence"] == 0.0

    def test_invalid_json(self):
        assert _parse_llm_json("not json at all") is None

    def test_missing_required_key(self):
        text = json.dumps({"factual_direction": 0.5, "summary": "partial"})
        assert _parse_llm_json(text) is None

    def test_not_a_dict(self):
        assert _parse_llm_json("[1, 2, 3]") is None

    def test_empty_string(self):
        assert _parse_llm_json("") is None

    @pytest.mark.asyncio
    async def test_valid_json_produces_haiku_result(self):
        agent = _make_agent_with_mock_client(_make_haiku_json(
            fact_dir=0.6, subj_dir=0.2, fact_conf=0.8, subj_conf=0.3,
        ))
        result = await agent.analyze("BTC", headlines=["BTC up 10%"])
        assert result.source == "haiku"
        assert result.factual_direction == 0.6
        assert result.factual_confidence == 0.8
        # Combined via SOTA policy
        assert result.direction != 0.0

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back(self):
        """Invalid LLM output -> parse error -> fallback to F&G or default."""
        agent = _make_agent_with_mock_client("This is not JSON at all")
        result = await agent.analyze("BTC", headlines=["BTC news"], fg_zscore=1.5)
        # Should fall back to F&G since haiku parse failed
        assert result.source == "f&g"
        assert result.source_status.get("haiku", {}).get("error") != ""

    @pytest.mark.asyncio
    async def test_invalid_json_no_fg_returns_default(self):
        """Invalid LLM + no F&G -> default neutral."""
        agent = _make_agent_with_mock_client("broken output")
        result = await agent.analyze("BTC", headlines=["BTC news"], fg_zscore=0.0)
        assert result.source == "default"
        assert result.direction == 0.0
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# 3) Headline filtering
# ---------------------------------------------------------------------------

class TestHeadlineFiltering:
    @pytest.mark.asyncio
    async def test_time_filter_removes_old_headlines(self):
        """Only headlines within the 4H window should be included.

        [P165] This seeded ONE fresh BTC headline and asserted a 5h-old one was
        excluded. `[#12]` (`sentiment_llm_agent.py:1267-1281`) since added an
        adaptive fallback: when the 4H pass yields **fewer than 3** headlines it
        re-scans at 8H, so a single fresh item guarantees the 5h item is pulled
        back in. The test was exercising the fallback while asserting the primary
        filter — it did not describe the code as written.

        Seed three fresh headlines so the 4H window is satisfied on its own and
        the expansion never triggers, which is the case the test name means.
        `test_adaptive_window_expands_when_sparse` below covers the other branch.
        """
        now = datetime.now(timezone.utc)

        @dataclass
        class FakeItem:
            title: str
            currencies: list
            published_at: datetime

        @dataclass
        class FakeData:
            recent_news: list

        fresh_items = [
            FakeItem(f"BTC Fresh {i}", ["BTC"], now - timedelta(hours=1))
            for i in range(3)
        ]
        old_item = FakeItem("BTC Old", ["BTC"], now - timedelta(hours=5))
        eth_item = FakeItem("ETH News", ["ETH"], now - timedelta(hours=1))

        fake_data = FakeData(recent_news=[*fresh_items, old_item, eth_item])

        with patch("agents.sentiment_llm_agent.fetch_headlines") as mock_fetch:
            # Actually test the real function by patching the CryptoPanic feed
            pass  # We'll test via direct function below

        # Direct test: mock the cryptopanic feed
        mock_feed = MagicMock()
        mock_feed.get_latest.return_value = fake_data
        mock_feed.fetch = AsyncMock()

        with patch("agents.sentiment_llm_agent.fetch_headlines", wraps=fetch_headlines):
            # We need to patch the import inside the function
            with patch.dict("sys.modules", {
                "data_mgmt.feeds.cryptopanic_feed": MagicMock(
                    get_cryptopanic_feed=MagicMock(return_value=mock_feed),
                ),
            }):
                from agents.sentiment_llm_agent import fetch_headlines as fh
                headlines = await fh("BTC")

        assert "BTC Fresh 0" in headlines
        assert "BTC Old" not in headlines  # older than 4H
        assert "ETH News" not in headlines  # wrong asset

    @pytest.mark.asyncio
    async def test_adaptive_window_expands_when_sparse(self):
        """[P165] The other branch of `[#12]`: under 3 hits, re-scan at 8H.

        Pins the behaviour that made the test above misleading, and pins its
        bound — 5h is pulled in, 9h is still excluded, so "expand" cannot quietly
        become "no time filter at all".
        """
        now = datetime.now(timezone.utc)

        @dataclass
        class FakeItem:
            title: str
            currencies: list
            published_at: datetime

        @dataclass
        class FakeData:
            recent_news: list

        fake_data = FakeData(recent_news=[
            FakeItem("BTC Fresh", ["BTC"], now - timedelta(hours=1)),
            FakeItem("BTC Five Hours", ["BTC"], now - timedelta(hours=5)),
            FakeItem("BTC Nine Hours", ["BTC"], now - timedelta(hours=9)),
        ])

        mock_feed = MagicMock()
        mock_feed.get_latest.return_value = fake_data
        mock_feed.fetch = AsyncMock()

        with patch.dict("sys.modules", {
            "data_mgmt.feeds.cryptopanic_feed": MagicMock(
                get_cryptopanic_feed=MagicMock(return_value=mock_feed),
            ),
        }):
            from agents.sentiment_llm_agent import fetch_headlines as fh
            headlines = await fh("BTC")

        assert "BTC Fresh" in headlines
        assert "BTC Five Hours" in headlines      # 4H -> 8H expansion
        assert "BTC Nine Hours" not in headlines  # beyond the expanded window

    @pytest.mark.asyncio
    async def test_dedup_headlines(self):
        """Duplicate titles should be deduplicated."""
        now = datetime.now(timezone.utc)

        @dataclass
        class FakeItem:
            title: str
            currencies: list
            published_at: datetime

        @dataclass
        class FakeData:
            recent_news: list

        items = [
            FakeItem("BTC Rally!", ["BTC"], now - timedelta(minutes=30)),
            FakeItem("BTC Rally!", ["BTC"], now - timedelta(minutes=45)),
            FakeItem("btc rally!", ["BTC"], now - timedelta(minutes=60)),  # case-insensitive dup
        ]
        fake_data = FakeData(recent_news=items)

        mock_feed = MagicMock()
        mock_feed.get_latest.return_value = fake_data
        mock_feed.fetch = AsyncMock()

        with patch.dict("sys.modules", {
            "data_mgmt.feeds.cryptopanic_feed": MagicMock(
                get_cryptopanic_feed=MagicMock(return_value=mock_feed),
            ),
        }):
            from agents.sentiment_llm_agent import fetch_headlines as fh
            headlines = await fh("BTC")

        assert len(headlines) == 1  # All three are same title (case-insensitive)


# ---------------------------------------------------------------------------
# 4) analyze_many timeout
# ---------------------------------------------------------------------------

class TestAnalyzeMany:
    @pytest.mark.asyncio
    async def test_timeout_returns_default(self):
        """If one asset call exceeds max_total_sec, it gets a default result."""
        agent = LLMSentimentAgent(api_key="")  # No API key -> will try F&G/default

        # Patch analyze to sleep forever for SOL, return instantly for BTC
        original_analyze = agent.analyze

        async def slow_analyze(asset, headlines=None, fg_zscore=0.0):
            if asset == "SOL":
                await asyncio.sleep(100)  # Will be cancelled by timeout
            return await original_analyze(asset, headlines=headlines, fg_zscore=fg_zscore)

        agent.analyze = slow_analyze

        results = await agent.analyze_many(
            assets=["BTC", "SOL"],
            fg_zscore=0.0,
            max_total_sec=0.5,
        )

        assert "BTC" in results
        assert "SOL" in results
        assert results["SOL"].source == "default"
        assert "timeout" in results["SOL"].summary

    @pytest.mark.asyncio
    async def test_all_assets_returned(self):
        """All requested assets must appear in the result dict."""
        agent = LLMSentimentAgent(api_key="")
        results = await agent.analyze_many(
            assets=["BTC", "ETH", "SOL"],
            fg_zscore=0.0,
            max_total_sec=5.0,
        )
        assert set(results.keys()) == {"BTC", "ETH", "SOL"}

    @pytest.mark.asyncio
    async def test_fg_zscore_dict(self):
        """Per-asset fg_zscore dict is supported."""
        agent = LLMSentimentAgent(api_key="")
        results = await agent.analyze_many(
            assets=["BTC", "ETH"],
            fg_zscore={"BTC": 2.0, "ETH": -1.5},
            max_total_sec=5.0,
        )
        # BTC should use f&g (zscore=2.0 > dead zone)
        assert results["BTC"].source == "f&g"
        assert results["BTC"].direction > 0
        # ETH should use f&g (zscore=-1.5)
        assert results["ETH"].source == "f&g"
        assert results["ETH"].direction < 0


# ---------------------------------------------------------------------------
# 5) Fail-closed
# ---------------------------------------------------------------------------

class TestFailClosed:
    @pytest.mark.asyncio
    async def test_no_api_no_headlines_dead_zone(self):
        """
        No API key + no headlines + fg_zscore in dead zone (|z|<0.5)
        => returns neutral with source="default" and confidence=0.
        """
        agent = LLMSentimentAgent(api_key="")  # No API key
        result = await agent.analyze("BTC", headlines=[], fg_zscore=0.3)

        assert result.source == "default"
        assert result.direction == 0.0
        assert result.confidence == 0.0
        assert result.coverage.get("factual") == "none"
        assert result.coverage.get("subjective") == "none"

    @pytest.mark.asyncio
    async def test_no_api_no_headlines_no_fg(self):
        """Completely empty state -> neutral."""
        agent = LLMSentimentAgent(api_key="")
        result = await agent.analyze("SOL")
        assert result.source == "default"
        assert result.direction == 0.0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_api_error_falls_back_to_fg(self):
        """API error -> fall back to F&G if available."""
        agent = _make_agent_with_mock_client(
            side_effect=RuntimeError("connection refused"),
        )
        result = await agent.analyze("BTC", headlines=["test"], fg_zscore=2.0)
        assert result.source == "f&g"

    @pytest.mark.asyncio
    async def test_api_error_no_fg_returns_default(self):
        """API error + no F&G -> default neutral."""
        agent = _make_agent_with_mock_client(
            side_effect=RuntimeError("connection refused"),
        )
        result = await agent.analyze("BTC", headlines=["test"], fg_zscore=0.0)
        assert result.source == "default"
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Observability fields
# ---------------------------------------------------------------------------

class TestObservability:
    @pytest.mark.asyncio
    async def test_to_dict_has_all_fields(self):
        """to_dict must include timestamp, data_age, cache_hit, source_status, coverage."""
        agent = _make_agent_with_mock_client()
        result = await agent.analyze("BTC", headlines=["BTC rally"])
        d = result.to_dict()

        assert "timestamp" in d
        assert "data_age_seconds" in d
        assert "cache_hit" in d
        assert "source_status" in d
        assert "coverage" in d
        assert d["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_haiku_source_status(self):
        """Haiku result should include structured source_status."""
        agent = _make_agent_with_mock_client()
        result = await agent.analyze("BTC", headlines=["BTC news"])

        ss = result.source_status
        assert "haiku" in ss
        assert ss["haiku"]["used"] is True
        assert ss["haiku"]["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_fg_fallback_coverage_labeled(self):
        """F&G fallback should be labeled as inferred/global in coverage."""
        agent = LLMSentimentAgent(api_key="")  # No Haiku
        result = await agent.analyze("BTC", fg_zscore=2.0)

        assert result.source == "f&g"
        assert "inferred" in result.coverage.get("factual", "")
        assert "inferred" in result.coverage.get("subjective", "")

    @pytest.mark.asyncio
    async def test_fg_factual_subjective_split(self):
        """F&G fallback should populate factual/subjective with reduced confidence."""
        agent = LLMSentimentAgent(api_key="")
        result = await agent.analyze("ETH", fg_zscore=2.5)

        # [P165] Was `factual_confidence < subjective_confidence`, from the era
        # when the split was 0.3 / 0.7. [FIX-AG9] (`sentiment_llm_agent.py:899`)
        # deliberately made it an even 0.5 / 0.5 — "F&G is ~50% factual
        # (market-wide metric), ~50% subjective (crowd behavior)" — so the old
        # assertion now contradicts the design. What the test name is actually
        # about is that both halves are populated and discounted, so pin that.
        assert result.factual_confidence > 0
        assert result.subjective_confidence > 0
        assert result.factual_confidence == result.subjective_confidence
        # Reduced: a global, inferred proxy must never reach full confidence,
        # even at a saturating zscore (2.5 clamps |z|/2.5 to 1.0).
        assert result.confidence < 1.0
        # Summary should mention "global" or "inferred"
        assert "global" in result.summary.lower() or "inferred" in result.summary.lower()


# ---------------------------------------------------------------------------
# SOTA combined policy
# ---------------------------------------------------------------------------

class TestCombinedPolicy:
    def test_zero_confidence_zero_direction(self):
        d, c = _compute_combined(0.5, -0.3, 0.0, 0.0)
        assert d == 0.0
        assert c == 0.0

    def test_weighted_average(self):
        d, c = _compute_combined(1.0, -1.0, 0.5, 0.5)
        # Equal weights -> direction = 0
        assert d == 0.0

    def test_purely_subjective_penalty(self):
        """When factual_conf < 0.1 and subjective is high, confidence penalized."""
        # Same max(conf) = 0.8, but one has factual support, the other doesn't
        _, conf_with_factual = _compute_combined(0.5, 0.5, 0.8, 0.8)
        _, conf_purely_subj = _compute_combined(0.5, 0.5, 0.05, 0.8)
        assert conf_purely_subj < conf_with_factual

    def test_clipping(self):
        d, c = _compute_combined(1.0, 1.0, 1.0, 1.0)
        assert -1.0 <= d <= 1.0
        assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_alias_classes(self):
        assert SentimentLLMAgent is LLMSentimentAgent

    def test_stub_types_exist(self):
        assert SentimentLevel is not None
        assert SentimentAgentConfig is not None
        assert SentimentSignal is not None
        assert SentimentLevel.NEUTRAL == "neutral"

    def test_singleton_aliases(self):
        assert get_sentiment_agent is get_llm_sentiment_agent

    def test_reset(self):
        reset_sentiment_agent()
        a = get_sentiment_agent()
        assert isinstance(a, LLMSentimentAgent)
        reset_sentiment_agent()

    def test_from_agents_init(self):
        from agents import SentimentLLMAgent as A
        from agents import get_sentiment_agent as B
        assert A is not None
        assert B is not None

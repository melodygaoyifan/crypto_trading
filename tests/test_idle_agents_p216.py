"""[P216] The idle-agent investigation, and the three things in it that were code.

Live measurement (21 consecutive ticks x 3 assets / 18h of `[AGENT-TRACE]`) found
12 of 18 agents emitting zero direction. Investigating each one individually, the
causes were mostly NOT wiring bugs — the agent layer is starved, not broken. Only
three items were actually fixable in code, and they are what this file pins:

  1. CC NEWS 429 STORM (the one with real consequences). On a 429 the feed
     returned `[]` without recording a backoff, and `Retry-After` was parsed
     straight into the log message and discarded — computed-but-unenforced, the
     P144 shape. The 5-min cache is written only on SUCCESS, so a rate-limited
     feed retried at full rate: 3 assets x every tick, forever, plus a fresh
     start on every restart because the state was in RAM (P154's lesson, fixed
     for CryptoPanic and never applied here). Consequence: `headline_count=0` ->
     `_c3_live` False -> `main.py:8529` deliberately zeroes
     `llm_sentiment_direction` ("fallback sentiment is observability only, not
     tradeable L3 alpha"). So an ADVISE agent was dark because of our own
     request pattern, with a valid 64-char API key.

  2. VOL_ALPHA EXTRACTOR MISMATCH (measurement only). `_extract_vol_alpha` reads
     `vol_alpha_implied_direction` / `vol_alpha_intensity`; `_attr_collected`
     passed `vol_alpha_direction` / `vol_alpha_bias`. The two key sets DID NOT
     INTERSECT, so attribution read 0.0 whatever the agent produced — a P3
     measurement bug independent of vol_alpha being directionally dead by design.

  3. SHORT_BIAS REGIME GATE (policy, made expressible — NOT changed). The skip
     set was a code literal covering ~93% of live ticks, so the agent was
     switched off almost always and no config recorded that as a choice.

Everything else was starvation (no headlines, no options data, no on-chain flow,
near-zero funding) or correct behaviour, and is deliberately not "fixed" here.
"""

import json
import os
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_FEED = _REPO / "data_mgmt" / "feeds" / "cryptocompare_news_feed.py"
_FSRC = _FEED.read_text(encoding="utf-8", errors="replace")
_MSRC = (_REPO / "main.py").read_text(encoding="utf-8-sig", errors="replace")


# ---------------------------------------------------------------------------
# 1. CC News backoff
# ---------------------------------------------------------------------------

class TestCCNewsHonoursItsRateLimit:

    def _feed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("CRYPTOCOMPARE_API_KEY", "k" * 64)
        from data_mgmt.feeds.cryptocompare_news_feed import CCNewsFeed
        return CCNewsFeed()

    def test_retry_after_is_acted_on_not_just_logged(self):
        """It was parsed into the warning string and thrown away."""
        i = _FSRC.index("if resp.status == 429:")
        w = _FSRC[i:i + 1200]
        assert "self._backoff_until = time.time() + _wait" in w, (
            "429 does not set a backoff — the next asset re-hits the API"
        )
        assert "self._persist_state()" in w

    def test_a_missing_retry_after_still_backs_off(self, tmp_path, monkeypatch):
        """The live 429s carry `Retry-After=None`, so a fallback is required or
        the fix does nothing in the exact case that produced it."""
        f = self._feed(tmp_path, monkeypatch)
        assert f._DEFAULT_BACKOFF_SEC > 0

    def test_backoff_survives_a_restart(self, tmp_path, monkeypatch):
        """An in-RAM rate limit is not a rate limit: it re-arms on every restart,
        and restart-heavy failure modes are when it matters most (P154)."""
        f = self._feed(tmp_path, monkeypatch)
        f._backoff_until = time.time() + 600
        f._persist_state()
        g = self._feed(tmp_path, monkeypatch)
        assert g.backoff_remaining_sec() > 500

    def test_an_expired_backoff_does_not_block(self, tmp_path, monkeypatch):
        """A persisted backoff must not wedge the feed shut forever."""
        f = self._feed(tmp_path, monkeypatch)
        f._backoff_until = time.time() - 10
        f._persist_state()
        g = self._feed(tmp_path, monkeypatch)
        assert g.backoff_remaining_sec() == 0

    @pytest.mark.asyncio
    async def test_while_backed_off_no_request_is_made(self, tmp_path, monkeypatch):
        f = self._feed(tmp_path, monkeypatch)
        f._backoff_until = time.time() + 600

        def _boom(*a, **k):
            raise AssertionError("called the API while backed off")

        monkeypatch.setattr(
            "data_mgmt.feeds.cryptocompare_news_feed.create_session", _boom)
        assert await f.fetch_headlines("BTC") == []

    @pytest.mark.asyncio
    async def test_cached_headlines_are_served_while_backed_off(self, tmp_path, monkeypatch):
        """Serving `[]` makes llm_sentiment fall back to f&g, which main.py
        deliberately treats as untradeable — so a backoff would silently keep
        the agent dark. Stale headlines beat no headlines."""
        f = self._feed(tmp_path, monkeypatch)
        f._cache["BTC"] = (time.time() - 10_000, ["stale-item"])
        f._backoff_until = time.time() + 600
        monkeypatch.setattr(
            "data_mgmt.feeds.cryptocompare_news_feed.create_session",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("no API call")))
        assert await f.fetch_headlines("BTC") == ["stale-item"]

    def test_corrupt_state_does_not_break_startup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("CRYPTOCOMPARE_API_KEY", "k" * 64)
        (tmp_path / "ccnews_state.json").write_text("{not json", encoding="utf-8")
        from data_mgmt.feeds.cryptocompare_news_feed import CCNewsFeed
        assert CCNewsFeed().backoff_remaining_sec() == 0

    def test_version_mismatch_discards(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("CRYPTOCOMPARE_API_KEY", "k" * 64)
        (tmp_path / "ccnews_state.json").write_text(
            json.dumps({"state_version": "old", "backoff_until": time.time() + 999}),
            encoding="utf-8")
        from data_mgmt.feeds.cryptocompare_news_feed import CCNewsFeed
        assert CCNewsFeed().backoff_remaining_sec() == 0

    def test_mock_mode_writes_no_state(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
        monkeypatch.delenv("CRYPTOCOMPARE_API_KEY", raising=False)
        from data_mgmt.feeds.cryptocompare_news_feed import CCNewsFeed
        CCNewsFeed()._persist_state()
        assert not (tmp_path / "ccnews_state.json").exists()


# ---------------------------------------------------------------------------
# 2. vol_alpha extractor
# ---------------------------------------------------------------------------

class TestVolAlphaAttributionCanSeeTheAgent:

    def test_collected_keys_intersect_extracted_keys(self):
        """The actual invariant: the dict handed to the extractor must contain
        the keys it reads. They did not intersect at all."""
        import re
        src = (_REPO / "agents" / "signal_envelope.py").read_text(encoding="utf-8")
        body = src[src.index("def _extract_vol_alpha"):src.index("def _extract_micro")]
        reads = set(re.findall(r'raw\.get\(\s*["\']([^"\']+)', body))

        i = _MSRC.index('"vol_alpha": {k: agent_signals.get(k, 0.0) for k in')
        collected = set(re.findall(r'"(vol_alpha_[a-z_]+)"', _MSRC[i:i + 500]))
        assert reads & collected, (
            f"attribution passes {sorted(collected)} but the extractor reads "
            f"{sorted(reads)} — vol_alpha reads 0.0 whatever the agent emits"
        )
        assert "vol_alpha_implied_direction" in collected

    def test_the_agent_really_emits_that_key(self):
        """Otherwise this fix just moves the mismatch."""
        src = (_REPO / "agents" / "volatility_alpha_agent.py").read_text(
            encoding="utf-8", errors="replace")
        assert '"vol_alpha_implied_direction"' in src

    def test_extraction_round_trip(self):
        from agents.signal_envelope import wrap_agent_signal
        env = wrap_agent_signal(
            "vol_alpha", "ADVISE",
            {"vol_alpha_implied_direction": -0.3, "vol_alpha_intensity": 0.5,
             "vol_alpha_direction_reason": "vol_expand+BEAR"},
            "BTC")
        assert env.direction == pytest.approx(-0.3)
        assert env.confidence == pytest.approx(0.5)

    def test_the_old_shape_would_have_read_zero(self):
        """Pins that this was a real defect, not a cosmetic rename."""
        from agents.signal_envelope import wrap_agent_signal
        env = wrap_agent_signal(
            "vol_alpha", "ADVISE",
            {"vol_alpha_direction": -0.3, "vol_alpha_bias": "bear"}, "BTC")
        assert env.direction == 0.0


# ---------------------------------------------------------------------------
# 3. short_bias regime gate — expressible, unchanged
# ---------------------------------------------------------------------------

class TestShortBiasSkipRegimesIsConfigurable:

    def test_the_field_is_declared_and_parsed(self):
        """P201: two flags read via getattr and never parsed were inert, so the
        config documenting them was a no-op. Both halves are required."""
        import dataclasses
        from main import ProductionConfig
        names = {f.name for f in dataclasses.fields(ProductionConfig)}
        assert "short_bias_skip_regimes" in names
        assert '"short_bias_skip_regimes"' in _MSRC or \
               'data["short_bias_skip_regimes"]' in _MSRC

    def test_the_default_preserves_todays_behaviour(self):
        """This change must not enable a short-signal agent by accident."""
        import dataclasses
        from main import ProductionConfig
        d = {f.name: f.default for f in dataclasses.fields(ProductionConfig)}
        assert d["short_bias_skip_regimes"] is None
        i = _MSRC.index("_SHORT_BIAS_SKIP_REGIMES = set(getattr(")
        w = _MSRC[i:i + 400]
        for r in ("QUIET_ACCUMULATION", "WEAK_CONSOLIDATION", "NEUTRAL_DRIFT"):
            assert r in w, f"{r} dropped from the fallback — behaviour changed"

    def test_absent_key_is_not_the_same_as_empty_list(self):
        """`[]` means "never skip" (agent always runs) and must be honoured;
        a missing key must fall back to the historical set. Collapsing them
        would silently enable the agent everywhere."""
        i = _MSRC.index("short_bias_skip_regimes=(")
        w = _MSRC[i:i + 320]
        assert "isinstance(data.get(\"short_bias_skip_regimes\"), list)" in w

    def test_the_live_profile_has_not_been_changed(self):
        cfg = json.loads((_REPO / "configs" / "live_high_risk.json").read_text(
            encoding="utf-8"))
        assert "short_bias_skip_regimes" not in cfg, (
            "enabling short_bias in the dominant regimes is a live behaviour "
            "change and an operator decision (P141)"
        )

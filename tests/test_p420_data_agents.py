"""[P420] Agent-layer fixes — fork-3 tasks 12-14.

 12. A Haiku account cap arriving on HTTP 429 got the 30s transient cooldown
     because the caller tested `status_code == 429` BEFORE the
     `usage_limit_` reason; the dated P345/P355 cooldown must win.
 13. The options agent's `_pcr_history` was RAM-only (5-sample z floor ->
     PCR z ~ 0 for ~20h after every restart); persisted via the shared
     warmup helper exactly like P371 did for micro.
 14. model_alpha `_get_model` fell back to ANOTHER asset's checkpoint; an
     asset without its own model is absence (P2).
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from collections import deque

import pytest


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("COINGLASS_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# 12 — Haiku quota cap on 429
# ---------------------------------------------------------------------------
class TestHaikuQuotaOn429:
    def _agent(self, monkeypatch, reason):
        from agents import sentiment_llm_agent as m
        a = m.SentimentLLMAgent(api_key="k")

        class _Boom:
            class messages:
                @staticmethod
                async def create(**kw):
                    raise RuntimeError("429 you have reached your usage limit")
        monkeypatch.setattr(a, "_get_client", lambda: _Boom())
        monkeypatch.setattr(m.SentimentLLMAgent, "_classify_haiku_error",
                            staticmethod(lambda e: (429, True, reason)))
        seen = {}

        def _open(reason, cooldown_sec=None):
            seen["reason"], seen["cooldown"] = reason, cooldown_sec
        monkeypatch.setattr(a, "_open_hard_disable", _open)
        return a, seen

    def test_usage_limit_on_429_takes_the_dated_cooldown(self, monkeypatch):
        a, seen = self._agent(monkeypatch, "usage_limit_429_cooldown=86400")
        out = asyncio.run(a._call_haiku("BTC", ["h1", "h2"]))
        assert out is None
        assert seen["reason"].startswith("usage_limit_")
        assert seen["cooldown"] == 86400.0, (
            f"a quota cap on 429 got the transient cooldown: {seen}")

    def test_a_plain_429_still_takes_the_short_retry_after_cooldown(
            self, monkeypatch):
        a, seen = self._agent(monkeypatch, "http_429_retry_after=12")
        asyncio.run(a._call_haiku("BTC", ["h1"]))
        assert seen["cooldown"] == 12.0

    def test_source_order_reason_before_status(self):
        from agents import sentiment_llm_agent as m
        src = inspect.getsource(m.SentimentLLMAgent._call_haiku)
        # comment-stripped: the P420 comment names the old order (P177)
        code = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("#"))
        assert code.index('reason.startswith("usage_limit_")') < \
            code.index("status_code == 429")


# ---------------------------------------------------------------------------
# 13 — options PCR history persistence
# ---------------------------------------------------------------------------
class TestOptionsPcrPersistence:
    def test_round_trip_across_a_restart(self, tmp_path):
        from agents.options_sentiment_agent import OptionsSentimentAgent
        a = OptionsSentimentAgent(api_key="k")
        for v in (0.6, 0.7, 0.65, 0.8, 0.75):
            a._pcr_history["BTC"].append(v)
        a._persist_pcr_history()
        assert (tmp_path / "v5_1_warmup" / "options_pcr_history.json").exists()
        b = OptionsSentimentAgent(api_key="k")
        assert list(b._pcr_history["BTC"]) == [0.6, 0.7, 0.65, 0.8, 0.75]
        assert b._pcr_history["BTC"].maxlen == 42

    def test_restore_only_prefills_never_double_appends(self, tmp_path):
        from agents.options_sentiment_agent import OptionsSentimentAgent
        a = OptionsSentimentAgent(api_key="k")
        a._pcr_history["ETH"].append(0.5)
        a._persist_pcr_history()
        b = OptionsSentimentAgent(api_key="k")
        assert len(b._pcr_history["ETH"]) == 1

    def test_missing_file_is_a_logged_cold_start(self, caplog):
        from agents.options_sentiment_agent import OptionsSentimentAgent
        with caplog.at_level(logging.INFO):
            a = OptionsSentimentAgent(api_key="k")
        assert len(a._pcr_history["BTC"]) == 0
        assert any("cold start" in r.getMessage() for r in caplog.records)

    def test_both_append_sites_persist(self):
        from agents import options_sentiment_agent as m
        src = inspect.getsource(m)
        i = src.index("self._pcr_history[asset].append(pcr_oi)")
        assert "_persist_pcr_history()" in src[i:i + 120]
        j = src.index("self._pcr_history[asset].append(pcr)")
        assert "_persist_pcr_history()" in src[j:j + 120]

    def test_bad_values_never_enter_a_restored_deque(self, tmp_path):
        from strategies._warmup_state import save
        from agents.options_sentiment_agent import OptionsSentimentAgent
        save("options_pcr_history", {"BTC": [0.9, -1.0, 0.0, 1.1]})
        a = OptionsSentimentAgent(api_key="k")
        assert list(a._pcr_history["BTC"]) == [0.9, 1.1]


# ---------------------------------------------------------------------------
# 14 — model_alpha per-asset only
# ---------------------------------------------------------------------------
class TestModelAlphaNoCrossAssetFallback:
    def _agent(self):
        from agents.model_alpha_agent import ModelAlphaAgent
        a = ModelAlphaAgent.__new__(ModelAlphaAgent)
        m = object()
        a._models_by_asset = {"BTC": m}
        a._model = m                       # the old fallback target
        a._model_loaded = True
        a.assets = ["BTC", "ETH", "SOL"]
        return a, m

    def test_an_asset_without_its_own_model_is_absent(self):
        a, m = self._agent()
        assert a._get_model("BTC") is m
        assert a._get_model("SOL") is None, (
            "SOL was handed BTC's checkpoint — cross-asset model application")

    def test_the_fallback_is_gone_from_the_source(self):
        from agents.model_alpha_agent import ModelAlphaAgent
        src = inspect.getsource(ModelAlphaAgent._get_model)
        code = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("#") and '"""' not in l)
        assert "or self._model" not in code

    def test_generate_intent_names_the_absence(self):
        from agents.model_alpha_agent import ModelAlphaAgent
        a, _ = self._agent()
        assert a.generate_intent(object(), "SOL") is None
        assert a._last_gating_reason == "no_model_for_asset"
        assert a.generate_intent(object(), "DOGE") is None
        assert a._last_gating_reason == "unknown_asset"

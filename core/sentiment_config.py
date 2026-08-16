"""
core/sentiment_config.py — live home for the sentiment-mode contract
====================================================================

[P165 2026-08-04] These four symbols (`SentimentConfig`, `get_sentiment_config`,
`sentiment_can_trigger_opportunity`, `is_sentiment_mock`) used to live in
`engine/compute/vllm_inference_wrapper.py`. That module was archived to
`archive/engine/compute/`, but `core/canonical_imports.py:229` kept importing
it from the old path — so `core.canonical_imports` has been UNIMPORTABLE for
the entire history of this repo.

The consequence was not cosmetic. `main.py:19861` wraps its whole
runtime-protection block in `except ImportError: logger.warning(...)`, so:

  * `activate_runtime_mode()` has NEVER run in production — the canonical-import
    enforcement it turns on was silently off in every process, and
  * the `[STARTUP] SENTIMENT_MODE=` line an operator greps for was never emitted.

Same family as P152/P155d: a guard that exists, is tested in isolation, and is
never actually reached. Here the swallow was one `except ImportError` five
lines below the only thing that could raise it.

**Semantics changed deliberately.** The archived `IS_MOCK` was
`not (VLLM_AVAILABLE or TRANSFORMERS_AVAILABLE)` — it described a local-vLLM
inference architecture this system no longer has. Live LLM sentiment is
Anthropic Haiku via `agents/sentiment_llm_agent.py`, which is gated on
`ANTHROPIC_API_KEY` (`sentiment_llm_agent.py:466,503`). Carrying the vLLM
definition forward would have reported MOCK on every production container that
has a perfectly working Haiku path. `IS_MOCK` now means what it says: no live
sentiment backend is reachable.

Note this is a STARTUP-time reading. Per-tick mock-ness is a property of the
feed, not of the process (`sentiment_llm_agent.py:1318` reads the CryptoPanic
feed's `_mock_mode`); this config does not and should not try to track that.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SentimentMode(Enum):
    """Unified sentiment mode enum — single source of truth."""

    REAL = "REAL"           # A live sentiment backend is reachable
    MOCK = "MOCK"           # No backend; synthetic/fallback sentiment only
    DISABLED = "DISABLED"   # Sentiment switched off entirely


def _live_backend_available() -> bool:
    """True when a real sentiment backend is configured.

    Today that is exactly one thing: an Anthropic key for the Haiku path in
    `SentimentLLMAgent`. Without it that agent logs "Haiku disabled, F&G
    fallback only" and the LLM sentiment signal degrades to the deterministic
    engine — which is precisely the state MOCK is meant to describe.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


@dataclass
class SentimentConfig:
    """Sentiment configuration with explicit mock-mode control.

    Invariant preserved from the archived module: **mock sentiment must not be
    able to trigger OPPORTUNITY mode.** Widening the system's risk posture off
    a synthetic signal is the failure this flag exists to prevent.
    """

    ENABLE_SENTIMENT: bool = True
    # [P287] Default is None-resolve-at-construction, NOT False. The old
    # `IS_MOCK: bool = False` made the __post_init__ resolution unreachable,
    # so a bare SentimentConfig() claimed REAL regardless of the API key —
    # the unsafe direction (mock must never look real; REAL-looking mock
    # could trigger OPPORTUNITY). An explicitly-passed IS_MOCK still wins.
    IS_MOCK: Optional[bool] = None
    DISABLE_OPPORTUNITY_TRIGGER_IN_MOCK: bool = True

    def __post_init__(self) -> None:
        # Resolved at construction, not at import, so tests and callers can
        # set the environment first. An explicitly-passed IS_MOCK still wins.
        if self.IS_MOCK is None:
            self.IS_MOCK = not _live_backend_available()

    def can_trigger_opportunity(self) -> bool:
        if not self.ENABLE_SENTIMENT:
            return False
        if self.IS_MOCK and self.DISABLE_OPPORTUNITY_TRIGGER_IN_MOCK:
            return False
        return True

    def get_status_string(self) -> str:
        if not self.ENABLE_SENTIMENT:
            return "DISABLED"
        if self.IS_MOCK:
            return "MOCK (triggers disabled)"
        return "REAL"

    @property
    def mode(self) -> SentimentMode:
        if not self.ENABLE_SENTIMENT:
            return SentimentMode.DISABLED
        return SentimentMode.MOCK if self.IS_MOCK else SentimentMode.REAL


_sentiment_config: Optional[SentimentConfig] = None


def get_sentiment_config() -> SentimentConfig:
    """Get (and on first call, build + announce) the process sentiment config."""
    global _sentiment_config
    if _sentiment_config is None:
        _sentiment_config = SentimentConfig(IS_MOCK=not _live_backend_available())
        # WARNING, not INFO: this is the line an operator greps to answer
        # "is sentiment real right now?", and it must survive production
        # log levels. Format kept byte-compatible with the archived module
        # so any existing log parsing keeps working.
        logger.warning(
            f"[STARTUP] SENTIMENT_MODE={_sentiment_config.mode.value} "
            f"CAN_TRIGGER_OPPORTUNITY={_sentiment_config.can_trigger_opportunity()} "
            f"IS_MOCK={_sentiment_config.IS_MOCK}"
        )
    return _sentiment_config


def get_sentiment_mode() -> SentimentMode:
    return get_sentiment_config().mode


def set_sentiment_config(config: SentimentConfig) -> None:
    global _sentiment_config
    _sentiment_config = config
    logger.info(
        f"[Sentiment] Config updated: status={config.get_status_string()}, "
        f"can_trigger_opportunity={config.can_trigger_opportunity()}"
    )


def reset_sentiment_config() -> None:
    """Drop the cached config (tests; environment changes)."""
    global _sentiment_config
    _sentiment_config = None


def sentiment_can_trigger_opportunity() -> bool:
    return get_sentiment_config().can_trigger_opportunity()


def is_sentiment_mock() -> bool:
    return get_sentiment_config().IS_MOCK

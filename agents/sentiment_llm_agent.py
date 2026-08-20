"""
================================================================================
HMATS v6.8 - LLM Sentiment Agent (Production-Grade)
================================================================================
Per-asset sentiment scoring via Anthropic Haiku on CryptoPanic headlines.
Replaces the single-market F&G zscore with per-asset LLM-derived direction.

Authority: ADVISE - outputs sentiment telemetry only.  Never places orders,
vetoes trades, or alters exposure directly.

Fallback chain (never blocks tick):
  1. CryptoPanic + Haiku -> per-asset direction + confidence
  2. F&G zscore (Layer 1) -> global direction (already in market_data)
  3. Hardcoded 0.0 ("no signal" - fusion weight -> 0)

Cost estimate: ~$0.34/month (45K tokens/day × 30 days at Haiku pricing).

CRITICAL: Sentiment NEVER vetoes or blocks a tick. Every code path has
bare except -> return neutral. Return 0.0 = "no signal" = fusion weight 0.

Production fixes (v6.8):
- Cache keyed per-asset (not global _cache_ts that aliases across assets)
- Headline time filtering (last 4H window) + dedup
- Structured source_status / coverage / data_age_seconds in output
- Robust _parse_llm_json with schema validation, fence stripping, clipping
- F&G fallback explicitly labeled as inferred/global, confidence degraded
- SOTA combined confidence: purely-subjective signal penalized
- analyze_many() with overall timeout for multi-asset calls
- LLM call: temperature=0, asyncio.Semaphore concurrency guard
================================================================================
"""

import logging
import os
import json
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SentimentResult:
    """Per-asset sentiment from LLM analysis."""
    asset: str
    direction: float = 0.0      # -1 (bearish) to +1 (bullish) - combined
    confidence: float = 0.0     # 0 to 1
    headline_count: int = 0     # How many headlines were analyzed
    source: str = "none"        # "haiku", "f&g", "default"
    summary: str = ""           # Short LLM-generated rationale
    timestamp: float = 0.0      # time.time() when generated
    # [C5] Factual vs Subjective split
    factual_direction: float = 0.0       # -1 to +1 (hard data: earnings, regs, hacks)
    subjective_direction: float = 0.0    # -1 to +1 (opinions: analyst calls, social buzz)
    factual_confidence: float = 0.0      # 0 to 1
    subjective_confidence: float = 0.0   # 0 to 1
    # [L3] Crowding risk - True when crowd sentiment is extreme and contrarian signal applies
    crowding_risk: bool = False
    # Observability (v6.8)
    cache_hit: bool = False
    source_status: Dict[str, Any] = field(default_factory=dict)
    coverage: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "asset": self.asset,
            "direction": round(self.direction, 4),
            "confidence": round(self.confidence, 4),
            "factual_direction": round(self.factual_direction, 4),
            "subjective_direction": round(self.subjective_direction, 4),
            "factual_confidence": round(self.factual_confidence, 4),
            "subjective_confidence": round(self.subjective_confidence, 4),
            "crowding_risk": self.crowding_risk,
            "headline_count": self.headline_count,
            "source": self.source,
            "summary": self.summary[:120],
            "timestamp": self.timestamp,
            "data_age_seconds": round(now - self.timestamp, 1) if self.timestamp > 0 else None,
            "cache_hit": self.cache_hit,
            "source_status": self.source_status,
            "coverage": self.coverage,
        }


# ============================================================================
# HAIKU PROMPT
# ============================================================================

_SYSTEM_PROMPT = """\
You are a crypto market sentiment analyzer for a SHORT-BIASED trading system.

CRITICAL FRAMING - our system profits from shorts, so interpret sentiment through that lens:
- Retail euphoria / moon talk = CONTRARIAN SHORT signal (bearish for us = negative sentiment)
- Fear / panic / crash talk = could be opportunity OR exhaustion, assess carefully
- Liquidation cascades = strong SHORT confirmation (very negative sentiment)
- Sarcasm / memes = noise, low confidence
- Institutional news (ETF, regulation) = assess actual market impact, not headline tone

Output a JSON object with exactly these fields:
{
  "factual_direction": <float -1.0 to 1.0>,
  "subjective_direction": <float -1.0 to 1.0>,
  "factual_confidence": <float 0.0 to 1.0>,
  "subjective_confidence": <float 0.0 to 1.0>,
  "crowding_risk": <bool>,
  "summary": "<one sentence>"
}

factual_direction: sentiment from hard data (regulations, hacks, exchange listings, protocol upgrades, on-chain metrics, liquidation data). -1=bearish, +1=bullish.
subjective_direction: sentiment from opinions/narrative (analyst calls, social buzz, influencer takes, FOMO/FUD). -1=bearish, +1=bullish.
factual_confidence: how clear the factual signal is (0=none, 1=very clear).
subjective_confidence: how clear the subjective signal is (0=none, 1=very clear).
crowding_risk: true when crowd positioning is extreme (retail euphoria, short crowding via negative funding, overleveraged longs). Our system reduces size when crowding is detected.
summary: one-sentence rationale.

CALIBRATION EXAMPLES from our trading history:

EXAMPLE 1:
Headlines: ["BTC breaks $100K! To the moon!", "Everyone buying Bitcoin right now"]
-> {"factual_direction": 0.0, "subjective_direction": -0.4, "factual_confidence": 0.1, "subjective_confidence": 0.6, "crowding_risk": true, "summary": "Retail euphoria, potential long crowding - contrarian short signal"}
WHY: No factual catalyst. Retail FOMO = subjective bearish for our short-bias system. Crowding risk active.

EXAMPLE 2:
Headlines: ["Massive long liquidations cascade on Binance, $500M wiped", "Longs getting rekt"]
-> {"factual_direction": -0.8, "subjective_direction": -0.6, "factual_confidence": 0.85, "subjective_confidence": 0.7, "crowding_risk": false, "summary": "Long liquidation cascade - strong short confirmation"}
WHY: Liquidation data is factual. Cascade is directional = high confidence short signal.

EXAMPLE 3:
Headlines: ["SEC approves spot Bitcoin ETF", "BlackRock ETF sees record inflows"]
-> {"factual_direction": 0.6, "subjective_direction": 0.3, "factual_confidence": 0.7, "subjective_confidence": 0.5, "crowding_risk": false, "summary": "Institutional adoption catalyst - genuine structural demand"}
WHY: Regulatory approval is factual. Real demand shift, not just narrative.

EXAMPLE 4:
Headlines: ["Bitcoin is dead for the 500th time", "Crypto winter forever"]
-> {"factual_direction": 0.0, "subjective_direction": 0.0, "factual_confidence": 0.0, "subjective_confidence": 0.1, "crowding_risk": false, "summary": "Sarcasm/meme content, no actionable signal"}
WHY: No real data. Sarcasm = noise. Very low confidence.

EXAMPLE 5:
Headlines: ["Tether reserves audit reveals $2B hole", "USDT depegs to $0.97"]
-> {"factual_direction": -0.9, "subjective_direction": -0.7, "factual_confidence": 0.9, "subjective_confidence": 0.6, "crowding_risk": false, "summary": "Systemic stablecoin risk - contagion threat"}
WHY: Hard factual data (audit, depeg). Systemic risk = short opportunity.

EXAMPLE 6:
Headlines: ["Fed holds rates steady as expected", "CPI comes in at 3.2% vs 3.1% expected"]
-> {"factual_direction": -0.1, "subjective_direction": 0.0, "factual_confidence": 0.3, "subjective_confidence": 0.1, "crowding_risk": false, "summary": "Macro data in line, minimal crypto impact"}
WHY: Expected outcome. Minimal signal.

EXAMPLE 7:
Headlines: ["Funding rates extremely negative across all exchanges", "Shorts paying 0.1% per 8h"]
-> {"factual_direction": 0.3, "subjective_direction": 0.2, "factual_confidence": 0.7, "subjective_confidence": 0.4, "crowding_risk": true, "summary": "Short crowding detected via funding - squeeze risk"}
WHY: Extreme negative funding = shorts crowded = our system should be cautious about adding shorts.

EXAMPLE 8:
Headlines: ["$BTC breakout confirmed!", "Bitcoin breaking resistance, next stop $120K"]
-> {"factual_direction": 0.0, "subjective_direction": -0.3, "factual_confidence": 0.1, "subjective_confidence": 0.5, "crowding_risk": true, "summary": "Retail calling breakout, potential bull trap - contrarian short signal"}
WHY: Retail confidently calling breakout = often the top. No factual catalyst. Contrarian SHORT signal, moderate confidence.

Output ONLY the JSON object. No markdown. No explanation. No preamble."""


def _build_user_prompt(
    asset: str,
    headlines: List[str],
    regime: str = "",
    funding_rate: float = 0.0,
) -> str:
    """Build the user prompt from headlines with regime context (Step 18.5 spec)."""
    joined = "\n".join(f"- {h}" for h in headlines[:50])
    # [#5] Regime-aware context per spec: same headline = different interpretation per regime
    context = f"Asset: {asset}\n"
    if regime:
        context += f"Current market regime: {regime}\n"
    if abs(funding_rate) > 0.0001:
        context += f"Current funding rate: {funding_rate:+.4f} (8h)\n"
    context += f"\nRecent headlines:\n{joined}\n\nAnalyze sentiment."
    return context


# ============================================================================
# ROBUST JSON PARSING
# ============================================================================

_REQUIRED_KEYS = ("factual_direction", "subjective_direction",
                   "factual_confidence", "subjective_confidence", "summary")
# crowding_risk is optional (defaults False) - prompt asks for it but don't fail if missing


def _parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse LLM JSON response with fence stripping, schema validation,
    type coercion, and range clipping.  Returns None on any failure.
    Never raises.
    """
    try:
        t = text.strip()
        # Strip code fences (```json ... ``` or ``` ... ```)
        if t.startswith("```"):
            t = t.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(t)
        if not isinstance(data, dict):
            return None

        # Verify required keys
        for key in _REQUIRED_KEYS:
            if key not in data:
                return None

        # Coerce + clip
        def _clip_float(v, lo, hi):
            return max(lo, min(hi, float(v)))

        data["factual_direction"] = _clip_float(data["factual_direction"], -1.0, 1.0)
        data["subjective_direction"] = _clip_float(data["subjective_direction"], -1.0, 1.0)
        data["factual_confidence"] = _clip_float(data["factual_confidence"], 0.0, 1.0)
        data["subjective_confidence"] = _clip_float(data["subjective_confidence"], 0.0, 1.0)
        data["crowding_risk"] = bool(data.get("crowding_risk", False))
        data["summary"] = str(data["summary"])[:200]

        return data
    except Exception:
        return None


# ============================================================================
# COMBINED DIRECTION/CONFIDENCE POLICY (SOTA-aligned)
# ============================================================================

def _compute_combined(
    fact_dir: float, subj_dir: float,
    fact_conf: float, subj_conf: float,
) -> tuple:
    """
    Compute combined direction and confidence from factual/subjective splits.

    Policy:
    - Direction = confidence-weighted average of factual and subjective.
    - Confidence = max(fact_conf, subj_conf) with a penalty when the signal
      is purely subjective (factual_confidence near zero).
    - If both confidences near zero -> direction forced to 0.0.

    Returns (direction, confidence) both clipped.
    """
    total_weight = fact_conf + subj_conf
    if total_weight < 0.01:
        return 0.0, 0.0

    direction = (fact_dir * fact_conf + subj_dir * subj_conf) / total_weight
    direction = max(-1.0, min(1.0, direction))

    confidence = max(fact_conf, subj_conf)
    # Penalty: purely subjective signal (factual < 0.1 but subjective high)
    if fact_conf < 0.1 and subj_conf > 0.3:
        confidence *= 0.7
    confidence = min(1.0, confidence)

    return round(direction, 4), round(confidence, 4)


_FACT_POSITIVE_KEYWORDS = {
    "approve": 0.8,
    "approval": 0.8,
    "approved": 0.8,
    "etf inflow": 0.9,
    "inflows": 0.6,
    "adoption": 0.6,
    "partnership": 0.5,
    "integrates": 0.4,
    "integration": 0.4,
    "upgrade": 0.4,
    "launch": 0.3,
    "record revenue": 0.7,
    "buyback": 0.6,
}

_FACT_NEGATIVE_KEYWORDS = {
    "hack": 0.9,
    "exploit": 0.9,
    "breach": 0.8,
    "lawsuit": 0.7,
    "sec probe": 0.8,
    "investigation": 0.7,
    "reject": 0.8,
    "rejected": 0.8,
    "outflow": 0.6,
    "liquidation": 0.6,
    "depeg": 0.9,
    "bankruptcy": 0.9,
    "delist": 0.8,
    "outage": 0.5,
    "downtime": 0.5,
    "drain": 0.7,
}

_SUBJ_EUPHORIA_KEYWORDS = {
    "to the moon": 0.9,
    "moon": 0.7,
    "ath": 0.6,
    "all-time high": 0.6,
    "rocket": 0.7,
    "breakout": 0.4,
    "parabolic": 0.7,
    "send it": 0.8,
    "bull run": 0.5,
    "mega pump": 0.8,
}

_SUBJ_PANIC_KEYWORDS = {
    "panic": 0.4,
    "crash": 0.4,
    "capitulation": 0.7,
    "bloodbath": 0.7,
    "rekt": 0.4,
    "fear": 0.3,
    "wipeout": 0.5,
}


def _score_headline_heuristic(asset: str, headlines: List[str], now_ts: float) -> Optional[SentimentResult]:
    """Deterministic headline fallback when CryptoPanic is live but Haiku is unavailable."""
    if not headlines:
        return None

    fact_score = 0.0
    subj_score = 0.0
    fact_hits = 0
    subj_hits = 0

    for headline in headlines[:50]:
        text = str(headline).strip().lower()
        if not text:
            continue
        for kw, weight in _FACT_POSITIVE_KEYWORDS.items():
            if kw in text:
                fact_score += weight
                fact_hits += 1
        for kw, weight in _FACT_NEGATIVE_KEYWORDS.items():
            if kw in text:
                fact_score -= weight
                fact_hits += 1
        for kw, weight in _SUBJ_EUPHORIA_KEYWORDS.items():
            if kw in text:
                # Short-biased framing: retail euphoria is contrarian bearish.
                subj_score -= weight
                subj_hits += 1
        for kw, weight in _SUBJ_PANIC_KEYWORDS.items():
            if kw in text:
                # Panic can mean squeeze exhaustion / reflexive bounce risk.
                subj_score += weight * 0.35
                subj_hits += 1

    if fact_hits == 0 and subj_hits == 0:
        return None

    fact_dir = max(-1.0, min(1.0, fact_score / 2.5))
    subj_dir = max(-1.0, min(1.0, subj_score / 2.0))
    fact_conf = min(0.85, 0.18 * fact_hits)
    subj_conf = min(0.70, 0.14 * subj_hits)
    combined_dir, combined_conf = _compute_combined(fact_dir, subj_dir, fact_conf, subj_conf)
    crowding_risk = subj_hits > 0 and abs(subj_score) >= 0.8
    summary = (
        f"Headline heuristic from {len(headlines)} headlines "
        f"(fact_hits={fact_hits}, subj_hits={subj_hits})"
    )

    return SentimentResult(
        asset=asset,
        direction=combined_dir,
        confidence=max(0.2, combined_conf),
        headline_count=len(headlines),
        source="headline_heuristic",
        summary=summary,
        timestamp=now_ts,
        factual_direction=round(fact_dir, 4),
        subjective_direction=round(subj_dir, 4),
        factual_confidence=round(fact_conf, 4),
        subjective_confidence=round(subj_conf, 4),
        crowding_risk=crowding_risk,
    )


def _score_cryptopanic_metrics(
    asset: str,
    metrics: Dict[str, Any],
    now_ts: float,
    data_age_s: Optional[float] = None,
) -> Optional[SentimentResult]:
    """Live CryptoPanic metrics fallback when titles are unavailable but feed metrics exist."""
    consensus = max(-1.0, min(1.0, float(metrics.get("sentiment_consensus", 0.0) or 0.0)))
    panic = max(0.0, min(1.0, float(metrics.get("panic_score", 0.0) or 0.0)))
    velocity = max(0.0, min(1.0, float(metrics.get("news_velocity", 0.0) or 0.0)))
    intensity = max(0.0, min(1.0, float(metrics.get("narrative_intensity", 0.0) or 0.0)))

    if velocity < 0.05 and abs(consensus) < 0.08 and panic < 0.20 and intensity < 0.10:
        return None

    age_penalty = 1.0
    if data_age_s is not None:
        if data_age_s > 1800:
            return None
        if data_age_s > 900:
            age_penalty = 0.6

    narrative_bias = 0.0
    if abs(consensus) >= 0.05:
        narrative_bias = intensity if consensus > 0 else -intensity

    factual_direction = max(-1.0, min(1.0, consensus - 0.80 * panic))
    subjective_direction = max(-1.0, min(1.0, 0.70 * consensus + 0.30 * narrative_bias - 0.35 * panic))
    factual_confidence = min(1.0, (0.30 + 0.40 * velocity + 0.30 * max(abs(consensus), panic)) * age_penalty)
    subjective_confidence = min(1.0, (0.25 + 0.35 * intensity + 0.25 * velocity + 0.15 * abs(consensus)) * age_penalty)
    combined_direction, combined_confidence = _compute_combined(
        factual_direction,
        subjective_direction,
        factual_confidence,
        subjective_confidence,
    )
    crowding_risk = panic > 0.80 or (intensity > 0.75 and abs(consensus) > 0.50)

    return SentimentResult(
        asset=asset,
        direction=combined_direction,
        confidence=combined_confidence,
        headline_count=0,
        source="cryptopanic_metrics",
        summary=(
            f"CryptoPanic metrics fallback consensus={consensus:+.2f}, "
            f"panic={panic:.2f}, velocity={velocity:.2f}, narrative={intensity:.2f}"
        ),
        timestamp=now_ts,
        factual_direction=round(factual_direction, 4),
        subjective_direction=round(subjective_direction, 4),
        factual_confidence=round(factual_confidence, 4),
        subjective_confidence=round(subjective_confidence, 4),
        crowding_risk=crowding_risk,
    )


# ============================================================================
# LLM SENTIMENT AGENT
# ============================================================================

def _parse_regain_utc(msg: str) -> Optional[str]:
    """[P329c] Pull the reset instant out of an Anthropic usage-limit message.

    "You will regain access on 2026-09-01 at 00:00 UTC." -> "2026-09-01T00:00Z"

    Returns None when the wording changes, and the caller then falls back to
    its normal hard-disable cooldown — a parse failure must never degrade to
    "retry immediately", which is the behaviour being fixed.
    """
    import re
    m = re.search(r"regain access on (\d{4}-\d{2}-\d{2})(?:\s+at\s+(\d{2}:\d{2}))?", msg)
    if not m:
        return None
    return f"{m.group(1)}T{m.group(2) or '00:00'}Z"


class LLMSentimentAgent:
    """
    Per-asset sentiment via Anthropic Haiku + CryptoPanic.

    Usage:
        agent = LLMSentimentAgent()
        result = await agent.analyze(asset="BTC", headlines=["BTC hits ATH", ...])
        # result.direction = 0.7, result.confidence = 0.8, result.source = "haiku"
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",  # [#6] Updated per Step 18 spec
        max_tokens: int = 200,
        timeout_sec: float = 8.0,
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model
        self._max_tokens = max_tokens
        self._timeout_sec = timeout_sec

        # Lazy-loaded Anthropic client
        self._client = None
        self._client_init_failed = False

        # Cache: asset -> SentimentResult.  Per-asset validity via result.timestamp.
        self._cache: Dict[str, SentimentResult] = {}
        self._cache_ttl: float = 14400.0  # [#11] 4 hours per Step 18 spec (was 1h)

        # Concurrency guard: avoid stampede from parallel BTC/ETH/SOL calls
        self._semaphore = asyncio.Semaphore(3)

        # Stats
        self._call_count: int = 0
        self._error_count: int = 0
        self._last_haiku_ok_ts: Optional[float] = None

        # Circuit breaker: consecutive Haiku failures -> cooldown
        self._consecutive_haiku_failures: int = 0
        self._circuit_breaker_threshold: int = 5     # open after N consecutive failures
        self._circuit_breaker_cooldown_sec: float = 300.0  # 5 min cooldown
        self._circuit_breaker_opened_at: Optional[float] = None
        self._haiku_hard_disable_until: Optional[float] = None
        self._haiku_hard_disable_reason: str = ""
        self._haiku_hard_disable_logged: bool = False
        self._haiku_hard_disable_cooldown_sec: float = 6 * 3600.0

        # V6: event emission callback (set by EventBus integration)
        self._event_callback = None

        # V6: rolling direction history for sentiment_shock detection
        self._prev_directions: Dict[str, float] = {}  # asset -> last direction

        if self._api_key:
            logger.info("[LLM_SENTIMENT] Initialized (Haiku ready)")
        else:
            logger.warning("[LLM_SENTIMENT] No ANTHROPIC_API_KEY - Haiku disabled, F&G fallback only")

    def set_event_callback(self, callback):
        """Set EventBus callback for SENTIMENT_TICK emission."""
        self._event_callback = callback

    def _compute_sentiment_shock(self, asset: str, direction: float) -> float:
        """Compute sentiment_shock: absolute delta from previous direction.

        A large shock (e.g., direction flipped from +0.6 to -0.4 -> shock=1.0)
        signals regime-relevant sentiment shift. Returns 0.0 on first call.
        """
        prev = self._prev_directions.get(asset, direction)
        shock = abs(direction - prev)
        self._prev_directions[asset] = direction
        return round(min(shock, 2.0), 4)

    def _emit_sentiment_tick(self, result: "SentimentResult"):
        """Publish SENTIMENT_TICK event if callback is wired."""
        if not self._event_callback:
            return
        try:
            self._event_callback("SENTIMENT_TICK", result.to_dict())
        except Exception:
            pass  # Never block tick

    # -----------------------------------------------------------------------
    # CIRCUIT BREAKER
    # -----------------------------------------------------------------------

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (Haiku calls blocked)."""
        _now = time.time()
        if self._haiku_hard_disable_until is not None:
            if _now < self._haiku_hard_disable_until:
                if not self._haiku_hard_disable_logged:
                    _remaining = int(max(0, self._haiku_hard_disable_until - _now))
                    logger.warning(
                        "[LLM_SENTIMENT] Haiku hard-disabled "
                        f"(reason={self._haiku_hard_disable_reason}, retry_in={_remaining}s)"
                    )
                    self._haiku_hard_disable_logged = True
                return True
            logger.info("[LLM_SENTIMENT] Haiku hard-disable expired, resuming attempts")
            self._haiku_hard_disable_until = None
            self._haiku_hard_disable_reason = ""
            self._haiku_hard_disable_logged = False

        if self._consecutive_haiku_failures < self._circuit_breaker_threshold:
            return False
        if self._circuit_breaker_opened_at is None:
            self._circuit_breaker_opened_at = time.time()
            logger.warning(
                f"[LLM_SENTIMENT] Circuit breaker OPEN after "
                f"{self._consecutive_haiku_failures} consecutive failures "
                f"(cooldown={self._circuit_breaker_cooldown_sec}s)"
            )
            return True
        elapsed = time.time() - self._circuit_breaker_opened_at
        if elapsed >= self._circuit_breaker_cooldown_sec:
            # Half-open: allow one attempt
            logger.info("[LLM_SENTIMENT] Circuit breaker HALF-OPEN (attempting recovery)")
            self._circuit_breaker_opened_at = None
            self._consecutive_haiku_failures = 0
            return False
        return True

    def _record_haiku_success(self):
        """Reset circuit breaker on success."""
        self._consecutive_haiku_failures = 0
        self._circuit_breaker_opened_at = None
        self._haiku_hard_disable_until = None
        self._haiku_hard_disable_reason = ""
        self._haiku_hard_disable_logged = False

    def _record_haiku_failure(self):
        """Increment consecutive failure counter."""
        self._consecutive_haiku_failures += 1

    def _open_hard_disable(self, reason: str, cooldown_sec: Optional[float] = None):
        """Open a long cooldown for non-retryable API failures."""
        cooldown = float(cooldown_sec or self._haiku_hard_disable_cooldown_sec)
        now = time.time()
        self._haiku_hard_disable_until = now + max(60.0, cooldown)
        self._haiku_hard_disable_reason = reason
        self._haiku_hard_disable_logged = False
        self._consecutive_haiku_failures = max(
            self._consecutive_haiku_failures,
            self._circuit_breaker_threshold,
        )
        self._circuit_breaker_opened_at = now
        logger.error(
            "[LLM_SENTIMENT] Non-retryable Haiku failure, entering hard-disable "
            f"for {int(cooldown)}s (reason={reason})"
        )

    @staticmethod
    def _classify_haiku_error(err: Exception) -> tuple:
        """Return (status_code, is_non_retryable, reason).

        [P24 2026-04-24] 429 used to fall through to "transient_error",
        which meant the next per-asset tick fired another request
        and burned more quota. Now 429 is treated as non-retryable
        for a short cooldown window via _open_hard_disable; reason
        encodes Retry-After if present so the caller can size the
        cooldown.
        """
        status_code = getattr(err, "status_code", None)
        if status_code is None:
            _resp = getattr(err, "response", None)
            status_code = getattr(_resp, "status_code", None) if _resp is not None else None
        msg = str(err).lower()
        # [P329c] A 400 carrying "you have reached your specified API usage
        # limits" is a HARD-DATED account cap, not a malformed request:
        #
        #   400 invalid_request_error: You have reached your specified API
        #   usage limits. You will regain access on 2026-09-01 at 00:00 UTC.
        #
        # 400 was in neither the non-retryable list nor the 429 branch, so it
        # fell to the bare-warning `else` with NO backoff — retried once per
        # asset per tick (~18 calls/day) for the whole capped period. That is
        # P293b/P319 exactly: a quota exhaustion retried like a transient.
        # Unlike P319 the reset is not guessed — the API states it — so the
        # cooldown is sized from the message when it can be parsed.
        if status_code == 400 and (
                "usage limit" in msg or "regain access" in msg):
            return 400, True, f"usage_limit_400_until={_parse_regain_utc(msg) or 'unparsed'}"
        if status_code in (401, 403, 404, 422):
            return status_code, True, f"http_{status_code}"
        if status_code == 429 or "429" in msg or "rate limit" in msg or "rate_limit" in msg:
            # Try to read Retry-After from the underlying response.
            retry_after = None
            try:
                _resp = getattr(err, "response", None)
                _hdrs = getattr(_resp, "headers", None) if _resp is not None else None
                if _hdrs is not None:
                    raw = _hdrs.get("retry-after") or _hdrs.get("Retry-After")
                    if raw:
                        retry_after = min(300.0, max(1.0, float(raw)))
            except (ValueError, TypeError, AttributeError):
                retry_after = None
            return 429, True, f"rate_limit_429_retry_after={retry_after or 'none'}"
        if "404" in msg and "not found" in msg:
            return status_code, True, "http_404_not_found"
        if "401" in msg or "unauthorized" in msg:
            return status_code, True, "http_401_unauthorized"
        if "403" in msg or "forbidden" in msg:
            return status_code, True, "http_403_forbidden"
        if "invalid model" in msg or ("model" in msg and "not found" in msg):
            return status_code, True, "invalid_model"
        return status_code, False, "transient_error"

    # -----------------------------------------------------------------------
    # HMATS COMPATIBILITY - generate_signal / to_agent_signal_dict
    # -----------------------------------------------------------------------

    def generate_signal(
        self,
        asset: str = "BTC",
        price: float = 0.0,
        regime: str = "",
    ) -> SentimentResult:
        """
        Synchronous HMATS generate_signal() wrapper.

        Returns cached result if available, else neutral.
        For async callers, use analyze() directly.
        """
        cached = self._cache.get(asset)
        if cached is not None:
            return cached
        return SentimentResult(
            asset=asset,
            source="default",
            summary="no_cached_result",
            timestamp=time.time(),
            source_status={"note": "sync_call_no_cache"},
            coverage={"factual": "none", "subjective": "none", "combined": "none"},
        )

    def to_agent_signal_dict(self, result: "SentimentResult") -> dict:
        """
        Convert SentimentResult to flat dict for Authority Fusion.

        V6 contract: includes sentiment_shock, missing_inputs, asof_timestamp.
        Keys prefixed with sentiment_ to avoid collision.
        """
        shock = self._compute_sentiment_shock(result.asset, result.direction)
        missing = [k for k, v in result.coverage.items() if v == "none"]
        data_age = round(time.time() - result.timestamp, 1) if result.timestamp > 0 else None

        # Emit SENTIMENT_TICK event
        self._emit_sentiment_tick(result)

        return {
            "sentiment_direction": result.direction,
            "sentiment_confidence": result.confidence,
            "sentiment_factual_direction": result.factual_direction,
            "sentiment_subjective_direction": result.subjective_direction,
            "sentiment_crowding_risk": result.crowding_risk,
            "sentiment_source": result.source,
            "sentiment_headline_count": result.headline_count,
            "sentiment_data_quality": (
                1.0 if result.source == "haiku"
                else 0.75 if result.source == "headline_heuristic"
                else 0.65 if result.source == "cryptopanic_metrics"
                else 0.5 if result.source == "f&g"
                else 0.0
            ),
            "sentiment_shock": shock,
            "missing_inputs": missing,
            "asof_timestamp": result.timestamp,
            "data_age_seconds": data_age,
            "asset": result.asset,
        }

    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------

    async def analyze(
        self,
        asset: str,
        headlines: Optional[List[str]] = None,
        fg_zscore: float = 0.0,
        cryptopanic_meta: Optional[Dict[str, Any]] = None,
        regime: str = "",             # [#5] GMM regime for context-aware analysis
        funding_rate: float = 0.0,    # [#5] Current funding rate for disambiguation
    ) -> SentimentResult:
        """
        Analyze sentiment for one asset. Never raises.

        Fallback chain:
          1. CryptoPanic headlines -> Haiku
          2. Deterministic headline heuristic
          3. Live CryptoPanic metrics
          4. F&G zscore -> direction + confidence
          5. Neutral (0.0, 0.0)

        Args:
            asset: "BTC", "ETH", or "SOL"
            headlines: News headline strings (from CryptoPanic or similar)
            fg_zscore: Fear & Greed zscore from Layer 1 (fallback)

        Returns:
            SentimentResult (always, never None)
        """
        now = time.time()

        # Per-asset cache check: validity based on the cached result's own timestamp.
        cached = self._cache.get(asset)
        if cached is not None and (now - cached.timestamp) < self._cache_ttl:
            # Return a copy with cache_hit=True (don't mutate stored object)
            hit = SentimentResult(
                asset=cached.asset,
                direction=cached.direction,
                confidence=cached.confidence,
                headline_count=cached.headline_count,
                source=cached.source,
                summary=cached.summary,
                timestamp=cached.timestamp,
                factual_direction=cached.factual_direction,
                subjective_direction=cached.subjective_direction,
                factual_confidence=cached.factual_confidence,
                subjective_confidence=cached.subjective_confidence,
                crowding_risk=cached.crowding_risk,
                cache_hit=True,
                source_status=cached.source_status,
                coverage=cached.coverage,
            )
            return hit

        # Track source statuses for this call
        haiku_status: Dict[str, Any] = {
            "enabled": bool(self._api_key),
            "used": False,
            "error": "",
            "latency_ms": 0,
            "last_ok_ts": self._last_haiku_ok_ts,
        }
        cryptopanic_status: Dict[str, Any] = {
            "available": headlines is not None and len(headlines or []) > 0,
            "headline_count": len(headlines) if headlines else 0,
            "window_seconds": None,
            "newest_headline_age_s": None,
            "error": "",
        }
        if cryptopanic_meta:
            cryptopanic_status.update({
                "available": bool(cryptopanic_meta.get("headlines")),
                "headline_count": len(cryptopanic_meta.get("headlines", []) or []),
                "window_seconds": cryptopanic_meta.get("window_seconds"),
                "newest_headline_age_s": cryptopanic_meta.get("newest_headline_age_s"),
                "window_unknown": bool(cryptopanic_meta.get("window_unknown", False)),
                "data_age_s": cryptopanic_meta.get("data_age_s"),
                "is_mock": bool(cryptopanic_meta.get("is_mock", False)),
                "metrics": cryptopanic_meta.get("metrics", {}),
                "error": cryptopanic_meta.get("error", ""),
            })
        fg_status: Dict[str, Any] = {
            "used": False,
            "fg_zscore": fg_zscore,
            "dead_zone_threshold": 0.5,
        }

        # Layer 1: Haiku + headlines (with circuit breaker)
        circuit_open = self._is_circuit_open()
        if headlines and len(headlines) > 0 and self._api_key and not circuit_open:
            try:
                t0 = time.monotonic()
                # P71: thread regime + funding_rate through to _call_haiku so
                # _build_user_prompt sees them. Previously these were referenced
                # inside _call_haiku without being in its signature → NameError.
                result = await self._call_haiku(
                    asset, headlines, regime=regime, funding_rate=funding_rate,
                )
                latency_ms = round((time.monotonic() - t0) * 1000)
                haiku_status["used"] = True
                haiku_status["latency_ms"] = latency_ms

                if result is not None:
                    self._record_haiku_success()
                    self._last_haiku_ok_ts = now
                    haiku_status["last_ok_ts"] = now
                    result.source_status = {
                        "haiku": haiku_status,
                        "cryptopanic": cryptopanic_status,
                        "fg_fallback": fg_status,
                    }
                    result.coverage = {
                        "factual": "llm",
                        "subjective": "llm",
                        "combined": "derived",
                    }
                    self._cache[asset] = result
                    return result
                else:
                    self._record_haiku_failure()
                    haiku_status["error"] = "parse_or_call_failed"
            except Exception as e:
                logger.debug(f"[LLM_SENTIMENT] Haiku failed for {asset}: {e}")
                self._record_haiku_failure()
                self._error_count += 1
                haiku_status["error"] = str(e)[:100]
        elif circuit_open:
            if self._haiku_hard_disable_until and time.time() < self._haiku_hard_disable_until:
                haiku_status["error"] = f"hard_disabled:{self._haiku_hard_disable_reason}"
                haiku_status["retry_after_sec"] = int(
                    self._haiku_hard_disable_until - time.time()
                )
            else:
                haiku_status["error"] = "circuit_breaker_open"

        # Layer 2: Deterministic headline heuristic fallback
        if headlines and len(headlines) > 0:
            try:
                result = _score_headline_heuristic(asset, headlines, now)
                if result is not None:
                    result.source_status = {
                        "haiku": haiku_status,
                        "cryptopanic": cryptopanic_status,
                        "fg_fallback": fg_status,
                        "heuristic": {"used": True, "headline_count": len(headlines)},
                    }
                    result.coverage = {
                        "factual": "heuristic_headlines",
                        "subjective": "heuristic_headlines",
                        "combined": "derived",
                    }
                    self._cache[asset] = result
                    return result
            except Exception as e:
                logger.debug(f"[LLM_SENTIMENT] headline heuristic failed for {asset}: {e}")

        # Layer 3: Live CryptoPanic metrics fallback
        _cp_metrics = cryptopanic_meta.get("metrics", {}) if isinstance(cryptopanic_meta, dict) else {}
        _cp_is_mock = bool(cryptopanic_meta.get("is_mock", False)) if isinstance(cryptopanic_meta, dict) else False
        _cp_age = cryptopanic_meta.get("data_age_s") if isinstance(cryptopanic_meta, dict) else None
        if _cp_metrics and not _cp_is_mock:
            try:
                result = _score_cryptopanic_metrics(asset, _cp_metrics, now, _cp_age)
                if result is not None:
                    result.source_status = {
                        "haiku": haiku_status,
                        "cryptopanic": cryptopanic_status,
                        "fg_fallback": fg_status,
                        "metrics_fallback": {"used": True, "data_age_s": _cp_age},
                    }
                    result.coverage = {
                        "factual": "cryptopanic_metrics",
                        "subjective": "cryptopanic_metrics",
                        "combined": "derived",
                    }
                    self._cache[asset] = result
                    return result
            except Exception as e:
                logger.debug(f"[LLM_SENTIMENT] cryptopanic metrics fallback failed for {asset}: {e}")

        # Layer 4: F&G zscore fallback
        # [FIX-AG9] Dead zone narrowed 0.5→0.3; confidence penalty reduced.
        # F&G is a strong contrarian indicator in crypto. Was neutered to 5-12% eff. confidence.
        if abs(fg_zscore) > 0.3:
            try:
                fg_status["used"] = True
                direction = max(-1.0, min(1.0, fg_zscore / 2.5))
                # F&G is global, not per-asset -> moderate discount, not halve
                confidence = min(abs(fg_zscore) / 2.5, 1.0) * 0.65

                # F&G is ~50% factual (market-wide metric), ~50% subjective (crowd behavior)
                fact_dir = direction
                subj_dir = direction
                fact_conf = confidence * 0.5   # [FIX-AG9] Was 0.3 (too low for market-wide data)
                subj_conf = confidence * 0.5   # [FIX-AG9] Was 0.7
                combined_dir, combined_conf = _compute_combined(
                    fact_dir, subj_dir, fact_conf, subj_conf,
                )

                source_status = {
                    "haiku": haiku_status,
                    "cryptopanic": cryptopanic_status,
                    "fg_fallback": fg_status,
                }
                coverage = {
                    "factual": "fallback_inferred_global",
                    "subjective": "fallback_inferred_global",
                    "combined": "derived",
                }

                result = SentimentResult(
                    asset=asset,
                    direction=combined_dir,
                    confidence=combined_conf,
                    headline_count=0,
                    source="f&g",
                    summary=f"F&G fallback zscore={fg_zscore:+.2f} (global, inferred)",
                    timestamp=now,
                    factual_direction=round(fact_dir, 4),
                    subjective_direction=round(subj_dir, 4),
                    factual_confidence=round(fact_conf, 4),
                    subjective_confidence=round(subj_conf, 4),
                    source_status=source_status,
                    coverage=coverage,
                )
                self._cache[asset] = result
                return result
            except Exception:
                pass

        # Layer 5: Neutral default
        return SentimentResult(
            asset=asset,
            direction=0.0,
            confidence=0.0,
            headline_count=0,
            source="default",
            summary="No sentiment data",
            timestamp=now,
            source_status={
                "haiku": haiku_status,
                "cryptopanic": cryptopanic_status,
                "fg_fallback": fg_status,
            },
            coverage={
                "factual": "none",
                "subjective": "none",
                "combined": "none",
            },
        )

    async def analyze_many(
        self,
        assets: List[str],
        fg_zscore: Any = 0.0,
        max_total_sec: float = 8.0,
    ) -> Dict[str, SentimentResult]:
        """
        Analyze multiple assets concurrently with an overall timeout.

        Args:
            assets: list of asset symbols (e.g. ["BTC", "ETH", "SOL"])
            fg_zscore: float (applied to all) or Dict[str, float] per-asset
            max_total_sec: hard deadline; unfinished assets get neutral fallback

        Returns:
            Dict mapping asset -> SentimentResult (always complete, never raises)
        """
        # Normalize fg_zscore to per-asset dict
        if isinstance(fg_zscore, (int, float)):
            fg_map = {a: float(fg_zscore) for a in assets}
        elif isinstance(fg_zscore, dict):
            fg_map = {a: float(fg_zscore.get(a, 0.0)) for a in assets}
        else:
            fg_map = {a: 0.0 for a in assets}

        async def _analyze_one(asset: str) -> tuple:
            try:
                headlines = await fetch_headlines(asset)
                result = await self.analyze(
                    asset=asset,
                    headlines=headlines,
                    fg_zscore=fg_map.get(asset, 0.0),
                )
                return asset, result
            except Exception:
                return asset, SentimentResult(
                    asset=asset, source="default",
                    summary="analyze error", timestamp=time.time(),
                    source_status={"error": "analyze_exception"},
                    coverage={"factual": "none", "subjective": "none", "combined": "none"},
                )

        results: Dict[str, SentimentResult] = {}
        tasks = {asset: asyncio.create_task(_analyze_one(asset)) for asset in assets}

        try:
            done, pending = await asyncio.wait(
                tasks.values(),
                timeout=max_total_sec,
            )
        except Exception:
            done, pending = set(), set(tasks.values())

        # Collect completed results
        for task in done:
            try:
                asset, result = task.result()
                results[asset] = result
            except Exception:
                pass

        # Cancel and fill in timed-out assets
        for task in pending:
            task.cancel()

        now = time.time()
        for asset in assets:
            if asset not in results:
                results[asset] = SentimentResult(
                    asset=asset,
                    source="default",
                    summary="timeout",
                    timestamp=now,
                    source_status={
                        "haiku": {"enabled": bool(self._api_key), "used": False,
                                  "error": "timeout", "latency_ms": 0, "last_ok_ts": None},
                        "cryptopanic": {"available": False, "headline_count": 0,
                                        "error": "timeout"},
                        "fg_fallback": {"used": False},
                    },
                    coverage={
                        "factual": "none",
                        "subjective": "none",
                        "combined": "none",
                    },
                )

        return results

    def get_status(self) -> Dict[str, Any]:
        """Agent status for monitoring."""
        return {
            "haiku_enabled": bool(self._api_key),
            "model": self._model,
            "calls": self._call_count,
            "errors": self._error_count,
            "cached_assets": list(self._cache.keys()),
            "last_haiku_ok_ts": self._last_haiku_ok_ts,
            "hard_disable": {
                "active": bool(
                    self._haiku_hard_disable_until
                    and time.time() < self._haiku_hard_disable_until
                ),
                "reason": self._haiku_hard_disable_reason,
                "retry_at_ts": self._haiku_hard_disable_until,
            },
            "circuit_breaker": {
                "open": self._is_circuit_open(),
                "consecutive_failures": self._consecutive_haiku_failures,
                "threshold": self._circuit_breaker_threshold,
                "cooldown_sec": self._circuit_breaker_cooldown_sec,
            },
        }

    # -----------------------------------------------------------------------
    # PRIVATE: Haiku call
    # -----------------------------------------------------------------------

    def _get_client(self):
        """Lazy-init Anthropic client."""
        if self._client is not None:
            return self._client
        if self._client_init_failed:
            return None
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            return self._client
        except Exception as e:
            logger.warning(f"[LLM_SENTIMENT] Anthropic SDK not available: {e}")
            self._client_init_failed = True
            return None

    async def _call_haiku(
        self,
        asset: str,
        headlines: List[str],
        regime: str = "",
        funding_rate: float = 0.0,
    ) -> Optional[SentimentResult]:
        """Call Haiku API with timeout + semaphore. Returns None on any failure.

        P71: regime + funding_rate were referenced inside this function but
        not in its signature, raising NameError on every call. The surrounding
        try/except returned None, silently falling back to the deterministic
        heuristic — L3 LLM sentiment was effectively dead since this code
        shipped. Threading the analyze() kwargs through fixes it.
        """
        client = self._get_client()
        if client is None:
            return None

        user_prompt = _build_user_prompt(asset, headlines, regime=regime, funding_rate=funding_rate)

        async with self._semaphore:
            try:
                response = await asyncio.wait_for(
                    client.messages.create(
                        model=self._model,
                        max_tokens=self._max_tokens,
                        temperature=0,
                        system=_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": user_prompt}],
                    ),
                    timeout=self._timeout_sec,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[LLM_SENTIMENT] Haiku timeout ({self._timeout_sec}s) for {asset}")
                self._error_count += 1
                return None
            except Exception as e:
                status_code, non_retryable, reason = self._classify_haiku_error(e)
                if status_code == 429:
                    # [P24 2026-04-24] 429: short cooldown driven by
                    # Retry-After (parsed in _classify_haiku_error). Fall back
                    # to 30s if header missing. Don't lock for the full
                    # haiku_hard_disable_cooldown_sec — that's for auth/model
                    # errors, not transient rate limits.
                    retry_after = 30.0
                    if "retry_after=" in reason:
                        _val = reason.split("retry_after=", 1)[1]
                        if _val and _val != "none":
                            try:
                                retry_after = float(_val)
                            except (ValueError, TypeError):
                                retry_after = 30.0
                    self._open_hard_disable(reason=reason, cooldown_sec=retry_after)
                    logger.warning(
                        f"[LLM_SENTIMENT] Haiku 429 rate limit for {asset}: "
                        f"cooldown={retry_after:.0f}s"
                    )
                elif status_code == 400 and reason.startswith("usage_limit_400"):
                    # [P329c] Back off until the instant the API named. A
                    # parse failure falls through to the standard hard-disable
                    # cooldown rather than to no backoff at all.
                    _cd = None
                    _until = reason.split("until=", 1)[1] if "until=" in reason else ""
                    if _until and _until != "unparsed":
                        try:
                            _dt = datetime.fromisoformat(
                                _until.replace("Z", "+00:00"))
                            _cd = max(
                                0.0,
                                (_dt - datetime.now(timezone.utc)).total_seconds())
                        except (ValueError, TypeError):
                            _cd = None
                    self._open_hard_disable(reason=reason, cooldown_sec=_cd)
                    logger.warning(
                        f"[LLM_SENTIMENT] Haiku ACCOUNT USAGE LIMIT for {asset} "
                        f"(regain access {_until or 'unknown'}). This is a "
                        f"BILLING state, not a fault: sentiment falls back to "
                        f"the headline heuristic and stays tradeable. Not "
                        f"retried until then."
                    )
                elif non_retryable:
                    self._open_hard_disable(reason=reason)
                    logger.warning(
                        f"[LLM_SENTIMENT] Haiku API non-retryable error for {asset}: "
                        f"status={status_code}, err={e}"
                    )
                else:
                    logger.warning(f"[LLM_SENTIMENT] Haiku API error for {asset}: {e}")
                self._error_count += 1
                return None

        self._call_count += 1

        # Parse response
        try:
            text = response.content[0].text.strip()
        except (IndexError, AttributeError):
            self._error_count += 1
            return None

        data = _parse_llm_json(text)
        if data is None:
            logger.warning(f"[LLM_SENTIMENT] Haiku parse error for {asset}")
            self._error_count += 1
            return None

        fact_dir = data["factual_direction"]
        subj_dir = data["subjective_direction"]
        fact_conf = data["factual_confidence"]
        subj_conf = data["subjective_confidence"]
        crowding = data["crowding_risk"]
        summary = data["summary"]

        combined_dir, combined_conf = _compute_combined(
            fact_dir, subj_dir, fact_conf, subj_conf,
        )

        return SentimentResult(
            asset=asset,
            direction=combined_dir,
            confidence=combined_conf,
            headline_count=len(headlines),
            source="haiku",
            summary=summary,
            timestamp=time.time(),
            factual_direction=round(fact_dir, 4),
            subjective_direction=round(subj_dir, 4),
            factual_confidence=round(fact_conf, 4),
            subjective_confidence=round(subj_conf, 4),
            crowding_risk=crowding,
        )


# ============================================================================
# HEADLINE FETCHER (CryptoPanic wrapper)
# ============================================================================

_HEADLINE_WINDOW = timedelta(hours=4)


async def fetch_headlines(
    asset: str,
    window: Optional[timedelta] = None,
) -> List[str]:
    """
    Fetch recent headlines for an asset from CryptoPanic.
    Returns empty list on any failure (never raises).

    Time-filters to `window` (default 4H) using NewsItem.published_at.
    Deduplicates by title.
    """
    try:
        # [P322f] Same switch, same reason as fetch_headlines_with_meta: read
        # it before constructing, so a disabled feed leaves no startup lines
        # that look like it ignored the flag. This function is CryptoPanic-only
        # (no CC News, no RSS), so declining here really is "no headlines" —
        # unlike its sibling, where an early return also killed two other
        # sources (P322d).
        try:
            from data_mgmt.feeds.cryptopanic_feed import is_feed_enabled
            if not is_feed_enabled():
                return []
        except Exception:  # noqa: silent-swallow — an unreadable switch must not disable a working feed; default to ON
            pass

        from data_mgmt.feeds.cryptopanic_feed import get_cryptopanic_feed
        feed = get_cryptopanic_feed()
        now = datetime.now(timezone.utc)
        win = window or _HEADLINE_WINDOW

        data = feed.get_latest()
        data_age_s = None
        cache_is_empty = False
        if data is not None and getattr(data, "timestamp", None) is not None:
            data_age_s = (now - data.timestamp).total_seconds()
            cache_is_empty = not bool(getattr(data, "recent_news", None))

        needs_refresh = (
            data is None
            or (data_age_s is not None and data_age_s > min(win.total_seconds(), 3600))
        )
        if needs_refresh:
            refreshed = await feed.fetch()
            if refreshed is not None:
                data = refreshed
                cache_is_empty = not bool(getattr(data, "recent_news", None))

        if data is None:
            logger.debug(f"[LLM_SENTIMENT] {asset}: CryptoPanic unavailable (no cached data)")
            return []

        cutoff = now - win

        seen_titles: set = set()
        headlines: List[str] = []

        for item in data.recent_news:
            if asset not in item.currencies:
                continue
            # Time filter
            if item.published_at < cutoff:
                continue
            # Dedup
            title_lower = item.title.strip().lower()
            if title_lower in seen_titles:
                continue
            seen_titles.add(title_lower)
            headlines.append(item.title)

        # [#12] Adaptive window: if < 3 headlines, expand to 8H per spec
        if len(headlines) < 3 and win.total_seconds() < 28800:
            expanded_win = timedelta(hours=8)
            expanded_cutoff = now - expanded_win
            for item in data.recent_news:
                if asset not in item.currencies:
                    continue
                if item.published_at < expanded_cutoff:
                    continue
                title_lower = item.title.strip().lower()
                if title_lower in seen_titles:
                    continue
                seen_titles.add(title_lower)
                headlines.append(item.title)
            if len(headlines) >= 3:
                logger.info(f"[LLM_SENTIMENT] {asset}: expanded to 8H window, got {len(headlines)} headlines")

        if not headlines:
            logger.debug(
                f"[LLM_SENTIMENT] {asset}: no CryptoPanic headlines in "
                f"window={int(win.total_seconds())}s, cached_news={len(getattr(data, 'recent_news', []))}, "
                f"cache_empty={cache_is_empty}, data_age_s={None if data_age_s is None else round(data_age_s, 1)}"
            )

        return headlines[:50]  # Cap at 50
    except Exception as e:
        logger.debug(f"[LLM_SENTIMENT] CryptoPanic fetch failed for {asset}: {e}")
        return []


async def fetch_headlines_with_meta(
    asset: str,
    window: Optional[timedelta] = None,
) -> Dict[str, Any]:
    """
    Like fetch_headlines but returns metadata for source_status.
    Never raises.
    """
    meta: Dict[str, Any] = {
        "headlines": [],
        "window_seconds": None,
        "newest_headline_age_s": None,
        "data_age_s": None,
        "is_mock": False,
        "metrics": {},
        "window_unknown": False,
        "error": "",
    }
    # [P322d] THE THREE SOURCES ARE INDEPENDENT, and the shared locals are
    # hoisted here so they cannot stop being reachable.
    #
    # This function blends CryptoPanic + CC News + RSS, but every one of the
    # CryptoPanic bail-outs below used to `return meta` — including the
    # PRE-EXISTING `data is None` branch. So whenever the FIRST source had
    # nothing, the other two never ran: a curated-feed outage silently took
    # the whole agent dark rather than degrading to the free sources that
    # exist precisely for that case. Latent while CryptoPanic answered;
    # permanent the moment it is disabled (P322b), which is how it surfaced —
    # SOL went `headlines=0 tradeable=False` on a deploy whose only intent
    # was to stop paying a vendor.
    #
    # CryptoPanic is now ONE OPTIONAL SOURCE OF THREE: it can decline, and
    # the blend continues.
    now = datetime.now(timezone.utc)
    win = window or _HEADLINE_WINDOW
    cutoff = now - win
    meta["window_seconds"] = win.total_seconds()
    seen_titles: set = set()
    headlines: List[str] = []
    newest_age: Optional[float] = None
    try:
        # [P322f] The switch is read BEFORE the singleton is built, so a
        # disabled feed is never CONSTRUCTED — not merely muted.
        #
        # Construction is cheap and issues no requests, so this is not about
        # cost. It is about the log: a disabled feed that still prints
        # `[CRYPTOPANIC] Initialized: mock=False` and `restored backoff ...`
        # at every boot reads exactly like a feed that ignored the flag, and
        # an operator checking whether the disable took effect has no way to
        # tell those apart from four INFO lines. Silence is the observable
        # that makes the config verifiable.
        try:
            from data_mgmt.feeds.cryptopanic_feed import is_feed_enabled
            _cp_on = is_feed_enabled()
        except Exception:  # noqa: silent-swallow — an unreadable switch must not disable a working feed; default to ON
            _cp_on = True

        feed = None
        if _cp_on:
            from data_mgmt.feeds.cryptopanic_feed import get_cryptopanic_feed
            feed = get_cryptopanic_feed()
            meta["is_mock"] = bool(getattr(feed, "_mock_mode", False))
        if not _cp_on:
            # [P322c] The operator disabled this feed. This function is one of
            # TWO sites that reach the singleton directly, which is why the
            # switch is module-level rather than a main.py handle.
            meta["reason"] = "cryptopanic_disabled_by_config"
            _cp_usable = False
        elif meta["is_mock"]:
            # [P322] MOCK HEADLINES MUST NEVER REACH HAIKU.
            #
            # `_fetch_mock` invents 3-10 items per currency titled
            # "Mock BTC News 0" and stamps them within the last 240 minutes —
            # i.e. FRESH ENOUGH to pass the 4h window and satisfy the _c3_live
            # starvation gate. `is_mock` was recorded in this meta dict and
            # consulted only by the METRICS path (`if _cp_metrics and not
            # _cp_is_mock`); the headline loop below never looked at it, so
            # fabricated titles were sent to Haiku as real news and their
            # reading came back labelled `status=live`.
            #
            # Latent today only because a key is configured — and the very
            # next operator action under consideration (cancel the plan,
            # remove the key) flips `mock_mode` on, because it is derived as
            # `not bool(cryptopanic_key)`. That is the P2/P223 class: a
            # fabricated value is indistinguishable from a measured one, and
            # here it would arrive already stamped fresh.
            #
            # Mock therefore means NO NEWS, never FAKE NEWS. The RSS blend
            # (P314) is unaffected and continues to carry the agent.
            meta["reason"] = "cryptopanic_mock_mode_no_headlines"
            _cp_usable = False
        else:
            _cp_usable = True

        data = feed.get_latest() if (_cp_usable and feed is not None) else None
        data_age_s = None
        cache_is_empty = False
        if data is not None and getattr(data, "timestamp", None) is not None:
            data_age_s = (now - data.timestamp).total_seconds()
            cache_is_empty = not bool(getattr(data, "recent_news", None))

        # A declined source must never issue a request.
        needs_refresh = _cp_usable and feed is not None and (
            data is None
            or (data_age_s is not None and data_age_s > min(win.total_seconds(), 3600))
        )
        # `feed is not None` is implied by needs_refresh and restated so
        # the narrowing survives the bool — same reason as the branch above.
        if needs_refresh and feed is not None:
            refreshed = await feed.fetch()
            if refreshed is not None:
                data = refreshed
                cache_is_empty = not bool(getattr(data, "recent_news", None))
                if getattr(data, "timestamp", None) is not None:
                    data_age_s = (now - data.timestamp).total_seconds()

        if data is None:
            # [P322d] CryptoPanic contributed nothing. This is RECORDED, not
            # returned: CC News and RSS below are independent sources and the
            # whole point of having them is this case. `error` keeps its
            # historical value so any consumer reading it is unaffected.
            meta["error"] = "no_data"
            _cp_usable = False

        # `data is not None` is redundant with `_cp_usable` (the branch above
        # clears the flag on a None payload) and is stated anyway so the
        # narrowing is visible to a reader and to the type checker — a bool
        # cannot carry that fact.
        if _cp_usable and data is not None:
            meta["data_age_s"] = round(data_age_s, 1) if data_age_s is not None else None
            _asset = asset.upper().replace("/USD", "")
            meta["metrics"] = {
                "sentiment_consensus": float(getattr(data, "sentiment_consensus", {}).get(_asset, 0.0) or 0.0),
                "panic_score": float(getattr(data, "panic_score", {}).get(_asset, 0.0) or 0.0),
                "news_velocity": float(getattr(data, "news_velocity", {}).get(_asset, 0.0) or 0.0),
                "narrative_intensity": float(getattr(data, "narrative_intensity", {}).get(_asset, 0.0) or 0.0),
            }


            for item in data.recent_news:
                if asset not in item.currencies:
                    continue

                # Check if published_at is usable
                if item.published_at is None:
                    meta["window_unknown"] = True
                    title_lower = item.title.strip().lower()
                    if title_lower not in seen_titles:
                        seen_titles.add(title_lower)
                        headlines.append(item.title)
                    continue

                # Time filter
                if item.published_at < cutoff:
                    continue

                age_s = (now - item.published_at).total_seconds()
                if newest_age is None or age_s < newest_age:
                    newest_age = age_s

                title_lower = item.title.strip().lower()
                if title_lower not in seen_titles:
                    seen_titles.add(title_lower)
                    headlines.append(item.title)

            meta["headlines"] = headlines[:50]
            meta["newest_headline_age_s"] = round(newest_age, 1) if newest_age is not None else None
            if not meta["headlines"] and cache_is_empty:
                meta["error"] = "empty_cache"

        # [CC_NEWS_BLEND] Add CryptoCompare News as secondary source (redundancy +
        # different editorial bias). Dedup by title, cap at 50 combined.
        try:
            from data_mgmt.feeds.cryptocompare_news_feed import get_cc_news_feed
            cc_feed = get_cc_news_feed()
            cc_items = await cc_feed.fetch_headlines(asset=asset, limit=25)
            cc_added = 0
            cc_aged_out = 0
            for ci in cc_items:
                tl = ci.title.strip().lower()
                if not tl or tl in seen_titles:
                    continue
                # [P265] The CC News blend must respect the SAME time window
                # the CryptoPanic items are filtered to. The old loop appended
                # regardless of age (the cutoff check guarded only the
                # newest_age STATISTIC) — with the 12h cache serving the API's
                # unbounded "latest" list, day-old corpus was fed to Haiku as
                # "Recent headlines" for ~3 consecutive ticks, and
                # headline_count > 0 kept the _c3_live starvation gate
                # satisfied on stale news — masking exactly the dried-up-
                # coverage condition the count exists to reveal. An item with
                # NO published_at is kept (unknown beats discarding the whole
                # secondary source) but counted separately.
                if ci.published_at and ci.published_at < cutoff:
                    cc_aged_out += 1
                    continue
                seen_titles.add(tl)
                headlines.append(ci.title)
                cc_added += 1
                if ci.published_at:
                    age_s = (now - ci.published_at).total_seconds()
                    if newest_age is None or age_s < newest_age:
                        newest_age = age_s
            meta["cc_news_aged_out"] = cc_aged_out
            meta["cc_news_added"] = cc_added
            meta["headlines"] = headlines[:50]
            if newest_age is not None:
                meta["newest_headline_age_s"] = round(newest_age, 1)
        except Exception as _cc_err:
            meta["cc_news_error"] = str(_cc_err)[:100]

        # [P293c][RSS_BLEND] Third source: public RSS. Free, keyless and
        # unquota'd, which matters because BOTH paid sources are currently
        # dark — CryptoPanic on a monthly-quota 429 until the month rolls,
        # CryptoCompare at 90/90 on an account cap that cannot be raised —
        # and `headline_count == 0` is what makes llm_sentiment untradeable
        # (main.py's _c3_live gate).
        #
        # Deliberately LAST and additive: it can only add headlines the two
        # curated sources did not supply, never displace them. Same dedup
        # and the SAME time window as the other two (P265) — an RSS item
        # older than the window is aged out exactly like a CC News one, so
        # this cannot satisfy the starvation gate with stale copy.
        try:
            from data_mgmt.feeds.rss_news_feed import get_rss_news_feed
            _rss = get_rss_news_feed()
            await _rss.fetch_if_stale()
            _rss_added = 0
            _rss_aged_out = 0
            _rss_undated = 0
            for ri in _rss.get_items(asset):
                tl = (ri.title or "").strip().lower()
                if not tl or tl in seen_titles:
                    continue
                if ri.published_at is None:
                    # Unknown age: keep (a headline with no date still beats
                    # no coverage) but count it separately so a feed that
                    # stops dating its items cannot silently pass as fresh.
                    _rss_undated += 1
                    meta["window_unknown"] = True
                elif ri.published_at < cutoff:
                    _rss_aged_out += 1
                    continue
                else:
                    age_s = (now - ri.published_at).total_seconds()
                    if newest_age is None or age_s < newest_age:
                        newest_age = age_s
                seen_titles.add(tl)
                headlines.append(ri.title)
                _rss_added += 1
            meta["rss_added"] = _rss_added
            meta["rss_aged_out"] = _rss_aged_out
            meta["rss_undated"] = _rss_undated
            meta["headlines"] = headlines[:50]
            if newest_age is not None:
                meta["newest_headline_age_s"] = round(newest_age, 1)
            if _rss_added:
                logger.info(
                    "[RSS_BLEND] %s: +%d headline(s) from public RSS "
                    "(aged_out=%d undated=%d, total=%d)",
                    asset, _rss_added, _rss_aged_out, _rss_undated,
                    len(meta["headlines"]),
                )
        except Exception as _rss_err:
            meta["rss_error"] = str(_rss_err)[:100]

        # The blend may have supplied what the primary could not — clear a
        # stale "empty" verdict so the caller does not read starvation off a
        # now-populated list.
        if meta.get("headlines") and meta.get("error") == "empty_cache":
            meta["error"] = ""

        return meta
    except Exception as e:
        meta["error"] = str(e)[:100]
        return meta


# ============================================================================
# SINGLETON
# ============================================================================

_llm_sentiment_agent: Optional[LLMSentimentAgent] = None


def get_llm_sentiment_agent() -> LLMSentimentAgent:
    """Get or create LLM sentiment agent singleton."""
    global _llm_sentiment_agent
    if _llm_sentiment_agent is None:
        _llm_sentiment_agent = LLMSentimentAgent()
    return _llm_sentiment_agent


def reset_llm_sentiment_agent():
    """Reset singleton (for testing)."""
    global _llm_sentiment_agent
    _llm_sentiment_agent = None


# ============================================================================
# BACKWARD COMPATIBILITY - old names used by agents/__init__.py, sota_integration
# ============================================================================

# Old v5.2 class name -> new LLMSentimentAgent
SentimentLLMAgent = LLMSentimentAgent

# Stubs for removed v5.2 types (tests/sota_integration import these)
class SentimentLevel:
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"

class SentimentAgentConfig:
    pass

class SentimentSignal:
    pass

# Old singleton names
get_sentiment_agent = get_llm_sentiment_agent

def reset_sentiment_agent():
    global _llm_sentiment_agent
    _llm_sentiment_agent = None

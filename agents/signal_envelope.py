"""
AgentSignalEnvelope - Standardized signal wrapper for per-agent attribution.

Wraps each agent's raw output into a uniform envelope that supports:
  - Consistent direction/confidence extraction across all agent types
  - Audit trail via to_audit_record()
  - Attribution scoring (correct/incorrect vs realized price move)

Usage:
    from agents.signal_envelope import AgentSignalEnvelope, wrap_agent_signal

    env = wrap_agent_signal("short_bias", "ADVISE", raw_dict, "BTC")
    record = env.to_audit_record()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class AgentSignalEnvelope:
    """Uniform wrapper around any agent's signal output."""

    agent_name: str          # "short_bias", "quant", "onchain_sol", ...
    authority: str           # "DECIDE" | "ADVISE" | "TRIGGER" | "VETO"
    timestamp: datetime
    asset: str               # "BTC" | "ETH" | "SOL"
    direction: float         # -1.0 to 1.0  (vol_alpha always 0.0)
    confidence: float        # 0.0 to 1.0
    reasoning: str           # one-line explanation
    data_quality: float      # 0.0 to 1.0
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    def to_audit_record(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict for JSONL logging."""
        return {
            "agent_name": self.agent_name,
            "authority": self.authority,
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "direction": self.direction,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "data_quality": self.data_quality,
            "raw_payload": _safe_serialize(self.raw_payload),
        }


# ---------------------------------------------------------------------------
# Agent-specific extraction helpers
# ---------------------------------------------------------------------------

def _extract_quant(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("quant_direction", 0.0)),
        "confidence": float(raw.get("quant_confidence", 0.0)),
        "reasoning": str(raw.get("quant_strategy", "")),
    }


def _extract_short_bias(raw: Dict) -> Dict[str, Any]:
    meta = raw.get("meta", {})
    dominant = meta.get("dominant_signal", "") if isinstance(meta, dict) else ""
    return {
        "direction": float(raw.get("direction", 0.0)),
        "confidence": float(raw.get("confidence", 0.0)),
        "reasoning": str(dominant),
    }


def _extract_sentiment(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("sentiment_direction", 0.0)),
        "confidence": float(raw.get("sentiment_confidence", 0.0)),
        "reasoning": str(raw.get("sentiment_source", "")),
    }


def _extract_onchain_sol(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("onchain_direction", 0.0)),
        "confidence": float(raw.get("onchain_confidence", 0.0)),
        "reasoning": str(raw.get("onchain_reasoning", "")),
    }


def _extract_vol_alpha(raw: Dict) -> Dict[str, Any]:
    # [UPG3] Use implied direction if available (regime-conditional)
    direction = float(raw.get("vol_alpha_implied_direction", 0.0))
    return {
        "direction": direction,
        "confidence": float(raw.get("vol_alpha_intensity", 0.0)) if direction != 0.0 else 0.0,
        "reasoning": str(raw.get("vol_alpha_direction_reason", raw.get("vol_alpha_bias", ""))),
    }


def _extract_micro(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("micro_direction", 0.0)),
        "confidence": float(raw.get("micro_confidence", 0.0)),
        "reasoning": str(raw.get("micro_primary_signal", "")),
    }


def _extract_model_alpha(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("model_alpha_direction", 0.0)),
        "confidence": float(raw.get("model_alpha_confidence", 0.0)),
        "reasoning": ", ".join(raw.get("model_alpha_reasons", [])),
    }


def _extract_v5_1_strats(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("v5_1_strats_direction", 0.0)),
        "confidence": float(raw.get("v5_1_strats_confidence", 0.0)),
        "reasoning": "v5.1 ensemble (micro/cascade/funding/ml_factor)",
    }


def _extract_kraken_quant(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("kq_direction", 0.0)),
        "confidence": float(raw.get("kq_confidence", 0.0)),
        "reasoning": str(raw.get("kq_primary_strategy", "")),
    }


# [FIX 2026-04-22] Extractors for 8 agents added to attribution in main.py:8299.
# Prior code routed these through "unknown_agent" fallback which zeroed out
# direction+confidence, corrupting attribution + hit-rate metrics.

def _extract_drl(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("direction", 0.0)),
        "confidence": float(raw.get("confidence", 0.0)),
        "reasoning": "drl_ensemble",
    }


def _extract_two_stage(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("direction", 0.0)),
        "confidence": float(raw.get("confidence", 0.0)),
        "reasoning": "two_stage_prior",
    }


def _extract_funding(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("direction", 0.0)),
        "confidence": float(raw.get("confidence", 0.0)),
        "reasoning": "funding_carry",
    }


def _extract_llm_sentiment(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("direction", 0.0)),
        "confidence": float(raw.get("confidence", 0.0)),
        "reasoning": "llm_haiku",
    }


def _extract_flow(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("direction", 0.0)),
        "confidence": float(raw.get("confidence", 0.0)),
        "reasoning": "whale_exchange_etf_flow",
    }


def _extract_onchain(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("direction", 0.0)),
        "confidence": float(raw.get("confidence", 0.0)),
        "reasoning": "btc_eth_onchain",
    }


def _extract_soldex(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("direction", 0.0)),
        "confidence": float(raw.get("confidence", 0.0)),
        "reasoning": "soldex_arb",
    }


def _extract_onchain_graph(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("direction", 0.0)),
        "confidence": float(raw.get("confidence", 0.0)),
        "reasoning": "sol_whale_cluster",
    }


# [P57 2026-04-25] whale + options were direction-producing per the authority
# matrix (rows 24 + 22) but missing from _EXTRACTORS — silent zeroing through
# the fallback at line 244.
def _extract_whale(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("direction", 0.0)),
        "confidence": float(raw.get("confidence", 0.0)),
        "reasoning": "whale_flow",
    }


def _extract_options(raw: Dict) -> Dict[str, Any]:
    return {
        "direction": float(raw.get("direction", 0.0)),
        "confidence": float(raw.get("confidence", 0.0)),
        "reasoning": "options_short_confirmation",
    }


_EXTRACTORS = {
    "quant": _extract_quant,
    "short_bias": _extract_short_bias,
    "sentiment": _extract_sentiment,
    "onchain_sol": _extract_onchain_sol,
    "vol_alpha": _extract_vol_alpha,
    "micro": _extract_micro,
    "model_alpha": _extract_model_alpha,
    "v5_1_strats": _extract_v5_1_strats,
    "kraken_quant": _extract_kraken_quant,
    # [FIX 2026-04-22] 8 new direction-producing agents
    "drl": _extract_drl,
    "two_stage": _extract_two_stage,
    "funding": _extract_funding,
    "llm_sentiment": _extract_llm_sentiment,
    "flow": _extract_flow,
    "onchain": _extract_onchain,
    "soldex": _extract_soldex,
    "onchain_graph": _extract_onchain_graph,
    # [P57 2026-04-25]
    "whale": _extract_whale,
    "options": _extract_options,
}


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def wrap_agent_signal(
    agent_name: str,
    authority: str,
    raw_dict: Dict[str, Any],
    asset: str,
) -> AgentSignalEnvelope:
    """Wrap an agent's raw signal dict into a standardized envelope.

    Parameters
    ----------
    agent_name : str
        One of: quant, short_bias, sentiment, onchain_sol, vol_alpha,
        micro, model_alpha, kraken_quant.
    authority : str
        "DECIDE", "ADVISE", "TRIGGER", or "VETO".
    raw_dict : dict
        The raw output from agent.to_agent_signal_dict().
    asset : str
        "BTC", "ETH", or "SOL".
    """
    extractor = _EXTRACTORS.get(agent_name)
    if extractor is not None:
        extracted = extractor(raw_dict)
    else:
        extracted = {
            "direction": 0.0,
            "confidence": 0.0,
            "reasoning": "unknown_agent",
        }

    # Data quality: try several common key patterns, fallback to 1.0
    data_quality = _extract_data_quality(raw_dict)

    return AgentSignalEnvelope(
        agent_name=agent_name,
        authority=authority,
        timestamp=datetime.now(timezone.utc),
        asset=asset,
        direction=float(extracted["direction"]),
        confidence=float(extracted["confidence"]),
        reasoning=str(extracted["reasoning"]),
        data_quality=data_quality,
        raw_payload=raw_dict,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_data_quality(raw: Dict) -> float:
    """Try multiple key patterns for data_quality, fallback to 1.0."""
    for key in (
        "data_quality",
        "onchain_data_quality",
        "sentiment_data_quality",
        "vol_alpha_data_quality",
        "micro_data_quality",
        "model_alpha_data_quality",
        "kq_data_quality",
    ):
        val = raw.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return 1.0


def _safe_serialize(obj: Any) -> Any:
    """Make an object JSON-serializable (best-effort)."""
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    try:
        import json
        json.dumps(obj)
        return obj
    except (TypeError, ValueError, OverflowError):
        return str(obj)
"""
================================================================================
HMATS v6.7 - P1-2C Confidence Gate (Staged)
================================================================================
Soft gate that checks overall confidence and agent alignment count before
allowing new T1 entries.  Profile-gated: ULTRA_AGGRESSIVE_5Y lowers thresholds
via staged schedule; HIGH_RISK keeps defaults (0.65 / 5).

NOT a hard veto - only blocks new entries / tranche escalation.
Does NOT override RiskGovernor, correlation, leverage, or exposure caps.
Does NOT change authority semantics (DECIDE / CONFIRM / VETO).
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# =============================================================================
# AGENT KEYS used for alignment counting
# =============================================================================

# Agents whose direction is checked for alignment
# Same set as partial_consensus CORE_AGENTS + short_bias + sentiment
_ALIGNMENT_AGENTS = [
    ("quant_direction",     "quant_confidence"),
    ("drl_direction",       None),                # DRL has no separate confidence key
    ("sentiment_direction", None),
    ("short_bias_direction", "short_bias_confidence"),
    ("two_stage_direction", "two_stage_confidence"),
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ConfidenceGateConfig:
    """Configuration for confidence gate."""
    min_confidence_default: float = 0.65   # HIGH_RISK default
    min_aligned_agents: int = 5            # HIGH_RISK default
    staged_schedule: List[Dict] = field(default_factory=list)
    stage_selector: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ConfidenceGateConfig":
        return cls(
            min_confidence_default=d.get("min_confidence_default", 0.65),
            min_aligned_agents=d.get("min_aligned_agents", 5),
            staged_schedule=d.get("staged_schedule", []),
            stage_selector=d.get("stage_selector", ""),
        )

    def resolve_min_confidence(self) -> float:
        """Resolve min_confidence from staged schedule."""
        for entry in self.staged_schedule:
            if entry.get("stage") == self.stage_selector:
                return entry.get("min_confidence", self.min_confidence_default)
        return self.min_confidence_default


@dataclass
class ConfidenceGateResult:
    """Result of confidence gate check."""
    passed: bool = True
    reason: str = ""                     # "" = passed, else "LOW_CONF" or "LOW_ALIGNMENT"
    min_confidence_used: float = 0.65
    min_aligned_used: int = 5
    overall_confidence: float = 0.0
    aligned_count: int = 0
    aligned_agents: List[str] = field(default_factory=list)
    stage_selector: str = ""


# =============================================================================
# CONFIDENCE GATE
# =============================================================================

class ConfidenceGate:
    """
    Soft gate: checks overall confidence and agent alignment.

    Usage:
        gate = ConfidenceGate(config)
        result = gate.check(intent_direction, quant_confidence, agent_signals)
        if not result.passed:
            # block new T1 entry (soft - not a hard veto)
    """

    def __init__(self, config: ConfidenceGateConfig):
        self.config = config
        self._min_confidence = config.resolve_min_confidence()

    def check(
        self,
        intent_direction: float,
        overall_confidence: float,
        agent_signals: Dict[str, Any],
    ) -> ConfidenceGateResult:
        """
        Check confidence gate.

        Args:
            intent_direction: decided direction from engine (-1 to +1)
            overall_confidence: quant_confidence (blended by engine)
            agent_signals: dict with agent direction/confidence keys

        Returns:
            ConfidenceGateResult
        """
        result = ConfidenceGateResult(
            min_confidence_used=self._min_confidence,
            min_aligned_used=self.config.min_aligned_agents,
            overall_confidence=overall_confidence,
            stage_selector=self.config.stage_selector,
        )

        # Count aligned agents
        direction_sign = 1 if intent_direction > 0 else (-1 if intent_direction < 0 else 0)
        if direction_sign == 0:
            # No direction -> gate passes (nothing to align on)
            result.aligned_count = 0
            return result

        aligned = []
        for dir_key, conf_key in _ALIGNMENT_AGENTS:
            agent_dir = agent_signals.get(dir_key, 0.0)
            if agent_dir == 0.0:
                continue  # neutral, doesn't count
            agent_sign = 1 if agent_dir > 0 else -1
            if agent_sign == direction_sign:
                aligned.append(dir_key.replace("_direction", ""))

        result.aligned_count = len(aligned)
        result.aligned_agents = aligned

        # Check thresholds
        if overall_confidence < self._min_confidence:
            result.passed = False
            result.reason = "LOW_CONF"
        elif result.aligned_count < self.config.min_aligned_agents:
            result.passed = False
            result.reason = "LOW_ALIGNMENT"

        return result

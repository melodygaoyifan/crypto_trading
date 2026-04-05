"""
================================================================================
HMATS P1.1 — Sentiment-Conditioned Participation Gate
================================================================================
Purpose: Allow sentiment to influence participation / aggression / confidence
         in aligned high-conviction states WITHOUT flipping trade direction.

Design:
  - Sentiment CANNOT create a trade or reverse direction
  - Sentiment CAN: boost confidence when aligned, suppress when misaligned
  - All effects are bounded multipliers [0.85, 1.15]
  - Default: DISABLED (flag-gated)
  - Observable in proof logs

Integration:
  Called after sentiment signals computed, before engine.decide().
  Output injected into agent_signals for fusion consumption.
================================================================================
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

logger = logging.getLogger("HMATS.SentimentGate")


@dataclass
class SentimentGateConfig:
    """Config for sentiment participation gate."""
    enabled: bool = False
    # Minimum |sentiment_zscore| to activate gate
    min_zscore: float = 1.5
    # When sentiment aligns with quant direction: boost multiplier
    aligned_boost: float = 1.10      # +10% confidence/exposure on alignment
    # When sentiment opposes quant direction: suppress multiplier
    opposed_suppress: float = 0.90   # -10% confidence/exposure on opposition
    # Maximum confidence multiplier (prevent runaway)
    max_mult: float = 1.15
    min_mult: float = 0.85
    # Regime filter: only active in these regimes (empty = all)
    active_regimes: tuple = ("MOMENTUM_RALLY", "PANIC_SELLOFF", "STEADY_UPTREND")


class SentimentParticipationGate:
    """
    Modulates participation/confidence based on sentiment alignment.

    Output keys in agent_signals:
      _sg_confidence_mult: float  — multiply into quant confidence
      _sg_proof: str              — human-readable explanation
    """

    def __init__(self, config: Optional[SentimentGateConfig] = None):
        self.config = config or SentimentGateConfig()

    def compute(
        self,
        agent_signals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute sentiment gate modifiers."""
        cfg = self.config
        if not cfg.enabled:
            return {"_sg_confidence_mult": 1.0, "_sg_proof": "disabled"}

        regime = agent_signals.get("regime_state", "UNKNOWN").upper()
        if cfg.active_regimes and regime not in cfg.active_regimes:
            return {"_sg_confidence_mult": 1.0, "_sg_proof": f"regime={regime} not gated"}

        sentiment_z = agent_signals.get("sentiment_zscore", 0.0)
        quant_dir = agent_signals.get("quant_direction", 0.0)

        if abs(sentiment_z) < cfg.min_zscore:
            return {"_sg_confidence_mult": 1.0, "_sg_proof": f"zscore={sentiment_z:.2f} below threshold"}

        # Direction alignment check
        sent_dir = 1.0 if sentiment_z > 0 else -1.0
        quant_sign = 1.0 if quant_dir > 0 else (-1.0 if quant_dir < 0 else 0.0)

        if quant_sign == 0.0:
            return {"_sg_confidence_mult": 1.0, "_sg_proof": "quant_dir=0"}

        aligned = (sent_dir * quant_sign) > 0

        if aligned:
            # Scale boost by zscore magnitude (stronger sentiment = more boost)
            scale = min(1.0, (abs(sentiment_z) - cfg.min_zscore) / 2.0)  # 0-1 over 2 zscore range
            mult = 1.0 + (cfg.aligned_boost - 1.0) * scale
        else:
            scale = min(1.0, (abs(sentiment_z) - cfg.min_zscore) / 2.0)
            mult = 1.0 + (cfg.opposed_suppress - 1.0) * scale

        mult = max(cfg.min_mult, min(cfg.max_mult, mult))

        proof = (
            f"P1.1 SENT_GATE: z={sentiment_z:+.2f} q={quant_dir:+.3f} "
            f"{'ALIGNED' if aligned else 'OPPOSED'} -> conf_mult={mult:.3f}"
        )

        return {"_sg_confidence_mult": mult, "_sg_proof": proof}


# Singleton
_instance: Optional[SentimentParticipationGate] = None

def get_sentiment_gate(config: Optional[SentimentGateConfig] = None) -> SentimentParticipationGate:
    global _instance
    if _instance is None:
        _instance = SentimentParticipationGate(config)
    return _instance

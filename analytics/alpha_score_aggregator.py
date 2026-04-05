"""
================================================================================
V10: AlphaScoreAggregator - Per-asset alpha scoring + alpha-weighted allocation.
================================================================================
Consumes: agent signal strength/direction/confidence per asset
Produces: per-asset alpha_score (0-1), alpha-weighted allocation

Usage in main.py 4H tick:
    from analytics.alpha_score_aggregator import get_alpha_aggregator
    scores = aggregator.compute(asset, agent_signals, regime, regime_confidence)
    allocation_weight = scores.alpha_score / total_alpha
================================================================================
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AlphaScore:
    """Per-asset alpha evaluation."""
    asset: str
    alpha_score: float = 0.0
    direction_agreement: float = 0.0
    confidence_weighted: float = 0.0
    regime_alignment: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AlphaScoreAggregator:
    """
    Aggregates multi-agent signals into per-asset alpha score.

    Replaces static equal-weight allocation with alpha-weighted distribution
    across BTC/ETH/SOL.
    """

    def __init__(self):
        self._history: Dict[str, List[AlphaScore]] = {}
        self._weights = {
            "quant": 0.35,
            "drl": 0.25,
            "sentiment": 0.15,
            "macro": 0.10,
            "flow": 0.15,
        }

    def compute(
        self,
        asset: str,
        agent_signals: Dict,
        regime: str = "unknown",
        regime_confidence: float = 0.5,
    ) -> AlphaScore:
        """Compute single-asset alpha score from agent signals."""
        try:
            directions = {}
            strengths = {}

            for agent_name, weight in self._weights.items():
                sig = agent_signals.get(agent_name, {})
                if isinstance(sig, dict):
                    d = sig.get("direction", 0)
                    s = sig.get("strength", 0)
                    c = sig.get("confidence", 0.5)
                elif hasattr(sig, 'direction'):
                    d = getattr(sig, 'direction', 0)
                    s = getattr(sig, 'strength', 0)
                    c = getattr(sig, 'confidence', 0.5)
                else:
                    continue

                directions[agent_name] = d
                strengths[agent_name] = abs(s) * c * weight

            if not strengths:
                return AlphaScore(asset=asset)

            # Direction agreement: fraction of agents aligned
            dir_signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in directions.values()]
            non_zero = [s for s in dir_signs if s != 0]
            if non_zero:
                majority = max(set(non_zero), key=non_zero.count)
                agreement = sum(1 for s in non_zero if s == majority) / len(non_zero)
            else:
                agreement = 0.0

            # Confidence-weighted signal strength
            total_strength = sum(strengths.values())

            # Regime alignment bonus
            regime_bonus = 0.0
            dominant_dir = 1 if sum(directions.values()) > 0 else -1
            regime_upper = regime.upper() if regime else ""
            if "RALLY" in regime_upper or "MOMENTUM" in regime_upper:
                regime_bonus = 0.2 if dominant_dir > 0 else -0.1
            elif "PANIC" in regime_upper or "SELLOFF" in regime_upper:
                regime_bonus = 0.2 if dominant_dir < 0 else -0.1
            elif "ACCUMULATION" in regime_upper:
                regime_bonus = 0.1 if dominant_dir > 0 else 0.0

            # Composite alpha_score
            alpha = (
                0.40 * agreement
                + 0.35 * min(total_strength, 1.0)
                + 0.15 * regime_confidence
                + 0.10 * max(regime_bonus + 0.5, 0)
            )
            alpha = max(0.0, min(alpha, 1.0))

            score = AlphaScore(
                asset=asset,
                alpha_score=alpha,
                direction_agreement=agreement,
                confidence_weighted=total_strength,
                regime_alignment=regime_bonus,
            )

            # History (bounded)
            if asset not in self._history:
                self._history[asset] = []
            self._history[asset].append(score)
            if len(self._history[asset]) > 100:
                self._history[asset] = self._history[asset][-50:]

            logger.debug(
                f"[V10] AlphaScore {asset}: {alpha:.3f} "
                f"(agree={agreement:.2f}, str={total_strength:.2f}, regime={regime_bonus:+.2f})"
            )

            return score

        except Exception as e:
            logger.debug(f"[V10] AlphaScore compute error for {asset}: {e}")
            return AlphaScore(asset=asset)

    def get_allocation_weights(self, scores: Dict[str, AlphaScore]) -> Dict[str, float]:
        """
        Convert per-asset alpha scores to allocation weights.

        Constraint: each asset allocation in [15%, 55%].
        """
        if not scores:
            return {}

        total = sum(s.alpha_score for s in scores.values())
        if total <= 0:
            n = len(scores)
            return {a: 1.0 / n for a in scores}

        raw = {a: s.alpha_score / total for a, s in scores.items()}

        # Clamp to [0.15, 0.55]
        clamped = {}
        for a, w in raw.items():
            clamped[a] = max(0.15, min(w, 0.55))

        # Re-normalize
        total_clamped = sum(clamped.values())
        return {a: w / total_clamped for a, w in clamped.items()}


# Singleton
_aggregator: Optional[AlphaScoreAggregator] = None


def get_alpha_aggregator() -> AlphaScoreAggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = AlphaScoreAggregator()
    return _aggregator
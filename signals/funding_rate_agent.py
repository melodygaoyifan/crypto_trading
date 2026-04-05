"""
================================================================================
HMATS v9 - Funding Rate Agent
================================================================================
Carries bias towards earning funding payments.
Registered as ADVISE authority in NORMAL mode.
[v9-PATCH-8]

Follow the extension pattern in: signals/authority_fusion_volalpha.py
================================================================================
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict

logger = logging.getLogger(__name__)

try:
    from signals.authority_fusion import (
        AgentSignal, Authority, AuthorityFusionEngine, get_fusion_engine,
    )
    AUTHORITY_AVAILABLE = True
except ImportError:
    AUTHORITY_AVAILABLE = False


@dataclass
class FundingRateSignal(AgentSignal if AUTHORITY_AVAILABLE else object):
    """Signal from funding rate analysis."""
    agent_name: str = "FundingRateAgent"
    funding_rate: float = 0.0
    carry_direction: str = "NEUTRAL"  # LONG_CARRY | SHORT_CARRY | NEUTRAL
    carry_confidence: float = 0.0


FUNDING_AUTHORITY = {
    "NORMAL": "ADVISE",
    "OPPORTUNITY": "CONFIRM",
    "NO_TRADE": "ADVISE",
}

FUNDING_THRESHOLD = 0.0002  # [FIX-AG1] Was 0.002 (never triggered). Kraken 8h funding rate typical range: ±0.0001-0.0008. 0.0002 = 0.02%/8h ≈ 22% annualized — threshold for meaningful carry.


class FundingRateAgent:
    """
    Generates carry-bias signals from perp funding rates.

    When funding is significantly positive (longs pay shorts),
    bias towards SHORT to earn carry. Vice versa for negative funding.
    """

    def __init__(self):
        self._last_rates: Dict[str, float] = {}
        logger.info("[v9-PATCH-8] FundingRateAgent initialized")

    def generate_signal(self, asset: str, funding_rate: Optional[float]) -> FundingRateSignal:
        """Generate a carry-bias signal from current funding rate."""
        if funding_rate is None:
            return FundingRateSignal(carry_direction="NEUTRAL")

        self._last_rates[asset] = funding_rate

        # [FIX-AG1] Scale to Kraken 8h rate range: 0.0002 (threshold) to 0.001 (extreme)
        _intensity = min(abs(funding_rate) / 0.001, 1.0)  # 0→1 over [0, 0.1%/8h]

        if funding_rate > FUNDING_THRESHOLD:
            # Longs pay shorts -> bias SHORT to earn carry
            return FundingRateSignal(
                funding_rate=funding_rate,
                carry_direction="SHORT_CARRY",
                carry_confidence=_intensity,
                direction=-0.3 * _intensity,
                confidence=_intensity * 0.5,
            )
        elif funding_rate < -FUNDING_THRESHOLD:
            # Shorts pay longs -> bias LONG to earn carry
            return FundingRateSignal(
                funding_rate=funding_rate,
                carry_direction="LONG_CARRY",
                carry_confidence=_intensity,
                direction=0.3 * _intensity,
                confidence=_intensity * 0.5,
            )
        return FundingRateSignal(carry_direction="NEUTRAL")

    def get_last_rates(self) -> Dict[str, float]:
        """Get last known funding rates per asset."""
        return dict(self._last_rates)

    def get_status(self) -> Dict:
        """Get agent status."""
        return {
            "last_rates": self._last_rates,
            "threshold": FUNDING_THRESHOLD,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_funding_agent: Optional[FundingRateAgent] = None


def get_funding_rate_agent() -> FundingRateAgent:
    """Get or create the singleton FundingRateAgent."""
    global _funding_agent
    if _funding_agent is None:
        _funding_agent = FundingRateAgent()
    return _funding_agent


def reset_funding_rate_agent():
    """Reset the singleton."""
    global _funding_agent
    _funding_agent = None
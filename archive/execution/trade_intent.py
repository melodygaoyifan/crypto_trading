"""
HMATS v3.2 - Trade Intent Layer
================================
Purpose: Decouple strategy decisions from execution mechanics.

Philosophy:
- Fusion outputs INTENT (what we want)
- Execution translates to ORDERS (how to get it)
- Execution failure ≠ Strategy failure

Intent is immutable once created. Execution logs its own success/failure separately.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, List
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """What the strategy wants to do."""
    FLAT = auto()           # Close all positions
    LONG = auto()           # Go long
    SHORT = auto()          # Go short
    REDUCE_LONG = auto()    # Reduce long exposure
    REDUCE_SHORT = auto()   # Reduce short exposure
    HOLD = auto()           # No change desired


class IntentUrgency(Enum):
    """How urgent is this intent."""
    IMMEDIATE = auto()      # Execute ASAP (CRACK window, stop-loss)
    NORMAL = auto()         # Standard execution window
    PATIENT = auto()        # Can wait for better fills


class IntentAuthority(Enum):
    """Who authorized this intent."""
    HARD_STATE = auto()     # NO_TRADE / Data invalid / Emergency
    AUTHORITY_QUANT = auto()    # Quant dominant decision
    AUTHORITY_DRL = auto()      # DRL dominant decision
    AUTHORITY_BLEND = auto()    # Weighted blend
    DEFENSIVE_OVERRIDE = auto() # Defense layer override
    OPPORTUNITY_MODE = auto()   # Aggressive opportunity


@dataclass(frozen=True)
class TradeIntent:
    """
    Immutable strategy intent.
    
    This is the OUTPUT of Fusion and INPUT to Execution.
    Once created, it cannot be modified - only replaced by a new intent.
    """
    # Identity
    intent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Core intent
    asset: str = "BTC"
    intent_type: IntentType = IntentType.HOLD
    target_exposure: float = 0.0  # -1.0 to 1.0 (short to long)
    
    # Context
    authority: IntentAuthority = IntentAuthority.AUTHORITY_BLEND
    urgency: IntentUrgency = IntentUrgency.NORMAL
    
    # Confidence & reasoning
    confidence: float = 0.5
    reasoning: str = ""
    
    # Constraints for execution
    max_slippage_bps: float = 50.0
    time_limit_seconds: float = 300.0  # 5 min default
    
    # Source signals (for audit)
    quant_signal: float = 0.0
    drl_signal: float = 0.0
    sentiment_bias: float = 0.0
    regime_name: str = "UNKNOWN"
    
    def to_dict(self) -> Dict:
        return {
            "intent_id": self.intent_id,
            "created_at": self.created_at.isoformat(),
            "asset": self.asset,
            "intent_type": self.intent_type.name,
            "target_exposure": self.target_exposure,
            "authority": self.authority.name,
            "urgency": self.urgency.name,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "max_slippage_bps": self.max_slippage_bps,
            "time_limit_seconds": self.time_limit_seconds,
            "quant_signal": self.quant_signal,
            "drl_signal": self.drl_signal,
            "sentiment_bias": self.sentiment_bias,
            "regime_name": self.regime_name
        }


class IntentOutcome(Enum):
    """Execution result for an intent."""
    PENDING = auto()
    FULLY_FILLED = auto()
    PARTIALLY_FILLED = auto()
    REJECTED_BY_EXECUTION = auto()  # Execution layer refused
    TIMED_OUT = auto()
    CANCELLED = auto()
    SUPERSEDED = auto()  # New intent replaced this one


@dataclass
class IntentExecutionReport:
    """
    Execution reports back on intent fulfillment.
    
    CRITICAL: Execution failure is NOT strategy failure.
    This is logged separately for analysis.
    """
    intent_id: str
    outcome: IntentOutcome
    
    # What we achieved
    achieved_exposure: float = 0.0
    fill_rate: float = 0.0  # 0-1
    
    # Execution metrics
    avg_slippage_bps: float = 0.0
    execution_time_ms: float = 0.0
    orders_placed: int = 0
    orders_filled: int = 0
    orders_cancelled: int = 0
    
    # Failure details (if any)
    failure_reason: str = ""
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def is_success(self) -> bool:
        return self.outcome in [IntentOutcome.FULLY_FILLED, IntentOutcome.PARTIALLY_FILLED]
    
    def to_dict(self) -> Dict:
        return {
            "intent_id": self.intent_id,
            "outcome": self.outcome.name,
            "achieved_exposure": self.achieved_exposure,
            "fill_rate": self.fill_rate,
            "avg_slippage_bps": self.avg_slippage_bps,
            "execution_time_ms": self.execution_time_ms,
            "orders_placed": self.orders_placed,
            "orders_filled": self.orders_filled,
            "orders_cancelled": self.orders_cancelled,
            "failure_reason": self.failure_reason,
            "is_success": self.is_success()
        }


class IntentFactory:
    """Factory for creating properly validated intents."""
    
    @staticmethod
    def create_flat_intent(
        asset: str,
        authority: IntentAuthority,
        reasoning: str,
        urgency: IntentUrgency = IntentUrgency.IMMEDIATE
    ) -> TradeIntent:
        """Create a FLAT intent (close all positions)."""
        return TradeIntent(
            asset=asset,
            intent_type=IntentType.FLAT,
            target_exposure=0.0,
            authority=authority,
            urgency=urgency,
            confidence=1.0,  # FLAT is always high confidence
            reasoning=reasoning
        )
    
    @staticmethod
    def create_directional_intent(
        asset: str,
        target_exposure: float,
        authority: IntentAuthority,
        confidence: float,
        reasoning: str,
        quant_signal: float = 0.0,
        drl_signal: float = 0.0,
        sentiment_bias: float = 0.0,
        regime_name: str = "UNKNOWN",
        urgency: IntentUrgency = IntentUrgency.NORMAL,
        max_slippage_bps: float = 50.0
    ) -> TradeIntent:
        """Create a directional intent."""
        # Determine intent type
        if abs(target_exposure) < 0.01:
            intent_type = IntentType.HOLD
        elif target_exposure > 0:
            intent_type = IntentType.LONG
        else:
            intent_type = IntentType.SHORT
        
        return TradeIntent(
            asset=asset,
            intent_type=intent_type,
            target_exposure=max(-1.0, min(1.0, target_exposure)),
            authority=authority,
            urgency=urgency,
            confidence=confidence,
            reasoning=reasoning,
            max_slippage_bps=max_slippage_bps,
            quant_signal=quant_signal,
            drl_signal=drl_signal,
            sentiment_bias=sentiment_bias,
            regime_name=regime_name
        )


class IntentJournal:
    """
    Records all intents and their outcomes.
    
    Purpose: Audit trail + strategy vs execution analysis.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.intents: Dict[str, TradeIntent] = {}
        self.reports: Dict[str, IntentExecutionReport] = {}
        self.history: List[str] = []  # intent_ids in order
    
    def record_intent(self, intent: TradeIntent):
        """Record a new intent."""
        self.intents[intent.intent_id] = intent
        self.history.append(intent.intent_id)
        
        # Trim history
        while len(self.history) > self.max_history:
            old_id = self.history.pop(0)
            self.intents.pop(old_id, None)
            self.reports.pop(old_id, None)
        
        logger.info(f"Intent recorded: {intent.intent_id} | {intent.asset} | "
                   f"{intent.intent_type.name} | target={intent.target_exposure:.2f} | "
                   f"authority={intent.authority.name}")
    
    def record_execution_report(self, report: IntentExecutionReport):
        """Record execution outcome for an intent."""
        self.reports[report.intent_id] = report
        
        if report.is_success():
            logger.info(f"Intent {report.intent_id} executed: "
                       f"fill_rate={report.fill_rate:.1%}, slippage={report.avg_slippage_bps:.1f}bps")
        else:
            logger.warning(f"Intent {report.intent_id} failed: "
                          f"{report.outcome.name} - {report.failure_reason}")
    
    def get_execution_stats(self, lookback: int = 100) -> Dict:
        """Get execution performance stats."""
        recent_ids = self.history[-lookback:]
        
        total = 0
        success = 0
        total_slippage = 0.0
        total_time = 0.0
        
        for intent_id in recent_ids:
            report = self.reports.get(intent_id)
            if report:
                total += 1
                if report.is_success():
                    success += 1
                    total_slippage += report.avg_slippage_bps
                    total_time += report.execution_time_ms
        
        return {
            "total_intents": total,
            "success_rate": success / total if total > 0 else 0,
            "avg_slippage_bps": total_slippage / success if success > 0 else 0,
            "avg_execution_time_ms": total_time / success if success > 0 else 0
        }


# Global journal instance
_intent_journal = IntentJournal()

def get_intent_journal() -> IntentJournal:
    return _intent_journal

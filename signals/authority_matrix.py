"""
HMATS v3.2 - Authority Matrix
==============================
Implements Strategy Constitution Section 7.

Authority Roles:
- DECIDE: Can set direction and/or exposure
- TRIGGER: Can change system mode
- VETO: Can force flat or block a direction
- ADVISE: Provides information but no direct control
- CAP: Can limit exposure magnitude

Hard Rules:
- Sentiment/Macro are NEVER linear multipliers
- Lead-Lag CANNOT set direction (only TRIGGER + exec mode)
- VETO > DECIDE > TRIGGER > CAP > ADVISE
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto
import logging

from orchestration.system_mode import SystemMode

logger = logging.getLogger(__name__)


class AuthorityRole(Enum):
    """Authority roles in precedence order."""
    VETO = 1      # Highest - can force flat
    DECIDE = 2    # Can set direction/exposure
    TRIGGER = 3   # Can change mode
    CAP = 4       # Can limit magnitude
    ADVISE = 5    # No direct control (lowest)


class Agent(Enum):
    """Trading agents in the system."""
    QUANT_BASE = "Quant(Base)"
    QUANT_AGGRESSIVE = "Quant(Aggressive)"
    DRL = "DRL"
    LEAD_LAG = "LeadLag"
    SENTIMENT = "Sentiment"
    MACRO = "Macro"


@dataclass
class AgentSignal:
    """Signal from an agent."""
    agent: Agent
    direction: float = 0.0  # -1 to 1
    confidence: float = 0.0  # 0 to 1
    
    # Agent-specific data
    entropy: float = 0.0  # For DRL
    shock_sigma: float = 0.0  # For Sentiment
    risk_appetite: float = 0.5  # For Macro
    leverage_cap: float = 1.0  # For Macro
    
    def is_bullish(self) -> bool:
        return self.direction > 0.3 and self.confidence > 0.5
    
    def is_bearish(self) -> bool:
        return self.direction < -0.3 and self.confidence > 0.5
    
    def is_neutral(self) -> bool:
        return abs(self.direction) <= 0.3 or self.confidence <= 0.5


@dataclass
class VetoResult:
    """Result of a veto check."""
    is_active: bool = False
    blocked_direction: Optional[str] = None  # "long", "short", None
    reason: str = ""
    agent: Optional[Agent] = None


@dataclass
class CapResult:
    """Result of a cap check."""
    is_active: bool = False
    max_exposure: float = 1.0
    leverage_multiplier: float = 1.0
    reason: str = ""
    agent: Optional[Agent] = None


@dataclass
class AuthorityDecision:
    """Decision from authority matrix."""
    deciding_agent: Agent
    direction: float
    exposure: float
    confidence: float
    
    # Applied modifications
    veto_applied: Optional[VetoResult] = None
    cap_applied: Optional[CapResult] = None
    
    # Trace
    authority_chain: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "deciding_agent": self.deciding_agent.value,
            "direction": self.direction,
            "exposure": self.exposure,
            "confidence": self.confidence,
            "veto_applied": self.veto_applied.reason if self.veto_applied and self.veto_applied.is_active else None,
            "cap_applied": self.cap_applied.reason if self.cap_applied and self.cap_applied.is_active else None,
            "authority_chain": self.authority_chain
        }


class AuthorityMatrix:
    """
    Authority matrix for decision making.
    
    From Strategy Constitution Section 7.
    
    Matrix (Agent × Mode -> Role):
    
    | Agent            | NORMAL        | OPPORTUNITY   | NO_TRADE |
    |------------------|---------------|---------------|----------|
    | Quant (Base)     | DECIDE        | DECIDE        | ADVISE   |
    | Quant (Aggressive)| ADVISE       | DECIDE        | ADVISE   |
    | DRL              | ADVISE/DECIDE*| DECIDE        | ADVISE   |
    | Lead-Lag         | TRIGGER       | TRIGGER       | ADVISE   |
    | Sentiment        | TRIGGER,VETO  | TRIGGER,VETO  | VETO     |
    | Macro            | CAP           | CAP           | VETO     |
    
    *DRL can DECIDE in NORMAL only when entropy < 0.5 AND regime is BULL/STRONG_BULL
    """
    
    # Authority matrix definition
    MATRIX = {
        Agent.QUANT_BASE: {
            SystemMode.NORMAL: [AuthorityRole.DECIDE],
            SystemMode.OPPORTUNITY: [AuthorityRole.DECIDE],
            SystemMode.NO_TRADE: [AuthorityRole.ADVISE]
        },
        Agent.QUANT_AGGRESSIVE: {
            SystemMode.NORMAL: [AuthorityRole.ADVISE],
            SystemMode.OPPORTUNITY: [AuthorityRole.DECIDE],
            SystemMode.NO_TRADE: [AuthorityRole.ADVISE]
        },
        Agent.DRL: {
            SystemMode.NORMAL: [AuthorityRole.ADVISE],  # Upgraded conditionally
            SystemMode.OPPORTUNITY: [AuthorityRole.DECIDE],
            SystemMode.NO_TRADE: [AuthorityRole.ADVISE]
        },
        Agent.LEAD_LAG: {
            SystemMode.NORMAL: [AuthorityRole.TRIGGER],
            SystemMode.OPPORTUNITY: [AuthorityRole.TRIGGER],
            SystemMode.NO_TRADE: [AuthorityRole.ADVISE]
        },
        Agent.SENTIMENT: {
            SystemMode.NORMAL: [AuthorityRole.TRIGGER, AuthorityRole.VETO],
            SystemMode.OPPORTUNITY: [AuthorityRole.TRIGGER, AuthorityRole.VETO],
            SystemMode.NO_TRADE: [AuthorityRole.VETO]
        },
        Agent.MACRO: {
            SystemMode.NORMAL: [AuthorityRole.CAP],
            SystemMode.OPPORTUNITY: [AuthorityRole.CAP],
            SystemMode.NO_TRADE: [AuthorityRole.VETO]
        }
    }
    
    # DRL upgrade conditions
    DRL_UPGRADE_ENTROPY_MAX = 0.5
    DRL_UPGRADE_REGIMES = ["BULL", "STRONG_BULL"]
    
    # Sentiment veto threshold
    SENTIMENT_VETO_SIGMA = 3.0  # Extreme shock enables veto
    
    # Macro risk thresholds
    MACRO_RISK_OFF_THRESHOLD = 0.3
    
    def __init__(self):
        self.last_decision: Optional[AuthorityDecision] = None
    
    def get_roles(self, agent: Agent, mode: SystemMode) -> List[AuthorityRole]:
        """Get roles for an agent in a mode."""
        return self.MATRIX.get(agent, {}).get(mode, [AuthorityRole.ADVISE])
    
    def resolve(
        self,
        mode: SystemMode,
        signals: Dict[Agent, AgentSignal],
        regime_name: str = "NEUTRAL",
        crack_direction: Optional[str] = None
    ) -> AuthorityDecision:
        """
        Resolve authority and produce decision.
        
        Precedence: VETO > DECIDE > TRIGGER > CAP > ADVISE
        
        Args:
            mode: Current system mode
            signals: Dict of agent -> signal
            regime_name: Current regime name (for DRL upgrade check)
            crack_direction: CRACK window direction if active
        
        Returns:
            AuthorityDecision with resolved direction and exposure
        """
        
        authority_chain = []
        
        # === Step 1: Check for VETO ===
        veto = self._check_vetos(mode, signals)
        if veto.is_active:
            authority_chain.append(f"VETO by {veto.agent.value}: {veto.reason}")
        
        # === Step 2: Find DECIDE agents ===
        deciders = self._get_deciders(mode, signals, regime_name)
        authority_chain.append(f"Eligible deciders: {[d.value for d in deciders]}")
        
        # === Step 3: Select primary decider ===
        primary_decider, direction, confidence = self._select_primary_decider(
            deciders, signals, crack_direction
        )
        authority_chain.append(f"Primary decider: {primary_decider.value}")
        
        # === Step 4: Calculate exposure ===
        exposure = abs(direction) * signals[primary_decider].confidence
        
        # === Step 5: Apply VETO if active ===
        if veto.is_active:
            if veto.blocked_direction == "long" and direction > 0:
                direction = 0
                exposure = 0
                authority_chain.append("VETO blocked long direction")
            elif veto.blocked_direction == "short" and direction < 0:
                direction = 0
                exposure = 0
                authority_chain.append("VETO blocked short direction")
        
        # === Step 6: Apply CAP (direction-aware for RISK_OFF) ===
        cap = self._check_caps(mode, signals, direction=direction)
        if cap.is_active:
            exposure = min(exposure, cap.max_exposure)
            exposure *= cap.leverage_multiplier
            authority_chain.append(f"CAP applied: max={cap.max_exposure:.2f}, mult={cap.leverage_multiplier:.2f}")
        
        # === Step 7: Build decision ===
        decision = AuthorityDecision(
            deciding_agent=primary_decider,
            direction=direction,
            exposure=exposure,
            confidence=confidence,
            veto_applied=veto if veto.is_active else None,
            cap_applied=cap if cap.is_active else None,
            authority_chain=authority_chain
        )
        
        self.last_decision = decision
        
        logger.info(f"Authority decision: {primary_decider.value} -> dir={direction:.2f}, exp={exposure:.2f}")
        
        return decision
    
    def _check_vetos(self, mode: SystemMode, signals: Dict[Agent, AgentSignal]) -> VetoResult:
        """Check for active vetos from Sentiment and Macro."""
        
        # Sentiment VETO (extreme shock)
        if Agent.SENTIMENT in signals:
            sentiment = signals[Agent.SENTIMENT]
            if abs(sentiment.shock_sigma) >= self.SENTIMENT_VETO_SIGMA:
                blocked = "short" if sentiment.shock_sigma > 0 else "long"
                return VetoResult(
                    is_active=True,
                    blocked_direction=blocked,
                    reason=f"Sentiment extreme shock {sentiment.shock_sigma:.1f}σ",
                    agent=Agent.SENTIMENT
                )
        
        # Macro VETO (only in NO_TRADE mode, or extreme risk-off)
        if Agent.MACRO in signals:
            macro = signals[Agent.MACRO]
            if mode == SystemMode.NO_TRADE:
                return VetoResult(
                    is_active=True,
                    blocked_direction=None,  # Block both
                    reason="NO_TRADE mode - Macro veto",
                    agent=Agent.MACRO
                )
        
        return VetoResult(is_active=False)
    
    def _check_caps(self, mode: SystemMode, signals: Dict[Agent, AgentSignal],
                    direction: float = 0) -> CapResult:
        """Check for active caps from Macro (direction-aware for RISK_OFF)."""

        if Agent.MACRO not in signals:
            return CapResult(is_active=False)

        macro = signals[Agent.MACRO]

        # Risk-off: direction-aware leverage (FIX5)
        # Shorts thrive in RISK_OFF (flight-to-quality), longs get penalized
        if macro.risk_appetite < self.MACRO_RISK_OFF_THRESHOLD:
            if direction < 0:
                # Shorts: RISK_OFF is the best shorting environment - boost leverage
                _mult = 1.2
                _reason = f"Macro risk-off SHORT BOOST: appetite={macro.risk_appetite:.2f}, mult=1.2x"
            else:
                # Longs (or flat): penalize heavily during risk-off
                _mult = 0.3
                _reason = f"Macro risk-off LONG PENALTY: appetite={macro.risk_appetite:.2f}, mult=0.3x"
            return CapResult(
                is_active=True,
                max_exposure=1.0,
                leverage_multiplier=_mult,
                reason=_reason,
                agent=Agent.MACRO
            )
        
        # Normal cap
        if macro.leverage_cap < 1.0:
            return CapResult(
                is_active=True,
                max_exposure=1.0,
                leverage_multiplier=macro.leverage_cap,
                reason=f"Macro leverage cap: {macro.leverage_cap:.2f}",
                agent=Agent.MACRO
            )
        
        return CapResult(is_active=False)
    
    def _get_deciders(
        self,
        mode: SystemMode,
        signals: Dict[Agent, AgentSignal],
        regime_name: str
    ) -> List[Agent]:
        """Get list of agents with DECIDE authority."""
        
        deciders = []
        
        for agent, roles_by_mode in self.MATRIX.items():
            roles = roles_by_mode.get(mode, [])
            
            if AuthorityRole.DECIDE in roles:
                # Check DRL upgrade condition
                if agent == Agent.DRL and mode == SystemMode.NORMAL:
                    if self._drl_can_decide(signals.get(Agent.DRL), regime_name):
                        deciders.append(agent)
                else:
                    deciders.append(agent)
        
        return deciders
    
    def _drl_can_decide(self, drl_signal: Optional[AgentSignal], regime_name: str) -> bool:
        """Check if DRL can upgrade to DECIDE in NORMAL mode."""
        
        if drl_signal is None:
            return False
        
        # Must have low entropy
        if drl_signal.entropy >= self.DRL_UPGRADE_ENTROPY_MAX:
            return False
        
        # Must be in bullish regime
        if regime_name not in self.DRL_UPGRADE_REGIMES:
            return False
        
        return True
    
    def _select_primary_decider(
        self,
        deciders: List[Agent],
        signals: Dict[Agent, AgentSignal],
        crack_direction: Optional[str]
    ) -> Tuple[Agent, float, float]:
        """
        Select primary decider from eligible deciders.
        
        Priority:
        1. In CRACK window: agent aligned with CRACK direction
        2. Highest confidence
        3. Quant (Base) as default
        """
        
        if not deciders:
            # Fallback to Quant Base
            quant = signals.get(Agent.QUANT_BASE, AgentSignal(agent=Agent.QUANT_BASE))
            return Agent.QUANT_BASE, quant.direction, quant.confidence
        
        # In CRACK window, prefer aligned agent
        if crack_direction:
            for agent in deciders:
                signal = signals.get(agent)
                if signal:
                    if crack_direction == "long" and signal.is_bullish():
                        return agent, signal.direction, signal.confidence
                    elif crack_direction == "short" and signal.is_bearish():
                        return agent, signal.direction, signal.confidence
        
        # Select by confidence
        best_agent = None
        best_confidence = -1
        best_direction = 0
        
        for agent in deciders:
            signal = signals.get(agent)
            if signal and signal.confidence > best_confidence:
                best_agent = agent
                best_confidence = signal.confidence
                best_direction = signal.direction
        
        if best_agent:
            return best_agent, best_direction, best_confidence
        
        # Fallback
        quant = signals.get(Agent.QUANT_BASE, AgentSignal(agent=Agent.QUANT_BASE))
        return Agent.QUANT_BASE, quant.direction, quant.confidence
    
    def validate_lead_lag_constraint(self, action: str) -> bool:
        """
        Validate that Lead-Lag is not setting direction.
        
        CRITICAL: Lead-Lag can NEVER set direction.
        Returns False if constraint violated.
        """
        
        forbidden_actions = ["set_direction", "set_exposure", "decide"]
        return action.lower() not in forbidden_actions
    
    def validate_sentiment_not_multiplier(self, usage: str) -> bool:
        """
        Validate that Sentiment is not used as linear multiplier.
        
        CRITICAL: Sentiment/Macro are NEVER linear multipliers.
        Returns False if constraint violated.
        """
        
        forbidden_usages = ["multiply", "scale", "factor", "linear"]
        return not any(f in usage.lower() for f in forbidden_usages)
    
    def get_matrix_summary(self) -> Dict:
        """Get summary of authority matrix."""
        summary = {}
        for agent in Agent:
            summary[agent.value] = {}
            for mode in SystemMode:
                roles = self.get_roles(agent, mode)
                summary[agent.value][mode.name] = [r.name for r in roles]
        return summary


# Singleton instance
_authority_matrix = AuthorityMatrix()

def get_authority_matrix() -> AuthorityMatrix:
    return _authority_matrix

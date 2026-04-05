"""
================================================================================
HMATS v6.2.3 - PARTIAL CONSENSUS ENTRY
================================================================================
Purpose: Allow reduced-size T1 entries when partial consensus exists

DEFINITION:
    Partial consensus exists when:
    1. The decider agent has a direction (!=0)
    2. At least ONE core agent (DECIDE or CONFIRM) agrees with direction
    3. NO agent with CONFIRM or VETO authority explicitly OPPOSES the direction
    4. Some agents may abstain (direction=0 is neutral, not opposition)

CONSTRAINTS (STRICTLY ENFORCED):
    OFF Does NOT bypass: veto logic, risk checks, trade gate, stop logic
    OFF Does NOT enable: Gambler Mode, leverage increase, max exposure increase
    OFF Does NOT allow: T2/T3/T4 escalation (T1 only, non-upgradeable)
    ON ONLY allows: Reduced position (30-50% of normal T1)

DISABLE CONDITIONS (any one triggers disable):
    - Correlation controls active (cap level != NORMAL)
    - Cascade exhaustion active (phase != NONE)
    - Volatility guard breached (DVOL zscore >= warning threshold)

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PartialConsensusResult:
    """Result of partial consensus check."""
    
    # Core result
    is_partial_consensus: bool = False
    
    # Scale factor to apply (0.3 to 0.5, or 1.0 if not partial)
    scale_factor: float = 1.0
    
    # Which agents participated in the consensus
    aligned_agents: List[str] = field(default_factory=list)
    neutral_agents: List[str] = field(default_factory=list)
    
    # Why partial consensus (or why not)
    reason: str = ""
    
    # Disable condition that blocked partial consensus
    disabled_by: Optional[str] = None
    
    # Flags for logging
    flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "is_partial_consensus": self.is_partial_consensus,
            "scale_factor": self.scale_factor,
            "aligned_agents": self.aligned_agents,
            "neutral_agents": self.neutral_agents,
            "reason": self.reason,
            "disabled_by": self.disabled_by,
            "flags": self.flags,
        }


@dataclass 
class DisableConditions:
    """Current state of disable conditions for partial consensus."""
    
    # Correlation control state
    correlation_cap_level: str = "NORMAL"  # NORMAL, ELEVATED, HIGH_CORR, CRISIS
    correlation_score: float = 0.0
    
    # Cascade exhaustion state  
    cascade_phase: str = "NONE"  # NONE, DETECTED, ACCELERATING, EXHAUSTING, STOP
    
    # Volatility state
    dvol_zscore: float = 0.0
    dvol_warning_threshold: float = 2.0
    
    def is_any_active(self) -> Tuple[bool, str]:
        """
        Check if any disable condition is active.
        
        Returns:
            (is_disabled, reason)
        """
        # Check correlation
        if self.correlation_cap_level not in ["NORMAL"]:
            return True, f"CORRELATION_CONTROLS_ACTIVE:{self.correlation_cap_level}"
        
        # Check cascade
        if self.cascade_phase not in ["NONE", "STOP"]:
            return True, f"CASCADE_ACTIVE:{self.cascade_phase}"
        
        # Check DVOL
        if self.dvol_zscore >= self.dvol_warning_threshold:
            return True, f"VOLATILITY_ELEVATED:zscore={self.dvol_zscore:.2f}"
        
        return False, ""


# =============================================================================
# CORE AGENTS (for consensus checking)
# =============================================================================

# These are the agents that matter for consensus
# DECIDE/CONFIRM authorities determine the direction
CORE_AGENTS = ["quant", "regime", "two_stage", "model_alpha"]

# Agents that can veto (their opposition matters)
VETO_CAPABLE_AGENTS = ["sentiment", "risk"]


# =============================================================================
# PARTIAL CONSENSUS CHECKER
# =============================================================================

class PartialConsensusChecker:
    """
    Checks for partial consensus conditions.
    
    SAFETY PRINCIPLES:
    1. Conservative by default (partial consensus = smaller position, not larger)
    2. Any opposition = no partial consensus
    3. Disable conditions ALWAYS respected
    4. Never bypasses veto logic
    """
    
    def __init__(self):
        self._check_count = 0
        self._partial_count = 0
        self._disabled_count = 0
        
        # Load configuration
        self._load_config()
        
        logger.info("[PARTIAL_CONSENSUS] Checker initialized")
    
    def _load_config(self):
        """Load configuration from sota_flags."""
        try:
            from configs.sota_flags import get_sota_flags
            flags = get_sota_flags()
            self.enabled = flags.ENABLE_PARTIAL_CONSENSUS_ENTRY
            self.max_scale = flags.PARTIAL_CONSENSUS_MAX_SCALE
            self.min_scale = flags.PARTIAL_CONSENSUS_MIN_SCALE
            self.min_confidence = flags.PARTIAL_CONSENSUS_MIN_CONFIDENCE
        except (ImportError, AttributeError):
            # Conservative defaults
            self.enabled = False
            self.max_scale = 0.5
            self.min_scale = 0.3
            self.min_confidence = 0.6
    
    def check(
        self,
        decider_agent: str,
        decider_direction: float,
        decider_confidence: float,
        signals: Dict[str, Any],  # agent_name -> AgentSignal
        authority_matrix: Dict[str, Any],  # agent_name -> Authority
        disable_conditions: Optional[DisableConditions] = None,
    ) -> PartialConsensusResult:
        """
        Check if partial consensus conditions are met.
        
        Args:
            decider_agent: Name of the DECIDE authority agent
            decider_direction: Direction from decider (-1 to 1)
            decider_confidence: Confidence from decider (0 to 1)
            signals: Dict of agent signals
            authority_matrix: Current authority matrix
            disable_conditions: Current disable condition states
            
        Returns:
            PartialConsensusResult with decision and metadata
        """
        self._check_count += 1
        
        result = PartialConsensusResult()
        
        # =================================================================
        # CHECK 0: Feature enabled?
        # =================================================================
        
        if not self.enabled:
            result.reason = "PARTIAL_CONSENSUS_DISABLED"
            return result
        
        # =================================================================
        # CHECK 1: Decider has direction?
        # =================================================================
        
        if abs(decider_direction) < 0.1:
            result.reason = "DECIDER_NEUTRAL"
            return result
        
        # =================================================================
        # CHECK 2: Check disable conditions FIRST (fail-closed)
        # =================================================================
        
        if disable_conditions:
            is_disabled, disable_reason = disable_conditions.is_any_active()
            if is_disabled:
                result.disabled_by = disable_reason
                result.reason = f"DISABLED:{disable_reason}"
                self._disabled_count += 1
                return result
        
        # =================================================================
        # CHECK 3: Analyze agent signals for consensus
        # =================================================================
        
        aligned = []
        neutral = []
        opposing = []
        
        direction_sign = 1 if decider_direction > 0 else -1
        
        for agent_name, signal in signals.items():
            if agent_name == decider_agent:
                continue  # Skip decider itself
            
            # Get signal direction
            agent_dir = getattr(signal, 'direction', 0.0)
            agent_confidence = getattr(signal, 'confidence', 0.0)
            
            # Get authority for this agent
            authority = authority_matrix.get(agent_name)
            if authority is None:
                continue
            
            authority_str = authority.value if hasattr(authority, 'value') else str(authority)
            
            # Check if this agent matters for consensus
            is_core = authority_str in ["DECIDE", "CONFIRM"]
            is_veto_capable = authority_str == "VETO"
            
            if not is_core and not is_veto_capable:
                continue  # Skip ADVISE, CAP, TRIGGER, EXECUTE agents
            
            # Classify the agent's stance
            if abs(agent_dir) < 0.1:
                # Neutral / abstaining
                neutral.append(agent_name)
            elif (agent_dir > 0) == (direction_sign > 0):
                # Same direction as decider
                if agent_confidence >= self.min_confidence:
                    aligned.append(agent_name)
                else:
                    neutral.append(agent_name)  # Low confidence = effectively neutral
            else:
                # OPPOSING direction
                opposing.append(agent_name)
        
        # =================================================================
        # CHECK 4: Any opposition?
        # =================================================================
        
        if opposing:
            result.reason = f"OPPOSITION_EXISTS:agents={opposing}"
            result.flags.append("HAS_OPPOSITION")
            return result
        
        # =================================================================
        # CHECK 5: At least one aligned agent?
        # =================================================================
        
        if not aligned:
            result.reason = "NO_ALIGNED_AGENTS"
            return result
        
        # =================================================================
        # CHECK 6: Is this actually partial consensus?
        # Full consensus = all CONFIRM agents aligned
        # Partial = at least one aligned, rest neutral
        # =================================================================
        
        # Count CONFIRM authorities
        confirm_count = 0
        confirm_aligned = 0
        
        for agent_name in signals.keys():
            authority = authority_matrix.get(agent_name)
            if authority is None:
                continue
            authority_str = authority.value if hasattr(authority, 'value') else str(authority)
            
            if authority_str == "CONFIRM":
                confirm_count += 1
                if agent_name in aligned:
                    confirm_aligned += 1
        
        # If all CONFIRM agents are aligned, this is FULL consensus, not partial
        if confirm_count > 0 and confirm_aligned == confirm_count:
            result.reason = "FULL_CONSENSUS"
            result.aligned_agents = aligned
            # Full consensus = normal sizing
            return result
        
        # =================================================================
        # SUCCESS: Partial consensus detected
        # =================================================================
        
        result.is_partial_consensus = True
        result.aligned_agents = aligned
        result.neutral_agents = neutral
        
        # Calculate scale factor based on alignment
        # More aligned agents = higher scale (up to max_scale)
        alignment_ratio = len(aligned) / max(1, len(aligned) + len(neutral))
        base_scale = self.min_scale + (alignment_ratio * (self.max_scale - self.min_scale))
        
        # Confidence adjustment: higher confidence = higher scale
        confidence_factor = min(1.0, decider_confidence / 0.8)  # 0.8+ = full factor
        result.scale_factor = base_scale * confidence_factor
        
        # Clamp to bounds
        result.scale_factor = max(self.min_scale, min(self.max_scale, result.scale_factor))
        
        result.reason = f"PARTIAL_CONSENSUS:aligned={aligned},neutral={neutral},scale={result.scale_factor:.2f}"
        result.flags.append("PARTIAL_CONSENSUS_ENTRY")
        result.flags.append("T1_ONLY")
        result.flags.append(f"SCALE_{int(result.scale_factor * 100)}PCT")
        
        self._partial_count += 1
        
        # Log to Shadow Ledger for audit trail
        self._log_to_shadow_ledger(
            decider_agent=decider_agent,
            decider_direction=decider_direction,
            decider_confidence=decider_confidence,
            aligned=aligned,
            neutral=neutral,
            scale_factor=result.scale_factor,
        )
        
        logger.info(
            f"[PARTIAL_CONSENSUS] DETECTED: scale={result.scale_factor:.2f}, "
            f"aligned={aligned}, neutral={neutral}"
        )
        
        return result
    
    def _log_to_shadow_ledger(
        self,
        decider_agent: str,
        decider_direction: float,
        decider_confidence: float,
        aligned: List[str],
        neutral: List[str],
        scale_factor: float,
    ):
        """Log partial consensus entry to ShadowLedger for audit trail."""
        try:
            from data_mgmt.shadow_ledger_manager import get_shadow_ledger
            ledger = get_shadow_ledger()
            if ledger:
                ledger.log_veto(
                    source="PARTIAL_CONSENSUS",
                    reason="PARTIAL_CONSENSUS_ENTRY",
                    details={
                        "decider_agent": decider_agent,
                        "decider_direction": decider_direction,
                        "decider_confidence": decider_confidence,
                        "aligned_agents": aligned,
                        "neutral_agents": neutral,
                        "scale_factor": scale_factor,
                        "tranche_limit": "T1_ONLY",
                        "escalation_blocked": True,
                    }
                )
        except (ImportError, Exception) as e:
            logger.debug(f"Shadow ledger logging skipped: {e}")
    
    def get_disable_conditions(
        self,
        correlation_cap_level: str = "NORMAL",
        correlation_score: float = 0.0,
        cascade_phase: str = "NONE",
        dvol_zscore: float = 0.0,
        dvol_warning_threshold: float = 2.0,
    ) -> DisableConditions:
        """Helper to create DisableConditions from individual parameters."""
        return DisableConditions(
            correlation_cap_level=correlation_cap_level,
            correlation_score=correlation_score,
            cascade_phase=cascade_phase,
            dvol_zscore=dvol_zscore,
            dvol_warning_threshold=dvol_warning_threshold,
        )
    
    def get_stats(self) -> Dict:
        """Get checker statistics."""
        return {
            "enabled": self.enabled,
            "total_checks": self._check_count,
            "partial_consensus_triggered": self._partial_count,
            "disabled_by_conditions": self._disabled_count,
            "partial_rate": (
                self._partial_count / max(1, self._check_count)
            ),
            "config": {
                "max_scale": self.max_scale,
                "min_scale": self.min_scale,
                "min_confidence": self.min_confidence,
            }
        }


# =============================================================================
# SINGLETON
# =============================================================================

_partial_consensus_checker: Optional[PartialConsensusChecker] = None


def get_partial_consensus_checker() -> PartialConsensusChecker:
    """Get or create singleton instance."""
    global _partial_consensus_checker
    if _partial_consensus_checker is None:
        _partial_consensus_checker = PartialConsensusChecker()
    return _partial_consensus_checker


def reset_partial_consensus_checker():
    """Reset singleton (for testing)."""
    global _partial_consensus_checker
    _partial_consensus_checker = None


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def check_partial_consensus(
    decider_agent: str,
    decider_direction: float,
    decider_confidence: float,
    signals: Dict[str, Any],
    authority_matrix: Dict[str, Any],
    disable_conditions: Optional[DisableConditions] = None,
) -> PartialConsensusResult:
    """Convenience function for partial consensus check."""
    checker = get_partial_consensus_checker()
    return checker.check(
        decider_agent=decider_agent,
        decider_direction=decider_direction,
        decider_confidence=decider_confidence,
        signals=signals,
        authority_matrix=authority_matrix,
        disable_conditions=disable_conditions,
    )


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PARTIAL CONSENSUS CHECKER TEST")
    print("=" * 60)
    
    # Create mock signals
    class MockSignal:
        def __init__(self, direction, confidence):
            self.direction = direction
            self.confidence = confidence
    
    class MockAuthority:
        def __init__(self, value):
            self.value = value
    
    checker = PartialConsensusChecker()
    checker.enabled = True  # Enable for testing
    
    # Test 1: Full consensus
    print("\n[Test 1] Full consensus")
    signals = {
        "quant": MockSignal(0.8, 0.9),
        "regime": MockSignal(0.6, 0.8),
        "two_stage": MockSignal(0.5, 0.7),
    }
    matrix = {
        "quant": MockAuthority("DECIDE"),
        "regime": MockAuthority("CONFIRM"),
        "two_stage": MockAuthority("CONFIRM"),
    }
    result = checker.check("quant", 0.8, 0.9, signals, matrix)
    print(f"  Result: partial={result.is_partial_consensus}, reason={result.reason}")
    
    # Test 2: Partial consensus (one neutral)
    print("\n[Test 2] Partial consensus (one neutral)")
    signals = {
        "quant": MockSignal(0.8, 0.9),
        "regime": MockSignal(0.6, 0.8),
        "two_stage": MockSignal(0.0, 0.2),  # Neutral
    }
    result = checker.check("quant", 0.8, 0.9, signals, matrix)
    print(f"  Result: partial={result.is_partial_consensus}, scale={result.scale_factor:.2f}, reason={result.reason}")
    
    # Test 3: Opposition exists
    print("\n[Test 3] Opposition exists")
    signals = {
        "quant": MockSignal(0.8, 0.9),
        "regime": MockSignal(-0.3, 0.7),  # Opposing!
        "two_stage": MockSignal(0.0, 0.2),
    }
    result = checker.check("quant", 0.8, 0.9, signals, matrix)
    print(f"  Result: partial={result.is_partial_consensus}, reason={result.reason}")
    
    # Test 4: Disabled by correlation
    print("\n[Test 4] Disabled by correlation")
    signals = {
        "quant": MockSignal(0.8, 0.9),
        "regime": MockSignal(0.0, 0.2),  # Neutral
    }
    disable = DisableConditions(correlation_cap_level="ELEVATED")
    result = checker.check("quant", 0.8, 0.9, signals, matrix, disable)
    print(f"  Result: partial={result.is_partial_consensus}, disabled_by={result.disabled_by}")
    
    print("\n" + "=" * 60)
    print(f"Stats: {checker.get_stats()}")

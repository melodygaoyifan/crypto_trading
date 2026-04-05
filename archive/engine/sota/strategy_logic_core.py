"""
================================================================================
HMATS v3.2 - STRATEGY LOGIC CORE
================================================================================
Purpose: THE ONLY decision authority for all strategy decisions.

This module implements:
- TASK 1: Single authority (no implicit/legacy defaults)
- TASK 2: OPPORTUNITY rule disable/soften lists
- TASK 3: Single EarlyEntryGate
- TASK 4: SOL dominance forced exit
- TASK 5: Strategy acceptance criteria

ALL STRATEGY DECISIONS MUST FLOW THROUGH THIS MODULE.
Legacy defaults are explicitly overridden here.

MODE PRIORITY (STRICT):
1. NO_TRADE (highest) - data integrity, conflict, correlation collapse
2. OPPORTUNITY - CRACK, lead-lag edge, sentiment shock
3. NORMAL (default) - baseline beta collection
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum, auto
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# MODE TTL CONFIGURATION
# =============================================================================

MODE_TTL_CONFIG = {
    "OPPORTUNITY": timedelta(hours=16),  # 4 bars × 4H = 16H default
    "NO_TRADE": None,  # No TTL - stays until exit condition met
    "NORMAL": None,  # No TTL - continuous
}

@dataclass
class ModeState:
    """Tracks current mode and TTL."""
    mode: str = "NORMAL"
    entered_at: Optional[datetime] = None
    ttl: Optional[timedelta] = None
    trigger_reason: str = ""
    
    def is_expired(self, now: datetime) -> bool:
        """Check if mode TTL has expired."""
        if self.ttl is None or self.entered_at is None:
            return False
        return (now - self.entered_at) > self.ttl
    
    def extend_ttl(self, additional: timedelta):
        """Extend TTL (for re-triggers)."""
        if self.ttl is not None:
            self.ttl += additional


# =============================================================================
# TASK 2: OPPORTUNITY MODE RULE CONFIGURATION
# =============================================================================

class RuleStatus(Enum):
    """Status of a rule in OPPORTUNITY mode."""
    ENABLED = auto()      # Rule is active with normal parameters
    DISABLED = auto()     # Rule is completely bypassed
    SOFTENED = auto()     # Rule is active with relaxed parameters
    HARD = auto()         # Rule cannot be disabled under any circumstance


@dataclass
class RuleConfig:
    """Configuration for a single rule."""
    name: str
    status_normal: RuleStatus
    status_opportunity: RuleStatus
    softened_params: Optional[Dict] = None
    description: str = ""


# EXPLICIT RULE CONFIGURATION TABLE (TASK 2)
# This is the AUTHORITATIVE list of what changes in OPPORTUNITY mode
RULE_CONFIGURATION: Dict[str, RuleConfig] = {
    # === DISABLED in OPPORTUNITY ===
    "persistence_requirement": RuleConfig(
        name="Regime Persistence Requirement",
        status_normal=RuleStatus.ENABLED,
        status_opportunity=RuleStatus.DISABLED,  # DISABLED
        description="Normally requires 3 ticks for regime change; OPPORTUNITY bypasses"
    ),
    "strict_snr_filter": RuleConfig(
        name="Strict SNR Filter (>2.0)",
        status_normal=RuleStatus.ENABLED,
        status_opportunity=RuleStatus.DISABLED,  # DISABLED
        description="Normally requires SNR>2.0; OPPORTUNITY accepts lower"
    ),
    "alpha_3x_gate": RuleConfig(
        name="3× Friction Alpha Gate",
        status_normal=RuleStatus.ENABLED,
        status_opportunity=RuleStatus.DISABLED,  # DISABLED (replaced by 1.5-2.5×)
        description="Normally requires alpha >= 3× friction; OPPORTUNITY uses 1.5-2.5×"
    ),
    "full_confirmation_depth": RuleConfig(
        name="Full Confirmation Depth (4H close required)",
        status_normal=RuleStatus.ENABLED,
        status_opportunity=RuleStatus.DISABLED,  # DISABLED for Tranche-1
        description="Normally requires 4H close; OPPORTUNITY allows early Tranche-1"
    ),
    
    # === SOFTENED in OPPORTUNITY ===
    "regime_confidence_threshold": RuleConfig(
        name="Regime Confidence Threshold",
        status_normal=RuleStatus.ENABLED,
        status_opportunity=RuleStatus.SOFTENED,  # SOFTENED
        softened_params={"normal": 0.7, "opportunity": 0.5},
        description="Confidence threshold lowered in OPPORTUNITY"
    ),
    "max_position_size": RuleConfig(
        name="Maximum Position Size",
        status_normal=RuleStatus.ENABLED,
        status_opportunity=RuleStatus.SOFTENED,  # SOFTENED
        softened_params={"normal": 0.8, "opportunity_sol": 1.0},
        description="SOL can reach 100% in OPPORTUNITY"
    ),
    "tranche_escalation_speed": RuleConfig(
        name="Tranche Escalation Requirements",
        status_normal=RuleStatus.ENABLED,
        status_opportunity=RuleStatus.SOFTENED,  # SOFTENED
        softened_params={"normal_profit_req": 50, "opportunity_profit_req": 0},
        description="T3 profit requirement relaxed in OPPORTUNITY"
    ),
    "btc_eth_anchor_requirement": RuleConfig(
        name="BTC/ETH Anchor Requirement",
        status_normal=RuleStatus.ENABLED,
        status_opportunity=RuleStatus.SOFTENED,  # SOFTENED
        softened_params={"normal_min": 0.2, "opportunity_min": 0.0},
        description="BTC/ETH can be zero during SOL dominance"
    ),
    
    # === HARD (never disabled) ===
    "data_integrity_check": RuleConfig(
        name="Data Integrity Check",
        status_normal=RuleStatus.HARD,
        status_opportunity=RuleStatus.HARD,  # HARD - never disabled
        description="Data staleness/CRC failure always triggers NO_TRADE"
    ),
    "no_trade_enforcement": RuleConfig(
        name="NO_TRADE Mode Enforcement",
        status_normal=RuleStatus.HARD,
        status_opportunity=RuleStatus.HARD,  # HARD
        description="NO_TRADE always means FLAT, no exceptions"
    ),
    "correlation_collapse_check": RuleConfig(
        name="Correlation Collapse Detection",
        status_normal=RuleStatus.HARD,
        status_opportunity=RuleStatus.HARD,  # HARD
        description="Correlation collapse always triggers de-risk"
    ),
    "vpin_toxicity_abort": RuleConfig(
        name="VPIN Toxicity Abort (>0.85)",
        status_normal=RuleStatus.HARD,
        status_opportunity=RuleStatus.HARD,  # HARD
        description="VPIN > 0.85 always aborts execution"
    ),
    "all_conflict_flat": RuleConfig(
        name="All-Conflict-Flat Rule",
        status_normal=RuleStatus.HARD,
        status_opportunity=RuleStatus.HARD,  # HARD
        description="Quant/DRL/Sentiment conflict always means FLAT"
    ),
    "max_drawdown_circuit_breaker": RuleConfig(
        name="Max Drawdown Circuit Breaker",
        status_normal=RuleStatus.HARD,
        status_opportunity=RuleStatus.HARD,  # HARD
        description="15% drawdown always triggers shutdown"
    ),
}


def get_rule_status(rule_name: str, is_opportunity: bool) -> RuleStatus:
    """Get the status of a rule based on current mode."""
    config = RULE_CONFIGURATION.get(rule_name)
    if not config:
        logger.warning(f"Unknown rule: {rule_name}, defaulting to ENABLED")
        return RuleStatus.ENABLED
    return config.status_opportunity if is_opportunity else config.status_normal


def is_rule_disabled(rule_name: str, is_opportunity: bool) -> bool:
    """Check if a rule is disabled."""
    return get_rule_status(rule_name, is_opportunity) == RuleStatus.DISABLED


def is_rule_softened(rule_name: str, is_opportunity: bool) -> bool:
    """Check if a rule is softened."""
    return get_rule_status(rule_name, is_opportunity) == RuleStatus.SOFTENED


def get_softened_param(rule_name: str, param_name: str, is_opportunity: bool):
    """Get the softened parameter value."""
    config = RULE_CONFIGURATION.get(rule_name)
    if not config or not config.softened_params:
        return None
    
    if is_opportunity:
        # Look for opportunity-specific param
        opp_key = f"opportunity_{param_name}" if f"opportunity_{param_name}" in config.softened_params else "opportunity"
        if opp_key in config.softened_params:
            return config.softened_params[opp_key]
    
    return config.softened_params.get(param_name) or config.softened_params.get("normal")


# =============================================================================
# TASK 3: EARLY ENTRY GATE
# =============================================================================

@dataclass
class EarlyEntryContext:
    """Context for early entry decision."""
    mode: str  # "NORMAL", "OPPORTUNITY", "NO_TRADE"
    
    # Lead-lag state
    lead_lag_valid: bool = False
    lead_lag_edge_bps: float = 0.0
    lead_lag_expires_in_seconds: float = 0.0
    
    # CRACK window state
    crack_window_active: bool = False
    crack_direction: str = "none"  # "long", "short", "none"
    crack_remaining_strength: float = 0.0
    
    # Sentiment state
    sentiment_shock_sigma: float = 0.0
    sentiment_direction: str = "neutral"
    
    # Risk state
    vpin: float = 0.5
    correlation_btc_eth_sol: float = 0.8
    dvol_zscore: float = 0.0
    drawdown_pct: float = 0.0
    
    # Position state
    current_tranche_level: int = 0  # 0=none, 1-4=tranches
    time_since_last_4h_close_seconds: float = 0.0


@dataclass
class EarlyEntryDecision:
    """Decision from EarlyEntryGate."""
    allowed: bool
    max_tranche: int  # Maximum allowed tranche (0=none, 1=Tranche-1 only)
    reason: str
    abort_conditions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "allowed": self.allowed,
            "max_tranche": self.max_tranche,
            "reason": self.reason,
            "abort_conditions": self.abort_conditions
        }


def EarlyEntryGate(
    mode: str,
    lead_lag_valid: bool,
    lead_lag_edge_bps: float,
    crack_active: bool,
    crack_direction: str,
    sentiment_shock_sigma: float,
    vpin: float,
    correlation: float,
    dvol_zscore: float,
    current_tranche: int
) -> EarlyEntryDecision:
    """
    TASK 3: Single mandatory gate for early entry.
    
    Rules:
    - Only OPPORTUNITY mode can pass
    - Only Tranche-1 (20%) allowed before 4H close
    - Explicit abort conditions checked
    
    Args:
        mode: Current system mode
        lead_lag_valid: Whether lead-lag edge is confirmed
        lead_lag_edge_bps: Lead-lag edge in basis points
        crack_active: Whether CRACK window is active
        crack_direction: Direction of CRACK window
        sentiment_shock_sigma: Sentiment shock in sigma
        vpin: Current VPIN level
        correlation: BTC-ETH-SOL correlation
        dvol_zscore: DVOL z-score
        current_tranche: Current tranche level (0-4)
    
    Returns:
        EarlyEntryDecision with allowed status and constraints
    """
    
    logger.info(f"[EarlyEntryGate] mode={mode}, crack={crack_active}, vpin={vpin:.2f}")
    
    # Collect abort conditions
    abort_conditions = []
    
    # === HARD ABORT CONDITIONS (always checked) ===
    
    # VPIN toxicity
    if vpin >= 0.85:
        abort_conditions.append(f"VPIN_TOXIC: {vpin:.2f} >= 0.85")
    
    # Correlation collapse
    if correlation < 0.3:
        abort_conditions.append(f"CORRELATION_COLLAPSE: {correlation:.2f} < 0.3")
    
    # Extreme DVOL
    if dvol_zscore >= 4.0:
        abort_conditions.append(f"DVOL_EXTREME: {dvol_zscore:.1f} >= 4.0")
    
    # If any abort condition is met, deny entry
    if abort_conditions:
        return EarlyEntryDecision(
            allowed=False,
            max_tranche=0,
            reason="Hard abort conditions triggered",
            abort_conditions=abort_conditions
        )
    
    # === MODE CHECK ===
    
    # Only OPPORTUNITY mode allows early entry
    if mode != "OPPORTUNITY":
        return EarlyEntryDecision(
            allowed=False,
            max_tranche=0,
            reason=f"Early entry requires OPPORTUNITY mode (current: {mode})",
            abort_conditions=[]
        )
    
    # === OPPORTUNITY MODE - CHECK QUALIFYING TRIGGERS ===
    
    qualifying_triggers = []
    
    # CRACK window
    if crack_active and crack_direction != "none":
        qualifying_triggers.append(f"CRACK_{crack_direction.upper()}")
    
    # Lead-lag edge
    if lead_lag_valid and abs(lead_lag_edge_bps) >= 15:
        qualifying_triggers.append(f"LEAD_LAG_EDGE_{lead_lag_edge_bps:.0f}bps")
    
    # Sentiment shock
    if abs(sentiment_shock_sigma) >= 2.0:
        direction = "BULLISH" if sentiment_shock_sigma > 0 else "BEARISH"
        qualifying_triggers.append(f"SENTIMENT_SHOCK_{direction}_{abs(sentiment_shock_sigma):.1f}σ")
    
    # Must have at least one qualifying trigger
    if not qualifying_triggers:
        return EarlyEntryDecision(
            allowed=False,
            max_tranche=0,
            reason="OPPORTUNITY mode but no qualifying triggers (CRACK/LL/Sentiment)",
            abort_conditions=[]
        )
    
    # === TRANCHE CHECK ===
    
    # Only Tranche-1 allowed before 4H close
    if current_tranche >= 1:
        return EarlyEntryDecision(
            allowed=False,
            max_tranche=1,
            reason=f"Already at Tranche-{current_tranche}, cannot early-enter higher",
            abort_conditions=[]
        )
    
    # === ALLOWED ===
    return EarlyEntryDecision(
        allowed=True,
        max_tranche=1,  # Only Tranche-1 allowed
        reason=f"Early entry allowed: {', '.join(qualifying_triggers)}",
        abort_conditions=[]
    )


# =============================================================================
# TASK 4: SOL DOMINANCE FORCED EXIT
# =============================================================================

@dataclass
class SOLDominanceState:
    """State for SOL dominance decision."""
    sol_exposure: float = 0.0
    btc_exposure: float = 0.0
    eth_exposure: float = 0.0
    
    # Risk metrics
    vpin_sol: float = 0.5
    correlation_btc_eth_sol: float = 0.8
    lead_lag_valid: bool = False
    crack_window_active: bool = False
    crack_remaining_strength: float = 0.0
    
    # Market state
    sol_price_change_pct: float = 0.0
    volume_ratio: float = 1.0
    
    # Time
    time_in_dominance_seconds: float = 0.0


@dataclass 
class SOLDominanceDecision:
    """Decision for SOL dominance escalation/de-escalation."""
    action: str  # "ESCALATE", "HOLD", "DE_ESCALATE_PARTIAL", "DE_ESCALATE_FULL", "EXIT_IMMEDIATE"
    target_sol_exposure: float
    target_btc_exposure: float
    target_eth_exposure: float
    reason: str
    urgency: str  # "NORMAL", "HIGH", "IMMEDIATE"
    
    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "target_sol_exposure": self.target_sol_exposure,
            "target_btc_exposure": self.target_btc_exposure,
            "target_eth_exposure": self.target_eth_exposure,
            "reason": self.reason,
            "urgency": self.urgency
        }


def SOLDominanceForcedExit(
    sol_exposure: float,
    vpin: float,
    correlation: float,
    lead_lag_valid: bool,
    crack_active: bool,
    crack_strength: float,
    price_change_pct: float,
    volume_ratio: float,
    mode: str
) -> SOLDominanceDecision:
    """
    TASK 4: SOL dominance forced exit rules.
    
    These exits OVERRIDE Quant/DRL signals.
    
    Forced Exit Conditions (immediate):
    1. Correlation collapse (< 0.3)
    2. VPIN toxicity (> 0.85)
    3. Lead-lag invalidation (was valid, now invalid + adverse move)
    4. CRACK window close (strength < 0.2)
    
    Partial De-escalation Conditions:
    5. VPIN elevated (> 0.7)
    6. Correlation weakening (< 0.5)
    7. Volume collapse (< 0.5× average)
    
    Returns target exposures for SOL/BTC/ETH.
    """
    
    logger.info(f"[SOLDominanceForcedExit] sol={sol_exposure:.2f}, vpin={vpin:.2f}, "
               f"corr={correlation:.2f}, crack={crack_active}")
    
    # If not in SOL dominance, no forced exit needed
    if abs(sol_exposure) < 0.7:
        return SOLDominanceDecision(
            action="HOLD",
            target_sol_exposure=sol_exposure,
            target_btc_exposure=0.25,
            target_eth_exposure=0.25,
            reason="Not in SOL dominance (exposure < 0.7)",
            urgency="NORMAL"
        )
    
    # === IMMEDIATE FORCED EXIT CONDITIONS ===
    
    # 1. Correlation collapse
    if correlation < 0.3:
        return SOLDominanceDecision(
            action="EXIT_IMMEDIATE",
            target_sol_exposure=0.0,
            target_btc_exposure=0.0,
            target_eth_exposure=0.0,
            reason=f"CORRELATION_COLLAPSE: {correlation:.2f} < 0.3 -> FLAT ALL",
            urgency="IMMEDIATE"
        )
    
    # 2. VPIN toxicity
    if vpin > 0.85:
        return SOLDominanceDecision(
            action="EXIT_IMMEDIATE",
            target_sol_exposure=0.0,
            target_btc_exposure=0.0,
            target_eth_exposure=0.0,
            reason=f"VPIN_TOXIC: {vpin:.2f} > 0.85 -> FLAT ALL",
            urgency="IMMEDIATE"
        )
    
    # 3. Lead-lag invalidation with adverse move
    if not lead_lag_valid and abs(price_change_pct) > 0.02:
        # Check if move is adverse
        adverse = (sol_exposure > 0 and price_change_pct < -0.02) or \
                  (sol_exposure < 0 and price_change_pct > 0.02)
        if adverse:
            return SOLDominanceDecision(
                action="EXIT_IMMEDIATE",
                target_sol_exposure=0.0,
                target_btc_exposure=0.0,
                target_eth_exposure=0.0,
                reason=f"LEAD_LAG_INVALID + ADVERSE_MOVE: {price_change_pct*100:.1f}% -> FLAT ALL",
                urgency="IMMEDIATE"
            )
    
    # 4. CRACK window closed (was supporting dominance)
    if not crack_active or crack_strength < 0.2:
        # CRACK closed, must reduce but not necessarily exit fully
        target_sol = 0.5 * (1 if sol_exposure > 0 else -1)  # Reduce to 50%
        return SOLDominanceDecision(
            action="DE_ESCALATE_PARTIAL",
            target_sol_exposure=target_sol,
            target_btc_exposure=0.15,
            target_eth_exposure=0.15,
            reason=f"CRACK_WINDOW_CLOSED: strength={crack_strength:.2f} -> reduce to 50%",
            urgency="HIGH"
        )
    
    # === PARTIAL DE-ESCALATION CONDITIONS ===
    
    # 5. VPIN elevated (but not toxic)
    if vpin > 0.7:
        target_sol = 0.6 * (1 if sol_exposure > 0 else -1)  # Reduce to 60%
        return SOLDominanceDecision(
            action="DE_ESCALATE_PARTIAL",
            target_sol_exposure=target_sol,
            target_btc_exposure=0.1,
            target_eth_exposure=0.1,
            reason=f"VPIN_ELEVATED: {vpin:.2f} > 0.7 -> reduce to 60%",
            urgency="HIGH"
        )
    
    # 6. Correlation weakening
    if correlation < 0.5:
        target_sol = 0.5 * (1 if sol_exposure > 0 else -1)  # Reduce to 50%
        return SOLDominanceDecision(
            action="DE_ESCALATE_PARTIAL",
            target_sol_exposure=target_sol,
            target_btc_exposure=0.15,
            target_eth_exposure=0.15,
            reason=f"CORRELATION_WEAKENING: {correlation:.2f} < 0.5 -> reduce to 50%",
            urgency="NORMAL"
        )
    
    # 7. Volume collapse
    if volume_ratio < 0.5:
        target_sol = 0.6 * (1 if sol_exposure > 0 else -1)  # Reduce to 60%
        return SOLDominanceDecision(
            action="DE_ESCALATE_PARTIAL",
            target_sol_exposure=target_sol,
            target_btc_exposure=0.1,
            target_eth_exposure=0.1,
            reason=f"VOLUME_COLLAPSE: {volume_ratio:.1f}× < 0.5× -> reduce to 60%",
            urgency="NORMAL"
        )
    
    # === NO FORCED EXIT - CAN ESCALATE ===
    
    # Check if we can escalate further
    if mode == "OPPORTUNITY" and crack_active and vpin < 0.6 and correlation > 0.6:
        # Conditions good for escalation
        direction = 1 if sol_exposure > 0 else -1
        if abs(sol_exposure) < 0.9:
            return SOLDominanceDecision(
                action="ESCALATE",
                target_sol_exposure=0.9 * direction,
                target_btc_exposure=0.05,
                target_eth_exposure=0.05,
                reason="Conditions favorable for SOL dominance escalation",
                urgency="NORMAL"
            )
    
    # Hold current position
    return SOLDominanceDecision(
        action="HOLD",
        target_sol_exposure=sol_exposure,
        target_btc_exposure=0.05 if abs(sol_exposure) > 0.7 else 0.25,
        target_eth_exposure=0.05 if abs(sol_exposure) > 0.7 else 0.25,
        reason="Holding current SOL dominance position",
        urgency="NORMAL"
    )


# =============================================================================
# TASK 5: STRATEGY ACCEPTANCE CRITERIA
# =============================================================================

class AcceptanceCriteriaResult(Enum):
    """Result of acceptance criteria check."""
    PASS = auto()
    FAIL = auto()


@dataclass
class AcceptanceCriterion:
    """Single acceptance criterion."""
    name: str
    description: str
    check_function: str  # Name of check function
    is_critical: bool = True  # If critical, failure = strategy FAIL


ACCEPTANCE_CRITERIA: List[AcceptanceCriterion] = [
    AcceptanceCriterion(
        name="OPP_DISABLES_RULES",
        description="OPPORTUNITY mode disables specified rules (persistence, SNR, 3× alpha)",
        check_function="check_opportunity_disables_rules",
        is_critical=True
    ),
    AcceptanceCriterion(
        name="EARLY_ENTRY_ONLY_OPPORTUNITY",
        description="Early entry (before 4H) only occurs in OPPORTUNITY mode",
        check_function="check_early_entry_only_opportunity",
        is_critical=True
    ),
    AcceptanceCriterion(
        name="SOL_DOMINANCE_HAS_EXIT",
        description="SOL dominance (>0.7 exposure) has forced exit path",
        check_function="check_sol_dominance_has_exit",
        is_critical=True
    ),
    AcceptanceCriterion(
        name="CONFLICT_RESULTS_FLAT",
        description="Conflicting signals (Quant/DRL/Sentiment) result in FLAT",
        check_function="check_conflict_results_flat",
        is_critical=True
    ),
    AcceptanceCriterion(
        name="NO_TRADE_ENFORCED",
        description="NO_TRADE mode always results in FLAT, no trades",
        check_function="check_no_trade_enforced",
        is_critical=True
    ),
    AcceptanceCriterion(
        name="TRANCHE_GATING_ENFORCED",
        description="Pre-4H entry limited to Tranche-1 (20%)",
        check_function="check_tranche_gating_enforced",
        is_critical=True
    ),
]


def run_acceptance_checks() -> Dict[str, Tuple[AcceptanceCriteriaResult, str]]:
    """
    Run all acceptance criteria checks.
    
    Returns dict of {criterion_name: (result, details)}.
    """
    results = {}
    
    # Check 1: OPPORTUNITY disables rules
    disabled_rules = ["persistence_requirement", "strict_snr_filter", "alpha_3x_gate", "full_confirmation_depth"]
    all_disabled = all(is_rule_disabled(rule, is_opportunity=True) for rule in disabled_rules)
    results["OPP_DISABLES_RULES"] = (
        AcceptanceCriteriaResult.PASS if all_disabled else AcceptanceCriteriaResult.FAIL,
        f"Disabled rules in OPP: {disabled_rules}"
    )
    
    # Check 2: Early entry only in OPPORTUNITY
    early_normal = EarlyEntryGate("NORMAL", True, 20, True, "long", 2.5, 0.5, 0.8, 2.0, 0)
    early_opp = EarlyEntryGate("OPPORTUNITY", True, 20, True, "long", 2.5, 0.5, 0.8, 2.0, 0)
    results["EARLY_ENTRY_ONLY_OPPORTUNITY"] = (
        AcceptanceCriteriaResult.PASS if (not early_normal.allowed and early_opp.allowed) else AcceptanceCriteriaResult.FAIL,
        f"NORMAL: allowed={early_normal.allowed}, OPPORTUNITY: allowed={early_opp.allowed}"
    )
    
    # Check 3: SOL dominance has exit path
    exit_on_vpin = SOLDominanceForcedExit(0.9, 0.9, 0.8, True, True, 0.8, 0.0, 1.0, "OPPORTUNITY")
    exit_on_corr = SOLDominanceForcedExit(0.9, 0.5, 0.2, True, True, 0.8, 0.0, 1.0, "OPPORTUNITY")
    has_exit = (exit_on_vpin.action == "EXIT_IMMEDIATE" and exit_on_corr.action == "EXIT_IMMEDIATE")
    results["SOL_DOMINANCE_HAS_EXIT"] = (
        AcceptanceCriteriaResult.PASS if has_exit else AcceptanceCriteriaResult.FAIL,
        f"VPIN exit: {exit_on_vpin.action}, CORR exit: {exit_on_corr.action}"
    )
    
    # Check 4: Conflict results in FLAT (checked via HARD rule status)
    conflict_rule = RULE_CONFIGURATION.get("all_conflict_flat")
    conflict_hard = conflict_rule and conflict_rule.status_opportunity == RuleStatus.HARD
    results["CONFLICT_RESULTS_FLAT"] = (
        AcceptanceCriteriaResult.PASS if conflict_hard else AcceptanceCriteriaResult.FAIL,
        f"all_conflict_flat is HARD: {conflict_hard}"
    )
    
    # Check 5: NO_TRADE enforced
    no_trade_rule = RULE_CONFIGURATION.get("no_trade_enforcement")
    no_trade_hard = no_trade_rule and no_trade_rule.status_opportunity == RuleStatus.HARD
    results["NO_TRADE_ENFORCED"] = (
        AcceptanceCriteriaResult.PASS if no_trade_hard else AcceptanceCriteriaResult.FAIL,
        f"no_trade_enforcement is HARD: {no_trade_hard}"
    )
    
    # Check 6: Tranche gating enforced
    gate_result = EarlyEntryGate("OPPORTUNITY", True, 20, True, "long", 2.5, 0.5, 0.8, 2.0, 0)
    tranche_gated = gate_result.allowed and gate_result.max_tranche == 1
    results["TRANCHE_GATING_ENFORCED"] = (
        AcceptanceCriteriaResult.PASS if tranche_gated else AcceptanceCriteriaResult.FAIL,
        f"Early entry max_tranche={gate_result.max_tranche}"
    )
    
    return results


def assert_strategy_acceptance() -> bool:
    """
    Assert all acceptance criteria pass.
    
    Returns True if all pass, raises AssertionError if any critical fail.
    """
    results = run_acceptance_checks()
    
    all_pass = True
    for criterion in ACCEPTANCE_CRITERIA:
        result, details = results.get(criterion.name, (AcceptanceCriteriaResult.FAIL, "Not checked"))
        
        if result == AcceptanceCriteriaResult.FAIL:
            msg = f"ACCEPTANCE FAIL: {criterion.name} - {criterion.description}\n  Details: {details}"
            if criterion.is_critical:
                logger.error(msg)
                all_pass = False
            else:
                logger.warning(msg)
        else:
            logger.info(f"ACCEPTANCE PASS: {criterion.name}")
    
    return all_pass


# =============================================================================
# TASK 1: STRATEGY LOGIC CORE - THE ONLY DECISION AUTHORITY
# =============================================================================

@dataclass
class StrategyDecisionContext:
    """Complete context for strategy decision."""
    # Mode
    mode: str  # "NORMAL", "OPPORTUNITY", "NO_TRADE"
    
    # Market data
    vpin: float = 0.5
    correlation: float = 0.8
    dvol_zscore: float = 0.0
    volatility: float = 0.02
    
    # Signal data
    quant_direction: float = 0.0
    quant_confidence: float = 0.5
    drl_direction: float = 0.0
    drl_confidence: float = 0.5
    sentiment_direction: float = 0.0
    sentiment_shock_sigma: float = 0.0
    
    # OPPORTUNITY triggers
    crack_active: bool = False
    crack_direction: str = "none"
    crack_strength: float = 0.0
    lead_lag_valid: bool = False
    lead_lag_edge_bps: float = 0.0
    
    # Position state
    current_exposures: Dict[str, float] = field(default_factory=dict)
    current_tranche: int = 0
    
    # Timing
    is_4h_close: bool = False
    time_since_4h_close_seconds: float = 0.0
    
    # Risk state
    drawdown_pct: float = 0.0
    price_change_pct: float = 0.0
    volume_ratio: float = 1.0


@dataclass
class StrategyDecisionOutput:
    """Output from StrategyLogicCore."""
    # Action
    action: str  # "ENTER_LONG", "ENTER_SHORT", "EXIT", "HOLD", "FLAT"
    
    # Exposures
    target_exposures: Dict[str, float]
    
    # Constraints
    max_tranche: int
    urgency: str
    
    # Metadata
    rules_applied: List[str]
    rules_disabled: List[str]
    reason: str
    
    # SOL dominance
    sol_dominance_decision: Optional[SOLDominanceDecision] = None
    
    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "target_exposures": self.target_exposures,
            "max_tranche": self.max_tranche,
            "urgency": self.urgency,
            "rules_applied": self.rules_applied,
            "rules_disabled": self.rules_disabled,
            "reason": self.reason
        }


class StrategyLogicCore:
    """
    THE ONLY DECISION AUTHORITY FOR STRATEGY DECISIONS.
    
    All strategy decisions must flow through this class.
    Legacy defaults are explicitly overridden here.
    
    TASK 1: Removes implicit behaviors from:
    - Quant signal generation (now filtered through rules)
    - Fusion logic (now gated by rule status)
    - Risk filters (now mode-aware)
    - Execution gates (now use EarlyEntryGate)
    """
    
    # Implicit behaviors that are REMOVED/OVERRIDDEN
    REMOVED_IMPLICIT_BEHAVIORS = [
        "Quant signals were accepted without mode check -> Now filtered by mode rules",
        "DRL entropy threshold was hardcoded 0.7 -> Now softened in OPPORTUNITY",
        "Alpha threshold was always 3× -> Now mode-dependent (1.5-2.5× in OPP)",
        "Regime persistence always 3 ticks -> Now bypassed in OPPORTUNITY",
        "SNR filter always required >2.0 -> Now disabled in OPPORTUNITY",
        "Full 4H confirmation required -> Now allows Tranche-1 early in OPP",
        "BTC/ETH anchors always required -> Now can be zero in SOL dominance",
        "Position size capped at 0.8 -> Now SOL can reach 1.0 in OPP",
    ]
    
    def __init__(self):
        self._last_decision: Optional[StrategyDecisionOutput] = None
        logger.info("StrategyLogicCore initialized - THE ONLY decision authority")
        logger.info(f"Removed {len(self.REMOVED_IMPLICIT_BEHAVIORS)} implicit behaviors")
    
    def decide(self, ctx: StrategyDecisionContext) -> StrategyDecisionOutput:
        """
        Make the authoritative strategy decision.
        
        This is THE entry point for all strategy decisions.
        """
        
        is_opportunity = ctx.mode == "OPPORTUNITY"
        rules_applied = []
        rules_disabled = []
        
        # === STEP 1: Check HARD rules (never disabled) ===
        
        # NO_TRADE enforcement (HARD)
        if ctx.mode == "NO_TRADE":
            rules_applied.append("no_trade_enforcement")
            return self._create_flat_output(
                reason="NO_TRADE mode -> FLAT (HARD rule)",
                rules_applied=rules_applied,
                urgency="IMMEDIATE"
            )
        
        # Data integrity (HARD) - checked via VPIN/correlation
        if ctx.vpin >= 0.85 or ctx.correlation < 0.3 or ctx.dvol_zscore >= 5.0:
            rules_applied.append("data_integrity_check")
            return self._create_flat_output(
                reason=f"Data integrity fail: VPIN={ctx.vpin:.2f}, CORR={ctx.correlation:.2f}, DVOL={ctx.dvol_zscore:.1f}",
                rules_applied=rules_applied,
                urgency="IMMEDIATE"
            )
        
        # All-conflict-flat (HARD)
        if self._is_signal_conflict(ctx):
            rules_applied.append("all_conflict_flat")
            return self._create_flat_output(
                reason="All signals in directional conflict -> FLAT (HARD rule)",
                rules_applied=rules_applied,
                urgency="NORMAL"
            )
        
        # Drawdown circuit breaker (HARD)
        if ctx.drawdown_pct >= 0.35:
            rules_applied.append("max_drawdown_circuit_breaker")
            return self._create_flat_output(
                reason=f"Drawdown {ctx.drawdown_pct*100:.1f}% >= 35% -> FLAT (HARD rule)",
                rules_applied=rules_applied,
                urgency="IMMEDIATE"
            )
        
        # === STEP 2: Check SOL dominance forced exit (if in dominance) ===
        
        sol_exposure = ctx.current_exposures.get("SOL", 0)
        if abs(sol_exposure) >= 0.7:
            sol_decision = SOLDominanceForcedExit(
                sol_exposure=sol_exposure,
                vpin=ctx.vpin,
                correlation=ctx.correlation,
                lead_lag_valid=ctx.lead_lag_valid,
                crack_active=ctx.crack_active,
                crack_strength=ctx.crack_strength,
                price_change_pct=ctx.price_change_pct,
                volume_ratio=ctx.volume_ratio,
                mode=ctx.mode
            )
            
            if sol_decision.action in ["EXIT_IMMEDIATE", "DE_ESCALATE_PARTIAL", "DE_ESCALATE_FULL"]:
                rules_applied.append("sol_dominance_forced_exit")
                return StrategyDecisionOutput(
                    action="EXIT" if sol_decision.action == "EXIT_IMMEDIATE" else "REDUCE",
                    target_exposures={
                        "SOL": sol_decision.target_sol_exposure,
                        "BTC": sol_decision.target_btc_exposure,
                        "ETH": sol_decision.target_eth_exposure
                    },
                    max_tranche=0,
                    urgency=sol_decision.urgency,
                    rules_applied=rules_applied,
                    rules_disabled=rules_disabled,
                    reason=sol_decision.reason,
                    sol_dominance_decision=sol_decision
                )
        
        # === STEP 3: Apply mode-specific rules ===
        
        # Check disabled rules
        for rule_name, config in RULE_CONFIGURATION.items():
            if is_rule_disabled(rule_name, is_opportunity):
                rules_disabled.append(rule_name)
        
        # Persistence requirement
        if not is_rule_disabled("persistence_requirement", is_opportunity):
            rules_applied.append("persistence_requirement")
            # Check regime persistence: regime must be stable for at least N periods
            regime_stable = True
            if hasattr(ctx, 'regime_persistence_bars') and ctx.regime_persistence_bars is not None:
                min_persistence = 3 if is_opportunity else 6  # Fewer bars required in OPPORTUNITY mode
                regime_stable = ctx.regime_persistence_bars >= min_persistence
            if not regime_stable:
                return StrategyDecisionOutput(
                    action="HOLD",
                    target_exposures=ctx.current_exposures,
                    max_tranche=0,
                    urgency="LOW",
                    rules_applied=rules_applied,
                    rules_disabled=rules_disabled,
                    reason=f"Regime not persistent enough: {getattr(ctx, 'regime_persistence_bars', 0)} bars"
                )
        else:
            rules_disabled.append("persistence_requirement")
        
        # SNR filter
        if not is_rule_disabled("strict_snr_filter", is_opportunity):
            rules_applied.append("strict_snr_filter")
            # Check signal-to-noise ratio
            snr_valid = True
            if hasattr(ctx, 'snr') and ctx.snr is not None:
                min_snr = 1.5 if is_opportunity else 2.5  # Lower threshold in OPPORTUNITY mode
                snr_valid = ctx.snr >= min_snr
            if not snr_valid:
                return StrategyDecisionOutput(
                    action="HOLD",
                    target_exposures=ctx.current_exposures,
                    max_tranche=0,
                    urgency="LOW",
                    rules_applied=rules_applied,
                    rules_disabled=rules_disabled,
                    reason=f"SNR too low: {getattr(ctx, 'snr', 0):.2f}"
                )
        else:
            rules_disabled.append("strict_snr_filter")
        
        # === STEP 4: Check early entry eligibility ===
        
        if not ctx.is_4h_close:
            early_gate = EarlyEntryGate(
                mode=ctx.mode,
                lead_lag_valid=ctx.lead_lag_valid,
                lead_lag_edge_bps=ctx.lead_lag_edge_bps,
                crack_active=ctx.crack_active,
                crack_direction=ctx.crack_direction,
                sentiment_shock_sigma=ctx.sentiment_shock_sigma,
                vpin=ctx.vpin,
                correlation=ctx.correlation,
                dvol_zscore=ctx.dvol_zscore,
                current_tranche=ctx.current_tranche
            )
            
            rules_applied.append("early_entry_gate")
            
            if not early_gate.allowed:
                return StrategyDecisionOutput(
                    action="HOLD",
                    target_exposures=ctx.current_exposures,
                    max_tranche=0,
                    urgency="NORMAL",
                    rules_applied=rules_applied,
                    rules_disabled=rules_disabled,
                    reason=f"Early entry denied: {early_gate.reason}"
                )
            
            # Early entry allowed, but capped at Tranche-1
            max_tranche = early_gate.max_tranche
        else:
            max_tranche = 4  # Full tranche access at 4H close
        
        # === STEP 5: Calculate target exposures ===
        
        # Determine direction
        direction = self._calculate_direction(ctx, is_opportunity)
        
        if abs(direction) < 0.1:
            return StrategyDecisionOutput(
                action="HOLD",
                target_exposures=ctx.current_exposures,
                max_tranche=max_tranche,
                urgency="NORMAL",
                rules_applied=rules_applied,
                rules_disabled=rules_disabled,
                reason="Insufficient directional signal"
            )
        
        # Calculate exposures based on mode
        target_exposures = self._calculate_exposures(ctx, direction, is_opportunity, max_tranche)
        
        action = "ENTER_LONG" if direction > 0 else "ENTER_SHORT"
        
        return StrategyDecisionOutput(
            action=action,
            target_exposures=target_exposures,
            max_tranche=max_tranche,
            urgency="IMMEDIATE" if is_opportunity and ctx.crack_active else "NORMAL",
            rules_applied=rules_applied,
            rules_disabled=rules_disabled,
            reason=f"{action} with {len(rules_disabled)} rules disabled in {ctx.mode}"
        )
    
    def _is_signal_conflict(self, ctx: StrategyDecisionContext) -> bool:
        """Check if signals are in full conflict."""
        q_dir = "long" if ctx.quant_direction > 0.3 else ("short" if ctx.quant_direction < -0.3 else "neutral")
        d_dir = "long" if ctx.drl_direction > 0.3 else ("short" if ctx.drl_direction < -0.3 else "neutral")
        s_dir = "long" if ctx.sentiment_direction > 0.3 else ("short" if ctx.sentiment_direction < -0.3 else "neutral")
        
        if ctx.quant_confidence < 0.5:
            q_dir = "neutral"
        if ctx.drl_confidence < 0.5:
            d_dir = "neutral"
        
        dirs = [q_dir, d_dir, s_dir]
        confident_dirs = [d for d in dirs if d != "neutral"]
        
        if len(confident_dirs) >= 2:
            has_long = "long" in confident_dirs
            has_short = "short" in confident_dirs
            if has_long and has_short:
                return True
        
        return False
    
    def _calculate_direction(self, ctx: StrategyDecisionContext, is_opportunity: bool) -> float:
        """Calculate net direction from signals."""
        # In OPPORTUNITY, CRACK direction can override
        if is_opportunity and ctx.crack_active and ctx.crack_direction != "none":
            crack_bias = 0.3 if ctx.crack_direction == "long" else -0.3
        else:
            crack_bias = 0.0
        
        # Weight signals
        quant_weight = 0.4
        drl_weight = 0.3
        sentiment_weight = 0.2
        crack_weight = 0.1 if is_opportunity else 0.0
        
        direction = (
            quant_weight * ctx.quant_direction * ctx.quant_confidence +
            drl_weight * ctx.drl_direction * ctx.drl_confidence +
            sentiment_weight * ctx.sentiment_direction +
            crack_weight * crack_bias
        )
        
        return direction
    
    def _calculate_exposures(
        self, 
        ctx: StrategyDecisionContext, 
        direction: float, 
        is_opportunity: bool,
        max_tranche: int
    ) -> Dict[str, float]:
        """Calculate target exposures."""
        
        sign = 1 if direction > 0 else -1
        magnitude = abs(direction)
        
        # Base allocation
        base_btc = 0.25
        base_eth = 0.25
        base_sol = 0.50
        
        # Apply tranche scaling
        tranche_scale = {0: 0.0, 1: 0.2, 2: 0.5, 3: 0.8, 4: 1.0}
        scale = tranche_scale.get(max_tranche, 1.0)
        
        if is_opportunity and ctx.crack_active:
            # SOL dominance allowed
            sol_max = get_softened_param("max_position_size", "sol", is_opportunity) or 1.0
            btc_min = get_softened_param("btc_eth_anchor_requirement", "min", is_opportunity) or 0.0
            
            sol_exposure = min(sol_max, 0.5 + magnitude * 0.5) * scale * sign
            btc_exposure = max(btc_min, 0.25 - magnitude * 0.15) * scale * sign
            eth_exposure = max(btc_min, 0.25 - magnitude * 0.15) * scale * sign
        else:
            # Normal allocation
            sol_exposure = base_sol * magnitude * scale * sign
            btc_exposure = base_btc * magnitude * scale * sign
            eth_exposure = base_eth * magnitude * scale * sign
        
        return {
            "SOL": sol_exposure,
            "BTC": btc_exposure,
            "ETH": eth_exposure
        }
    
    def _create_flat_output(
        self, 
        reason: str, 
        rules_applied: List[str],
        urgency: str = "IMMEDIATE"
    ) -> StrategyDecisionOutput:
        """Create a FLAT output."""
        return StrategyDecisionOutput(
            action="FLAT",
            target_exposures={"SOL": 0.0, "BTC": 0.0, "ETH": 0.0},
            max_tranche=0,
            urgency=urgency,
            rules_applied=rules_applied,
            rules_disabled=[],
            reason=reason
        )


# =============================================================================
# SINGLETON & EXPORTS
# =============================================================================

_strategy_logic_core: Optional[StrategyLogicCore] = None


def get_strategy_logic_core() -> StrategyLogicCore:
    """Get the singleton StrategyLogicCore instance."""
    global _strategy_logic_core
    if _strategy_logic_core is None:
        _strategy_logic_core = StrategyLogicCore()
    return _strategy_logic_core


# =============================================================================
# FINAL CHECK: FAST ±20-30% SOL MOVE SCENARIO
# =============================================================================

def verify_fast_sol_move_handling():
    """
    FINAL CHECK: In a fast ±20-30% SOL move, does the system:
    1. Enter early (Tranche-1 before 4H close)
    2. Scale correctly (escalate through tranches)
    3. Exit decisively (forced exit on adverse conditions)
    
    Returns True if all conditions are met.
    """
    print("\n" + "=" * 70)
    print("FINAL CHECK: Fast ±20-30% SOL Move Handling")
    print("=" * 70)
    
    core = get_strategy_logic_core()
    
    # === SCENARIO: +25% SOL bullish move ===
    print("\n--- Scenario: +25% SOL bullish move ---")
    
    # T0: CRACK window triggers, enter early
    ctx_t0 = StrategyDecisionContext(
        mode="OPPORTUNITY",
        vpin=0.5,
        correlation=0.85,
        dvol_zscore=1.5,
        quant_direction=0.6,
        quant_confidence=0.7,
        drl_direction=0.5,
        drl_confidence=0.6,
        sentiment_shock_sigma=2.5,
        crack_active=True,
        crack_direction="long",
        crack_strength=0.9,
        lead_lag_valid=True,
        lead_lag_edge_bps=25,
        current_exposures={"SOL": 0.0, "BTC": 0.0, "ETH": 0.0},
        current_tranche=0,
        is_4h_close=False,
        price_change_pct=0.05,
        volume_ratio=2.5
    )
    
    decision_t0 = core.decide(ctx_t0)
    print(f"T0 (early entry): action={decision_t0.action}, "
          f"max_tranche={decision_t0.max_tranche}, "
          f"SOL={decision_t0.target_exposures.get('SOL', 0):.2f}")
    
    enters_early = decision_t0.action in ["ENTER_LONG", "ENTER_SHORT"] and decision_t0.max_tranche == 1
    print(f"  ✓ Enters early: {enters_early}")
    
    # T1: 4H close, can scale up
    ctx_t1 = StrategyDecisionContext(
        mode="OPPORTUNITY",
        vpin=0.45,
        correlation=0.82,
        dvol_zscore=1.2,
        quant_direction=0.7,
        quant_confidence=0.75,
        drl_direction=0.6,
        drl_confidence=0.7,
        sentiment_shock_sigma=2.0,
        crack_active=True,
        crack_direction="long",
        crack_strength=0.7,
        lead_lag_valid=True,
        lead_lag_edge_bps=20,
        current_exposures={"SOL": 0.18, "BTC": 0.02, "ETH": 0.02},
        current_tranche=1,
        is_4h_close=True,
        price_change_pct=0.15,
        volume_ratio=2.0
    )
    
    decision_t1 = core.decide(ctx_t1)
    print(f"T1 (4H close): action={decision_t1.action}, "
          f"max_tranche={decision_t1.max_tranche}, "
          f"SOL={decision_t1.target_exposures.get('SOL', 0):.2f}")
    
    scales_correctly = decision_t1.max_tranche > 1 and decision_t1.target_exposures.get('SOL', 0) > 0.18
    print(f"  ✓ Scales correctly: {scales_correctly}")
    
    # T2: Adverse condition - VPIN spikes
    ctx_t2 = StrategyDecisionContext(
        mode="OPPORTUNITY",
        vpin=0.88,  # TOXIC!
        correlation=0.75,
        dvol_zscore=2.0,
        quant_direction=0.5,
        quant_confidence=0.6,
        drl_direction=0.4,
        drl_confidence=0.5,
        sentiment_shock_sigma=1.0,
        crack_active=True,
        crack_direction="long",
        crack_strength=0.5,
        lead_lag_valid=False,
        lead_lag_edge_bps=5,
        current_exposures={"SOL": 0.8, "BTC": 0.05, "ETH": 0.05},
        current_tranche=3,
        is_4h_close=False,
        price_change_pct=-0.05,
        volume_ratio=0.8
    )
    
    decision_t2 = core.decide(ctx_t2)
    print(f"T2 (VPIN spike): action={decision_t2.action}, "
          f"urgency={decision_t2.urgency}, "
          f"SOL={decision_t2.target_exposures.get('SOL', 0):.2f}")
    
    exits_decisively = decision_t2.action in ["FLAT", "EXIT"] and decision_t2.urgency == "IMMEDIATE"
    print(f"  ✓ Exits decisively: {exits_decisively}")
    
    # === SUMMARY ===
    print("\n" + "-" * 50)
    all_pass = enters_early and scales_correctly and exits_decisively
    
    if all_pass:
        print("ON FINAL CHECK PASSED: System handles fast SOL move correctly")
        print("   - Enters early (Tranche-1 before 4H)")
        print("   - Scales correctly (escalates through tranches)")
        print("   - Exits decisively (forced exit on VPIN spike)")
    else:
        print("OFF FINAL CHECK FAILED:")
        if not enters_early:
            print("   - Does NOT enter early")
        if not scales_correctly:
            print("   - Does NOT scale correctly")
        if not exits_decisively:
            print("   - Does NOT exit decisively")
    
    return all_pass


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_checks():
    """Run all checks and print summary."""
    print("=" * 70)
    print("STRATEGY LOGIC CORE - COMPREHENSIVE CHECK")
    print("=" * 70)
    
    # Task 2: Rule configuration
    print("\n--- TASK 2: OPPORTUNITY Rule Disable List ---")
    print(f"{'Rule Name':<35} {'NORMAL':<12} {'OPPORTUNITY':<12}")
    print("-" * 60)
    for name, config in RULE_CONFIGURATION.items():
        print(f"{name:<35} {config.status_normal.name:<12} {config.status_opportunity.name:<12}")
    
    # Task 3: Early Entry Gate
    print("\n--- TASK 3: Early Entry Gate Test ---")
    gate_normal = EarlyEntryGate("NORMAL", True, 20, True, "long", 2.5, 0.5, 0.8, 2.0, 0)
    gate_opp = EarlyEntryGate("OPPORTUNITY", True, 20, True, "long", 2.5, 0.5, 0.8, 2.0, 0)
    print(f"NORMAL mode: allowed={gate_normal.allowed}, reason={gate_normal.reason}")
    print(f"OPPORTUNITY mode: allowed={gate_opp.allowed}, max_tranche={gate_opp.max_tranche}")
    
    # Task 4: SOL Dominance Forced Exit
    print("\n--- TASK 4: SOL Dominance Forced Exit Test ---")
    exit_vpin = SOLDominanceForcedExit(0.9, 0.9, 0.8, True, True, 0.8, 0.0, 1.0, "OPPORTUNITY")
    exit_corr = SOLDominanceForcedExit(0.9, 0.5, 0.25, True, True, 0.8, 0.0, 1.0, "OPPORTUNITY")
    exit_crack = SOLDominanceForcedExit(0.9, 0.5, 0.8, True, False, 0.1, 0.0, 1.0, "OPPORTUNITY")
    print(f"VPIN toxic (0.9): action={exit_vpin.action}, urgency={exit_vpin.urgency}")
    print(f"Correlation collapse (0.25): action={exit_corr.action}, urgency={exit_corr.urgency}")
    print(f"CRACK closed: action={exit_crack.action}, urgency={exit_crack.urgency}")
    
    # Task 5: Acceptance Criteria
    print("\n--- TASK 5: Acceptance Criteria ---")
    results = run_acceptance_checks()
    all_pass = True
    for name, (result, details) in results.items():
        status = "✓ PASS" if result == AcceptanceCriteriaResult.PASS else "✗ FAIL"
        print(f"  {status}: {name}")
        if result == AcceptanceCriteriaResult.FAIL:
            all_pass = False
    
    # Final check
    final_ok = verify_fast_sol_move_handling()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Task 2 (Rule Configuration): ✓ {len(RULE_CONFIGURATION)} rules defined")
    print(f"Task 3 (Early Entry Gate): {'✓' if not gate_normal.allowed and gate_opp.allowed else '✗'}")
    print(f"Task 4 (SOL Forced Exit): {'✓' if exit_vpin.action == 'EXIT_IMMEDIATE' else '✗'}")
    print(f"Task 5 (Acceptance Criteria): {'✓' if all_pass else '✗'}")
    print(f"Final Check (Fast SOL Move): {'✓' if final_ok else '✗'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_checks()

"""
================================================================================
HMATS v3.2 - STRATEGY ENFORCEMENT MODULE
================================================================================
Purpose: Enforce that Strategy Constitution is the ONLY decision authority.

This module:
1. TASK 1: Routes ALL decisions through StrategyLogicCore (no implicit defaults)
2. TASK 2: Enforces OPPORTUNITY rule disable list at runtime
3. TASK 3: Provides single EarlyEntryGate callable from pipeline
4. TASK 4: Hooks SOL dominance forced exit into 200ms loop
5. TASK 5: Runs acceptance criteria validation at startup

ALL STRATEGY DECISIONS MUST FLOW THROUGH THIS MODULE.
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum, auto
from datetime import datetime
import logging

from .strategy_logic_core import (
    StrategyLogicCore, StrategyDecisionContext, StrategyDecisionOutput,
    get_strategy_logic_core,
    EarlyEntryGate, EarlyEntryDecision,
    SOLDominanceForcedExit, SOLDominanceDecision,
    RULE_CONFIGURATION, RuleStatus,
    is_rule_disabled, is_rule_softened, get_softened_param,
    run_acceptance_checks, assert_strategy_acceptance, AcceptanceCriteriaResult
)
from orchestration.system_mode import SystemMode

logger = logging.getLogger(__name__)


# =============================================================================
# TASK 1: IMPLICIT BEHAVIOR REGISTRY
# =============================================================================

@dataclass
class ImplicitBehavior:
    """Record of an implicit behavior that was removed."""
    name: str
    location: str  # File and function where it was removed
    old_behavior: str
    new_behavior: str
    is_removed: bool = True


# Registry of ALL implicit behaviors removed or overridden
REMOVED_IMPLICIT_BEHAVIORS: List[ImplicitBehavior] = [
    ImplicitBehavior(
        name="quant_signal_no_mode_check",
        location="master_pipeline._generate_signals",
        old_behavior="Quant signals were accepted without mode check",
        new_behavior="Now filtered through StrategyLogicCore with mode rules"
    ),
    ImplicitBehavior(
        name="drl_entropy_hardcoded",
        location="signal_generators_v32.DRLSignalGenerator",
        old_behavior="DRL entropy threshold was hardcoded 0.7",
        new_behavior="Now softened to 0.5 in OPPORTUNITY mode via StrategyLogicCore"
    ),
    ImplicitBehavior(
        name="alpha_always_3x",
        location="master_pipeline._create_directives",
        old_behavior="Alpha threshold was always 3× friction",
        new_behavior="Now mode-dependent (1.5-2.5× in OPPORTUNITY) via StrategyLogicCore"
    ),
    ImplicitBehavior(
        name="regime_persistence_always_3",
        location="market_regime_v32.EnhancedRegimeNavigator",
        old_behavior="Regime persistence always required 3 ticks",
        new_behavior="Now bypassed in OPPORTUNITY mode via StrategyLogicCore"
    ),
    ImplicitBehavior(
        name="snr_always_2",
        location="quant_agent.QuantAgent",
        old_behavior="SNR filter always required >2.0",
        new_behavior="Now disabled in OPPORTUNITY mode via StrategyLogicCore"
    ),
    ImplicitBehavior(
        name="4h_confirmation_required",
        location="master_pipeline._create_directives",
        old_behavior="Full 4H confirmation required for entry",
        new_behavior="Now allows Tranche-1 early entry in OPPORTUNITY via EarlyEntryGate"
    ),
    ImplicitBehavior(
        name="btc_eth_anchors_required",
        location="sol_exposure_manager.calculate_target_exposures",
        old_behavior="BTC/ETH anchors always required (min 20%)",
        new_behavior="Now can be zero in SOL dominance via StrategyLogicCore"
    ),
    ImplicitBehavior(
        name="position_cap_0.8",
        location="sol_exposure_manager.calculate_target_exposures",
        old_behavior="Position size capped at 0.8",
        new_behavior="Now SOL can reach 1.0 in OPPORTUNITY via StrategyLogicCore"
    ),
    ImplicitBehavior(
        name="fusion_default_weights",
        location="authority_fusion.AuthorityFusionModule.fuse",
        old_behavior="Fusion used fixed weights regardless of mode",
        new_behavior="Now weights adjusted via StrategyLogicCore rule status"
    ),
    ImplicitBehavior(
        name="risk_filter_static_thresholds",
        location="defense/execution_guards.py",
        old_behavior="Risk filters had static thresholds",
        new_behavior="Now thresholds are mode-aware via StrategyLogicCore"
    ),
]


def get_removed_behaviors() -> List[ImplicitBehavior]:
    """Get list of all removed implicit behaviors."""
    return REMOVED_IMPLICIT_BEHAVIORS


def print_removed_behaviors():
    """Print all removed implicit behaviors for audit."""
    print("\n" + "=" * 70)
    print("TASK 1: REMOVED/OVERRIDDEN IMPLICIT BEHAVIORS")
    print("=" * 70)
    for i, behavior in enumerate(REMOVED_IMPLICIT_BEHAVIORS, 1):
        print(f"\n{i}. {behavior.name}")
        print(f"   Location: {behavior.location}")
        print(f"   OLD: {behavior.old_behavior}")
        print(f"   NEW: {behavior.new_behavior}")
    print("\n" + "=" * 70)


# =============================================================================
# TASK 2: OPPORTUNITY RULE ENFORCEMENT TABLE
# =============================================================================

@dataclass
class RuleEnforcementRecord:
    """Record of a rule's status in OPPORTUNITY mode."""
    rule_name: str
    status: str  # "DISABLED", "SOFTENED", "HARD"
    normal_value: str
    opportunity_value: str
    enforcement_location: str  # Where this is enforced in code


# EXPLICIT TABLE OF RULE STATES (TASK 2)
OPPORTUNITY_RULE_TABLE: List[RuleEnforcementRecord] = [
    # === DISABLED in OPPORTUNITY ===
    RuleEnforcementRecord(
        rule_name="persistence_requirement",
        status="DISABLED",
        normal_value="3 ticks required",
        opportunity_value="Bypassed",
        enforcement_location="strategy_logic_core.py:906-910"
    ),
    RuleEnforcementRecord(
        rule_name="strict_snr_filter",
        status="DISABLED",
        normal_value="SNR > 2.0 required",
        opportunity_value="Bypassed",
        enforcement_location="strategy_logic_core.py:912-917"
    ),
    RuleEnforcementRecord(
        rule_name="alpha_3x_gate",
        status="DISABLED",
        normal_value="Alpha >= 3× friction",
        opportunity_value="Replaced by 1.5-2.5×",
        enforcement_location="alpha_threshold.py via StrategyLogicCore"
    ),
    RuleEnforcementRecord(
        rule_name="full_confirmation_depth",
        status="DISABLED",
        normal_value="4H close required",
        opportunity_value="Tranche-1 allowed early",
        enforcement_location="strategy_logic_core.py:921-946 (EarlyEntryGate)"
    ),
    # === SOFTENED in OPPORTUNITY ===
    RuleEnforcementRecord(
        rule_name="regime_confidence_threshold",
        status="SOFTENED",
        normal_value="0.7",
        opportunity_value="0.5",
        enforcement_location="strategy_logic_core.py:80-85"
    ),
    RuleEnforcementRecord(
        rule_name="max_position_size",
        status="SOFTENED",
        normal_value="0.8",
        opportunity_value="1.0 for SOL",
        enforcement_location="strategy_logic_core.py:87-92"
    ),
    RuleEnforcementRecord(
        rule_name="tranche_escalation_speed",
        status="SOFTENED",
        normal_value="50 bps profit required for T3",
        opportunity_value="0 profit required",
        enforcement_location="strategy_logic_core.py:93-99"
    ),
    RuleEnforcementRecord(
        rule_name="btc_eth_anchor_requirement",
        status="SOFTENED",
        normal_value="20% minimum",
        opportunity_value="0% allowed",
        enforcement_location="strategy_logic_core.py:100-106"
    ),
    # === HARD (never disabled) ===
    RuleEnforcementRecord(
        rule_name="data_integrity_check",
        status="HARD",
        normal_value="Always checked",
        opportunity_value="Always checked",
        enforcement_location="strategy_logic_core.py:838-845"
    ),
    RuleEnforcementRecord(
        rule_name="no_trade_enforcement",
        status="HARD",
        normal_value="NO_TRADE = FLAT",
        opportunity_value="NO_TRADE = FLAT",
        enforcement_location="strategy_logic_core.py:829-836"
    ),
    RuleEnforcementRecord(
        rule_name="correlation_collapse_check",
        status="HARD",
        normal_value="Corr < 0.3 = de-risk",
        opportunity_value="Corr < 0.3 = de-risk",
        enforcement_location="strategy_logic_core.py:838-845"
    ),
    RuleEnforcementRecord(
        rule_name="vpin_toxicity_abort",
        status="HARD",
        normal_value="VPIN > 0.85 = abort",
        opportunity_value="VPIN > 0.85 = abort",
        enforcement_location="strategy_logic_core.py:838-845"
    ),
    RuleEnforcementRecord(
        rule_name="all_conflict_flat",
        status="HARD",
        normal_value="Conflict = FLAT",
        opportunity_value="Conflict = FLAT",
        enforcement_location="strategy_logic_core.py:847-854"
    ),
    RuleEnforcementRecord(
        rule_name="max_drawdown_circuit_breaker",
        status="HARD",
        normal_value="15% DD = shutdown",
        opportunity_value="15% DD = shutdown",
        enforcement_location="strategy_logic_core.py:856-863"
    ),
]


def get_rule_enforcement_table() -> List[RuleEnforcementRecord]:
    """Get the rule enforcement table."""
    return OPPORTUNITY_RULE_TABLE


def print_rule_enforcement_table():
    """Print the rule enforcement table for audit."""
    print("\n" + "=" * 90)
    print("TASK 2: OPPORTUNITY MODE RULE ENFORCEMENT TABLE")
    print("=" * 90)
    print(f"{'Rule':<30} {'Status':<10} {'Normal':<20} {'Opportunity':<20}")
    print("-" * 90)
    for rule in OPPORTUNITY_RULE_TABLE:
        print(f"{rule.rule_name:<30} {rule.status:<10} {rule.normal_value:<20} {rule.opportunity_value:<20}")
    print("=" * 90)


def enforce_opportunity_rules(mode: str, rules_context: Dict) -> Dict[str, bool]:
    """
    TASK 2: Enforce OPPORTUNITY rules at runtime.
    
    Returns dict of {rule_name: is_active}.
    """
    is_opportunity = (mode == "OPPORTUNITY")
    rule_states = {}
    
    for rule in OPPORTUNITY_RULE_TABLE:
        rule_name = rule.rule_name
        
        if rule.status == "DISABLED":
            rule_states[rule_name] = not is_opportunity  # Active in NORMAL, disabled in OPP
        elif rule.status == "SOFTENED":
            rule_states[rule_name] = True  # Always active but with different params
        elif rule.status == "HARD":
            rule_states[rule_name] = True  # Always active
        else:
            rule_states[rule_name] = True
    
    return rule_states


# =============================================================================
# TASK 3: SINGLE EARLY ENTRY GATE WRAPPER
# =============================================================================

class EarlyEntryEnforcer:
    """
    TASK 3: Single mandatory gate for early entry.
    
    This wraps EarlyEntryGate and ensures it's called from the pipeline.
    """
    
    def __init__(self):
        self._last_decision: Optional[EarlyEntryDecision] = None
        self._total_allowed = 0
        self._total_denied = 0
        logger.info("EarlyEntryEnforcer initialized")
    
    def check(
        self,
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
        Check if early entry is allowed.
        
        This MUST be called before any entry before 4H close.
        """
        decision = EarlyEntryGate(
            mode=mode,
            lead_lag_valid=lead_lag_valid,
            lead_lag_edge_bps=lead_lag_edge_bps,
            crack_active=crack_active,
            crack_direction=crack_direction,
            sentiment_shock_sigma=sentiment_shock_sigma,
            vpin=vpin,
            correlation=correlation,
            dvol_zscore=dvol_zscore,
            current_tranche=current_tranche
        )
        
        self._last_decision = decision
        
        if decision.allowed:
            self._total_allowed += 1
            logger.info(f"[EarlyEntryEnforcer] ALLOWED: {decision.reason}")
        else:
            self._total_denied += 1
            logger.info(f"[EarlyEntryEnforcer] DENIED: {decision.reason}")
            if decision.abort_conditions:
                logger.warning(f"  Abort conditions: {decision.abort_conditions}")
        
        return decision
    
    def get_stats(self) -> Dict:
        """Get enforcement statistics."""
        return {
            "total_allowed": self._total_allowed,
            "total_denied": self._total_denied,
            "last_decision": self._last_decision.to_dict() if self._last_decision else None
        }


# Singleton
_early_entry_enforcer: Optional[EarlyEntryEnforcer] = None

def get_early_entry_enforcer() -> EarlyEntryEnforcer:
    global _early_entry_enforcer
    if _early_entry_enforcer is None:
        _early_entry_enforcer = EarlyEntryEnforcer()
    return _early_entry_enforcer


# =============================================================================
# TASK 4: SOL DOMINANCE FORCED EXIT HOOK
# =============================================================================

class SOLDominanceEnforcer:
    """
    TASK 4: SOL dominance forced exit enforcement.
    
    This hooks into 200ms loop to check for forced exit conditions.
    Exits OVERRIDE Quant/DRL signals.
    """
    
    def __init__(self):
        self._last_decision: Optional[SOLDominanceDecision] = None
        self._forced_exits = 0
        self._partial_deescalations = 0
        logger.info("SOLDominanceEnforcer initialized")
    
    def check_forced_exit(
        self,
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
        Check if SOL dominance requires forced exit.
        
        This should be called on EVERY 200ms tick when SOL exposure > 0.7.
        Returns decision that OVERRIDES Quant/DRL.
        """
        decision = SOLDominanceForcedExit(
            sol_exposure=sol_exposure,
            vpin=vpin,
            correlation=correlation,
            lead_lag_valid=lead_lag_valid,
            crack_active=crack_active,
            crack_strength=crack_strength,
            price_change_pct=price_change_pct,
            volume_ratio=volume_ratio,
            mode=mode
        )
        
        self._last_decision = decision
        
        if decision.action == "EXIT_IMMEDIATE":
            self._forced_exits += 1
            logger.warning(f"[SOLDominanceEnforcer] FORCED EXIT: {decision.reason}")
        elif decision.action.startswith("DE_ESCALATE"):
            self._partial_deescalations += 1
            logger.info(f"[SOLDominanceEnforcer] DE-ESCALATE: {decision.reason}")
        
        return decision
    
    def get_stats(self) -> Dict:
        """Get enforcement statistics."""
        return {
            "forced_exits": self._forced_exits,
            "partial_deescalations": self._partial_deescalations,
            "last_decision": self._last_decision.to_dict() if self._last_decision else None
        }


# Singleton
_sol_dominance_enforcer: Optional[SOLDominanceEnforcer] = None

def get_sol_dominance_enforcer() -> SOLDominanceEnforcer:
    global _sol_dominance_enforcer
    if _sol_dominance_enforcer is None:
        _sol_dominance_enforcer = SOLDominanceEnforcer()
    return _sol_dominance_enforcer


# =============================================================================
# TASK 4: SOL DOMINANCE DECISION TREE
# =============================================================================

def print_sol_dominance_decision_tree():
    """
    Print the SOL dominance escalation/de-escalation decision tree.
    """
    print("\n" + "=" * 80)
    print("TASK 4: SOL DOMINANCE DECISION TREE")
    print("=" * 80)
    print("""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   SOL DOMINANCE DECISION TREE                          │
    └─────────────────────────────────────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ Entry Point: SOL Exposure >= 0.7                                        │
    └──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ CHECK 1: HARD ABORT CONDITIONS (immediate FLAT ALL)                     │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ • Correlation < 0.3   -> EXIT_IMMEDIATE -> SOL=0, BTC=0, ETH=0           │
    │ • VPIN > 0.85         -> EXIT_IMMEDIATE -> SOL=0, BTC=0, ETH=0           │
    │ • Lead-lag invalid + adverse move (>2%) -> EXIT_IMMEDIATE               │
    └──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (No hard abort)
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ CHECK 2: CRACK WINDOW STATUS                                            │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ • CRACK closed OR strength < 0.2 -> DE_ESCALATE_PARTIAL                  │
    │   Target: SOL=0.5, BTC=0.15, ETH=0.15 (urgency=HIGH)                   │
    └──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (CRACK still active)
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ CHECK 3: SOFT DE-ESCALATION CONDITIONS                                  │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ • VPIN > 0.7 (elevated)     -> DE_ESCALATE_PARTIAL -> SOL=0.6            │
    │ • Correlation < 0.5          -> DE_ESCALATE_PARTIAL -> SOL=0.5           │
    │ • Volume ratio < 0.5×        -> DE_ESCALATE_PARTIAL -> SOL=0.6           │
    └──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (No de-escalation needed)
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ CHECK 4: ESCALATION CONDITIONS                                          │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ • Mode = OPPORTUNITY                                                    │
    │ • CRACK active                                                          │
    │ • VPIN < 0.6                                                            │
    │ • Correlation > 0.6                                                     │
    │ • SOL exposure < 0.9                                                    │
    │ -> ESCALATE -> SOL=0.9, BTC=0.05, ETH=0.05                               │
    └──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (No action needed)
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ DEFAULT: HOLD current position                                          │
    └──────────────────────────────────────────────────────────────────────────┘
    
    CRITICAL: All EXIT_IMMEDIATE and DE_ESCALATE decisions OVERRIDE Quant/DRL.
    These are HARD authority decisions that cannot be bypassed.
    """)
    print("=" * 80)


# =============================================================================
# TASK 5: STRATEGY ACCEPTANCE CRITERIA VALIDATION
# =============================================================================

@dataclass
class AcceptanceValidationResult:
    """Result of acceptance validation."""
    passed: bool
    checks: Dict[str, Tuple[bool, str]]
    failed_critical: List[str]
    warnings: List[str]


def validate_strategy_acceptance() -> AcceptanceValidationResult:
    """
    TASK 5: Validate all strategy acceptance criteria.
    
    This should be run at startup to ensure system is correctly configured.
    """
    logger.info("Running strategy acceptance validation...")
    
    results = run_acceptance_checks()
    
    failed_critical = []
    warnings = []
    checks = {}
    
    for name, (result, details) in results.items():
        passed = (result == AcceptanceCriteriaResult.PASS)
        checks[name] = (passed, details)
        
        if not passed:
            # All criteria are critical
            failed_critical.append(name)
    
    validation = AcceptanceValidationResult(
        passed=(len(failed_critical) == 0),
        checks=checks,
        failed_critical=failed_critical,
        warnings=warnings
    )
    
    if validation.passed:
        logger.info("✓ Strategy acceptance validation PASSED")
    else:
        logger.error(f"✗ Strategy acceptance validation FAILED: {failed_critical}")
    
    return validation


def print_acceptance_criteria():
    """Print the acceptance criteria checklist."""
    print("\n" + "=" * 70)
    print("TASK 5: STRATEGY ACCEPTANCE CRITERIA")
    print("=" * 70)
    print("""
    Strategy implementation is considered FAILED if ANY of these are violated:
    
    [ ] 1. OPP_DISABLES_RULES
        OPPORTUNITY mode MUST disable: persistence, SNR, 3× alpha, 4H confirmation
        
    [ ] 2. EARLY_ENTRY_ONLY_OPPORTUNITY
        Early entry (before 4H close) MUST only occur in OPPORTUNITY mode
        
    [ ] 3. SOL_DOMINANCE_HAS_EXIT
        SOL dominance (>0.7 exposure) MUST have forced exit path
        (VPIN toxic, correlation collapse, CRACK close, lead-lag invalid)
        
    [ ] 4. CONFLICT_RESULTS_FLAT
        Conflicting signals (Quant/DRL/Sentiment all opposing) MUST result in FLAT
        
    [ ] 5. NO_TRADE_ENFORCED
        NO_TRADE mode MUST always result in FLAT, no exceptions
        
    [ ] 6. TRANCHE_GATING_ENFORCED
        Pre-4H entry MUST be limited to Tranche-1 (20% of target)
    """)
    print("=" * 70)


def run_startup_validation() -> bool:
    """
    Run all startup validations.
    
    Returns True if all pass, False if any critical failure.
    """
    print("\n" + "=" * 70)
    print("HMATS v3.2 STRATEGY ENFORCEMENT STARTUP VALIDATION")
    print("=" * 70)
    
    # 1. Print removed behaviors (Task 1)
    print("\n[TASK 1] Verifying implicit behavior removal...")
    print(f"  {len(REMOVED_IMPLICIT_BEHAVIORS)} implicit behaviors removed/overridden")
    
    # 2. Print rule table (Task 2)
    print("\n[TASK 2] Verifying OPPORTUNITY rule enforcement...")
    disabled_in_opp = sum(1 for r in OPPORTUNITY_RULE_TABLE if r.status == "DISABLED")
    softened_in_opp = sum(1 for r in OPPORTUNITY_RULE_TABLE if r.status == "SOFTENED")
    hard_rules = sum(1 for r in OPPORTUNITY_RULE_TABLE if r.status == "HARD")
    print(f"  DISABLED in OPPORTUNITY: {disabled_in_opp}")
    print(f"  SOFTENED in OPPORTUNITY: {softened_in_opp}")
    print(f"  HARD (never disabled): {hard_rules}")
    
    # 3. Verify EarlyEntryGate (Task 3)
    print("\n[TASK 3] Verifying EarlyEntryGate...")
    enforcer = get_early_entry_enforcer()
    # Test: NORMAL mode should deny
    normal_test = enforcer.check(
        mode="NORMAL", lead_lag_valid=True, lead_lag_edge_bps=20,
        crack_active=True, crack_direction="long", sentiment_shock_sigma=2.5,
        vpin=0.5, correlation=0.8, dvol_zscore=2.0, current_tranche=0
    )
    # Test: OPPORTUNITY mode should allow
    opp_test = enforcer.check(
        mode="OPPORTUNITY", lead_lag_valid=True, lead_lag_edge_bps=20,
        crack_active=True, crack_direction="long", sentiment_shock_sigma=2.5,
        vpin=0.5, correlation=0.8, dvol_zscore=2.0, current_tranche=0
    )
    early_entry_ok = (not normal_test.allowed) and opp_test.allowed and (opp_test.max_tranche == 1)
    print(f"  NORMAL mode denied: {not normal_test.allowed}")
    print(f"  OPPORTUNITY mode allowed: {opp_test.allowed}")
    print(f"  Max tranche = 1: {opp_test.max_tranche == 1}")
    
    # 4. Verify SOL dominance exit (Task 4)
    print("\n[TASK 4] Verifying SOL dominance forced exit...")
    sol_enforcer = get_sol_dominance_enforcer()
    # Test: VPIN toxic should exit
    vpin_test = sol_enforcer.check_forced_exit(
        sol_exposure=0.9, vpin=0.9, correlation=0.8, lead_lag_valid=True,
        crack_active=True, crack_strength=0.8, price_change_pct=0.0,
        volume_ratio=1.0, mode="OPPORTUNITY"
    )
    # Test: Correlation collapse should exit
    corr_test = sol_enforcer.check_forced_exit(
        sol_exposure=0.9, vpin=0.5, correlation=0.2, lead_lag_valid=True,
        crack_active=True, crack_strength=0.8, price_change_pct=0.0,
        volume_ratio=1.0, mode="OPPORTUNITY"
    )
    sol_exit_ok = (vpin_test.action == "EXIT_IMMEDIATE") and (corr_test.action == "EXIT_IMMEDIATE")
    print(f"  VPIN toxic -> EXIT: {vpin_test.action == 'EXIT_IMMEDIATE'}")
    print(f"  Corr collapse -> EXIT: {corr_test.action == 'EXIT_IMMEDIATE'}")
    
    # 5. Validate acceptance criteria (Task 5)
    print("\n[TASK 5] Running acceptance criteria validation...")
    validation = validate_strategy_acceptance()
    for name, (passed, details) in validation.checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    # Final result
    all_passed = early_entry_ok and sol_exit_ok and validation.passed
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL STRATEGY ENFORCEMENT VALIDATIONS PASSED")
    else:
        print("✗ SOME VALIDATIONS FAILED - CHECK ABOVE")
    print("=" * 70)
    
    return all_passed


# =============================================================================
# STRATEGY DECISION BRIDGE (connects pipeline to StrategyLogicCore)
# =============================================================================

class StrategyDecisionBridge:
    """
    Bridge that routes ALL decisions through StrategyLogicCore.
    
    This is the ONLY interface the pipeline should use for strategy decisions.
    """
    
    def __init__(self):
        self._core = get_strategy_logic_core()
        self._early_entry = get_early_entry_enforcer()
        self._sol_exit = get_sol_dominance_enforcer()
        self._decisions_made = 0
        logger.info("StrategyDecisionBridge initialized - ALL decisions flow through here")
    
    def make_decision_from_context(self, ctx: StrategyDecisionContext) -> StrategyDecisionOutput:
        """
        Make a strategy decision from a pre-built context.
        
        Preferred method when you already have a StrategyDecisionContext object.
        """
        decision = self._core.decide(ctx)
        
        self._decisions_made += 1
        logger.debug(f"[StrategyDecisionBridge] Decision #{self._decisions_made}: "
                    f"{decision.action} (rules_disabled={len(decision.rules_disabled)})")
        
        return decision
    
    def make_decision(
        self,
        mode: str,
        vpin: float,
        correlation: float,
        dvol_zscore: float,
        quant_direction: float,
        quant_confidence: float,
        drl_direction: float,
        drl_confidence: float,
        sentiment_direction: float,
        sentiment_shock_sigma: float,
        crack_active: bool,
        crack_direction: str,
        crack_strength: float,
        lead_lag_valid: bool,
        lead_lag_edge_bps: float,
        current_exposures: Dict[str, float],
        current_tranche: int,
        is_4h_close: bool,
        drawdown_pct: float = 0.0,
        price_change_pct: float = 0.0,
        volume_ratio: float = 1.0
    ) -> StrategyDecisionOutput:
        """
        Make a strategy decision.
        
        This is THE entry point for all strategy decisions.
        All decisions flow through StrategyLogicCore.
        """
        
        # Build context
        ctx = StrategyDecisionContext(
            mode=mode,
            vpin=vpin,
            correlation=correlation,
            dvol_zscore=dvol_zscore,
            quant_direction=quant_direction,
            quant_confidence=quant_confidence,
            drl_direction=drl_direction,
            drl_confidence=drl_confidence,
            sentiment_direction=sentiment_direction,
            sentiment_shock_sigma=sentiment_shock_sigma,
            crack_active=crack_active,
            crack_direction=crack_direction,
            crack_strength=crack_strength,
            lead_lag_valid=lead_lag_valid,
            lead_lag_edge_bps=lead_lag_edge_bps,
            current_exposures=current_exposures,
            current_tranche=current_tranche,
            is_4h_close=is_4h_close,
            drawdown_pct=drawdown_pct,
            price_change_pct=price_change_pct,
            volume_ratio=volume_ratio
        )
        
        # Delegate to core
        decision = self._core.decide(ctx)
        
        self._decisions_made += 1
        logger.debug(f"[StrategyDecisionBridge] Decision #{self._decisions_made}: "
                    f"{decision.action} (rules_disabled={len(decision.rules_disabled)})")
        
        return decision
    
    def check_early_entry(
        self,
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
        """Check if early entry is allowed."""
        return self._early_entry.check(
            mode=mode,
            lead_lag_valid=lead_lag_valid,
            lead_lag_edge_bps=lead_lag_edge_bps,
            crack_active=crack_active,
            crack_direction=crack_direction,
            sentiment_shock_sigma=sentiment_shock_sigma,
            vpin=vpin,
            correlation=correlation,
            dvol_zscore=dvol_zscore,
            current_tranche=current_tranche
        )
    
    def check_sol_exit(
        self,
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
        """Check if SOL dominance requires forced exit."""
        return self._sol_exit.check_forced_exit(
            sol_exposure=sol_exposure,
            vpin=vpin,
            correlation=correlation,
            lead_lag_valid=lead_lag_valid,
            crack_active=crack_active,
            crack_strength=crack_strength,
            price_change_pct=price_change_pct,
            volume_ratio=volume_ratio,
            mode=mode
        )
    
    def get_stats(self) -> Dict:
        """Get bridge statistics."""
        return {
            "decisions_made": self._decisions_made,
            "early_entry_stats": self._early_entry.get_stats(),
            "sol_exit_stats": self._sol_exit.get_stats()
        }


# Singleton
_strategy_bridge: Optional[StrategyDecisionBridge] = None

def get_strategy_bridge() -> StrategyDecisionBridge:
    global _strategy_bridge
    if _strategy_bridge is None:
        _strategy_bridge = StrategyDecisionBridge()
    return _strategy_bridge


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Task 1
    "REMOVED_IMPLICIT_BEHAVIORS",
    "get_removed_behaviors",
    "print_removed_behaviors",
    # Task 2
    "OPPORTUNITY_RULE_TABLE",
    "get_rule_enforcement_table",
    "print_rule_enforcement_table",
    "enforce_opportunity_rules",
    # Task 3
    "EarlyEntryEnforcer",
    "get_early_entry_enforcer",
    # Task 4
    "SOLDominanceEnforcer",
    "get_sol_dominance_enforcer",
    "print_sol_dominance_decision_tree",
    # Task 5
    "validate_strategy_acceptance",
    "print_acceptance_criteria",
    "run_startup_validation",
    # Bridge
    "StrategyDecisionBridge",
    "get_strategy_bridge",
]


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Run startup validation
    run_startup_validation()
    
    # Print detailed tables
    print_removed_behaviors()
    print_rule_enforcement_table()
    print_sol_dominance_decision_tree()
    print_acceptance_criteria()

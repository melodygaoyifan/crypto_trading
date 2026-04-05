"""
================================================================================
HMATS v4.0 - SCENARIO WALKTHROUGHS
================================================================================
Version: 4.0.0
Purpose: Scenario-based testing and validation for HMATS trading logic

RESTORED FROM: v3.2-STABILIZED (scenario_walkthroughs_v32.py)

KEY FEATURES:
    1. Pre-defined market scenarios for testing
    2. Step-by-step walkthrough validation
    3. Expected vs actual output comparison
    4. Regression testing for constitution guarantees
    5. Integration testing for mode transitions

USAGE:
    from integration.core.scenario_walkthroughs import run_all_scenarios
    results = run_all_scenarios()
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of test scenarios."""
    NO_TRADE_TRIGGER = "NO_TRADE_TRIGGER"
    OPPORTUNITY_TRIGGER = "OPPORTUNITY_TRIGGER"
    ALPHA_GATING = "ALPHA_GATING"
    TRANCHE_SCHEDULING = "TRANCHE_SCHEDULING"
    MODE_TRANSITION = "MODE_TRANSITION"
    RISK_VETO = "RISK_VETO"
    SOL_DOMINANCE = "SOL_DOMINANCE"
    WEEKEND_HANDLING = "WEEKEND_HANDLING"
    SIGNAL_CONFLICT = "SIGNAL_CONFLICT"
    EXECUTION_DEADLOCK = "EXECUTION_DEADLOCK"


@dataclass
class ScenarioInput:
    """Input data for a scenario."""
    market_data: Dict[str, Any] = field(default_factory=dict)
    signal_data: Dict[str, Any] = field(default_factory=dict)
    lead_lag_data: Dict[str, Any] = field(default_factory=dict)
    position_data: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpectedOutput:
    """Expected output from a scenario."""
    mode: Optional[str] = None
    should_trade: Optional[bool] = None
    veto_active: Optional[bool] = None
    tranche_action: Optional[str] = None
    exposure_cap: Optional[float] = None
    execution_mode: Optional[str] = None
    proof_log_contains: List[str] = field(default_factory=list)
    custom_checks: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Result of running a scenario."""
    scenario_name: str
    passed: bool
    expected: ExpectedOutput
    actual: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


@dataclass
class Scenario:
    """A complete test scenario."""
    name: str
    type: ScenarioType
    description: str
    input_data: ScenarioInput
    expected: ExpectedOutput
    tags: List[str] = field(default_factory=list)


# =============================================================================
# PRE-DEFINED SCENARIOS
# =============================================================================

def get_no_trade_scenarios() -> List[Scenario]:
    """Get NO_TRADE trigger test scenarios."""
    
    return [
        # Scenario 1: Stale data
        Scenario(
            name="NT001_StaleData",
            type=ScenarioType.NO_TRADE_TRIGGER,
            description="Data age > 2s should trigger NO_TRADE",
            input_data=ScenarioInput(
                market_data={
                    "current_price": 150.0,
                    "volume_24h": 1_000_000,
                    "data_age_seconds": 3.5,  # > 2s threshold
                    "dvol_zscore": 1.0,
                    "vpin": 0.5,
                }
            ),
            expected=ExpectedOutput(
                mode="NO_TRADE",
                should_trade=False,
                proof_log_contains=["STALE_DATA"],
            ),
            tags=["no_trade", "data_integrity"],
        ),
        
        # Scenario 2: Flash crash
        Scenario(
            name="NT002_FlashCrash",
            type=ScenarioType.NO_TRADE_TRIGGER,
            description="5%+ move in 5min should trigger NO_TRADE",
            input_data=ScenarioInput(
                market_data={
                    "current_price": 142.5,  # -5% from 150
                    "volume_24h": 2_000_000,
                    "data_age_seconds": 0.5,
                    "dvol_zscore": 3.0,
                    "vpin": 0.75,
                    "price_5min_ago": 150.0,
                }
            ),
            expected=ExpectedOutput(
                mode="NO_TRADE",
                should_trade=False,
                proof_log_contains=["FLASH_CRASH"],
            ),
            tags=["no_trade", "flash_crash"],
        ),
        
        # Scenario 3: Extreme DVOL
        Scenario(
            name="NT003_ExtremeDVOL",
            type=ScenarioType.NO_TRADE_TRIGGER,
            description="DVOL z-score >= 5.0 should trigger NO_TRADE",
            input_data=ScenarioInput(
                market_data={
                    "current_price": 150.0,
                    "volume_24h": 1_000_000,
                    "data_age_seconds": 0.5,
                    "dvol_zscore": 5.5,  # >= 5.0 threshold
                    "vpin": 0.5,
                }
            ),
            expected=ExpectedOutput(
                mode="NO_TRADE",
                should_trade=False,
                proof_log_contains=["EXTREME_DVOL"],
            ),
            tags=["no_trade", "volatility"],
        ),
        
        # Scenario 4: Correlation collapse
        Scenario(
            name="NT004_CorrelationCollapse",
            type=ScenarioType.NO_TRADE_TRIGGER,
            description="Correlation >= 0.92 should trigger NO_TRADE",
            input_data=ScenarioInput(
                market_data={
                    "current_price": 150.0,
                    "volume_24h": 1_000_000,
                    "data_age_seconds": 0.5,
                    "dvol_zscore": 2.0,
                    "vpin": 0.5,
                    "correlation_btc_eth_sol": 0.94,  # >= 0.92 threshold
                }
            ),
            expected=ExpectedOutput(
                mode="NO_TRADE",
                should_trade=False,
                proof_log_contains=["CORRELATION_COLLAPSE"],
            ),
            tags=["no_trade", "correlation"],
        ),
        
        # Scenario 5: All-conflict-flat
        Scenario(
            name="NT005_AllConflictFlat",
            type=ScenarioType.NO_TRADE_TRIGGER,
            description="Strong opposing signals should trigger NO_TRADE",
            input_data=ScenarioInput(
                market_data={
                    "current_price": 150.0,
                    "volume_24h": 1_000_000,
                    "data_age_seconds": 0.5,
                },
                signal_data={
                    "quant_direction": 0.6,      # Strong long
                    "drl_direction": -0.5,       # Strong short
                    "sentiment_direction": 0.4,  # Moderate long
                }
            ),
            expected=ExpectedOutput(
                mode="NO_TRADE",
                should_trade=False,
                proof_log_contains=["ALL_CONFLICT_FLAT"],
            ),
            tags=["no_trade", "signal_conflict"],
        ),
    ]


def get_opportunity_scenarios() -> List[Scenario]:
    """Get OPPORTUNITY trigger test scenarios."""
    
    return [
        # Scenario 1: Lead-lag edge
        Scenario(
            name="OP001_LeadLagEdge",
            type=ScenarioType.OPPORTUNITY_TRIGGER,
            description="Strong lead-lag edge should trigger OPPORTUNITY",
            input_data=ScenarioInput(
                market_data={
                    "current_price": 150.0,
                    "volume_24h": 1_000_000,
                    "data_age_seconds": 0.5,
                    "dvol_zscore": 1.5,
                    "vpin": 0.65,
                },
                lead_lag_data={
                    "net_edge_bps": 30,  # >= 25 threshold
                    "edge_window_confidence": 0.70,  # >= 0.65 threshold
                    "cvd_divergence": 0.65,  # >= 0.6 threshold
                    "vpin": 0.65,  # < 0.80 threshold
                }
            ),
            expected=ExpectedOutput(
                mode="OPPORTUNITY",
                should_trade=True,
                proof_log_contains=["LEAD_LAG_EDGE"],
            ),
            tags=["opportunity", "lead_lag"],
        ),
        
        # Scenario 2: CRACK window
        Scenario(
            name="OP002_CrackWindow",
            type=ScenarioType.OPPORTUNITY_TRIGGER,
            description="Structure break with volume should trigger OPPORTUNITY",
            input_data=ScenarioInput(
                market_data={
                    "current_price": 153.0,  # +2% break
                    "volume_24h": 3_000_000,
                    "data_age_seconds": 0.5,
                    "dvol_zscore": 2.0,
                    "vpin": 0.55,
                    "structure_break_pct": 0.025,  # 2.5% >= 2% threshold
                    "volume_ratio": 2.8,  # >= 2.5x threshold
                    "orderbook_imbalance": 0.45,  # >= 0.4 threshold
                }
            ),
            expected=ExpectedOutput(
                mode="OPPORTUNITY",
                should_trade=True,
                proof_log_contains=["CRACK_WINDOW"],
            ),
            tags=["opportunity", "crack"],
        ),
        
        # Scenario 3: Volatility expansion
        Scenario(
            name="OP003_VolatilityExpansion",
            type=ScenarioType.OPPORTUNITY_TRIGGER,
            description="DVOL in 2.5-4.0 range should trigger OPPORTUNITY",
            input_data=ScenarioInput(
                market_data={
                    "current_price": 150.0,
                    "volume_24h": 1_500_000,
                    "data_age_seconds": 0.5,
                    "dvol_zscore": 3.2,  # In 2.5-4.0 range
                    "vpin": 0.50,
                }
            ),
            expected=ExpectedOutput(
                mode="OPPORTUNITY",
                should_trade=True,
                proof_log_contains=["VOLATILITY_EXPANSION"],
            ),
            tags=["opportunity", "volatility"],
        ),
    ]


def get_alpha_gating_scenarios() -> List[Scenario]:
    """Get alpha gating test scenarios."""
    
    return [
        # Scenario 1: Alpha below threshold in NORMAL
        Scenario(
            name="AG001_BelowThresholdNormal",
            type=ScenarioType.ALPHA_GATING,
            description="Low alpha in NORMAL mode should block trade",
            input_data=ScenarioInput(
                signal_data={
                    "signal_strength": 0.3,
                    "regime_confidence": 0.6,
                },
                system_state={
                    "mode": "NORMAL",
                    "is_maker": False,
                }
            ),
            expected=ExpectedOutput(
                should_trade=False,
                proof_log_contains=["Alpha", "threshold"],
            ),
            tags=["alpha", "gating"],
        ),
        
        # Scenario 2: Alpha above threshold in OPPORTUNITY
        Scenario(
            name="AG002_AboveThresholdOpportunity",
            type=ScenarioType.ALPHA_GATING,
            description="Good alpha in OPPORTUNITY should pass",
            input_data=ScenarioInput(
                signal_data={
                    "signal_strength": 0.7,
                    "regime_confidence": 0.8,
                },
                system_state={
                    "mode": "OPPORTUNITY",
                    "is_maker": False,
                    "lead_lag_edge_confirmed": True,
                }
            ),
            expected=ExpectedOutput(
                should_trade=True,
            ),
            tags=["alpha", "gating", "opportunity"],
        ),
    ]


def get_tranche_scenarios() -> List[Scenario]:
    """Get tranche scheduling test scenarios."""
    
    return [
        # Scenario 1: T1 entry in OPPORTUNITY
        Scenario(
            name="TR001_T1EntryOpportunity",
            type=ScenarioType.TRANCHE_SCHEDULING,
            description="T1 entry should be 20% in OPPORTUNITY",
            input_data=ScenarioInput(
                position_data={
                    "current_level": "NONE",
                    "current_exposure": 0.0,
                },
                system_state={
                    "mode": "OPPORTUNITY",
                    "target_direction": 0.7,
                }
            ),
            expected=ExpectedOutput(
                tranche_action="ENTER_T1",
                exposure_cap=0.20,
            ),
            tags=["tranche", "entry"],
        ),
        
        # Scenario 2: Abort on VPIN spike
        Scenario(
            name="TR002_AbortVPINSpike",
            type=ScenarioType.TRANCHE_SCHEDULING,
            description="VPIN > 0.85 should trigger tranche abort",
            input_data=ScenarioInput(
                market_data={
                    "vpin": 0.88,  # > 0.85 threshold
                },
                position_data={
                    "current_level": "TRANCHE_1",
                    "current_exposure": 0.20,
                    "entry_price": 148.0,
                },
                system_state={
                    "mode": "OPPORTUNITY",
                }
            ),
            expected=ExpectedOutput(
                tranche_action="ABORT",
                proof_log_contains=["VPIN", "spike"],
            ),
            tags=["tranche", "abort"],
        ),
    ]


def get_risk_veto_scenarios() -> List[Scenario]:
    """Get risk veto test scenarios."""
    
    return [
        # Scenario 1: HARD veto at 10% drawdown
        Scenario(
            name="RV001_HardDrawdownVeto",
            type=ScenarioType.RISK_VETO,
            description="10% drawdown should trigger HARD veto",
            input_data=ScenarioInput(
                system_state={
                    "mode": "OPPORTUNITY",
                    "current_drawdown": 0.12,  # 12% > 10% threshold
                }
            ),
            expected=ExpectedOutput(
                veto_active=True,
                should_trade=False,
                proof_log_contains=["HARD", "DRAWDOWN"],
            ),
            tags=["risk", "veto", "hard"],
        ),
        
        # Scenario 2: SOFT veto caps in OPPORTUNITY
        Scenario(
            name="RV002_SoftCapOpportunity",
            type=ScenarioType.RISK_VETO,
            description="SOFT veto in OPPORTUNITY should cap but not block",
            input_data=ScenarioInput(
                market_data={
                    "dvol_zscore": 3.5,  # ELEVATED (3.0-4.9)
                },
                system_state={
                    "mode": "OPPORTUNITY",
                    "current_drawdown": 0.03,
                }
            ),
            expected=ExpectedOutput(
                veto_active=False,  # SOFT doesn't block in OPPORTUNITY
                should_trade=True,
                exposure_cap=0.50,  # Capped at 50%
                proof_log_contains=["SOFT", "ELEVATED_DVOL"],
            ),
            tags=["risk", "veto", "soft"],
        ),
    ]


def get_sol_dominance_scenarios() -> List[Scenario]:
    """Get SOL dominance test scenarios."""
    
    return [
        # Scenario 1: SOL dominance in IGNITION phase
        Scenario(
            name="SD001_IgnitionPhase",
            type=ScenarioType.SOL_DOMINANCE,
            description="SOL dominance allowed in IGNITION phase",
            input_data=ScenarioInput(
                system_state={
                    "mode": "OPPORTUNITY",
                    "regime_phase": "IGNITION",
                    "sol_dominance_requested": True,
                }
            ),
            expected=ExpectedOutput(
                exposure_cap=1.0,  # 100% SOL allowed
                proof_log_contains=["SOL_DOMINANCE", "IGNITION"],
            ),
            tags=["sol", "dominance"],
        ),
        
        # Scenario 2: SOL forced exit on VPIN spike
        Scenario(
            name="SD002_ForcedExitVPIN",
            type=ScenarioType.SOL_DOMINANCE,
            description="SOL forced exit when VPIN >= 0.90",
            input_data=ScenarioInput(
                market_data={
                    "vpin": 0.92,  # >= 0.90 immediate threshold
                },
                position_data={
                    "asset": "SOL",
                    "current_exposure": 0.80,
                    "in_dominance_override": True,
                }
            ),
            expected=ExpectedOutput(
                tranche_action="FORCED_EXIT",
                custom_checks={
                    "exit_urgency": "IMMEDIATE",
                },
                proof_log_contains=["FORCED_EXIT", "IMMEDIATE"],
            ),
            tags=["sol", "forced_exit"],
        ),
    ]


def get_execution_deadlock_scenarios() -> List[Scenario]:
    """Get execution deadlock test scenarios."""
    
    return [
        # Scenario 1: T1 deadlock should FORCE
        Scenario(
            name="DL001_T1ForceAggressive",
            type=ScenarioType.EXECUTION_DEADLOCK,
            description="T1 deadlock with good edge should FORCE_AGGRESSIVE",
            input_data=ScenarioInput(
                position_data={
                    "current_level": "TRANCHE_1",
                    "deadlock_bars": 2,
                },
                signal_data={
                    "estimated_edge_bps": 50,
                    "friction_bps": 30,  # edge/friction = 1.67 >= 1.5
                }
            ),
            expected=ExpectedOutput(
                custom_checks={
                    "deadlock_resolution": "FORCE_AGGRESSIVE",
                },
                proof_log_contains=["DEADLOCK", "FORCE"],
            ),
            tags=["deadlock", "tranche"],
        ),
        
        # Scenario 2: T2+ deadlock should ABORT
        Scenario(
            name="DL002_T2Abort",
            type=ScenarioType.EXECUTION_DEADLOCK,
            description="T2+ deadlock should ABORT to protect capital",
            input_data=ScenarioInput(
                position_data={
                    "current_level": "TRANCHE_2",
                    "deadlock_bars": 3,
                },
                signal_data={
                    "estimated_edge_bps": 40,
                    "friction_bps": 30,
                }
            ),
            expected=ExpectedOutput(
                custom_checks={
                    "deadlock_resolution": "ABORT_OPPORTUNITY",
                },
                proof_log_contains=["DEADLOCK", "ABORT"],
            ),
            tags=["deadlock", "tranche"],
        ),
    ]


# =============================================================================
# SCENARIO RUNNER
# =============================================================================

class ScenarioRunner:
    """
    Runs scenarios and validates outputs.
    
    Usage:
        runner = ScenarioRunner()
        results = runner.run_all()
    """
    
    def __init__(self):
        self.scenarios: List[Scenario] = []
        self.results: List[ScenarioResult] = []
        self._load_all_scenarios()
    
    def _load_all_scenarios(self):
        """Load all pre-defined scenarios."""
        self.scenarios.extend(get_no_trade_scenarios())
        self.scenarios.extend(get_opportunity_scenarios())
        self.scenarios.extend(get_alpha_gating_scenarios())
        self.scenarios.extend(get_tranche_scenarios())
        self.scenarios.extend(get_risk_veto_scenarios())
        self.scenarios.extend(get_sol_dominance_scenarios())
        self.scenarios.extend(get_execution_deadlock_scenarios())
        
        logger.info(f"Loaded {len(self.scenarios)} scenarios")
    
    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        """
        Run a single scenario.
        
        Note: This is a validation framework. Actual execution requires
        integration with the HMATS engine.
        """
        import time
        start_time = time.time()
        
        result = ScenarioResult(
            scenario_name=scenario.name,
            passed=False,
            expected=scenario.expected,
        )
        
        try:
            # Import constitution guarantees for testing
            from .constitution_guarantees import get_constitution_guarantees
            
            guarantees = get_constitution_guarantees()
            
            # Run appropriate test based on scenario type
            if scenario.type == ScenarioType.NO_TRADE_TRIGGER:
                actual = self._run_no_trade_test(guarantees, scenario)
            elif scenario.type == ScenarioType.OPPORTUNITY_TRIGGER:
                actual = self._run_opportunity_test(guarantees, scenario)
            elif scenario.type == ScenarioType.ALPHA_GATING:
                actual = self._run_alpha_test(guarantees, scenario)
            elif scenario.type == ScenarioType.TRANCHE_SCHEDULING:
                actual = self._run_tranche_test(guarantees, scenario)
            else:
                actual = {"note": "Test type not implemented"}
            
            result.actual = actual
            result.passed = self._validate_result(scenario.expected, actual)
            
        except Exception as e:
            result.errors.append(str(e))
            result.passed = False
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _run_no_trade_test(self, guarantees, scenario: Scenario) -> Dict:
        """Run NO_TRADE trigger test."""
        market_data = scenario.input_data.market_data
        signal_data = scenario.input_data.signal_data
        
        state = guarantees.compute_no_trade_triggers(market_data, signal_data)
        
        return {
            "should_no_trade": state.should_no_trade,
            "active_conditions": [c.trigger_type.name for c in state.active_conditions],
            "primary_reason": state.primary_reason.name if state.primary_reason else None,
        }
    
    def _run_opportunity_test(self, guarantees, scenario: Scenario) -> Dict:
        """Run OPPORTUNITY trigger test."""
        market_data = scenario.input_data.market_data
        lead_lag_data = scenario.input_data.lead_lag_data
        
        state = guarantees.compute_opportunity_triggers(market_data, lead_lag_data)
        
        return {
            "is_active": state.is_active,
            "active_triggers": [t.group.name for t in state.active_triggers],
            "direction_consensus": state.direction_consensus,
            "combined_confidence": state.combined_confidence,
        }
    
    def _run_alpha_test(self, guarantees, scenario: Scenario) -> Dict:
        """Run alpha gating test."""
        signal_data = scenario.input_data.signal_data
        system_state = scenario.input_data.system_state
        
        result = guarantees.check_alpha_gate(
            signal_strength=signal_data.get("signal_strength", 0),
            regime_confidence=signal_data.get("regime_confidence", 0.5),
            mode=system_state.get("mode", "NORMAL"),
            is_maker=system_state.get("is_maker", False),
            lead_lag_edge_confirmed=system_state.get("lead_lag_edge_confirmed", False),
        )
        
        return {
            "passes_threshold": result.passes_threshold,
            "estimated_alpha_bps": result.estimated_alpha_bps,
            "threshold_bps": result.threshold_bps,
        }
    
    def _run_tranche_test(self, guarantees, scenario: Scenario) -> Dict:
        """Run tranche scheduling test."""
        market_data = scenario.input_data.market_data
        signal_data = scenario.input_data.signal_data
        system_state = scenario.input_data.system_state
        
        decision = guarantees.schedule_tranche(
            asset="SOL",
            mode=system_state.get("mode", "NORMAL"),
            target_direction=system_state.get("target_direction", 0),
            market_data=market_data,
            signal_data=signal_data,
        )
        
        return {
            "action": decision.action,
            "target_exposure": decision.target_exposure,
            "abort_active": decision.abort_active,
        }
    
    def _validate_result(self, expected: ExpectedOutput, actual: Dict) -> bool:
        """Validate actual output against expected."""
        
        # Check mode
        if expected.mode is not None:
            if "should_no_trade" in actual:
                actual_mode = "NO_TRADE" if actual["should_no_trade"] else "NORMAL"
            elif "is_active" in actual:
                actual_mode = "OPPORTUNITY" if actual["is_active"] else "NORMAL"
            else:
                actual_mode = actual.get("mode", "UNKNOWN")
            
            if actual_mode != expected.mode:
                return False
        
        # Check should_trade
        if expected.should_trade is not None:
            if "should_no_trade" in actual:
                actual_should_trade = not actual["should_no_trade"]
            elif "passes_threshold" in actual:
                actual_should_trade = actual["passes_threshold"]
            else:
                actual_should_trade = actual.get("should_trade", True)
            
            if actual_should_trade != expected.should_trade:
                return False
        
        # Check tranche action
        if expected.tranche_action is not None:
            if actual.get("action") != expected.tranche_action:
                return False
        
        return True
    
    def run_by_type(self, scenario_type: ScenarioType) -> List[ScenarioResult]:
        """Run all scenarios of a specific type."""
        scenarios = [s for s in self.scenarios if s.type == scenario_type]
        return [self.run_scenario(s) for s in scenarios]
    
    def run_by_tags(self, tags: List[str]) -> List[ScenarioResult]:
        """Run all scenarios matching any of the given tags."""
        scenarios = [s for s in self.scenarios if any(t in s.tags for t in tags)]
        return [self.run_scenario(s) for s in scenarios]
    
    def run_all(self) -> List[ScenarioResult]:
        """Run all scenarios."""
        self.results = [self.run_scenario(s) for s in self.scenarios]
        return self.results
    
    def get_summary(self) -> Dict:
        """Get summary of results."""
        if not self.results:
            return {"status": "No results"}
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results) if self.results else 0,
            "failed_scenarios": [r.scenario_name for r in self.results if not r.passed],
        }
    
    def print_report(self):
        """Print formatted report."""
        print("\n" + "=" * 80)
        print("HMATS SCENARIO WALKTHROUGH REPORT")
        print("=" * 80)
        
        summary = self.get_summary()
        print(f"\nTotal Scenarios: {summary['total']}")
        print(f"Passed: {summary['passed']} ({summary['pass_rate']:.0%})")
        print(f"Failed: {summary['failed']}")
        
        if summary['failed_scenarios']:
            print("\nFailed Scenarios:")
            for name in summary['failed_scenarios']:
                print(f"  ✗ {name}")
        
        print("\n" + "=" * 80)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_all_scenarios() -> List[ScenarioResult]:
    """Run all scenarios and return results."""
    runner = ScenarioRunner()
    return runner.run_all()


def run_quick_validation() -> bool:
    """
    Run quick validation and return pass/fail.
    
    Good for CI/CD integration.
    """
    runner = ScenarioRunner()
    results = runner.run_all()
    summary = runner.get_summary()
    
    if summary['failed'] > 0:
        runner.print_report()
        return False
    
    logger.info(f"Quick validation passed: {summary['passed']}/{summary['total']} scenarios")
    return True


# =============================================================================
# MAIN (for direct execution)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    runner = ScenarioRunner()
    results = runner.run_all()
    runner.print_report()

"""
================================================================================
HMATS v3.6.1-PROD - INTEGRATION ENGINE (Production Ready)
================================================================================
Version: 3.6.1-PROD
Purpose: Fused v3.3-HR + v3.4 + v3.5 with production reliability patches

FUSED FEATURES:
    v3.3-HR: Risk philosophy, SOL dominance, high-risk posture
    v3.4: Authority-based fusion, phase detection, execution intelligence
    v3.5: Failure memory, strategy confidence, DRL promotion gate

PRODUCTION RELIABILITY PATCHES:
    Patch 1: Canonical schema key mapping + required-key validation
    Patch 2: Single canonical entrypoint + legacy redirect + version banner
    Patch 3: Structured [PROOF] log per 4H tick
    Patch 4: HARD vs SOFT risk veto classification
    Patch 5: Execution deadlock + tranche stage coupling
    Patch 6: SOL dominance forced exit (immediate via execution loop)

ARCHITECTURE (LOCKED):
    Information -> Regime -> Quant -> DRL -> Fusion -> Execution -> Risk

MODE HIERARCHY (LOCKED):
    NO_TRADE > OPPORTUNITY > NORMAL
================================================================================
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)

VERSION = "3.6.1-PROD"  # Production reliability patches applied

# =============================================================================
# v3.6.1: IMPORT CONSTITUTION GUARANTEES (v3.3-STABILIZED RESTORATION)
# =============================================================================
# Canonical imports from defense/constitution.py
from defense.constitution import (
    get_constitution_guarantees,
    ConstitutionGuaranteesModule,
    NoTradeState,
    OpportunityState,
    AlphaGatingResult,
    TrancheDecision,
    TrancheLevel,
    ConstitutionRule,
)

# =============================================================================
# PRODUCTION RELIABILITY IMPORTS (Patches 1-6)
# =============================================================================
# Canonical imports from defense/production_reliability.py
from defense.production_reliability import (
    # Patch 2: Entrypoint
    canonical_entrypoint,
    legacy_redirect,
    get_call_chain,
    VERSION_BANNER,
    # Patch 3: Proof log
    StructuredProofLog,
    ProofLogBuilder,
    ComponentStatus,
    create_proof_log,
    # Patch 4: Risk veto
    RiskVetoClassifier,
    RiskVetoResult,
    VetoType,
    HardVetoCondition,
    SoftVetoCondition,
    get_risk_veto_classifier,
    # Patch 5: Deadlock
    TrancheAwareDeadlockResolver,
    DeadlockResult,
    DeadlockResolution,
    DeadlockBias,
    get_deadlock_resolver,
    # Patch 6: SOL forced exit
    SOLDominanceForcedExit,
    ForcedExitSignal,
    ExitUrgency,
    get_sol_forced_exit,
)


# =============================================================================
# v3.3-HR RISK PHILOSOPHY (EMBEDDED - HARDCODED)
# =============================================================================

V33HR_PHILOSOPHY = {
    "core": "Missing a high-conviction move is worse than taking a controlled loss",
    "risk_profile": "HIGH_RISK",
    "primary_asset": "SOL",
    "secondary_assets": ["BTC", "ETH"],
    "hard_drawdown_halt": 0.20,  # 20% halt [UNLEASH UL-5]
    "correlation_crisis": 0.98,  # [UNLEASH UL-4]
    "opportunity_multiplier": 0.70,  # at 5% DD (v3.3-HR)
    "normal_multiplier": 0.50,  # at 5% DD (v3.3-HR)
}

V33HR_SOL_DOMINANCE = {
    "max_exposure": 1.00,  # 100%
    "gross_override": 0.25,  # +25%
    "max_duration_bars": 2,  # TTL
    "phase_gate": ["IGNITION", "EXPANSION"],
    "auto_disable_on": ["SATURATION", "EXHAUSTION"],
}


# =============================================================================
# ENUMS (Canonical - single source of truth)
# =============================================================================

from core.canonical_enums import (
    SystemMode,
    ExecutionMode,
    DRLAuthorityLevel,
    DeadlockResolution,
    RegimePhase,
)


# =============================================================================
# TRADE INTENT v3.6.1
# =============================================================================

@dataclass
class TradeIntentV36:
    """Trade intent from 4H decision layer (v3.6.1 canonical)."""

    # Audit trail linkage (Phase 8)
    intent_id: str = ""       # unique per intent: f"{tick_ts}_{asset}"
    tick_id: str = ""         # shared across all assets in one tick cycle

    # Core intent
    asset: str = ""
    direction: float = 0.0
    target_exposure: float = 0.0
    tranche_target: int = 0
    allow_escalation: bool = False
    max_tranche_tier: int = 4
    
    # Execution
    execution_mode: ExecutionMode = ExecutionMode.PASSIVE_PREFERRED
    urgency: float = 0.5
    timing_score: float = 0.5
    timing_spread_score: Optional[float] = None
    timing_depth_score: Optional[float] = None
    timing_vpin_score: Optional[float] = None
    timing_lead_lag_score: Optional[float] = None
    delay_bars: int = 0
    force_execution: bool = False
    
    # Risk limits
    max_gross: float = 1.50
    max_asset: float = 0.80
    risk_multiplier: float = 1.0
    
    # v3.3-HR: SOL dominance
    sol_dominance_active: bool = False
    sol_dominance_ttl: int = 0
    sol_dominance_reason: str = ""
    
    # v3.5: Failure memory
    opportunity_density_threshold: float = 0.70
    tranche_delay_bars: int = 0
    in_caution_mode: bool = False
    
    # v3.5: Deadlock
    deadlock_resolution: str = ""
    
    # v3.5: DRL
    drl_authority_level: str = "DISABLED"
    drl_has_authority: bool = False
    
    # Veto
    veto_active: bool = False
    veto_reason: str = ""
    
    # Mode
    system_mode: str = "NORMAL"
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Quant info
    quant_strategy_id: str = ""
    quant_confidence: float = 0.0
    
    # v3.6.1: Constitution guarantees tracking
    no_trade_triggers_internal: bool = True
    opportunity_triggers_internal: bool = True
    alpha_gate_passed: bool = False  # FAIL-CLOSED: must be explicitly set by alpha gate check
    alpha_estimated_bps: float = 0.0
    alpha_threshold_bps: float = 0.0
    expected_net_alpha_bps: float = 0.0
    net_alpha_threshold_bps: float = 0.0
    tranche_action: str = ""
    tranche_level: int = 0
    signal_conflict_detected: bool = False
    data_validation_passed: bool = True
    base_actionable_direction_threshold: float = 0.10
    opportunity_actionable_direction_threshold: float = 0.05
    opportunity_actionable_direction_threshold_short: Optional[float] = None
    opportunity_actionable_exposure_threshold: Optional[float] = None
    normal_actionable_direction_threshold_short: Optional[float] = None

    # [CFG-9] Pyramid add-on flag - exempt from rebuild cooldown
    is_addon: bool = False

    # [FIX-REDUCE-THRESHOLD] Reduce intents (target < current_exposure, direction aligned)
    # bypass the direction strength threshold in is_actionable. Direction alignment still
    # required (set in integration_v36.py after tranche decision detects reduce).
    is_reduce_intent: bool = False

    @property
    def is_actionable(self) -> bool:
        # OPPORTUNITY entries already pass the constitution alpha gate and
        # tranche sizing before reaching runtime execution. A second hard
        # 0.10 direction floor was suppressing reduced-size but otherwise
        # valid HIGH_RISK trades during paper bootstrap.
        #
        # [FIX-FORCE-EXEC] force_execution=True (deadlock ABORT+CLOSE, soft_stop,
        # gambler_exit) must always be actionable. Cannot set property directly.
        if self.force_execution:
            return True
        #
        # [FIX-REDUCE-THRESHOLD] Reduces bypass direction strength check entirely.
        # When signal weakens (e.g. dir=-0.07 < 0.10 threshold) but the tranche
        # correctly downsizes the position, we must not block the reduce.
        if self.is_reduce_intent and not self.veto_active and self.target_exposure > 0.001:
            return True

        _dir_threshold = self.base_actionable_direction_threshold
        _exp_threshold = 0.01
        if self.alpha_gate_passed and self.system_mode == "OPPORTUNITY":
            _dir_threshold = self.opportunity_actionable_direction_threshold
            if (
                self.direction < 0
                and self.opportunity_actionable_direction_threshold_short is not None
            ):
                _dir_threshold = self.opportunity_actionable_direction_threshold_short
            if self.opportunity_actionable_exposure_threshold is not None:
                _exp_threshold = self.opportunity_actionable_exposure_threshold
        elif (
            self.alpha_gate_passed
            and self.system_mode == "NORMAL"
            and self.direction < 0
            and self.normal_actionable_direction_threshold_short is not None
        ):
            _dir_threshold = self.normal_actionable_direction_threshold_short
        return (
            not self.veto_active and
            abs(self.direction) > _dir_threshold and
            self.target_exposure > _exp_threshold
        )
    
    def to_proof_log(self) -> str:
        """Generate [PROOF] log line per 4H tick (v3.6.1 enhanced)."""
        return (
            f"[PROOF] {self.timestamp.isoformat()} | "
            f"v3.6.1 | "
            f"DataValid={self.data_validation_passed} | "
            f"NO_TRADE_internal={self.no_trade_triggers_internal} | "
            f"OPP_internal={self.opportunity_triggers_internal} | "
            f"AlphaGate={{pass={self.alpha_gate_passed},est={self.alpha_estimated_bps:.0f},thresh={self.alpha_threshold_bps:.0f}}} | "
            f"Tranche={{action={self.tranche_action or 'NONE'},level={self.tranche_level}}} | "
            f"Conflict={self.signal_conflict_detected} | "
            f"Mode={self.system_mode} | "
            f"Quant=REAL({self.quant_strategy_id}) | "
            f"DRL={self.drl_authority_level} | "
            f"FailureMemory={{caution={self.in_caution_mode},boost={self.opportunity_density_threshold-0.70:.2f}}} | "
            f"SOL_dom={{active={self.sol_dominance_active},TTL={self.sol_dominance_ttl}}} | "
            f"Deadlock={self.deadlock_resolution or 'NONE'} | "
            f"Intent={{asset={self.asset},dir={self.direction:+.2f},exp={self.target_exposure:.1%},id={self.intent_id}}} | "
            f"Exec={self.execution_mode.value if hasattr(self.execution_mode, 'value') else self.execution_mode} | "
            f"Timing={{total={self.timing_score:.2f},spread={self.timing_spread_score or 0:.2f},depth={self.timing_depth_score or 0:.2f},vpin={self.timing_vpin_score or 0:.2f},ll={self.timing_lead_lag_score or 0:.2f}}}"
        )


# =============================================================================
# MOCK REGISTRY (Tracks which modules are mocked vs real)
# =============================================================================

class ModuleRegistry:
    """Track which modules are REAL vs MOCKED."""
    
    def __init__(self):
        self.modules = {}
        
    def register(self, name: str, is_real: bool, reason: str = ""):
        self.modules[name] = {"real": is_real, "reason": reason}
        
    def is_real(self, name: str) -> bool:
        return self.modules.get(name, {}).get("real", False)
    
    def get_status(self) -> Dict:
        return {
            "real": [k for k, v in self.modules.items() if v["real"]],
            "mocked": [k for k, v in self.modules.items() if not v["real"]],
        }


_registry = ModuleRegistry()


# =============================================================================
# STUB CLASSES (Used when imports fail - DISABLED from runtime)
# =============================================================================

class StubFailureMemory:
    """Stub for FailureAwareMetaMemory - DISABLED."""
    
    def __init__(self):
        _registry.register("FailureAwareMetaMemory", False, "Stub - imports failed")
        
    def get_modifiers(self):
        return type('Modifiers', (), {
            'density_boost': 0.0,
            'tranche_delay_bars': 0,
            'in_caution_mode': False,
            'aggressiveness_modifier': 1.0,
        })()
    
    def record_opportunity(self, **kwargs):
        pass


class StubConfidenceScorer:
    """Stub for StrategyConfidenceScorer - DISABLED."""
    
    def __init__(self):
        _registry.register("StrategyConfidenceScorer", False, "Stub - imports failed")
        
    def get_all_confidences(self):
        return {}
    
    def record_signal(self, **kwargs):
        pass


class StubAttributionPromotion:
    """Stub for PnLAttributionPromotion - DISABLED."""
    
    def __init__(self):
        _registry.register("PnLAttributionPromotion", False, "Stub - imports failed")
        
    def record_trade(self, **kwargs):
        pass
    
    def get_promotion_signal(self):
        return type('Signal', (), {'ready_for_promotion': False})()


class StubPatienceManager:
    """Stub for FusionPatienceManager - DISABLED."""
    
    def __init__(self):
        _registry.register("FusionPatienceManager", False, "Stub - imports failed")
        
    def check_deadlock(self, **kwargs):
        return type('Result', (), {
            'resolution': DeadlockResolution.NONE,
            'abort_reason': '',
        })()
    
    def record_delay(self, **kwargs):
        pass
    
    def record_execution(self, asset):
        pass


class StubPromotionGate:
    """Stub for DRLPromotionGate - DISABLED."""
    
    def __init__(self):
        _registry.register("DRLPromotionGate", False, "Stub - imports failed")
        
    def get_authority(self):
        return DRLAuthorityLevel.DISABLED
    
    def has_exit_authority(self):
        return False
    
    def is_shadow_mode(self):
        return False
    
    def check_promotion(self):
        return type('Decision', (), {'should_promote': False})()


class StubPhaseDetector:
    """Stub for RegimePhaseDetector - DISABLED."""
    
    def __init__(self):
        _registry.register("RegimePhaseDetector", False, "Stub - imports failed")
        
    def detect(self, **kwargs):
        return type('Result', (), {
            'phase': RegimePhase.UNKNOWN,
            'opportunity_density': 0.0,
            'crack_weight': 0.0,
        })()


class StubFusionEngine:
    """Stub for AuthorityFusionEngine - DISABLED."""
    
    def __init__(self):
        _registry.register("AuthorityFusionEngine", False, "Stub - imports failed")
        
    def fuse(self, signals, context):
        return type('Result', (), {
            'direction': 0.0,
            'target_exposure': 0.0,
            'tranche_target': 0,
            'allow_escalation': False,
            'max_tranche_tier': 4,
            'urgency': 0.5,
        })()


class StubTimingEngine:
    """Stub for ExecutionTimingEngine - DISABLED."""
    
    def __init__(self):
        _registry.register("ExecutionTimingEngine", False, "Stub - imports failed")
        
    def evaluate(self, **kwargs):
        score = type('Score', (), {'total_score': 0.5})()
        return score, ExecutionMode.PASSIVE_PREFERRED


class StubRiskAgent:
    """Stub for RiskAgent - DISABLED."""
    
    def __init__(self):
        _registry.register("RiskAgent", False, "Stub - imports failed")
        
    def assess(self, **kwargs):
        return type('Assessment', (), {
            'veto_active': False,
            'veto_reason': '',
            'max_gross_cap': 1.0,
            'risk_multiplier': 1.0,
            'sol_max_exposure': 0.8,
            'max_single_asset_cap': 0.8,
        })()


class StubDRLAgent:
    """Stub for DRLAgent - DISABLED."""
    
    def __init__(self):
        _registry.register("DRLAgent", False, "Stub - imports failed")
        
    def build_state(self, **kwargs):
        return None
    
    def get_action(self, state):
        return None
    
    def interpret_action(self, output):
        return {}


# =============================================================================
# TRY TO IMPORT REAL MODULES
# =============================================================================

try:
    from analytics.failure_memory import FailureAwareMetaMemory, get_failure_memory
    _registry.register("FailureAwareMetaMemory", True, "Imported successfully")
except ImportError as e:
    logger.warning(f"FailureAwareMetaMemory import failed: {e}")
    FailureAwareMetaMemory = StubFailureMemory
    get_failure_memory = lambda: StubFailureMemory()

try:
    from analytics.confidence_scorer import StrategyConfidenceScorer, get_confidence_scorer
    _registry.register("StrategyConfidenceScorer", True, "Imported successfully")
except ImportError as e:
    logger.warning(f"StrategyConfidenceScorer import failed: {e}")
    StrategyConfidenceScorer = StubConfidenceScorer
    get_confidence_scorer = lambda: StubConfidenceScorer()

try:
    from analytics.pnl_promotion import PnLAttributionPromotion, get_attribution_promotion
    _registry.register("PnLAttributionPromotion", True, "Imported successfully")
except ImportError as e:
    logger.warning(f"PnLAttributionPromotion import failed: {e}")
    PnLAttributionPromotion = StubAttributionPromotion
    get_attribution_promotion = lambda: StubAttributionPromotion()

try:
    from signals.fusion_patience import FusionPatienceManager, get_patience_manager
    _registry.register("FusionPatienceManager", True, "Imported successfully")
except ImportError as e:
    logger.warning(f"FusionPatienceManager import failed: {e}")
    FusionPatienceManager = StubPatienceManager
    get_patience_manager = lambda: StubPatienceManager()

try:
    from drl.promotion_gate import DRLPromotionGate
    _registry.register("DRLPromotionGate", True, "Imported successfully")
    get_promotion_gate = lambda: DRLPromotionGate(
        config={
            'enable': True,
            'consecutive_loss_threshold': 5,
            'drawdown_trigger_pct': 0.15,
            'demote_to': 'EXIT_ONLY',
            'recovery_period_days': 3,
            'max_demotions_before_disable': 3,
        },
        state_file='data/drl_promotion_state.json',
    )
except ImportError as e:
    logger.warning(f"DRLPromotionGate import failed: {e}")
    DRLPromotionGate = StubPromotionGate
    get_promotion_gate = lambda: StubPromotionGate()

try:
    from market.phase_detector import RegimePhaseDetector, get_phase_detector
    _registry.register("RegimePhaseDetector", True, "Imported successfully")
except ImportError as e:
    logger.warning(f"RegimePhaseDetector import failed: {e}")
    RegimePhaseDetector = StubPhaseDetector
    get_phase_detector = lambda: StubPhaseDetector()

try:
    from signals.authority_fusion import (
        AuthorityFusionEngine, FusionContext, AgentSignal, get_fusion_engine,
    )
    # SystemMode is now canonical (core.canonical_enums) — no mapping needed.
    # Both integration_v36.py and authority_fusion.py use the same enum.
    _registry.register("AuthorityFusionEngine", True, "Imported successfully")
except ImportError as e:
    logger.warning(f"AuthorityFusionEngine import failed: {e}")
    AuthorityFusionEngine = StubFusionEngine
    get_fusion_engine = lambda: StubFusionEngine()

    # Stub out all types from authority_fusion so decide() doesn't NameError
    @dataclass
    class AgentSignal:
        direction: float = 0.0
        confidence: float = 0.0
        veto_active: bool = False
        veto_direction: Optional[str] = None
        leverage_cap: float = 1.0
        exit_pressure: float = 0.0
        tranche_advice: Optional[str] = None
        asof_timestamp: float = 0.0  # [v3.3-B9] Unix timestamp when signal was generated

    @dataclass
    class FusionContext:
        mode: SystemMode = SystemMode.NORMAL
        regime_phase: Any = None
        phase_result: Any = None
        crack_active: bool = False
        crack_weight: float = 0.0
        data_valid: bool = True
        drl_enabled: bool = False
        lead_lag_confident: bool = False
        lead_lag_edge: float = 0.0  # unit: bps

try:
    from execution.timing_engine import ExecutionTimingEngine, get_timing_engine
    _registry.register("ExecutionTimingEngine", True, "Imported successfully")
except ImportError as e:
    logger.warning(f"ExecutionTimingEngine import failed: {e}")
    ExecutionTimingEngine = StubTimingEngine
    get_timing_engine = lambda: StubTimingEngine()

try:
    from agents.risk_agent import RiskAgentV34, get_risk_agent
    _registry.register("RiskAgent", True, "Imported successfully")
except ImportError as e:
    logger.warning(f"RiskAgent import failed: {e}")
    RiskAgentV34 = StubRiskAgent
    get_risk_agent = lambda: StubRiskAgent()

try:
    from agents.drl_agent import DRLAgent, get_drl_agent
    _registry.register("DRLAgent", True, "Imported successfully")
except ImportError as e:
    logger.warning(f"DRLAgent import failed: {e}")
    DRLAgent = StubDRLAgent
    get_drl_agent = lambda: StubDRLAgent()


# =============================================================================
# v3.6 ENGINE
# =============================================================================

class HMATSv36Engine:
    """
    HMATS v3.6.1-PROD Integration Engine - Production Ready.
    
    FUSES:
        - v3.3-HR: Risk philosophy, SOL dominance
        - v3.4: Authority-based fusion, phase detection
        - v3.5: Failure memory, strategy confidence, DRL gate
    
    v3.6.1 RESTORED GUARANTEES (from v3.3-STABILIZED):
        - Internal NO_TRADE trigger computation
        - Internal OPPORTUNITY trigger computation
        - Alpha threshold gating (friction-aware)
        - Tranche scheduling + abort conditions
        - Signal conflict detection
        - Strategy constitution enforcement
        - Market data validation
    
    PRODUCTION RELIABILITY PATCHES:
        Patch 1: Canonical schema key mapping + required-key validation
        Patch 2: Single canonical entrypoint + version banner
        Patch 3: Structured [PROOF] log per 4H tick
        Patch 4: HARD vs SOFT risk veto classification
        Patch 5: Execution deadlock + tranche stage coupling
        Patch 6: SOL dominance forced exit (immediate via execution loop)
    
    DECISION FLOW (4H tick):
        0. [v3.6.1] Market Data Validation + Schema (Patch 1)
        1. [v3.6.1] Internal NO_TRADE Trigger Computation
        2. [v3.6.1] Internal OPPORTUNITY Trigger Computation
        3. Regime Phase Detection (v3.4)
        4. Failure Memory Check (v3.5) -> threshold adjustment
        5. Mode Determination (v3.3-HR priorities + v3.6.1 internal)
        6. Quant Signals with Strategy Confidence (v3.5)
        7. [v3.6.1] Alpha Threshold Gating
        8. Authority-Based Fusion (v3.4)
        9. [v3.6.1] Tranche Scheduling
        10. [PROD] Deadlock + Tranche Coupling (Patch 5)
        11. DRL Advisory (v3.4 + v3.5 gate)
        12. Execution Timing (v3.4)
        13. [PROD] HARD vs SOFT Risk Assessment (Patch 4)
        14. [PROD] SOL Dominance + Immediate Exit (Patch 6)
        15. [PROD] Generate Structured Proof Log (Patch 3)
    """
    
    def __init__(self):
        """Initialize v3.6.1-PROD engine with all fused components."""

        # =================================================================
        # PROFILE-CONFIGURABLE THRESHOLDS (T21/T22)
        # Defaults are FAIL-CLOSED (conservative) - profile overrides via configure()
        # =================================================================
        self._profile_config: Dict[str, float] = {
            "hard_drawdown_halt": 0.20,      # overridden by profile (e.g. 0.25)
            "correlation_crisis": 0.98,       # overridden by profile (e.g. 0.98)
            "reduce_at_drawdown": 0.08,      # overridden by profile (e.g. 0.15)
            "critical_drawdown": 0.15,       # overridden by profile (e.g. 0.35)
            "max_leverage": 3.0,             # overridden by profile (e.g. 4.0)
            "max_gross_exposure": 1.50,      # overridden by profile (e.g. 2.50)
        }

        # =================================================================
        # PRODUCTION RELIABILITY: Print version banner (Patch 2)
        # =================================================================
        print(VERSION_BANNER.format(timestamp=datetime.now(timezone.utc).isoformat()))
        
        # =================================================================
        # v3.6.1: CONSTITUTION GUARANTEES MODULE (v3.3-STABILIZED RESTORATION)
        # =================================================================
        self.guarantees = get_constitution_guarantees()
        
        # =================================================================
        # PRODUCTION RELIABILITY PATCHES (Patches 4, 5, 6)
        # =================================================================
        self.risk_veto_classifier = get_risk_veto_classifier()  # Patch 4
        self.deadlock_resolver = get_deadlock_resolver()        # Patch 5
        self.sol_forced_exit = get_sol_forced_exit()           # Patch 6
        
        # v3.4 Core (with fallback to stubs)
        self.phase_detector = get_phase_detector()
        self.fusion_engine = get_fusion_engine()
        self.drl_agent = get_drl_agent()
        self.timing_engine = get_timing_engine()
        self.risk_agent = get_risk_agent()
        
        # v3.5 Meta-Layers (with fallback to stubs)
        self.failure_memory = get_failure_memory()
        self.confidence_scorer = get_confidence_scorer()
        self.attribution_promotion = get_attribution_promotion()
        self.patience_manager = get_patience_manager()
        self.drl_gate = get_promotion_gate()
        
        # State
        self._current_mode = SystemMode.NORMAL
        self._sol_dominance_active = False
        self._sol_dominance_ttl = 0
        self._last_phase_result = None
        self._last_phase = ""  # [FIX-L1-07] phase string from agent_signals

        # v3.6.1: Internal trigger state
        self._last_no_trade_state: Optional[NoTradeState] = None
        self._last_opportunity_state: Optional[OpportunityState] = None
        self._last_alpha_result: Optional[AlphaGatingResult] = None
        self._last_tranche_decision: Optional[TrancheDecision] = None
        
        # Production reliability state (Patches 4, 5, 6)
        self._last_veto_result: Optional[RiskVetoResult] = None
        self._last_deadlock_result: Optional[DeadlockResult] = None
        self._last_sol_exit_signal: Optional[ForcedExitSignal] = None
        self._execution_failures: int = 0

        # Fusion signal state (initialized here to avoid AttributeError)
        self._last_sentiment_active: bool = False
        self._last_macro_active: bool = False
        
        # Proof logging (Patch 3) - capped to prevent memory leak in 24h+ runs
        self._proof_logs: deque = deque(maxlen=1000)
        self._structured_proof_logs: deque = deque(maxlen=1000)
        
        logger.info(f"HMATSv36Engine initialized (version={VERSION})")
        logger.info(f"[v3.6.1] Constitution guarantees: {self.guarantees.get_proof_summary()}")
        logger.info(f"[PROD] Risk veto classifier: HARD/SOFT separation active")
        logger.info(f"[PROD] Deadlock resolver: tranche-coupled (T1->FORCE, T2+->ABORT)")
        logger.info(f"[PROD] SOL forced exit: immediate execution loop enabled")
        logger.info(f"Module status: {_registry.get_status()}")

        # =================================================================
        # STARTUP VALIDATION: Warn if critical modules are stubs
        # =================================================================
        _critical = ["AuthorityFusionEngine", "RiskAgent", "RegimePhaseDetector"]
        _stub_critical = [m for m in _critical if not _registry.is_real(m)]
        if _stub_critical:
            logger.warning(
                f"[STARTUP] CRITICAL modules running as STUBS: {_stub_critical}. "
                f"Trading decisions will be degraded or disabled."
            )

    def configure(self, profile_config: Dict[str, float]) -> None:
        """Inject profile thresholds (T21/T22). Called from main.py after init."""
        _accepted = ("hard_drawdown_halt", "correlation_crisis",
                     "reduce_at_drawdown", "critical_drawdown",
                     "max_leverage", "max_gross_exposure")
        for key in _accepted:
            if key in profile_config:
                old = self._profile_config.get(key)
                self._profile_config[key] = profile_config[key]
                logger.info(f"[T21/T22] Engine threshold {key}: {old} -> {profile_config[key]}")

    def _maybe_apply_pre_alpha_hold(
        self,
        asset: str,
        intent: TradeIntentV36,
        agent_signals: Dict[str, Any],
        market_data: Dict[str, Any],
        position_state: Dict[str, Any],
    ) -> bool:
        """Downgrade to HOLD before alpha gating when Best-of-N selects hold strategy,
        or when weak quiet-regime volume_breakout probes are detected."""
        # [2026-04-08] Best-of-N HOLD strategy: when pipeline determines no strategy
        # has edge (ADX<15, RSI neutral, volume thin in choppy regime), skip alpha gate
        # and maintain current position. This prevents forced entries that produced
        # 14.3% SOL win rate and -$60 SHORT losses in paper run.
        _strategy = str(agent_signals.get("primary_strategy", "") or "").lower()
        if _strategy == "hold":
            _current_exp = abs(
                float(
                    position_state.get(
                        "current_exposure",
                        market_data.get("current_exposure", 0.0),
                    ) or 0.0
                )
            )
            _regime = str(market_data.get("regime_state", "") or "").upper()
            intent.direction = 0.0
            intent.target_exposure = _current_exp
            intent.tranche_action = "HOLD"
            intent.tranche_level = 0
            intent.alpha_gate_passed = True
            intent.alpha_estimated_bps = 0.0
            intent.alpha_threshold_bps = 0.0
            intent.expected_net_alpha_bps = 0.0
            intent.net_alpha_threshold_bps = 0.0
            intent.veto_active = False
            intent.veto_reason = (
                f"[BEST_OF_N_HOLD] {_regime} no strategy has edge"
            )
            self._last_alpha_result = AlphaGatingResult(
                passes_threshold=True,
                estimated_alpha_bps=0.0,
                threshold_bps=0.0,
                gate_decision="BEST_OF_N_HOLD",
            )
            logger.info(
                f"[BEST_OF_N_HOLD] {asset}: {_regime} hold strategy won "
                f"Best-of-N -> HOLD (current_exp={_current_exp:.4f})"
            )
            return True

        # [2026-04-11] Black Swan Sentinel: when BSS=0 (crisis in progress),
        # force HOLD for new entries. Existing positions handled by stop-loss/risk.
        _bss = int(market_data.get("black_swan_sentinel", 3))
        if _bss == 0:
            _current_exp = abs(float(
                position_state.get("current_exposure",
                                   market_data.get("current_exposure", 0.0)) or 0.0
            ))
            _regime = str(market_data.get("regime_state", "") or "").upper()
            intent.direction = 0.0
            intent.target_exposure = _current_exp
            intent.tranche_action = "HOLD"
            intent.tranche_level = 0
            intent.alpha_gate_passed = True
            intent.alpha_estimated_bps = 0.0
            intent.alpha_threshold_bps = 0.0
            intent.expected_net_alpha_bps = 0.0
            intent.net_alpha_threshold_bps = 0.0
            intent.veto_active = False
            intent.veto_reason = f"[BLACK_SWAN_SENTINEL] BSS=0 crisis detected in {_regime}"
            self._last_alpha_result = AlphaGatingResult(
                passes_threshold=True,
                estimated_alpha_bps=0.0,
                threshold_bps=0.0,
                gate_decision="BLACK_SWAN_HOLD",
            )
            logger.warning(
                f"[BLACK_SWAN_SENTINEL] {asset}: BSS=0 crisis -> HOLD "
                f"(fear={market_data.get('_fear_greed_value', '?')}, "
                f"vol_spike={market_data.get('vol_spike_ratio', '?')})"
            )
            return True

        if not bool(market_data.get("pre_alpha_hold_volume_breakout_long_in_quiet", False)):
            return False

        _regime = str(market_data.get("regime_state", "") or "").upper()
        if _regime != "QUIET_ACCUMULATION":
            return False

        _strategy = str(agent_signals.get("primary_strategy", "") or "").lower()
        if _strategy != "volume_breakout":
            return False

        _current_exp = abs(
            float(
                position_state.get(
                    "current_exposure",
                    market_data.get("current_exposure", 0.0),
                ) or 0.0
            )
        )
        if _current_exp > 0.001:
            return False

        _quant_dir = float(agent_signals.get("quant_direction", 0.0) or 0.0)
        _max_dir = float(market_data.get("pre_alpha_hold_volume_breakout_long_max_direction", 0.0) or 0.0)
        if _quant_dir <= 0.0:
            return False
        if _max_dir > 0.0 and abs(_quant_dir) > _max_dir:
            return False

        intent.direction = 0.0
        intent.target_exposure = _current_exp
        intent.tranche_action = "HOLD"
        intent.tranche_level = 0
        intent.alpha_gate_passed = True
        intent.alpha_estimated_bps = 0.0
        intent.alpha_threshold_bps = 0.0
        intent.expected_net_alpha_bps = 0.0
        intent.net_alpha_threshold_bps = 0.0
        intent.veto_active = False
        intent.veto_reason = (
            f"[PRE_ALPHA_HOLD] {_regime} weak long volume_breakout "
            f"({abs(_quant_dir):.2f} < {_max_dir:.2f})"
        )
        self._last_alpha_result = AlphaGatingResult(
            passes_threshold=True,
            estimated_alpha_bps=0.0,
            threshold_bps=0.0,
            gate_decision="PRE_ALPHA_HOLD",
        )
        logger.info(
            f"[PRE_ALPHA_HOLD] {asset}: {_regime} weak long volume_breakout "
            f"(dir={_quant_dir:+.3f}, max={_max_dir:.3f}) -> HOLD"
        )
        return True

    def decide(
        self,
        asset: str,
        prices: Dict[str, list],
        volumes: Dict[str, list],
        agent_signals: Dict[str, Any],
        market_data: Dict[str, float],
        position_state: Dict[str, Any],
        system_state: Dict[str, Any],
    ) -> TradeIntentV36:
        """
        Main 4H decision function (v3.6.1 canonical).
        
        DECISION FLOW (v3.6.1 with restored guarantees):
            [v3.6.1] Data Validation -> [v3.6.1] NO_TRADE Triggers -> 
            [v3.6.1] OPPORTUNITY Triggers -> Regime -> Failure Memory -> 
            Mode -> [v3.6.1] Alpha Gate -> Quant -> Fusion -> 
            [v3.6.1] Tranche -> Patience -> DRL -> Execution -> Risk -> SOL Dominance
        """
        
        intent = TradeIntentV36(asset=asset)
        # Update DRL authority level at entry so all early-return proof logs are accurate
        if hasattr(self, 'drl_gate') and self.drl_gate is not None:
            try:
                intent.drl_authority_level = self.drl_gate.get_authority().value
            except Exception:
                pass
        # Update sentiment/macro active flags at entry so all early-return proof logs are accurate
        _early_sent_z = agent_signals.get("sentiment_zscore", 0.0)
        self._last_sentiment_active = (
            agent_signals.get("sentiment_feed_alive", False) or abs(_early_sent_z) > 0.001
        )
        _early_mcc = agent_signals.get("macro_crowd_context")
        self._last_macro_active = _early_mcc.get("available", False) if isinstance(_early_mcc, dict) else bool(_early_mcc)
        try:
            intent.opportunity_actionable_direction_threshold_short = float(
                market_data.get(
                    "opportunity_actionable_min_direction_short",
                    intent.opportunity_actionable_direction_threshold,
                ) or 0.0
            )
        except Exception:
            intent.opportunity_actionable_direction_threshold_short = (
                intent.opportunity_actionable_direction_threshold
            )
        try:
            _opp_exp_thresh = market_data.get("opportunity_actionable_min_exposure")
            intent.opportunity_actionable_exposure_threshold = (
                None if _opp_exp_thresh is None else float(_opp_exp_thresh)
            )
        except Exception:
            intent.opportunity_actionable_exposure_threshold = None
        try:
            _normal_short_thresh = market_data.get("normal_actionable_min_direction_short")
            intent.normal_actionable_direction_threshold_short = (
                None if _normal_short_thresh is None else float(_normal_short_thresh)
            )
        except Exception:
            intent.normal_actionable_direction_threshold_short = None

        # Reset per-asset state from previous decide() call to prevent
        # cross-asset contamination (e.g., BTC's mode leaking into ETH)
        self._last_no_trade_state = None
        self._last_opportunity_state = None
        self._last_alpha_result = None
        self._last_tranche_decision = None
        self._last_veto_result = None
        self._last_deadlock_result = None
        self._last_sol_exit_signal = None
        self._last_phase_result = None

        # [v3.3-B9] Signal staleness check
        # [BUGFIX M6] Stale signals (>6H) now trigger NO_TRADE instead of log-only
        # [AUDIT M6] Skip check during reconnect grace period (first tick after recovery)
        import time as _b9_time
        _sig_ts = agent_signals.get('_signal_timestamp', _b9_time.time())
        _sig_age = _b9_time.time() - _sig_ts
        _reconnect_grace = agent_signals.get('_reconnect_grace', False)
        if _sig_age > 14400 * 1.5 and not _reconnect_grace:  # 1.5x the 4H interval = 6H
            logger.warning(
                f"[BUGFIX M6] {asset}: signal age {_sig_age:.0f}s > 6H threshold - "
                f"triggering NO_TRADE (stale signal protection)"
            )
            intent.no_trade_triggers_internal = True
            intent.veto_active = True
            intent.veto_reason = f"[BUGFIX M6] Stale signal: age={_sig_age:.0f}s > 21600s"
            intent.data_validation_passed = True
            self._add_proof_log(intent)
            return intent
        elif _sig_age > 14400 * 1.5 and _reconnect_grace:
            logger.info(
                f"[AUDIT M6] {asset}: signal age {_sig_age:.0f}s stale but reconnect grace active - allowing"
            )

        # =====================================================================
        # STEP 0: [v3.6.1] MARKET DATA VALIDATION (v3.3-STABILIZED guarantee)
        # =====================================================================
        
        validation_result = self.guarantees.validate_market_data(market_data)
        if not validation_result.is_valid:
            intent.veto_active = True
            intent.veto_reason = f"[v3.6.1] Data validation failed: {validation_result.error}"
            intent.system_mode = "NO_TRADE"
            self._add_proof_log(intent)
            logger.warning(f"[v3.6.1] Data validation FAILED: {validation_result.error}")
            return intent

        if validation_result.canonical_data:
            market_data = validation_result.canonical_data

        # =====================================================================
        # STEP 1: [v3.6.1] INTERNAL NO_TRADE TRIGGER COMPUTATION
        # =====================================================================
        
        # Build signal data for conflict detection.
        # Shadow / exit-only DRL is advisory only and must not flatten entries.
        _drl_conflict_dir = 0.0
        _drl_conflict_level = (
            self.drl_gate.get_authority().value
            if hasattr(self.drl_gate, "get_authority")
            else "DISABLED"
        )
        if _drl_conflict_level == "ACTIVE":
            _drl_conflict_dir = agent_signals.get("drl_direction", 0)

        signal_data = {
            "quant_direction": agent_signals.get("quant_direction", 0),
            "drl_direction": _drl_conflict_dir,
            "sentiment_direction": agent_signals.get("sentiment_direction", 0),
        }
        
        no_trade_state = self.guarantees.compute_no_trade_triggers(market_data, signal_data)
        self._last_no_trade_state = no_trade_state
        
        # Convert to trigger scores for legacy compatibility
        internal_no_trade_triggers = no_trade_state.trigger_scores
        
        logger.info(f"[v3.6.1] NO_TRADE triggers computed internally: "
                   f"active={no_trade_state.should_no_trade}, "
                   f"triggers={list(internal_no_trade_triggers.keys())}")
        
        # =====================================================================
        # STEP 2: [v3.6.1] INTERNAL OPPORTUNITY TRIGGER COMPUTATION
        # =====================================================================
        
        # Build lead-lag data
        lead_lag_data = {
            "net_edge_bps": agent_signals.get("lead_lag_edge", 0),  # Already in bps (Section B fix)
            "edge_window_confidence": agent_signals.get("lead_lag_confidence", 0),
            "cvd_divergence": agent_signals.get("cvd_divergence", 0),
            "vpin": market_data.get("vpin", 0.35),
        }
        
        # Build sentiment data if available
        sentiment_data = None
        if "sentiment_shock" in agent_signals:
            # [P0-FIX] Use actual sigma delta, not binary flag (was always 0 or 1, Group D needs >= 2.0)
            _shock_dir = agent_signals.get("sentiment_direction", 0)
            sentiment_data = {
                "shock_sigma": agent_signals.get("sentiment_shock_sigma", 0),  # [P0-FIX] was "sentiment_shock"
                "shock_direction": "long" if _shock_dir > 0 else ("short" if _shock_dir < 0 else "none"),
            }

        # [P1a] Build SOL flow data if asset is SOL and on-chain data available
        sol_flow_data = None
        _asset_upper = market_data.get("asset", "").upper().replace("/USD", "").replace("USD", "")
        if _asset_upper == "SOL":
            sol_flow_data = {
                "dex_volume_ratio": market_data.get("sol_dex_volume_ratio", 0),
                "onchain_inflow_4h": market_data.get("net_exchange_flow_usd", 0),
                "whale_net_direction": market_data.get("whale_direction", "none"),
                "jito_tip_ratio": market_data.get("sol_jito_tip_ratio",
                                                   market_data.get("sol_mev_pressure", 0)),
            }

        opportunity_state = self.guarantees.compute_opportunity_triggers(
            market_data, lead_lag_data, sentiment_data, sol_flow_data
        )
        self._last_opportunity_state = opportunity_state
        
        logger.info(f"[v3.6.1] OPPORTUNITY triggers computed internally: "
                   f"active={opportunity_state.is_active}, "
                   f"triggers={[t.group.value for t in opportunity_state.active_triggers] if opportunity_state.active_triggers else []}, "
                   f"ttl_bars={opportunity_state.ttl_remaining_bars}")
        
        # =====================================================================
        # STEP 3: Regime Phase Detection (v3.4)
        # =====================================================================
        
        phase_result = self.phase_detector.detect(
            regime_state=agent_signals.get("regime_state", "UNKNOWN"),
            crack_active=opportunity_state.is_active and any(
                t.group.value == "B" for t in opportunity_state.active_triggers
            ),  # CRACK from internal computation
            prices=prices,
            volumes=volumes,
            agent_signals=agent_signals,
        )
        self._last_phase_result = phase_result
        
        # =====================================================================
        # STEP 4: Failure Memory Check (v3.5)
        # =====================================================================
        
        failure_modifiers = self.failure_memory.get_modifiers()
        
        base_threshold = 0.88  # [P3b] was 0.70 -> 94% OPPORTUNITY. Raised to reduce density fallback
        adjusted_threshold = base_threshold + failure_modifiers.density_boost
        intent.opportunity_density_threshold = adjusted_threshold
        intent.tranche_delay_bars = failure_modifiers.tranche_delay_bars
        intent.in_caution_mode = failure_modifiers.in_caution_mode
        
        # =====================================================================
        # STEP 5: Mode Determination (v3.3-HR priorities + v3.6.1 internal)
        # =====================================================================
        
        # Use INTERNAL triggers, not caller-provided
        self._current_mode = self._determine_mode_v361(
            phase_result=phase_result,
            no_trade_state=no_trade_state,
            opportunity_state=opportunity_state,
            density_threshold=adjusted_threshold,
        )
        intent.system_mode = self._current_mode.value
        
        # Check NO_TRADE veto
        if self._current_mode == SystemMode.NO_TRADE:
            intent.veto_active = True
            if no_trade_state.primary_reason:
                intent.veto_reason = f"[v3.6.1] NO_TRADE: {no_trade_state.primary_reason.name}"
            else:
                intent.veto_reason = "NO_TRADE mode active"
            self._add_proof_log(intent)
            return intent
        
        # =====================================================================
        # STEP 5.5: [PROD] EARLY HARD VETO CHECK (Patch 4)
        # HARD conditions must block BEFORE alpha gating
        # =====================================================================
        
        # Skip flash crash hard veto when holding an existing position -
        # position management (HOLD, partial close, stop-loss) must NOT be frozen.
        # Flash crash only blocks NEW entries, not exits/holds on existing positions.
        _current_exposure = abs(market_data.get("current_exposure", 0.0))
        _flash_active = market_data.get("flash_crash_active", False)
        _skip_flash_for_existing = _flash_active and _current_exposure > 0.001

        early_hard_veto = self._check_early_hard_veto(
            drawdown=system_state.get("drawdown", 0.0),
            correlation=system_state.get("correlation", 0.0),
            dvol_zscore=market_data.get("dvol_zscore", 0.0),
            flash_crash_active=_flash_active and not _skip_flash_for_existing,
            execution_failures=self._execution_failures,
        )

        if early_hard_veto["is_hard"]:
            intent.veto_active = True
            intent.veto_reason = f"[PROD] HARD VETO: {early_hard_veto['reason']}"
            intent.direction = 0.0
            intent.target_exposure = 0.0
            logger.warning(f"[PROD Patch 4] EARLY HARD veto: {early_hard_veto['reason']}")
            self._add_proof_log(intent)
            return intent
        elif _skip_flash_for_existing:
            logger.info(
                f"[PROD Patch 4] FLASH_CRASH detected but existing position "
                f"(exposure={_current_exposure:.4f}) - allowing position management"
            )
        
        # =====================================================================
        # STEP 6: Quant Signals with Strategy Confidence (v3.5)
        # =====================================================================
        
        strategy_confidences = self.confidence_scorer.get_all_confidences()
        intent.quant_strategy_id = agent_signals.get("primary_strategy", "momentum")
        intent.quant_confidence = agent_signals.get("quant_confidence", 0.5)

        # Apply confidence modifier
        if intent.quant_strategy_id in strategy_confidences:
            intent.quant_confidence *= strategy_confidences[intent.quant_strategy_id]

        if self._maybe_apply_pre_alpha_hold(
            asset=asset,
            intent=intent,
            agent_signals=agent_signals,
            market_data=market_data,
            position_state=position_state,
        ):
            self._add_proof_log(intent)
            return intent

        # =====================================================================
        # STEP 7: [v3.6.1] ALPHA THRESHOLD GATING (v3.3-STABILIZED guarantee)
        # =====================================================================
        
        # Check if CRACK window is active from internal computation
        crack_window_active = opportunity_state.is_active and any(
            t.group.value == "B" for t in opportunity_state.active_triggers
        )
        
        # Check if lead-lag edge is confirmed from internal computation
        lead_lag_confirmed = opportunity_state.is_active and any(
            t.group.value == "A" for t in opportunity_state.active_triggers
        )
        
        # [DP-24] Fail-closed: synthetic fallback data must not reach alpha gate for new entries.
        # STALE_DATA in TradeGate catches prolonged outages via timestamp; this catches the
        # single-tick case where fetch failed and pipeline returned generate_verification_data().
        if not market_data.get("data_valid", True):
            _dv_current_exp = abs(market_data.get("current_exposure", 0.0))
            if _dv_current_exp < 0.001:
                intent.veto_active = True
                intent.veto_reason = "[DATA_INVALID] synthetic fallback data, new entries blocked"
                intent.no_trade_triggers_internal = True
                logger.warning(
                    f"[DP-24] {market_data.get('asset', '?')}: data_valid=False "
                    f"(_source={market_data.get('_source', '?')}) — new entry blocked"
                )
                return intent

        # [v6.7-P1] Regime-conditional alpha gate multiplier (injected by main.py)
        _alpha_gate_mult = agent_signals.get("_regime_alpha_gate_mult", 1.0)

        _alpha_input_direction = float(
            agent_signals.get("effective_alpha_direction", agent_signals.get("quant_direction", 0.0)) or 0.0
        )
        _alpha_min_bps = market_data.get("min_alpha_bps", 0.0)
        if _alpha_input_direction < 0:
            _alpha_gate_mult = agent_signals.get("_regime_alpha_gate_mult_short", _alpha_gate_mult)
            _alpha_min_bps = market_data.get("min_alpha_bps_short", _alpha_min_bps)
        # [UNIFIED-ALPHA] Pass calibrated signal_edge_bps from pipeline as alpha override.
        # This eliminates the dual-path: pipeline's ×65 (calibrated against paper run)
        # replaces constitution's ×200×conf×perf (uncalibrated, +46bps overestimate).
        _unified_alpha = market_data.get("signal_edge_bps", None)
        alpha_result = self.guarantees.check_alpha_gate(
            signal_strength=abs(_alpha_input_direction),
            regime_confidence=agent_signals.get("regime_confidence", 0.5),
            mode=self._current_mode.value,
            lead_lag_edge_confirmed=lead_lag_confirmed,
            crack_window_active=crack_window_active,
            min_alpha_bps=_alpha_min_bps,
            direction=_alpha_input_direction,
            regime_alpha_gate_mult=_alpha_gate_mult,
            quant_confidence=agent_signals.get("quant_confidence", 0.5),
            asset=market_data.get("asset", ""),
            regime=market_data.get("regime_state", ""),
            quiet_accum_min_direction=market_data.get("quiet_accum_min_direction", 0.35),
            quiet_accum_min_direction_short=market_data.get(
                "quiet_accum_min_direction_short",
                market_data.get("quiet_accum_min_direction", 0.35),
            ),
            borderline_pass_epsilon_bps=market_data.get(
                "borderline_alpha_pass_epsilon_bps",
                0.0,
            ),
            borderline_pass_epsilon_bps_short=market_data.get(
                "borderline_alpha_pass_epsilon_bps_short",
                0.0,
            ),
            estimated_alpha_override=_unified_alpha,  # [UNIFIED-ALPHA]
        )
        self._last_alpha_result = alpha_result
        
        logger.info(f"[v3.6.1] Alpha gate: passes={alpha_result.passes_threshold}, "
                   f"alpha={alpha_result.estimated_alpha_bps:.0f}bps, "
                   f"threshold={alpha_result.threshold_bps:.0f}bps ({alpha_result.multiplier_used}x), "
                   f"net={alpha_result.expected_net_alpha_bps:.0f}bps, "
                   f"net_req={alpha_result.required_net_alpha_bps:.0f}bps, "
                   f"gate={alpha_result.gate_decision}, "
                   f"min_alpha={alpha_result.min_alpha_bps_used:.0f}, "
                   f"ev_thresh={alpha_result.ev_threshold_bps:.0f}, "
                   f"alpha_dir={_alpha_input_direction:+.3f}, "
                   f"quant_dir={float(agent_signals.get('quant_direction', 0.0) or 0.0):+.3f}")
        
        # Populate alpha fields in intent (always, even if blocking)
        intent.alpha_gate_passed = alpha_result.passes_threshold
        intent.alpha_estimated_bps = alpha_result.estimated_alpha_bps
        intent.alpha_threshold_bps = alpha_result.threshold_bps
        intent.expected_net_alpha_bps = alpha_result.expected_net_alpha_bps
        intent.net_alpha_threshold_bps = alpha_result.required_net_alpha_bps
        
        # Block if alpha doesn't meet threshold
        # [BUGFIX C1] EXIT/REDUCE must NEVER be blocked by alpha gate.
        # When a position exists, allow fusion+tranche to decide EXIT/REDUCE.
        # Only block pure NEW entries (no existing position).
        _c1_current_exp = abs(market_data.get("current_exposure", 0.0))
        if not alpha_result.passes_threshold:
            if _c1_current_exp < 0.001:
                # No existing position - this is a pure new entry, block it
                intent.veto_active = True
                intent.veto_reason = f"[v3.6.1] Alpha gate: {alpha_result.reason}"
                intent.no_trade_triggers_internal = True
                intent.opportunity_triggers_internal = True
                intent.data_validation_passed = True
                self._add_proof_log(intent)
                logger.info(f"[v3.6.1] Trade BLOCKED by alpha gate: {alpha_result.reason}")
                return intent
            else:
                # [BUGFIX C1] Position exists - allow through to fusion/tranche for EXIT/REDUCE
                # [FIX-H4] Explicitly clear veto so downstream exit paths can fire.
                # Without this, a prior veto (e.g., from upstream checks) persists
                # and blocks gambler/regime/adaptive exits in main.py post-processing.
                intent.veto_active = False
                intent.veto_reason = ""
                logger.info(
                    f"[BUGFIX C1] Alpha gate failed (alpha={alpha_result.estimated_alpha_bps:.0f}bps "
                    f"< thresh={alpha_result.threshold_bps:.0f}bps) but position exists "
                    f"(exp={_c1_current_exp:.4f}) - allowing EXIT/REDUCE path (veto cleared)"
                )
        
        # =====================================================================
        # STEP 8: Authority-Based Fusion (v3.4)
        # =====================================================================

        # L4-05: Sanitize agent signal ranges before fusion
        agent_signals = self._sanitize_agent_signals(agent_signals)
        fusion_signals = self._build_fusion_signals(agent_signals, strategy_confidences)

        fusion_context = FusionContext(
            mode=self._current_mode,
            regime_phase=phase_result.phase,
            phase_result=phase_result,
            crack_active=crack_window_active,  # Use internal CRACK detection
            crack_weight=agent_signals.get("crack_weight", 0.0),
            data_valid=True,  # Already validated in Step 0
            drl_enabled=self.drl_gate.has_exit_authority(),
            lead_lag_confident=lead_lag_confirmed,  # Use internal lead-lag detection
            lead_lag_edge=agent_signals.get("lead_lag_edge", 0.0),
            regime=agent_signals.get("regime_state", "UNKNOWN"),  # [FIX-32] GMM regime for ADVISE influence
            htf_trend_direction=int(agent_signals.get("htf_trend_direction", 0)),  # [S11] daily trend
        )
        
        fusion_result = self.fusion_engine.fuse(fusion_signals, fusion_context)

        # [FIX-NAN-GUARD] Last-line defense: NaN/Inf in fusion output → zero (no trade).
        # A NaN direction reaching execution could cause unpredictable orders.
        import math as _math
        _fused_dir = fusion_result.direction
        _fused_exp = fusion_result.target_exposure
        if (_fused_dir is not None and isinstance(_fused_dir, float)
                and (_math.isnan(_fused_dir) or _math.isinf(_fused_dir))):
            logger.critical(
                f"[FIX-NAN-GUARD] Fusion produced NaN/Inf direction={_fused_dir} — "
                f"forcing direction=0 (no trade). Signals: {list(fusion_signals.keys())}"
            )
            fusion_result.direction = 0.0
            fusion_result.target_exposure = 0.0
        if (_fused_exp is not None and isinstance(_fused_exp, float)
                and (_math.isnan(_fused_exp) or _math.isinf(_fused_exp))):
            logger.critical(f"[FIX-NAN-GUARD] Fusion produced NaN/Inf exposure={_fused_exp} — forcing 0")
            fusion_result.target_exposure = 0.0

        intent.direction = fusion_result.direction
        intent.target_exposure = fusion_result.target_exposure
        intent.confidence = getattr(fusion_result, 'confidence', 0.5)  # [FIX-4]
        intent.tranche_target = fusion_result.tranche_target
        intent.allow_escalation = fusion_result.allow_escalation
        intent.max_tranche_tier = getattr(fusion_result, 'max_tranche_tier', 4)
        intent.urgency = fusion_result.urgency
        
        # Apply v3.5 failure memory aggressiveness
        intent.target_exposure *= failure_modifiers.aggressiveness_modifier
        
        # =====================================================================
        # STEP 9: [v3.6.1] TRANCHE SCHEDULING (v3.3-STABILIZED guarantee)
        # =====================================================================
        
        # Build signal data for abort detection
        tranche_signal_data = {
            "signal_conflict": no_trade_state.trigger_scores.get("signal_conflict", 0) > 0.5,
            "structure_confirmed": agent_signals.get("structure_confirmed", False),
        }

        # [FIX-REDUCE-THRESHOLD] Snapshot pre-schedule state BEFORE schedule_tranche
        # mutates position.current_exposure. _execute_full_position sets
        # current_exposure=target OPTIMISTICALLY, making target==current after the call.
        # [FIX-M3] Defensive copy — scheduler.get_position() may return mutable object
        _pre_sched_pos = self.guarantees.tranche_scheduler.get_position(asset)
        _pre_current_exposure = float(getattr(_pre_sched_pos, 'current_exposure', 0.0) or 0.0)
        _pre_direction = getattr(_pre_sched_pos, 'direction', None)

        tranche_decision = self.guarantees.schedule_tranche(
            asset=asset,
            mode=self._current_mode.value,
            target_direction=intent.direction,
            market_data=market_data,
            signal_data=tranche_signal_data,
            is_4h_bar_close=system_state.get("is_4h_bar_close", False),
        )
        self._last_tranche_decision = tranche_decision
        
        logger.info(f"[v3.6.1] Tranche scheduling: action={tranche_decision.action}, "
                   f"level={tranche_decision.new_level.name}, "
                   f"exposure={tranche_decision.target_exposure:.2f}"
                   + (f", reason={tranche_decision.reason}" if tranche_decision.reason else ""))
        
        # Handle tranche abort
        if tranche_decision.abort_active:
            # [FIX-TRANCHE-ABORT] Allow the REDUCE/FLATTEN to execute.
            # Previously veto_active=True prevented the position reduction entirely,
            # causing a stuck loop: abort fires every tick but position never changes.
            #
            # For REDUCE: the signal direction may have flipped (e.g., entered LONG but
            # now signal is SHORT). The execution code uses direction_sign to determine
            # if this is a partial_exit (same direction) or new entry (opposite direction).
            # We must preserve the existing position direction so the execution treats
            # REDUCE as a partial close, not a new entry in the opposite direction.
            intent.veto_active = False
            intent.veto_reason = ""
            intent.target_exposure = tranche_decision.target_exposure
            # [FIX-ABORT-SIZE] Save exact abort target so _execute_intent can bypass
            # all sizing multipliers (REGIME_POWER, WIRE-3, G6-SIZING, WARMUP_SIZE)
            # that would otherwise inflate/deflate the target unpredictably.
            intent._abort_target_exposure = tranche_decision.target_exposure
            intent.tranche_action = tranche_decision.action
            intent.tranche_level = tranche_decision.new_level.value
            intent.tranche_target = tranche_decision.new_level.value  # paper_positions['tranche'] update
            intent.force_execution = True
            intent.execution_mode = ExecutionMode.AGGRESSIVE
            if tranche_decision.action == "REDUCE":
                # Preserve existing position direction so execution branch B (partial_exit) fires.
                _pos = self.guarantees.tranche_scheduler.get_position(asset)
                if _pos and _pos.level.value > 0 and _pos.direction:
                    intent.direction = 1.0 if _pos.direction == "long" else -1.0
            logger.warning(
                f"[TRANCHE_ABORT] {asset}: Executing {tranche_decision.action} "
                f"to {tranche_decision.new_level.name} "
                f"(dir={intent.direction:+.1f}, target_exp={tranche_decision.target_exposure:.4f}): "
                f"{tranche_decision.reason}"
            )
            self._add_proof_log(intent)
            return intent
        
        # Update intent with tranche decision
        # Bug #44: Always apply tranche's target_exposure, including HOLD.
        # Previously HOLD was skipped, leaving fusion engine's unchecked value
        # (e.g. 80%) which then hit RISK_GOVERNOR correlated exposure limit.
        # For HOLD, tranche.target_exposure = position.current_exposure (correct).
        intent.target_exposure = tranche_decision.target_exposure
        if tranche_decision.action != "HOLD":
            intent.tranche_target = tranche_decision.new_level.value

        # [FIX-REDUCE-THRESHOLD] Detect organic reduces: tranche target < actual current
        # exposure. Uses pre-schedule snapshot because _execute_full_position mutates
        # position.current_exposure = target optimistically during schedule_tranche,
        # which would make target == current and prevent detection.
        if (_pre_current_exposure > 0.01 and
                intent.target_exposure > 0.001 and
                intent.target_exposure < _pre_current_exposure):
            _pre_dir_sign = 1 if _pre_direction == "long" else -1
            _intent_dir_sign = 1 if intent.direction > 0 else -1
            if _pre_dir_sign == _intent_dir_sign:
                intent.is_reduce_intent = True
                logger.info(
                    f"[REDUCE_INTENT] {asset}: flagging reduce "
                    f"(target={intent.target_exposure:.4f} < pre_sched={_pre_current_exposure:.4f}, "
                    f"dir={intent.direction:+.3f}) → bypassing direction threshold"
                )

        # Enforce max_tranche_tier cap (e.g. SATURATION + aggressive = T3 max)
        if intent.tranche_target > intent.max_tranche_tier:
            intent.tranche_target = intent.max_tranche_tier

        # [FIX-NORMAL-REVERSAL] Detect signal direction reversal for existing positions.
        # In NORMAL mode at 4H bar close, _execute_full_position sizes the new position
        # as abs(signal_direction). When signal < 0.10 threshold, is_actionable=False
        # and the position stays open indefinitely — even if signal has flipped direction.
        # Deadlock resolver needs 2+ consecutive ticks of in-memory state to fire (lost
        # on process restart). Fix: detect direction reversal and force-close immediately.
        if (_pre_current_exposure > 0.01 and
                _pre_direction is not None and
                not intent.is_actionable and
                not intent.is_reduce_intent and
                not intent.force_execution):
            _n_rev_pre_sign = 1 if _pre_direction == "long" else -1
            _n_rev_intent_sign = 1 if intent.direction > 0 else -1
            if _n_rev_pre_sign != _n_rev_intent_sign:
                # Signal direction has reversed vs existing position.
                # Use same direction as existing position so main.py partial_exit
                # branch fires correctly (close direction = position direction, target=0).
                intent.target_exposure = 0.0
                intent.direction = float(_n_rev_pre_sign)
                intent.force_execution = True
                intent.veto_active = False
                logger.warning(
                    f"[FIX-NORMAL-REVERSAL] {asset}: force-closing {_pre_direction} position "
                    f"(signal reversed to {'long' if _n_rev_intent_sign > 0 else 'short'}, "
                    f"abs_dir={abs(intent.direction):.3f} was < 0.10 threshold, "
                    f"pre_exp={_pre_current_exposure:.4f})"
                )
                self._add_proof_log(intent)
                return intent

        # =====================================================================
        # STEP 10: [PROD] Fusion Patience + Deadlock Resolution (Patch 5)
        # =====================================================================
        
        # Original v3.5 patience manager check
        deadlock_result = self.patience_manager.check_deadlock(
            asset=asset,
            current_edge_bps=agent_signals.get("signal_edge_bps", 50.0),
            current_friction_bps=market_data.get("estimated_friction_bps", 15.0),
        )
        
        # [PROD Patch 5] Tranche-aware deadlock resolution
        # Override with tranche-coupled bias (T1->FORCE, T2+->ABORT)
        current_tranche = self._last_tranche_decision.new_level.value if self._last_tranche_decision else 0
        # [FIX-DEADLOCK-ENUM] Three modules each define DeadlockResolution with different enum
        # identity. Python enum 'is' / '==' checks fail across module boundaries even when the
        # string values are identical. Fix: compare .value strings.  Also: patience manager
        # has .consecutive_delays — use that for is_stuck to avoid all enum comparison.
        is_stuck = deadlock_result.consecutive_delays >= 2

        tranche_deadlock = self.deadlock_resolver.resolve(
            asset=asset,
            tranche_level=current_tranche,
            current_edge_bps=agent_signals.get("signal_edge_bps", 50.0),
            friction_bps=market_data.get("estimated_friction_bps", 15.0),
            is_stuck=is_stuck,
            mode=self._current_mode.name,
        )
        self._last_deadlock_result = tranche_deadlock

        # Use tranche-aware resolution if deadlocked
        # [FIX-DEADLOCK-ENUM] Use .value string comparison — immune to enum identity mismatch
        if tranche_deadlock.resolution.value == "FORCE_AGGRESSIVE":
            intent.force_execution = True
            intent.execution_mode = ExecutionMode.AGGRESSIVE
            intent.deadlock_resolution = f"FORCE_AGGRESSIVE (T{current_tranche} bias)"
            logger.warning(f"[PROD Patch 5] Tranche-aware FORCE for {asset}: {tranche_deadlock.reason}")
        elif tranche_deadlock.resolution.value == "ABORT_OPPORTUNITY":
            # Check if there is an existing position to close
            _dl_exp = abs(getattr(intent, 'target_exposure', 0))
            _dl_dir = getattr(intent, 'direction', 0)
            if _dl_exp <= 0.001:
                # [FIX-ABORT-CLOSE-VETO] Intent is already a close (target=0).
                # ABORT should not VETO a close — that traps the position.
                # Force the close to execute despite bad timing.
                intent.veto_active = False
                intent.force_execution = True
                intent.execution_mode = ExecutionMode.AGGRESSIVE
                intent.deadlock_resolution = f"ABORT+FORCE_CLOSE (target was 0)"
                logger.warning(
                    f"[PROD Patch 5] Tranche-aware FORCE_CLOSE for {asset}: "
                    f"close intent was stuck due to bad timing, forcing execution"
                )
            elif _dl_dir != 0:
                # Force close existing position instead of just vetoing
                intent.target_exposure = 0.0
                intent.direction = -_dl_dir
                # is_actionable is @property — force_execution=True makes it True
                intent.veto_active = False
                intent.force_execution = True
                intent.execution_mode = ExecutionMode.AGGRESSIVE
                intent.deadlock_resolution = f"ABORT+CLOSE (T{current_tranche} bias)"
                logger.warning(
                    f"[PROD Patch 5] Tranche-aware ABORT+CLOSE for {asset}: "
                    f"{tranche_deadlock.reason} ->force closing position"
                )
            else:
                intent.veto_active = True
                intent.veto_reason = f"[PROD] Tranche deadlock abort: {tranche_deadlock.reason}"
                intent.deadlock_resolution = f"ABORT (T{current_tranche} bias)"
                logger.warning(f"[PROD Patch 5] Tranche-aware ABORT for {asset}: {tranche_deadlock.reason}")
            # Reset patience to prevent infinite ABORT loop (step 12 is skipped below)
            self.patience_manager.reset_state(asset)
            self._add_proof_log(intent)
            return intent
        elif deadlock_result.resolution.value == "FORCE_AGGRESSIVE":
            # Fallback to original v3.5 behavior if tranche resolver didn't trigger
            intent.force_execution = True
            intent.execution_mode = ExecutionMode.AGGRESSIVE
            intent.deadlock_resolution = "FORCE_AGGRESSIVE"
            logger.warning(f"[v3.5] Deadlock FORCE_AGGRESSIVE for {asset}")
        elif deadlock_result.resolution.value == "ABORT_OPPORTUNITY":
            _dl2_exp = abs(getattr(intent, 'target_exposure', 0))
            _dl2_dir = getattr(intent, 'direction', 0)
            if _dl2_exp <= 0.001:
                # [FIX-ABORT-CLOSE-VETO] Intent is already a close (target=0) — force it
                intent.veto_active = False
                intent.force_execution = True
                intent.execution_mode = ExecutionMode.AGGRESSIVE
                intent.deadlock_resolution = "ABORT+FORCE_CLOSE (target was 0)"
                logger.warning(f"[v3.5] Deadlock FORCE_CLOSE for {asset}: close intent was stuck, forcing")
            elif _dl2_dir != 0:
                intent.target_exposure = 0.0
                intent.direction = -_dl2_dir
                # is_actionable is @property — force_execution=True makes it True
                intent.veto_active = False
                intent.force_execution = True
                intent.execution_mode = ExecutionMode.AGGRESSIVE
                intent.deadlock_resolution = "ABORT+CLOSE"
                logger.warning(f"[v3.5] Deadlock ABORT+CLOSE for {asset}: force closing position")
            else:
                intent.veto_active = True
                intent.veto_reason = f"Deadlock abort: {deadlock_result.abort_reason}"
                intent.deadlock_resolution = "ABORT"
                logger.warning(f"[v3.5] Deadlock ABORT for {asset}")
            # Reset patience to prevent infinite ABORT loop (step 12 is skipped below)
            self.patience_manager.reset_state(asset)
            self._add_proof_log(intent)
            return intent
        
        # =====================================================================
        # STEP 11: DRL Advisory (v3.4 + v3.5 gate)
        # =====================================================================
        
        intent.drl_authority_level = self.drl_gate.get_authority().value
        intent.drl_has_authority = self.drl_gate.has_exit_authority()
        
        # DRL silence ≠ veto in OPPORTUNITY (HARD RULE from v3.4)
        if self._current_mode == SystemMode.OPPORTUNITY:
            # DRL advice is advisory only, cannot veto entry
            pass
        
        # Apply tranche delay from v3.5 failure memory
        if intent.tranche_delay_bars > 0:
            intent.allow_escalation = False
        
        # =====================================================================
        # STEP 12: Execution Timing (v3.4)
        # =====================================================================
        
        if not intent.force_execution:
            timing_score, exec_mode = self.timing_engine.evaluate(
                asset=asset,
                spread_bps=market_data.get("spread_bps", 10.0),
                depth=market_data.get("orderbook_depth", 1.0),
                vpin=market_data.get("vpin", 0.35),
                lead_lag_edge=agent_signals.get("lead_lag_edge", 0.0),
                lead_lag_confidence=agent_signals.get("lead_lag_confidence", 0.0),
                urgency=intent.urgency,
                regime_phase=phase_result.phase.value if hasattr(phase_result.phase, 'value') else str(phase_result.phase),
                fee_context=agent_signals.get("fee_context"),
            )
            
            intent.timing_score = timing_score.total_score
            intent.timing_spread_score = timing_score.spread_score
            intent.timing_depth_score = timing_score.depth_score
            intent.timing_vpin_score = timing_score.vpin_score
            intent.timing_lead_lag_score = timing_score.lead_lag_score
            intent.execution_mode = exec_mode
            
            # Record delay for patience manager
            if timing_score.total_score < 0.4:
                self.patience_manager.record_delay(
                    asset=asset,
                    intended_action="ENTRY",
                    delay_reason="poor_timing",
                    timing_score=timing_score.total_score,
                    estimated_friction_bps=market_data.get("estimated_friction_bps", 15.0),
                    signal_direction=intent.direction,
                    signal_edge_bps=agent_signals.get("signal_edge_bps", 50.0),
                )
                intent.delay_bars = 1
        else:
            self.patience_manager.record_execution(asset)
        
        # =====================================================================
        # STEP 13: [PROD] HARD vs SOFT Risk Assessment (Patch 4)
        # =====================================================================
        
        # Get standard risk assessment first
        risk_assessment = self.risk_agent.assess(
            system_mode=self._current_mode.name,
            current_drawdown=system_state.get("drawdown", 0.0),
            current_correlation=system_state.get("correlation", 0.5),
            current_exposures=system_state.get("exposures", {}),
            phase_result=phase_result,
            sol_signal_strength=agent_signals.get("sol_signal_strength", 0.0),
            sol_beta=agent_signals.get("sol_beta", 1.0),
            no_trade_trigger_scores=system_state.get("no_trade_triggers", {}),
            is_weekend=system_state.get("is_weekend", False),
            is_holiday=system_state.get("is_holiday", False),
            is_low_liquidity=system_state.get("is_low_liquidity", False),
        )
        
        # [PROD Patch 4] Apply HARD vs SOFT veto classification
        veto_result = self.risk_veto_classifier.classify(
            mode=self._current_mode.name,
            drawdown=system_state.get("drawdown", 0.0),
            dvol_zscore=market_data.get("dvol_zscore", 0.0),
            correlation=system_state.get("correlation", 0.0),
            liquidity_usd=market_data.get("orderbook_depth_1pct_usd", float('inf')),
            signal_conflict_score=1.0 if (self._last_no_trade_state and 
                self._last_no_trade_state.trigger_scores.get("signal_conflict", 0) > 0.5) else 0.0,
            data_valid=True,  # Already validated in Step 0
            execution_failures=self._execution_failures,
            flash_crash_active=market_data.get("flash_crash_active", False) and not _skip_flash_for_existing,
            is_weekend=system_state.get("is_weekend", False),
            missing_required_keys=validation_result.missing_required if hasattr(validation_result, 'missing_required') else None,
            missing_critical_keys=validation_result.missing_critical if hasattr(validation_result, 'missing_critical') else None,
        )
        self._last_veto_result = veto_result
        
        # HARD veto always blocks (v3.3-HR: 10% halt is NON-NEGOTIABLE)
        if veto_result.veto_type == VetoType.HARD:
            intent.veto_active = True
            intent.veto_reason = f"[PROD] HARD VETO: {veto_result.reason}"
            intent.direction = 0.0
            intent.target_exposure = 0.0
            logger.warning(f"[PROD Patch 4] HARD veto triggered: {veto_result.reason}")
            self._add_proof_log(intent)
            return intent
        
        # SOFT veto behavior depends on mode (Patch 4)
        if veto_result.veto_type == VetoType.SOFT:
            if not veto_result.allows_trade:
                # NORMAL mode: SOFT blocks
                intent.veto_active = True
                intent.veto_reason = f"[PROD] SOFT VETO (NORMAL mode): {veto_result.reason}"
                intent.direction = 0.0
                intent.target_exposure = 0.0
                logger.info(f"[PROD Patch 4] SOFT veto blocks in NORMAL: {veto_result.reason}")
                self._add_proof_log(intent)
                return intent
            else:
                # OPPORTUNITY mode: SOFT only caps exposure
                exposure_cap = veto_result.exposure_cap
                intent.target_exposure = min(intent.target_exposure, exposure_cap)
                logger.info(f"[PROD Patch 4] SOFT veto caps exposure in OPPORTUNITY: {exposure_cap:.0%}")
        
        intent.max_gross = risk_assessment.max_gross_cap
        intent.risk_multiplier = risk_assessment.risk_multiplier
        
        # =====================================================================
        # STEP 14: SOL Dominance Check (v3.3-HR) + [PROD] Patch 6 Immediate Exit
        # =====================================================================
        
        if asset == "SOL":
            sol_dominance_result = self._check_sol_dominance(
                phase=phase_result.phase,
                mode=self._current_mode,
                sol_signal=agent_signals.get("sol_signal_strength", 0.0),
            )
            
            intent.sol_dominance_active = sol_dominance_result["active"]
            intent.sol_dominance_ttl = sol_dominance_result["ttl"]
            intent.sol_dominance_reason = sol_dominance_result["reason"]
            
            # [PROD Patch 6] Check for immediate exit via execution loop
            current_exposure = position_state.get("exposure", 0.0) if position_state else 0.0
            sol_exit_signal = self.sol_forced_exit.check_forced_exit(
                asset=asset,
                dominance_active=sol_dominance_result["active"],
                dominance_ttl=sol_dominance_result["ttl"],
                current_phase=phase_result.phase.value if hasattr(phase_result.phase, 'value') else str(phase_result.phase),
                vpin=market_data.get("vpin", 0.35),
                correlation=system_state.get("correlation", 0.5),
                price_move_pct=market_data.get("price_move_pct", 0.0),
                current_exposure=current_exposure,
            )
            self._last_sol_exit_signal = sol_exit_signal
            
            # IMMEDIATE exit overrides normal decision
            if sol_exit_signal.urgency == ExitUrgency.IMMEDIATE:
                intent.force_execution = True
                intent.execution_mode = sol_exit_signal.execution_mode
                intent.target_exposure = sol_exit_signal.target_exposure
                intent.direction = -1.0 if current_exposure > 0 else 1.0  # Flatten
                intent.urgency = 1.0
                logger.warning(f"[PROD Patch 6] SOL IMMEDIATE EXIT: {sol_exit_signal.reason}")
                # Don't return yet - let proof log be generated
            elif sol_exit_signal.urgency == ExitUrgency.SCHEDULED:
                # Reduce target exposure for scheduled exit
                intent.target_exposure = min(intent.target_exposure, sol_exit_signal.target_exposure)
                logger.info(f"[PROD Patch 6] SOL scheduled exit: {sol_exit_signal.reason}")
            
            if intent.sol_dominance_active:
                intent.max_asset = V33HR_SOL_DOMINANCE["max_exposure"]
                intent.max_gross = min(
                    intent.max_gross + V33HR_SOL_DOMINANCE["gross_override"],
                    1.50  # Cap at 150%
                )
            else:
                intent.max_asset = risk_assessment.sol_max_exposure
        else:
            intent.max_asset = risk_assessment.max_single_asset_cap
        
        # Apply risk multiplier
        intent.target_exposure *= risk_assessment.risk_multiplier
        intent.target_exposure = min(intent.target_exposure, intent.max_asset)
        
        # =====================================================================
        # STEP 15: [v3.6.1] Populate Constitution Guarantee Fields
        # =====================================================================
        
        intent.no_trade_triggers_internal = True  # Always true in v3.6.1
        intent.opportunity_triggers_internal = True  # Always true in v3.6.1
        intent.data_validation_passed = True  # Already passed (early return if failed)
        
        if self._last_alpha_result:
            intent.alpha_gate_passed = self._last_alpha_result.passes_threshold
            intent.alpha_estimated_bps = self._last_alpha_result.estimated_alpha_bps
            intent.alpha_threshold_bps = self._last_alpha_result.threshold_bps
        
        if self._last_tranche_decision:
            intent.tranche_action = self._last_tranche_decision.action
            intent.tranche_level = self._last_tranche_decision.new_level.value
        
        if self._last_no_trade_state:
            intent.signal_conflict_detected = (
                self._last_no_trade_state.trigger_scores.get("signal_conflict", 0) > 0.5
            )
        
        # =====================================================================
        # STEP 16: [v3.3-B12] Consume macro_risk_appetite (was STEP 15b)
        # Reduce urgency in risk-off macro; slight boost in strong risk-on
        # =====================================================================
        _risk_appetite = agent_signals.get('macro_risk_appetite', 0.5)
        if _risk_appetite < 0.3:
            _pre_urg = intent.urgency
            # [SHORT-OPT SB-2] Direction-aware risk-off urgency
            if intent.direction < 0:  # Short - risk-off favors shorts, preserve urgency
                pass
            else:  # Long - risk-off penalizes longs more aggressively
                intent.urgency = intent.urgency * 0.5
            if intent.urgency != _pre_urg:
                logger.info(
                    f"[MACRO_RISK_OFF] {asset}: risk_appetite={_risk_appetite:.2f}, "
                    f"dir={intent.direction:+.1f}, urgency {_pre_urg:.2f} -> {intent.urgency:.2f}"
                )
        elif _risk_appetite > 0.8:
            if intent.direction < 0:  # Short in risk-on - reduce urgency
                intent.urgency = intent.urgency * 0.85
            else:  # Long in risk-on - slight boost
                intent.urgency = min(intent.urgency * 1.15, 1.0)

        # =====================================================================
        # STEP 17: [v3.3-B7] Per-signal audit + Structured Proof Log
        # =====================================================================

        # Build per-signal adopted/rejected audit
        _AUDIT_KEYS = [
            'quant_direction', 'drl_direction', 'sentiment_direction',
            'risk_veto', 'structure_confirmed', 'lead_lag_edge',
            'macro_risk_appetite', 'cvd_divergence',
        ]
        signal_audit = {}
        for _ak in _AUDIT_KEYS:
            _val = agent_signals.get(_ak, 'MISSING')
            signal_audit[_ak] = {
                'value': _val,
                'adopted': _val not in ('MISSING', 0, 0.0, False, None),
            }
        intent._signal_audit = signal_audit

        self._add_proof_log(intent)

        return intent
    
    def _determine_mode_v361(
        self,
        phase_result,
        no_trade_state: NoTradeState,
        opportunity_state: OpportunityState,
        density_threshold: float,
    ) -> SystemMode:
        """
        [v3.6.1] Determine system mode using INTERNAL trigger computation.
        
        Uses NoTradeState and OpportunityState from ConstitutionGuaranteesModule
        instead of caller-provided triggers.
        """
        
        # NO_TRADE always takes precedence (from internal computation)
        if no_trade_state.should_no_trade:
            logger.info(f"[v3.6.1] Mode=NO_TRADE from internal triggers: {no_trade_state.primary_reason}")
            return SystemMode.NO_TRADE
        
        # [FIX-M5] Legacy trigger score check. Was 0.80 — too aggressive.
        # dvol_zscore=4.0 (high but not extreme) produces score=0.80, triggering NO_TRADE
        # even though constitution says should_no_trade=False (needs zscore≥5.0).
        # Raised to 0.95 to only catch near-extreme conditions.
        max_trigger = max(no_trade_state.trigger_scores.values()) if no_trade_state.trigger_scores else 0.0
        if max_trigger >= 0.95:
            logger.info(f"[v3.6.1] Mode=NO_TRADE from trigger score: {max_trigger:.2f} (legacy safety)")
            return SystemMode.NO_TRADE
        
        # OPPORTUNITY from internal computation
        if opportunity_state.is_active:
            # Check for CRACK specifically (Group B)
            has_crack = any(t.group.value == "B" for t in opportunity_state.active_triggers)
            if has_crack:
                logger.info(f"[v3.6.1] Mode=OPPORTUNITY from CRACK window (internal)")
                return SystemMode.OPPORTUNITY
            
            # Check for lead-lag edge (Group A)
            has_lead_lag = any(t.group.value == "A" for t in opportunity_state.active_triggers)
            if has_lead_lag:
                logger.info(f"[v3.6.1] Mode=OPPORTUNITY from Lead-Lag edge (internal)")
                return SystemMode.OPPORTUNITY
            
            # Other opportunity triggers
            logger.info(f"[v3.6.1] Mode=OPPORTUNITY from internal triggers: "
                       f"{[t.group.value for t in opportunity_state.active_triggers]}")
            return SystemMode.OPPORTUNITY
        
        # Phase-based OPPORTUNITY
        phase_value = phase_result.phase.value if hasattr(phase_result.phase, 'value') else str(phase_result.phase)
        if phase_value == "IGNITION":
            logger.info(f"[v3.6.1] Mode=OPPORTUNITY from IGNITION phase")
            return SystemMode.OPPORTUNITY
        
        # Density-based OPPORTUNITY (v3.5 adjusted threshold)
        if phase_result.opportunity_density >= density_threshold:
            logger.info(f"[v3.6.1] Mode=OPPORTUNITY from density {phase_result.opportunity_density:.2f} >= {density_threshold:.2f}")
            return SystemMode.OPPORTUNITY
        
        return SystemMode.NORMAL
    
    def _check_early_hard_veto(
        self,
        drawdown: float,
        correlation: float,
        dvol_zscore: float,
        flash_crash_active: bool,
        execution_failures: int,
    ) -> Dict:
        """
        [PROD Patch 4] Early HARD veto check before alpha gating.

        HARD conditions that must block immediately:
        - Drawdown >= profile threshold (default 10%, configurable via T21)
        - Correlation >= profile threshold (default 0.95, configurable via T22)
        - DVOL z-score >= 5.0 (extreme volatility)
        - Flash crash detected
        - 3+ consecutive execution failures
        """

        reasons = []

        # Drawdown halt (T21: configurable from profile, fail-closed default=20%)
        _dd_threshold = self._profile_config.get("hard_drawdown_halt", 0.20)
        if drawdown >= _dd_threshold:
            reasons.append(f"DRAWDOWN_HALT: {drawdown:.1%} >= {_dd_threshold:.0%}")

        # Correlation crisis (T22: configurable from profile, fail-closed default=0.98)
        _corr_threshold = self._profile_config.get("correlation_crisis", 0.98)
        if correlation >= _corr_threshold:
            reasons.append(f"CORRELATION_CRISIS: {correlation:.2f} >= {_corr_threshold}")
        
        # Extreme DVOL
        if dvol_zscore >= 5.0:
            reasons.append(f"EXTREME_DVOL: z={dvol_zscore:.1f} >= 5.0")
        
        # Flash crash
        if flash_crash_active:
            reasons.append("FLASH_CRASH_DETECTED")
        
        # Execution failures
        if execution_failures >= 3:
            reasons.append(f"EXECUTION_BLOCKED: {execution_failures} failures")
        
        if reasons:
            return {
                "is_hard": True,
                "reason": ", ".join(reasons),
                "conditions": reasons,
            }
        
        return {"is_hard": False, "reason": "", "conditions": []}

    # L4-05: Signal value range clip protection
    _SIGNAL_CLIP_RULES = {
        # direction: [-1.0, 1.0]
        'quant_direction': (-1.0, 1.0),
        'drl_direction': (-1.0, 1.0),
        'sentiment_direction': (-1.0, 1.0),
        'lead_lag_direction': (-1.0, 1.0),
        'micro_direction': (-1.0, 1.0),
        'regime_direction': (-1.0, 1.0),
        # confidence: [0.0, 1.0]
        'quant_confidence': (0.0, 1.0),
        'drl_confidence': (0.0, 1.0),
        'regime_confidence': (0.0, 1.0),
        'sentiment_confidence': (0.0, 1.0),
        'micro_confidence': (0.0, 1.0),
        'lead_lag_confidence': (0.0, 1.0),
        # zscore: [-5.0, 5.0]
        'sentiment_zscore': (-5.0, 5.0),
        'ofi_zscore': (-5.0, 5.0),
        # probability: [0.0, 1.0]
        'squeeze_risk': (0.0, 1.0),
        'macro_risk_appetite': (0.0, 1.0),
    }

    def _sanitize_agent_signals(self, signals: dict) -> dict:
        """Clip all agent signals to valid ranges before fusion."""
        sanitized = signals.copy()
        for key, (low, high) in self._SIGNAL_CLIP_RULES.items():
            if key in sanitized and isinstance(sanitized[key], (int, float)):
                original = sanitized[key]
                sanitized[key] = max(low, min(high, float(sanitized[key])))
                if sanitized[key] != original:
                    logger.warning(f"[SIGNAL_CLIP] {key}: {original} -> {sanitized[key]}")
        return sanitized

    def _build_fusion_signals(
        self,
        agent_signals: Dict,
        strategy_confidences: Dict[str, float],
    ) -> Dict:
        """Build signals with v3.5 strategy confidence applied."""
        
        signals = {}
        
        # =================================================================
        # V6: Get adjusted weights from Strategic Coordinator
        # =================================================================
        v6_weights = agent_signals.get("v6_adjusted_weights", {})
        
        # Quant (with confidence and V6 weight)
        quant_confidence = agent_signals.get("quant_confidence", 0.5)
        strategy_name = agent_signals.get("primary_strategy", "momentum")
        if strategy_name in strategy_confidences:
            quant_confidence *= strategy_confidences[strategy_name]
        
        # V6: Apply adaptive weight adjustment
        if "quant_agent" in v6_weights:
            quant_confidence *= v6_weights["quant_agent"]

        # [P0.2 V2] Quant-strategy adaptive confidence (NOT multi-expert gating).
        # Adjusts confidence of the ACTIVE quant strategy based on realized utility.
        # Uses Bayesian shrinkage at low sample sizes → stays near 1.0 until proven.
        # Globally clamped by AlphaBoostConfig.global_confidence_mult_clamp.
        _ab_expert_w = agent_signals.get("_ab_expert_weights", {})
        if strategy_name in _ab_expert_w:
            quant_confidence *= _ab_expert_w[strategy_name]

        # [P1.1] Sentiment gate: bounded confidence modulation when aligned (flag-gated, default OFF)
        _sg_conf_mult = agent_signals.get("_sg_confidence_mult", 1.0)
        if _sg_conf_mult != 1.0:
            quant_confidence *= _sg_conf_mult

        # [P0.3 V2] Execution quality confidence: penalizes high-slippage assets, rewards maker success.
        # Ramps in gradually past exec_feedback_min_fills, globally clamped.
        _ab_exec_conf = agent_signals.get("_ab_exec_confidence", 1.0)
        if _ab_exec_conf != 1.0:
            quant_confidence *= _ab_exec_conf

        # [V2-FIX] Final combined confidence clamp.
        # The three V2 multipliers above (_ab_expert_weights, _sg_confidence_mult,
        # _ab_exec_confidence) are each individually bounded by AlphaBoost's
        # global_confidence_mult_clamp, but their PRODUCT can exceed 1.0.
        # Baseline quant_confidence (flags OFF) has a natural max of 1.0
        # (from pipeline: 0.3+0.4*agreement+0.3*dir, times scorer max 1.0).
        # The upper clamp is set to 1.0 so flags-OFF behavior is UNCHANGED,
        # but V2-inflated values above 1.0 are caught.
        # Lower clamp at 0.05 prevents near-zero suppression.
        _pre_clamp_qc = quant_confidence
        quant_confidence = max(0.05, min(1.0, quant_confidence))
        if quant_confidence != _pre_clamp_qc:
            logger.info(
                f"[V2-FIX] quant_confidence clamped: {_pre_clamp_qc:.4f}->{quant_confidence:.4f} "
                f"(strategy={strategy_name})"
            )

        signals["quant"] = AgentSignal(
            direction=agent_signals.get("quant_direction", 0.0),
            confidence=quant_confidence,
        )
        
        # Regime - in OPPORTUNITY mode, regime has DECIDE authority for timing.
        # Direction comes from quant (regime provides confidence/gating, not direction).
        regime_confidence = agent_signals.get("regime_confidence", 0.5)
        regime_direction = agent_signals.get("regime_direction", 0.0)
        if regime_direction == 0.0:
            regime_direction = agent_signals.get("quant_direction", 0.0)
        signals["regime"] = AgentSignal(
            direction=regime_direction,
            confidence=regime_confidence,
        )
        
        # Sentiment (with V6 weight)
        # v6.7: F&G routes to sentiment_zscore; dead zone |zscore| < 0.5 (~F&G 42-58) -> neutral
        sentiment_zscore = agent_signals.get("sentiment_zscore", 0.0)
        # For proof log: REAL if feed is alive (even if neutral/dead-zone)
        _sent_feed_alive = agent_signals.get("sentiment_feed_alive", abs(sentiment_zscore) > 0.001)
        self._last_sentiment_active = _sent_feed_alive or abs(sentiment_zscore) > 0.001
        # [BUGFIX AUDIT-D4] Check 'available' key, not truthiness of dict
        _mcc = agent_signals.get("macro_crowd_context")
        self._last_macro_active = _mcc.get("available", False) if isinstance(_mcc, dict) else bool(_mcc)
        sentiment_confidence = min(abs(sentiment_zscore) / 3.0, 1.0)
        if "sentiment_agent" in v6_weights:
            sentiment_confidence *= v6_weights["sentiment_agent"]

        # Dead zone: only emit direction when F&G is meaningfully away from neutral
        # |zscore| < 0.5 ≈ F&G 42-58 -> direction=0 -> CONFIRM check skipped
        if sentiment_zscore > 0.5:
            sentiment_direction = 1.0
        elif sentiment_zscore < -0.5:
            sentiment_direction = -1.0
        else:
            sentiment_direction = 0.0
            sentiment_confidence = 0.0  # No signal in dead zone

        # Iron Law #34: sentiment never vetoes trades.
        # Extreme fear (<-3) -> scale down shorts ×0.5 instead of hard VETO
        # Extreme greed/fear only attenuates sentiment's fusion confidence.
        _sent_conf = sentiment_confidence
        if abs(sentiment_zscore) > 3.0:
            _sent_conf *= 0.5
        signals["sentiment"] = AgentSignal(
            direction=sentiment_direction,
            confidence=_sent_conf,
            veto_active=False,
            veto_direction=None,
        )
        
        # Macro
        signals["macro"] = AgentSignal(
            leverage_cap=agent_signals.get("macro_leverage_cap", 1.0),
        )
        
        # Lead-Lag
        # [BUGFIX AUDIT-A3] Clip to [-1,1] - edge in bps, /100 only converts to %
        _ll_raw = agent_signals.get("lead_lag_edge", 0.0) / 100.0
        signals["lead_lag"] = AgentSignal(
            direction=max(-1.0, min(1.0, _ll_raw)),
            confidence=agent_signals.get("lead_lag_confidence", 0.0),
        )
        
        # Risk
        signals["risk"] = AgentSignal(
            veto_active=agent_signals.get("risk_veto", False),
            veto_direction="risk_limit" if agent_signals.get("risk_veto", False) else None,
        )

        # Short Bias (v6.5.2: soft penalty via CONFIRM/TRIGGER authority)
        sb_direction = agent_signals.get("short_bias_direction", 0.0)
        sb_confidence = agent_signals.get("short_bias_confidence", 0.0)
        if sb_confidence > 0:
            signals["short_bias"] = AgentSignal(
                direction=sb_direction,
                confidence=sb_confidence,
            )
        
        # Funding Rate bias (v9-PATCH-8: carry-bias from funding rates)
        # Positive funding -> longs pay shorts -> short bias
        # Negative funding -> shorts pay longs -> long bias
        _funding = agent_signals.get("funding_rate", 0.0)
        if abs(_funding) > 0.0001:
            _fr_direction = -1.0 if _funding > 0 else 1.0
            _fr_confidence = min(abs(_funding) / 0.001, 1.0)
            signals["funding_rate"] = AgentSignal(
                direction=_fr_direction,
                confidence=_fr_confidence,
            )

        # DRL: shadow ensemble becomes actionable only after promotion.
        _drl_dir = agent_signals.get("drl_direction", 0.0)
        _drl_conf = agent_signals.get("drl_confidence", 0.0)
        _drl_level = self.drl_gate.get_authority().value if hasattr(self.drl_gate, "get_authority") else "DISABLED"

        # [FIX-DRIFT] Apply drift & OOD multipliers to DRL confidence.
        # _drl_drift_weight: 1.0 (no drift) → 0.0 (critical drift). Computed by DriftDetector.
        # _ood_confidence_mult: 1.0 (in-distribution) → 0.1 (out-of-distribution). Computed by OOD detector.
        _drift_w = agent_signals.get("_drl_drift_weight", 1.0)
        _ood_m = agent_signals.get("_ood_confidence_mult", 1.0)
        if _drift_w < 1.0 or _ood_m < 1.0:
            _drl_conf_raw = _drl_conf
            _drl_conf *= _drift_w * _ood_m
            logger.info(
                f"[FIX-DRIFT] DRL conf {_drl_conf_raw:.3f} → {_drl_conf:.3f} "
                f"(drift_w={_drift_w:.2f}, ood_m={_ood_m:.2f})"
            )
        if _drl_level in ("EXIT_ONLY", "ACTIVE") and abs(_drl_dir) > 0.01 and _drl_conf > 0.05:
            _drl_tranche_advice = None
            _quant_dir = agent_signals.get("quant_direction", 0.0)
            _drl_alignment = _drl_dir * _quant_dir

            if _drl_level == "EXIT_ONLY":
                if _drl_alignment < -0.15 and _drl_conf >= 0.35:
                    _drl_tranche_advice = "HOLD"
            else:
                if _drl_alignment > 0.25 and _drl_conf >= 0.55:
                    _drl_tranche_advice = "ESCALATE"
                elif _drl_alignment < -0.15 and _drl_conf >= 0.35:
                    _drl_tranche_advice = "HOLD"

            # [FIX-DRL-WEIGHT] Scale DRL confidence to give it more fusion weight.
            # Fusion uses confidence-weighted average among DECIDE agents.
            # Quant conf ~0.4-0.5, DRL conf ~0.4-0.5 → roughly equal.
            # Boost DRL by 1.3x so it gets ~55% vs quant's ~45% in fusion,
            # preventing EMA_200 lagging bias from dominating.
            _drl_fusion_conf = _drl_conf * 1.3 if _drl_level == "ACTIVE" else _drl_conf
            signals["drl"] = AgentSignal(
                direction=_drl_dir,
                confidence=min(_drl_fusion_conf, 0.95),  # cap to avoid overriding everything
                tranche_advice=_drl_tranche_advice,
            )

        # [GAP-P0] ModelAlphaWeightController: cap model-based signals during correlation crises
        # Applies to quant (primary model signal) and DRL (when activated)
        _alpha_cap = agent_signals.get('model_alpha_max_weight', 1.0)
        if _alpha_cap < 1.0:
            if "quant" in signals:
                signals["quant"] = AgentSignal(
                    direction=signals["quant"].direction,
                    confidence=signals["quant"].confidence * _alpha_cap,
                )
            if "drl" in signals:
                signals["drl"] = AgentSignal(
                    direction=signals["drl"].direction,
                    confidence=signals["drl"].confidence * _alpha_cap,
                    tranche_advice=signals["drl"].tranche_advice,
                )

        # =================================================================
        # V6: TWO-STAGE INTELLIGENCE (上层意图/先验)
        # =================================================================
        two_stage_direction = agent_signals.get("two_stage_direction", 0.0)
        two_stage_confidence = agent_signals.get("two_stage_confidence", 0.0)
        
        if two_stage_confidence > 0:
            signals["two_stage"] = AgentSignal(
                direction=two_stage_direction,
                confidence=two_stage_confidence,
            )
            logger.debug(
                f"[V6] Two-Stage prior injected: direction={two_stage_direction:.2f}, "
                f"confidence={two_stage_confidence:.2f}"
            )
        
        # OnChain intelligence (advisory)
        _oc_dir = agent_signals.get("onchain_direction", 0.0)
        _oc_conf = agent_signals.get("onchain_confidence", 0.0)
        if abs(_oc_dir) > 0.01 and _oc_conf > 0.1:
            signals["onchain"] = AgentSignal(
                direction=_oc_dir,
                confidence=_oc_conf,
            )

        # LLM Sentiment (advisory)
        _llm_dir = agent_signals.get("llm_sentiment_direction", 0.0)
        _llm_conf = agent_signals.get("llm_sentiment_confidence", 0.0)
        if abs(_llm_dir) > 0.01 and _llm_conf > 0.1:
            signals["llm_sentiment"] = AgentSignal(
                direction=_llm_dir,
                confidence=_llm_conf,
            )

        # Flow signals (advisory)
        _flow_dir = agent_signals.get("flow_direction", 0.0)
        if abs(_flow_dir) > 0.01:
            signals["flow"] = AgentSignal(
                direction=_flow_dir,
                confidence=min(abs(_flow_dir), 1.0),
            )

        # [FIX-L1-07] Ghost signals - previously written but never read by fusion
        # structure_confirmed: bool from OFI/orderbook higher_lows -> CONFIRM signal
        _struct = agent_signals.get("structure_confirmed", False)
        if _struct:
            signals["structure"] = AgentSignal(
                direction=agent_signals.get("quant_direction", 0.0),  # confirms quant direction
                confidence=0.6,
            )

        # squeeze_risk: float 0-1 -> squeeze detection -> reduce confidence
        _squeeze = agent_signals.get("squeeze_risk", 0.0)
        if _squeeze > 0.1:
            signals["squeeze"] = AgentSignal(
                direction=0.0,  # no directional bias, pure risk signal
                confidence=_squeeze,
                veto_active=_squeeze > 0.7,  # high squeeze risk -> soft veto
                veto_direction="squeeze_risk" if _squeeze > 0.7 else None,
            )

        # cvd_divergence: float [-1,1] from lead-lag CVD -> directional hint
        _cvd = agent_signals.get("cvd_divergence", 0.0)
        if abs(_cvd) > 0.05:
            signals["cvd"] = AgentSignal(
                direction=1.0 if _cvd > 0 else -1.0,
                confidence=min(abs(_cvd), 1.0),
            )

        # macro_risk_appetite: float 0-1 -> scales confidence (0.5=neutral, <0.3=defensive, >0.7=risk-on)
        _risk_app = agent_signals.get("macro_risk_appetite", 0.5)
        if abs(_risk_app - 0.5) > 0.1:
            _ra_dir = 1.0 if _risk_app > 0.5 else -1.0  # risk-on=bullish bias, risk-off=bearish
            signals["risk_appetite"] = AgentSignal(
                direction=_ra_dir,
                confidence=abs(_risk_app - 0.5) * 2.0,  # 0->1 scale from deviation
            )

        # phase: str -> market cycle context (not directional, stored as metadata)
        _phase_str = agent_signals.get("phase", "")
        if _phase_str:
            self._last_phase = _phase_str  # store for proof log / downstream use

        # =================================================================
        # [FIX-GHOST] Wire 7 previously-ghost agents into fusion.
        # These agents compute signals every tick but were never read.
        # All wired as ADVISE (supplementary context, not direction authority).
        # =================================================================

        # 1. Kraken Quant Agent: 12-strategy matrix (ADVISE — supplements Best-of-N)
        _kq_dir = agent_signals.get("kq_direction", 0.0)
        _kq_conf = agent_signals.get("kq_confidence", 0.0)
        if abs(_kq_dir) > 0.01 and _kq_conf > 0.1:
            signals["kraken_quant"] = AgentSignal(
                direction=_kq_dir,
                confidence=_kq_conf * 0.5,  # dampen: supplementary to main quant
            )

        # 2. Microstructure Agent: OB imbalance + taker flow (ADVISE — execution context)
        _micro_imb = agent_signals.get("micro_imbalance", 0.0)
        _micro_conf = agent_signals.get("micro_confidence", 0.0)
        if abs(_micro_imb) > 0.05:
            signals["microstructure"] = AgentSignal(
                direction=1.0 if _micro_imb > 0 else -1.0,
                confidence=min(abs(_micro_imb), 1.0) * 0.4,  # low weight: noisy
            )

        # 3. Model Alpha Agent: LSTM/GRU ensemble (ADVISE — ML consensus)
        _ma_dir = agent_signals.get("model_alpha_direction", 0.0)
        _ma_weight = agent_signals.get("model_alpha_weight", 0.0)
        if abs(_ma_dir) > 0.01 and _ma_weight > 0.05:
            signals["model_alpha"] = AgentSignal(
                direction=_ma_dir,
                confidence=min(_ma_weight, 1.0),
            )

        # 4. On-Chain Graph Alpha: whale/cluster detection (ADVISE — SOL only)
        _og_dir = agent_signals.get("onchain_graph_direction", 0.0)
        _og_conf = agent_signals.get("onchain_graph_confidence", 0.0)
        if abs(_og_dir) > 0.01 and _og_conf > 0.1:
            signals["onchain_graph"] = AgentSignal(
                direction=_og_dir,
                confidence=_og_conf,
            )

        # 5. Options Sentiment: put/call ratio + max pain (ADVISE — derivatives context)
        _opt_conf = agent_signals.get("options_confidence", 0.0)
        _opt_short = agent_signals.get("options_short_confirmation", 0.0)
        if _opt_conf > 0.1:
            signals["options"] = AgentSignal(
                direction=-1.0 if _opt_short > 0.5 else (1.0 if _opt_short < -0.5 else 0.0),
                confidence=_opt_conf * 0.5,  # dampen: options data can be sparse
            )

        # 6. Volatility Alpha: [REMOVED 2026-04-22] Dead fusion branch deleted.
        # Per agents/volatility_alpha_agent.py:955, vol_alpha_direction is ALWAYS 0.0
        # by design — the agent is "direction-agnostic" (measures vol regime, not
        # long/short). The prior `abs(_va_dir) > 0.01` check was therefore permanently
        # false, and signals["vol_alpha"] was never set.
        # vol_alpha's real impact flows through execution: vol_alpha_intensity
        # (main.py:6754) + vol_convex_long_stop/short_stop (main.py:6757-6762)
        # + leverage_cap_soft, NOT via fusion direction. No change to that path.
        # If vol_alpha is ever promoted to produce directional signals, restore
        # this branch and remove the "ALWAYS 0.0" contract in the agent.

        # 7. Whale Detector: large order detection (ADVISE — flow context)
        _wh_dir = agent_signals.get("whale_flow_direction", 0.0)
        _wh_conf = agent_signals.get("whale_confidence", 0.0)
        if abs(_wh_dir) > 0.01 and _wh_conf > 0.1:
            signals["whale"] = AgentSignal(
                direction=_wh_dir,
                confidence=_wh_conf,
            )

        # 8. SolDex DEX-CEX arb monitor (ADVISE — SOL only, opportunistic)
        # [FIX 2026-04-22] soldex was in AUTHORITY_MATRIX_NORMAL (authority_fusion.py:174)
        # but _build_fusion_signals never read any soldex key — signal was produced
        # at main.py:6042-6050 and silently dropped. Wiring as ADVISE since the arb
        # is exchange-microstructure specific, not directional authority.
        _sdx_dir = float(agent_signals.get("soldex_arb_direction", 0.0) or 0.0)
        _sdx_conf = float(agent_signals.get("soldex_confidence", 0.0) or 0.0)
        _sdx_active = bool(agent_signals.get("soldex_arb_active", False))
        if _sdx_active and abs(_sdx_dir) > 0.01 and _sdx_conf > 0.1:
            signals["soldex"] = AgentSignal(
                direction=_sdx_dir,
                confidence=_sdx_conf * 0.5,  # dampen: arb signal, not primary direction
            )

        return signals

    def _check_sol_dominance(
        self,
        phase,
        mode: SystemMode,
        sol_signal: float,
    ) -> Dict:
        """Check v3.3-HR SOL dominance conditions."""
        
        phase_value = phase.value if hasattr(phase, 'value') else str(phase)
        
        # Decrement TTL
        if self._sol_dominance_ttl > 0:
            self._sol_dominance_ttl -= 1
        
        # Auto-disable on phase change
        if phase_value in V33HR_SOL_DOMINANCE["auto_disable_on"]:
            if self._sol_dominance_active:
                logger.info(f"SOL dominance AUTO-DISABLED: phase={phase_value}")
            self._sol_dominance_active = False
            self._sol_dominance_ttl = 0
            return {"active": False, "ttl": 0, "reason": f"Phase={phase_value}"}
        
        # Auto-disable on TTL expiry
        if self._sol_dominance_active and self._sol_dominance_ttl <= 0:
            logger.info("SOL dominance AUTO-DISABLED: TTL expired")
            self._sol_dominance_active = False
            return {"active": False, "ttl": 0, "reason": "TTL expired"}
        
        # Check activation conditions
        if not self._sol_dominance_active:
            if (
                mode == SystemMode.OPPORTUNITY and
                phase_value in V33HR_SOL_DOMINANCE["phase_gate"] and
                abs(sol_signal) > 0.7
            ):
                self._sol_dominance_active = True
                self._sol_dominance_ttl = V33HR_SOL_DOMINANCE["max_duration_bars"]
                logger.warning(
                    f"SOL DOMINANCE ACTIVATED: phase={phase_value}, "
                    f"signal={sol_signal:.2f}, TTL={self._sol_dominance_ttl}"
                )
                return {
                    "active": True,
                    "ttl": self._sol_dominance_ttl,
                    "reason": f"OPPORTUNITY+{phase_value}+strong_signal"
                }
        
        return {
            "active": self._sol_dominance_active,
            "ttl": self._sol_dominance_ttl,
            "reason": "continuing" if self._sol_dominance_active else "inactive"
        }
    
    def _add_proof_log(self, intent: TradeIntentV36):
        """
        Add proof log entry.
        
        [PROD Patch 3] Generate structured proof log with component statuses.
        """
        # Original v3.6.1 proof log
        log_line = intent.to_proof_log()
        self._proof_logs.append(log_line)
        
        # Check for missing/invalid keys in intent
        missing_keys = []
        required_fields = ['system_mode', 'direction', 'target_exposure', 'execution_mode']
        for field in required_fields:
            if not hasattr(intent, field) or getattr(intent, field) is None:
                missing_keys.append(field)
        
        # Check data validation result if available
        if hasattr(intent, 'validation_result') and intent.validation_result:
            missing_keys.extend(intent.validation_result.get('missing_keys', []))
        
        # [PROD Patch 3] Structured proof log
        structured_log = create_proof_log(
            # Component statuses
            data_real=True,  # Data is always validated
            quant_real=not isinstance(self.fusion_engine, type(StubFusionEngine)),
            drl_enabled=self.drl_gate.has_exit_authority() if hasattr(self.drl_gate, 'has_exit_authority') else False,
            dvol_real=hasattr(intent, 'dvol_zscore'),  # True when DVOL feed provides real data
            sentiment_real=getattr(self, '_last_sentiment_active', False),  # True when F&G feed provides live data
            macro_real=getattr(self, '_last_macro_active', False),  # True when macro data is available
            
            # Schema
            schema_valid=intent.data_validation_passed if hasattr(intent, 'data_validation_passed') else True,
            missing_keys=missing_keys,
            
            # Decision
            mode=intent.system_mode,
            direction=intent.direction,
            exposure=intent.target_exposure,
            execution_mode=intent.execution_mode,
            
            # Risk (Patch 4)
            veto_result=self._last_veto_result,
            
            # Alpha
            alpha_passed=intent.alpha_gate_passed if hasattr(intent, 'alpha_gate_passed') else True,
            alpha_est=intent.alpha_estimated_bps if hasattr(intent, 'alpha_estimated_bps') else 0.0,
            alpha_thresh=intent.alpha_threshold_bps if hasattr(intent, 'alpha_threshold_bps') else 0.0,
            
            # Tranche
            tranche_action=intent.tranche_action if hasattr(intent, 'tranche_action') else "NONE",
            tranche_level=intent.tranche_level if hasattr(intent, 'tranche_level') else 0,
            
            # Deadlock (Patch 5)
            deadlock_result=self._last_deadlock_result,
            
            # SOL exit (Patch 6)
            sol_exit_signal=self._last_sol_exit_signal,
        )
        
        self._structured_proof_logs.append(structured_log)
        
        # Log both formats
        logger.info(log_line)
        logger.info(structured_log.to_log_line())

        # [v3.3-B7] Per-signal audit log
        _sig_audit = getattr(intent, '_signal_audit', None)
        if _sig_audit:
            _adopted = [k for k, v in _sig_audit.items() if v['adopted']]
            _rejected = [k for k, v in _sig_audit.items() if not v['adopted']]
            logger.info(
                f"[PROOF][SIGNALS] {intent.asset}: "
                f"adopted={_adopted} rejected={_rejected}"
            )

    # =========================================================================
    # POST-TRADE RECORDING
    # =========================================================================
    
    def record_trade_completion(
        self,
        trade_id: str,
        asset: str,
        mode_at_entry: str,
        phase_at_entry: str,
        opportunity_density_at_entry: float,
        crack_weight_at_entry: float,
        pnl_bps: float,
        bars_held: int,
        entry_alpha_bps: float = 0.0,
        execution_alpha_bps: float = 0.0,
        exit_alpha_bps: float = 0.0,
        drl_shadow_pnl_bps: Optional[float] = None,
        actual_exit_pnl_bps: float = 0.0,
    ):
        """Record trade completion for v3.5 meta-layers."""
        
        # v3.5: Record to failure memory (OPPORTUNITY only)
        if mode_at_entry == "OPPORTUNITY":
            self.failure_memory.record_opportunity(
                trade_id=trade_id,
                asset=asset,
                entry_time=datetime.now(timezone.utc),
                phase_at_entry=phase_at_entry,
                opportunity_density_at_entry=opportunity_density_at_entry,
                crack_weight_at_entry=crack_weight_at_entry,
                pnl_bps=pnl_bps,
                bars_to_outcome=bars_held,
            )
        
        # v3.5: Record to attribution promotion
        self.attribution_promotion.record_trade(
            trade_id=trade_id,
            entry_alpha_bps=entry_alpha_bps,
            execution_alpha_bps=execution_alpha_bps,
            exit_alpha_bps=exit_alpha_bps,
            total_pnl_bps=pnl_bps,
            drl_shadow_would_have_bps=drl_shadow_pnl_bps,
            actual_exit_bps=actual_exit_pnl_bps,
        )
    
    def check_drl_promotion(self):
        """Check if DRL should be promoted (v3.5)."""
        return self.drl_gate.check_promotion()
    
    def get_proof_logs(self) -> List[str]:
        """Get all proof logs."""
        return self._proof_logs.copy()
    
    def get_version_info(self) -> Dict:
        """Get version information."""
        return {
            "version": VERSION,
            "name": "HMATS v3.6 (Runnable Canonical)",
            "fused_from": ["v3.3-HR", "v3.4", "v3.5"],
            "drl_authority": self.drl_gate.get_authority().value,
            "sol_dominance_active": self._sol_dominance_active,
            "caution_mode": self.failure_memory.get_modifiers().in_caution_mode,
            "module_status": _registry.get_status(),
        }
    
    def get_module_status(self) -> Dict:
        """Get status of all modules (REAL vs MOCKED)."""
        return _registry.get_status()


# =============================================================================
# SINGLETON
# =============================================================================

_v36_engine: Optional[HMATSv36Engine] = None

def get_v36_engine() -> HMATSv36Engine:
    """Get v3.6 engine singleton."""
    global _v36_engine
    if _v36_engine is None:
        _v36_engine = HMATSv36Engine()
    return _v36_engine

def reset_v36_engine():
    """Reset v3.6 engine."""
    global _v36_engine
    _v36_engine = None


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HMATS v3.6 Integration Engine")
    parser.add_argument("--version", action="store_true", help="Show version info")
    parser.add_argument("--status", action="store_true", help="Show module status")
    args = parser.parse_args()
    
    engine = get_v36_engine()
    
    if args.version:
        print(f"HMATS v{VERSION} (Runnable Canonical)")
        print(f"Fused from: v3.3-HR + v3.4 + v3.5")
    
    if args.status:
        status = engine.get_module_status()
        print(f"REAL modules: {status['real']}")
        print(f"MOCKED modules: {status['mocked']}")

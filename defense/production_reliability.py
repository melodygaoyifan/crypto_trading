"""
================================================================================
HMATS v3.6.1 - PRODUCTION RELIABILITY PATCHES
================================================================================
Version: 3.6.1-PROD
Purpose: Minimal patches for production reliability

PATCHES IMPLEMENTED:
    Patch 1: Canonical schema key mapping + required-key validation
    Patch 2: Single canonical entrypoint + legacy redirect + version banner
    Patch 3: Structured [PROOF] log per 4H tick
    Patch 4: HARD vs SOFT risk veto classification
    Patch 5: Execution deadlock + tranche stage coupling
    Patch 6: SOL dominance forced exit (immediate via execution loop)
================================================================================
"""

import logging
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any, Callable
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from functools import wraps

logger = logging.getLogger(__name__)

VERSION = "3.6.1-PROD"
VERSION_BANNER = f"""
================================================================================
    HMATS v{VERSION} - Hierarchical Multi-Agent Trading System
    Build: Production Reliability Patches Applied
    Timestamp: {{timestamp}}
================================================================================
"""


# =============================================================================
# PATCH 2: CANONICAL ENTRYPOINT + VERSION BANNER + CALL-CHAIN PROOF
# =============================================================================

class CallChainProof:
    """Tracks call chain for debugging and audit."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._chain = deque(maxlen=100)  # [FIX-46] was unbounded list
            cls._instance._entry_time = None
        return cls._instance
    
    def reset(self):
        self._chain = deque(maxlen=100)  # [FIX-46]
        self._entry_time = datetime.now(timezone.utc)
    
    def record(self, component: str, action: str):
        self._chain.append({
            "component": component,
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def to_proof_log(self) -> str:
        if not self._chain:
            return "CallChain=EMPTY"
        chain_str = " -> ".join(f"{c['component']}:{c['action']}" for c in self._chain[-5:])
        return f"CallChain=[{chain_str}]"


_call_chain = CallChainProof()


def canonical_entrypoint(func: Callable) -> Callable:
    """
    Decorator marking a function as the canonical entrypoint.
    Prints version banner and initializes call-chain proof.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Print version banner on first call
        if not hasattr(wrapper, '_banner_printed'):
            print(VERSION_BANNER.format(timestamp=datetime.now(timezone.utc).isoformat()))
            wrapper._banner_printed = True
        
        # Reset call chain
        _call_chain.reset()
        _call_chain.record("ENTRYPOINT", func.__name__)
        
        # Log entry
        logger.info(f"[ENTRYPOINT] v{VERSION} | {func.__name__} | {datetime.now(timezone.utc).isoformat()}")
        
        return func(*args, **kwargs)
    
    wrapper._is_canonical_entrypoint = True
    return wrapper


def legacy_redirect(canonical_func: Callable) -> Callable:
    """
    Decorator for legacy entrypoints that redirects to canonical.
    
    Usage:
        @legacy_redirect(canonical_decide)
        def old_decide(...):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.warning(
                f"[LEGACY_REDIRECT] {func.__name__} is deprecated. "
                f"Redirecting to {canonical_func.__name__}"
            )
            _call_chain.record("LEGACY", f"{func.__name__}->{canonical_func.__name__}")
            return canonical_func(*args, **kwargs)
        
        wrapper._is_legacy = True
        wrapper._canonical = canonical_func
        return wrapper
    return decorator


class EntrypointRegistry:
    """Registry of valid entrypoints."""
    
    _canonical: Optional[str] = None
    _legacy_map: Dict[str, str] = {}
    
    @classmethod
    def set_canonical(cls, name: str):
        cls._canonical = name
        logger.info(f"[ENTRYPOINT] Canonical entrypoint set: {name}")
    
    @classmethod
    def register_legacy(cls, legacy_name: str, canonical_name: str):
        cls._legacy_map[legacy_name] = canonical_name
    
    @classmethod
    def get_canonical(cls) -> Optional[str]:
        return cls._canonical
    
    @classmethod
    def is_legacy(cls, name: str) -> bool:
        return name in cls._legacy_map


# =============================================================================
# PATCH 3: STRUCTURED [PROOF] LOG PER 4H TICK
# =============================================================================

class ComponentStatus(Enum):
    """Status of system components."""
    REAL = "REAL"           # Live data/signals
    DISABLED = "DISABLED"   # Component disabled
    MOCK = "MOCK"           # Mock/simulated data
    ERROR = "ERROR"         # Component error
    STALE = "STALE"         # Data is stale


@dataclass
class StructuredProofLog:
    """
    Structured proof log for 4H tick.
    
    Patch 3: Lists REAL/DISABLED status for all components.
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: str = VERSION
    
    # Component statuses (Patch 3)
    data_status: ComponentStatus = ComponentStatus.DISABLED
    quant_status: ComponentStatus = ComponentStatus.DISABLED
    drl_status: ComponentStatus = ComponentStatus.DISABLED
    dvol_status: ComponentStatus = ComponentStatus.DISABLED
    sentiment_status: ComponentStatus = ComponentStatus.DISABLED
    macro_status: ComponentStatus = ComponentStatus.DISABLED
    
    # Schema validation (Patch 1)
    schema_valid: bool = True
    schema_missing_keys: List[str] = field(default_factory=list)
    
    # Mode and intent
    mode: str = "UNKNOWN"
    intent_direction: float = 0.0
    intent_exposure: float = 0.0
    
    # Execution
    execution_mode: str = "UNKNOWN"
    tranche_action: str = "NONE"
    tranche_level: int = 0
    
    # Risk veto (Patch 4)
    veto_type: str = "NONE"  # HARD, SOFT, NONE
    veto_conditions: List[str] = field(default_factory=list)
    exposure_cap: float = 1.0
    
    # Alpha gate
    alpha_passed: bool = True
    alpha_estimated_bps: float = 0.0
    alpha_threshold_bps: float = 0.0
    
    # Deadlock (Patch 5)
    deadlock_bars: int = 0
    deadlock_resolution: str = "NONE"
    
    # SOL dominance (Patch 6)
    sol_forced_exit: bool = False
    sol_exit_urgency: str = "NONE"
    
    # Call chain (Patch 2)
    call_chain: str = ""
    
    def to_log_line(self) -> str:
        """Generate single-line structured proof log."""
        return (
            f"[PROOF] {self.timestamp.isoformat()} | v{self.version} | "
            f"Data={self.data_status.value} | Quant={self.quant_status.value} | "
            f"DRL={self.drl_status.value} | DVOL={self.dvol_status.value} | "
            f"Sentiment={self.sentiment_status.value} | Macro={self.macro_status.value} | "
            f"Schema={{valid={self.schema_valid},missing={len(self.schema_missing_keys)}}} | "
            f"Mode={self.mode} | "
            f"Intent={{dir={self.intent_direction:+.2f},exp={self.intent_exposure:.1%}}} | "
            f"Exec={self.execution_mode} | "
            f"Veto={{type={self.veto_type},cap={self.exposure_cap:.0%}}} | "
            f"Alpha={{pass={self.alpha_passed},est={self.alpha_estimated_bps:.0f},thresh={self.alpha_threshold_bps:.0f}}} | "
            f"Tranche={{action={self.tranche_action},level={self.tranche_level}}} | "
            f"Deadlock={{bars={self.deadlock_bars},res={self.deadlock_resolution}}} | "
            f"SOL={{exit={self.sol_forced_exit},urgency={self.sol_exit_urgency}}}"
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "components": {
                "data": self.data_status.value,
                "quant": self.quant_status.value,
                "drl": self.drl_status.value,
                "dvol": self.dvol_status.value,
                "sentiment": self.sentiment_status.value,
                "macro": self.macro_status.value,
            },
            "schema": {
                "valid": self.schema_valid,
                "missing_keys": self.schema_missing_keys,
            },
            "decision": {
                "mode": self.mode,
                "intent_direction": self.intent_direction,
                "intent_exposure": self.intent_exposure,
                "execution_mode": self.execution_mode,
            },
            "risk": {
                "veto_type": self.veto_type,
                "veto_conditions": self.veto_conditions,
                "exposure_cap": self.exposure_cap,
            },
            "alpha": {
                "passed": self.alpha_passed,
                "estimated_bps": self.alpha_estimated_bps,
                "threshold_bps": self.alpha_threshold_bps,
            },
            "tranche": {
                "action": self.tranche_action,
                "level": self.tranche_level,
            },
            "deadlock": {
                "bars": self.deadlock_bars,
                "resolution": self.deadlock_resolution,
            },
            "sol_dominance": {
                "forced_exit": self.sol_forced_exit,
                "exit_urgency": self.sol_exit_urgency,
            },
        }


class ProofLogBuilder:
    """Builder for structured proof logs."""
    
    def __init__(self):
        self._log = StructuredProofLog()
    
    def set_components(
        self,
        data: ComponentStatus = ComponentStatus.DISABLED,
        quant: ComponentStatus = ComponentStatus.DISABLED,
        drl: ComponentStatus = ComponentStatus.DISABLED,
        dvol: ComponentStatus = ComponentStatus.DISABLED,
        sentiment: ComponentStatus = ComponentStatus.DISABLED,
        macro: ComponentStatus = ComponentStatus.DISABLED,
    ) -> 'ProofLogBuilder':
        self._log.data_status = data
        self._log.quant_status = quant
        self._log.drl_status = drl
        self._log.dvol_status = dvol
        self._log.sentiment_status = sentiment
        self._log.macro_status = macro
        return self
    
    def set_schema(self, valid: bool, missing_keys: List[str] = None) -> 'ProofLogBuilder':
        self._log.schema_valid = valid
        self._log.schema_missing_keys = missing_keys or []
        return self
    
    def set_decision(self, mode: str, direction: float, exposure: float) -> 'ProofLogBuilder':
        self._log.mode = mode
        self._log.intent_direction = direction
        self._log.intent_exposure = exposure
        return self
    
    def set_execution(self, mode: str, tranche_action: str, tranche_level: int) -> 'ProofLogBuilder':
        self._log.execution_mode = mode
        self._log.tranche_action = tranche_action
        self._log.tranche_level = tranche_level
        return self
    
    def set_veto(self, veto_type: str, conditions: List[str], cap: float) -> 'ProofLogBuilder':
        self._log.veto_type = veto_type
        self._log.veto_conditions = conditions
        self._log.exposure_cap = cap
        return self
    
    def set_alpha(self, passed: bool, estimated: float, threshold: float) -> 'ProofLogBuilder':
        self._log.alpha_passed = passed
        self._log.alpha_estimated_bps = estimated
        self._log.alpha_threshold_bps = threshold
        return self
    
    def set_deadlock(self, bars: int, resolution: str) -> 'ProofLogBuilder':
        self._log.deadlock_bars = bars
        self._log.deadlock_resolution = resolution
        return self
    
    def set_sol_exit(self, forced: bool, urgency: str) -> 'ProofLogBuilder':
        self._log.sol_forced_exit = forced
        self._log.sol_exit_urgency = urgency
        return self
    
    def set_call_chain(self, chain: str) -> 'ProofLogBuilder':
        self._log.call_chain = chain
        return self
    
    def build(self) -> StructuredProofLog:
        self._log.timestamp = datetime.now(timezone.utc)
        return self._log


# =============================================================================
# PATCH 4: HARD vs SOFT RISK VETO
# =============================================================================

class VetoType(Enum):
    """Risk veto classification."""
    NONE = "NONE"
    SOFT = "SOFT"  # Caps exposure in OPPORTUNITY, blocks in NORMAL
    HARD = "HARD"  # Always blocks, regardless of mode


class HardVetoCondition(Enum):
    """Conditions that ALWAYS trigger HARD veto (non-negotiable)."""
    DRAWDOWN_HALT = auto()          # DD >= 10%
    CORRELATION_CRISIS = auto()      # Correlation >= 0.95
    DATA_INTEGRITY_FAIL = auto()     # CRC mismatch, shadow ledger
    EXECUTION_BLOCKED = auto()       # 3+ consecutive failures
    FLASH_CRASH = auto()             # >= 5% in 5 minutes
    EXTREME_DVOL = auto()            # DVOL z-score >= 5.0
    ALL_CONFLICT_FLAT = auto()       # All signals in direct conflict
    MISSING_REQUIRED_KEYS = auto()   # Schema: missing required keys


class SoftVetoCondition(Enum):
    """Conditions that trigger SOFT veto (cap exposure in OPPORTUNITY)."""
    ELEVATED_DVOL = auto()           # DVOL z-score 3.0-4.9
    LIQUIDITY_WARNING = auto()       # Depth $100K-$300K
    CORRELATION_ELEVATED = auto()    # Correlation 0.85-0.94
    PARTIAL_CONFLICT = auto()        # 2/3 signals conflict
    WEEKEND_LIQUIDITY = auto()       # Weekend reduced liquidity
    MISSING_CRITICAL_KEYS = auto()   # Schema: missing critical keys


@dataclass
class RiskVetoResult:
    """Result of risk veto classification."""
    veto_type: VetoType = VetoType.NONE
    hard_conditions: List[HardVetoCondition] = field(default_factory=list)
    soft_conditions: List[SoftVetoCondition] = field(default_factory=list)
    exposure_cap: float = 1.0  # For SOFT veto, maximum allowed exposure
    allows_trade: bool = True
    reason: str = ""
    
    def to_proof_str(self) -> str:
        hard_str = ",".join(c.name for c in self.hard_conditions) or "NONE"
        soft_str = ",".join(c.name for c in self.soft_conditions) or "NONE"
        return f"Veto={self.veto_type.value}|HARD=[{hard_str}]|SOFT=[{soft_str}]|cap={self.exposure_cap:.0%}"


class RiskVetoClassifier:
    """
    Classifies risk conditions into HARD vs SOFT veto.

    Patch 4: SOFT veto in OPPORTUNITY mode caps exposure but doesn't block.
             HARD veto always blocks regardless of mode.

    Accepts optional weekend_config dict to override weekend soft veto cap.
    Profile-gated: without config, defaults to HIGH_RISK behavior (0.40 cap).
    """

    # HARD veto thresholds (non-negotiable)
    HARD_THRESHOLDS = {
        "drawdown": 0.20,           # [UNLEASH UL-5] synced with RULETABLE
        "correlation": 0.98,        # [UNLEASH UL-4] synced with RULETABLE
        "dvol_zscore": 5.0,         # Extreme volatility
        "flash_crash_pct": 0.05,    # 5% in 5 minutes
        "execution_failures": 3,    # Consecutive failures
    }

    # SOFT veto thresholds (negotiable in OPPORTUNITY)
    SOFT_THRESHOLDS = {
        "dvol_zscore_min": 3.0,
        "dvol_zscore_max": 4.9,
        "correlation_min": 0.85,
        "correlation_max": 0.94,
        "liquidity_warning": 300_000,
        "liquidity_critical": 100_000,
    }

    # Default exposure caps for SOFT conditions
    _DEFAULT_SOFT_EXPOSURE_CAPS = {
        SoftVetoCondition.ELEVATED_DVOL: 0.50,
        SoftVetoCondition.LIQUIDITY_WARNING: 0.60,
        SoftVetoCondition.CORRELATION_ELEVATED: 0.70,
        SoftVetoCondition.PARTIAL_CONFLICT: 0.50,
        SoftVetoCondition.WEEKEND_LIQUIDITY: 0.40,
        SoftVetoCondition.MISSING_CRITICAL_KEYS: 0.30,
    }

    def __init__(self, weekend_config: Optional[Dict] = None):
        # Build instance-level caps from defaults + any weekend override
        self.SOFT_EXPOSURE_CAPS = dict(self._DEFAULT_SOFT_EXPOSURE_CAPS)
        self._weekend_config = weekend_config or {}

        # Override weekend soft veto cap from profile config
        weekend_cap = self._weekend_config.get("weekend_soft_veto_cap")
        if weekend_cap is not None:
            self.SOFT_EXPOSURE_CAPS[SoftVetoCondition.WEEKEND_LIQUIDITY] = float(weekend_cap)
            logger.info(
                f"[RiskVetoClassifier] Weekend soft veto cap overridden: "
                f"0.40 -> {weekend_cap:.2f}"
            )
    
    def classify(
        self,
        mode: str,
        drawdown: float = 0.0,
        dvol_zscore: float = 0.0,
        correlation: float = 0.0,
        liquidity_usd: float = float('inf'),
        signal_conflict_score: float = 0.0,  # 0=none, 0.5=partial, 1.0=full
        data_valid: bool = True,
        execution_failures: int = 0,
        flash_crash_active: bool = False,
        is_weekend: bool = False,
        missing_required_keys: List[str] = None,
        missing_critical_keys: List[str] = None,
    ) -> RiskVetoResult:
        """
        Classify risk conditions.
        
        Patch 4: SOFT may cap exposure in OPPORTUNITY, but must NOT
        become HARD unless hard conditions are actually met.
        """
        
        hard_conditions = []
        soft_conditions = []
        
        # === CHECK HARD CONDITIONS (always block) ===
        
        if drawdown >= self.HARD_THRESHOLDS["drawdown"]:
            hard_conditions.append(HardVetoCondition.DRAWDOWN_HALT)
        
        if correlation >= self.HARD_THRESHOLDS["correlation"]:
            hard_conditions.append(HardVetoCondition.CORRELATION_CRISIS)
        
        if dvol_zscore >= self.HARD_THRESHOLDS["dvol_zscore"]:
            hard_conditions.append(HardVetoCondition.EXTREME_DVOL)
        
        if not data_valid:
            hard_conditions.append(HardVetoCondition.DATA_INTEGRITY_FAIL)
        
        if execution_failures >= self.HARD_THRESHOLDS["execution_failures"]:
            hard_conditions.append(HardVetoCondition.EXECUTION_BLOCKED)
        
        if flash_crash_active:
            hard_conditions.append(HardVetoCondition.FLASH_CRASH)
        
        if signal_conflict_score >= 1.0:
            hard_conditions.append(HardVetoCondition.ALL_CONFLICT_FLAT)
        
        if missing_required_keys:
            hard_conditions.append(HardVetoCondition.MISSING_REQUIRED_KEYS)
        
        # === CHECK SOFT CONDITIONS (only if no HARD conditions) ===
        
        if not hard_conditions:
            if self.SOFT_THRESHOLDS["dvol_zscore_min"] <= dvol_zscore < self.SOFT_THRESHOLDS["dvol_zscore_max"]:
                soft_conditions.append(SoftVetoCondition.ELEVATED_DVOL)
            
            if self.SOFT_THRESHOLDS["liquidity_critical"] <= liquidity_usd < self.SOFT_THRESHOLDS["liquidity_warning"]:
                soft_conditions.append(SoftVetoCondition.LIQUIDITY_WARNING)
            
            if self.SOFT_THRESHOLDS["correlation_min"] <= correlation < self.SOFT_THRESHOLDS["correlation_max"]:
                soft_conditions.append(SoftVetoCondition.CORRELATION_ELEVATED)
            
            if 0.5 <= signal_conflict_score < 1.0:
                soft_conditions.append(SoftVetoCondition.PARTIAL_CONFLICT)
            
            if is_weekend:
                soft_conditions.append(SoftVetoCondition.WEEKEND_LIQUIDITY)
            
            if missing_critical_keys:
                soft_conditions.append(SoftVetoCondition.MISSING_CRITICAL_KEYS)
        
        # === DETERMINE VETO TYPE AND RESULT ===
        
        if hard_conditions:
            # HARD veto - always blocks
            return RiskVetoResult(
                veto_type=VetoType.HARD,
                hard_conditions=hard_conditions,
                exposure_cap=0.0,
                allows_trade=False,
                reason=f"HARD veto: {[c.name for c in hard_conditions]}"
            )
        
        if soft_conditions:
            # Calculate minimum exposure cap from all soft conditions
            min_cap = min(
                self.SOFT_EXPOSURE_CAPS.get(c, 1.0) 
                for c in soft_conditions
            )
            soft_names = [c.name for c in soft_conditions]
            weekend_only_soft = (
                len(soft_conditions) == 1
                and soft_conditions[0] == SoftVetoCondition.WEEKEND_LIQUIDITY
            )
            
            correlation_only_soft = (
                len(soft_conditions) == 1
                and soft_conditions[0] == SoftVetoCondition.CORRELATION_ELEVATED
            )
            correlation_weekend_only_soft = (
                len(soft_conditions) == 2
                and set(soft_conditions)
                == {
                    SoftVetoCondition.CORRELATION_ELEVATED,
                    SoftVetoCondition.WEEKEND_LIQUIDITY,
                }
            )

            if mode == "OPPORTUNITY":
                # Patch 4: In OPPORTUNITY, SOFT only caps exposure
                return RiskVetoResult(
                    veto_type=VetoType.SOFT,
                    soft_conditions=soft_conditions,
                    exposure_cap=min_cap,
                    allows_trade=True,  # ALLOW with cap
                    reason=f"SOFT veto (OPPORTUNITY mode): {soft_names} cap at {min_cap:.0%}"
                )
            elif weekend_only_soft:
                # Weekend liquidity is already handled by the dedicated weekend
                # manager. In NORMAL mode, blocking here duplicates that gate
                # and causes systematic 0-fill weekends even when alpha passes.
                return RiskVetoResult(
                    veto_type=VetoType.SOFT,
                    soft_conditions=soft_conditions,
                    exposure_cap=min_cap,
                    allows_trade=True,
                    reason=f"SOFT weekend cap (NORMAL mode): {soft_names} cap at {min_cap:.0%}"
                )
            elif correlation_only_soft:
                # Correlation already has downstream size/cap governors.
                # Blocking here duplicates that protection and suppresses
                # valid same-side portfolio trades in NORMAL mode.
                return RiskVetoResult(
                    veto_type=VetoType.SOFT,
                    soft_conditions=soft_conditions,
                    exposure_cap=min_cap,
                    allows_trade=True,
                    reason=f"SOFT correlation cap (NORMAL mode): {soft_names} cap at {min_cap:.0%}"
                )
            elif correlation_weekend_only_soft:
                # Weekend liquidity and elevated correlation are already
                # downstream-capped by dedicated weekend/correlation governors.
                # Blocking here creates systematic no-fill windows for
                # otherwise valid same-side high-risk trades.
                return RiskVetoResult(
                    veto_type=VetoType.SOFT,
                    soft_conditions=soft_conditions,
                    exposure_cap=min_cap,
                    allows_trade=True,
                    reason=f"SOFT correlation+weekend cap (NORMAL mode): {soft_names} cap at {min_cap:.0%}"
                )
            else:
                # In NORMAL mode, SOFT caps exposure (was: blocks entirely)
                # The individual caps (30-70%) already provide adequate protection.
                # Hard-blocking kills trade frequency and biases toward no-action.
                return RiskVetoResult(
                    veto_type=VetoType.SOFT,
                    soft_conditions=soft_conditions,
                    exposure_cap=min_cap,
                    allows_trade=True,
                    reason=f"SOFT veto (NORMAL mode): {soft_names} cap at {min_cap:.0%}"
                )
        
        # No veto
        return RiskVetoResult(
            veto_type=VetoType.NONE,
            exposure_cap=1.0,
            allows_trade=True
        )


# =============================================================================
# PATCH 5: EXECUTION DEADLOCK + TRANCHE COUPLING
# =============================================================================

class DeadlockBias(Enum):
    """Bias for deadlock resolution based on tranche stage."""
    FORCE = "FORCE"      # T1: bias toward FORCE_AGGRESSIVE
    ABORT = "ABORT"      # T2+: bias toward ABORT_OPPORTUNITY
    NEUTRAL = "NEUTRAL"  # No strong bias


from core.canonical_enums import DeadlockResolution  # Single source of truth


@dataclass
class DeadlockResult:
    """Result of deadlock resolution with tranche coupling."""
    resolution: DeadlockResolution = DeadlockResolution.NONE
    bias: DeadlockBias = DeadlockBias.NEUTRAL
    tranche_level: int = 0
    deadlock_bars: int = 0
    edge_decay_pct: float = 0.0
    reason: str = ""


class TrancheAwareDeadlockResolver:
    """
    Couples execution deadlock resolution with tranche stage.
    
    Patch 5:
        - T1: Bias toward FORCE_AGGRESSIVE (small position, can afford aggression)
        - T2+: Bias toward ABORT_OPPORTUNITY (larger position, protect capital)
    """
    
    # Deadlock thresholds per tranche
    T1_FORCE_BARS = 2      # T1: force after 2 bars
    T2_PLUS_ABORT_BARS = 3  # T2+: abort after 3 bars
    GENERAL_ABORT_BARS = 4  # General timeout
    
    # Edge decay
    EDGE_DECAY_PER_BAR = 0.15  # 15% decay per bar
    MIN_EDGE_FOR_FORCE = 1.5   # Minimum edge/friction ratio to force
    
    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._initial_edges: Dict[str, float] = {}
    
    def resolve(
        self,
        asset: str,
        tranche_level: int,
        current_edge_bps: float,
        friction_bps: float,
        is_stuck: bool,
        mode: str,
    ) -> DeadlockResult:
        """
        Resolve deadlock with tranche-aware bias.
        
        Patch 5: T1 biases FORCE, T2+ biases ABORT.
        """
        
        # Track deadlock state
        if is_stuck:
            self._counters[asset] = self._counters.get(asset, 0) + 1
            if asset not in self._initial_edges:
                self._initial_edges[asset] = current_edge_bps
        else:
            self._counters[asset] = 0
            self._initial_edges.pop(asset, None)
        
        deadlock_bars = self._counters.get(asset, 0)
        
        # No deadlock
        if deadlock_bars == 0:
            return DeadlockResult(
                resolution=DeadlockResolution.NONE,
                tranche_level=tranche_level,
                deadlock_bars=0
            )
        
        # Calculate decayed edge
        decay = 1.0 - (self.EDGE_DECAY_PER_BAR * deadlock_bars)
        decay = max(0.1, decay)  # Floor at 10%
        decayed_edge = current_edge_bps * decay
        edge_friction_ratio = decayed_edge / friction_bps if friction_bps > 0 else 0
        
        # Determine bias based on tranche level
        if tranche_level <= 1:
            bias = DeadlockBias.FORCE
            threshold = self.T1_FORCE_BARS
        else:
            bias = DeadlockBias.ABORT
            threshold = self.T2_PLUS_ABORT_BARS
        
        # Not yet at threshold
        if deadlock_bars < threshold:
            return DeadlockResult(
                resolution=DeadlockResolution.NONE,
                bias=bias,
                tranche_level=tranche_level,
                deadlock_bars=deadlock_bars,
                edge_decay_pct=(1.0 - decay) * 100,
                reason=f"Deadlock {deadlock_bars}/{threshold} bars, waiting"
            )
        
        # At threshold - apply resolution based on bias
        if bias == DeadlockBias.FORCE:
            # T1: try to force if edge still covers friction
            if edge_friction_ratio >= self.MIN_EDGE_FOR_FORCE:
                return DeadlockResult(
                    resolution=DeadlockResolution.FORCE_AGGRESSIVE,
                    bias=bias,
                    tranche_level=tranche_level,
                    deadlock_bars=deadlock_bars,
                    edge_decay_pct=(1.0 - decay) * 100,
                    reason=f"T1 FORCE: edge/friction={edge_friction_ratio:.1f}x still viable"
                )
            else:
                # Edge degraded too much
                return DeadlockResult(
                    resolution=DeadlockResolution.ABORT_OPPORTUNITY,
                    bias=bias,
                    tranche_level=tranche_level,
                    deadlock_bars=deadlock_bars,
                    edge_decay_pct=(1.0 - decay) * 100,
                    reason=f"T1 ABORT: edge degraded to {edge_friction_ratio:.1f}x"
                )
        
        else:
            # T2+: abort to protect capital
            return DeadlockResult(
                resolution=DeadlockResolution.ABORT_OPPORTUNITY,
                bias=bias,
                tranche_level=tranche_level,
                deadlock_bars=deadlock_bars,
                edge_decay_pct=(1.0 - decay) * 100,
                reason=f"T{tranche_level} ABORT: protecting larger position"
            )
    
    def reset(self, asset: str):
        """Reset deadlock state for asset."""
        self._counters.pop(asset, None)
        self._initial_edges.pop(asset, None)


# =============================================================================
# PATCH 6: SOL DOMINANCE FORCED EXIT (IMMEDIATE VIA EXECUTION LOOP)
# =============================================================================

class ExitUrgency(Enum):
    """Urgency level for forced exits."""
    NONE = "NONE"               # No exit needed
    SCHEDULED = "SCHEDULED"     # Exit at next 4H bar
    IMMEDIATE = "IMMEDIATE"     # Exit NOW via execution loop (200ms)


@dataclass
class ForcedExitSignal:
    """Signal for forced exit."""
    should_exit: bool = False
    urgency: ExitUrgency = ExitUrgency.NONE
    reason: str = ""
    target_exposure: float = 0.0
    execution_mode: str = "AGGRESSIVE"
    triggered_at: Optional[datetime] = None
    
    def to_execution_loop_signal(self) -> Optional[Dict]:
        """Convert to signal for execution loop (200ms frequency)."""
        if self.urgency != ExitUrgency.IMMEDIATE:
            return None
        
        return {
            "action": "FORCED_EXIT",
            "target_exposure": self.target_exposure,
            "execution_mode": self.execution_mode,
            "reason": self.reason,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
        }


class SOLDominanceForcedExit:
    """
    SOL dominance forced exit with immediate execution capability.
    
    Patch 6: Can trigger IMMEDIATELY via execution loop, not waiting for 4H.
    
    IMMEDIATE triggers (bypass 4H wait):
        - TTL expired (dominance was active, now expired)
        - Phase transition to SATURATION/EXHAUSTION/CRASH
        - Correlation spike >= 0.92
        - VPIN spike >= 0.90
        - Flash move >= 3%
    
    SCHEDULED triggers (wait for 4H):
        - TTL approaching (1 bar remaining)
        - Signal weakening gradually
    """
    
    # Immediate exit thresholds
    IMMEDIATE_VPIN_THRESHOLD = 0.90
    IMMEDIATE_CORRELATION_THRESHOLD = 0.92
    IMMEDIATE_FLASH_MOVE_PCT = 0.03  # 3%
    
    # Danger phases requiring immediate exit
    DANGER_PHASES = {"SATURATION", "EXHAUSTION", "CRASH"}
    
    # Safe phases
    SAFE_PHASES = {"IGNITION", "EXPANSION", "ACCUMULATION"}
    
    def __init__(self):
        self._active_dominance: Dict[str, datetime] = {}
        self._last_phase: Dict[str, str] = {}
    
    def check_forced_exit(
        self,
        asset: str,
        dominance_active: bool,
        dominance_ttl: int,
        current_phase: str,
        vpin: float,
        correlation: float,
        price_move_pct: float,  # Recent price move %
        current_exposure: float,
    ) -> ForcedExitSignal:
        """
        Check if forced exit should trigger.
        
        Patch 6: Returns IMMEDIATE for urgent conditions,
        allowing execution loop to act without waiting for 4H.
        """
        
        now = datetime.now(timezone.utc)
        
        # No position -> no exit needed
        if current_exposure <= 0.01:
            return ForcedExitSignal(should_exit=False)
        
        # Track dominance state
        if dominance_active and asset not in self._active_dominance:
            self._active_dominance[asset] = now
        
        # === CHECK IMMEDIATE EXIT CONDITIONS ===
        
        # 1. VPIN spike -> IMMEDIATE
        if vpin >= self.IMMEDIATE_VPIN_THRESHOLD:
            logger.warning(f"[PATCH6] IMMEDIATE EXIT: VPIN {vpin:.2f} >= {self.IMMEDIATE_VPIN_THRESHOLD}")
            return ForcedExitSignal(
                should_exit=True,
                urgency=ExitUrgency.IMMEDIATE,
                reason=f"VPIN spike: {vpin:.2f}",
                target_exposure=0.0,
                execution_mode="AGGRESSIVE",
                triggered_at=now
            )
        
        # 2. Correlation spike -> IMMEDIATE
        if correlation >= self.IMMEDIATE_CORRELATION_THRESHOLD:
            logger.warning(f"[PATCH6] IMMEDIATE EXIT: Correlation {correlation:.2f} >= {self.IMMEDIATE_CORRELATION_THRESHOLD}")
            return ForcedExitSignal(
                should_exit=True,
                urgency=ExitUrgency.IMMEDIATE,
                reason=f"Correlation spike: {correlation:.2f}",
                target_exposure=0.0,
                execution_mode="AGGRESSIVE",
                triggered_at=now
            )
        
        # 3. Flash move -> IMMEDIATE
        if abs(price_move_pct) >= self.IMMEDIATE_FLASH_MOVE_PCT:
            logger.warning(f"[PATCH6] IMMEDIATE EXIT: Flash move {price_move_pct:.1%}")
            return ForcedExitSignal(
                should_exit=True,
                urgency=ExitUrgency.IMMEDIATE,
                reason=f"Flash move: {price_move_pct:.1%}",
                target_exposure=0.0,
                execution_mode="AGGRESSIVE",
                triggered_at=now
            )
        
        # 4. Phase transition to danger zone -> IMMEDIATE
        previous_phase = self._last_phase.get(asset, "UNKNOWN")
        self._last_phase[asset] = current_phase
        
        if (current_phase in self.DANGER_PHASES and 
            previous_phase in self.SAFE_PHASES):
            logger.warning(f"[PATCH6] IMMEDIATE EXIT: Phase {previous_phase} -> {current_phase}")
            return ForcedExitSignal(
                should_exit=True,
                urgency=ExitUrgency.IMMEDIATE,
                reason=f"Phase transition: {previous_phase} -> {current_phase}",
                target_exposure=0.0,
                execution_mode="AGGRESSIVE",
                triggered_at=now
            )
        
        # 5. Dominance TTL expired -> IMMEDIATE
        if asset in self._active_dominance and not dominance_active:
            del self._active_dominance[asset]
            logger.warning(f"[PATCH6] IMMEDIATE EXIT: Dominance TTL expired")
            return ForcedExitSignal(
                should_exit=True,
                urgency=ExitUrgency.IMMEDIATE,
                reason="Dominance TTL expired",
                target_exposure=0.0,
                execution_mode="PASSIVE_PREFERRED",  # Less urgent
                triggered_at=now
            )
        
        # === CHECK SCHEDULED EXIT CONDITIONS ===
        
        # 6. TTL approaching (1 bar left) -> SCHEDULED
        if dominance_active and dominance_ttl == 1:
            return ForcedExitSignal(
                should_exit=True,
                urgency=ExitUrgency.SCHEDULED,
                reason="Dominance TTL: 1 bar remaining",
                target_exposure=current_exposure * 0.5,  # Reduce 50%
                execution_mode="PASSIVE_PREFERRED",
                triggered_at=now
            )
        
        # No exit needed
        return ForcedExitSignal(should_exit=False)
    
    def get_execution_loop_signal(
        self,
        asset: str,
        dominance_active: bool,
        dominance_ttl: int,
        current_phase: str,
        vpin: float,
        correlation: float,
        price_move_pct: float,
        current_exposure: float,
    ) -> Optional[Dict]:
        """
        Get signal for execution loop (200ms frequency).
        
        Patch 6: Returns non-None if IMMEDIATE action needed.
        This bypasses the 4H decision frequency.
        """
        
        exit_signal = self.check_forced_exit(
            asset, dominance_active, dominance_ttl, current_phase,
            vpin, correlation, price_move_pct, current_exposure
        )
        
        return exit_signal.to_execution_loop_signal()
    
    def reset(self, asset: str):
        """Reset state for asset."""
        self._active_dominance.pop(asset, None)
        self._last_phase.pop(asset, None)


# =============================================================================
# SINGLETON ACCESSORS
# =============================================================================

_risk_veto_classifier: Optional[RiskVetoClassifier] = None
_deadlock_resolver: Optional[TrancheAwareDeadlockResolver] = None
_sol_forced_exit: Optional[SOLDominanceForcedExit] = None


def get_risk_veto_classifier(weekend_config: Optional[Dict] = None) -> RiskVetoClassifier:
    """Get singleton RiskVetoClassifier.

    Refresh the singleton when weekend_config changes so profile overlays do not
    silently keep the first constructor inputs.
    """
    global _risk_veto_classifier
    if _risk_veto_classifier is None:
        _risk_veto_classifier = RiskVetoClassifier(weekend_config=weekend_config)
    elif weekend_config is not None and getattr(_risk_veto_classifier, "_weekend_config", {}) != (weekend_config or {}):
        logger.info("[RiskVetoClassifier] weekend_config changed; refreshing singleton instance")
        _risk_veto_classifier = RiskVetoClassifier(weekend_config=weekend_config)
    return _risk_veto_classifier


def get_deadlock_resolver() -> TrancheAwareDeadlockResolver:
    """Get singleton TrancheAwareDeadlockResolver."""
    global _deadlock_resolver
    if _deadlock_resolver is None:
        _deadlock_resolver = TrancheAwareDeadlockResolver()
    return _deadlock_resolver


def get_sol_forced_exit() -> SOLDominanceForcedExit:
    """Get singleton SOLDominanceForcedExit."""
    global _sol_forced_exit
    if _sol_forced_exit is None:
        _sol_forced_exit = SOLDominanceForcedExit()
    return _sol_forced_exit


def get_call_chain() -> CallChainProof:
    """Get singleton CallChainProof."""
    return _call_chain


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

@dataclass
class ProofLogParams:
    """
    Parameters for creating a structured proof log.
    
    Use this dataclass instead of passing 20+ individual arguments to create_proof_log().
    
    Example:
        params = ProofLogParams(
            data_real=True,
            quant_real=True,
            mode="NORMAL",
            direction=1.0
        )
        log = create_proof_log_from_params(params)
    """
    # Component statuses
    data_real: bool = False
    quant_real: bool = False
    drl_enabled: bool = False
    dvol_real: bool = False
    sentiment_real: bool = False
    macro_real: bool = False
    
    # Schema
    schema_valid: bool = True
    missing_keys: List[str] = field(default_factory=list)
    
    # Decision
    mode: str = "UNKNOWN"
    direction: float = 0.0
    exposure: float = 0.0
    execution_mode: str = "UNKNOWN"
    
    # Risk
    veto_result: Optional['RiskVetoResult'] = None
    
    # Alpha
    alpha_passed: bool = True
    alpha_est: float = 0.0
    alpha_thresh: float = 0.0
    
    # Tranche
    tranche_action: str = "NONE"
    tranche_level: int = 0
    
    # Deadlock
    deadlock_result: Optional['DeadlockResult'] = None
    
    # SOL exit
    sol_exit_signal: Optional['ForcedExitSignal'] = None


def create_proof_log_from_params(params: ProofLogParams) -> 'StructuredProofLog':
    """Create proof log from a params object."""
    return create_proof_log(
        data_real=params.data_real,
        quant_real=params.quant_real,
        drl_enabled=params.drl_enabled,
        dvol_real=params.dvol_real,
        sentiment_real=params.sentiment_real,
        macro_real=params.macro_real,
        schema_valid=params.schema_valid,
        missing_keys=params.missing_keys,
        mode=params.mode,
        direction=params.direction,
        exposure=params.exposure,
        execution_mode=params.execution_mode,
        veto_result=params.veto_result,
        alpha_passed=params.alpha_passed,
        alpha_est=params.alpha_est,
        alpha_thresh=params.alpha_thresh,
        tranche_action=params.tranche_action,
        tranche_level=params.tranche_level,
        deadlock_result=params.deadlock_result,
        sol_exit_signal=params.sol_exit_signal,
    )


def create_proof_log(
    # Component statuses
    data_real: bool = False,
    quant_real: bool = False,
    drl_enabled: bool = False,
    dvol_real: bool = False,
    sentiment_real: bool = False,
    macro_real: bool = False,
    
    # Schema
    schema_valid: bool = True,
    missing_keys: List[str] = None,
    
    # Decision
    mode: str = "UNKNOWN",
    direction: float = 0.0,
    exposure: float = 0.0,
    execution_mode: str = "UNKNOWN",
    
    # Risk
    veto_result: Optional[RiskVetoResult] = None,
    
    # Alpha
    alpha_passed: bool = True,
    alpha_est: float = 0.0,
    alpha_thresh: float = 0.0,
    
    # Tranche
    tranche_action: str = "NONE",
    tranche_level: int = 0,
    
    # Deadlock
    deadlock_result: Optional[DeadlockResult] = None,
    
    # SOL exit
    sol_exit_signal: Optional[ForcedExitSignal] = None,
) -> StructuredProofLog:
    """Helper to create structured proof log."""
    
    builder = ProofLogBuilder()
    
    # Set component statuses
    builder.set_components(
        data=ComponentStatus.REAL if data_real else ComponentStatus.DISABLED,
        quant=ComponentStatus.REAL if quant_real else ComponentStatus.DISABLED,
        drl=ComponentStatus.REAL if drl_enabled else ComponentStatus.DISABLED,
        dvol=ComponentStatus.REAL if dvol_real else ComponentStatus.DISABLED,
        sentiment=ComponentStatus.REAL if sentiment_real else ComponentStatus.DISABLED,
        macro=ComponentStatus.REAL if macro_real else ComponentStatus.DISABLED,
    )
    
    builder.set_schema(schema_valid, missing_keys or [])
    builder.set_decision(mode, direction, exposure)
    builder.set_execution(execution_mode, tranche_action, tranche_level)
    
    if veto_result:
        builder.set_veto(
            veto_result.veto_type.value,
            [c.name for c in veto_result.hard_conditions + veto_result.soft_conditions],
            veto_result.exposure_cap
        )
    else:
        builder.set_veto("NONE", [], 1.0)
    
    builder.set_alpha(alpha_passed, alpha_est, alpha_thresh)
    
    if deadlock_result:
        builder.set_deadlock(
            deadlock_result.deadlock_bars,
            deadlock_result.resolution.value
        )
    else:
        builder.set_deadlock(0, "NONE")
    
    if sol_exit_signal and sol_exit_signal.should_exit:
        builder.set_sol_exit(True, sol_exit_signal.urgency.value)
    else:
        builder.set_sol_exit(False, "NONE")
    
    builder.set_call_chain(_call_chain.to_proof_log())
    
    return builder.build()

"""
================================================================================
HMATS v3.6.1 - PRODUCTION RELIABILITY PATCHES (Patch-Only Diffs)
================================================================================
Version: 3.6.1-PROD-PATCHES
Purpose: Minimal patches for production reliability (applied via import hooks)

PATCHES:
    Patch 1: Canonical schema key mapping + required-key validation + proof log
    Patch 2: Single canonical entrypoint lock + version banner + call-chain proof
    Patch 3: Structured [PROOF] log per 4H tick with REAL/DISABLED status
    Patch 4: HARD vs SOFT risk veto split (SOFT caps in OPPORTUNITY, HARD blocks)
    Patch 5: Execution deadlock + tranche stage coupling (T1->FORCE, T2+->ABORT)
    Patch 6: SOL dominance forced exit via execution loop (200ms, not 4H)

USAGE:
    from integration.core.production_reliability_patches import apply_all_patches
    apply_all_patches()
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Callable, Any
from datetime import datetime
from enum import Enum, auto
from functools import wraps

logger = logging.getLogger(__name__)

PATCH_VERSION = "3.6.1-PROD-PATCHES"

# =============================================================================
# PATCH 1: CANONICAL SCHEMA KEY MAPPING + REQUIRED-KEY VALIDATION + PROOF LOG
# =============================================================================

class SchemaValidationLevel(Enum):
    """Validation level for schema keys."""
    REQUIRED = "REQUIRED"    # Missing -> always NO_TRADE
    CRITICAL = "CRITICAL"    # Missing -> NO_TRADE in NORMAL, warning in OPPORTUNITY
    OPTIONAL = "OPTIONAL"    # Missing -> use default, log warning


@dataclass
class SchemaKeySpec:
    """Specification for a schema key."""
    name: str
    level: SchemaValidationLevel
    data_type: type
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    aliases: List[str] = field(default_factory=list)


# Canonical schema specification (Patch 1)
CANONICAL_SCHEMA_SPEC: List[SchemaKeySpec] = [
    # REQUIRED keys (missing -> always NO_TRADE)
    SchemaKeySpec("current_price", SchemaValidationLevel.REQUIRED, float, min_value=0.0001, max_value=1e9),
    SchemaKeySpec("volume_24h", SchemaValidationLevel.REQUIRED, float, min_value=0),
    SchemaKeySpec("data_age_seconds", SchemaValidationLevel.REQUIRED, float, min_value=0, max_value=10.0, aliases=["age_seconds", "staleness"]),
    
    # CRITICAL keys (missing -> NO_TRADE in NORMAL, warning in OPPORTUNITY)
    SchemaKeySpec("dvol_zscore", SchemaValidationLevel.CRITICAL, float, default=0.0, aliases=["dvol", "dvol_z"]),
    SchemaKeySpec("vpin", SchemaValidationLevel.CRITICAL, float, default=0.5, min_value=0, max_value=1),
    SchemaKeySpec("correlation_btc_eth_sol", SchemaValidationLevel.CRITICAL, float, default=0.0, min_value=-1, max_value=1, aliases=["correlation"]),
    SchemaKeySpec("orderbook_depth_1pct_usd", SchemaValidationLevel.CRITICAL, float, default=float('inf'), min_value=0, aliases=["depth_1pct", "orderbook_depth"]),
    
    # OPTIONAL keys (missing -> use default, log warning)
    SchemaKeySpec("binance_price", SchemaValidationLevel.OPTIONAL, float, default=0.0),
    SchemaKeySpec("kraken_price", SchemaValidationLevel.OPTIONAL, float, default=0.0),
    SchemaKeySpec("spread_bps", SchemaValidationLevel.OPTIONAL, float, default=10.0, min_value=0),
    SchemaKeySpec("structure_break_pct", SchemaValidationLevel.OPTIONAL, float, default=0.0),
    SchemaKeySpec("volume_ratio", SchemaValidationLevel.OPTIONAL, float, default=1.0, min_value=0),
    SchemaKeySpec("orderbook_imbalance", SchemaValidationLevel.OPTIONAL, float, default=0.0, min_value=-1, max_value=1),
    SchemaKeySpec("dvol_1h_change", SchemaValidationLevel.OPTIONAL, float, default=0.0),
    SchemaKeySpec("estimated_friction_bps", SchemaValidationLevel.OPTIONAL, float, default=15.0, min_value=0),
    SchemaKeySpec("price_move_pct", SchemaValidationLevel.OPTIONAL, float, default=0.0),
]


@dataclass
class SchemaValidationProof:
    """Proof log for schema validation (Patch 1)."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_valid: bool = True
    mode_at_validation: str = "UNKNOWN"
    
    # Missing keys by level
    missing_required: List[str] = field(default_factory=list)
    missing_critical: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    
    # Applied defaults
    defaults_applied: Dict[str, Any] = field(default_factory=dict)
    
    # Resolved aliases
    aliases_resolved: Dict[str, str] = field(default_factory=dict)
    
    # Validation outcome
    triggers_no_trade: bool = False
    no_trade_reason: str = ""
    
    def to_proof_log(self) -> str:
        """Generate [PROOF] log line for schema validation."""
        return (
            f"[PROOF:SCHEMA] {self.timestamp.isoformat()} | "
            f"valid={self.is_valid} | mode={self.mode_at_validation} | "
            f"missing_required={len(self.missing_required)} | "
            f"missing_critical={len(self.missing_critical)} | "
            f"missing_optional={len(self.missing_optional)} | "
            f"aliases_resolved={len(self.aliases_resolved)} | "
            f"defaults_applied={len(self.defaults_applied)} | "
            f"NO_TRADE={self.triggers_no_trade}"
            + (f" | reason={self.no_trade_reason}" if self.no_trade_reason else "")
        )


class CanonicalSchemaValidator:
    """
    Enhanced schema validator with proof logging (Patch 1).
    
    CRITICAL: Missing required keys MUST trigger NO_TRADE + proof log.
    """
    
    def __init__(self, spec: List[SchemaKeySpec] = None):
        self.spec = spec or CANONICAL_SCHEMA_SPEC
        self._build_alias_map()
    
    def _build_alias_map(self):
        """Build alias -> canonical key mapping."""
        self.alias_map: Dict[str, str] = {}
        for key_spec in self.spec:
            for alias in key_spec.aliases:
                self.alias_map[alias] = key_spec.name
    
    def validate(self, data: Dict, mode: str = "NORMAL") -> tuple:
        """
        Validate data against canonical schema.
        
        Returns:
            (canonical_data, proof) where:
            - canonical_data: Dict with aliases resolved and defaults applied
            - proof: SchemaValidationProof with validation details
        """
        proof = SchemaValidationProof(mode_at_validation=mode)
        canonical_data = {}
        
        # Step 1: Resolve aliases
        for key, value in data.items():
            canonical_key = self.alias_map.get(key, key)
            if canonical_key != key:
                proof.aliases_resolved[key] = canonical_key
            canonical_data[canonical_key] = value
        
        # Step 2: Check each key in spec
        for key_spec in self.spec:
            if key_spec.name not in canonical_data:
                # Key is missing
                if key_spec.level == SchemaValidationLevel.REQUIRED:
                    proof.missing_required.append(key_spec.name)
                elif key_spec.level == SchemaValidationLevel.CRITICAL:
                    proof.missing_critical.append(key_spec.name)
                    # Apply default
                    if key_spec.default is not None:
                        canonical_data[key_spec.name] = key_spec.default
                        proof.defaults_applied[key_spec.name] = key_spec.default
                else:  # OPTIONAL
                    proof.missing_optional.append(key_spec.name)
                    # Apply default
                    if key_spec.default is not None:
                        canonical_data[key_spec.name] = key_spec.default
                        proof.defaults_applied[key_spec.name] = key_spec.default
        
        # Step 3: Determine if NO_TRADE should trigger
        if proof.missing_required:
            proof.is_valid = False
            proof.triggers_no_trade = True
            proof.no_trade_reason = f"MISSING_REQUIRED: {proof.missing_required}"
        elif proof.missing_critical and mode == "NORMAL":
            proof.is_valid = False
            proof.triggers_no_trade = True
            proof.no_trade_reason = f"MISSING_CRITICAL_IN_NORMAL: {proof.missing_critical}"
        
        # Log proof
        logger.info(proof.to_proof_log())
        
        return canonical_data, proof


# Singleton
_schema_validator: Optional[CanonicalSchemaValidator] = None

def get_schema_validator() -> CanonicalSchemaValidator:
    """Get singleton schema validator."""
    global _schema_validator
    if _schema_validator is None:
        _schema_validator = CanonicalSchemaValidator()
    return _schema_validator


# =============================================================================
# PATCH 2: SINGLE CANONICAL ENTRYPOINT + VERSION BANNER + CALL-CHAIN PROOF
# =============================================================================

class EntrypointLock:
    """
    Entrypoint lock that ensures only one canonical entrypoint exists (Patch 2).
    """
    
    _canonical: Optional[str] = None
    _legacy_redirects: Dict[str, str] = {}
    _call_chain: List[Dict] = []
    _banner_printed: bool = False
    
    VERSION_BANNER = """
================================================================================
    HMATS v{version} - Hierarchical Multi-Agent Trading System
    Build: {build}
    Entrypoint: {entrypoint}
    Timestamp: {timestamp}
================================================================================
"""
    
    @classmethod
    def set_canonical(cls, name: str, version: str = "3.6.1-PROD"):
        """Lock the canonical entrypoint."""
        if cls._canonical is not None and cls._canonical != name:
            raise RuntimeError(
                f"Canonical entrypoint already locked to '{cls._canonical}'. "
                f"Cannot change to '{name}'."
            )
        cls._canonical = name
        logger.info(f"[ENTRYPOINT] Canonical entrypoint LOCKED: {name}")
    
    @classmethod
    def register_legacy(cls, legacy_name: str, redirect_to: str):
        """Register a legacy entrypoint that redirects to canonical."""
        cls._legacy_redirects[legacy_name] = redirect_to
        logger.info(f"[ENTRYPOINT] Legacy redirect registered: {legacy_name} -> {redirect_to}")
    
    @classmethod
    def print_banner(cls, version: str, entrypoint: str):
        """Print version banner (only once)."""
        if not cls._banner_printed:
            print(cls.VERSION_BANNER.format(
                version=version,
                build="Production Reliability Patches Applied",
                entrypoint=entrypoint,
                timestamp=datetime.utcnow().isoformat()
            ))
            cls._banner_printed = True
    
    @classmethod
    def record_call(cls, component: str, action: str):
        """Record a call in the call chain."""
        cls._call_chain.append({
            "component": component,
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        })
        # Keep only last 10 calls
        if len(cls._call_chain) > 10:
            cls._call_chain = cls._call_chain[-10:]
    
    @classmethod
    def get_call_chain_proof(cls) -> str:
        """Get call chain as proof string."""
        if not cls._call_chain:
            return "CallChain=EMPTY"
        chain = " -> ".join(f"{c['component']}:{c['action']}" for c in cls._call_chain[-5:])
        return f"CallChain=[{chain}]"
    
    @classmethod
    def reset(cls):
        """Reset for testing."""
        cls._canonical = None
        cls._legacy_redirects = {}
        cls._call_chain = []
        cls._banner_printed = False


def canonical_entrypoint_patch(version: str = "3.6.1-PROD"):
    """
    Decorator that marks and locks a function as the canonical entrypoint (Patch 2).
    
    Usage:
        @canonical_entrypoint_patch()
        def decide(self, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Lock entrypoint
            EntrypointLock.set_canonical(func.__name__, version)
            
            # Print banner on first call
            EntrypointLock.print_banner(version, func.__name__)
            
            # Record call
            EntrypointLock.record_call("ENTRYPOINT", func.__name__)
            
            # Log entry
            logger.info(f"[PROOF:ENTRY] v{version} | {func.__name__} | {datetime.utcnow().isoformat()} | {EntrypointLock.get_call_chain_proof()}")
            
            return func(*args, **kwargs)
        
        wrapper._is_canonical_entrypoint = True
        wrapper._entrypoint_version = version
        return wrapper
    return decorator


def legacy_redirect_patch(canonical_func_name: str):
    """
    Decorator that redirects legacy entrypoints to canonical (Patch 2).
    
    Usage:
        @legacy_redirect_patch("decide")
        def old_decide(self, ...):
            return self.decide(...)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Register redirect
            EntrypointLock.register_legacy(func.__name__, canonical_func_name)
            
            # Log redirect
            logger.warning(f"[LEGACY_REDIRECT] {func.__name__} -> {canonical_func_name}")
            EntrypointLock.record_call("LEGACY", f"{func.__name__}->{canonical_func_name}")
            
            # Call canonical
            canonical_func = getattr(self, canonical_func_name)
            return canonical_func(*args, **kwargs)
        
        wrapper._is_legacy_entrypoint = True
        wrapper._redirects_to = canonical_func_name
        return wrapper
    return decorator


# =============================================================================
# PATCH 3: STRUCTURED [PROOF] LOG PER 4H TICK
# =============================================================================

class ComponentStatusPatch(Enum):
    """Component status for proof log (Patch 3)."""
    REAL = "REAL"           # Live data/signals
    DISABLED = "DISABLED"   # Component disabled
    MOCK = "MOCK"           # Mock/simulated
    ERROR = "ERROR"         # Component error
    STALE = "STALE"         # Data is stale


@dataclass
class TickProofLog:
    """
    Structured proof log per 4H tick (Patch 3).
    
    Lists REAL/DISABLED status for:
    - Data, Quant, DRL, DVOL, Sentiment, Macro
    - Mode, Intent, Execution
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: str = PATCH_VERSION
    
    # Component statuses (REAL/DISABLED)
    data_status: ComponentStatusPatch = ComponentStatusPatch.DISABLED
    quant_status: ComponentStatusPatch = ComponentStatusPatch.DISABLED
    drl_status: ComponentStatusPatch = ComponentStatusPatch.DISABLED
    dvol_status: ComponentStatusPatch = ComponentStatusPatch.DISABLED
    sentiment_status: ComponentStatusPatch = ComponentStatusPatch.DISABLED
    macro_status: ComponentStatusPatch = ComponentStatusPatch.DISABLED
    
    # Decision state
    mode: str = "UNKNOWN"
    intent_direction: float = 0.0
    intent_exposure: float = 0.0
    
    # Execution state
    execution_mode: str = "UNKNOWN"
    tranche_action: str = "NONE"
    tranche_level: int = 0
    
    # Risk veto (Patch 4)
    veto_type: str = "NONE"  # HARD, SOFT, NONE
    veto_reason: str = ""
    exposure_cap: float = 1.0
    
    # Alpha gate
    alpha_passed: bool = True
    alpha_est_bps: float = 0.0
    alpha_thresh_bps: float = 0.0
    
    # Deadlock (Patch 5)
    deadlock_bars: int = 0
    deadlock_resolution: str = "NONE"
    tranche_bias: str = "NEUTRAL"  # FORCE, ABORT, NEUTRAL
    
    # SOL forced exit (Patch 6)
    sol_forced_exit: bool = False
    sol_exit_urgency: str = "NONE"  # NONE, SCHEDULED, IMMEDIATE
    
    # Schema (Patch 1)
    schema_valid: bool = True
    schema_missing_count: int = 0
    
    # Call chain (Patch 2)
    call_chain: str = ""
    
    def to_log_line(self) -> str:
        """Generate single-line [PROOF] log."""
        return (
            f"[PROOF] {self.timestamp.isoformat()} | v{self.version} | "
            f"Data={self.data_status.value} | Quant={self.quant_status.value} | "
            f"DRL={self.drl_status.value} | DVOL={self.dvol_status.value} | "
            f"Sentiment={self.sentiment_status.value} | Macro={self.macro_status.value} | "
            f"Mode={self.mode} | "
            f"Intent={{dir={self.intent_direction:+.2f},exp={self.intent_exposure:.1%}}} | "
            f"Exec={self.execution_mode} | "
            f"Veto={{type={self.veto_type},cap={self.exposure_cap:.0%}}} | "
            f"Alpha={{pass={self.alpha_passed},est={self.alpha_est_bps:.0f},thresh={self.alpha_thresh_bps:.0f}}} | "
            f"Tranche={{action={self.tranche_action},level={self.tranche_level},bias={self.tranche_bias}}} | "
            f"Deadlock={{bars={self.deadlock_bars},res={self.deadlock_resolution}}} | "
            f"SOL={{exit={self.sol_forced_exit},urgency={self.sol_exit_urgency}}} | "
            f"Schema={{valid={self.schema_valid},missing={self.schema_missing_count}}} | "
            f"{self.call_chain}"
        )


class TickProofLogBuilder:
    """Builder for TickProofLog (Patch 3)."""
    
    def __init__(self):
        self._log = TickProofLog()
    
    def set_components(
        self,
        data: ComponentStatusPatch = ComponentStatusPatch.DISABLED,
        quant: ComponentStatusPatch = ComponentStatusPatch.DISABLED,
        drl: ComponentStatusPatch = ComponentStatusPatch.DISABLED,
        dvol: ComponentStatusPatch = ComponentStatusPatch.DISABLED,
        sentiment: ComponentStatusPatch = ComponentStatusPatch.DISABLED,
        macro: ComponentStatusPatch = ComponentStatusPatch.DISABLED,
    ) -> 'TickProofLogBuilder':
        self._log.data_status = data
        self._log.quant_status = quant
        self._log.drl_status = drl
        self._log.dvol_status = dvol
        self._log.sentiment_status = sentiment
        self._log.macro_status = macro
        return self
    
    def set_decision(self, mode: str, direction: float, exposure: float) -> 'TickProofLogBuilder':
        self._log.mode = mode
        self._log.intent_direction = direction
        self._log.intent_exposure = exposure
        return self
    
    def set_execution(self, mode: str, tranche_action: str, tranche_level: int) -> 'TickProofLogBuilder':
        self._log.execution_mode = mode
        self._log.tranche_action = tranche_action
        self._log.tranche_level = tranche_level
        return self
    
    def set_veto(self, veto_type: str, reason: str, cap: float) -> 'TickProofLogBuilder':
        self._log.veto_type = veto_type
        self._log.veto_reason = reason
        self._log.exposure_cap = cap
        return self
    
    def set_alpha(self, passed: bool, est_bps: float, thresh_bps: float) -> 'TickProofLogBuilder':
        self._log.alpha_passed = passed
        self._log.alpha_est_bps = est_bps
        self._log.alpha_thresh_bps = thresh_bps
        return self
    
    def set_deadlock(self, bars: int, resolution: str, bias: str) -> 'TickProofLogBuilder':
        self._log.deadlock_bars = bars
        self._log.deadlock_resolution = resolution
        self._log.tranche_bias = bias
        return self
    
    def set_sol_exit(self, forced: bool, urgency: str) -> 'TickProofLogBuilder':
        self._log.sol_forced_exit = forced
        self._log.sol_exit_urgency = urgency
        return self
    
    def set_schema(self, valid: bool, missing_count: int) -> 'TickProofLogBuilder':
        self._log.schema_valid = valid
        self._log.schema_missing_count = missing_count
        return self
    
    def set_call_chain(self, chain: str) -> 'TickProofLogBuilder':
        self._log.call_chain = chain
        return self
    
    def build(self) -> TickProofLog:
        self._log.timestamp = datetime.utcnow()
        self._log.call_chain = EntrypointLock.get_call_chain_proof()
        return self._log


# =============================================================================
# PATCH 4: HARD vs SOFT RISK VETO CLASSIFICATION
# =============================================================================

class HardVetoConditionPatch(Enum):
    """HARD veto conditions (always block, regardless of mode)."""
    DRAWDOWN_HALT = auto()         # DD >= 10%
    CORRELATION_CRISIS = auto()     # Correlation >= 0.95
    EXTREME_DVOL = auto()           # DVOL z-score >= 5.0
    FLASH_CRASH = auto()            # >= 5% in 5 minutes
    DATA_INTEGRITY_FAIL = auto()    # CRC/schema fail
    EXECUTION_BLOCKED = auto()      # 3+ consecutive failures
    ALL_CONFLICT_FLAT = auto()      # All signals in direct conflict
    MISSING_REQUIRED_KEYS = auto()  # Schema: missing required


class SoftVetoConditionPatch(Enum):
    """SOFT veto conditions (cap in OPPORTUNITY, block in NORMAL)."""
    ELEVATED_DVOL = auto()          # DVOL z-score 3.0-4.9
    LIQUIDITY_WARNING = auto()      # Depth $100K-$300K
    CORRELATION_ELEVATED = auto()   # Correlation 0.85-0.94
    PARTIAL_CONFLICT = auto()       # 2/3 signals conflict
    WEEKEND_LIQUIDITY = auto()      # Weekend reduced liquidity
    MISSING_CRITICAL_KEYS = auto()  # Schema: missing critical


@dataclass
class VetoClassificationResult:
    """Result of HARD vs SOFT veto classification (Patch 4)."""
    veto_type: str = "NONE"  # HARD, SOFT, NONE
    hard_conditions: List[HardVetoConditionPatch] = field(default_factory=list)
    soft_conditions: List[SoftVetoConditionPatch] = field(default_factory=list)
    exposure_cap: float = 1.0  # For SOFT, max allowed exposure
    allows_trade: bool = True
    reason: str = ""
    
    def to_proof_str(self) -> str:
        hard_str = ",".join(c.name for c in self.hard_conditions) or "NONE"
        soft_str = ",".join(c.name for c in self.soft_conditions) or "NONE"
        return f"Veto={{type={self.veto_type}|HARD=[{hard_str}]|SOFT=[{soft_str}]|cap={self.exposure_cap:.0%}}}"


class HardSoftVetoClassifier:
    """
    HARD vs SOFT risk veto classifier (Patch 4).

    CRITICAL RULE:
        - In OPPORTUNITY mode, SOFT may cap exposure but MUST NOT become HARD
          unless hard conditions are actually met.
        - HARD veto ALWAYS blocks, regardless of mode.

    Accepts optional weekend_config dict to override weekend soft veto cap.
    """

    # HARD thresholds (non-negotiable)
    HARD_THRESHOLDS = {
        "drawdown": 0.10,           # 10% halt
        "correlation": 0.95,        # Crisis
        "dvol_zscore": 5.0,         # Extreme
        "flash_crash_pct": 0.05,    # 5% in 5 min
        "execution_failures": 3,
    }

    # SOFT thresholds
    SOFT_THRESHOLDS = {
        "dvol_min": 3.0,
        "dvol_max": 4.9,
        "correlation_min": 0.85,
        "correlation_max": 0.94,
        "liquidity_min": 100_000,
        "liquidity_max": 300_000,
    }

    # Default exposure caps for SOFT conditions
    _DEFAULT_SOFT_CAPS = {
        SoftVetoConditionPatch.ELEVATED_DVOL: 0.50,
        SoftVetoConditionPatch.LIQUIDITY_WARNING: 0.60,
        SoftVetoConditionPatch.CORRELATION_ELEVATED: 0.70,
        SoftVetoConditionPatch.PARTIAL_CONFLICT: 0.50,
        SoftVetoConditionPatch.WEEKEND_LIQUIDITY: 0.40,
        SoftVetoConditionPatch.MISSING_CRITICAL_KEYS: 0.30,
    }

    def __init__(self, weekend_config: Optional[Dict] = None):
        self.SOFT_CAPS = dict(self._DEFAULT_SOFT_CAPS)
        self._weekend_config = weekend_config or {}

        weekend_cap = self._weekend_config.get("weekend_soft_veto_cap")
        if weekend_cap is not None:
            self.SOFT_CAPS[SoftVetoConditionPatch.WEEKEND_LIQUIDITY] = float(weekend_cap)
    
    def classify(
        self,
        mode: str,
        drawdown: float = 0.0,
        dvol_zscore: float = 0.0,
        correlation: float = 0.0,
        liquidity_usd: float = float('inf'),
        signal_conflict_score: float = 0.0,
        data_valid: bool = True,
        execution_failures: int = 0,
        flash_crash_active: bool = False,
        is_weekend: bool = False,
        missing_required: List[str] = None,
        missing_critical: List[str] = None,
    ) -> VetoClassificationResult:
        """
        Classify risk conditions into HARD vs SOFT.
        
        Patch 4: SOFT in OPPORTUNITY caps but DOES NOT BLOCK.
        """
        hard = []
        soft = []
        
        # === HARD CONDITIONS (always block) ===
        
        if drawdown >= self.HARD_THRESHOLDS["drawdown"]:
            hard.append(HardVetoConditionPatch.DRAWDOWN_HALT)
        
        if correlation >= self.HARD_THRESHOLDS["correlation"]:
            hard.append(HardVetoConditionPatch.CORRELATION_CRISIS)
        
        if dvol_zscore >= self.HARD_THRESHOLDS["dvol_zscore"]:
            hard.append(HardVetoConditionPatch.EXTREME_DVOL)
        
        if not data_valid:
            hard.append(HardVetoConditionPatch.DATA_INTEGRITY_FAIL)
        
        if execution_failures >= self.HARD_THRESHOLDS["execution_failures"]:
            hard.append(HardVetoConditionPatch.EXECUTION_BLOCKED)
        
        if flash_crash_active:
            hard.append(HardVetoConditionPatch.FLASH_CRASH)
        
        if signal_conflict_score >= 1.0:
            hard.append(HardVetoConditionPatch.ALL_CONFLICT_FLAT)
        
        if missing_required:
            hard.append(HardVetoConditionPatch.MISSING_REQUIRED_KEYS)
        
        # === SOFT CONDITIONS (only if no HARD) ===
        
        if not hard:
            if self.SOFT_THRESHOLDS["dvol_min"] <= dvol_zscore < self.SOFT_THRESHOLDS["dvol_max"]:
                soft.append(SoftVetoConditionPatch.ELEVATED_DVOL)
            
            if self.SOFT_THRESHOLDS["liquidity_min"] <= liquidity_usd < self.SOFT_THRESHOLDS["liquidity_max"]:
                soft.append(SoftVetoConditionPatch.LIQUIDITY_WARNING)
            
            if self.SOFT_THRESHOLDS["correlation_min"] <= correlation < self.SOFT_THRESHOLDS["correlation_max"]:
                soft.append(SoftVetoConditionPatch.CORRELATION_ELEVATED)
            
            if 0.5 <= signal_conflict_score < 1.0:
                soft.append(SoftVetoConditionPatch.PARTIAL_CONFLICT)
            
            if is_weekend:
                soft.append(SoftVetoConditionPatch.WEEKEND_LIQUIDITY)
            
            if missing_critical:
                soft.append(SoftVetoConditionPatch.MISSING_CRITICAL_KEYS)
        
        # === DETERMINE RESULT ===
        
        if hard:
            return VetoClassificationResult(
                veto_type="HARD",
                hard_conditions=hard,
                exposure_cap=0.0,
                allows_trade=False,
                reason=f"HARD: {[c.name for c in hard]}"
            )
        
        if soft:
            min_cap = min(self.SOFT_CAPS.get(c, 1.0) for c in soft)
            
            # Patch 4 CRITICAL: In OPPORTUNITY, SOFT only caps, does NOT block
            if mode == "OPPORTUNITY":
                return VetoClassificationResult(
                    veto_type="SOFT",
                    soft_conditions=soft,
                    exposure_cap=min_cap,
                    allows_trade=True,  # ALLOW with cap
                    reason=f"SOFT (OPPORTUNITY): cap={min_cap:.0%}"
                )
            else:
                # In NORMAL, SOFT blocks
                return VetoClassificationResult(
                    veto_type="SOFT",
                    soft_conditions=soft,
                    exposure_cap=0.0,
                    allows_trade=False,
                    reason=f"SOFT (NORMAL): blocks"
                )
        
        return VetoClassificationResult()


# Singleton
_veto_classifier: Optional[HardSoftVetoClassifier] = None

def get_veto_classifier(weekend_config: Optional[Dict] = None) -> HardSoftVetoClassifier:
    global _veto_classifier
    if _veto_classifier is None:
        _veto_classifier = HardSoftVetoClassifier(weekend_config=weekend_config)
    return _veto_classifier


# =============================================================================
# PATCH 5: EXECUTION DEADLOCK + TRANCHE STAGE COUPLING
# =============================================================================

class TrancheBias(Enum):
    """Deadlock resolution bias based on tranche stage (Patch 5)."""
    FORCE = "FORCE"      # T1: bias toward FORCE_AGGRESSIVE
    ABORT = "ABORT"      # T2+: bias toward ABORT_OPPORTUNITY
    NEUTRAL = "NEUTRAL"  # No bias


@dataclass
class TrancheDeadlockResult:
    """Result of tranche-coupled deadlock resolution (Patch 5)."""
    resolution: str = "CONTINUE"  # CONTINUE, FORCE_AGGRESSIVE, ABORT_OPPORTUNITY
    bias: TrancheBias = TrancheBias.NEUTRAL
    tranche_level: int = 0
    deadlock_bars: int = 0
    edge_decay_pct: float = 0.0
    reason: str = ""


class TrancheDeadlockResolver:
    """
    Execution deadlock resolver coupled with tranche stage (Patch 5).
    
    RULE:
        - T1 (20% entry): Bias toward FORCE_AGGRESSIVE (small position, can afford aggression)
        - T2+ (30%+ entry): Bias toward ABORT_OPPORTUNITY (protect larger position)
    """
    
    # Thresholds per tranche
    T1_FORCE_BARS = 2      # T1: force after 2 bars
    T2_PLUS_ABORT_BARS = 3  # T2+: abort after 3 bars
    
    # Edge decay
    EDGE_DECAY_PER_BAR = 0.15  # 15% per bar
    MIN_EDGE_FOR_FORCE = 1.5   # edge/friction ratio
    
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
    ) -> TrancheDeadlockResult:
        """
        Resolve deadlock with tranche-aware bias (Patch 5).
        
        T1 -> bias FORCE, T2+ -> bias ABORT.
        """
        
        # Track deadlock
        if is_stuck:
            self._counters[asset] = self._counters.get(asset, 0) + 1
            if asset not in self._initial_edges:
                self._initial_edges[asset] = current_edge_bps
        else:
            self._counters[asset] = 0
            self._initial_edges.pop(asset, None)
        
        bars = self._counters.get(asset, 0)
        
        if bars == 0:
            return TrancheDeadlockResult(tranche_level=tranche_level)
        
        # Calculate decayed edge
        decay = max(0.1, 1.0 - self.EDGE_DECAY_PER_BAR * bars)
        decayed_edge = current_edge_bps * decay
        edge_ratio = decayed_edge / friction_bps if friction_bps > 0 else 0
        
        # Determine bias
        if tranche_level <= 1:
            bias = TrancheBias.FORCE
            threshold = self.T1_FORCE_BARS
        else:
            bias = TrancheBias.ABORT
            threshold = self.T2_PLUS_ABORT_BARS
        
        # Not at threshold yet
        if bars < threshold:
            return TrancheDeadlockResult(
                resolution="CONTINUE",
                bias=bias,
                tranche_level=tranche_level,
                deadlock_bars=bars,
                edge_decay_pct=(1.0 - decay) * 100,
                reason=f"Deadlock {bars}/{threshold}, waiting"
            )
        
        # At threshold - apply bias
        if bias == TrancheBias.FORCE:
            if edge_ratio >= self.MIN_EDGE_FOR_FORCE:
                return TrancheDeadlockResult(
                    resolution="FORCE_AGGRESSIVE",
                    bias=bias,
                    tranche_level=tranche_level,
                    deadlock_bars=bars,
                    edge_decay_pct=(1.0 - decay) * 100,
                    reason=f"T1 FORCE: edge/friction={edge_ratio:.1f}x"
                )
            else:
                return TrancheDeadlockResult(
                    resolution="ABORT_OPPORTUNITY",
                    bias=bias,
                    tranche_level=tranche_level,
                    deadlock_bars=bars,
                    edge_decay_pct=(1.0 - decay) * 100,
                    reason=f"T1 ABORT: edge degraded to {edge_ratio:.1f}x"
                )
        else:
            # T2+: protect capital
            return TrancheDeadlockResult(
                resolution="ABORT_OPPORTUNITY",
                bias=bias,
                tranche_level=tranche_level,
                deadlock_bars=bars,
                edge_decay_pct=(1.0 - decay) * 100,
                reason=f"T{tranche_level} ABORT: protecting position"
            )
    
    def reset(self, asset: str):
        self._counters.pop(asset, None)
        self._initial_edges.pop(asset, None)


# Singleton
_deadlock_resolver: Optional[TrancheDeadlockResolver] = None

def get_deadlock_resolver_patch() -> TrancheDeadlockResolver:
    global _deadlock_resolver
    if _deadlock_resolver is None:
        _deadlock_resolver = TrancheDeadlockResolver()
    return _deadlock_resolver


# =============================================================================
# PATCH 6: SOL DOMINANCE FORCED EXIT VIA EXECUTION LOOP (200ms)
# =============================================================================

class ExitUrgencyPatch(Enum):
    """Exit urgency level (Patch 6)."""
    NONE = "NONE"               # No exit
    SCHEDULED = "SCHEDULED"     # Exit at next 4H bar
    IMMEDIATE = "IMMEDIATE"     # Exit NOW via 200ms execution loop


@dataclass
class SOLForcedExitSignal:
    """
    SOL dominance forced exit signal (Patch 6).
    
    IMMEDIATE urgency bypasses 4H wait and triggers via execution loop.
    """
    should_exit: bool = False
    urgency: ExitUrgencyPatch = ExitUrgencyPatch.NONE
    reason: str = ""
    target_exposure: float = 0.0
    execution_mode: str = "AGGRESSIVE"
    triggered_at: Optional[datetime] = None
    
    def to_execution_loop_signal(self) -> Optional[Dict]:
        """Convert to signal for 200ms execution loop."""
        if self.urgency != ExitUrgencyPatch.IMMEDIATE:
            return None
        
        return {
            "action": "FORCED_EXIT",
            "target_exposure": self.target_exposure,
            "execution_mode": self.execution_mode,
            "reason": self.reason,
            "urgency": "IMMEDIATE",
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
        }


class SOLDominanceForcedExitPatch:
    """
    SOL dominance forced exit with immediate execution loop capability (Patch 6).
    
    IMMEDIATE triggers (bypass 4H wait, trigger via 200ms loop):
        - TTL expired
        - Phase transition to SATURATION/EXHAUSTION/CRASH
        - Correlation spike >= 0.92
        - VPIN spike >= 0.90
        - Flash move >= 3%
    
    SCHEDULED triggers (wait for 4H):
        - TTL approaching (1 bar)
        - Signal weakening
    """
    
    # IMMEDIATE thresholds
    IMMEDIATE_VPIN = 0.90
    IMMEDIATE_CORRELATION = 0.92
    IMMEDIATE_FLASH_MOVE = 0.03  # 3%
    
    DANGER_PHASES = {"SATURATION", "EXHAUSTION", "CRASH"}
    SAFE_PHASES = {"IGNITION", "EXPANSION", "ACCUMULATION"}
    
    def __init__(self):
        self._active: Dict[str, datetime] = {}
        self._last_phase: Dict[str, str] = {}
        self._execution_loop_signals: List[Dict] = []
    
    def check_forced_exit(
        self,
        asset: str,
        dominance_active: bool,
        dominance_ttl: int,
        current_phase: str,
        vpin: float,
        correlation: float,
        price_move_pct: float,
        current_exposure: float,
    ) -> SOLForcedExitSignal:
        """Check if forced exit should trigger."""
        
        now = datetime.utcnow()
        
        # No position -> no exit
        if current_exposure <= 0.01:
            return SOLForcedExitSignal()
        
        # Track dominance
        if dominance_active and asset not in self._active:
            self._active[asset] = now
        
        # === IMMEDIATE TRIGGERS ===
        
        # VPIN spike
        if vpin >= self.IMMEDIATE_VPIN:
            signal = SOLForcedExitSignal(
                should_exit=True,
                urgency=ExitUrgencyPatch.IMMEDIATE,
                reason=f"VPIN spike: {vpin:.2f}",
                target_exposure=0.0,
                execution_mode="AGGRESSIVE",
                triggered_at=now
            )
            self._add_to_execution_loop(signal)
            return signal
        
        # Correlation spike
        if correlation >= self.IMMEDIATE_CORRELATION:
            signal = SOLForcedExitSignal(
                should_exit=True,
                urgency=ExitUrgencyPatch.IMMEDIATE,
                reason=f"Correlation spike: {correlation:.2f}",
                target_exposure=0.0,
                execution_mode="AGGRESSIVE",
                triggered_at=now
            )
            self._add_to_execution_loop(signal)
            return signal
        
        # Flash move
        if abs(price_move_pct) >= self.IMMEDIATE_FLASH_MOVE:
            signal = SOLForcedExitSignal(
                should_exit=True,
                urgency=ExitUrgencyPatch.IMMEDIATE,
                reason=f"Flash move: {price_move_pct:.1%}",
                target_exposure=0.0,
                execution_mode="AGGRESSIVE",
                triggered_at=now
            )
            self._add_to_execution_loop(signal)
            return signal
        
        # Phase transition to danger
        prev_phase = self._last_phase.get(asset, "UNKNOWN")
        self._last_phase[asset] = current_phase
        
        if current_phase in self.DANGER_PHASES and prev_phase in self.SAFE_PHASES:
            signal = SOLForcedExitSignal(
                should_exit=True,
                urgency=ExitUrgencyPatch.IMMEDIATE,
                reason=f"Phase transition: {prev_phase} -> {current_phase}",
                target_exposure=0.0,
                execution_mode="AGGRESSIVE",
                triggered_at=now
            )
            self._add_to_execution_loop(signal)
            return signal
        
        # TTL expired
        if asset in self._active and not dominance_active:
            del self._active[asset]
            signal = SOLForcedExitSignal(
                should_exit=True,
                urgency=ExitUrgencyPatch.IMMEDIATE,
                reason="Dominance TTL expired",
                target_exposure=0.0,
                execution_mode="PASSIVE_PREFERRED",
                triggered_at=now
            )
            self._add_to_execution_loop(signal)
            return signal
        
        # === SCHEDULED TRIGGERS ===
        
        # TTL approaching
        if dominance_active and dominance_ttl == 1:
            return SOLForcedExitSignal(
                should_exit=True,
                urgency=ExitUrgencyPatch.SCHEDULED,
                reason="Dominance TTL: 1 bar remaining",
                target_exposure=current_exposure * 0.5,
                execution_mode="PASSIVE_PREFERRED",
                triggered_at=now
            )
        
        return SOLForcedExitSignal()
    
    def _add_to_execution_loop(self, signal: SOLForcedExitSignal):
        """Add signal to execution loop queue."""
        exec_signal = signal.to_execution_loop_signal()
        if exec_signal:
            self._execution_loop_signals.append(exec_signal)
            logger.warning(f"[PATCH6] Added IMMEDIATE exit to execution loop: {exec_signal['reason']}")
    
    def get_execution_loop_signals(self) -> List[Dict]:
        """Get and clear pending execution loop signals."""
        signals = self._execution_loop_signals.copy()
        self._execution_loop_signals = []
        return signals
    
    def reset(self, asset: str):
        self._active.pop(asset, None)
        self._last_phase.pop(asset, None)


# Singleton
_sol_forced_exit: Optional[SOLDominanceForcedExitPatch] = None

def get_sol_forced_exit_patch() -> SOLDominanceForcedExitPatch:
    global _sol_forced_exit
    if _sol_forced_exit is None:
        _sol_forced_exit = SOLDominanceForcedExitPatch()
    return _sol_forced_exit


# =============================================================================
# EXECUTION LOOP HOOK (Patch 6)
# =============================================================================

class ExecutionLoopHook:
    """
    Hook into execution loop for immediate actions (Patch 6).
    
    This allows SOL forced exit to trigger at 200ms frequency,
    not waiting for 4H decision.
    """
    
    _callbacks: List[Callable] = []
    _last_check: Optional[datetime] = None
    CHECK_INTERVAL_MS = 200
    
    @classmethod
    def register_callback(cls, callback: Callable):
        """Register a callback to be called every 200ms."""
        cls._callbacks.append(callback)
        logger.info(f"[PATCH6] Registered execution loop callback: {callback.__name__}")
    
    @classmethod
    def tick(cls) -> List[Dict]:
        """
        Called every 200ms by execution loop.
        Returns list of immediate actions to execute.
        """
        now = datetime.utcnow()
        
        if cls._last_check:
            elapsed_ms = (now - cls._last_check).total_seconds() * 1000
            if elapsed_ms < cls.CHECK_INTERVAL_MS:
                return []
        
        cls._last_check = now
        
        actions = []
        for callback in cls._callbacks:
            try:
                result = callback()
                if result:
                    actions.extend(result if isinstance(result, list) else [result])
            except Exception as e:
                logger.error(f"[PATCH6] Execution loop callback error: {e}")
        
        return actions
    
    @classmethod
    def reset(cls):
        cls._callbacks = []
        cls._last_check = None


# Register SOL forced exit with execution loop
def _sol_exit_callback() -> List[Dict]:
    """Callback for SOL forced exit signals."""
    return get_sol_forced_exit_patch().get_execution_loop_signals()


ExecutionLoopHook.register_callback(_sol_exit_callback)


# =============================================================================
# APPLY ALL PATCHES
# =============================================================================

def apply_all_patches():
    """
    Apply all production reliability patches.
    
    This initializes singletons and registers hooks.
    """
    logger.info(f"[PATCHES] Applying all production reliability patches (v{PATCH_VERSION})")
    
    # Patch 1: Schema validator
    get_schema_validator()
    logger.info("[PATCH1] Canonical schema validator initialized")
    
    # Patch 2: Entrypoint lock
    EntrypointLock.reset()
    logger.info("[PATCH2] Entrypoint lock reset")
    
    # Patch 3: Proof log builder (no init needed)
    logger.info("[PATCH3] Structured proof log ready")
    
    # Patch 4: Veto classifier
    get_veto_classifier()
    logger.info("[PATCH4] HARD/SOFT veto classifier initialized")
    
    # Patch 5: Deadlock resolver
    get_deadlock_resolver_patch()
    logger.info("[PATCH5] Tranche-aware deadlock resolver initialized")
    
    # Patch 6: SOL forced exit
    get_sol_forced_exit_patch()
    logger.info("[PATCH6] SOL forced exit with execution loop initialized")
    
    logger.info(f"[PATCHES] All patches applied successfully")
    
    return {
        "version": PATCH_VERSION,
        "patches": [
            "Patch1: Canonical schema + required-key validation + proof log",
            "Patch2: Single canonical entrypoint + version banner + call chain",
            "Patch3: Structured [PROOF] log per 4H tick",
            "Patch4: HARD vs SOFT risk veto split",
            "Patch5: Execution deadlock + tranche coupling",
            "Patch6: SOL dominance forced exit via execution loop",
        ]
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Patch 1
    "SchemaValidationLevel",
    "SchemaKeySpec",
    "SchemaValidationProof",
    "CanonicalSchemaValidator",
    "get_schema_validator",
    "CANONICAL_SCHEMA_SPEC",
    
    # Patch 2
    "EntrypointLock",
    "canonical_entrypoint_patch",
    "legacy_redirect_patch",
    
    # Patch 3
    "ComponentStatusPatch",
    "TickProofLog",
    "TickProofLogBuilder",
    
    # Patch 4
    "HardVetoConditionPatch",
    "SoftVetoConditionPatch",
    "VetoClassificationResult",
    "HardSoftVetoClassifier",
    "get_veto_classifier",
    
    # Patch 5
    "TrancheBias",
    "TrancheDeadlockResult",
    "TrancheDeadlockResolver",
    "get_deadlock_resolver_patch",
    
    # Patch 6
    "ExitUrgencyPatch",
    "SOLForcedExitSignal",
    "SOLDominanceForcedExitPatch",
    "get_sol_forced_exit_patch",
    "ExecutionLoopHook",
    
    # Apply
    "apply_all_patches",
    "PATCH_VERSION",
]

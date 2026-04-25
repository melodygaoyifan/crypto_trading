"""
================================================================================
HMATS v3.4 - HUMAN OVERRIDE PROTOCOL
================================================================================
Version: 3.4.0
Type: META-LAYER (not an agent, does not generate signals)
Purpose: Allow controlled human overrides with safety constraints

FUNCTION:
    - Allow aggressiveness bias adjustments
    - Allow mode suppression (e.g., disable OPPORTUNITY)
    - All overrides are TIME-LIMITED and LOGGED
    - Overrides CANNOT bypass hard risk vetoes

ALLOWED OVERRIDES:
    1. Aggressiveness Bias (-50% to +30%)
    2. Mode Suppression (disable OPPORTUNITY temporarily)
    3. Asset Restriction (reduce exposure to specific assets)
    4. Execution Mode Force (force PASSIVE_ONLY)

PROHIBITED (CANNOT OVERRIDE):
    - Hard risk vetoes (10% drawdown, correlation crisis)
    - Stop loss levels
    - Maximum position limits
    - Data integrity checks

CRITICAL CONSTRAINTS:
    - All overrides expire automatically (max 24 hours)
    - All overrides are logged with timestamp and reason
    - Multiple active overrides stack according to rules
    - System must explicitly check override status
    - Override removal requires explicit action OR expiry

WHY THIS EXISTS:
    Even well-designed systems need occasional human intervention.
    This provides a CONTROLLED way to do so without breaking safety.
    Example: After major news event, human may want to reduce
    aggressiveness temporarily while system adapts.
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class OverrideType(Enum):
    """Types of human overrides."""
    AGGRESSIVENESS_BIAS = "AGGRESSIVENESS_BIAS"
    MODE_SUPPRESSION = "MODE_SUPPRESSION"
    ASSET_RESTRICTION = "ASSET_RESTRICTION"
    EXECUTION_MODE_FORCE = "EXECUTION_MODE_FORCE"
    TRANCHE_LIMIT = "TRANCHE_LIMIT"


class OverrideStatus(Enum):
    """Override status."""
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


@dataclass
class HumanOverride:
    """A single human override."""
    
    override_id: str
    override_type: OverrideType
    
    # Value depends on type:
    # AGGRESSIVENESS_BIAS: float (-0.5 to +0.3)
    # MODE_SUPPRESSION: str (mode name to suppress)
    # ASSET_RESTRICTION: Dict[str, float] (asset -> max exposure)
    # EXECUTION_MODE_FORCE: str (execution mode to force)
    # TRANCHE_LIMIT: int (max tranche level)
    value: any
    
    # Timing
    # [P50 2026-04-25] datetime.utcnow() returns naive; downstream comparisons
    # use timezone-aware datetimes. After restart with persisted override state,
    # comparisons raised TypeError silently. Use aware default to match.
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    duration_hours: float = 4.0  # Default 4 hours
    
    # Metadata
    reason: str = ""
    created_by: str = "human"
    
    # Status
    status: OverrideStatus = OverrideStatus.ACTIVE
    cancelled_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(hours=self.duration_hours)
    
    @property
    def is_active(self) -> bool:
        if self.status != OverrideStatus.ACTIVE:
            return False
        if datetime.now(timezone.utc) > self.expires_at:
            self.status = OverrideStatus.EXPIRED
            return False
        return True
    
    @property
    def remaining_hours(self) -> float:
        if not self.is_active:
            return 0.0
        remaining = (self.expires_at - datetime.now(timezone.utc)).total_seconds() / 3600
        return max(0.0, remaining)
    
    def to_dict(self) -> Dict:
        return {
            "override_id": self.override_id,
            "type": self.override_type.value,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "reason": self.reason,
            "status": self.status.value,
            "remaining_hours": self.remaining_hours,
        }


@dataclass
class OverrideEffects:
    """Combined effects of all active overrides."""
    
    # Aggressiveness
    aggressiveness_multiplier: float = 1.0
    
    # Mode suppression
    suppressed_modes: Set[str] = field(default_factory=set)
    
    # Asset restrictions
    asset_max_exposure: Dict[str, float] = field(default_factory=dict)
    
    # Execution
    forced_execution_mode: Optional[str] = None
    
    # Tranche
    max_tranche_level: int = 4
    
    # Flags
    any_override_active: bool = False
    override_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "aggressiveness_multiplier": self.aggressiveness_multiplier,
            "suppressed_modes": list(self.suppressed_modes),
            "asset_restrictions": self.asset_max_exposure,
            "forced_execution_mode": self.forced_execution_mode,
            "max_tranche": self.max_tranche_level,
            "any_active": self.any_override_active,
            "count": self.override_count,
        }


@dataclass
class OverrideConfig:
    """Configuration for override limits."""
    
    # Aggressiveness bounds
    min_aggressiveness_bias: float = -0.5  # Max 50% reduction
    max_aggressiveness_bias: float = 0.3   # Max 30% increase
    
    # Duration limits
    max_override_hours: float = 24.0       # Max 24 hours
    default_override_hours: float = 4.0    # Default 4 hours
    
    # Stacking rules
    allow_multiple_same_type: bool = False  # Only one of each type
    
    # Asset restriction limits
    min_asset_exposure: float = 0.0        # Can restrict to 0%
    
    # Tranche limits
    min_tranche_limit: int = 1             # Must allow at least T1


# Protected items that CANNOT be overridden
PROTECTED_ITEMS = [
    "hard_drawdown_stop",
    "correlation_crisis_veto",
    "data_integrity_check",
    "stop_loss_levels",
    "maximum_position_limit",
]


class HumanOverrideProtocol:
    """
    Human override protocol for controlled manual intervention.
    
    USAGE:
        protocol = HumanOverrideProtocol()
        
        # Add an override
        override_id = protocol.add_override(
            override_type=OverrideType.AGGRESSIVENESS_BIAS,
            value=-0.3,  # Reduce aggressiveness by 30%
            duration_hours=4,
            reason="Post-CPI volatility expected",
        )
        
        # Get combined effects
        effects = protocol.get_effects()
        
        # Apply in system
        effective_aggressiveness = base * effects.aggressiveness_multiplier
        
        # Cancel override
        protocol.cancel_override(override_id)
    
    CRITICAL SAFETY:
        - Overrides CANNOT bypass hard risk vetoes
        - All overrides are time-limited
        - All overrides are logged
    """
    
    def __init__(self, config: Optional[OverrideConfig] = None):
        self.config = config or OverrideConfig()
        
        # Active overrides
        self._overrides: Dict[str, HumanOverride] = {}
        
        # Override log (persisted)
        self._log: List[Dict] = []
        
        # Counter for IDs
        self._counter: int = 0
        
        logger.info("HumanOverrideProtocol initialized")

    def check(self):
        """Pre-tick check: expire stale overrides and log active ones."""
        self._cleanup_expired()
        active = [o for o in self._overrides.values() if o.is_active]
        if active:
            logger.info(f"[HUMAN_OVERRIDE] {len(active)} active override(s)")

    def add_override(
        self,
        override_type: OverrideType,
        value: any,
        duration_hours: float = None,
        reason: str = "",
        created_by: str = "human",
    ) -> str:
        """
        Add a new override.
        
        Args:
            override_type: Type of override
            value: Override value (type-dependent)
            duration_hours: How long override lasts
            reason: Reason for override (REQUIRED for audit)
            created_by: Who created the override
        
        Returns:
            Override ID
        
        Raises:
            ValueError: If override violates constraints
        """
        
        # Validate
        self._validate_override(override_type, value)
        
        # Check for existing same-type override
        if not self.config.allow_multiple_same_type:
            for existing in self._overrides.values():
                if existing.override_type == override_type and existing.is_active:
                    # Cancel existing
                    self.cancel_override(existing.override_id, "Replaced by new override")
        
        # Create override
        self._counter += 1
        override_id = f"override_{self._counter}_{override_type.value}"
        
        duration = duration_hours or self.config.default_override_hours
        duration = min(duration, self.config.max_override_hours)
        
        override = HumanOverride(
            override_id=override_id,
            override_type=override_type,
            value=value,
            duration_hours=duration,
            reason=reason,
            created_by=created_by,
        )
        
        self._overrides[override_id] = override
        
        # Log
        self._log_event("OVERRIDE_ADDED", override)
        
        logger.warning(
            f"HUMAN OVERRIDE ADDED: {override_type.value}={value}, "
            f"duration={duration:.1f}h, reason='{reason}'"
        )
        
        return override_id
    
    def _validate_override(self, override_type: OverrideType, value: any):
        """Validate override before adding."""
        
        if override_type == OverrideType.AGGRESSIVENESS_BIAS:
            if not isinstance(value, (int, float)):
                raise ValueError("Aggressiveness bias must be a number")
            if value < self.config.min_aggressiveness_bias:
                raise ValueError(f"Aggressiveness bias cannot be below {self.config.min_aggressiveness_bias}")
            if value > self.config.max_aggressiveness_bias:
                raise ValueError(f"Aggressiveness bias cannot exceed {self.config.max_aggressiveness_bias}")
        
        elif override_type == OverrideType.MODE_SUPPRESSION:
            if value not in ["OPPORTUNITY", "NORMAL"]:
                raise ValueError("Can only suppress OPPORTUNITY or NORMAL modes")
            # Cannot suppress NO_TRADE (that would bypass safety)
        
        elif override_type == OverrideType.ASSET_RESTRICTION:
            if not isinstance(value, dict):
                raise ValueError("Asset restriction must be a dict {asset: max_exposure}")
            for asset, exposure in value.items():
                if exposure < self.config.min_asset_exposure:
                    raise ValueError(f"Asset exposure cannot be below {self.config.min_asset_exposure}")
        
        elif override_type == OverrideType.TRANCHE_LIMIT:
            if not isinstance(value, int):
                raise ValueError("Tranche limit must be an integer")
            if value < self.config.min_tranche_limit:
                raise ValueError(f"Tranche limit cannot be below {self.config.min_tranche_limit}")
    
    def cancel_override(
        self,
        override_id: str,
        reason: str = "Manual cancellation",
    ) -> bool:
        """
        Cancel an override.
        
        Returns True if override was found and cancelled.
        """
        
        if override_id not in self._overrides:
            return False
        
        override = self._overrides[override_id]
        
        if override.status != OverrideStatus.ACTIVE:
            return False
        
        override.status = OverrideStatus.CANCELLED
        override.cancelled_at = datetime.now(timezone.utc)
        
        # Log
        self._log_event("OVERRIDE_CANCELLED", override, reason)
        
        logger.info(f"Override cancelled: {override_id}, reason='{reason}'")
        
        return True
    
    def get_effects(self) -> OverrideEffects:
        """
        Get combined effects of all active overrides.
        
        WHERE CONSUMED:
            - authority_fusion_v34.py: Mode suppression, aggressiveness
            - execution_timing_v34.py: Forced execution mode
            - tranche_manager: Tranche limits
            - risk_agent_v34.py: Asset restrictions
        
        CRITICAL: Effects are ADDITIVE for aggressiveness,
        UNION for suppression, MIN for limits.
        """
        
        # Clean expired overrides
        self._cleanup_expired()
        
        effects = OverrideEffects()
        
        active_overrides = [o for o in self._overrides.values() if o.is_active]
        
        if not active_overrides:
            return effects
        
        effects.any_override_active = True
        effects.override_count = len(active_overrides)
        
        for override in active_overrides:
            if override.override_type == OverrideType.AGGRESSIVENESS_BIAS:
                # Additive bias
                effects.aggressiveness_multiplier += override.value
            
            elif override.override_type == OverrideType.MODE_SUPPRESSION:
                effects.suppressed_modes.add(override.value)
            
            elif override.override_type == OverrideType.ASSET_RESTRICTION:
                for asset, max_exp in override.value.items():
                    current = effects.asset_max_exposure.get(asset, 1.0)
                    effects.asset_max_exposure[asset] = min(current, max_exp)
            
            elif override.override_type == OverrideType.EXECUTION_MODE_FORCE:
                effects.forced_execution_mode = override.value
            
            elif override.override_type == OverrideType.TRANCHE_LIMIT:
                effects.max_tranche_level = min(effects.max_tranche_level, override.value)
        
        # Bound aggressiveness
        effects.aggressiveness_multiplier = max(
            1.0 + self.config.min_aggressiveness_bias,  # 0.5 minimum
            min(
                1.0 + self.config.max_aggressiveness_bias,  # 1.3 maximum
                effects.aggressiveness_multiplier
            )
        )
        
        return effects
    
    def _cleanup_expired(self):
        """Clean up expired overrides."""
        
        for override in self._overrides.values():
            if override.status == OverrideStatus.ACTIVE:
                if datetime.now(timezone.utc) > override.expires_at:
                    override.status = OverrideStatus.EXPIRED
                    self._log_event("OVERRIDE_EXPIRED", override)
                    logger.info(f"Override expired: {override.override_id}")
    
    def _log_event(self, event_type: str, override: HumanOverride, extra: str = ""):
        """Log an override event."""
        
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "override": override.to_dict(),
            "extra": extra,
        }
        
        self._log.append(entry)
    
    def get_active_overrides(self) -> List[HumanOverride]:
        """Get list of active overrides."""
        self._cleanup_expired()
        return [o for o in self._overrides.values() if o.is_active]
    
    def get_override(self, override_id: str) -> Optional[HumanOverride]:
        """Get a specific override."""
        return self._overrides.get(override_id)
    
    def get_log(self, last_n: int = 100) -> List[Dict]:
        """Get override log."""
        return self._log[-last_n:]
    
    def is_mode_suppressed(self, mode: str) -> bool:
        """Check if a mode is suppressed."""
        effects = self.get_effects()
        return mode in effects.suppressed_modes
    
    def cancel_all(self, reason: str = "Bulk cancellation"):
        """Cancel all active overrides."""
        
        for override_id in list(self._overrides.keys()):
            self.cancel_override(override_id, reason)
        
        logger.warning(f"All overrides cancelled: {reason}")
    
    def reset(self):
        """Full reset."""
        self._overrides.clear()
        self._log.clear()
        self._counter = 0


# ============================================================================
# SAFETY CHECK INTEGRATION
# ============================================================================

def check_override_safety(effects: OverrideEffects, system_state: Dict) -> Tuple[bool, str]:
    """
    Verify that overrides don't bypass safety.
    
    This should be called by Risk Agent to ensure human overrides
    don't compromise system safety.
    
    ALWAYS PROTECTED (overrides CANNOT affect):
        - Hard drawdown stops
        - Correlation crisis vetoes
        - Data integrity checks
        - Stop loss levels
    """
    
    # Check if system is in crisis state
    drawdown = system_state.get("drawdown", 0.0)
    correlation = system_state.get("correlation", 0.0)
    data_valid = system_state.get("data_valid", True)
    
    # Hard drawdown - CANNOT be overridden [UNLEASH UL-5: synced 20%]
    if drawdown >= 0.20:
        if effects.aggressiveness_multiplier > 0.0:
            return False, "Override cannot bypass 20% drawdown halt"

    # Correlation crisis - CANNOT be overridden [UNLEASH UL-4: synced 0.98]
    if correlation >= 0.98:
        if effects.aggressiveness_multiplier > 0.0:
            return False, "Override cannot bypass correlation crisis"
    
    # Data integrity - CANNOT be overridden
    if not data_valid:
        return False, "Override cannot bypass data integrity check"
    
    return True, "Override effects validated"


# Singleton
_override_protocol: Optional[HumanOverrideProtocol] = None

def get_override_protocol() -> HumanOverrideProtocol:
    global _override_protocol
    if _override_protocol is None:
        _override_protocol = HumanOverrideProtocol()
    return _override_protocol

def reset_override_protocol():
    global _override_protocol
    _override_protocol = None

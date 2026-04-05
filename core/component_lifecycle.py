"""
HMATS v3.2 - Generic Component Lifecycle
Pattern: DISABLED -> SHADOW -> ADVISORY -> ACTIVE
With auto-demotion safety net.

[v3.2-B15] Generalizes DRL PromotionGate for any new component.
Usage:
    lifecycle = ComponentLifecycle("PassiveAggressive")
    lifecycle.promote("SHADOW")  # Start observing
    ...
    if lifecycle.is_active():
        # Use component output for real decisions
    elif lifecycle.is_shadow():
        # Log what would happen
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class LifecycleLevel(Enum):
    DISABLED = "DISABLED"    # Completely off
    SHADOW = "SHADOW"        # Runs + logs, zero decision impact
    ADVISORY = "ADVISORY"    # Output in Decision Log, not in fusion
    ACTIVE = "ACTIVE"        # Full authority


@dataclass
class DemotionConfig:
    enable: bool = True
    consecutive_errors_threshold: int = 5
    demote_to: str = "SHADOW"
    recovery_period_hours: int = 24


class ComponentLifecycle:
    VALID_LEVELS = {"DISABLED", "SHADOW", "ADVISORY", "ACTIVE"}
    PROMOTION_ORDER = ["DISABLED", "SHADOW", "ADVISORY", "ACTIVE"]

    def __init__(self, name: str, demotion_config: Optional[DemotionConfig] = None):
        self.name = name
        self._level = LifecycleLevel.DISABLED
        self._consecutive_errors = 0
        self._demoted_at: Optional[datetime] = None
        self._promoted_at: Dict[str, datetime] = {}
        self.demotion_config = demotion_config or DemotionConfig()
        logger.info(f"[Lifecycle:{name}] Initialized at DISABLED")

    @property
    def level(self) -> LifecycleLevel:
        return self._level

    def is_disabled(self) -> bool:
        return self._level == LifecycleLevel.DISABLED

    def is_shadow(self) -> bool:
        return self._level == LifecycleLevel.SHADOW

    def is_advisory(self) -> bool:
        return self._level == LifecycleLevel.ADVISORY

    def is_active(self) -> bool:
        return self._level == LifecycleLevel.ACTIVE

    def promote(self, target: str):
        if target not in self.VALID_LEVELS:
            raise ValueError(f"Invalid level: {target}")
        old = self._level.value
        self._level = LifecycleLevel(target)
        self._promoted_at[target] = datetime.now(timezone.utc)
        self._consecutive_errors = 0
        logger.info(f"[Lifecycle:{self.name}] {old} -> {target}")

    def record_error(self):
        self._consecutive_errors += 1
        if (self.demotion_config.enable and
            self._consecutive_errors >= self.demotion_config.consecutive_errors_threshold):
            self._demote(f"consecutive_errors={self._consecutive_errors}")

    def record_success(self):
        self._consecutive_errors = 0

    def _demote(self, reason: str):
        old = self._level.value
        self._level = LifecycleLevel(self.demotion_config.demote_to)
        self._demoted_at = datetime.now(timezone.utc)
        logger.warning(f"[Lifecycle:{self.name}] AUTO-DEMOTE {old} -> {self._level.value}: {reason}")

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "level": self._level.value,
            "consecutive_errors": self._consecutive_errors,
            "demoted_at": self._demoted_at.isoformat() if self._demoted_at else None,
            "history": {k: v.isoformat() for k, v in self._promoted_at.items()},
        }

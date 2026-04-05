"""Compatibility authority-gate wrapper for DRL lifecycle control.

This bridges legacy `drl.authority_gate` imports to the maintained
`drl.promotion_gate.DRLPromotionGate` implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from .promotion_gate import DRLPromotionGate

logger = logging.getLogger(__name__)

from core.canonical_enums import DRLAuthorityLevel


@dataclass
class PromotionGateConfig:
    """Config for underlying promotion gate behavior."""

    enable: bool = True
    consecutive_loss_threshold: int = 5
    drawdown_trigger_pct: float = 0.15
    demote_to: str = "EXIT_ONLY"
    recovery_period_days: int = 3
    max_demotions_before_disable: int = 3
    demotion_window_days: int = 14
    state_file: str = "data/drl_promotion_state.json"


@dataclass
class FailClosedConfig:
    """Fail-closed policy placeholder for backward compatibility."""

    enabled: bool = True
    default_level: str = "DISABLED"


def _to_level(value: Union[str, Enum, DRLAuthorityLevel]) -> DRLAuthorityLevel:
    if isinstance(value, DRLAuthorityLevel):
        return value
    if isinstance(value, Enum):
        value = value.value
    value = str(value).upper()
    try:
        return DRLAuthorityLevel(value)
    except Exception:
        return DRLAuthorityLevel.DISABLED


class DRLAuthorityGate:
    """Adapter exposing legacy authority-gate API."""

    def __init__(
        self,
        authority_level: Union[DRLAuthorityLevel, str] = DRLAuthorityLevel.SHADOW,
        promotion_config: Optional[PromotionGateConfig] = None,
        fail_closed_config: Optional[FailClosedConfig] = None,
    ):
        self.promotion_config = promotion_config or PromotionGateConfig()
        self.fail_closed_config = fail_closed_config or FailClosedConfig()

        self._gate = DRLPromotionGate(
            config={
                "enable": self.promotion_config.enable,
                "consecutive_loss_threshold": self.promotion_config.consecutive_loss_threshold,
                "drawdown_trigger_pct": self.promotion_config.drawdown_trigger_pct,
                "demote_to": self.promotion_config.demote_to,
                "recovery_period_days": self.promotion_config.recovery_period_days,
                "max_demotions_before_disable": self.promotion_config.max_demotions_before_disable,
                "demotion_window_days": self.promotion_config.demotion_window_days,
            },
            state_file=self.promotion_config.state_file,
        )

        requested = _to_level(authority_level)
        current = self.get_authority()
        if current == DRLAuthorityLevel.DISABLED and requested != DRLAuthorityLevel.DISABLED:
            # Keep legacy startup behavior: if gate is fresh, start from requested
            # level (usually SHADOW) instead of hard-disabled.
            self.promote(requested)

    def get_authority(self) -> DRLAuthorityLevel:
        return _to_level(self._gate.get_authority_level())

    def has_exit_authority(self) -> bool:
        return self.get_authority() in (DRLAuthorityLevel.EXIT_ONLY, DRLAuthorityLevel.ACTIVE)

    def is_shadow_mode(self) -> bool:
        return self.get_authority() == DRLAuthorityLevel.SHADOW

    def check_promotion(self) -> dict:
        return self._gate.check_promotion()

    def get_status(self) -> dict:
        status = self._gate.get_status()
        status["authority_level"] = self.get_authority().value
        return status

    def promote(self, level: Union[DRLAuthorityLevel, str]) -> None:
        lvl = _to_level(level)
        self._gate.promote(lvl.value)

    def record_trade(self, pnl: float, drl_contributed: bool = True) -> None:
        self._gate.record_trade(pnl=pnl, drl_contributed=drl_contributed)

    def print_banner(self) -> None:
        level = self.get_authority().value
        print(
            "\n============================================================\n"
            f"DRL_AUTHORITY_MODE={level}\n"
            f"   Exit authority: {'ON' if self.has_exit_authority() else 'OFF'}\n"
            "============================================================\n"
        )


_drl_authority_gate: Optional[DRLAuthorityGate] = None


def get_drl_authority_gate(
    authority_level: Union[DRLAuthorityLevel, str] = DRLAuthorityLevel.SHADOW,
    promotion_config: Optional[PromotionGateConfig] = None,
    fail_closed_config: Optional[FailClosedConfig] = None,
) -> DRLAuthorityGate:
    """Get singleton authority gate."""
    global _drl_authority_gate
    if _drl_authority_gate is None:
        _drl_authority_gate = DRLAuthorityGate(
            authority_level=authority_level,
            promotion_config=promotion_config,
            fail_closed_config=fail_closed_config,
        )
    return _drl_authority_gate


def reset_drl_authority_gate() -> None:
    """Reset singleton authority gate (test utility)."""
    global _drl_authority_gate
    _drl_authority_gate = None


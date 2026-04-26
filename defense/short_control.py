"""
================================================================================
HMATS v6.7 - P5-01 Short Control (Profile-Gated)
================================================================================
Profile-gated override for short position constraints.

ULTRA_AGGRESSIVE_5Y:
  - max_short_exposure: 1.00 (up from 0.30)
  - allow_short_in_bull: True (override bull-regime blocks)
  - Squeeze protection: PRESERVED (squeeze_risk > 0.60 -> don't override)
  - Funding extreme: PRESERVED (funding_rate > 0.003 -> don't override)

HIGH_RISK (disabled):
  - No overrides - existing P0/V6 short blocks apply as-is

Architecture:
  Runs AFTER P0 SHORT BLOCK + V6 SHORT FILTER.
  If both blocked shorts due to bull-regime reasons AND no squeeze/funding
  extreme is active -> un-veto and restore exposure.
  If exposure was capped below ULTRA max -> re-cap to ULTRA max.
  Does NOT bypass RiskGovernor/LeverageGuard (those run later).
================================================================================
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ShortControlConfig:
    """Configuration for short control."""
    enabled: bool = False
    max_short_exposure: float = 0.30
    allow_short_in_bull: bool = False
    squeeze_protection_enabled: bool = True
    squeeze_override_threshold: float = 0.60
    funding_extreme_threshold: float = 0.003  # 0.3% per 8h

    @classmethod
    def from_dict(cls, d: dict) -> "ShortControlConfig":
        return cls(
            enabled=d.get("enabled", False),
            max_short_exposure=d.get("max_short_exposure", 0.30),
            allow_short_in_bull=d.get("allow_short_in_bull", False),
            squeeze_protection_enabled=d.get("squeeze_protection_enabled", True),
            squeeze_override_threshold=d.get("squeeze_override_threshold", 0.60),
            funding_extreme_threshold=d.get("funding_extreme_threshold", 0.003),
        )


@dataclass
class ShortControlResult:
    """Result of short control evaluation."""
    action: str = "PASS"       # PASS / OVERRIDE / SCALE / SKIP
    override_veto: bool = False
    clamped: bool = False
    old_exposure: float = 0.0
    new_exposure: float = 0.0
    max_short_used: float = 0.30
    allow_bull_used: bool = False
    is_bull_regime: bool = False
    squeeze_active: bool = False
    funding_extreme: bool = False
    reason: str = ""


# =============================================================================
# Bull regime detection - same labels used by EnhancedRegimeNavigator
# and regime_tensor.market_regime.name
# =============================================================================

_BULL_REGIMES = frozenset([
    "STRONG_BULL", "PARABOLIC", "BULL_TREND",
    "MOMENTUM_RALLY", "STRONG_TREND", "IGNITION", "EXPANSION",
])


# =============================================================================
# SHORT CONTROL
# =============================================================================

class ShortControl:
    """
    Profile-gated short position controller.

    Runs AFTER P0 SHORT BLOCK + V6 SHORT FILTER.
    When ULTRA: overrides bull-regime-only blocks and raises short exposure cap.
    Squeeze and funding extreme protections always preserved.

    Usage:
        sc = ShortControl(config)
        result = sc.evaluate(
            intent_direction=intent.direction,
            target_exposure=intent.target_exposure,
            is_vetoed=intent.veto_active,
            veto_reason=intent.veto_reason,
            regime=market_data.get('regime_state', ''),
            squeeze_risk=agent_signals.get('squeeze_risk', 0.0),
            funding_rate=market_data.get('funding_rate', 0.0),
        )
        if result.override_veto:
            intent.veto_active = False
            intent.veto_reason = ""
        if result.clamped:
            intent.target_exposure = result.new_exposure
    """

    def __init__(self, config: ShortControlConfig):
        self.config = config

    def evaluate(
        self,
        *,
        intent_direction: float,
        target_exposure: float,
        is_vetoed: bool = False,
        veto_reason: str = "",
        regime: str = "",
        squeeze_risk: float = 0.0,
        funding_rate: float = 0.0,
    ) -> ShortControlResult:
        """
        Evaluate short control overrides.

        Args:
            intent_direction: Decided direction (-1 = short)
            target_exposure: Current target_exposure (positive magnitude)
            is_vetoed: Whether intent is currently vetoed
            veto_reason: Current veto reason string
            regime: Current regime name (from GMM or navigator)
            squeeze_risk: Squeeze risk score (0-1)
            funding_rate: Funding rate (e.g. 0.001 = 0.1%/8h)

        Returns:
            ShortControlResult with override/scale decisions
        """
        regime_upper = regime.upper() if regime else ""
        is_bull = regime_upper in _BULL_REGIMES

        result = ShortControlResult(
            old_exposure=target_exposure,
            max_short_used=self.config.max_short_exposure,
            allow_bull_used=self.config.allow_short_in_bull,
            is_bull_regime=is_bull,
        )

        # Only applies to shorts
        if intent_direction >= 0:
            result.action = "SKIP"
            result.reason = "not_short"
            return result

        # Check safety conditions (never overridden)
        if self.config.squeeze_protection_enabled and squeeze_risk >= self.config.squeeze_override_threshold:
            result.squeeze_active = True
        if abs(funding_rate) >= self.config.funding_extreme_threshold:
            result.funding_extreme = True

        # --- Override bull-regime veto ---
        if is_vetoed and self.config.allow_short_in_bull:
            _is_short_veto = (
                "SHORT BLOCK" in veto_reason
                or "SHORT FILTER" in veto_reason
            )
            if _is_short_veto and not result.squeeze_active and not result.funding_extreme:
                result.override_veto = True
                result.action = "OVERRIDE"
                result.reason = f"ULTRA_allow_bull_short: regime={regime_upper}"
                # P84: was logger.info — veto override is a defense-in-depth
                # breach (the SHORT BLOCK was authorized somewhere upstream;
                # this code is choosing to override it). main.py:10109-10113
                # already logs the same override at WARNING for operator
                # visibility. Promoting here too for consistency.
                logger.warning(
                    f"[P5-01] Short veto OVERRIDDEN: regime={regime_upper}, "
                    f"squeeze={squeeze_risk:.2f}, funding={funding_rate:.4%}"
                )
            elif _is_short_veto:
                # Would override but safety conditions active
                result.action = "PASS"
                result.reason = (
                    f"veto_kept: squeeze={'Y' if result.squeeze_active else 'N'}"
                    f" funding={'Y' if result.funding_extreme else 'N'}"
                )

        # --- Exposure cap ---
        if target_exposure > self.config.max_short_exposure:
            result.clamped = True
            result.new_exposure = self.config.max_short_exposure
            if result.action == "OVERRIDE":
                result.action = "OVERRIDE+SCALE"
            elif result.action in ("PASS", "SKIP"):
                result.action = "SCALE"
            result.reason += f" clamped:{target_exposure:.4f}->{self.config.max_short_exposure:.4f}"
        else:
            result.new_exposure = target_exposure

        if result.action == "PASS" and not result.reason:
            result.reason = "no_override_needed"

        return result

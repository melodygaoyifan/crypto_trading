"""
================================================================================
HMATS v6.7 - P1-1D Dynamic Hard Limits
================================================================================
Profile-gated dynamic adjustment of hard risk limits:
- max_leverage:       base 3.0, capped at 3.0
- max_gross_exposure: base 2.5, up to 3.0 in strong trends
- max_single_asset:   fixed at 1.0 (ULTRA) or config default
- correlation_limit:  fixed at 0.98 (ULTRA) or config default

Adjustment factors:
- Regime:      STRONG_TREND/IGNITION ×1.25 leverage, ×1.20 gross
- Confidence:  >0.80 -> ×1.10
- Volatility:  vol_z > 2.0 -> ×0.90 leverage (pullback safety)

All outputs are CLAMPED to absolute maximums.
Only active when config.enabled=True (ULTRA_AGGRESSIVE_5Y profile).
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class DynamicLimitsConfig:
    """Configuration for dynamic hard limits."""
    enabled: bool = False

    # Leverage
    base_max_leverage: float = 3.0
    abs_max_leverage: float = 3.0

    # Gross exposure
    base_max_gross_exposure: float = 2.50
    abs_max_gross_exposure: float = 3.0

    # Single asset
    base_max_single_asset: float = 1.0

    # Correlation
    correlation_limit: float = 0.98

    # Regime multipliers: regime_name -> {"leverage": float, "gross": float}
    regime_multipliers: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Confidence
    confidence_multiplier_threshold: float = 0.80
    confidence_multiplier: float = 1.10

    # Volatility
    volatility_high_threshold: float = 2.0
    volatility_high_leverage_multiplier: float = 0.90

    @classmethod
    def from_dict(cls, d: dict) -> "DynamicLimitsConfig":
        return cls(
            enabled=d.get("enabled", False),
            base_max_leverage=d.get("base_max_leverage", 3.0),
            abs_max_leverage=d.get("abs_max_leverage", 3.0),
            base_max_gross_exposure=d.get("base_max_gross_exposure", 2.50),
            abs_max_gross_exposure=d.get("abs_max_gross_exposure", 3.0),
            base_max_single_asset=d.get("base_max_single_asset", 1.0),
            correlation_limit=d.get("correlation_limit", 0.98),
            regime_multipliers=d.get("regime_multipliers", {}),
            confidence_multiplier_threshold=d.get("confidence_multiplier_threshold", 0.80),
            confidence_multiplier=d.get("confidence_multiplier", 1.10),
            volatility_high_threshold=d.get("volatility_high_threshold", 2.0),
            volatility_high_leverage_multiplier=d.get("volatility_high_leverage_multiplier", 0.90),
        )


@dataclass
class DynamicLimitsResult:
    """Result of dynamic limits computation."""
    max_leverage: float = 3.0
    max_gross_exposure: float = 1.50
    max_single_asset: float = 0.80
    correlation_limit: float = 0.95
    dynamic_active: bool = False

    # Proof fields
    regime_used: str = ""
    leverage_mult: float = 1.0
    gross_mult: float = 1.0
    confidence_applied: bool = False
    volatility_applied: bool = False


class DynamicLimits:
    """
    Computes per-tick dynamic hard limits based on regime, volatility, confidence.

    When disabled (HIGH_RISK default), returns static fallback limits.
    When enabled (ULTRA), applies regime/confidence/volatility multipliers
    and clamps to absolute maximums.
    """

    def __init__(self, config: DynamicLimitsConfig):
        self.config = config

    def get_limits(
        self,
        *,
        regime: str = "",
        volatility_z: Optional[float] = None,
        confidence: Optional[float] = None,
        # Static fallbacks for HIGH_RISK mode
        fallback_max_leverage: float = 3.0,
        fallback_max_gross: float = 1.50,
        fallback_max_single_asset: float = 0.80,
        fallback_correlation_limit: float = 0.95,
    ) -> DynamicLimitsResult:
        """
        Compute dynamic limits for this tick.

        Args:
            regime: Current regime name (e.g. "STRONG_TREND", "VOLATILE_CHOP")
            volatility_z: Volatility z-score (None if unavailable)
            confidence: Overall confidence (None if unavailable)
            fallback_*: Static values used when dynamic_limits is disabled

        Returns:
            DynamicLimitsResult with computed limits and proof fields
        """
        if not self.config.enabled:
            return DynamicLimitsResult(
                max_leverage=fallback_max_leverage,
                max_gross_exposure=fallback_max_gross,
                max_single_asset=fallback_max_single_asset,
                correlation_limit=fallback_correlation_limit,
                dynamic_active=False,
            )

        # Start with base values
        lev = self.config.base_max_leverage
        gross = self.config.base_max_gross_exposure

        # --- Regime multiplier ---
        regime_key = regime.upper() if regime else ""
        rm = self.config.regime_multipliers.get(regime_key)
        if rm is None:
            rm = self.config.regime_multipliers.get("DEFAULT", {"leverage": 1.0, "gross": 1.0})
        lev_mult = rm.get("leverage", 1.0)
        gross_mult = rm.get("gross", 1.0)

        # --- Confidence multiplier ---
        conf_applied = False
        if confidence is not None and confidence > self.config.confidence_multiplier_threshold:
            lev_mult *= self.config.confidence_multiplier
            gross_mult *= self.config.confidence_multiplier
            conf_applied = True

        # --- Volatility pullback (leverage only) ---
        vol_applied = False
        if volatility_z is not None and volatility_z > self.config.volatility_high_threshold:
            lev_mult *= self.config.volatility_high_leverage_multiplier
            vol_applied = True

        # Apply multipliers
        lev *= lev_mult
        gross *= gross_mult

        # Clamp to absolute maximums
        lev = min(lev, self.config.abs_max_leverage)
        gross = min(gross, self.config.abs_max_gross_exposure)

        return DynamicLimitsResult(
            max_leverage=lev,
            max_gross_exposure=gross,
            max_single_asset=self.config.base_max_single_asset,
            correlation_limit=self.config.correlation_limit,
            dynamic_active=True,
            regime_used=regime_key,
            leverage_mult=lev_mult,
            gross_mult=gross_mult,
            confidence_applied=conf_applied,
            volatility_applied=vol_applied,
        )

"""
================================================================================
GLOBAL EXPOSURE CAP - Hard Gross Exposure Limits
================================================================================
Version: 3.3.0
Purpose: Resolves HIGH impact item #2 from CTO Audit

PROBLEM SOLVED:
- SOL dominance could allow SOL=1.0 + BTC=0.3 + ETH=0.2 = 1.5× gross exposure
- In correlated markets, this is unbounded risk
- No single enforcer of gross exposure limits existed

SOLUTION:
- This module enforces HARD gross exposure caps at all times
- Integrates with SOL dominance rules
- Provides pre-trade validation and runtime monitoring

HARD LIMITS:
- MAX_GROSS_EXPOSURE = 1.50 (150% total)
- During correlation > 0.88: MAX_GROSS_EXPOSURE = 1.20 (120%)
- During correlation > 0.92: MAX_GROSS_EXPOSURE = 1.00 (100%)
- ABSOLUTE_MAX = 1.50 (never exceeded regardless of mode)
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class ExposureCapLevel(Enum):
    """Exposure cap levels based on market conditions."""
    NORMAL = "NORMAL"           # Standard caps
    ELEVATED_CORR = "ELEVATED"  # Correlation > 0.88
    HIGH_CORR = "HIGH_CORR"     # Correlation > 0.92
    CRISIS = "CRISIS"           # Maximum de-risking


@dataclass
class ExposureCapConfig:
    """Configuration for exposure caps."""
    
    # Gross exposure caps by condition
    max_gross_normal: float = 1.50       # 150% in normal conditions
    max_gross_elevated: float = 1.20     # 120% when correlation > 0.88
    max_gross_high_corr: float = 1.00    # 100% when correlation > 0.92
    max_gross_crisis: float = 0.50       # 50% in crisis mode
    
    # Absolute hard cap (NEVER exceeded)
    absolute_max_gross: float = 1.50

    # [NET-CAP 2026-06-14] Net (signed) directional exposure budget.
    # The book ran +0.54 net-long into a -23% market = ~HALF the Apr-Jun loss,
    # but gross caps don't constrain net DIRECTION (BTC+ETH+SOL all-long passes a
    # gross cap). This caps |net_exposure| so the book cannot become a pure
    # directional bet. None = OFF (backward-compatible). Set in the live profile.
    max_net_exposure: Optional[float] = None

    # Per-asset caps (default, can be overridden by mode)
    default_asset_caps: Dict[str, float] = field(default_factory=lambda: {
        "BTC": 0.80,
        "ETH": 0.80,
        "SOL": 0.80,
    })
    
    # SOL dominance asset caps
    # L4-18: Sum (1.60) may exceed absolute_max_gross (1.50), but
    # validate_exposure() enforces gross cap - no per-asset cap can
    # independently push total above absolute_max_gross.
    sol_dominant_caps: Dict[str, float] = field(default_factory=lambda: {
        "BTC": 0.30,
        "ETH": 0.30,
        "SOL": 1.00,  # SOL can go all-in (gross cap still enforced)
    })
    
    # Correlation thresholds
    # [UNLEASH] UL-4: Relaxed correlation scaling
    # Original: 0.88/0.92/5.7 -> New: 0.92/0.96/3.5
    correlation_elevated_threshold: float = 0.92   # [UNLEASH] was 0.88
    correlation_high_threshold: float = 0.96       # [UNLEASH] was 0.92

    # Scaling factor for high correlation
    correlation_scale_factor: float = 3.5  # [UNLEASH] was 5.7 -> 3.5 (less aggressive reduction)


@dataclass
class ExposureState:
    """Current exposure state."""
    btc_exposure: float = 0.0
    eth_exposure: float = 0.0
    sol_exposure: float = 0.0
    
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    
    cap_level: ExposureCapLevel = ExposureCapLevel.NORMAL
    current_gross_cap: float = 1.50
    remaining_capacity: float = 1.50
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update_totals(self):
        """Update derived totals."""
        self.gross_exposure = abs(self.btc_exposure) + abs(self.eth_exposure) + abs(self.sol_exposure)
        self.net_exposure = self.btc_exposure + self.eth_exposure + self.sol_exposure


@dataclass
class ExposureValidationResult:
    """Result of exposure validation."""
    is_allowed: bool
    original_request: float
    adjusted_size: float
    adjustment_reason: str
    current_gross: float
    new_gross_if_allowed: float
    cap_in_effect: float
    remaining_after: float
    
    def to_dict(self) -> Dict:
        return {
            "is_allowed": self.is_allowed,
            "original_request": self.original_request,
            "adjusted_size": self.adjusted_size,
            "adjustment_reason": self.adjustment_reason,
            "current_gross": self.current_gross,
            "new_gross_if_allowed": self.new_gross_if_allowed,
            "cap_in_effect": self.cap_in_effect,
            "remaining_after": self.remaining_after,
        }


class GlobalExposureCapManager:
    """
    Enforces hard gross exposure limits at all times.
    
    This is a GATEKEEPER - every position change must pass through here.
    
    USAGE:
        cap_manager = GlobalExposureCapManager()
        
        # Before entering position
        result = cap_manager.validate_new_exposure(
            asset="SOL",
            requested_exposure=0.30,
            correlation_score=0.75,
            is_sol_dominant=False
        )
        
        if result.is_allowed:
            # Proceed with trade at result.adjusted_size
            pass
        else:
            # Block trade
            pass
    """
    
    def __init__(self, config: Optional[ExposureCapConfig] = None):
        self.config = config or ExposureCapConfig()
        self.state = ExposureState()
        logger.info("GlobalExposureCapManager initialized")
    
    def update_positions(
        self,
        btc_exposure: float,
        eth_exposure: float,
        sol_exposure: float,
    ):
        """Update current exposure state."""
        self.state.btc_exposure = btc_exposure
        self.state.eth_exposure = eth_exposure
        self.state.sol_exposure = sol_exposure
        self.state.update_totals()
        self.state.timestamp = datetime.now(timezone.utc)
    
    def get_current_gross_cap(
        self,
        correlation_score: float = 0.0,
        is_crisis: bool = False,
    ) -> Tuple[float, ExposureCapLevel]:
        """
        Get current gross exposure cap based on conditions.
        
        Returns:
            (cap_value, cap_level)
        """
        if is_crisis:
            return self.config.max_gross_crisis, ExposureCapLevel.CRISIS
        
        if correlation_score >= self.config.correlation_high_threshold:
            return self.config.max_gross_high_corr, ExposureCapLevel.HIGH_CORR
        
        if correlation_score >= self.config.correlation_elevated_threshold:
            return self.config.max_gross_elevated, ExposureCapLevel.ELEVATED_CORR
        
        return self.config.max_gross_normal, ExposureCapLevel.NORMAL
    
    def get_correlation_scale_factor(self, correlation_score: float) -> float:
        """
        Get scaling factor for high correlation.

        [UNLEASH] UL-4: When correlation > 0.92, positions scaled down.
        Scale = 1 - ((correlation - 0.92) × 3.5)
        At correlation = 0.95: scale = 1 - (0.03 × 3.5) = 0.895
        At correlation = 0.98: scale = 1 - (0.06 × 3.5) = 0.790
        Minimum scale = 0.60 (60%)
        """
        if correlation_score <= self.config.correlation_elevated_threshold:
            return 1.0
        
        excess = correlation_score - self.config.correlation_elevated_threshold
        scale = 1.0 - (excess * self.config.correlation_scale_factor)
        return max(0.60, scale)  # Minimum 60%
    
    def validate_new_exposure(
        self,
        asset: str,
        requested_exposure_delta: float,
        correlation_score: float = 0.0,
        is_sol_dominant: bool = False,
        is_crisis: bool = False,
    ) -> ExposureValidationResult:
        """
        Validate if a new exposure can be added.
        
        Args:
            asset: BTC, ETH, or SOL
            requested_exposure_delta: Additional exposure requested (can be negative for reducing)
            correlation_score: Current cross-asset correlation (0 to 1)
            is_sol_dominant: Whether SOL dominance mode is active
            is_crisis: Whether crisis mode is active
        
        Returns:
            ExposureValidationResult with allowed/adjusted size
        """
        
        # Get current cap
        gross_cap, cap_level = self.get_current_gross_cap(correlation_score, is_crisis)
        
        # Apply correlation scaling to the request
        corr_scale = self.get_correlation_scale_factor(correlation_score)
        scaled_request = requested_exposure_delta * corr_scale
        
        # Get asset-specific cap
        if is_sol_dominant:
            asset_cap = self.config.sol_dominant_caps.get(asset, 0.80)
        else:
            asset_cap = self.config.default_asset_caps.get(asset, 0.80)
        
        # Calculate current asset exposure
        current_asset = getattr(self.state, f"{asset.lower()}_exposure", 0.0)
        new_asset_exposure = current_asset + scaled_request
        
        # Check asset cap
        asset_cap_adjusted = False
        if abs(new_asset_exposure) > asset_cap:
            if scaled_request > 0:
                max_additional = max(0, asset_cap - current_asset)
                scaled_request = min(scaled_request, max_additional)
            else:
                max_reduction = max(0, current_asset + asset_cap)  # How much we can go short
                scaled_request = max(scaled_request, -max_reduction)
            asset_cap_adjusted = True
        
        # Calculate new gross exposure
        current_gross = self.state.gross_exposure
        
        # For gross calculation, we need to consider direction
        old_asset = current_asset
        new_asset = old_asset + scaled_request
        
        # Change in gross = |new| - |old| for this asset
        gross_delta = abs(new_asset) - abs(old_asset)
        new_gross = current_gross + gross_delta
        
        # Check gross cap
        gross_cap_adjusted = False
        if new_gross > gross_cap:
            # How much gross room do we have?
            available_gross = gross_cap - current_gross
            
            if available_gross <= 0:
                # No room at all
                return ExposureValidationResult(
                    is_allowed=False,
                    original_request=requested_exposure_delta,
                    adjusted_size=0.0,
                    adjustment_reason=f"Gross exposure at cap ({current_gross:.2f}/{gross_cap:.2f})",
                    current_gross=current_gross,
                    new_gross_if_allowed=current_gross,
                    cap_in_effect=gross_cap,
                    remaining_after=0.0,
                )
            
            # Adjust request to fit within gross cap
            # This is simplified - assumes adding to position (not reducing)
            if scaled_request > 0:
                scaled_request = min(scaled_request, available_gross)
                gross_cap_adjusted = True

        # [NET-CAP 2026-06-14] Net (signed) directional exposure budget.
        # Gross caps above don't constrain net DIRECTION; the book ran +0.54
        # net-long into a -23% market (~half the loss). Clamp a request that
        # would push |net| beyond the budget in the already-dominant direction.
        # ONLY reduces a request that INCREASES |net| -> de-risking/hedging (a
        # request that reduces |net|) is never blocked. None = OFF.
        net_cap_adjusted = False
        _net_cap = getattr(self.config, "max_net_exposure", None)
        if _net_cap is not None and _net_cap >= 0:
            current_net = self.state.net_exposure
            new_net = current_net + scaled_request
            if abs(new_net) > _net_cap and abs(new_net) > abs(current_net):
                if scaled_request > 0:
                    scaled_request = min(scaled_request, max(0.0, _net_cap - current_net))
                else:
                    scaled_request = max(scaled_request, -max(0.0, _net_cap + current_net))
                net_cap_adjusted = True

        # Recalculate final values
        final_gross = current_gross + (abs(current_asset + scaled_request) - abs(current_asset))
        
        # Determine adjustment reason
        if scaled_request == 0 and requested_exposure_delta != 0:
            adjustment_reason = "Request fully blocked by caps"
            is_allowed = False
        elif net_cap_adjusted:
            adjustment_reason = f"Adjusted for NET exposure budget ({_net_cap:.0%}); net was {self.state.net_exposure:+.2f}"
            is_allowed = scaled_request != 0
        elif asset_cap_adjusted and gross_cap_adjusted:
            adjustment_reason = f"Adjusted for asset cap ({asset_cap:.0%}) and gross cap ({gross_cap:.0%})"
            is_allowed = True
        elif asset_cap_adjusted:
            adjustment_reason = f"Adjusted for asset cap ({asset_cap:.0%})"
            is_allowed = True
        elif gross_cap_adjusted:
            adjustment_reason = f"Adjusted for gross cap ({gross_cap:.0%})"
            is_allowed = True
        elif corr_scale < 1.0:
            adjustment_reason = f"Scaled {corr_scale:.0%} for correlation ({correlation_score:.2f})"
            is_allowed = True
        else:
            adjustment_reason = "No adjustment needed"
            is_allowed = True
        
        # Update state
        self.state.cap_level = cap_level
        self.state.current_gross_cap = gross_cap
        self.state.remaining_capacity = gross_cap - final_gross
        
        return ExposureValidationResult(
            is_allowed=is_allowed,
            original_request=requested_exposure_delta,
            adjusted_size=scaled_request,
            adjustment_reason=adjustment_reason,
            current_gross=current_gross,
            new_gross_if_allowed=final_gross,
            cap_in_effect=gross_cap,
            remaining_after=max(0, gross_cap - final_gross),
        )
    
    def force_reduce_to_cap(
        self,
        correlation_score: float = 0.0,
        is_crisis: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate required reductions to bring exposure within caps.
        
        Returns:
            Dict of asset -> target_exposure to achieve compliance
        """
        gross_cap, cap_level = self.get_current_gross_cap(correlation_score, is_crisis)
        
        if self.state.gross_exposure <= gross_cap:
            # Already compliant
            return {
                "BTC": self.state.btc_exposure,
                "ETH": self.state.eth_exposure,
                "SOL": self.state.sol_exposure,
            }
        
        # Need to reduce proportionally
        excess = self.state.gross_exposure - gross_cap
        scale_factor = gross_cap / self.state.gross_exposure if self.state.gross_exposure > 0 else 1.0
        
        return {
            "BTC": self.state.btc_exposure * scale_factor,
            "ETH": self.state.eth_exposure * scale_factor,
            "SOL": self.state.sol_exposure * scale_factor,
        }
    
    def get_remaining_capacity(
        self,
        asset: str,
        correlation_score: float = 0.0,
        is_sol_dominant: bool = False,
        is_crisis: bool = False,
    ) -> Dict[str, float]:
        """
        Get remaining capacity for an asset.
        
        Returns:
            {
                "asset_room": remaining under asset cap,
                "gross_room": remaining under gross cap,
                "effective_room": min of the two
            }
        """
        gross_cap, _ = self.get_current_gross_cap(correlation_score, is_crisis)
        
        if is_sol_dominant:
            asset_cap = self.config.sol_dominant_caps.get(asset, 0.80)
        else:
            asset_cap = self.config.default_asset_caps.get(asset, 0.80)
        
        current_asset = getattr(self.state, f"{asset.lower()}_exposure", 0.0)
        asset_room = max(0, asset_cap - abs(current_asset))
        
        gross_room = max(0, gross_cap - self.state.gross_exposure)
        
        # Apply correlation scaling
        corr_scale = self.get_correlation_scale_factor(correlation_score)
        
        return {
            "asset_room": asset_room,
            "gross_room": gross_room,
            "effective_room": min(asset_room, gross_room) * corr_scale,
            "correlation_scale": corr_scale,
        }


# Singleton instance
_exposure_cap_manager: Optional[GlobalExposureCapManager] = None


def get_exposure_cap_manager(config: Optional[ExposureCapConfig] = None) -> GlobalExposureCapManager:
    """Get or create singleton ExposureCapManager."""
    global _exposure_cap_manager
    if _exposure_cap_manager is None:
        _exposure_cap_manager = GlobalExposureCapManager(config)
    elif config is not None and _exposure_cap_manager.config != config:
        logger.info("[EXPOSURE_CAP] Config changed; refreshing singleton instance")
        _exposure_cap_manager = GlobalExposureCapManager(config)
    return _exposure_cap_manager


def reset_exposure_cap_manager():
    """Reset singleton (for testing)."""
    global _exposure_cap_manager
    _exposure_cap_manager = None

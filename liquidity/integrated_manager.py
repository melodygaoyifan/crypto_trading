"""
================================================================================
HMATS v4.0-COMPLETE - INTEGRATED WEEKEND LIQUIDITY MANAGER
================================================================================
Version: 4.0.0-INTEGRATED
Purpose: Unified weekend/holiday/low-liquidity management with constitution integration

MIGRATION NOTES:
    - Combines v3.3-STABILIZED (core/weekend_liquidity_manager.py)
    - With v3.6 structure (integration/core/weekend_liquidity_manager.py)
    - Integrates with constitution_guarantees.py and risk_agent.py

KEY FEATURES:
    1. Weekend detection and position caps (Friday 21:00 - Sunday 21:00 UTC)
    2. Holiday detection with configurable list
    3. Low-liquidity hours (22:00-02:00 UTC transition)
    4. SOL-specific DEX liquidity adjustments
    5. v3.3-HR OPPORTUNITY exemption rules
    6. Constitution guarantee integration
    7. Risk agent soft/hard veto coordination

CRITICAL RULES:
    - Weekend caps OVERRIDE OPPORTUNITY mode exposure (except v3.3-HR exemption)
    - SOL dominance is CAPPED during reduced periods
    - This module provides CAPS, not SIGNALS
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime, timezone, timedelta
from enum import Enum
import calendar

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class LiquidityPeriod(Enum):
    """Liquidity period classification."""
    NORMAL = "NORMAL"                 # Normal trading hours
    LOW_LIQUIDITY = "LOW_LIQUIDITY"   # Reduced liquidity (transition hours)
    FRIDAY_LATE = "FRIDAY_LATE"       # Friday after cutoff
    WEEKEND = "WEEKEND"               # Full weekend
    SUNDAY_EARLY = "SUNDAY_EARLY"     # Sunday before resume
    HOLIDAY = "HOLIDAY"               # Market holiday


class LiquidityQuality(Enum):
    """Liquidity quality assessment."""
    EXCELLENT = "EXCELLENT"     # Normal weekday, high volume
    GOOD = "GOOD"               # Normal weekday, moderate volume
    REDUCED = "REDUCED"         # Weekend or low volume
    POOR = "POOR"               # Weekend + low volume
    CRITICAL = "CRITICAL"       # Extreme conditions


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class IntegratedLiquidityConfig:
    """
    Unified configuration for liquidity-based exposure management.
    
    Combines v3.3 and v3.6 configurations.
    """
    
    # === Weekend Times (UTC) ===
    friday_cutoff_hour: int = 21       # Friday 21:00 UTC
    sunday_resume_hour: int = 21       # Sunday 21:00 UTC
    
    # === Low Liquidity Hours (UTC) ===
    low_liquidity_start_hour: int = 22  # 22:00 UTC
    low_liquidity_end_hour: int = 2     # 02:00 UTC
    
    # === Exposure Caps by Period ===
    cap_normal: float = 1.50            # 150% gross exposure
    cap_low_liquidity: float = 0.70     # 70% gross exposure
    cap_weekend: float = 0.50           # 50% gross exposure
    cap_holiday: float = 0.30           # 30% gross exposure
    cap_friday_late: float = 0.60       # 60% gross exposure
    cap_sunday_early: float = 0.40      # 40% gross exposure
    
    # === Per-Asset Caps ===
    asset_cap_weekend: Dict[str, float] = field(default_factory=lambda: {
        "BTC": 0.30,
        "ETH": 0.25,
        "SOL": 0.25,  # More conservative for SOL
    })
    
    asset_cap_low_liquidity: Dict[str, float] = field(default_factory=lambda: {
        "BTC": 0.50,
        "ETH": 0.40,
        "SOL": 0.60,  # SOL slightly higher - active on-chain
    })
    
    # === SOL Dominance Caps ===
    sol_dominant_cap_weekend: float = 0.50   # Max 50% SOL even in dominance
    sol_dominant_cap_low_liq: float = 0.70   # Max 70% SOL in low liquidity
    
    # === SOL DEX Adjustments ===
    sol_dex_volume_multiplier: float = 0.60  # DEX volume drops 40% on weekend
    sol_max_position_multiplier: float = 0.40  # Extra conservative
    
    # === Spread/Slippage ===
    spread_multiplier_weekend: float = 1.50    # 50% more tolerance
    spread_multiplier_low_liq: float = 1.20    # 20% more tolerance
    expected_slippage_multiplier: float = 1.80  # 80% more slippage expected
    
    # === Entry Requirements ===
    # [P42 2026-04-25] See weekend_manager.py P42 — defaults aligned with
    # 24/7 crypto markets. min_alpha = weekend_min_alpha_bps * mult.
    weekend_min_alpha_bps: float = 20.0         # was hardcoded 33 in formula
    min_alpha_multiplier_weekend: float = 1.0   # was 2.0
    min_confidence_weekend: float = 0.50        # was 0.75
    
    # === Time-based Reduction ===
    reduce_at_hours: int = 12    # Reduce after 12h into weekend
    flatten_at_hours: int = 36   # Flatten after 36h (Sunday afternoon)
    
    # === Holidays (month, day) ===
    holidays: List[Tuple[int, int]] = field(default_factory=lambda: [
        (12, 25),  # Christmas
        (12, 26),  # Boxing Day
        (1, 1),    # New Year
        (7, 4),    # July 4th (US)
    ])


# =============================================================================
# STATE
# =============================================================================

@dataclass
class IntegratedLiquidityState:
    """Current liquidity state with full tracking."""
    
    # Period classification
    period: LiquidityPeriod = LiquidityPeriod.NORMAL
    quality: LiquidityQuality = LiquidityQuality.GOOD
    
    # Boolean flags
    is_weekend: bool = False
    is_low_liquidity: bool = False
    is_holiday: bool = False
    
    # Time tracking
    hours_into_period: float = 0.0
    hours_until_normal: float = 0.0
    
    # Exposure caps
    gross_cap: float = 1.50
    per_asset_caps: Dict[str, float] = field(default_factory=dict)
    sol_dominance_cap: float = 1.0
    
    # Multipliers
    position_multiplier: float = 1.0
    spread_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    
    # Metadata
    last_update: Optional[datetime] = None
    next_transition: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period.value,
            "quality": self.quality.value,
            "is_weekend": self.is_weekend,
            "is_low_liquidity": self.is_low_liquidity,
            "is_holiday": self.is_holiday,
            "hours_into_period": self.hours_into_period,
            "hours_until_normal": self.hours_until_normal,
            "gross_cap": self.gross_cap,
            "sol_dominance_cap": self.sol_dominance_cap,
            "position_multiplier": self.position_multiplier,
        }


# =============================================================================
# MAIN MANAGER CLASS
# =============================================================================

class IntegratedWeekendLiquidityManager:
    """
    Integrated weekend/holiday/low-liquidity manager.
    
    Combines v3.3 and v3.6 functionality with v4 integration.
    
    Usage:
        manager = get_integrated_liquidity_manager()
        state = manager.update()
        
        # Check if entry allowed
        allowed, reason = manager.validate_entry(
            asset="SOL",
            target_exposure=0.30,
            is_opportunity=True
        )
        
        # Get adjusted exposure cap
        cap = manager.get_exposure_cap("SOL", is_opportunity=True)
    """
    
    def __init__(
        self,
        config: Optional[IntegratedLiquidityConfig] = None,
        weekend_config: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or IntegratedLiquidityConfig()
        self.state = IntegratedLiquidityState()
        self._constitution_module: Optional[Any] = None
        self._risk_agent: Optional[Any] = None
        self._weekend_config = weekend_config or {}

        # Apply weekend_config overrides to IntegratedLiquidityConfig fields
        if self._weekend_config:
            wc = self._weekend_config
            if "weekend_max_exposure" in wc:
                self.config.cap_weekend = float(wc["weekend_max_exposure"])
                self.config.cap_friday_late = min(1.0, float(wc["weekend_max_exposure"]))
                self.config.cap_sunday_early = min(1.0, float(wc["weekend_max_exposure"]))
            if "per_asset_weekend_caps" in wc:
                self.config.asset_cap_weekend = {
                    k: float(v) for k, v in wc["per_asset_weekend_caps"].items()
                }
            if "reduce_at_hours" in wc:
                self.config.reduce_at_hours = int(wc["reduce_at_hours"])
            if "flatten_at_hours" in wc:
                self.config.flatten_at_hours = int(wc["flatten_at_hours"])
            if "min_alpha_multiplier_weekend" in wc:
                self.config.min_alpha_multiplier_weekend = float(wc["min_alpha_multiplier_weekend"])
            if "weekend_min_alpha_bps" in wc:
                # [P42 2026-04-25] Allow operator to tune the base bps floor.
                self.config.weekend_min_alpha_bps = float(wc["weekend_min_alpha_bps"])
            if "min_confidence_weekend" in wc:
                self.config.min_confidence_weekend = float(wc["min_confidence_weekend"])
            logger.info(
                f"[v4.0-INTEGRATED] Weekend config override applied: "
                f"cap_weekend={self.config.cap_weekend:.2f}, "
                f"reduce_at={self.config.reduce_at_hours}h, "
                f"flatten_at={self.config.flatten_at_hours}h"
            )

        logger.info("[v4.0-INTEGRATED] IntegratedWeekendLiquidityManager initialized")
    
    # =========================================================================
    # INTEGRATION WITH V4 MODULES
    # =========================================================================
    
    def set_constitution_module(self, module: Any):
        """Set reference to constitution guarantees module."""
        self._constitution_module = module
        logger.debug("Constitution module linked to liquidity manager")
    
    def set_risk_agent(self, agent: Any):
        """Set reference to risk agent."""
        self._risk_agent = agent
        logger.debug("Risk agent linked to liquidity manager")
    
    # =========================================================================
    # STATE UPDATE
    # =========================================================================
    
    def update(self, utc_now: Optional[datetime] = None) -> IntegratedLiquidityState:
        """
        Update liquidity state based on current time.
        
        Args:
            utc_now: Current UTC time (defaults to datetime.now(timezone.utc))
        
        Returns:
            Updated IntegratedLiquidityState
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)
        
        self.state.last_update = utc_now
        
        # Determine period and flags
        self._determine_period(utc_now)
        
        # Calculate time metrics
        self._calculate_time_metrics(utc_now)
        
        # Update caps and multipliers
        self._update_caps_and_multipliers()
        
        # Determine quality
        self._determine_quality()
        
        return self.state
    
    def _determine_period(self, utc_now: datetime):
        """Determine current liquidity period."""
        
        weekday = utc_now.weekday()  # 0=Monday, 6=Sunday
        hour = utc_now.hour
        
        # Check holiday first
        if self._is_holiday(utc_now):
            self.state.period = LiquidityPeriod.HOLIDAY
            self.state.is_holiday = True
            self.state.is_weekend = False
            self.state.is_low_liquidity = False
            return
        
        self.state.is_holiday = False
        
        # Check weekend
        if weekday == 5:  # Saturday
            self.state.period = LiquidityPeriod.WEEKEND
            self.state.is_weekend = True
        elif weekday == 6:  # Sunday
            if hour < self.config.sunday_resume_hour:
                self.state.period = LiquidityPeriod.SUNDAY_EARLY
                self.state.is_weekend = True
            else:
                self.state.period = LiquidityPeriod.NORMAL
                self.state.is_weekend = False
        elif weekday == 4:  # Friday
            if hour >= self.config.friday_cutoff_hour:
                self.state.period = LiquidityPeriod.FRIDAY_LATE
                self.state.is_weekend = True
            else:
                self.state.period = LiquidityPeriod.NORMAL
                self.state.is_weekend = False
        else:
            self.state.period = LiquidityPeriod.NORMAL
            self.state.is_weekend = False
        
        # Check low liquidity hours (applies during weekdays)
        if not self.state.is_weekend and not self.state.is_holiday:
            if self._is_low_liquidity_hours(hour):
                self.state.period = LiquidityPeriod.LOW_LIQUIDITY
                self.state.is_low_liquidity = True
            else:
                self.state.is_low_liquidity = False
        else:
            self.state.is_low_liquidity = False
    
    def _is_holiday(self, utc_now: datetime) -> bool:
        """Check if current date is a holiday."""
        month_day = (utc_now.month, utc_now.day)
        return month_day in self.config.holidays
    
    def _is_low_liquidity_hours(self, hour: int) -> bool:
        """Check if within low liquidity hours."""
        start = self.config.low_liquidity_start_hour
        end = self.config.low_liquidity_end_hour
        
        if start > end:
            # Crosses midnight (e.g., 22:00 - 02:00)
            return hour >= start or hour < end
        else:
            return start <= hour < end
    
    def _calculate_time_metrics(self, utc_now: datetime):
        """Calculate time-based metrics."""
        
        if self.state.is_weekend:
            self.state.hours_into_period = self._hours_into_weekend(utc_now)
            self.state.hours_until_normal = self._hours_until_normal(utc_now)
        elif self.state.is_holiday:
            # Assume holiday lasts all day
            hours_left = 24 - utc_now.hour
            self.state.hours_into_period = utc_now.hour
            self.state.hours_until_normal = hours_left
        elif self.state.is_low_liquidity:
            self.state.hours_into_period = self._hours_into_low_liquidity(utc_now)
            self.state.hours_until_normal = self._hours_until_low_liquidity_end(utc_now)
        else:
            self.state.hours_into_period = 0.0
            self.state.hours_until_normal = 0.0
    
    def _hours_into_weekend(self, utc_now: datetime) -> float:
        """Calculate hours since weekend started."""
        weekday = utc_now.weekday()
        hour = utc_now.hour
        
        if weekday == 4 and hour >= self.config.friday_cutoff_hour:
            return hour - self.config.friday_cutoff_hour
        elif weekday == 5:
            friday_hours = 24 - self.config.friday_cutoff_hour
            return friday_hours + hour
        elif weekday == 6 and hour < self.config.sunday_resume_hour:
            friday_hours = 24 - self.config.friday_cutoff_hour
            return friday_hours + 24 + hour
        return 0.0
    
    def _hours_until_normal(self, utc_now: datetime) -> float:
        """Calculate hours until normal trading resumes."""
        weekday = utc_now.weekday()
        hour = utc_now.hour
        
        if weekday == 4 and hour >= self.config.friday_cutoff_hour:
            hours_until_saturday = 24 - hour
            return hours_until_saturday + 24 + self.config.sunday_resume_hour
        elif weekday == 5:
            hours_remaining = 24 - hour
            return hours_remaining + self.config.sunday_resume_hour
        elif weekday == 6 and hour < self.config.sunday_resume_hour:
            return self.config.sunday_resume_hour - hour
        return 0.0
    
    def _hours_into_low_liquidity(self, utc_now: datetime) -> float:
        """Calculate hours into low liquidity period."""
        hour = utc_now.hour
        start = self.config.low_liquidity_start_hour
        
        if hour >= start:
            return hour - start
        else:
            return (24 - start) + hour
    
    def _hours_until_low_liquidity_end(self, utc_now: datetime) -> float:
        """Calculate hours until low liquidity ends."""
        hour = utc_now.hour
        end = self.config.low_liquidity_end_hour
        
        if hour < end:
            return end - hour
        else:
            return (24 - hour) + end
    
    def _update_caps_and_multipliers(self):
        """Update exposure caps and multipliers based on period."""
        
        period = self.state.period
        hours_in = self.state.hours_into_period
        
        # Base caps by period
        if period == LiquidityPeriod.HOLIDAY:
            self.state.gross_cap = self.config.cap_holiday
            self.state.per_asset_caps = {"BTC": 0.20, "ETH": 0.15, "SOL": 0.15}
            self.state.sol_dominance_cap = 0.30
            self.state.position_multiplier = 0.30
            self.state.spread_multiplier = 2.0
            self.state.slippage_multiplier = 2.5
            
        elif period == LiquidityPeriod.WEEKEND:
            self.state.gross_cap = self.config.cap_weekend
            self.state.per_asset_caps = self.config.asset_cap_weekend.copy()
            self.state.sol_dominance_cap = self.config.sol_dominant_cap_weekend
            self.state.position_multiplier = self.config.cap_weekend
            self.state.spread_multiplier = self.config.spread_multiplier_weekend
            self.state.slippage_multiplier = self.config.expected_slippage_multiplier
            
            # Progressive reduction
            if hours_in >= self.config.reduce_at_hours:
                reduction = 0.1 * ((hours_in - self.config.reduce_at_hours) / 12)
                self.state.position_multiplier *= (1.0 - min(reduction, 0.3))
            
        elif period == LiquidityPeriod.FRIDAY_LATE:
            self.state.gross_cap = self.config.cap_friday_late
            self.state.per_asset_caps = self.config.asset_cap_weekend.copy()
            self.state.sol_dominance_cap = 0.60
            self.state.position_multiplier = 0.60
            self.state.spread_multiplier = 1.30
            self.state.slippage_multiplier = 1.50
            
        elif period == LiquidityPeriod.SUNDAY_EARLY:
            self.state.gross_cap = self.config.cap_sunday_early
            self.state.per_asset_caps = {"BTC": 0.25, "ETH": 0.20, "SOL": 0.20}
            self.state.sol_dominance_cap = 0.40
            self.state.position_multiplier = 0.40
            self.state.spread_multiplier = 1.60
            self.state.slippage_multiplier = 1.90
            
        elif period == LiquidityPeriod.LOW_LIQUIDITY:
            self.state.gross_cap = self.config.cap_low_liquidity
            self.state.per_asset_caps = self.config.asset_cap_low_liquidity.copy()
            self.state.sol_dominance_cap = self.config.sol_dominant_cap_low_liq
            self.state.position_multiplier = 0.70
            self.state.spread_multiplier = self.config.spread_multiplier_low_liq
            self.state.slippage_multiplier = 1.30
            
        else:  # NORMAL
            self.state.gross_cap = self.config.cap_normal
            self.state.per_asset_caps = {"BTC": 0.60, "ETH": 0.50, "SOL": 0.90}
            self.state.sol_dominance_cap = 1.0
            self.state.position_multiplier = 1.0
            self.state.spread_multiplier = 1.0
            self.state.slippage_multiplier = 1.0
    
    def _determine_quality(self):
        """Determine overall liquidity quality."""
        
        if self.state.is_holiday:
            self.state.quality = LiquidityQuality.CRITICAL
        elif self.state.period == LiquidityPeriod.WEEKEND:
            if self.state.hours_into_period > 24:
                self.state.quality = LiquidityQuality.POOR
            else:
                self.state.quality = LiquidityQuality.REDUCED
        elif self.state.period == LiquidityPeriod.SUNDAY_EARLY:
            self.state.quality = LiquidityQuality.POOR
        elif self.state.period == LiquidityPeriod.FRIDAY_LATE:
            self.state.quality = LiquidityQuality.REDUCED
        elif self.state.is_low_liquidity:
            self.state.quality = LiquidityQuality.REDUCED
        else:
            self.state.quality = LiquidityQuality.GOOD
    
    # =========================================================================
    # EXPOSURE VALIDATION
    # =========================================================================
    
    def validate_entry(
        self,
        asset: str,
        target_exposure: float,
        is_opportunity: bool = False,
        estimated_alpha_bps: float = 0.0,
        confidence: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Validate if a new entry is allowed.
        
        Args:
            asset: Asset symbol (BTC, ETH, SOL)
            target_exposure: Target exposure (0.0 to 1.0)
            is_opportunity: Whether in OPPORTUNITY mode
            estimated_alpha_bps: Estimated alpha in basis points
            confidence: Signal confidence (0.0 to 1.0)
        
        Returns:
            (allowed, reason) tuple
        """
        
        # Always allow if normal period
        if self.state.period == LiquidityPeriod.NORMAL:
            return (True, "Normal period - entry allowed")
        
        # Check v3.3-HR OPPORTUNITY exemption
        if is_opportunity and self._check_opportunity_exemption(asset, target_exposure):
            return (True, "v3.3-HR OPPORTUNITY exemption - entry allowed with cap")
        
        # Check gross cap
        if target_exposure > self.state.gross_cap:
            return (
                False,
                f"Exposure {target_exposure:.0%} > {self.state.period.value} cap {self.state.gross_cap:.0%}"
            )
        
        # Check per-asset cap
        asset_cap = self.state.per_asset_caps.get(asset, 0.30)
        if target_exposure > asset_cap:
            return (
                False,
                f"{asset} exposure {target_exposure:.0%} > period cap {asset_cap:.0%}"
            )
        
        # Check alpha requirement during reduced periods
        # [P42 2026-04-25] Was 33 * mult — base now configurable.
        if self.state.quality in [LiquidityQuality.REDUCED, LiquidityQuality.POOR]:
            min_alpha = self.config.weekend_min_alpha_bps * self.config.min_alpha_multiplier_weekend
            if estimated_alpha_bps < min_alpha:
                return (
                    False,
                    f"Alpha {estimated_alpha_bps:.0f}bps < min required {min_alpha:.0f}bps"
                )
            
            if confidence < self.config.min_confidence_weekend:
                return (
                    False,
                    f"Confidence {confidence:.0%} < min required {self.config.min_confidence_weekend:.0%}"
                )
        
        return (True, f"Entry validated for {self.state.period.value}")
    
    def _check_opportunity_exemption(self, asset: str, target_exposure: float) -> bool:
        """
        Check v3.3-HR OPPORTUNITY exemption.
        
        During OPPORTUNITY mode, weekend caps are partially relaxed:
        - Gross cap: 70% (vs 50% normal weekend)
        - SOL dominance: 70% (vs 50% normal weekend)
        """
        
        # OPPORTUNITY exemption caps (higher than weekend)
        opp_gross_cap = 0.70
        opp_sol_cap = 0.70
        
        if asset == "SOL":
            return target_exposure <= opp_sol_cap
        else:
            return target_exposure <= opp_gross_cap
    
    def get_exposure_cap(
        self,
        asset: str,
        is_opportunity: bool = False,
    ) -> float:
        """
        Get maximum allowed exposure for an asset.
        
        Args:
            asset: Asset symbol
            is_opportunity: Whether in OPPORTUNITY mode
        
        Returns:
            Maximum allowed exposure (0.0 to 1.0)
        """
        
        # Normal period = no cap
        if self.state.period == LiquidityPeriod.NORMAL:
            if asset == "SOL":
                return 0.90  # SOL dominance limit
            return 0.60
        
        # OPPORTUNITY exemption
        if is_opportunity:
            return 0.70 if asset == "SOL" else 0.50
        
        # Period-based cap
        return self.state.per_asset_caps.get(asset, 0.30)
    
    def get_sol_adjustments(self) -> Dict[str, float]:
        """
        Get SOL-specific adjustments for weekend/low-liquidity.
        
        Returns dict with adjustment multipliers.
        """
        
        if not (self.state.is_weekend or self.state.is_low_liquidity):
            return {
                "volume_multiplier": 1.0,
                "position_multiplier": 1.0,
                "spread_multiplier": 1.0,
                "dex_shift_expected": False,
            }
        
        return {
            "volume_multiplier": self.config.sol_dex_volume_multiplier,
            "position_multiplier": self.config.sol_max_position_multiplier,
            "spread_multiplier": self.state.spread_multiplier,
            "dex_shift_expected": self.state.is_weekend,
        }
    
    # =========================================================================
    # CONSTITUTION INTEGRATION
    # =========================================================================
    
    def get_soft_veto_condition(self) -> Optional[str]:
        """
        Get soft veto condition for risk agent integration.
        
        Returns condition name if applicable, None otherwise.
        """
        
        if self.state.period == LiquidityPeriod.NORMAL:
            return None
        
        if self.state.is_weekend:
            return "WEEKEND_LIQUIDITY"
        elif self.state.is_holiday:
            return "HOLIDAY_LIQUIDITY"
        elif self.state.is_low_liquidity:
            return "LOW_LIQUIDITY_HOURS"
        
        return None
    
    def get_constitution_override(self) -> Optional[Dict[str, Any]]:
        """
        Get constitution override parameters.
        
        Returns dict to be applied to constitution guarantees.
        """
        
        if self.state.period == LiquidityPeriod.NORMAL:
            return None
        
        return {
            "max_gross_exposure": self.state.gross_cap,
            "max_single_asset_exposure": max(self.state.per_asset_caps.values()),
            "sol_dominance_cap": self.state.sol_dominance_cap,
            "spread_tolerance_multiplier": self.state.spread_multiplier,
            "reason": f"Liquidity period: {self.state.period.value}",
        }
    
    # =========================================================================
    # PROOF LOG
    # =========================================================================
    
    def get_proof_log(self) -> str:
        """Generate proof log for liquidity state."""
        
        return (
            f"[LIQUIDITY] period={self.state.period.value} | "
            f"quality={self.state.quality.value} | "
            f"weekend={self.state.is_weekend} | "
            f"holiday={self.state.is_holiday} | "
            f"low_liq={self.state.is_low_liquidity} | "
            f"hours_in={self.state.hours_into_period:.1f} | "
            f"hours_until_normal={self.state.hours_until_normal:.1f} | "
            f"gross_cap={self.state.gross_cap:.0%} | "
            f"sol_cap={self.state.sol_dominance_cap:.0%} | "
            f"pos_mult={self.state.position_multiplier:.2f}"
        )


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_integrated_manager: Optional[IntegratedWeekendLiquidityManager] = None


def get_integrated_liquidity_manager(
    weekend_config: Optional[Dict[str, Any]] = None,
) -> IntegratedWeekendLiquidityManager:
    """Get singleton integrated liquidity manager. Pass weekend_config on first call."""
    global _integrated_manager
    if _integrated_manager is None:
        _integrated_manager = IntegratedWeekendLiquidityManager(
            weekend_config=weekend_config,
        )
    return _integrated_manager


def reset_integrated_liquidity_manager():
    """Reset singleton (for testing)."""
    global _integrated_manager
    _integrated_manager = None


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Alias for v3.6 import path
WeekendLiquidityManager = IntegratedWeekendLiquidityManager
get_weekend_liquidity_manager = get_integrated_liquidity_manager

# Alias for v3.3 import path
LiquidityManager = IntegratedWeekendLiquidityManager
get_liquidity_manager = get_integrated_liquidity_manager


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("HMATS v4.0-COMPLETE - INTEGRATED LIQUIDITY MANAGER TEST")
    print("=" * 70)
    
    manager = get_integrated_liquidity_manager()
    state = manager.update()
    
    print(f"\nCurrent State:")
    print(f"  Period: {state.period.value}")
    print(f"  Quality: {state.quality.value}")
    print(f"  Is Weekend: {state.is_weekend}")
    print(f"  Is Holiday: {state.is_holiday}")
    print(f"  Gross Cap: {state.gross_cap:.0%}")
    print(f"  SOL Dominance Cap: {state.sol_dominance_cap:.0%}")
    
    print(f"\nProof Log:")
    print(f"  {manager.get_proof_log()}")
    
    # Test entry validation
    print(f"\nEntry Validation Tests:")
    for exposure in [0.30, 0.50, 0.70]:
        allowed, reason = manager.validate_entry(
            asset="SOL",
            target_exposure=exposure,
            is_opportunity=False,
            estimated_alpha_bps=80,
            confidence=0.80
        )
        print(f"  SOL {exposure:.0%}: {'ON' if allowed else 'OFF'} - {reason}")

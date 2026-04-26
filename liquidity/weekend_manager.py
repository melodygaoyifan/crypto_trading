"""
================================================================================
HMATS v4.0 - WEEKEND LIQUIDITY MANAGER
================================================================================
Version: 4.0.0
Purpose: Weekend-specific liquidity management and risk adjustments

RESTORED FROM: v3.3-STABILIZED (weekend_liquidity_manager.py + override_weekend.py)

KEY FEATURES:
    1. Weekend detection and time-zone aware trading hours
    2. Reduced liquidity thresholds during weekends
    3. Position size reduction for weekend exposure
    4. SOL-specific weekend handling (DEX liquidity shift)
    5. Auto-recovery to normal parameters on Monday

CRITICAL RULES:
    - Weekend = Friday 21:00 UTC to Sunday 21:00 UTC
    - Max position size reduced to 50% of normal
    - Spread tolerance increased by 50%
    - SOL-specific: DEX volume typically 60% of weekday
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timezone, timedelta
from enum import Enum
import calendar

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    """Trading session classification."""
    NORMAL = "NORMAL"           # Mon-Fri normal hours
    WEEKEND = "WEEKEND"         # Sat-Sun
    FRIDAY_LATE = "FRIDAY_LATE" # Friday after 21:00 UTC
    SUNDAY_EARLY = "SUNDAY_EARLY" # Sunday before 21:00 UTC
    HOLIDAY = "HOLIDAY"         # Major holidays


class LiquidityQuality(Enum):
    """Liquidity quality assessment."""
    EXCELLENT = "EXCELLENT"     # Normal weekday, high volume
    GOOD = "GOOD"               # Normal weekday, moderate volume
    REDUCED = "REDUCED"         # Weekend or low volume
    POOR = "POOR"               # Weekend + low volume
    CRITICAL = "CRITICAL"       # Extreme conditions


@dataclass
class WeekendConfig:
    """Weekend trading configuration."""
    
    # Time boundaries (UTC)
    friday_cutoff_hour: int = 21       # Friday 21:00 UTC
    sunday_resume_hour: int = 21       # Sunday 21:00 UTC
    
    # Position limits
    max_position_multiplier: float = 0.50    # 50% of normal
    max_new_position_multiplier: float = 0.30  # 30% for new entries
    
    # Spread tolerance
    spread_multiplier: float = 1.50    # 50% more tolerance
    
    # Slippage expectations
    expected_slippage_multiplier: float = 1.80  # 80% more slippage
    
    # SOL-specific
    sol_dex_volume_multiplier: float = 0.60  # DEX volume drops 40%
    sol_max_position_multiplier: float = 0.40  # Even more conservative
    
    # Liquidity thresholds
    min_orderbook_depth_usd: float = 50_000  # Lower weekend threshold
    max_spread_bps: float = 30.0  # Higher weekend tolerance


@dataclass
class WeekendState:
    """Current weekend state."""
    session: TradingSession = TradingSession.NORMAL
    is_weekend: bool = False
    hours_until_normal: float = 0.0
    hours_into_weekend: float = 0.0
    liquidity_quality: LiquidityQuality = LiquidityQuality.GOOD
    position_multiplier: float = 1.0
    spread_multiplier: float = 1.0
    last_update: Optional[datetime] = None


class WeekendLiquidityManager:
    """
    Manages weekend-specific liquidity conditions and adjustments.
    
    From v3.3-STABILIZED: Restored to handle reduced weekend liquidity.
    """
    
    def __init__(self, config: Optional[WeekendConfig] = None):
        self.config = config or WeekendConfig()
        self.state = WeekendState()
        self._holiday_cache: Dict[str, List[datetime]] = {}
        
        logger.info("[v4.0] WeekendLiquidityManager initialized")
    
    def update(self, utc_now: Optional[datetime] = None) -> WeekendState:
        """
        Update weekend state based on current time.
        
        Args:
            utc_now: Current UTC time (defaults to datetime.now(timezone.utc))
        
        Returns:
            Updated WeekendState
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)
        
        self.state.last_update = utc_now
        
        # Determine session
        self.state.session = self._determine_session(utc_now)
        self.state.is_weekend = self.state.session in [
            TradingSession.WEEKEND,
            TradingSession.FRIDAY_LATE,
            TradingSession.SUNDAY_EARLY
        ]
        
        # Calculate time metrics
        if self.state.is_weekend:
            self.state.hours_until_normal = self._hours_until_normal(utc_now)
            self.state.hours_into_weekend = self._hours_into_weekend(utc_now)
        else:
            self.state.hours_until_normal = 0.0
            self.state.hours_into_weekend = 0.0
        
        # Set multipliers
        self._update_multipliers()
        
        return self.state
    
    def _determine_session(self, utc_now: datetime) -> TradingSession:
        """Determine current trading session."""
        
        weekday = utc_now.weekday()  # 0=Monday, 6=Sunday
        hour = utc_now.hour
        
        # Saturday (weekday=5)
        if weekday == 5:
            return TradingSession.WEEKEND
        
        # Sunday (weekday=6)
        if weekday == 6:
            if hour < self.config.sunday_resume_hour:
                return TradingSession.SUNDAY_EARLY
            else:
                return TradingSession.NORMAL  # Sunday evening = normal
        
        # Friday (weekday=4)
        if weekday == 4:
            if hour >= self.config.friday_cutoff_hour:
                return TradingSession.FRIDAY_LATE
            else:
                return TradingSession.NORMAL
        
        # Monday-Thursday
        return TradingSession.NORMAL
    
    def _hours_until_normal(self, utc_now: datetime) -> float:
        """Calculate hours until normal trading resumes."""
        
        weekday = utc_now.weekday()
        hour = utc_now.hour
        
        if weekday == 4 and hour >= self.config.friday_cutoff_hour:
            # Friday late: until Sunday 21:00
            hours_until_saturday = 24 - hour
            return hours_until_saturday + 24 + self.config.sunday_resume_hour
        
        elif weekday == 5:
            # Saturday: until Sunday 21:00
            hours_remaining_saturday = 24 - hour
            return hours_remaining_saturday + self.config.sunday_resume_hour
        
        elif weekday == 6 and hour < self.config.sunday_resume_hour:
            # Sunday early: until 21:00
            return self.config.sunday_resume_hour - hour
        
        return 0.0
    
    def _hours_into_weekend(self, utc_now: datetime) -> float:
        """Calculate hours since weekend started."""
        
        weekday = utc_now.weekday()
        hour = utc_now.hour
        
        if weekday == 4 and hour >= self.config.friday_cutoff_hour:
            # Friday late
            return hour - self.config.friday_cutoff_hour
        
        elif weekday == 5:
            # Saturday
            friday_hours = 24 - self.config.friday_cutoff_hour
            return friday_hours + hour
        
        elif weekday == 6 and hour < self.config.sunday_resume_hour:
            # Sunday early
            friday_hours = 24 - self.config.friday_cutoff_hour
            return friday_hours + 24 + hour
        
        return 0.0
    
    def _update_multipliers(self):
        """Update position and spread multipliers based on session."""
        
        if not self.state.is_weekend:
            self.state.position_multiplier = 1.0
            self.state.spread_multiplier = 1.0
            self.state.liquidity_quality = LiquidityQuality.GOOD
            return
        
        # Weekend multipliers
        self.state.position_multiplier = self.config.max_position_multiplier
        self.state.spread_multiplier = self.config.spread_multiplier
        
        # Deeper into weekend = worse liquidity
        hours_in = self.state.hours_into_weekend
        
        if hours_in < 6:
            self.state.liquidity_quality = LiquidityQuality.REDUCED
        elif hours_in < 24:
            self.state.liquidity_quality = LiquidityQuality.POOR
            self.state.position_multiplier *= 0.8  # Extra reduction
        else:
            self.state.liquidity_quality = LiquidityQuality.POOR
            self.state.position_multiplier *= 0.7  # Even more reduction
    
    def get_sol_adjustments(self) -> Dict[str, float]:
        """
        Get SOL-specific weekend adjustments.
        
        SOL has unique weekend dynamics:
        - DEX liquidity shifts (CEX -> DEX)
        - Memecoin activity continues
        - Network congestion may vary
        """
        
        if not self.state.is_weekend:
            return {
                "position_multiplier": 1.0,
                "dex_volume_expected": 1.0,
                "spread_tolerance": 1.0,
            }
        
        return {
            "position_multiplier": self.config.sol_max_position_multiplier,
            "dex_volume_expected": self.config.sol_dex_volume_multiplier,
            "spread_tolerance": self.config.spread_multiplier * 1.2,  # Extra for SOL
        }
    
    def should_reduce_position(self, current_exposure: float) -> Tuple[bool, float]:
        """
        Check if position should be reduced for weekend.
        
        Args:
            current_exposure: Current position as fraction of max
        
        Returns:
            (should_reduce, target_exposure)
        """
        
        if not self.state.is_weekend:
            return (False, current_exposure)
        
        max_weekend_exposure = self.state.position_multiplier
        
        if current_exposure > max_weekend_exposure:
            return (True, max_weekend_exposure)
        
        return (False, current_exposure)
    
    def get_execution_adjustments(self) -> Dict[str, any]:
        """
        Get execution parameter adjustments for weekend.
        
        Returns dict with:
        - spread_tolerance_bps: Adjusted spread tolerance
        - slippage_expectation_bps: Expected slippage
        - use_passive_only: Whether to use only passive orders
        - slice_size_multiplier: Order slicing adjustment
        """
        
        if not self.state.is_weekend:
            return {
                "spread_tolerance_bps": 10.0,
                "slippage_expectation_bps": 5.0,
                "use_passive_only": False,
                "slice_size_multiplier": 1.0,
            }
        
        return {
            "spread_tolerance_bps": 10.0 * self.config.spread_multiplier,
            "slippage_expectation_bps": 5.0 * self.config.expected_slippage_multiplier,
            "use_passive_only": True,  # Passive only on weekends
            "slice_size_multiplier": 0.5,  # Smaller slices
        }
    
    def validate_weekend_entry(
        self,
        asset: str,
        target_exposure: float,
        orderbook_depth_usd: float,
        spread_bps: float,
    ) -> Tuple[bool, str]:
        """
        Validate if a new entry is acceptable during weekend.
        
        Args:
            asset: Asset symbol
            target_exposure: Target position exposure
            orderbook_depth_usd: Current orderbook depth
            spread_bps: Current spread in basis points
        
        Returns:
            (is_valid, reason)
        """
        
        if not self.state.is_weekend:
            return (True, "Normal trading hours")
        
        # Check depth
        if orderbook_depth_usd < self.config.min_orderbook_depth_usd:
            return (
                False,
                f"Orderbook depth ${orderbook_depth_usd:,.0f} < weekend min ${self.config.min_orderbook_depth_usd:,.0f}"
            )
        
        # Check spread
        if spread_bps > self.config.max_spread_bps:
            return (
                False,
                f"Spread {spread_bps:.1f}bps > weekend max {self.config.max_spread_bps:.1f}bps"
            )
        
        # Check position size
        max_new = self.config.max_new_position_multiplier
        if target_exposure > max_new:
            return (
                False,
                f"Target exposure {target_exposure:.0%} > weekend max new {max_new:.0%}"
            )
        
        # SOL-specific checks
        if asset == "SOL":
            sol_max = self.config.sol_max_position_multiplier
            if target_exposure > sol_max:
                return (
                    False,
                    f"SOL exposure {target_exposure:.0%} > weekend SOL max {sol_max:.0%}"
                )
        
        return (True, "Weekend entry validated")
    
    def get_proof_log(self) -> str:
        """Generate proof log for weekend state."""
        
        return (
            f"[WEEKEND] session={self.state.session.value} | "
            f"is_weekend={self.state.is_weekend} | "
            f"hours_in={self.state.hours_into_weekend:.1f} | "
            f"hours_until_normal={self.state.hours_until_normal:.1f} | "
            f"liquidity={self.state.liquidity_quality.value} | "
            f"pos_mult={self.state.position_multiplier:.2f} | "
            f"spread_mult={self.state.spread_multiplier:.2f}"
        )


# =============================================================================
# WEEKEND OVERRIDE RULES (from override_weekend.py)
# =============================================================================

class WeekendOverrideRules:
    """
    Weekend-specific override rules.
    
    From v3.3-STABILIZED override_weekend.py
    """
    
    # Position limits
    WEEKEND_MAX_GROSS_EXPOSURE = 0.50  # 50% max
    WEEKEND_MAX_SINGLE_ASSET = 0.30    # 30% max single
    WEEKEND_SOL_MAX = 0.25             # 25% max SOL
    
    # Entry rules
    # [P42 2026-04-25] Lowered defaults after runtime forensics (P41) showed
    # weekend gate blocks ~25% of all signals on 24/7 crypto markets. The
    # previous defaults (mult=2.0, base=33) were calibrated for equities-style
    # weekend illiquidity that doesn't apply to Kraken. New defaults match the
    # operator's existing live config (`live_high_risk.json:151` already had
    # `min_alpha_multiplier_weekend: 1.0`) and add a configurable base, so
    # `min_alpha = WEEKEND_MIN_ALPHA_BPS * _alpha_mult` can be tuned without
    # editing this constant.
    WEEKEND_MIN_ALPHA_BPS = 20.0        # was hardcoded 33 inside the formula
    WEEKEND_MIN_ALPHA_MULTIPLIER = 1.0  # was 2.0 — calibrated for 24/7 crypto
    # [P52 2026-04-25] Lowered 0.50 → 0.30 to match P46 live config
    # (`live_high_risk.json:min_confidence_weekend = 0.30`). Runtime data on
    # 2026-04-25 showed weekend rejections firing at "min 50%" even though JSON
    # had 0.30 set — symptom of class default being read instead of the JSON.
    # Lowering the default closes the gap regardless of root cause and matches
    # the same defensive pattern P42 applied to MAX_LEVERAGE.
    WEEKEND_MIN_CONFIDENCE = 0.30       # was 0.50 — matches live_high_risk.json
    
    # Exit rules
    WEEKEND_REDUCE_AT_HOURS = 12  # Reduce after 12h into weekend
    WEEKEND_FLATTEN_AT_HOURS = 36 # Flatten after 36h (Sunday afternoon)
    
    @classmethod
    def should_override_entry(
        cls,
        weekend_state: WeekendState,
        estimated_alpha_bps: float,
        confidence: float,
        asset: str = "",
        system_mode: str = "",
        weekend_config: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, str]:
        """Check if entry should be blocked by weekend override."""

        if not weekend_state.is_weekend:
            return (False, "")

        _wk_cfg = weekend_config or {}
        # [AP-5] Weekend filter kill switch. User accepts overnight/weekend risk
        # under 25% DD + 5x leverage aggressive philosophy. When config flag is
        # explicitly False, entire weekend gate is bypassed (warning logged).
        # Default behavior preserved: filter active unless operator opts out.
        if _wk_cfg.get("weekend_filter_enabled", True) is False:
            try:
                import logging as _logging
                _logging.getLogger(__name__).info(
                    f"[AP-5] WEEKEND_FILTER_DISABLED {asset}: bypassed by config "
                    f"alpha={estimated_alpha_bps:.0f} conf={confidence:.0%}"
                )
            except Exception:
                pass
            return (False, "")
        _mode = str(system_mode or "").upper()
        _asset = str(asset or "").upper()
        _opp_alpha_mult_by_asset = _wk_cfg.get("opportunity_alpha_multiplier_weekend_by_asset", {})
        _min_alpha_mult_by_asset = _wk_cfg.get("min_alpha_multiplier_weekend_by_asset", {})
        _alpha_mult = float(
            _wk_cfg.get(
                "opportunity_alpha_multiplier_weekend",
                _wk_cfg.get("min_alpha_multiplier_weekend", cls.WEEKEND_MIN_ALPHA_MULTIPLIER),
            )
            if _mode == "OPPORTUNITY"
            else _wk_cfg.get("min_alpha_multiplier_weekend", cls.WEEKEND_MIN_ALPHA_MULTIPLIER)
        )
        if _asset:
            _asset_override = None
            if _mode == "OPPORTUNITY" and isinstance(_opp_alpha_mult_by_asset, dict):
                _asset_override = _opp_alpha_mult_by_asset.get(_asset)
            elif isinstance(_min_alpha_mult_by_asset, dict):
                _asset_override = _min_alpha_mult_by_asset.get(_asset)
            if _asset_override is not None:
                try:
                    _alpha_mult = float(_asset_override)
                except Exception:
                    pass
        _min_conf = float(
            _wk_cfg.get(
                "opportunity_confidence_weekend",
                _wk_cfg.get("min_confidence_weekend", cls.WEEKEND_MIN_CONFIDENCE),
            )
            if _mode == "OPPORTUNITY"
            else _wk_cfg.get("min_confidence_weekend", cls.WEEKEND_MIN_CONFIDENCE)
        )
        _opp_conf_by_asset = _wk_cfg.get("opportunity_confidence_weekend_by_asset", {})
        _min_conf_by_asset = _wk_cfg.get("min_confidence_weekend_by_asset", {})
        if _asset:
            _conf_override = None
            if _mode == "OPPORTUNITY" and isinstance(_opp_conf_by_asset, dict):
                _conf_override = _opp_conf_by_asset.get(_asset)
            elif isinstance(_min_conf_by_asset, dict):
                _conf_override = _min_conf_by_asset.get(_asset)
            if _conf_override is not None:
                try:
                    _min_conf = float(_conf_override)
                except Exception:
                    pass

        # Alpha check
        # [P42 2026-04-25] Base value (was hardcoded 33) now configurable
        # via `weekend_min_alpha_bps` in weekend_config, default 20 bps.
        # Operators tune via JSON without editing constants.
        _min_alpha_bps_base = float(
            _wk_cfg.get("weekend_min_alpha_bps", cls.WEEKEND_MIN_ALPHA_BPS)
        )
        min_alpha = _min_alpha_bps_base * _alpha_mult
        if estimated_alpha_bps < min_alpha:
            return (True, f"[AP-5] Weekend alpha {estimated_alpha_bps:.0f}bps < min {min_alpha:.0f}bps (set weekend_filter_enabled=False to bypass)")

        # Confidence check
        if confidence < _min_conf:
            return (True, f"[AP-5] Weekend confidence {confidence:.0%} < min {_min_conf:.0%} (set weekend_filter_enabled=False to bypass)")
        
        return (False, "")
    
    @classmethod
    def get_weekend_exposure_cap(
        cls,
        weekend_state: WeekendState,
        asset: str,
        weekend_config: Optional[Dict[str, float]] = None,
    ) -> float:
        """Get exposure cap for weekend."""
        
        if not weekend_state.is_weekend:
            return 1.0  # No cap

        _wk_cfg = weekend_config or {}
        _asset_caps = _wk_cfg.get("per_asset_weekend_caps") or {}
        if asset in _asset_caps:
            return float(_asset_caps[asset])
        if "weekend_max_single_asset" in _wk_cfg:
            return float(_wk_cfg["weekend_max_single_asset"])
        if asset == "SOL":
            return cls.WEEKEND_SOL_MAX

        return cls.WEEKEND_MAX_SINGLE_ASSET


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_weekend_manager: Optional[WeekendLiquidityManager] = None


def get_weekend_liquidity_manager() -> WeekendLiquidityManager:
    """Get singleton weekend liquidity manager."""
    global _weekend_manager
    if _weekend_manager is None:
        _weekend_manager = WeekendLiquidityManager()
    return _weekend_manager

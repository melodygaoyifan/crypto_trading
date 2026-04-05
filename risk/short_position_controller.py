"""
================================================================================
HMATS v6.0.0 - Short Position Risk Controller
================================================================================
Purpose: Specialized risk control for short positions with unlimited loss protection

Key Features:
1. Single-trade stop loss (max 10%)
2. Portfolio short exposure limits
3. Short squeeze detection integration
4. Funding rate monitoring
5. Automatic position reduction on risk signals

SOTA Compliance:
- Addresses "unlimited loss risk" for short strategies
- Implements strict stop-loss discipline
- Monitors squeeze risk indicators
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ShortRiskLevel(Enum):
    """Short position risk level."""
    NORMAL = "NORMAL"           # Full trading allowed
    ELEVATED = "ELEVATED"       # Reduced position sizes
    HIGH = "HIGH"               # Exit-only mode
    CRITICAL = "CRITICAL"       # Force flatten


class SqueezeRiskLevel(Enum):
    """Short squeeze risk level."""
    LOW = "LOW"                 # < 30%
    MODERATE = "MODERATE"       # 30-50%
    HIGH = "HIGH"               # 50-70%
    EXTREME = "EXTREME"         # > 70%


class FundingRateState(Enum):
    """Funding rate state for shorts."""
    FAVORABLE = "FAVORABLE"     # Negative rate (shorts receive)
    NEUTRAL = "NEUTRAL"         # Near zero
    UNFAVORABLE = "UNFAVORABLE" # Positive rate (shorts pay)
    EXTREME = "EXTREME"         # Very high positive rate


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ShortRiskConfig:
    """Configuration for short position risk control."""
    
    # Single trade stop loss
    max_single_loss_pct: float = 0.10      # 10% max loss per trade
    warning_single_loss_pct: float = 0.05  # 5% warning threshold
    
    # Portfolio short exposure
    max_short_exposure_pct: float = 0.60   # 60% max short exposure
    warning_short_exposure_pct: float = 0.40  # 40% warning
    
    # Portfolio stop loss
    max_portfolio_loss_pct: float = 0.15   # 15% max portfolio loss
    reduce_at_portfolio_loss_pct: float = 0.10  # 10% reduce positions
    
    # Squeeze risk thresholds
    squeeze_warning_threshold: float = 0.50
    squeeze_reduce_threshold: float = 0.70
    squeeze_flatten_threshold: float = 0.80
    
    # Funding rate thresholds (8-hour rate)
    funding_warning_threshold: float = 0.001   # 0.1%
    funding_reduce_threshold: float = 0.002    # 0.2%
    funding_halt_threshold: float = 0.003      # 0.3%
    
    # Position limits
    max_leverage: float = 3.0
    max_single_position_pct: float = 0.30  # 30% max in single asset
    
    # Recovery settings
    cooldown_after_stop_hours: float = 4.0
    require_manual_restart_on_critical: bool = True


@dataclass
class ShortPositionState:
    """State of a short position."""
    asset: str
    entry_price: float
    current_price: float
    size: float
    entry_time: datetime
    unrealized_pnl_pct: float = 0.0
    squeeze_risk: float = 0.0
    funding_rate: float = 0.0
    
    @property
    def is_profitable(self) -> bool:
        return self.unrealized_pnl_pct > 0
    
    @property
    def time_held_hours(self) -> float:
        return (datetime.now(timezone.utc) - self.entry_time).total_seconds() / 3600


@dataclass
class ShortRiskAssessment:
    """Result of short position risk assessment."""
    risk_level: ShortRiskLevel
    squeeze_level: SqueezeRiskLevel
    funding_state: FundingRateState
    
    allow_new_shorts: bool = True
    reduce_position_pct: float = 0.0  # 0-100%
    force_flatten: bool = False
    
    reasons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'risk_level': self.risk_level.value,
            'squeeze_level': self.squeeze_level.value,
            'funding_state': self.funding_state.value,
            'allow_new_shorts': self.allow_new_shorts,
            'reduce_position_pct': self.reduce_position_pct,
            'force_flatten': self.force_flatten,
            'reasons': self.reasons,
            'recommendations': self.recommendations,
        }


# =============================================================================
# MAIN CONTROLLER
# =============================================================================

class ShortPositionController:
    """
    Specialized risk controller for short positions.
    
    Addresses SOTA requirements:
    - Unlimited loss protection (strict stop-loss)
    - Squeeze risk monitoring
    - Funding rate management
    - Position sizing based on risk
    """
    
    def __init__(self, config: Optional[ShortRiskConfig] = None):
        self.config = config or ShortRiskConfig()
        
        # State tracking
        self.active_positions: Dict[str, ShortPositionState] = {}
        self.daily_pnl: float = 0.0
        self.daily_start_equity: float = 0.0
        self.peak_equity: float = 0.0
        
        # Risk history
        self.squeeze_history: deque = deque(maxlen=100)
        self.funding_history: deque = deque(maxlen=100)
        self.loss_events: List[Dict] = []
        
        # Control state
        self.risk_level = ShortRiskLevel.NORMAL
        self.last_stop_trigger: Optional[datetime] = None
        self.is_halted = False
        self.halt_reason: Optional[str] = None
        
        logger.info(f"ShortPositionController initialized with config: {self.config}")
    
    def assess_risk(
        self,
        positions: Dict[str, ShortPositionState],
        account_equity: float,
        market_data: Dict[str, Any],
    ) -> ShortRiskAssessment:
        """
        Comprehensive risk assessment for short positions.
        
        Args:
            positions: Current short positions
            account_equity: Current account equity
            market_data: Market data including squeeze indicators
        
        Returns:
            ShortRiskAssessment with recommendations
        """
        self.active_positions = positions
        reasons = []
        recommendations = []
        
        # 1. Check single position losses
        single_loss_check = self._check_single_position_losses(positions)
        if single_loss_check['force_stop']:
            reasons.append(f"Single position loss exceeded {self.config.max_single_loss_pct:.1%}")
            recommendations.append(f"Force close position in {single_loss_check['worst_asset']}")
        
        # 2. Check portfolio short exposure
        total_short_exposure = sum(
            abs(p.size * p.current_price) / account_equity 
            for p in positions.values()
        )
        
        if total_short_exposure > self.config.max_short_exposure_pct:
            reasons.append(f"Short exposure {total_short_exposure:.1%} exceeds limit")
            recommendations.append("Reduce short exposure")
        
        # 3. Check squeeze risk
        squeeze_risk = market_data.get('squeeze_risk', 0.0)
        self.squeeze_history.append(squeeze_risk)
        squeeze_level = self._classify_squeeze_risk(squeeze_risk)
        
        if squeeze_risk > self.config.squeeze_warning_threshold:
            reasons.append(f"Squeeze risk elevated: {squeeze_risk:.1%}")
        
        # 4. Check funding rate
        funding_rate = market_data.get('funding_rate', 0.0)
        self.funding_history.append(funding_rate)
        funding_state = self._classify_funding_rate(funding_rate)
        
        if funding_rate > self.config.funding_warning_threshold:
            reasons.append(f"High funding rate: {funding_rate:.4%}")
        
        # 5. Determine overall risk level and actions
        risk_level, allow_new, reduce_pct, force_flatten = self._determine_risk_level(
            single_loss_check=single_loss_check,
            total_exposure=total_short_exposure,
            squeeze_risk=squeeze_risk,
            funding_rate=funding_rate,
        )
        
        # 6. Build recommendations
        if not allow_new:
            recommendations.append("No new short positions allowed")
        if reduce_pct > 0:
            recommendations.append(f"Reduce positions by {reduce_pct:.0%}")
        if force_flatten:
            recommendations.append("FLATTEN ALL SHORT POSITIONS")
        
        self.risk_level = risk_level
        
        return ShortRiskAssessment(
            risk_level=risk_level,
            squeeze_level=squeeze_level,
            funding_state=funding_state,
            allow_new_shorts=allow_new,
            reduce_position_pct=reduce_pct,
            force_flatten=force_flatten,
            reasons=reasons,
            recommendations=recommendations,
        )
    
    def check_stop_loss(
        self,
        asset: str,
        entry_price: float,
        current_price: float,
        position_size: float,
    ) -> Tuple[bool, str, float]:
        """
        Check if stop loss should be triggered for a short position.
        
        Args:
            asset: Asset symbol
            entry_price: Entry price
            current_price: Current price
            position_size: Position size (negative for short)
        
        Returns:
            (should_stop, reason, loss_pct)
        """
        # For short: loss when price goes UP
        loss_pct = (current_price - entry_price) / entry_price
        
        if loss_pct >= self.config.max_single_loss_pct:
            reason = f"Max single loss {self.config.max_single_loss_pct:.1%} reached"
            self._record_stop_event(asset, loss_pct, reason)
            return True, reason, loss_pct
        
        if loss_pct >= self.config.warning_single_loss_pct:
            logger.warning(
                f"[SHORT RISK] {asset} approaching stop: loss={loss_pct:.2%}"
            )
        
        return False, "", loss_pct
    
    def get_position_size_multiplier(
        self,
        squeeze_risk: float,
        funding_rate: float,
        current_exposure: float,
    ) -> float:
        """
        Calculate position size multiplier based on current risk conditions.
        
        Returns multiplier [0, 1]:
        - 1.0: Full position allowed
        - 0.5: Half position
        - 0.0: No new positions
        """
        multiplier = 1.0
        
        # Squeeze risk adjustment
        if squeeze_risk > self.config.squeeze_flatten_threshold:
            multiplier = 0.0
        elif squeeze_risk > self.config.squeeze_reduce_threshold:
            multiplier *= 0.25
        elif squeeze_risk > self.config.squeeze_warning_threshold:
            multiplier *= 0.50
        
        # Funding rate adjustment
        if funding_rate > self.config.funding_halt_threshold:
            multiplier = 0.0
        elif funding_rate > self.config.funding_reduce_threshold:
            multiplier *= 0.50
        elif funding_rate > self.config.funding_warning_threshold:
            multiplier *= 0.75
        
        # Exposure adjustment
        if current_exposure > self.config.max_short_exposure_pct:
            multiplier = 0.0
        elif current_exposure > self.config.warning_short_exposure_pct:
            multiplier *= 0.50
        
        # Cooldown check
        if self._in_cooldown():
            multiplier = 0.0
        
        return multiplier
    
    def update_daily_pnl(self, pnl: float, current_equity: float):
        """Update daily P&L tracking."""
        self.daily_pnl = pnl
        
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Check daily loss limit
        if self.daily_start_equity > 0:
            daily_loss_pct = -pnl / self.daily_start_equity
            if daily_loss_pct > self.config.reduce_at_portfolio_loss_pct:
                self.risk_level = ShortRiskLevel.HIGH
                logger.warning(
                    f"[SHORT RISK] Daily loss {daily_loss_pct:.2%} exceeded threshold"
                )
    
    def reset_daily(self, current_equity: float):
        """Reset daily tracking (call at start of trading day)."""
        self.daily_pnl = 0.0
        self.daily_start_equity = current_equity
        logger.info(f"ShortPositionController daily reset: equity={current_equity:,.2f}")
    
    def halt(self, reason: str):
        """Halt all short trading."""
        self.is_halted = True
        self.halt_reason = reason
        self.risk_level = ShortRiskLevel.CRITICAL
        logger.critical(f"[SHORT RISK] HALTED: {reason}")
    
    def resume(self, confirm: bool = False) -> Tuple[bool, str]:
        """Resume trading after halt."""
        if not self.is_halted:
            return True, "Not halted"
        
        if self.config.require_manual_restart_on_critical and not confirm:
            return False, "Manual confirmation required (confirm=True)"
        
        self.is_halted = False
        self.halt_reason = None
        self.risk_level = ShortRiskLevel.NORMAL
        logger.info("[SHORT RISK] Trading resumed")
        return True, "Resumed"
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _check_single_position_losses(
        self,
        positions: Dict[str, ShortPositionState],
    ) -> Dict:
        """Check individual position losses."""
        worst_loss = 0.0
        worst_asset = None
        force_stop = False
        
        for asset, pos in positions.items():
            loss_pct = (pos.current_price - pos.entry_price) / pos.entry_price
            if loss_pct > worst_loss:
                worst_loss = loss_pct
                worst_asset = asset
            
            if loss_pct >= self.config.max_single_loss_pct:
                force_stop = True
        
        return {
            'worst_loss': worst_loss,
            'worst_asset': worst_asset,
            'force_stop': force_stop,
        }
    
    def _classify_squeeze_risk(self, squeeze_risk: float) -> SqueezeRiskLevel:
        """Classify squeeze risk level."""
        if squeeze_risk < 0.30:
            return SqueezeRiskLevel.LOW
        elif squeeze_risk < 0.50:
            return SqueezeRiskLevel.MODERATE
        elif squeeze_risk < 0.70:
            return SqueezeRiskLevel.HIGH
        else:
            return SqueezeRiskLevel.EXTREME
    
    def _classify_funding_rate(self, funding_rate: float) -> FundingRateState:
        """Classify funding rate state."""
        if funding_rate < 0:
            return FundingRateState.FAVORABLE
        elif funding_rate < self.config.funding_warning_threshold:
            return FundingRateState.NEUTRAL
        elif funding_rate < self.config.funding_halt_threshold:
            return FundingRateState.UNFAVORABLE
        else:
            return FundingRateState.EXTREME
    
    def _determine_risk_level(
        self,
        single_loss_check: Dict,
        total_exposure: float,
        squeeze_risk: float,
        funding_rate: float,
    ) -> Tuple[ShortRiskLevel, bool, float, bool]:
        """
        Determine overall risk level and actions.
        
        Returns:
            (risk_level, allow_new_shorts, reduce_pct, force_flatten)
        """
        # Default: Normal
        risk_level = ShortRiskLevel.NORMAL
        allow_new = True
        reduce_pct = 0.0
        force_flatten = False
        
        # Check for critical conditions
        if single_loss_check['force_stop']:
            risk_level = ShortRiskLevel.CRITICAL
            force_flatten = True
        
        if squeeze_risk >= self.config.squeeze_flatten_threshold:
            risk_level = ShortRiskLevel.CRITICAL
            force_flatten = True
        
        # Check for high risk
        if squeeze_risk >= self.config.squeeze_reduce_threshold:
            risk_level = ShortRiskLevel.HIGH
            allow_new = False
            reduce_pct = 0.50
        
        if funding_rate >= self.config.funding_halt_threshold:
            risk_level = ShortRiskLevel.HIGH
            allow_new = False
        
        # Check for elevated risk
        if squeeze_risk >= self.config.squeeze_warning_threshold:
            if risk_level == ShortRiskLevel.NORMAL:
                risk_level = ShortRiskLevel.ELEVATED
            reduce_pct = max(reduce_pct, 0.25)
        
        if funding_rate >= self.config.funding_reduce_threshold:
            if risk_level == ShortRiskLevel.NORMAL:
                risk_level = ShortRiskLevel.ELEVATED
            reduce_pct = max(reduce_pct, 0.25)
        
        if total_exposure >= self.config.max_short_exposure_pct:
            allow_new = False
        
        return risk_level, allow_new, reduce_pct, force_flatten
    
    def _record_stop_event(self, asset: str, loss_pct: float, reason: str):
        """Record a stop loss event."""
        self.loss_events.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'asset': asset,
            'loss_pct': loss_pct,
            'reason': reason,
        })
        self.last_stop_trigger = datetime.now(timezone.utc)
        logger.warning(f"[STOP LOSS] {asset}: {loss_pct:.2%} - {reason}")
    
    def _in_cooldown(self) -> bool:
        """Check if in cooldown period after stop loss."""
        if self.last_stop_trigger is None:
            return False
        
        cooldown_end = self.last_stop_trigger + timedelta(
            hours=self.config.cooldown_after_stop_hours
        )
        return datetime.now(timezone.utc) < cooldown_end
    
    def get_stats(self) -> Dict:
        """Get controller statistics."""
        return {
            'risk_level': self.risk_level.value,
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason,
            'active_positions': len(self.active_positions),
            'daily_pnl': self.daily_pnl,
            'loss_events_count': len(self.loss_events),
            'avg_squeeze_risk': np.mean(self.squeeze_history) if self.squeeze_history else 0,
            'avg_funding_rate': np.mean(self.funding_history) if self.funding_history else 0,
            'in_cooldown': self._in_cooldown(),
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_instance: Optional[ShortPositionController] = None


def get_short_controller(
    config: Optional[ShortRiskConfig] = None
) -> ShortPositionController:
    """Get or create the singleton short position controller."""
    global _instance
    if _instance is None:
        _instance = ShortPositionController(config)
    elif config is not None and _instance.config != config:
        logger.info("[SHORT_CONTROLLER] Config changed; refreshing singleton instance")
        _instance = ShortPositionController(config)
    return _instance


def reset_short_controller():
    """Reset the singleton instance."""
    global _instance
    _instance = None

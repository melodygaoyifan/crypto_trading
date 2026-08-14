"""
================================================================================
HMATS v5.1.0 - SOTA Risk Controller
================================================================================
Purpose: Advanced risk management implementing SOTA checklist requirements

Key Features (from Upgrade Checklist):
1. Dynamic Stop-Loss with regime awareness
2. Correlation Collapse Detection with auto risk-off
3. Kill-Switch implementation with validation
4. Adaptive Risk by Regime
5. Volatility Targeting integration

Author: HMATS SOTA Upgrade
Version: 5.1.0-SOTA
================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, List, Tuple, Any, Callable
from datetime import datetime, timezone, timedelta
from collections import deque
import json

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class RiskState(Enum):
    """System-wide risk state."""
    NORMAL = "NORMAL"                    # Full trading allowed
    REDUCED = "REDUCED"                  # Position sizes reduced
    EXIT_ONLY = "EXIT_ONLY"              # Only exits allowed
    HALTED = "HALTED"                    # All trading halted
    KILL_SWITCH = "KILL_SWITCH"          # Emergency shutdown


class CorrelationState(Enum):
    """Correlation regime state."""
    NORMAL = "NORMAL"                    # Expected correlations
    DECOUPLING = "DECOUPLING"            # Assets diverging unexpectedly
    CRISIS = "CRISIS"                    # Extreme correlation (panic)
    RECOVERING = "RECOVERING"            # Returning to normal


class KillSwitchReason(Enum):
    """Reasons for kill-switch activation."""
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_DRAWDOWN = "max_drawdown"
    CORRELATION_CRISIS = "correlation_crisis"
    MANUAL_TRIGGER = "manual_trigger"
    SYSTEM_ERROR = "system_error"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SOTARiskConfig:
    """SOTA risk management configuration.

    [CFG-7] Drawdown gradient (ordered, non-overlapping with engine layer):
        Engine layer:  reduce=8% -> halt=20%  (ProductionConfig / canonical_config)
        This layer:    reduce=15% -> halt=25% -> critical=35%
        Kill-switch:   35% (canonical_config.KILL_SWITCH_DRAWDOWN, sota_flags authoritative)
    """

    # Drawdown thresholds - risk controller layer (above engine's 8%/20%)
    reduce_threshold: float = 0.15           # 15% -> reduce position sizes
    halt_threshold: float = 0.25             # 25% -> halt new trades
    critical_threshold: float = 0.35         # 35% -> critical alert

    # Daily limits - relaxed for crypto volatility
    max_daily_loss: float = 0.08             # 8% daily loss limit
    max_daily_trades: int = 20

    # Kill-switch triggers - [CFG-7] aligned with sota_flags (0.35)
    kill_switch_drawdown: float = 0.35       # 35% triggers kill-switch (was 0.30, now matches sota_flags)
    kill_switch_daily_loss: float = 0.10     # 10% daily loss triggers (matches DAILY_LOSS_HALT)
    require_manual_restart: bool = True
    
    # Correlation collapse detection
    normal_btc_eth_corr: float = 0.85
    normal_btc_sol_corr: float = 0.75
    collapse_threshold: float = 0.30         # Correlation drops > 30%
    spike_threshold: float = 0.98            # Correlation > 98% = crisis
    correlation_lookback: int = 24           # Hours for correlation calc
    
    # Regime-aware adjustments
    high_vol_position_cap: float = 0.10      # 10% max in high vol
    low_vol_position_boost: float = 1.2      # 20% boost in low vol
    
    # Volatility targeting
    target_annual_vol: float = 0.25          # 25% annual target
    vol_lookback_days: int = 20
    
    # Stop-loss regime multipliers
    stop_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "bull_trend": 1.2,
        "bear_trend": 0.8,
        "high_volatility": 1.5,
        "low_volatility": 0.7,
        "sideways": 1.0
    })

    # R6: Regime-conditional drawdown threshold multipliers
    # >1.0 = more tolerance (higher DD before state change)
    # <1.0 = stricter (earlier state change)
    # Kill-switch is NEVER scaled - absolute safety floor
    regime_dd_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "STRONG_TREND": 1.30,
        "MOMENTUM_RALLY": 1.20,
        "IGNITION": 1.20,
        "QUIET_ACCUMULATION": 1.00,
        "WEAK_CONSOLIDATION": 0.90,
        "VOLATILE_CHOP": 0.80,
        "PANIC_SELLOFF": 0.60,
        "EXTREME_VOLATILITY": 0.70,
    })


# =============================================================================
# SOTA RISK CONTROLLER
# =============================================================================

class SOTARiskController:
    """
    Advanced risk controller implementing SOTA checklist requirements.
    
    Responsibilities:
    1. Monitor drawdown and daily P&L
    2. Detect correlation collapse/crisis
    3. Manage kill-switch logic
    4. Provide regime-aware risk parameters
    5. Coordinate with volatility targeting
    """
    
    def __init__(self, config: SOTARiskConfig = None):
        self.config = config or SOTARiskConfig()
        
        # State tracking
        self.risk_state = RiskState.NORMAL
        self.correlation_state = CorrelationState.NORMAL
        self.kill_switch_active = False
        self.kill_switch_reason: Optional[KillSwitchReason] = None
        self.kill_switch_time: Optional[datetime] = None
        
        # P&L tracking
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_drawdown = 0.0

        # R6: Current regime for DD threshold scaling
        self._current_regime: str = ""
        
        # Correlation tracking
        self.price_history: Dict[str, deque] = {
            "BTC": deque(maxlen=1000),
            "ETH": deque(maxlen=1000),
            "SOL": deque(maxlen=1000)
        }
        self.correlation_history: deque = deque(maxlen=100)
        
        # Event log — bounded to prevent memory leak in long-running sessions
        self.risk_events: deque = deque(maxlen=500)
        
        # Callbacks for alerts
        self._alert_callbacks: List[Callable] = []
        
        logger.info(f"[SOTA] RiskController initialized with config: "
                   f"kill_switch_dd={self.config.kill_switch_drawdown}, "
                   f"correlation_spike={self.config.spike_threshold}")
    
    # =========================================================================
    # [P253] PERSISTENCE
    # =========================================================================
    # peak_equity, kill_switch_active and risk_state were RAM-only: every
    # restart re-anchored the peak to current equity (erasing accumulated
    # drawdown) and silently CLEARED an active HALT/kill-switch — the one 35%
    # kill switch in the system self-disarmed on every deploy (P150 class).

    def to_dict(self) -> dict:
        return {
            "risk_state": self.risk_state.value,
            "kill_switch_active": bool(self.kill_switch_active),
            "kill_switch_reason": (
                self.kill_switch_reason.value
                if self.kill_switch_reason is not None else None),
            "kill_switch_time": (
                self.kill_switch_time.isoformat()
                if self.kill_switch_time is not None else None),
            "peak_equity": float(self.peak_equity),
        }

    def from_dict(self, data: dict) -> None:
        """Restore persisted state. Deliberately one-directional: this can
        RAISE the peak and RE-ARM a halt/kill-switch, but a malformed payload
        can never lower the peak or clear a live halt — the conservative
        failure direction for a loss-capping control."""
        if not isinstance(data, dict):
            return
        try:
            saved_peak = float(data.get("peak_equity") or 0.0)
            if saved_peak > self.peak_equity:
                self.peak_equity = saved_peak
        except (TypeError, ValueError):  # noqa: silent-swallow — P223-class value coercion; a malformed peak keeps the current (never-lower) value, which is the stated contract
            pass
        try:
            rs = data.get("risk_state")
            if rs:
                restored = RiskState(rs)
                # Only restore a MORE restrictive state than NORMAL; never
                # relax the current one.
                if restored != RiskState.NORMAL and self.risk_state == RiskState.NORMAL:
                    self.risk_state = restored
        except (ValueError, KeyError):  # noqa: silent-swallow — unknown enum keeps the current state; restore is one-directional by contract
            pass
        if data.get("kill_switch_active"):
            self.kill_switch_active = True
            try:
                r = data.get("kill_switch_reason")
                self.kill_switch_reason = KillSwitchReason(r) if r else self.kill_switch_reason
            except (ValueError, KeyError):  # noqa: silent-swallow — the ARMED bit above is the load-bearing restore; reason is best-effort metadata
                pass
            try:
                t = data.get("kill_switch_time")
                self.kill_switch_time = (
                    datetime.fromisoformat(t) if t else self.kill_switch_time)
            except (TypeError, ValueError):  # noqa: silent-swallow — same: metadata only, the CRITICAL log below still fires
                pass
            logger.critical(
                "[SOTA] Kill switch RESTORED as ACTIVE from persisted state "
                f"(reason={data.get('kill_switch_reason')}, "
                f"since={data.get('kill_switch_time')}) — manual reset required, "
                "a restart no longer clears it")

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register_alert_callback(self, callback: Callable):
        """Register callback for risk alerts."""
        self._alert_callbacks.append(callback)
    
    def _emit_alert(self, alert_type: str, message: str, severity: str = "WARNING"):
        """Emit alert to all registered callbacks."""
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": alert_type,
            "message": message,
            "severity": severity,
            "risk_state": self.risk_state.value,
            "correlation_state": self.correlation_state.value
        }
        
        self.risk_events.append(alert)
        
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.log(
            logging.CRITICAL if severity == "CRITICAL" else logging.WARNING,
            f"[SOTA ALERT] {alert_type}: {message}"
        )
    
    # =========================================================================
    # EQUITY & DRAWDOWN TRACKING
    # =========================================================================
    
    def update_equity(self, equity: float, realized_pnl_today: float = None,
                      regime: str = ""):
        """
        Update equity and check drawdown thresholds.

        Args:
            equity: Current account equity
            realized_pnl_today: Today's realized P&L (optional)
            regime: Current regime name for R6 threshold scaling (optional)
        """
        self.current_equity = equity

        # R6: Update regime for threshold scaling
        if regime:
            self._current_regime = regime

        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Calculate drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity

        # Update daily P&L if provided
        if realized_pnl_today is not None:
            self.daily_pnl = realized_pnl_today

        # Check thresholds
        self._check_drawdown_thresholds()
        self._check_daily_loss()
    
    def _check_drawdown_thresholds(self):
        """Check drawdown against regime-scaled thresholds and update risk state.

        R6: Thresholds are scaled by regime_dd_multipliers.
        Kill-switch is NEVER regime-scaled (absolute safety).
        Floors/ceilings prevent extreme scaling.
        """
        old_state = self.risk_state

        if self.kill_switch_active:
            return  # Kill-switch takes precedence

        # R6: Regime-scaled thresholds
        regime_mult = self.config.regime_dd_multipliers.get(self._current_regime, 1.0)
        eff_reduce = max(0.07, self.config.reduce_threshold * regime_mult)
        eff_halt = max(0.12, self.config.halt_threshold * regime_mult)
        eff_critical = min(0.50, self.config.critical_threshold * regime_mult)

        # Kill-switch is NEVER regime-scaled
        if self.current_drawdown >= self.config.kill_switch_drawdown:
            self._activate_kill_switch(KillSwitchReason.MAX_DRAWDOWN)
        elif self.current_drawdown >= eff_halt:
            self.risk_state = RiskState.HALTED
        elif self.current_drawdown >= eff_reduce:
            self.risk_state = RiskState.REDUCED
        else:
            # Only return to normal if correlation is also normal
            if self.correlation_state == CorrelationState.NORMAL:
                self.risk_state = RiskState.NORMAL

        if old_state != self.risk_state:
            self._emit_alert(
                "RISK_STATE_CHANGE",
                f"Risk state changed: {old_state.value} -> {self.risk_state.value} "
                f"(dd={self.current_drawdown:.2%} regime={self._current_regime} "
                f"mult={regime_mult:.2f} "
                f"eff={eff_reduce:.1%}/{eff_halt:.1%}/{eff_critical:.1%})",
                "CRITICAL" if self.risk_state in [RiskState.HALTED, RiskState.KILL_SWITCH] else "WARNING"
            )
    
    def _check_daily_loss(self):
        """Check daily loss limit."""
        if self.peak_equity > 0:
            daily_loss_pct = abs(min(0, self.daily_pnl)) / self.peak_equity
            
            if daily_loss_pct >= self.config.kill_switch_daily_loss:
                self._activate_kill_switch(KillSwitchReason.MAX_DAILY_LOSS)
            elif daily_loss_pct >= self.config.max_daily_loss:
                if self.risk_state == RiskState.NORMAL:
                    self.risk_state = RiskState.REDUCED
                    self._emit_alert(
                        "DAILY_LOSS_WARNING",
                        f"Daily loss limit approaching: {daily_loss_pct:.2%}",
                        "WARNING"
                    )
    
    # =========================================================================
    # CORRELATION COLLAPSE DETECTION
    # =========================================================================
    
    def update_prices(self, prices: Dict[str, float]):
        """
        Update price history and check for correlation collapse.
        
        Args:
            prices: Dict of {symbol: price}
        """
        timestamp = datetime.now(timezone.utc)
        
        for symbol, price in prices.items():
            if symbol in self.price_history:
                self.price_history[symbol].append({
                    "timestamp": timestamp,
                    "price": price
                })
        
        # Check correlation if we have enough data
        if all(len(h) >= 50 for h in self.price_history.values()):
            self._check_correlation_collapse()
    
    def _calculate_correlation(self, symbol1: str, symbol2: str, window: int = 50) -> float:
        """Calculate rolling correlation between two assets."""
        try:
            prices1 = [p["price"] for p in list(self.price_history[symbol1])[-window:]]
            prices2 = [p["price"] for p in list(self.price_history[symbol2])[-window:]]
            
            if len(prices1) < window or len(prices2) < window:
                return np.nan
            
            # Calculate returns
            returns1 = np.diff(np.log(prices1))
            returns2 = np.diff(np.log(prices2))
            
            # Correlation
            corr = np.corrcoef(returns1, returns2)[0, 1]
            return corr
            
        except Exception as e:
            # [P96 2026-04-26] Promoted DEBUG → WARNING. Correlation
            # collapse detection is mission-critical; if _calculate_correlation
            # silently returns NaN due to numpy issues / malformed prices,
            # _check_correlation_collapse() returns early and the gate
            # goes blind. Operator running at INFO previously had no
            # visibility. Includes asset names + exception type for triage.
            logger.warning(
                f"[SOTA_RISK] Correlation calculation FAILED "
                f"({type(e).__name__}: {e}); returning NaN. "
                f"Correlation collapse check will be skipped this tick — "
                f"investigate price-history quality."
            )
            return np.nan
    
    def _check_correlation_collapse(self):
        """Check for correlation collapse or crisis conditions."""
        btc_eth_corr = self._calculate_correlation("BTC", "ETH")
        btc_sol_corr = self._calculate_correlation("BTC", "SOL")
        
        if np.isnan(btc_eth_corr) or np.isnan(btc_sol_corr):
            return
        
        # Store for history
        self.correlation_history.append({
            "timestamp": datetime.now(timezone.utc),
            "btc_eth": btc_eth_corr,
            "btc_sol": btc_sol_corr
        })
        
        old_state = self.correlation_state
        
        # Check for correlation crisis (extremely high correlation = panic)
        if btc_eth_corr >= self.config.spike_threshold or btc_sol_corr >= self.config.spike_threshold:
            self.correlation_state = CorrelationState.CRISIS
            
            if old_state != CorrelationState.CRISIS:
                self._emit_alert(
                    "CORRELATION_CRISIS",
                    f"Correlation spike detected! BTC-ETH={btc_eth_corr:.3f}, "
                    f"BTC-SOL={btc_sol_corr:.3f}. Possible market panic.",
                    "CRITICAL"
                )
                
                # Set exit-only mode
                if self.risk_state == RiskState.NORMAL:
                    self.risk_state = RiskState.EXIT_ONLY
                    
                # Check if kill-switch threshold
                if btc_eth_corr >= 0.99 and btc_sol_corr >= 0.99:
                    self._activate_kill_switch(KillSwitchReason.CORRELATION_CRISIS)
        
        # Check for correlation collapse (assets decoupling unexpectedly)
        elif (abs(btc_eth_corr - self.config.normal_btc_eth_corr) > self.config.collapse_threshold or
              abs(btc_sol_corr - self.config.normal_btc_sol_corr) > self.config.collapse_threshold):
            
            self.correlation_state = CorrelationState.DECOUPLING
            
            if old_state not in [CorrelationState.DECOUPLING, CorrelationState.CRISIS]:
                self._emit_alert(
                    "CORRELATION_COLLAPSE",
                    f"Correlation breakdown! BTC-ETH={btc_eth_corr:.3f} "
                    f"(normal={self.config.normal_btc_eth_corr}), "
                    f"BTC-SOL={btc_sol_corr:.3f} (normal={self.config.normal_btc_sol_corr})",
                    "WARNING"
                )
                
                # Set exit-only mode
                if self.risk_state == RiskState.NORMAL:
                    self.risk_state = RiskState.EXIT_ONLY
        
        # Check for recovery
        elif old_state in [CorrelationState.CRISIS, CorrelationState.DECOUPLING]:
            self.correlation_state = CorrelationState.RECOVERING
        
        # Return to normal
        elif old_state == CorrelationState.RECOVERING:
            self.correlation_state = CorrelationState.NORMAL
            
            # Can return to normal trading if drawdown is also OK
            if self.current_drawdown < self.config.reduce_threshold:
                self.risk_state = RiskState.NORMAL
                self._emit_alert(
                    "CORRELATION_RECOVERED",
                    "Correlation returned to normal. Resuming normal trading.",
                    "INFO"
                )
    
    # =========================================================================
    # KILL-SWITCH
    # =========================================================================
    
    def _activate_kill_switch(self, reason: KillSwitchReason):
        """Activate the kill-switch."""
        if self.kill_switch_active:
            return  # Already active
        
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        self.kill_switch_time = datetime.now(timezone.utc)
        self.risk_state = RiskState.KILL_SWITCH
        
        self._emit_alert(
            "KILL_SWITCH_ACTIVATED",
            f"KILL-SWITCH ACTIVATED! Reason: {reason.value}. "
            f"All positions should be flattened. "
            f"Manual restart required: {self.config.require_manual_restart}",
            "CRITICAL"
        )
        
        logger.critical(
            f"[KILL-SWITCH] Activated at {self.kill_switch_time} "
            f"due to {reason.value}. Drawdown={self.current_drawdown:.2%}, "
            f"Daily P&L={self.daily_pnl:.2f}"
        )
    
    def reset_kill_switch(self, confirmation_code: str = None) -> bool:
        """
        Reset kill-switch (requires confirmation if configured).
        
        Args:
            confirmation_code: Required confirmation code if manual restart required
            
        Returns:
            True if successfully reset
        """
        if not self.kill_switch_active:
            return True
        
        if self.config.require_manual_restart:
            expected_code = f"RESET-{self.kill_switch_time.strftime('%Y%m%d')}"
            if confirmation_code != expected_code:
                logger.warning(
                    f"Kill-switch reset denied. Expected code: {expected_code}"
                )
                return False
        
        self.kill_switch_active = False
        self.kill_switch_reason = None
        self.kill_switch_time = None
        self.risk_state = RiskState.REDUCED  # Start in reduced mode
        
        self._emit_alert(
            "KILL_SWITCH_RESET",
            "Kill-switch has been reset. Starting in REDUCED mode.",
            "INFO"
        )
        
        return True
    
    # =========================================================================
    # REGIME-AWARE RISK PARAMETERS
    # =========================================================================
    
    def get_regime_stop_multiplier(self, regime: str) -> float:
        """Get stop-loss multiplier for current regime."""
        return self.config.stop_multipliers.get(regime, 1.0)
    
    def get_position_size_multiplier(self, regime: str, volatility: float = None) -> float:
        """
        Get position size multiplier based on regime and volatility.
        
        Args:
            regime: Current market regime
            volatility: Current realized volatility (annualized)
            
        Returns:
            Multiplier to apply to base position size
        """
        multiplier = 1.0
        
        # Risk state adjustment
        if self.risk_state == RiskState.REDUCED:
            multiplier *= 0.5
        elif self.risk_state in [RiskState.EXIT_ONLY, RiskState.HALTED, RiskState.KILL_SWITCH]:
            return 0.0  # No new positions
        
        # Regime adjustment
        regime_multipliers = {
            "bull_trend": 1.0,
            "bear_trend": 0.7,
            "high_volatility": 0.5,
            "low_volatility": 1.2,
            "sideways": 0.6
        }
        multiplier *= regime_multipliers.get(regime, 1.0)
        
        # Volatility targeting adjustment
        if volatility and self.config.target_annual_vol > 0:
            vol_ratio = self.config.target_annual_vol / max(volatility, 0.01)
            vol_ratio = np.clip(vol_ratio, 0.3, 2.0)  # Limit adjustment
            multiplier *= vol_ratio
        
        # Correlation state adjustment
        if self.correlation_state == CorrelationState.CRISIS:
            multiplier *= 0.3
        elif self.correlation_state == CorrelationState.DECOUPLING:
            multiplier *= 0.5
        
        return max(0.0, min(multiplier, 2.0))  # Cap at 2x
    
    # =========================================================================
    # TRADE AUTHORIZATION
    # =========================================================================
    
    def can_open_position(self, symbol: str, side: str, size: float) -> Tuple[bool, str]:
        """
        Check if a new position can be opened.
        
        Returns:
            (allowed, reason)
        """
        # Kill-switch check
        if self.kill_switch_active:
            return False, f"Kill-switch active: {self.kill_switch_reason.value}"
        
        # Risk state check
        if self.risk_state == RiskState.KILL_SWITCH:
            return False, "System in KILL_SWITCH state"
        
        if self.risk_state == RiskState.HALTED:
            return False, f"Trading halted: drawdown={self.current_drawdown:.2%}"
        
        if self.risk_state == RiskState.EXIT_ONLY:
            return False, f"Exit-only mode: correlation={self.correlation_state.value}"
        
        # Correlation crisis
        if self.correlation_state == CorrelationState.CRISIS:
            return False, "Correlation crisis detected - no new positions"
        
        return True, "Approved"
    
    def can_exit_position(self, symbol: str, side: str, size: float) -> Tuple[bool, str]:
        """
        Check if a position exit is allowed.
        
        Exits are almost always allowed, even in kill-switch mode.
        
        Returns:
            (allowed, reason)
        """
        # Exits are always allowed (we want to be able to flatten)
        return True, "Exits always allowed"
    
    # =========================================================================
    # STATUS & REPORTING
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current risk controller status."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_state": self.risk_state.value,
            "correlation_state": self.correlation_state.value,
            "kill_switch_active": self.kill_switch_active,
            "kill_switch_reason": self.kill_switch_reason.value if self.kill_switch_reason else None,
            "current_drawdown": self.current_drawdown,
            "daily_pnl": self.daily_pnl,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            # [P265] list() first — slicing a deque raises TypeError, so the
            # first caller of get_status() with any event recorded crashed
            "recent_alerts": list(self.risk_events)[-5:] if self.risk_events else []
        }
    
    def get_dashboard_flags(self) -> List[Dict[str, str]]:
        """Get flags to display on dashboard."""
        flags = []
        
        if self.kill_switch_active:
            flags.append({
                "type": "CRITICAL",
                "message": f"KILL-SWITCH: {self.kill_switch_reason.value}",
                "color": "red"
            })
        
        if self.risk_state == RiskState.HALTED:
            flags.append({
                "type": "ERROR",
                "message": f"TRADING HALTED: DD={self.current_drawdown:.1%}",
                "color": "red"
            })
        elif self.risk_state == RiskState.EXIT_ONLY:
            flags.append({
                "type": "WARNING",
                "message": "EXIT-ONLY MODE",
                "color": "orange"
            })
        elif self.risk_state == RiskState.REDUCED:
            flags.append({
                "type": "WARNING",
                "message": f"REDUCED SIZE: DD={self.current_drawdown:.1%}",
                "color": "yellow"
            })
        
        if self.correlation_state == CorrelationState.CRISIS:
            flags.append({
                "type": "CRITICAL",
                "message": "CORRELATION CRISIS",
                "color": "red"
            })
        elif self.correlation_state == CorrelationState.DECOUPLING:
            flags.append({
                "type": "WARNING",
                "message": "CORRELATION BREAKDOWN",
                "color": "orange"
            })
        
        return flags


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_sota_risk_controller: Optional[SOTARiskController] = None


def get_sota_risk_controller(config: SOTARiskConfig = None) -> SOTARiskController:
    """Get or create the global SOTA risk controller."""
    global _sota_risk_controller
    if _sota_risk_controller is None:
        _sota_risk_controller = SOTARiskController(config)
    elif config is not None and _sota_risk_controller.config != config:
        logger.info("[SOTA_RISK] Config changed; refreshing singleton instance")
        _sota_risk_controller = SOTARiskController(config)
    return _sota_risk_controller


def reset_sota_risk_controller():
    """Reset the global SOTA risk controller."""
    global _sota_risk_controller
    _sota_risk_controller = None

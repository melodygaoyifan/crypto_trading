"""
================================================================================
HMATS v6.1.0 - Runtime State Provider
================================================================================
Purpose: Centralized runtime state for dashboard and external consumers

This module provides a singleton that exposes real runtime state:
- Position data
- Risk status
- Kill switch state
- Trade gate status
- Regime information
- Recent trades/alerts
- Shadow ledger summary

Usage:
    from core.runtime_state import get_runtime_state, RuntimeStateProvider
    
    # In main loop after each tick:
    state = get_runtime_state()
    state.update_positions(positions)
    state.update_risk_status(risk_data)
    
    # In dashboard:
    state = get_runtime_state()
    positions = state.get_positions()
    risk_status = state.get_risk_status()

================================================================================
"""

import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PositionSnapshot:
    """Snapshot of a single position."""
    asset: str
    side: str  # "long", "short", "flat"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    tranche: int
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "asset": self.asset,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "tranche": self.tranche,
            "timestamp": self.timestamp,
        }


@dataclass
class RiskSnapshot:
    """Snapshot of risk status."""
    risk_state: str = "UNKNOWN"
    correlation_state: str = "UNKNOWN"
    kill_switch_active: bool = False
    kill_switch_reason: str = ""
    current_drawdown: float = 0.0
    max_drawdown_allowed: float = 0.35    # 35% (was 15%)
    daily_pnl: float = 0.0
    daily_pnl_limit: float = -0.10       # -10% (was -5%)
    peak_equity: float = 0.0
    current_equity: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "risk_state": self.risk_state,
            "correlation_state": self.correlation_state,
            "kill_switch_active": self.kill_switch_active,
            "kill_switch_reason": self.kill_switch_reason,
            "current_drawdown": self.current_drawdown,
            "max_drawdown_allowed": self.max_drawdown_allowed,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_limit": self.daily_pnl_limit,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "timestamp": self.timestamp,
        }


@dataclass
class GateSnapshot:
    """Snapshot of trade gate status."""
    gate_mode: str = "ALLOW"  # ALLOW, NO_TRADE, EXIT_ONLY
    human_override_active: bool = False
    human_override_reason: str = ""
    stale_data_detected: bool = False
    dvol_mode: str = "NORMAL"  # NORMAL, CAUTIOUS, DEFENSIVE
    rate_limit_warning: bool = False
    last_rejection_reason: str = ""
    rejections_last_hour: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "gate_mode": self.gate_mode,
            "human_override_active": self.human_override_active,
            "human_override_reason": self.human_override_reason,
            "stale_data_detected": self.stale_data_detected,
            "dvol_mode": self.dvol_mode,
            "rate_limit_warning": self.rate_limit_warning,
            "last_rejection_reason": self.last_rejection_reason,
            "rejections_last_hour": self.rejections_last_hour,
            "timestamp": self.timestamp,
        }


@dataclass
class RegimeSnapshot:
    """Snapshot of regime information."""
    current_regime: str = "UNKNOWN"
    regime_confidence: float = 0.0
    btc_eth_correlation: float = 0.0
    btc_sol_correlation: float = 0.0
    realized_volatility: float = 0.0
    system_mode: str = "NORMAL"  # NORMAL, OPPORTUNITY, NO_TRADE
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "current_regime": self.current_regime,
            "regime_confidence": self.regime_confidence,
            "btc_eth_correlation": self.btc_eth_correlation,
            "btc_sol_correlation": self.btc_sol_correlation,
            "realized_volatility": self.realized_volatility,
            "system_mode": self.system_mode,
            "timestamp": self.timestamp,
        }


@dataclass
class StrategyWeightSnapshot:
    """Snapshot of strategy weights."""
    weights: Dict[str, float] = field(default_factory=dict)
    last_update: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "weights": self.weights,
            "last_update": self.last_update,
        }


@dataclass
class AlertEntry:
    """Single alert entry."""
    timestamp: str
    alert_type: str
    message: str
    severity: str  # INFO, WARNING, CRITICAL
    asset: Optional[str] = None
    details: Dict = field(default_factory=dict)


class RuntimeStateProvider:
    """
    Centralized runtime state provider.
    
    Thread-safe singleton that aggregates all runtime state
    for dashboard and external consumers.
    """
    
    MAX_ALERTS = 100
    MAX_TRADES = 100
    
    def __init__(self):
        self._lock = threading.RLock()
        
        # State snapshots
        self._positions: Dict[str, PositionSnapshot] = {}
        self._risk: RiskSnapshot = RiskSnapshot()
        self._gate: GateSnapshot = GateSnapshot()
        self._regime: RegimeSnapshot = RegimeSnapshot()
        self._strategy_weights: StrategyWeightSnapshot = StrategyWeightSnapshot()
        
        # Recent events
        self._alerts: deque = deque(maxlen=self.MAX_ALERTS)
        self._recent_trades: deque = deque(maxlen=self.MAX_TRADES)
        self._ledger_summary: Dict = {}
        
        # Metadata
        self._last_tick_time: float = 0.0
        self._tick_count: int = 0
        self._mode: str = "UNKNOWN"  # verify, paper, live
        self._version: str = "6.1.0"
        self._startup_time: float = time.time()
        
        logger.info("[RuntimeState] Provider initialized")
    
    # =========================================================================
    # UPDATE METHODS (called by main loop)
    # =========================================================================
    
    def update_tick(self, tick_count: int, mode: str):
        """Update tick metadata."""
        with self._lock:
            self._tick_count = tick_count
            self._last_tick_time = time.time()
            self._mode = mode
    
    def update_position(
        self,
        asset: str,
        side: str,
        size: float,
        entry_price: float,
        current_price: float,
        unrealized_pnl: float,
        realized_pnl: float,
        tranche: int = 0,
    ):
        """Update position for an asset."""
        with self._lock:
            self._positions[asset] = PositionSnapshot(
                asset=asset,
                side=side,
                size=size,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                tranche=tranche,
            )
    
    def update_risk_status(
        self,
        risk_state: str = None,
        correlation_state: str = None,
        kill_switch_active: bool = None,
        kill_switch_reason: str = None,
        current_drawdown: float = None,
        daily_pnl: float = None,
        peak_equity: float = None,
        current_equity: float = None,
    ):
        """Update risk status."""
        with self._lock:
            if risk_state is not None:
                self._risk.risk_state = risk_state
            if correlation_state is not None:
                self._risk.correlation_state = correlation_state
            if kill_switch_active is not None:
                self._risk.kill_switch_active = kill_switch_active
            if kill_switch_reason is not None:
                self._risk.kill_switch_reason = kill_switch_reason
            if current_drawdown is not None:
                self._risk.current_drawdown = current_drawdown
            if daily_pnl is not None:
                self._risk.daily_pnl = daily_pnl
            if peak_equity is not None:
                self._risk.peak_equity = peak_equity
            if current_equity is not None:
                self._risk.current_equity = current_equity
            self._risk.timestamp = time.time()
    
    def update_gate_status(
        self,
        gate_mode: str = None,
        human_override_active: bool = None,
        human_override_reason: str = None,
        stale_data_detected: bool = None,
        dvol_mode: str = None,
        rate_limit_warning: bool = None,
        last_rejection_reason: str = None,
    ):
        """Update trade gate status."""
        with self._lock:
            if gate_mode is not None:
                self._gate.gate_mode = gate_mode
            if human_override_active is not None:
                self._gate.human_override_active = human_override_active
            if human_override_reason is not None:
                self._gate.human_override_reason = human_override_reason
            if stale_data_detected is not None:
                self._gate.stale_data_detected = stale_data_detected
            if dvol_mode is not None:
                self._gate.dvol_mode = dvol_mode
            if rate_limit_warning is not None:
                self._gate.rate_limit_warning = rate_limit_warning
            if last_rejection_reason is not None:
                self._gate.last_rejection_reason = last_rejection_reason
                self._gate.rejections_last_hour += 1
            self._gate.timestamp = time.time()
    
    def update_regime(
        self,
        current_regime: str = None,
        regime_confidence: float = None,
        btc_eth_correlation: float = None,
        btc_sol_correlation: float = None,
        realized_volatility: float = None,
        system_mode: str = None,
    ):
        """Update regime information."""
        with self._lock:
            if current_regime is not None:
                self._regime.current_regime = current_regime
            if regime_confidence is not None:
                self._regime.regime_confidence = regime_confidence
            if btc_eth_correlation is not None:
                self._regime.btc_eth_correlation = btc_eth_correlation
            if btc_sol_correlation is not None:
                self._regime.btc_sol_correlation = btc_sol_correlation
            if realized_volatility is not None:
                self._regime.realized_volatility = realized_volatility
            if system_mode is not None:
                self._regime.system_mode = system_mode
            self._regime.timestamp = time.time()
    
    def update_strategy_weights(self, weights: Dict[str, float]):
        """Update strategy weights."""
        with self._lock:
            self._strategy_weights.weights = weights.copy()
            self._strategy_weights.last_update = time.time()
    
    def add_alert(
        self,
        alert_type: str,
        message: str,
        severity: str,
        asset: str = None,
        details: Dict = None,
    ):
        """Add an alert."""
        with self._lock:
            alert = AlertEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                alert_type=alert_type,
                message=message,
                severity=severity,
                asset=asset,
                details=details or {},
            )
            self._alerts.append(alert)
    
    def add_trade(self, trade_data: Dict):
        """Add a trade record."""
        with self._lock:
            self._recent_trades.append({
                **trade_data,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
    
    def update_ledger_summary(self, summary: Dict):
        """Update shadow ledger summary."""
        with self._lock:
            self._ledger_summary = summary.copy()
    
    # =========================================================================
    # GET METHODS (called by dashboard)
    # =========================================================================
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get all positions."""
        with self._lock:
            return {
                asset: pos.to_dict()
                for asset, pos in self._positions.items()
            }
    
    def get_risk_status(self) -> Dict:
        """Get risk status."""
        with self._lock:
            return self._risk.to_dict()
    
    def get_gate_status(self) -> Dict:
        """Get trade gate status."""
        with self._lock:
            return self._gate.to_dict()
    
    def get_regime_info(self) -> Dict:
        """Get regime information."""
        with self._lock:
            return self._regime.to_dict()
    
    def get_strategy_weights(self) -> Dict:
        """Get strategy weights."""
        with self._lock:
            return self._strategy_weights.to_dict()
    
    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts."""
        with self._lock:
            alerts = list(self._alerts)[-limit:]
            return [
                {
                    "timestamp": a.timestamp,
                    "type": a.alert_type,
                    "message": a.message,
                    "severity": a.severity,
                    "asset": a.asset,
                    "details": a.details,
                }
                for a in reversed(alerts)
            ]
    
    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Get recent trades."""
        with self._lock:
            return list(self._recent_trades)[-limit:]
    
    def get_ledger_summary(self) -> Dict:
        """Get shadow ledger summary."""
        with self._lock:
            return self._ledger_summary.copy()
    
    def get_metadata(self) -> Dict:
        """Get runtime metadata."""
        with self._lock:
            return {
                "version": self._version,
                "mode": self._mode,
                "tick_count": self._tick_count,
                "last_tick_time": self._last_tick_time,
                "last_tick_age_seconds": time.time() - self._last_tick_time if self._last_tick_time > 0 else -1,
                "uptime_seconds": time.time() - self._startup_time,
            }
    
    def get_full_state(self) -> Dict:
        """Get complete runtime state (for debugging/export)."""
        with self._lock:
            return {
                "metadata": self.get_metadata(),
                "positions": self.get_positions(),
                "risk": self.get_risk_status(),
                "gate": self.get_gate_status(),
                "regime": self.get_regime_info(),
                "strategy_weights": self.get_strategy_weights(),
                "recent_alerts": self.get_recent_alerts(20),
                "recent_trades": self.get_recent_trades(20),
                "ledger_summary": self.get_ledger_summary(),
            }
    
    def export_to_file(self, path: str = "data/runtime_state.json"):
        """Export current state to file (atomic write)."""
        state = self.get_full_state()
        # [P37 2026-04-24] Was open(w, encoding="utf-8")+json.dump → corrupt on crash.
        from core.state_persistence import save_state
        if save_state(path, state):
            logger.debug(f"[RuntimeState] Exported to {path}")
        else:
            logger.warning(f"[RuntimeState] Failed to export to {path}")


# =============================================================================
# SINGLETON
# =============================================================================

_runtime_state: Optional[RuntimeStateProvider] = None
_runtime_state_lock = threading.Lock()


def get_runtime_state() -> RuntimeStateProvider:
    """Get singleton runtime state provider."""
    global _runtime_state
    if _runtime_state is None:
        with _runtime_state_lock:
            if _runtime_state is None:
                _runtime_state = RuntimeStateProvider()
    return _runtime_state


def reset_runtime_state():
    """Reset singleton (for testing)."""
    global _runtime_state
    with _runtime_state_lock:
        _runtime_state = None


if __name__ == "__main__":
    # Test
    state = get_runtime_state()
    
    # Simulate updates
    state.update_tick(1, "paper")
    state.update_position("SOL", "long", 100, 98.5, 102.3, 380.0, 0.0, 2)
    state.update_risk_status(
        risk_state="NORMAL",
        current_drawdown=0.023,
        current_equity=100380.0,
    )
    state.add_alert("TRADE_EXECUTED", "BUY 100 SOL @ 98.5", "INFO", "SOL")
    
    # Export
    print(json.dumps(state.get_full_state(), indent=2, default=str))

"""
================================================================================
RISK MANAGER - Professional Risk Management
================================================================================
Version: 3.1.2
Purpose: Production-grade risk management for crypto trading

Key Features:
- Risk-based position sizing (1% account risk per trade)
- Drawdown control (5%/10%/15% thresholds)
- Correlation adjustment for portfolio risk
- Dynamic regime-based risk parameters
================================================================================
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np
from collections import deque


class DrawdownStatus(Enum):
    """Drawdown severity levels."""
    NORMAL = "NORMAL"           # < 5% - Normal trading
    REDUCED = "REDUCED"         # 5-10% - Reduce position sizes by 50%
    HALTED = "HALTED"           # 10-15% - Stop trading
    CRITICAL = "CRITICAL"       # > 15% - Human review required


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Per-trade risk
    risk_per_trade: float = 0.02          # 2% of account per trade (was 1%)
    max_position_pct: float = 0.40        # 40% max single position (was 20%)

    # Drawdown thresholds - relaxed for profit maximization
    reduce_at_drawdown: float = 0.15      # Reduce at 15% drawdown (was 5%)
    halt_at_drawdown: float = 0.25        # Halt at 25% drawdown (was 10%)
    critical_drawdown: float = 0.35       # Critical at 35% drawdown (was 15%)

    # Position limits
    max_positions: int = 3                # Max concurrent positions
    max_correlated_exposure: float = 0.60 # 60%: crypto inherently high-corr (was 0.50)

    # Daily limits - relaxed for crypto volatility
    max_daily_loss: float = 0.08          # 8% max daily loss (was 3%)
    max_daily_trades: int = 20            # Max trades per day (was 10)
    
    # Correlation
    correlation_threshold: float = 0.7    # Assets above this are "correlated"
    correlation_reduction: float = 0.5    # Reduce size by 50% for correlated


@dataclass
class PositionRisk:
    """Risk metrics for a single position."""
    symbol: str
    entry_price: float
    current_price: float
    stop_loss: float
    position_size: float
    side: str  # "long" or "short"
    unrealized_pnl: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    
    def calculate_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.position_size
        else:
            return (self.entry_price - self.current_price) * self.position_size


def get_regime_params(regime: str) -> Dict[str, float]:
    """
    Get risk parameters adjusted for market regime.
    
    Different regimes require different risk approaches:
    - Trending: Higher position sizes, wider stops
    - Ranging: Smaller positions, tighter stops
    - Volatile: Much smaller positions, very wide stops
    """
    regime_params = {
        "bull_trend": {
            "risk_multiplier": 1.0,
            "stop_multiplier": 1.0,
            "max_positions": 3
        },
        "bear_trend": {
            "risk_multiplier": 0.8,
            "stop_multiplier": 1.2,
            "max_positions": 2
        },
        "sideways_chop": {
            "risk_multiplier": 0.6,
            "stop_multiplier": 0.8,
            "max_positions": 2
        },
        "extreme_volatility": {
            "risk_multiplier": 0.3,
            "stop_multiplier": 2.0,
            "max_positions": 1
        },
        "breakout_up": {
            "risk_multiplier": 1.2,
            "stop_multiplier": 1.5,
            "max_positions": 3
        },
        "breakout_down": {
            "risk_multiplier": 0.8,
            "stop_multiplier": 1.5,
            "max_positions": 2
        },
        "accumulation": {
            "risk_multiplier": 0.7,
            "stop_multiplier": 1.0,
            "max_positions": 2
        },
        "distribution": {
            "risk_multiplier": 0.5,
            "stop_multiplier": 1.2,
            "max_positions": 1
        },
    }
    
    return regime_params.get(regime, {
        "risk_multiplier": 0.5,
        "stop_multiplier": 1.0,
        "max_positions": 2
    })


class RiskManager:
    """
    Production-grade risk management system.
    
    Key Responsibilities:
    1. Position sizing based on account risk
    2. Drawdown monitoring and control
    3. Correlation-adjusted exposure
    4. Regime-based risk adjustment
    
    Position Sizing Math:
    ====================
    Risk Amount = Account Balance × Risk Per Trade
    Position Size = Risk Amount / (Entry Price - Stop Loss)
    
    Example:
    - Account: $10,000
    - Risk per trade: 1% = $100
    - Entry: $45,000
    - Stop Loss: $44,000 (2.2% below entry)
    - Risk distance: $1,000
    - Position Size: $100 / $1,000 = 0.1 BTC
    
    This ensures we never lose more than 1% of account on any single trade.
    """
    
    def __init__(self, config: Optional[RiskConfig] = None,
                 correlation_source=None, sota_risk_ref=None):
        """Initialize RiskManager with configuration.

        Args:
            config: Risk management configuration
            correlation_source: Optional CorrelationRealtimeController for live
                correlation data. Falls back to hardcoded matrix if None.
            sota_risk_ref: Optional SOTARiskController for unified DD tracking.
                Can also be set later via set_sota_risk_ref().
        """
        self.config = config or RiskConfig()
        self.logger = logging.getLogger(__name__)

        # R1: Live correlation source (CorrelationRealtimeController or None)
        self._correlation_source = correlation_source

        # R2: SOTA risk controller reference for unified DD tracking
        self._sota_risk = sota_risk_ref

        # Account state
        self.initial_balance: float = 0.0
        self.current_balance: float = 0.0
        self.peak_balance: float = 0.0

        # Position tracking
        self.positions: Dict[str, PositionRisk] = {}

        # Performance tracking
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.last_trade_date: Optional[datetime] = None

        # Drawdown tracking
        self.current_drawdown: float = 0.0
        self.max_drawdown: float = 0.0
        self.drawdown_status: DrawdownStatus = DrawdownStatus.NORMAL

        # Trade history for correlation
        self.trade_history: deque = deque(maxlen=100)

        # Correlation matrix (simplified) - fallback when no live source
        self.correlation_matrix: Dict[Tuple[str, str], float] = {
            ("BTC/USDT:USDT", "ETH/USDT:USDT"): 0.85,
            ("ETH/USDT:USDT", "BTC/USDT:USDT"): 0.85,
            ("BTC/USDT:USDT", "SOL/USDT:USDT"): 0.75,
            ("SOL/USDT:USDT", "BTC/USDT:USDT"): 0.75,
            ("ETH/USDT:USDT", "SOL/USDT:USDT"): 0.80,
            ("SOL/USDT:USDT", "ETH/USDT:USDT"): 0.80,
        }
    
    # R2: Deferred wiring for SOTA risk controller
    def set_sota_risk_ref(self, ref) -> None:
        """Set reference to SOTARiskController for unified DD tracking.

        Can be called after __init__ since p0_integrator may be created later.
        """
        self._sota_risk = ref
        self.logger.info("[RISK_DELEG] SOTARiskController reference set")

    @staticmethod
    def _strip_to_base(symbol: str) -> str:
        """Extract base asset from full ccxt symbol. 'BTC/USDT:USDT' -> 'BTC'"""
        return symbol.split("/")[0] if "/" in symbol else symbol

    def initialize(self, balance: float) -> None:
        """Initialize risk manager with account balance."""
        self.initial_balance = balance
        self.current_balance = balance
        self.peak_balance = balance
        self.logger.info(f"RiskManager initialized with balance: ${balance:,.2f}")
    
    def update_balance(self, new_balance: float) -> None:
        """Update current balance and recalculate drawdown."""
        self.current_balance = new_balance

        # Update peak
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

        # R2: Delegate DD computation to SOTA when available
        if self._sota_risk:
            try:
                self._sota_risk.update_equity(new_balance)
                self.current_drawdown = self._sota_risk.current_drawdown
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            except Exception as e:
                self.logger.warning(f"[RISK_DELEG] SOTA DD sync failed: {e}")
                # Fall through to legacy computation
                if self.peak_balance > 0:
                    self.current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
                    self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        else:
            # Legacy standalone mode
            if self.peak_balance > 0:
                self.current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # Update status
        self._update_drawdown_status()
    
    def _update_drawdown_status(self) -> None:
        """Update drawdown status based on current drawdown AND SOTA risk state."""
        # Drawdown-based status (legacy logic)
        if self.current_drawdown >= self.config.critical_drawdown:
            dd_status = DrawdownStatus.CRITICAL
        elif self.current_drawdown >= self.config.halt_at_drawdown:
            dd_status = DrawdownStatus.HALTED
        elif self.current_drawdown >= self.config.reduce_at_drawdown:
            dd_status = DrawdownStatus.REDUCED
        else:
            dd_status = DrawdownStatus.NORMAL

        # R2: SOTA override - take the MORE restrictive of the two
        sota_status = self._get_sota_drawdown_status()
        if sota_status is not None:
            severity = [DrawdownStatus.NORMAL, DrawdownStatus.REDUCED,
                        DrawdownStatus.HALTED, DrawdownStatus.CRITICAL]
            dd_idx = severity.index(dd_status)
            sota_idx = severity.index(sota_status)
            self.drawdown_status = severity[max(dd_idx, sota_idx)]
            if sota_idx > dd_idx:
                self.logger.debug(
                    f"[RISK_DELEG] SOTA override: {dd_status.value}->{self.drawdown_status.value}"
                )
        else:
            self.drawdown_status = dd_status

    def _get_sota_drawdown_status(self) -> Optional[DrawdownStatus]:
        """Map SOTARiskController.risk_state to DrawdownStatus."""
        if self._sota_risk is None:
            return None
        try:
            from risk.sota_risk_controller import RiskState
            state = self._sota_risk.risk_state
            mapping = {
                RiskState.NORMAL: DrawdownStatus.NORMAL,
                RiskState.REDUCED: DrawdownStatus.REDUCED,
                RiskState.EXIT_ONLY: DrawdownStatus.HALTED,
                RiskState.HALTED: DrawdownStatus.HALTED,
                RiskState.KILL_SWITCH: DrawdownStatus.CRITICAL,
            }
            return mapping.get(state)
        except Exception:
            return None
    
    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on risk limits.
        
        Returns:
            Tuple of (allowed, reason)
        """
        # Check drawdown status
        if self.drawdown_status == DrawdownStatus.CRITICAL:
            return False, f"CRITICAL drawdown ({self.current_drawdown:.1%}) - Human review required"
        
        if self.drawdown_status == DrawdownStatus.HALTED:
            return False, f"Trading HALTED due to drawdown ({self.current_drawdown:.1%})"
        
        # Check position limits
        if len(self.positions) >= self.config.max_positions:
            return False, f"Max positions reached ({self.config.max_positions})"
        
        # Check daily limits
        if self._is_new_trading_day():
            self._reset_daily_counters()
        
        if self.daily_trades >= self.config.max_daily_trades:
            return False, f"Max daily trades reached ({self.config.max_daily_trades})"
        
        if self.current_balance > 0 and self.daily_pnl <= -self.current_balance * self.config.max_daily_loss:
            return False, f"Max daily loss reached ({self.config.max_daily_loss:.1%})"
        
        return True, "Trading allowed"
    
    # =========================================================================
    # v5.1.0-HARDENED: check_trade_allowed() interface for main.py compatibility
    # =========================================================================
    
    def check_trade_allowed(
        self,
        symbol: str,
        direction: float,
        size: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Check if a specific trade is allowed.
        
        v5.1.0-HARDENED: Glue-layer fix - main.py calls check_trade_allowed()
        but RiskManager only had can_trade().
        
        Args:
            symbol: Asset symbol
            direction: Trade direction (+1 long, -1 short)
            size: Target exposure/position size
            **kwargs: Additional context
            
        Returns:
            Dict with 'approved' bool and 'reason' string
        """
        # First check basic trading permission
        can_trade, reason = self.can_trade()
        if not can_trade:
            return {"approved": False, "reason": reason}
        
        # Check position-specific constraints
        side = "long" if direction > 0 else "short"
        
        # Check if we already have a position in this symbol
        if symbol in self.positions:
            existing = self.positions[symbol]
            # Check if trying to increase in same direction (allow)
            # or reverse (need to close first)
            if (existing.side == "long" and direction < 0) or \
               (existing.side == "short" and direction > 0):
                # Allow reversal/close
                pass
        
        # Check correlation exposure
        correlated_exposure = self._calculate_correlated_exposure(symbol)
        if correlated_exposure + abs(size) > self.config.max_correlated_exposure:
            return {
                "approved": False,
                "reason": f"Correlated exposure would exceed limit ({correlated_exposure + abs(size):.1%} > {self.config.max_correlated_exposure:.1%})"
            }
        
        # Check size against max position
        max_size = self.current_balance * self.config.max_position_pct
        if abs(size) > max_size:
            return {
                "approved": True,
                "reason": f"Size capped to max position ({self.config.max_position_pct:.0%})",
                "adjusted_size": max_size * (1 if direction > 0 else -1)
            }
        
        return {"approved": True, "reason": "Trade approved"}
    
    def _calculate_correlated_exposure(self, symbol: str) -> float:
        """Calculate total exposure in correlated assets."""
        # Simplified correlation check for crypto (all correlated)
        total = 0.0
        for pos in self.positions.values():
            total += abs(pos.position_size * pos.current_price)
        return total / max(1.0, self.current_balance)
    
    def calculate_position_size(self,
                               symbol: str,
                               entry_price: float,
                               stop_loss: float,
                               side: str,
                               regime: str = "unknown") -> Dict[str, Any]:
        """
        Calculate risk-based position size.
        
        This is the core position sizing algorithm that ensures
        we never risk more than the configured percentage per trade.
        
        Args:
            symbol: Trading pair
            entry_price: Planned entry price
            stop_loss: Stop loss price
            side: "long" or "short"
            regime: Current market regime
            
        Returns:
            Dict with position_size, risk_amount, and other details
        """
        # Validate inputs
        if entry_price <= 0 or stop_loss <= 0:
            return {"error": "Invalid prices", "position_size": 0}
        
        # Calculate risk distance
        if side == "long":
            risk_distance = entry_price - stop_loss
            if risk_distance <= 0:
                return {"error": "Stop loss must be below entry for long", "position_size": 0}
        else:
            risk_distance = stop_loss - entry_price
            if risk_distance <= 0:
                return {"error": "Stop loss must be above entry for short", "position_size": 0}
        
        # Risk per trade as percentage of account
        risk_pct = risk_distance / entry_price
        
        # Get regime-adjusted risk
        regime_params = get_regime_params(regime)
        risk_multiplier = regime_params["risk_multiplier"]
        
        # Apply drawdown reduction
        if self.drawdown_status == DrawdownStatus.REDUCED:
            risk_multiplier *= 0.5
            self.logger.warning("Position size reduced due to drawdown")
        
        # Calculate base risk amount
        base_risk_pct = self.config.risk_per_trade * risk_multiplier
        risk_amount = self.current_balance * base_risk_pct
        
        # Calculate position size
        position_size = risk_amount / risk_distance
        
        # Apply correlation adjustment (regime-scaled)
        correlation_factor = self._get_correlation_adjustment(symbol, regime=regime)
        position_size *= correlation_factor
        
        # Apply max position limit
        max_position_value = self.current_balance * self.config.max_position_pct
        max_position_size = max_position_value / entry_price
        position_size = min(position_size, max_position_size)
        
        # Recalculate actual risk
        actual_risk = position_size * risk_distance
        actual_risk_pct = actual_risk / self.current_balance
        
        return {
            "position_size": position_size,
            "position_value": position_size * entry_price,
            "risk_amount": actual_risk,
            "risk_pct": actual_risk_pct,
            "risk_distance": risk_distance,
            "risk_distance_pct": risk_pct,
            "regime_multiplier": risk_multiplier,
            "correlation_factor": correlation_factor,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "side": side,
            "regime": regime,
        }
    
    def _get_pair_correlation(self, sym_a: str, sym_b: str) -> float:
        """Get correlation between two symbols.

        R1: Uses live correlation source if available, falls back to hardcoded.
        """
        # Try live correlation source first
        if self._correlation_source is not None:
            try:
                matrix = self._correlation_source.compute_correlation_matrix("short")
                if matrix is not None:
                    base_a = self._strip_to_base(sym_a)
                    base_b = self._strip_to_base(sym_b)
                    # [P47 2026-04-25] Was `matrix.get(base_a, base_b)` — dict.get(k, d)
                    # has TWO args (key, default), so `base_b` was being passed as the
                    # DEFAULT value, not as a second key. Live correlation lookup ALWAYS
                    # returned `base_b` (a string like "ETH"), `live_corr is not None`
                    # was True, then `f"{live_corr:.3f}"` would have raised TypeError on
                    # the string format — but the surrounding `try: ... except: pass`
                    # swallowed it. Net: live correlation has been silently broken
                    # since this code was written, always falling back to the static
                    # 6-pair hardcoded matrix below.
                    live_corr = matrix.get((base_a, base_b))
                    if live_corr is not None:
                        try:
                            live_corr_f = float(live_corr)
                        except (TypeError, ValueError):
                            live_corr_f = None
                        if live_corr_f is not None:
                            self.logger.debug(
                                f"[CORR] Live={live_corr_f:.3f} for {sym_a}/{sym_b}"
                            )
                            return live_corr_f
            except Exception:
                pass  # Fall through to hardcoded

        # Fallback: hardcoded static matrix
        corr = self.correlation_matrix.get((sym_a, sym_b), 0.0)
        self.logger.debug(f"[CORR] Fallback static={corr:.3f} for {sym_a}/{sym_b}")
        return corr

    def _get_correlation_adjustment(self, symbol: str, regime: str = "") -> float:
        """
        Adjust position size based on correlation with existing positions.

        Correlations are scaled by regime: in stress regimes all assets
        move together (correlation -> 1.0), reducing diversification benefit.
        R1: Now uses live correlation when available.
        """
        if not self.positions:
            return 1.0

        # Regime scaling: stress regimes have higher realized correlation
        regime_corr_scale = {
            "PANIC_SELLOFF": 1.15,       # Correlations spike in panic
            "EXTREME_VOLATILITY": 1.10,
            "VOLATILE_CHOP": 1.05,
            "MOMENTUM_RALLY": 0.95,      # Slight decorrelation in rallies
            "QUIET_ACCUMULATION": 0.90,
            "WEAK_CONSOLIDATION": 1.0,
        }.get(regime, 1.0)

        max_correlation = 0.0
        for existing_symbol in self.positions.keys():
            base_corr = self._get_pair_correlation(symbol, existing_symbol)
            adjusted_corr = min(base_corr * regime_corr_scale, 0.99)
            max_correlation = max(max_correlation, adjusted_corr)

        if max_correlation >= self.config.correlation_threshold:
            return self.config.correlation_reduction

        return 1.0
    
    def register_position(self, position: PositionRisk) -> None:
        """Register a new position."""
        self.positions[position.symbol] = position
        self.daily_trades += 1
        self.last_trade_date = datetime.now(timezone.utc)
        
        self.trade_history.append({
            "symbol": position.symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "position_size": position.position_size,
            "timestamp": datetime.now(timezone.utc)
        })
        
        self.logger.info(
            f"Registered position: {position.symbol} {position.side} "
            f"@ ${position.entry_price:,.2f} x {position.position_size:.4f}"
        )
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[float]:
        """Close a position and return realized P&L."""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        position.current_price = exit_price
        pnl = position.calculate_pnl()
        
        # Update daily P&L
        self.daily_pnl += pnl
        
        # Update balance
        self.update_balance(self.current_balance + pnl)
        
        # Remove position
        del self.positions[symbol]
        
        self.logger.info(
            f"Closed position: {symbol} @ ${exit_price:,.2f} | "
            f"P&L: ${pnl:,.2f} ({pnl/self.current_balance:.1%})"
        )
        
        return pnl
    
    def update_position_price(self, symbol: str, current_price: float) -> None:
        """Update current price for a position."""
        if symbol in self.positions:
            self.positions[symbol].current_price = current_price
            self.positions[symbol].unrealized_pnl = self.positions[symbol].calculate_pnl()
    
    def get_total_exposure(self) -> float:
        """Get total portfolio exposure as percentage of account."""
        total_value = sum(
            p.position_size * p.current_price for p in self.positions.values()
        )
        return total_value / self.current_balance if self.current_balance > 0 else 0
    
    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions."""
        return sum(p.calculate_pnl() for p in self.positions.values())
    
    def _is_new_trading_day(self) -> bool:
        """Check if it's a new trading day (UTC boundary)."""
        if self.last_trade_date is None:
            return True
        return datetime.now(timezone.utc).date() > self.last_trade_date.date()

    def record_realized_pnl(self, pnl: float) -> None:
        """Record realized PnL from a closed position. Updates daily_pnl for loss limit checks."""
        self.daily_pnl += pnl
        self.daily_trades += 1
        self.logger.info(f"[RISK] Realized PnL: ${pnl:+,.2f} | daily_pnl=${self.daily_pnl:+,.2f} | trades={self.daily_trades}")

    def _reset_daily_counters(self) -> None:
        """Reset daily counters at UTC midnight."""
        prev_pnl = self.daily_pnl
        prev_trades = self.daily_trades
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.logger.info(
            f"[DAILY_RESET] UTC midnight reset: "
            f"prev_daily_pnl=${prev_pnl:,.2f}, prev_trades={prev_trades}, "
            f"new_date={datetime.now(timezone.utc).date()}"
        )
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        return {
            "balance": {
                "initial": self.initial_balance,
                "current": self.current_balance,
                "peak": self.peak_balance,
            },
            "drawdown": {
                "current": self.current_drawdown,
                "max": self.max_drawdown,
                "status": self.drawdown_status.value,
            },
            "positions": {
                "count": len(self.positions),
                "max": self.config.max_positions,
                "exposure": self.get_total_exposure(),
                "unrealized_pnl": self.get_total_unrealized_pnl(),
            },
            "daily": {
                "pnl": self.daily_pnl,
                "trades": self.daily_trades,
                "max_trades": self.config.max_daily_trades,
            },
            "config": {
                "risk_per_trade": self.config.risk_per_trade,
                "max_position_pct": self.config.max_position_pct,
                "reduce_at_dd": self.config.reduce_at_drawdown,
                "halt_at_dd": self.config.halt_at_drawdown,
            }
        }


if __name__ == "__main__":
    # Test risk manager
    rm = RiskManager()
    rm.initialize(10000)
    
    # Test position sizing
    result = rm.calculate_position_size(
        symbol="BTC/USDT:USDT",
        entry_price=45000,
        stop_loss=44000,
        side="long",
        regime="bull_trend"
    )
    
    print("Position Sizing Test:")
    print(f"  Entry: ${result['entry_price']:,.2f}")
    print(f"  Stop Loss: ${result['stop_loss']:,.2f}")
    print(f"  Risk Distance: ${result['risk_distance']:,.2f} ({result['risk_distance_pct']:.2%})")
    print(f"  Position Size: {result['position_size']:.4f} BTC")
    print(f"  Position Value: ${result['position_value']:,.2f}")
    print(f"  Risk Amount: ${result['risk_amount']:,.2f}")
    print(f"  Risk %: {result['risk_pct']:.2%}")
    print(f"  Regime Multiplier: {result['regime_multiplier']}")
    print(f"  Correlation Factor: {result['correlation_factor']}")

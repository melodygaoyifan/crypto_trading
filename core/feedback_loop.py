"""
================================================================================
HMATS v5.1.0 - FEEDBACK LOOP
================================================================================
Version: 5.1.0-HARDENED
Purpose: Feedback loop for execution -> risk -> agents

This module closes the loop by:
1. Recording execution outcomes
2. Attributing PnL to strategies/signals
3. Tracking strategy aging/decay
4. Feeding results back to agents and risk

FEEDBACK CHAIN:
    Execution -> FeedbackLoop -> Risk (position updates)
                            -> Agents (signal performance)
                            -> Analytics (PnL attribution)

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Callable
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from collections import deque
import json

logger = logging.getLogger(__name__)


# =============================================================================
# FEEDBACK TYPES
# =============================================================================

class FeedbackType(Enum):
    """Types of feedback events."""
    TRADE_OPENED = auto()
    TRADE_CLOSED = auto()
    TRADE_MODIFIED = auto()
    PNL_UPDATE = auto()
    SIGNAL_OUTCOME = auto()
    STRATEGY_DECAY = auto()
    RISK_ADJUSTMENT = auto()


@dataclass
class FeedbackEvent:
    """A feedback event."""
    timestamp: datetime
    event_type: FeedbackType
    
    # Trade info
    trade_id: str = ""
    asset: str = ""
    
    # Signal info
    signal_source: str = ""
    signal_direction: float = 0.0
    signal_confidence: float = 0.0
    
    # Execution info
    entry_price: float = 0.0
    exit_price: float = 0.0
    size: float = 0.0
    
    # Outcome
    pnl_realized: float = 0.0
    pnl_unrealized: float = 0.0
    pnl_bps: float = 0.0
    bars_held: int = 0
    
    # Attribution
    entry_alpha_bps: float = 0.0
    execution_alpha_bps: float = 0.0
    exit_alpha_bps: float = 0.0
    
    # Mode/regime at entry
    mode_at_entry: str = ""
    regime_at_entry: str = ""
    phase_at_entry: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.name,
            "trade_id": self.trade_id,
            "asset": self.asset,
            "signal_source": self.signal_source,
            "pnl_realized": self.pnl_realized,
            "pnl_bps": self.pnl_bps,
            "bars_held": self.bars_held,
            "mode_at_entry": self.mode_at_entry,
            "metadata": self.metadata,
        }


# =============================================================================
# STRATEGY PERFORMANCE
# =============================================================================

@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy/signal source."""
    strategy_id: str
    
    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # PnL
    total_pnl_bps: float = 0.0
    avg_win_bps: float = 0.0
    avg_loss_bps: float = 0.0
    max_win_bps: float = 0.0
    max_loss_bps: float = 0.0
    
    # Recent performance (decay window)
    recent_trades: int = 0
    recent_pnl_bps: float = 0.0
    
    # Win rate
    win_rate: float = 0.0
    
    # Aging
    days_since_creation: int = 0
    days_since_last_win: int = 0
    
    # Confidence adjustment
    confidence_multiplier: float = 1.0
    
    def update_win_rate(self):
        """Update win rate calculation."""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
    
    def calculate_confidence_multiplier(self) -> float:
        """Calculate confidence multiplier based on performance."""
        base = 1.0
        
        # Win rate adjustment
        if self.win_rate > 0.6:
            base *= 1.1
        elif self.win_rate < 0.4:
            base *= 0.8
        
        # Recent performance
        if self.recent_trades > 5:
            recent_avg = self.recent_pnl_bps / self.recent_trades
            if recent_avg > 10:
                base *= 1.1
            elif recent_avg < -10:
                base *= 0.85
        
        # Decay (days since last win)
        if self.days_since_last_win > 30:
            decay = 0.95 ** ((self.days_since_last_win - 30) / 7)
            base *= max(0.5, decay)
        
        self.confidence_multiplier = max(0.3, min(1.5, base))
        return self.confidence_multiplier


# =============================================================================
# OPEN POSITION TRACKING
# =============================================================================

@dataclass
class OpenPosition:
    """Tracked open position."""
    trade_id: str
    asset: str
    entry_time: datetime
    entry_price: float
    size: float
    direction: float
    
    # Signal info
    signal_source: str
    signal_direction: float
    signal_confidence: float
    
    # Mode/regime at entry
    mode_at_entry: str
    regime_at_entry: str
    phase_at_entry: str
    
    # Current state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    bars_held: int = 0
    
    def update_price(self, price: float):
        """Update current price and unrealized PnL."""
        self.current_price = price
        if self.direction > 0:
            self.unrealized_pnl = (price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.size


# =============================================================================
# FEEDBACK LOOP
# =============================================================================

class FeedbackLoop:
    """
    Feedback loop for execution -> risk -> agents.
    
    Responsibilities:
    1. Track all open positions
    2. Record trade outcomes
    3. Attribute PnL to strategies
    4. Calculate strategy aging/decay
    5. Provide feedback to agents
    """
    
    def __init__(
        self,
        max_history: int = 1000,
        decay_window_days: int = 30
    ):
        """Initialize feedback loop."""
        self.max_history = max_history
        self.decay_window_days = decay_window_days
        
        # Position tracking
        self._open_positions: Dict[str, OpenPosition] = {}
        
        # Trade history
        self._trade_history: deque = deque(maxlen=max_history)
        
        # Strategy performance
        self._strategy_performance: Dict[str, StrategyPerformance] = {}
        
        # Feedback subscribers
        self._subscribers: Dict[FeedbackType, List[Callable]] = {
            t: [] for t in FeedbackType
        }
        
        # Statistics
        self._total_trades: int = 0
        self._total_pnl: float = 0.0
        self._today_trades: int = 0
        self._today_pnl: float = 0.0
        self._last_reset: datetime = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
        
        logger.info("FeedbackLoop initialized")
    
    # =========================================================================
    # POSITION TRACKING
    # =========================================================================
    
    def open_position(
        self,
        trade_id: str,
        asset: str,
        entry_price: float,
        size: float,
        direction: float,
        signal_source: str = "",
        signal_direction: float = 0.0,
        signal_confidence: float = 0.0,
        mode: str = "",
        regime: str = "",
        phase: str = ""
    ) -> FeedbackEvent:
        """Record a new position."""
        position = OpenPosition(
            trade_id=trade_id,
            asset=asset,
            entry_time=datetime.now(timezone.utc),
            entry_price=entry_price,
            size=size,
            direction=direction,
            signal_source=signal_source,
            signal_direction=signal_direction,
            signal_confidence=signal_confidence,
            mode_at_entry=mode,
            regime_at_entry=regime,
            phase_at_entry=phase,
            current_price=entry_price,
        )
        
        self._open_positions[trade_id] = position
        
        event = FeedbackEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=FeedbackType.TRADE_OPENED,
            trade_id=trade_id,
            asset=asset,
            signal_source=signal_source,
            signal_direction=signal_direction,
            signal_confidence=signal_confidence,
            entry_price=entry_price,
            size=size,
            mode_at_entry=mode,
            regime_at_entry=regime,
            phase_at_entry=phase,
        )
        
        self._notify_subscribers(event)
        logger.info(f"[FEEDBACK] Opened position: {trade_id} {asset} @ {entry_price}")
        
        return event
    
    def close_position(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = ""
    ) -> Optional[FeedbackEvent]:
        """Close a position and record outcome."""
        if trade_id not in self._open_positions:
            logger.warning(f"[FEEDBACK] Position not found: {trade_id}")
            return None
        
        position = self._open_positions.pop(trade_id)
        
        # Calculate PnL
        if position.direction > 0:
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size
        
        pnl_bps = (pnl / (position.entry_price * position.size)) * 10000 if position.entry_price > 0 and position.size > 0 else 0
        
        # Calculate bars held (simplified - assumes 4H bars)
        time_held = datetime.now(timezone.utc) - position.entry_time
        bars_held = int(time_held.total_seconds() / 14400) + 1
        
        # Create feedback event
        event = FeedbackEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=FeedbackType.TRADE_CLOSED,
            trade_id=trade_id,
            asset=position.asset,
            signal_source=position.signal_source,
            signal_direction=position.signal_direction,
            signal_confidence=position.signal_confidence,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            pnl_realized=pnl,
            pnl_bps=pnl_bps,
            bars_held=bars_held,
            mode_at_entry=position.mode_at_entry,
            regime_at_entry=position.regime_at_entry,
            phase_at_entry=position.phase_at_entry,
            metadata={"exit_reason": exit_reason},
        )
        
        # Record to history
        self._trade_history.append(event)
        
        # Update statistics
        self._total_trades += 1
        self._total_pnl += pnl
        self._check_day_reset()
        self._today_trades += 1
        self._today_pnl += pnl
        
        # Update strategy performance
        self._update_strategy_performance(event)
        
        # Notify subscribers
        self._notify_subscribers(event)
        
        logger.info(
            f"[FEEDBACK] Closed position: {trade_id} {position.asset} | "
            f"PnL: {pnl:.2f} ({pnl_bps:.1f}bps) | Bars: {bars_held}"
        )
        
        return event
    
    def update_position_price(self, asset: str, price: float):
        """Update current price for all positions in asset."""
        for position in self._open_positions.values():
            if position.asset == asset:
                position.update_price(price)
                position.bars_held += 1
    
    def increment_bars(self):
        """Increment bars held for all positions."""
        for position in self._open_positions.values():
            position.bars_held += 1
    
    # =========================================================================
    # STRATEGY PERFORMANCE
    # =========================================================================
    
    def _update_strategy_performance(self, event: FeedbackEvent):
        """Update strategy performance from trade event."""
        strategy_id = event.signal_source or "unknown"
        
        if strategy_id not in self._strategy_performance:
            self._strategy_performance[strategy_id] = StrategyPerformance(
                strategy_id=strategy_id
            )
        
        perf = self._strategy_performance[strategy_id]
        
        # Update counts
        perf.total_trades += 1
        perf.total_pnl_bps += event.pnl_bps
        
        if event.pnl_bps > 0:
            perf.winning_trades += 1
            perf.avg_win_bps = (
                (perf.avg_win_bps * (perf.winning_trades - 1) + event.pnl_bps) 
                / perf.winning_trades
            )
            perf.max_win_bps = max(perf.max_win_bps, event.pnl_bps)
            perf.days_since_last_win = 0
        else:
            perf.losing_trades += 1
            perf.avg_loss_bps = (
                (perf.avg_loss_bps * (perf.losing_trades - 1) + event.pnl_bps)
                / perf.losing_trades
            )
            perf.max_loss_bps = min(perf.max_loss_bps, event.pnl_bps)
        
        # Update recent window
        perf.recent_trades += 1
        perf.recent_pnl_bps += event.pnl_bps
        
        # Update derived metrics
        perf.update_win_rate()
        perf.calculate_confidence_multiplier()
    
    def get_strategy_performance(self, strategy_id: str) -> Optional[StrategyPerformance]:
        """Get performance for a strategy."""
        return self._strategy_performance.get(strategy_id)
    
    def get_all_strategy_performance(self) -> Dict[str, StrategyPerformance]:
        """Get performance for all strategies."""
        return self._strategy_performance.copy()
    
    def get_confidence_multiplier(self, strategy_id: str) -> float:
        """Get confidence multiplier for a strategy (for signal adjustment)."""
        perf = self._strategy_performance.get(strategy_id)
        if perf:
            return perf.confidence_multiplier
        return 1.0  # Default
    
    # =========================================================================
    # SUBSCRIBERS
    # =========================================================================
    
    def subscribe(self, event_type: FeedbackType, callback: Callable):
        """Subscribe to feedback events."""
        self._subscribers[event_type].append(callback)
    
    def _notify_subscribers(self, event: FeedbackEvent):
        """Notify all subscribers of event.

        [P50 2026-04-25] Was logger.warning() then continued — same shape as
        P15 (silent record_trade_completed AttributeError swallowed). Track
        consecutive failures per callback, escalate to ERROR after 5 in a row,
        and log the callback's qualname so operator can identify which subscriber
        is broken — not just "Subscriber error: ...".
        """
        for callback in self._subscribers[event.event_type]:
            try:
                callback(event)
                # Reset failure counter on success.
                _qn = getattr(callback, "__qualname__", str(callback))
                if hasattr(self, "_subscriber_failures") and _qn in self._subscriber_failures:
                    self._subscriber_failures.pop(_qn, None)
            except Exception as e:
                _qn = getattr(callback, "__qualname__", str(callback))
                if not hasattr(self, "_subscriber_failures"):
                    self._subscriber_failures = {}
                self._subscriber_failures[_qn] = self._subscriber_failures.get(_qn, 0) + 1
                _n = self._subscriber_failures[_qn]
                if _n >= 5:
                    logger.error(
                        f"[FEEDBACK] Subscriber {_qn} has failed {_n}× in a row "
                        f"(latest: {type(e).__name__}: {e}). Likely AttributeError "
                        f"from method-name mismatch (P15-shape). Check the callback's "
                        f"target type for the expected method."
                    )
                else:
                    logger.warning(f"[FEEDBACK] Subscriber {_qn} error ({_n}× consecutive): {e}")
    
    # =========================================================================
    # TICK RESULT RECORDING
    # =========================================================================
    
    def record(self, tick_result):
        """
        Record a tick result (from runtime spine).
        
        This is the main integration point with RuntimeSpine.
        """
        if tick_result.execution_result and tick_result.execution_result.success:
            exec_result = tick_result.execution_result
            fused = tick_result.fused_signal
            regime = tick_result.regime_state
            
            # Determine if opening or closing
            if fused and exec_result.new_exposure > 0:
                # Opening or adding to position
                trade_id = exec_result.order_id or f"T{tick_result.tick_id}"
                self.open_position(
                    trade_id=trade_id,
                    asset=exec_result.asset,
                    entry_price=exec_result.filled_price,
                    size=exec_result.filled_size,
                    direction=fused.direction,
                    signal_source=fused.authority_source,
                    signal_direction=fused.direction,
                    signal_confidence=fused.confidence,
                    mode=tick_result.system_mode.value,
                    regime=regime.regime if regime else "",
                    phase=regime.phase.value if regime else "",
                )
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def _check_day_reset(self):
        """Check if we need to reset daily stats."""
        now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
        if now > self._last_reset:
            self._today_trades = 0
            self._today_pnl = 0.0
            self._last_reset = now
    
    def get_statistics(self) -> Dict:
        """Get feedback loop statistics."""
        self._check_day_reset()
        return {
            "total_trades": self._total_trades,
            "total_pnl": self._total_pnl,
            "today_trades": self._today_trades,
            "today_pnl": self._today_pnl,
            "open_positions": len(self._open_positions),
            "history_size": len(self._trade_history),
            "strategies_tracked": len(self._strategy_performance),
        }
    
    def get_open_positions(self) -> Dict[str, OpenPosition]:
        """Get all open positions."""
        return self._open_positions.copy()
    
    def get_recent_trades(self, n: int = 10) -> List[FeedbackEvent]:
        """Get n most recent trades."""
        return list(self._trade_history)[-n:]


# =============================================================================
# SINGLETON
# =============================================================================

_feedback_loop: Optional[FeedbackLoop] = None


def get_feedback_loop() -> FeedbackLoop:
    """Get feedback loop singleton."""
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = FeedbackLoop()
    return _feedback_loop


def reset_feedback_loop():
    """Reset feedback loop singleton."""
    global _feedback_loop
    _feedback_loop = None


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    loop = FeedbackLoop()
    
    # Test opening a position
    print("\n=== Opening position ===")
    loop.open_position(
        trade_id="T001",
        asset="SOL",
        entry_price=185.0,
        size=100.0,
        direction=1.0,
        signal_source="quant",
        signal_direction=0.7,
        signal_confidence=0.65,
        mode="OPPORTUNITY",
        regime="TRENDING",
        phase="EXPANSION"
    )
    
    # Test updating price
    print("\n=== Updating price ===")
    loop.update_position_price("SOL", 190.0)
    
    positions = loop.get_open_positions()
    for tid, pos in positions.items():
        print(f"  {tid}: unrealized={pos.unrealized_pnl:.2f}")
    
    # Test closing position
    print("\n=== Closing position ===")
    event = loop.close_position("T001", exit_price=192.0, exit_reason="target_hit")
    
    if event:
        print(f"  PnL: {event.pnl_realized:.2f} ({event.pnl_bps:.1f}bps)")
        print(f"  Bars held: {event.bars_held}")
    
    # Check strategy performance
    print("\n=== Strategy Performance ===")
    perf = loop.get_strategy_performance("quant")
    if perf:
        print(f"  Trades: {perf.total_trades}")
        print(f"  Win rate: {perf.win_rate:.1%}")
        print(f"  Confidence multiplier: {perf.confidence_multiplier:.2f}")
    
    # Statistics
    print("\n=== Statistics ===")
    stats = loop.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")

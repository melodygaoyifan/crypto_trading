#!/usr/bin/env python3
"""
================================================================================
HMATS v5.1 - EventBus Integration Layer
================================================================================
Purpose: Connect EventBus to all core system components for unified event flow

P1 SOTA Requirement:
- 接入 5 类事件: MarketDataUpdate, RegimeUpdate, SignalGenerated, 
  OrderSubmitted, RiskVeto
- 每个 4H tick 输出可 replay 的 event log

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        EventBus                                  │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
    │  │ Market   │  │ Regime   │  │ Signal   │  │ Order    │        │
    │  │ Data     │  │ Update   │  │ Fusion   │  │ Flow     │        │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
    └───────┼─────────────┼─────────────┼─────────────┼───────────────┘
            │             │             │             │
    ┌───────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
    │ Kraken WS   │ │ Regime    │ │ Authority │ │ Execution │
    │ Data Feed   │ │ Navigator │ │ Fusion    │ │ Manager   │
    └─────────────┘ └───────────┘ └───────────┘ └───────────┘

Author: HMATS v5.1 P1 Fix
Version: 5.1.0
================================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto
from pathlib import Path
import json

from .event_bus import (
    EventBus, Event, EventType, EventPriority,
    SignalEvent, OrderEvent
)

logger = logging.getLogger(__name__)


# =============================================================================
# INTEGRATION EVENT TYPES (Extended for P1 compliance)
# =============================================================================

class IntegrationEventType(Enum):
    """Extended event types for full system integration."""
    # Market Data (Category 1)
    MARKET_DATA_UPDATE = auto()
    OHLCV_RECEIVED = auto()
    ORDERBOOK_SNAPSHOT = auto()
    TRADE_FLOW_UPDATE = auto()
    LIQUIDITY_UPDATE = auto()
    
    # Regime (Category 2)
    REGIME_UPDATE = auto()
    PHASE_TRANSITION = auto()
    VOLATILITY_REGIME_CHANGE = auto()
    CORRELATION_UPDATE = auto()
    
    # Signal (Category 3)
    SIGNAL_GENERATED = auto()
    SIGNAL_FUSED = auto()
    AUTHORITY_DECISION = auto()
    OPPORTUNITY_DETECTED = auto()
    NO_TRADE_TRIGGERED = auto()
    
    # Order (Category 4)
    ORDER_SUBMITTED = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELLED = auto()
    ORDER_REJECTED = auto()
    TRANCHE_SCHEDULED = auto()
    
    # Risk (Category 5)
    RISK_VETO = auto()
    TRADE_GATE_DECISION = auto()
    STOP_LOSS_TRIGGERED = auto()
    EXPOSURE_ALERT = auto()
    DRAWDOWN_WARNING = auto()


@dataclass
class TickEvent:
    """4H tick event with all components."""
    tick_id: str
    timestamp: datetime
    mode: str  # NORMAL, OPPORTUNITY, NO_TRADE
    regime: str
    signals: Dict[str, Any]
    risk_state: Dict[str, Any]
    execution_state: Dict[str, Any]
    proof_log: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "tick_id": self.tick_id,
            "timestamp": self.timestamp.isoformat(),
            "mode": self.mode,
            "regime": self.regime,
            "signals": self.signals,
            "risk_state": self.risk_state,
            "execution_state": self.execution_state,
            "proof_log": self.proof_log
        }


# =============================================================================
# EVENT EMITTERS - Adapters for each subsystem
# =============================================================================

class MarketDataEmitter:
    """Emits market data events to EventBus."""
    
    def __init__(self, event_bus: EventBus):
        self.bus = event_bus
        self.source = "market_data"
        
    def emit_price_update(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: Optional[datetime] = None
    ):
        """Emit price update event."""
        event = Event(
            event_type=EventType.PRICE_UPDATE,
            source=self.source,
            priority=EventPriority.HIGH,
            timestamp=timestamp or datetime.now(timezone.utc),
            payload={
                "symbol": symbol,
                "price": price,
                "volume": volume
            }
        )
        self.bus.publish(event)
        
    def emit_orderbook_update(
        self,
        symbol: str,
        bids: List[tuple],
        asks: List[tuple],
        timestamp: Optional[datetime] = None
    ):
        """Emit orderbook snapshot event."""
        event = Event(
            event_type=EventType.ORDERBOOK_UPDATE,
            source=self.source,
            priority=EventPriority.NORMAL,
            timestamp=timestamp or datetime.now(timezone.utc),
            payload={
                "symbol": symbol,
                "bids": bids[:5],  # Top 5
                "asks": asks[:5],
                "spread": asks[0][0] - bids[0][0] if bids and asks else 0
            }
        )
        self.bus.publish(event)


class RegimeEmitter:
    """Emits regime change events to EventBus."""
    
    def __init__(self, event_bus: EventBus):
        self.bus = event_bus
        self.source = "regime_navigator"
        self._last_regime = None
        
    def emit_regime_update(
        self,
        regime: str,
        phase: str,
        confidence: float,
        volatility: float,
        metadata: Optional[Dict] = None
    ):
        """Emit regime update event."""
        is_transition = self._last_regime != regime
        self._last_regime = regime
        
        event = Event(
            event_type=EventType.REGIME_CHANGED if is_transition else EventType.MODE_CHANGED,
            source=self.source,
            priority=EventPriority.HIGH if is_transition else EventPriority.NORMAL,
            payload={
                "regime": regime,
                "phase": phase,
                "confidence": confidence,
                "volatility": volatility,
                "is_transition": is_transition,
                **(metadata or {})
            }
        )
        self.bus.publish(event)
        
        if is_transition:
            logger.info(f"[REGIME] Transition: {self._last_regime} -> {regime}")


class SignalEmitter:
    """Emits signal and fusion events to EventBus."""
    
    def __init__(self, event_bus: EventBus):
        self.bus = event_bus
        self.source = "signal_fusion"
        
    def emit_signal_generated(
        self,
        agent: str,
        symbol: str,
        direction: float,
        confidence: float,
        signal_type: str = "quant"
    ):
        """Emit individual agent signal."""
        event = SignalEvent(
            agent=agent,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            extra_data={"signal_type": signal_type}
        )
        self.bus.publish(event)
        
    def emit_signal_fused(
        self,
        symbol: str,
        fused_direction: float,
        fused_confidence: float,
        authority_source: str,
        contributing_agents: List[str],
        mode: str
    ):
        """Emit fused signal result."""
        event = Event(
            event_type=EventType.FUSION_COMPLETED,
            source=self.source,
            priority=EventPriority.NORMAL,
            payload={
                "symbol": symbol,
                "direction": fused_direction,
                "confidence": fused_confidence,
                "authority_source": authority_source,
                "contributing_agents": contributing_agents,
                "mode": mode
            }
        )
        self.bus.publish(event)


class OrderEmitter:
    """Emits order flow events to EventBus."""
    
    def __init__(self, event_bus: EventBus):
        self.bus = event_bus
        self.source = "execution_manager"
        
    def emit_order_submitted(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market"
    ):
        """Emit order submission event."""
        event = OrderEvent(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            event_type=EventType.ORDER_SUBMITTED,
            extra_data={
                "price": price,
                "order_type": order_type
            }
        )
        self.bus.publish(event)
        
    def emit_order_filled(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        fee: float = 0.0
    ):
        """Emit order fill event."""
        event = OrderEvent(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            event_type=EventType.ORDER_FILLED,
            extra_data={
                "fill_price": fill_price,
                "fee": fee
            }
        )
        self.bus.publish(event)


class RiskEmitter:
    """Emits risk veto and trade gate events to EventBus."""
    
    def __init__(self, event_bus: EventBus):
        self.bus = event_bus
        self.source = "risk_manager"
        
    def emit_risk_veto(
        self,
        veto_type: str,  # HARD or SOFT
        reason: str,
        threshold: float,
        actual_value: float,
        symbol: Optional[str] = None
    ):
        """Emit risk veto event."""
        event = Event(
            event_type=EventType.RISK_ALERT,
            source=self.source,
            priority=EventPriority.CRITICAL,
            payload={
                "veto_type": veto_type,
                "reason": reason,
                "threshold": threshold,
                "actual_value": actual_value,
                "symbol": symbol,
                "decision": "BLOCKED"
            }
        )
        self.bus.publish(event)
        logger.warning(f"[RISK VETO] {veto_type}: {reason} (threshold={threshold}, actual={actual_value})")
        
    def emit_trade_gate_decision(
        self,
        allowed: bool,
        gate_checks: Dict[str, bool],
        mode: str
    ):
        """Emit trade gate decision event."""
        event = Event(
            event_type=EventType.MODE_CHANGED,
            source="trade_gate",
            priority=EventPriority.HIGH,
            payload={
                "allowed": allowed,
                "gate_checks": gate_checks,
                "mode": mode,
                "decision": "ALLOWED" if allowed else "BLOCKED"
            }
        )
        self.bus.publish(event)


# =============================================================================
# TICK LOG BUILDER - Creates structured 4H tick logs
# =============================================================================

class TickLogBuilder:
    """
    Builds structured tick logs for each 4H decision cycle.
    
    SOTA Requirement: "每个 4H tick 输出可 replay 的 event log"
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs/ticks")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_tick: Optional[TickEvent] = None
        self._tick_counter = 0
        
    def start_tick(self, mode: str, regime: str) -> str:
        """Start a new 4H tick cycle."""
        self._tick_counter += 1
        tick_id = f"TICK_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}_{self._tick_counter:04d}"
        
        self._current_tick = TickEvent(
            tick_id=tick_id,
            timestamp=datetime.now(timezone.utc),
            mode=mode,
            regime=regime,
            signals={},
            risk_state={},
            execution_state={},
            proof_log={}
        )
        
        logger.info(f"[TICK] Started: {tick_id} | Mode: {mode} | Regime: {regime}")
        return tick_id
        
    def add_signal(self, agent: str, signal_data: Dict):
        """Add agent signal to current tick."""
        if self._current_tick:
            self._current_tick.signals[agent] = signal_data
            
    def add_risk_state(self, risk_data: Dict):
        """Add risk state to current tick."""
        if self._current_tick:
            self._current_tick.risk_state = risk_data
            
    def add_execution_state(self, execution_data: Dict):
        """Add execution state to current tick."""
        if self._current_tick:
            self._current_tick.execution_state = execution_data
            
    def add_proof(self, key: str, value: Any):
        """Add proof log entry."""
        if self._current_tick:
            self._current_tick.proof_log[key] = value
            
    def finalize_tick(self) -> Dict:
        """Finalize and save tick log."""
        if not self._current_tick:
            return {}
            
        tick_data = self._current_tick.to_dict()
        
        # Save to file
        log_file = self.log_dir / f"{self._current_tick.tick_id}.json"
        with open(log_file, 'w') as f:
            json.dump(tick_data, f, indent=2, default=str)
            
        logger.info(f"[TICK] Finalized: {self._current_tick.tick_id} -> {log_file}")
        
        result = tick_data
        self._current_tick = None
        return result


# =============================================================================
# EVENT INTEGRATION MANAGER - Coordinates all emitters
# =============================================================================

class EventIntegrationManager:
    """
    Coordinates all event emitters and provides unified access.
    
    Usage:
        manager = EventIntegrationManager()
        
        # Use individual emitters
        manager.market.emit_price_update("SOL/USD", 150.0, 1000.0)
        manager.signal.emit_signal_generated("quant", "SOL/USD", 0.8, 0.9)
        
        # Start tick cycle
        tick_id = manager.tick_log.start_tick("NORMAL", "trending")
        # ... trading logic ...
        manager.tick_log.finalize_tick()
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        log_dir: Optional[Path] = None,
        enable_logging: bool = True
    ):
        # Create or use provided EventBus
        self.bus = event_bus or EventBus(
            log_dir=log_dir,
            enable_logging=enable_logging
        )
        
        # Initialize all emitters
        self.market = MarketDataEmitter(self.bus)
        self.regime = RegimeEmitter(self.bus)
        self.signal = SignalEmitter(self.bus)
        self.order = OrderEmitter(self.bus)
        self.risk = RiskEmitter(self.bus)
        
        # Tick log builder
        self.tick_log = TickLogBuilder(log_dir=log_dir)
        
        # Statistics
        self._stats = {
            "ticks_processed": 0,
            "events_emitted": 0,
            "risk_vetoes": 0
        }
        
        logger.info("[EventIntegration] Initialized with 5 emitters")
        
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """Subscribe to specific event type."""
        self.bus.subscribe(event_type, handler)
        
    def subscribe_all(self, handler: Callable[[Event], None]):
        """Subscribe to all events."""
        self.bus.subscribe_all(handler)
        
    def start_tick_cycle(self, mode: str, regime: str) -> str:
        """Start a new 4H tick cycle with event tracking."""
        tick_id = self.tick_log.start_tick(mode, regime)
        self._stats["ticks_processed"] += 1
        return tick_id
        
    def finalize_tick_cycle(self) -> Dict:
        """Finalize current tick cycle and return log."""
        return self.tick_log.finalize_tick()
        
    def get_stats(self) -> Dict:
        """Get integration statistics."""
        return {
            **self._stats,
            "bus_stats": self.bus._stats if hasattr(self.bus, '_stats') else {}
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_integration_manager: Optional[EventIntegrationManager] = None


def get_event_integration(
    log_dir: Optional[Path] = None,
    enable_logging: bool = True
) -> EventIntegrationManager:
    """Get singleton EventIntegrationManager instance."""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = EventIntegrationManager(
            log_dir=log_dir,
            enable_logging=enable_logging
        )
    return _integration_manager


def reset_event_integration():
    """Reset singleton instance."""
    global _integration_manager
    _integration_manager = None


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    'EventIntegrationManager',
    'MarketDataEmitter',
    'RegimeEmitter', 
    'SignalEmitter',
    'OrderEmitter',
    'RiskEmitter',
    'TickLogBuilder',
    'TickEvent',
    'IntegrationEventType',
    'get_event_integration',
    'reset_event_integration',
]

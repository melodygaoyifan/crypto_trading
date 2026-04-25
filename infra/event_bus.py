#!/usr/bin/env python3
"""
================================================================================
HMATS v5.0 - EventBus Core Infrastructure with Causation Tracking
================================================================================
Purpose: Unified event routing for agent signals, fusion, execution flow
         Supports live trading and backtest replay modes
         
SOTA Features (v5.0):
1. Standard event types for all system components
2. Publish/Subscribe pattern with typed handlers
3. Event logging for replay/debugging
4. Async and sync execution modes
5. Priority-based event delivery
6. **NEW** Causation chain tracking (event_id, correlation_id, causation_id)
7. **NEW** SignalToExecutionSaga for full flow tracking
8. **NEW** Event sourcing support

Architecture:
    Agent Signal -> EventBus -> Signal Fusion -> EventBus -> Execution
                     ↓                           ↓
                Event Log ────────────────► Replay Engine
                
    Causation Chain:
    SIGNAL_GEN (corr_id: A1) ─causes-> FUSION_COMPLETED (parent: A1-sig)
                                          ↓
                                    ORDER_SUBMITTED (parent: A1-fus)
                                          ↓
                                    ORDER_FILLED (corr_id: A1)

Author: HMATS v5.0 SOTA Upgrade
Version: 5.0.0
================================================================================
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from queue import Queue, PriorityQueue
from typing import Any, Callable, Dict, List, Optional, Set, Union
import gzip


logger = logging.getLogger(__name__)


# =============================================================================
# EVENT TYPES
# =============================================================================

class EventType(Enum):
    """Standard event types in HMATS system."""
    # Agent Signals
    SIGNAL_GENERATED = auto()       # Agent produces a trading signal
    SIGNAL_UPDATED = auto()         # Signal strength/direction changed
    SIGNAL_EXPIRED = auto()         # Signal no longer valid
    
    # Signal Fusion
    FUSION_STARTED = auto()         # Fusion process begins
    FUSION_COMPLETED = auto()       # Final fused signal ready
    AUTHORITY_RESOLVED = auto()     # Authority matrix decision made
    
    # Order Flow
    ORDER_REQUESTED = auto()        # Execution requested
    ORDER_SUBMITTED = auto()        # Order sent to exchange
    ORDER_FILLED = auto()           # Order fully filled
    ORDER_PARTIAL_FILL = auto()     # Order partially filled
    ORDER_CANCELLED = auto()        # Order cancelled
    ORDER_REJECTED = auto()         # Order rejected by exchange
    
    # Position Events
    POSITION_OPENED = auto()        # New position opened
    POSITION_UPDATED = auto()       # Position size changed
    POSITION_CLOSED = auto()        # Position fully closed
    
    # Risk Events
    RISK_ALERT = auto()             # Risk threshold triggered
    STOP_TRIGGERED = auto()         # Stop loss hit
    DRAWDOWN_WARNING = auto()       # Drawdown threshold reached
    EXPOSURE_EXCEEDED = auto()      # Exposure limit exceeded
    
    # System Events
    MODE_CHANGED = auto()           # System mode transition
    REGIME_CHANGED = auto()         # Market regime changed
    HEARTBEAT = auto()              # System alive check
    ERROR = auto()                  # System error occurred
    
    # Market Data
    PRICE_UPDATE = auto()           # Price tick received
    ORDERBOOK_UPDATE = auto()       # Orderbook snapshot
    TRADE_FLOW = auto()             # Trade flow event


class EventPriority(Enum):
    """Event priority levels (lower = higher priority)."""
    CRITICAL = 0    # Risk alerts, stop triggers
    HIGH = 1        # Order events, position changes
    NORMAL = 2      # Signals, fusion results
    LOW = 3         # Heartbeats, logging


# =============================================================================
# EVENT DATA STRUCTURES
# =============================================================================

import uuid as _uuid

@dataclass
class Event:
    """
    Base event structure with causation tracking (v5.0 Enhanced).
    
    Causation Fields:
    - event_id: Unique identifier for this event
    - correlation_id: Groups related events in a saga
    - causation_id: Direct parent event that caused this one
    """
    event_type: EventType
    # [P51 2026-04-25] datetime.utcnow() returns naive; if any consumer
    # compares event.timestamp with timezone.utc-aware values (replay,
    # ordering checks across saga boundaries), TypeError silently. Use
    # aware default to match the rest of the codebase post-P39/P40/P50.
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""               # Component that generated event
    target: Optional[str] = None   # Target component (None = broadcast)
    priority: EventPriority = EventPriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Causation tracking (v5.0)
    event_id: str = field(default_factory=lambda: str(_uuid.uuid4()))
    correlation_id: Optional[str] = None  # Saga ID (groups related events)
    causation_id: Optional[str] = None    # Direct parent event ID
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value < other.priority.value
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.name,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "target": self.target,
            "priority": self.priority.name,
            "payload": self.payload,
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Event':
        """Create event from dictionary."""
        return cls(
            event_type=EventType[data["event_type"]],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data.get("source", ""),
            target=data.get("target"),
            priority=EventPriority[data.get("priority", "NORMAL")],
            payload=data.get("payload", {}),
            event_id=data.get("event_id", str(_uuid.uuid4())),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id")
        )
    
    def create_child(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        source: Optional[str] = None,
        priority: Optional[EventPriority] = None
    ) -> 'Event':
        """
        Create a child event (maintains causation chain).
        
        The child event will:
        - Have same correlation_id as parent
        - Have parent's event_id as its causation_id
        - Get a new unique event_id
        
        Args:
            event_type: Type of child event
            payload: Child event payload
            source: Source component (default: self.target or self.source)
            priority: Priority (default: same as parent)
            
        Returns:
            New Event linked to this one via causation chain
        """
        return Event(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            source=source or self.target or self.source,
            payload=payload,
            priority=priority or self.priority,
            correlation_id=self.correlation_id or self.event_id,
            causation_id=self.event_id
        )
    
    def __str__(self) -> str:
        return f"Event({self.event_type.name}, src={self.source}, ts={self.timestamp.strftime('%H:%M:%S.%f')[:-3]})"


@dataclass
class SignalEvent(Event):
    """Signal-specific event with standard fields."""
    def __init__(
        self,
        agent: str,
        symbol: str,
        direction: float,
        confidence: float,
        **kwargs
    ):
        payload = {
            "agent": agent,
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            **kwargs.pop("extra_data", {})
        }
        super().__init__(
            event_type=EventType.SIGNAL_GENERATED,
            source=agent,
            payload=payload,
            **kwargs
        )


@dataclass 
class OrderEvent(Event):
    """Order-specific event with standard fields."""
    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        event_type: EventType = EventType.ORDER_SUBMITTED,
        **kwargs
    ):
        payload = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            **kwargs.pop("extra_data", {})
        }
        super().__init__(
            event_type=event_type,
            priority=EventPriority.HIGH,
            payload=payload,
            **kwargs
        )


# =============================================================================
# EVENT HANDLER TYPE
# =============================================================================

EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Any]  # Can be coroutine


# =============================================================================
# EVENT BUS IMPLEMENTATION
# =============================================================================

class EventBus:
    """
    Central event routing hub for HMATS system.
    
    Features:
    - Typed subscription (by EventType)
    - Wildcard subscription (receive all events)
    - Sync and async handler support
    - Event logging for replay
    - Thread-safe operations
    
    Usage:
        bus = EventBus()
        
        # Subscribe to specific events
        bus.subscribe(EventType.SIGNAL_GENERATED, handle_signal)
        
        # Subscribe to all events
        bus.subscribe_all(my_logger)
        
        # Publish event
        bus.publish(Event(EventType.SIGNAL_GENERATED, ...))
        
        # Async mode
        await bus.publish_async(event)
    """
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        enable_logging: bool = True,
        replay_mode: bool = False,
        max_queue_size: int = 10000
    ):
        """
        Initialize EventBus.
        
        Args:
            log_dir: Directory for event logs
            enable_logging: Whether to log events to disk
            replay_mode: If True, events sorted by timestamp for backtesting
            max_queue_size: Maximum events in queue
        """
        self._subscribers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._async_subscribers: Dict[EventType, List[AsyncEventHandler]] = defaultdict(list)
        self._wildcard_subscribers: List[EventHandler] = []
        self._async_wildcard_subscribers: List[AsyncEventHandler] = []
        
        self._lock = threading.RLock()
        self._event_queue: Queue = Queue(maxsize=max_queue_size)
        self._priority_queue: PriorityQueue = PriorityQueue(maxsize=max_queue_size)
        
        self._enable_logging = enable_logging
        self._log_dir = log_dir or Path("logs/events")
        self._log_file = None
        self._replay_mode = replay_mode
        self._replay_events: List[Event] = []
        
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._stats = {
            "published": 0,
            "delivered": 0,
            "errors": 0,
            "by_type": defaultdict(int)
        }
        
        if enable_logging:
            self._init_logging()

    def _increment_stat(self, key: str, event_type: Optional[EventType] = None) -> None:
        """Thread-safe stats update helper."""
        with self._lock:
            self._stats[key] += 1
            if event_type is not None:
                self._stats["by_type"][event_type.name] += 1

    def _log_event(self, event: Event) -> None:
        """Thread-safe event log write helper."""
        if not self._enable_logging:
            return

        payload = json.dumps(event.to_dict()) + "\n"
        with self._lock:
            if not self._log_file:
                return
            self._log_file.write(payload)
            self._log_file.flush()
    
    def _init_logging(self):
        """Initialize event logging (plain JSONL, no gzip - crash-safe)."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_path = self._log_dir / f"events_{timestamp}.jsonl"
        self._log_file = open(log_path, 'w', encoding='utf-8')
        logger.info(f"EventBus logging to {log_path}")
    
    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
        is_async: bool = False
    ) -> None:
        """
        Subscribe to specific event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Callback function for events
            is_async: Whether handler is async coroutine
        """
        with self._lock:
            if is_async:
                self._async_subscribers[event_type].append(handler)
            else:
                self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed to {event_type.name}: {handler.__name__}")
    
    def subscribe_all(
        self,
        handler: EventHandler,
        is_async: bool = False
    ) -> None:
        """Subscribe to all events (wildcard)."""
        with self._lock:
            if is_async:
                self._async_wildcard_subscribers.append(handler)
            else:
                self._wildcard_subscribers.append(handler)
        logger.debug(f"Wildcard subscription: {handler.__name__}")
    
    def unsubscribe(
        self,
        event_type: EventType,
        handler: EventHandler
    ) -> bool:
        """Unsubscribe handler from event type."""
        with self._lock:
            try:
                if handler in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(handler)
                    return True
                if handler in self._async_subscribers[event_type]:
                    self._async_subscribers[event_type].remove(handler)
                    return True
            except ValueError:
                pass
        return False
    
    def publish(
        self,
        event: Event,
        use_priority_queue: bool = False
    ) -> None:
        """
        Publish event synchronously.
        
        Args:
            event: Event to publish
            use_priority_queue: Use priority ordering
        """
        self._increment_stat("published", event.event_type)
        
        # Log event
        try:
            self._log_event(event)
        except Exception as e:
            logger.warning(f"Failed to log event: {e}")

        # In replay mode, queue for ordered delivery
        if self._replay_mode:
            with self._lock:
                self._replay_events.append(event)
            return
        
        # Deliver to handlers
        self._deliver_sync(event)
    
    def _deliver_sync(self, event: Event) -> None:
        """Deliver event to synchronous handlers.

        Copies handler lists under lock then releases before calling handlers.
        This prevents deadlock if a handler subscribes/unsubscribes during delivery,
        and avoids blocking other threads during potentially slow handler execution.
        """
        with self._lock:
            handlers = list(self._subscribers.get(event.event_type, []))
            wildcards = list(self._wildcard_subscribers)

        for handler in handlers:
            try:
                handler(event)
                self._increment_stat("delivered")
            except Exception as e:
                self._increment_stat("errors")
                logger.error(f"Handler error for {event.event_type}: {e}")

        for handler in wildcards:
            try:
                handler(event)
                self._increment_stat("delivered")
            except Exception as e:
                self._increment_stat("errors")
                logger.error(f"Wildcard handler error: {e}")
    
    async def publish_async(self, event: Event) -> None:
        """
        Publish event asynchronously.
        
        Args:
            event: Event to publish
        """
        self._increment_stat("published", event.event_type)
        
        # Log event
        try:
            self._log_event(event)
        except Exception as e:
            logger.warning(f"Failed to log event: {e}")

        # Deliver to sync handlers first
        self._deliver_sync(event)
        
        # Then async handlers
        await self._deliver_async(event)
    
    async def _deliver_async(self, event: Event) -> None:
        """Deliver event to async handlers.

        Copies handler lists under lock then releases before creating tasks.
        """
        with self._lock:
            async_handlers = list(self._async_subscribers.get(event.event_type, []))
            async_wildcards = list(self._async_wildcard_subscribers)

        tasks = []
        for handler in async_handlers:
            tasks.append(asyncio.create_task(self._safe_async_call(handler, event)))
        for handler in async_wildcards:
            tasks.append(asyncio.create_task(self._safe_async_call(handler, event)))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _safe_async_call(
        self,
        handler: AsyncEventHandler,
        event: Event
    ) -> None:
        """Safely call async handler."""
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
            self._increment_stat("delivered")
        except Exception as e:
            self._increment_stat("errors")
            logger.error(f"Async handler error for {event.event_type}: {e}")
    
    # =========================================================================
    # REPLAY MODE
    # =========================================================================
    
    def load_replay_events(self, log_file: Path) -> int:
        """
        Load events from log file for replay.
        
        Args:
            log_file: Path to event log file
            
        Returns:
            Number of events loaded
        """
        self._replay_events = []
        
        opener = gzip.open if str(log_file).endswith('.gz') else open
        with opener(log_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    event = Event.from_dict(data)
                    self._replay_events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to parse event: {e}")
        
        # Sort by timestamp
        self._replay_events.sort(key=lambda e: e.timestamp)
        logger.info(f"Loaded {len(self._replay_events)} events for replay")
        return len(self._replay_events)
    
    def replay(
        self,
        speed: float = 1.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> None:
        """
        Replay events in order.
        
        Args:
            speed: Replay speed multiplier (1.0 = real-time)
            start_time: Start from this timestamp
            end_time: Stop at this timestamp
        """
        if not self._replay_events:
            logger.warning("No events loaded for replay")
            return
        
        events = self._replay_events
        
        # Filter by time range
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        logger.info(f"Replaying {len(events)} events at {speed}x speed")
        
        prev_time = None
        for event in events:
            # Simulate time delay
            if prev_time and speed > 0:
                delay = (event.timestamp - prev_time).total_seconds() / speed
                if delay > 0:
                    time.sleep(min(delay, 5.0))  # Cap at 5 seconds
            
            # Deliver event
            self._deliver_sync(event)
            prev_time = event.timestamp
    
    async def replay_async(
        self,
        speed: float = 1.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> None:
        """Async version of replay."""
        if not self._replay_events:
            return
        
        events = self._replay_events
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        prev_time = None
        for event in events:
            if prev_time and speed > 0:
                delay = (event.timestamp - prev_time).total_seconds() / speed
                if delay > 0:
                    await asyncio.sleep(min(delay, 5.0))
            
            self._deliver_sync(event)
            await self._deliver_async(event)
            prev_time = event.timestamp
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Get event bus statistics."""
        with self._lock:
            return {
                "published": self._stats["published"],
                "delivered": self._stats["delivered"],
                "errors": self._stats["errors"],
                "by_type": dict(self._stats["by_type"]),
                "subscriber_count": sum(len(h) for h in self._subscribers.values()),
                "async_subscriber_count": sum(len(h) for h in self._async_subscribers.values()),
                "wildcard_count": len(self._wildcard_subscribers) + len(self._async_wildcard_subscribers),
                "replay_events_loaded": len(self._replay_events)
            }
    
    def clear_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._stats = {
                "published": 0,
                "delivered": 0,
                "errors": 0,
                "by_type": defaultdict(int)
            }
    
    def close(self) -> None:
        """Close event bus and cleanup resources."""
        with self._lock:
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            self._running = False
        logger.info("EventBus closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get global EventBus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def init_event_bus(
    log_dir: Optional[Path] = None,
    enable_logging: bool = True,
    replay_mode: bool = False
) -> EventBus:
    """Initialize global EventBus with configuration."""
    global _global_event_bus
    _global_event_bus = EventBus(
        log_dir=log_dir,
        enable_logging=enable_logging,
        replay_mode=replay_mode
    )
    return _global_event_bus


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def publish_signal(
    agent: str,
    symbol: str,
    direction: float,
    confidence: float,
    **extra_data
) -> None:
    """Convenience function to publish signal event."""
    event = SignalEvent(
        agent=agent,
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        extra_data=extra_data
    )
    get_event_bus().publish(event)


def publish_order(
    order_id: str,
    symbol: str,
    side: str,
    quantity: float,
    event_type: EventType = EventType.ORDER_SUBMITTED,
    **extra_data
) -> None:
    """Convenience function to publish order event."""
    event = OrderEvent(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        event_type=event_type,
        extra_data=extra_data
    )
    get_event_bus().publish(event)


def publish_risk_alert(
    alert_type: str,
    severity: str,
    message: str,
    **extra_data
) -> None:
    """Convenience function to publish risk alert."""
    event = Event(
        event_type=EventType.RISK_ALERT,
        priority=EventPriority.CRITICAL,
        source="risk_manager",
        payload={
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            **extra_data
        }
    )
    get_event_bus().publish(event)


def publish_position_event(
    symbol: str,
    event_type: EventType,
    position_size: float,
    entry_price: float = 0.0,
    current_price: float = 0.0,
    pnl: float = 0.0,
    **extra_data
) -> None:
    """Convenience function to publish position events."""
    event = Event(
        event_type=event_type,
        priority=EventPriority.HIGH,
        source="position_manager",
        payload={
            "symbol": symbol,
            "position_size": position_size,
            "entry_price": entry_price,
            "current_price": current_price,
            "pnl": pnl,
            **extra_data
        }
    )
    get_event_bus().publish(event)


def publish_system_event(
    event_type: EventType,
    message: str,
    **extra_data
) -> None:
    """Convenience function to publish system events."""
    event = Event(
        event_type=event_type,
        priority=EventPriority.NORMAL,
        source="system",
        payload={
            "message": message,
            **extra_data
        }
    )
    get_event_bus().publish(event)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Basic test
    logging.basicConfig(level=logging.INFO)
    
    bus = EventBus(enable_logging=False)
    
    received = []
    
    def signal_handler(event: Event):
        received.append(event)
        print(f"Received: {event}")
    
    bus.subscribe(EventType.SIGNAL_GENERATED, signal_handler)
    
    # Publish test event
    event = SignalEvent(
        agent="QuantBase",
        symbol="SOL/USD",
        direction=0.8,
        confidence=0.75
    )
    bus.publish(event)
    
    print(f"\nStats: {bus.get_stats()}")
    print(f"Received {len(received)} events")
    
    bus.close()

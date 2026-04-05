"""
================================================================================
HMATS v5.1.0 - Event Replay System
================================================================================
Purpose: Formalized event replay mode using EventBus logs

SOTA Requirement:
- Formalize event replay mode using EventBus logs.
- Simulate edge-case events (e.g., WS disconnect, API rate limit) in backtest 
  to test resilience.

Features:
1. Load and parse EventBus logs
2. Chronological event replay
3. Edge-case event injection
4. Speed control (real-time, accelerated)
5. State checkpoint/restore

Author: HMATS SOTA Upgrade
Version: 5.1.0-SOTA
================================================================================
"""

import logging
import asyncio
import json
import gzip
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Generator
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import random

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class ReplaySpeed(Enum):
    """Replay speed modes."""
    REALTIME = "realtime"       # 1x speed
    FAST = "fast"               # 10x speed
    TURBO = "turbo"             # 100x speed
    INSTANT = "instant"         # As fast as possible


class EventType(Enum):
    """Types of events in the system."""
    MARKET_DATA = "market_data"
    TRADE = "trade"
    ORDER = "order"
    SIGNAL = "signal"
    RISK = "risk"
    SYSTEM = "system"
    ERROR = "error"


class InjectedFaultType(Enum):
    """Types of faults that can be injected."""
    WS_DISCONNECT = "ws_disconnect"
    WS_RECONNECT = "ws_reconnect"
    API_RATE_LIMIT = "api_rate_limit"
    API_TIMEOUT = "api_timeout"
    API_ERROR = "api_error"
    DATA_GAP = "data_gap"
    LATENCY_SPIKE = "latency_spike"
    EXCHANGE_MAINTENANCE = "exchange_maintenance"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ReplayEvent:
    """A single event for replay."""
    timestamp: datetime
    event_type: EventType
    data: Dict
    source: str = "replay"
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "type": self.event_type.value,
            "data": self.data,
            "source": self.source
        }


@dataclass
class InjectedFault:
    """A fault to inject during replay."""
    fault_type: InjectedFaultType
    trigger_time: datetime
    duration_seconds: float = 0
    parameters: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "type": self.fault_type.value,
            "trigger_time": self.trigger_time.isoformat(),
            "duration": self.duration_seconds,
            "parameters": self.parameters
        }


@dataclass
class ReplayCheckpoint:
    """Checkpoint for replay state."""
    checkpoint_id: str
    timestamp: datetime
    event_index: int
    system_state: Dict
    
    def to_dict(self) -> Dict:
        return {
            "id": self.checkpoint_id,
            "timestamp": self.timestamp.isoformat(),
            "event_index": self.event_index
        }


@dataclass
class ReplayStats:
    """Statistics from replay session."""
    total_events: int = 0
    events_processed: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    faults_injected: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    replay_duration_seconds: float = 0
    simulated_duration_seconds: float = 0
    
    def to_dict(self) -> Dict:
        return {
            "total_events": self.total_events,
            "events_processed": self.events_processed,
            "events_by_type": self.events_by_type,
            "faults_injected": self.faults_injected,
            "replay_duration": self.replay_duration_seconds,
            "simulated_duration": self.simulated_duration_seconds
        }


# =============================================================================
# EVENT LOG PARSER
# =============================================================================

class EventLogParser:
    """Parses EventBus log files."""
    
    def __init__(self, log_dir: str = "logs/events"):
        self.log_dir = Path(log_dir)
    
    def list_log_files(
        self,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[Path]:
        """List available log files in date range."""
        files = []
        
        for pattern in ["events_*.jsonl", "events_*.jsonl.gz"]:
            for f in self.log_dir.glob(pattern):
                # Parse date from filename (events_YYYYMMDD_HHMMSS.jsonl)
                try:
                    date_str = f.stem.replace("events_", "").split(".")[0]
                    file_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                    
                    if start_date and file_date < start_date:
                        continue
                    if end_date and file_date > end_date:
                        continue
                    
                    files.append(f)
                except ValueError:
                    continue
        
        return sorted(files)
    
    def parse_file(self, file_path: Path) -> Generator[ReplayEvent, None, None]:
        """Parse events from a log file."""
        opener = gzip.open if str(file_path).endswith('.gz') else open
        
        with opener(file_path, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Parse timestamp
                    ts_str = data.get("tick_timestamp") or data.get("timestamp")
                    if ts_str:
                        timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    else:
                        continue
                    
                    # Determine event type
                    event_type = self._classify_event(data)
                    
                    yield ReplayEvent(
                        timestamp=timestamp,
                        event_type=event_type,
                        data=data,
                        source=str(file_path.name)
                    )
                    
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"Skipping malformed line: {e}")
    
    def _classify_event(self, data: Dict) -> EventType:
        """Classify event based on its content."""
        if "price" in data or "bid" in data or "ask" in data:
            return EventType.MARKET_DATA
        if "trade_id" in data or "fill" in data:
            return EventType.TRADE
        if "order_id" in data or "order_type" in data:
            return EventType.ORDER
        if "signal" in data or "direction" in data:
            return EventType.SIGNAL
        if "risk" in data or "veto" in data:
            return EventType.RISK
        if "error" in data:
            return EventType.ERROR
        return EventType.SYSTEM


# =============================================================================
# FAULT INJECTOR
# =============================================================================

class FaultInjector:
    """Injects simulated faults during replay."""
    
    def __init__(self):
        self.scheduled_faults: List[InjectedFault] = []
        self.active_faults: List[InjectedFault] = []
        self.fault_history: List[Dict] = []
    
    def schedule_fault(self, fault: InjectedFault):
        """Schedule a fault for injection."""
        self.scheduled_faults.append(fault)
        self.scheduled_faults.sort(key=lambda f: f.trigger_time)
    
    def schedule_random_faults(
        self,
        start_time: datetime,
        end_time: datetime,
        fault_types: List[InjectedFaultType] = None,
        count: int = 5
    ):
        """Schedule random faults within a time range."""
        fault_types = fault_types or list(InjectedFaultType)
        duration = (end_time - start_time).total_seconds()
        
        for _ in range(count):
            offset = random.random() * duration
            trigger_time = start_time + timedelta(seconds=offset)
            fault_type = random.choice(fault_types)
            
            # Set reasonable durations
            durations = {
                InjectedFaultType.WS_DISCONNECT: random.uniform(1, 10),
                InjectedFaultType.API_RATE_LIMIT: random.uniform(5, 30),
                InjectedFaultType.API_TIMEOUT: random.uniform(1, 5),
                InjectedFaultType.DATA_GAP: random.uniform(10, 60),
                InjectedFaultType.LATENCY_SPIKE: random.uniform(0.5, 2),
                InjectedFaultType.EXCHANGE_MAINTENANCE: random.uniform(60, 300),
            }
            
            fault = InjectedFault(
                fault_type=fault_type,
                trigger_time=trigger_time,
                duration_seconds=durations.get(fault_type, 5)
            )
            
            self.schedule_fault(fault)
    
    def check_faults(self, current_time: datetime) -> List[InjectedFault]:
        """Check for faults that should trigger at current time."""
        triggered = []
        
        # Check scheduled faults
        while self.scheduled_faults and self.scheduled_faults[0].trigger_time <= current_time:
            fault = self.scheduled_faults.pop(0)
            triggered.append(fault)
            self.active_faults.append(fault)
            self.fault_history.append({
                "fault": fault.to_dict(),
                "triggered_at": current_time.isoformat()
            })
        
        # Expire old faults
        self.active_faults = [
            f for f in self.active_faults
            if (current_time - f.trigger_time).total_seconds() < f.duration_seconds
        ]
        
        return triggered
    
    def is_fault_active(self, fault_type: InjectedFaultType) -> bool:
        """Check if a fault type is currently active."""
        return any(f.fault_type == fault_type for f in self.active_faults)
    
    def get_active_faults(self) -> List[InjectedFaultType]:
        """Get list of currently active fault types."""
        return [f.fault_type for f in self.active_faults]


# =============================================================================
# EVENT REPLAY ENGINE
# =============================================================================

class EventReplayEngine:
    """
    Replays historical events for backtesting and simulation.
    
    Features:
    - Chronological event replay
    - Speed control
    - Fault injection
    - Checkpointing
    """
    
    def __init__(self, log_dir: str = "logs/events"):
        self.parser = EventLogParser(log_dir)
        self.fault_injector = FaultInjector()
        
        # Event handling
        self._handlers: Dict[EventType, List[Callable]] = {t: [] for t in EventType}
        self._fault_handlers: Dict[InjectedFaultType, Callable] = {}
        
        # State
        self.events: List[ReplayEvent] = []
        self.current_index: int = 0
        self.current_time: Optional[datetime] = None
        self.speed: ReplaySpeed = ReplaySpeed.FAST
        
        # Checkpoints
        self.checkpoints: Dict[str, ReplayCheckpoint] = {}
        
        # Stats
        self.stats = ReplayStats()
        
        # Control
        self._running = False
        self._paused = False
        
        logger.info("[EventReplay] Engine initialized")
    
    # =========================================================================
    # EVENT LOADING
    # =========================================================================
    
    def load_events(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        event_types: List[EventType] = None
    ):
        """Load events from log files."""
        self.events = []
        
        files = self.parser.list_log_files(start_date, end_date)
        logger.info(f"[EventReplay] Loading from {len(files)} files")
        
        for file_path in files:
            for event in self.parser.parse_file(file_path):
                if event_types and event.event_type not in event_types:
                    continue
                self.events.append(event)
        
        # Sort by timestamp
        self.events.sort(key=lambda e: e.timestamp)
        
        self.stats.total_events = len(self.events)
        
        if self.events:
            self.current_time = self.events[0].timestamp
            self.stats.start_time = self.events[0].timestamp
            self.stats.end_time = self.events[-1].timestamp
            self.stats.simulated_duration_seconds = (
                self.events[-1].timestamp - self.events[0].timestamp
            ).total_seconds()
        
        logger.info(f"[EventReplay] Loaded {len(self.events)} events")
    
    # =========================================================================
    # HANDLER REGISTRATION
    # =========================================================================
    
    def register_handler(self, event_type: EventType, handler: Callable):
        """Register a handler for an event type."""
        self._handlers[event_type].append(handler)
    
    def register_fault_handler(self, fault_type: InjectedFaultType, handler: Callable):
        """Register a handler for a fault type."""
        self._fault_handlers[fault_type] = handler
    
    # =========================================================================
    # REPLAY CONTROL
    # =========================================================================
    
    async def start(self, speed: ReplaySpeed = None):
        """Start the replay."""
        if speed:
            self.speed = speed
        
        self._running = True
        self._paused = False
        self.stats.events_processed = 0
        
        replay_start = datetime.utcnow()
        
        logger.info(f"[EventReplay] Starting replay at {self.speed.value} speed")
        
        while self._running and self.current_index < len(self.events):
            if self._paused:
                await asyncio.sleep(0.1)
                continue
            
            event = self.events[self.current_index]
            
            # Check for faults
            triggered_faults = self.fault_injector.check_faults(event.timestamp)
            for fault in triggered_faults:
                await self._handle_fault(fault)
            
            # Skip events during data gap fault
            if self.fault_injector.is_fault_active(InjectedFaultType.DATA_GAP):
                self.current_index += 1
                continue
            
            # Process event
            await self._process_event(event)
            
            # Update stats
            self.stats.events_processed += 1
            type_key = event.event_type.value
            self.stats.events_by_type[type_key] = self.stats.events_by_type.get(type_key, 0) + 1
            
            # Speed control
            if self.current_index + 1 < len(self.events):
                next_event = self.events[self.current_index + 1]
                delay = (next_event.timestamp - event.timestamp).total_seconds()
                
                if self.speed == ReplaySpeed.REALTIME:
                    await asyncio.sleep(delay)
                elif self.speed == ReplaySpeed.FAST:
                    await asyncio.sleep(delay / 10)
                elif self.speed == ReplaySpeed.TURBO:
                    await asyncio.sleep(delay / 100)
                # INSTANT: no delay
                else:
                    await asyncio.sleep(0)  # Yield to event loop
            
            self.current_index += 1
            self.current_time = event.timestamp
        
        self.stats.replay_duration_seconds = (datetime.utcnow() - replay_start).total_seconds()
        self._running = False
        
        logger.info(f"[EventReplay] Completed: {self.stats.events_processed} events")
    
    def pause(self):
        """Pause the replay."""
        self._paused = True
        logger.info("[EventReplay] Paused")
    
    def resume(self):
        """Resume the replay."""
        self._paused = False
        logger.info("[EventReplay] Resumed")
    
    def stop(self):
        """Stop the replay."""
        self._running = False
        logger.info("[EventReplay] Stopped")
    
    async def _process_event(self, event: ReplayEvent):
        """Process a single event."""
        # Add latency if fault active
        if self.fault_injector.is_fault_active(InjectedFaultType.LATENCY_SPIKE):
            await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Call handlers
        for handler in self._handlers.get(event.event_type, []):
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[EventReplay] Handler error: {e}")
    
    async def _handle_fault(self, fault: InjectedFault):
        """Handle an injected fault."""
        self.stats.faults_injected += 1
        
        handler = self._fault_handlers.get(fault.fault_type)
        if handler:
            try:
                result = handler(fault)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[EventReplay] Fault handler error: {e}")
        
        logger.info(f"[EventReplay] Injected fault: {fault.fault_type.value}")
    
    # =========================================================================
    # CHECKPOINTING
    # =========================================================================
    
    def create_checkpoint(self, checkpoint_id: str, system_state: Dict = None):
        """Create a checkpoint."""
        checkpoint = ReplayCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=self.current_time or datetime.utcnow(),
            event_index=self.current_index,
            system_state=system_state or {}
        )
        
        self.checkpoints[checkpoint_id] = checkpoint
        logger.info(f"[EventReplay] Created checkpoint: {checkpoint_id}")
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict]:
        """Restore to a checkpoint."""
        checkpoint = self.checkpoints.get(checkpoint_id)
        if not checkpoint:
            logger.error(f"[EventReplay] Unknown checkpoint: {checkpoint_id}")
            return None
        
        self.current_index = checkpoint.event_index
        self.current_time = checkpoint.timestamp
        
        logger.info(f"[EventReplay] Restored checkpoint: {checkpoint_id}")
        return checkpoint.system_state
    
    # =========================================================================
    # QUERIES
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Get replay statistics."""
        return self.stats.to_dict()
    
    def get_progress(self) -> Dict:
        """Get current replay progress."""
        return {
            "current_index": self.current_index,
            "total_events": len(self.events),
            "progress_pct": self.current_index / len(self.events) * 100 if self.events else 0,
            "current_time": self.current_time.isoformat() if self.current_time else None,
            "running": self._running,
            "paused": self._paused,
            "active_faults": [f.value for f in self.fault_injector.get_active_faults()]
        }


# =============================================================================
# SINGLETON
# =============================================================================

_replay_engine: Optional[EventReplayEngine] = None


def get_replay_engine(log_dir: str = "logs/events") -> EventReplayEngine:
    """Get or create the global replay engine."""
    global _replay_engine
    if _replay_engine is None:
        _replay_engine = EventReplayEngine(log_dir)
    return _replay_engine


def reset_replay_engine():
    """Reset the global replay engine."""
    global _replay_engine
    if _replay_engine:
        _replay_engine.stop()
    _replay_engine = None

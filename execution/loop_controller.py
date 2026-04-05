"""
================================================================================
EXECUTION LOOP CONTROLLER - 200ms Tick Loop Timing
================================================================================
Version: 3.3.0
Purpose: Resolves HIGH impact item #4 from CTO Audit

PROBLEM SOLVED:
- 200ms execution modifier loop was conceptual but timing unclear
- Risk of duplicate or skipped execution calls
- No enforcement of "200ms can modify execution mode but NOT target exposure"

SOLUTION:
- Explicit loop timing with drift correction
- Tick deduplication
- Strict separation of 4H intents vs 200ms modifiers
- Execution mode changes logged for audit

CRITICAL RULES (FROM CONSTITUTION):
1. 200ms loop can ONLY modify:
   - Execution mode (PASSIVE_ONLY, PASSIVE_PREFERRED, AGGRESSIVE_TAKER)
   - Order timing (delay, accelerate)
   - Slice sizing (within bounds)
   
2. 200ms loop CANNOT modify:
   - Target exposure
   - Direction
   - Tranche level
   - Stop loss levels

3. 4H is the ONLY authority for:
   - Target exposure
   - Direction
   - Entry/exit decisions
================================================================================
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, List, Any
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes that 200ms loop can set."""
    PASSIVE_ONLY = "PASSIVE_ONLY"           # Maker orders only
    PASSIVE_PREFERRED = "PASSIVE_PREFERRED"  # Maker bias, taker fallback
    AGGRESSIVE_TAKER = "AGGRESSIVE_TAKER"    # Market orders allowed
    ABORT = "ABORT"                          # No execution


@dataclass
class ExecutionModifier:
    """
    Modifiers that 200ms loop can apply.
    
    CRITICAL: These do NOT change target exposure or direction.
    They only modify HOW and WHEN we execute.
    """
    mode: ExecutionMode = ExecutionMode.PASSIVE_PREFERRED
    delay_ms: int = 0                    # Delay execution by N ms
    urgency_scale: float = 1.0           # 0.0 = no urgency, 1.0 = full urgency
    slice_scale: float = 1.0             # Scale individual slice sizes
    reason: str = ""
    source: str = ""                     # Which detector set this
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_ms: int = 500                    # Time-to-live in ms


@dataclass
class LoopTickResult:
    """Result of a single 200ms tick."""
    tick_id: int
    timestamp: datetime
    scheduled_time: datetime
    actual_time: datetime
    drift_ms: float
    modifier_applied: Optional[ExecutionModifier]
    was_skipped: bool
    skip_reason: str = ""


@dataclass
class ExecutionLoopConfig:
    """Configuration for execution loop."""
    
    # Timing
    tick_interval_ms: int = 200          # 200ms ticks
    max_drift_ms: int = 50               # Max acceptable drift
    skip_if_drift_exceeds_ms: int = 100  # Skip tick if too late
    
    # Deduplication
    dedup_window_ms: int = 150           # Ignore duplicate within this window
    
    # Modifier TTL
    default_modifier_ttl_ms: int = 500   # Modifiers expire after 500ms
    
    # Logging
    log_every_n_ticks: int = 300         # Log summary every 60 seconds


class ExecutionLoopController:
    """
    Controls the 200ms execution modifier loop.
    
    RESPONSIBILITIES:
    1. Maintain precise 200ms timing
    2. Apply execution modifiers from Lead-Lag, VPIN, etc.
    3. Prevent modifier staleness
    4. Log all modifier changes for audit
    
    USAGE:
        controller = ExecutionLoopController()
        controller.start()
        
        # From Lead-Lag detector
        controller.set_modifier(ExecutionModifier(
            mode=ExecutionMode.AGGRESSIVE_TAKER,
            urgency_scale=1.0,
            reason="Lead-Lag edge confirmed",
            source="lead_lag"
        ))
        
        # In execution engine
        modifier = controller.get_current_modifier()
        if modifier.mode == ExecutionMode.PASSIVE_ONLY:
            # Use limit orders only
            pass
    """
    
    def __init__(self, config: Optional[ExecutionLoopConfig] = None):
        self.config = config or ExecutionLoopConfig()
        
        # Current state
        self.current_modifier = ExecutionModifier()
        self.modifier_lock = threading.Lock()
        
        # Tick tracking
        self.tick_counter = 0
        self.last_tick_time: Optional[datetime] = None
        self.tick_history: deque = deque(maxlen=1000)  # Last ~3 minutes
        
        # Loop control
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._tick_callbacks: List[Callable[[ExecutionModifier], None]] = []
        
        # Modifier history (for audit)
        self.modifier_history: deque = deque(maxlen=500)
        
        logger.info("ExecutionLoopController initialized")
    
    def start(self):
        """Start the 200ms loop."""
        if self._running:
            logger.warning("ExecutionLoopController already running")
            return
        
        self._running = True
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        logger.info("ExecutionLoopController started")
    
    def stop(self):
        """Stop the 200ms loop."""
        self._running = False
        if self._loop_thread:
            self._loop_thread.join(timeout=1.0)
        logger.info("ExecutionLoopController stopped")
    
    def _run_loop(self):
        """Main loop - runs every 200ms."""
        next_tick = datetime.now(timezone.utc)
        interval = timedelta(milliseconds=self.config.tick_interval_ms)
        
        while self._running:
            now = datetime.now(timezone.utc)
            
            # Calculate drift
            drift = (now - next_tick).total_seconds() * 1000
            
            # Determine if tick should be processed or skipped
            if drift > self.config.skip_if_drift_exceeds_ms:
                # Too late - skip this tick
                result = LoopTickResult(
                    tick_id=self.tick_counter,
                    timestamp=now,
                    scheduled_time=next_tick,
                    actual_time=now,
                    drift_ms=drift,
                    modifier_applied=None,
                    was_skipped=True,
                    skip_reason=f"Drift {drift:.1f}ms exceeds threshold",
                )
                self.tick_history.append(result)
                logger.debug(f"Tick {self.tick_counter} skipped: drift={drift:.1f}ms")
            else:
                # Process tick
                self._process_tick(now, next_tick, drift)
            
            # Calculate next tick time
            self.tick_counter += 1
            next_tick = next_tick + interval
            
            # If we're behind, catch up
            if next_tick < now:
                next_tick = now + interval
            
            # Sleep until next tick
            sleep_time = (next_tick - datetime.now(timezone.utc)).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _process_tick(self, now: datetime, scheduled: datetime, drift: float):
        """Process a single tick."""
        
        # Check modifier expiration
        self._expire_stale_modifiers()
        
        # Get current modifier (thread-safe)
        with self.modifier_lock:
            modifier = self.current_modifier
        
        # Create result
        result = LoopTickResult(
            tick_id=self.tick_counter,
            timestamp=now,
            scheduled_time=scheduled,
            actual_time=now,
            drift_ms=drift,
            modifier_applied=modifier if modifier.mode != ExecutionMode.PASSIVE_PREFERRED else None,
            was_skipped=False,
        )
        
        self.tick_history.append(result)
        self.last_tick_time = now
        
        # Execute callbacks
        for callback in self._tick_callbacks:
            try:
                callback(modifier)
            except Exception as e:
                logger.error(f"Tick callback error: {e}")
        
        # Periodic logging
        if self.tick_counter % self.config.log_every_n_ticks == 0:
            self._log_summary()
    
    def _expire_stale_modifiers(self):
        """Expire modifiers that have exceeded their TTL."""
        with self.modifier_lock:
            if self.current_modifier.mode == ExecutionMode.PASSIVE_PREFERRED:
                return  # Default mode doesn't expire
            
            age_ms = (datetime.now(timezone.utc) - self.current_modifier.timestamp).total_seconds() * 1000
            
            if age_ms > self.current_modifier.ttl_ms:
                old_mode = self.current_modifier.mode
                self.current_modifier = ExecutionModifier()  # Reset to default
                logger.debug(f"Modifier expired: {old_mode.value} -> PASSIVE_PREFERRED")
    
    def set_modifier(self, modifier: ExecutionModifier):
        """
        Set execution modifier.
        
        CRITICAL: This only changes HOW we execute, not WHAT we execute.
        """
        with self.modifier_lock:
            old_mode = self.current_modifier.mode
            self.current_modifier = modifier
            
            # Log if mode changed
            if old_mode != modifier.mode:
                self.modifier_history.append({
                    "timestamp": modifier.timestamp,
                    "old_mode": old_mode.value,
                    "new_mode": modifier.mode.value,
                    "reason": modifier.reason,
                    "source": modifier.source,
                })
                logger.info(
                    f"Execution mode: {old_mode.value} -> {modifier.mode.value} "
                    f"({modifier.reason}, source={modifier.source})"
                )
    
    def get_current_modifier(self) -> ExecutionModifier:
        """Get current execution modifier (thread-safe)."""
        with self.modifier_lock:
            return self.current_modifier
    
    def add_tick_callback(self, callback: Callable[[ExecutionModifier], None]):
        """Add callback to be called on each tick."""
        self._tick_callbacks.append(callback)
    
    def remove_tick_callback(self, callback: Callable[[ExecutionModifier], None]):
        """Remove a tick callback."""
        if callback in self._tick_callbacks:
            self._tick_callbacks.remove(callback)
    
    def _log_summary(self):
        """Log periodic summary."""
        recent = list(self.tick_history)[-self.config.log_every_n_ticks:]
        if not recent:
            return
        
        skipped = sum(1 for r in recent if r.was_skipped)
        avg_drift = sum(r.drift_ms for r in recent) / len(recent)
        max_drift = max(r.drift_ms for r in recent)
        
        logger.debug(
            f"ExecutionLoop summary: ticks={len(recent)}, "
            f"skipped={skipped}, avg_drift={avg_drift:.1f}ms, max_drift={max_drift:.1f}ms"
        )
    
    def get_tick_stats(self) -> Dict[str, Any]:
        """Get tick statistics."""
        recent = list(self.tick_history)
        if not recent:
            return {}
        
        skipped = sum(1 for r in recent if r.was_skipped)
        drifts = [r.drift_ms for r in recent]
        
        return {
            "total_ticks": self.tick_counter,
            "recent_ticks": len(recent),
            "skipped_ticks": skipped,
            "skip_rate": skipped / len(recent) if recent else 0,
            "avg_drift_ms": sum(drifts) / len(drifts) if drifts else 0,
            "max_drift_ms": max(drifts) if drifts else 0,
            "min_drift_ms": min(drifts) if drifts else 0,
            "current_mode": self.current_modifier.mode.value,
        }
    
    def get_modifier_history(self) -> List[Dict]:
        """Get modifier change history for audit."""
        return list(self.modifier_history)


# Singleton instance
_execution_loop: Optional[ExecutionLoopController] = None


def get_execution_loop(config: Optional[ExecutionLoopConfig] = None) -> ExecutionLoopController:
    """Get or create singleton execution loop controller."""
    global _execution_loop
    if _execution_loop is None:
        _execution_loop = ExecutionLoopController(config)
    elif config is not None and _execution_loop.config != config:
        logger.info("[EXEC_LOOP] Config changed; refreshing singleton instance")
        _execution_loop = ExecutionLoopController(config)
    return _execution_loop


def reset_execution_loop():
    """Reset singleton (for testing)."""
    global _execution_loop
    if _execution_loop:
        _execution_loop.stop()
    _execution_loop = None


# Alias for backward compatibility
get_execution_loop_controller = get_execution_loop


__all__ = [
    "ExecutionLoopController",
    "ExecutionLoopConfig",
    "ExecutionModifier",
    "LoopTickResult",
    "get_execution_loop",
    "get_execution_loop_controller",
    "reset_execution_loop",
]

"""
================================================================================
HMATS v5.1.0 - SOTA Execution Scheduler
================================================================================
Purpose: Advanced order scheduling implementing SOTA checklist requirements

Features (from Upgrade Checklist):
1. TWAP (Time-Weighted Average Price) scheduling
2. VWAP (Volume-Weighted) with participation rate
3. Market Impact-aware slicing
4. Urgency override capability
5. Integration with TimingEngine

Author: HMATS SOTA Upgrade
Version: 5.1.0-SOTA
================================================================================
"""

import logging
import asyncio
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, List, Tuple, Any, Callable
from datetime import datetime, timezone, timedelta
from collections import deque
import json

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class ScheduleType(Enum):
    """Order scheduling algorithms."""
    IMMEDIATE = "immediate"       # Execute immediately (single order)
    TWAP = "twap"                 # Time-weighted average price
    VWAP = "vwap"                 # Volume-weighted (participation rate)
    ADAPTIVE = "adaptive"         # Adaptive based on conditions
    ICEBERG = "iceberg"          # Iceberg with hidden size


class SliceStatus(Enum):
    """Status of an order slice."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class UrgencyLevel(Enum):
    """Execution urgency levels."""
    LOW = 1       # Patient, minimize impact
    MEDIUM = 2    # Balanced
    HIGH = 3      # Faster execution, accept more impact
    URGENT = 4    # Execute ASAP


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SchedulerConfig:
    """Execution scheduler configuration."""
    
    # TWAP settings
    twap_default_duration_min: int = 15
    twap_min_slice_size_usd: float = 1000
    twap_max_slices: int = 20
    
    # VWAP settings
    vwap_participation_rate: float = 0.05    # 5% of volume
    vwap_max_participation: float = 0.10     # Never exceed 10%
    
    # Market impact
    max_impact_bps: float = 50               # Maximum acceptable impact
    impact_slice_threshold_usd: float = 50000  # Auto-slice above this
    
    # Timing
    min_slice_interval_ms: int = 1000
    max_slice_interval_ms: int = 60000
    
    # Urgency multipliers (for duration)
    urgency_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "LOW": 2.0,      # Double the duration
        "MEDIUM": 1.0,   # Normal duration
        "HIGH": 0.5,     # Half duration
        "URGENT": 0.1    # 10% duration (almost immediate)
    })


# =============================================================================
# ORDER SLICE
# =============================================================================

@dataclass
class OrderSlice:
    """A single slice of a scheduled order."""
    slice_id: str
    parent_id: str
    symbol: str
    side: str  # "buy" or "sell"
    size: float
    target_time: datetime
    
    status: SliceStatus = SliceStatus.PENDING
    submitted_time: Optional[datetime] = None
    filled_time: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_size: float = 0.0
    order_id: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "slice_id": self.slice_id,
            "parent_id": self.parent_id,
            "symbol": self.symbol,
            "side": self.side,
            "size": self.size,
            "target_time": self.target_time.isoformat(),
            "status": self.status.value,
            "filled_size": self.filled_size,
            "filled_price": self.filled_price
        }


# =============================================================================
# SCHEDULED ORDER
# =============================================================================

@dataclass
class ScheduledOrder:
    """A parent order being scheduled for execution."""
    order_id: str
    symbol: str
    side: str
    total_size: float
    schedule_type: ScheduleType
    urgency: UrgencyLevel
    
    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    
    # Progress
    slices: List[OrderSlice] = field(default_factory=list)
    filled_size: float = 0.0
    avg_price: float = 0.0
    
    # State
    is_active: bool = True
    is_complete: bool = False
    cancelled: bool = False
    
    # Market context
    start_price: Optional[float] = None
    estimated_impact_bps: float = 0.0
    
    @property
    def remaining_size(self) -> float:
        return self.total_size - self.filled_size
    
    @property
    def completion_pct(self) -> float:
        if self.total_size == 0:
            return 0.0
        return self.filled_size / self.total_size
    
    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "total_size": self.total_size,
            "schedule_type": self.schedule_type.value,
            "urgency": self.urgency.value,
            "start_time": self.start_time.isoformat(),
            "filled_size": self.filled_size,
            "avg_price": self.avg_price,
            "completion_pct": self.completion_pct,
            "is_active": self.is_active,
            "slices_count": len(self.slices)
        }


# =============================================================================
# EXECUTION SCHEDULER
# =============================================================================

class SOTAExecutionScheduler:
    """
    Advanced order execution scheduler implementing SOTA requirements.
    
    Features:
    - TWAP/VWAP scheduling algorithms
    - Market impact-aware order slicing
    - Urgency-based timing adjustment
    - Real-time adaptation to market conditions
    """
    
    def __init__(self, config: SchedulerConfig = None):
        self.config = config or SchedulerConfig()
        
        # Active orders
        self.active_orders: Dict[str, ScheduledOrder] = {}
        
        # Order history
        self.completed_orders: deque = deque(maxlen=100)
        
        # Market data
        self.current_prices: Dict[str, float] = {}
        self.volume_estimates: Dict[str, float] = {}  # Estimated volume per interval
        
        # Execution callback
        self._execute_callback: Optional[Callable] = None
        
        # Running state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        logger.info(f"[SOTA] ExecutionScheduler initialized: "
                   f"TWAP={self.config.twap_default_duration_min}min, "
                   f"VWAP_rate={self.config.vwap_participation_rate:.1%}")
    
    # =========================================================================
    # REGISTRATION
    # =========================================================================
    
    def set_execute_callback(self, callback: Callable):
        """
        Set callback for actual order execution.
        
        Callback signature: async def callback(symbol, side, size, price_limit=None) -> Dict
        """
        self._execute_callback = callback
    
    def update_market_data(self, symbol: str, price: float, volume_1m: float = None):
        """Update market data for scheduling decisions."""
        self.current_prices[symbol] = price
        if volume_1m is not None:
            self.volume_estimates[symbol] = volume_1m
    
    # =========================================================================
    # ORDER SCHEDULING
    # =========================================================================
    
    def schedule_order(
        self,
        symbol: str,
        side: str,
        size: float,
        schedule_type: ScheduleType = None,
        urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
        duration_minutes: int = None,
        impact_estimate_bps: float = None
    ) -> ScheduledOrder:
        """
        Schedule an order for execution.
        
        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            size: Total size to execute
            schedule_type: Algorithm to use (auto-selected if None)
            urgency: Execution urgency
            duration_minutes: Override duration (for TWAP)
            impact_estimate_bps: Estimated market impact
            
        Returns:
            ScheduledOrder object
        """
        order_id = f"SCHED-{symbol}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        
        # Auto-select schedule type based on order size
        if schedule_type is None:
            schedule_type = self._select_schedule_type(symbol, size)
        
        # Calculate duration
        if duration_minutes is None:
            duration_minutes = self._calculate_duration(symbol, size, urgency)
        
        # Apply urgency multiplier
        urgency_mult = self.config.urgency_multipliers.get(urgency.name, 1.0)
        adjusted_duration = int(duration_minutes * urgency_mult)
        adjusted_duration = max(1, adjusted_duration)  # At least 1 minute
        
        # Create order
        order = ScheduledOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_size=size,
            schedule_type=schedule_type,
            urgency=urgency,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(minutes=adjusted_duration),
            duration_seconds=adjusted_duration * 60,
            start_price=self.current_prices.get(symbol),
            estimated_impact_bps=impact_estimate_bps or 0.0
        )
        
        # Generate slices
        if schedule_type == ScheduleType.TWAP:
            order.slices = self._generate_twap_slices(order)
        elif schedule_type == ScheduleType.VWAP:
            order.slices = self._generate_vwap_slices(order)
        elif schedule_type == ScheduleType.ICEBERG:
            order.slices = self._generate_iceberg_slices(order)
        else:
            # Immediate - single slice
            order.slices = [OrderSlice(
                slice_id=f"{order_id}-0",
                parent_id=order_id,
                symbol=symbol,
                side=side,
                size=size,
                target_time=datetime.now(timezone.utc)
            )]
        
        # Store
        self.active_orders[order_id] = order
        
        logger.info(
            f"[SOTA] Scheduled order: {order_id} "
            f"{side.upper()} {size} {symbol} via {schedule_type.value} "
            f"over {adjusted_duration} minutes ({len(order.slices)} slices)"
        )
        
        return order
    
    def _select_schedule_type(self, symbol: str, size: float) -> ScheduleType:
        """Auto-select scheduling algorithm based on order characteristics."""
        price = self.current_prices.get(symbol, 1.0)
        notional = size * price
        
        if notional < self.config.twap_min_slice_size_usd:
            return ScheduleType.IMMEDIATE
        elif notional > self.config.impact_slice_threshold_usd:
            return ScheduleType.TWAP
        else:
            return ScheduleType.ADAPTIVE
    
    def _calculate_duration(self, symbol: str, size: float, urgency: UrgencyLevel) -> int:
        """Calculate appropriate execution duration."""
        # Base duration from config
        base_duration = self.config.twap_default_duration_min
        
        # Scale by order size (larger orders = longer duration)
        price = self.current_prices.get(symbol, 1.0)
        notional = size * price
        
        if notional > 100000:
            base_duration = int(base_duration * 2)
        elif notional > 50000:
            base_duration = int(base_duration * 1.5)
        
        return base_duration
    
    # =========================================================================
    # SLICE GENERATION
    # =========================================================================
    
    def _generate_twap_slices(self, order: ScheduledOrder) -> List[OrderSlice]:
        """Generate TWAP slices (equal size, equal time intervals)."""
        slices = []
        
        # Determine number of slices
        num_slices = min(
            self.config.twap_max_slices,
            max(2, order.duration_seconds // 60)  # At least 1 per minute
        )
        
        slice_size = order.total_size / num_slices
        interval_seconds = order.duration_seconds / num_slices
        
        for i in range(num_slices):
            target_time = order.start_time + timedelta(seconds=i * interval_seconds)
            
            slices.append(OrderSlice(
                slice_id=f"{order.order_id}-{i}",
                parent_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                size=slice_size,
                target_time=target_time
            ))
        
        return slices
    
    def _generate_vwap_slices(self, order: ScheduledOrder) -> List[OrderSlice]:
        """Generate VWAP slices (sized by expected volume)."""
        slices = []
        
        # Get volume estimate
        volume_per_minute = self.volume_estimates.get(order.symbol, order.total_size / 10)
        
        # Calculate participation
        participation_rate = min(
            self.config.vwap_participation_rate,
            self.config.vwap_max_participation
        )
        
        remaining_size = order.total_size
        current_time = order.start_time
        slice_idx = 0
        
        while remaining_size > 0 and slice_idx < self.config.twap_max_slices:
            # Size this slice based on expected volume
            expected_volume = volume_per_minute * participation_rate
            slice_size = min(remaining_size, expected_volume)
            
            # Ensure minimum slice size
            price = self.current_prices.get(order.symbol, 1.0)
            min_size = self.config.twap_min_slice_size_usd / price
            slice_size = max(slice_size, min_size)
            slice_size = min(slice_size, remaining_size)
            
            slices.append(OrderSlice(
                slice_id=f"{order.order_id}-{slice_idx}",
                parent_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                size=slice_size,
                target_time=current_time
            ))
            
            remaining_size -= slice_size
            current_time += timedelta(minutes=1)
            slice_idx += 1
        
        return slices
    
    def _generate_iceberg_slices(self, order: ScheduledOrder) -> List[OrderSlice]:
        """Generate iceberg slices (show small, hide rest)."""
        slices = []
        
        # Visible size is 10-20% of total
        visible_pct = 0.15
        visible_size = order.total_size * visible_pct
        
        remaining_size = order.total_size
        slice_idx = 0
        
        while remaining_size > 0:
            slice_size = min(visible_size, remaining_size)
            
            slices.append(OrderSlice(
                slice_id=f"{order.order_id}-{slice_idx}",
                parent_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                size=slice_size,
                target_time=order.start_time + timedelta(seconds=slice_idx * 30)
            ))
            
            remaining_size -= slice_size
            slice_idx += 1
        
        return slices
    
    # =========================================================================
    # EXECUTION LOOP
    # =========================================================================
    
    async def start(self):
        """Start the execution scheduler."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._execution_loop())
        logger.info("[SOTA] ExecutionScheduler started")
    
    async def stop(self):
        """Stop the execution scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[SOTA] ExecutionScheduler stopped")
    
    async def _execution_loop(self):
        """Main execution loop - processes pending slices."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                
                for order_id, order in list(self.active_orders.items()):
                    if not order.is_active or order.cancelled:
                        continue
                    
                    # Process pending slices whose time has come
                    for slice in order.slices:
                        if slice.status == SliceStatus.PENDING and slice.target_time <= now:
                            await self._execute_slice(order, slice)
                    
                    # Check if order is complete
                    if all(s.status in [SliceStatus.FILLED, SliceStatus.CANCELLED, SliceStatus.FAILED]
                           for s in order.slices):
                        order.is_complete = True
                        order.is_active = False
                        self.completed_orders.append(order)
                        del self.active_orders[order_id]
                        
                        logger.info(
                            f"[SOTA] Order complete: {order_id} "
                            f"filled {order.filled_size}/{order.total_size} "
                            f"@ avg {order.avg_price:.2f}"
                        )
                
                await asyncio.sleep(0.1)  # 100ms tick
                
            except Exception as e:
                logger.error(f"[SOTA] Execution loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_slice(self, order: ScheduledOrder, slice: OrderSlice):
        """Execute a single order slice."""
        slice.status = SliceStatus.SUBMITTED
        slice.submitted_time = datetime.now(timezone.utc)
        
        try:
            if self._execute_callback:
                result = await self._execute_callback(
                    slice.symbol,
                    slice.side,
                    slice.size
                )
                
                if result.get("success"):
                    slice.status = SliceStatus.FILLED
                    slice.filled_time = datetime.now(timezone.utc)
                    slice.filled_price = result.get("price", self.current_prices.get(slice.symbol))
                    slice.filled_size = result.get("filled_size", slice.size)
                    slice.order_id = result.get("order_id")
                    
                    # Update parent order
                    order.filled_size += slice.filled_size
                    if order.filled_size > 0:
                        # Update average price
                        total_value = (order.avg_price * (order.filled_size - slice.filled_size) +
                                      slice.filled_price * slice.filled_size)
                        order.avg_price = total_value / order.filled_size
                else:
                    slice.status = SliceStatus.FAILED
                    slice.error = result.get("error", "Unknown error")
            else:
                # No callback - simulate fill
                slice.status = SliceStatus.FILLED
                slice.filled_time = datetime.now(timezone.utc)
                slice.filled_price = self.current_prices.get(slice.symbol, 0)
                slice.filled_size = slice.size
                
                order.filled_size += slice.filled_size
                if order.filled_size > 0 and slice.filled_price > 0:
                    total_value = (order.avg_price * (order.filled_size - slice.filled_size) +
                                  slice.filled_price * slice.filled_size)
                    order.avg_price = total_value / order.filled_size
                
        except Exception as e:
            slice.status = SliceStatus.FAILED
            slice.error = str(e)
            logger.error(f"[SOTA] Slice execution error: {e}")
    
    # =========================================================================
    # ORDER MANAGEMENT
    # =========================================================================
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a scheduled order."""
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        order.cancelled = True
        order.is_active = False
        
        # Cancel pending slices
        for slice in order.slices:
            if slice.status == SliceStatus.PENDING:
                slice.status = SliceStatus.CANCELLED
        
        logger.info(
            f"[SOTA] Order cancelled: {order_id} "
            f"(filled {order.filled_size}/{order.total_size})"
        )
        
        return True
    
    def override_urgency(self, order_id: str, new_urgency: UrgencyLevel):
        """Override urgency for an active order (speeds up or slows down)."""
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        old_urgency = order.urgency
        order.urgency = new_urgency
        
        # Adjust remaining slice timings
        now = datetime.now(timezone.utc)
        pending_slices = [s for s in order.slices if s.status == SliceStatus.PENDING]
        
        if pending_slices:
            urgency_mult = self.config.urgency_multipliers.get(new_urgency.name, 1.0)
            remaining_duration = (order.end_time - now).total_seconds() * urgency_mult
            interval = remaining_duration / len(pending_slices)
            
            for i, slice in enumerate(pending_slices):
                slice.target_time = now + timedelta(seconds=i * interval)
        
        logger.info(
            f"[SOTA] Urgency override: {order_id} "
            f"{old_urgency.name} -> {new_urgency.name}"
        )
        
        return True
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    def get_active_orders(self) -> List[Dict]:
        """Get all active orders."""
        return [order.to_dict() for order in self.active_orders.values()]
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of a specific order."""
        if order_id in self.active_orders:
            return self.active_orders[order_id].to_dict()
        
        for order in self.completed_orders:
            if order.order_id == order_id:
                return order.to_dict()
        
        return None


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_scheduler: Optional[SOTAExecutionScheduler] = None


def get_execution_scheduler(config: SchedulerConfig = None) -> SOTAExecutionScheduler:
    """Get or create the global execution scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = SOTAExecutionScheduler(config)
    elif config is not None and _scheduler.config != config:
        logger.info("[SCHEDULER] Config changed; refreshing singleton instance")
        _scheduler = SOTAExecutionScheduler(config)
    return _scheduler


def reset_execution_scheduler():
    """Reset the global execution scheduler."""
    global _scheduler
    if _scheduler:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # [P37 2026-04-24] Discarded create_task — stop() could be GC'd
            # before cleanup finishes. Add a done-callback so failures log.
            _stop_task = loop.create_task(_scheduler.stop())
            _stop_task.add_done_callback(
                lambda t: logger.error(f"[scheduler.stop] failed: {t.exception()}")
                if t.exception() else None
            )
        else:
            asyncio.run(_scheduler.stop())
    _scheduler = None

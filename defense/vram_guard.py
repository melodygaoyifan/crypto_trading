"""
================================================================================
VRAM GUARD - GPU Memory Management and OOM Prevention
================================================================================
Version: 3.2.0
Purpose: Prevent out-of-memory errors on RTX 5090 (32GB VRAM)

Features:
- VRAM usage monitoring
- KV-cache cleanup for LLM inference
- Batch size adaptation
- OOM prevention with automatic model offloading
- Memory pool management
================================================================================
"""

import time
import logging
import threading
import gc
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable

logger = logging.getLogger("HMATS.VRAMGuard")

# Try to import GPU libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class MemoryPressure(Enum):
    """Memory pressure levels."""
    LOW = auto()        # < 50% VRAM used
    NORMAL = auto()     # 50-70% VRAM used
    HIGH = auto()       # 70-85% VRAM used
    CRITICAL = auto()   # > 85% VRAM used
    OOM_IMMINENT = auto()  # > 95% VRAM used


class ModelPriority(Enum):
    """Model priority for offloading decisions."""
    CRITICAL = auto()   # Never offload (microstructure)
    HIGH = auto()       # Offload last (DRL)
    MEDIUM = auto()     # Offload if needed (SLM)
    LOW = auto()        # Offload first (cache, embeddings)


@dataclass
class VRAMAllocation:
    """VRAM allocation record."""
    name: str
    size_bytes: int
    priority: ModelPriority
    allocated_at: float
    last_used: float
    offloadable: bool = True


@dataclass
class VRAMGuardConfig:
    """VRAM guard configuration."""
    # Memory thresholds (fraction of total VRAM)
    low_threshold: float = 0.50
    normal_threshold: float = 0.70
    high_threshold: float = 0.85
    critical_threshold: float = 0.95
    
    # Total VRAM (RTX 5090)
    total_vram_gb: float = 32.0
    
    # Reserved VRAM for safety
    reserved_vram_gb: float = 2.0
    
    # KV-cache settings
    max_kv_cache_gb: float = 8.0
    kv_cache_cleanup_threshold: float = 0.80
    
    # Batch size adaptation
    min_batch_size: int = 1
    max_batch_size: int = 64
    batch_reduce_on_high: float = 0.5
    
    # Monitoring
    check_interval_seconds: float = 1.0


@dataclass
class VRAMStatus:
    """Current VRAM status."""
    total_bytes: int
    used_bytes: int
    free_bytes: int
    utilization: float
    pressure: MemoryPressure
    timestamp: float = field(default_factory=time.time)


class VRAMMonitor:
    """Monitor GPU VRAM usage."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._last_status: Optional[VRAMStatus] = None
        
    def get_status(self) -> VRAMStatus:
        """Get current VRAM status."""
        total = 0
        used = 0
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                total = torch.cuda.get_device_properties(self.device_id).total_memory
                used = torch.cuda.memory_allocated(self.device_id)
            except Exception as e:
                logger.debug(f"Torch VRAM query failed: {e}")
        
        elif CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                used = mempool.used_bytes()
                total = cp.cuda.Device(self.device_id).mem_info[1]
            except Exception as e:
                logger.debug(f"CuPy VRAM query failed: {e}")
        
        if total == 0:
            # Fallback to config default
            total = int(32 * 1024**3)  # 32GB
            used = 0
        
        free = total - used
        utilization = used / total if total > 0 else 0.0
        
        # Determine pressure level
        if utilization >= 0.95:
            pressure = MemoryPressure.OOM_IMMINENT
        elif utilization >= 0.85:
            pressure = MemoryPressure.CRITICAL
        elif utilization >= 0.70:
            pressure = MemoryPressure.HIGH
        elif utilization >= 0.50:
            pressure = MemoryPressure.NORMAL
        else:
            pressure = MemoryPressure.LOW
        
        status = VRAMStatus(
            total_bytes=total,
            used_bytes=used,
            free_bytes=free,
            utilization=utilization,
            pressure=pressure
        )
        
        self._last_status = status
        return status


class KVCacheManager:
    """Manage KV-cache for LLM inference."""
    
    def __init__(self, max_cache_bytes: int):
        self.max_cache_bytes = max_cache_bytes
        self.current_cache_bytes = 0
        self.cache_entries: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def allocate(self, key: str, size_bytes: int) -> bool:
        """Allocate KV-cache entry."""
        with self._lock:
            if self.current_cache_bytes + size_bytes > self.max_cache_bytes:
                return False
            
            self.cache_entries[key] = size_bytes
            self.current_cache_bytes += size_bytes
            return True
    
    def deallocate(self, key: str):
        """Deallocate KV-cache entry."""
        with self._lock:
            if key in self.cache_entries:
                self.current_cache_bytes -= self.cache_entries[key]
                del self.cache_entries[key]
    
    def clear(self):
        """Clear all KV-cache."""
        with self._lock:
            self.cache_entries.clear()
            self.current_cache_bytes = 0
            
            # Force garbage collection
            gc.collect()
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if CUPY_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
    
    def get_usage(self) -> Tuple[int, int]:
        """Get (used, max) cache bytes."""
        return self.current_cache_bytes, self.max_cache_bytes


class BatchSizeAdapter:
    """Adapt batch size based on memory pressure."""
    
    def __init__(
        self,
        min_batch: int = 1,
        max_batch: int = 64,
        initial_batch: int = 32
    ):
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.current_batch = initial_batch
        self._lock = threading.Lock()
    
    def adapt(self, pressure: MemoryPressure) -> int:
        """Adapt batch size based on pressure."""
        with self._lock:
            if pressure == MemoryPressure.OOM_IMMINENT:
                self.current_batch = self.min_batch
            elif pressure == MemoryPressure.CRITICAL:
                self.current_batch = max(self.min_batch, self.current_batch // 2)
            elif pressure == MemoryPressure.HIGH:
                self.current_batch = max(self.min_batch, int(self.current_batch * 0.75))
            elif pressure == MemoryPressure.LOW:
                self.current_batch = min(self.max_batch, self.current_batch + 4)
            # NORMAL: keep current
            
            return self.current_batch
    
    def get_batch_size(self) -> int:
        """Get current batch size."""
        return self.current_batch


class VRAMGuard:
    """
    Main VRAM guard coordinating memory management.
    
    Prevents OOM by:
    - Monitoring VRAM usage
    - Clearing KV-cache when needed
    - Adapting batch sizes
    - Offloading low-priority models
    """
    
    def __init__(self, config: VRAMGuardConfig = None):
        self.config = config or VRAMGuardConfig()
        
        # Components
        self.monitor = VRAMMonitor()
        self.kv_cache = KVCacheManager(
            max_cache_bytes=int(self.config.max_kv_cache_gb * 1024**3)
        )
        self.batch_adapter = BatchSizeAdapter(
            min_batch=self.config.min_batch_size,
            max_batch=self.config.max_batch_size
        )
        
        # Allocation tracking
        self.allocations: Dict[str, VRAMAllocation] = {}
        
        # Callbacks
        self.on_high_pressure: Optional[Callable] = None
        self.on_critical_pressure: Optional[Callable] = None
        self.on_oom_imminent: Optional[Callable] = None
        
        # Background monitoring
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        self._lock = threading.Lock()
        
        logger.info(f"VRAMGuard initialized: {self.config.total_vram_gb}GB total")
    
    def start_monitoring(self):
        """Start background VRAM monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("VRAM monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                self.check_and_respond()
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            
            time.sleep(self.config.check_interval_seconds)
    
    def check_and_respond(self) -> VRAMStatus:
        """Check VRAM and respond to pressure."""
        status = self.monitor.get_status()
        
        with self._lock:
            # Adapt batch size
            self.batch_adapter.adapt(status.pressure)
            
            # Respond to pressure
            if status.pressure == MemoryPressure.OOM_IMMINENT:
                logger.critical(f"OOM imminent! VRAM: {status.utilization:.1%}")
                self._emergency_cleanup()
                if self.on_oom_imminent:
                    self.on_oom_imminent(status)
                    
            elif status.pressure == MemoryPressure.CRITICAL:
                logger.warning(f"Critical VRAM pressure: {status.utilization:.1%}")
                self._aggressive_cleanup()
                if self.on_critical_pressure:
                    self.on_critical_pressure(status)
                    
            elif status.pressure == MemoryPressure.HIGH:
                logger.info(f"High VRAM pressure: {status.utilization:.1%}")
                self._gentle_cleanup()
                if self.on_high_pressure:
                    self.on_high_pressure(status)
        
        return status
    
    def _emergency_cleanup(self):
        """Emergency cleanup to prevent OOM."""
        logger.warning("Executing emergency VRAM cleanup")
        
        # Clear all KV-cache
        self.kv_cache.clear()
        
        # Offload low-priority allocations
        for name, alloc in sorted(
            self.allocations.items(),
            key=lambda x: x[1].priority.value,
            reverse=True  # LOW priority first
        ):
            if alloc.offloadable and alloc.priority.value >= ModelPriority.MEDIUM.value:
                self._offload_allocation(name)
        
        # Force GC
        gc.collect()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _aggressive_cleanup(self):
        """Aggressive cleanup for critical pressure."""
        # Clear KV-cache if above threshold
        kv_used, kv_max = self.kv_cache.get_usage()
        if kv_used / max(kv_max, 1) > self.config.kv_cache_cleanup_threshold:
            self.kv_cache.clear()
        
        # GC
        gc.collect()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _gentle_cleanup(self):
        """Gentle cleanup for high pressure."""
        gc.collect()
    
    def _offload_allocation(self, name: str):
        """Offload allocation to CPU."""
        if name not in self.allocations:
            return
        
        alloc = self.allocations[name]
        logger.info(f"Offloading allocation: {name} ({alloc.size_bytes / 1024**2:.1f}MB)")
        
        # In production, would move tensor to CPU
        del self.allocations[name]
    
    def register_allocation(
        self,
        name: str,
        size_bytes: int,
        priority: ModelPriority = ModelPriority.MEDIUM,
        offloadable: bool = True
    ):
        """Register VRAM allocation."""
        with self._lock:
            self.allocations[name] = VRAMAllocation(
                name=name,
                size_bytes=size_bytes,
                priority=priority,
                allocated_at=time.time(),
                last_used=time.time(),
                offloadable=offloadable
            )
    
    def unregister_allocation(self, name: str):
        """Unregister VRAM allocation."""
        with self._lock:
            self.allocations.pop(name, None)
    
    def touch_allocation(self, name: str):
        """Mark allocation as recently used."""
        with self._lock:
            if name in self.allocations:
                self.allocations[name].last_used = time.time()
    
    def get_recommended_batch_size(self) -> int:
        """Get recommended batch size."""
        return self.batch_adapter.get_batch_size()
    
    def can_allocate(self, size_bytes: int) -> bool:
        """Check if allocation is safe."""
        status = self.monitor.get_status()
        available = status.free_bytes - int(self.config.reserved_vram_gb * 1024**3)
        return size_bytes < available
    
    def get_status(self) -> Dict:
        """Get complete VRAM guard status."""
        vram_status = self.monitor.get_status()
        kv_used, kv_max = self.kv_cache.get_usage()
        
        return {
            'vram': {
                'total_gb': vram_status.total_bytes / 1024**3,
                'used_gb': vram_status.used_bytes / 1024**3,
                'free_gb': vram_status.free_bytes / 1024**3,
                'utilization': vram_status.utilization,
                'pressure': vram_status.pressure.name
            },
            'kv_cache': {
                'used_gb': kv_used / 1024**3,
                'max_gb': kv_max / 1024**3,
                'utilization': kv_used / max(kv_max, 1)
            },
            'batch_size': self.batch_adapter.get_batch_size(),
            'allocations': len(self.allocations),
            'monitoring': self._monitoring
        }

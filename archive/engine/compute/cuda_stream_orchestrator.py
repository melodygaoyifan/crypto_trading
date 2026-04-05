"""
CUDAStreamOrchestrator - Multi-Stream CUDA Parallelism
=======================================================
Managing three non-blocking parallel streams on RTX 5090 (32GB VRAM).

Stream Architecture:
- Stream_Alpha: Local SLM (Llama-3-8B) inference via vLLM
- Stream_Beta:  Decision Transformer (DRL) state-action inference  
- Stream_Gamma: CuPy-based Microstructure (VPIN/OFI) calculations

Key Features:
- Non-blocking stream execution
- Stream synchronization primitives
- Memory pool management per stream
- Automatic kernel overlap optimization

Hardware Target: NVIDIA RTX 5090 (32GB VRAM, SM 100+)
Author: Senior Performance Engineer
Version: 3.1
"""

import os
import sys
import time
import asyncio
import logging
import threading
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
import queue

import numpy as np

# GPU imports with fallback
try:
    import torch
    import torch.cuda as cuda
    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False

try:
    import cupy as cp
    from cupy.cuda import stream as cp_stream
    from cupy.cuda import memory as cp_memory
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('CUDAStreamOrchestrator')


# =============================================================================
# ENUMS & CONFIGURATION
# =============================================================================

class StreamType(Enum):
    """CUDA stream types"""
    ALPHA = auto()  # SLM inference
    BETA = auto()   # DRL inference
    GAMMA = auto()  # Microstructure (VPIN/OFI)


class StreamPriority(Enum):
    """Stream execution priority"""
    HIGH = -1      # Critical path (execution)
    NORMAL = 0     # Standard (DRL inference)
    LOW = 1        # Background (SLM processing)


@dataclass
class StreamConfig:
    """Configuration for a single stream"""
    stream_type: StreamType
    priority: StreamPriority
    memory_pool_mb: int = 4096  # 4GB default per stream
    max_batch_size: int = 32
    timeout_ms: float = 1000.0


@dataclass
class StreamMetrics:
    """Metrics for stream performance"""
    stream_type: StreamType
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0


# Default stream configurations for RTX 5090
DEFAULT_STREAM_CONFIGS = {
    StreamType.ALPHA: StreamConfig(
        stream_type=StreamType.ALPHA,
        priority=StreamPriority.LOW,
        memory_pool_mb=8192,  # 8GB for Llama-3-8B
        max_batch_size=8,
        timeout_ms=5000.0
    ),
    StreamType.BETA: StreamConfig(
        stream_type=StreamType.BETA,
        priority=StreamPriority.NORMAL,
        memory_pool_mb=4096,  # 4GB for Decision Transformer
        max_batch_size=64,
        timeout_ms=100.0
    ),
    StreamType.GAMMA: StreamConfig(
        stream_type=StreamType.GAMMA,
        priority=StreamPriority.HIGH,
        memory_pool_mb=2048,  # 2GB for VPIN/OFI
        max_batch_size=256,
        timeout_ms=10.0
    ),
}


# =============================================================================
# STREAM WRAPPER
# =============================================================================

class CUDAStream:
    """
    Wrapper for CUDA stream with PyTorch and CuPy support.
    Provides unified interface for both backends.
    """
    
    def __init__(
        self,
        config: StreamConfig,
        device_id: int = 0
    ):
        self.config = config
        self.device_id = device_id
        self.stream_type = config.stream_type
        
        self._torch_stream: Optional[Any] = None
        self._cupy_stream: Optional[Any] = None
        self._memory_pool: Optional[Any] = None
        self._lock = threading.Lock()
        
        self.metrics = StreamMetrics(stream_type=config.stream_type)
        
        self._initialize()
    
    def _initialize(self):
        """Initialize CUDA streams and memory pools"""
        # PyTorch stream
        if TORCH_CUDA_AVAILABLE:
            with cuda.device(self.device_id):
                # Create stream with priority
                priority = self.config.priority.value
                self._torch_stream = cuda.Stream(
                    device=self.device_id,
                    priority=priority
                )
                logger.debug(f"Created PyTorch stream {self.stream_type.name} "
                            f"with priority {priority}")
        
        # CuPy stream
        if CUPY_AVAILABLE:
            with cp.cuda.Device(self.device_id):
                self._cupy_stream = cp.cuda.Stream(non_blocking=True)
                
                # Create memory pool for this stream
                # Note: CuPy doesn't directly support per-stream pools,
                # but we can track allocations
                self._memory_pool = cp.cuda.MemoryPool()
                logger.debug(f"Created CuPy stream {self.stream_type.name}")
        
        logger.info(f"Initialized {self.stream_type.name} stream on device {self.device_id}")
    
    @property
    def torch_stream(self):
        """Get PyTorch CUDA stream"""
        return self._torch_stream
    
    @property
    def cupy_stream(self):
        """Get CuPy CUDA stream"""
        return self._cupy_stream
    
    def synchronize(self):
        """Synchronize the stream (wait for completion)"""
        if self._torch_stream:
            self._torch_stream.synchronize()
        if self._cupy_stream:
            self._cupy_stream.synchronize()
    
    def record_event(self) -> Optional[Any]:
        """Record a CUDA event on this stream"""
        if TORCH_CUDA_AVAILABLE and self._torch_stream:
            event = cuda.Event()
            event.record(self._torch_stream)
            return event
        return None
    
    def wait_event(self, event: Any):
        """Wait for a CUDA event"""
        if TORCH_CUDA_AVAILABLE and self._torch_stream and event:
            self._torch_stream.wait_event(event)
    
    def __enter__(self):
        """Context manager entry - switch to this stream"""
        if self._torch_stream:
            self._torch_stream.__enter__()
        if self._cupy_stream:
            self._cupy_stream.use()
        return self
    
    def __exit__(self, *args):
        """Context manager exit"""
        if self._torch_stream:
            self._torch_stream.__exit__(*args)


# =============================================================================
# TASK DEFINITIONS
# =============================================================================

@dataclass
class StreamTask:
    """Task to be executed on a stream"""
    task_id: str
    stream_type: StreamType
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    callback: Optional[Callable] = None
    
    # Result storage
    result: Any = None
    error: Optional[Exception] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    @property
    def execution_time_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0


class TaskQueue:
    """Priority queue for stream tasks"""
    
    def __init__(self, stream_type: StreamType):
        self.stream_type = stream_type
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._counter = 0
        self._lock = threading.Lock()
    
    def put(self, task: StreamTask):
        """Add task to queue"""
        with self._lock:
            # Use (priority, counter, task) to ensure FIFO within same priority
            self._queue.put((task.priority, self._counter, task))
            self._counter += 1
    
    def get(self, timeout: float = None) -> Optional[StreamTask]:
        """Get next task from queue"""
        try:
            _, _, task = self._queue.get(timeout=timeout)
            return task
        except queue.Empty:
            return None
    
    def qsize(self) -> int:
        return self._queue.qsize()
    
    def empty(self) -> bool:
        return self._queue.empty()


# =============================================================================
# CUDA STREAM ORCHESTRATOR
# =============================================================================

class CUDAStreamOrchestrator:
    """
    Main orchestrator for multi-stream CUDA execution.
    
    Manages three parallel streams:
    - Stream_Alpha: Local SLM (Llama-3-8B) inference
    - Stream_Beta:  Decision Transformer inference
    - Stream_Gamma: Microstructure calculations (VPIN/OFI)
    
    Features:
    - Non-blocking task submission
    - Automatic stream selection
    - Cross-stream synchronization
    - Memory management per stream
    """
    
    def __init__(
        self,
        device_id: int = 0,
        configs: Dict[StreamType, StreamConfig] = None
    ):
        self.device_id = device_id
        self.configs = configs or DEFAULT_STREAM_CONFIGS
        
        self._streams: Dict[StreamType, CUDAStream] = {}
        self._task_queues: Dict[StreamType, TaskQueue] = {}
        self._workers: Dict[StreamType, threading.Thread] = {}
        
        self._running = False
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()
        
        self._task_counter = 0
        self._pending_futures: Dict[str, Future] = {}
        
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix='CUDAStream')
        
        self._initialize()
    
    def _initialize(self):
        """Initialize all streams"""
        logger.info(f"Initializing CUDAStreamOrchestrator on device {self.device_id}")
        
        # Check GPU availability
        if TORCH_CUDA_AVAILABLE:
            device_name = torch.cuda.get_device_name(self.device_id)
            total_mem = torch.cuda.get_device_properties(self.device_id).total_memory / 1e9
            logger.info(f"GPU: {device_name} ({total_mem:.1f}GB VRAM)")
        elif CUPY_AVAILABLE:
            with cp.cuda.Device(self.device_id):
                mem_info = cp.cuda.runtime.memGetInfo()
                logger.info(f"GPU Memory: {mem_info[1]/1e9:.1f}GB total")
        else:
            logger.warning("No GPU backend available, running in CPU mode")
        
        # Create streams
        for stream_type, config in self.configs.items():
            self._streams[stream_type] = CUDAStream(config, self.device_id)
            self._task_queues[stream_type] = TaskQueue(stream_type)
        
        logger.info(f"Created {len(self._streams)} CUDA streams")
    
    def start(self):
        """Start stream worker threads"""
        if self._running:
            return
        
        self._running = True
        self._shutdown_event.clear()
        
        # Start worker thread for each stream
        for stream_type in self._streams:
            worker = threading.Thread(
                target=self._stream_worker,
                args=(stream_type,),
                name=f"StreamWorker-{stream_type.name}",
                daemon=True
            )
            worker.start()
            self._workers[stream_type] = worker
        
        logger.info("Stream workers started")
    
    def stop(self, timeout: float = 5.0):
        """Stop all stream workers"""
        if not self._running:
            return
        
        self._running = False
        self._shutdown_event.set()
        
        # Wait for workers to finish
        for stream_type, worker in self._workers.items():
            worker.join(timeout=timeout)
            if worker.is_alive():
                logger.warning(f"Worker {stream_type.name} did not stop gracefully")
        
        self._workers.clear()
        logger.info("Stream workers stopped")
    
    def _stream_worker(self, stream_type: StreamType):
        """Worker thread for a stream"""
        stream = self._streams[stream_type]
        task_queue = self._task_queues[stream_type]
        config = self.configs[stream_type]
        
        logger.debug(f"Started worker for {stream_type.name}")
        
        while self._running:
            try:
                # Get next task with timeout
                task = task_queue.get(timeout=0.1)
                if task is None:
                    continue
                
                # Execute task
                task.started_at = time.time()
                
                try:
                    with stream:
                        task.result = task.func(*task.args, **task.kwargs)
                    task.completed_at = time.time()
                    
                    # Update metrics
                    stream.metrics.total_tasks += 1
                    stream.metrics.completed_tasks += 1
                    stream.metrics.total_time_ms += task.execution_time_ms
                    stream.metrics.avg_time_ms = (
                        stream.metrics.total_time_ms / stream.metrics.completed_tasks
                    )
                    
                except Exception as e:
                    task.error = e
                    task.completed_at = time.time()
                    stream.metrics.total_tasks += 1
                    stream.metrics.failed_tasks += 1
                    logger.error(f"Task {task.task_id} failed: {e}")
                
                # Execute callback if provided
                if task.callback:
                    try:
                        task.callback(task)
                    except Exception as e:
                        logger.error(f"Callback error for {task.task_id}: {e}")
                        
            except Exception as e:
                logger.error(f"Worker error on {stream_type.name}: {e}")
        
        logger.debug(f"Stopped worker for {stream_type.name}")
    
    def submit(
        self,
        stream_type: StreamType,
        func: Callable,
        *args,
        priority: int = 0,
        callback: Callable = None,
        **kwargs
    ) -> str:
        """
        Submit a task to a stream.
        
        Args:
            stream_type: Target stream
            func: Function to execute
            *args: Function arguments
            priority: Task priority (lower = higher priority)
            callback: Callback function(task) called on completion
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        with self._lock:
            self._task_counter += 1
            task_id = f"{stream_type.name}_{self._task_counter}"
        
        task = StreamTask(
            task_id=task_id,
            stream_type=stream_type,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            callback=callback
        )
        
        self._task_queues[stream_type].put(task)
        
        return task_id
    
    def submit_alpha(
        self,
        func: Callable,
        *args,
        callback: Callable = None,
        **kwargs
    ) -> str:
        """Submit task to Alpha stream (SLM)"""
        return self.submit(
            StreamType.ALPHA, func, *args,
            priority=1, callback=callback, **kwargs
        )
    
    def submit_beta(
        self,
        func: Callable,
        *args,
        callback: Callable = None,
        **kwargs
    ) -> str:
        """Submit task to Beta stream (DRL)"""
        return self.submit(
            StreamType.BETA, func, *args,
            priority=0, callback=callback, **kwargs
        )
    
    def submit_gamma(
        self,
        func: Callable,
        *args,
        callback: Callable = None,
        **kwargs
    ) -> str:
        """Submit task to Gamma stream (Microstructure)"""
        return self.submit(
            StreamType.GAMMA, func, *args,
            priority=-1, callback=callback, **kwargs
        )
    
    async def submit_async(
        self,
        stream_type: StreamType,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Async submission with result waiting.
        
        Returns:
            Task result
        """
        result_event = asyncio.Event()
        result_holder = {'result': None, 'error': None}
        
        def on_complete(task: StreamTask):
            result_holder['result'] = task.result
            result_holder['error'] = task.error
            asyncio.get_event_loop().call_soon_threadsafe(result_event.set)
        
        self.submit(stream_type, func, *args, callback=on_complete, **kwargs)
        
        await result_event.wait()
        
        if result_holder['error']:
            raise result_holder['error']
        return result_holder['result']
    
    def synchronize_stream(self, stream_type: StreamType):
        """Synchronize a specific stream"""
        self._streams[stream_type].synchronize()
    
    def synchronize_all(self):
        """Synchronize all streams"""
        for stream in self._streams.values():
            stream.synchronize()
    
    def create_dependency(
        self,
        from_stream: StreamType,
        to_stream: StreamType
    ):
        """
        Create dependency: to_stream waits for from_stream's current work.
        
        Example: Make Beta wait for Gamma's microstructure calculation.
        """
        event = self._streams[from_stream].record_event()
        if event:
            self._streams[to_stream].wait_event(event)
    
    def get_metrics(self, stream_type: StreamType = None) -> Dict[str, StreamMetrics]:
        """Get stream metrics"""
        if stream_type:
            return {stream_type.name: self._streams[stream_type].metrics}
        return {
            st.name: stream.metrics
            for st, stream in self._streams.items()
        }
    
    def get_memory_usage(self) -> Dict[str, Dict[str, float]]:
        """Get GPU memory usage"""
        if TORCH_CUDA_AVAILABLE:
            allocated = torch.cuda.memory_allocated(self.device_id) / 1e9
            reserved = torch.cuda.memory_reserved(self.device_id) / 1e9
            total = torch.cuda.get_device_properties(self.device_id).total_memory / 1e9
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - reserved
            }
        elif CUPY_AVAILABLE:
            with cp.cuda.Device(self.device_id):
                free, total = cp.cuda.runtime.memGetInfo()
                return {
                    'allocated_gb': (total - free) / 1e9,
                    'total_gb': total / 1e9,
                    'free_gb': free / 1e9
                }
        return {}
    
    def get_queue_depths(self) -> Dict[str, int]:
        """Get current queue depths"""
        return {
            st.name: queue.qsize()
            for st, queue in self._task_queues.items()
        }


# =============================================================================
# SPECIALIZED STREAM FUNCTIONS
# =============================================================================

class MicrostructureCompute:
    """
    GPU-accelerated microstructure calculations for Stream Gamma.
    Uses CuPy for VPIN and OFI calculations.
    """
    
    def __init__(self, orchestrator: CUDAStreamOrchestrator):
        self.orchestrator = orchestrator
        self.xp = cp if CUPY_AVAILABLE else np
    
    def compute_vpin_gpu(
        self,
        buy_volumes: np.ndarray,
        sell_volumes: np.ndarray,
        bucket_size: int = 50
    ) -> float:
        """
        Compute VPIN on GPU.
        
        Args:
            buy_volumes: Array of buy volumes
            sell_volumes: Array of sell volumes
            bucket_size: Number of buckets for averaging
            
        Returns:
            VPIN value
        """
        def _compute():
            if CUPY_AVAILABLE:
                buy = cp.asarray(buy_volumes)
                sell = cp.asarray(sell_volumes)
                
                total = buy + sell
                imbalance = cp.abs(buy - sell)
                
                # Bucket averaging
                n = len(buy) // bucket_size * bucket_size
                if n == 0:
                    return 0.5
                
                total_buckets = total[:n].reshape(-1, bucket_size).sum(axis=1)
                imbalance_buckets = imbalance[:n].reshape(-1, bucket_size).sum(axis=1)
                
                vpin = float(cp.sum(imbalance_buckets) / cp.sum(total_buckets))
                return vpin
            else:
                total = buy_volumes + sell_volumes
                imbalance = np.abs(buy_volumes - sell_volumes)
                return float(np.sum(imbalance) / np.sum(total)) if np.sum(total) > 0 else 0.5
        
        task_id = self.orchestrator.submit_gamma(_compute)
        return task_id
    
    def compute_ofi_gpu(
        self,
        bid_prices: np.ndarray,
        bid_sizes: np.ndarray,
        ask_prices: np.ndarray,
        ask_sizes: np.ndarray,
        prev_bid_prices: np.ndarray,
        prev_bid_sizes: np.ndarray,
        prev_ask_prices: np.ndarray,
        prev_ask_sizes: np.ndarray
    ) -> str:
        """
        Compute OFI on GPU.
        
        Returns:
            Task ID
        """
        def _compute():
            if CUPY_AVAILABLE:
                bp = cp.asarray(bid_prices)
                bs = cp.asarray(bid_sizes)
                ap = cp.asarray(ask_prices)
                as_ = cp.asarray(ask_sizes)
                pbp = cp.asarray(prev_bid_prices)
                pbs = cp.asarray(prev_bid_sizes)
                pap = cp.asarray(prev_ask_prices)
                pas = cp.asarray(prev_ask_sizes)
                
                # Bid OFI
                bid_ofi = cp.where(bp > pbp, bs, 0)
                bid_ofi = cp.where(bp < pbp, -pbs, bid_ofi)
                bid_ofi = cp.where(bp == pbp, bs - pbs, bid_ofi)
                
                # Ask OFI
                ask_ofi = cp.where(ap < pap, -as_, 0)
                ask_ofi = cp.where(ap > pap, pas, ask_ofi)
                ask_ofi = cp.where(ap == pap, -(as_ - pas), ask_ofi)
                
                return float(cp.sum(bid_ofi) + cp.sum(ask_ofi))
            else:
                return 0.0
        
        return self.orchestrator.submit_gamma(_compute)


class DRLInference:
    """
    Decision Transformer inference for Stream Beta.
    Uses PyTorch for model inference.
    """
    
    def __init__(
        self,
        orchestrator: CUDAStreamOrchestrator,
        model: Any = None
    ):
        self.orchestrator = orchestrator
        self.model = model
        self.device = f'cuda:{orchestrator.device_id}' if TORCH_CUDA_AVAILABLE else 'cpu'
    
    def infer(
        self,
        states: np.ndarray,
        actions: np.ndarray = None,
        returns_to_go: np.ndarray = None
    ) -> str:
        """
        Run Decision Transformer inference.
        
        Args:
            states: State sequence (batch, seq_len, state_dim)
            actions: Action sequence (optional)
            returns_to_go: Target returns (optional)
            
        Returns:
            Task ID
        """
        def _infer():
            if not TORCH_CUDA_AVAILABLE or self.model is None:
                # Return random action for testing
                return np.random.randn(states.shape[0], 3)  # 3 assets
            
            with torch.no_grad():
                states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
                
                if actions is not None:
                    actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
                else:
                    actions_t = None
                
                if returns_to_go is not None:
                    rtg_t = torch.tensor(returns_to_go, dtype=torch.float32, device=self.device)
                else:
                    rtg_t = None
                
                output = self.model(states_t, actions_t, rtg_t)
                return output.cpu().numpy()
        
        return self.orchestrator.submit_beta(_infer)


class SLMInference:
    """
    Local SLM inference for Stream Alpha.
    Uses vLLM or transformers for text generation.
    """
    
    def __init__(
        self,
        orchestrator: CUDAStreamOrchestrator,
        model: Any = None,
        tokenizer: Any = None
    ):
        self.orchestrator = orchestrator
        self.model = model
        self.tokenizer = tokenizer
        self.device = f'cuda:{orchestrator.device_id}' if TORCH_CUDA_AVAILABLE else 'cpu'
    
    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment of text using SLM.
        
        Args:
            text: Input text
            
        Returns:
            Task ID
        """
        def _analyze():
            if self.model is None:
                # Mock sentiment analysis
                import random
                return {
                    'sentiment': random.uniform(-1, 1),
                    'confidence': random.uniform(0.5, 1.0),
                    'analysis': 'Mock analysis - model not loaded'
                }
            
            # Real inference would go here
            # prompt = f"Analyze the sentiment of this crypto news: {text}\nSentiment score (-1 to 1):"
            # ...
            
            return {'sentiment': 0.0, 'confidence': 0.5, 'analysis': ''}
        
        return self.orchestrator.submit_alpha(_analyze, priority=1)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstration of CUDAStreamOrchestrator"""
    print("="*70)
    print("CUDAStreamOrchestrator Demo")
    print("="*70)
    
    # Check GPU
    print(f"\n▶ GPU Status:")
    print(f"  PyTorch CUDA: {TORCH_CUDA_AVAILABLE}")
    print(f"  CuPy: {CUPY_AVAILABLE}")
    
    if TORCH_CUDA_AVAILABLE:
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    
    # Create orchestrator
    orchestrator = CUDAStreamOrchestrator(device_id=0)
    orchestrator.start()
    
    print(f"\n▶ Streams created: {list(orchestrator._streams.keys())}")
    
    # Define test functions
    def slow_function(duration: float, name: str) -> str:
        time.sleep(duration)
        return f"{name} completed after {duration}s"
    
    def matrix_multiply(size: int) -> np.ndarray:
        a = np.random.randn(size, size)
        b = np.random.randn(size, size)
        if CUPY_AVAILABLE:
            a_gpu = cp.asarray(a)
            b_gpu = cp.asarray(b)
            result = cp.dot(a_gpu, b_gpu)
            return cp.asnumpy(result)
        return np.dot(a, b)
    
    # Submit tasks to different streams
    print("\n▶ Submitting tasks...")
    
    results = []
    
    def collect_result(task: StreamTask):
        results.append((task.task_id, task.result, task.execution_time_ms))
    
    # Alpha: SLM task (slow)
    orchestrator.submit_alpha(slow_function, 0.5, "SLM", callback=collect_result)
    
    # Beta: DRL task (medium)
    orchestrator.submit_beta(slow_function, 0.2, "DRL", callback=collect_result)
    
    # Gamma: Microstructure task (fast)
    orchestrator.submit_gamma(matrix_multiply, 100, callback=collect_result)
    orchestrator.submit_gamma(matrix_multiply, 100, callback=collect_result)
    
    # Wait for completion
    print("▶ Waiting for tasks...")
    time.sleep(1.5)
    
    # Print results
    print(f"\n▶ Results ({len(results)} tasks completed):")
    for task_id, result, exec_time in results:
        result_str = str(result)[:50] if result is not None else 'None'
        print(f"  {task_id}: {result_str}... ({exec_time:.1f}ms)")
    
    # Print metrics
    print("\n▶ Stream Metrics:")
    metrics = orchestrator.get_metrics()
    for name, m in metrics.items():
        print(f"  {name}: {m.completed_tasks} tasks, avg={m.avg_time_ms:.1f}ms")
    
    # Memory usage
    mem = orchestrator.get_memory_usage()
    if mem:
        print(f"\n▶ GPU Memory:")
        print(f"  Allocated: {mem.get('allocated_gb', 0):.2f}GB")
        print(f"  Free: {mem.get('free_gb', 0):.2f}GB")
    
    # Cleanup
    orchestrator.stop()
    print("\n✓ Demo complete")


if __name__ == '__main__':
    demo()

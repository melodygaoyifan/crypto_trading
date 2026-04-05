"""
================================================================================
HMATS v4.0-COMPLETE - IDLE MODULE INTEGRATOR (UPDATED)
================================================================================
Version: 4.0.0-UPDATED
Purpose: Wire up idle modules + market quiet period detection + resource management

MIGRATION NOTES:
    - Updated from v3.2 idle_module_integrator.py
    - Integrated with v4 integrated_system.py
    - Added market quiet period detection
    - Added compute resource management hooks

INTEGRATED MODULES:
    - macro_integrator -> risk cap / leverage adjustment
    - SentimentIntegrator -> trigger/veto/mode-switch (NOT linear multipliers)
    - GlobalContextInformer -> context aggregation
    - StateVectorAugmentor -> DRL state enhancement

IDLE PERIOD FEATURES:
    - Detect market quiet periods (low volatility, low volume)
    - Release non-critical compute resources
    - Background task scheduling
    - Module pause/resume hooks

PHILOSOPHY:
    - Sentiment/Macro are TRIGGER/VETO/MODE-SWITCH, NOT linear multipliers
    - Sentiment shock -> OpportunityMode
    - Chaos + conflict -> NO_TRADE
    - Macro adjusts risk cap / leverage multiplier, NEVER flips direction
    - Idle periods = opportunity to run background tasks
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
import time
import threading

logger = logging.getLogger(__name__)

# Try numpy import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# =============================================================================
# ENUMS
# =============================================================================

class MarketActivityLevel(Enum):
    """Market activity classification for idle detection."""
    ACTIVE = "ACTIVE"           # Normal trading activity
    MODERATE = "MODERATE"       # Reduced activity
    QUIET = "QUIET"             # Low activity (potential idle)
    IDLE = "IDLE"               # Very low activity (release resources)


class ModuleStatus(Enum):
    """Status of managed modules."""
    ACTIVE = "ACTIVE"           # Running normally
    PAUSED = "PAUSED"           # Temporarily paused
    SUSPENDED = "SUSPENDED"     # Suspended for resource saving
    ERROR = "ERROR"             # Error state


# =============================================================================
# IDLE DETECTION CONFIG
# =============================================================================

@dataclass
class IdleDetectionConfig:
    """Configuration for idle period detection."""
    
    # Volatility thresholds (realized vol, annualized)
    vol_quiet_threshold: float = 0.15      # Below = QUIET
    vol_idle_threshold: float = 0.08       # Below = IDLE
    
    # Volume thresholds (ratio to 20-day average)
    volume_quiet_threshold: float = 0.50   # Below = QUIET
    volume_idle_threshold: float = 0.25    # Below = IDLE
    
    # Spread thresholds
    spread_quiet_threshold: float = 3.0    # Above = QUIET (bps)
    spread_idle_threshold: float = 5.0     # Above = IDLE
    
    # Time thresholds (minutes to confirm idle)
    quiet_confirmation_minutes: int = 15   # Must be quiet for 15min
    idle_confirmation_minutes: int = 30    # Must be idle for 30min
    
    # Module management
    pausable_modules: List[str] = field(default_factory=lambda: [
        "state_vector_augmentor",
        "quant_data_collector",
        "sentiment_collector",
        "macro_data_integrator",
    ])
    
    suspendable_modules: List[str] = field(default_factory=lambda: [
        "vllm_inference_wrapper",
        "cuda_stream_orchestrator",
    ])


# =============================================================================
# IDLE STATE
# =============================================================================

@dataclass
class IdleState:
    """Current idle/activity state."""
    
    activity_level: MarketActivityLevel = MarketActivityLevel.ACTIVE
    
    # Metrics
    current_volatility: float = 0.0
    current_volume_ratio: float = 1.0
    current_spread_bps: float = 2.0
    
    # Duration tracking
    quiet_since: Optional[datetime] = None
    idle_since: Optional[datetime] = None
    
    # Resource management
    paused_modules: List[str] = field(default_factory=list)
    suspended_modules: List[str] = field(default_factory=list)
    background_tasks_running: List[str] = field(default_factory=list)
    
    # Flags
    is_quiet: bool = False
    is_idle: bool = False
    resources_released: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "activity_level": self.activity_level.value,
            "volatility": self.current_volatility,
            "volume_ratio": self.current_volume_ratio,
            "spread_bps": self.current_spread_bps,
            "is_quiet": self.is_quiet,
            "is_idle": self.is_idle,
            "paused_modules": self.paused_modules,
            "suspended_modules": self.suspended_modules,
        }


# =============================================================================
# BACKGROUND TASK MANAGER
# =============================================================================

@dataclass
class BackgroundTask:
    """Represents a background task to run during idle."""
    name: str
    function: Callable
    priority: int = 0  # Higher = more important
    last_run: Optional[datetime] = None
    interval_minutes: int = 60  # Minimum interval between runs
    requires_idle: bool = True  # Only run in IDLE state


class BackgroundTaskManager:
    """Manages background tasks to run during idle periods."""
    
    def __init__(self):
        self._tasks: List[BackgroundTask] = []
        self._running: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
    
    def register_task(
        self,
        name: str,
        function: Callable,
        priority: int = 0,
        interval_minutes: int = 60,
        requires_idle: bool = True
    ):
        """Register a background task."""
        task = BackgroundTask(
            name=name,
            function=function,
            priority=priority,
            interval_minutes=interval_minutes,
            requires_idle=requires_idle,
        )
        with self._lock:
            self._tasks.append(task)
            self._tasks.sort(key=lambda t: -t.priority)
        logger.debug(f"Registered background task: {name}")
    
    def get_runnable_tasks(
        self,
        is_idle: bool,
        is_quiet: bool
    ) -> List[BackgroundTask]:
        """Get tasks that can run now."""
        now = datetime.utcnow()
        runnable = []
        
        with self._lock:
            for task in self._tasks:
                # Check if already running
                if task.name in self._running:
                    continue
                
                # Check state requirement
                if task.requires_idle and not is_idle:
                    if not is_quiet:
                        continue
                
                # Check interval
                if task.last_run:
                    elapsed = (now - task.last_run).total_seconds() / 60
                    if elapsed < task.interval_minutes:
                        continue
                
                runnable.append(task)
        
        return runnable
    
    def run_task(self, task: BackgroundTask) -> bool:
        """Run a background task in a separate thread."""
        
        def _run():
            try:
                logger.info(f"Starting background task: {task.name}")
                task.function()
                task.last_run = datetime.utcnow()
                logger.info(f"Completed background task: {task.name}")
            except Exception as e:
                logger.error(f"Background task {task.name} failed: {e}")
            finally:
                with self._lock:
                    self._running.pop(task.name, None)
        
        with self._lock:
            if task.name in self._running:
                return False
            
            thread = threading.Thread(target=_run, daemon=True)
            self._running[task.name] = thread
            thread.start()
        
        return True
    
    def get_running_tasks(self) -> List[str]:
        """Get list of currently running task names."""
        with self._lock:
            return list(self._running.keys())


# =============================================================================
# MODULE MANAGER
# =============================================================================

class ModuleManager:
    """Manages pausing/resuming modules during idle periods."""
    
    def __init__(self):
        self._modules: Dict[str, Any] = {}
        self._status: Dict[str, ModuleStatus] = {}
        self._pause_hooks: Dict[str, Callable] = {}
        self._resume_hooks: Dict[str, Callable] = {}
    
    def register_module(
        self,
        name: str,
        module: Any,
        pause_hook: Optional[Callable] = None,
        resume_hook: Optional[Callable] = None,
    ):
        """Register a module for management."""
        self._modules[name] = module
        self._status[name] = ModuleStatus.ACTIVE
        
        if pause_hook:
            self._pause_hooks[name] = pause_hook
        if resume_hook:
            self._resume_hooks[name] = resume_hook
        
        logger.debug(f"Registered module: {name}")
    
    def pause_module(self, name: str) -> bool:
        """Pause a module."""
        if name not in self._modules:
            return False
        
        if self._status.get(name) == ModuleStatus.PAUSED:
            return True
        
        try:
            if name in self._pause_hooks:
                self._pause_hooks[name]()
            self._status[name] = ModuleStatus.PAUSED
            logger.info(f"Paused module: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause {name}: {e}")
            self._status[name] = ModuleStatus.ERROR
            return False
    
    def resume_module(self, name: str) -> bool:
        """Resume a paused module."""
        if name not in self._modules:
            return False
        
        if self._status.get(name) == ModuleStatus.ACTIVE:
            return True
        
        try:
            if name in self._resume_hooks:
                self._resume_hooks[name]()
            self._status[name] = ModuleStatus.ACTIVE
            logger.info(f"Resumed module: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume {name}: {e}")
            self._status[name] = ModuleStatus.ERROR
            return False
    
    def suspend_module(self, name: str) -> bool:
        """Suspend a module (deeper than pause, releases resources)."""
        if name not in self._modules:
            return False
        
        if self._status.get(name) == ModuleStatus.SUSPENDED:
            return True
        
        try:
            # First pause
            self.pause_module(name)
            
            # Then release resources if module has method
            module = self._modules[name]
            if hasattr(module, 'release_resources'):
                module.release_resources()
            
            self._status[name] = ModuleStatus.SUSPENDED
            logger.info(f"Suspended module: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to suspend {name}: {e}")
            self._status[name] = ModuleStatus.ERROR
            return False
    
    def get_status(self, name: str) -> ModuleStatus:
        """Get module status."""
        return self._status.get(name, ModuleStatus.ERROR)
    
    def get_paused_modules(self) -> List[str]:
        """Get list of paused module names."""
        return [n for n, s in self._status.items() if s == ModuleStatus.PAUSED]
    
    def get_suspended_modules(self) -> List[str]:
        """Get list of suspended module names."""
        return [n for n, s in self._status.items() if s == ModuleStatus.SUSPENDED]


# =============================================================================
# MAIN IDLE MODULE INTEGRATOR
# =============================================================================

class IdleModuleIntegrator:
    """
    Main integrator for idle detection and module management.
    
    Combines:
    1. Market activity detection
    2. Module pause/resume management
    3. Background task scheduling
    4. Integration with v4 IntegratedTradingSystem
    
    Usage:
        integrator = get_idle_module_integrator()
        
        # Update with market data
        state = integrator.update(market_data)
        
        # Check if modules should be paused
        if state.is_idle:
            integrator.enter_idle_mode()
        
        # Resume when active
        if state.activity_level == MarketActivityLevel.ACTIVE:
            integrator.exit_idle_mode()
    """
    
    def __init__(self, config: Optional[IdleDetectionConfig] = None):
        self.config = config or IdleDetectionConfig()
        self.state = IdleState()
        
        self.module_manager = ModuleManager()
        self.task_manager = BackgroundTaskManager()
        
        # Integration references
        self._integrated_system: Optional[Any] = None
        
        logger.info("[v4.0-UPDATED] IdleModuleIntegrator initialized")
    
    # =========================================================================
    # INTEGRATION WITH V4
    # =========================================================================
    
    def set_integrated_system(self, system: Any):
        """Set reference to IntegratedTradingSystem."""
        self._integrated_system = system
        logger.debug("IntegratedTradingSystem linked to idle integrator")
    
    def register_system_modules(self):
        """
        Auto-register modules from IntegratedTradingSystem.
        
        Call after setting integrated_system.
        """
        if not self._integrated_system:
            logger.warning("IntegratedTradingSystem not set")
            return
        
        # Register known modules
        for module_name in self.config.pausable_modules:
            if hasattr(self._integrated_system, module_name):
                module = getattr(self._integrated_system, module_name)
                self.module_manager.register_module(module_name, module)
    
    # =========================================================================
    # ACTIVITY DETECTION
    # =========================================================================
    
    def update(self, market_data: Dict) -> IdleState:
        """
        Update idle state based on market data.
        
        Args:
            market_data: Dict with volatility, volume, spread info
        
        Returns:
            Updated IdleState
        """
        
        now = datetime.utcnow()
        
        # Extract metrics
        self.state.current_volatility = market_data.get('realized_vol', 0.20)
        self.state.current_volume_ratio = market_data.get('volume_ratio', 1.0)
        self.state.current_spread_bps = market_data.get('spread_bps', 2.0)
        
        # Determine activity level
        self._determine_activity_level()
        
        # Track duration
        self._track_duration(now)
        
        # Check if confirmed idle
        self._check_confirmed_idle(now)
        
        return self.state
    
    def _determine_activity_level(self):
        """Determine current activity level from metrics."""
        
        vol = self.state.current_volatility
        vol_ratio = self.state.current_volume_ratio
        spread = self.state.current_spread_bps
        
        # Check IDLE conditions (all must be met)
        if (vol < self.config.vol_idle_threshold and
            vol_ratio < self.config.volume_idle_threshold and
            spread > self.config.spread_idle_threshold):
            self.state.activity_level = MarketActivityLevel.IDLE
            
        # Check QUIET conditions (2 of 3 must be met)
        elif sum([
            vol < self.config.vol_quiet_threshold,
            vol_ratio < self.config.volume_quiet_threshold,
            spread > self.config.spread_quiet_threshold,
        ]) >= 2:
            self.state.activity_level = MarketActivityLevel.QUIET
            
        # Check MODERATE (1 condition met)
        elif (vol < self.config.vol_quiet_threshold or
              vol_ratio < self.config.volume_quiet_threshold or
              spread > self.config.spread_quiet_threshold):
            self.state.activity_level = MarketActivityLevel.MODERATE
            
        else:
            self.state.activity_level = MarketActivityLevel.ACTIVE
    
    def _track_duration(self, now: datetime):
        """Track duration in quiet/idle state."""
        
        if self.state.activity_level in [MarketActivityLevel.QUIET, MarketActivityLevel.IDLE]:
            if self.state.quiet_since is None:
                self.state.quiet_since = now
        else:
            self.state.quiet_since = None
            self.state.is_quiet = False
        
        if self.state.activity_level == MarketActivityLevel.IDLE:
            if self.state.idle_since is None:
                self.state.idle_since = now
        else:
            self.state.idle_since = None
            self.state.is_idle = False
    
    def _check_confirmed_idle(self, now: datetime):
        """Check if idle/quiet is confirmed (duration threshold met)."""
        
        # Check quiet confirmation
        if self.state.quiet_since:
            quiet_duration = (now - self.state.quiet_since).total_seconds() / 60
            if quiet_duration >= self.config.quiet_confirmation_minutes:
                self.state.is_quiet = True
        
        # Check idle confirmation
        if self.state.idle_since:
            idle_duration = (now - self.state.idle_since).total_seconds() / 60
            if idle_duration >= self.config.idle_confirmation_minutes:
                self.state.is_idle = True
    
    # =========================================================================
    # IDLE MODE MANAGEMENT
    # =========================================================================
    
    def enter_idle_mode(self):
        """
        Enter idle mode - pause/suspend modules, start background tasks.
        
        Called when market is confirmed idle.
        """
        
        logger.info("Entering idle mode")
        
        # Pause pausable modules
        for module_name in self.config.pausable_modules:
            if self.module_manager.pause_module(module_name):
                if module_name not in self.state.paused_modules:
                    self.state.paused_modules.append(module_name)
        
        # Suspend suspendable modules if deep idle
        if self.state.is_idle:
            for module_name in self.config.suspendable_modules:
                if self.module_manager.suspend_module(module_name):
                    if module_name not in self.state.suspended_modules:
                        self.state.suspended_modules.append(module_name)
            self.state.resources_released = True
        
        # Start background tasks
        self._run_background_tasks()
    
    def exit_idle_mode(self):
        """
        Exit idle mode - resume all modules.
        
        Called when market becomes active again.
        """
        
        logger.info("Exiting idle mode")
        
        # Resume paused modules
        for module_name in list(self.state.paused_modules):
            if self.module_manager.resume_module(module_name):
                self.state.paused_modules.remove(module_name)
        
        # Resume suspended modules
        for module_name in list(self.state.suspended_modules):
            if self.module_manager.resume_module(module_name):
                self.state.suspended_modules.remove(module_name)
        
        self.state.resources_released = False
    
    def _run_background_tasks(self):
        """Run eligible background tasks."""
        
        runnable = self.task_manager.get_runnable_tasks(
            self.state.is_idle,
            self.state.is_quiet
        )
        
        for task in runnable:
            if self.task_manager.run_task(task):
                self.state.background_tasks_running.append(task.name)
    
    # =========================================================================
    # PROOF LOG
    # =========================================================================
    
    def get_proof_log(self) -> str:
        """Generate proof log for idle state."""
        
        return (
            f"[IDLE] activity={self.state.activity_level.value} | "
            f"vol={self.state.current_volatility:.2f} | "
            f"vol_ratio={self.state.current_volume_ratio:.2f} | "
            f"spread={self.state.current_spread_bps:.1f}bps | "
            f"is_quiet={self.state.is_quiet} | "
            f"is_idle={self.state.is_idle} | "
            f"paused={len(self.state.paused_modules)} | "
            f"suspended={len(self.state.suspended_modules)} | "
            f"bg_tasks={len(self.state.background_tasks_running)}"
        )


# =============================================================================
# RE-EXPORT ORIGINAL FUNCTIONALITY
# =============================================================================

# Import original components for backward compatibility
try:
    from engine.idle_module_integrator import (
        SentimentState,
        SentimentIntegrator,
        MacroState,
        MacroIntegratorWrapper,
        GlobalContextAggregator,
        get_sentiment_integrator,
        get_macro_wrapper,
        get_context_aggregator,
    )
except ImportError:
    # Define stubs if original not available
    pass


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_idle_integrator: Optional[IdleModuleIntegrator] = None


def get_idle_module_integrator() -> IdleModuleIntegrator:
    """Get singleton idle module integrator."""
    global _idle_integrator
    if _idle_integrator is None:
        _idle_integrator = IdleModuleIntegrator()
    return _idle_integrator


def reset_idle_module_integrator():
    """Reset singleton (for testing)."""
    global _idle_integrator
    _idle_integrator = None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("HMATS v4.0-COMPLETE - IDLE MODULE INTEGRATOR TEST")
    print("=" * 70)
    
    integrator = get_idle_module_integrator()
    
    # Simulate market data
    print("\n--- Active Market ---")
    state = integrator.update({
        'realized_vol': 0.25,
        'volume_ratio': 1.2,
        'spread_bps': 2.0,
    })
    print(f"Activity: {state.activity_level.value}")
    print(f"Is Quiet: {state.is_quiet}")
    print(f"Is Idle: {state.is_idle}")
    
    print("\n--- Quiet Market ---")
    state = integrator.update({
        'realized_vol': 0.12,
        'volume_ratio': 0.4,
        'spread_bps': 4.0,
    })
    print(f"Activity: {state.activity_level.value}")
    
    print("\n--- Idle Market ---")
    state = integrator.update({
        'realized_vol': 0.06,
        'volume_ratio': 0.2,
        'spread_bps': 6.0,
    })
    print(f"Activity: {state.activity_level.value}")
    
    print(f"\nProof Log: {integrator.get_proof_log()}")

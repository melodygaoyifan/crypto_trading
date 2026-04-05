"""
================================================================================
HMATS v5.1.0 - Plugin Registry System
================================================================================
Purpose: Dynamic registration of strategy, risk, and execution modules

SOTA Requirement:
- Add a plugin registry: enable new strategy/risk/execution modules to be 
  registered dynamically via config.

Features:
1. Module discovery and registration
2. Config-driven plugin loading
3. Plugin lifecycle management
4. Dependency resolution
5. Hot-reload support (development)

Author: HMATS SOTA Upgrade
Version: 5.1.0-SOTA
================================================================================
"""

import logging
import importlib
import importlib.util
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Any, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
import inspect

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class PluginType(Enum):
    """Types of plugins supported."""
    STRATEGY = "strategy"
    RISK = "risk"
    EXECUTION = "execution"
    SIGNAL = "signal"
    DATA = "data"
    ANALYTICS = "analytics"


class PluginStatus(Enum):
    """Plugin lifecycle status."""
    REGISTERED = "registered"
    LOADED = "loaded"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


# =============================================================================
# PLUGIN INTERFACES
# =============================================================================

class PluginInterface(ABC):
    """Base interface for all plugins."""
    
    # Plugin metadata (override in subclasses)
    PLUGIN_NAME: str = "unnamed"
    PLUGIN_VERSION: str = "1.0.0"
    PLUGIN_TYPE: PluginType = PluginType.STRATEGY
    PLUGIN_DESCRIPTION: str = ""
    PLUGIN_AUTHOR: str = ""
    
    # Dependencies (list of plugin names this depends on)
    DEPENDENCIES: List[str] = []
    
    @abstractmethod
    def initialize(self, config: Dict) -> bool:
        """Initialize the plugin. Return True on success."""
        pass
    
    def shutdown(self):
        """Clean up resources."""
        pass
    
    def get_status(self) -> Dict:
        """Get plugin status."""
        return {"name": self.PLUGIN_NAME, "version": self.PLUGIN_VERSION}


class StrategyPlugin(PluginInterface):
    """Interface for strategy plugins."""
    PLUGIN_TYPE = PluginType.STRATEGY
    
    def initialize(self, config: Dict) -> bool:
        """Default initialization."""
        return True
    
    @abstractmethod
    def generate_signal(self, market_data: Dict) -> Dict:
        """Generate trading signal from market data."""
        pass


class RiskPlugin(PluginInterface):
    """Interface for risk management plugins."""
    PLUGIN_TYPE = PluginType.RISK
    
    def initialize(self, config: Dict) -> bool:
        """Default initialization."""
        return True
    
    @abstractmethod
    def evaluate_risk(self, position: Dict, market_data: Dict) -> Dict:
        """Evaluate risk for a position."""
        pass
    
    @abstractmethod
    def get_position_limit(self, asset: str, context: Dict) -> float:
        """Get position limit for an asset."""
        pass


class ExecutionPlugin(PluginInterface):
    """Interface for execution plugins."""
    PLUGIN_TYPE = PluginType.EXECUTION
    
    def initialize(self, config: Dict) -> bool:
        """Default initialization."""
        return True
    
    @abstractmethod
    def plan_execution(self, order: Dict, market_state: Dict) -> Dict:
        """Plan order execution."""
        pass


class SignalPlugin(PluginInterface):
    """Interface for signal generation plugins."""
    PLUGIN_TYPE = PluginType.SIGNAL
    
    def initialize(self, config: Dict) -> bool:
        """Default initialization."""
        return True
    
    @abstractmethod
    def process(self, data: Dict) -> Optional[Dict]:
        """Process data and optionally generate signal."""
        pass


# =============================================================================
# PLUGIN METADATA
# =============================================================================

@dataclass
class PluginMetadata:
    """Metadata about a registered plugin."""
    name: str
    version: str
    plugin_type: PluginType
    description: str
    author: str
    dependencies: List[str]
    
    # Registration info
    module_path: str
    class_name: str
    
    # Status
    status: PluginStatus = PluginStatus.REGISTERED
    error_message: str = ""
    
    # Runtime
    instance: Optional[PluginInterface] = None
    loaded_at: Optional[datetime] = None
    config: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "version": self.version,
            "type": self.plugin_type.value,
            "description": self.description,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "error": self.error_message or None
        }


# =============================================================================
# PLUGIN REGISTRY
# =============================================================================

class PluginRegistry:
    """
    Central registry for all HMATS plugins.
    
    Responsibilities:
    1. Discover plugins from configured paths
    2. Manage plugin lifecycle (load, initialize, shutdown)
    3. Resolve dependencies
    4. Provide plugin instances to the system
    """
    
    def __init__(self, config_path: str = None):
        self.plugins: Dict[str, PluginMetadata] = {}
        self.plugin_paths: List[Path] = []
        self.config_path = config_path
        
        # Plugin instances by type
        self.by_type: Dict[PluginType, List[str]] = {t: [] for t in PluginType}
        
        # Load config if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        logger.info("[PluginRegistry] Initialized")
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    def _load_config(self, config_path: str):
        """Load plugin configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Add plugin paths
            for path in config.get("plugin_paths", []):
                self.add_plugin_path(path)
            
            # Auto-load configured plugins
            for plugin_config in config.get("plugins", []):
                if plugin_config.get("enabled", True):
                    self.register_from_config(plugin_config)
                    
        except Exception as e:
            logger.error(f"[PluginRegistry] Failed to load config: {e}")
    
    def add_plugin_path(self, path: str):
        """Add a directory to search for plugins."""
        p = Path(path)
        if p.exists() and p.is_dir():
            self.plugin_paths.append(p)
            logger.info(f"[PluginRegistry] Added plugin path: {path}")
    
    # =========================================================================
    # REGISTRATION
    # =========================================================================
    
    def register(
        self,
        plugin_class: Type[PluginInterface],
        config: Dict = None
    ) -> bool:
        """
        Register a plugin class.
        
        Args:
            plugin_class: The plugin class to register
            config: Plugin configuration
            
        Returns:
            True if registered successfully
        """
        try:
            # Extract metadata from class
            name = getattr(plugin_class, 'PLUGIN_NAME', plugin_class.__name__)
            version = getattr(plugin_class, 'PLUGIN_VERSION', '1.0.0')
            plugin_type = getattr(plugin_class, 'PLUGIN_TYPE', PluginType.STRATEGY)
            description = getattr(plugin_class, 'PLUGIN_DESCRIPTION', '')
            author = getattr(plugin_class, 'PLUGIN_AUTHOR', '')
            dependencies = getattr(plugin_class, 'DEPENDENCIES', [])
            
            # Check if already registered
            if name in self.plugins:
                logger.warning(f"[PluginRegistry] Plugin already registered: {name}")
                return False
            
            # Create metadata
            metadata = PluginMetadata(
                name=name,
                version=version,
                plugin_type=plugin_type,
                description=description,
                author=author,
                dependencies=dependencies,
                module_path=plugin_class.__module__,
                class_name=plugin_class.__name__,
                config=config or {}
            )
            
            # Store reference to class for instantiation
            metadata._class = plugin_class
            
            self.plugins[name] = metadata
            self.by_type[plugin_type].append(name)
            
            logger.info(f"[PluginRegistry] Registered: {name} v{version} ({plugin_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"[PluginRegistry] Registration failed: {e}")
            return False
    
    def register_from_config(self, config: Dict) -> bool:
        """
        Register a plugin from configuration.
        
        Config format:
        {
            "module": "path.to.module",
            "class": "PluginClassName",
            "config": {...}
        }
        """
        try:
            module_path = config["module"]
            class_name = config["class"]
            plugin_config = config.get("config", {})
            
            # Import module
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            
            return self.register(plugin_class, plugin_config)
            
        except Exception as e:
            logger.error(f"[PluginRegistry] Failed to register from config: {e}")
            return False
    
    def discover_plugins(self):
        """Discover and register plugins from plugin paths."""
        for path in self.plugin_paths:
            for py_file in path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                
                try:
                    # Load module
                    spec = importlib.util.spec_from_file_location(
                        py_file.stem, py_file
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find plugin classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, PluginInterface) and 
                            obj is not PluginInterface and
                            not name.endswith('Plugin')):  # Skip base classes
                            self.register(obj)
                            
                except Exception as e:
                    logger.debug(f"[PluginRegistry] Could not load {py_file}: {e}")
    
    # =========================================================================
    # LIFECYCLE
    # =========================================================================
    
    def load(self, name: str) -> bool:
        """
        Load a plugin (instantiate but don't initialize).
        
        Args:
            name: Plugin name
            
        Returns:
            True if loaded successfully
        """
        if name not in self.plugins:
            logger.error(f"[PluginRegistry] Unknown plugin: {name}")
            return False
        
        metadata = self.plugins[name]
        
        # Check dependencies
        for dep in metadata.dependencies:
            if dep not in self.plugins:
                metadata.status = PluginStatus.ERROR
                metadata.error_message = f"Missing dependency: {dep}"
                return False
            if self.plugins[dep].status not in [PluginStatus.LOADED, PluginStatus.ACTIVE]:
                # Try to load dependency first
                if not self.load(dep):
                    metadata.status = PluginStatus.ERROR
                    metadata.error_message = f"Failed to load dependency: {dep}"
                    return False
        
        try:
            # Instantiate
            plugin_class = metadata._class
            metadata.instance = plugin_class()
            metadata.status = PluginStatus.LOADED
            metadata.loaded_at = datetime.utcnow()
            
            logger.info(f"[PluginRegistry] Loaded: {name}")
            return True
            
        except Exception as e:
            metadata.status = PluginStatus.ERROR
            metadata.error_message = str(e)
            logger.error(f"[PluginRegistry] Failed to load {name}: {e}")
            return False
    
    def initialize(self, name: str, config: Dict = None) -> bool:
        """
        Initialize a loaded plugin.
        
        Args:
            name: Plugin name
            config: Configuration to pass to plugin
            
        Returns:
            True if initialized successfully
        """
        if name not in self.plugins:
            return False
        
        metadata = self.plugins[name]
        
        if metadata.status != PluginStatus.LOADED:
            if not self.load(name):
                return False
        
        try:
            # Merge configs
            full_config = {**metadata.config, **(config or {})}
            
            # Initialize
            success = metadata.instance.initialize(full_config)
            
            if success:
                metadata.status = PluginStatus.ACTIVE
                logger.info(f"[PluginRegistry] Initialized: {name}")
            else:
                metadata.status = PluginStatus.ERROR
                metadata.error_message = "Initialization returned False"
            
            return success
            
        except Exception as e:
            metadata.status = PluginStatus.ERROR
            metadata.error_message = str(e)
            logger.error(f"[PluginRegistry] Failed to initialize {name}: {e}")
            return False
    
    def shutdown(self, name: str):
        """Shutdown a plugin."""
        if name not in self.plugins:
            return
        
        metadata = self.plugins[name]
        
        if metadata.instance:
            try:
                metadata.instance.shutdown()
            except Exception as e:
                logger.error(f"[PluginRegistry] Shutdown error for {name}: {e}")
        
        metadata.status = PluginStatus.DISABLED
        logger.info(f"[PluginRegistry] Shutdown: {name}")
    
    def shutdown_all(self):
        """Shutdown all active plugins."""
        for name in list(self.plugins.keys()):
            if self.plugins[name].status == PluginStatus.ACTIVE:
                self.shutdown(name)
    
    # =========================================================================
    # ACCESS
    # =========================================================================
    
    def get(self, name: str) -> Optional[PluginInterface]:
        """Get a plugin instance."""
        metadata = self.plugins.get(name)
        if metadata and metadata.status == PluginStatus.ACTIVE:
            return metadata.instance
        return None
    
    def get_by_type(self, plugin_type: PluginType) -> List[PluginInterface]:
        """Get all active plugins of a type."""
        instances = []
        for name in self.by_type.get(plugin_type, []):
            instance = self.get(name)
            if instance:
                instances.append(instance)
        return instances
    
    def list_plugins(self, plugin_type: PluginType = None) -> List[Dict]:
        """List all registered plugins."""
        plugins = []
        for name, metadata in self.plugins.items():
            if plugin_type is None or metadata.plugin_type == plugin_type:
                plugins.append(metadata.to_dict())
        return plugins
    
    def get_status(self) -> Dict:
        """Get registry status."""
        by_status = {}
        for metadata in self.plugins.values():
            status = metadata.status.value
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            "total_plugins": len(self.plugins),
            "by_status": by_status,
            "by_type": {t.value: len(names) for t, names in self.by_type.items()},
            "plugin_paths": [str(p) for p in self.plugin_paths]
        }


# =============================================================================
# GLOBAL REGISTRY
# =============================================================================

_registry: Optional[PluginRegistry] = None


def get_plugin_registry(config_path: str = None) -> PluginRegistry:
    """Get or create the global plugin registry."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry(config_path)
    return _registry


def reset_plugin_registry():
    """Reset the global registry."""
    global _registry
    if _registry:
        _registry.shutdown_all()
    _registry = None


# =============================================================================
# DECORATORS FOR PLUGIN REGISTRATION
# =============================================================================

def register_plugin(plugin_type: PluginType = PluginType.STRATEGY):
    """Decorator to auto-register a plugin class."""
    def decorator(cls):
        cls.PLUGIN_TYPE = plugin_type
        # Will be registered when registry is created and discovers plugins
        return cls
    return decorator

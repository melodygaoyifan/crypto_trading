"""
================================================================================
HMATS Cloud Configuration Loader
================================================================================
Version: 5.1.0-CLOUD
Purpose: Unified configuration loading with environment variable overrides

DESIGN PRINCIPLES:
    - Environment variables ALWAYS take precedence over config files
    - Fail-closed on missing required configuration
    - Support both local development and cloud deployment
    - Single-exchange mode enforcement

USAGE:
    from core.cloud_config import CloudConfig, get_cloud_config
    
    config = get_cloud_config()
    model_dir = config.get_model_dir()
    log_dir = config.get_log_dir()

================================================================================
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

class DeployMode(Enum):
    """Deployment mode."""
    LOCAL = "local"
    CLOUD = "cloud"
    DOCKER = "docker"


# Environment variable prefix
ENV_PREFIX = "HMATS_"

# Required environment variables for live trading
REQUIRED_ENV_LIVE = [
    "KRAKEN_API_KEY",
    "KRAKEN_API_SECRET",
]


# =============================================================================
# CLOUD CONFIG
# =============================================================================

@dataclass
class CloudConfig:
    """
    Cloud-optimized configuration with environment variable support.
    
    Priority order:
        1. Environment variables (highest)
        2. Config file values
        3. Default values (lowest)
    """
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path("/opt/hmats"))
    model_dir: Path = field(default_factory=lambda: Path("/opt/hmats/models/current"))
    log_dir: Path = field(default_factory=lambda: Path("/var/log/hmats"))
    state_dir: Path = field(default_factory=lambda: Path("/var/lib/hmats"))
    config_dir: Path = field(default_factory=lambda: Path("/opt/hmats/configs"))
    
    # Exchange (SINGLE EXCHANGE ONLY)
    exchange: str = "kraken"
    
    # Runtime
    mode: str = "paper"
    log_level: str = "INFO"
    initial_capital: float = 10000.0  # [CFG-4] was 100K, actual $10K
    
    # Flags
    stdout_logging: bool = True
    file_logging: bool = True
    single_exchange_mode: bool = True
    
    # Raw config data
    _raw_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'CloudConfig':
        """Load config from file with environment overrides."""
        config = cls()
        
        # Load base config file
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config._raw_config = json.load(f)
            logger.info(f"Loaded config from {config_path}")
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
        
        # Apply environment overrides
        config._apply_env_overrides()
        
        # Validate
        config._validate()
        
        return config
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        runtime = self._raw_config.get("runtime", {})
        
        # Path overrides
        self.base_dir = Path(os.getenv(
            runtime.get("base_dir_env", "HMATS_BASE_DIR"),
            runtime.get("base_dir_default", "/opt/hmats")
        ))
        
        self.model_dir = Path(os.getenv(
            runtime.get("model_dir_env", "HMATS_MODEL_DIR"),
            runtime.get("model_dir_default", str(self.base_dir / "models/current"))
        ))
        
        self.log_dir = Path(os.getenv(
            runtime.get("log_dir_env", "HMATS_LOG_DIR"),
            runtime.get("log_dir_default", "/var/log/hmats")
        ))
        
        self.state_dir = Path(os.getenv(
            runtime.get("state_dir_env", "HMATS_STATE_DIR"),
            runtime.get("state_dir_default", "/var/lib/hmats")
        ))
        
        self.config_dir = Path(os.getenv(
            runtime.get("config_dir_env", "HMATS_CONFIG_DIR"),
            runtime.get("config_dir_default", str(self.base_dir / "configs"))
        ))
        
        # Logging overrides
        logging_cfg = self._raw_config.get("logging", {})
        self.log_level = os.getenv(
            logging_cfg.get("level_env", "HMATS_LOG_LEVEL"),
            logging_cfg.get("level_default", "INFO")
        )
        self.stdout_logging = logging_cfg.get("stdout_enabled", True)
        self.file_logging = logging_cfg.get("file_enabled", True)
        
        # Capital override
        risk_cfg = self._raw_config.get("risk", {})
        capital_str = os.getenv(
            risk_cfg.get("initial_capital_env", "HMATS_INITIAL_CAPITAL"),
            str(risk_cfg.get("initial_capital_default", 10000.0))
        )
        self.initial_capital = float(capital_str)
        
        # Exchange (SINGLE EXCHANGE ENFORCED)
        exchange_cfg = self._raw_config.get("exchange", {})
        self.exchange = exchange_cfg.get("_active", "kraken")
        self.single_exchange_mode = exchange_cfg.get("_mode", "SINGLE_EXCHANGE") == "SINGLE_EXCHANGE"
    
    def _validate(self):
        """Validate configuration."""
        errors = []
        
        # Validate single exchange mode
        if not self.single_exchange_mode:
            logger.warning("[WARN]️  Multi-exchange mode detected but not supported!")
            self.single_exchange_mode = True
        
        # Log single exchange mode
        logger.info(f"🔒 SINGLE_EXCHANGE_MODE=ON (active: {self.exchange})")
        
        if errors:
            raise ValueError(f"Configuration errors: {errors}")
    
    def get_model_dir(self) -> Path:
        """Get model directory."""
        return self.model_dir
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path for a specific model."""
        return self.model_dir / model_name
    
    def get_log_dir(self) -> Path:
        """Get log directory."""
        return self.log_dir
    
    def get_state_dir(self) -> Path:
        """Get state directory."""
        return self.state_dir
    
    def get_log_file(self, name: str = "hmats.log") -> Path:
        """Get log file path."""
        return self.log_dir / name
    
    def get_state_file(self, name: str = "hmats_state.json") -> Path:
        """Get state file path."""
        return self.state_dir / name
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        dirs = [
            self.log_dir,
            self.state_dir,
            self.log_dir / "events",
            self.log_dir / "proof",
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory: {d}")
    
    def check_live_requirements(self) -> bool:
        """Check if all requirements for live trading are met."""
        missing = []
        
        for env_var in REQUIRED_ENV_LIVE:
            if not os.getenv(env_var):
                missing.append(env_var)
        
        if missing:
            logger.error(f"Missing required environment variables for live: {missing}")
            return False
        
        return True
    
    def get_kraken_credentials(self) -> tuple:
        """Get Kraken API credentials from environment."""
        api_key = os.getenv("KRAKEN_API_KEY", "")
        api_secret = os.getenv("KRAKEN_API_SECRET", "")
        return api_key, api_secret
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary."""
        return {
            "base_dir": str(self.base_dir),
            "model_dir": str(self.model_dir),
            "log_dir": str(self.log_dir),
            "state_dir": str(self.state_dir),
            "config_dir": str(self.config_dir),
            "exchange": self.exchange,
            "mode": self.mode,
            "log_level": self.log_level,
            "initial_capital": self.initial_capital,
            "single_exchange_mode": self.single_exchange_mode,
        }


# =============================================================================
# MODEL LOADER WITH VALIDATION
# =============================================================================

@dataclass
class ModelValidationResult:
    """Result of model validation."""
    valid: bool
    model_path: Path
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ModelLoader:
    """
    Model loader with validation and fail-closed behavior.
    
    Ensures models exist and are valid before use.
    If validation fails, system enters NO_TRADE mode.
    """
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.model_dir = config.get_model_dir()
        self._loaded_models: Dict[str, Any] = {}
        self._validation_results: Dict[str, ModelValidationResult] = {}
    
    def validate_model(
        self,
        model_name: str,
        required_files: List[str],
        min_size_bytes: int = 1000,
    ) -> ModelValidationResult:
        """
        Validate a model directory.
        
        Args:
            model_name: Name of model (subdirectory)
            required_files: List of required files
            min_size_bytes: Minimum file size for validation
            
        Returns:
            ModelValidationResult
        """
        model_path = self.model_dir / model_name
        result = ModelValidationResult(valid=True, model_path=model_path)
        
        # Check directory exists
        if not model_path.exists():
            result.valid = False
            result.errors.append(f"Model directory not found: {model_path}")
            return result
        
        # Check required files
        for filename in required_files:
            file_path = model_path / filename
            
            if not file_path.exists():
                result.valid = False
                result.errors.append(f"Required file missing: {file_path}")
                continue
            
            # Check file size
            size = file_path.stat().st_size
            if size < min_size_bytes:
                result.valid = False
                result.errors.append(
                    f"File too small: {file_path} ({size} bytes < {min_size_bytes})"
                )
        
        self._validation_results[model_name] = result
        
        if result.valid:
            logger.info(f"✓ Model validated: {model_name}")
        else:
            logger.error(f"✗ Model validation failed: {model_name}")
            for err in result.errors:
                logger.error(f"  - {err}")
        
        return result
    
    def validate_all_models(self) -> bool:
        """
        Validate all configured models.
        
        Returns:
            True if all required models are valid
        """
        models_cfg = self.config._raw_config.get("models", {})
        all_valid = True
        
        for model_name, model_cfg in models_cfg.items():
            if model_name.startswith("_"):
                continue
            
            if not model_cfg.get("enabled", True):
                continue
            
            required_files = []
            if "config_file" in model_cfg:
                required_files.append(model_cfg["config_file"])
            if "weights_file" in model_cfg:
                required_files.append(model_cfg["weights_file"])
            
            if not required_files:
                continue
            
            result = self.validate_model(
                model_cfg.get("subdir", model_name),
                required_files,
            )
            
            if not result.valid and model_cfg.get("required_for_live", False):
                all_valid = False
        
        return all_valid
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get validated model path."""
        result = self._validation_results.get(model_name)
        if result and result.valid:
            return result.model_path
        return None


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_cloud_logging(config: CloudConfig):
    """
    Setup logging for cloud deployment.
    
    Supports both stdout (for container logs) and file output.
    """
    # Ensure log directory exists
    config.ensure_directories()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s │ %(name)-25s │ %(levelname)-8s │ %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Stdout handler (for container logs)
    if config.stdout_logging:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(stdout_handler)
    
    # File handler
    if config.file_logging:
        log_file = config.get_log_file()
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    logger.info(f"Logging configured: level={config.log_level}, stdout={config.stdout_logging}, file={config.file_logging}")


# =============================================================================
# SINGLETON
# =============================================================================

_cloud_config: Optional[CloudConfig] = None
_cloud_config_path: Optional[Path] = None


def _resolve_cloud_config_path(config_path: Optional[Path] = None) -> Path:
    """Resolve the effective cloud config path for the current process."""
    if config_path is not None:
        return Path(config_path)

    env_config = os.getenv("HMATS_CONFIG_FILE")
    if env_config:
        return Path(env_config)

    candidates = [
        Path("configs/cloud_production.json"),
        Path("/opt/hmats/configs/cloud_production.json"),
        Path("configs/production.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_cloud_config(config_path: Optional[Path] = None) -> CloudConfig:
    """Get cloud configuration singleton."""
    global _cloud_config, _cloud_config_path

    resolved_path = _resolve_cloud_config_path(config_path)

    if _cloud_config is None:
        _cloud_config = CloudConfig.from_file(resolved_path)
        _cloud_config_path = resolved_path
    elif _cloud_config_path != resolved_path:
        logger.info(f"Cloud config path changed ({_cloud_config_path} -> {resolved_path}); refreshing singleton")
        _cloud_config = CloudConfig.from_file(resolved_path)
        _cloud_config_path = resolved_path
    return _cloud_config


def reset_cloud_config():
    """Reset cloud configuration singleton."""
    global _cloud_config, _cloud_config_path
    _cloud_config = None
    _cloud_config_path = None


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cloud configuration utility")
    parser.add_argument("--config", type=Path, help="Config file path")
    parser.add_argument("--show", action="store_true", help="Show current config")
    parser.add_argument("--validate-models", action="store_true", help="Validate models")
    parser.add_argument("--ensure-dirs", action="store_true", help="Ensure directories exist")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    config = get_cloud_config(args.config)
    
    if args.show:
        print(json.dumps(config.to_dict(), indent=2))
    
    if args.validate_models:
        loader = ModelLoader(config)
        valid = loader.validate_all_models()
        sys.exit(0 if valid else 1)
    
    if args.ensure_dirs:
        config.ensure_directories()
        print("Directories ensured")

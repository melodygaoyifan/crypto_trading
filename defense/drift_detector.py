"""
================================================================================
DATA DRIFT DETECTOR & HOT RESTART MANAGER
================================================================================
Version: 3.2.0
Purpose: Detect data distribution drift and enable stateful restarts

Features:
- DRL latent space drift detection (p-value monitoring)
- Feature distribution monitoring
- Auto-downweight DRL on drift detection
- Hot restart with state preservation
- Checkpoint management
================================================================================
"""

import time
import json
import logging
import pickle
import hashlib
import threading
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from collections import deque
import numpy as np

logger = logging.getLogger("HMATS.DriftDetector")


class DriftSeverity(Enum):
    """Drift severity levels."""
    NONE = auto()
    MINOR = auto()      # Slight shift, monitor
    MODERATE = auto()   # Noticeable shift, reduce DRL weight
    SEVERE = auto()     # Major shift, disable DRL
    CRITICAL = auto()   # System-wide drift, halt trading


class DriftType(Enum):
    """Types of drift detected."""
    FEATURE_DISTRIBUTION = auto()
    LATENT_SPACE = auto()
    PREDICTION_ACCURACY = auto()
    REGIME_MISMATCH = auto()
    TEMPORAL = auto()


@dataclass
class DriftEvent:
    """Single drift detection event."""
    timestamp: float
    drift_type: DriftType
    severity: DriftSeverity
    feature_name: str = ""
    p_value: float = 1.0
    kl_divergence: float = 0.0
    details: Dict = field(default_factory=dict)


@dataclass
class DriftDetectorConfig:
    """Drift detector configuration."""
    # P-value thresholds
    p_value_warning: float = 0.05
    p_value_moderate: float = 0.01
    p_value_severe: float = 0.001
    
    # Window sizes
    reference_window: int = 1000      # Samples for reference distribution
    detection_window: int = 100       # Samples for current distribution
    
    # Update frequency
    check_interval_seconds: float = 60.0
    
    # DRL weight adjustments
    drl_weight_on_minor: float = 0.9
    drl_weight_on_moderate: float = 0.5
    drl_weight_on_severe: float = 0.1
    drl_weight_on_critical: float = 0.0


class FeatureDistributionTracker:
    """Track feature distribution over time."""
    
    def __init__(self, feature_name: str, reference_window: int = 1000):
        self.feature_name = feature_name
        self.reference_window = reference_window
        
        self.reference_data: deque = deque(maxlen=reference_window)
        self.current_data: deque = deque(maxlen=100)
        
        self.reference_stats: Optional[Dict] = None
        self._lock = threading.Lock()
    
    def add_sample(self, value: float):
        """Add sample to current window."""
        with self._lock:
            self.current_data.append(value)
    
    def update_reference(self):
        """Update reference distribution from current data."""
        with self._lock:
            self.reference_data.extend(self.current_data)
            if len(self.reference_data) >= 100:
                data = np.array(list(self.reference_data))
                self.reference_stats = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'percentiles': np.percentile(data, [25, 50, 75]).tolist()
                }
    
    def check_drift(self) -> Tuple[bool, float, DriftSeverity]:
        """
        Check for distribution drift using KS test approximation.
        
        Returns:
            (drift_detected, p_value, severity)
        """
        with self._lock:
            if len(self.current_data) < 30 or self.reference_stats is None:
                return False, 1.0, DriftSeverity.NONE
            
            current = np.array(list(self.current_data))
            ref_mean = self.reference_stats['mean']
            ref_std = max(self.reference_stats['std'], 1e-10)
            
            # Z-score of current mean
            current_mean = np.mean(current)
            z_score = abs(current_mean - ref_mean) / (ref_std / np.sqrt(len(current)))
            
            # Approximate p-value (two-tailed)
            p_value = 2 * (1 - self._normal_cdf(z_score))
            
            # Determine severity
            if p_value < 0.001:
                severity = DriftSeverity.SEVERE
            elif p_value < 0.01:
                severity = DriftSeverity.MODERATE
            elif p_value < 0.05:
                severity = DriftSeverity.MINOR
            else:
                severity = DriftSeverity.NONE
            
            drift_detected = severity != DriftSeverity.NONE
            
            return drift_detected, p_value, severity
    
    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximate normal CDF."""
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


class LatentSpaceDriftDetector:
    """
    Detect drift in DRL model's latent space.
    
    Monitors the distribution of latent representations
    to detect when input data has shifted from training distribution.
    """
    
    def __init__(self, latent_dim: int = 64, reference_window: int = 5000):
        self.latent_dim = latent_dim
        self.reference_window = reference_window
        
        # Reference latent vectors (from training or initial period)
        self.reference_latents: deque = deque(maxlen=reference_window)
        self.current_latents: deque = deque(maxlen=500)
        
        # Reference statistics per dimension
        self.reference_means: Optional[np.ndarray] = None
        self.reference_stds: Optional[np.ndarray] = None
        self.reference_covariance: Optional[np.ndarray] = None
        
        self._lock = threading.Lock()
    
    def add_latent(self, latent_vector: np.ndarray):
        """Add latent vector observation."""
        with self._lock:
            if latent_vector.shape[0] != self.latent_dim:
                logger.warning(f"Latent dim mismatch: {latent_vector.shape[0]} vs {self.latent_dim}")
                return
            
            self.current_latents.append(latent_vector)
    
    def update_reference(self):
        """Update reference distribution."""
        with self._lock:
            self.reference_latents.extend(self.current_latents)
            
            if len(self.reference_latents) >= 100:
                data = np.array(list(self.reference_latents))
                self.reference_means = np.mean(data, axis=0)
                self.reference_stds = np.std(data, axis=0)
                
                # Compute covariance (for Mahalanobis distance)
                try:
                    self.reference_covariance = np.cov(data.T)
                except Exception:
                    self.reference_covariance = np.eye(self.latent_dim)
    
    def check_drift(self) -> Tuple[bool, float, DriftSeverity]:
        """
        Check for latent space drift.
        
        Returns:
            (drift_detected, p_value_approx, severity)
        """
        with self._lock:
            if len(self.current_latents) < 50 or self.reference_means is None:
                return False, 1.0, DriftSeverity.NONE
            
            current = np.array(list(self.current_latents))
            current_means = np.mean(current, axis=0)
            
            # Per-dimension z-scores
            safe_stds = np.where(self.reference_stds > 1e-10, self.reference_stds, 1e-10)
            z_scores = np.abs(current_means - self.reference_means) / safe_stds
            
            # Max z-score across dimensions
            max_z = np.max(z_scores)
            avg_z = np.mean(z_scores)
            
            # Approximate p-value from max z
            p_value = 2 * (1 - FeatureDistributionTracker._normal_cdf(max_z))
            
            # Determine severity
            if max_z > 4.0 or avg_z > 2.5:
                severity = DriftSeverity.SEVERE
            elif max_z > 3.0 or avg_z > 2.0:
                severity = DriftSeverity.MODERATE
            elif max_z > 2.0 or avg_z > 1.5:
                severity = DriftSeverity.MINOR
            else:
                severity = DriftSeverity.NONE
            
            return severity != DriftSeverity.NONE, p_value, severity


class DataDriftDetector:
    """
    Main drift detector coordinating all drift detection.
    
    Monitors feature distributions and DRL latent space,
    automatically adjusts DRL weight on drift detection.
    """
    
    def __init__(self, config: DriftDetectorConfig = None):
        self.config = config or DriftDetectorConfig()
        
        # Feature trackers
        self.feature_trackers: Dict[str, FeatureDistributionTracker] = {}
        
        # Latent space tracker
        self.latent_detector = LatentSpaceDriftDetector()
        
        # Drift events history
        self.drift_events: List[DriftEvent] = []
        
        # Current drift state
        self.current_severity = DriftSeverity.NONE
        self.current_drl_weight_multiplier = 1.0
        
        # Last check time
        self.last_check_time = 0.0
        
        self._lock = threading.Lock()
        
        logger.info("DataDriftDetector initialized")
    
    def add_feature(self, name: str, value: float):
        """Add feature observation."""
        with self._lock:
            if name not in self.feature_trackers:
                self.feature_trackers[name] = FeatureDistributionTracker(name)
            
            self.feature_trackers[name].add_sample(value)
    
    def add_features(self, features: Dict[str, float]):
        """Add multiple feature observations."""
        for name, value in features.items():
            self.add_feature(name, value)
    
    def add_latent(self, latent_vector: np.ndarray):
        """Add DRL latent vector observation."""
        self.latent_detector.add_latent(latent_vector)
    
    def check_all_drift(self) -> Tuple[DriftSeverity, Dict]:
        """
        Check all drift sources.
        
        Returns:
            (max_severity, details)
        """
        with self._lock:
            now = time.time()
            
            # Rate limit checks
            if now - self.last_check_time < self.config.check_interval_seconds:
                return self.current_severity, {}
            
            self.last_check_time = now
            
            max_severity = DriftSeverity.NONE
            details = {'features': {}, 'latent': {}}
            
            # Check feature distributions
            for name, tracker in self.feature_trackers.items():
                detected, p_value, severity = tracker.check_drift()
                
                details['features'][name] = {
                    'drift': detected,
                    'p_value': p_value,
                    'severity': severity.name
                }
                
                if detected and severity.value > max_severity.value:
                    max_severity = severity
                    
                    self.drift_events.append(DriftEvent(
                        timestamp=now,
                        drift_type=DriftType.FEATURE_DISTRIBUTION,
                        severity=severity,
                        feature_name=name,
                        p_value=p_value
                    ))
            
            # Check latent space
            detected, p_value, severity = self.latent_detector.check_drift()
            details['latent'] = {
                'drift': detected,
                'p_value': p_value,
                'severity': severity.name
            }
            
            if detected and severity.value > max_severity.value:
                max_severity = severity
                
                self.drift_events.append(DriftEvent(
                    timestamp=now,
                    drift_type=DriftType.LATENT_SPACE,
                    severity=severity,
                    p_value=p_value
                ))
            
            # Update DRL weight multiplier
            self._update_drl_weight(max_severity)
            
            self.current_severity = max_severity
            
            # Trim events
            if len(self.drift_events) > 1000:
                self.drift_events = self.drift_events[-1000:]
            
            return max_severity, details
    
    def _update_drl_weight(self, severity: DriftSeverity):
        """Update DRL weight multiplier based on drift severity."""
        if severity == DriftSeverity.NONE:
            # Gradually recover
            self.current_drl_weight_multiplier = min(
                1.0,
                self.current_drl_weight_multiplier + 0.01
            )
        elif severity == DriftSeverity.MINOR:
            self.current_drl_weight_multiplier = self.config.drl_weight_on_minor
        elif severity == DriftSeverity.MODERATE:
            self.current_drl_weight_multiplier = self.config.drl_weight_on_moderate
        elif severity == DriftSeverity.SEVERE:
            self.current_drl_weight_multiplier = self.config.drl_weight_on_severe
        else:  # CRITICAL
            self.current_drl_weight_multiplier = self.config.drl_weight_on_critical
    
    def get_drl_weight_multiplier(self) -> float:
        """Get current DRL weight multiplier."""
        return self.current_drl_weight_multiplier
    
    def update_references(self):
        """Update reference distributions from current data."""
        with self._lock:
            for tracker in self.feature_trackers.values():
                tracker.update_reference()
            self.latent_detector.update_reference()
    
    def get_status(self) -> Dict:
        """Get drift detector status."""
        with self._lock:
            return {
                'current_severity': self.current_severity.name,
                'drl_weight_multiplier': self.current_drl_weight_multiplier,
                'tracked_features': len(self.feature_trackers),
                'recent_drift_events': len([
                    e for e in self.drift_events
                    if time.time() - e.timestamp < 3600
                ]),
                'total_drift_events': len(self.drift_events)
            }


# ==============================================================================
# HOT RESTART MANAGER
# ==============================================================================

@dataclass
class SystemCheckpoint:
    """System checkpoint for hot restart."""
    checkpoint_id: str
    timestamp: float
    version: str
    
    # State data
    shadow_ledger_state: Dict = field(default_factory=dict)
    pending_orders: List[Dict] = field(default_factory=list)
    positions: Dict = field(default_factory=dict)
    regime_state: Dict = field(default_factory=dict)
    signal_state: Dict = field(default_factory=dict)
    
    # Metrics
    equity: float = 0.0
    total_trades: int = 0
    
    # Checksums
    checksum: str = ""


class HotRestartManager:
    """
    Manage hot restarts with state preservation.
    
    Features:
    - Periodic checkpointing
    - State recovery on restart
    - Integrity verification
    - Graceful degradation if state invalid
    """
    
    CHECKPOINT_VERSION = "3.2.0"
    
    def __init__(
        self,
        checkpoint_dir: Path = None,
        checkpoint_interval_seconds: float = 60.0,
        max_checkpoints: int = 10
    ):
        self.checkpoint_dir = checkpoint_dir or Path("/tmp/hmats_checkpoints")
        self.checkpoint_interval = checkpoint_interval_seconds
        self.max_checkpoints = max_checkpoints
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.last_checkpoint_time = 0.0
        self.checkpoint_count = 0
        
        self._lock = threading.Lock()
        
        logger.info(f"HotRestartManager initialized: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        shadow_ledger_state: Dict = None,
        pending_orders: List[Dict] = None,
        positions: Dict = None,
        regime_state: Dict = None,
        signal_state: Dict = None,
        equity: float = 0.0,
        total_trades: int = 0
    ) -> Optional[str]:
        """Save system checkpoint."""
        with self._lock:
            now = time.time()
            
            # Rate limit
            if now - self.last_checkpoint_time < self.checkpoint_interval:
                return None
            
            self.checkpoint_count += 1
            checkpoint_id = f"cp_{int(now)}_{self.checkpoint_count:06d}"
            
            checkpoint = SystemCheckpoint(
                checkpoint_id=checkpoint_id,
                timestamp=now,
                version=self.CHECKPOINT_VERSION,
                shadow_ledger_state=shadow_ledger_state or {},
                pending_orders=pending_orders or [],
                positions=positions or {},
                regime_state=regime_state or {},
                signal_state=signal_state or {},
                equity=equity,
                total_trades=total_trades
            )
            
            # Calculate checksum
            checkpoint.checksum = self._calculate_checksum(checkpoint)
            
            # Save to file
            filepath = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(checkpoint, f)
                
                self.last_checkpoint_time = now
                
                # Cleanup old checkpoints
                self._cleanup_old_checkpoints()
                
                logger.debug(f"Saved checkpoint: {checkpoint_id}")
                return checkpoint_id
                
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                return None
    
    def load_latest_checkpoint(self) -> Optional[SystemCheckpoint]:
        """Load most recent valid checkpoint."""
        with self._lock:
            checkpoints = sorted(
                self.checkpoint_dir.glob("cp_*.pkl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            for filepath in checkpoints:
                try:
                    with open(filepath, 'rb') as f:
                        checkpoint = pickle.load(f)
                    
                    # Verify checksum
                    if self._verify_checksum(checkpoint):
                        logger.info(f"Loaded checkpoint: {checkpoint.checkpoint_id}")
                        return checkpoint
                    else:
                        logger.warning(f"Checksum mismatch: {filepath}")
                        
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
            
            logger.warning("No valid checkpoint found")
            return None
    
    def _calculate_checksum(self, checkpoint: SystemCheckpoint) -> str:
        """Calculate checkpoint checksum."""
        data = json.dumps({
            'id': checkpoint.checkpoint_id,
            'timestamp': checkpoint.timestamp,
            'version': checkpoint.version,
            'equity': checkpoint.equity,
            'trades': checkpoint.total_trades
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _verify_checksum(self, checkpoint: SystemCheckpoint) -> bool:
        """Verify checkpoint integrity."""
        expected = self._calculate_checksum(checkpoint)
        return checkpoint.checksum == expected
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max limit."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("cp_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for filepath in checkpoints[self.max_checkpoints:]:
            try:
                filepath.unlink()
                logger.debug(f"Removed old checkpoint: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to remove {filepath}: {e}")
    
    def get_status(self) -> Dict:
        """Get restart manager status."""
        with self._lock:
            checkpoints = list(self.checkpoint_dir.glob("cp_*.pkl"))
            return {
                'checkpoint_dir': str(self.checkpoint_dir),
                'checkpoint_count': len(checkpoints),
                'last_checkpoint_time': self.last_checkpoint_time,
                'checkpoint_interval': self.checkpoint_interval,
                'version': self.CHECKPOINT_VERSION
            }

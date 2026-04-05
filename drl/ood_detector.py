"""
Stage 9.3: Mahalanobis-distance OOD detector for DRL authority degradation.

Principle:
  - Fit mean + covariance on TRAINING SET features only (Rule #31)
  - At runtime, observations beyond the threshold -> soft-degrade DRL weight
  - Only hard-switch to EXIT_ONLY after consecutive OOD >= N steps (Rule #32)

Why Mahalanobis:
  - Accounts for feature correlations (better than per-dim z-score)
  - Cheap to evaluate (one matrix multiply per obs)
  - Decoupled from GMM regime model (avoids coupling)
"""

import logging
from collections import deque
from typing import Optional

import numpy as np
from scipy.spatial.distance import mahalanobis

logger = logging.getLogger(__name__)


class MahalanobisOODDetector:
    """Mahalanobis-distance based OOD detector.

    Fitted on training-set features (scaled, 122-dim without env state).
    At runtime: distance > threshold -> is_ood, confidence_mult decays.
    """

    def __init__(
        self,
        percentile_threshold: float = 97.5,
        hard_switch_steps: int = 6,
        soft_decay_scale: float = 2.0,
        shadow_percentile: float = 95.0,
        shadow_min_samples: int = 2,
        shadow_buffer_size: int = 96,
        shadow_margin: float = 1.10,
    ):
        self.percentile_threshold = percentile_threshold
        self.hard_switch_steps = hard_switch_steps
        self.soft_decay_scale = soft_decay_scale
        self.shadow_percentile = shadow_percentile
        self.shadow_min_samples = max(1, int(shadow_min_samples))
        self.shadow_buffer_size = max(self.shadow_min_samples, int(shadow_buffer_size))
        self.shadow_margin = max(1.0, float(shadow_margin))

        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._threshold: float = 0.0
        self._consecutive_ood_count: int = 0
        self._is_fitted: bool = False
        self._shadow_distances = deque(maxlen=self.shadow_buffer_size)

    def fit(self, train_features: np.ndarray):
        """Fit distribution parameters on training set features.

        Args:
            train_features: (N, D) scaled feature matrix.
                            Must be only feature columns (no env state dims).
        """
        assert train_features.ndim == 2, f"Expected 2D, got {train_features.ndim}D"

        self._mean = np.mean(train_features, axis=0)
        cov = np.cov(train_features, rowvar=False)

        # Regularize to prevent singular matrix
        cov += np.eye(cov.shape[0]) * 1e-6
        self._cov_inv = np.linalg.inv(cov)

        # Compute distances for all training samples to set threshold
        distances = np.array([
            mahalanobis(x, self._mean, self._cov_inv)
            for x in train_features
        ])

        self._threshold = float(np.percentile(distances, self.percentile_threshold))
        self._is_fitted = True

        logger.info(
            f"[OOD] Fitted on {len(train_features)} samples, "
            f"dim={train_features.shape[1]}, "
            f"threshold={self._threshold:.2f} "
            f"(p{self.percentile_threshold})"
        )

    def _confidence_mult_for(self, dist: float, threshold: float) -> float:
        if dist <= threshold:
            return 1.0
        if threshold <= 0 or self.soft_decay_scale <= 0:
            return 0.1
        excess_ratio = (dist - threshold) / (threshold * self.soft_decay_scale)
        confidence_mult = 1.0 / (1.0 + excess_ratio ** 2)
        return max(float(confidence_mult), 0.1)

    def _shadow_threshold(self) -> tuple[float, bool]:
        if len(self._shadow_distances) < self.shadow_min_samples:
            return self._threshold, False

        recent = np.asarray(self._shadow_distances, dtype=np.float64)
        recent = recent[np.isfinite(recent)]
        if recent.size < self.shadow_min_samples:
            return self._threshold, False

        adaptive = float(np.percentile(recent, self.shadow_percentile))
        if not np.isfinite(adaptive) or adaptive <= 0:
            return self._threshold, False

        return max(self._threshold, adaptive * self.shadow_margin), True

    def _score_internal(self, obs_features: np.ndarray, *, update_state: bool) -> dict:
        """Common scorer with optional mutation of the consecutive-OOD counter."""
        if not self._is_fitted:
            return {
                'distance': 0.0, 'threshold': 0.0, 'is_ood': False,
                'confidence_mult': 1.0, 'consecutive_ood': 0, 'hard_switch': False,
            }

        dist = float(mahalanobis(obs_features, self._mean, self._cov_inv))
        is_ood = dist > self._threshold

        consecutive_ood = self._consecutive_ood_count
        if update_state:
            if is_ood:
                self._consecutive_ood_count += 1
            else:
                self._consecutive_ood_count = 0
            consecutive_ood = self._consecutive_ood_count
        elif is_ood:
            consecutive_ood = self._consecutive_ood_count + 1
        else:
            consecutive_ood = 0

        confidence_mult = self._confidence_mult_for(dist, self._threshold)

        hard_switch = consecutive_ood >= self.hard_switch_steps

        return {
            'distance': dist,
            'threshold': self._threshold,
            'is_ood': is_ood,
            'confidence_mult': float(confidence_mult),
            'consecutive_ood': consecutive_ood,
            'hard_switch': hard_switch,
        }

    def score(self, obs_features: np.ndarray) -> dict:
        """Score and update the consecutive-OOD state."""
        return self._score_internal(obs_features, update_state=True)

    def peek_score(self, obs_features: np.ndarray) -> dict:
        """Score without mutating the consecutive-OOD state.

        Used when DRL is telemetry-only so shadow monitoring does not accumulate
        synthetic hard-switch state.
        """
        return self._score_internal(obs_features, update_state=False)

    def score_shadow(self, obs_features: np.ndarray) -> dict:
        """Telemetry-only score with adaptive threshold calibration."""
        base = self._score_internal(obs_features, update_state=False)
        dist = float(base['distance'])
        if np.isfinite(dist):
            self._shadow_distances.append(dist)

        adaptive_threshold, calibrated = self._shadow_threshold()
        calibrating = not calibrated
        is_ood = False if calibrating else dist > adaptive_threshold
        confidence_mult = 1.0 if calibrating else self._confidence_mult_for(dist, adaptive_threshold)

        return {
            'distance': dist,
            'distance_raw': dist,
            'threshold': adaptive_threshold,
            'threshold_raw': base['threshold'],
            'is_ood': is_ood,
            'confidence_mult': confidence_mult,
            'consecutive_ood': 0,
            'hard_switch': False,
            'shadow_calibrated': calibrated,
            'shadow_calibrating': calibrating,
            'shadow_samples': len(self._shadow_distances),
        }

    def reset_consecutive(self):
        """Reset consecutive OOD counter (e.g. on new episode)."""
        self._consecutive_ood_count = 0

    def save(self, path: str):
        """Save fitted parameters to .npz file."""
        np.savez(
            path,
            mean=self._mean,
            cov_inv=self._cov_inv,
            threshold=self._threshold,
            percentile_threshold=self.percentile_threshold,
            hard_switch_steps=self.hard_switch_steps,
            soft_decay_scale=self.soft_decay_scale,
            shadow_percentile=self.shadow_percentile,
            shadow_min_samples=self.shadow_min_samples,
            shadow_buffer_size=self.shadow_buffer_size,
            shadow_margin=self.shadow_margin,
        )
        logger.info(f"[OOD] Saved detector to {path}")

    @classmethod
    def load(cls, path: str) -> 'MahalanobisOODDetector':
        """Load fitted detector from .npz file."""
        data = np.load(path)
        detector = cls(
            percentile_threshold=float(data['percentile_threshold']),
            hard_switch_steps=int(data['hard_switch_steps']),
            soft_decay_scale=float(data['soft_decay_scale']),
            shadow_percentile=(
                float(data['shadow_percentile'])
                if 'shadow_percentile' in data.files else 95.0
            ),
            shadow_min_samples=(
                int(data['shadow_min_samples'])
                if 'shadow_min_samples' in data.files else 2
            ),
            shadow_buffer_size=(
                int(data['shadow_buffer_size'])
                if 'shadow_buffer_size' in data.files else 96
            ),
            shadow_margin=(
                float(data['shadow_margin'])
                if 'shadow_margin' in data.files else 1.10
            ),
        )
        detector._mean = data['mean']
        detector._cov_inv = data['cov_inv']
        detector._threshold = float(data['threshold'])
        detector._is_fitted = True
        logger.info(
            f"[OOD] Loaded detector from {path}, "
            f"dim={detector._mean.shape[0]}, threshold={detector._threshold:.2f}"
        )
        return detector

"""
================================================================================
HMATS v7 - Per-Regime DRL Router
================================================================================
Routes DRL inference through per-regime specialist TQC models.
Uses GMM posterior probabilities as natural mixture weights.

    final_action = sum(posterior_i * specialist_action_i for each regime_i)

Fallback: regimes without enough training data use the universal model.

Usage:
    router = RegimeRouter("BTC", "models/drl_per_regime/BTC/regime_router_manifest.json")
    action = router.predict(obs_121d, gmm_posteriors_kd)
================================================================================
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class RegimeRouter:
    """
    Routes DRL inference through per-regime specialist models.

    Each semantic group (TRENDING_UP, VOLATILE, ACCUMULATION, etc.) may have
    its own TQC model trained exclusively on that regime's data. Groups with
    insufficient data fall back to the universal (all-regime) model.

    GMM posteriors are the natural router weights - no separate router training
    required.
    """

    def __init__(self, asset: str, manifest_path: str):
        self.asset = asset
        self.manifest_path = manifest_path
        self._models: Dict[str, object] = {}   # group_name -> TQC model
        self._universal_model = None
        self._regime_id_to_group: Dict[str, str] = {}
        self._loaded = False
        self._load_count = 0
        self._fallback_count = 0

    def load(self) -> bool:
        """
        Load manifest and all models. Returns True on success.
        Call once at startup (lazy or explicit).
        """
        try:
            manifest_file = Path(self.manifest_path)
            if not manifest_file.exists():
                logger.warning(
                    f"[REGIME_ROUTER] Manifest not found: {self.manifest_path}"
                )
                return False

            with open(manifest_file, "r") as f:
                self._manifest = json.load(f)

            self._regime_id_to_group = self._manifest.get("regime_id_to_group", {})

            # Load universal model first
            universal_info = self._manifest.get("models", {}).get("universal")
            if universal_info and universal_info.get("path"):
                self._universal_model = self._load_tqc(universal_info["path"])
                if self._universal_model is None:
                    logger.error("[REGIME_ROUTER] Universal model failed to load")
                    return False
            else:
                logger.error("[REGIME_ROUTER] No universal model in manifest")
                return False

            # Load per-group models
            for group_name, info in self._manifest.get("models", {}).items():
                if group_name == "universal":
                    self._models[group_name] = self._universal_model
                    continue

                if info.get("is_fallback", True):
                    self._models[group_name] = self._universal_model
                    self._fallback_count += 1
                    logger.debug(
                        f"[REGIME_ROUTER] {group_name}: fallback to universal "
                        f"({info.get('fallback_reason', 'N/A')})"
                    )
                else:
                    model = self._load_tqc(info["path"])
                    if model is not None:
                        self._models[group_name] = model
                        self._load_count += 1
                    else:
                        self._models[group_name] = self._universal_model
                        self._fallback_count += 1
                        logger.warning(
                            f"[REGIME_ROUTER] {group_name}: load failed, using universal"
                        )

            self._loaded = True
            logger.info(
                f"[REGIME_ROUTER] {self.asset}: loaded {self._load_count} specialists + "
                f"{self._fallback_count} fallbacks + 1 universal"
            )
            return True

        except Exception as e:
            logger.error(f"[REGIME_ROUTER] Load failed: {e}")
            return False

    def predict(
        self,
        obs: np.ndarray,
        gmm_posteriors: np.ndarray,
        deterministic: bool = True,
    ) -> float:
        """
        Predict action using posterior-weighted specialist ensemble.

        Args:
            obs: observation vector (121 dims), already scaled
            gmm_posteriors: GMM predict_proba output (k dims, sums to 1)
            deterministic: whether to use deterministic policy

        Returns:
            Blended action in [-1.0, 1.0]
        """
        if not self._loaded:
            if not self.load():
                # Total failure - return neutral
                return 0.0

        # Map GMM posteriors -> semantic group weights
        group_weights = self._posteriors_to_group_weights(gmm_posteriors)

        # Get action from each group's model, weighted by posterior
        blended_action = 0.0
        total_weight = 0.0

        for group_name, weight in group_weights.items():
            if weight < 0.01:
                continue

            model = self._models.get(group_name)
            if model is None:
                model = self._universal_model

            try:
                action, _ = model.predict(obs, deterministic=deterministic)
                act_val = float(action[0]) if hasattr(action, '__len__') else float(action)
                blended_action += weight * act_val
                total_weight += weight
            except Exception as e:
                logger.debug(f"[REGIME_ROUTER] predict failed for {group_name}: {e}")
                continue

        if total_weight > 0:
            blended_action /= total_weight

        return float(np.clip(blended_action, -1.0, 1.0))

    def predict_per_group(
        self,
        obs: np.ndarray,
        gmm_posteriors: np.ndarray,
        deterministic: bool = True,
    ) -> Dict[str, dict]:
        """
        Get per-group actions and weights (for debugging/logging).

        Returns dict of {group_name: {"action": float, "weight": float, "is_fallback": bool}}
        """
        if not self._loaded:
            self.load()

        group_weights = self._posteriors_to_group_weights(gmm_posteriors)
        results = {}

        for group_name, weight in group_weights.items():
            model = self._models.get(group_name, self._universal_model)
            is_fallback = model is self._universal_model and group_name != "universal"

            try:
                action, _ = model.predict(obs, deterministic=deterministic)
                act_val = float(action[0]) if hasattr(action, '__len__') else float(action)
            except Exception:
                act_val = 0.0

            results[group_name] = {
                "action": act_val,
                "weight": weight,
                "is_fallback": is_fallback,
            }

        return results

    def _posteriors_to_group_weights(
        self, posteriors: np.ndarray
    ) -> Dict[str, float]:
        """
        Convert per-regime GMM posteriors to per-semantic-group weights.

        Multiple GMM regimes can map to the same semantic group - their
        posteriors are summed.
        """
        group_weights: Dict[str, float] = defaultdict(float)

        for regime_id, prob in enumerate(posteriors):
            group = self._regime_id_to_group.get(str(regime_id), "LOW_VOL_RANGE")
            group_weights[group] += float(prob)

        return dict(group_weights)

    def _load_tqc(self, path_str: str) -> Optional[object]:
        """Load a TQC model. Returns None on failure."""
        try:
            from sb3_contrib import TQC

            path = Path(path_str)
            # SB3 auto-appends .zip - strip if present
            load_path = str(path).replace(".zip", "")
            model = TQC.load(load_path)
            return model
        except Exception as e:
            logger.warning(f"[REGIME_ROUTER] TQC.load failed for {path_str}: {e}")
            return None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_status(self) -> Dict:
        """Status for monitoring."""
        return {
            "asset": self.asset,
            "loaded": self._loaded,
            "specialists": self._load_count,
            "fallbacks": self._fallback_count,
            "groups": list(self._models.keys()),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_routers: Dict[str, RegimeRouter] = {}


def get_regime_router(asset: str) -> Optional[RegimeRouter]:
    """
    Get or create a RegimeRouter for an asset.
    Returns None if manifest doesn't exist.
    """
    if asset in _routers:
        return _routers[asset]

    manifest_path = f"models/drl_per_regime/{asset}/regime_router_manifest.json"
    if not Path(manifest_path).exists():
        return None

    router = RegimeRouter(asset, manifest_path)
    if router.load():
        _routers[asset] = router
        return router
    return None


def reset_routers():
    """Reset all cached routers."""
    global _routers
    _routers = {}

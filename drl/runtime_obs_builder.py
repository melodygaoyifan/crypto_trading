"""
RuntimeObsBuilder - builds 126-dim observation vector at runtime.

Matches training pipeline exactly:
  102 base features (FeatureEngineer) + 5 wavelet + 7 external + 8 regime_proba = 122
  + 4 env state = 126 dims

Usage:
    builder = RuntimeObsBuilder("configs/feature_manifest.json", "models/retrained")
    obs = builder.build_obs("BTC", ohlcv_df, market_data, gmm_probs, env_state)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("HMATS.ObsBuilder")

# Best folds per asset (must match drl/ensemble.py BEST_FOLDS)
# [FIX 2026-04-22] ETH fold_1 -> fold_3 (stale metadata). See ensemble.py.
BEST_FOLDS = {"BTC": "fold_3", "ETH": "fold_3", "SOL": "fold_3"}


def _fix_drl_shadowing():
    """Remove training/ from sys.path to prevent drl module shadowing.

    training/train_drl_full.py line 74 adds training/ to sys.path,
    which makes `import drl` find training/drl/ instead of top-level drl/.
    """
    training_dir = str(Path(__file__).resolve().parent.parent / "training")
    if training_dir in sys.path:
        sys.path.remove(training_dir)
    # Clear stale drl entries that came from training/drl/
    for key in list(sys.modules.keys()):
        mod = sys.modules.get(key)
        if mod and key.startswith("drl") and hasattr(mod, "__file__") and mod.__file__:
            if "training" in str(mod.__file__):
                sys.modules.pop(key, None)


class RuntimeObsBuilder:
    """Builds 126-dim DRL observation from live market data."""

    def __init__(
        self,
        manifest_path: str = "configs/feature_manifest.json",
        include_fv2: bool = False,  # [P1b] serve a 139-obs (fv2-era) model
        scaler_dir: str = "models/retrained",
        assets: Optional[List[str]] = None,
    ):
        assets = assets or ["BTC", "ETH", "SOL"]
        self._manifest_path = Path(manifest_path)
        self._scaler_dir = Path(scaler_dir)

        # Load feature manifest
        self._all_features: List[str] = []
        self._base_features: List[str] = []
        self._denoised_features: List[str] = []
        self._external_features: List[str] = []
        self._regime_proba_features: List[str] = []
        self._no_scale: set = set()
        self._include_fv2 = bool(include_fv2)
        self._load_manifest()
        if self._include_fv2:
            # [P1b] Mirror the trainer's --include-fv2 EXACTLY: the 13 fv2
            # features are inserted BEFORE the trailing regime_proba block
            # (the FiLM extractor slices regime as obs[-12:-4], END-relative).
            from data_mgmt.flow_features import FV2_COLUMNS
            tail = self._all_features[-8:]
            assert all(c.startswith("regime_proba") for c in tail), (
                "manifest tail is not the 8 regime_proba columns — the FiLM "
                "slice contract has changed; do not insert fv2 blindly")
            self._fv2_features = sorted(FV2_COLUMNS)
            self._all_features = (self._all_features[:-8]
                                  + self._fv2_features + tail)
        else:
            self._fv2_features = []

        # Per-asset scalers
        self._scalers: Dict[str, dict] = {}
        for asset in assets:
            self._load_scaler(asset)

        # Feature engineer (lazy import to avoid training/ shadowing)
        self._feature_engineer = None
        self._init_feature_engineer()

        logger.info(
            f"[ObsBuilder] Ready: {len(self._all_features)} features, "
            f"{len(self._scalers)} asset scalers"
        )

    def _load_manifest(self):
        """Load feature manifest from JSON."""
        if not self._manifest_path.exists():
            logger.warning(f"[ObsBuilder] Manifest not found: {self._manifest_path}")
            return
        with open(self._manifest_path) as f:
            manifest = json.load(f)
        self._all_features = manifest.get("all_features", [])
        self._base_features = manifest.get("base_features", [])
        self._denoised_features = manifest.get("denoised_features", [])
        self._external_features = manifest.get("external_features", [])
        self._regime_proba_features = manifest.get("regime_proba_features", [])
        self._no_scale = set(manifest.get("no_scale_features", []))
        logger.info(
            f"[ObsBuilder] Manifest: {len(self._all_features)} features, "
            f"{len(self._no_scale)} no_scale"
        )

    def _load_scaler(self, asset: str):
        """Load per-asset feature scaler from training output."""
        best_fold = BEST_FOLDS.get(asset, "fold_1")
        scaler_path = (
            self._scaler_dir / asset / best_fold / "logs" / "LSTM_FILM_A" / "feature_scaler.pkl"
        )
        if not scaler_path.exists():
            logger.warning(f"[ObsBuilder] Scaler not found for {asset}: {scaler_path}")
            return

        try:
            import joblib
            from infra.safe_torch_load import safe_joblib_load
            scaler_obj = safe_joblib_load(scaler_path, joblib_module=joblib)
            if isinstance(scaler_obj, dict):
                self._scalers[asset] = scaler_obj
            else:
                # Backward compat: raw RobustScaler
                self._scalers[asset] = {
                    "scaler": scaler_obj,
                    "scale_cols": [f for f in self._all_features if f not in self._no_scale],
                    "pass_cols": list(self._no_scale),
                }
            logger.info(f"[ObsBuilder] Scaler loaded for {asset} ({scaler_path})")
        except Exception as e:
            logger.warning(f"[ObsBuilder] Failed to load scaler for {asset}: {e}")

    def _init_feature_engineer(self):
        """Load FeatureEngineer from training/drl/features.py via importlib."""
        try:
            import importlib.util
            features_path = Path(__file__).resolve().parent.parent / "training" / "drl" / "features.py"
            if not features_path.exists():
                logger.warning(f"[ObsBuilder] features.py not found: {features_path}")
                return
            spec = importlib.util.spec_from_file_location("_training_drl_features", str(features_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._feature_engineer = mod.FeatureEngineer(state_dim=128, cross_asset_dim=16)
            logger.info("[ObsBuilder] FeatureEngineer loaded from training/drl/features.py")
        except Exception as e:
            logger.warning(f"[ObsBuilder] FeatureEngineer unavailable: {e}. Base features disabled.")

    def build_obs(
        self,
        asset: str,
        ohlcv_df: Optional[pd.DataFrame],
        market_data: dict,
        gmm_probs: list,
        env_state: list,
        fv2: "Optional[dict]" = None,   # [P1b] required when include_fv2
    ) -> "Optional[np.ndarray]":
        """Build 126-dim observation vector.

        Args:
            asset: "BTC", "ETH", or "SOL"
            ohlcv_df: DataFrame with columns [timestamp, open, high, low, close, volume]
            market_data: raw market_data dict from tick (contains wavelet features etc.)
            gmm_probs: list of regime probabilities from GMM (len <= 8)
            env_state: [position_ratio, position_direction, pnl_ratio, drawdown]

        Returns:
            np.ndarray of shape (126,) float32
        """
        if self._include_fv2 and not fv2:
            # A 139-obs model without fv2 values cannot infer. Refuse loudly
            # rather than zero-fill: 13 fabricated zeros is a fake observation
            # (P160/P170 — absence must never look like a measurement).
            logger.warning(f"[ObsBuilder] {asset}: fv2 features unavailable "
                           f"this tick — obs NOT built, model must skip")
            return None
        features_122 = np.zeros(len(self._all_features), dtype=np.float32)

        # 1. Base features (102) via FeatureEngineer
        if self._feature_engineer is not None and ohlcv_df is not None and len(ohlcv_df) > 0:
            try:
                computed_df = self._feature_engineer.compute_features(ohlcv_df)
                last_row = computed_df.iloc[-1]
                for i, feat_name in enumerate(self._base_features):
                    if feat_name in last_row.index:
                        val = last_row[feat_name]
                        features_122[i] = float(val) if pd.notna(val) else 0.0
            except Exception as e:
                logger.debug(f"[ObsBuilder] {asset}: FeatureEngineer failed: {e}")

        # 2. Denoised features (5) - from market_data wavelet cache
        base_count = len(self._base_features)
        for j, feat_name in enumerate(self._denoised_features):
            idx = base_count + j
            val = market_data.get(feat_name, 0.0)
            features_122[idx] = float(val) if val is not None else 0.0

        # 3. External features (7) - derived upstream from live feeds/trade activity
        ext_offset = base_count + len(self._denoised_features)
        for j, feat_name in enumerate(self._external_features):
            idx = ext_offset + j
            val = market_data.get(feat_name, 0.0)
            features_122[idx] = float(val) if val is not None else 0.0

        # 3b. [P1b] fv2 flow features (13) — from the BinanceFlowFeed
        fv2_offset = ext_offset + len(self._external_features)
        for j, feat_name in enumerate(self._fv2_features):
            val = (fv2 or {}).get(feat_name)
            features_122[fv2_offset + j] = float(val) if val is not None else 0.0

        # 4. Regime proba (8) - from GMM predictions
        regime_offset = fv2_offset + len(self._fv2_features)
        for j in range(len(self._regime_proba_features)):
            idx = regime_offset + j
            if j < len(gmm_probs):
                features_122[idx] = float(gmm_probs[j])

        # 5. Scale features
        features_122 = self._scale_features(asset, features_122)

        # 6. Append env state [position_ratio, position_direction, pnl_ratio, drawdown]
        env = np.array(env_state, dtype=np.float32)[:4]
        if len(env) < 4:
            env = np.pad(env, (0, 4 - len(env)))

        # 7. Concatenate and clean
        obs = np.concatenate([features_122, env])
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = np.clip(obs, -10.0, 10.0)

        return obs.astype(np.float32)

    def _scale_features(self, asset: str, features: np.ndarray) -> np.ndarray:
        """Apply per-asset RobustScaler to scalable features, clip to [-10, 10]."""
        scaler_meta = self._scalers.get(asset)
        if scaler_meta is None:
            return features

        scaler = scaler_meta.get("scaler")
        scale_cols = scaler_meta.get("scale_cols", [])
        if scaler is None or not scale_cols:
            return features

        # Build index mapping: which positions in all_features are scalable
        scale_indices = []
        for col in scale_cols:
            if col in self._all_features:
                scale_indices.append(self._all_features.index(col))

        if not scale_indices:
            return features

        # Extract scalable values, transform, clip, put back
        scalable = features[scale_indices].reshape(1, -1)
        try:
            scaled = scaler.transform(scalable)
            scaled = np.clip(scaled, -10.0, 10.0)
            features[scale_indices] = scaled[0]
        except Exception as e:
            logger.debug(f"[ObsBuilder] {asset}: scaler.transform failed: {e}")

        return features

    @property
    def feature_cols(self) -> List[str]:
        """Return the full 122-feature column list for OOD detector compatibility."""
        return list(self._all_features)

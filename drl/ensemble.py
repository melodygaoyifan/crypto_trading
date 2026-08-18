"""
TQC (LSTM_FILM_A) inference wrapper with obs stacking.

Grid search results (Mar 2026):
  DT adds no value for any asset -> TQC-only for all.
  This module provides TQC inference with n_stack=8 frame stacking.

Usage (runtime):
    from drl.ensemble import TQCInference, load_best_model

    tqc = load_best_model(asset="BTC")
    result = tqc.predict(obs_126, regime="MOMENTUM_RALLY")

Backward compat aliases:
    TQCDTEnsemble = TQCInference
    load_best_ensemble = load_best_model
"""

import importlib
import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("TQC_Inference")

# Frame stacking config (must match training)
N_STACK = 8
SINGLE_OBS_DIM = 126  # 122 features + 4 env state

# Best folds from training results
# [FIX 2026-04-22] ETH fold_1 -> fold_3. results.json reports fold_1 reward=1400
# but train_rows=0/train_time=0 — stale checkpoint with bogus metadata.
# fold_3 (reward=1029, train_rows=10028) is the real best. Same fix in
# training/drl/oracle_tqc_teacher.py BEST_FOLDS.
BEST_FOLDS = {"BTC": "fold_3", "ETH": "fold_3", "SOL": "fold_3"}

# Backward compat - main.py imports this
REGIME_WEIGHTS = {}


def _register_training_modules():
    """Register training/models as 'models' in sys.modules for cloudpickle compat.

    TQC models were pickled from training/ dir with imports like
    'from models.film_extractor import LSTMFiLMPosAExtractor'.
    At load time from project root, we need 'models' -> 'training.models'.
    """
    if "models" not in sys.modules or sys.modules["models"] is None:
        try:
            training_models = importlib.import_module("training.models")
            sys.modules["models"] = training_models
            film = importlib.import_module("training.models.film_extractor")
            sys.modules["models.film_extractor"] = film
            feat = importlib.import_module("training.models.feature_extractors")
            sys.modules["models.feature_extractors"] = feat
        except Exception as e:
            logger.warning(f"Failed to register models alias: {e}")


@dataclass
class TQCResult:
    """Result from TQC prediction."""
    action: float                # Action [-1, 1]
    tqc_action: float            # Same as action (compat)
    regime: str                  # Current regime
    confidence: float            # |action| * 0.5
    tqc_uncertainty_ratio: float = 0.0  # IQR/|median| of TQC quantiles
    # Backward compat fields (always zero/default)
    dt_prediction: float = 0.0
    dt_confidence: float = 0.0
    tqc_weight: float = 1.0
    dt_weight: float = 0.0
    agreement: float = 1.0
    dt_mode: str = "DISABLED"


# Backward compat alias
EnsembleResult = TQCResult


class TQCInference:
    """TQC (LSTM_FILM_A) inference with obs stacking.

    Architecture:
        1. Maintains deque(maxlen=8) for frame stacking
        2. Zero-pads if buffer < 8 frames
        3. Feeds stacked (1, 1008) obs to TQC.predict(deterministic=True)
        4. Quantile spread for uncertainty estimation
    """

    def __init__(
        self,
        tqc_path: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        device: str = "auto",
        asset: Optional[str] = None,
        # Ignored backward compat params
        dt_path: Optional[str] = None,
        dt_mode: str = "DISABLED",
    ):
        self.feature_cols = feature_cols or []
        self.dt_mode = "DISABLED"
        # [AP-12] Asset tag for unrounded inference log; falls back to "?" if caller
        # doesn't supply it (drl_drift validation script uses load_best_model which does).
        self._asset = asset or "?"

        self._tqc_model = None
        self._obs_buffer: deque = deque(maxlen=N_STACK)
        # [P148 2026-06-14] restore the frame buffer from a quick prior restart so
        # we don't zero-pad for 8 ticks (32h) every time the engine restarts.
        self._restore_buffer()

        if tqc_path:
            self._load_tqc(tqc_path, device)

    # ---- [P148] frame-buffer persistence (no 32h warmup after a restart) -------
    _BUFFER_MAX_AGE_SEC = 12 * 3600  # restore only if fresh enough to stay temporally consecutive (~3 bars)

    def _buffer_path(self) -> str:
        d = os.environ.get("HMATS_DATA_DIR", "data")
        return os.path.join(d, f"drl_obs_buffer_{self._asset}.npz")

    def _restore_buffer(self) -> None:
        try:
            p = self._buffer_path()
            if not os.path.exists(p):
                return
            z = np.load(p, allow_pickle=False)
            age = time.time() - float(z["saved_ts"])
            if age > self._BUFFER_MAX_AGE_SEC:
                logger.info(f"[DRL_BUFFER] {self._asset}: saved buffer stale "
                            f"({age/3600:.1f}h) -> warmup fresh (zero-pad)")
                return
            _dim = getattr(self, "single_obs_dim", SINGLE_OBS_DIM)  # [P1b]
            for f in z["frames"]:
                arr = np.asarray(f, dtype=np.float32)
                if len(arr) != _dim:
                    # A buffer persisted under a different obs width (e.g. a
                    # 126-frame buffer restored into a 139 model after a
                    # Rung-3 deploy) would poison the stack. Warmup fresh.
                    logger.warning(
                        f"[DRL_BUFFER] {self._asset}: saved frames are "
                        f"{len(arr)}-dim but the model expects {_dim} — "
                        f"discarding buffer, warming up fresh (P1b)")
                    self._obs_buffer.clear()
                    return
                self._obs_buffer.append(arr)
            logger.info(f"[DRL_BUFFER] {self._asset}: restored {len(self._obs_buffer)}/{N_STACK} "
                        f"frames (age {age/3600:.1f}h) -> no 32h warmup")
        except Exception as e:  # noqa: silent-swallow — bad/old buffer file; fall back to empty (warmup) + log
            logger.warning(f"[DRL_BUFFER] {self._asset}: restore failed: {type(e).__name__}: {e}")

    def _save_buffer(self) -> None:
        try:
            if not self._obs_buffer:
                return
            frames = np.stack([np.asarray(f, dtype=np.float32) for f in self._obs_buffer])
            p = self._buffer_path()
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            np.savez(p, frames=frames, saved_ts=np.float64(time.time()))
        except Exception as e:  # noqa: silent-swallow — persistence is best-effort, never break inference
            logger.debug(f"[DRL_BUFFER] {self._asset}: save failed: {type(e).__name__}: {e}")

    def _load_tqc(self, path: str, device: str = "auto"):
        """Load TQC model with LSTM_FILM_A support."""
        try:
            from sb3_contrib import TQC
            import torch

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # [FIX-DRL-VALIDATE] Validate model file exists and has non-trivial size
            # before attempting load. Prevents silent degrade from corrupt/truncated files.
            _zip_path = path if path.endswith(".zip") else path + ".zip"
            _model_path = Path(_zip_path)
            if not _model_path.exists():
                logger.error(f"[FIX-DRL-VALIDATE] Model file NOT FOUND: {_zip_path}")
                return
            _file_size = _model_path.stat().st_size
            if _file_size < 10_000:  # Healthy TQC models are >1MB
                logger.error(
                    f"[FIX-DRL-VALIDATE] Model file suspiciously small: "
                    f"{_zip_path} ({_file_size} bytes). Possible corruption."
                )
                return

            # Register module aliases so cloudpickle can deserialize the FiLM
            # extractor class natively. Do NOT pass custom_objects for policy_kwargs
            # because SB3's json_to_data REPLACES (not updates) the entire dict,
            # which drops n_quantiles, net_arch, features_extractor_kwargs, etc.
            _register_training_modules()

            load_path = path[:-4] if path.endswith(".zip") else path
            self._tqc_model = TQC.load(load_path, device=device)

            # [P1b] The checkpoint declares its own input width. Old models
            # are 126x8=1008; fv2-era models (P200-LADDER Rung 3) are
            # 139x8=1112. Deriving from the model's observation_space makes
            # the loader dimension-agnostic while VALIDATING the result —
            # an unexpected width is a hard refuse, not a squeeze.
            try:
                _total = int(np.prod(self._tqc_model.observation_space.shape))
                if _total % N_STACK != 0:
                    raise ValueError(f"obs width {_total} not divisible by n_stack {N_STACK}")
                self.single_obs_dim = _total // N_STACK
                if self.single_obs_dim not in (126, 139):
                    raise ValueError(
                        f"checkpoint single_obs_dim {self.single_obs_dim} is neither "
                        f"126 (legacy) nor 139 (fv2) — refusing to serve it")
            except Exception as _dim_err:
                logger.error(f"[P1b] obs-dim derivation failed: {_dim_err}")
                self._tqc_model = None
                return
            if self.single_obs_dim != SINGLE_OBS_DIM:
                logger.warning(
                    f"[P1b] checkpoint uses single_obs_dim={self.single_obs_dim} "
                    f"(fv2-era); the obs builder MUST supply fv2 features or "
                    f"prediction will be skipped every tick")
            # _restore_buffer ran in __init__ BEFORE the dim was known —
            # re-validate the restored frames against the true width now.
            if self._obs_buffer and len(self._obs_buffer[0]) != self.single_obs_dim:
                logger.warning(
                    f"[DRL_BUFFER] {self._asset}: restored frames are "
                    f"{len(self._obs_buffer[0])}-dim but the checkpoint expects "
                    f"{self.single_obs_dim} — discarding, warming up fresh (P1b)")
                self._obs_buffer.clear()

            # Smoke test: predict on zeros to verify model is functional
            try:
                _test_obs = np.zeros((1, self.single_obs_dim * N_STACK), dtype=np.float32)
                _test_action, _ = self._tqc_model.predict(_test_obs, deterministic=True)
                if np.isnan(_test_action).any() or np.isinf(_test_action).any():
                    logger.error(f"[FIX-DRL-VALIDATE] Model produces NaN/Inf on smoke test: {path}")
                    self._tqc_model = None
                    return
            except Exception as _smoke_err:
                logger.error(f"[FIX-DRL-VALIDATE] Smoke test FAILED: {_smoke_err}")
                self._tqc_model = None
                return

            logger.info(f"TQC loaded + validated: {path} (device={device}, size={_file_size:,}B)")
        except Exception as e:
            logger.error(f"[FIX-DRL-VALIDATE] Failed to load TQC model: {e}")

    @property
    def tqc_available(self) -> bool:
        return self._tqc_model is not None

    @property
    def dt_available(self) -> bool:
        return False

    def predict(
        self,
        obs: np.ndarray,
        regime: str = "WEAK_CONSOLIDATION",
        dt_features: Optional[np.ndarray] = None,  # ignored, backward compat
    ) -> TQCResult:
        """Get TQC prediction with obs stacking.

        Args:
            obs: 126-dim observation vector (122 features + 4 env state).
            regime: Current market regime name.

        Returns:
            TQCResult with action and metadata.
        """
        tqc_action = 0.0
        tqc_uncertainty = 0.0

        if self._tqc_model is not None:
            try:
                obs_126 = obs.astype(np.float32)
                _dim = getattr(self, "single_obs_dim", SINGLE_OBS_DIM)  # [P1b]
                _expected_stacked = _dim * N_STACK
                if len(obs_126) == _expected_stacked:
                    # [FIX-M7] Pre-stacked observation — validate exact dimension
                    stacked = obs_126.reshape(1, -1)
                elif len(obs_126) != _dim:
                    logger.error(
                        f"[TQC] Invalid obs dimension {len(obs_126)}, "
                        f"expected {_dim} or {_expected_stacked}"
                    )
                    stacked = np.zeros((1, _expected_stacked), dtype=np.float32)
                else:
                    self._obs_buffer.append(obs_126.copy())
                    self._save_buffer()  # [P148] persist so a restart doesn't zero-pad 32h
                    if len(self._obs_buffer) < N_STACK:
                        padded = [np.zeros(_dim, dtype=np.float32)] * (
                            N_STACK - len(self._obs_buffer)
                        )
                        padded.extend(self._obs_buffer)
                    else:
                        padded = list(self._obs_buffer)
                    stacked = np.concatenate(padded).reshape(1, -1)  # (1, 1008)

                action, _ = self._tqc_model.predict(stacked, deterministic=True)
                tqc_action = float(np.clip(action[0], -1.0, 1.0))
            except Exception as e:
                logger.warning(f"TQC predict failed: {e}")

            # Quantile spread for uncertainty estimation
            try:
                import torch as _th
                with _th.no_grad():
                    _obs_t = self._tqc_model.policy.obs_to_tensor(stacked)[0]
                    _act_t = _th.tensor([[tqc_action]], dtype=_th.float32,
                                        device=_obs_t.device)
                    if hasattr(self._tqc_model, 'critic'):
                        _q_out = self._tqc_model.critic(_obs_t, _act_t)
                        _all_q = _th.cat([q.squeeze() for q in _q_out])
                        _q25 = float(_th.quantile(_all_q, 0.25))
                        _q75 = float(_th.quantile(_all_q, 0.75))
                        _med = float(_th.median(_all_q))
                        tqc_uncertainty = (_q75 - _q25) / (abs(_med) + 1e-8)
            except Exception as _unc_err:
                # P71: previously bare `except: pass` left tqc_uncertainty=0.0
                # silently → fusion saw full DRL conviction even on critic
                # disagreement / OOM / device errors. Log so silent
                # overconfidence is observable in heartbeat logs. The
                # downstream confidence calc at line 246+ already handles
                # tqc_uncertainty=0.0 correctly (no discount applied).
                logger.debug(
                    f"[TQC_UNCERTAINTY] critic-quantile read failed "
                    f"({type(_unc_err).__name__}: {_unc_err}); "
                    f"falling back to undiscounted confidence"
                )

        # [FIX-DRL-CONF] Incorporate uncertainty into confidence so fusion sees
        # reduced conviction when critics disagree. Previously confidence = |action| × 0.5
        # with uncertainty only applied post-decide (too late for direction decision).
        # Now: confidence = |action| × 0.5 × (1 / (1 + uncertainty_ratio))
        # Examples: unc=0.0 → ×1.0 (full), unc=0.5 → ×0.67, unc=2.0 → ×0.33
        _base_conf = abs(tqc_action) * 0.5
        if tqc_uncertainty > 0:
            _unc_discount = 1.0 / (1.0 + tqc_uncertainty)
            _adj_conf = _base_conf * _unc_discount
        else:
            _adj_conf = _base_conf

        # [AP-12] Unrounded DRL inference log. Counterpart to the rounded
        # [BEST_OF_N_HOLD_OVERRIDE] punch-through log in integration_v36 which
        # uses :.2f and made TEST 5 of the drift-validation suite read as
        # COLLAPSED. Logged here at the predict() return site so operators have
        # ground-truth precision when auditing live behavior.
        if self._tqc_model is not None:
            logger.info(
                f"[DRL_INFERENCE] {self._asset} "
                f"dir={tqc_action:+.4f} conf={_adj_conf:+.4f} "
                f"unc={tqc_uncertainty:.4f} regime={regime}"
            )

        return TQCResult(
            action=tqc_action,
            tqc_action=tqc_action,
            regime=regime,
            confidence=_adj_conf,
            tqc_uncertainty_ratio=tqc_uncertainty,
        )


# Backward compat alias
TQCDTEnsemble = TQCInference


def load_best_model(
    asset: str,
    output_dir: str = "models/retrained",
    device: str = "auto",
) -> TQCInference:
    """Load the best TQC model for an asset."""
    base = Path(output_dir) / asset
    results_path = base / "results.json"
    tqc_path = None

    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
        # [FIX 2026-04-24 P4-regression] BEST_FOLDS must override results.json.
        # results.json can report stale best_fold (e.g. ETH fold_1 with
        # train_rows=0 from an aborted run), while BEST_FOLDS is the hand-verified
        # authoritative map. Previously `results.get("best_fold", BEST_FOLDS...)`
        # made results.json win — ETH inference loaded fold_1 model but
        # ObsBuilder/OOD used fold_3 scaler (mixed-fold pairing => broken math).
        best_fold = BEST_FOLDS.get(asset) or results.get("best_fold", "fold_3")

        for subdir in ["LSTM_FILM_A", "ULTIMATE"]:
            candidate = base / best_fold / "logs" / subdir / "best_model" / "best_model.zip"
            if candidate.exists():
                tqc_path = str(candidate)
                break
            candidate = base / best_fold / "logs" / subdir / "best_model.zip"
            if candidate.exists():
                tqc_path = str(candidate)
                break

    if tqc_path is None:
        logger.warning(f"No TQC model found for {asset} in {output_dir}")

    manifest_path = Path("configs/feature_manifest.json")
    feature_cols = None
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        feature_cols = manifest.get("all_features", [])

    tqc = TQCInference(
        tqc_path=tqc_path,
        feature_cols=feature_cols,
        device=device,
        asset=asset,
    )

    logger.info(f"TQC[{asset}]: {'READY' if tqc.tqc_available else 'NOT FOUND'}")
    return tqc


# Backward compat alias
load_best_ensemble = load_best_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TQC Inference")
    parser.add_argument("--asset", required=True, choices=["BTC", "ETH", "SOL"])
    parser.add_argument("--output", default="models/retrained")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    tqc = load_best_model(
        asset=args.asset,
        output_dir=args.output,
        device=args.device,
    )

    obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
    result = tqc.predict(obs, regime="MOMENTUM_RALLY")
    print(f"\nTest prediction:")
    print(f"  Action: {result.action:.4f}")
    print(f"  Uncertainty: {result.tqc_uncertainty_ratio:.4f}")
    print(f"  Confidence: {result.confidence:.4f}")

"""
HMATS v5.1 Phase 6 - ML Factor Fusion Agent (SHADOW MODE)
==========================================================

Loads a trained factor autoencoder (training/factor_extraction/) and at
each tick produces a {ml_factor_direction, ml_factor_confidence,
ml_factor_data_quality} signal from the live 122-feature observation.

Authority level: ADVISE (Iron Law 6 — quant DECIDE retained, this is
ADVISE-only). Until 30d shadow accumulates and Phase 10 promotion
gates clear, the agent runs in SHADOW MODE: signals are emitted to
data/strategy_shadow/ml_factor_*.jsonl JSONL but NOT injected into
agent_signals fusion.

Iron Law contracts:
  1. obs_dim=126 unchanged at runtime — agent reads existing market_data
     features, produces a NEW signal at agent layer (post-fusion-gate).
  3. training/ touch limited to factor_extraction/ subdir (constitutional
     override per __init__.py of that subdir).
  5. DRL ACTIVE preserved — this agent does not touch DRL inference.
  6. ADVISE only — quant DECIDE retained.
  7. Shadow >=30d before promotion via the existing StrategyShadow harness.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)


REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "models" / "factor_extraction"


class ShadowSignal(NamedTuple):
    strategy_name: str
    asset: str
    direction: float
    confidence: float
    reason: str
    diagnostics: Dict[str, Any]


def NEUTRAL(strategy_name: str, asset: str, reason: str = "insufficient_data") -> ShadowSignal:
    return ShadowSignal(strategy_name, asset, 0.0, 0.0, reason, {})


def _try_import_torch():
    try:
        import torch  # noqa
        return True
    except ImportError:  # noqa: silent-swallow
        # Optional dep; agent fails closed (returns NEUTRAL) when torch absent.
        # Caller surfaces this via _load_error="torch_unavailable".
        return False


@dataclass
class MLFactorFusionAgent:
    """ADVISE-authority agent that converts encoded latent factors → direction.

    Loaded encoder + per-asset feature scaler (median + IQR from training).
    On each tick:
        1. Extract 122-feature vector from market_data
        2. Encode → latent vector (latent_dim ~= 5)
        3. Combine via per-dim IC-weighted projection: direction = sign of
           sum(latent[k] * sign(ic_per_dim[k])) over PROMOTED dims
        4. Confidence = clamp(weighted magnitude / saturation, 0, 0.50)

    Iron Law 6: confidence cap = 0.50 keeps this strictly ADVISE-tier.
    Falls back to NEUTRAL on any missing input or model load failure.
    """
    asset: str
    model_path: Optional[Path] = None
    confidence_cap: float = 0.50
    saturation_z: float = 2.0

    _encoder: Any = field(default=None, init=False)
    _median: Optional[List[float]] = field(default=None, init=False)
    _iqr: Optional[List[float]] = field(default=None, init=False)
    _input_dim: int = field(default=0, init=False)
    _latent_dim: int = field(default=0, init=False)
    _ic_per_dim: List[float] = field(default_factory=list, init=False)
    _promoted_dims: List[int] = field(default_factory=list, init=False)
    _loaded: bool = field(default=False, init=False)
    _load_error: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not _try_import_torch():
            self._load_error = "torch_unavailable"
            return
        if self.model_path is None:
            self.model_path = MODEL_DIR / self.asset / "factor_autoencoder.pt"
        if not self.model_path.exists():
            self._load_error = f"model_missing:{self.model_path}"
            return

        try:
            import torch
            from training.factor_extraction.autoencoder_factor import build_autoencoder
            blob = torch.load(self.model_path, weights_only=False)
            self._input_dim = int(blob["input_dim"])
            self._latent_dim = int(blob["latent_dim"])
            self._median = list(map(float, blob["median"]))
            self._iqr = list(map(float, blob["iqr"]))
            encoder, _decoder = build_autoencoder(
                input_dim=self._input_dim, latent_dim=self._latent_dim
            )
            encoder.load_state_dict(blob["encoder"])
            encoder.eval()
            self._encoder = encoder
            self._loaded = True
            # [P147-c] IC table embedded in the model blob (training re-saves it
            # after computing per-latent IC). Without it, latent->direction
            # projection has no promoted dims -> NEUTRAL("no_promoted_dims...").
            if blob.get("ic_per_dim") is not None and blob.get("promoted_dims") is not None:
                self.load_ic_report(blob["ic_per_dim"], blob["promoted_dims"])
        except Exception as e:
            self._load_error = f"load_failed:{type(e).__name__}:{e}"
            logger.warning(
                f"[ML_FACTOR] {self.asset} encoder load failed: {self._load_error}"
            )

    def load_ic_report(self, ic_per_dim: List[float], promoted_dims: List[int]) -> None:
        """Operator-set: pass IC table from training report so the agent knows
        which latent dims to use for direction projection."""
        self._ic_per_dim = list(ic_per_dim)
        self._promoted_dims = list(promoted_dims)

    def _extract_features(self, market_data: Dict[str, Any]) -> Optional[List[float]]:
        """Build the 122-feature vector matching training schema. Loads the
        canonical feature order from configs/feature_manifest.json."""
        manifest_path = REPO / "configs" / "feature_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:  # noqa: silent-swallow
            return None
        all_features = manifest.get("all_features", [])
        if len(all_features) != self._input_dim:
            return None
        # Fail-closed: any missing feature → NEUTRAL upstream
        vec: List[float] = []
        for fname in all_features:
            v = market_data.get(fname)
            if v is None:
                return None
            try:
                f = float(v)
                if f != f:
                    return None
            except (TypeError, ValueError):  # noqa: silent-swallow
                return None
            vec.append(f)
        return vec

    def evaluate(self, asset: str, market_data: Dict[str, Any]) -> ShadowSignal:
        if not self._loaded:
            return NEUTRAL("ml_factor", asset, self._load_error or "not_loaded")
        if asset != self.asset:
            return NEUTRAL("ml_factor", asset, f"agent_bound_to_{self.asset}")

        feats = self._extract_features(market_data)
        if feats is None:
            return NEUTRAL("ml_factor", asset, "missing_features")

        try:
            import torch
            x = torch.tensor([feats], dtype=torch.float32)
            # Apply per-feature robust scaling
            for i, (m, q) in enumerate(zip(self._median, self._iqr)):
                x[0, i] = (x[0, i] - m) / max(q, 1e-6)
            with torch.no_grad():
                z = self._encoder(x).cpu().numpy()[0]
        except Exception as e:  # noqa: silent-swallow
            # Inference failure is fail-closed: NEUTRAL with reason. Per-tick
            # WARN would spam if model is malformed; reason is in JSONL ledger.
            return NEUTRAL("ml_factor", asset, f"inference_error:{type(e).__name__}")

        # Direction: sum of (latent[k] × sign(ic[k])) over promoted dims
        if not self._promoted_dims or not self._ic_per_dim:
            return NEUTRAL("ml_factor", asset, "no_promoted_dims_or_ic_table")

        weighted_sum = 0.0
        for k in self._promoted_dims:
            if k >= len(self._ic_per_dim):
                continue
            sign = 1.0 if self._ic_per_dim[k] >= 0 else -1.0
            weighted_sum += sign * float(z[k])

        magnitude = abs(weighted_sum)
        if magnitude < 0.10:
            return NEUTRAL("ml_factor", asset, f"weak_signal({weighted_sum:+.3f})")

        direction = 1.0 if weighted_sum > 0 else -1.0
        confidence = min(self.confidence_cap, magnitude / self.saturation_z)

        return ShadowSignal(
            strategy_name="ml_factor",
            asset=asset,
            direction=direction,
            confidence=confidence,
            reason=f"ml_factor_projection(magnitude={magnitude:.3f})",
            diagnostics={
                "latent_dim": self._latent_dim,
                "promoted_dims": list(self._promoted_dims),
                "weighted_sum": weighted_sum,
                "input_dim": self._input_dim,
            },
        )


def build_ml_factor_agents(assets: Tuple[str, ...] = ("BTC", "ETH", "SOL")) -> List[MLFactorFusionAgent]:
    """Construct ML factor agents for each asset. Failed loads return NEUTRAL
    on every evaluate() — fail-closed, no crash."""
    return [MLFactorFusionAgent(asset=a) for a in assets]


class MLFactorDispatcher:
    """Wrap per-asset MLFactorFusionAgent instances behind the
    `evaluate(asset, market_data)` interface that the existing
    MicrostructureShadowHarness expects.

    On asset mismatch (no agent for that asset), returns NEUTRAL.
    """
    def __init__(self, assets: Tuple[str, ...] = ("BTC", "ETH", "SOL")) -> None:
        self._agents: Dict[str, MLFactorFusionAgent] = {
            a: MLFactorFusionAgent(asset=a) for a in assets
        }

    def evaluate(self, asset: str, market_data: Dict[str, Any]) -> ShadowSignal:
        agent = self._agents.get(asset)
        if agent is None:
            return NEUTRAL("ml_factor", asset, f"no_agent_for_{asset}")
        return agent.evaluate(asset, market_data)

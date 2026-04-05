"""
================================================================================
HMATS P1.2 — DRL Shadow Diagnostics & Promotion Readiness
================================================================================
Purpose: Better shadow diagnostics for DRL so uncertainty / OOD / drift
         produce clear promotion-readiness signals.

Design:
  - DRL remains SHADOW in live. This module does NOT promote DRL.
  - Tracks shadow accuracy, directional agreement, confidence calibration
  - Produces a "promotion readiness score" for operator review
  - Observable in proof logs every tick

Integration:
  Called after DRL shadow inference, reads existing OOD/drift data.
  Logs diagnostics. Does not affect trading decisions.
================================================================================
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from collections import deque

logger = logging.getLogger("HMATS.DRLShadow")


@dataclass
class DRLShadowConfig:
    enabled: bool = False
    window_size: int = 50     # Rolling window for accuracy tracking
    min_observations: int = 10  # Minimum before producing readiness score


@dataclass
class ShadowObservation:
    timestamp: float
    asset: str
    drl_direction: float
    drl_confidence: float
    quant_direction: float
    realized_direction: float = 0.0  # Filled in at next tick
    ood_score: float = 0.0
    drift_weight: float = 1.0


class DRLShadowDiagnostics:
    """
    Track DRL shadow performance for promotion readiness.

    Output in proof log only — no trading impact.
    """

    def __init__(self, config: Optional[DRLShadowConfig] = None):
        self.config = config or DRLShadowConfig()
        self._observations: Dict[str, deque] = {}  # per-asset
        self._pending: Dict[str, Optional[ShadowObservation]] = {}  # last obs awaiting outcome

    def record_shadow_inference(
        self,
        asset: str,
        drl_direction: float,
        drl_confidence: float,
        quant_direction: float,
        ood_score: float = 0.0,
        drift_weight: float = 1.0,
    ) -> None:
        """Record DRL shadow output for this tick."""
        if not self.config.enabled:
            return

        if asset not in self._observations:
            self._observations[asset] = deque(maxlen=self.config.window_size)

        # Close previous pending observation with realized direction
        # (approximated as: did price move in the direction DRL predicted?)
        # This will be filled in by record_outcome() at next tick

        obs = ShadowObservation(
            timestamp=time.time(),
            asset=asset,
            drl_direction=drl_direction,
            drl_confidence=drl_confidence,
            quant_direction=quant_direction,
            ood_score=ood_score,
            drift_weight=drift_weight,
        )
        self._pending[asset] = obs

    def record_price_change(self, asset: str, price_change_pct: float) -> None:
        """Called at tick N+1 to fill in realized direction of tick N's prediction."""
        if not self.config.enabled:
            return
        pending = self._pending.get(asset)
        if pending is None:
            return
        pending.realized_direction = 1.0 if price_change_pct > 0 else (-1.0 if price_change_pct < 0 else 0.0)
        if asset not in self._observations:
            self._observations[asset] = deque(maxlen=self.config.window_size)
        self._observations[asset].append(pending)
        self._pending[asset] = None

    def get_readiness_report(self, asset: str) -> Dict[str, Any]:
        """Produce promotion readiness diagnostics for proof log."""
        if not self.config.enabled:
            return {"enabled": False}

        obs = self._observations.get(asset, deque())
        n = len(obs)

        if n < self.config.min_observations:
            return {
                "n_observations": n,
                "readiness": "INSUFFICIENT_DATA",
                "min_required": self.config.min_observations,
            }

        # Direction accuracy: how often DRL predicted price direction correctly
        correct = sum(1 for o in obs if o.drl_direction * o.realized_direction > 0)
        dir_accuracy = correct / n

        # Agreement with quant: how often DRL agrees with primary signal
        agree = sum(1 for o in obs if o.drl_direction * o.quant_direction > 0)
        quant_agreement = agree / n

        # Average confidence
        avg_conf = sum(o.drl_confidence for o in obs) / n

        # OOD frequency: how often OOD score was elevated
        ood_elevated = sum(1 for o in obs if o.ood_score > 10.0) / n

        # Promotion readiness score (0-1)
        # Needs: >55% accuracy, >40% quant agreement, <30% OOD, avg conf > 0.2
        score = 0.0
        if dir_accuracy > 0.55:
            score += 0.35
        elif dir_accuracy > 0.50:
            score += 0.15
        if quant_agreement > 0.40:
            score += 0.25
        if ood_elevated < 0.30:
            score += 0.20
        if avg_conf > 0.20:
            score += 0.20

        readiness = "NOT_READY"
        if score >= 0.80 and n >= 30:
            readiness = "READY"
        elif score >= 0.60:
            readiness = "APPROACHING"

        return {
            "n_observations": n,
            "direction_accuracy": round(dir_accuracy, 3),
            "quant_agreement": round(quant_agreement, 3),
            "avg_confidence": round(avg_conf, 3),
            "ood_elevated_pct": round(ood_elevated, 3),
            "readiness_score": round(score, 2),
            "readiness": readiness,
        }

    def to_dict(self) -> dict:
        return {
            "observations": {
                k: [
                    {
                        "ts": o.timestamp, "asset": o.asset,
                        "drl_dir": o.drl_direction, "drl_conf": o.drl_confidence,
                        "quant_dir": o.quant_direction, "realized_dir": o.realized_direction,
                        "ood": o.ood_score, "drift_w": o.drift_weight,
                    }
                    for o in v
                ]
                for k, v in self._observations.items()
            }
        }

    def from_dict(self, data: dict) -> None:
        if not data:
            return
        for k, obs_list in data.get("observations", {}).items():
            self._observations[k] = deque(maxlen=self.config.window_size)
            for o in obs_list:
                self._observations[k].append(ShadowObservation(
                    timestamp=o.get("ts", 0),
                    asset=o.get("asset", k),
                    drl_direction=o.get("drl_dir", 0),
                    drl_confidence=o.get("drl_conf", 0),
                    quant_direction=o.get("quant_dir", 0),
                    realized_direction=o.get("realized_dir", 0),
                    ood_score=o.get("ood", 0),
                    drift_weight=o.get("drift_w", 1.0),
                ))


# Singleton
_instance: Optional[DRLShadowDiagnostics] = None

def get_drl_shadow_diagnostics(config: Optional[DRLShadowConfig] = None) -> DRLShadowDiagnostics:
    global _instance
    if _instance is None:
        _instance = DRLShadowDiagnostics(config)
    return _instance

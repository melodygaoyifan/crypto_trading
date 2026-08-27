"""[P414c] Live jump-model regime SHADOW (observation-only, Iron Law 7).

The disciplined first step of the eventual GMM->jump regime swap (a P215
campaign). Offline evidence is GO: a jump penalty cuts regime CHURN vs the GMM
(~80% batch, ~59-66% online/filtered leak-free) with a vocabulary that maps
CLEAN to the control tables (P414c export). This harness runs the ONLINE
filtered jump regime alongside the live GMM every tick and logs its label + a
running churn comparison, so the cutover -- which changes what every
regime-conditional control reads (trend gate, smart beta, ADVISE weights,
kraken_quant buckets) -- is a validated, gated flip (P141), never a blind swap.

It changes NO live behaviour: it reads the GMM's own feature vector
(market_data['_gmm_raw_features'], stashed at the serve site for PARITY -- no
train/serve skew) + the GMM's live regime, runs a forward-only DP filter with
the split-aware exported centroids (data <= t, the honest live label), and
writes only a log line. The filter cost vector is PERSISTED so the regime stays
warm across a restart (P301) instead of cold-starting each deploy.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_STATE_VERSION = "jumpregime_state_v1"


class JumpRegimeShadow:
    def __init__(self, data_dir: str = "data",
                 repo_root: Optional[Path] = None):
        root = repo_root or Path(__file__).resolve().parent.parent
        self._models: Dict[str, dict] = {}
        self._cost: Dict[str, List[float]] = {}       # forward-DP cost vector
        self._last_label: Dict[str, int] = {}
        self._jsw: Dict[str, int] = {}                # rolling switch counters
        self._gsw: Dict[str, int] = {}
        self._nseen: Dict[str, int] = {}
        self._last_gmm: Dict[str, str] = {}
        self._state_path = Path(data_dir) / "jumpregime_state.json"
        cfg_dir = root / "configs" / "jumpregime"
        if cfg_dir.exists():
            for p in cfg_dir.glob("*.json"):
                try:
                    m = json.loads(p.read_text(encoding="utf-8"))
                    need = ("centroids", "scaler_mean", "scaler_std",
                            "state_to_name", "lambda")
                    if all(k in m for k in need):
                        self._models[m.get("asset", p.stem)] = m
                    else:
                        logger.warning(f"[JUMP-REGIME] {p.name}: incomplete "
                                       f"artifact — skipped")
                except Exception as e:  # noqa: silent-swallow — a bad artifact must not break construction
                    logger.warning(f"[JUMP-REGIME] {p.name}: unreadable "
                                   f"({type(e).__name__})")
        if self._models:
            logger.info(f"[JUMP-REGIME] loaded {sorted(self._models)} "
                        f"(P414c shadow; observation-only)")
        self._restore_state()

    # ---------------- persistence (P301: keep the filter warm) ----------
    def _restore_state(self) -> None:
        try:
            if not self._state_path.exists():
                return
            pay = json.loads(self._state_path.read_text(encoding="utf-8"))
            if pay.get("v") != _STATE_VERSION:
                return
            for a, st in (pay.get("assets") or {}).items():
                c = st.get("cost")
                if isinstance(c, list) and a in self._models \
                        and len(c) == len(self._models[a]["centroids"]):
                    self._cost[a] = [float(x) for x in c]
                    self._last_label[a] = int(st.get("last_label", -1))
            if self._cost:
                logger.info(f"[JUMP-REGIME] restored filter state "
                            f"{sorted(self._cost)} — warm across the restart")
        except Exception as e:  # noqa: silent-swallow — corrupt state = cold start
            logger.warning(f"[JUMP-REGIME] state unreadable "
                           f"({type(e).__name__}) — cold start")

    def _persist_state(self) -> None:
        if getattr(self, "_state_path", None) is None:
            return
        try:
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "v": _STATE_VERSION, "saved_ts": time.time(),
                "assets": {a: {"cost": self._cost.get(a),
                               "last_label": self._last_label.get(a, -1)}
                           for a in self._cost}}), encoding="utf-8")
            tmp.replace(self._state_path)
        except Exception as e:  # noqa: silent-swallow — persist failure must not break the tick
            logger.warning(f"[JUMP-REGIME] persist failed ({type(e).__name__})")

    # ---------------- pure online filter step ----------------
    @staticmethod
    def _sqdist(scaled: List[float], centroid: List[float]) -> float:
        return sum((scaled[i] - centroid[i]) ** 2 for i in range(len(centroid)))

    def step(self, asset: str, raw_features: List[float]):
        """Forward-only DP filter (data <= t): returns (label, name, switched).
        Uses ONLY the current features + the persisted cost vector — the honest
        live label a control would read, not a batch-smoothed one."""
        m = self._models[asset]
        mean, std = m["scaler_mean"], m["scaler_std"]
        cents = m["centroids"]
        k = len(cents)
        lam = float(m["lambda"])
        scaled = [(float(raw_features[i]) - mean[i]) / (std[i] or 1e-9)
                  for i in range(len(mean))]
        d = [self._sqdist(scaled, cents[j]) for j in range(k)]
        prev = self._cost.get(asset)
        if prev is None or len(prev) != k:
            cost = list(d)
        else:
            cost = [d[j] + min(prev[s] + (0.0 if s == j else lam)
                               for s in range(k)) for j in range(k)]
        label = min(range(k), key=lambda j: cost[j])
        self._cost[asset] = cost
        switched = self._last_label.get(asset, label) != label
        self._last_label[asset] = label
        name = m["state_to_name"].get(str(label), f"JUMP_{label}")
        return label, name, switched

    # ---------------- tick ----------------
    def tick(self, features_by_asset: Dict[str, list],
             gmm_regime_by_asset: Dict[str, str]) -> list:
        summary = []
        for asset in self._models:
            try:
                feats = features_by_asset.get(asset)
                if not feats:
                    continue
                _, name, switched = self.step(asset, list(feats))
                # churn accounting vs the GMM's live regime
                gmm = gmm_regime_by_asset.get(asset)
                self._nseen[asset] = self._nseen.get(asset, 0) + 1
                if switched:
                    self._jsw[asset] = self._jsw.get(asset, 0) + 1
                if gmm is not None and self._last_gmm.get(asset) not in (None, gmm):
                    self._gsw[asset] = self._gsw.get(asset, 0) + 1
                if gmm is not None:
                    self._last_gmm[asset] = gmm
                n = self._nseen[asset]
                js, gs = self._jsw.get(asset, 0), self._gsw.get(asset, 0)
                summary.append(f"{asset}={name}"
                               + ("*" if switched else "")
                               + f" (jump {js}/{n} vs gmm {gs}/{n} switches)")
            except Exception as e:  # noqa: silent-swallow — per-asset fail-soft, observation-only
                summary.append(f"{asset}=ERR({type(e).__name__})")
        if summary:
            logger.info("[JUMP-REGIME] " + " | ".join(summary))
        self._persist_state()
        return summary

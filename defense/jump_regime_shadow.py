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
from datetime import datetime, timezone
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
        # [P420] the P414c cutover needs a LEDGER, not a log line: one row per
        # asset per decision tick, the same strategy_shadow/ convention the
        # P166 scorer reads. Observation-only (Iron Law 7).
        self._ledger_dir = Path(data_dir) / "strategy_shadow"
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
                    # [P420] the churn counters were RAM-only, so every
                    # restart (7 today) zeroed the "jump X/N vs gmm Y/N"
                    # comparison the shadow exists to accumulate. Restored
                    # beside the cost vector; absent keys (a pre-P420 file)
                    # restore as 0 = cold counters, never a fabricated
                    # history.
                    for _k, _d in (("nseen", self._nseen), ("jsw", self._jsw),
                                   ("gsw", self._gsw)):
                        try:
                            _d[a] = int(st.get(_k, 0) or 0)
                        except (TypeError, ValueError):  # noqa: silent-swallow — a bad counter restores as 0 (logged below)
                            _d[a] = 0
                            logger.warning(f"[JUMP-REGIME] {a}: counter "
                                           f"{_k!r} unreadable — restored as 0")
                    _lg = st.get("last_gmm")
                    if isinstance(_lg, str) and _lg:
                        self._last_gmm[a] = _lg
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
                               "last_label": self._last_label.get(a, -1),
                               # [P420] counters ride the same atomic write
                               "nseen": self._nseen.get(a, 0),
                               "jsw": self._jsw.get(a, 0),
                               "gsw": self._gsw.get(a, 0),
                               "last_gmm": self._last_gmm.get(a)}
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
    def _append_ledger_row(self, asset: str, row: dict) -> None:
        """[P420] One JSONL row per asset per decision tick — the evidence
        the P414c cutover can be judged on. Never raises: a ledger that
        cannot be written must not break the tick (logged, observation-only).
        """
        ledger_dir = getattr(self, "_ledger_dir", None)
        if ledger_dir is None:
            return
        try:
            ledger_dir.mkdir(parents=True, exist_ok=True)
            with (ledger_dir / f"jumpregime_{asset}.jsonl").open(
                    "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except Exception as e:  # noqa: silent-swallow — logged; ledger write failure must not break the tick
            logger.warning(f"[JUMP-REGIME] {asset}: ledger append failed "
                           f"({type(e).__name__}: {e})")

    def tick(self, features_by_asset: Dict[str, list],
             gmm_regime_by_asset: Dict[str, str],
             fallback_by_asset: Optional[Dict[str, object]] = None) -> list:
        """[P420] `fallback_by_asset[asset]` truthy (the pipeline's
        `market_data['_gmm_fallback']`, e.g. the OOD ADX proxy) SKIPS the
        asset: its "gmm label" is then the ADX proxy, and counting a proxy
        label as a GMM switch would corrupt the churn comparison. The
        pipeline also withholds the feature stash on that path, so the skip
        holds even when the caller does not pass the map."""
        summary = []
        now = time.time()
        for asset in self._models:
            try:
                feats = features_by_asset.get(asset)
                if not feats:
                    continue
                if fallback_by_asset and fallback_by_asset.get(asset):
                    summary.append(f"{asset}=SKIP(gmm_fallback="
                                   f"{fallback_by_asset.get(asset)})")
                    continue
                label, name, switched = self.step(asset, list(feats))
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
                self._append_ledger_row(asset, {
                    "ts": now,
                    "iso": datetime.fromtimestamp(
                        now, tz=timezone.utc).isoformat(),
                    "asset": asset,
                    "strategy": "jumpregime",
                    "jump_label": int(label),
                    "jump_name": name,
                    "jump_switched": bool(switched),
                    "gmm_label": gmm,
                    "jump_switches": js,
                    "gmm_switches": gs,
                    "n": n,
                })
            except Exception as e:  # noqa: silent-swallow — per-asset fail-soft, observation-only
                summary.append(f"{asset}=ERR({type(e).__name__})")
        if summary:
            logger.info("[JUMP-REGIME] " + " | ".join(summary))
        self._persist_state()
        return summary

"""[P409] Forward shadow of the held BTC ridge (observation-only, Iron Law 7).

The trained-forecaster question, resolved per-asset instead of pooled (operator:
"we don't have to trade what isn't tradeable"). On the flat CDE fee a per-bar
forecaster is dead (P385), but HELD (deadband on the trailing-z of the
prediction, band 1.0, ~59 flips/yr -- the operator's "hold longer" lever, P386)
a small BTC ridge CLEARS walk-forward: 4-fold sum +46% vs buy-hold +9%, steadier
on 8 features than 137 (P409). ETH/SOL fail even held and are NOT built.

This harness records, every 4H tick, the direction the held ridge would hold, to
data/strategy_shadow/ridgeshadow_BTC.jsonl, so compute_shadow_ic scores it on
FORWARD returns against the P166 cost-aware gate. It changes NO live position --
a seat is a P141 decision taken only if the forward read certifies.

Deterministic (ridge = closed form) so it cannot die of seed fragility the way
the withdrawn mlpshadow did (P285c). It REPRODUCES the validated recipe exactly:
a CAUSAL trailing z (rolling z_window, min z_min) of the prediction -- NOT a
fixed sig, which would be a different unvalidated signal (P164/P214 skew). The
prediction buffer is PERSISTED so a restart keeps the z warm instead of
re-warming ~17 days each time (the P301 lesson); rows written before the buffer
fills carry warmup_transient so the exam ledger stays honest (P287). confidence =
|direction| (P224/P236), so a held flat contributes zero, never a saturated 1.0.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

# [P310] SINGLE SOURCE for the `strategy` field this module writes. A consumer
# (analytics/shadow_ic) must classify this exact name, never restate a prefix.
SHADOW_STRATEGY_NAMES = frozenset({"ridgeshadow"})

logger = logging.getLogger(__name__)

_STATE_VERSION = "ridgeshadow_state_v1"


class RidgeShadow:
    def __init__(self, data_dir: str = "data",
                 repo_root: Optional[Path] = None):
        root = repo_root or Path(__file__).resolve().parent.parent
        self._dir = Path(data_dir) / "strategy_shadow"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._state_path = Path(data_dir) / "ridgeshadow_state.json"
        self._models: Dict[str, dict] = {}
        # asset -> {cur: float, buffer: [float,...]} (buffer = trailing preds)
        self._state: Dict[str, dict] = {}
        self._transient: Dict[str, bool] = {}   # True while z is still warming
        self._warned: Dict[str, str] = {}
        self._last_records: Dict[str, dict] = {}   # seat feed: full-coverage,
        cfg_dir = root / "configs" / "ridgeshadow"                # non-warmup
        if cfg_dir.exists():
            for p in cfg_dir.glob("*.json"):
                try:
                    m = json.loads(p.read_text(encoding="utf-8"))
                    need = ("feature_names", "scaler_mean", "scaler_scale",
                            "coef", "intercept", "deadband", "z_window",
                            "z_min")
                    missing = [k for k in need if k not in m]
                    if missing:
                        logger.warning(f"[RIDGE-SHADOW] {p.name}: export "
                                       f"missing {missing} — leg silent")
                        continue
                    self._models[m.get("asset", p.stem)] = m
                except Exception as e:
                    logger.warning(f"[RIDGE-SHADOW] {p.name}: unreadable "
                                   f"({type(e).__name__}) — leg silent")
        if self._models:
            logger.info(f"[RIDGE-SHADOW] loaded: {sorted(self._models)} "
                        f"(P409 held-ridge, BTC-only)")
        self._restore_state()

    # ---------------- state persistence (P301: keep the z warm) ----------
    def _restore_state(self) -> None:
        try:
            if not self._state_path.exists():
                return
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if payload.get("v") != _STATE_VERSION:
                logger.warning(f"[RIDGE-SHADOW] state version mismatch "
                               f"({payload.get('v')}) — cold start")
                return
            for asset, st in (payload.get("assets") or {}).items():
                buf = [float(x) for x in (st.get("buffer") or [])]
                self._state[asset] = {"cur": float(st.get("cur", 0.0) or 0.0),
                                      "buffer": buf}
                z_min = int(self._models.get(asset, {}).get("z_min", 100))
                self._transient[asset] = len(buf) < z_min
            if self._state:
                logger.info(f"[RIDGE-SHADOW] restored {sorted(self._state)} "
                            f"— trailing-z stays warm across the restart (P301)")
        except Exception as e:  # noqa: silent-swallow — corrupt state = cold start, warned; warmup rows are transient-marked
            logger.warning(f"[RIDGE-SHADOW] state unreadable "
                           f"({type(e).__name__}) — cold start; rows carry "
                           f"warmup_transient until the z-buffer fills")

    def _persist_state(self) -> None:
        if getattr(self, "_state_path", None) is None:   # P85 fixture defense
            return
        try:
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "v": _STATE_VERSION, "saved_ts": time.time(),
                "assets": self._state}), encoding="utf-8")
            tmp.replace(self._state_path)
        except Exception as e:  # noqa: silent-swallow — persist failure must not break the tick
            logger.warning(f"[RIDGE-SHADOW] state persist failed "
                           f"({type(e).__name__}) — restart re-warms the z")

    # ---------------- pure model math ----------------
    @staticmethod
    def forward(m: dict, x: list) -> float:
        """Ridge in stdlib: scale then dot(coef) + intercept -> raw pred."""
        coef = m["coef"]
        pred = float(m["intercept"])
        for i in range(len(coef)):
            sc = m["scaler_scale"][i] or 1e-12
            pred += ((x[i] - m["scaler_mean"][i]) / sc) * coef[i]
        return pred

    def decide(self, asset: str, raw_pred: float):
        """Append the prediction, compute the CAUSAL trailing z, and apply the
        deadband-hold. Returns (cur, z_or_None, warmup_bool). No DI gate — the
        hold comes from the deadband (the validated P386 recipe)."""
        m = self._models[asset]
        st = self._state.setdefault(asset, {"cur": 0.0, "buffer": []})
        buf = st["buffer"]
        buf.append(float(raw_pred))
        z_win = int(m["z_window"])
        if len(buf) > z_win:
            del buf[:len(buf) - z_win]   # keep only the trailing window
        z_min = int(m["z_min"])
        if len(buf) < z_min:
            self._transient[asset] = True
            self._persist_state()
            return st["cur"], None, True
        mu = sum(buf) / len(buf)
        var = sum((v - mu) ** 2 for v in buf) / len(buf)
        sd = var ** 0.5 or 1e-9
        z = (float(raw_pred) - mu) / sd
        band = float(m["deadband"])
        if z > band:
            st["cur"] = 1.0
        elif z < -band:
            st["cur"] = -1.0
        # else: HOLD st["cur"] (the deadband-hold — the "hold longer" lever)
        self._transient[asset] = False
        self._persist_state()
        return st["cur"], z, False

    # ---------------- tick ----------------
    def tick(self, features_by_asset: Dict[str, dict],
             presence_by_asset: Dict[str, dict]) -> list:
        summary = []
        for asset, m in self._models.items():
            try:
                feats = features_by_asset.get(asset) or {}
                pres = presence_by_asset.get(asset) or {}
                missing = [nm for nm in m["feature_names"] if not pres.get(nm)]
                if missing:
                    self._write(asset, 0.0, None, warmup=False,
                                coverage_note=f"missing:{missing[:6]}")
                    summary.append(f"{asset}=FLAT(cov-{len(missing)})")
                    self._transition_log(
                        f"cov:{asset}",
                        f"[RIDGE-SHADOW] {asset}: {len(missing)}/"
                        f"{len(m['feature_names'])} features uncovered "
                        f"({missing[:6]}...) — recording flat (P248 parity)")
                    continue
                x = [float(feats[nm]) for nm in m["feature_names"]]
                raw = self.forward(m, x)
                cur, z, warmup = self.decide(asset, raw)
                d = 1.0 if cur >= 0.5 else (-1.0 if cur <= -0.5 else 0.0)
                self._write(asset, d, z, warmup=warmup)
                if not warmup:   # seat feed: never a warming flat (P2)
                    self._last_records[asset] = {
                        "direction": d, "z": z, "ts": time.time()}
                summary.append(f"{asset}={d:+.0f}"
                               + (f"(z={z:+.2f})" if z is not None
                                  else "(warmup)"))
            except Exception as e:  # noqa: silent-swallow — per-asset fail-soft
                summary.append(f"{asset}=ERR({type(e).__name__})")
        if summary:
            logger.info("[RIDGE-SHADOW] " + " | ".join(summary))
        return summary

    def last_direction(self, asset: str, max_age_s: float = 6 * 3600):
        """Most recent full-coverage, non-warmup emit, or None (P285 seat
        contract, P156 staleness bound). None = NO SEAT INPUT, never flat."""
        rec = getattr(self, "_last_records", {}).get(asset)
        if not rec:
            return None
        age = time.time() - float(rec.get("ts", 0.0) or 0.0)
        if age > max_age_s:
            return None
        return (float(rec.get("direction", 0.0) or 0.0), rec.get("z"), age)

    def _write(self, asset: str, direction: float, z, warmup: bool,
               coverage_note=None) -> dict:
        rec = {"ts": time.time(),
               "iso": datetime.now(timezone.utc).isoformat(),
               "strategy": "ridgeshadow", "asset": asset,
               "direction": float(direction),
               "confidence": abs(float(direction)),   # P224: flat -> 0
               "z": None if z is None else round(float(z), 4),
               "coverage_note": coverage_note,
               "warmup_transient": bool(warmup)}
        with open(self._dir / f"ridgeshadow_{asset}.jsonl", "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def _transition_log(self, key: str, msg: str) -> None:
        if self._warned.get(key) != msg:
            self._warned[key] = msg
            logger.warning(msg)

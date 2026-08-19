"""[P284] BTC mlp_small Rung-3 SHADOW — the pooled-certified candidate's
30-day live forward exam (OBSERVATION-ONLY, Iron Law 7).

P283b certified mlp_small on BTC under the LIVE derivatives expression
(pooled +165.8%, Sharpe 1.52, CI[0.33,2.77], DSR 0.90) — with the recorded
caveats (multiply-read windows, single seed) that make the FORWARD ledger
the only remaining exam. This harness records, every tick, the direction
the certified model would trade, through the P166 gate at ~09-15.

PARITY DISCIPLINE (what killed the SOL ridge, done right here):
- The model is a JSON weight export (training/scripts/export_mlp_shadow.py
  — the weekly re-run IS the candidate's refit cadence; no pickle, P5).
- Features are SINGLE-SOURCED: 22 of 24 from the runtime obs builder's raw
  pre-scaling stash (the exact live computation the DRL obs uses, P172),
  the 2 fv2 members from the live BinanceFlowFeed. **100% coverage of all
  24 names or the tick records flat with the missing names** — a partial
  vector is a different model (P248's coverage_note rule).
- Decision cadence: the certified DI=4 (16h) via epoch bins
  (floor(ts/4h) % 4 == 0); between decision bins the HELD direction is
  recorded, exactly like positions_from_z holds `last`. The BIN arithmetic
  is deterministic across restarts; the HELD position is NOT — it is
  therefore PERSISTED (data/mlpshadow_state.json, P287, P154 pattern):
  pre-P287 every deploy reset `cur` to 0.0 and wrote up to ~12h of
  phantom-flat rows into this exam ledger as real claims. Rows emitted
  before persisted state is available carry `restart_transient: true` so
  the scorer/review can filter them. Known honest caveats carried in every
  row: the live funding_rate_zscore is the runtime rolling-z (a P214-class
  formulation skew vs the parquet's daily z), and epoch bins need not
  align with the training folds' bar-index phase — the ledger judges the
  live composite, which is the point.
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

# [P310] SINGLE SOURCE for the names this module writes into a record's
# `strategy` field. Consumers (analytics/shadow_ic) must not restate
# them: P309 keyed its allowlists on LEDGER-FILE PREFIXES instead, so
# two families were silently never pooled and an archive section never
# rendered. A conformance test asserts every consumer name is one of
# these, and that every one of these is classified by a consumer.
SHADOW_STRATEGY_NAMES = frozenset({"mlpshadow"})


logger = logging.getLogger(__name__)

BAR_SEC = 4 * 3600
_STATE_VERSION = "mlpshadow_state_v1"


class MlpShadow:
    def __init__(self, data_dir: str = "data",
                 repo_root: Optional[Path] = None):
        root = repo_root or Path(__file__).resolve().parent.parent
        self._dir = Path(data_dir) / "strategy_shadow"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._state_path = Path(data_dir) / "mlpshadow_state.json"
        self._models: Dict[str, dict] = {}
        self._state: Dict[str, dict] = {}   # asset -> {cur, last_bin}
        # [P287] asset -> True while the held position is a cold-start
        # fabrication rather than a decided/restored value; cleared on the
        # first decision bin or on a successful restore. Rows written while
        # True are stamped restart_transient so the ledger stays honest.
        self._transient: Dict[str, bool] = {}
        self._warned: Dict[str, str] = {}
        # [P285] seat feed: the last FULL-COVERAGE emit per asset. A
        # coverage-gap tick must NEVER refresh this — the seat treating
        # "cannot compute" as a fresh flat is the P2 absence!=flat trap.
        self._last_records: Dict[str, dict] = {}
        cfg_dir = root / "configs" / "mlpshadow"
        if cfg_dir.exists():
            for p in cfg_dir.glob("*.json"):
                try:
                    m = json.loads(p.read_text(encoding="utf-8"))
                    need = ("feature_names", "scaler_mean", "scaler_scale",
                            "w1", "b1", "w2", "b2", "sig", "deadband",
                            "decision_interval")
                    missing = [k for k in need if k not in m]
                    if missing:
                        logger.warning(f"[MLP-SHADOW] {p.name}: export "
                                       f"missing {missing} — leg silent")
                        continue
                    self._models[m.get("asset", p.stem)] = m
                except Exception as e:
                    logger.warning(f"[MLP-SHADOW] {p.name}: unreadable "
                                   f"({type(e).__name__}) — leg silent")
        if self._models:
            logger.info(f"[MLP-SHADOW] loaded: "
                        f"{sorted(self._models)} (P283b certified)")
        self._restore_state()

    # ---------------- [P287] held-position persistence ----------------
    def _restore_state(self) -> None:
        """Restore per-asset {cur, last_bin} (P154 pattern). Corrupt or
        version-mismatched files degrade to a cold start with a warning —
        and cold-start assets stay marked transient until their first
        decision bin, so the ledger never claims a fabricated flat as a
        held position."""
        try:
            if not self._state_path.exists():
                return
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if payload.get("v") != _STATE_VERSION:
                logger.warning(f"[MLP-SHADOW] state version mismatch "
                               f"({payload.get('v')}) — cold start")
                return
            age_s = time.time() - float(payload.get("saved_ts", 0.0) or 0.0)
            for asset, st in (payload.get("assets") or {}).items():
                cur = float(st.get("cur", 0.0) or 0.0)
                lb = st.get("last_bin")
                self._state[asset] = {
                    "cur": cur,
                    "last_bin": None if lb is None else int(lb)}
                self._transient[asset] = False
            if self._state:
                logger.info(f"[MLP-SHADOW] held positions restored for "
                            f"{sorted(self._state)} (saved {age_s/3600.0:.1f}h "
                            f"ago) — restarts no longer fabricate flat rows "
                            f"(P287)")
        except Exception as e:  # noqa: silent-swallow — corrupt state = cold start, warned; the exam ledger marks the transient rows
            logger.warning(f"[MLP-SHADOW] state file unreadable "
                           f"({type(e).__name__}) — cold start; rows carry "
                           f"restart_transient until the next decision bin")

    def _persist_state(self) -> None:
        # [P85] defended: test fixtures (and any partial construction)
        # build this object without __init__ — a missing path must degrade
        # to no-persist, never to an ERR row in the exam ledger
        if getattr(self, "_state_path", None) is None:
            return
        try:
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "v": _STATE_VERSION, "saved_ts": time.time(),
                "assets": self._state}), encoding="utf-8")
            tmp.replace(self._state_path)
        except Exception as e:  # noqa: silent-swallow — persistence failure must not break the tick; the next restart is transient-marked anyway
            logger.warning(f"[MLP-SHADOW] state persist failed "
                           f"({type(e).__name__}) — a restart before the "
                           f"next decision bin will record transient rows")

    # ---------------- pure model math ----------------
    @staticmethod
    def forward(m: dict, x: list) -> float:
        """The certified MLP in stdlib math: scale -> relu(W1x+b1) ->
        W2h+b2 -> /sig. Deterministic, no sklearn dependency at serve."""
        xs = [(xi - mu) / (sc or 1e-12) for xi, mu, sc in
              zip(x, m["scaler_mean"], m["scaler_scale"])]
        h = []
        for j in range(len(m["b1"])):
            a = m["b1"][j] + sum(xs[i] * m["w1"][i][j]
                                 for i in range(len(xs)))
            h.append(a if a > 0 else 0.0)
        pred = m["b2"] + sum(h[j] * m["w2"][j] for j in range(len(h)))
        return pred / (m["sig"] or 1e-9)

    def decide(self, asset: str, z: float, now_ts: float) -> float:
        """DI-cadence position: decide on decision bins, hold between —
        the live replica of positions_from_z's `last` semantics."""
        m = self._models[asset]
        st = self._state.setdefault(asset, {"cur": 0.0, "last_bin": None})
        bin_idx = int(now_ts // BAR_SEC)
        if bin_idx % int(m["decision_interval"]) == 0 \
                and st["last_bin"] != bin_idx:
            st["last_bin"] = bin_idx
            st["cur"] = (0.0 if abs(z) < float(m["deadband"])
                         else max(-1.0, min(1.0, z)))
            # [P287] a decided value is never transient, and it must
            # survive the next restart (once per 16h per asset — cheap).
            # [P85] setdefault-style defense: partially-constructed
            # instances (test fixtures) lack the map.
            if not hasattr(self, "_transient"):
                self._transient = {}
            self._transient[asset] = False
            self._persist_state()
        return st["cur"]

    # ---------------- tick ----------------
    def tick(self, features_by_asset: Dict[str, dict],
             presence_by_asset: Dict[str, dict]) -> list:
        summary = []
        for asset, m in self._models.items():
            try:
                feats = features_by_asset.get(asset) or {}
                pres = presence_by_asset.get(asset) or {}
                missing = [nm for nm in m["feature_names"]
                           if not pres.get(nm)]
                if missing:
                    rec = self._write(asset, 0.0, None,
                                      coverage_note=f"missing:{missing[:6]}")
                    summary.append(f"{asset}=FLAT(cov-{len(missing)})")
                    self._transition_log(
                        f"cov:{asset}",
                        f"[MLP-SHADOW] {asset}: {len(missing)}/24 features "
                        f"uncovered ({missing[:6]}...) — recording flat; "
                        f"this is the parity work-list (P248 rule)")
                    continue
                x = [float(feats[nm]) for nm in m["feature_names"]]
                z = self.forward(m, x)
                # LIVE expression: the certified decisive cell is the
                # sign-quantized mapping — the ledger claims what the
                # sleeve would monetize
                cont = self.decide(asset, z, time.time())
                d = (1.0 if cont >= 0.15 else
                     (-1.0 if cont <= -0.15 else 0.0))
                self._write(asset, d, z, cont=cont)
                # [P285] seat feed — full-coverage emits only (the coverage
                # branch above `continue`s past this line by construction)
                if not hasattr(self, "_last_records"):
                    self._last_records = {}
                self._last_records[asset] = {
                    "direction": d, "z": z, "ts": time.time()}
                summary.append(f"{asset}={d:+.0f}(z={z:+.2f})")
            except Exception as e:  # noqa: silent-swallow — per-asset fail-soft; summary shows the miss
                summary.append(f"{asset}=ERR({type(e).__name__})")
        if summary:
            logger.info("[MLP-SHADOW] " + " | ".join(summary))
        return summary

    def last_direction(self, asset: str, max_age_s: float = 6 * 3600):
        """[P285] The model's most recent FULL-COVERAGE emit, or None.

        Returns ``(direction, z, age_s)`` when a full-coverage record exists
        and is younger than ``max_age_s`` (6h = one 4H tick + slack, the
        P156 staleness-bound rule). None means "no fresh opinion" — the
        seat consumer must treat that as NO SEAT INPUT, never as flat (P2):
        the model's deadband flat IS a position; the model being absent or
        coverage-gapped is not. Between DI=4 decision bins the direction is
        the HELD one — exactly what the certified expression trades.

        One-tick lag by construction: tick() runs at loop level after
        decide, so a seat consumer inside decide reads the previous loop's
        emit — immaterial at the 16h decision horizon, but a fact.
        """
        rec = getattr(self, "_last_records", {}).get(asset)
        if not rec:
            return None
        age = time.time() - float(rec.get("ts", 0.0) or 0.0)
        if age > max_age_s:
            return None
        return (float(rec.get("direction", 0.0) or 0.0),
                rec.get("z"), age)

    def _write(self, asset: str, direction: float, z, cont=None,
               coverage_note=None) -> dict:
        rec = {"ts": time.time(),
               "iso": datetime.now(timezone.utc).isoformat(),
               "strategy": "mlpshadow", "asset": asset,
               "direction": float(direction),
               "confidence": abs(float(direction)),
               "z": None if z is None else round(float(z), 4),
               "cont_pos": None if cont is None else round(float(cont), 4),
               "coverage_note": coverage_note,
               # [P287] True while the held position is a cold-start
               # fabrication (no restored state, no decision bin yet) —
               # phantom-flat rows are filterable, not silent claims.
               # [P85] getattr-defended for partially-constructed instances.
               "restart_transient": bool(
                   getattr(self, "_transient", {}).get(asset, True))}
        with open(self._dir / f"mlpshadow_{asset}.jsonl", "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def _transition_log(self, key: str, msg: str) -> None:
        if self._warned.get(key) != msg:
            self._warned[key] = msg
            logger.warning(msg)

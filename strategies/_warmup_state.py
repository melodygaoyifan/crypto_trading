"""
================================================================================
HMATS [P301] - persisted warmup history for the v5.1 shadow strategies
================================================================================

WHY THIS EXISTS
    P199 recorded that 6 of 9 v5.1 strategies scored KILL with
    `n_directional = 0`, and P204 demoted the blend on that verdict. P300 then
    read the per-row `reason` fields on the server and found the silence has
    two different causes wearing one symptom:

        microstructure : z_below_threshold(+0.00) / no_prev_price   <- starved
        cascade        : quiet(composite=0.00)                      <- starved
        funding        : history_warmup(1/12), regime_warmup(1/30)  <- THIS

    Those counters read **1**, not 11 or 29. They are per-process `deque`s
    that start empty on every construction, so the warmup restarts at 1 with
    every deploy. At 3 funding observations per day a 12-observation warmup
    needs ~4 days of uninterrupted uptime and the 30-observation one needs
    ~10 days - and the 2026-08-08 operator note already recorded that this
    engine has never had 20 hours. The strategy is not quiet; it has never
    been allowed to finish waking up.

    Same class as P154 (CryptoPanic rate-limit state), P148 (DRL frame
    buffer), P150 (sleeve drawdown baseline) and P293b's own summary: "a
    limiter that re-arms on restart is not a limiter". Here: a warmup that
    re-arms on restart is not a warmup, it is a permanent NEUTRAL.

WHAT IT DOES NOT DO
    It does not make the strategy fire, and it is not evidence that the
    strategy is any good - `v5_1_strategies_live` stays false. It removes the
    reason its ledger has been empty, so the P166 gate can eventually judge
    something other than an artifact of the deploy cadence.

FAIL DIRECTIONS
    * A missing, unreadable or malformed state file restores NOTHING and the
      strategy warms up exactly as it does today. Never a fabricated history:
      a synthetic distribution would give the z-score a scale nobody measured
      (P2/P199).
    * Observations carry timestamps and anything older than `max_age_sec` is
      DROPPED on restore. A funding distribution from two months ago is not
      the current regime, and silently treating it as such would be worse
      than warming up again.
    * Writes are atomic (os.replace) so a crash mid-write cannot leave a
      truncated history that parses.
================================================================================
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("HMATS.WarmupState")

_STATE_VERSION = "v5_1_warmup_v1"
DEFAULT_MAX_AGE_SEC = 30 * 24 * 3600.0     # 30 days


def _state_dir() -> str:
    return os.path.join(os.environ.get("HMATS_DATA_DIR", "data"), "v5_1_warmup")


def state_path(name: str) -> str:
    return os.path.join(_state_dir(), f"{name}.json")


def save(name: str, history: Dict[str, "object"]) -> bool:
    """Persist {asset: iterable_of_values} with a write timestamp.

    Returns True on success. Never raises: a warmup that cannot be saved must
    not take a tick down.
    """
    try:
        os.makedirs(_state_dir(), exist_ok=True)
        now = time.time()
        payload = {
            "version": _STATE_VERSION,
            "saved_ts": now,
            "series": {
                str(a): [float(v) for v in list(vals)]
                for a, vals in (history or {}).items()
            },
        }
        d = _state_dir()
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, state_path(name))
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:  # noqa: silent-swallow — tmp cleanup only
                    pass
        return True
    except Exception as e:  # noqa: silent-swallow — logged
        logger.warning("[WARMUP] %s: save failed (%s: %s) — the warmup will "
                       "restart at 1 after the next restart", name,
                       type(e).__name__, e)
        return False


def load(name: str,
         max_age_sec: float = DEFAULT_MAX_AGE_SEC) -> Dict[str, List[float]]:
    """Restore {asset: [values]}, or {} when unavailable or too old.

    {} means "warm up from scratch", which is exactly today's behaviour - so
    every failure path here is a no-op rather than a risk.
    """
    p = state_path(name)
    try:
        if not os.path.exists(p):
            return {}
        with open(p, encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("version") != _STATE_VERSION:
            logger.info("[WARMUP] %s: state version mismatch — warming up "
                        "from scratch", name)
            return {}
        age = time.time() - float(payload.get("saved_ts") or 0.0)
        if age > max_age_sec:
            logger.info("[WARMUP] %s: saved history is %.1f days old (> %.1f) "
                        "— dropped; a stale distribution is not the current "
                        "regime", name, age / 86400.0, max_age_sec / 86400.0)
            return {}
        out: Dict[str, List[float]] = {}
        for a, vals in (payload.get("series") or {}).items():
            try:
                clean = [float(v) for v in vals
                         if isinstance(v, (int, float)) and float(v) == float(v)]
            except (TypeError, ValueError):  # noqa: silent-swallow — row dropped
                continue
            if clean:
                out[str(a)] = clean
        if out:
            logger.info("[WARMUP] %s: restored %s", name,
                        ", ".join(f"{a}={len(v)}" for a, v in sorted(out.items())))
        return out
    except Exception as e:  # noqa: silent-swallow — logged
        logger.warning("[WARMUP] %s: restore failed (%s: %s) — warming up "
                       "from scratch", name, type(e).__name__, e)
        return {}

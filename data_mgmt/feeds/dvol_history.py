"""
================================================================================
HMATS [P306] - DVOL history, so `market_data["dvol"]` can be a Z-SCORE
================================================================================

WHY THIS EXISTS
    P265d found that `market_data["dvol"]` has never had a producer while the
    P0 guard reading it force-flattens the book. P293 shipped the Deribit feed
    that CAN produce it. P298 then tried to enable `dvol_to_market_data` and
    found the units bug that made enabling it dangerous:

        defense/constitution.py:81   "dvol": "dvol_zscore"      <- alias
        defense/constitution.py:410  dvol_zscore >= 5.0  ->  EXTREME_DVOL

    Deribit publishes DVOL as an INDEX LEVEL (BTC ~34, ETH ~46), not a
    z-score. Publishing the level under a key the constitution aliases to
    `dvol_zscore` would have read as z=34 and fired EXTREME_DVOL on EVERY
    tick - and `EXTREME_DVOL` is not in the sleeve HOLD set, so it would have
    permanently flattened the book. The flag stayed off for want of a
    denominator, not for want of data.

WHAT THIS SUPPLIES
    The denominator. Deribit's `get_volatility_index_data` accepts an
    arbitrary time range, so the trailing distribution does not have to be
    ACCRUED - it can be fetched. Probed 2026-08-18: 401 daily rows per
    currency back to 2025-07-14. That means the z-score is real from the
    first tick after deploy, unlike a series that has to fill up (the P301
    warmup class, which is what made three other signals look dead).

    Live reading at build time: BTC 34.13 against a trailing-year
    [33.8, 82.6] and ETH 45.69 against [45.7, 95.8] - both sitting at the
    BOTTOM of their own year, i.e. z is around -1.5, nowhere near +5. So the
    honest z-score is not merely safe to publish, it is the opposite of what
    the raw level asserted.

SOL GETS NOTHING, DELIBERATELY
    Deribit lists ZERO SOL options and no SOL DVOL (P293). No key is written
    for SOL. An absence stays an absence (P2); fabricating a "market-average"
    volatility index for an asset nobody quotes one for is exactly the defect
    this file exists to undo.

FAIL DIRECTIONS
    * Fewer than `min_samples` usable rows -> `zscore()` returns None and the
      caller writes NO key. A z computed off a handful of points is a number
      with a measurement's name, which is the whole problem here.
    * Zero/degenerate dispersion -> None, never a divide-by-epsilon spike.
    * Fetch failure -> the persisted history is used; if it is too old to be
      the current regime the feed refuses rather than scoring against a stale
      distribution (P156).
    * Nothing here raises into a tick.
================================================================================
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import tempfile
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("HMATS.DvolHistory")

_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
_UA = "hmats/1.0 (+dvol-history)"
_STATE_VERSION = "dvol_history_v1"

SUPPORTED = ("BTC", "ETH")
DEFAULT_MIN_SAMPLES = 60
DEFAULT_WINDOW_DAYS = 365
DEFAULT_REFRESH_SEC = 12 * 3600.0
DEFAULT_MAX_HISTORY_AGE_SEC = 7 * 24 * 3600.0   # newest row must be recent


def _state_path() -> str:
    return os.path.join(os.environ.get("HMATS_DATA_DIR", "data"),
                        "dvol_history.json")


class DvolHistory:
    """Trailing daily DVOL per currency, persisted, with a real z-score."""

    def __init__(self,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 window_days: int = DEFAULT_WINDOW_DAYS,
                 refresh_sec: float = DEFAULT_REFRESH_SEC,
                 max_history_age_sec: float = DEFAULT_MAX_HISTORY_AGE_SEC,
                 timeout_sec: float = 20.0,
                 fetch_days: int = 400):
        self._min_samples = int(min_samples)
        self._window_days = int(window_days)
        self._refresh = float(refresh_sec)
        self._max_hist_age = float(max_history_age_sec)
        self._timeout = float(timeout_sec)
        self._fetch_days = int(fetch_days)
        # currency -> [(day_ts, close)] ascending
        self._series: Dict[str, List[Tuple[float, float]]] = {}
        self._last_fetch: float = 0.0
        self._warned: Dict[str, str] = {}
        self._restore()

    # ---------------------------------------------------------------- public
    def sample_count(self, currency: str) -> int:
        return len(self._series.get(str(currency).upper(), []))

    def refresh_if_stale(self) -> bool:
        """Fetch when the cached history is older than the refresh window."""
        if (time.time() - self._last_fetch) < self._refresh and self._series:
            return False
        ok = False
        for cur in SUPPORTED:
            if self._fetch(cur):
                ok = True
        self._last_fetch = time.time()
        if ok:
            self._persist()
        return ok

    def zscore(self, currency: str, value: float) -> Optional[float]:
        """Z of `value` against the trailing window. None when unscoreable."""
        cur = str(currency).upper()
        rows = self._recent(cur)
        if len(rows) < self._min_samples:
            self._warn_once(
                cur, "thin",
                "only {} usable daily rows (< {}) - no z-score is published, "
                "so no key is written".format(len(rows), self._min_samples))
            return None
        newest = rows[-1][0]
        age = time.time() - newest
        if age > self._max_hist_age:
            self._warn_once(
                cur, "stale",
                "newest DVOL row is {:.1f} days old - refusing to score "
                "against a stale distribution".format(age / 86400.0))
            return None
        vals = [v for _, v in rows]
        try:
            mu = statistics.fmean(vals)
            sd = statistics.pstdev(vals)
        except statistics.StatisticsError:  # noqa: silent-swallow - unscoreable
            return None
        if not (sd > 1e-9):
            self._warn_once(cur, "flat",
                            "trailing DVOL has no dispersion - unscoreable")
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):  # noqa: silent-swallow - value coercion
            return None
        if v != v:
            return None
        self._warned.pop(cur, None)
        return (v - mu) / sd

    def span_str(self, currency: str) -> str:
        rows = self._recent(str(currency).upper())
        if not rows:
            return "n=0"
        vals = [v for _, v in rows]
        return "n={} range=[{:.1f},{:.1f}] mean={:.1f}".format(
            len(rows), min(vals), max(vals), statistics.fmean(vals))

    # --------------------------------------------------------------- private
    def _recent(self, cur: str) -> List[Tuple[float, float]]:
        rows = self._series.get(cur) or []
        if not rows:
            return []
        cutoff = time.time() - self._window_days * 86400.0
        return [r for r in rows if r[0] >= cutoff]

    def _warn_once(self, cur: str, cause: str, msg: str) -> None:
        if self._warned.get(cur) == cause:
            return
        self._warned[cur] = cause
        logger.warning("[DVOL-HIST] %s: %s", cur, msg)

    def _fetch(self, cur: str) -> bool:
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - self._fetch_days * 86400 * 1000
        q = urllib.parse.urlencode({
            "currency": cur,
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
            "resolution": 86400,
        })
        try:
            req = urllib.request.Request(_URL + "?" + q,
                                         headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.load(resp)
        except Exception as e:  # noqa: silent-swallow - logged, keeps old series
            self._warn_once(
                cur, "http",
                "fetch failed ({}: {}) - keeping the persisted history".format(
                    type(e).__name__, e))
            return False
        rows = self._parse(payload)
        if not rows:
            self._warn_once(cur, "empty", "fetch returned no usable rows")
            return False
        merged = dict(self._series.get(cur) or [])
        merged.update(dict(rows))          # merge, never overwrite (P266)
        self._series[cur] = sorted(merged.items())
        self._warned.pop(cur, None)
        return True

    @staticmethod
    def _parse(payload) -> List[Tuple[float, float]]:
        """Deribit rows are [ts_ms, open, high, low, close]."""
        try:
            rows = ((payload or {}).get("result") or {}).get("data") or []
            out: List[Tuple[float, float]] = []
            for r in rows:
                if not isinstance(r, (list, tuple)) or len(r) < 5:
                    continue
                try:
                    ts = float(r[0]) / 1000.0
                    close = float(r[4])
                except (TypeError, ValueError):  # noqa: silent-swallow - row drop
                    continue
                if close > 0 and close == close:
                    out.append((ts, close))
            out.sort(key=lambda x: x[0])
            return out
        except Exception:  # noqa: silent-swallow - malformed payload -> no rows
            return []

    def _persist(self) -> None:
        try:
            path = _state_path()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            payload = {
                "version": _STATE_VERSION,
                "saved_ts": time.time(),
                "series": {c: [[t, v] for t, v in rows]
                           for c, rows in self._series.items()},
            }
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                                       suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:  # noqa: silent-swallow - tmp cleanup only
                        pass
        except Exception as e:  # noqa: silent-swallow - logged, cache-only loss
            logger.warning("[DVOL-HIST] persist failed (%s: %s) - the history "
                           "will be refetched after the next restart",
                           type(e).__name__, e)

    def _restore(self) -> None:
        try:
            path = _state_path()
            if not os.path.exists(path):
                return
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            if payload.get("version") != _STATE_VERSION:
                return
            for c, rows in (payload.get("series") or {}).items():
                clean = []
                for r in rows or []:
                    try:
                        clean.append((float(r[0]), float(r[1])))
                    except (TypeError, ValueError, IndexError):  # noqa: silent-swallow
                        continue
                if clean:
                    self._series[str(c).upper()] = sorted(clean)
            self._last_fetch = float(payload.get("saved_ts") or 0.0)
            if self._series:
                logger.info("[DVOL-HIST] restored %s", ", ".join(
                    "{}={}".format(c, len(v))
                    for c, v in sorted(self._series.items())))
        except Exception as e:  # noqa: silent-swallow - logged, cold start
            logger.warning("[DVOL-HIST] restore failed (%s: %s) - starting "
                           "cold", type(e).__name__, e)


_SINGLETON: Optional[DvolHistory] = None


def get_dvol_history(**kwargs) -> DvolHistory:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = DvolHistory(**kwargs)
    return _SINGLETON


def reset_dvol_history() -> None:
    global _SINGLETON
    _SINGLETON = None

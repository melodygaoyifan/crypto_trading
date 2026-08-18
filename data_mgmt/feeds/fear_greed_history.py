"""
================================================================================
HMATS [P293] - Fear & Greed HISTORY (free, keyless) + real z-score
================================================================================

Closes the measured gap: `market_data["sentiment_zscore"]` is NOT a z-score.

main.py computes it as

        fg_direction = (fg_value - 50) / 50.0
        sentiment_zscore = fg_direction * 3.0

which is a FIXED LINEAR RESCALE of today's Fear & Greed reading. It has no
mean, no standard deviation and no history behind it, because the feed calls
`https://api.alternative.me/fng/?limit=1` — one row. Fusion then derives the
sentiment agent's confidence as `min(|zscore| / 3.0, 1.0)`, so the strength
of a live ADVISE input (weight 0.10) is decided by a hardcoded scale rather
than by where today's reading sits in the market's own recent distribution.

The same endpoint serves the whole series for free: `?limit=0` returned
**3116 daily rows back to 2018-02-01** on a live probe (2026-08-17). So the
honest quantity was one query away the entire time.

WHAT THIS MODULE DOES
    - fetches and PERSISTS the daily F&G series (survives restarts, P154)
    - computes a real trailing z-score and percentile
    - REFUSES (returns None) below `min_samples` rather than emitting a
      confident-looking number from a thin window (P199: no data is not a
      verdict)

WHAT IT DELIBERATELY DOES NOT DO
    It does not decide which z-score the engine uses. Swapping the live
    `sentiment_zscore` changes the confidence of a fusion-consumed agent —
    a live behaviour change, so it is gated on config (default OFF) and
    both values are logged side by side until forward evidence says which
    is better. See P141.
================================================================================
"""

import json
import logging
import os
import statistics
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("HMATS.FearGreedHistory")

_STATE_VERSION = "fng_history_v1"
_API_URL = "https://api.alternative.me/fng/?limit=0&format=json"

# A z-score needs a distribution. Below this many days we have an average of
# very little, and a confident-looking number from it is worse than none.
DEFAULT_MIN_SAMPLES = 60
DEFAULT_WINDOW_DAYS = 365


@dataclass
class FearGreedStats:
    """Result of scoring one reading against the persisted history."""
    value: float
    zscore: Optional[float]
    percentile: Optional[float]
    window_days: int
    n_samples: int
    mean: Optional[float] = None
    stdev: Optional[float] = None
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.zscore is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "zscore": self.zscore,
            "percentile": self.percentile,
            "window_days": self.window_days,
            "n_samples": self.n_samples,
            "mean": self.mean,
            "stdev": self.stdev,
            "reason": self.reason,
        }


class FearGreedHistory:
    """Persisted daily Fear & Greed series with real distribution stats."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        refresh_interval_sec: float = 21600.0,   # 6h; the index updates daily
        window_days: int = DEFAULT_WINDOW_DAYS,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ):
        self._data_dir = data_dir or os.environ.get("HMATS_DATA_DIR", "data")
        self._path = os.path.join(self._data_dir, "fear_greed_history.json")
        self.refresh_interval_sec = refresh_interval_sec
        self.window_days = window_days
        self.min_samples = min_samples

        # date (YYYY-MM-DD) -> value, so re-fetching cannot duplicate a day
        self._series: Dict[str, float] = {}
        self._last_refresh: Optional[datetime] = None

        self._restore()

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def sample_count(self) -> int:
        return len(self._series)

    def cache_age_sec(self) -> Optional[float]:
        if self._last_refresh is None:
            return None
        _t = self._last_refresh
        if _t.tzinfo is None:
            _t = _t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - _t).total_seconds()

    def needs_refresh(self) -> bool:
        _age = self.cache_age_sec()
        return _age is None or _age >= self.refresh_interval_sec

    async def refresh_if_stale(self) -> bool:
        """Fetch the series when stale. Returns True if new rows landed.

        Never raises: a history refresh failing must not disturb a tick.
        """
        if not self.needs_refresh():
            return False
        try:
            rows = await self._fetch_series()
        except Exception as e:
            logger.warning(
                f"[FNG_HISTORY] refresh failed: {type(e).__name__}: {e} — "
                f"serving {len(self._series)} cached day(s)"
            )
            return False

        if not rows:
            # Do NOT stamp a failed fetch as fresh (the P265f defect).
            logger.warning("[FNG_HISTORY] refresh returned no rows — cache kept")
            return False

        before = len(self._series)
        for day, val in rows:
            self._series[day] = val
        self._last_refresh = datetime.now(timezone.utc)
        added = len(self._series) - before
        self._persist()
        logger.info(
            f"[FNG_HISTORY] refreshed: {len(self._series)} days "
            f"(+{added} new), span {self.span_str()}"
        )
        return added > 0

    def span_str(self) -> str:
        if not self._series:
            return "empty"
        days = sorted(self._series)
        return f"{days[0]} -> {days[-1]}"

    def score(
        self,
        value: float,
        window_days: Optional[int] = None,
    ) -> FearGreedStats:
        """Score a reading against the trailing window.

        Returns stats with zscore=None (and a reason) when the window is too
        thin or degenerate — never a fabricated 0.0, which would read as
        "perfectly average" (P2).
        """
        w = int(window_days or self.window_days)
        try:
            v = float(value)
        except (TypeError, ValueError):  # noqa: silent-swallow
            # [P293] Non-numeric input -> an explicit refusal record whose
            # `reason` field names the cause; never a silent 0.0.
            return FearGreedStats(
                value=float("nan"), zscore=None, percentile=None,
                window_days=w, n_samples=0, reason="value_not_numeric",
            )

        window = self._recent_values(w)
        n = len(window)
        if n < self.min_samples:
            return FearGreedStats(
                value=v, zscore=None, percentile=None, window_days=w,
                n_samples=n,
                reason=f"insufficient_history({n}<{self.min_samples})",
            )

        mean = statistics.fmean(window)
        try:
            sd = statistics.stdev(window)
        except statistics.StatisticsError:
            sd = 0.0
        if not (sd > 1e-9):
            # A constant window has no scale; dividing by it is the
            # Sharpe-of-a-constant defect (P265g).
            return FearGreedStats(
                value=v, zscore=None, percentile=None, window_days=w,
                n_samples=n, mean=mean, stdev=sd, reason="degenerate_stdev",
            )

        z = (v - mean) / sd
        below = sum(1 for x in window if x < v)
        pct = below / n
        return FearGreedStats(
            value=v, zscore=z, percentile=pct, window_days=w,
            n_samples=n, mean=mean, stdev=sd, reason="ok",
        )

    # =========================================================================
    # INTERNALS
    # =========================================================================

    def _recent_values(self, window_days: int) -> List[float]:
        if not self._series:
            return []
        days = sorted(self._series)[-max(1, window_days):]
        return [self._series[d] for d in days]

    async def _fetch_series(self) -> List[Tuple[str, float]]:
        """Fetch the full daily series. Returns [(YYYY-MM-DD, value)]."""
        import aiohttp
        from data_mgmt.feeds._http import create_session

        async with create_session() as session:
            async with session.get(
                _API_URL, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"[FNG_HISTORY] HTTP {resp.status}")
                    return []
                payload = await resp.json(content_type=None)

        return self._parse_payload(payload)

    @staticmethod
    def _parse_payload(payload: Any) -> List[Tuple[str, float]]:
        """Parse the alternative.me shape into (day, value) pairs."""
        rows = (payload or {}).get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []
        out: List[Tuple[str, float]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            _raw_val = r.get("value")
            _raw_ts = r.get("timestamp")
            if _raw_val is None or _raw_ts is None:
                continue  # [P293] drop a malformed row, keep the series
            try:
                val = float(_raw_val)
                ts = int(_raw_ts)
            except (TypeError, ValueError):  # noqa: silent-swallow
                continue  # [P293] drop a malformed row, keep the series
            if not (0.0 <= val <= 100.0):
                continue
            day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            out.append((day, val))
        return out

    def _persist(self) -> None:
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            payload = {
                "version": _STATE_VERSION,
                "last_refresh": (
                    self._last_refresh.isoformat() if self._last_refresh else None
                ),
                "series": self._series,
            }
            fd, tmp = tempfile.mkstemp(dir=self._data_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                os.replace(tmp, self._path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:  # noqa: silent-swallow
                        pass  # [P293] tmp cleanup only; the atomic replace
                        # above already succeeded or raised.
        except Exception as e:
            # Loud: this file is what makes the z-score survive a restart.
            logger.error(
                f"[FNG_HISTORY] persist FAILED ({type(e).__name__}: {e}) — "
                f"history will be refetched after the next restart"
            )

    def _restore(self) -> None:
        try:
            if not os.path.exists(self._path):
                return
            with open(self._path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if payload.get("version") != _STATE_VERSION:
                logger.warning(
                    f"[FNG_HISTORY] state version mismatch "
                    f"({payload.get('version')} != {_STATE_VERSION}) — cold start"
                )
                return
            series = payload.get("series") or {}
            if isinstance(series, dict):
                for k, v in series.items():
                    try:
                        self._series[str(k)] = float(v)
                    except (TypeError, ValueError):  # noqa: silent-swallow
                        continue  # [P293] skip one corrupt persisted day
            lr = payload.get("last_refresh")
            if lr:
                try:
                    _t = datetime.fromisoformat(lr)
                    if _t.tzinfo is None:
                        _t = _t.replace(tzinfo=timezone.utc)
                    self._last_refresh = _t
                except ValueError:
                    self._last_refresh = None
            logger.info(
                f"[FNG_HISTORY] restored {len(self._series)} day(s), "
                f"span {self.span_str()}"
            )
        except Exception as e:
            logger.warning(
                f"[FNG_HISTORY] restore failed ({type(e).__name__}: {e}) — cold start"
            )
            self._series = {}
            self._last_refresh = None


# =============================================================================
# SINGLETON
# =============================================================================

_fng_history_instance: Optional[FearGreedHistory] = None


def get_fear_greed_history(**kwargs) -> FearGreedHistory:
    global _fng_history_instance
    if _fng_history_instance is None:
        _fng_history_instance = FearGreedHistory(**kwargs)
    return _fng_history_instance


def reset_fear_greed_history():
    global _fng_history_instance
    _fng_history_instance = None


if __name__ == "__main__":
    import asyncio

    async def _main():
        logging.basicConfig(level=logging.INFO)
        h = FearGreedHistory()
        await h.refresh_if_stale()
        print("samples:", h.sample_count(), "span:", h.span_str())
        for v in (31.0, 50.0, 75.0, 10.0):
            s = h.score(v)
            legacy = (v - 50) / 50.0 * 3.0
            print(
                f"  F&G={v:5.1f}  legacy_z={legacy:+.3f}  "
                f"real_z={('%+.3f' % s.zscore) if s.zscore is not None else 'None':>7}  "
                f"pct={('%.3f' % s.percentile) if s.percentile is not None else 'None':>6}  "
                f"n={s.n_samples} {s.reason}"
            )

    asyncio.run(_main())

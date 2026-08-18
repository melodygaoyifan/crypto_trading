"""
================================================================================
HMATS [P306] - the real 1-hour price change, replacing a fixed rescale
================================================================================

WHAT WAS WRONG
    `data_mgmt/market_data_pipeline.py` published

        raw['price_change_1h_pct'] = raw['price_change_4h_pct'] / 4.0

    and a second copy at :2133 (`ret_1h = ret_4h / 4.0`) fed a feature vector.
    That is a CONSTANT RESCALE wearing a measurement's name - the same shape as
    the P293 sentiment "z-score" that was really `(fg-50)/50*3`. Its
    consequence is specific and bad: a genuine burst INSIDE a 4H bar is
    invisible. A -3% hour inside a flat 4H bar reports -0.0%, and

        risk/cascade_exhaustion_governor.py:323
            metrics.price_change_1h_pct <= -price_drop_detect_pct   (-3%)

    is one of the two conditions that can DETECT a cascade at all. The other
    is the liquidation window P304/P305 just repaired. Both were derived by
    dividing a slower aggregate, which is precisely the operation that erases
    a burst.

WHY KRAKEN PUBLIC OHLC
    The live tick is 4H, so no buffer of tick samples can ever contain a 1h
    observation - the resolution simply is not in the data the engine already
    holds. Kraken's public OHLC endpoint serves `interval=60` keyless and
    unmetered, and `defense/regime_book_shadow.py` already reads the same
    endpoint at `interval=240`; its `KRAKEN_PAIRS` map is IMPORTED here rather
    than restated, so the two cannot disagree about which pair an asset is
    (P172 - one implementation, and P133/P135/P253d's record that SOL is
    deliberately not a plain USD pair everywhere).

    Probed 2026-08-18: 721 hourly rows per pair (~30 days) on BTC/ETH/SOL.

WHAT IT RETURNS
    The return of the last COMPLETED hourly candle, as a FRACTION - the unit
    the pipeline's `price_change_4h_pct` and the governor's thresholds both
    use (`price_drop_detect_pct = 0.03` means 3%). Never a percent: the same
    key is written as a percent at main.py's [P147-b] enrich path, which is a
    latent unit disagreement corrected in the same commit as this module.

FAIL DIRECTIONS (all of them resolve to "no reading", never to a number)
    * The newest candle Kraken returns is the IN-PROGRESS hour. It is dropped
      (P253c) - reading a partial bar as a completed one is how a quiet hour
      reports as a violent one at :01 past.
    * If the newest COMPLETED candle is older than `max_age_sec`, the feed
      refuses. A frozen feed asserting an hours-old move is P156's defect.
    * Any transport, shape or parse failure returns None. The caller keeps the
      old behaviour; it never substitutes 0.0, which would claim "the price
      did not move" - a measurement nobody made (P2).
================================================================================
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

from defense.regime_book_shadow import KRAKEN_PAIRS

logger = logging.getLogger("HMATS.KrakenHourly")

_URL = "https://api.kraken.com/0/public/OHLC?pair={pair}&interval=60"
_UA = "hmats/1.0 (+kraken-hourly)"

DEFAULT_TTL_SEC = 900.0            # the tick is 4H; this only bounds retries
DEFAULT_MAX_AGE_SEC = 2.5 * 3600   # a completed hour older than this is stale


class KrakenHourlyReturns:
    """Last-completed-hour return per asset, as a fraction. Fail-soft."""

    def __init__(self,
                 ttl_sec: float = DEFAULT_TTL_SEC,
                 max_age_sec: float = DEFAULT_MAX_AGE_SEC,
                 timeout_sec: float = 12.0):
        self._ttl = float(ttl_sec)
        self._max_age = float(max_age_sec)
        self._timeout = float(timeout_sec)
        # asset -> (fetched_at, return_or_None, close_ts_or_None)
        self._cache: Dict[str, Tuple[float, Optional[float], Optional[float]]] = {}
        self._warned: Dict[str, str] = {}

    # ---------------------------------------------------------------- public
    def get(self, asset: str) -> Optional[float]:
        """Return the last completed hour's return (fraction), or None."""
        a = str(asset).upper()
        now = time.time()
        hit = self._cache.get(a)
        if hit is not None and (now - hit[0]) < self._ttl:
            return self._refuse_if_stale(a, hit[1], hit[2], now)
        val, close_ts = self._fetch(a)
        self._cache[a] = (now, val, close_ts)
        return self._refuse_if_stale(a, val, close_ts, now)

    def close_ts(self, asset: str) -> Optional[float]:
        hit = self._cache.get(str(asset).upper())
        return hit[2] if hit else None

    # --------------------------------------------------------------- private
    def _refuse_if_stale(self, asset: str, val: Optional[float],
                         close_ts: Optional[float],
                         now: float) -> Optional[float]:
        if val is None or close_ts is None:
            return None
        age = now - float(close_ts)
        if age > self._max_age:
            self._warn_once(
                asset, "stale",
                "newest completed hour is {:.1f}h old (> {:.1f}h) - "
                "no 1h reading this tick".format(
                    age / 3600.0, self._max_age / 3600.0),
            )
            return None
        return val

    def _warn_once(self, asset: str, cause: str, msg: str) -> None:
        if self._warned.get(asset) == cause:
            return
        self._warned[asset] = cause
        logger.warning("[KRAKEN-1H] %s: %s", asset, msg)

    def _fetch(self, asset: str) -> Tuple[Optional[float], Optional[float]]:
        pair = KRAKEN_PAIRS.get(asset)
        if not pair:
            self._warn_once(asset, "nopair",
                            "no Kraken pair mapped - no 1h reading is possible")
            return None, None
        try:
            req = urllib.request.Request(_URL.format(pair=pair),
                                         headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.load(resp)
        except Exception as e:  # noqa: silent-swallow - logged, degrades to None
            self._warn_once(
                asset, "http",
                "fetch failed ({}: {}) - no 1h reading; the caller keeps "
                "its prior behaviour".format(type(e).__name__, e))
            return None, None
        rows = self._rows(payload)
        if len(rows) < 3:
            self._warn_once(asset, "shape",
                            "only {} hourly rows - need >= 3".format(len(rows)))
            return None, None
        now = time.time()
        # Drop the in-progress hour: keep only candles whose CLOSE has already
        # happened (open_ts + 3600 <= now). Kraken stamps the row at the OPEN.
        done = [r for r in rows if (r[0] + 3600.0) <= now]
        if len(done) < 2:
            self._warn_once(asset, "nodone",
                            "no two completed hourly candles")
            return None, None
        prev_close = done[-2][1]
        last_ts, last_close = done[-1]
        if prev_close <= 0:
            return None, None
        self._warned.pop(asset, None)
        return (last_close - prev_close) / prev_close, last_ts + 3600.0

    @staticmethod
    def _rows(payload) -> List[Tuple[float, float]]:
        """[(open_ts, close_price)] ascending; [] on any odd shape."""
        try:
            result = (payload or {}).get("result") or {}
            keys = [k for k in result if k != "last"]
            if not keys:
                return []
            out: List[Tuple[float, float]] = []
            for r in result[keys[0]]:
                if isinstance(r, (list, tuple)) and len(r) >= 5:
                    out.append((float(r[0]), float(r[4])))
            out.sort(key=lambda x: x[0])
            return out
        except Exception:  # noqa: silent-swallow - malformed payload -> no rows
            return []


_SINGLETON: Optional[KrakenHourlyReturns] = None


def get_hourly_returns(**kwargs) -> KrakenHourlyReturns:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = KrakenHourlyReturns(**kwargs)
    return _SINGLETON


def reset_hourly_returns() -> None:
    global _SINGLETON
    _SINGLETON = None

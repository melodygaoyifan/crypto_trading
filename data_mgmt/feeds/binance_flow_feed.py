"""[P1a] Live fv2 flow features from Binance spot klines.

Purpose: serve the CURRENT bar's 13 fv2_* features to a 139-obs model
(P200-LADDER Rung 3). The math is data_mgmt/flow_features.py — the SAME
functions training uses, so train/serve parity is identity, not a claim.

Data source: api.binance.com spot klines (1h), which carry quote_volume,
count and taker_buy volumes. REACHABILITY IS REGION-DEPENDENT: verified
HTTP 200 from the Hetzner box, HTTP 451 (geo-blocked) from the US operator
laptop — so this feed works in production and fails soft in local runs.
That failure is LOUD (P160): a 139-obs model without fv2 cannot infer, and
"feed down" must never look like "features are zero".

Fetch cost: 2 paginated kline calls per asset per 4H tick (~1,150 1h bars
covers the 180-bar z-window plus warmup), cached per bar. Public endpoint,
no key, weight is well inside Binance's public limits.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import pandas as pd
import requests

from data_mgmt.flow_features import REF, latest_fv2_vector

logger = logging.getLogger(__name__)

_BASE = "https://api.binance.com/api/v3/klines"
_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
_NEED_BARS = 1150          # 180*4h z-window + warmup, in 1h bars, with margin
_KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_volume", "count", "taker_buy_base",
               "taker_buy_quote", "ignore"]


class BinanceFlowFeed:
    """Per-tick fv2 provider. `latest(asset)` returns a dict of the 13
    features or None — absence is explicit, never zero-filled."""

    def __init__(self, cache_ttl_sec: float = 3600.0):
        self._cache: Dict[str, tuple] = {}   # asset -> (ts, dict|None)
        self._cache_ttl = cache_ttl_sec
        self._fail_logged = False

    def _fetch_1h(self, symbol: str) -> Optional[pd.DataFrame]:
        frames = []
        end_time = None
        try:
            for _ in range(2):  # 2 x 1000-bar pages
                params = {"symbol": symbol, "interval": "1h", "limit": 1000}
                if end_time:
                    params["endTime"] = end_time
                r = requests.get(_BASE, params=params, timeout=15)
                if r.status_code != 200:
                    if not self._fail_logged:
                        logger.warning(
                            f"[FLOW-FEED] klines HTTP {r.status_code} for "
                            f"{symbol} — fv2 features UNAVAILABLE this tick "
                            f"(451 = geo-block: expected off-server). A "
                            f"139-obs model cannot run without them.")
                        self._fail_logged = True
                    return None
                batch = r.json()
                if not batch:
                    break
                df = pd.DataFrame(batch, columns=_KLINE_COLS)
                frames.append(df)
                end_time = int(batch[0][0]) - 1
                if len(batch) < 1000:
                    break
                time.sleep(0.1)
        except Exception as e:
            if not self._fail_logged:
                logger.warning(f"[FLOW-FEED] fetch failed for {symbol}: "
                               f"{type(e).__name__}: {e} — fv2 UNAVAILABLE")
                self._fail_logged = True
            return None
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        _first = int(df["open_time"].iloc[0])
        unit = "us" if _first > 1e14 else "ms"
        out = pd.DataFrame({
            "timestamp": pd.to_datetime(df["open_time"].astype("int64"), unit=unit),
        })
        for c in ("open", "high", "low", "close", "volume", "quote_volume",
                  "count", "taker_buy_base", "taker_buy_quote"):
            out[c] = df[c].astype(float)
        out = out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        # drop the still-forming final 1h bar: a partial bar's flow totals are
        # not comparable to completed bars and would wobble the z-scores
        return out.iloc[:-1] if len(out) > 1 else out

    def latest(self, asset: str) -> Optional[Dict[str, float]]:
        now = time.time()
        cached = self._cache.get(asset)
        if cached and now - cached[0] < self._cache_ttl:
            return cached[1]
        result = None
        sym = _SYMBOLS.get(asset)
        ref = _SYMBOLS.get(REF.get(asset, ""))
        if sym and ref:
            raw_self = self._fetch_1h(sym)
            raw_ref = self._fetch_1h(ref) if raw_self is not None else None
            if raw_self is not None and raw_ref is not None \
                    and len(raw_self) >= _NEED_BARS * 0.8:
                vec = latest_fv2_vector(raw_self, raw_ref, asset)
                if vec is not None:
                    result = {k: float(v) for k, v in vec.items()}
                    self._fail_logged = False
        if result is None and not self._fail_logged:
            logger.warning(f"[FLOW-FEED] {asset}: fv2 vector UNAVAILABLE "
                           f"(insufficient data or warmup) — models needing "
                           f"fv2 must skip this tick, not zero-fill")
            self._fail_logged = True
        self._cache[asset] = (now, result)
        return result


_singleton: Optional[BinanceFlowFeed] = None


def get_flow_feed() -> BinanceFlowFeed:
    global _singleton
    if _singleton is None:
        _singleton = BinanceFlowFeed()
    return _singleton

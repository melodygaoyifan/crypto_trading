"""[P407] Live Deribit 25-delta skew signal (Laevitas) -> contrarian direction.

Validated over 6.6y (2020-2026) of Laevitas Deribit option data pulled through
the logged-in dashboard backend: CONTRARIAN 25-delta skew is the first genuine
direction edge of the whole model campaign -- robust across deadband/window
(16/16 BTC, 14/16 ETH OOS cells), era-stable per-year (+6/7 BTC, +5/7 ETH,
INCLUDING +21.7%/+81.8% in the 2022 crash while buy-and-hold lost -82%/-74%),
cross-asset consistent, low-turnover (~15 flips/yr so it survives the flat CDE
fee). The multi-feature ridge OVERFIT (BTC lockbox -59%); the SIGNAL is the
edge, expressed as a robust rule -- same lesson as SMA200 beating every trained
model.

Runtime needs only the trailing ~60 days (for the z-score), which the standard
Laevitas API key serves within its 3-month cap -- the deep history was only for
validation. Fail-safe by construction: no key / fetch failure / stale / warmup
-> NOT fresh, so the seat is skipped and the incumbent certified book stands
(an absent feed must never move a live position, P2).

skew_25d = call_iv - put_iv ; negative = puts rich = fear.
Contrarian: skew much BELOW its trailing mean (extra fear) -> LONG (+1);
skew much ABOVE (calls rich / greed) -> SHORT (-1); inside the deadband -> hold.
"""
import os, json, time, logging, urllib.request, urllib.parse
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_BASE = "https://apiv2.laevitas.ch/api/v1/options/vol-surface/by-tenor"
_TENOR = 30            # 30-day skew (the standard, and what the probe used)
_WIN = 30             # trailing z-score window (pre-committed default)
_BAND = 1.0           # deadband in z units (pre-committed default)
_MIN_OBS = 15         # below this -> warmup, not fresh
_STALE_DAYS = 3       # latest skew older than this -> stale, not fresh
_FETCH_DAYS = 70      # pull a bit more than _WIN so the z always has a full window
_CACHE_TTL = 3 * 3600  # refetch at most every 3h (the loop is 4H)


class SkewFlowSignal:
    def __init__(self, data_dir: str = "data"):
        self._data_dir = data_dir
        self._state_path = os.path.join(data_dir, "skew_seat_state.json")
        self._key = os.environ.get("LAEVITAS_API_KEY", "").strip()
        self._cache = {}   # asset -> {"ts": epoch, "dir": float, "fresh": bool, "z": float}
        self._hold = {}    # asset -> last directional position (deadband hold)
        try:
            if os.path.exists(self._state_path):
                with open(self._state_path, "r", encoding="utf-8") as f:
                    st = json.load(f)
                self._hold = {k: float(v) for k, v in (st.get("hold") or {}).items()}
        except Exception as e:
            logger.warning(f"[SKEW] state restore failed ({type(e).__name__}); cold start")

    def _save(self):
        try:
            tmp = self._state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"hold": self._hold, "saved_at": time.time()}, f)
            os.replace(tmp, self._state_path)
        except Exception as e:
            logger.warning(f"[SKEW] state save failed ({type(e).__name__})")

    def _fetch_trailing(self, asset: str):
        """Return sorted list of daily 30d skew_25d for the last ~70 days, or None."""
        if not self._key:
            return None
        end = time.strftime("%Y-%m-%d", time.gmtime())
        start = time.strftime("%Y-%m-%d", time.gmtime(time.time() - _FETCH_DAYS * 86400))
        q = urllib.parse.urlencode({
            "exchange": "deribit", "currency": asset.upper(),
            "resolution": "1d", "start": start, "end": end,
            "limit": 1000, "sort_dir": "ASC"})
        req = urllib.request.Request(
            f"{_BASE}?{q}",
            headers={"X-API-Key": self._key,
                     "Accept": "application/json",
                     "User-Agent": "Mozilla/5.0 (hmats skew signal)"})
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                d = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            logger.warning(f"[SKEW] {asset}: fetch failed ({type(e).__name__}: {e})")
            return None
        rows = [x for x in d.get("data", [])
                if x.get("tenor") == _TENOR and x.get("skew_25d") is not None
                and x.get("date")]
        rows.sort(key=lambda x: x["date"])
        return rows or None

    def seat_direction(self, asset: str) -> Optional[Tuple[float, bool]]:
        """(direction, fresh) for the skew seat, or None on hard failure.

        direction in {-1,0,+1} (contrarian, deadband-held). fresh=False whenever
        the reading cannot be trusted (no key / fetch fail / warmup / stale) ->
        the caller skips the seat and the incumbent stands.
        """
        now = time.time()
        c = self._cache.get(asset)
        if c and (now - c["ts"]) < _CACHE_TTL:
            return (c["dir"], c["fresh"])
        rows = self._fetch_trailing(asset)
        if not rows or len(rows) < _MIN_OBS:
            self._cache[asset] = {"ts": now, "dir": 0.0, "fresh": False, "z": 0.0}
            return (0.0, False)
        # staleness: latest skew must be recent
        try:
            _d = rows[-1]["date"]
            if isinstance(_d, (int, float)):        # ms epoch (deep-history feed)
                latest = float(_d) / 1000.0
            else:                                    # ISO string (live by-tenor API)
                import datetime as _dt
                latest = _dt.datetime.fromisoformat(
                    str(_d).replace("Z", "+00:00")).timestamp()
            age_days = (now - latest) / 86400.0
        except Exception:
            age_days = 999
        vals = [float(x["skew_25d"]) for x in rows]
        w = vals[-_WIN:] if len(vals) >= _WIN else vals
        import statistics
        mu = statistics.fmean(w[:-1]) if len(w) > 1 else w[-1]
        sd = statistics.pstdev(w[:-1]) if len(w) > 2 else 0.0
        z = 0.0 if sd == 0 else (vals[-1] - mu) / sd
        prev = self._hold.get(asset, 0.0)
        if z < -_BAND:
            direction = 1.0          # extra fear -> contrarian long
        elif z > _BAND:
            direction = -1.0         # greed -> contrarian short
        else:
            direction = prev         # deadband: hold
        fresh = age_days <= _STALE_DAYS
        if fresh and direction != prev:
            self._hold[asset] = direction
            self._save()
        self._cache[asset] = {"ts": now, "dir": direction, "fresh": fresh, "z": z}
        if not fresh:
            logger.warning(f"[SKEW] {asset}: stale (latest {age_days:.1f}d old) -> seat skipped")
        return (direction, fresh)

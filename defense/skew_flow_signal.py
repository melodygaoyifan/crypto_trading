"""[P407] Live Deribit 25-delta skew signal (Laevitas) -> direction seat.

Validated over 6.6y (2020-2026) of Laevitas Deribit option data pulled through
the logged-in dashboard backend: the 25-delta skew z-deadband rule is the first
genuine direction edge of the whole model campaign -- robust across
deadband/window (16/16 BTC, 14/16 ETH OOS cells), era-stable per-year (+6/7
BTC, +5/7 ETH, INCLUDING +21.7%/+81.8% in the 2022 crash while buy-and-hold
lost -82%/-74%), cross-asset consistent, low-turnover (~15 flips/yr so it
survives the flat CDE fee). The multi-feature ridge OVERFIT (BTC lockbox -59%);
the SIGNAL is the edge, expressed as a robust rule -- same lesson as SMA200
beating every trained model.

Runtime needs only the trailing ~60 days (for the z-score), which the standard
Laevitas API key serves within its 3-month cap -- the deep history was only for
validation. Fail-safe by construction: no key / fetch failure / stale / warmup
-> NOT fresh, so the seat is skipped and the incumbent certified book stands
(an absent feed must never move a live position, P2).

SIGN CONVENTION [P420 -- corrected; the sign itself is UNCHANGED]
------------------------------------------------------------------
The Laevitas field is  skew_25d = put_25d_iv - call_25d_iv  (verified on every
row of the deep-history and the apiv2 by-tenor series alike: e.g. 2026-08-17
tenor 30, put 35.10 - call 31.48 = +3.62). POSITIVE = puts rich (fear);
NEGATIVE = calls rich (upside chase). The rule maps:

    z < -BAND  (skew far BELOW its trailing mean = calls unusually rich)  -> +1 LONG
    z > +BAND  (skew far ABOVE its trailing mean = puts unusually rich)   -> -1 SHORT
    inside the band                                                      -> hold

i.e. the seat RIDES call-richness and SELLS put-richness -- a momentum-shaped
reading of positioning, NOT the contrarian reading the earlier docstring
described ("call_iv - put_iv; negative = fear -> long" had the field's sign
backwards). The validation (training/skew_seat_calibration.py, P407f) walks
the SAME series with the SAME mapping (`contra = -z; +1 iff contra > BAND`),
so live == validated and the 6.6y evidence stands exactly as measured. DO NOT
flip the sign: flipping it would deploy the UNVALIDATED reading. The strategy
label stays `skew_contra` (renaming breaks the P317/P407c vocabulary pins);
see STRATEGY_LABEL below.

SERIES CAVEAT [P420] -- the live series is NOT the calibration series
----------------------------------------------------------------------
Live reads apiv2 `by-tenor` `skew_25d`; the 680/1080 bps/RT edge was measured
on the dashboard-backend series (training/training_data/laevitas_skew/*.json),
which apiv2 does NOT reproduce (per-expiry 25SEP26 skew_25d on 2026-08-17 =
3.64, by-tenor 3.62, dashboard 14.34). Measured over the 67-day overlap
(2026-06-18..08-24): raw corr 0.90-0.95, z-corr 0.62-0.84, and the +-1
deadband DECISIONS agree BTC 46/59, ETH 46/59 (opposite fires 0-1). So the
live seat trades a close cousin of the validated signal, not the signal
itself. Deploy-side responses, all here: (1) the live z-window is aligned to
the calibration EXACTLY (trailing 30 EXCLUDING the current day, min 8 obs --
it was 29 and min 3); (2) every tick's raw z25/z10/blended z and the tenor-30
skew_25d the seat read are recorded into the skewetf_{ASSET}.jsonl row via
`last_diag()`, so the forward ledger can be re-scored later against a
re-fetched calibration series; (3) the ~70-day by-tenor window fetched every
3h is BANKED into `data/laevitas_apiv2_skew_{ASSET}.jsonl` (merge-by-date
union, P266) -- in ~6 months that archive lets training/skew_seat_calibration
be re-run on the RUNTIME series, i.e. a runtime-parity recalibration (the P395
accumulate-then-probe pattern). Mock/absent rows are never written.

DEADBAND HOLD ACROSS A RESTART [P420]
-------------------------------------
The held direction is persisted (P154). A persisted hold older than
HOLD_MAX_AGE_DAYS is NOT restored blindly -- a -1 held N days ago could
flatten a long book today through the de-risk path with no current signal
behind it. Instead the hold is REPLAYED from the trailing fetched window
(walk the ~70 days forward through the band rule from state 0): deterministic
and restart-invariant, it reproduces what a continuous process would hold. If
the replay cannot run (no rows) the hold starts at 0.0 -> direction 0 -> no
seat -> the incumbent stands. An ABSENT state file (true cold start) also
starts at 0.0 -- absence is never a position (P2).
"""
import os, json, time, logging, urllib.request, urllib.parse
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# [P420] The ledger/attribution label the seat writes (main.py, core.seat_alpha,
# analytics.strategy_aging all pin it). "contra" is a HISTORICAL MISNOMER --
# see SIGN CONVENTION above: the rule rides call-richness, it is not contrarian.
# Kept verbatim because renaming it breaks the P317/P407c vocabulary pins and
# would split the forward ledger under two names.
STRATEGY_LABEL = "skew_contra"

_BASE = "https://apiv2.laevitas.ch/api/v1/options/vol-surface/by-tenor"
_TENOR = 30            # 30-day skew (the standard, and what the probe used)
_WIN = 30             # trailing z-score window EXCLUDING the current day (== calibration _Z_WIN)
_Z_MIN = 8            # [P420] min trailing obs for a z (== calibration _Z_MIN; was 3)
_BAND = 1.0           # deadband in z units (pre-committed default)
_MIN_OBS = 15         # below this -> warmup, not fresh
_STALE_DAYS = 3       # latest skew older than this -> stale, not fresh
_FETCH_DAYS = 70      # pull a bit more than _WIN so the z always has a full window
_CACHE_TTL = 3 * 3600  # refetch at most every 3h (the loop is 4H)
HOLD_MAX_AGE_DAYS = 1.0  # [P420] a persisted hold older than this is REPLAYED, not restored

# [P420] the by-tenor fields banked into the runtime archive (absent -> None,
# never fabricated). `date` is normalised to YYYY-MM-DD and is the merge key.
_ARCHIVE_FIELDS = ("skew_25d", "skew_10d", "call_25d_iv", "put_25d_iv", "atm_iv")

# [P420] Per-process diagnostics of the last z computation per asset, read by
# the skew+ETF combo shadow so the skewetf_{ASSET}.jsonl row carries the raw
# inputs the seat decided on (re-scoreable later against a re-fetched
# calibration series). Module-level so the combo shadow -- constructed
# separately in main.py with no reference to the signal object -- can read it
# without a main.py change. One SkewFlowSignal per process by construction.
_LAST_DIAG: Dict[str, Dict[str, object]] = {}


def last_diag(asset: str) -> Optional[Dict[str, object]]:
    """[P420] The last z-diagnostic computed for `asset` in this process, or
    None if the seat has not computed one (absence stays absent, P2)."""
    d = _LAST_DIAG.get(str(asset).upper())
    return dict(d) if d else None


def _row_date_iso(d) -> Optional[str]:
    """Normalise a by-tenor/deep-history date (ms epoch or ISO) to YYYY-MM-DD."""
    try:
        if isinstance(d, (int, float)):
            return time.strftime("%Y-%m-%d", time.gmtime(float(d) / 1000.0))
        s = str(d)
        return s[:10] if len(s) >= 10 and s[4] == "-" and s[7] == "-" else None
    except Exception:  # noqa: silent-swallow -- an unparseable date yields None; the caller skips the row
        return None


def zscore_trailing(series: List[float], win: int = _WIN,
                    min_obs: int = _Z_MIN) -> float:
    """[P420] z of the LAST value vs the `win` values strictly BEFORE it.

    This is the calibration's `_zseries` convention exactly (training/
    skew_seat_calibration.py: window = sig[max(0, i-30):i], < 8 obs -> 0.0).
    The pre-P420 live form used series[-30:] then w[:-1] (29 obs, min 3) --
    a train/serve skew on the live decider's own window (P164/P214 class).
    """
    if len(series) < 2:
        return 0.0
    import statistics
    w = series[:-1][-win:]
    if len(w) < min_obs:
        return 0.0
    mu = statistics.fmean(w)
    sd = statistics.pstdev(w)
    return 0.0 if sd == 0 else (series[-1] - mu) / sd


def band_direction(z: float, prev: float, band: float = _BAND) -> float:
    """[P420] The deadband rule (== calibration `_positions_from_z`)."""
    if z < -band:
        return 1.0          # calls unusually rich vs trailing -> ride it LONG
    if z > band:
        return -1.0         # puts unusually rich vs trailing -> SHORT
    return float(prev)      # inside the band: hold


def replay_hold(vals: List[float], vals10: Optional[List[float]] = None,
                min_obs10: int = _MIN_OBS) -> float:
    """[P420] Walk the window forward through the band rule from state 0 and
    return the hold a CONTINUOUS process would carry INTO the last bar (i.e.
    the state after processing every bar except the last). Deterministic.

    The blend rule matches seat_direction: when the 10d series is long enough
    the z is the mean of the 25d and 10d z's, else 25d alone.
    """
    use10 = bool(vals10) and len(vals10) >= min_obs10 and len(vals10) == len(vals)
    prev = 0.0
    for i in range(1, len(vals)):          # state entering bar i == pos after bar i-1
        z25 = zscore_trailing(vals[:i])
        z = (z25 + zscore_trailing(vals10[:i])) / 2.0 if use10 else z25
        prev = band_direction(z, prev)
    return prev


class SkewFlowSignal:
    def __init__(self, data_dir: str = "data"):
        self._data_dir = data_dir
        self._state_path = os.path.join(data_dir, "skew_seat_state.json")
        self._key = os.environ.get("LAEVITAS_API_KEY", "").strip()
        self._cache = {}   # asset -> {"ts": epoch, "dir": float, "fresh": bool, "z": float}
        self._hold = {}    # asset -> last directional position (deadband hold)
        # [P420] assets whose persisted hold is too old to trust: replayed from
        # the fetched window on the next seat_direction instead of restored.
        self._replay_pending: set = set()
        try:
            if os.path.exists(self._state_path):
                with open(self._state_path, "r", encoding="utf-8") as f:
                    st = json.load(f)
                hold = {k: float(v) for k, v in (st.get("hold") or {}).items()}
                saved_at = st.get("saved_at")
                age_days = ((time.time() - float(saved_at)) / 86400.0
                            if isinstance(saved_at, (int, float)) else None)
                if age_days is None or age_days > HOLD_MAX_AGE_DAYS:
                    # [P420] stale (or unstamped) hold: do NOT seat it blindly.
                    self._replay_pending = set(hold.keys())
                    logger.warning(
                        "[SKEW] persisted hold is %s -- will REPLAY from the "
                        "trailing window instead of restoring %s",
                        ("unstamped" if age_days is None
                         else f"{age_days:.1f}d old (> {HOLD_MAX_AGE_DAYS:g}d)"),
                        hold)
                else:
                    self._hold = hold
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

    # ------------------------------------------------------------------
    # [P420] runtime-series archive (the path to a runtime-parity recalibration)
    # ------------------------------------------------------------------
    def _archive_path(self, asset: str) -> str:
        return os.path.join(self._data_dir, f"laevitas_apiv2_skew_{asset.upper()}.jsonl")

    def bank_rows(self, asset: str, rows) -> int:
        """[P420] Merge fetched by-tenor rows into the per-asset archive
        (one row per date; a re-fetched date REPLACES the old row -- P266
        union semantics, new wins). Atomic rewrite. Returns the archive's row
        count, or -1 on failure. Fail-soft and never raises: this runs beside
        the decision path and a ledger must not be able to stop a tick.
        Nothing is written when `rows` is empty (a failed fetch banks nothing)."""
        if not rows:
            return -1
        path = self._archive_path(asset)
        try:
            merged: Dict[str, dict] = {}
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except Exception:  # noqa: silent-swallow -- a corrupt line is dropped at the rewrite; counted below
                            continue
                        if isinstance(r, dict) and r.get("date"):
                            merged[str(r["date"])] = r
            before = len(merged)
            added = 0
            for x in rows:
                if not isinstance(x, dict) or x.get("skew_25d") is None:
                    continue
                d = _row_date_iso(x.get("date"))
                if not d:
                    continue
                rec = {"date": d, "tenor": _TENOR}
                for k in _ARCHIVE_FIELDS:
                    v = x.get(k)
                    rec[k] = float(v) if isinstance(v, (int, float)) else None
                if d not in merged:
                    added += 1
                merged[d] = rec
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                for d in sorted(merged):
                    fh.write(json.dumps(merged[d]) + "\n")
            os.replace(tmp, path)
            if added:
                logger.info("[SKEW] %s: archive +%d new date(s) (%d -> %d rows)",
                            asset, added, before, len(merged))
            return len(merged)
        except Exception as e:
            logger.warning(f"[SKEW] {asset}: archive write failed ({type(e).__name__}: {e}) "
                           f"-- runtime series not banked this tick")
            return -1

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

        direction in {-1,0,+1} (band rule, deadband-held -- see SIGN CONVENTION
        in the module docstring). fresh=False whenever the reading cannot be
        trusted (no key / fetch fail / warmup / stale) -> the caller skips the
        seat and the incumbent stands.
        """
        now = time.time()
        if getattr(self, "_last_good", None) is None:
            self._last_good: Dict[str, Dict[str, float]] = {}
        lg_map: Dict[str, Dict[str, float]] = self._last_good
        c = self._cache.get(asset)
        if c and (now - c["ts"]) < _CACHE_TTL:
            return (c["dir"], c["fresh"])
        rows = self._fetch_trailing(asset)
        if rows:
            self.bank_rows(asset, rows)  # [P420] runtime archive; fail-soft
        if not rows or len(rows) < _MIN_OBS:
            # [P418] A transient FETCH failure is not data staleness. Handing
            # the seat to a different decider for one tick is a direction
            # change with no signal change (a potential fee round trip); if
            # the last GOOD computation is younger than the staleness bound,
            # carry the held direction instead (the P287 family-carry rule).
            lg = lg_map.get(asset)
            if lg and (now - lg["ts"]) / 86400.0 <= _STALE_DAYS:
                logger.warning(
                    f"[SKEW] {asset}: fetch failed -- carrying held direction "
                    f"{lg['dir']:+.0f} (last good "
                    f"{(now - lg['ts'])/3600.0:.1f}h old)")
                self._cache[asset] = {"ts": now, "dir": lg["dir"],
                                      "fresh": True, "z": lg.get("z", 0.0)}
                return (lg["dir"], True)
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
        # [P407g] BLEND 25d + 10d skew z-scores. The 10d tail slice (paid-for
        # Laevitas data, previously unused) makes the signal 6/6 era-stable on
        # BOTH assets vs 5/6 for 25d alone (measured over 6.6y). Both come from
        # the same by-tenor row, so this adds no request. Fail-safe: if 10d is
        # missing/thin, fall back to 25d-only so the live seat never breaks.
        # [P420] z-window == calibration: trailing 30 EXCLUDING current, min 8.
        vals = [float(x["skew_25d"]) for x in rows if x.get("skew_25d") is not None]
        vals10 = [float(x["skew_10d"]) for x in rows if x.get("skew_10d") is not None]
        use10 = len(vals10) >= _MIN_OBS
        z25 = zscore_trailing(vals)
        z10 = zscore_trailing(vals10) if use10 else None
        z = (z25 + z10) / 2.0 if use10 else z25
        hold_source = "persisted"
        if asset in self._replay_pending:
            # [P420] stale persisted hold: reproduce what a continuous process
            # would carry into THIS bar, from the same window it would have seen.
            try:
                replayed = replay_hold(vals, vals10 if len(vals10) == len(vals) else None)
            except Exception as e:  # noqa: silent-swallow -- a failed replay starts at 0.0 (no seat), logged
                logger.warning(f"[SKEW] {asset}: hold replay failed ({type(e).__name__}) -- starting at 0.0")
                replayed = 0.0
            logger.warning(
                f"[SKEW] {asset}: hold REPLAYED from the trailing window -> "
                f"{replayed:+.0f} (stale persisted hold discarded)")
            self._hold[asset] = replayed
            self._replay_pending.discard(asset)
            hold_source = "replayed"
        prev = self._hold.get(asset, 0.0)
        direction = band_direction(z, prev)
        fresh = age_days <= _STALE_DAYS
        if fresh and (direction != prev or hold_source == "replayed"):
            self._hold[asset] = direction
            self._save()
        self._cache[asset] = {"ts": now, "dir": direction, "fresh": fresh, "z": z}
        # [P420] raw inputs of THIS decision, for the skewetf ledger row
        _LAST_DIAG[asset.upper()] = {
            "ts": now,
            "skew_25d": vals[-1] if vals else None,
            "skew_10d": vals10[-1] if vals10 else None,
            "z25": round(float(z25), 4),
            "z10": None if z10 is None else round(float(z10), 4),
            "z": round(float(z), 4),
            "n_rows": len(vals),
            "hold_source": hold_source,
            "fresh": bool(fresh),
            "direction": float(direction),
        }
        if fresh:
            lg_map[asset] = {"ts": now, "dir": direction, "z": z}  # [P418]
        if not fresh:
            logger.warning(f"[SKEW] {asset}: stale (latest {age_days:.1f}d old) -> seat skipped")
        return (direction, fresh)

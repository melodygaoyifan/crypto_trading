"""[P392] Macro factor lab — which macro factors MOVE BTC/ETH/SOL, and does any
factor LEAD at a horizon a 4H system could act on. MEASUREMENT ONLY: no live
path reads this module; it writes two reports under training/reports/.

DATA
  Crypto : training/training_data/drl_training/{A}_4H_ohlcv.parquet via the
           funding_legs_lab.load_closes precedent (~6y of 4H closes, UTC).
  Macro  : FRED daily/weekly history (DGS10, DFII10, SP500, VIXCLS, DTWEXBGS,
           WALCL, RRPONTSYD), fetched with the FRED_API_KEY from the repo .env
           (never printed) and CACHED raw to
           training/reports/macro_factor_series_p392.json so the run is
           reproducible. An unfetchable series is a NAMED gap per series,
           never a silent skip (P2).

CAUSALITY CONVENTION (stated once, enforced in code, pinned by tests):
  A FRED value stamped business day B is published AFTER day B (usually the
  next business morning). Conservatively, this lab treats the value of
  business day B as KNOWN only from B + PUBLICATION_LAG_BDAYS (=2) BUSINESS
  days at 00:00 UTC. (Calendar shift(2) would leak: a Friday observation
  would read as known Sunday 00:00 although it is published Monday morning;
  Friday + 2 business days = Tuesday 00:00 is conservative everywhere.)
  So for the tradeable-lead tests, the predictor available at day D 00:00 UTC
  is the factor CHANGE of the business day two business days earlier — the
  task's "a 4H bar in day D may read the macro change of day D-2 at best".
  Liquidity series (WALCL weekly / RRPONTSYD daily) use a 7-CALENDAR-day lag
  (H.4.1 is published the Thursday after its Wednesday stamp; one full week
  is conservative).
  CONTEMPORANEOUS betas deliberately use SAME-DAY alignment (no lag): they
  measure co-movement, not tradability.

WEEKENDS / HOLIDAYS: macro levels exist only on business days. Changes are
  computed business-day-to-business-day (Friday->Monday is ONE change, never
  zero-filled). Crypto trades 24/7; for factor regressions the crypto return
  is aligned to the SAME business-day grid (close(B_prev)->close(B), spanning
  the same calendar interval as the factor change). Crypto days with no fresh
  macro information are ABSENT from the factor samples — a fabricated 0.0
  change is not an observation (P2); the lead sample contains only days on
  which a fresh factor change first became known.

ERA CONVENTION (stated choice): funding_legs_lab.ERAS bar-index bands mapped
  to DATES per asset via that asset's own 4H index —
  pre_design [bar 800, 3000), design [3000, 9100), validation [9100, end).
  On BTC that is ~2020-12-12 / 2021-12-14 / 2024-09-25. "Middle era" below
  means design; "recent era" means validation.

COST BAR (P166 arithmetic at the MEASURED CDE round-trip costs, P315/P334):
  RT_COST_BPS = {BTC: 19.7, ETH: 29.0, SOL: 29.0}.
  Implied edge of a Spearman IC: edge_bps = 0.7979 * 2*sin(pi*IC/6) * sigma_fwd_bps
  where sigma_fwd_bps is the std of the cell's own forward returns.
  Required: edge >= 2 * RT cost  =>  IC_req = (6/pi) * asin(cost/(0.7979*sigma));
  if cost/(0.7979*sigma) >= 1 the cell is unclearable at any IC (IC_req = inf).
  Significance: overlap-corrected t (P231): t = IC * sqrt(n_eff - 1),
  n_eff = n / h, h = horizon / median inter-sample gap (>= 1).

PRE-COMMITTED VERDICT RULES (P260 — written before the first run; the run
reports these as they fall, never adjusts them):
  A lead cell (asset, factor, horizon, era) PASSES iff
      n_eff >= 30  AND  |t| >= 2.0  AND  |IC| >= IC_req(era sigma).
  A factor is TRADEABLE-LEAD for an asset/horizon iff its lead cell PASSES in
  BOTH the design (middle) AND validation (recent) eras AND the IC sign
  agrees across the two eras.
  Else it is BETA-ONLY (risk context) iff its CONTEMPORANEOUS beta has
  |t| >= 2.0 in BOTH design and validation eras with the same beta sign.
  Else NOISE.
  EVENT WINDOWS report the vol multiple (mean |4H return| in the event bars
  vs the same-hours unconditional mean |return|); NO direction claim unless
  the event-bar mean signed return has |t| >= 2.

ANTI-VACUITY (P174): the run itself includes (a) a shuffled-factor control
  (must produce |IC| ~ 0) and (b) a PLANTED-lead control — a synthetic factor
  constructed so that, after the full publication-lag alignment, the
  predictor equals the forward return; the lab must report it with IC ~ 1,
  proving the alignment does not lag a real signal away (P164 family).

EVENT DATES: 2026 FOMC decision days are hardcoded (2026-01-28, 03-18,
  04-29, 06-17, 07-29; decision 14:00 ET lands in the 16:00 UTC 4H bar,
  presser in the 16:00/20:00 bars — both bars are the window). The CPI-day
  set is SKIPPED: exact 2025-2026 BLS release dates could not be verified
  from this environment, and running on guessed dates would measure nothing.

Usage:  python -X utf8 training/scripts/macro_factor_lab.py [--refetch]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.funding_legs_lab import ERAS, load_closes  # noqa: E402

REPORT_DIR = REPO / "training" / "reports"
CACHE_PATH = REPORT_DIR / "macro_factor_series_p392.json"
REPORT_PATH = REPORT_DIR / "macro_factor_lab_p392.json"

ASSETS = ("BTC", "ETH", "SOL")

# [P315/P334] measured CDE round-trip costs, bps.
RT_COST_BPS = {"BTC": 19.7, "ETH": 29.0, "SOL": 29.0}

# Factor definitions: series_id -> (kind of change). "diff" = arithmetic
# change (yields in pp, VIX in points); "ret" = percent change.
DAILY_FACTORS = {
    "DGS10": "diff",      # 10y nominal yield, pp
    "DFII10": "diff",     # 10y REAL yield, pp
    "SP500": "ret",       # S&P 500
    "VIXCLS": "diff",     # VIX, points
    "DTWEXBGS": "ret",    # broad dollar
}
LIQUIDITY_FACTORS = ("WALCL", "RRPONTSYD")
ALL_SERIES = tuple(DAILY_FACTORS) + LIQUIDITY_FACTORS

# Publication-lag conventions (see module docstring).
PUBLICATION_LAG_BDAYS = 2          # daily factors: business days
LIQUIDITY_LAG_DAYS = 7             # weekly balance-sheet series: calendar days
LIQUIDITY_CHANGE_DAYS = 28         # "4-week change"
LIQUIDITY_FWD_DAYS = 14            # 2-week-forward return

FOMC_2026 = ("2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29")
EVENT_BAR_HOURS_UTC = (16, 20)     # decision + presser 4H bars

OBS_START = "2020-01-01"

MIN_N_EFF = 30.0
T_BAR = 2.0
EDGE_MARGIN = 2.0                  # edge >= EDGE_MARGIN * RT cost


# =============================================================================
# FRED fetch + cache
# =============================================================================

def load_fred_key(env_path: Path | None = None) -> str | None:
    """Hand-parse FRED_API_KEY from the repo .env. Never printed by callers."""
    p = env_path or (REPO / ".env")
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("FRED_API_KEY"):
                _, _, v = line.partition("=")
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
    except OSError:
        return None
    return None


def fetch_fred_series(series_id: str, api_key: str,
                      obs_start: str = OBS_START) -> list[dict]:
    q = urllib.parse.urlencode({
        "series_id": series_id, "api_key": api_key, "file_type": "json",
        "observation_start": obs_start,
    })
    url = f"https://api.stlouisfed.org/fred/series/observations?{q}"
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.loads(r.read().decode("utf-8"))
    obs = payload.get("observations")
    if not isinstance(obs, list):
        raise ValueError(f"{series_id}: malformed FRED payload (no observations)")
    return [{"date": o["date"], "value": o["value"]} for o in obs]


def load_or_fetch_macro(refetch: bool = False) -> tuple[dict, dict]:
    """Returns (series_obs, status). status[series] = 'fetched'|'cache'|error."""
    cache: dict = {}
    if CACHE_PATH.exists() and not refetch:
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}
    key = load_fred_key()
    series, status = {}, {}
    for sid in ALL_SERIES:
        if sid in cache and cache[sid].get("observations"):
            series[sid] = cache[sid]["observations"]
            status[sid] = "cache"
            continue
        if not key:
            status[sid] = "UNFETCHABLE: FRED_API_KEY absent from .env"
            continue
        try:
            series[sid] = fetch_fred_series(sid, key)
            status[sid] = "fetched"
            cache[sid] = {"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "observations": series[sid]}
            time.sleep(0.4)  # politeness; FRED allows 120 req/min
        except Exception as e:  # noqa: BLE001 — per-series named gap, never a silent skip (P2)
            status[sid] = f"UNFETCHABLE: {type(e).__name__}: {e}"
    if any(v == "fetched" for v in status.values()):
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, indent=1), encoding="utf-8",
                              newline="\n")
    return series, status


def obs_to_series(obs: list[dict]) -> pd.Series:
    """Raw FRED obs -> float Series indexed by tz-aware UTC midnight; '.' dropped."""
    idx, vals = [], []
    for o in obs:
        v = o.get("value", ".")
        if v in (".", "", None):
            continue  # a missing print is ABSENT, not 0 (P2)
        idx.append(pd.Timestamp(o["date"], tz="UTC"))
        vals.append(float(v))
    return pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()


# =============================================================================
# Alignment machinery (all pure; pinned by tests)
# =============================================================================

def bd_changes(level: pd.Series, kind: str) -> pd.Series:
    """Business-observation-to-business-observation changes. One change per
    consecutive observation pair (Friday->Monday is ONE change), stamped at
    the LATER observation day. Never zero-filled."""
    level = level.dropna()
    if kind == "ret":
        ch = level.pct_change()
    else:
        ch = level.diff()
    return ch.dropna()


def known_from(dates: pd.DatetimeIndex,
               lag_bdays: int = PUBLICATION_LAG_BDAYS) -> pd.DatetimeIndex:
    """The 00:00 UTC instant from which the value stamped at each date may be
    read. Business-day shift: a Friday observation is usable from Tuesday
    00:00 UTC, never Sunday (see module docstring)."""
    return pd.DatetimeIndex([d + pd.offsets.BusinessDay(lag_bdays) for d in dates])


def daily_closes(closes_4h: pd.Series) -> pd.Series:
    """Last 4H close of each UTC calendar day (= that day's 24:00 close),
    reindexed to the full calendar range with ffill (a level carried over a
    rare missing bar is honest; a change is never fabricated from it)."""
    d = closes_4h.groupby(closes_4h.index.normalize()).last()
    full = pd.date_range(d.index[0], d.index[-1], freq="D", tz="UTC")
    return d.reindex(full).ffill()


def align_lead(changes: pd.Series, dclose: pd.Series, horizon_days: int,
               lag_bdays: int | None = PUBLICATION_LAG_BDAYS,
               lag_days: int | None = None) -> pd.DataFrame:
    """Tradeable-lead alignment. For each factor change (stamped obs day B),
    the sample day D is the first 00:00 UTC at which it is known
    (B + lag). Predictor x = the change; forward return
    fwd = close(D + horizon - 1) / close(D - 1) - 1  (position takeable at
    D 00:00, marked at the D-1 close). Sample contains ONLY days where a
    fresh change first became known — absent macro days are excluded, never
    zero-filled."""
    changes = changes.dropna()
    if lag_days is not None:
        kf = pd.DatetimeIndex([d + pd.Timedelta(days=lag_days)
                               for d in changes.index])
    else:
        kf = known_from(changes.index, lag_bdays or 0)
    rows = []
    for obs_d, d, x in zip(changes.index, kf, changes.to_numpy()):
        d = d.normalize()
        base_day = d - pd.Timedelta(days=1)
        end_day = d + pd.Timedelta(days=horizon_days - 1)
        if base_day not in dclose.index or end_day not in dclose.index:
            continue
        base, end = dclose.loc[base_day], dclose.loc[end_day]
        if not (np.isfinite(base) and np.isfinite(end)) or base <= 0:
            continue
        rows.append((d, obs_d, float(x), float(end / base - 1.0)))
    return pd.DataFrame(rows, columns=["sample_date", "obs_date", "x", "fwd"]
                        ).set_index("sample_date")


def spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def overlap_t(ic: float, n: int, h: float) -> float:
    """[P231] overlap-corrected t: n_eff = n / h."""
    n_eff = n / max(1.0, h)
    if not np.isfinite(ic) or n_eff <= 1:
        return float("nan")
    return float(ic * math.sqrt(n_eff - 1.0))


def implied_edge_bps(ic: float, sigma_fwd_bps: float) -> float:
    """[P166] expected edge of a Spearman IC against forward-return vol."""
    return 0.7979 * 2.0 * math.sin(math.pi * abs(ic) / 6.0) * sigma_fwd_bps


def required_ic(cost_rt_bps: float, sigma_fwd_bps: float,
                margin: float = EDGE_MARGIN) -> float:
    """Smallest |IC| whose implied edge >= margin * RT cost. inf if the
    cell is unclearable at any IC (asin domain)."""
    if sigma_fwd_bps <= 0:
        return float("inf")
    ratio = (margin * cost_rt_bps) / (2.0 * 0.7979 * sigma_fwd_bps)
    if ratio >= 1.0:
        return float("inf")
    return (6.0 / math.pi) * math.asin(ratio)


def _ols_beta(x: np.ndarray, y: np.ndarray) -> float:
    vx = np.var(x)
    if vx == 0:
        return float("nan")
    return float(np.cov(x, y, ddof=0)[0, 1] / vx)


def block_bootstrap_beta(x: np.ndarray, y: np.ndarray, block: int = 20,
                         reps: int = 500, seed: int = 0) -> tuple[float, float]:
    """(beta, t) with a circular 20-day block bootstrap SE — the
    'Newey-West-ish' robustness the task asks for."""
    n = len(x)
    beta = _ols_beta(x, y)
    if n < 3 * block or not np.isfinite(beta):
        return beta, float("nan")
    rng = np.random.default_rng(seed)
    nblocks = int(math.ceil(n / block))
    betas = []
    for _ in range(reps):
        starts = rng.integers(0, n, nblocks)
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        betas.append(_ols_beta(x[idx], y[idx]))
    se = float(np.std(betas))
    return beta, (beta / se if se > 0 else float("nan"))


def contemporaneous_cell(changes: pd.Series, dclose: pd.Series,
                         start: pd.Timestamp, end: pd.Timestamp | None) -> dict:
    """Same-day co-movement: crypto return over the SAME business interval as
    each factor change (close(B_prev)->close(B))."""
    ch = changes.dropna()
    ch = ch[ch.index >= start]
    if end is not None:
        ch = ch[ch.index < end]
    if len(ch) < 10:
        return {"n": int(len(ch)), "note": "insufficient"}
    # The change stamped at obs day B spans from the PREVIOUS observation day;
    # the crypto return is taken over the SAME interval (close->close), so a
    # Friday->Monday factor change faces the Friday->Monday crypto return.
    obs_days = ch.index
    xs, ys = [], []
    for pd_day, cd, x in zip(obs_days[:-1].normalize(), obs_days[1:].normalize(),
                             ch.to_numpy()[1:]):
        if pd_day in dclose.index and cd in dclose.index:
            b, e = dclose.loc[pd_day], dclose.loc[cd]
            if np.isfinite(b) and np.isfinite(e) and b > 0:
                xs.append(float(x))
                ys.append(float(e / b - 1.0))
    x_arr, y_arr = np.asarray(xs), np.asarray(ys)
    if len(x_arr) < 10:
        return {"n": int(len(x_arr)), "note": "insufficient"}
    beta, t = block_bootstrap_beta(x_arr, y_arr)
    corr = float(np.corrcoef(x_arr, y_arr)[0, 1])
    return {"n": int(len(x_arr)), "beta": beta, "t": t,
            "corr": corr, "r2": corr * corr}


def lead_cell(changes: pd.Series, dclose: pd.Series, horizon_days: int,
              cost_rt_bps: float, start: pd.Timestamp,
              end: pd.Timestamp | None,
              lag_bdays: int | None = PUBLICATION_LAG_BDAYS,
              lag_days: int | None = None,
              aligned: pd.DataFrame | None = None) -> dict:
    al = aligned if aligned is not None else align_lead(
        changes, dclose, horizon_days, lag_bdays=lag_bdays, lag_days=lag_days)
    al = al[al.index >= start]
    if end is not None:
        al = al[al.index < end]
    n = len(al)
    if n < 10:
        return {"n": n, "note": "insufficient"}
    gaps = al.index.to_series().diff().dt.days.dropna()
    med_gap = float(gaps.median()) if len(gaps) else 1.0
    h = max(1.0, horizon_days / max(1.0, med_gap))
    ic = spearman_ic(al["x"].to_numpy(), al["fwd"].to_numpy())
    t = overlap_t(ic, n, h)
    sigma = float(np.std(al["fwd"].to_numpy())) * 1e4
    ic_req = required_ic(cost_rt_bps, sigma)
    edge = implied_edge_bps(ic, sigma) if np.isfinite(ic) else float("nan")
    n_eff = n / h
    passes = (n_eff >= MIN_N_EFF and np.isfinite(ic) and np.isfinite(t)
              and abs(t) >= T_BAR and abs(ic) >= ic_req)
    return {"n": n, "n_eff": round(n_eff, 1), "h": round(h, 2),
            "ic": None if not np.isfinite(ic) else round(ic, 4),
            "t": None if not np.isfinite(t) else round(t, 2),
            "sigma_fwd_bps": round(sigma, 1),
            "ic_required": None if not np.isfinite(ic_req) else round(ic_req, 4),
            "implied_edge_bps": None if not np.isfinite(edge) else round(edge, 1),
            "passes": bool(passes)}


def decide_verdict(lead_design: dict, lead_valid: dict,
                   cont_design: dict, cont_valid: dict) -> str:
    """PRE-COMMITTED verdict rule (module docstring). Pure; pinned by tests."""
    ld_ok = bool(lead_design.get("passes"))
    lv_ok = bool(lead_valid.get("passes"))
    if ld_ok and lv_ok:
        s1, s2 = lead_design.get("ic") or 0.0, lead_valid.get("ic") or 0.0
        if s1 * s2 > 0:
            return "TRADEABLE-LEAD"
    def _beta_sig(c):
        t, b = c.get("t"), c.get("beta")
        return (t is not None and b is not None
                and np.isfinite(t) and abs(t) >= T_BAR)
    if _beta_sig(cont_design) and _beta_sig(cont_valid):
        if (cont_design.get("beta") or 0.0) * (cont_valid.get("beta") or 0.0) > 0:
            return "BETA-ONLY"
    return "NOISE"


# =============================================================================
# Controls (anti-vacuity, P174)
# =============================================================================

def planted_factor(dclose: pd.Series, horizon_days: int = 1,
                   lag_bdays: int = PUBLICATION_LAG_BDAYS) -> pd.Series:
    """A synthetic factor whose change stamped at business day B equals the
    forward return the aligned sample will measure at known_from(B). A
    correct alignment must recover IC ~ 1; an alignment that lags the signal
    away cannot (P164 family)."""
    bdays = pd.bdate_range(dclose.index[0], dclose.index[-1], tz="UTC")
    kf = known_from(bdays, lag_bdays)
    vals, idx = [], []
    for b, d in zip(bdays, kf):
        d = d.normalize()
        base_day = d - pd.Timedelta(days=1)
        end_day = d + pd.Timedelta(days=horizon_days - 1)
        if base_day in dclose.index and end_day in dclose.index:
            base, end = dclose.loc[base_day], dclose.loc[end_day]
            if np.isfinite(base) and np.isfinite(end) and base > 0:
                idx.append(b)
                vals.append(float(end / base - 1.0))
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


def shuffled_control(changes: pd.Series, dclose: pd.Series,
                     horizon_days: int = 1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    shuf = pd.Series(rng.permutation(changes.dropna().to_numpy()),
                     index=changes.dropna().index)
    al = align_lead(shuf, dclose, horizon_days)
    ic = spearman_ic(al["x"].to_numpy(), al["fwd"].to_numpy())
    return {"n": int(len(al)), "ic": None if not np.isfinite(ic) else round(ic, 4)}


# =============================================================================
# Event windows
# =============================================================================

def event_bars(closes_4h: pd.Series, event_days: tuple[str, ...],
               hours: tuple[int, ...] = EVENT_BAR_HOURS_UTC) -> dict:
    r = closes_4h.pct_change().dropna()
    hrs = r.index.hour
    same_hours = r[np.isin(hrs, hours)]
    days = {pd.Timestamp(d, tz="UTC").normalize() for d in event_days}
    ev = same_hours[[ts.normalize() in days for ts in same_hours.index]]
    if len(ev) == 0 or len(same_hours) == 0:
        return {"n_event_bars": int(len(ev)), "note": "no event bars in range"}
    mult = float(np.mean(np.abs(ev)) / np.mean(np.abs(same_hours)))
    mean_r = float(np.mean(ev))
    sd = float(np.std(ev, ddof=1)) if len(ev) > 1 else float("nan")
    t = mean_r / (sd / math.sqrt(len(ev))) if sd and sd > 0 else float("nan")
    return {"n_event_bars": int(len(ev)),
            "vol_multiple": round(mult, 2),
            "mean_signed_return_bps": round(mean_r * 1e4, 1),
            "direction_t": None if not np.isfinite(t) else round(t, 2),
            "direction_claim": bool(np.isfinite(t) and abs(t) >= T_BAR)}


# =============================================================================
# main
# =============================================================================

def _era_dates(closes_4h: pd.Series) -> dict:
    idx = closes_4h.index
    out = {}
    for name, (a, b) in ERAS.items():
        start = idx[a]
        end = idx[b] if (b is not None and b < len(idx)) else None
        out[name] = (start, end)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args(argv)

    series_obs, status = load_or_fetch_macro(refetch=args.refetch)
    print("[P392] series status:")
    for sid in ALL_SERIES:
        print(f"  {sid:10s} {status.get(sid)}")

    levels = {sid: obs_to_series(series_obs[sid])
              for sid in ALL_SERIES if sid in series_obs}
    changes = {sid: bd_changes(levels[sid], DAILY_FACTORS[sid])
               for sid in DAILY_FACTORS if sid in levels}

    report: dict = {
        "meta": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "publication_lag_bdays": PUBLICATION_LAG_BDAYS,
            "liquidity_lag_days": LIQUIDITY_LAG_DAYS,
            "rt_cost_bps": RT_COST_BPS,
            "edge_margin": EDGE_MARGIN,
            "era_convention": "funding_legs_lab.ERAS bar bands mapped to dates per asset",
            "cpi_events": "SKIPPED — exact 2025-2026 BLS release dates unverifiable here",
            "verdict_rule": "pre-committed in module docstring before first run",
        },
        "series_status": status,
        "contemporaneous": {}, "lead": {}, "liquidity": {},
        "events": {}, "controls": {}, "verdicts": {},
    }

    for asset in ASSETS:
        c4 = load_closes(asset)
        dc = daily_closes(c4)
        eras = _era_dates(c4)
        cost = RT_COST_BPS[asset]

        # -- contemporaneous
        cont_a: dict = {}
        for sid, ch in changes.items():
            cont_a[sid] = {}
            for era, (s, e) in eras.items():
                cont_a[sid][era] = contemporaneous_cell(ch, dc, s, e)
        report["contemporaneous"][asset] = cont_a

        # -- lead
        lead_a: dict = {}
        for sid, ch in changes.items():
            lead_a[sid] = {}
            for hz in (1, 5):
                al = align_lead(ch, dc, hz)
                lead_a[sid][f"{hz}d"] = {}
                for era, (s, e) in eras.items():
                    lead_a[sid][f"{hz}d"][era] = lead_cell(
                        ch, dc, hz, cost, s, e, aligned=al)
                lead_a[sid][f"{hz}d"]["pooled"] = lead_cell(
                    ch, dc, hz, cost, eras["pre_design"][0], None, aligned=al)
        report["lead"][asset] = lead_a

        # -- liquidity (4-week change, 1-week lag, 2-week forward)
        liq_a: dict = {}
        for sid in LIQUIDITY_FACTORS:
            if sid not in levels:
                liq_a[sid] = {"note": status.get(sid, "absent")}
                continue
            lv = levels[sid].dropna()
            ch4w = (lv - lv.reindex(lv.index - pd.Timedelta(days=LIQUIDITY_CHANGE_DAYS),
                                    method="ffill").to_numpy()).dropna()
            al = align_lead(ch4w, dc, LIQUIDITY_FWD_DAYS,
                            lag_bdays=None, lag_days=LIQUIDITY_LAG_DAYS)
            liq_a[sid] = {}
            for era, (s, e) in eras.items():
                liq_a[sid][era] = lead_cell(
                    ch4w, dc, LIQUIDITY_FWD_DAYS, cost, s, e,
                    lag_bdays=None, lag_days=LIQUIDITY_LAG_DAYS, aligned=al)
            liq_a[sid]["pooled"] = lead_cell(
                ch4w, dc, LIQUIDITY_FWD_DAYS, cost, eras["pre_design"][0], None,
                lag_bdays=None, lag_days=LIQUIDITY_LAG_DAYS, aligned=al)
        report["liquidity"][asset] = liq_a

        # -- events (FOMC only; CPI skipped, see meta)
        report["events"][asset] = {"FOMC_2026": event_bars(c4, FOMC_2026)}

        # -- controls
        ctl = {"planted_lead": None, "shuffled": None}
        pf = planted_factor(dc)
        al = align_lead(pf, dc, 1)
        ctl["planted_lead"] = {
            "n": int(len(al)),
            "ic": round(spearman_ic(al["x"].to_numpy(), al["fwd"].to_numpy()), 4)}
        if "SP500" in changes:
            ctl["shuffled"] = shuffled_control(changes["SP500"], dc)
        report["controls"][asset] = ctl
        if ctl["planted_lead"]["ic"] < 0.9:
            print(f"[P392][WARN] {asset}: planted-lead IC "
                  f"{ctl['planted_lead']['ic']} < 0.9 — alignment suspect (P174)")

        # -- verdicts
        verd_a = {}
        for sid in changes:
            for hz in (1, 5):
                v = decide_verdict(
                    lead_a[sid][f"{hz}d"]["design"],
                    lead_a[sid][f"{hz}d"]["validation"],
                    cont_a[sid]["design"], cont_a[sid]["validation"])
                verd_a[f"{sid}_{hz}d"] = v
        for sid in LIQUIDITY_FACTORS:
            if sid in levels and "design" in liq_a.get(sid, {}):
                ld, lvv = liq_a[sid]["design"], liq_a[sid]["validation"]
                ok = (ld.get("passes") and lvv.get("passes")
                      and (ld.get("ic") or 0) * (lvv.get("ic") or 0) > 0)
                verd_a[f"{sid}_14d"] = "TRADEABLE-LEAD" if ok else "NOISE"
        report["verdicts"][asset] = verd_a

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=1, default=str),
                           encoding="utf-8", newline="\n")

    # ---- printed table ----
    def _f(v, w=7):
        return f"{v:>{w}}" if v is not None else " " * (w - 4) + "  — "
    print("\n[P392] CONTEMPORANEOUS betas (same-day, per era)")
    print(f"{'asset':6s}{'factor':10s}{'era':12s}{'n':>6s}{'beta':>10s}{'t':>7s}{'corr':>7s}{'R2':>6s}")
    for asset in ASSETS:
        for sid in changes:
            for era in ERAS:
                c = report["contemporaneous"][asset][sid][era]
                if "beta" not in c:
                    continue
                print(f"{asset:6s}{sid:10s}{era:12s}{c['n']:>6d}"
                      f"{c['beta']:>10.4f}{c['t']:>7.2f}{c['corr']:>7.3f}{c['r2']:>6.3f}")
    print("\n[P392] LEAD cells (publication-lagged; PASS needs n_eff>=30, |t|>=2, |IC|>=IC_req)")
    print(f"{'asset':6s}{'factor':10s}{'hz':4s}{'era':12s}{'n':>5s}{'IC':>8s}{'t':>7s}"
          f"{'IC_req':>8s}{'edge':>7s}{'sigma':>7s}{'pass':>6s}")
    for asset in ASSETS:
        for sid in changes:
            for hz in ("1d", "5d"):
                for era in list(ERAS) + ["pooled"]:
                    c = report["lead"][asset][sid][hz][era]
                    if "ic" not in c:
                        continue
                    print(f"{asset:6s}{sid:10s}{hz:4s}{era:12s}{c['n']:>5d}"
                          f"{_f(c['ic'], 8)}{_f(c['t'])}{_f(c['ic_required'], 8)}"
                          f"{_f(c['implied_edge_bps'])}{c['sigma_fwd_bps']:>7.0f}"
                          f"{'PASS' if c['passes'] else '.':>6s}")
    print("\n[P392] LIQUIDITY (4w change, 1w lag, 14d fwd)")
    for asset in ASSETS:
        for sid in LIQUIDITY_FACTORS:
            cell = report["liquidity"][asset].get(sid, {})
            for era in list(ERAS) + ["pooled"]:
                c = cell.get(era)
                if not c or "ic" not in c:
                    continue
                print(f"{asset:6s}{sid:10s}14d {era:12s}{c['n']:>5d}"
                      f"{_f(c['ic'], 8)}{_f(c['t'])}{_f(c['ic_required'], 8)}"
                      f"{'PASS' if c['passes'] else '.':>6s}")
    print("\n[P392] FOMC-2026 event bars (16:00/20:00 UTC)")
    for asset in ASSETS:
        e = report["events"][asset]["FOMC_2026"]
        print(f"  {asset}: {e}")
    print("\n[P392] controls")
    for asset in ASSETS:
        print(f"  {asset}: planted={report['controls'][asset]['planted_lead']}"
              f" shuffled={report['controls'][asset]['shuffled']}")
    print("\n[P392] VERDICTS (pre-committed rule)")
    for asset in ASSETS:
        for k, v in report["verdicts"][asset].items():
            if v != "NOISE":
                print(f"  {asset} {k}: {v}")
    noise_n = sum(1 for a in ASSETS
                  for v in report["verdicts"][a].values() if v == "NOISE")
    print(f"  (NOISE cells: {noise_n})")
    print(f"\n[P392] report -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

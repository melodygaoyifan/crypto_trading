"""
================================================================================
HMATS [P296] - The sentiment exam, settled on HISTORY instead of a 30-day clock
================================================================================

WHY THIS EXISTS
    P293e shipped `sentvariant_*.jsonl`: three competing readings of Fear &
    Greed recorded live, to be judged by the P166 cost-aware gate after 30
    days. P293g then measured that such a clock CANNOT FIRE - at a 16h
    horizon, |t| = IC*sqrt(n_eff-1) with n_eff = n/h needs ~370 days of one
    asset to certify an economically adequate IC. A 30-day ledger against a
    16h claim is a check that cannot pass.

    But the three claims need no forward data at all. The same free endpoint
    that serves today's reading serves 3,116 days back to 2018-02-01, and the
    three variants have ZERO FREE PARAMETERS - every threshold is inherited
    verbatim from the live code, not fitted here. So the usual objection to
    backtesting this repo has learned the hard way (P164 leak, P243/P244
    era-collapse, P259b, P281, P283b->P285c) does not apply with its usual
    force: there is no search, so there is nothing for a search to overfit.
    Multiplicity is 3, not thousands.

    That is the difference between this and the campaigns that produced false
    positives: they FIT on the window they then read. This one only READS.

DERIVATIVES EXPRESSION (the operator's instruction: focus on derivatives)
    The book is Coinbase US perp-style futures, so the exam is scored the way
    the venue actually settles, not on paper returns:

      * position is SIGN-QUANTIZED to {-1, 0, +1} - the sleeve sizes by sign
        and discards magnitude (P273/P293d), so a continuous claim would be
        measuring a strategy the system cannot express;
      * taker cost per leg = HALF the measured CDE spread (P289: full spreads
        BTC 2.0 / ETH 5.5 / SOL 4.0 bps) + the 3bps Coinbase taker fee;
      * FUNDING CARRY is charged every bar a position is held, at the P245
        convention (8h rate / 2 per 4H bar, longs pay when funding is
        positive). Binance funding is a documented PROXY for the CDE contract
        and P218 measured that the two can differ in SIGN - so carry is
        reported as its own column and the verdict is stated both ways.

    Both are the P283b "decisive cell": LIVE expression, WITH carry.

CAUSALITY
    The F&G value stamped day D drives positions only from D+1 00:00 UTC
    onward. The historical z-score for day D is scored against the trailing
    window ENDING at D - which is exactly what the live feed does, since it
    has just fetched today's value - and is therefore knowable at D. Nothing
    reads a bar it could not have seen. A construction test asserts it.

PRE-COMMITTED VERDICT (fixed BEFORE the first run - P285b)
    A variant PASSES only if ALL of:
      1. net after-cost, after-carry return > 0 on the pooled book;
      2. the block-bootstrap 90% CI on its Sharpe EXCLUDES zero;
      3. it is positive in BOTH halves of the sample (era stability - the
         P243/P244/P259b lesson: a full-sample winner that dies in one half
         is an era artifact, not an edge);
      4. it beats FLAT (no position) after costs, which is the only baseline
         a long/flat/short sentiment rule has to clear.
    Failing any one is a FAIL, whatever the headline number says.

    A RANDOM-SIGN control is scored alongside as an anti-vacuity check
    (P174): if the harness cannot produce a losing verdict for coin flips
    charged real costs, its passes mean nothing.

USAGE
    python -X utf8 training/sentiment_history_lab.py
    python -X utf8 training/sentiment_history_lab.py --assets BTC,ETH,SOL
================================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

# [P172] SINGLE SOURCE. The variant rules are imported from the module the
# LIVE ledger uses, never restated here - a second copy is how a lab and the
# thing it certifies start disagreeing about what was tested.
from defense.sentiment_variant_shadow import (
    STRATEGY_CONTRARIAN,
    STRATEGY_MOMENTUM_HIST,
    STRATEGY_MOMENTUM_LINEAR,
    contrarian_direction,
    momentum_direction,
)
from data_mgmt.feeds.fear_greed_history import FearGreedHistory

# [P289] Measured CDE full spreads. The taker pays HALF.
CDE_SPREAD_BPS = {"BTC": 2.0, "ETH": 5.5, "SOL": 4.0}
COINBASE_TAKER_FEE_BPS = 3.0
DEFAULT_SPREAD_BPS = 5.0          # unmeasured asset -> the expensive side

BARS_PER_DAY = 6                  # 4H
BARS_PER_YEAR = BARS_PER_DAY * 365

PRICE_DIR = REPO / "training" / "training_data" / "drl_training"
FUNDING_DIR = REPO / "training" / "training_data" / "coinglass_history"
FNG_PATH = REPO / "data" / "fear_greed_history.json"


# =============================================================================
# LOADERS
# =============================================================================

def load_fng(path: Path = FNG_PATH) -> pd.Series:
    """Daily F&G as a date-indexed Series."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    series = payload.get("series") or {}
    if not series:
        raise SystemExit(f"no F&G series in {path}")
    s = pd.Series(
        {pd.Timestamp(k, tz="UTC"): float(v) for k, v in series.items()}
    ).sort_index()
    return s


def load_closes(asset: str) -> pd.Series:
    """4H closes, UTC-indexed."""
    p = PRICE_DIR / f"{asset}_4H_ohlcv.parquet"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        tcol = next((c for c in df.columns
                     if "time" in c.lower() or "date" in c.lower()), None)
        if tcol is None:
            raise SystemExit(f"{p}: no timestamp column")
        df = df.set_index(pd.to_datetime(df[tcol]))
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    ccol = next((c for c in df.columns if c.lower() == "close"), None)
    if ccol is None:
        raise SystemExit(f"{p}: no close column")
    return pd.Series(df[ccol].to_numpy(dtype=float), index=idx).sort_index()


def load_funding_daily(asset: str) -> Optional[pd.Series]:
    """Daily funding rate (8h convention), or None when unavailable.

    None is NOT zero: a missing rate means the carry leg cannot be priced for
    that asset, and the caller says so rather than quietly charging nothing.
    """
    p = FUNDING_DIR / f"{asset}_funding_1d.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    tcol = next((c for c in df.columns if "time" in c.lower()), None)
    vcol = next((c for c in df.columns if c.lower() == "funding_close"), None)
    if tcol is None or vcol is None:
        return None
    idx = pd.to_datetime(df[tcol])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize("UTC")
    return pd.Series(df[vcol].to_numpy(dtype=float),
                     index=pd.DatetimeIndex(idx)).sort_index()


# =============================================================================
# SIGNAL CONSTRUCTION (causal)
# =============================================================================

def build_daily_directions(fng: pd.Series,
                           window_days: int = 365,
                           min_samples: int = 60) -> pd.DataFrame:
    """Per DAY, the three variants' directions.

    The historical z reuses the LIVE scorer (`FearGreedHistory.score`) fed a
    series truncated at that day, so the lab cannot drift from the runtime
    (P172) and cannot see the future (P164).
    """
    hist = FearGreedHistory(data_dir=str(Path(os.devnull).parent),
                            window_days=window_days,
                            min_samples=min_samples)
    rows = []
    running: Dict[str, float] = {}
    for ts, val in fng.items():
        day = ts.strftime("%Y-%m-%d")
        running[day] = float(val)          # today is knowable today
        hist._series = running
        stats = hist.score(float(val))
        z_lin = (float(val) - 50.0) / 50.0 * 3.0
        rows.append({
            "day": ts,
            "fg": float(val),
            "z_linear": z_lin,
            "z_hist": stats.zscore,
            STRATEGY_MOMENTUM_LINEAR: momentum_direction(z_lin),
            STRATEGY_MOMENTUM_HIST: (momentum_direction(stats.zscore)
                                     if stats.zscore is not None else np.nan),
            STRATEGY_CONTRARIAN: contrarian_direction(float(val)),
        })
    return pd.DataFrame(rows).set_index("day")


# =============================================================================
# DERIVATIVES BACKTEST
# =============================================================================

def per_leg_cost_bps(asset: str) -> float:
    spread = CDE_SPREAD_BPS.get(asset.upper(), DEFAULT_SPREAD_BPS)
    return spread / 2.0 + COINBASE_TAKER_FEE_BPS


def run_asset(asset: str,
              directions: pd.DataFrame,
              strategy: str,
              charge_carry: bool = True) -> Optional[pd.DataFrame]:
    """Bar-by-bar perp PnL for one asset/variant. Returns None if unusable."""
    closes = load_closes(asset)
    ret = closes.pct_change().shift(-1)          # return of the NEXT bar

    # The value stamped day D drives bars from D+1 00:00 UTC (publication lag)
    d = directions[[strategy]].copy()
    d.index = d.index + pd.Timedelta(days=1)
    pos = d[strategy].reindex(closes.index, method="ffill")

    df = pd.DataFrame({"close": closes, "ret": ret, "pos": pos}).dropna(
        subset=["ret", "pos"])
    if df.empty:
        return None

    leg_bps = per_leg_cost_bps(asset)
    turn = df["pos"].diff().abs().fillna(df["pos"].abs())
    cost = turn * (leg_bps / 1e4)

    carry = pd.Series(0.0, index=df.index)
    fund = load_funding_daily(asset)
    carry_priced = fund is not None
    if charge_carry and carry_priced:
        f_bar = fund.reindex(df.index, method="ffill") / 2.0   # [P245] 8h/2
        # long pays when funding is positive
        carry = -df["pos"] * f_bar.fillna(0.0)

    df["gross"] = df["pos"] * df["ret"]
    df["cost"] = cost
    df["carry"] = carry
    df["net"] = df["gross"] - df["cost"] + df["carry"]
    df.attrs["carry_priced"] = carry_priced
    return df


def _sharpe(x: pd.Series) -> float:
    sd = x.std(ddof=1)
    if not (sd > 0):
        return 0.0
    return float(x.mean() / sd * math.sqrt(BARS_PER_YEAR))


def _block_bootstrap_ci(x: np.ndarray, block: int = 90,
                        n: int = 2000, lo: float = 5.0,
                        hi: float = 95.0, seed: int = 7) -> Tuple[float, float]:
    """Percentile CI on the annualized Sharpe, blocks preserving autocorr."""
    rng = np.random.default_rng(seed)
    m = len(x)
    if m < block * 3:
        return (float("nan"), float("nan"))
    nblocks = m // block
    out = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, m - block, size=nblocks)
        samp = np.concatenate([x[s:s + block] for s in starts])
        sd = samp.std(ddof=1)
        out[i] = 0.0 if sd <= 0 else samp.mean() / sd * math.sqrt(BARS_PER_YEAR)
    return (float(np.percentile(out, lo)), float(np.percentile(out, hi)))


def evaluate(strategy: str, assets: List[str], directions: pd.DataFrame,
             charge_carry: bool = True) -> Dict[str, Any]:
    """Pooled equal-weight book + per-asset detail."""
    frames = {}
    for a in assets:
        f = run_asset(a, directions, strategy, charge_carry=charge_carry)
        if f is not None and not f.empty:
            frames[a] = f
    if not frames:
        return {"strategy": strategy, "error": "no usable asset"}

    pooled = pd.concat({a: f["net"] for a, f in frames.items()},
                       axis=1).dropna(how="all")
    book = pooled.mean(axis=1).dropna()          # equal-weight book

    halves = np.array_split(book.to_numpy(), 2)
    lo, hi = _block_bootstrap_ci(book.to_numpy())
    total_gross = float(sum(f["gross"].sum() for f in frames.values())
                        / max(len(frames), 1))
    total_cost = float(sum(f["cost"].sum() for f in frames.values())
                       / max(len(frames), 1))
    total_carry = float(sum(f["carry"].sum() for f in frames.values())
                        / max(len(frames), 1))

    res = {
        "strategy": strategy,
        "assets": sorted(frames),
        "bars": int(len(book)),
        "years": round(len(book) / BARS_PER_YEAR, 2),
        "net_return_pct": round(float(book.sum()) * 100, 2),
        "gross_return_pct": round(total_gross * 100, 2),
        "cost_pct": round(total_cost * 100, 2),
        "carry_pct": round(total_carry * 100, 2),
        "sharpe": round(_sharpe(book), 3),
        "sharpe_ci90": [round(lo, 3), round(hi, 3)],
        "half1_return_pct": round(float(halves[0].sum()) * 100, 2),
        "half2_return_pct": round(float(halves[1].sum()) * 100, 2),
        "exposure_frac": round(float(
            np.mean([f["pos"].abs().mean() for f in frames.values()])), 3),
        "flips_per_asset": round(float(
            np.mean([f["pos"].diff().abs().gt(0).sum() for f in frames.values()])), 1),
        "carry_priced": all(f.attrs.get("carry_priced") for f in frames.values()),
    }
    res["verdict"], res["blockers"] = _verdict(res)
    return res


def _verdict(r: Dict[str, Any]) -> Tuple[str, List[str]]:
    """The PRE-COMMITTED rule. See the module docstring; not edited after
    seeing a result."""
    b = []
    if r["net_return_pct"] <= 0:
        b.append(f"net after cost+carry {r['net_return_pct']:+.2f}% <= 0")
    lo, hi = r["sharpe_ci90"]
    if not (lo == lo and hi == hi):
        b.append("Sharpe CI unavailable (sample too short)")
    elif lo <= 0 <= hi:
        b.append(f"Sharpe CI [{lo:+.2f},{hi:+.2f}] includes zero")
    if r["half1_return_pct"] <= 0 or r["half2_return_pct"] <= 0:
        b.append(f"era-unstable: halves {r['half1_return_pct']:+.2f}% / "
                 f"{r['half2_return_pct']:+.2f}%")
    return ("PASS" if not b else "FAIL"), b


def random_control(assets: List[str], directions: pd.DataFrame,
                   seed: int = 11,
                   flip_prob: Optional[float] = None,
                   label: str = "random_control") -> Dict[str, Any]:
    """[P174] Random signs charged real costs. If this can PASS, nothing here
    means anything.

    `flip_prob` matches the control's TURNOVER to a real variant's. Without
    it the control changes sign on ~2/3 of days and is destroyed by costs
    alone (measured: 1,485 flips/asset and a 96% cost drag), which tests
    "does churn lose money" - a question nobody asked - instead of "can luck
    pass this gate". A control that fails for the wrong reason is not a
    control.
    """
    rng = random.Random(seed)
    d = directions.copy()
    if flip_prob is None:
        vals = [rng.choice([-1.0, 0.0, 1.0]) for _ in range(len(d))]
    else:
        vals, cur = [], rng.choice([-1.0, 1.0])
        for _ in range(len(d)):
            if rng.random() < flip_prob:
                cur = rng.choice([-1.0, 0.0, 1.0])
            vals.append(cur)
    d["__random__"] = vals
    r = evaluate("__random__", assets, d)
    r["strategy"] = label
    return r


def baseline_buy_and_hold(assets: List[str], directions: pd.DataFrame,
                          charge_carry: bool = True) -> Dict[str, Any]:
    """[P182] The baseline this lab shipped WITHOUT, and the one that matters.

    Every variant here is long-biased and the sample is 2020-2026, so "beats
    flat" is nearly free: a rule that is long two-thirds of the time in a
    bull market scores well for reasons that have nothing to do with
    sentiment. P182 added exactly this baseline to the DRL trainer after
    "under-performs buy-and-hold" turned out not to be an outcome that
    harness could produce. Charged the same carry and one entry leg.
    """
    d = directions.copy()
    d["__bh__"] = 1.0
    r = evaluate("__bh__", assets, d, charge_carry=charge_carry)
    r["strategy"] = "buy_and_hold"
    return r


# =============================================================================
# MAIN
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", default="BTC,ETH,SOL")
    ap.add_argument("--no-carry", action="store_true",
                    help="report the same exam without the funding leg")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)
    assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()]

    print("=" * 78)
    print("PRE-COMMITTED VERDICT (fixed before this ran):")
    print("  PASS iff net>0 after cost AND carry, Sharpe CI excludes zero,")
    print("  BOTH halves positive, and it beats flat. Any miss = FAIL.")
    print("=" * 78)

    fng = load_fng()
    print(f"\nF&G history : {len(fng)} days  "
          f"{fng.index.min().date()} -> {fng.index.max().date()}")
    directions = build_daily_directions(fng)
    usable_hist = int(directions[STRATEGY_MOMENTUM_HIST].notna().sum())
    print(f"variants    : linear={len(directions)} days, "
          f"historical={usable_hist} days (the rest are below min_samples)")

    results = []
    for strat in (STRATEGY_MOMENTUM_LINEAR, STRATEGY_MOMENTUM_HIST,
                  STRATEGY_CONTRARIAN):
        results.append(evaluate(strat, assets, directions,
                                charge_carry=not args.no_carry))
    # Baselines. B&H is the decisive one for a long-biased rule (P182).
    results.append(baseline_buy_and_hold(assets, directions,
                                         charge_carry=not args.no_carry))
    results.append(random_control(assets, directions))
    # Turnover-matched control: same flip rate as the live rule, random signs.
    _lin = next((x for x in results
                 if x["strategy"] == STRATEGY_MOMENTUM_LINEAR), None)
    if _lin and not _lin.get("error") and len(directions):
        _fp = min(1.0, max(0.0, _lin["flips_per_asset"] / max(len(directions), 1)))
        results.append(random_control(assets, directions, seed=23,
                                      flip_prob=_fp,
                                      label="random_turnover_matched"))

    print(f"\nDERIVATIVES EXPRESSION: sign-quantized +/-1, taker = half-spread"
          f" + {COINBASE_TAKER_FEE_BPS:.0f}bps, "
          f"carry {'OFF' if args.no_carry else 'ON (P245)'}")
    print(f"per-leg cost bps: "
          + ", ".join(f"{a}={per_leg_cost_bps(a):.2f}" for a in assets))

    hdr = (f"\n{'variant':<24}{'years':>6}{'net%':>10}{'gross%':>9}"
           f"{'cost%':>8}{'carry%':>8}{'Sharpe':>8}{'CI90':>18}"
           f"{'expo':>7}{'verdict':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r.get("error"):
            print(f"{r['strategy']:<24}{r['error']}")
            continue
        ci = f"[{r['sharpe_ci90'][0]:+.2f},{r['sharpe_ci90'][1]:+.2f}]"
        print(f"{r['strategy']:<24}{r['years']:>6.1f}{r['net_return_pct']:>10.2f}"
              f"{r['gross_return_pct']:>9.2f}{r['cost_pct']:>8.2f}"
              f"{r['carry_pct']:>8.2f}{r['sharpe']:>8.2f}{ci:>18}"
              f"{r['exposure_frac']:>7.2f}{r['verdict']:>9}")

    print("\nBLOCKERS")
    for r in results:
        if r.get("error"):
            continue
        if r["verdict"] == "PASS":
            print(f"  {r['strategy']:<24} (none)")
        for x in r.get("blockers", []):
            print(f"  {r['strategy']:<24} {x}")

    print("\nHALVES (era stability)")
    for r in results:
        if not r.get("error"):
            print(f"  {r['strategy']:<24}"
                  f"h1={r['half1_return_pct']:>8.2f}%   "
                  f"h2={r['half2_return_pct']:>8.2f}%   "
                  f"flips/asset={r['flips_per_asset']:.0f}")

    out = args.output or str(REPO / "training" / "reports" /
                             "sentiment_history_lab_p296.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "fng_days": int(len(fng)),
        "fng_span": [str(fng.index.min().date()), str(fng.index.max().date())],
        "assets": assets,
        "carry": not args.no_carry,
        "cost_model": {"cde_full_spread_bps": CDE_SPREAD_BPS,
                       "taker_fee_bps": COINBASE_TAKER_FEE_BPS},
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

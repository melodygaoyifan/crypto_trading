"""
================================================================================
HMATS [P297] - The BTC funding legs, settled on 6 years of funding history
================================================================================

THE OPEN QUESTION THIS CLOSES
    P262 certified the trend/hold MECHANISM on two out-of-selection axes (the
    2017-2020 virgin era and five never-fitted assets) and then said plainly
    what it could NOT certify:

        "The BTC funding legs - the entire excess of +33.6% over trend-only's
         +16.6% - remain uncertified: virgin-era evidence is thin-and-negative
         (active only ~9 usable months, and they SUBTRACTED in 2020), DSR says
         the window cannot vouch for them, and P244 already measured
         funding-cell era-instability."

    So the BTC book's whole margin over the certified mechanism rests on the
    one component nobody could certify - and it is live. The funding history
    to settle it (2,191 daily rows, 2020-08 -> 2026-08) has been on disk the
    whole time.

WHAT IS ACTUALLY UNDER TEST: THE INCREMENT, NOT THE BOOK
    `book` and `trend_only` agree in every BULL bar (both hold long), so
    comparing their totals mostly compares the trend mechanism to itself. The
    funding legs live ONLY in the bear and peace cells. This lab scores the
    INCREMENT - book minus trend_only, bar by bar - which is the marginal PnL
    of the funding cells and nothing else (the P259 overlay-increment
    framing).

    The decisive control follows from that: RANDOM SIGNS IN THE SAME CELLS,
    with the same turnover. If a coin flip does as well in the bear/peace
    bars, the funding signal adds nothing and the +33.6% was the trend
    mechanism plus noise.

SINGLE SOURCE (P172)
    `regime_label`, `causal_funding_z` and `book_target` are IMPORTED from
    `defense/regime_book_shadow.py` - the module the live ledger runs. This
    lab cannot test a book that differs from the deployed one, and the
    funding z keeps its documented no-in-progress-day rule (the P247-F1 leak).

DERIVATIVES EXPRESSION
    Same decisive cell as P296: sign-quantized +/-1 (the sleeve sizes by sign),
    taker = half the measured CDE spread (P289) + 3bps fee, funding carry
    charged on every held bar (P245, Binance proxy with the P218 sign caveat).
    A short in a positive-funding regime EARNS carry, which matters here
    because the funding legs are short-biased by construction.

PRE-COMMITTED VERDICT (fixed BEFORE the first run - P285b)
    The funding legs PASS only if ALL of:
      1. the INCREMENT is net positive after cost and carry on the full
         sample;
      2. it is positive in the DESIGN era AND in at least one era it was not
         selected on (they were chosen on design, so design alone proves
         nothing - P244/P259b);
      3. the block-bootstrap 90% CI on the increment's Sharpe excludes zero;
      4. it beats the SAME-CELL random control, which is the only baseline
         that isolates the funding signal from the regime mask it rides on.
    Any miss is a FAIL. A FAIL means the BTC book should be trend-only, which
    is the P262-certified mechanism - i.e. failing here REMOVES a component,
    it does not stop the book trading.

USAGE
    python -X utf8 training/funding_legs_lab.py
    python -X utf8 training/funding_legs_lab.py --asset BTC
================================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

# [P172] the LIVE book, imported - never restated
from defense.regime_book_shadow import (
    FUND_Z_WINDOW,
    MIN_BARS,
    book_target,
    causal_funding_z,
    regime_label,
)

CDE_SPREAD_BPS = {"BTC": 2.0, "ETH": 5.5, "SOL": 4.0}
COINBASE_TAKER_FEE_BPS = 3.0
BARS_PER_DAY = 6
BARS_PER_YEAR = BARS_PER_DAY * 365

PRICE_DIR = REPO / "training" / "training_data" / "drl_training"
FUNDING_DIR = REPO / "training" / "training_data" / "coinglass_history"

# [P250/P262] the lab's era convention, as bar indices into the 4H series.
ERAS = {"pre_design": (800, 3000), "design": (3000, 9100), "validation": (9100, None)}


def per_leg_cost_bps(asset: str) -> float:
    return CDE_SPREAD_BPS.get(asset.upper(), 5.0) / 2.0 + COINBASE_TAKER_FEE_BPS


# [P315] CDE charges a FLAT FEE PER CONTRACT, not a percentage of notional.
# Measured from the venue's own reported fees (data/fill_quality.jsonl):
# BTC ~$0.60/ct -> 9.4bps at $64k, ETH ~$0.26/ct -> 13.8bps at $1.9k. The
# constant above (3bps) understates the real charge ~3x, and because the fee
# is flat its bps cost moves INVERSELY with price: at BTC $10k a 0.01 nano
# contract is ~$100 of notional, so the same $0.60 is ~60bps. No single bps
# constant can price six years of history — hence a per-bar SERIES.
FEE_MODEL = "per_contract"          # "legacy_3bps" reproduces the P297 run


def per_leg_cost_series(asset: str, closes):
    """Per-leg cost as a FRACTION, per bar. Half-spread + the honest fee."""
    import pandas as pd  # local: keeps the legacy path import-free
    half_spread = CDE_SPREAD_BPS.get(asset.upper(), 5.0) / 2.0 / 1e4
    if FEE_MODEL == "legacy_3bps":
        return pd.Series(half_spread + COINBASE_TAKER_FEE_BPS / 1e4,
                         index=closes.index)
    from core.cde_fees import CDE_FEE_PER_CONTRACT_USD, _contract_sizes
    a = asset.upper()
    per_ct = CDE_FEE_PER_CONTRACT_USD.get(a, {}).get("taker")
    cs = _contract_sizes().get(a)
    if not per_ct or not cs:
        raise SystemExit(
            f"[P315] no per-contract fee/contract size for {a}. Refusing to "
            f"fall back to the 3bps constant silently — that is the defect "
            f"this model exists to correct (P167: never undercharge).")
    fee_frac = per_ct / (cs * closes.astype(float))
    return half_spread + fee_frac


# =============================================================================
# DATA
# =============================================================================

def load_closes(asset: str) -> pd.Series:
    p = PRICE_DIR / f"{asset}_4H_ohlcv.parquet"
    df = pd.read_parquet(p)
    if not isinstance(df.index, pd.DatetimeIndex):
        tcol = next(c for c in df.columns
                    if "time" in c.lower() or "date" in c.lower())
        df = df.set_index(pd.to_datetime(df[tcol]))
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    ccol = next(c for c in df.columns if c.lower() == "close")
    return pd.Series(df[ccol].to_numpy(dtype=float), index=idx).sort_index()


def load_funding_daily(asset: str) -> pd.Series:
    p = FUNDING_DIR / f"{asset}_funding_1d.parquet"
    df = pd.read_parquet(p)
    tcol = next(c for c in df.columns if "time" in c.lower())
    idx = pd.to_datetime(df[tcol])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize("UTC")
    return pd.Series(df["funding_close"].to_numpy(dtype=float),
                     index=pd.DatetimeIndex(idx)).sort_index()


# =============================================================================
# POSITION SERIES
# =============================================================================

def build_positions(asset: str, closes: pd.Series,
                    funding: pd.Series) -> pd.DataFrame:
    """Bar-by-bar regime, funding z, book target and trend-only target.

    Causality: the funding z at bar t uses daily closes strictly BEFORE t's
    UTC date, i.e. the last COMPLETED day - the module's own documented rule
    and the P247-F1 leak it was written to avoid.
    """
    c = closes.to_numpy(dtype=float)
    days = funding.index.normalize()
    fmap = {d: v for d, v in zip(days, funding.to_numpy(dtype=float))}
    ordered_days = sorted(fmap)

    rows = []
    for i, ts in enumerate(closes.index):
        if i < MIN_BARS:
            continue
        window = c[max(0, i - MIN_BARS): i + 1]
        reg = regime_label(list(window))
        # strictly completed days only
        cutoff = ts.normalize()
        hist = [fmap[d] for d in ordered_days if d < cutoff]
        fz = causal_funding_z(hist) if len(hist) >= FUND_Z_WINDOW else None
        bt, leg = book_target(asset, reg, fz)
        trend = 1.0 if reg == "bull" else 0.0
        rows.append({"ts": ts, "i": i, "regime": reg, "funding_z": fz,
                     "book": float(bt), "trend": trend, "leg": leg})
    return pd.DataFrame(rows).set_index("ts")


def pnl(asset: str, pos: pd.Series, closes: pd.Series,
        funding: pd.Series) -> pd.DataFrame:
    ret = closes.pct_change().shift(-1).reindex(pos.index)
    # [P315] price-dependent per-leg cost (see per_leg_cost_series).
    leg = per_leg_cost_series(asset, closes).reindex(pos.index).ffill()
    turn = pos.diff().abs().fillna(pos.abs())
    f_bar = funding.reindex(pos.index, method="ffill") / 2.0   # [P245]
    df = pd.DataFrame({
        "gross": pos * ret,
        "cost": turn * leg,
        "carry": -pos * f_bar.fillna(0.0),
    }).dropna(subset=["gross"])
    df["net"] = df["gross"] - df["cost"] + df["carry"]
    return df


def _sharpe(x: pd.Series) -> float:
    sd = x.std(ddof=1)
    return 0.0 if not (sd > 0) else float(x.mean() / sd * math.sqrt(BARS_PER_YEAR))


def _ci(x: np.ndarray, block: int = 90, n: int = 2000, seed: int = 7):
    rng = np.random.default_rng(seed)
    m = len(x)
    if m < block * 3:
        return (float("nan"), float("nan"))
    nb = m // block
    out = np.empty(n)
    for i in range(n):
        st = rng.integers(0, m - block, size=nb)
        s = np.concatenate([x[a:a + block] for a in st])
        sd = s.std(ddof=1)
        out[i] = 0.0 if sd <= 0 else s.mean() / sd * math.sqrt(BARS_PER_YEAR)
    return (float(np.percentile(out, 5)), float(np.percentile(out, 95)))


def summarize(name: str, net: pd.Series, frame: pd.DataFrame) -> Dict[str, Any]:
    lo, hi = _ci(net.to_numpy())
    return {
        "name": name,
        "bars": int(len(net)),
        "net_pct": round(float(net.sum()) * 100, 2),
        "gross_pct": round(float(frame["gross"].sum()) * 100, 2),
        "cost_pct": round(float(frame["cost"].sum()) * 100, 2),
        "carry_pct": round(float(frame["carry"].sum()) * 100, 2),
        "sharpe": round(_sharpe(net), 3),
        "ci90": [round(lo, 3), round(hi, 3)],
    }


# =============================================================================
# MAIN
# =============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)
    A = args.asset.upper()

    print("=" * 78)
    print("PRE-COMMITTED VERDICT (fixed before this ran):")
    print("  The funding legs PASS only if the INCREMENT (book - trend_only) is")
    print("  net>0 after cost+carry, positive in DESIGN *and* a non-design era,")
    print("  its Sharpe CI excludes zero, AND it beats a same-cell random control.")
    print("  A FAIL means the book should be trend-only - the P262-certified")
    print("  mechanism - so failing REMOVES a component, it does not stop trading.")
    print("=" * 78)

    closes = load_closes(A)
    funding = load_funding_daily(A)
    print(f"\n{A}: {len(closes)} 4H bars {closes.index.min().date()} -> "
          f"{closes.index.max().date()}")
    print(f"{A}: {len(funding)} daily funding rows "
          f"{funding.index.min().date()} -> {funding.index.max().date()}")

    pos = build_positions(A, closes, funding)
    active = pos["book"] != pos["trend"]
    print(f"\nregime census: " + ", ".join(
        f"{k}={v}" for k, v in pos['regime'].value_counts().items()))
    print(f"funding legs differ from trend-only on {int(active.sum())} of "
          f"{len(pos)} bars ({active.mean()*100:.1f}%)")
    print("leg census: " + ", ".join(
        f"{k}={v}" for k, v in pos['leg'].value_counts().items()))

    book_p = pnl(A, pos["book"], closes, funding)
    trend_p = pnl(A, pos["trend"], closes, funding)
    bh_p = pnl(A, pd.Series(1.0, index=pos.index), closes, funding)
    inc = (book_p["net"] - trend_p["net"]).dropna()
    inc_frame = (book_p[["gross", "cost", "carry"]]
                 - trend_p[["gross", "cost", "carry"]])

    # same-cell random control: random signs ONLY where the legs are active
    rng = random.Random(17)
    rnd = pos["trend"].copy()
    rnd[active] = [rng.choice([-1.0, 0.0, 1.0]) for _ in range(int(active.sum()))]
    rnd_p = pnl(A, rnd, closes, funding)
    rnd_inc = (rnd_p["net"] - trend_p["net"]).dropna()
    rnd_inc_frame = (rnd_p[["gross", "cost", "carry"]]
                     - trend_p[["gross", "cost", "carry"]])

    rows = [
        summarize("book (trend + funding legs)", book_p["net"], book_p),
        summarize("trend_only (P262-certified)", trend_p["net"], trend_p),
        summarize("buy_and_hold", bh_p["net"], bh_p),
        summarize("INCREMENT (funding legs)", inc, inc_frame),
        summarize("increment: random same-cell", rnd_inc, rnd_inc_frame),
    ]
    hdr = (f"\n{'series':<30}{'net%':>10}{'gross%':>9}{'cost%':>8}"
           f"{'carry%':>8}{'Sharpe':>8}{'CI90':>18}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ci = f"[{r['ci90'][0]:+.2f},{r['ci90'][1]:+.2f}]"
        print(f"{r['name']:<30}{r['net_pct']:>10.2f}{r['gross_pct']:>9.2f}"
              f"{r['cost_pct']:>8.2f}{r['carry_pct']:>8.2f}"
              f"{r['sharpe']:>8.2f}{ci:>18}")

    print(f"\n{'ERA':<14}{'book%':>10}{'trend%':>10}{'increment%':>12}"
          f"{'rand inc%':>11}{'B&H%':>10}")
    print("-" * 67)
    era_inc = {}
    for era, (a, b) in ERAS.items():
        idx = pos["i"]
        m = (idx >= a) & ((idx < b) if b else True)
        m = m.reindex(inc.index).fillna(False)
        if not m.any():
            continue
        era_inc[era] = float(inc[m].sum())
        print(f"{era:<14}{book_p['net'][m].sum()*100:>10.2f}"
              f"{trend_p['net'][m].sum()*100:>10.2f}"
              f"{inc[m].sum()*100:>12.2f}{rnd_inc[m].sum()*100:>11.2f}"
              f"{bh_p['net'][m].sum()*100:>10.2f}")

    # verdict
    incr = rows[3]
    rnd_r = rows[4]
    blockers = []
    if incr["net_pct"] <= 0:
        blockers.append(f"increment {incr['net_pct']:+.2f}% <= 0")
    lo, hi = incr["ci90"]
    if lo != lo or (lo <= 0 <= hi):
        blockers.append(f"increment Sharpe CI [{lo:+.2f},{hi:+.2f}] includes zero")
    design = era_inc.get("design", 0.0)
    non_design = [v for k, v in era_inc.items() if k != "design"]
    if design <= 0:
        blockers.append(f"design-era increment {design*100:+.2f}% <= 0")
    if not any(v > 0 for v in non_design):
        blockers.append("no non-design era is positive (selected-era only)")
    if incr["net_pct"] <= rnd_r["net_pct"]:
        blockers.append(f"does not beat same-cell random "
                        f"({incr['net_pct']:+.2f}% vs {rnd_r['net_pct']:+.2f}%)")
    verdict = "PASS" if not blockers else "FAIL"

    print(f"\nVERDICT: funding legs {verdict}")
    for b in blockers:
        print(f"  blocker: {b}")
    if verdict == "FAIL":
        print("  => the BTC book should be TREND-ONLY (the P262-certified")
        print("     mechanism). This removes a component; it does not stop trading.")

    # [P315] The report name carries the FEE MODEL. Before this, any re-run
    # overwrote `funding_legs_lab_p297_{A}.json` in place — so re-pricing the
    # lab silently replaced the very artifact that recorded P297's verdict,
    # and the two numbers could never be compared afterwards. A verdict whose
    # inputs can be overwritten without trace is not auditable (the P299/P309
    # rule about archives, applied to the lab's own output).
    _suffix = "p297" if FEE_MODEL == "legacy_3bps" else "p315_per_contract"
    out = args.output or str(REPO / "training" / "reports" /
                             f"funding_legs_lab_{_suffix}_{A}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "asset": A, "rows": rows, "era_increment": era_inc,
        "verdict": verdict, "blockers": blockers,
        "active_bars": int(active.sum()), "total_bars": int(len(pos)),
    }, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

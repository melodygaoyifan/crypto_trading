"""
================================================================================
HMATS [P301] - The five certified breadth assets, judged as PERPS with carry
================================================================================

WHAT WAS MISSING
    P262 certified the trend/hold mechanism on data no selection ever touched:
    the five never-fitted assets (XRP, ADA, LTC, DOGE, BNB) beat flat 5/5
    after 10bps round-trip costs. That is the strongest out-of-selection
    evidence in the project - and it was measured WITHOUT funding, because
    `coinglass_history` carried funding for BTC/ETH/SOL only.

    P296 then measured what that omission is worth on a long-biased perp
    book: a permanent long paid -59.7% in funding over six years. So "beats
    flat" on a spot-shaped return says little about whether these would earn
    on the venue we actually trade. P301 fetched their funding (2,191 daily
    rows each, 2020-08 -> 2026-07, same span as the majors) and re-runs the
    exam the way the book settles.

SINGLE SOURCE (P172)
    `regime_label` and `book_target` are IMPORTED from
    `defense/regime_book_shadow.py`. For a breadth asset `book_target` is the
    P262-certified mechanism verbatim - long in bull, flat otherwise - so this
    exam cannot test a book that differs from the one the harness records.

PRICES
    Resampled from the 6-year raw 60m archives, NOT from the
    `*_4H_ohlcv.parquet` files, which hold only 725 Kraken-sourced bars for
    the breadth assets (P271 built those for the September scorer). The
    resample convention is validated against a known-good 4H parquet: BTC
    reproduces 13,146 overlapping bars at 0.0 max abs diff.

COSTS - AND THE ONE HONEST ASSUMPTION
    Fees and carry are measured. The SPREAD is not: P289 probed CDE spreads
    for BTC/ETH/SOL only, and P291's breadth probe read contract specs, not
    books. Thin alts trade wider, so this assumes a 10bps FULL spread (5bps
    to the taker) - roughly 2.5x BTC's measured 2.0 - and reports a
    sensitivity at 2x that. An unmeasured cost is assumed EXPENSIVE (P167).

PRE-COMMITTED VERDICT (fixed BEFORE the first run - P285b)
    An asset PASSES only if ALL of:
      1. net after cost AND carry > 0;
      2. it beats BUY-AND-HOLD after the same costs and carry (the baseline
         P182 exists to force, and the one P296 showed a long-biased rule
         clears only by accident against `flat`);
      3. positive in BOTH halves of the sample (era stability);
      4. it beats a TURNOVER-MATCHED random control on the same asset.
    Any miss is a FAIL. A FAIL does not retire the P262 certification - that
    was a claim about beating FLAT on spot-shaped returns, and this is a
    different, harder question about a perp book.

USAGE
    python -X utf8 training/breadth_exam.py
    python -X utf8 training/breadth_exam.py --spread-bps 20
================================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from defense.regime_book_shadow import MIN_BARS, book_target, regime_label

RAW_DIR = REPO / "training" / "training_data" / "raw"
FUNDING_DIR = REPO / "training" / "training_data" / "coinglass_history"

BREADTH = ("XRP", "ADA", "LTC", "DOGE", "BNB")
MAJORS = ("BTC", "ETH", "SOL")
MEASURED_SPREAD_BPS = {"BTC": 2.0, "ETH": 5.5, "SOL": 4.0}   # [P289]
TAKER_FEE_BPS = 3.0
BARS_PER_YEAR = 6 * 365


def load_closes(asset: str) -> pd.Series:
    """4H closes resampled from the 6-year 60m archive."""
    df = pd.read_parquet(RAW_DIR / f"{asset}_60m.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        tcol = next(c for c in df.columns if "time" in c.lower())
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df[tcol])))
    s = df["close"].astype(float).resample("4h", origin="start_day").last().dropna()
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s


def load_funding(asset: str) -> Optional[pd.Series]:
    p = FUNDING_DIR / f"{asset}_funding_1d.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    idx = pd.to_datetime(df["timestamp"])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize("UTC")
    # [P301] .to_numpy(), NOT the Series. Passing a Series as `data` together
    # with an `index` makes pandas ALIGN on the Series' own index rather than
    # replace it, so every value came back NaN -> fillna(0.0) -> carry
    # reported as exactly 0.0% for all eight assets. A silent zero in the one
    # column this exam was built to add; caught only because 0.0 across the
    # board is not a plausible six-year funding bill.
    return pd.Series(df["funding_close"].to_numpy(dtype=float),
                     index=pd.DatetimeIndex(idx)).sort_index()


def leg_bps(asset: str, spread_bps: float) -> float:
    full = MEASURED_SPREAD_BPS.get(asset, spread_bps)
    return full / 2.0 + TAKER_FEE_BPS


def book_positions(asset: str, closes: pd.Series) -> pd.Series:
    c = closes.to_numpy(dtype=float)
    out, idx = [], []
    for i, ts in enumerate(closes.index):
        if i < MIN_BARS:
            continue
        reg = regime_label(list(c[max(0, i - MIN_BARS): i + 1]))
        # breadth assets take the trend-only branch inside book_target;
        # funding_z is irrelevant for them and passed as None deliberately.
        tgt, _leg = book_target(asset, reg, None)
        out.append(float(tgt))
        idx.append(ts)
    return pd.Series(out, index=pd.DatetimeIndex(idx))


def pnl(asset: str, pos: pd.Series, closes: pd.Series,
        funding: Optional[pd.Series], spread_bps: float) -> pd.DataFrame:
    ret = closes.pct_change().shift(-1).reindex(pos.index)
    lb = leg_bps(asset, spread_bps) / 1e4
    turn = pos.diff().abs().fillna(pos.abs())
    carry = pd.Series(0.0, index=pos.index)
    if funding is not None:
        carry = -pos * (funding.reindex(pos.index, method="ffill").fillna(0.0) / 2.0)
    df = pd.DataFrame({"gross": pos * ret, "cost": turn * lb,
                       "carry": carry}).dropna(subset=["gross"])
    df["net"] = df["gross"] - df["cost"] + df["carry"]
    return df


def sharpe(x: pd.Series) -> float:
    sd = x.std(ddof=1)
    return 0.0 if not (sd > 0) else float(x.mean() / sd * math.sqrt(BARS_PER_YEAR))


def run(asset: str, spread_bps: float, seed: int = 5) -> Dict[str, Any]:
    closes = load_closes(asset)
    funding = load_funding(asset)
    pos = book_positions(asset, closes)
    book = pnl(asset, pos, closes, funding, spread_bps)
    bh = pnl(asset, pd.Series(1.0, index=pos.index), closes, funding, spread_bps)

    rng = random.Random(seed)
    flips = int(pos.diff().abs().gt(0).sum())
    fp = flips / max(len(pos), 1)
    vals, cur = [], rng.choice([0.0, 1.0])
    for _ in range(len(pos)):
        if rng.random() < fp:
            cur = rng.choice([0.0, 1.0])
        vals.append(cur)
    rnd = pnl(asset, pd.Series(vals, index=pos.index), closes, funding, spread_bps)

    halves = np.array_split(book["net"].to_numpy(), 2)
    blockers = []
    if book["net"].sum() <= 0:
        blockers.append("net <= 0")
    if book["net"].sum() <= bh["net"].sum():
        blockers.append(f"does not beat buy-and-hold "
                        f"({book['net'].sum()*100:+.1f}% vs {bh['net'].sum()*100:+.1f}%)")
    if halves[0].sum() <= 0 or halves[1].sum() <= 0:
        blockers.append(f"era-unstable ({halves[0].sum()*100:+.1f}% / "
                        f"{halves[1].sum()*100:+.1f}%)")
    if book["net"].sum() <= rnd["net"].sum():
        blockers.append("does not beat turnover-matched random")

    return {
        "asset": asset, "bars": int(len(book)),
        "years": round(len(book) / BARS_PER_YEAR, 1),
        "leg_bps": round(leg_bps(asset, spread_bps), 2),
        "carry_priced": funding is not None,
        "net_pct": round(float(book["net"].sum()) * 100, 1),
        "gross_pct": round(float(book["gross"].sum()) * 100, 1),
        "cost_pct": round(float(book["cost"].sum()) * 100, 1),
        "carry_pct": round(float(book["carry"].sum()) * 100, 1),
        "sharpe": round(sharpe(book["net"]), 2),
        "bh_net_pct": round(float(bh["net"].sum()) * 100, 1),
        "bh_carry_pct": round(float(bh["carry"].sum()) * 100, 1),
        "rand_net_pct": round(float(rnd["net"].sum()) * 100, 1),
        "exposure": round(float(pos.abs().mean()), 2),
        "flips": flips,
        "h1_pct": round(float(halves[0].sum()) * 100, 1),
        "h2_pct": round(float(halves[1].sum()) * 100, 1),
        "verdict": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spread-bps", type=float, default=10.0,
                    help="ASSUMED full spread for unmeasured (breadth) assets")
    ap.add_argument("--assets", default=",".join(BREADTH + MAJORS))
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)
    assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()]

    print("=" * 80)
    print("PRE-COMMITTED VERDICT: PASS iff net>0 after cost AND carry, BEATS")
    print("BUY-AND-HOLD, positive in both halves, and beats a turnover-matched")
    print("random control. A FAIL does not retire the P262 certification - that")
    print("was 'beats FLAT' on spot-shaped returns; this is a perp book.")
    print(f"Unmeasured spreads assumed {args.spread_bps:.0f}bps FULL "
          f"(BTC/ETH/SOL use the P289 measurements).")
    print("=" * 80)

    rows = [run(a, args.spread_bps) for a in assets]
    hdr = (f"\n{'asset':<6}{'yrs':>5}{'leg':>6}{'net%':>9}{'B&H%':>9}"
           f"{'rand%':>9}{'carry%':>9}{'Sh':>7}{'expo':>6}{'flips':>7}  verdict")
    print(hdr)
    print("-" * (len(hdr) + 6))
    for r in rows:
        print(f"{r['asset']:<6}{r['years']:>5.1f}{r['leg_bps']:>6.2f}"
              f"{r['net_pct']:>9.1f}{r['bh_net_pct']:>9.1f}{r['rand_net_pct']:>9.1f}"
              f"{r['carry_pct']:>9.1f}{r['sharpe']:>7.2f}{r['exposure']:>6.2f}"
              f"{r['flips']:>7}  {r['verdict']}")

    print("\nBLOCKERS")
    for r in rows:
        if r["verdict"] == "PASS":
            print(f"  {r['asset']:<6} (none)")
        for b in r["blockers"]:
            print(f"  {r['asset']:<6} {b}")

    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    print(f"\n{n_pass}/{len(rows)} PASS")

    out = args.output or str(REPO / "training" / "reports" / "breadth_exam_p301.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"assumed_spread_bps": args.spread_bps, "results": rows}, indent=2),
        encoding="utf-8")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

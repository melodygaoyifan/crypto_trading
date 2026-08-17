#!/usr/bin/env python3
"""[P290] Realized-fill cost review — the standing instrument for the CDE
spread table (P230 rule: a bar without an instrument becomes "whoever
re-greps by hand").

Reads data/fill_quality.jsonl (written by CoinbaseSleeve._record_fill_quality
on every sleeve fill) and reports, per asset: filled-leg count, maker share,
median/mean realized slippage per leg by liquidity class, and median
decision-time spread — compared against the CDE spread table the alpha gate
charges (P289).

REFUSES (exit 2) below --min-n filled records total: missing data is never a
verdict (P199/P278). Runs stdlib-only so it works in-container where the
ledger lives (P213 note: this ledger is on the hmats-data volume, so unlike
compute_shadow_ic this reader IS meaningful server-side).

RE-DERIVATION RULE (pre-committed): the CDE_SPREAD_BPS constants may only be
LOWERED on >= 20 filled legs for that asset AND a new P-entry recording the
change — never from this report alone, and never from unresolved rows.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

# [P290] Local restatement of FrictionComponents.CDE_SPREAD_BPS — this script
# must stay stdlib (no defense/ import at runtime), so the two copies are
# pinned equal by tests/test_p290_fill_quality.py (P192 two-file guard).
CDE_SPREAD_BPS = {"BTC": 2.0, "ETH": 5.5, "SOL": 4.0}


def load(path: str):
    rows = []
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def _med(vals):
    return round(statistics.median(vals), 3) if vals else None


def _mean(vals):
    return round(statistics.fmean(vals), 3) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path",
                    default=os.path.join(
                        os.environ.get("HMATS_DATA_DIR", "data"),
                        "fill_quality.jsonl"))
    ap.add_argument("--min-n", type=int, default=20,
                    help="minimum FILLED records before any verdict (P278)")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"REFUSING TO REPORT: no ledger at {args.path}")
        print("  The ledger is written by the sleeve on every fill (P290);")
        print("  'no file' means no fills recorded yet OR the wrong host —")
        print("  it lives on the hmats-data volume (run in-container or scp).")
        return 2

    rows, bad = load(args.path)
    filled = [r for r in rows if r.get("status") == "filled"
              and isinstance(r.get("realized_slippage_bps"), (int, float))]
    unresolved = [r for r in rows if r.get("status") != "filled"]

    print(f"fill_quality.jsonl: {len(rows)} rows "
          f"({len(filled)} filled+priced, {len(unresolved)} unresolved, "
          f"{bad} unparseable)")

    if len(filled) < args.min_n:
        print(f"REFUSING VERDICT: {len(filled)} filled records < "
              f"min_n={args.min_n} (P278: the fee/spread verdict needs "
              f"accrual, not extrapolation). Keep accruing.")
        return 2

    assets = sorted({r.get("asset") for r in filled if r.get("asset")})
    print()
    print(f"{'asset':6s} {'n':>4s} {'maker%':>7s} {'med_slip':>9s} "
          f"{'mean_slip':>10s} {'med_spread':>11s}  vs CDE table")
    for a in assets:
        ar = [r for r in filled if r.get("asset") == a]
        mk = [r for r in ar if r.get("liquidity") == "maker"]
        slips = [float(r["realized_slippage_bps"]) for r in ar]
        spreads = [float(r["decision_spread_bps"]) for r in ar
                   if isinstance(r.get("decision_spread_bps"), (int, float))]
        charged = CDE_SPREAD_BPS.get(a)
        note = ""
        if charged is not None:
            ms = _med(slips)
            if ms is not None and len(ar) >= 20 and ms < charged:
                note = (f"charged {charged}bps > realized {ms}bps at "
                        f"n={len(ar)} — re-derivation ELIGIBLE (needs a "
                        f"P-entry, never automatic)")
            elif charged is not None:
                note = f"charged {charged}bps (n={len(ar)}, hold)"
        print(f"{a:6s} {len(ar):>4d} {100*len(mk)/len(ar):>6.1f}% "
              f"{str(_med(slips)):>9s} {str(_mean(slips)):>10s} "
              f"{str(_med(spreads)):>11s}  {note}")
    print()
    print("Per-liquidity medians (slippage bps/leg, + = paid worse than mid):")
    for liq in ("maker", "taker_cross", "direct"):
        lr = [float(r["realized_slippage_bps"]) for r in filled
              if r.get("liquidity") == liq]
        if lr:
            print(f"  {liq:12s} n={len(lr):<4d} median={_med(lr)} "
                  f"mean={_mean(lr)}")
    print()
    print("RULE: CDE_SPREAD_BPS may only be LOWERED on >=20 filled legs per")
    print("asset AND a new P-entry — this report informs, it never edits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

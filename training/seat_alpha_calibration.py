"""[P326] The PRODUCER for core.seat_alpha.REGIMEBOOK_ALPHA_BY_ERA.

Those constants gate live trading — BTC stops entering because its era-median
is 24.1bps/round-trip against a ~42bps threshold — and until now they were
hand-copied literals whose derivation lived only in a docstring sentence:

    "training/funding_legs_lab, 6y, honest per-contract fees,
     gross bps per round trip = 2 x gross per unit turnover"

That sentence is not enough to reproduce them. Re-deriving from it produced
BTC 240.3 / 58.4 / 44.0 against the shipped 2.3 / 68.5 / 24.1 — a control
failure that invalidated an entire comparison until the convention was
recovered by enumeration. This module makes the constants reproducible, which
is the P310/P312 rule applied to a NUMBER rather than to a record shape: when a
consumer restates a producer's value, the restatement is the defect waiting to
happen.

THE CONVENTION, in full, because each clause changes the answer:

  1. edge = 2 x (sum of GROSS bar PnL) / (sum of |delta position|), in bps.
     GROSS only — not net. Charging cost here would double-count: the gate
     compares this against a friction it computes itself.
  2. Eras index the POSITIONS frame directly (funding_legs_lab.ERAS applied
     with .iloc, no adjustment for the MIN_BARS warmup that build_positions
     drops). Correcting for that offset moves BTC pre_design 2.3 -> 243.2.
  3. Turnover treats the FIRST bar of a window as zero turnover: a position
     already standing when the window opens was not entered inside it. This
     only matters where a window opens mid-position — ETH's book is +1 at the
     pre_design open and flat at the other two, which is exactly why only that
     cell disagreed (245.9 vs the shipped 251.7).

VERIFIED: with all three clauses, ETH reproduces 251.7 / 88.1 / 52.1 and BTC
reproduces 2.3 / 68.5 / 24.1 — every era, exactly.

WHAT IT IS FOR, beyond re-deriving what we already have: `--series trend`
prices an ALTERNATIVE position series through the identical convention, so
"would dropping BTC's funding legs let it trade?" is answerable rather than
arguable. (It was asked, and the answer is no: BTC trend-only measures
-3.6 / +137.3 / -10.4, an era-median of -3.6 against the book's +24.1. The legs
are expensive and uncertified, and they are also what makes BTC's per-trade
edge era-STABLE.)

    python -X utf8 training/seat_alpha_calibration.py --verify
    python -X utf8 training/seat_alpha_calibration.py --series trend

Exit codes:  0 = matches the shipped table   2 = refused (no data)
             3 = DRIFT vs the shipped table, or a non-verify report
Operator-local: needs training/training_data (P213).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Clause 1-3 are implemented HERE; the price/position/era machinery is imported
# from the lab that produced the shipped numbers (P172) so this cannot drift
# from it.
TOLERANCE_BPS = 0.15


def round_trip_edge_bps(gross, pos) -> Optional[float]:
    """2 x sum(gross) / sum(|dpos|), in bps — clauses 1 and 3.

    `gross` and `pos` are aligned per-bar series (any sequence type with the
    pandas .diff/.abs/.sum API). Returns None when the window has no turnover,
    because "the position never moved" is not an edge of zero — dividing by it
    would fabricate one (P2).
    """
    turn = pos.diff().abs().fillna(0.0)   # clause 3: first bar contributes 0
    t = float(turn.sum())
    if not (t > 0):
        return None
    return 2.0 * float(gross.sum()) / t * 1e4


def calibrate(asset: str, series: str = "book") -> Dict[str, Optional[float]]:
    """Per-era edge for `asset`'s `series` ("book" or "trend")."""
    import training.funding_legs_lab as lab

    closes = lab.load_closes(asset)
    funding = lab.load_funding_daily(asset)
    pos_df = lab.build_positions(asset, closes, funding)
    out: Dict[str, Optional[float]] = {}
    for era, (lo, hi) in lab.ERAS.items():
        # clause 2: eras index the POSITIONS frame directly
        p = pos_df[series].iloc[lo:hi] if hi else pos_df[series].iloc[lo:]
        if len(p) < 5:
            out[era] = None
            continue
        df = lab.pnl(asset, p, closes, funding)
        out[era] = round_trip_edge_bps(df["gross"], p.reindex(df.index))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assets", default="BTC,ETH,SOL")
    ap.add_argument("--series", default="book", choices=("book", "trend"))
    ap.add_argument("--verify", action="store_true",
                    help="compare against core.seat_alpha and exit 3 on drift")
    args = ap.parse_args(argv)

    try:
        from core.seat_alpha import REGIMEBOOK_ALPHA_BY_ERA as SHIPPED
    except Exception as e:  # noqa: silent-swallow — reported and refused below
        print(f"REFUSING: cannot import the shipped table: {e}",
              file=sys.stderr)
        return 2

    if args.verify and args.series != "book":
        print("REFUSING: --verify compares the BOOK series, which is what the "
              "shipped table calibrates. Verifying a different series against "
              "it would report drift that is really a different question.",
              file=sys.stderr)
        return 2

    drift = []
    print(f"{'asset':<6}{'era':<13}{'measured':>11}{'shipped':>10}   series="
          f"{args.series}")
    for asset in [a.strip().upper() for a in args.assets.split(",") if a.strip()]:
        try:
            cells = calibrate(asset, args.series)
        except FileNotFoundError as e:
            print(f"REFUSING: price/funding history missing for {asset} ({e}). "
                  f"This tool is operator-local (P213); 'no data' is not "
                  f"'the calibration changed'.", file=sys.stderr)
            return 2
        ship = SHIPPED.get(asset, {})
        for era, v in cells.items():
            s = ship.get(era)
            m = "-" if v is None else f"{v:+.1f}"
            sv = "-" if s is None else f"{s:+.1f}"
            flag = ""
            if args.verify and v is not None and s is not None:
                if abs(v - s) > TOLERANCE_BPS:
                    flag = "   <== DRIFT"
                    drift.append((asset, era, v, s))
            print(f"{asset:<6}{era:<13}{m:>11}{sv:>10}{flag}")
        if cells:
            vals = sorted(x for x in cells.values() if x is not None)
            if vals:
                med = vals[len(vals) // 2] if len(vals) % 2 else (
                    (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2.0)
                print(f"{asset:<6}{'MEDIAN':<13}{med:>+11.1f}"
                      f"{(ship.get('__median__') or ''):>10}")

    if args.verify:
        if drift:
            print(f"\nDRIFT vs core.seat_alpha in {len(drift)} cell(s). Either "
                  f"the data moved or the convention did — do NOT edit the "
                  f"shipped constants to match without deciding which.",
                  file=sys.stderr)
            return 3
        print("\nOK — reproduces core.seat_alpha.REGIMEBOOK_ALPHA_BY_ERA "
              "within %.2fbps on every cell." % TOLERANCE_BPS)
        return 0
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

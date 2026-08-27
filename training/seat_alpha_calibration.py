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

[P420] THREE HOLES IN --verify, CLOSED
  1. It compared per-ERA cells only. The value the LIVE gate reads is the
     era-MEDIAN (`REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP`), and a median can move
     without the per-cell check naming it as the thing that moved (BTC's
     validation 24.1 -> 25.5 re-ordered the eras and the median became 25.5).
     It now compares the computed median to the shipped per-RT constant too.
  2. An asset ABSENT from the shipped table produced zero comparisons and an
     OK — a vacuous pass (P174). --verify now REFUSES (exit 2) on an unlisted
     asset; calibrating a NEW asset is a non-verify run.
  3. Its input was OVERWRITTEN by scripts/september_check.py (Kraken prints
     merged into {ASSET}_4H_ohlcv.parquet), which is how the shipped XRP/BNB
     table drifted the day after it shipped. The report JSON now stamps the
     input parquets' sha256 + row counts (training/provenance.py) so a mutated
     input reads as DATA drift and a changed convention as CODE drift.
  Every run also LEDGERS its validation-era read (training/splits.py) — the
  per-asset validation edge feeds a live gate constant and was unledgered.
--assets defaults to the shipped table's keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
REPORT_DIR = REPO / "training" / "reports"


def input_stamp(asset: str) -> Dict[str, object]:
    """[P420] sha256 + row count of the two inputs the calibration reads, so a
    report can tell a mutated input from a changed convention."""
    import training.funding_legs_lab as lab
    out: Dict[str, object] = {}
    for tag, path in (("closes", lab.PRICE_DIR / f"{asset}_4H_ohlcv.parquet"),
                      ("funding", lab.FUNDING_DIR / f"{asset}_funding_1d.parquet")):
        if not path.exists():
            out[tag] = {"path": str(path), "missing": True}
            continue
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            import pandas as pd
            rows = int(len(pd.read_parquet(path)))
        except Exception as e:  # noqa: silent-swallow — surfaced in the stamp itself
            rows = f"unreadable: {type(e).__name__}"
        try:
            rel = str(path.relative_to(REPO))
        except ValueError:
            rel = str(path)
        out[tag] = {"path": rel, "sha256": h, "rows": rows}
    return out


def median_of(cells: Dict[str, Optional[float]]) -> Optional[float]:
    vals = sorted(x for x in cells.values() if x is not None)
    if not vals:
        return None
    if len(vals) % 2:
        return vals[len(vals) // 2]
    return (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2.0


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


def calibrate(asset: str, series: str = "book",
              ledger: bool = True) -> Dict[str, Optional[float]]:
    """Per-era edge for `asset`'s `series` ("book", "trend" or "donchian").

    [P420] `ledger=True` records the VALIDATION-era read in the window-usage
    ledger (training/splits.py): this read feeds a live gate constant and
    every spend of the unread era must be visible (P332/P382). Indices are
    the bar index `i` of the 4H series (clause 2's positions frame carries it;
    MIN_BARS warmup rows are absent), the same axis the other ledger rows use."""
    import training.funding_legs_lab as lab

    closes = lab.load_closes(asset)
    funding = lab.load_funding_daily(asset)
    pos_df = lab.build_positions(asset, closes, funding)
    if ledger:
        try:
            from training.splits import record_window_usage
            v_lo = lab.ERAS["validation"][0]
            v_hi = int(pos_df["i"].iloc[-1]) + 1
            prior = record_window_usage(
                f"seat_alpha_calibration:{series}", asset, v_lo, v_hi,
                f"validation:seat_alpha {series} per-era edge (live gate "
                f"constant producer, P320/P420)")
            if prior:
                print(f"[WINDOW-LEDGER] {asset}: validation window read by "
                      f"{prior} other experiment(s) before — discount "
                      f"accordingly (P260).")
        except Exception as e:  # noqa: silent-swallow — the ledger must never block the producer; surfaced
            print(f"[WINDOW-LEDGER] WARNING: could not record the {asset} "
                  f"validation read ({type(e).__name__}: {e})")
    if series == "donchian":
        # [P419] the canonical Donchian-100 labels (defense.trend_rule_shadow
        # -- the same math the live leg runs, P172), aligned to the chassis
        # frame exactly like the lab that produced the shipped ETH rows.
        import pandas as _pd
        from defense.trend_rule_shadow import donchian_labels as _dl
        _don = _dl(closes.to_numpy(dtype=float))
        pos_df = pos_df.copy()
        pos_df["donchian"] = _pd.Series(
            _don, index=closes.index).reindex(pos_df.index)
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
    ap.add_argument("--assets", default=None,
                    help="comma list; default = the shipped table's keys "
                         "(core.seat_alpha.REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP)")
    ap.add_argument("--series", default="book",  # [P419]
                    choices=("book", "trend", "donchian"))
    ap.add_argument("--verify", action="store_true",
                    help="compare against core.seat_alpha and exit 3 on drift")
    ap.add_argument("--no-ledger", action="store_true",
                    help="[P420] do not record the validation-era read "
                         "(tests only)")
    ap.add_argument("--report", default=None,
                    help="[P420] report JSON path (default: training/reports/"
                         "seat_alpha_calibration_<series>[_verify].json)")
    args = ap.parse_args(argv)

    try:
        from core.seat_alpha import REGIMEBOOK_ALPHA_BY_ERA as SHIPPED
        # [P420] the per-RT MEDIAN the live gate reads — compared explicitly
        from core.seat_alpha import REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP as SHIPPED_RT
    except Exception as e:  # noqa: silent-swallow — reported and refused below
        print(f"REFUSING: cannot import the shipped table: {e}",
              file=sys.stderr)
        return 2

    # [P419] the shipped table is per-asset series-mapped (ETH = donchian);
    # --verify uses the map, so the flag's --series is only for exploration.
    try:
        from core.seat_alpha import REGIMEBOOK_SERIES_BY_ASSET as SERIES_MAP
    except Exception:  # noqa: silent-swallow — an older core without the map verifies the BOOK series
        SERIES_MAP = {}
    if args.verify and args.series != "book":
        print("REFUSING: --verify picks each asset's series from "
              "core.seat_alpha.REGIMEBOOK_SERIES_BY_ASSET; passing --series "
              "with --verify would compare a different question.",
              file=sys.stderr)
        return 2

    assets = ([a.strip().upper() for a in args.assets.split(",") if a.strip()]
              if args.assets else list(SHIPPED_RT))
    if args.verify:
        # [P420] an asset the shipped table does not carry yields ZERO
        # comparisons — an OK there is the P174 vacuous pass. Refuse.
        unlisted = [a for a in assets if a not in SHIPPED_RT or a not in SHIPPED]
        if unlisted:
            print(f"REFUSING: --verify has nothing to compare for {unlisted} — "
                  f"not in core.seat_alpha's shipped table. Calibrating a NEW "
                  f"asset is a non-verify run; an OK on zero comparisons is "
                  f"not a verification (P174/P420).", file=sys.stderr)
            return 2

    drift = []
    report: Dict[str, object] = {"verify": bool(args.verify),
                                 "tolerance_bps": TOLERANCE_BPS, "assets": {}}
    print(f"{'asset':<6}{'era':<13}{'measured':>11}{'shipped':>10}   series="
          f"{args.series}")
    for asset in assets:
        _series = (SERIES_MAP.get(asset, "book")  # [P419] per-asset series
                   if args.verify else args.series)
        try:
            cells = calibrate(asset, _series, ledger=not args.no_ledger)
        except FileNotFoundError as e:
            print(f"REFUSING: price/funding history missing for {asset} ({e}). "
                  f"This tool is operator-local (P213); 'no data' is not "
                  f"'the calibration changed'.", file=sys.stderr)
            return 2
        ship = SHIPPED.get(asset, {})
        for era, v in cells.items():
            s_ = ship.get(era)
            m = "-" if v is None else f"{v:+.1f}"
            sv = "-" if s_ is None else f"{s_:+.1f}"
            flag = ""
            if args.verify and v is not None and s_ is not None:
                if abs(v - s_) > TOLERANCE_BPS:
                    flag = "   <== DRIFT"
                    drift.append((asset, era, v, s_))
            print(f"{asset:<6}{era:<13}{m:>11}{sv:>10}{flag}")
        med = median_of(cells)
        ship_rt = SHIPPED_RT.get(asset)
        med_flag = ""
        if args.verify and med is not None and ship_rt is not None:
            # [P420] the value the LIVE gate reads — compared explicitly, not
            # inferred from the cells: BTC's validation cell moved 24.1->25.5
            # and the MEDIAN followed it to 25.5 while the old check only
            # named the cell.
            if abs(med - ship_rt) > TOLERANCE_BPS:
                med_flag = "   <== DRIFT (the value the gate reads)"
                drift.append((asset, "MEDIAN", med, ship_rt))
        med_s = "-" if med is None else f"{med:+.1f}"
        rt_s = "-" if ship_rt is None else f"{ship_rt:+.1f}"
        print(f"{asset:<6}{'MEDIAN':<13}{med_s:>11}{rt_s:>10}{med_flag}")
        report["assets"][asset] = {
            "series": _series, "cells": cells, "median": med,
            "shipped_cells": ship or None, "shipped_median": ship_rt,
            "inputs": input_stamp(asset),
        }

    # [P420] provenance: git + input hashes, written on EVERY run so the
    # report answers "which data produced this number" (GP0/P200).
    try:
        from training.provenance import provenance_stamp
        import training.funding_legs_lab as lab
        files = []
        for a in assets:
            files += [lab.PRICE_DIR / f"{a}_4H_ohlcv.parquet",
                      lab.FUNDING_DIR / f"{a}_funding_1d.parquet"]
        report["provenance"] = provenance_stamp(
            data_files=files, config={"assets": assets, "series": args.series,
                                      "verify": bool(args.verify)})
    except Exception as e:  # noqa: silent-swallow — a stamp failure is written into the report, never hidden
        report["provenance"] = {"error": f"{type(e).__name__}: {e}"}
    report["drift"] = [{"asset": a, "cell": c, "measured": v, "shipped": s_}
                       for a, c, v, s_ in drift]
    rp = Path(args.report) if args.report else (
        REPORT_DIR / f"seat_alpha_calibration_{args.series}"
                     f"{'_verify' if args.verify else ''}.json")
    try:
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(f"report -> {rp}")
    except OSError as e:
        print(f"WARNING: could not write the report ({e})", file=sys.stderr)

    if args.verify:
        if drift:
            print(f"\nDRIFT vs core.seat_alpha in {len(drift)} cell(s). Either "
                  f"the data moved or the convention did — do NOT edit the "
                  f"shipped constants to match without deciding which. The "
                  f"report's input sha256/rows say which (P420).",
                  file=sys.stderr)
            return 3
        print("\nOK — reproduces core.seat_alpha.REGIMEBOOK_ALPHA_BY_ERA and the "
              "per-RT MEDIAN within %.2fbps on every cell." % TOLERANCE_BPS)
        return 0
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

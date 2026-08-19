"""[P307c] Head-to-head: the seat that drives the book TODAY vs the trend
signal it displaced.

THE QUESTION, STATED SO IT CAN BE FALSIFIED
    Since the P302/P303 seat restoration, the live quant slot is taken by the
    REGIMEBOOK target where the book is directional, with whale filling flat
    cells and trend as the fallback (P298 seat precedence). "Is that better
    than trend?" only means something if both are scored on the same bars,
    with the same costs, in the same expression the sleeve actually trades.

    trend   = sign(close - SMA200), i.e. +/-1 — the live trend seat, and the
              sleeve sizes by SIGN, so a magnitude model would answer a
              question nobody asked
    book    = defense/regime_book_shadow's target via the SAME vectorised
              helper the mechanism lab uses (imported, not restated — P172),
              which is the P250 roster: BTC full book (hold-bull /
              funding_short-bear / funding_contrarian-peace), ETH trend-only,
              SOL hold-bull-only

    Both run through the shared chassis (1-bar delay, per-side costs) and are
    reported in the DESIGN era [3000, 9100) and the PRE-DESIGN era
    [800, 3000). A mechanism that wins in only one era is era-fragile and the
    comparison is not settled (P243/P244).

WHAT THIS CANNOT ANSWER, AND MUST NOT BE READ AS ANSWERING
    Nothing here is forward evidence. The books were SELECTED on the design
    era (P250), so their design-era numbers are in-sample by construction and
    the pre-design column is the only out-of-selection read in this file. The
    6-year BTC certification with a same-cell random control and era
    stability is P297's, not this lab's. And the live seat has been running
    for days, not months — its own ledger is the exam.

    CORRECTION TO MY OWN FRAMING, measured rather than assumed. I expected
    ETH and SOL to be degenerate ("the book IS trend-only"). They are not:
    the two disagree on 66-67% of bars. Decomposed over the design era:

        asset  both LONG  trend SHORT->book FLAT  both SHORT  trend LONG->book FLAT
        BTC         2221                    2249         436                    287
        ETH         1662                    3456           0                     982
        SOL         1827                    3321           0                     950

    So the comparison is dominated by ONE structural difference — trend is
    always in the market and the book refuses to short — plus a second,
    smaller one nobody had stated: the regime labeler's BULL cell agrees with
    `close > SMA200` on only ~83-84% of bars, so the book's long leg is
    stricter than trend's, not identical to it.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if REPO.name == "training":
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

from training.mechanism_lab import book_targets, pnl_after_cost   # noqa: E402
from training.regime_model_lab import _ctx, DESIGN                # noqa: E402
from training.train_supervised_full import COST_BPS               # noqa: E402

DS, DE = DESIGN
PRE = (800, 3000)
SMA_N = 200
REPORT = REPO / "training" / "reports" / "seat_headtohead_p307c.json"


def _sma_sign(close: np.ndarray) -> np.ndarray:
    pos = np.zeros(len(close))
    for i in range(SMA_N, len(close)):
        pos[i] = 1.0 if close[i] > close[i - SMA_N:i].mean() else -1.0
    return pos


def _score(close, pos, cost, lo, hi, carry_rate=None):
    """Net after cost AND funding carry.

    [P307c] Carry is not optional here. The sleeve trades PERPS, trend is
    SHORT on roughly half of all bars while the book is FLAT there, and P296
    measured carry as the DOMINANT drag on a long-biased perp book (-59.7%
    over six years). Scoring price PnL alone would compare the two mechanisms
    on a cash flow the venue does not settle — and it would do so
    asymmetrically, because the two hold opposite positions in exactly the
    cells where they disagree. Convention imported from the labs: positive
    rate means LONGS PAY, charged on the position held into the bar (P245).
    """
    r = pnl_after_cost(close, pos, cost, lo, hi)
    carry = 0.0
    if carry_rate is not None:
        c = np.zeros(len(pos))
        c[1:] = -pos[:-1] * np.nan_to_num(carry_rate[1:])
        carry = float(np.nansum(c[lo:hi]))
    return {"net": round(r["net"] + carry, 4), "net_ex_carry": r["net"],
            "carry": round(carry, 4), "gross": r["gross"], "cost": r["cost"],
            "turnover": r["turnover_units"],
            "exposure": round(float(np.mean(np.abs(pos[lo:hi]))), 3)}


def run(assets=("BTC", "ETH", "SOL")) -> dict:
    out = {"generated": datetime.now(timezone.utc).isoformat(),
           "sma": SMA_N, "eras": {"design": [DS, DE], "pre_design": list(PRE)},
           "assets": {}}
    for a in assets:
        ctx = _ctx(a)
        close, n, cost = ctx["close"], ctx["n"], COST_BPS[a]
        trend = _sma_sign(close)
        book = book_targets(a, ctx["lab"], ctx["fz"])
        # buy_and_hold: the baseline that must always be scored (P182), and
        # the one that caught P296's "beats flat" bar being free.
        bh = np.ones(n)
        rec = {"cost_rt_bps": cost,
               "bars_where_they_differ": int(np.sum(book != trend)),
               "share_differing": round(float(np.mean(book != trend)), 4),
               "candidates": {}}
        for name, pos in (("trend_sma200", trend), ("book", book),
                          ("buy_and_hold", bh)):
            rec["candidates"][name] = {
                "design": _score(close, pos, cost, DS, DE, ctx["carry_rate"]),
                "pre_design": _score(close, pos, cost, *PRE,
                                     carry_rate=ctx["carry_rate"]),
            }
        t, b = rec["candidates"]["trend_sma200"], rec["candidates"]["book"]
        rec["book_minus_trend"] = {
            "design": round(b["design"]["net"] - t["design"]["net"], 4),
            "pre_design": round(b["pre_design"]["net"] - t["pre_design"]["net"], 4),
        }
        rec["verdict"] = (
            "DEGENERATE (book == trend on every bar)"
            if rec["bars_where_they_differ"] == 0 else
            "BOOK" if (rec["book_minus_trend"]["design"] > 0
                       and rec["book_minus_trend"]["pre_design"] > 0) else
            "TREND" if (rec["book_minus_trend"]["design"] < 0
                        and rec["book_minus_trend"]["pre_design"] < 0) else
            "SPLIT (era-dependent — not settled)")
        out["assets"][a] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="BTC,ETH,SOL")
    args = ap.parse_args()
    rep = run(tuple(x.strip().upper() for x in args.assets.split(",")))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    for a, r in rep["assets"].items():
        print(f"\n=== {a}   differ on {r['bars_where_they_differ']} bars "
              f"({r['share_differing']:.1%})   cost={r['cost_rt_bps']}bps rt")
        print(f"  {'candidate':<14}{'design net':>12}{'carry':>9}{'expo':>7}"
              f"{'pre net':>11}{'carry':>9}{'expo':>7}")
        for nm, c in r["candidates"].items():
            d, p = c["design"], c["pre_design"]
            print(f"  {nm:<14}{d['net']:>+12.3f}{d['carry']:>+9.3f}"
                  f"{d['exposure']:>7.2f}{p['net']:>+11.3f}{p['carry']:>+9.3f}"
                  f"{p['exposure']:>7.2f}")
        print(f"  book - trend: design {r['book_minus_trend']['design']:+.3f}  "
              f"pre {r['book_minus_trend']['pre_design']:+.3f}   -> {r['verdict']}")
    print(f"\nreport -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

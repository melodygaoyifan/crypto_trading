"""[P307d] Is being FLAT better than being SHORT? The one question that
separates the live trend seat from the regime book.

WHY THIS AND NOT ANOTHER BOOK-VS-TREND RUN
    P307c measured the two head to head and got SPLIT on all three assets —
    but its decomposition showed that two thirds of the disagreement is a
    single structural choice: trend is always in the market (+/-1) and the
    book refuses to short. Comparing the two composites again would keep
    re-measuring that choice mixed with everything else. So this lab holds
    the LONG LEG CONSTANT across every arm (+1 iff close > SMA200) and varies
    only what happens in the bars where trend would be short.

    That isolation is the point. The long leg is not in dispute — the arms
    are byte-identical there — so every difference reported below is the
    short leg and nothing else.

ARMS (all share the same long leg; they differ ONLY in the short cells)
    long_flat        0 in the short cells                 (the book's answer)
    long_short       -1 in the short cells                (the live trend seat)
    funding_gated    -1 only where causal funding z > 1.0 (the BTC book's own
                     bear rule, generalised to all three assets so it can be
                     falsified somewhere other than where it was designed)
    random_samecell  random signs in the SAME cells, turnover-matched
                     (P297's discriminator: a control a coin flip cannot fake)
    buy_and_hold     always +1                            (P182 — always score
                     the do-nothing baseline; P296 caught a bar being free
                     precisely because it was not scored)

ECONOMICS
    1-bar delay, per-side costs and FUNDING CARRY, all from the shared
    chassis (imported, not restated — P172). Carry is decisive here rather
    than incidental: a short COLLECTS funding when longs pay, so the short
    leg's whole case may rest on carry rather than on price direction, and
    the decomposition below reports the two separately so that can be seen
    instead of inferred.

VERDICT, PRE-COMMITTED BEFORE THE RUN
    For each asset, SHORT EARNS iff the short-leg increment (arm minus
    long_flat) is:
      (1) positive in the DESIGN era, AND
      (2) positive in the PRE-DESIGN era, AND
      (3) greater than the same-cell random control in both.
    Anything else is FLAT STANDS — i.e. the book's refusal to short is right
    and the live trend seat is giving that edge away.

    Deliberately strict on era stability, because every mechanism this repo
    has certified on one era and deployed has died on the other (P243/P244),
    and deliberately NOT keyed on a t-statistic: P297 established that at
    this sample size a |t| >= 2 bar rejects buy-and-hold too, so it does not
    discriminate. The random control and era stability are properties a coin
    flip cannot fake; a significance bar the sample cannot support is not.
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

from training.mechanism_lab import pnl_after_cost      # noqa: E402
from training.regime_model_lab import _ctx, DESIGN     # noqa: E402
from training.train_supervised_full import COST_BPS    # noqa: E402

DS, DE = DESIGN
PRE = (800, 3000)
SMA_N = 200
FUNDING_Z_SHORT = 1.0          # the BTC book's own bear-leg threshold (P250)
N_RANDOM = 200                 # control draws
SEED = 20260818
REPORT = REPO / "training" / "reports" / "short_leg_lab_p307d.json"


def _bull(close: np.ndarray) -> np.ndarray:
    """+1 where close > trailing SMA200, else 0. The LONG LEG, held constant
    across every arm so the comparison is the short leg alone."""
    out = np.zeros(len(close), dtype=bool)
    for i in range(SMA_N, len(close)):
        out[i] = close[i] > close[i - SMA_N:i].mean()
    return out


def _carry(pos, carry_rate, lo, hi) -> float:
    c = np.zeros(len(pos))
    c[1:] = -pos[:-1] * np.nan_to_num(carry_rate[1:])
    return float(np.nansum(c[lo:hi]))


def _score(close, pos, cost, carry_rate, lo, hi) -> dict:
    r = pnl_after_cost(close, pos, cost, lo, hi)
    car = _carry(pos, carry_rate, lo, hi)
    return {"net": round(r["net"] + car, 4), "price_pnl": r["gross"],
            "cost": r["cost"], "carry": round(car, 4),
            "turnover": r["turnover_units"],
            "exposure": round(float(np.mean(np.abs(pos[lo:hi]))), 3)}


def _short_cell_decomposition(close, short_cells, carry_rate, lo, hi) -> dict:
    """What a -1 held ONLY in the short cells earns, split into its parts.

    Reported because the short leg's case may rest entirely on collecting
    funding rather than on price direction, and those have different
    futures: a funding regime can invert (P244 measured exactly that), a
    price edge is a different claim.
    """
    r1 = np.zeros(len(close))
    r1[:-1] = close[1:] / close[:-1] - 1.0
    seg = slice(lo, min(hi - 1, len(close) - 1))
    m = short_cells[seg]
    price = float(np.nansum(-r1[seg][m]))
    pos = np.where(short_cells, -1.0, 0.0)
    return {"bars": int(m.sum()),
            "price_pnl_from_shorts": round(price, 4),
            "carry_collected": round(_carry(pos, carry_rate, lo, hi), 4)}


def run(assets=("BTC", "ETH", "SOL")) -> dict:
    rng = np.random.default_rng(SEED)
    out = {"generated": datetime.now(timezone.utc).isoformat(),
           "sma": SMA_N, "funding_z_short": FUNDING_Z_SHORT,
           "eras": {"design": [DS, DE], "pre_design": list(PRE)},
           "n_random": N_RANDOM, "seed": SEED, "assets": {}}
    for a in assets:
        ctx = _ctx(a)
        close, n, cost, cr = ctx["close"], ctx["n"], COST_BPS[a], ctx["carry_rate"]
        bull = _bull(close)
        short_cells = (~bull) & (np.arange(n) >= SMA_N)
        fz = np.nan_to_num(ctx["fz"], nan=0.0)
        has_fz = ~np.isnan(ctx["fz"])

        arms = {
            "long_flat": np.where(bull, 1.0, 0.0),
            "long_short": np.where(bull, 1.0, np.where(short_cells, -1.0, 0.0)),
            "funding_gated": np.where(
                bull, 1.0,
                np.where(short_cells & has_fz & (fz > FUNDING_Z_SHORT), -1.0, 0.0)),
            "buy_and_hold": np.ones(n),
        }
        rec = {"cost_rt_bps": cost,
               "short_cells_design": int(short_cells[DS:DE].sum()),
               "short_cells_pre": int(short_cells[PRE[0]:PRE[1]].sum()),
               "arms": {}, "increment_vs_long_flat": {}, "decomposition": {}}
        for nm, pos in arms.items():
            rec["arms"][nm] = {"design": _score(close, pos, cost, cr, DS, DE),
                               "pre_design": _score(close, pos, cost, cr, *PRE)}
        base = rec["arms"]["long_flat"]
        for nm in ("long_short", "funding_gated", "buy_and_hold"):
            rec["increment_vs_long_flat"][nm] = {
                era: round(rec["arms"][nm][era]["net"] - base[era]["net"], 4)
                for era in ("design", "pre_design")}
        for era, (lo, hi) in (("design", (DS, DE)), ("pre_design", PRE)):
            rec["decomposition"][era] = _short_cell_decomposition(
                close, short_cells, cr, lo, hi)

        # --- same-cell, turnover-matched random control (P297) -------------
        ctrl = {}
        for era, (lo, hi) in (("design", (DS, DE)), ("pre_design", PRE)):
            incs = []
            for _ in range(N_RANDOM):
                signs = rng.choice((-1.0, 1.0), size=n)
                pos = np.where(bull, 1.0, np.where(short_cells, signs, 0.0))
                incs.append(_score(close, pos, cost, cr, lo, hi)["net"]
                            - base[era]["net"])
            incs = np.array(incs)
            ctrl[era] = {"mean": round(float(incs.mean()), 4),
                         "p05": round(float(np.percentile(incs, 5)), 4),
                         "p95": round(float(np.percentile(incs, 95)), 4)}
        rec["random_samecell_increment"] = ctrl

        def _verdict(nm):
            i = rec["increment_vs_long_flat"][nm]
            beats_ctrl = all(i[e] > ctrl[e]["mean"] for e in i)
            positive = all(i[e] > 0 for e in i)
            return ("SHORT EARNS" if (positive and beats_ctrl) else
                    "FLAT STANDS")
        rec["verdict"] = {nm: _verdict(nm)
                          for nm in ("long_short", "funding_gated")}
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
        print(f"\n=== {a}   short cells: {r['short_cells_design']} design / "
              f"{r['short_cells_pre']} pre   cost={r['cost_rt_bps']}bps rt")
        print(f"  {'arm':<16}{'design net':>12}{'carry':>9}"
              f"{'pre net':>11}{'carry':>9}")
        for nm, c in r["arms"].items():
            d, p = c["design"], c["pre_design"]
            print(f"  {nm:<16}{d['net']:>+12.3f}{d['carry']:>+9.3f}"
                  f"{p['net']:>+11.3f}{p['carry']:>+9.3f}")
        print("  short-leg increment vs long_flat:")
        for nm, i in r["increment_vs_long_flat"].items():
            v = r["verdict"].get(nm, "")
            print(f"    {nm:<16}design {i['design']:>+8.3f}   "
                  f"pre {i['pre_design']:>+8.3f}   {v}")
        c = r["random_samecell_increment"]
        print(f"    {'random same-cell':<16}design {c['design']['mean']:>+8.3f} "
              f"[{c['design']['p05']:+.2f},{c['design']['p95']:+.2f}]  "
              f"pre {c['pre_design']['mean']:>+8.3f} "
              f"[{c['pre_design']['p05']:+.2f},{c['pre_design']['p95']:+.2f}]")
        for era, dd in r["decomposition"].items():
            print(f"    [{era}] pure short leg: price {dd['price_pnl_from_shorts']:+.3f}"
                  f"  carry {dd['carry_collected']:+.3f}  over {dd['bars']} bars")
    print(f"\nreport -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

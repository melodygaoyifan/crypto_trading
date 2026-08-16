#!/usr/bin/env python3
"""[P276] Exit-rule lab — the sleeve's profit-taking gap (P275), attacked
through the ladder instead of by porting the Kraken exit stack.

WHY
---
P275's venue-binding audit found the sleeve has NO profit-taking logic: its
only exits are signal flatten/flip, the 10% venue stop, halts, and
FastRisk. The old excuse (±1 contract cannot express partials) weakened the
day P274 sized ETH at 3+ contracts. But the Kraken exit stack (exit_alpha /
PROFIT_SCHEDULE / soft_stop) was never validated on this venue's economics
— porting it would deploy an unmeasured strategy (the exact P29/P200
mistake). So: measure first. Two mechanisms, chosen for EXPRESSIBILITY at
the sleeve's granularity:

  trail  — trailing exit: leave the position when price retraces
           K x rolling-vol from its best level since entry. FULL exit —
           expressible at ANY contract count, all three books.
  scaleout — take 1/3 off at +P x rolling-vol of favorable move, hold the
           rest for the book's own exit. Fractional — expressible only
           where sizing >= 3 contracts (ETH under the P274 live config),
           so it is tested on ETH ONLY.

DISCIPLINE (unchanged): design era selection + pre-design stability, NO
validation-era bar scored (chassis assert). A mechanism that wins BOTH eras
earns the single ledgered validation read (P259b ordering) BEFORE any
forward-ledger wiring; nothing here deploys anything.

Run (operator-local, needs training parquets):
    python -X utf8 training/exit_rule_lab.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from training.mechanism_lab import (  # noqa: E402
    book_targets, pnl_after_cost, DS, DE, PRE)
from training.entry_filter_lab import _roll_vol  # noqa: E402
from training.regime_model_lab import _ctx  # noqa: E402
from training.train_supervised_full import COST_BPS  # noqa: E402

REPORT = REPO / "training" / "reports" / "exit_rule_lab_p276.json"


def apply_trailing_exit(raw: np.ndarray, close: np.ndarray,
                        vol: np.ndarray, k: float) -> np.ndarray:
    """Overlay a trailing exit on the book's position path.

    While the book holds a position, track the best close in the trade's
    favor; exit to FLAT when price retraces k x vol (fraction terms) from
    that best. After a trailing exit, re-entry waits for the BOOK to go
    flat and signal again (no immediate re-entry against the exit — the
    churn trap). Missing vol never triggers an exit (absence is not a
    signal, P2) and never blocks one the book itself orders.
    """
    n = len(raw)
    out = np.zeros(n)
    cur, best, locked = 0.0, np.nan, False
    for i in range(n):
        want = raw[i]
        if locked:
            # stay flat until the book itself releases (goes flat)
            if want == 0.0:
                locked = False
            out[i] = 0.0
            cur = 0.0
            continue
        if want != cur:
            cur = want
            best = close[i] if cur != 0.0 else np.nan
        elif cur != 0.0:
            best = (max(best, close[i]) if cur > 0
                    else min(best, close[i]))
            v = vol[i]
            if np.isfinite(v) and v > 0 and np.isfinite(best) and best > 0:
                retrace = ((best - close[i]) / best if cur > 0
                           else (close[i] - best) / best)
                if retrace >= k * v:
                    cur = 0.0
                    locked = True
        out[i] = cur
    return out


def apply_scaleout(raw: np.ndarray, close: np.ndarray, vol: np.ndarray,
                   p: float, frac: float = 1.0 / 3.0) -> np.ndarray:
    """Overlay: once the favorable move since entry exceeds p x vol, reduce
    the position by `frac` (granularity: ETH at 3ct can shed exactly 1/3);
    the remainder follows the book's own exit."""
    n = len(raw)
    out = np.zeros(n)
    cur, entry, scaled = 0.0, np.nan, False
    for i in range(n):
        want = raw[i]
        if want != cur or want == 0.0:
            cur = want
            entry = close[i] if cur != 0.0 else np.nan
            scaled = False
            out[i] = cur
            continue
        gain = ((close[i] - entry) / entry if cur > 0
                else (entry - close[i]) / entry)
        v = vol[i]
        if (not scaled and np.isfinite(v) and v > 0
                and np.isfinite(gain) and gain >= p * v):
            scaled = True
        out[i] = cur * (1.0 - frac) if scaled else cur
    return out


def _eras(close, pos, cost):
    return {"design": pnl_after_cost(close, pos, cost, DS, DE),
            "pre_design": pnl_after_cost(close, pos, cost, *PRE)}


def stage(mechanism: str, assets) -> dict:
    res: dict = {}
    for a in assets:
        c = _ctx(a)
        close = c["close"]
        vol = _roll_vol(close)
        raw = book_targets(a, c["lab"], c["fz"])
        base = _eras(close, raw, COST_BPS[a])
        rows, best_k, best_net = {}, None, -1e9
        grid = (2.0, 3.0, 4.0) if mechanism == "trail" else (1.0, 2.0, 3.0)
        for k in grid:
            pos = (apply_trailing_exit(raw, close, vol, k)
                   if mechanism == "trail"
                   else apply_scaleout(raw, close, vol, k))
            r = _eras(close, pos, COST_BPS[a])
            rows[f"k{k}"] = r
            if r["design"]["net"] > best_net:
                best_k, best_net = k, r["design"]["net"]
        bd = rows[f"k{best_k}"]["design"]["net"]
        bp = rows[f"k{best_k}"]["pre_design"]["net"]
        earns = (bd > base["design"]["net"]
                 and bp > base["pre_design"]["net"])
        res[a] = {"base": base, "grid": rows, "selected_k": best_k,
                  "verdict": ("EARNS BOTH ERAS" if earns else
                              "ERA-FRAGILE" if bd > base["design"]["net"]
                              else "base stands")}
        print(f"[{mechanism}] {a}: base d={base['design']['net']:+.4f}/p="
              f"{base['pre_design']['net']:+.4f} | best k={best_k} "
              f"d={bd:+.4f}/p={bp:+.4f} -> {res[a]['verdict']}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["trail", "scaleout", "all"])
    args = ap.parse_args()
    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "design_era": [DS, DE], "pre_design_era": list(PRE),
              "note": "DESIGN-era selection + pre-design stability ONLY; "
                      "no validation bar scored (chassis assert). Winners "
                      "earn the single P259b validation read BEFORE any "
                      "wiring. scaleout is ETH-only: the sole asset whose "
                      "P274 sizing (3ct) makes 1/3 partials expressible."}
    if args.stage in ("trail", "all"):
        report["trail"] = stage("trail", ("BTC", "ETH", "SOL"))
    if args.stage in ("scaleout", "all"):
        report["scaleout"] = stage("scaleout", ("ETH",))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1, default=str),
                      encoding="utf-8")
    print(f"report: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

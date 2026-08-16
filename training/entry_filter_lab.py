#!/usr/bin/env python3
"""[P270] Entry-filter lab — two mechanisms from the 2026-08-15 external
research pass, tested on the P250 books, DESIGN ERA ONLY.

WHY THESE TWO
-------------
The 2025-26 net-of-cost literature supports exactly two ways of improving a
trend book's entries without destroying the edge:
  1. vol management (FRL 2025: risk-managed crypto momentum Sharpe
     1.12 -> 1.42, robust to costs) — at +/-1-contract granularity the only
     expressible form is an ENTRY-SKIP rule, which is NOT what P256's
     volfilter tested (that was a low-vol position filter on SOL; this is
     the high-vol side, entries only);
  2. higher-timeframe confirmation (Quantpedia 2025: hourly MACD + daily
     trend filter, net Sharpe 0.33 -> 0.80 from FEWER, filtered entries).

DISCIPLINE (unchanged from P256/P263)
-------------------------------------
- Design era [3000, 9100) for selection; pre-design [800, 3000) for the
  era-stability read. NO validation-era bar is ever scored (hard-asserted
  in the imported pnl_after_cost). A mechanism EARNS only by beating the
  base book in BOTH eras; anything else is recorded and dropped.
- Entry filters NEVER force an exit (the P195/P236 asymmetry): a held
  position rides; a FLIP whose entry leg is blocked degrades to a flatten
  (the exit half of a flip is never blocked).
- Chassis imported from mechanism_lab (single source, P172) — book targets,
  cost arithmetic and era bounds cannot drift from what P256 measured.

Run (operator-local, needs training parquets):
    python -X utf8 training/entry_filter_lab.py
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
from training.regime_model_lab import _ctx  # noqa: E402
from training.train_supervised_full import COST_BPS  # noqa: E402

REPORT = REPO / "training" / "reports" / "entry_filter_lab_p270.json"


# ---------------------------------------------------------------------------
# the one mechanism both stages share: allow/deny NEW ENTRIES only
# ---------------------------------------------------------------------------

def apply_entry_filter(raw: np.ndarray, allow: np.ndarray) -> np.ndarray:
    """Position path with entries gated by `allow` (bool per bar).

    - flat -> nonzero with allow[i] False: entry BLOCKED (stay flat)
    - held position with raw unchanged: untouched (never a forced exit)
    - book exits (raw -> 0): always honored
    - FLIP with allow[i] False: degrades to FLATTEN — the exit leg of a
      flip is never blocked (P195), only its entry leg
    Missing/NaN allow reads as False for ENTERING only — absence of the
    filter input must not open positions the filter would have vetoed, and
    must equally never force an exit (P2: absence is not a signal).
    """
    n = len(raw)
    out = np.zeros(n)
    cur = 0.0
    ok = np.where(np.isnan(allow.astype(float)), 0.0, allow.astype(float))
    for i in range(n):
        want = raw[i]
        if want == cur:
            out[i] = cur
            continue
        if want == 0.0:                       # exit — always honored
            cur = 0.0
        elif cur == 0.0:                      # entry from flat
            cur = want if ok[i] else 0.0
        else:                                 # flip
            cur = want if ok[i] else 0.0      # blocked entry leg -> flatten
        out[i] = cur
    return out


def _roll_vol(close: np.ndarray, window: int = 20) -> np.ndarray:
    n = len(close)
    r1 = np.zeros(n)
    r1[1:] = np.log(close[1:] / close[:-1])
    return pd.Series(r1).rolling(window).std().values


# ---------------------------------------------------------------------------
# stage 1: high-vol entry skip
# ---------------------------------------------------------------------------

def stage_volskip(assets=("BTC", "ETH", "SOL")) -> dict:
    res: dict = {}
    for a in assets:
        c = _ctx(a)
        close = c["close"]
        n = len(close)
        vol = _roll_vol(close)
        # causal expanding quantile of PAST vol only (shift(1))
        v = pd.Series(vol)
        raw = book_targets(a, c["lab"], c["fz"])
        base_d = pnl_after_cost(close, raw, COST_BPS[a], DS, DE)
        base_p = pnl_after_cost(close, raw, COST_BPS[a], *PRE)
        rows = {}
        best_q, best_net = None, -1e9
        for q in (0.67, 0.80, 0.90):
            thr = np.full(n, np.nan)
            thr[400:] = v.expanding(min_periods=300).quantile(q).shift(1).values[400:]
            allow = ~np.isnan(vol) & ~np.isnan(thr) & (vol <= thr)
            filt = apply_entry_filter(raw, allow)
            d = pnl_after_cost(close, filt, COST_BPS[a], DS, DE)
            p = pnl_after_cost(close, filt, COST_BPS[a], *PRE)
            rows[f"q{q}"] = {"design": d, "pre_design": p}
            if d["net"] > best_net:
                best_q, best_net = q, d["net"]
        bd = rows[f"q{best_q}"]["design"]["net"]
        bp = rows[f"q{best_q}"]["pre_design"]["net"]
        earns = bd > base_d["net"] and bp > base_p["net"]
        res[a] = {"base": {"design": base_d, "pre_design": base_p},
                  "grid": rows, "selected_q": best_q,
                  "verdict": ("EARNS BOTH ERAS" if earns else
                              "ERA-FRAGILE" if bd > base_d["net"]
                              else "base stands")}
        print(f"[volskip] {a}: base d={base_d['net']:+.4f}/p="
              f"{base_p['net']:+.4f} | best q={best_q} d={bd:+.4f}/p="
              f"{bp:+.4f} -> {res[a]['verdict']}")
    return res


# ---------------------------------------------------------------------------
# stage 2: higher-timeframe (slower-trend) confirmation
# ---------------------------------------------------------------------------

def stage_htf(assets=("BTC", "ETH", "SOL")) -> dict:
    res: dict = {}
    for a in assets:
        c = _ctx(a)
        close = c["close"]
        raw = book_targets(a, c["lab"], c["fz"])
        base_d = pnl_after_cost(close, raw, COST_BPS[a], DS, DE)
        base_p = pnl_after_cost(close, raw, COST_BPS[a], *PRE)
        rows = {}
        best_w, best_net = None, -1e9
        for w in (400, 600, 900):   # 4H bars: ~67d / ~100d / ~150d
            sma = pd.Series(close).rolling(w).mean().values
            # agreement is DIRECTIONAL: a long entry needs close > slow SMA,
            # a short entry close < it. Computed per bar against the raw
            # target's sign; where raw is 0 the allow value is irrelevant.
            with np.errstate(invalid="ignore"):
                allow = np.where(raw > 0, close > sma,
                                 np.where(raw < 0, close < sma, True))
            allow = allow & ~np.isnan(sma)
            filt = apply_entry_filter(raw, allow)
            d = pnl_after_cost(close, filt, COST_BPS[a], DS, DE)
            p = pnl_after_cost(close, filt, COST_BPS[a], *PRE)
            rows[f"sma{w}"] = {"design": d, "pre_design": p}
            if d["net"] > best_net:
                best_w, best_net = w, d["net"]
        bd = rows[f"sma{best_w}"]["design"]["net"]
        bp = rows[f"sma{best_w}"]["pre_design"]["net"]
        earns = bd > base_d["net"] and bp > base_p["net"]
        res[a] = {"base": {"design": base_d, "pre_design": base_p},
                  "grid": rows, "selected_w": best_w,
                  "verdict": ("EARNS BOTH ERAS" if earns else
                              "ERA-FRAGILE" if bd > base_d["net"]
                              else "base stands")}
        print(f"[htf] {a}: base d={base_d['net']:+.4f}/p={base_p['net']:+.4f}"
              f" | best sma{best_w} d={bd:+.4f}/p={bp:+.4f} -> "
              f"{res[a]['verdict']}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["volskip", "htf", "all"])
    args = ap.parse_args()
    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "design_era": [DS, DE], "pre_design_era": list(PRE),
              "note": "DESIGN-ERA selection + pre-design stability ONLY. "
                      "No validation-era bar scored (hard-asserted). A "
                      "mechanism earns a forward ledger only on BOTH-era "
                      "wins; nothing here deploys anything."}
    if args.stage in ("volskip", "all"):
        report["volskip"] = stage_volskip()
    if args.stage in ("htf", "all"):
        report["htf"] = stage_htf()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1, default=str),
                      encoding="utf-8")
    print(f"report: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""[P288-C] The unread-era probe for the TREND-RULE CHALLENGERS.

The P288-B sweep found DONCHIAN-100 and EMA-ENSEMBLE dethrone SMA200 on
ETH/SOL under the house rules — but only on the design/pre-design windows,
which are inside the selected sample. The SMA200 incumbent's certification
additionally rests on two out-of-selection axes (P262): the VIRGIN ERA
(2017-11 -> 2020-08-09, before the parquet start — no selection ever
touched it) and FIVE NEVER-FITTED ASSETS (2020-2026, assets no selection
ever touched). This probe sits the two dethroning challengers on exactly
those axes, so a winner's evidence becomes comparable to the incumbent's.

SINGLE-SOURCE (P172): the data machinery (vision-archive fetch w/ local
cache, causal labels, pnl convention) is IMPORTED from
training/unread_era_probe.py (the P262 probe), and the challenger labelers
are IMPORTED from training/trend_rule_lab.py — a re-typed lookback would
be a different candidate.

COST CONVENTION: identical to the P262 probe it extends — Part A charges
BTC 6 / ETH 8 bps RT, Part B charges 10 bps RT. (The directive's flat
"10bps" matches Part B; Part A keeps the original convention because the
incumbent-control reproduction is against the recorded P262 numbers,
which were produced at 6/8. Deviation recorded, not hidden.)

HARD GUARD (inherited): Part A scores NO bar on/after 2020-08-09.

PRE-COMMITTED VERDICT PER CHALLENGER (written before the first run):
  MATCHES-INCUMBENT-CERTIFICATION iff BOTH
    (a) virgin era: after-cost net > 0 (beats flat) on BOTH BTC and ETH,
        AND the 2018-style crash-dodge is visible: 2018 net >= -0.35 AND
        2018 net >= (2018 B&H + 0.20) on both assets;
    (b) never-fitted assets: after-cost net > 0 (beats flat) on 5/5.
  PARTIAL if exactly one of (a)/(b) holds.  FAILS-TRANSFER otherwise.
  Head-to-head vs SMA200 is REPORTED in every cell but does not decide —
  out-of-selection evidence is about the mechanism being real, not about
  winning every era.

CONTROL: the SMA200 trend-only cells must reproduce the recorded P262
numbers (BTC +0.9115 / ETH +1.8603 virgin era; breadth 5/5 beats-flat)
within 0.02 — same code, same cached archives, so an exact match is
expected; a miss means the machinery drifted and the run REFUSES.

Run:  python -X utf8 training/scripts/unread_era_probe_rules_p288.py
Report: training/reports/unread_era_probe_rules_p288.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if REPO.name == "training":
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

from training.unread_era_probe import (  # noqa: E402
    monthly_klines, causal_labels, pnl, PARQUET_START_MS,
    COST_RT, XASSET_COST_RT)
from training.trend_rule_lab import (  # noqa: E402
    lab_donchian, lab_ema_ensemble)

OUT = REPO / "training" / "reports" / "unread_era_probe_rules_p288.json"

# Recorded P262 control values (training/reports/unread_era_probe_p261.json)
CONTROL_A = {"BTC": 0.9115, "ETH": 1.8603}
CONTROL_B = {"BNBUSDT": 4.0608, "XRPUSDT": 2.7613, "LTCUSDT": 0.6755,
             "DOGEUSDT": 6.7165, "ADAUSDT": 3.905}
CTRL_TOL = 0.02

RULES = (
    ("SMA200", None),                    # incumbent: (lab == 1) from labels
    ("DONCHIAN-100", lab_donchian),
    ("EMA-ENSEMBLE", lab_ema_ensemble),
)


def positions(name, fn, close, lab):
    if fn is None:
        return np.where(lab == 1, 1.0, 0.0)
    return np.asarray(fn(close), dtype=float)


def causality_selfcheck(close: np.ndarray) -> None:
    """P164 construction test on the imported challengers."""
    t0 = min(2000, len(close) - 100)
    fut = close.copy()
    fut[t0:] *= np.linspace(3.0, 0.1, len(fut) - t0)
    for name, fn in RULES[1:]:
        a = np.nan_to_num(fn(close)[:t0])
        b = np.nan_to_num(fn(fut)[:t0])
        if not np.array_equal(a, b):
            raise SystemExit(f"[P288-C] CAUSALITY VIOLATION in {name}")
    print("[P288-C] causality construction test: PASS")


def part_a():
    res = {}
    for a, sym in [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")]:
        rows = monthly_klines(sym, 2017, 2020)
        rows = [r for r in rows if r[0] < PARQUET_START_MS]   # HARD GUARD
        ts = np.array([r[0] for r in rows])
        close = np.array([r[1] for r in rows])
        assert ts.max() < PARQUET_START_MS, "virgin-era guard violated"
        lab = causal_labels(close)
        dates = [datetime.fromtimestamp(t / 1000, tz=timezone.utc
                                        ).date().isoformat() for t in ts]
        lo, hi = 600, len(close)
        if a == "BTC":
            causality_selfcheck(close)
        cells = {}
        import pandas as pd
        yr = pd.Series([d[:4] for d in dates])
        for name, fn in RULES:
            pos = positions(name, fn, close, lab)
            net = pnl(close, pos, COST_RT[a], lo, hi)
            by_year = {}
            for y in sorted(yr.unique()):
                idx = np.where(yr.values == y)[0]
                l2, h2 = max(lo, idx.min()), idx.max() + 1
                if h2 - l2 < 100:
                    continue
                by_year[y] = {
                    "net": pnl(close, pos, COST_RT[a], l2, h2),
                    "bh": round(close[h2 - 1] / close[l2] - 1.0, 4)}
            cells[name] = {"net": net, "by_year": by_year}
            print(f"[A] {a} {name:13s} net={net:+.4f}  "
                  + "  ".join(f"{y}:{v['net']:+.3f}" for y, v in by_year.items()))
        ctrl = cells["SMA200"]["net"]
        if abs(ctrl - CONTROL_A[a]) > CTRL_TOL:
            raise SystemExit(
                f"[P288-C] CONTROL FAILED {a}: SMA200 {ctrl} vs recorded "
                f"{CONTROL_A[a]} — machinery drifted, refusing")
        res[a] = {"span": [dates[lo], dates[-1]], "bars": hi - lo - 1,
                  "buy_hold": round(close[hi - 1] / close[lo] - 1.0, 4),
                  "cost_rt_bps": COST_RT[a], "cells": cells,
                  "control_reproduced": True}
    return res


def part_b():
    res = {}
    beats_flat = {name: 0 for name, _ in RULES}
    for sym in ("BNBUSDT", "XRPUSDT", "LTCUSDT", "DOGEUSDT", "ADAUSDT"):
        rows = monthly_klines(sym, 2020, 2026)
        if len(rows) < 2000:
            res[sym] = {"error": f"only {len(rows)} bars — NOT-TESTED"}
            continue
        close = np.array([r[1] for r in rows])
        lab = causal_labels(close)
        lo, hi = 600, len(close)
        cells = {}
        for name, fn in RULES:
            pos = positions(name, fn, close, lab)
            net = pnl(close, pos, XASSET_COST_RT, lo, hi)
            cells[name] = net
            beats_flat[name] += int(net > 0)
        ctrl = cells["SMA200"]
        if abs(ctrl - CONTROL_B[sym]) > CTRL_TOL:
            raise SystemExit(
                f"[P288-C] CONTROL FAILED {sym}: {ctrl} vs {CONTROL_B[sym]}")
        res[sym] = {"bars": hi - lo,
                    "buy_hold": round(close[hi - 1] / close[lo] - 1.0, 4),
                    **{name: cells[name] for name, _ in RULES}}
        print(f"[B] {sym:9s} " + "  ".join(
            f"{name}:{cells[name]:+.3f}" for name, _ in RULES))
    res["_beats_flat"] = {name: f"{beats_flat[name]}/5" for name, _ in RULES}
    return res


def verdict(res_a, res_b):
    out = {}
    for name, _ in RULES[1:]:
        a_ok = True
        for asset in ("BTC", "ETH"):
            c = res_a[asset]["cells"][name]
            y18 = c["by_year"].get("2018")
            a_ok &= c["net"] > 0
            a_ok &= y18 is not None and y18["net"] >= -0.35
            a_ok &= y18 is not None and y18["net"] >= y18["bh"] + 0.20
        b_ok = res_b["_beats_flat"][name] == "5/5"
        out[name] = ("MATCHES-INCUMBENT-CERTIFICATION" if (a_ok and b_ok)
                     else "PARTIAL" if (a_ok or b_ok) else "FAILS-TRANSFER")
        out[name + "_legs"] = {"virgin_era": bool(a_ok), "breadth_5of5": bool(b_ok)}
    return out


def main() -> int:
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "guard": "Part A scores NO bar on/after 2020-08-09",
        "verdict_rule": ("MATCHES iff virgin-era net>0 both assets w/ 2018 "
                         "crash-dodge (net>=-0.35 AND >= bh+0.20) AND "
                         "breadth beats-flat 5/5; PARTIAL if one leg; "
                         "head-to-head vs SMA200 reported, not deciding"),
    }
    report["A_virgin_era"] = part_a()
    report["B_never_fitted_assets"] = part_b()
    report["verdicts"] = verdict(report["A_virgin_era"],
                                 report["B_never_fitted_assets"])
    OUT.write_text(json.dumps(report, indent=1, default=str),
                   encoding="utf-8")
    print(f"\nreport: {OUT}")
    print("verdicts:", {k: v for k, v in report["verdicts"].items()
                        if not k.endswith("_legs")})
    return 0


if __name__ == "__main__":
    sys.exit(main())

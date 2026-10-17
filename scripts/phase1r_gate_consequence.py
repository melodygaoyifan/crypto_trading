"""[P1R-6] Consequence table for "alpha-gate friction := realized RT all-in".

    python -X utf8 scripts/phase1r_gate_consequence.py

Read-only. Inputs: data/phase0/gate_replay_ticks.json (per-tick alpha
estimate, threshold, multiplier and friction parsed from the live decision
log since 2026-08-26) and reports/phase0_params.json (realized RT friction,
current-tier projection). For each asset: today's friction (median of the
last 14 days of ticks), the realized figure, the threshold each implies at
the recorded multiplier, and how many of the last 60 days' directional ticks
pass under each. Changes nothing (the switch is an OPERATOR DECISION).
Writes reports/phase1r_gate_consequence.json.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None) -> int:
    tp = os.path.join(REPO, "data", "phase0", "gate_replay_ticks.json")
    pp = os.path.join(REPO, "reports", "phase0_params.json")
    if not (os.path.exists(tp) and os.path.exists(pp)):
        print("REFUSING: inputs missing"); return 2
    ticks = json.load(open(tp, encoding="utf-8"))
    rec = json.load(open(pp, encoding="utf-8"))["recommended"]
    proj = rec["friction_rt_bps_current_tier_projection"]
    realized = rec["friction_rt_bps"]
    last = max(t["t"] for t in ticks)
    end = dt.datetime.fromisoformat(last)
    w60 = (end - dt.timedelta(days=60)).isoformat()
    w14 = (end - dt.timedelta(days=14)).isoformat()
    out = {"window_60d": [w60, last], "assets": {}}
    for a in ("BTC", "ETH", "SOL"):
        rows = [t for t in ticks if t["a"] == a and t.get("fric") and t.get("mult")
                and t.get("thr") is not None and t["thr"] < 1e6 and t["t"] >= w60]
        if not rows:
            out["assets"][a] = {"status": "NO_TICKS"}
            continue
        f_now = st.median([t["fric"] for t in rows if t["t"] >= w14] or [t["fric"] for t in rows])
        mult = st.median([t["mult"] for t in rows])
        # The decision log carries thr/fric only on VETOED ticks (the veto
        # string); a passing tick has neither. Passing ticks therefore count
        # as passes under any LOWER threshold; vetoed directional ticks are
        # re-judged against the new threshold.
        directional = [t for t in rows if (t.get("alpha") or 0) > 0]
        passed_old = [t for t in ticks if t["a"] == a and t["t"] >= w60 and t.get("passed")
                      and abs(t.get("dir_final") or 0) > 0]
        res = {"n_vetoed_ticks_60d": len(rows), "n_vetoed_directional_60d": len(directional),
               "n_passed_directional_60d": len(passed_old),
               "friction_now_rt_bps": round(f_now, 2), "multiplier_median": mult,
               "threshold_now_bps": round(mult * f_now, 1),
               "pass_now": len(passed_old)}
        for name, fr in (("realized_jun_oct", realized[a]), ("current_tier_projection", proj[a])):
            # replace ONLY the friction term; the multiplier (N x regime mult) is
            # carried per tick as thr/fric so the regime scaling is unchanged
            def thr_new(t):
                return t["thr"] * fr / t["fric"]
            res[name] = {"friction_rt_bps": fr,
                         "threshold_bps_median": round(st.median([thr_new(t) for t in rows]), 1),
                         "pass": len(passed_old) + sum(1 for t in directional if t["alpha"] >= thr_new(t)),
                         "newly_passing_vetoed_ticks": sum(1 for t in directional if t["alpha"] >= thr_new(t)),
                         "direction": ("LOOSENS" if fr < f_now else "TIGHTENS")}
        out["assets"][a] = res
    p = os.path.join(REPO, "reports", "phase1r_gate_consequence.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""[P1-2] Replay the decision log under the realized all-in friction.

PRE-COMMITTED RULE (fixed before the first run):
  * Only the FRICTION term of each tick's gate threshold is replaced:
        new_thr = old_thr x new_friction / old_friction
    so the mode multiplier N, the short discount and the regime multiplier
    are carried unchanged by the ratio.
  * Old friction/threshold: parsed from the tick's own veto string when the
    gate rejected; for passing ticks (no string) from the nearest rejected
    tick of the same asset within 72h (label APPROX); else UNREPLAYABLE.
  * NEWLY BLOCKED set (old pass, new fail): sum of REALIZED net PnL of the
    round trips that entered within 60 min after those ticks. If that sum is
    > 0 with |t| >= 2 -> verdict STOP (do not enable enforce).
  * NEWLY ALLOWED set (old fail, new pass) — a LOOSENING: counterfactual
    signed forward return over the model's hold horizon minus the new
    friction. enforce is only defensible if its mean is > 0; otherwise the
    verdict is KEEP_SHADOW.
Inputs (operator-local; pulled from the server, P213):
  --ticks   JSON list of {t,a,alpha,passed,eff_dir,thr,fric}   (diagnostics)
  --closes  JSON {asset: [[epoch_s, close], ...]} 4H closes
  --rts     data/phase0/roundtrips.parquet
  --funding data/phase0/funding_raw.parquet
Exit 0 report written / 2 refusal.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
P0 = os.path.join(REPO, "data", "phase0")


def _ts(s):
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", default=os.path.join(P0, "gate_replay_ticks.json"))
    ap.add_argument("--closes", default=os.path.join(P0, "closes_4h.json"))
    ap.add_argument("--rts", default=os.path.join(P0, "roundtrips.parquet"))
    ap.add_argument("--funding", default=os.path.join(P0, "funding_raw.parquet"))
    ap.add_argument("--out", default=os.path.join(REPO, "reports", "phase1_friction_replay.json"))
    ap.add_argument("--latency-rt-bps", type=float, default=4.0,
                    help="the gate's existing latency, both legs (2.0/leg)")
    a = ap.parse_args(argv)
    for p in (a.ticks, a.closes, a.rts):
        if not os.path.exists(p):
            print(f"REFUSING: {p} missing"); return 2
    import numpy as np
    import pandas as pd
    from execution.cost.realized_friction import RealizedFrictionModel
    model = RealizedFrictionModel(state_path=os.path.join(P0, "_replay_state_unused.json"))
    ticks = [t for t in json.load(open(a.ticks, encoding="utf-8"))
             if t.get("a") in ("BTC", "ETH", "SOL") and (t.get("alpha") or 0) > 0]
    closes = {k: sorted((int(x), float(c)) for x, c in v)
              for k, v in json.load(open(a.closes, encoding="utf-8")).items()}
    lo = min(v[0][0] for v in closes.values()); hi = max(v[-1][0] for v in closes.values())
    RT = pd.read_parquet(a.rts)
    FU = pd.read_parquet(a.funding) if os.path.exists(a.funding) else None

    def close_at(asset, t, k=0):
        v = closes[asset]; xs = [x for x, _ in v]
        i = bisect.bisect_right(xs, t) - 1
        if i < 0 or i + k >= len(v): return None
        return v[i][1], v[i + k][1]

    def fund_at(asset, t):
        if FU is None: return None
        g = FU[(FU.asset == asset) & (FU.ts <= t)]
        if g.empty or t - g.ts.iloc[-1] > 12 * 3600: return None
        return float(g.hourly_rate.iloc[-1])

    for t in ticks:
        t["ts"] = _ts(t["t"])
    rej = {}
    for t in ticks:
        if t.get("fric"):
            rej.setdefault(t["a"], []).append((t["ts"], t["fric"], t["thr"]))
    for v in rej.values():
        v.sort()
    rows = []
    for t in ticks:
        if not (lo <= t["ts"] <= hi): continue
        asset = t["a"]
        side = float(t.get("eff_dir") or 0.0)
        if t.get("fric"):
            old_f, old_thr, src = t["fric"], t["thr"], "PARSED"
        elif not t.get("passed"):
            continue   # rejected for a NON-friction reason: friction cannot change it
        else:
            v = rej.get(asset, [])
            if not v: continue
            j = min(range(len(v)), key=lambda k: abs(v[k][0] - t["ts"]))
            if abs(v[j][0] - t["ts"]) > 72 * 3600: continue
            old_f, old_thr, src = v[j][1], v[j][2], "APPROX"
        h = model.hold_horizon_bars(asset)
        hb = max(1, int(round(h)))
        c = close_at(asset, t["ts"], hb)
        if c is None or c[0] <= 0: continue
        new_f, br = model.friction_rt_bps(asset, side, latency_rt_bps=a.latency_rt_bps,
                                          funding_rate_hourly=fund_at(asset, t["ts"]),
                                          price=c[0], hold_horizon_bars=h)
        new_thr = old_thr * new_f / old_f if old_f > 0 else math.inf
        old_pass = bool(t["passed"])
        new_pass = t["alpha"] >= new_thr
        fwd = (np.sign(side) * (c[1] / c[0] - 1.0) * 1e4) if side else 0.0
        rows.append(dict(ts=t["ts"], asset=asset, alpha=t["alpha"], side=side, src=src,
                         old_fric=old_f, new_fric=new_f, old_thr=old_thr, new_thr=new_thr,
                         old_pass=old_pass, new_pass=new_pass, fwd_bps=fwd, hold_bars=hb))
    D = pd.DataFrame(rows)
    if D.empty:
        print("REFUSING: no replayable ticks in the price window"); return 2
    RT["entry_ts"] = RT.entry_iso.map(_ts)

    def realized_for(sub):
        # each round trip counted ONCE even if several (restart-duplicate)
        # ticks precede it
        ids = set()
        for _, r in sub.iterrows():
            m = RT[(RT.asset == r.asset) & (RT.entry_ts >= r.ts) & (RT.entry_ts <= r.ts + 3600)]
            ids.update(m.rt_id.tolist())
        m = RT[RT.rt_id.isin(ids)]
        return (m.gross_pnl_usd - m.fee_usd).tolist()

    def stats(x):
        x = [float(v) for v in x]
        n = len(x)
        if n == 0: return dict(n=0)
        m = float(np.mean(x)); sd = float(np.std(x, ddof=1)) if n > 1 else float("nan")
        return dict(n=n, sum=float(np.sum(x)), mean=m,
                    t=(m / (sd / math.sqrt(n)) if n > 1 and sd > 0 else None))

    res = {"window": [dt.datetime.fromtimestamp(lo, dt.timezone.utc).isoformat(),
                      dt.datetime.fromtimestamp(hi, dt.timezone.utc).isoformat()],
           "per_asset": {}, "rule": "see module docstring"}
    verdict = "KEEP_SHADOW"
    for asset, g in D.groupby("asset"):
        nb = g[g.old_pass & ~g.new_pass]; na = g[~g.old_pass & g.new_pass]
        rb = realized_for(nb)
        sb = stats(rb)
        sa = stats((na.fwd_bps - na.new_fric).tolist())
        res["per_asset"][asset] = dict(
            ticks=len(g), approx_ticks=int((g.src == "APPROX").sum()),
            old_pass=int(g.old_pass.sum()), new_pass=int(g.new_pass.sum()),
            median_old_friction=float(g.old_fric.median()), median_new_friction=float(g.new_fric.median()),
            newly_blocked_ticks=len(nb), newly_blocked_realized_net_usd=sb,
            newly_allowed_ticks=len(na), newly_allowed_fwd_net_bps=sa)
        if sb.get("n", 0) > 1 and sb["sum"] > 0 and abs(sb.get("t") or 0) >= 2:
            verdict = "STOP"
    allowed = D[~D.old_pass & D.new_pass]
    if verdict != "STOP" and len(allowed):
        res["newly_allowed_pooled"] = stats((allowed.fwd_bps - allowed.new_fric).tolist())
    res["verdict"] = verdict
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

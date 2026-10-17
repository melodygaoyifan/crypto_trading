"""[P430] Quantified gate-fee parity for the P427 fee model, with and without
the ratchet. Replaces "the two must agree" with counts of what changes.

    python -X utf8 scripts/p430_gate_fee_parity.py [--days 60]

Read-only. Inputs: data/phase0/gate_replay_ticks.json (live 4H gate
decisions parsed from the decision log, P1R-6) and Kraken public 4H OHLC
(the tick's price, for the schedule's fixed per-contract term).

Deployed gate (c191da9): taker leg fee = max(3.0 modelled, P334 table).
P427 as written:          max(P1R-1 modelled 10.0, venue schedule).
P427 + P430 ratchet:      max(P427, P334 table).
Only the fee term moves; spread, latency, hold, the 1.1x / regime multiplier
are carried per tick via thr/fric (the P1R-6 method).

Counted over the window, per asset:
  newly_pass  — a tick the deployed gate VETOED that the new fee passes
                (a LOOSENING; the operator rule: must be 0 on every asset
                other than where it is a tightening)
  newly_fail  — a tick the deployed gate passed that the new fee vetoes
                (a tightening). Passing ticks carry no thr/fric in the log,
                so their deployed friction is APPROXIMATED from the nearest
                vetoed tick of the same asset within 72h; ticks with none
                are reported as `newly_fail_unknown`, never as 0.
Exit 3 if the ratchet variant newly passes ANY tick (the deploy gate), else 0;
2 on missing inputs.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import os
import re
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from core.cde_fees import cde_fee_bps, load_venue_fee_schedule, venue_fee_bps  # noqa: E402

PAIRS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD"}
LEG_RE = re.compile(r"friction ([\d.]+)bps = 2x([\d.]+)bps/leg \+ ([\d.]+)bps hold")
MODELLED_OLD, MODELLED_NEW = 3.0, 10.0


def kraken_4h(asset):
    url = f"https://api.kraken.com/0/public/OHLC?pair={PAIRS[asset]}&interval=240"
    with urllib.request.urlopen(url, timeout=20) as r:
        js = json.loads(r.read().decode("utf-8"))
    res = js["result"]
    key = next(k for k in res if k != "last")
    rows = sorted((int(c[0]), float(c[4])) for c in res[key])
    return [r[0] for r in rows], [r[1] for r in rows]


def price_at(ts_list, px_list, ts):
    i = bisect.bisect_right(ts_list, ts) - 1
    return px_list[i] if i >= 0 else None


VOLUME_USD = 0.0  # rolling 30d traded notional -> fee tier; 0 = Intro (dearest)


def fees(asset, px):
    tbl = cde_fee_bps(asset, px, is_maker=False)[0]
    ven = venue_fee_bps(asset, px, is_maker=False, rolling_30d_volume_usd=VOLUME_USD)[0]
    old = max(MODELLED_OLD, tbl)
    # P1R-1's modelled taker leg is the schedule's pct rate AT THE TIER
    # (10.0 Intro, 8.0 from $10k), not a constant
    vfs = load_venue_fee_schedule()
    mod_new = vfs.rate_bps(False, VOLUME_USD) if vfs else MODELLED_NEW
    new = max(mod_new, ven)
    return old, new, max(new, tbl)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=60.0)
    ap.add_argument("--volume-usd", type=float, default=0.0,
                    help="rolling 30d volume for the tier (live 2026-10-17: 10413 -> 8.0bps tier)")
    args = ap.parse_args(argv)
    global VOLUME_USD
    VOLUME_USD = args.volume_usd
    tp = os.path.join(REPO, "data", "phase0", "gate_replay_ticks.json")
    if not os.path.exists(tp):
        print("REFUSING: data/phase0/gate_replay_ticks.json missing (not a verdict)")
        return 2
    ticks = json.load(open(tp, encoding="utf-8"))
    end = max(dt.datetime.fromisoformat(t["t"]) for t in ticks)
    start = end - dt.timedelta(days=args.days)
    out = {"window": [start.isoformat(), end.isoformat()], "volume_usd": VOLUME_USD, "assets": {}}
    any_loosen = False
    for a in ("BTC", "ETH", "SOL"):
        ts_l, px_l = kraken_4h(a)
        rows = []
        skipped = 0
        for t in ticks:
            if t["a"] != a:
                continue
            tt = dt.datetime.fromisoformat(t["t"])
            # An alpha-gate veto ZEROES dir_final, so "the book wanted a
            # position" is the PRE-gate direction (eff_dir); dir_final is the
            # fallback for rows without it.
            pre = t.get("eff_dir")
            if pre is None:
                pre = t.get("dir_final")
            if tt < start or abs(pre or 0) <= 0:
                continue
            m = LEG_RE.search(str(t.get("veto") or ""))
            if m:
                px0 = price_at(ts_l, px_l, tt.timestamp())
                # a logged leg BELOW the deployed table fee predates the
                # per-contract fee arming (P321, 2026-08-19): not the deployed
                # gate, so it cannot be re-priced as one
                if px0 is None or float(m.group(2)) < fees(a, px0)[0]:
                    skipped += 1
                    continue
            rows.append({"ts": tt.timestamp(), "alpha": float(t.get("alpha") or 0.0),
                         "thr": t.get("thr"), "m": m})
        vetoed = [r for r in rows if r["m"] and r["thr"] is not None and r["thr"] < 1e6]
        vts = [r["ts"] for r in vetoed]
        res = {"n_directional": len(rows), "n_vetoed_by_alpha_gate": len(vetoed),
               "n_skipped_pre_p321_leg": skipped}
        # The counts can only observe a loosening on a DIRECTIONAL vetoed tick;
        # an asset whose book was flat on every veto (SOL, this window) yields
        # 0 by construction (P174). The structural check that can fail: the
        # per-leg fee change over EVERY priced tick of the window (directional
        # or not). min >= 0 means no tick, at any alpha, can newly pass.
        all_deltas = {1: [], 2: []}
        for t in ticks:
            if t["a"] != a:
                continue
            tt = dt.datetime.fromisoformat(t["t"])
            if tt < start:
                continue
            px = price_at(ts_l, px_l, tt.timestamp())
            if px is None:
                continue
            f = fees(a, px)
            all_deltas[1].append(f[1] - f[0])
            all_deltas[2].append(f[2] - f[0])
        res["counts_observable"] = bool(vetoed)
        res["min_leg_delta_bps_all_ticks"] = {
            "p427_as_written": min(all_deltas[1]) if all_deltas[1] else None,
            "p427_with_ratchet": min(all_deltas[2]) if all_deltas[2] else None}
        if not all_deltas[2] or min(all_deltas[2]) < 0:
            any_loosen = True
        for name, idx in (("p427_as_written", 1), ("p427_with_ratchet", 2)):
            npass = nfail = nfail_unknown = 0
            leg_deltas = []
            for r in rows:
                px = price_at(ts_l, px_l, r["ts"])
                if px is None:
                    continue
                f = fees(a, px)
                d_leg = f[idx] - f[0]
                leg_deltas.append(d_leg)
                if r in vetoed:
                    fric, leg, hold = (float(x) for x in r["m"].groups())
                    fric_new = 2 * (leg + d_leg) + hold
                    thr_new = r["thr"] * fric_new / fric
                    if r["alpha"] >= thr_new:
                        npass += 1
                else:
                    i = bisect.bisect_left(vts, r["ts"])
                    near = [vetoed[j] for j in (i - 1, i) if 0 <= j < len(vetoed)
                            and abs(vetoed[j]["ts"] - r["ts"]) <= 72 * 3600]
                    if not near:
                        nfail_unknown += 1
                        continue
                    v = min(near, key=lambda x: abs(x["ts"] - r["ts"]))
                    fric, leg, hold = (float(x) for x in v["m"].groups())
                    thr_old = v["thr"]
                    thr_new = thr_old * (2 * (leg + d_leg) + hold) / fric
                    if r["alpha"] >= thr_old and r["alpha"] < thr_new:
                        nfail += 1
            res[name] = {"leg_delta_bps_median": (sorted(leg_deltas)[len(leg_deltas) // 2]
                                                  if leg_deltas else None),
                         "newly_pass": npass, "newly_fail": nfail,
                         "newly_fail_unknown": nfail_unknown}
            if name == "p427_with_ratchet" and npass > 0:
                any_loosen = True
        out["assets"][a] = res
    print(json.dumps(out, indent=1))
    rp = os.path.join(REPO, "reports", f"p430_gate_fee_parity_vol{int(VOLUME_USD)}.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("VERDICT:", "LOOSENS (deploy blocked)" if any_loosen else "ratchet never loosens")
    return 3 if any_loosen else 0


if __name__ == "__main__":
    sys.exit(main())

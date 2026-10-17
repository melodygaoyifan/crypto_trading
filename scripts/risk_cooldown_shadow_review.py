#!/usr/bin/env python3
"""[P429] Shadow read for the sleeve risk-exit rebuild cooldown.

Reads data/risk_exit_cooldown_shadow.jsonl (written by CoinbaseSleeve) and
answers the one question the enforce decision needs: in the shadow window,
did the would-be-suppressed rebuilds pay for themselves?

For every WOULD_SUPPRESS row (a manage call that would have re-added size a
watchdog reduce/exit had just removed):
  * AVOIDED round trip: another watchdog trigger (risk_exit or risk_trigger)
    on the same asset landed inside the cooldown window after it. Enforce
    would have saved the re-add leg AND the next urgent reduce leg.
  * OPPORTUNITY cost: the signed price move (Kraken public 1h close) from the
    would-suppress time to the window end, times the contracts not re-added.
    Positive = enforce would have forgone that much upside.

Pre-committed verdict (fixed before any shadow row exists):
  NOT_YET           ledger span < 7 days (operator 2026-10-18: 48h holds
                    only 1-3 events; read at day 7-14), or zero would_suppress rows
  ENFORCE_ELIGIBLE  span >= 7 days AND avoided round trips >= 1 AND the summed
                    opportunity cost (bps of the re-added notional) is below
                    the summed avoided friction (2 legs x 11bps per avoided
                    round trip, the P427 BTC taker leg, conservative)
  KEEP_SHADOW       otherwise
ENFORCE_ELIGIBLE is still an operator flip (P141): set
coinbase_risk_exit_cooldown_mode: enforce + redeploy.

Read-only, stdlib-only. Run on the host against the data volume:
  python3 scripts/risk_cooldown_shadow_review.py \
      --ledger /var/lib/docker/volumes/hmats-data/_data/risk_exit_cooldown_shadow.jsonl
Exit 0 always on a readable ledger, 2 if the ledger is missing/empty.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any, Dict, List, Optional

LEG_BPS = 11.0          # conservative taker leg (P427 BTC 11.35bps)
MIN_SPAN_H = 7 * 24.0   # [P430] operator: 48h holds 1-3 events; read at 7-14 days
KRAKEN_PAIRS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD",
                "XRP": "XRPUSD", "BNB": "BNBUSD"}


def load(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def kraken_closes(asset: str) -> Dict[int, float]:
    pair = KRAKEN_PAIRS.get(asset)
    if not pair:
        return {}
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=60"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            js = json.loads(r.read().decode("utf-8"))
        res = js.get("result") or {}
        key = next((k for k in res if k != "last"), None)
        return {int(c[0]): float(c[4]) for c in res.get(key, [])}
    except Exception as e:  # noqa: silent-swallow — reported; opportunity cost becomes unknown, never 0
        print(f"  [warn] Kraken OHLC for {asset} unavailable: {e}")
        return {}


def close_at(closes: Dict[int, float], ts: float) -> Optional[float]:
    if not closes:
        return None
    keys = [k for k in closes if k <= ts]
    return closes[max(keys)] if keys else None


def review(rows: List[Dict[str, Any]], fetch=kraken_closes) -> Dict[str, Any]:
    rows = sorted(rows, key=lambda r: float(r.get("ts", 0)))
    span_h = ((float(rows[-1]["ts"]) - float(rows[0]["ts"])) / 3600.0
              if rows else 0.0)
    sup = [r for r in rows if r.get("event") in ("would_suppress", "suppressed")]
    risk = [r for r in rows if r.get("event") in ("risk_exit", "risk_trigger")]
    price_cache: Dict[str, Dict[int, float]] = {}
    avoided = 0
    opp_bps_sum = 0.0
    opp_unknown = 0
    detail = []
    seen_cycle = set()
    for s in sup:
        a, t0 = s.get("asset"), float(s["ts"])
        window_s = float(s.get("window_h") or 24.0) * 3600.0
        exit_ts = t0 - float(s.get("age_h") or 0.0) * 3600.0
        t_end = exit_ts + window_s
        key = (a, round(exit_ts))
        later = [r for r in risk if r.get("asset") == a
                 and t0 < float(r["ts"]) <= t_end]
        hit = bool(later)
        # one rebuild per risk exit: repeated would_suppress rows on later
        # ticks of the same cooldown are the same avoided cycle
        if key in seen_cycle:
            continue
        seen_cycle.add(key)
        if hit:
            avoided += 1
        if a not in price_cache:
            price_cache[a] = fetch(a)
        p0, p1 = close_at(price_cache[a], t0), close_at(price_cache[a], min(t_end, t0 + window_s))
        cur, tgt = float(s.get("current") or 0), float(s.get("target") or 0)
        delta = tgt - float(s.get("held_target") or 0)
        if p0 and p1 and delta:
            sign = 1.0 if delta > 0 else -1.0
            opp = sign * (p1 / p0 - 1.0) * 1e4
            opp_bps_sum += opp
        else:
            opp = None
            opp_unknown += 1
        detail.append({"asset": a, "iso": s.get("iso"), "reason": s.get("reason"),
                       "current": cur, "target": tgt, "retrigger_in_window": hit,
                       "opportunity_bps": opp})
    friction_saved_bps = avoided * 2 * LEG_BPS
    if span_h < MIN_SPAN_H or not sup:
        verdict = "NOT_YET"
    elif avoided >= 1 and opp_unknown == 0 and opp_bps_sum < friction_saved_bps:
        verdict = "ENFORCE_ELIGIBLE"
    else:
        verdict = "KEEP_SHADOW"
    return {"span_h": span_h, "risk_exits": sum(1 for r in risk if r.get("event") == "risk_exit"),
            "risk_triggers_noop": sum(1 for r in risk if r.get("event") == "risk_trigger"),
            "rebuild_cycles": len(detail), "avoided_round_trips": avoided,
            "friction_saved_bps": friction_saved_bps,
            "opportunity_bps_sum": opp_bps_sum, "opportunity_unknown": opp_unknown,
            "verdict": verdict, "detail": detail}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger", default="data/risk_exit_cooldown_shadow.jsonl")
    args = ap.parse_args(argv)
    try:
        rows = load(args.ledger)
    except FileNotFoundError:
        print(f"REFUSING: ledger {args.ledger} not found — no shadow evidence "
              f"yet (this is not a verdict)")
        return 2
    if not rows:
        print(f"REFUSING: ledger {args.ledger} is empty (not a verdict)")
        return 2
    out = review(rows)
    for k in ("span_h", "risk_exits", "risk_triggers_noop", "rebuild_cycles",
              "avoided_round_trips", "friction_saved_bps", "opportunity_bps_sum",
              "opportunity_unknown"):
        v = out[k]
        print(f"{k:24s} {v:.1f}" if isinstance(v, float) else f"{k:24s} {v}")
    for d in out["detail"]:
        print(f"  {d['iso']} {d['asset']} {d['reason']} cur={d['current']:+.0f} "
              f"tgt={d['target']:+.0f} retrigger={d['retrigger_in_window']} "
              f"opp_bps={d['opportunity_bps']}")
    print(f"VERDICT: {out['verdict']}  (enforce is an operator flip, P141)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

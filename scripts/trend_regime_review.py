"""[P198] Forward-evidence readout for the trend regime gate.

Reads data/trend_regime_shadow.jsonl (written every tick by
core/trend_decision_layer.py when trend_regime_gate != "off"), joins each
directional trend signal to its next-4H-bar forward return (Kraken public
OHLC), and prints per-regime performance plus the gated-vs-kept comparison.

This is the number that decides whether `trend_regime_gate` gets promoted
from "shadow" to "enforce". The gate regimes were chosen from the
2026-06-14 -> 2026-08-06 window (WEAK_CONSOLIDATION -9.1bps/tick,
NEUTRAL_DRIFT -68.9bps/tick) — that window is IN-SAMPLE, so only the
FORWARD accumulation this script reads counts as promotion evidence.
Aim for >= 3-4 weeks (>= ~250 directional ticks) before deciding.

Usage (operator laptop — the shadow file must be pulled from the server,
scripts/ is not baked into the engine image per P190):
    scp hmats:/var/lib/docker/volumes/hmats-data/_data/trend_regime_shadow.jsonl /tmp/
    python -X utf8 scripts/trend_regime_review.py --file /tmp/trend_regime_shadow.jsonl

Stdlib-only by design.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from datetime import datetime

PAIRS = {"BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSDT"}
BAR_SEC = 4 * 3600


def fetch_ohlc(pair: str):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=240"
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.load(r)
    key = [k for k in d["result"] if k != "last"][0]
    rows = d["result"][key]
    # [P265] Drop the in-progress last candle (see agent_ic_review.fetch_closes).
    rows = rows[:-1]
    return [int(x[0]) for x in rows], [float(x[4]) for x in rows]


def fwd_return(ts_list, close_list, epoch_ts, h=1):
    """Return over the next h bars after the bar containing epoch_ts."""
    # last bar with open <= epoch_ts (ts_list ascending)
    lo, hi = 0, len(ts_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if ts_list[mid] <= epoch_ts:
            lo = mid + 1
        else:
            hi = mid
    i = lo - 1
    if i < 0 or i + h >= len(close_list):
        return None
    return (close_list[i + h] - close_list[i]) / close_list[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/trend_regime_shadow.jsonl")
    args = ap.parse_args()

    try:
        lines = open(args.file, encoding="utf-8").read().splitlines()
    except OSError as e:
        print(f"ERROR: cannot read {args.file}: {e}")
        return 1
    recs = []
    for ln in lines:
        if not ln.strip():
            continue
        try:
            recs.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    if not recs:
        print(f"ERROR: {args.file} holds no records — the shadow has not "
              f"accumulated. Nothing to review; this is NOT evidence either way.")
        return 1

    ohlc = {a: fetch_ohlc(p) for a, p in PAIRS.items()}

    # regime -> [n, hits, sum_signed, list_signed]; split by gated flag too
    by_regime: dict = {}
    gated_srs, kept_srs = [], []
    n_flat = 0
    t0, t1 = None, None
    for r in recs:
        try:
            ep = datetime.fromisoformat(r["ts"]).timestamp()
        except (KeyError, ValueError):
            continue
        t0 = ep if t0 is None else min(t0, ep)
        t1 = ep if t1 is None else max(t1, ep)
        sig = float(r.get("trend_sig") or 0.0)
        if abs(sig) < 1e-9:
            n_flat += 1
            continue
        a = r.get("asset")
        if a not in ohlc:
            continue
        fr = fwd_return(*ohlc[a], ep, 1)
        if fr is None:
            continue
        sr = (1 if sig > 0 else -1) * fr
        reg = str(r.get("regime") or "UNKNOWN")
        st = by_regime.setdefault(reg, [0, 0, []])
        st[0] += 1
        st[1] += 1 if sr > 0 else 0
        st[2].append(sr)
        (gated_srs if r.get("gated") else kept_srs).append(sr)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    span_d = (t1 - t0) / 86400 if (t0 and t1) else 0.0
    n_dir = sum(v[0] for v in by_regime.values())
    print(f"records: {len(recs)}  span: {span_d:.1f}d  directional: {n_dir}  flat: {n_flat}")
    if n_dir < 250:
        print(f"NOTE: {n_dir} directional ticks < 250 — accumulation is thin; "
              f"treat every number below as provisional.")
    print(f"\n{'regime':<22}{'n':>6}{'hit':>8}{'avg_bps/4h':>12}{'total_bps':>11}")
    for reg, (n, hits, srs) in sorted(by_regime.items(), key=lambda kv: -kv[1][0]):
        print(f"{reg:<22}{n:>6}{hits / n:>8.1%}{_mean(srs) * 1e4:>12.1f}"
              f"{sum(srs) * 1e4:>11.0f}")

    print("\ngate verdict (forward, next-4H-bar signed return):")
    print(f"  GATED ticks (gate would have blocked): n={len(gated_srs)}  "
          f"avg={_mean(gated_srs) * 1e4:+.1f}bps  total={sum(gated_srs) * 1e4:+.0f}bps")
    print(f"  KEPT ticks  (gate would have traded):  n={len(kept_srs)}  "
          f"avg={_mean(kept_srs) * 1e4:+.1f}bps  total={sum(kept_srs) * 1e4:+.0f}bps")
    if gated_srs and kept_srs:
        g, k = _mean(gated_srs) * 1e4, _mean(kept_srs) * 1e4
        if len(gated_srs) >= 100 and g < 0 <= k:
            print("  -> forward evidence SUPPORTS the gate (blocked ticks lose, "
                  "kept ticks don't). Promotion is `trend_regime_gate: \"enforce\"`.")
        elif len(gated_srs) >= 100 and g >= 0:
            print("  -> forward evidence does NOT support the gate (blocked ticks "
                  "were not losing forward). Do not promote; keep shadowing or drop.")
        else:
            print("  -> insufficient or mixed evidence — keep accumulating.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

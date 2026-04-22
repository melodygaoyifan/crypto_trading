"""Deep-dive on quant agent's 33% hit-rate anomaly.

Hit-rate = direction × next-tick price_change > 0. Expected random = 50%.
At 33% over N=368 we have 17pp below random — suggests systematic
inversion, time-window mismatch, or regime-dependent failure.

Breaks down by:
  - asset (BTC / ETH / SOL)
  - regime
  - direction sign (LONG +1 / SHORT -1)
  - confidence bucket

Usage:
    docker exec hmats-trader python -X utf8 scripts/quant_hit_rate_deep_dive.py
    python -X utf8 scripts/quant_hit_rate_deep_dive.py --base . --days 14
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _load(path: Path):
    out = []
    try:
        with open(path, "r", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                line = line.replace(": Infinity,", ": 1e12,").replace(": -Infinity,", ": -1e12,")
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/opt/hmats")
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()

    out_dir = Path(args.base) / "logs" / "attribution"
    files = sorted(out_dir.glob("outcomes_*.jsonl"))[-args.days:]
    print("=" * 76)
    print(f"  quant hit-rate deep-dive  ({len(files)} outcome files)")
    print("=" * 76)

    # Collect every quant outcome with its context
    # rec = {asset, regime, direction, confidence, price_change_pct, correct}
    records = []
    for f in files:
        for rec in _load(f):
            asset = rec.get("asset", "?")
            regime = rec.get("regime", "?")
            pc = rec.get("price_change_pct", 0.0)
            for o in rec.get("outcomes", []):
                if o.get("agent_name") != "quant":
                    continue
                d = float(o.get("direction", 0.0) or 0.0)
                if abs(d) < 0.05:
                    continue  # inactive
                records.append({
                    "asset": asset,
                    "regime": regime,
                    "direction": d,
                    "confidence": float(o.get("confidence", 0.0) or 0.0),
                    "pc": float(pc),
                    "correct": bool(o.get("correct", False)),
                })

    n = len(records)
    if n == 0:
        print("  no quant outcomes — nothing to analyze")
        return
    hr = 100 * sum(1 for r in records if r["correct"]) / n
    print(f"\n  total active quant signals: {n}   overall hit-rate: {hr:.1f}%")

    def _bucket(name: str, key):
        groups = defaultdict(list)
        for r in records:
            groups[key(r)].append(r["correct"])
        print(f"\n  --- by {name} ---")
        print(f"  {'bucket':<22} {'N':>6} {'hit%':>7}  {'delta vs 50%':>14}")
        for k, vs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            n = len(vs)
            hit = 100 * sum(vs) / n
            delta = hit - 50.0
            flag = " [!]" if (n >= 20 and abs(delta) > 10) else ""
            print(f"  {str(k)[:22]:<22} {n:>6} {hit:>6.1f}% {delta:>+13.1f}pp{flag}")

    _bucket("asset", lambda r: r["asset"])
    _bucket("regime", lambda r: r["regime"])
    _bucket("direction sign", lambda r: "LONG" if r["direction"] > 0 else "SHORT")

    # confidence bucket
    def _conf_bucket(c):
        if c < 0.2:
            return "low(<0.2)"
        if c < 0.4:
            return "mid(0.2-0.4)"
        if c < 0.6:
            return "mid-hi(0.4-0.6)"
        return "high(>=0.6)"
    _bucket("confidence", lambda r: _conf_bucket(r["confidence"]))

    # Joint: asset × direction-sign — the most telling split
    print(f"\n  --- asset × direction sign (most diagnostic) ---")
    joint = defaultdict(list)
    for r in records:
        joint[(r["asset"], "LONG" if r["direction"] > 0 else "SHORT")].append(r["correct"])
    print(f"  {'asset':<6} {'dir':<6} {'N':>6} {'hit%':>7}  {'delta':>8}")
    for k, vs in sorted(joint.items()):
        n = len(vs); hit = 100 * sum(vs) / n
        print(f"  {k[0]:<6} {k[1]:<6} {n:>6} {hit:>6.1f}%  {hit-50:>+7.1f}pp")

    # Avg realized pc per signal direction — is quant calling shorts in rallies?
    print(f"\n  --- avg realized price-change by direction ---")
    long_pc = [r["pc"] for r in records if r["direction"] > 0]
    short_pc = [r["pc"] for r in records if r["direction"] < 0]
    if long_pc:
        print(f"  LONG  calls (N={len(long_pc):>4}): avg realized pc={statistics.mean(long_pc):+.3f}%  "
              f"(positive = correct call)")
    if short_pc:
        print(f"  SHORT calls (N={len(short_pc):>4}): avg realized pc={statistics.mean(short_pc):+.3f}%  "
              f"(negative = correct call)")

    # Direction × price_change_pct mean — should be > 0 if agent has edge
    edge = [r["direction"] * r["pc"] for r in records]
    if edge:
        m = statistics.mean(edge)
        print(f"\n  mean(direction × pc) = {m:+.4f}  "
              f"({'POSITIVE edge' if m > 0 else 'NEGATIVE edge — signal is inverted or anti-predictive'})")

    # Directional bias: are calls overwhelmingly LONG or SHORT?
    long_n = sum(1 for r in records if r["direction"] > 0)
    short_n = n - long_n
    print(f"\n  directional balance: LONG={long_n} ({100*long_n/n:.0f}%)  "
          f"SHORT={short_n} ({100*short_n/n:.0f}%)")
    if long_n / max(n, 1) > 0.8:
        print("  [!] >80% LONG calls — agent may be stuck in long-bias")
    elif short_n / max(n, 1) > 0.8:
        print("  [!] >80% SHORT calls — agent may be stuck in short-bias")

    print("\n" + "=" * 76)
    print("  interpretation guide:")
    print("    - hit-rate ~50% across all buckets  →  no signal, but no bug")
    print("    - hit-rate ~67% uniformly inverted   →  sign flip bug somewhere")
    print("    - hit-rate bad in ONE regime/asset   →  strategy selection mis-fit")
    print("    - hit-rate drops with high conf       →  miscalibrated confidence")
    print("=" * 76)


if __name__ == "__main__":
    main()

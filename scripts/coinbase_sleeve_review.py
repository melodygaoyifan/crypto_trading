#!/usr/bin/env python
"""[P150] Coinbase sleeve forward-PnL review — the "real test" readout.

Reads data/coinbase_sleeve_pnl.jsonl (one record per 4H tick, written by
CoinbaseSleeve.log_pnl_point) and prints the live verdict so the
"if it earns we scale / if it doesn't the loss is capped" decision is made on
REAL data, not a backtest.

Usage:
    python -X utf8 scripts/coinbase_sleeve_review.py
    python -X utf8 scripts/coinbase_sleeve_review.py --file data/coinbase_sleeve_pnl.jsonl

On the cloud host:
    docker cp scripts/coinbase_sleeve_review.py hmats-engine:/tmp/
    docker exec hmats-engine python -X utf8 /tmp/coinbase_sleeve_review.py \
        --file /opt/hmats/data/coinbase_sleeve_pnl.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone


def _load(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _fmt_ts(ts):
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join(
        os.environ.get("HMATS_DATA_DIR", "data"), "coinbase_sleeve_pnl.jsonl"))
    args = ap.parse_args()

    if not os.path.exists(args.file):
        print(f"[NO DATA] {args.file} does not exist yet.")
        print("  The sleeve writes one record per 4H tick once Coinbase routing is live.")
        return 0

    rows = _load(args.file)
    if not rows:
        print(f"[NO DATA] {args.file} is empty.")
        return 0

    first, last = rows[0], rows[-1]
    start_eq = first.get("start_equity_usd") or first.get("equity_usd") or 0.0
    cur_eq = last.get("equity_usd") or 0.0
    pnl_usd = last.get("pnl_usd", cur_eq - start_eq)
    pnl_pct = last.get("pnl_pct", (pnl_usd / start_eq) if start_eq else 0.0)

    # span / cadence
    t0, t1 = first.get("ts"), last.get("ts")
    span_days = ((t1 - t0) / 86400.0) if (t0 and t1) else 0.0

    # max drawdown across the equity path
    peak = -math.inf
    max_dd = 0.0
    eqs = [r.get("equity_usd", 0.0) for r in rows]
    for e in eqs:
        peak = max(peak, e)
        if peak > 0:
            max_dd = max(max_dd, (peak - e) / peak)

    # simple per-tick return stats (informational; tiny n early on)
    rets = []
    for a, b in zip(eqs, eqs[1:]):
        if a and a > 0:
            rets.append((b - a) / a)
    n = len(rets)
    mean_r = sum(rets) / n if n else 0.0
    if n > 1:
        var = sum((r - mean_r) ** 2 for r in rets) / (n - 1)
        std_r = math.sqrt(var)
    else:
        std_r = 0.0
    # 4H bars -> 6/day * 365 = 2190 bars/yr
    ann_sharpe = (mean_r / std_r * math.sqrt(2190)) if std_r > 0 else 0.0

    halted = last.get("halted", False)
    positions = last.get("positions") or {}
    pos_str = ", ".join(f"{a}={c}" for a, c in positions.items()) or "FLAT"

    print("=" * 64)
    print("  COINBASE SLEEVE — FORWARD PnL REVIEW  (the real test)")
    print("=" * 64)
    print(f"  window        : {_fmt_ts(t0)} -> {_fmt_ts(t1)} UTC  ({span_days:.1f} days, {len(rows)} ticks)")
    print(f"  start equity  : ${start_eq:,.2f}")
    print(f"  current equity: ${cur_eq:,.2f}")
    print(f"  cumulative PnL: ${pnl_usd:+,.2f}  ({pnl_pct * 100:+.2f}%)")
    print(f"  max drawdown  : {max_dd * 100:.1f}%   (halt fires at 15%)")
    print(f"  realized Sharpe(annualized, naive): {ann_sharpe:+.2f}   [n={n} ticks — noisy until ~30+]")
    print(f"  positions now : {pos_str}")
    print(f"  HALTED        : {halted}" + (f"  <-- sleeve stopped" if halted else ""))
    print("-" * 64)

    # verdict — honest, sample-size aware
    if halted:
        verdict = "HALTED — loss cap engaged. Loss is bounded; do NOT scale. Investigate before reset."
    elif span_days < 14 or n < 20:
        verdict = (f"TOO EARLY — only {span_days:.1f}d / {n} ticks. Need ~2-4wk of live data "
                   "before the earn/don't-earn call. Keep bounded (1 contract/asset).")
    elif pnl_pct > 0.03 and max_dd < 0.10:
        verdict = (f"EARNING — +{pnl_pct*100:.1f}% over {span_days:.0f}d, DD {max_dd*100:.0f}% < cap. "
                   "Candidate to SCALE (raise max_contracts_per_asset by 1, re-evaluate).")
    elif pnl_pct < -0.05:
        verdict = (f"LOSING — {pnl_pct*100:.1f}% over {span_days:.0f}d. Loss is capped at 15%; "
                   "consider flattening / demoting trend before the cap.")
    else:
        verdict = (f"FLAT/INCONCLUSIVE — {pnl_pct*100:+.1f}% over {span_days:.0f}d. "
                   "No clear edge yet; hold bounded and keep collecting.")
    print(f"  VERDICT: {verdict}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

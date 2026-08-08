"""[P230] Realized sleeve beta vs BTC — a live number for the book's one
documented, still-unfixed risk.

The Apr-Jun forensic priced ~half the -25% as BETA (+0.54 net-long into a
-23% market, P143/P144), P201 records that no net-exposure control binds the
sleeve's actual book, and the only beta tool in the tree
(scripts/run_beta_audit.py) reads a Kraken-era file that does not exist — its
last report PRE-DATES the loss it existed to measure (P227b made it refuse
loudly, but it still cannot see the sleeve). Since P150 the sleeve has been
writing an equity point every 4H tick to data/coinbase_sleeve_pnl.jsonl.
This reads that series, aligns it to 4H BTC closes from Kraken's PUBLIC OHLC
endpoint, and reports realized beta / alpha / R^2 plus a long/flat exposure
split — measurement only, no orders, no state.

WHERE THIS RUNS (P213): in-container on the live server, where the data
volume is:
    docker exec hmats-engine python -X utf8 \
        analytics/sleeve_attribution/sleeve_beta_review.py
or operator-local with --pnl-file pointing at an scp'd copy. Missing file or
unreachable price API => REFUSAL (exit 2), never an empty report.

READ THE NUMBER HONESTLY: with the sleeve capped at +/-1 nano contract per
asset, beta is bounded by (gross notional / equity); the interesting output
is the SIGN persistence (how much of the time the book is net-long) and the
alpha intercept — the same decomposition the June forensic did by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

MIN_POINTS = 12  # ~2 days of 4H ticks; below this a regression is decoration


def _refuse(msg: str) -> None:
    print(f"REFUSING TO REPORT: {msg}", file=sys.stderr)
    sys.exit(2)


def load_pnl(path: Path, window_days: int) -> list[dict]:
    if not path.exists():
        _refuse(
            f"{path} does not exist here. The sleeve writes it on the LIVE "
            f"data volume — run in-container or scp the file and pass "
            f"--pnl-file. 'File missing' is not 'sleeve flat'."
        )
    cutoff = datetime.now(timezone.utc).timestamp() - window_days * 86400
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if float(r["ts"]) >= cutoff and r.get("equity_usd"):
                rows.append(r)
        except Exception:  # noqa: silent-swallow — per-record JSONL
            # skip; MIN_POINTS refusal below reports the aggregate
            continue
    if len(rows) < MIN_POINTS:
        _refuse(f"only {len(rows)} usable points in {window_days}d "
                f"(need >= {MIN_POINTS}) — widen --window-days.")
    return rows


def fetch_btc_closes() -> tuple[list[float], list[float]]:
    url = "https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=240"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            payload = json.loads(r.read().decode())
    except Exception as e:
        _refuse(f"Kraken public OHLC unreachable: {type(e).__name__}: {e} — "
                f"no benchmark, no beta. (Network refusal, distinct from "
                f"'no sleeve data'.)")
    if payload.get("error"):
        _refuse(f"Kraken OHLC error: {payload['error']}")
    key = next(k for k in payload["result"] if k != "last")
    rows = payload["result"][key]
    return [float(x[0]) for x in rows], [float(x[4]) for x in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pnl-file", default="data/coinbase_sleeve_pnl.jsonl")
    ap.add_argument("--window-days", type=int, default=60)
    args = ap.parse_args()

    rows = load_pnl(Path(args.pnl_file), args.window_days)
    ts_list, closes = fetch_btc_closes()

    # Pair consecutive sleeve points that are ~one 4H bar apart, and compute
    # the benchmark return over the SAME wall-clock span.
    sleeve_rets, btc_rets, longs = [], [], 0
    for prev, cur in zip(rows, rows[1:]):
        dt = cur["ts"] - prev["ts"]
        if not (0.5 * 14400 <= dt <= 2.5 * 14400):
            continue  # restart gap — a stitched return would be fiction
        e0, e1 = float(prev["equity_usd"]), float(cur["equity_usd"])
        if e0 <= 0:
            continue
        i0 = bisect_right(ts_list, prev["ts"]) - 1
        i1 = bisect_right(ts_list, cur["ts"]) - 1
        if i0 < 0 or i1 <= i0 or i1 >= len(closes):
            continue
        sleeve_rets.append(e1 / e0 - 1.0)
        btc_rets.append(closes[i1] / closes[i0] - 1.0)
        net = sum(float(v or 0) for v in (prev.get("positions") or {}).values())
        if net > 0:
            longs += 1
    n = len(sleeve_rets)
    if n < MIN_POINTS:
        _refuse(f"only {n} contiguous 4H return pairs after gap filtering "
                f"(need >= {MIN_POINTS}).")

    mb = sum(btc_rets) / n
    ms = sum(sleeve_rets) / n
    var_b = sum((b - mb) ** 2 for b in btc_rets)
    if var_b == 0:
        _refuse("benchmark variance is zero over the window.")
    beta = sum((b - mb) * (s - ms) for b, s in zip(btc_rets, sleeve_rets)) / var_b
    alpha_4h = ms - beta * mb
    ss_tot = sum((s - ms) ** 2 for s in sleeve_rets)
    ss_res = sum((s - (alpha_4h + beta * b)) ** 2
                 for s, b in zip(sleeve_rets, btc_rets))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    vol_4h = math.sqrt(ss_tot / n)

    print(f"Sleeve beta vs BTC — {n} contiguous 4H pairs, "
          f"{args.window_days}d window")
    print(f"  beta            {beta:+.3f}")
    print(f"  alpha           {alpha_4h * 1e4:+.1f} bps/4h "
          f"({alpha_4h * 6 * 1e4:+.0f} bps/day)")
    print(f"  R^2             {r2:.3f}   (1-R^2 = idiosyncratic/churn share)")
    print(f"  sleeve vol      {vol_4h * 1e4:.1f} bps/4h")
    print(f"  net-long share  {longs}/{n} bars ({100.0 * longs / n:.0f}%) — "
          f"the +0.54 problem was DIRECTION persistence, watch this line")
    print(f"  equity          ${rows[0]['equity_usd']:,.2f} -> "
          f"${rows[-1]['equity_usd']:,.2f}")
    print("\nContext: P144's 0.50 net budget is enforced by sleeve_exposure "
          "(P208); this measures what the book actually DID. Small n makes "
          "beta noisy — trend, not gospel, below ~180 pairs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

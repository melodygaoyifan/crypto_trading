"""[P404] Gated live-timing / leak check for the P402 ETF-flow shadow.

The ETF signal is LOW-TURNOVER (~24-33 trades/yr), so a short forward window can
never statistically CERTIFY it (P293g clock) — the 2y backtest (P400) + the
complementarity result (P404, etf_sma_overlay_check.py) are the primary evidence.
What the live shadow CAN confirm quickly is the one caveat P400 flagged: that the
live feed's publication timing is leak-free and matches the lag-1 assumption.

This check reads the live etfflow ledger (post-P402 rows, identified by the
`z_score` field) and, once ~2 weeks of completed flow-days have accrued, verifies:
  (1) NO in-progress-day leak: every row's flow_day is strictly BEFORE the row's
      own tick date (the shadow never traded today's still-updating flow, P253c).
  (2) effective lag distribution: how many days after a flow-day it first became
      the 'newest completed' the shadow used — ~1 (weekend-adjusted) matches the
      backtest's lag-1; routinely >3 would mean the feed publishes late (weaker,
      conservative, NOT a leak).
Below the threshold it prints 'accumulating N/THRESHOLD' and exits 0, so the
payoff is read on schedule (P333) rather than from memory.

Exit: 0 accumulating or LEAK-FREE; 2 refuse (no ledger / no post-P402 rows);
      3 LEAK DETECTED (a flow_day on/after its own tick date) — actionable.

Usage: python etfflow_timing_check.py [--ledger-dir DIR] [--min-days N]
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DIR = "data/strategy_shadow"
MIN_DAYS = 10   # ~2 weeks incl. weekends of completed ETF flow-days


def _rows(path: Path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        # post-P402 rows carry z_score; raw-sign era rows do not (P402 discontinuity)
        if "z_score" in r and r.get("flow_day"):
            out.append(r)
    return out


def _date(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc).date()
    except Exception:
        return None


def main() -> int:
    ld = sys.argv[sys.argv.index("--ledger-dir") + 1] if "--ledger-dir" in sys.argv else DEFAULT_DIR
    md = int(sys.argv[sys.argv.index("--min-days") + 1]) if "--min-days" in sys.argv else MIN_DAYS
    base = Path(ld)
    print(f"[P404-TIMING] ETF-flow shadow leak/timing check (ledger={ld})")
    any_ledger = False
    leak = False
    for asset in ("BTC", "ETH"):
        rows = _rows(base / f"etfflow_{asset}.jsonl")
        if rows:
            any_ledger = True
        flow_days = sorted({r["flow_day"] for r in rows})
        if len(flow_days) < md:
            print(f"  {asset}: accumulating {len(flow_days)}/{md} completed flow-days "
                  f"(post-P402 rows={len(rows)})")
            continue
        # (1) leak test + (2) effective lag (first tick date that used each flow_day)
        first_use = {}
        for r in rows:
            fd = _date(r["flow_day"]); td = _date(r.get("iso"))
            if fd is None or td is None:
                continue
            if fd >= td:                      # used today's/future flow -> LEAK
                leak = True
                print(f"  {asset}: LEAK — flow_day {r['flow_day']} >= tick date {td}")
            k = r["flow_day"]
            if k not in first_use or td < first_use[k]:
                first_use[k] = td
        lags = sorted((fu - _date(fd)).days for fd, fu in first_use.items() if _date(fd))
        if lags:
            import statistics
            med = statistics.median(lags)
            print(f"  {asset}: {len(flow_days)} flow-days | effective lag days "
                  f"min={lags[0]} median={med} max={lags[-1]} "
                  f"({'~lag-1 (matches backtest)' if med <= 2 else 'feed publishes late — conservative, weaker'})")
        print(f"  {asset}: leak-free={'NO' if leak else 'YES'}")
    if not any_ledger:
        print("  REFUSED — no etfflow ledger with post-P402 rows found"); return 2
    if leak:
        print("  VERDICT: LEAK DETECTED — do NOT arm; investigate the feed timing"); return 3
    print("  VERDICT: leak-free so far; keep accruing to the arming decision (P141)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

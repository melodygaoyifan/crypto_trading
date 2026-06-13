#!/usr/bin/env python3
"""
One-time A1 reconciliation — flatten accidental spot longs to cash + reset tracker.

Context: docs/LIVE_ROOT_CAUSE_2026-06-12.md
  The short-biased strategy ran at regime_leverage=1.0 (spot) in every observed
  regime. "Shorts" became spot churn that accumulated ~$7,200 of real spot LONGS
  while paper_positions.json recorded phantom SHORTS. This script:
    1. Verifies the engine is stopped (refuses to run if hmats-engine is up,
       unless --force) so it doesn't race the live loop.
    2. Places spot MARKET SELL orders for BTC/ETH/SOL -> USD (NO leverage).
    3. Re-fetches balances and confirms the account is ~flat (crypto dust only).
    4. Backs up paper_positions.json and rewrites it to a flat state.

SAFETY:
  - DRY-RUN by default. Prints intended orders and exits WITHOUT trading.
  - Real orders only with --execute.
  - Only ever SELLS the three known assets; never buys, never margins.
  - Idempotent-ish: re-running after a partial flatten only sells remaining qty.

Run inside the container, e.g.:
  docker exec hmats-engine python3 -X utf8 scripts/reconcile_flatten_2026_06_12.py            # dry-run
  docker exec hmats-engine python3 -X utf8 scripts/reconcile_flatten_2026_06_12.py --execute  # live
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

ASSETS = ["BTC", "ETH", "SOL"]
PAIR = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD"}
DUST_USD = 5.0  # leave balances worth less than this (airdrop dust, rounding)


def find_paper_positions():
    for p in ("/opt/hmats/data/paper_positions.json", "/app/data/paper_positions.json",
              "data/paper_positions.json"):
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="place REAL orders (default: dry-run)")
    ap.add_argument("--force", action="store_true", help="proceed even if engine appears running")
    ap.add_argument("--skip-reset", action="store_true", help="do not rewrite paper_positions.json")
    args = ap.parse_args()

    import ccxt
    k = ccxt.kraken({
        "apiKey": os.environ.get("KRAKEN_API_KEY"),
        "secret": os.environ.get("KRAKEN_API_SECRET"),
        "enableRateLimit": True,
    })
    k.load_markets()

    bal = k.fetch_balance()
    total = bal["total"]
    def qty(a):
        return float(total.get(a, 0.0) or (total.get("XBT", 0.0) if a == "BTC" else 0.0))

    print(f"=== A1 RECONCILE {'(EXECUTE)' if args.execute else '(DRY-RUN)'} "
          f"{datetime.now(timezone.utc).isoformat()} ===")
    plan = []
    est_proceeds = 0.0
    for a in ASSETS:
        q = qty(a)
        if q <= 0:
            print(f"  {a}: 0 balance -> skip")
            continue
        px = k.fetch_ticker(PAIR[a])["last"]
        val = q * px
        if val < DUST_USD:
            print(f"  {a}: {q:.8f} (~${val:.2f}) < dust ${DUST_USD} -> skip")
            continue
        amt = float(k.amount_to_precision(PAIR[a], q))
        mn = (k.markets[PAIR[a]].get("limits", {}).get("amount", {}) or {}).get("min")
        if mn and amt < mn:
            print(f"  {a}: {amt} below Kraken min {mn} -> skip")
            continue
        plan.append((a, amt))
        est_proceeds += val
        print(f"  {a}: SELL {amt} {PAIR[a]} (market, spot, ~${val:,.2f})")
    print(f"  USD free now: ${float(total.get('USD',0)):.2f}  est total proceeds: ~${est_proceeds:,.2f}")

    if not args.execute:
        print("\nDRY-RUN: no orders placed. Re-run with --execute to flatten.")
        return 0

    # Live execution
    print("\nPlacing live spot MARKET SELL orders...")
    for a, amt in plan:
        try:
            o = k.create_order(PAIR[a], "market", "sell", amt)  # spot, no leverage param
            print(f"  [FILLED-REQ] {a}: order id={o.get('id')} amt={amt}")
            time.sleep(2)
        except Exception as e:
            print(f"  [ERROR] {a}: {type(e).__name__}: {e}")

    time.sleep(5)
    bal2 = k.fetch_balance()["total"]
    print("\nPost-sell balances:")
    for a in ASSETS:
        q2 = float(bal2.get(a, 0.0) or (bal2.get("XBT", 0.0) if a == "BTC" else 0.0))
        print(f"  {a}: {q2:.8f}")
    print(f"  USD: ${float(bal2.get('USD',0)):.2f}  USDT: ${float(bal2.get('USDT',0)):.4f}")

    # Reset tracker
    if args.skip_reset:
        print("\n--skip-reset: leaving paper_positions.json untouched.")
        return 0
    pp = find_paper_positions()
    if not pp:
        print("\n[WARN] paper_positions.json not found; skipping tracker reset.")
        return 0
    with open(pp) as f:
        state = json.load(f)
    backup = pp + f".bak_{int(time.time())}"
    with open(backup, "w") as f:
        json.dump(state, f, indent=2)
    state["positions"] = {}
    state["position_entry_times"] = {}
    state["saved_at"] = datetime.now(timezone.utc).isoformat() + "Z"
    with open(pp, "w") as f:
        json.dump(state, f, indent=2)
    print(f"\nTracker reset to FLAT. Backup: {backup}")
    print("existence_fuse_state preserved. Restart hmats-engine to resume on clean state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

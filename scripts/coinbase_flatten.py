#!/usr/bin/env python3
"""
Coinbase flatten — close ALL open Coinbase perp positions to flat.

Use to clean up any orphaned Coinbase position (e.g. after a routing rollback
left a position the engine no longer manages).

SAFETY: DRY-RUN by default (shows positions, places nothing). --execute closes
every open position via the sleeve (opposite-side order to flat).

Run via `!` so YOU trigger the live order(s):
  ! python -X utf8 scripts/coinbase_flatten.py            # dry-run (just shows)
  ! python -X utf8 scripts/coinbase_flatten.py --execute  # CLOSE all to flat
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def run(execute: bool):
    from exchange.coinbase_adapter import CoinbaseAdapter
    from exchange.coinbase_sleeve import CoinbaseSleeve

    s = CoinbaseSleeve(CoinbaseAdapter())
    if not s.is_ready():
        print("[flatten] sleeve not ready")
        return 2
    pos = s.reconcile_positions()
    if not pos:
        print("[flatten] no open Coinbase positions — already FLAT.")
        return 0
    print("[flatten] open positions:")
    for a, p in pos.items():
        print(f"  {a}: {p['side']} {p['contracts']}ct (signed={p['signed_contracts']}) "
              f"uPnL=${p.get('unrealized_pnl', 0):.2f}")

    if not execute:
        print("[flatten] DRY-RUN: nothing closed. Re-run with --execute to flatten all.")
        return 0

    for a in list(pos.keys()):
        # bypass the contract-cap gate for a pure close (target 0 is always a
        # reduction; can_trade allows it since resulting size = 0)
        r = await s.execute_target(a, 0)
        print(f"[flatten] {a}: {r['status']} pos_after={r.get('position_after')}")
    await asyncio.sleep(5)
    s.reconcile_positions()
    print("[flatten] final:", s._last_positions or "FLAT")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="close all open positions")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args.execute)))

#!/usr/bin/env python3
"""
Coinbase exit-management validation — proves manage_to_signal opens AND flattens.

This is the controlled, watched test of the fix for the orphaned-position gap.
It does NOT enable autonomous trading; it directly drives the sleeve's per-tick
management primitive with two synthetic signals and checks the outcome:

  STEP 1: strong directional signal (dir=+0.50) -> target +1 -> OPENS 1 ETH long
  STEP 2: hold/neutral signal     (dir=+0.05) -> target  0 -> FLATTENS to 0

If STEP 2 returns to FLAT, the exit gap is fixed live. ~$166 notional, ~$1-2.

SAFETY: DRY-RUN by default. --execute runs the live open+flatten.
Run via `!`:
  ! python -X utf8 scripts/coinbase_manage_validate.py            # dry-run
  ! python -X utf8 scripts/coinbase_manage_validate.py --execute  # LIVE
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ASSET = "ETH"


async def run(execute: bool):
    from exchange.coinbase_adapter import CoinbaseAdapter
    from exchange.coinbase_sleeve import CoinbaseSleeve

    s = CoinbaseSleeve(CoinbaseAdapter())
    if not s.is_ready():
        print("[manage-validate] sleeve not ready")
        return 2
    s.reconcile_positions()
    if s.signed_contracts(ASSET) != 0:
        print(f"[manage-validate] {ASSET} not flat to start ({s.signed_contracts(ASSET)}ct) "
              f"— run scripts/coinbase_flatten.py first.")
        return 1
    print(f"[manage-validate] {ASSET} flat. plan: dir=+0.50 -> OPEN +1 ; dir=+0.05 -> FLATTEN")
    print(f"  target_for_signal(0.50)={CoinbaseSleeve.target_for_signal(0.50)} "
          f"target_for_signal(0.05)={CoinbaseSleeve.target_for_signal(0.05)}")

    if not execute:
        print("[manage-validate] DRY-RUN: nothing placed.")
        return 0

    print("\n[STEP 1] directional signal dir=+0.50 -> manage_to_signal")
    r1 = await s.manage_to_signal(ASSET, 0.50)
    print("  ->", r1["status"], r1.get("reason", ""))
    await asyncio.sleep(5)
    s.reconcile_positions()
    print("  position:", s.position(ASSET))
    if s.signed_contracts(ASSET) != 1:
        print("  STEP 1 did not open as expected — check + flatten manually.")

    print("\n[STEP 2] HOLD signal dir=+0.05 -> manage_to_signal (must FLATTEN)")
    r2 = await s.manage_to_signal(ASSET, 0.05)
    print("  ->", r2["status"], r2.get("reason", ""))
    await asyncio.sleep(5)
    s.reconcile_positions()
    flat = s.signed_contracts(ASSET) == 0
    print("  position:", s.position(ASSET) or "FLAT")
    print(f"\n[manage-validate] RESULT: open-then-flatten {'PASSED' if flat else 'FAILED — FLATTEN MANUALLY'}")
    return 0 if flat else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="run the live open+flatten test")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args.execute)))

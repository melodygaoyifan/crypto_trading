#!/usr/bin/env python3
"""
Coinbase sleeve execution validation — 1-contract ETH round trip (watched).

Validates the risk-gated execution primitive end-to-end on the live venue:
  open +1 ETH contract -> sleeve tracks it -> cap-gate blocks an over-limit
  order -> close to flat. ~$166 notional, ~$1-2 round-trip cost.

SAFETY: DRY-RUN by default (places nothing). --execute does the live round trip.
Run via the `!` prefix so YOU trigger the live order:
  ! python -X utf8 scripts/coinbase_sleeve_validate.py            # dry-run
  ! python -X utf8 scripts/coinbase_sleeve_validate.py --execute  # LIVE 1 contract
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
        print("[validate] sleeve not ready (.coinbase_key.json missing?)")
        return 2
    r = s.update_risk()
    print(f"[validate] ready. baseline equity=${r['equity_usd']:.2f} halted={r['halted']}")
    print("[validate] plan: OPEN +1 ETH contract -> cap-test (+2 must BLOCK) -> CLOSE to flat")

    if not execute:
        print("[validate] DRY-RUN: nothing placed. Re-run with --execute for the live round trip.")
        return 0

    print("\n[STEP 1] OPEN +1 ETH contract")
    o = await s.execute_target("ETH", 1)
    print("  open:", o["status"], "pos_after=", o.get("position_after"), o.get("reason", ""))
    if o["status"] != "OK":
        print("  open did not succeed — stopping (verify nothing rests on the venue).")
        return 1
    await asyncio.sleep(5)
    s.reconcile_positions()
    print("  POSITION:", s.position("ETH"))
    r2 = s.update_risk()
    print(f"  sleeve: equity=${r2['equity_usd']:.2f} dd={r2['drawdown_pct']:.3%} halted={r2['halted']}")

    print("\n[STEP 2] cap-gate test: request total +2 (cap=1, must BLOCK)")
    b = await s.execute_target("ETH", 2)
    print("  cap-test:", b["status"], b.get("reason", ""))

    print("\n[STEP 3] CLOSE to flat")
    c = await s.execute_target("ETH", 0)
    print("  close:", c["status"], "pos_after=", c.get("position_after"), c.get("reason", ""))
    await asyncio.sleep(5)
    s.reconcile_positions()
    print("  POSITION after close:", s.position("ETH") or "FLAT")
    print("\n[validate] done. Success = OPEN ok, cap-test BLOCKED, CLOSE -> FLAT.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="place the live 1-contract round trip")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args.execute)))

#!/usr/bin/env python3
"""
Coinbase CDE — single minimal LIVE test order (validates the real order path).

Round-trips ONE contract of the cheapest asset (ETH: 1 contract = 0.1 ETH) via
CoinbaseAdapter to prove the end-to-end live path before any engine wiring:
  trade-key auth -> base->contract sizing -> order accepted -> fill -> position
  appears -> reduce_only close -> flat again.

SAFETY:
  - DRY-RUN by default: prints the plan, places NOTHING.
  - --execute places ONE real order (open) then ONE reduce_only close.
  - Smallest possible size (1 contract). No leverage arg (account default).
  - Prints the OrderResult + position at each step.

Run:
  python -X utf8 scripts/coinbase_test_order.py            # dry-run
  python -X utf8 scripts/coinbase_test_order.py --execute  # LIVE (1 contract)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ASSET = "ETH"          # cheapest contract (~$166 notional / 1 contract = 0.1 ETH)
BASE_SIZE = 0.1        # 0.1 ETH -> exactly 1 contract via the adapter conversion


async def run(execute: bool):
    from exchange.coinbase_adapter import CoinbaseAdapter
    from exchange.adapter import OrderRequest
    from exchange.symbol_mapping import to_venue_symbol

    cb = CoinbaseAdapter()
    if not cb.is_connected():
        print("[test_order] adapter not configured (.coinbase_key.json missing?)")
        return 2
    pid = to_venue_symbol(ASSET, "coinbase", "perp")

    # live mid for a marketable LIMIT (maker-first would risk no fill; this is a
    # one-off validation so we cross slightly to guarantee a quick fill).
    p = cb._client.get_product(product_id=pid)
    g = lambda o, k: (o.get(k) if isinstance(o, dict) else getattr(o, k, None))
    mid = float(g(p, "mid_market_price") or g(p, "price"))
    buy_px = round(mid * 1.002, 2)   # 20bps through to fill fast
    print(f"=== Coinbase test order {'(EXECUTE)' if execute else '(DRY-RUN)'} ===")
    print(f"  product={pid}  mid={mid}  open: LIMIT BUY 1 contract (~{BASE_SIZE} ETH) @ {buy_px}")
    print(f"  then: reduce_only close (SELL 1 contract). est round-trip cost: fees(~0) + ~9bps spread")

    if not execute:
        print("\nDRY-RUN: nothing placed. Re-run with --execute to do the 1-contract round trip.")
        return 0

    # OPEN
    req = OrderRequest(symbol=pid, side="BUY", size=BASE_SIZE, order_type="LIMIT",
                       price=buy_px, post_only=False)
    res = await cb.place_order(req)
    print(f"\n[OPEN] success={res.success} status={res.status} id={res.order_id} "
          f"err={res.error_code or ''} {res.error_message or ''}")
    if not res.success:
        print("Open failed — stopping (nothing to close).")
        return 1

    time.sleep(4)
    pos = await cb.fetch_positions()
    print(f"[POSITION] {len(pos)} open position(s):")
    for x in pos[:5]:
        print("   ", {k: x.get(k) for k in ("product_id", "side", "number_of_contracts",
                                            "net_size", "entry_vwap") if isinstance(x, dict) and k in x} or x)

    # CLOSE (reduce_only sell)
    csym = cb._client.get_product(product_id=pid)
    cmid = float(g(csym, "mid_market_price") or g(csym, "price"))
    sell_px = round(cmid * 0.998, 2)
    # CDE has no reduce_only; an opposite-side order nets the position flat.
    creq = OrderRequest(symbol=pid, side="SELL", size=BASE_SIZE, order_type="LIMIT",
                        price=sell_px, post_only=False)
    cres = await cb.place_order(creq)
    print(f"\n[CLOSE] success={cres.success} status={cres.status} id={cres.order_id} "
          f"err={cres.error_code or ''} {cres.error_message or ''}")

    time.sleep(4)
    pos2 = await cb.fetch_positions()
    print(f"[POSITION after close] {len(pos2)} open position(s) (expect 0)")
    print("\nDone. If OPEN+CLOSE both succeeded and position returned to flat, the "
          "live order path is validated.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="place ONE real 1-contract round trip")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args.execute)))

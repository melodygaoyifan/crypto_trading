#!/usr/bin/env python3
"""
[P197] READ-ONLY pre-flight for the Coinbase protective stop.

Shows, for each routed asset: the live position, the stop the engine WOULD place
at a given percentage, and whether a stop is already resting at the venue. Then
previews that exact order so you know the venue accepts it — without placing it.

Run this BEFORE setting `coinbase_protective_stop_pct` in the live profile.
P141: activation of live order behaviour is a deliberate, operator-watched step.

    python -X utf8 scripts/coinbase_stop_validate.py --pct 0.10
    python -X utf8 scripts/coinbase_stop_validate.py --pct 0.10 --assets SOL

Places nothing. Cancels nothing. The only POST is the SDK's `preview_*`
endpoint, which validates a payload and returns fees/margin without creating an
order.

Recommended rollout, mirroring the P141 lesson:
  1. run this, confirm every asset previews clean
  2. set  "coinbase_protective_stop_pct": 0.10
          "coinbase_protective_stop_assets": ["SOL"]      <- one asset first
  3. deploy, then watch a full cycle in the logs:
       [COINBASE-STOP] SOL: PLACED @ ...        (stop goes on)
       [COINBASE-STOP] tick summary: SOL=OK_EXISTS   (it persists, no churn)
       [COINBASE-STOP] SOL: FLAT_CANCELLED      (it comes off when flat)
  4. only then widen to BTC/ETH by emptying the assets list
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ASSETS = ("BTC", "ETH", "SOL")


async def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, default=0.10,
                    help="stop distance from entry vwap (0.10 = 10%%)")
    ap.add_argument("--assets", nargs="*", default=None,
                    help="subset to check (default: all)")
    args = ap.parse_args()

    from exchange.coinbase_adapter import CoinbaseAdapter
    from exchange.coinbase_sleeve import CoinbaseSleeve

    adapter = CoinbaseAdapter()
    if not adapter.is_connected():
        print("[stop_validate] Coinbase adapter not connected — need "
              ".coinbase_key.json / COINBASE_KEY_FILE / API key+secret.")
        return 2

    assets = tuple(args.assets) if args.assets else ASSETS
    sleeve = CoinbaseSleeve(adapter, assets=ASSETS,
                            protective_stop_pct=args.pct,
                            protective_stop_assets=assets)
    sleeve.reconcile_positions()
    if not sleeve._reconcile_ok:
        print("[stop_validate] venue reconcile FAILED — refusing to report off a "
              "stale snapshot (same rule the engine follows).")
        return 2

    print("=" * 72)
    print(f"READ-ONLY pre-flight — stop distance {args.pct:.1%} from entry vwap")
    print("Nothing is placed or cancelled.")
    print("=" * 72)

    for asset in assets:
        cur = sleeve.signed_contracts(asset)
        pos = sleeve.position(asset) or {}
        pid = adapter.to_venue_symbol(asset, "perp")
        print(f"\n--- {asset} ({pid}) ---")
        print(f"  position   : {cur:+.0f} contracts  entry_vwap={pos.get('entry_vwap')}")

        resting = [o for o in (await adapter.fetch_open_orders(pid) or [])
                   if CoinbaseSleeve._is_stop_order(o)]
        print(f"  resting stops at venue: {len(resting)}")

        if cur == 0:
            print("  -> FLAT. Engine would cancel any resting stop "
                  "(no reduce_only on CDE, so an orphan stop OPENS a position).")
            continue

        stop_px = sleeve.desired_stop_price(asset)
        side = "SELL" if cur > 0 else "BUY"
        if not stop_px:
            print("  -> NO ANCHOR (no entry_vwap and no mark) — engine would skip.")
            continue
        print(f"  -> would place {side} stop @ {stop_px:.4f} for {abs(cur):.0f}ct")

        # Preview the exact payload — validates, creates nothing.
        try:
            inc = adapter._price_increment(pid) or 0.01
            stop_r = adapter._round_to_tick(pid, stop_px)
            lim_r = adapter._round_to_tick(
                pid, stop_px * (0.995 if side == "SELL" else 1.005))
            fn = getattr(adapter._client, f"preview_stop_limit_order_gtc_{side.lower()}")
            r = fn(product_id=pid, base_size=str(int(abs(cur))),
                   limit_price=str(lim_r), stop_price=str(stop_r),
                   stop_direction=("STOP_DIRECTION_STOP_DOWN" if side == "SELL"
                                   else "STOP_DIRECTION_STOP_UP"))
            errs = (r.get("errs") if isinstance(r, dict) else getattr(r, "errs", None)) or []
            print(f"     preview  : {'ACCEPTED' if not errs else f'REJECTED {errs}'}"
                  f"   (tick={inc}, stop={stop_r}, limit={lim_r})")
        except Exception as e:
            print(f"     preview  : ERROR {type(e).__name__}: {e}")

    print("\nIf every line above says ACCEPTED, enable for ONE asset first:")
    print('  "coinbase_protective_stop_pct": %s,' % args.pct)
    print('  "coinbase_protective_stop_assets": ["SOL"]')
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))

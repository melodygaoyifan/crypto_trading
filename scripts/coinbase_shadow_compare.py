#!/usr/bin/env python3
"""
Coinbase CDE vs Kraken — READ-ONLY shadow parity comparison (v5.1 Phase 2 SHADOW).

For BTC/ETH/SOL, compares the Coinbase US Perpetual-Style Futures (CDE) venue
against the current Kraken spot venue on: mid price, top-of-book spread (bps),
top-1 depth ($), and funding (Coinbase perp only; Kraken spot has none).

Read-only: market data + funding only. No orders, no money moved. Also
exercises CoinbaseAdapter.fetch_funding_rate + fetch_orderbook against live
responses (SHADOW validation of the adapter).

Run:  python -X utf8 scripts/coinbase_shadow_compare.py
Needs the read-only .coinbase_key.json (gitignored).
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ASSETS = ["BTC", "ETH", "SOL"]
KRAKEN_SPOT = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD"}


def _g(o, k, d=None):
    return (o.get(k, d) if isinstance(o, dict) else getattr(o, k, d))


async def coinbase_side(asset):
    from exchange.coinbase_adapter import CoinbaseAdapter
    from exchange.symbol_mapping import to_venue_symbol
    cb = CoinbaseAdapter()
    if not cb.is_connected():
        return {"error": "coinbase not configured"}
    pid = to_venue_symbol(asset, "coinbase", "perp")
    out = {"product_id": pid}
    # price + contract_size via raw product
    try:
        p = cb._client.get_product(product_id=pid)
        fpd = _g(p, "future_product_details") or {}
        out["mid"] = float(_g(p, "mid_market_price") or _g(p, "price"))
        out["index"] = float(_g(fpd, "index_price") or 0) or None
        out["contract_size"] = float(_g(fpd, "contract_size") or 0)
    except Exception as e:
        out["price_err"] = f"{type(e).__name__}: {e}"
    # funding via adapter (validates the fix)
    try:
        fr = await cb.fetch_funding_rate(pid)
        out["funding_8h"] = fr.funding_rate_8h
        out["funding_period_h"] = fr.funding_period_hours
    except Exception as e:
        out["funding_err"] = f"{type(e).__name__}: {e}"
    # orderbook via adapter (validates parsing)
    try:
        ob = await cb.fetch_orderbook(pid, depth=10)
        bid = ob["bids"][0] if ob["bids"] else None
        ask = ob["asks"][0] if ob["asks"] else None
        if bid and ask:
            mid = (bid[0] + ask[0]) / 2
            out["spread_bps"] = (ask[0] - bid[0]) / mid * 1e4
            cs = out.get("contract_size") or 1.0
            # depth in USD = contracts * contract_size * price
            out["bid_depth_usd"] = bid[1] * cs * bid[0]
            out["ask_depth_usd"] = ask[1] * cs * ask[0]
    except Exception as e:
        out["ob_err"] = f"{type(e).__name__}: {e}"
    return out


def kraken_side(asset):
    import ccxt
    k = ccxt.kraken({"enableRateLimit": True})
    sym = KRAKEN_SPOT[asset]
    out = {"symbol": sym}
    try:
        t = k.fetch_ticker(sym)
        bid, ask = t["bid"], t["ask"]
        mid = (bid + ask) / 2
        out["mid"] = mid
        out["spread_bps"] = (ask - bid) / mid * 1e4
        ob = k.fetch_order_book(sym, 10)
        if ob["bids"] and ob["asks"]:
            out["bid_depth_usd"] = ob["bids"][0][1] * ob["bids"][0][0]
            out["ask_depth_usd"] = ob["asks"][0][1] * ob["asks"][0][0]
    except Exception as e:
        out["err"] = f"{type(e).__name__}: {e}"
    return out


def fmt(v, nd=2):
    return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else str(v)


async def main():
    print("=== Coinbase CDE (perp) vs Kraken (spot) — read-only parity ===\n")
    for a in ASSETS:
        cb = await coinbase_side(a)
        kr = kraken_side(a)
        print(f"--- {a} ---  CB={cb.get('product_id')}  KR={kr.get('symbol')}")
        cb_mid, kr_mid = cb.get("mid"), kr.get("mid")
        if isinstance(cb_mid, (int, float)) and isinstance(kr_mid, (int, float)):
            basis_bps = (cb_mid - kr_mid) / kr_mid * 1e4
            print(f"  mid:        CB {fmt(cb_mid)}   KR {fmt(kr_mid)}   basis {basis_bps:+.1f}bps")
        print(f"  spread_bps: CB {fmt(cb.get('spread_bps'))}   KR {fmt(kr.get('spread_bps'))}")
        print(f"  top depth$: CB bid {fmt(cb.get('bid_depth_usd'),0)} / ask {fmt(cb.get('ask_depth_usd'),0)}"
              f"   KR bid {fmt(kr.get('bid_depth_usd'),0)} / ask {fmt(kr.get('ask_depth_usd'),0)}")
        print(f"  funding(CB perp, 8h): {fmt(cb.get('funding_8h'),6)} "
              f"(period {fmt(cb.get('funding_period_h'),1)}h)   [Kraken spot: none]")
        for k in ("price_err", "funding_err", "ob_err", "error"):
            if cb.get(k):
                print(f"  CB {k}: {cb[k]}")
        if kr.get("err"):
            print(f"  KR err: {kr['err']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

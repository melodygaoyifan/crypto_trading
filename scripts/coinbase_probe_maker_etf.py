#!/usr/bin/env python3
"""
[P270] READ-ONLY probe, three questions before any code is designed:

1. Does CDE HONOR post_only on the nano perp contracts?
   Two previews per asset (creates nothing):
     a) post_only BUY far BELOW the bid (cannot cross)  -> must accept
     b) post_only BUY far ABOVE the ask (would cross)   -> must REJECT
   (a) alone proves only that the field parses; (b) rejecting proves the venue
   actually enforces maker-only. A venue that accepts both is IGNORING the
   field — and a "maker-first" ladder built on it would silently pay taker
   forever while the code reads as maker (the P169 shape).

2. What perp-style contracts does CDE list beyond BTC/ETH/SOL?
   (breadth question — trend/hold transfers to unread assets per P262)

3. Does the CoinGlass plan we already pay for serve BTC spot-ETF flow history?
   (candidate new daily signal basis; probe the endpoint before designing —
   the P218 rule)

Auth: same convention as the sibling probes (.coinbase_key.json /
COINBASE_KEY_FILE / env pair); COINGLASS_API_KEY from env. Read-only keys
suffice. Run (operator, via `!` or docker exec):
    python -X utf8 scripts/coinbase_probe_maker_etf.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from decimal import Decimal, ROUND_DOWN

ASSETS = {
    "BTC": "BIP-20DEC30-CDE",
    "ETH": "ETP-20DEC30-CDE",
    "SOL": "SLP-20DEC30-CDE",
}


def _g(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_tick(value: Decimal, inc: Decimal) -> Decimal:
    if inc <= 0:
        return value
    return (value / inc).to_integral_value(rounding=ROUND_DOWN) * inc


def _client():
    try:
        from coinbase.rest import RESTClient  # type: ignore
    except Exception:
        print("[probe] coinbase-advanced-py not installed")
        return None
    key_file = os.environ.get("COINBASE_KEY_FILE")
    default_kf = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".coinbase_key.json")
    if not key_file and os.path.exists(default_kf):
        key_file = default_kf
    if key_file and os.path.exists(key_file):
        return RESTClient(key_file=key_file)
    ak = os.environ.get("COINBASE_API_KEY")
    sk = os.environ.get("COINBASE_API_SECRET")
    if ak and sk:
        return RESTClient(api_key=ak, api_secret=sk)
    print("[probe] no Coinbase credentials found")
    return None


def probe_post_only(client) -> None:
    print("=" * 72)
    print("PROBE 1: post_only enforcement (preview only — creates nothing)")
    print("=" * 72)
    if not hasattr(client, "preview_limit_order_gtc"):
        print("  SDK lacks preview_limit_order_gtc — cannot settle by preview.")
        return
    for asset, pid in ASSETS.items():
        try:
            prod = client.get_product(product_id=pid)
            bid = Decimal(str(_g(prod, "price") or 0))
            inc = Decimal(str(_g(prod, "price_increment") or "0"))
            if bid <= 0:
                print(f"  {asset}: no price — skip")
                continue
            far_below = _to_tick(bid * Decimal("0.90"), inc)
            far_above = _to_tick(bid * Decimal("1.10"), inc)
            for label, px, expect in (
                    ("non-crossing (must ACCEPT)", far_below, "accept"),
                    ("crossing (must REJECT if honored)", far_above, "reject")):
                resp = client.preview_limit_order_gtc(
                    product_id=pid, side="BUY", base_size="1",
                    limit_price=str(px), post_only=True)
                errs = _g(resp, "errs") or _g(resp, "errors") or []
                print(f"  {asset} post_only BUY @{px} [{label}]: errs={errs}")
        except Exception as e:
            print(f"  {asset}: preview failed: {type(e).__name__}: {e}")


def probe_listings(client) -> None:
    print("=" * 72)
    print("PROBE 2: CDE perp-style contract listings")
    print("=" * 72)
    try:
        prods = client.get_products(product_type="FUTURE")
        rows = _g(prods, "products") or []
        n = 0
        for p in rows:
            pid = _g(p, "product_id", "")
            if not str(pid).endswith("-CDE"):
                continue
            fut = _g(p, "future_product_details")
            print(f"  {pid:<22} price={_g(p, 'price')} "
                  f"vol24h={_g(p, 'volume_24h')} "
                  f"contract_size={_g(fut, 'contract_size') if fut else '?'} "
                  f"expiry_type={_g(fut, 'contract_expiry_type') if fut else '?'}")
            n += 1
        print(f"  total CDE contracts: {n}")
    except Exception as e:
        print(f"  listings failed: {type(e).__name__}: {e}")


def probe_coinglass_etf() -> None:
    print("=" * 72)
    print("PROBE 3: CoinGlass ETF flow endpoints (GET only)")
    print("=" * 72)
    key = os.environ.get("COINGLASS_API_KEY", "")
    if not key:
        print("  COINGLASS_API_KEY not set — skip")
        return
    candidates = [
        "https://open-api-v3.coinglass.com/api/etf/bitcoin/list",
        "https://open-api-v3.coinglass.com/api/etf/bitcoin/flow-history",
        "https://open-api-v3.coinglass.com/api/etf/bitcoin/history",
        "https://open-api-v3.coinglass.com/api/etf/bitcoin/net-assets/history",
    ]
    for url in candidates:
        try:
            req = urllib.request.Request(url, headers={
                "accept": "application/json", "CG-API-KEY": key,
                "coinglassSecret": key})
            with urllib.request.urlopen(req, timeout=15) as r:
                body = r.read().decode("utf-8", "replace")
            d = json.loads(body)
            code = d.get("code")
            data = d.get("data")
            sample = ""
            if isinstance(data, list) and data:
                sample = json.dumps(data[-1])[:220]
            elif isinstance(data, dict):
                sample = json.dumps(data)[:220]
            print(f"  {url.split('/api/')[-1]:<36} code={code} "
                  f"rows={len(data) if isinstance(data, list) else 'obj'}")
            if sample:
                print(f"    sample: {sample}")
        except Exception as e:
            print(f"  {url.split('/api/')[-1]:<36} FAILED: "
                  f"{type(e).__name__}: {e}")


def main() -> int:
    client = _client()
    if client is not None:
        probe_post_only(client)
        probe_listings(client)
    probe_coinglass_etf()
    return 0


if __name__ == "__main__":
    sys.exit(main())

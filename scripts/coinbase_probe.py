#!/usr/bin/env python3
"""
Coinbase US Perpetual-Style Futures — READ-ONLY account/product probe.

Resolves the last v5.1 Phase-2 prep unknowns against the operator's live
account, WITHOUT placing any order or moving any money:
  1. exact perpetual `product_id`s (for the symbol map)
  2. confirms BTC / ETH / SOL are tradable perps on this account
  3. USDC balance in the perpetuals portfolio (margin availability)
  4. per-product leverage caps + funding info (if exposed)

Safe: only GET endpoints (list products, get accounts). A read-only CDP key
is sufficient and recommended.

Auth: Coinbase Advanced Trade uses CDP API keys (ES256 JWT). The official
`coinbase-advanced-py` SDK handles signing. Install + provide creds:

    pip install coinbase-advanced-py
    export COINBASE_API_KEY='organizations/.../apiKeys/...'   # CDP key name
    export COINBASE_API_SECRET='-----BEGIN EC PRIVATE KEY-----\n...'  # PEM

Run:
    python -X utf8 scripts/coinbase_probe.py
"""
from __future__ import annotations

import json
import os
import sys

ASSETS = ("BTC", "ETH", "SOL")


def main() -> int:
    try:
        from coinbase.rest import RESTClient  # type: ignore
    except Exception:
        print("[coinbase_probe] coinbase-advanced-py not installed.\n"
              "  pip install coinbase-advanced-py")
        return 2

    # Preferred: a downloaded CDP key JSON file (no secret handling in shell).
    # Default location is gitignored. Override with COINBASE_KEY_FILE.
    key_file = os.environ.get("COINBASE_KEY_FILE")
    default_kf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              ".coinbase_key.json")
    if not key_file and os.path.exists(default_kf):
        key_file = default_kf

    if key_file and os.path.exists(key_file):
        client = RESTClient(key_file=key_file)
    else:
        key = os.environ.get("COINBASE_API_KEY")
        secret = os.environ.get("COINBASE_API_SECRET")
        if not key or not secret:
            print("[coinbase_probe] No credentials found. Provide EITHER:\n"
                  "  (a) a downloaded CDP key JSON at .coinbase_key.json "
                  "(or set COINBASE_KEY_FILE=path), OR\n"
                  "  (b) COINBASE_API_KEY + COINBASE_API_SECRET env vars.\n"
                  "  A READ-ONLY key is sufficient.")
            return 2
        client = RESTClient(api_key=key, api_secret=secret)

    # --- 1/2: list perpetual futures products ---------------------------------
    print("=== Perpetual futures products (product_type=FUTURE) ===")
    found = {}
    try:
        # SDK paginates; get_products supports product_type filter.
        resp = client.get_products(product_type="FUTURE")
        products = getattr(resp, "products", None) or resp.get("products", [])
        for p in products:
            pid = p.get("product_id") if isinstance(p, dict) else getattr(p, "product_id", "")
            fcm = p.get("future_product_details", {}) if isinstance(p, dict) else {}
            expiry_type = (fcm or {}).get("contract_expiry_type", "")
            base = (p.get("base_currency_id") if isinstance(p, dict)
                    else getattr(p, "base_currency_id", "")) or ""
            is_perp = "PERP" in str(pid).upper() or str(expiry_type).upper() == "PERPETUAL"
            if not is_perp:
                continue
            print(f"  product_id={pid:24} base={base:6} expiry={expiry_type}")
            for a in ASSETS:
                if a in str(pid).upper() or a == str(base).upper():
                    found[a] = pid
    except Exception as e:
        print(f"  [ERROR] get_products failed: {type(e).__name__}: {e}")

    print("\n=== HMATS asset -> Coinbase perp product_id ===")
    for a in ASSETS:
        print(f"  {a}: {found.get(a, '*** NOT FOUND ***')}")

    # --- 3: USDC margin balance ----------------------------------------------
    print("\n=== Accounts (USDC margin availability) ===")
    try:
        accts = client.get_accounts()
        rows = getattr(accts, "accounts", None) or accts.get("accounts", [])
        for ac in rows:
            cur = ac.get("currency") if isinstance(ac, dict) else getattr(ac, "currency", "")
            bal = ac.get("available_balance", {}) if isinstance(ac, dict) else {}
            val = (bal or {}).get("value", "0")
            if cur in ("USDC", "USD") or float(val or 0) > 0:
                print(f"  {cur}: {val}")
    except Exception as e:
        print(f"  [ERROR] get_accounts failed: {type(e).__name__}: {e}")

    print("\nDone. Paste the product_id mapping back to finalize "
          "exchange/symbol_mapping.py (coinbase/perp).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

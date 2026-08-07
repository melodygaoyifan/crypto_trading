#!/usr/bin/env python3
"""
[P195] Coinbase US perp — READ-ONLY probe: does this venue support a
server-side protective stop on the contracts HMATS actually trades?

WHY THIS EXISTS
---------------
The Coinbase sleeve has NO server-side protection. Every exit is a client-side
API call made by the engine on the 4H tick; `exchange/coinbase_adapter.py` has
only MARKET and LIMIT branches. If the process dies, BTC/ETH/SOL perp exposure
sits with nothing resting at the venue to close it.

Before building anything, we need to know whether a resting stop is even
possible here. Public docs conflict: Coinbase Advanced markets
Limit/Market/Stop-Limit/Bracket and the nano contracts are the ones we trade,
but the perpetuals page says Market and Limit only with "more order types" in
progress, and notes "different rules for bracket orders on derivatives markets".

This script ANSWERS THAT QUESTION AND NOTHING ELSE. It places no orders, cancels
nothing, and moves no money — GET endpoints only.

TWO LANDMINES THIS PROBE EXISTS TO DE-RISK
------------------------------------------
1. `OrderRequest` carries a `stop_price` field and documents a "STOP" type
   (exchange/adapter.py:48-50), but the Coinbase adapter IGNORES `stop_price`
   entirely — `order_type="STOP"` falls through the `else` into a plain GTC
   limit. Any future stop work must fix that first or it will silently place
   the WRONG ORDER while looking correct.
2. `reduce_only` is rejected by this venue. coinbase_adapter.py:206-208 records
   "CDE rejects reduce_only ('unknown field' — confirmed via live test
   2026-06-13)… a close is just an opposite-side order." So a protective stop
   here cannot be reduce-only and could, in principle, OPEN a position if the
   underlying one is already gone. That has to be designed around.

Auth: same convention as the sibling scripts — a downloaded CDP key JSON at
.coinbase_key.json (gitignored), or COINBASE_KEY_FILE, or
COINBASE_API_KEY + COINBASE_API_SECRET. A READ-ONLY key is sufficient.

Run (operator, via `!`):
    python -X utf8 scripts/coinbase_probe_stop_support.py
"""
from __future__ import annotations

import json
import os
import sys

ASSETS = ("BTC", "ETH", "SOL")

# The contracts the sleeve actually trades (CLAUDE.md runtime table).
EXPECTED_PRODUCTS = {
    "BTC": "BIP-20DEC30-CDE",
    "ETH": "ETP-20DEC30-CDE",
    "SOL": "SLP-20DEC30-CDE",
}

# Order configurations we care about, in the SDK's naming.
STOP_CONFIG_KEYS = (
    "stop_limit_stop_limit_gtc",
    "stop_limit_stop_limit_gtd",
    "trigger_bracket_gtc",
    "trigger_bracket_gtd",
)


def _g(obj, name, default=None):
    """Attribute-or-key accessor — the SDK returns objects in some paths and
    plain dicts in others."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _client():
    try:
        from coinbase.rest import RESTClient  # type: ignore
    except Exception:
        print("[probe] coinbase-advanced-py not installed.\n"
              "  pip install coinbase-advanced-py==1.8.3")
        return None

    key_file = os.environ.get("COINBASE_KEY_FILE")
    default_kf = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".coinbase_key.json")
    if not key_file and os.path.exists(default_kf):
        key_file = default_kf
    if key_file and os.path.exists(key_file):
        return RESTClient(key_file=key_file)

    key = os.environ.get("COINBASE_API_KEY")
    secret = os.environ.get("COINBASE_API_SECRET")
    if not key or not secret:
        print("[probe] No credentials. Provide a CDP key JSON at "
              ".coinbase_key.json (or COINBASE_KEY_FILE=path), or "
              "COINBASE_API_KEY + COINBASE_API_SECRET. READ-ONLY is enough.")
        return None
    return RESTClient(api_key=key, api_secret=secret)


def main() -> int:
    client = _client()
    if client is None:
        return 2

    print("=" * 72)
    print("READ-ONLY probe — no orders are placed, nothing is cancelled.")
    print("=" * 72)

    verdict = {}

    for asset in ASSETS:
        pid = EXPECTED_PRODUCTS[asset]
        print(f"\n--- {asset}  ({pid}) ---")
        try:
            prod = client.get_product(product_id=pid)
        except Exception as e:
            print(f"  [ERROR] get_product failed: {type(e).__name__}: {e}")
            verdict[asset] = "UNKNOWN (product fetch failed)"
            continue

        print(f"  status              : {_g(prod, 'status')}")
        print(f"  trading_disabled    : {_g(prod, 'trading_disabled')}")
        print(f"  product_type        : {_g(prod, 'product_type')}")
        fpd = _g(prod, "future_product_details") or {}
        if fpd:
            print(f"  contract_expiry_type: {_g(fpd, 'contract_expiry_type')}")
            print(f"  venue               : {_g(fpd, 'contract_display_name') or _g(fpd, 'group_description')}")

        # The authoritative signal, when the venue exposes it: the set of order
        # configurations the product accepts.
        cfgs = (_g(prod, "supported_order_configurations")
                or _g(prod, "order_configurations")
                or _g(fpd, "supported_order_configurations"))
        if cfgs:
            names = [str(c) for c in cfgs] if isinstance(cfgs, (list, tuple)) else [str(cfgs)]
            print(f"  supported order cfgs: {names}")
            has_stop = any(any(k in n for k in STOP_CONFIG_KEYS) for n in names)
            verdict[asset] = "STOP SUPPORTED" if has_stop else "NO STOP CONFIG LISTED"
        else:
            print("  supported order cfgs: (not exposed by this endpoint)")
            verdict[asset] = "INCONCLUSIVE from product metadata"

    # --- can we validate an order without submitting it? ---------------------
    print("\n" + "=" * 72)
    print("Preview / dry-run capability")
    print("=" * 72)
    preview_attrs = [a for a in dir(client)
                     if "preview" in a.lower() and not a.startswith("_")]
    if preview_attrs:
        print(f"  SDK exposes: {preview_attrs}")
        print("  A preview endpoint CAN confirm stop acceptance without submitting.")
        print("  NOT exercised here: previewing is still a POST, and this script's")
        print("  contract is GET-only. Run it deliberately as a follow-up once the")
        print("  payload shape is agreed.")
    else:
        print("  No preview/dry-run method on this SDK version.")
        print("  => Whether a stop is ACCEPTED cannot be settled without submitting")
        print("     a real order. This script will NOT do that. Decide explicitly")
        print("     before any live test, and prefer the smallest contract.")

    # --- summary -------------------------------------------------------------
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    for asset in ASSETS:
        print(f"  {asset}: {verdict.get(asset, 'UNKNOWN')}")
    print("\nReminders before building on this:")
    print("  * exchange/coinbase_adapter.py has no STOP branch — order_type='STOP'")
    print("    silently becomes a GTC LIMIT. Fix that FIRST or the stop is a lie.")
    print("  * reduce_only is rejected by CDE, so a resting stop cannot be")
    print("    reduce-only and could open a position if the original is gone.")
    print("  * If stops are unsupported, the honest outcome is to document the")
    print("    residual risk (0.32x gross leverage, liquidation unreachable) and stop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
nothing, and moves no money.

RESULT (run 2026-08-07 against the live account): **STOPS ARE SUPPORTED.**
A protective stop-limit previews clean on all three contracts —
`errs: []`, and `order_margin_total = 0`, i.e. the venue recognises them as
position-REDUCING rather than as new exposure:

    BTC  SELL 1ct stop=57805  limit=57515   accepted, fee 0.64, lev 3.3
    ETH  SELL 1ct stop=1710.0 limit=1701.0  accepted, fee 0.27, lev 3.0
    SOL  BUY  1ct stop=79.81  limit=80.20   accepted, fee 0.42, lev 2.7

Endpoint note: product metadata does NOT expose supported order configurations,
so the question cannot be settled by GET alone. This uses the SDK's **preview**
endpoint, which validates a payload and returns fees/margin WITHOUT creating an
order. That is a POST, but it is non-mutating — and it is what the P195 plan
sanctioned ("metadata or a preview/validate endpoint"); only *submitting* was
ruled out. This script never calls a non-preview order method.

TWO PAYLOAD FACTS THIS PROBE ESTABLISHED (both cost a round of wrong answers)
----------------------------------------------------------------------------
* `base_size` is in **CONTRACTS**, not base currency: `base_increment = 1` and
  `base_min_size = 1` on all three, with `contract_size` (0.01 / 0.1 / 5) as
  separate metadata. Passing 0.01 for BTC yields
  PREVIEW_INVALID_BASE_SIZE_TOO_SMALL. `CoinbaseAdapter.place_order` already
  converts correctly (`contracts = int(round(size / cs))`) — bypass the adapter
  and you will get this wrong.
* Prices must be a **multiple of `price_increment`** (BTC 5, ETH 0.5, SOL 0.01),
  not merely rounded to its decimal places. `Decimal.quantize(Decimal("5"))`
  rounds to whole numbers, NOT to multiples of 5, and yields
  PREVIEW_INVALID_PRICE_PRECISION. Use `(v / inc).to_integral_value() * inc`, or
  reuse `CoinbaseAdapter._round_to_tick`.

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
from decimal import Decimal, ROUND_DOWN

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


def _to_tick(value: Decimal, inc: Decimal) -> Decimal:
    """Round DOWN to a MULTIPLE of `inc`.

    Not the same as quantize(): `Decimal.quantize(Decimal("5"))` rounds to whole
    numbers, not to multiples of 5, and CDE then rejects the price with
    PREVIEW_INVALID_PRICE_PRECISION. BTC's tick is 5, ETH's is 0.5.
    """
    if inc <= 0:
        return value
    return (value / inc).to_integral_value(rounding=ROUND_DOWN) * inc


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

    # --- the decisive test: preview a protective stop (creates nothing) ------
    print("\n" + "=" * 72)
    print("Protective stop-limit PREVIEW (validates only — no order is created)")
    print("=" * 72)
    if not hasattr(client, "preview_stop_limit_order_gtc_sell"):
        print("  This SDK has no preview_stop_limit_* method, so acceptance cannot")
        print("  be settled without submitting a real order. This script will NOT")
        print("  do that. Upgrade coinbase-advanced-py, or decide explicitly.")
        return 0

    # Direction matches what each LIVE position would need:
    #   long  -> protective SELL stop BELOW the mark
    #   short -> protective BUY  stop ABOVE the mark
    LIVE_SIDE = {"BTC": "sell", "ETH": "sell", "SOL": "buy"}
    for asset in ASSETS:
        pid = EXPECTED_PRODUCTS[asset]
        side = LIVE_SIDE[asset]
        try:
            prod = client.get_product(product_id=pid)
            inc = Decimal(str(_g(prod, "price_increment") or "0.01"))
            mid = Decimal(str(_g(prod, "mid_market_price") or _g(prod, "price")))
            mult = Decimal("0.90") if side == "sell" else Decimal("1.10")
            stop = _to_tick(mid * mult, inc)
            limit = _to_tick(
                stop * (Decimal("0.995") if side == "sell" else Decimal("1.005")), inc)
            fn = getattr(client, f"preview_stop_limit_order_gtc_{side}")
            resp = fn(
                product_id=pid,
                base_size="1",  # CONTRACTS — see module docstring
                limit_price=str(limit),
                stop_price=str(stop),
                stop_direction=("STOP_DIRECTION_STOP_DOWN" if side == "sell"
                                else "STOP_DIRECTION_STOP_UP"),
            )
            errs = _g(resp, "errs") or []
            print(f"\n  {asset}: {side.upper()} 1ct  stop={stop} limit={limit} (mid={mid})")
            print(f"    errs   : {errs}")
            if errs:
                verdict[asset] = f"REJECTED {errs}"
            else:
                print(f"    margin={_g(resp, 'order_margin_total')} "
                      f"fee={_g(resp, 'commission_total')} lev={_g(resp, 'leverage')}")
                # margin 0 => the venue treats it as reducing, not new exposure.
                verdict[asset] = "STOP SUPPORTED (preview accepted)"
        except Exception as e:
            print(f"\n  {asset}: [ERROR] {type(e).__name__}: {str(e)[:300]}")
            verdict[asset] = "UNKNOWN (preview call failed)"

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

"""
Verify live_phase1.json configuration against Kraken API.

Tests:
1. Config loads and all required fields present
2. Kraken API credentials valid (fetch balance)
3. All trading pairs accessible (fetch tickers)
4. Config values within sane bounds
5. Dead-man switch API accessible

Usage:
    python -X utf8 scripts/verify_live_config.py
    python -X utf8 scripts/verify_live_config.py --config configs/live_phase1.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import ccxt

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results = []

def test(name, status, detail=""):
    results.append((name, status, detail))
    icon = {"PASS": "+", "FAIL": "!", "WARN": "~"}[status]
    print(f"  [{icon}] {name}: {status}" + (f" - {detail}" if detail else ""))


def _resolve_config(config_arg: str | None) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not config_arg:
        return os.path.join(root, "configs", "live_high_risk.json")
    if not os.path.isabs(config_arg):
        return os.path.join(root, config_arg)
    return config_arg


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify a live HMATS config against Kraken API.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config path relative to repo root or absolute path. Defaults to configs/live_high_risk.json",
    )
    args = parser.parse_args(argv)

    config_path = _resolve_config(args.config)

    # =========================================================================
    # Section 1: Config Loading
    # =========================================================================
    print("\n[SECTION 1] Configuration Loading")
    print("-" * 50)

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        test("Config loads", PASS, config_path)
    except Exception as e:
        test("Config loads", FAIL, str(e))
        return

    # Required top-level keys
    required_keys = ["mode", "assets", "exchange", "risk", "execution", "regime_leverage"]
    for key in required_keys:
        if key in config:
            test(f"Key '{key}' present", PASS)
        else:
            test(f"Key '{key}' present", FAIL, "MISSING")

    if config.get("mode") != "LIVE":
        test("Mode is LIVE", FAIL, f"Got: {config.get('mode')}")
    else:
        test("Mode is LIVE", PASS)

    # =========================================================================
    # Section 2: Risk Parameter Validation
    # =========================================================================
    print("\n[SECTION 2] Risk Parameters")
    print("-" * 50)

    risk = config.get("risk", {})

    # Max position per asset
    max_pos = risk.get("max_position_pct", 0)
    if 0.05 <= max_pos <= 0.30:
        test("max_position_pct", PASS, f"{max_pos*100:.0f}%")
    else:
        test("max_position_pct", WARN, f"{max_pos*100:.0f}% (outside 5-30% range)")

    # Max total exposure
    max_exp = risk.get("max_exposure_total", 0)
    if 0.20 <= max_exp <= 0.80:
        test("max_exposure_total", PASS, f"{max_exp*100:.0f}%")
    else:
        test("max_exposure_total", WARN, f"{max_exp*100:.0f}%")

    # Max drawdown
    max_dd = risk.get("max_drawdown", 0)
    if 0.05 <= max_dd <= 0.20:
        test("max_drawdown", PASS, f"{max_dd*100:.0f}%")
    else:
        test("max_drawdown", WARN, f"{max_dd*100:.0f}%")

    # Daily loss limit
    dll = risk.get("daily_loss_limit", 0)
    if 0.01 <= dll <= 0.05:
        test("daily_loss_limit", PASS, f"{dll*100:.1f}%")
    else:
        test("daily_loss_limit", WARN, f"{dll*100:.1f}%")

    # Max leverage
    max_lev = risk.get("max_leverage", 1)
    if 1 <= max_lev <= 5:
        test("max_leverage", PASS, f"{max_lev}x")
    else:
        test("max_leverage", WARN, f"{max_lev}x (outside 1-5x range)")

    # Per-asset caps
    caps = risk.get("per_asset_caps", {})
    for asset in config.get("assets", []):
        cap = caps.get(asset, "MISSING")
        if cap == "MISSING":
            test(f"per_asset_cap[{asset}]", FAIL, "Not defined")
        elif 0.05 <= cap <= 0.25:
            test(f"per_asset_cap[{asset}]", PASS, f"{cap*100:.0f}%")
        else:
            test(f"per_asset_cap[{asset}]", WARN, f"{cap*100:.0f}%")

    # =========================================================================
    # Section 3: Regime Leverage Validation
    # =========================================================================
    print("\n[SECTION 3] Regime Leverage")
    print("-" * 50)

    rl = config.get("regime_leverage", {})
    expected_regimes = ["VOLATILE_CHOP", "MOMENTUM_RALLY", "PANIC_SELLOFF",
                        "WEAK_CONSOLIDATION", "QUIET_ACCUMULATION", "EXTREME_VOLATILITY"]
    for regime in expected_regimes:
        lev = rl.get(regime, "MISSING")
        if lev == "MISSING":
            test(f"regime_leverage[{regime}]", WARN, "Not defined (will use 1x)")
        elif 1 <= lev <= risk.get("max_leverage", 3):
            test(f"regime_leverage[{regime}]", PASS, f"{lev}x")
        else:
            test(f"regime_leverage[{regime}]", FAIL, f"{lev}x exceeds max_leverage={max_lev}x")

    # =========================================================================
    # Section 4: Kraken API Connectivity
    # =========================================================================
    print("\n[SECTION 4] Kraken API Connectivity")
    print("-" * 50)

    exch_config = config.get("exchange", {}).get("kraken", {})
    api_key = os.environ.get(exch_config.get("api_key_env", "KRAKEN_API_KEY"), "")
    api_secret = os.environ.get(exch_config.get("api_secret_env", "KRAKEN_API_SECRET"), "")

    if not api_key or not api_secret:
        test("API credentials", FAIL, "KRAKEN_API_KEY or KRAKEN_API_SECRET not set")
        return

    test("API credentials", PASS, f"Key: {api_key[:8]}...{api_key[-4:]}")

    exchange = ccxt.kraken({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    })

    # Fetch balance
    try:
        balance = exchange.fetch_balance()
        usd_free = balance.get("USD", {}).get("free", 0)
        usd_total = balance.get("USD", {}).get("total", 0)
        test("Fetch balance", PASS, f"USD free=${usd_free:,.2f} total=${usd_total:,.2f}")
    except Exception as e:
        test("Fetch balance", FAIL, str(e))

    # Fetch tickers for each asset
    for asset in config.get("assets", []):
        symbol = f"{asset}/USD"
        try:
            ticker = exchange.fetch_ticker(symbol)
            price = ticker.get("last", 0)
            test(f"Fetch ticker {symbol}", PASS, f"${price:,.2f}")
        except Exception as e:
            test(f"Fetch ticker {symbol}", FAIL, str(e))

    # Dead-man switch API
    try:
        result = exchange.cancel_all_orders_after(60000)
        test("Dead-man switch SET", PASS, f"trigger={result['result']['triggerTime']}")
        # Immediately disable
        exchange.cancel_all_orders_after(0)
        test("Dead-man switch DISABLE", PASS)
    except Exception as e:
        test("Dead-man switch", FAIL, str(e))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    passes = sum(1 for _, s, _ in results if s == PASS)
    fails = sum(1 for _, s, _ in results if s == FAIL)
    warns = sum(1 for _, s, _ in results if s == WARN)
    print(f"RESULTS: {passes} PASS, {warns} WARN, {fails} FAIL")

    if fails == 0:
        print(f"VERDICT: {os.path.basename(config_path)} is READY for deployment")
    else:
        print(f"VERDICT: FIX failures before deploying {os.path.basename(config_path)}")
    print("=" * 60)

    return fails == 0


if __name__ == "__main__":
    success = main(sys.argv[1:])
    sys.exit(0 if success else 1)

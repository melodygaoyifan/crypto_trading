#!/usr/bin/env python
"""
futures_vs_spot_fee_analysis.py - Quantify Kraken Futures vs Spot savings (S13)

Analyzes:
  1. Fee savings (Futures maker 0.02% vs Spot maker 0.25%)
  2. Funding rate carry income on short positions
  3. Liquidity comparison (spread quality)
  4. Total annual impact estimate

Usage:
    # With default assumptions (no paper trading data needed)
    python scripts/futures_vs_spot_fee_analysis.py

    # With actual paper trading logs
    python scripts/futures_vs_spot_fee_analysis.py --shadow-ledger data/shadow_ledger

Output:
    scripts/output/futures_vs_spot_report.json
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fee_analysis")


# ---------------------------------------------------------------------------
# Fee Structures
# ---------------------------------------------------------------------------

@dataclass
class FeeStructure:
    """Fee tiers for a market type."""
    maker_bps: float
    taker_bps: float
    label: str


SPOT_FEE = FeeStructure(maker_bps=25.0, taker_bps=40.0, label="Kraken Spot")
SPOT_FEE_FREE = FeeStructure(maker_bps=0.0, taker_bps=0.0, label="Kraken+ Free Tier")
FUTURES_FEE = FeeStructure(maker_bps=2.0, taker_bps=5.0, label="Kraken Futures")

# Kraken+ fee-free volume
FREE_VOLUME_MONTHLY = 10_000.0  # $10K/month


# ---------------------------------------------------------------------------
# Default Assumptions (when no paper trading data)
# ---------------------------------------------------------------------------

@dataclass
class Assumptions:
    """Default assumptions for fee analysis."""
    account_size_usd: float = 10_000.0
    avg_trades_per_day: float = 2.0
    avg_position_size_usd: float = 2_000.0
    avg_hold_hours: float = 24.0        # 6 bars × 4H
    pct_limit_orders: float = 0.70      # 70% maker, 30% taker
    pct_short_trades: float = 0.65      # 65% short bias
    analysis_days: int = 30
    # Funding rate assumption (annualized % for shorts in positive funding)
    avg_positive_funding_rate_8h: float = 0.005  # 0.5% per 8h = typical crypto

    # Spread assumptions (bps)
    spot_spread_bps: Dict[str, float] = field(default_factory=lambda: {
        "BTC": 2.0, "ETH": 3.0, "SOL": 5.0,
    })
    futures_spread_bps: Dict[str, float] = field(default_factory=lambda: {
        "BTC": 3.5, "ETH": 5.0, "SOL": 12.0,
    })


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_fee_savings(assumptions: Assumptions) -> Dict[str, Any]:
    """Calculate fee savings from spot -> futures migration."""
    total_trades = assumptions.avg_trades_per_day * assumptions.analysis_days
    total_volume = total_trades * assumptions.avg_position_size_usd
    monthly_volume = total_volume  # ~30 days

    # Spot fees (accounting for free tier)
    if monthly_volume <= FREE_VOLUME_MONTHLY:
        spot_maker = SPOT_FEE_FREE.maker_bps
        spot_taker = SPOT_FEE_FREE.taker_bps
        fee_tier = "free"
    else:
        # First $10K free, rest at standard rate
        free_pct = FREE_VOLUME_MONTHLY / monthly_volume
        spot_maker = SPOT_FEE.maker_bps * (1 - free_pct)
        spot_taker = SPOT_FEE.taker_bps * (1 - free_pct)
        fee_tier = "standard"

    # Average fee per trade (weighted maker/taker)
    pct_maker = assumptions.pct_limit_orders
    pct_taker = 1.0 - pct_maker

    spot_avg_fee_bps = spot_maker * pct_maker + spot_taker * pct_taker
    futures_avg_fee_bps = FUTURES_FEE.maker_bps * pct_maker + FUTURES_FEE.taker_bps * pct_taker

    # Round-trip = open + close
    spot_rt_bps = spot_avg_fee_bps * 2
    futures_rt_bps = futures_avg_fee_bps * 2

    spot_total_fees_usd = total_trades * assumptions.avg_position_size_usd * spot_rt_bps / 10_000
    futures_total_fees_usd = total_trades * assumptions.avg_position_size_usd * futures_rt_bps / 10_000
    savings_usd = spot_total_fees_usd - futures_total_fees_usd

    return {
        "period_days": assumptions.analysis_days,
        "total_trades": int(total_trades),
        "total_volume_usd": round(total_volume, 2),
        "spot_fee_tier": fee_tier,
        "spot_avg_roundtrip_bps": round(spot_rt_bps, 2),
        "futures_avg_roundtrip_bps": round(futures_rt_bps, 2),
        "spot_total_fees_usd": round(spot_total_fees_usd, 2),
        "futures_total_fees_usd": round(futures_total_fees_usd, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_annualized_usd": round(savings_usd * 365 / assumptions.analysis_days, 2),
    }


def analyze_funding_carry(assumptions: Assumptions) -> Dict[str, Any]:
    """Estimate funding rate carry income on short positions."""
    total_trades = assumptions.avg_trades_per_day * assumptions.analysis_days
    short_trades = total_trades * assumptions.pct_short_trades
    avg_hold_periods_8h = assumptions.avg_hold_hours / 8.0

    # Each 8h period with positive funding = carry income
    carry_per_trade = (
        assumptions.avg_position_size_usd
        * assumptions.avg_positive_funding_rate_8h / 100.0
        * avg_hold_periods_8h
    )
    total_carry = carry_per_trade * short_trades

    # Annualize
    annualized = total_carry * 365 / assumptions.analysis_days
    annualized_pct = annualized / assumptions.account_size_usd * 100

    return {
        "short_trades": int(short_trades),
        "avg_hold_8h_periods": round(avg_hold_periods_8h, 1),
        "carry_per_trade_usd": round(carry_per_trade, 4),
        "total_carry_usd": round(total_carry, 2),
        "annualized_carry_usd": round(annualized, 2),
        "annualized_carry_pct": round(annualized_pct, 2),
    }


def analyze_liquidity(assumptions: Assumptions) -> Dict[str, Any]:
    """Compare spread quality between spot and futures."""
    results = {}
    for asset in ["BTC", "ETH", "SOL"]:
        spot_bps = assumptions.spot_spread_bps.get(asset, 5.0)
        futures_bps = assumptions.futures_spread_bps.get(asset, 10.0)
        ratio = futures_bps / spot_bps if spot_bps > 0 else 999

        if ratio <= 1.3:
            verdict = "GOOD"
        elif ratio <= 2.0:
            verdict = "ACCEPTABLE"
        elif ratio <= 3.0:
            verdict = "MARGINAL"
        else:
            verdict = "WARNING"

        # Liquidity cost = spread difference × total volume for this asset
        # Assuming equal distribution across assets
        trades_per_asset = assumptions.avg_trades_per_day * assumptions.analysis_days / 3
        extra_cost_usd = (
            trades_per_asset * assumptions.avg_position_size_usd
            * (futures_bps - spot_bps) / 10_000 * 2  # round-trip
        )

        results[asset] = {
            "spot_spread_bps": spot_bps,
            "futures_spread_bps": futures_bps,
            "ratio": round(ratio, 2),
            "verdict": verdict,
            "extra_spread_cost_usd": round(max(0, extra_cost_usd), 2),
        }

    return results


def generate_recommendation(
    fee_savings: Dict, carry: Dict, liquidity: Dict
) -> Dict[str, Any]:
    """Generate migration recommendation."""
    savings_annual = fee_savings["savings_annualized_usd"]
    carry_annual = carry["annualized_carry_usd"]

    total_spread_cost = sum(
        v["extra_spread_cost_usd"] for v in liquidity.values()
    )
    spread_cost_annual = total_spread_cost * 365 / fee_savings["period_days"]

    net_annual_impact = savings_annual + carry_annual - spread_cost_annual

    # Per-asset recommendation
    asset_recs = {}
    for asset, liq in liquidity.items():
        if liq["verdict"] in ("GOOD", "ACCEPTABLE"):
            asset_recs[asset] = "MIGRATE"
        elif liq["verdict"] == "MARGINAL":
            asset_recs[asset] = "MONITOR"
        else:
            asset_recs[asset] = "STAY_SPOT"

    migrate_assets = [a for a, r in asset_recs.items() if r == "MIGRATE"]
    if len(migrate_assets) == 3:
        overall = "MIGRATE_ALL"
    elif len(migrate_assets) > 0:
        overall = f"MIGRATE_{'_'.join(migrate_assets)}_ONLY"
    else:
        overall = "STAY_SPOT"

    return {
        "overall": overall,
        "per_asset": asset_recs,
        "net_annual_impact_usd": round(net_annual_impact, 2),
        "breakdown": {
            "fee_savings_annual_usd": round(savings_annual, 2),
            "carry_income_annual_usd": round(carry_annual, 2),
            "extra_spread_cost_annual_usd": round(spread_cost_annual, 2),
        },
    }


# ---------------------------------------------------------------------------
# Shadow ledger reader (optional)
# ---------------------------------------------------------------------------

def read_shadow_ledger(ledger_dir: str) -> Optional[Dict]:
    """Read actual trading data from shadow ledger. Returns None if unavailable."""
    ledger_path = Path(ledger_dir)
    if not ledger_path.exists():
        logger.info(f"Shadow ledger not found at {ledger_dir}, using assumptions")
        return None

    trades = []
    for fpath in sorted(ledger_path.glob("*.jsonl")):
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("type") == "FILL":
                            trades.append(record)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

    if not trades:
        return None

    logger.info(f"Read {len(trades)} FILL records from shadow ledger")

    # Compute actual stats
    total_volume = sum(abs(t.get("notional_usd", 0)) for t in trades)
    avg_size = total_volume / len(trades) if trades else 0

    return {
        "trades": len(trades),
        "total_volume_usd": total_volume,
        "avg_position_size_usd": avg_size,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Futures vs Spot Fee Analysis")
    parser.add_argument("--shadow-ledger", type=str, default="data/shadow_ledger")
    parser.add_argument("--account-size", type=float, default=10_000.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    assumptions = Assumptions(account_size_usd=args.account_size)

    # Try to read actual trading data
    actual = read_shadow_ledger(args.shadow_ledger)
    if actual:
        assumptions.avg_trades_per_day = actual["trades"] / assumptions.analysis_days
        assumptions.avg_position_size_usd = actual["avg_position_size_usd"]

    fee_savings = analyze_fee_savings(assumptions)
    carry = analyze_funding_carry(assumptions)
    liquidity = analyze_liquidity(assumptions)
    recommendation = generate_recommendation(fee_savings, carry, liquidity)

    report = {
        "analysis_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "assumptions_used": "actual" if actual else "default",
        "fee_comparison": fee_savings,
        "funding_carry": carry,
        "liquidity": liquidity,
        "recommendation": recommendation,
    }

    # Write output
    out_path = args.output
    if out_path is None:
        os.makedirs(os.path.join(_PROJECT_ROOT, "scripts", "output"), exist_ok=True)
        out_path = os.path.join(_PROJECT_ROOT, "scripts", "output", "futures_vs_spot_report.json")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    logger.info(f"Report written to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Futures vs Spot Fee Analysis - {report['analysis_date']}")
    print(f"{'='*60}")
    print(f"Data source: {report['assumptions_used']}")

    fs = fee_savings
    print(f"\nFEE SAVINGS ({fs['period_days']}d)")
    print(f"  Trades: {fs['total_trades']}, Volume: ${fs['total_volume_usd']:,.0f}")
    print(f"  Spot fees:    ${fs['spot_total_fees_usd']:>8.2f} ({fs['spot_avg_roundtrip_bps']:.1f} bps RT)")
    print(f"  Futures fees: ${fs['futures_total_fees_usd']:>8.2f} ({fs['futures_avg_roundtrip_bps']:.1f} bps RT)")
    print(f"  Savings:      ${fs['savings_usd']:>8.2f} (${fs['savings_annualized_usd']:,.0f}/yr)")

    c = carry
    print(f"\nFUNDING CARRY")
    print(f"  Short trades: {c['short_trades']}, Avg hold: {c['avg_hold_8h_periods']:.0f} × 8h")
    print(f"  Total carry:  ${c['total_carry_usd']:>8.2f} (${c['annualized_carry_usd']:,.0f}/yr, {c['annualized_carry_pct']:.1f}%)")

    print(f"\nLIQUIDITY")
    for asset, liq in liquidity.items():
        print(f"  {asset}: spot={liq['spot_spread_bps']:.1f}bps, futures={liq['futures_spread_bps']:.1f}bps -> {liq['verdict']}")

    rec = recommendation
    print(f"\nRECOMMENDATION: {rec['overall']}")
    print(f"  Net annual impact: ${rec['net_annual_impact_usd']:+,.0f}")
    for asset, r in rec["per_asset"].items():
        print(f"  {asset}: {r}")

    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()

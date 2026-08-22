"""[P340] What drawdown of SLEEVE EQUITY should each certified book produce?

WHY THIS EXISTS. The sleeve halt is a percentage of sleeve equity, but every
certified figure in this repo is an ADDITIVE per-bar sum at 1.0 exposure (P301
states the convention explicitly). Those are different quantities, and the
record already carries two irreconcilable numbers for the same thing --
P301's breadth table says SOL maxDD -160.8, P321/P325 say -199. Neither is a
sleeve-equity drawdown, so neither can be compared to a 15% halt without the
conversion this script performs.

WHAT IT COMPUTES. The P262-certified trend-only rule (close > SMA200 -> long,
else flat) at the LIVE per-asset fraction, COMPOUNDED, so the result is a real
equity drawdown directly comparable to `max_sleeve_drawdown_pct`.

Costs are charged the P315 way (per-contract, the honest basis) at the
measured CDE half-spreads, and funding carry is charged on every held bar
(P245). Both make the drawdown WORSE, which is the safe direction for sizing
a halt: an under-costed backtest understates the drawdown you must tolerate.

Deliberately NOT a promotion instrument -- it re-expresses an already-certified
mechanism in the units the halt is written in. It changes no config and takes
no view on whether the book should trade.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "training" / "training_data" / "drl_training"

SMA_WINDOW = 200

# Live fractions (configs/live_high_risk.json coinbase_target_fraction_by_asset)
# [P370] moved from flat 0.15 x3 to vol-parity on the strategy-threshold
# audit (48% of book risk sat in SOL at flat 0.15). NOTE for any reader of
# the P340 cold-start figure ("12% of cold starts trip the 15% halt"): it
# was computed at flat 0.15 AND at a 15% halt; both have since moved (halt
# is 25% as of P370), so that figure is stale in BOTH inputs and must be
# re-derived before being quoted again.
LIVE_FRACTION = {"BTC": 0.20, "ETH": 0.15, "SOL": 0.095}

# P289 measured CDE full spreads; taker pays half. Per-leg cost in bps.
HALF_SPREAD_BPS = {"BTC": 1.0, "ETH": 2.75, "SOL": 2.0}
# P315: the fee is a FLAT per-contract charge, ~9.4bps (BTC) / 13.8 (ETH) at
# current prices. Charged per leg, on top of the spread.
PER_CONTRACT_FEE_BPS = {"BTC": 9.4, "ETH": 13.8, "SOL": 17.4}


def trend_only_position(close: pd.Series) -> pd.Series:
    """The P262/P247-certified rule: long above the 200-bar SMA, else flat.

    Shifted one bar: the decision uses information available at the close of
    the PREVIOUS bar, never the bar it acts in (P164).
    """
    sma = close.rolling(SMA_WINDOW, min_periods=SMA_WINDOW).mean()
    pos = (close > sma).astype(float)
    return pos.shift(1).fillna(0.0)


def equity_curve(close: pd.Series, pos: pd.Series, asset: str,
                 fraction: float, carry_bps_per_bar: float) -> pd.Series:
    """Compounded sleeve-equity curve at the live fraction."""
    ret = close.pct_change().fillna(0.0)
    gross = pos * ret * fraction

    leg_bps = HALF_SPREAD_BPS[asset] + PER_CONTRACT_FEE_BPS[asset]
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (leg_bps / 1e4) * fraction

    carry = pos.abs() * (carry_bps_per_bar / 1e4) * fraction

    net = gross - cost - carry
    return (1.0 + net).cumprod()


def max_drawdown(curve: pd.Series) -> float:
    peak = curve.cummax()
    return float(((curve / peak) - 1.0).min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="BTC,ETH,SOL")
    ap.add_argument("--halt", type=float, default=0.15,
                    help="max_sleeve_drawdown_pct to compare against")
    ap.add_argument("--carry-bps-per-bar", type=float, default=0.5,
                    help="funding charged per held 4H bar (P245 convention)")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    out = {"halt": args.halt, "carry_bps_per_bar": args.carry_bps_per_bar,
           "rows": []}
    print(f"{'asset':6s} {'frac':>5s} {'bars':>6s} {'net%':>9s} "
          f"{'maxDD(sleeve)':>14s} {'vs halt':>9s} {'frac to fit':>12s}")
    for asset in [a.strip().upper() for a in args.assets.split(",") if a.strip()]:
        f = DATA / f"{asset}_4H_ohlcv.parquet"
        if not f.exists():
            print(f"  {asset}: REFUSING — no price series at {f}")
            return 2
        df = pd.read_parquet(f)
        close = df["close"].astype(float).reset_index(drop=True)
        pos = trend_only_position(close)
        frac = LIVE_FRACTION.get(asset, 0.15)

        curve = equity_curve(close, pos, asset, frac, args.carry_bps_per_bar)
        dd = max_drawdown(curve)
        net = float(curve.iloc[-1] - 1.0)

        # The fraction at which this book's worst drawdown equals the halt.
        # Linear in fraction to first order, so scale by the ratio.
        fit = frac * (args.halt / abs(dd)) if dd < 0 else float("nan")

        flag = "OK" if abs(dd) < args.halt else "BREACHES"
        print(f"{asset:6s} {frac:5.2f} {len(df):6d} {100*net:8.1f}% "
              f"{100*dd:13.1f}% {flag:>9s} {fit:12.3f}")
        out["rows"].append({
            "asset": asset, "fraction": frac, "bars": int(len(df)),
            "net_pct": round(100 * net, 2),
            "max_dd_sleeve_pct": round(100 * dd, 2),
            "breaches_halt": bool(abs(dd) >= args.halt),
            "fraction_that_fits_halt": None if dd >= 0 else round(fit, 4),
        })

    if args.report:
        Path(args.report).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

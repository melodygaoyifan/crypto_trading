"""[P373] Does cross-sectional momentum across the 8 assets beat buy-and-hold?
The one offline-backtestable September candidate that was never run.

WHY THIS, WHY NOW. The operator asked "do we have to wait for September." The
answer for most candidates is no — they are backtestable from data on disk and
have ALREADY been run, each coming back not-a-profit-lever: donchian/emaens
(P288, risk-preference only), banded (P259b, failed validation), funding legs
(P297, fails the one criterion that also rejects B&H), sentiment (P296, ~=B&H),
breadth (P301, 0/8 loses to B&H on return). `xsmom` is the single exception —
P277 built it as a FORWARD ledger and nobody backtested it, though its input
(6y hourly closes for all 8 assets) has been on disk the whole time (P296's
own triage lists it as offline-backtestable).

THE BAR, pre-committed before the first number (the operator's real objective):
  xsmom EARNS iff its net-of-cost total return beats EQUAL-WEIGHT buy-and-hold
  of the same 8-asset universe, positive in >= 2 of 3 eras. Sharpe/drawdown are
  reported but do NOT decide — the question on the table is "make money," i.e.
  beat doing nothing on RETURN. This is the bar every other candidate failed.

HONEST CAVEATS, stated before the result:
  * EXECUTABILITY: the sleeve routes only BTC/ETH/SOL (P292: the other 5 have
    no SYMBOL_MAP perp entry and thin volume). An 8-asset xsmom is a SIGNAL
    test, not an executable strategy without a routing+sizing expansion that is
    itself an operator decision. A 3-asset variant is also run, but "3
    correlated majors are not a cross-section" (P277).
  * Hourly closes; costs are the measured CDE round trip where known
    (BTC 27.7 / ETH 44.0 / SOL 41.0 bps) and an assumed 45bps for the 5
    unmeasured alts (P167: an unmeasured cost is assumed expensive). Shorts on
    alts are charged the same round trip; funding carry is NOT modelled here
    (it would only make the short legs worse, so the long-only verdict is the
    conservative one and is the one that decides).
  * Additive per-bar return sums, the convention of this repo's labs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "training" / "training_data" / "raw"
UNIVERSE = ("BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "LTC")
COST_RT = {"BTC": 27.7e-4, "ETH": 44.0e-4, "SOL": 41.0e-4,
           "BNB": 45e-4, "XRP": 45e-4, "ADA": 45e-4, "DOGE": 45e-4, "LTC": 45e-4}
ERAS = {"2020-22": ("2020-01-01", "2023-01-01"),
        "2023-24": ("2023-01-01", "2025-01-01"),
        "2025-26": ("2025-01-01", "2027-01-01")}
LOOKBACK_BARS = 30 * 24          # 30-day trailing momentum, hourly
REBAL_BARS = 24                  # rebalance daily


def load_panel(assets) -> pd.DataFrame:
    cols = {}
    for a in assets:
        df = pd.read_parquet(RAW / f"{a}_60m.parquet")[["timestamp", "close"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        cols[a] = df.set_index("timestamp")["close"]
    panel = pd.DataFrame(cols).sort_index().ffill().dropna()
    return panel


def backtest(panel: pd.DataFrame, assets, *, k: int, long_only: bool) -> dict:
    px = panel[list(assets)].to_numpy(float)
    ts = panel.index
    n, m = px.shape
    ret = np.vstack([np.zeros((1, m)), px[1:] / px[:-1] - 1.0])
    mom = np.full((n, m), np.nan)
    mom[LOOKBACK_BARS:] = px[LOOKBACK_BARS:] / px[:-LOOKBACK_BARS] - 1.0
    costs = np.array([COST_RT[a] for a in assets])

    pos = np.zeros(m)
    pnl = np.zeros(n)
    for i in range(LOOKBACK_BARS + 1, n):
        pnl[i] = float(np.dot(pos, ret[i]))
        if (i % REBAL_BARS) == 0 and not np.isnan(mom[i]).any():
            order = np.argsort(mom[i])
            new = np.zeros(m)
            longs = order[-k:]
            new[longs] = 1.0 / k
            if not long_only:
                shorts = order[:k]
                new[shorts] = -1.0 / k
            pnl[i] -= float(np.dot(np.abs(new - pos), costs))
            pos = new
    return _stats(pnl, ts)


def bnh(panel: pd.DataFrame, assets) -> dict:
    px = panel[list(assets)].to_numpy(float)
    ts = panel.index
    ret = np.vstack([np.zeros((1, px.shape[1])), px[1:] / px[:-1] - 1.0])
    w = np.full(px.shape[1], 1.0 / px.shape[1])
    pnl = ret @ w
    return _stats(pnl, ts)


def _stats(pnl: np.ndarray, ts: pd.Index) -> dict:
    yrs = len(pnl) / 24 / 365.25
    eq = np.cumprod(1 + pnl)
    dd = (np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)
    sd = pnl.std()
    out = {"total_pct": round(float(pnl.sum() * 100), 1),
           "sharpe": round(float(pnl.mean() / sd * np.sqrt(24 * 365.25)), 2) if sd > 0 else 0.0,
           "maxdd_pct": round(float(dd.max() * 100), 1),
           "eras": {}}
    for name, (a, b) in ERAS.items():
        mask = np.asarray((ts >= a) & (ts < b))
        if mask.sum() > 24 * 30:
            out["eras"][name] = round(float(pnl[mask].sum() * 100), 1)
    return out


def main() -> int:
    results = {}
    # 8-asset signal test
    panel8 = load_panel(UNIVERSE)
    b8 = bnh(panel8, UNIVERSE)
    results["universe8"] = {"years": round(len(panel8) / 24 / 365.25, 2),
                            "buy_and_hold_eqw": b8, "variants": {}}
    for k in (2, 3):
        for lo in (True, False):
            tag = f"top{k}_{'long_only' if lo else 'long_short'}"
            results["universe8"]["variants"][tag] = backtest(panel8, UNIVERSE, k=k, long_only=lo)
    # 3-asset (executable today) variant
    panel3 = load_panel(("BTC", "ETH", "SOL"))
    b3 = bnh(panel3, ("BTC", "ETH", "SOL"))
    results["universe3_executable"] = {"buy_and_hold_eqw": b3,
        "top1_long_only": backtest(panel3, ("BTC", "ETH", "SOL"), k=1, long_only=True)}

    (REPO / "training" / "reports" / "xsmom_backtest_p373.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    W = 82
    print("=" * W)
    print("  XSMOM BACKTEST — cross-sectional momentum vs equal-weight buy-and-hold")
    print("  BAR: beat B&H on TOTAL RETURN, positive in >=2 of 3 eras (pre-committed)")
    print("=" * W)
    u = results["universe8"]
    print(f"\n8-ASSET universe ({u['years']}y)   [signal test; not executable w/o expansion, P292]")
    bh = u["buy_and_hold_eqw"]
    print(f"  {'variant':22s} {'net %':>9s} {'Sharpe':>7s} {'maxDD%':>8s}   eras")
    print(f"  {'buy_and_hold (eqw)':22s} {bh['total_pct']:+9.1f} {bh['sharpe']:>7.2f} {bh['maxdd_pct']:>8.1f}   "
          + " ".join(f"{v:+.0f}" for v in bh['eras'].values()))
    for tag, s in u["variants"].items():
        beats = s["total_pct"] > bh["total_pct"]
        eras_pos = sum(1 for v in s["eras"].values() if v > 0)
        verdict = "EARNS" if (beats and eras_pos >= 2) else "fails"
        print(f"  {tag:22s} {s['total_pct']:+9.1f} {s['sharpe']:>7.2f} {s['maxdd_pct']:>8.1f}   "
              + " ".join(f"{v:+.0f}" for v in s['eras'].values())
              + f"   [{verdict}: vs B&H {'+' if beats else '-'}, {eras_pos}/3 eras+]")
    e = results["universe3_executable"]
    print(f"\n3-ASSET executable (BTC/ETH/SOL) — 'not really a cross-section' (P277)")
    print(f"  {'buy_and_hold (eqw)':22s} {e['buy_and_hold_eqw']['total_pct']:+9.1f}")
    print(f"  {'top1_long_only':22s} {e['top1_long_only']['total_pct']:+9.1f}   "
          f"vs B&H {'BEATS' if e['top1_long_only']['total_pct']>e['buy_and_hold_eqw']['total_pct'] else 'LOSES'}")
    print(f"\nreport -> training/reports/xsmom_backtest_p373.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

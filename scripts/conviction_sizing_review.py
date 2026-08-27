#!/usr/bin/env python3
"""[WS2] Forward reader for the conviction-sizing shadow — the PnL increment of
the conviction-sized book over the 1x trend base, from the accrued ledger.

A sizing overlay is judged on PnL, not rank-IC, so this is the honest instrument
(the forward version of training/conviction_sizing_lab.py). Self-contained:
forward returns come from the ledger's own recorded close, no external fetch.

    python -X utf8 scripts/conviction_sizing_review.py [--data-dir data] [--cost-bps-btc 27.7 --cost-bps-eth 44.0]

Exit 2 if a ledger is missing/too thin (no verdict from thin data, P199/P348).
"""
import argparse
import json
import sys
from pathlib import Path

ASSETS = ("BTC", "ETH")
MIN_ROWS = 30                        # below this, no verdict (thin)

# [P374] measured per-RT market impact, the one term no registry carries; the
# fee and spread terms below are DERIVED from their single sources so a fee
# revision (P315 rule) propagates here without a hand edit.
IMPACT_BPS = {"BTC": 6.38, "ETH": 9.72}


def _derive_cost():
    """[P420] honest CDE round-trip bps = 2 x taker fee (core.cde_fees) +
    full spread (defense.constitution CDE_SPREAD_BPS_MEASURED) + measured
    impact (P374). The old hand-copied {27.7, 44.0} is what this reproduces
    (within 0.5bps — the P374 decomposition used the measured 1.58/5.26bps
    spreads where the registry carries the conservative 2.0/5.5); the parity
    with training/conviction_sizing_lab.COST_BPS is pinned by test so the
    forward reader and the lab it forward-tests cannot silently diverge."""
    try:
        from core.cde_fees import CDE_FEE_BPS
        from defense.constitution import CDE_SPREAD_BPS_MEASURED
        return {a: round(2.0 * CDE_FEE_BPS[a]["taker"]
                         + CDE_SPREAD_BPS_MEASURED[a] + IMPACT_BPS[a], 1)
                for a in ASSETS}
    except Exception as e:  # noqa: silent-swallow — logged; the recorded P374 figures are the degraded state
        print(f"WARNING: could not derive the CDE cost ({type(e).__name__}: {e}) "
              f"— using the recorded P374 figures 27.7/44.0 (P420)")
        return {"BTC": 27.7, "ETH": 44.0}


COST = _derive_cost()


def _maxdd(cum):
    # high-water mark starts at 0 (equity before any bar accrues), so a book
    # that only ever loses still shows its full drawdown from the start.
    peak = 0.0
    worst = 0.0
    for x in cum:
        peak = max(peak, x)
        worst = min(worst, x - peak)
    return worst


def _book(close, pos, cost_bps):
    """Forward net + maxDD of a position series over the ledger's own closes."""
    n = len(close)
    net = 0.0
    cum = []
    running = 0.0
    prev = 0.0
    for t in range(n - 1):
        ret = (close[t + 1] / close[t] - 1.0) if close[t] else 0.0
        fee = abs(pos[t] - prev) * (cost_bps / 2.0) / 1e4
        bar = pos[t] * ret - fee
        net += bar
        running += bar
        cum.append(running)
        prev = pos[t]
    return net, _maxdd(cum)


def review(data_dir="data", cost=None):
    cost = cost or COST
    d = Path(data_dir) / "conviction_shadow"
    any_thin = False
    print(f"{'asset':6s} {'rows':>5s} {'base net/DD':>16s} {'conv net/DD':>16s} "
          f"{'increment':>10s} {'verdict':>8s}")
    print("-" * 70)
    for a in ASSETS:
        f = d / f"convsize_{a}.jsonl"
        if not f.exists():
            print(f"{a:6s} (no ledger at {f})"); any_thin = True; continue
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [r for r in rows if r.get("reason") == "ok"]  # skip warmup rows
        if len(rows) < MIN_ROWS:
            print(f"{a:6s} {len(rows):>5d}  (thin — need >={MIN_ROWS}, no verdict)")
            any_thin = True
            continue
        close = [float(r["close"]) for r in rows]
        base = [float(r["base_pos"]) for r in rows]
        conv = [float(r["conv_pos"]) for r in rows]
        bn, bd = _book(close, base, cost[a])
        cn, cd = _book(close, conv, cost[a])
        inc = cn - bn
        verdict = "conv+" if inc > 0 else "base+"
        print(f"{a:6s} {len(rows):>5d} {bn:>+8.4f}/{bd:>+6.3f} "
              f"{cn:>+8.4f}/{cd:>+6.3f} {inc:>+10.4f} {verdict:>8s}")
    print("\nincrement = conv net - base net (forward, honest CDE fee); "
          "positive = conviction sizing is earning forward")
    return 2 if any_thin else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cost-bps-btc", type=float, default=COST["BTC"])
    ap.add_argument("--cost-bps-eth", type=float, default=COST["ETH"])
    args = ap.parse_args()
    return review(args.data_dir,
                  {"BTC": args.cost_bps_btc, "ETH": args.cost_bps_eth})


if __name__ == "__main__":
    sys.exit(main())

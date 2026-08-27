"""[P419] Stop-loss distance sweep on the live trend book, 6 years, honest RT.

THE OPERATOR'S QUESTION: "is our stop set right? volatility is normal -- if
we sell on every dip we never make money." The live facts first: the 10%
entry-anchored venue stop has NEVER FILLED in the retained live window (5
placements, 0 fires) -- today it is pure process-death insurance, not a
shake-out cost. This lab asks what each stop distance WOULD have done over
6 years so the setting rests on measurement, not intuition.

Chassis: the P369 risk_control_audit_lab's own replay (P172 -- same book,
same costs, same eras), which already models the stop, the 2-tick re-entry
cooldown and flip persistence. Book = long/flat SMA200 trend (the certified
mechanism the stop actually sits on).

PRE-COMMITTED VERDICT RULE (before the first number):
  * A stop distance EARNS ITS PLACE iff its 6y cost vs no-stop is small
    (>= -0.15 net summed across assets) AND it improves the tail (p05 or
    worst-24h) in EVERY era it fires in (P369's own four-part rule).
  * If NO distance earns tail protection, the recommendation is: keep the
    widest cheap distance as process-death INSURANCE (a stop that never
    fires costs nothing and still covers a dead engine), and explicitly do
    NOT tighten.
  * Whipsaw metric reported per distance: fires/yr and cost/yr -- the
    operator's "sell on every dip" fear, quantified.
Caveat: fixed-% distances per asset; per-asset optimal distance IS the
vol-scaling answer at asset granularity (SOL needs a wider stop than BTC for
the same shake-out probability). Hourly resolution (P369's stated caveat).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from training.risk_control_audit_lab import load, run, stats  # noqa: E402

REPORT = REPO / "training" / "reports" / "stop_sweep_p419.json"
GRID = (0.05, 0.08, 0.10, 0.15, 0.20, None)


def main() -> int:
    results: dict = {"grid": [g if g is not None else "none" for g in GRID],
                     "assets": {}}
    summary_delta: dict = {str(g): 0.0 for g in GRID}
    for asset in ("BTC", "ETH", "SOL"):
        df = load(asset)
        ts = df["timestamp"]
        years = len(df) / 24 / 365.25
        base = run(df, asset)                      # NO stop -- the reference
        b = stats(base["pnl"], ts, years)
        row: dict = {"years": round(years, 2), "no_stop": b, "stops": {}}
        for g in GRID:
            if g is None:
                continue
            r = run(df, asset, stop_pct=g)
            s = stats(r["pnl"], ts, years)
            fires = r["fires"]["stop"]
            row["stops"][str(g)] = {
                "total_pct": s["total_pct"],
                "delta_vs_nostop_pct": round(s["total_pct"] - b["total_pct"], 1),
                "fires_per_yr": round(fires / years, 1),
                "p05_24h_bps": s["p05_24h_bps"],
                "p05_delta_bps": round(s["p05_24h_bps"] - b["p05_24h_bps"], 1),
                "worst_24h_bps": s["worst_24h_bps"],
                "worst_delta_bps": round(
                    s["worst_24h_bps"] - b["worst_24h_bps"], 1),
                "eras_delta": {e: round(s["eras"][e] - b["eras"][e], 1)
                               for e in s["eras"]},
            }
            summary_delta[str(g)] += s["total_pct"] - b["total_pct"]
        results["assets"][asset] = row
        print(f"{asset}: no_stop={b['total_pct']:+.1f}% "
              f"worst24h={b['worst_24h_bps']:+.0f}bps")
        for g in GRID:
            if g is None:
                continue
            st = row["stops"][str(g)]
            print(f"  stop {g:.0%}: net {st['delta_vs_nostop_pct']:+.1f}pp "
                  f"fires/yr {st['fires_per_yr']:.1f} "
                  f"worst24h {st['worst_delta_bps']:+.0f}bps "
                  f"eras {st['eras_delta']}")
    results["summary_net_delta_3asset"] = {
        k: round(v, 1) for k, v in summary_delta.items() if k != "None"}
    REPORT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["summary_net_delta_3asset"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

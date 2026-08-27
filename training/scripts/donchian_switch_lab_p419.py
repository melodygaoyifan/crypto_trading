"""[P419] The Donchian trend-leg switch, judged and calibrated on the chassis.

OPERATOR INSTRUCTION: "调整一下我们的K线" -- after the external net-of-cost
survey graded Donchian/channel breakout as the ONE K-line family with multiple
independent net-of-cost passes, converging with the internal P288 lab
(donchian beat SMA200 on ETH/SOL at ~1/4 the turnover, transfer-validated on
5 never-fitted assets; BTC: SMA200 stands).

WHAT THIS LAB DOES (all through the deployed chassis, P172 -- the exact
funding_legs_lab build/pnl and the calibrator's round-trip convention, so
nothing here can drift from what the seat actually asserts):
  1. Re-derives the P288 verdict per asset per era at honest per-leg CDE
     costs + funding carry: donchian long/flat vs the SMA200 trend leg.
  2. Computes the DONCHIAN book's per-round-trip era edges through the SAME
     three-clause convention as the shipped seat_alpha table (P326) -- the
     number the alpha gate must assert if the leg switches (P320: a seat's
     asserted edge is the measured edge of the DEPLOYED rule, never an
     inherited calibration from a different rule).

PRE-COMMITTED VERDICT RULE (before the first number):
  * Switch an asset's trend leg to donchian iff donchian NET >= sma NET in
    at least 2 of 3 eras AND over the full window, AND the donchian per-RT
    era-MEDIAN clears the asset's honest friction (~25-29bps RT) with the
    same margin the incumbent clears it.
  * BTC is expected to keep SMA200 (P288); a surprise flip would be a
    finding to record, not to act on.
  * The known trade-off travels with any switch: later channel exits ->
    deeper bear-year drawdowns (P288's crash-dodge caveat) -- the operator
    accepted this direction explicitly ("if we sell on every dip we never
    make money").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import training.funding_legs_lab as lab                     # noqa: E402
from training.seat_alpha_calibration import round_trip_edge_bps  # noqa: E402
from defense.trend_rule_shadow import donchian_labels       # noqa: E402

REPORT = REPO / "training" / "reports" / "donchian_switch_lab_p419.json"


def main() -> int:
    results: dict = {"assets": {}}
    for asset in ("BTC", "ETH", "SOL"):
        closes = lab.load_closes(asset)
        funding = lab.load_funding_daily(asset)
        pos_df = lab.build_positions(asset, closes, funding)

        # donchian labels over the FULL close series, then aligned to the
        # chassis frame (build_positions drops the first MIN_BARS warmup bars)
        don_full = donchian_labels(closes.to_numpy(dtype=float))
        don = pd.Series(don_full, index=closes.index).reindex(pos_df.index)

        sma = pos_df["trend"]

        row: dict = {"eras": {}, "turnover": {}, "rt_edge_bps": {}}
        for name, series in (("sma", sma), ("donchian", don)):
            p = lab.pnl(asset, series, closes, funding)
            row.setdefault("net_total", {})[name] = round(
                float(p["net"].sum()), 3)
            row["turnover"][name] = round(
                float(series.diff().abs().fillna(0.0).sum()), 1)
            for era, (lo, hi) in lab.ERAS.items():
                pe = series.iloc[lo:hi] if hi else series.iloc[lo:]
                dfp = lab.pnl(asset, pe, closes, funding)
                row["eras"].setdefault(era, {})[name] = round(
                    float(dfp["net"].sum()), 3)
                if name == "donchian":
                    row["rt_edge_bps"][era] = round_trip_edge_bps(
                        dfp["gross"], pe.reindex(dfp.index))

        # verdict per the pre-committed rule
        d_full = row["net_total"]["donchian"]
        s_full = row["net_total"]["sma"]
        era_wins = sum(
            1 for e in lab.ERAS
            if row["eras"][e]["donchian"] >= row["eras"][e]["sma"])
        med_src = sorted(v for v in row["rt_edge_bps"].values()
                         if v is not None)
        med = (None if not med_src else
               med_src[len(med_src) // 2] if len(med_src) % 2 else
               (med_src[len(med_src) // 2 - 1] + med_src[len(med_src) // 2]) / 2)
        row["rt_edge_median"] = None if med is None else round(med, 1)
        friction = {"BTC": 27.7, "ETH": 29.0, "SOL": 29.0}[asset]
        row["verdict"] = (
            "SWITCH" if (d_full >= s_full and era_wins >= 2
                         and med is not None and med > friction * 1.5)
            else "SMA_STANDS")
        results["assets"][asset] = row
        print(f"{asset}: sma={s_full:+.3f} don={d_full:+.3f} "
              f"era_wins={era_wins}/3 rt_median="
              f"{row['rt_edge_median']} -> {row['verdict']}")
        for e in lab.ERAS:
            print(f"    {e}: sma {row['eras'][e]['sma']:+.3f} "
                  f"don {row['eras'][e]['donchian']:+.3f} "
                  f"rt {row['rt_edge_bps'][e]}")

    REPORT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

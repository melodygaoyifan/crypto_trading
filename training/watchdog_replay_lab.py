"""[P369] Does the inter-tick emergency exit ADD or DESTROY value? Six years, replayed.

THE QUESTION. FastRiskTick fires EXIT_ONLY when the price is >= 3% from the 4H
anchor, flattening the sleeve. Measured live 2026-08-20..22: 33 completed
flattens in ~2 days, against books designed for ~6 (ETH) and ~13 (SOL) round
trips PER YEAR. The trigger is direction-blind — `abs(px - anchor)/anchor` — so
a +8% rally fires the emergency exit exactly as hard as a -8% crash, and the
books were LONG (ETH 22/22, SOL 23/23 over the retained window).

WHY THIS DOES NOT NEED A FORWARD SHADOW. P367 shipped the velocity replacement
shadow-first because P366's evidence was a single 24h sample. But the deciding
question is not the firing RATE (already measured: ~650x) — it is whether
flattening on drift is a good trade. That is an event study on price history,
and six years of hourly bars for all three assets are already on disk.

PRE-COMMITTED VERDICT RULE (written before the first run):
  The drift trigger EARNS its place iff, pooled over flatten events while long,
  the mean forward return is NEGATIVE by MORE than the round-trip cost of
  flattening and re-entering — i.e. it avoided a loss larger than it cost.
  Direction-blindness is COSTLY iff the favourable-side (rally) subset shows a
  POSITIVE mean forward return: those are winners being cut.
  Neither leg is judged on a t-stat: at this event count a |t|>=2 bar rejects
  almost everything (P297), so the discriminators are SIGN, MAGNITUDE vs cost,
  and STABILITY across the three assets and across eras.

HONEST RESOLUTION CAVEAT, stated before the numbers. The live loop evaluates
every ~34s; this replays hourly. Consequences, in opposite directions:
  * DRIFT is well approximated. Drift is the cumulative move from a 4H anchor,
    so hourly sampling changes mainly WHEN inside the bar the threshold is
    first crossed, not WHETHER. Firing counts here are a LOWER bound.
  * VELOCITY is OVERSTATED. A 1-hour step is far larger than a 34s step, so
    the velocity arm fires much more here than it would live. That biases the
    comparison IN FAVOUR of drift — so if velocity still looks better, the
    result is conservative.

Costs are the measured CDE round trip (P315/P334): fee is a PERCENTAGE of
notional, so contract size is irrelevant to bps.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RAW = REPO / "training" / "training_data" / "raw"
ASSETS = ("BTC", "ETH", "SOL")

# [P315/P334] measured CDE round-trip cost in bps: 2 legs x (fee + spread + latency)
COST_RT_BPS = {"BTC": 27.7, "ETH": 44.0, "SOL": 41.0}
THRESHOLD = 0.03          # FastRiskTick PRICE_MOVE_THRESHOLD
FWD_HOURS = (4, 12, 24)   # forward horizons to score


def load(asset: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW / f"{asset}_60m.parquet")
    df = df[["timestamp", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_anchor(df: pd.DataFrame) -> pd.DataFrame:
    """The 4H anchor: the close at the most recent 4H boundary AT OR BEFORE
    this bar. Causal by construction — never reads a future bar."""
    h = df["timestamp"].dt.hour
    is_boundary = (h % 4) == 0
    anchor = df["close"].where(is_boundary)
    df["anchor"] = anchor.ffill()
    df["anchor_age_h"] = df.groupby(is_boundary.cumsum()).cumcount()
    return df.dropna(subset=["anchor"]).reset_index(drop=True)


def replay(asset: str) -> dict:
    df = add_anchor(load(asset))
    px = df["close"].to_numpy(float)
    anc = df["anchor"].to_numpy(float)

    drift = (px - anc) / anc                      # SIGNED
    vel = np.concatenate([[0.0], np.diff(px) / px[:-1]])

    out = {"asset": asset, "bars": int(len(df)),
           "cost_rt_bps": COST_RT_BPS[asset], "horizons": {}}

    for arm, sig in (("drift", drift), ("velocity", vel)):
        fires = np.abs(sig) >= THRESHOLD
        # the live control only acts when a position exists; the books are
        # overwhelmingly LONG (ETH 22/22, SOL 23/23), so score the long case.
        arm_out = {"fire_rate_pct": round(100.0 * fires.mean(), 3),
                   "n_fires": int(fires.sum())}
        for H in FWD_HOURS:
            fwd = np.full(len(px), np.nan)
            fwd[:-H] = (px[H:] - px[:-H]) / px[:-H]
            m = fires & ~np.isnan(fwd)
            if m.sum() == 0:
                continue
            f_all = fwd[m] * 1e4                     # bps, long-position PnL
            adverse = m & (sig < 0)                  # fired on a DOWN move
            favour = m & (sig > 0)                   # fired on an UP move (rally)
            arm_out[f"h{H}"] = {
                "n": int(m.sum()),
                # value of flattening = -(forward return) - cost
                "mean_fwd_bps": round(float(f_all.mean()), 1),
                "flatten_value_bps": round(float(-f_all.mean() - COST_RT_BPS[asset]), 1),
                "n_adverse": int(adverse.sum()),
                "adverse_fwd_bps": (round(float(fwd[adverse].mean() * 1e4), 1)
                                    if adverse.sum() else None),
                "n_favourable": int(favour.sum()),
                "favourable_fwd_bps": (round(float(fwd[favour].mean() * 1e4), 1)
                                       if favour.sum() else None),
            }
        out["horizons"][arm] = arm_out

    # ---- THE FAIR TEST OF A SAFETY CONTROL -----------------------------
    # A stop is not judged on the mean; it is judged on whether it truncates
    # the LEFT TAIL. Compare the forward-return distribution after an adverse
    # drift fire against the unconditional distribution: if the control fires
    # ahead of the genuinely bad outcomes, the conditional p05/min should be
    # materially WORSE than unconditional (i.e. it is catching real crashes).
    H = 24
    fwd = np.full(len(px), np.nan)
    fwd[:-H] = (px[H:] - px[:-H]) / px[:-H]
    ok = ~np.isnan(fwd)
    adverse = (drift <= -THRESHOLD) & ok
    out["tail"] = {
        "horizon_h": H,
        "uncond_p05_bps": round(float(np.percentile(fwd[ok], 5) * 1e4), 1),
        "uncond_min_bps": round(float(fwd[ok].min() * 1e4), 1),
        "adverse_p05_bps": round(float(np.percentile(fwd[adverse], 5) * 1e4), 1)
        if adverse.sum() else None,
        "adverse_min_bps": round(float(fwd[adverse].min() * 1e4), 1)
        if adverse.sum() else None,
        "n_adverse": int(adverse.sum()),
    }

    # ---- ERA STABILITY (P243/P244: era-fragility is disqualifying) ------
    ts = df["timestamp"]
    eras = {"2020-22": ts < "2023-01-01",
            "2023-24": (ts >= "2023-01-01") & (ts < "2025-01-01"),
            "2025-26": ts >= "2025-01-01"}
    H4 = 4
    f4 = np.full(len(px), np.nan)
    f4[:-H4] = (px[H4:] - px[:-H4]) / px[:-H4]
    fires = (np.abs(drift) >= THRESHOLD) & ~np.isnan(f4)
    out["eras"] = {}
    for name, mask in eras.items():
        m = fires & mask.to_numpy()
        if m.sum() < 20:
            continue
        out["eras"][name] = {
            "n": int(m.sum()),
            "flatten_value_bps": round(
                float(-f4[m].mean() * 1e4 - COST_RT_BPS[asset]), 1),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training/reports/watchdog_replay_p369.json")
    a = ap.parse_args()

    results = [replay(x) for x in ASSETS]

    print("=" * 78)
    print("  WATCHDOG REPLAY — does flattening on a 3% move add value?")
    print(f"  6y hourly, long-position scoring, cost = measured CDE round trip")
    print("=" * 78)
    for r in results:
        print(f"\n{r['asset']}  ({r['bars']:,} bars, RT cost {r['cost_rt_bps']}bps)")
        for arm in ("drift", "velocity"):
            d = r["horizons"][arm]
            print(f"  {arm:9s} fires on {d['fire_rate_pct']:6.3f}% of bars "
                  f"(n={d['n_fires']:,})")
            for H in FWD_HOURS:
                k = f"h{H}"
                if k not in d:
                    continue
                e = d[k]
                verdict = "SAVES" if e["flatten_value_bps"] > 0 else "COSTS"
                print(f"    +{H:>2}h  fwd={e['mean_fwd_bps']:+8.1f}bps  "
                      f"flatten={e['flatten_value_bps']:+8.1f}bps [{verdict}]"
                      f"   adverse n={e['n_adverse']:<5} "
                      f"{str(e['adverse_fwd_bps']):>9}   "
                      f"rally n={e['n_favourable']:<5} "
                      f"{str(e['favourable_fwd_bps']):>9}")

    print("\n" + "=" * 78)
    print("  TAIL TEST — does it truncate the left tail? (24h fwd, adverse fires)")
    print("=" * 78)
    for r in results:
        t = r["tail"]
        print(f"  {r['asset']:4s} uncond p05={t['uncond_p05_bps']:>9.1f} "
              f"min={t['uncond_min_bps']:>10.1f}   |   "
              f"after adverse fire p05={t['adverse_p05_bps']:>9.1f} "
              f"min={t['adverse_min_bps']:>10.1f}  (n={t['n_adverse']:,})")

    print("\n" + "=" * 78)
    print("  ERA STABILITY — flatten value at +4h, by era")
    print("=" * 78)
    for r in results:
        cells = "  ".join(f"{k}: {v['flatten_value_bps']:+8.1f} (n={v['n']:,})"
                          for k, v in r["eras"].items())
        print(f"  {r['asset']:4s} {cells}")

    p = REPO / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nreport -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

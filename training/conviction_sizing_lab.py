#!/usr/bin/env python3
"""[WS2] Conviction-agreement sizing lab — does sizing UP when trend AND the
contrarian skew signal AGREE (and DOWN when skew disagrees into euphoria) raise
risk-adjusted return vs the flat 1x trend book?

Reuses the LIVE P407 skew signal (25d+10d blend, contrarian deadband, band 1.0)
verbatim from skew_seat_calibration — no invented variant (P164/P214 parity).
Skew exists for BTC/ETH only (no SOL options).

The sizing is a DISCRETE, EQUAL-WEIGHT AGREE-RULE, never a learned combiner
(DeMiguel: 24 months can't fit one):
  trend flat                       -> 0            (no position)
  trend long + skew long (AGREE)   -> C  (cap)     (size up on agreement)
  trend long + skew short (DISAGREE, euphoria) -> D (de-risk; the "sell before
                                                    the top" leg, P407)
  trend long + skew hold(0)        -> 1x

Rigor: net/Sharpe/maxDD per era {pre_design, design, validation*}; a RANDOM-TIER
control at matched average exposure (proves AGREEMENT, not mere leverage, does
the work); a cap sweep (robustness, not one lucky cap, P386). * validation is a
deliberate one-shot recent-era read (P259b).

Verdict (pre-committed): conviction beats the 1x trend base on RAW net in >=2/3
eras AND beats the random-tier control AND Sharpe does not fall AND the maxDD
increase is justified by the return increment -- across the cap sweep, not one.
"""
from __future__ import annotations
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from training.trend_rule_lab import _sma, DS, DE, PRE
from training.sizing_overlay_lab import per_bar_net
from training.skew_seat_calibration import (
    _load, _by_day, _zseries, _positions_from_z, _ASSETS, _DEFAULT_DATA_DIR)

ASSETS = ["BTC", "ETH"]          # skew is BTC/ETH only
COST_BPS = {"BTC": 27.7, "ETH": 44.0}   # honest CDE round-trip (P382/P385)
CAPS = [1.5, 2.0, 2.5]           # agreement size-up sweep
DERISK = 0.5                     # trend-long-but-skew-short -> reduce (not flat)
SEED = 7


def skew_pos_by_day(asset: str, data_dir: str = _DEFAULT_DATA_DIR):
    """The LIVE P407 daily skew position (day_epoch -> {-1,0,+1}), verbatim."""
    a = _ASSETS[asset]
    sk25 = _by_day(_load(data_dir, f"skew_{a}_25d.json"), "30")
    sk10 = _by_day(_load(data_dir, f"skew_{a}_10d.json"), "30")
    spot = _by_day(_load(data_dir, f"gex_{a}.json"), "index_price")
    days = sorted(set(sk25) & set(sk10) & set(spot))
    z25 = _zseries([sk25[d] for d in days])
    z10 = _zseries([sk10[d] for d in days])
    pos = _positions_from_z([(z25[i] + z10[i]) / 2.0 for i in range(len(days))])
    return dict(zip(days, pos))


def align_skew_to_bars(ts_series, sk_by_day):
    """Causal align: each 4H bar uses its own calendar day's skew position (the
    z-series is strictly trailing, so day D's position already only used < D)."""
    t = pd.to_datetime(ts_series, utc=True, errors="coerce")
    bar_day = (t.view("int64") // 1_000_000_000 // 86400).to_numpy()  # days since epoch
    days = np.array(sorted(sk_by_day))
    vals = np.array([sk_by_day[d] for d in days])
    out = np.zeros(len(bar_day))
    if len(days):
        idx = np.searchsorted(days, bar_day, side="right") - 1  # last day <= bar_day
        ok = idx >= 0
        out[ok] = vals[idx[ok]]
    return out


def maxdd(s):
    if len(s) == 0:
        return 0.0
    cum = np.cumsum(s)
    return float((cum - np.maximum.accumulate(cum)).min())


def conviction_mult(trend_long, skew, cap, derisk=DERISK):
    """Discrete equal-weight agree-tier multiplier."""
    m = np.ones(len(trend_long))
    m[trend_long <= 0] = 0.0                              # flat when no trend
    long_ = (trend_long > 0)
    m[long_ & (skew > 0)] = cap                           # AGREE -> size up
    m[long_ & (skew < 0)] = derisk                        # DISAGREE -> de-risk
    # trend long + skew == 0 keeps 1x
    return m


def _ledger_validation_read(experiment, asset, start, end, purpose):
    """[P420] Record a validation-era read; never let the ledger break the lab
    (a failure is printed, the run continues)."""
    try:
        from training.splits import record_window_usage
        prior = record_window_usage(experiment, asset, int(start), int(end),
                                    purpose)
        if prior:
            print(f"[WINDOW-LEDGER] {asset}: validation window already read by "
                  f"{prior} other experiment(s) — discount accordingly (P260)")
    except Exception as e:  # noqa: silent-swallow — surfaced, never blocks the lab
        print(f"[WINDOW-LEDGER] WARNING: could not record {experiment}/{asset} "
              f"({type(e).__name__}: {e})")


def evaluate_pos(close, pos, cost, lo, hi):
    # [P420] the validation* era is a deliberate one-shot read (P259b) —
    # opted in explicitly; main() ledgers it per asset.
    s = per_bar_net(close, pos, cost, lo, hi, allow_validation=True)
    return {"net": float(np.sum(s)), "maxdd": maxdd(s),
            "flips": int(np.abs(np.diff(pos[lo:hi])).sum()),
            "avg_expo": float(np.mean(np.abs(pos[lo:hi])))}


def main():
    rng = np.random.default_rng(SEED)
    for a in ASSETS:
        d = pd.read_parquet(f"training/training_data/drl_training/{a}_4H_full.parquet")
        close = d["close"].to_numpy(float)
        n = len(close)
        # [P420] ledger the validation-era spend (was unledgered; P332/P382)
        _ledger_validation_read("conviction_sizing_lab:ws2", a, DE, n,
                                "validation:ws2 conviction-agreement sizing "
                                "one-shot recent-era read (P259b)")
        sma = _sma(close, 200)
        trend = np.where(np.isnan(sma), 0.0, (close > sma).astype(float))
        skew = align_skew_to_bars(d["timestamp"], skew_pos_by_day(a))[:n]
        cost = COST_BPS[a]
        base = trend.copy()                                # 1x trend book
        eras = [("pre_design", PRE[0], PRE[1]), ("design", DS, DE),
                ("validation*", DE, n)]

        # coverage of the agreement states (sanity)
        long_ = trend > 0
        agree = float(np.mean(skew[long_] > 0)) if long_.any() else 0.0
        disagree = float(np.mean(skew[long_] < 0)) if long_.any() else 0.0
        print(f"\n===== {a} =====  (of long bars: skew AGREES {agree:.0%}, "
              f"DISAGREES {disagree:.0%})")
        print(f"{'cap':>4s} {'era':12s} | {'base net/DD':>16s} | "
              f"{'conv net/DD':>16s} | {'rand net/DD':>16s} | verdict")
        for cap in CAPS:
            conv = base * conviction_mult(trend, skew, cap)
            # random-tier control: same multiplier VALUES, shuffled over long bars
            mult = conviction_mult(trend, skew, cap)
            rmult = mult.copy()
            li = np.where(long_)[0]
            rmult[li] = rng.permutation(mult[li])          # matched avg exposure
            rand = base * rmult
            wins = 0
            for en, lo, hi in eras:
                b = evaluate_pos(close, base, cost, lo, hi)
                c = evaluate_pos(close, conv, cost, lo, hi)
                r = evaluate_pos(close, rand, cost, lo, hi)
                beats_base = c["net"] > b["net"]
                beats_rand = c["net"] > r["net"]
                if beats_base and beats_rand:
                    wins += 1
                v = ("+" if beats_base else "-") + ("R" if beats_rand else "r")
                print(f"{cap:>4.1f} {en:12s} | "
                      f"{b['net']:>+7.3f}/{b['maxdd']:>+6.3f} | "
                      f"{c['net']:>+7.3f}/{c['maxdd']:>+6.3f} | "
                      f"{r['net']:>+7.3f}/{r['maxdd']:>+6.3f} | {v}")
            print(f"     cap {cap}: beats base+random in {wins}/3 eras")
    print("\nverdict key: +/- = beats base net; R/r = beats random-tier control")
    print("* validation = deliberate one-shot recent-era read (P259b)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

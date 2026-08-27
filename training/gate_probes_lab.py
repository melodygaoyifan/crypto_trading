#!/usr/bin/env python3
"""[A/B probes] Cheap Rung-0 tests of the not-yet-done levers, reusing the WS2
conviction infrastructure. Each asks a falsifiable question with honest CDE
fees, per era {pre_design, design, validation*}, and (where meaningful) a
control. The honest prior (research 2026-08-26): the direction-seat search is
EXHAUSTED, so A1/A3/A4 are expected mostly noise — they earn only as GATES that
improve the WS2 book's drawdown/Sharpe, never as new alpha.

  A1  skew TERM-STRUCTURE (short vs long tenor) as a de-risk gate on WS2
  A3  dealer GAMMA (GEX) sign as a size gate on WS2
  A4  skew-MOMENTUM vs skew-LEVEL as the conviction signal
  B2  WS2 for SOL: trend + REGIME agreement (SOL has no options skew)
  B4  a WIDE crash-stop on the WS2 book (drawdown enabler)

* validation = deliberate one-shot recent-era read (P259b).
"""
from __future__ import annotations
import sys
sys.path.insert(0, ".")
import json
import numpy as np
import pandas as pd

from training.trend_rule_lab import _sma, DS, DE, PRE
from training.sizing_overlay_lab import per_bar_net
from training.conviction_sizing_lab import (
    skew_pos_by_day, align_skew_to_bars, conviction_mult, maxdd, COST_BPS)
from training.skew_seat_calibration import _load, _by_day, _zseries, _ASSETS

ERAS = [("pre_design", PRE[0], PRE[1]), ("design", DS, DE)]  # validation added per-call
DATA = "training/training_data/laevitas_skew"


def _close(a):
    d = pd.read_parquet(f"training/training_data/drl_training/{a}_4H_full.parquet")
    return d["close"].to_numpy(float), d["timestamp"], d


def _ev(close, pos, cost, lo, hi):
    s = per_bar_net(close, pos, cost, lo, hi)
    return float(np.sum(s)), maxdd(s)


def _trend(close):
    sma = _sma(close, 200)
    return np.where(np.isnan(sma), 0.0, (close > sma).astype(float))


def _skew(a, ts, n):
    return align_skew_to_bars(ts, skew_pos_by_day(a))[:n]


def _daily_field_to_bars(rows_by_day, ts, n):
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    bar_day = (t.view("int64") // 1_000_000_000 // 86400).to_numpy()
    days = np.array(sorted(rows_by_day))
    vals = np.array([rows_by_day[d] for d in days])
    out = np.full(n, np.nan)
    if len(days):
        idx = np.searchsorted(days, bar_day, side="right") - 1
        ok = idx >= 0
        out[ok] = vals[idx[ok]]
    return out


def _report(title, close, base, variant, cost, n, note=""):
    eras = ERAS + [("validation*", DE, n)]
    print(f"\n[{title}] {note}")
    print(f"  {'era':12s} {'base net/DD':>16s} {'variant net/DD':>16s} {'Δnet':>8s}")
    wins = 0
    for en, lo, hi in eras:
        if hi <= lo:
            continue
        bn, bd = _ev(close, base, cost, lo, hi)
        vn, vd = _ev(close, variant, cost, lo, hi)
        dd_ok = vd >= bd - 1e-9      # variant must not deepen drawdown
        if vn > bn:
            wins += 1
        print(f"  {en:12s} {bn:>+8.3f}/{bd:>+6.3f} {vn:>+8.3f}/{vd:>+6.3f} "
              f"{vn - bn:>+8.3f}{'' if dd_ok else ' (DD deeper)'}")
    print(f"  -> variant beats base net in {wins}/3 eras")
    return wins


# ---- A1: skew term-structure slope as a de-risk gate on WS2 ----
def probe_A1():
    for a in ("BTC", "ETH"):
        close, ts, _ = _close(a)
        n = len(close)
        trend = _trend(close)
        skew = _skew(a, ts, n)
        base = trend * conviction_mult(trend, skew, 2.0)
        # skew term slope = short-tenor(30) minus long-tenor(180); align to bars
        sk30 = _by_day(_load(DATA, f"skew_{_ASSETS[a]}_25d.json"), "30")
        sk180 = _by_day(_load(DATA, f"skew_{_ASSETS[a]}_25d.json"), "180")
        days = sorted(set(sk30) & set(sk180))
        slope = {d: sk30[d] - sk180[d] for d in days}
        slope_b = _daily_field_to_bars(slope, ts, n)
        z = np.array(_zseries(list(np.nan_to_num(slope_b))))
        # GATE: when the slope z is extreme (front skew unusually rich vs back =
        # near-term fear spike) cap conviction at 1x (de-risk). Rung-0.
        gate = base.copy()
        gate[(base > 1.0) & (z > 1.0)] = 1.0
        _report(f"A1 {a} skew-term de-risk gate", close, base, gate, COST_BPS[a], n,
                "does gating WS2 on front-vs-back skew help?")


# ---- A3: dealer gamma (GEX) sign as a size gate ----
def probe_A3():
    for a in ("BTC", "ETH"):
        close, ts, _ = _close(a)
        n = len(close)
        trend = _trend(close)
        skew = _skew(a, ts, n)
        base = trend * conviction_mult(trend, skew, 2.0)
        gex = _by_day(_load(DATA, f"gex_{_ASSETS[a]}.json"), "gex")
        gexb = _daily_field_to_bars(gex, ts, n)
        # GATE: negative dealer gamma = amplifying/unstable -> cap conviction 1x.
        gate = base.copy()
        gate[(base > 1.0) & (gexb < 0)] = 1.0
        _report(f"A3 {a} GEX de-risk gate", close, base, gate, COST_BPS[a], n,
                "cap size-up when dealer gamma is negative (unstable)?")


# ---- A4: skew MOMENTUM vs skew LEVEL as the conviction signal ----
def probe_A4():
    for a in ("BTC", "ETH"):
        close, ts, _ = _close(a)
        n = len(close)
        trend = _trend(close)
        skew_level = _skew(a, ts, n)  # the live level-contrarian signal (base)
        base = trend * conviction_mult(trend, skew_level, 2.0)
        # skew MOMENTUM: sign of the change in raw 30d skew (rising fear -> long)
        sk30 = _by_day(_load(DATA, f"skew_{_ASSETS[a]}_25d.json"), "30")
        days = sorted(sk30)
        raw = _daily_field_to_bars({d: sk30[d] for d in days}, ts, n)
        dmom = np.zeros(n)
        dmom[1:] = np.sign(np.diff(np.nan_to_num(raw))) * -1.0  # falling skew(more fear)->long
        mom_pos = trend * conviction_mult(trend, dmom, 2.0)
        _report(f"A4 {a} skew-MOMENTUM vs LEVEL", close, base, mom_pos, COST_BPS[a], n,
                "does momentum-of-skew size better than level-of-skew?")


# ---- B2: WS2 for SOL via trend + REGIME agreement ----
def probe_B2():
    close, ts, d = _close("SOL")
    n = len(close)
    trend = _trend(close)
    base = trend.copy()  # 1x trend book (SOL has no skew)
    regime = d["regime"].to_numpy()
    # bullish GMM regimes for SOL: read from the deployed vocabulary if present
    try:
        names = json.loads(open("models/regime_classifier/SOL/gmm_config.json",
                                 encoding="utf-8").read()).get("regime_names") or []
    except Exception:
        names = []
    bull_ids = {i for i, nm in enumerate(names)
                if any(k in str(nm).upper() for k in ("UPTREND", "RALLY", "BULL",
                                                       "MOMENTUM", "ACCUMULATION"))}
    reg_long = np.array([1.0 if int(r) in bull_ids else 0.0 for r in regime])
    conv = np.ones(n)
    conv[trend <= 0] = 0.0
    conv[(trend > 0) & (reg_long > 0)] = 2.0   # trend AND bullish regime -> size up
    print(f"\n[B2 SOL trend+regime] bull regime ids={sorted(bull_ids)} "
          f"names={[names[i] for i in sorted(bull_ids)] if names else '(none)'}")
    _report("B2 SOL trend+regime conviction", close, base, conv, COST_BPS.get("SOL", 41.0), n,
            "size up when trend long AND GMM regime bullish?")


# ---- B4: a WIDE crash-stop on the WS2 book ----
def probe_B4():
    for a in ("BTC", "ETH"):
        close, ts, _ = _close(a)
        n = len(close)
        trend = _trend(close)
        skew = _skew(a, ts, n)
        base = trend * conviction_mult(trend, skew, 2.0)
        # wide stop: if price falls > STOP_PCT below the trailing 20-bar high while
        # in a position, go flat until the trend re-enters (a crash circuit-breaker).
        for STOP in (0.15, 0.20):
            hi20 = pd.Series(close).rolling(20).max().shift(1).to_numpy()
            stopped = base.copy()
            active = True
            for i in range(n):
                if not np.isnan(hi20[i]) and close[i] < hi20[i] * (1 - STOP):
                    active = False
                if base[i] == 0.0:      # trend flat re-arms the stop
                    active = True
                if not active:
                    stopped[i] = 0.0
            _report(f"B4 {a} wide crash-stop {int(STOP*100)}%", close, base, stopped,
                    COST_BPS[a], n, "does a wide stop cut drawdown without gutting return?")


def main():
    print("=" * 70)
    print("A/B GATE PROBES — Rung-0, honest CDE fees, per era (validation = 1-shot)")
    print("=" * 70)
    probe_A1(); probe_A3(); probe_A4(); probe_B2(); probe_B4()
    print("\nNOTE: a GATE earns only if it cuts drawdown/raises Sharpe without "
          "gutting return; a variant that just lowers return is dead (expected "
          "for most — the direction search is exhausted).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

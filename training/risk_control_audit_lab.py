"""[P369] End-to-end audit of every risk threshold that binds the live sleeve,
each replayed over six years of hourly data.

THE RULE THIS RUNS UNDER. Every verdict criterion is written HERE, before the
first number is read. A threshold chosen after seeing the result is selection,
not evidence (P340's window-shopping lesson; P297's pre-committed verdicts).

WHAT "MAKES SENSE" MEANS, stated per control. A risk control is NOT judged on
its mean PnL alone — a stop that costs a little and truncates the left tail
is doing its job. So each control is scored on FOUR things:
  1. COST    — mean PnL delta vs the uncontrolled book (bps/yr of notional)
  2. TAIL    — does it improve the 5th-percentile / worst 24h outcome?
  3. RATE    — how often it fires per year (an "emergency" control that fires
               weekly is a volatility meter with a liquidation button)
  4. STABLE  — is the verdict the same sign in every era (2020-22, 23-24, 25-26)?
               Era-fragility is disqualifying (P243/P244).
A control EARNS ITS PLACE iff it buys tail protection at a cost the book can
carry, fires rarely enough to be an emergency, and does so in every era.

CONTROLS AUDITED (the ones that actually bind the Coinbase sleeve; the Kraken
exit stack is dormant post-Phase-B and is NOT scored — P275):
  A. FastRiskTick price-move EXIT_ONLY (3% drift from 4H anchor)   [P366/P367]
  B. FastRiskTick vol-spike REDUCE_50 (2x 4H-anchor vol)
  C. Venue-resting protective stop (10% from ENTRY)                 [P197]
  D. Sleeve drawdown halt (15% of invested basis, sticky)           [P150/P294]
  E. Flip persistence (2 consecutive 4H ticks before a sign flip)   [P198]
  F. Re-entry cooldown (2 ticks after a flatten)                    [P232]

NOT audited here, and why: the alpha gate (it is a COST model, not a risk
control, and P318/P320/P334 already certified its two sides); the existence
fuse (28-day window needs years of independent samples to evaluate — P209's
own caveat); the depth-drop trigger (needs order-book history; none on disk).

BOOK UNDER CONTROL. The certified regimebook positions are not available
hourly, so the book is the simplest thing the sleeve actually holds: a
long/flat trend book (+1 when close > SMA200, else flat) — the P262-certified
mechanism and the one every control sits on top of in practice (ETH 22/22
long, SOL 23/23 long over the retained live window).

RESOLUTION CAVEAT. Hourly bars; the live loop is ~34s. Drift-style triggers
are well approximated (drift accumulates); velocity-style triggers are
OVERSTATED (an hourly step is far larger than a 34s step). Stated once, here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RAW = REPO / "training" / "training_data" / "raw"
ASSETS = ("BTC", "ETH", "SOL")
COST_RT = {"BTC": 27.7e-4, "ETH": 44.0e-4, "SOL": 41.0e-4}   # measured CDE RT, fraction
ERAS = {"2020-22": ("2020-01-01", "2023-01-01"),
        "2023-24": ("2023-01-01", "2025-01-01"),
        "2025-26": ("2025-01-01", "2027-01-01")}


# ---------------------------------------------------------------- data ----
def load(asset: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW / f"{asset}_60m.parquet")[["timestamp", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    h = df["timestamp"].dt.hour
    df["is4h"] = (h % 4) == 0
    df["anchor"] = df["close"].where(df["is4h"]).ffill()
    df["sma200"] = df["close"].rolling(200 * 4).mean()      # 200 4H bars
    # 4H-anchored realized vol proxy: std of hourly returns over the prior 4H bar
    r = df["close"].pct_change()
    df["vol_4h"] = r.rolling(4).std().where(df["is4h"]).ffill()
    df["vol_now"] = r.rolling(4).std()
    return df.dropna().reset_index(drop=True)


# ---------------------------------------------------------------- engine --
def run(df: pd.DataFrame, asset: str, *, stop_pct=None, drift_exit=None,
        vol_mult=None, dd_halt=None, flip_persist=0, reentry_cool=0,
        cooldown_sec=3600) -> dict:
    """One pass of the book with a chosen set of controls armed.

    Returns per-bar PnL series (for tail stats) plus event counts.
    Signal: long iff close > SMA200 (the certified trend mechanism).
    All controls act ONLY on a held long; the sign is never reversed.
    """
    px = df["close"].to_numpy(float)
    anc = df["anchor"].to_numpy(float)
    sma = df["sma200"].to_numpy(float)
    is4h = df["is4h"].to_numpy(bool)
    v4 = df["vol_4h"].to_numpy(float)
    vnow = df["vol_now"].to_numpy(float)
    cost = COST_RT[asset]
    n = len(px)
    ret = np.concatenate([[0.0], np.diff(px) / px[:-1]])

    pnl = np.zeros(n)
    held = False
    entry = np.nan
    size = 1.0                       # 1.0 = full, 0.5 after a REDUCE_50
    cd_ticks = 0                     # re-entry cooldown, in 4H ticks
    want_sign_streak = 0             # flip-persistence counter
    last_reduce_t = -1e18
    equity = 1.0; peak_basis = 1.0; halted = False
    n_fires = {"stop": 0, "drift": 0, "vol": 0, "halt": 0, "flip_deferred": 0}

    for i in range(1, n):
        if held:
            pnl[i] = ret[i] * size
            equity *= (1.0 + ret[i] * size)
        # ---- intra-tick controls (every bar) ----
        if held and stop_pct is not None and px[i] <= entry * (1 - stop_pct):
            held = False; pnl[i] -= cost; n_fires["stop"] += 1
            cd_ticks = reentry_cool
        if held and drift_exit is not None and abs(px[i] - anc[i]) / anc[i] >= drift_exit:
            held = False; pnl[i] -= cost; n_fires["drift"] += 1
            cd_ticks = reentry_cool
        if (held and vol_mult is not None and size > 0.5 and v4[i] > 0
                and vnow[i] > vol_mult * v4[i] and (i - last_reduce_t) * 3600 >= cooldown_sec):
            size = 0.5; pnl[i] -= cost / 2; n_fires["vol"] += 1; last_reduce_t = i
        # Peak-anchored (a high-water mark that RATCHETS), the semantics of a
        # drawdown HALT. The first cut compared against a fixed start basis
        # of 1.0, which is an inception anchor — it never ratcheted, so on a
        # book that had already gained, a 45-84% peak-to-trough drawdown
        # read as 0 fires. Caught by the sanity check, not by reading.
        peak_basis = max(peak_basis, equity)
        if dd_halt is not None and not halted and equity < peak_basis * (1 - dd_halt):
            halted = True; n_fires["halt"] += 1
            if held:
                held = False; pnl[i] -= cost
        # ---- decision tick (4H boundary) ----
        if is4h[i]:
            if cd_ticks > 0:
                cd_ticks -= 1
            want = px[i] > sma[i]
            if halted:
                continue
            if want and not held:
                if cd_ticks == 0:
                    held = True; entry = px[i]; size = 1.0; pnl[i] -= 0  # entry leg charged at exit as RT
            elif (not want) and held:
                want_sign_streak += 1
                if want_sign_streak >= max(1, flip_persist):
                    held = False; pnl[i] -= cost; want_sign_streak = 0
                else:
                    n_fires["flip_deferred"] += 1
            else:
                want_sign_streak = 0
            if held and size < 1.0 and is4h[i]:
                size = 1.0          # REDUCE_50 is restored at the next decision tick
    return {"pnl": pnl, "fires": n_fires}


def stats(pnl: np.ndarray, ts: pd.Series, years: float) -> dict:
    # 24h forward window PnL for tail stats: rolling 24-bar sums
    s = pd.Series(pnl)
    r24 = s.rolling(24).sum().dropna().to_numpy()
    out = {"total_pct": round(float(pnl.sum() * 100), 1),
           "per_yr_pct": round(float(pnl.sum() * 100 / years), 2),
           "p05_24h_bps": round(float(np.percentile(r24, 5) * 1e4), 1),
           "worst_24h_bps": round(float(r24.min() * 1e4), 1),
           "eras": {}}
    for name, (a, b) in ERAS.items():
        m = ((ts >= a) & (ts < b)).to_numpy()
        if m.sum() > 24 * 30:
            out["eras"][name] = round(float(pnl[m].sum() * 100), 1)
    return out


def audit(asset: str) -> dict:
    df = load(asset)
    ts = df["timestamp"]
    years = len(df) / 24 / 365.25
    base = run(df, asset)
    b = stats(base["pnl"], ts, years)
    res = {"asset": asset, "years": round(years, 2), "baseline": b, "controls": {}}

    def score(name, fires_key, **kw):
        r = run(df, asset, **kw)
        s = stats(r["pnl"], ts, years)
        fires = r["fires"][fires_key]
        era_sign_stable = all(
            (s["eras"][e] - b["eras"][e]) < 0 for e in s["eras"]) or all(
            (s["eras"][e] - b["eras"][e]) >= 0 for e in s["eras"])
        res["controls"][name] = {
            "fires_per_yr": round(fires / years, 1),
            "cost_pct_per_yr": round(s["per_yr_pct"] - b["per_yr_pct"], 2),
            "p05_24h_delta_bps": round(s["p05_24h_bps"] - b["p05_24h_bps"], 1),
            "worst_24h_delta_bps": round(s["worst_24h_bps"] - b["worst_24h_bps"], 1),
            "era_deltas": {e: round(s["eras"][e] - b["eras"][e], 1) for e in s["eras"]},
            "era_sign_stable": era_sign_stable,
            "total_with": s["total_pct"], "total_base": b["total_pct"],
        }

    # A. price-move EXIT_ONLY at the live 3%, and a sweep
    for thr in (0.03, 0.05, 0.07, 0.10):
        score(f"A_drift_exit_{int(thr*100)}pct", "drift", drift_exit=thr, reentry_cool=2)
    # B. vol-spike REDUCE_50
    for m in (2.0, 3.0, 4.0):
        score(f"B_vol_reduce_{m:.0f}x", "vol", vol_mult=m)
    # C. venue protective stop from ENTRY
    for sp in (0.05, 0.10, 0.15, 0.20):
        score(f"C_entry_stop_{int(sp*100)}pct", "stop", stop_pct=sp, reentry_cool=2)
    # D. sleeve drawdown halt (sticky; simulated as permanent from first trip)
    for dd in (0.10, 0.15, 0.25):
        score(f"D_dd_halt_{int(dd*100)}pct", "halt", dd_halt=dd)
    # E. flip persistence
    for fp in (1, 2, 3):
        score(f"E_flip_persist_{fp}", "flip_deferred", flip_persist=fp)
    # F. re-entry cooldown (only meaningful with something that flattens)
    for rc in (0, 2, 4):
        score(f"F_reentry_cool_{rc}_with_10pct_stop", "stop", stop_pct=0.10, reentry_cool=rc)
    return res


def main() -> int:
    results = [audit(a) for a in ASSETS]
    out = REPO / "training" / "reports" / "risk_control_audit_p369.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    W = 96
    print("=" * W)
    print("  RISK CONTROL AUDIT — each control vs the uncontrolled long/flat trend book, 6y hourly")
    print("  cost = %/yr of notional vs baseline   tail = 24h p05 delta (bps, + is better)   stable = same sign all eras")
    print("=" * W)
    for r in results:
        b = r["baseline"]
        print(f"\n{r['asset']}  baseline {b['total_pct']:+.1f}% over {r['years']}y  "
              f"p05_24h={b['p05_24h_bps']:.0f}bps  worst_24h={b['worst_24h_bps']:.0f}bps")
        print(f"  {'control':34s} {'fires/yr':>8s} {'cost %/yr':>10s} {'p05 Δ':>8s} {'worst Δ':>9s} {'stable':>7s}  eras")
        for name, c in r["controls"].items():
            eras = " ".join(f"{v:+.0f}" for v in c["era_deltas"].values())
            print(f"  {name:34s} {c['fires_per_yr']:8.1f} {c['cost_pct_per_yr']:+10.2f} "
                  f"{c['p05_24h_delta_bps']:+8.0f} {c['worst_24h_delta_bps']:+9.0f} "
                  f"{'yes' if c['era_sign_stable'] else 'NO':>7s}  [{eras}]")
    print(f"\nreport -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

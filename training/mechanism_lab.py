"""[P256] Mechanism lab — exploit-mechanics experiments on the P250 books.

Four stages, one chassis, DESIGN ERA ONLY [3000, 9100). No validation-era
bar is ever scored (hard-asserted). The candidates here are MECHANISMS
wrapping the already-selected books, per the 2026-08-10 research verdict
(docs/RESEARCH_PLAN_2026-08-10.md): partial adjustment, jump-model regime
switching, fv2-as-tilt, and the SOL vol filter. Winners become shadow
candidates through the standard P166 path — nothing here promotes anything.

Stages:
  adjust    Asymmetric exit-confirmation / flip-persistence / min-hold sweep
            on the book target series (the Garleanu-Pedersen degeneration at
            +/-1-contract granularity: cheap to hold, expensive to change).
  jump      Online jump-penalized 3-state switch (Shu/Mulvey shape: penalty
            chosen by CV on STRATEGY performance with costs + delay) vs the
            SMA200 switch, driving the same books.
  fv2tilt   Walk-forward ridge on fv2_* cols (16h) gating BOOK ENTRIES only
            (ma_filter semantics: holds/exits never touched). BTC + SOL
            (ETH fv2 measured dead).
  volfilter SOL hold-bull suppressed in the causal bottom vol-tercile
            (P249: -59.8% of SOL PnL concentrates in LOW vol). Reports the
            pre-design era too (era-stability, the P243 probe discipline).

Cost model: COST_BPS[asset]/2 per side per unit |dpos| (the round-trip
number split across legs), 1-bar execution delay everywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
if REPO.name == "training":
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

from training.regime_model_lab import _ctx, DESIGN  # noqa: E402
from training.train_supervised_full import COST_BPS, DATA_DIR  # noqa: E402

DS, DE = DESIGN                     # [3000, 9100)
PRE = (800, 3000)                   # pre-design era (era-stability reads)
REPORT = REPO / "training" / "reports" / "mechanism_lab_p256.json"


# ---------------------------------------------------------------------------
# chassis
# ---------------------------------------------------------------------------

def book_targets(asset: str, lab: np.ndarray, fz: np.ndarray) -> np.ndarray:
    """P250 winners, vectorized. lab: 0=peace 1=bull 2=bear (regime_model_lab
    REGIME_ID). fz: causal funding z per bar (NaN = no history -> flat)."""
    n = len(lab)
    t = np.zeros(n)
    bull, bear, peace = lab == 1, lab == 2, lab == 0
    t[bull] = 1.0
    if asset == "BTC":
        fzz = np.where(np.isnan(fz), 0.0, fz)
        has = ~np.isnan(fz)
        t[bear & has & (fzz > 1.0)] = -1.0
        t[peace & has & (fzz > 0.5)] = -1.0
        t[peace & has & (fzz < -0.5)] = 1.0
    # ETH: trend-only (bull hold, else flat). SOL: hold-bull (degraded book).
    return t


def pnl_after_cost(close: np.ndarray, pos: np.ndarray, cost_rt_bps: float,
                   lo: int, hi: int) -> dict:
    """Total after-cost return of holding pos[t] over t->t+1, bars [lo, hi).
    1-bar delay is built in: pos[t] is decided from info at t and earns the
    t->t+1 return. Hard guard: never scores a bar at/after the validation
    era start.

    [P260] Window-edge conventions, DOCUMENTED per the fresh-mind review
    (deliberate, not accidental — changing them would shift every recorded
    lab number by <= one round-trip of cost per window):
      * the window's INITIAL position is established cost-free
        (prepend=pos[lo]) — a position inherited from before the window
        belongs to the prior window's ledger;
      * a position change on the window's LAST bar is charged (the trade
        happens) even though its forward return falls outside the window.
    Net effect ~1 trade of cost per multi-thousand-bar window, direction
    mildly pro-candidate; identical for every candidate and baseline, so
    comparisons are unaffected."""
    assert hi <= DE, "validation-era read attempted — refused"
    r1 = np.zeros_like(close)
    r1[:-1] = close[1:] / close[:-1] - 1.0
    seg = slice(lo, hi - 1)
    gross = float(np.nansum(pos[seg] * r1[seg]))
    dpos = np.abs(np.diff(pos[lo:hi], prepend=pos[lo]))
    cost = float(np.nansum(dpos) * (cost_rt_bps / 2.0) / 1e4)
    return {"gross": round(gross, 4), "cost": round(cost, 4),
            "net": round(gross - cost, 4),
            "turnover_units": round(float(np.nansum(dpos)), 1)}


def apply_adjust(raw: np.ndarray, k_exit: int, k_flip: int,
                 min_hold: int) -> np.ndarray:
    """Asymmetric mechanism: an EXIT (target->0) must persist k_exit bars, a
    FLIP must persist k_flip bars, and a fresh entry holds >= min_hold bars
    (exits by flip still honor k_flip). Entries from flat are instant —
    the P195 asymmetry inverted: entering is the cheap-to-defer leg here,
    but the book's entries are regime-level (rare), so they stay instant."""
    n = len(raw)
    out = np.zeros(n)
    cur, streak, want_prev, held = 0.0, 0, 0.0, 0
    for i in range(n):
        want = raw[i]
        if want == cur:
            streak, want_prev = 0, want
            held += 1
            out[i] = cur
            continue
        # a change is requested
        streak = streak + 1 if want == want_prev else 1
        want_prev = want
        need = k_flip if (want != 0 and cur != 0) else (
            k_exit if want == 0 else 1)
        if cur != 0 and held < min_hold and want == 0:
            out[i] = cur
            continue
        if streak >= need:
            cur, streak, held = want, 0, 0
        out[i] = cur
    return out


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_adjust(assets):
    res = {}
    for a in assets:
        c = _ctx(a)
        raw = book_targets(a, c["lab"], c["fz"])
        base = pnl_after_cost(c["close"], raw, COST_BPS[a], DS, DE)
        rows = []
        for ke in (1, 2, 3):
            for kf in (1, 2):
                for mh in (0, 6):
                    pos = apply_adjust(raw, ke, kf, mh)
                    p = pnl_after_cost(c["close"], pos, COST_BPS[a], DS, DE)
                    rows.append({"k_exit": ke, "k_flip": kf, "min_hold": mh,
                                 **p})
        rows.sort(key=lambda r: -r["net"])
        res[a] = {"base_raw": base, "grid": rows[:6],
                  "verdict": ("MECHANISM EARNS" if rows[0]["net"] > base["net"]
                              and (rows[0]["k_exit"], rows[0]["k_flip"],
                                   rows[0]["min_hold"]) != (1, 1, 0)
                              else "raw book stands")}
        print(f"[adjust] {a}: raw net={base['net']:+.4f} "
              f"best={rows[0]['net']:+.4f} "
              f"(ke={rows[0]['k_exit']},kf={rows[0]['k_flip']},"
              f"mh={rows[0]['min_hold']}) -> {res[a]['verdict']}")
    return res


def _jump_states(close: np.ndarray, lam: float, refit: int = 500,
                 warm: int = 800) -> np.ndarray:
    """Online 3-state jump model: k-means centroids refit on ALL PAST data
    every `refit` bars; states assigned ONLINE with a greedy switch penalty
    (s_t = argmin ||x_t - mu_j||^2 + lam * 1[j != s_{t-1}]) — causal by
    construction. States are mapped to peace/bull/bear (0/1/2) by centroid
    momentum ordering at each refit."""
    from sklearn.cluster import KMeans
    n = len(close)
    mom = np.full(n, np.nan)
    mom[540:] = close[540:] / close[:-540] - 1.0
    sma = pd.Series(close).rolling(200).mean().values
    dist = np.where(sma > 0, close / sma - 1.0, np.nan)
    r1 = np.zeros(n)
    r1[1:] = np.log(close[1:] / close[:-1])
    vol = pd.Series(r1).rolling(20).std().values
    F = np.column_stack([mom, dist, vol])
    lab = np.zeros(n, dtype=int)
    mu = None
    order = None
    norm = None
    prev_raw = -1
    for i in range(n):
        if i >= warm and (i == warm or (i - warm) % refit == 0):
            hist = F[:i]
            ok = ~np.isnan(hist).any(axis=1)
            if ok.sum() > 300:
                _mean = hist[ok].mean(0)
                _std = hist[ok].std(0) + 1e-12
                km = KMeans(n_clusters=3, n_init=5, random_state=42).fit(
                    (hist[ok] - _mean) / _std)
                mu = km.cluster_centers_
                # momentum ordering (standardized col 0): low->bear(2),
                # high->bull(1), middle->peace(0)
                o = np.argsort(mu[:, 0])
                order = {int(o[0]): 2, int(o[1]): 0, int(o[2]): 1}
                norm = (_mean, _std)
                prev_raw = -1   # centroids changed; no switch anchor
        if mu is None or np.isnan(F[i]).any():
            lab[i] = 0
            prev_raw = -1
            continue
        x = (F[i] - norm[0]) / norm[1]
        d = ((mu - x) ** 2).sum(1)
        pen = np.array([0.0 if k == prev_raw else lam for k in range(3)])
        raw_s = int(np.argmin(d + pen))
        prev_raw = raw_s
        lab[i] = order[raw_s]
    return lab


def stage_jump(assets):
    res = {}
    split = DS + int((DE - DS) * 2 / 3)     # select on first 2/3 of design
    for a in assets:
        c = _ctx(a)
        base_pos = book_targets(a, c["lab"], c["fz"])
        base_sel = pnl_after_cost(c["close"], base_pos, COST_BPS[a], DS, split)
        base_hold = pnl_after_cost(c["close"], base_pos, COST_BPS[a], split, DE)
        rows = []
        for lam in (0.0, 5.0, 20.0, 50.0, 100.0):
            jl = _jump_states(c["close"], lam)
            pos = book_targets(a, jl, c["fz"])
            sel = pnl_after_cost(c["close"], pos, COST_BPS[a], DS, split)
            rows.append({"lam": lam, "sel_net": sel["net"], "_lab": jl})
        rows.sort(key=lambda r: -r["sel_net"])
        best = rows[0]
        pos = book_targets(a, best["_lab"], c["fz"])
        hold = pnl_after_cost(c["close"], pos, COST_BPS[a], split, DE)
        res[a] = {
            "sma200_select_net": base_sel["net"],
            "sma200_holdout_net": base_hold["net"],
            "jump_lambda": best["lam"],
            "jump_select_net": best["sel_net"],
            "jump_holdout_net": hold["net"],
            "grid": [{k: r[k] for k in ("lam", "sel_net")} for r in rows],
            "verdict": ("JUMP EARNS (holdout)" if hold["net"] >
                        base_hold["net"] else "SMA200 stands"),
        }
        print(f"[jump] {a}: sma holdout={base_hold['net']:+.4f} "
              f"jump(lam={best['lam']}) holdout={hold['net']:+.4f} "
              f"-> {res[a]['verdict']}")
    return res


def stage_fv2tilt(assets=("BTC", "SOL")):
    from sklearn.linear_model import Ridge
    res = {}
    for a in assets:
        c = _ctx(a)
        df = pd.read_parquet(DATA_DIR / f"{a}_4H_full.parquet")
        fcols = [x for x in df.columns if x.startswith("fv2_")]
        if not fcols:
            res[a] = {"error": "no fv2 columns"}
            continue
        Xf = df[fcols].values.astype(float)
        close = c["close"]
        n = len(close)
        y = np.full(n, np.nan)
        y[:-4] = close[4:] / close[:-4] - 1.0          # 16h target
        pred = np.full(n, np.nan)
        for t0 in range(DS, DE, 500):                   # walk-forward
            tr_end = t0 - 12                            # purge gap
            ok = ~np.isnan(Xf[:tr_end]).any(1) & ~np.isnan(y[:tr_end])
            if ok.sum() < 500:
                continue
            mu, sd = Xf[:tr_end][ok].mean(0), Xf[:tr_end][ok].std(0) + 1e-12
            m = Ridge(alpha=30.0).fit((Xf[:tr_end][ok] - mu) / sd,
                                      y[:tr_end][ok])
            hi = min(t0 + 500, DE)
            seg = ~np.isnan(Xf[t0:hi]).any(1)
            p = np.full(hi - t0, np.nan)
            p[seg] = m.predict((Xf[t0:hi][seg] - mu) / sd)
            pred[t0:hi] = p
        raw = book_targets(a, c["lab"], c["fz"])
        tilt = raw.copy()
        entering = np.zeros(n, dtype=bool)
        entering[1:] = (raw[:-1] == 0) & (raw[1:] != 0)
        oppose = ~np.isnan(pred) & (np.sign(pred) * raw < 0)
        block = entering & oppose
        # blocked entries stay flat until the book target changes again
        cur = 0.0
        for i in range(n):
            if raw[i] == cur:
                tilt[i] = cur
                continue
            if block[i]:
                tilt[i] = 0.0
                cur = 0.0 if raw[i] != 0 else cur
                # remain flat; re-evaluate on next target change
                continue
            cur = raw[i]
            tilt[i] = cur
        base = pnl_after_cost(c["close"], raw, COST_BPS[a], DS, DE)
        t = pnl_after_cost(c["close"], tilt, COST_BPS[a], DS, DE)
        res[a] = {"base": base, "tilted": t,
                  "entries_blocked": int(block[DS:DE].sum()),
                  "verdict": ("TILT EARNS" if t["net"] > base["net"]
                              else "base book stands")}
        print(f"[fv2tilt] {a}: base={base['net']:+.4f} tilt={t['net']:+.4f} "
              f"blocked={res[a]['entries_blocked']} -> {res[a]['verdict']}")
    return res


def stage_volfilter():
    a = "SOL"
    c = _ctx(a)
    close = c["close"]
    n = len(close)
    r1 = np.zeros(n)
    r1[1:] = np.log(close[1:] / close[:-1])
    vol = pd.Series(r1).rolling(20).std().values
    # causal terciles: expanding 33rd percentile of PAST vol only
    low_thr = np.full(n, np.nan)
    v = pd.Series(vol)
    low_thr[400:] = v.expanding(min_periods=300).quantile(0.33).shift(1).values[400:]
    raw = book_targets(a, c["lab"], c["fz"])
    filt = raw.copy()
    lowvol = ~np.isnan(vol) & ~np.isnan(low_thr) & (vol < low_thr)
    filt[lowvol] = 0.0
    out = {}
    for name, (lo, hi) in {"design": (DS, DE), "pre_design": PRE}.items():
        out[name] = {
            "base": pnl_after_cost(close, raw, COST_BPS[a], lo, hi),
            "filtered": pnl_after_cost(close, filt, COST_BPS[a], lo, hi),
        }
        print(f"[volfilter] SOL {name}: base={out[name]['base']['net']:+.4f} "
              f"filtered={out[name]['filtered']['net']:+.4f}")
    earns_design = out["design"]["filtered"]["net"] > out["design"]["base"]["net"]
    earns_pre = out["pre_design"]["filtered"]["net"] > out["pre_design"]["base"]["net"]
    out["verdict"] = ("FILTER EARNS BOTH ERAS" if earns_design and earns_pre
                      else "ERA-FRAGILE" if earns_design else "base stands")
    print(f"[volfilter] SOL verdict: {out['verdict']}")
    return {a: out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["adjust", "jump", "fv2tilt", "volfilter", "all"])
    args = ap.parse_args()
    assets = ("BTC", "ETH", "SOL")
    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "design_era": [DS, DE],
              "note": "DESIGN-ERA ONLY. No validation-era bar scored "
                      "(hard-asserted in pnl_after_cost)."}
    if args.stage in ("adjust", "all"):
        report["adjust"] = stage_adjust(assets)
    if args.stage in ("jump", "all"):
        report["jump"] = stage_jump(assets)
    if args.stage in ("fv2tilt", "all"):
        report["fv2tilt"] = stage_fv2tilt()
    if args.stage in ("volfilter", "all"):
        report["volfilter"] = stage_volfilter()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if REPORT.exists():
        try:
            existing = json.loads(REPORT.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(report)
    REPORT.write_text(json.dumps(existing, indent=1, default=str),
                      encoding="utf-8")
    print(f"report: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""[P263] The pooled banded lab — the retrain designed from everything the
campaign learned, examined by every unread window we know exists.

WHAT EACH LESSON CONTRIBUTES (operator: "combine what you learned and
retrain"):
  * parameters-vs-samples (P258): the ridge is fit POOLED across BTC+ETH+SOL
    (~3x effective samples for ~11 parameters), on a VOL-SCALED target (the
    one dev finding that repeatedly won: P242/P243).
  * information ceiling (P258): expression only through the shared banded
    mechanism (defense/regime_book_shadow.banded_step — single source), in
    OVERLAY form (P259: standalone failed BTC's era check; the book keeps
    every cell where it has an opinion).
  * era-conditionality (P259b): shorter refit (250 bars) so the model tracks
    the CURRENT era, and the exam ladder below.
  * live parity (P248/P259): features restricted to the live-computable
    close+funding set from birth (funding_z zeroed for pooling symmetry).
  * the P259b ORDERING lesson, structural this time: the validation read is
    stage FIVE, allowed only after every other unread exam passes; the
    forward ledger only after that.

THE EXAM LADDER (each gate must pass BEFORE the next is looked at):
  1. DESIGN [3000,9100): band-parameter selection (12 combos, home assets).
  2. PRE-DESIGN [800,3000): era gate #1 (overlay >= book on >=2/3 assets).
  3. VIRGIN ERA (BTC/ETH 2017-11 -> 2020-08, P262 cache — data predating
     every decision in this repo): era gate #2.
  4. CROSS-ASSET TRANSFER (BNB/XRP/LTC/DOGE/ADA — DELIBERATELY EXCLUDED
     from pooling so they remain out-of-selection): banded standalone must
     be positive after 10bps RT on >=3/5.
  5. ONE ledgered validation read (only if 1-4 pass; refusing to spend the
     window on a candidate that already failed cheaper exams).
Failures STOP the ladder and are reported — a failed gate is a result.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if REPO.name == "training":
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

from sklearn.linear_model import Ridge  # noqa: E402

from training.regime_model_lab import _ctx, DESIGN  # noqa: E402
from training.train_supervised_full import COST_BPS  # noqa: E402
from training.mechanism_lab import book_targets  # noqa: E402
from training.banded_forecast_lab import close_features  # noqa: E402
from training.unread_era_probe import monthly_klines, causal_labels  # noqa: E402
from defense.regime_book_shadow import banded_step  # noqa: E402
from training.splits import record_window_usage  # noqa: E402

DS, DE = DESIGN
PRE = (800, 3000)
H = 4
REFIT = 250            # short refit: track the CURRENT era (P259b lesson)
GAP = 12
ALPHA = 50.0
XCOST = 10.0
OUT = REPO / "training" / "reports" / "pooled_banded_lab_p263.json"
HOME = ("BTC", "ETH", "SOL")
ALTS = ("BNBUSDT", "XRPUSDT", "LTCUSDT", "DOGEUSDT", "ADAUSDT")


def _series(asset):
    c = _ctx(asset)
    close = c["close"]
    X, names = close_features(close, np.full(len(close), np.nan))  # fz->0
    return close, c["lab"], c["fz"], X


def _vol20(close):
    import pandas as pd
    lr = np.zeros(len(close))
    lr[1:] = np.log(close[1:] / close[:-1])
    return pd.Series(lr).rolling(20).std().values


def pooled_forecast(datasets, n_end_by_asset):
    """Walk-forward pooled ridge on a VOL-SCALED 16h target. At each refit
    boundary t, train on ALL home assets' bars < t-GAP; predict [t, t+REFIT)
    per asset. Returns {asset: pred array} (pred = vol-scaled units)."""
    preds = {a: np.full(len(d["close"]), np.nan) for a, d in datasets.items()}
    max_n = max(n_end_by_asset.values())
    for t0 in range(DS - 500, max_n, REFIT):
        Xtr, ytr = [], []
        for a, d in datasets.items():
            close, X, vol = d["close"], d["X"], d["vol"]
            n = len(close)
            y = np.full(n, np.nan)
            y[:-H] = (close[H:] / close[:-H] - 1.0)
            y = y / (vol * np.sqrt(H) + 1e-9)            # vol-scaled target
            hi_tr = min(t0 - GAP, n)
            ok = (~np.isnan(X[:hi_tr]).any(1) & ~np.isnan(y[:hi_tr])
                  & ~np.isnan(vol[:hi_tr]))
            if ok.sum() > 300:
                Xtr.append(X[:hi_tr][ok])
                ytr.append(y[:hi_tr][ok])
        if not Xtr or sum(len(x) for x in Xtr) < 1500:
            continue
        Xp = np.vstack(Xtr)
        yp = np.concatenate(ytr)
        mu, sd = Xp.mean(0), Xp.std(0) + 1e-12
        m = Ridge(alpha=ALPHA).fit((Xp - mu) / sd, yp)
        for a, d in datasets.items():
            n = n_end_by_asset[a]
            lo, hi = t0, min(t0 + REFIT, n)
            if lo >= hi:
                continue
            X = d["X"]
            seg = ~np.isnan(X[lo:hi]).any(1)
            p = np.full(hi - lo, np.nan)
            p[seg] = m.predict((X[lo:hi][seg] - mu) / sd)
            preds[a][lo:hi] = p
    return preds


def banded_pos(pred, lab, te, tx):
    import pandas as pd
    sig = pd.Series(pred).rolling(500, min_periods=200).std().shift(1).values
    with np.errstate(invalid="ignore", divide="ignore"):
        s = pred / (sig + 1e-12)
    st: dict = {}
    return np.array([banded_step(st, float(x) if x == x else float("nan"),
                                 te, tx, False, int(l))
                     for x, l in zip(s, lab)])


def pnl(close, pos, cost_rt, lo, hi):
    r1 = np.zeros_like(close)
    r1[:-1] = close[1:] / close[:-1] - 1.0
    seg = slice(lo, hi - 1)
    gross = float(np.nansum(pos[seg] * r1[seg]))
    dpos = np.abs(np.diff(pos[lo:hi], prepend=pos[lo]))
    return round(gross - float(np.nansum(dpos) * (cost_rt / 2.0) / 1e4), 4)


def main() -> int:
    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "design": [DS, DE], "refit": REFIT,
              "ladder": ["design-select", "pre-design", "virgin-era",
                         "cross-asset", "validation-read", "forward"]}
    # ---- home datasets (pooled fitting universe) ----
    datasets = {}
    for a in HOME:
        close, lab, fz, X = _series(a)
        datasets[a] = {"close": close, "lab": lab, "fz": fz, "X": X,
                       "vol": _vol20(close)}
    n_end = {a: len(d["close"]) for a, d in datasets.items()}
    preds = pooled_forecast(datasets, n_end)

    # ---- Stage 1: DESIGN selection (overlay vs book, sum across home) ----
    def overlay_for(a, te, tx):
        d = datasets[a]
        posb = banded_pos(preds[a], d["lab"], te, tx)
        book = book_targets(a, d["lab"], d["fz"])
        return np.where(book != 0.0, book, posb), book
    grid = []
    for te in (1.0, 1.5, 2.0):
        for tx in (0.25, 0.5):
            tot, per = 0.0, {}
            for a in HOME:
                ov, bk = overlay_for(a, te, tx)
                x = pnl(datasets[a]["close"], ov, COST_BPS[a], DS, DE)
                b = pnl(datasets[a]["close"], bk, COST_BPS[a], DS, DE)
                per[a] = {"overlay": x, "book": b}
                tot += x - b
            grid.append({"te": te, "tx": tx, "sum_increment": round(tot, 4),
                         "per": per})
    grid.sort(key=lambda r: -r["sum_increment"])
    best = grid[0]
    te, tx = best["te"], best["tx"]
    report["stage1_design"] = {"grid_top3": grid[:3], "selected": [te, tx]}
    print(f"[1 design] selected te={te} tx={tx} "
          f"sum_increment={best['sum_increment']:+.4f}")
    if best["sum_increment"] <= 0:
        report["verdict"] = "DEAD AT STAGE 1 — overlay adds nothing in-design"
        OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(report["verdict"])
        return 0

    # ---- Stage 2: PRE-DESIGN gate ----
    wins = 0
    s2 = {}
    for a in HOME:
        ov, bk = overlay_for(a, te, tx)
        x = pnl(datasets[a]["close"], ov, COST_BPS[a], *PRE)
        b = pnl(datasets[a]["close"], bk, COST_BPS[a], *PRE)
        s2[a] = {"overlay": x, "book": b}
        wins += x >= b
    report["stage2_pre_design"] = {**s2, "wins": f"{wins}/3",
                                   "pass": wins >= 2}
    print(f"[2 pre-design] wins {wins}/3 -> {'PASS' if wins >= 2 else 'FAIL'}")
    if wins < 2:
        report["verdict"] = "DEAD AT STAGE 2 — pre-design era"
        OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(report["verdict"])
        return 0

    # ---- Stage 3: VIRGIN ERA (BTC/ETH 2017-2020, P262 cache) ----
    s3 = {}
    v_wins = 0
    PARQ_MS = int(datetime(2020, 8, 9, tzinfo=timezone.utc).timestamp() * 1000)
    for a, sym in [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")]:
        rows = [r for r in monthly_klines(sym, 2017, 2020)
                if r[0] < PARQ_MS]
        close = np.array([r[1] for r in rows])
        lab = causal_labels(close)
        X, _ = close_features(close, np.full(len(close), np.nan))
        vol = _vol20(close)
        vd = {"v": {"close": close, "lab": lab, "X": X, "vol": vol}}
        # walk-forward INSIDE the virgin era only (self-contained past)
        pv = pooled_forecast(
            {"v": vd["v"]}, {"v": len(close)}) if False else None
        # simpler: refit walk-forward on this series alone (the pooled model
        # cannot be used — it was fit on post-2020 data, which is the FUTURE
        # relative to this era; using it would be a look-ahead)
        pred = np.full(len(close), np.nan)
        y = np.full(len(close), np.nan)
        y[:-H] = (close[H:] / close[:-H] - 1.0) / (vol[:-H] * 2 + 1e-9)
        for t0 in range(800, len(close), REFIT):
            ok = (~np.isnan(X[:t0 - GAP]).any(1) & ~np.isnan(y[:t0 - GAP]))
            if ok.sum() < 300:
                continue
            mu, sd = X[:t0 - GAP][ok].mean(0), X[:t0 - GAP][ok].std(0) + 1e-12
            m = Ridge(alpha=ALPHA).fit((X[:t0 - GAP][ok] - mu) / sd,
                                       y[:t0 - GAP][ok])
            hi = min(t0 + REFIT, len(close))
            seg = ~np.isnan(X[t0:hi]).any(1)
            p = np.full(hi - t0, np.nan)
            p[seg] = m.predict((X[t0:hi][seg] - mu) / sd)
            pred[t0:hi] = p
        posb = banded_pos(pred, lab, te, tx)
        book = np.where(lab == 1, 1.0, 0.0)   # virgin book = trend-only leg
        ov = np.where(book != 0.0, book, posb)
        x = pnl(close, ov, COST_BPS[a], 800, len(close))
        b = pnl(close, book, COST_BPS[a], 800, len(close))
        s3[a] = {"overlay": x, "book": b}
        v_wins += x >= b
    report["stage3_virgin"] = {**s3, "wins": f"{v_wins}/2",
                               "pass": v_wins >= 1}
    print(f"[3 virgin] wins {v_wins}/2 -> {'PASS' if v_wins >= 1 else 'FAIL'}")
    if v_wins < 1:
        report["verdict"] = "DEAD AT STAGE 3 — virgin era"
        OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(report["verdict"])
        return 0

    # ---- Stage 4: CROSS-ASSET TRANSFER (never in pooling) ----
    s4 = {}
    x_wins = 0
    for sym in ALTS:
        rows = monthly_klines(sym, 2020, 2026)
        if len(rows) < 2000:
            s4[sym] = {"error": "insufficient data"}
            continue
        close = np.array([r[1] for r in rows])
        lab = causal_labels(close)
        X, _ = close_features(close, np.full(len(close), np.nan))
        vol = _vol20(close)
        pred = np.full(len(close), np.nan)
        y = np.full(len(close), np.nan)
        y[:-H] = (close[H:] / close[:-H] - 1.0) / (vol[:-H] * 2 + 1e-9)
        for t0 in range(800, len(close), REFIT):
            ok = (~np.isnan(X[:t0 - GAP]).any(1) & ~np.isnan(y[:t0 - GAP]))
            if ok.sum() < 300:
                continue
            mu, sd = X[:t0 - GAP][ok].mean(0), X[:t0 - GAP][ok].std(0) + 1e-12
            m = Ridge(alpha=ALPHA).fit((X[:t0 - GAP][ok] - mu) / sd,
                                       y[:t0 - GAP][ok])
            hi = min(t0 + REFIT, len(close))
            seg = ~np.isnan(X[t0:hi]).any(1)
            p = np.full(hi - t0, np.nan)
            p[seg] = m.predict((X[t0:hi][seg] - mu) / sd)
            pred[t0:hi] = p
        posb = banded_pos(pred, lab, te, tx)
        net = pnl(close, posb, XCOST, 800, len(close))
        s4[sym] = {"banded_net": net, "positive": net > 0}
        x_wins += net > 0
    report["stage4_cross_asset"] = {**s4, "wins": f"{x_wins}/5",
                                    "pass": x_wins >= 3}
    print(f"[4 cross-asset] positive {x_wins}/5 -> "
          f"{'PASS' if x_wins >= 3 else 'FAIL'}")
    if x_wins < 3:
        report["verdict"] = "DEAD AT STAGE 4 — cross-asset transfer"
        OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(report["verdict"])
        return 0

    # ---- Stage 5: THE validation read (ledgered) ----
    s5 = {}
    for a in HOME:
        d = datasets[a]
        n = len(d["close"])
        ov, bk = overlay_for(a, te, tx)
        x = pnl(d["close"], ov, COST_BPS[a], 9100, n)
        b = pnl(d["close"], bk, COST_BPS[a], 9100, n)
        record_window_usage("pooled_banded_p263", a, 9100, n,
                            "validation read #1 for this candidate — reached "
                            "only after pre-design+virgin+cross-asset gates")
        s5[a] = {"overlay": x, "book": b, "increment": round(x - b, 4)}
        print(f"[5 validation] {a}: overlay={x:+.4f} book={b:+.4f} "
              f"increment={x - b:+.4f}")
    report["stage5_validation"] = s5
    inc_pos = sum(1 for a in HOME if s5[a]["increment"] > 0)
    report["verdict"] = (
        f"SURVIVED ALL FIVE GATES on {inc_pos}/3 validation increments"
        if inc_pos >= 2 else
        "DEAD AT STAGE 5 — validation era (the P259b outcome, honestly "
        "reached through the full ladder)")
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(report["verdict"])
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

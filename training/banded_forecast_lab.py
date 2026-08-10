"""[P259] Banded-forecast lab — a model family DESIGNED around the three
measured failure mechanisms (operator spec 2026-08-10):

  1. INFORMATION CEILING (IC 0.03-0.09 vs ~0.13 per-bar break-even): the
     forecast is expressed ONLY through an asymmetric band — enter when the
     normalized forecast exceeds t_enter sigmas, hold until it decays below
     t_exit (t_exit << t_enter). The model pays costs only when its signal
     genuinely clears them; most bars it holds and pays nothing.
  2. PARAMETERS vs SAMPLES: ridge on <=10 features (~11 parameters), high
     alpha, walk-forward refit — nothing to memorize eras with.
  3. REGIME-CONDITIONALITY: an optional gate restricts direction by the
     causal regime label (longs in bull/peace, shorts in bear/peace), so the
     model does not fight the one structure the data measurably has.

  LIVE-PARITY BY CONSTRUCTION: every feature is computable from the
  regime-book harness's OWN inputs (self-fetched 4H closes + persisted daily
  funding history). No parquet-only columns — the constraint whose violation
  killed the SOL bear ridge (P248: runtime-safe subset measured -1.93% vs
  full-feature +5.50%; no honest reduced variant existed). This family is
  reduced from birth.

Honesty protocol: selection on the DESIGN ERA only [3000, 9100); the top
combo is then reported on the PRE-DESIGN era [800, 3000) for era stability
(the P243 probe). NO validation-era bar is scored (hard-asserted) — that
window is ~spent (6+ ledgered reads); survivors go to the FORWARD ledger,
which is the exam that counts (P166).
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
from training.train_supervised_full import COST_BPS  # noqa: E402
from training.mechanism_lab import book_targets, pnl_after_cost  # noqa: E402

DS, DE = DESIGN
PRE = (800, 3000)
H = 4           # 16h forecast horizon (bars)
REFIT = 500
GAP = 12
RIDGE_ALPHA = 50.0
REPORT = REPO / "training" / "reports" / "banded_forecast_lab_p259.json"


def close_features(close: np.ndarray, fz: np.ndarray) -> tuple:
    """The LIVE-COMPUTABLE feature set: closes + daily funding z only.
    Must stay portable to defense/regime_book_shadow verbatim."""
    n = len(close)
    c = pd.Series(close)
    lr = np.log(c / c.shift(1))
    feats = {
        "ret_6": c.pct_change(6),
        "ret_24": c.pct_change(24),
        "ret_72": c.pct_change(72),
        "ret_168": c.pct_change(168),
        "mom_540": c.pct_change(540),
        "dist_sma200": c / c.rolling(200).mean() - 1.0,
        "vol_20": lr.rolling(20).std(),
        "vol_120": lr.rolling(120).std(),
        "vol_ratio": lr.rolling(20).std() / (lr.rolling(120).std() + 1e-12),
        "funding_z": pd.Series(np.where(np.isnan(fz), 0.0, fz)),
    }
    names = list(feats)
    X = np.column_stack([feats[k].values for k in names])
    return X, names


def walk_forward_forecast(X: np.ndarray, close: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import Ridge
    n = len(close)
    y = np.full(n, np.nan)
    y[:-H] = close[H:] / close[:-H] - 1.0
    pred = np.full(n, np.nan)
    for t0 in range(DS - 500, DE, REFIT):   # start early enough to cover DS
        tr_end = t0 - GAP
        ok = ~np.isnan(X[:tr_end]).any(1) & ~np.isnan(y[:tr_end])
        if ok.sum() < 800:
            continue
        mu, sd = X[:tr_end][ok].mean(0), X[:tr_end][ok].std(0) + 1e-12
        m = Ridge(alpha=RIDGE_ALPHA).fit((X[:tr_end][ok] - mu) / sd,
                                         y[:tr_end][ok])
        hi = min(t0 + REFIT, DE)
        seg = ~np.isnan(X[t0:hi]).any(1)
        p = np.full(hi - t0, np.nan)
        p[seg] = m.predict((X[t0:hi][seg] - mu) / sd)
        pred[t0:hi] = p
    # pre-design coverage for the era check: one fit on data before PRE end
    # is impossible causally (no history) — instead walk forward inside PRE
    for t0 in range(PRE[0] + 800, PRE[1], REFIT):
        tr_end = t0 - GAP
        ok = ~np.isnan(X[:tr_end]).any(1) & ~np.isnan(y[:tr_end])
        if ok.sum() < 500:
            continue
        mu, sd = X[:tr_end][ok].mean(0), X[:tr_end][ok].std(0) + 1e-12
        m = Ridge(alpha=RIDGE_ALPHA).fit((X[:tr_end][ok] - mu) / sd,
                                         y[:tr_end][ok])
        hi = min(t0 + REFIT, PRE[1])
        seg = ~np.isnan(X[t0:hi]).any(1)
        p = np.full(hi - t0, np.nan)
        p[seg] = m.predict((X[t0:hi][seg] - mu) / sd)
        pred[t0:hi] = p
    return pred


def banded_positions(pred: np.ndarray, lab: np.ndarray, t_enter: float,
                     t_exit: float, regime_gated: bool) -> np.ndarray:
    """Normalize the forecast by its trailing dispersion, then apply the
    asymmetric band via the SHARED step function (defense/regime_book_shadow
    .banded_step) — one source of truth, so the forward ledger runs the
    byte-identical mechanism the lab selected (P172 one-resolver rule)."""
    from defense.regime_book_shadow import banded_step
    n = len(pred)
    ps = pd.Series(pred)
    sig = ps.rolling(500, min_periods=200).std().shift(1).values
    with np.errstate(invalid="ignore", divide="ignore"):
        s = pred / (sig + 1e-12)
    pos = np.zeros(n)
    state: dict = {}
    for i in range(n):
        pos[i] = banded_step(state, float(s[i]) if s[i] == s[i] else
                             float("nan"), t_enter, t_exit, regime_gated,
                             int(lab[i]))
    return pos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="BTC,ETH,SOL")
    args = ap.parse_args()
    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "design_era": [DS, DE], "pre_design": list(PRE),
              "protocol": "select on design era only; top combo reported on "
                          "pre-design for era stability; NO validation reads "
                          "— survivors go to the forward ledger"}
    for a in args.assets.split(","):
        c = _ctx(a)
        close, lab, fz = c["close"], c["lab"], c["fz"]
        X, names = close_features(close, fz)
        pred = walk_forward_forecast(X, close)
        book = book_targets(a, lab, fz)
        book_design = pnl_after_cost(close, book, COST_BPS[a], DS, DE)
        book_pre = pnl_after_cost(close, book, COST_BPS[a], *PRE)
        rows = []
        for te in (1.0, 1.5, 2.0):
            for tx in (0.25, 0.5):
                for gated in (False, True):
                    pos = banded_positions(pred, lab, te, tx, gated)
                    p = pnl_after_cost(close, pos, COST_BPS[a], DS, DE)
                    rows.append({"t_enter": te, "t_exit": tx,
                                 "gated": gated, **p})
        rows.sort(key=lambda r: -r["net"])
        best = rows[0]
        pos = banded_positions(pred, lab, best["t_enter"], best["t_exit"],
                               best["gated"])
        pre = pnl_after_cost(close, pos, COST_BPS[a], *PRE)
        # overlay variant: book where book is non-flat, banded model where
        # the book is flat (peace/bear cells) — the model only ADDS in cells
        # the book leaves empty, so the book's edge is never diluted
        overlay = np.where(book != 0.0, book, pos)
        ov_design = pnl_after_cost(close, overlay, COST_BPS[a], DS, DE)
        ov_pre = pnl_after_cost(close, overlay, COST_BPS[a], *PRE)
        report[a] = {
            "features": names,
            "book": {"design": book_design, "pre": book_pre},
            "banded_best": {**{k: best[k] for k in
                               ("t_enter", "t_exit", "gated", "net",
                                "turnover_units")},
                            "pre_design": pre},
            "grid_top3": [{k: r[k] for k in ("t_enter", "t_exit", "gated",
                                             "net", "turnover_units")}
                          for r in rows[:3]],
            "overlay": {"design": ov_design, "pre": ov_pre},
            "verdicts": {
                "banded_vs_book_design": best["net"] > book_design["net"],
                "banded_era_stable": pre["net"] > 0,
                "overlay_vs_book_design":
                    ov_design["net"] > book_design["net"],
                "overlay_era_stable": ov_pre["net"] > book_pre["net"],
            },
        }
        print(f"{a}: book design={book_design['net']:+.4f} pre={book_pre['net']:+.4f} | "
              f"banded best={best['net']:+.4f} (te={best['t_enter']},"
              f"tx={best['t_exit']},g={best['gated']}) pre={pre['net']:+.4f} | "
              f"overlay design={ov_design['net']:+.4f} pre={ov_pre['net']:+.4f}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1, default=str),
                      encoding="utf-8")
    print(f"report: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

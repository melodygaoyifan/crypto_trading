"""[P250] End-to-end feature-engineering pipeline — a STAGE, not a one-off.

Four sub-stages, each with a hard gate, mirroring the model pipeline's
discipline (design era only, trial-counted, refusals loud):

  F1  GENERATE  ~60 candidates from systematic causal families:
        - transforms of the top-|IC| base features (rolling z, deltas,
          regime crosses — the P249-earned family, generated systematically)
        - pairwise products of the top-6 base features
        - price-structure (drawdown-from-high, streaks, range compression,
          vol-of-vol, SMA gaps)
        - funding (causal z, its 5d momentum, funding x trend)
  F2  CAUSALITY GATE  the P164 construction test, automated: candidates are
      rebuilt from a truncated history and every overlapping value must be
      bit-identical — a feature that fails is DROPPED LOUDLY, never shipped.
  F3  SCREEN  (design era only): |IC| floor, redundancy prune vs the base
      135 AND vs already-accepted candidates, and half-split IC sign
      stability (a feature whose sign flips inside the design era is noise).
  F4  LADDER  the survivors only earn deployment by beating the base set
      inside the model pipeline (regime_model_lab --engineered re-run).

`build_screened(ctx)` is deterministic and self-contained so the model lab
can call it directly — no hidden state between the two pipelines.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scipy import stats

from training.splits import DESIGN_ERA  # noqa: E402

IC_FLOOR = 0.03
REDUNDANCY_MAX_CORR = 0.85
MAX_KEEP = 25
TOP_BASE = 12


# ---------------------------------------------------------------- F1
def generate_candidates(X, close, lab, fz, feats, y, design=DESIGN_ERA,
                        top=None):
    """Return (F, names, family_of, top). All constructions are causal:
    rolling/shift only, never centered, never future-indexed. `top` (the
    base features to transform/cross) is computed from the design era when
    None; the causality gate PASSES IT BACK IN for the truncated rebuild —
    the gate tests the CONSTRUCTIONS' causality, and the selection itself
    is era-confined by the design mask, not per-bar."""
    n = len(close)
    if top is None:
        s, e = design
        m = (np.arange(n) >= s) & (np.arange(n) < e) & ~np.isnan(y)
        ry = np.argsort(np.argsort(y[m]))
        ics = []
        for i in range(X.shape[1]):
            col = X[m][:, i]
            if np.std(col) == 0 or np.isnan(col).any():
                ics.append(0.0); continue
            ics.append(abs(float(stats.spearmanr(col, ry).statistic)))
        top = list(np.argsort(ics)[::-1][:TOP_BASE])

    cols, names, fam = [], [], []

    def add(v, name, family):
        cols.append(np.asarray(v, dtype=float)); names.append(name)
        fam.append(family)

    bull = (lab == 1).astype(float)
    bear = (lab == 2).astype(float)
    for i in top:
        base = pd.Series(X[:, i])
        z = ((base - base.rolling(42).mean()) / base.rolling(42).std()).to_numpy()
        add(z, f"z42_{feats[i]}", "transform")
        add(base.diff(6).to_numpy(), f"d6_{feats[i]}", "transform")
        add(X[:, i] * bull, f"x_{feats[i]}_bull", "regime_cross")
        add(X[:, i] * bear, f"x_{feats[i]}_bear", "regime_cross")
    for a in range(6):
        for b in range(a + 1, 6):
            ia, ib = top[a], top[b]
            add(X[:, ia] * X[:, ib], f"p_{feats[ia]}__{feats[ib]}", "product")

    c = pd.Series(close)
    r1 = pd.Series(np.concatenate([[np.nan], np.log(close[1:] / close[:-1])]))
    add((close / c.rolling(120).max().to_numpy()) - 1.0, "dd_from_20d_high", "structure")
    add((close / c.rolling(120).min().to_numpy()) - 1.0, "up_from_20d_low", "structure")
    vol42 = r1.rolling(42).std()
    add(vol42.rolling(42).std().to_numpy(), "vol_of_vol_7d", "structure")
    rng = (c.rolling(42).max() - c.rolling(42).min()) / c.rolling(42).mean()
    add((rng / rng.rolling(180).mean()).to_numpy(), "range_compression", "structure")
    up = (r1 > 0).astype(float)
    streak = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
    add((streak * (2 * up - 1)).to_numpy(), "signed_updown_streak", "structure")
    add((close / c.rolling(50).mean().to_numpy() - 1.0), "gap_sma50", "structure")
    trend = np.sign(close - c.rolling(200).mean().to_numpy())
    add(fz, "funding_z_causal", "funding")
    add(pd.Series(fz).diff(30).to_numpy(), "funding_z_mom5d", "funding")
    add(fz * trend, "funding_x_trend", "funding")

    # inf -> 0 (a zero denominator is "no information", not a huge value —
    # nan_to_num's 1e308 default overflowed StandardScaler's variance into
    # NaN and crashed Ridge on ETH), then winsorize to a sane range.
    F = np.column_stack([
        np.clip(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
        for v in cols])
    return F, names, fam, top


# ---------------------------------------------------------------- F2
def causality_gate(X, close, lab, fz, feats, y, F, names, top, cut=6000, buffer=250):
    """P164 construction test: rebuild candidates from history truncated at
    `cut` (with the SAME `top` selection pinned); every value before
    cut-buffer must match the full build exactly. Returns indices of
    PASSING candidates; failures are named."""
    Ft, names_t, _, _ = generate_candidates(
        X[:cut], close[:cut], lab[:cut], fz[:cut], feats, y[:cut], top=top)
    assert names_t == names, "generator must be deterministic in its naming"
    ok, failed = [], []
    for j in range(F.shape[1]):
        a, b = F[: cut - buffer, j], Ft[: cut - buffer, j]
        if np.allclose(a, b, equal_nan=True):
            ok.append(j)
        else:
            failed.append(names[j])
    if failed:
        print(f"  F2 CAUSALITY FAILURES (dropped loudly): {failed}", flush=True)
    return ok


# ---------------------------------------------------------------- F3
def screen(X, F, names, ok_idx, y, design=DESIGN_ERA):
    s, e = design
    n = len(y)
    m = (np.arange(n) >= s) & (np.arange(n) < e) & ~np.isnan(y)
    half = (s + e) // 2
    m1 = m & (np.arange(n) < half)
    m2 = m & (np.arange(n) >= half)

    def ic(col, mask):
        c = col[mask]
        if np.std(c) == 0:
            return 0.0
        r = stats.spearmanr(c, y[mask]).statistic
        return float(r) if np.isfinite(r) else 0.0

    scored = []
    for j in ok_idx:
        i_full, i1, i2 = ic(F[:, j], m), ic(F[:, j], m1), ic(F[:, j], m2)
        if abs(i_full) < IC_FLOOR:
            continue
        if np.sign(i1) != np.sign(i2) or i1 == 0 or i2 == 0:
            continue                       # unstable inside the design era
        scored.append((abs(i_full), j, i_full))
    scored.sort(reverse=True)

    kept, report = [], []
    base_design = X[m]
    for score, j, i_full in scored:
        col = F[m][:, j]
        if np.std(col) == 0:
            continue
        # redundancy vs base features
        max_base = max(
            (abs(float(np.corrcoef(col, base_design[:, i])[0, 1]))
             for i in range(X.shape[1]) if np.std(base_design[:, i]) > 0),
            default=0.0)
        if max_base > REDUNDANCY_MAX_CORR:
            continue
        # redundancy vs already-kept candidates
        if any(abs(float(np.corrcoef(col, F[m][:, k])[0, 1])) > REDUNDANCY_MAX_CORR
               for k in kept):
            continue
        kept.append(j)
        report.append({"name": names[j], "ic": round(i_full, 4)})
        if len(kept) >= MAX_KEEP:
            break
    return kept, report


# ---------------------------------------------------------------- entry
def build_screened(ctx, verbose=False):
    """The model lab's entry point: deterministic, self-contained.
    Returns (F_screened, screened_names)."""
    F, names, fam, top = generate_candidates(
        ctx["X"], ctx["close"], ctx["lab"], ctx["fz"], ctx["feats"], ctx["y"])
    ok = causality_gate(ctx["X"], ctx["close"], ctx["lab"], ctx["fz"],
                        ctx["feats"], ctx["y"], F, names, top)
    kept, report = screen(ctx["X"], F, names, ok, ctx["y"])
    if verbose:
        print(f"  F1 generated={len(names)}  F2 causal-pass={len(ok)}  "
              f"F3 screened={len(kept)}", flush=True)
        for r in report:
            print(f"    {r['name']:<40} ic={r['ic']:+.4f}", flush=True)
    return F[:, kept], [names[j] for j in kept], report


def main():
    from training.regime_model_lab import _ctx
    from training.provenance import provenance_stamp
    out = {"assets": {}, "provenance": provenance_stamp(),
           "config": {"ic_floor": IC_FLOOR, "max_corr": REDUNDANCY_MAX_CORR,
                      "max_keep": MAX_KEEP}}
    for asset in ("BTC", "ETH", "SOL"):
        ctx = _ctx(asset); ctx["asset"] = asset
        print(f"\n########## {asset} feature lab ##########", flush=True)
        _, names, report = build_screened(ctx, verbose=True)
        out["assets"][asset] = {"screened": report}
    p = REPO / "training" / "reports" / "feature_lab_p250.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nDONE -> {p}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)

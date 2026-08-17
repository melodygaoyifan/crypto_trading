"""[P288-A] Long-horizon edge probe — the never-tried axis, measured.

WHY THIS EXISTS
    The P288 design-axis audit found that every trained model and every probe
    the project ever ran used targets at 4h/16h (edge_probe.py: HORIZONS=(1,4);
    train_supervised_full.py: H=4) — while the only certified strategies (the
    trend/hold rule books) monetize a MULTI-WEEK horizon. "No forecast edge"
    is therefore a 4h/16h fact, not a measured fact about the horizons the
    books actually live at. The required-IC bar FALLS with horizon (the same
    round-trip cost amortizes over a larger sigma_fwd); nobody has measured
    whether the 48h+ cells clear it. This probe closes that cell at the
    cheapest honest tier: ridge only, two feature groups, strict window.

PRE-REGISTERED DESIGN (fixed before the first run; no knobs, no search)
    horizons   : {12, 24, 42} bars = 48h / 96h / 168h forward returns,
                 plus h=4 (16h) as the REPRODUCTION ANCHOR ONLY — it must
                 land within +/-0.01 of the recorded strict-probe BTC
                 ALL/ridge IC (0.088, training/reports/edge_probe_strict_
                 p281.txt) or the run stops and diagnoses instead of
                 reporting. The 16h cells are already-recorded evidence;
                 they are re-printed for continuity, not re-adjudicated.
    model      : ridge ONLY (the proven generalizer; every richer family is
                 measured dead on this basis — P241/P250/P281/P286).
    groups     : ALL and external (the two that mattered historically).
    window     : MIN_TRAIN=7200 — every prediction past the split-aware GMM
                 fit boundary (the falsification setting, P200-LADDER).
    machinery  : imported from edge_probe.py (walk_forward, spearman, data /
                 manifest resolution) — single-source, P172. This file adds
                 only the horizon loop, the corrected statistics, and the
                 verdict logic.

THE STATISTICS (the load-bearing part at long horizons)
    sigma_fwd(h)   : measured from the SAME joined (prediction, forward
                     return) pairs each cell scores.
    required IC(h) : solves  0.7979 * 2*sin(pi*IC/6) * sigma_fwd(h)
                             >= 2.0 * COST_RT_BPS         (P166 arithmetic,
                     exact 2sin form, 2x margin — i.e. the bar the September
                     forward gate itself applies; NOTE stricter than
                     edge_probe.py's 1x-margin linearized bar).
    overlap t      : t = IC * sqrt(n_eff - 1), n_eff = n_oos / h  (P231 —
                     h-bar forward returns sampled per 4H bar overlap h-fold;
                     the uncorrected t is inflated ~sqrt(h)).
    detectable IC  : 2 / sqrt(n_eff - 1) — the minimum IC that can reach
                     |t|>=2 at this cell's effective sample size. This is
                     the cell's POWER LIMIT and is reported beside every
                     verdict.

PRE-COMMITTED VERDICT RULES (written before the run; the UNDECIDABLE
comparison is resolved to the directive's evident intent — when the power
limit sits ABOVE the bar, a non-pass says nothing about bar-level ICs, so
it must not be recorded as a decisive FAIL; the directive's literal
sentence inverted the comparison, which would have marked the decisive
16h cells undecidable):
    EDGE_CANDIDATE : IC > 0  AND  |t_corrected| >= 2
                     AND  edge(IC) >= 2 * COST_RT_BPS.
    UNDECIDABLE    : not EDGE_CANDIDATE and detectable_ic > required_ic —
                     the probe lacks the power to adjudicate the bar at this
                     horizon; `point_above_bar` flags whether the point
                     estimate at least sits at/above the bar (a hypothesis,
                     never evidence).
    FAIL           : not EDGE_CANDIDATE and detectable_ic <= required_ic —
                     the cell had the power to certify a bar-level IC and
                     did not.

CAVEATS (printed into the report)
    * These windows are many-times-read (P260): an EDGE_CANDIDATE here is a
      Rung-0 hypothesis earning at most a forward ledger — P282's standing
      caveat applies (EDGE_CANDIDATE preceded every dead campaign; it is a
      necessary gate carrying near-zero positive information).
    * A FAIL is decisive only for this cheap tier (ridge on these features).
    * An UNDECIDABLE is a statement about statistical power, not the market.

Usage:  python -X utf8 training/scripts/edge_probe_longhorizon.py
Writes: training/reports/edge_probe_longhorizon_p288.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import edge_probe as ep  # noqa: E402  (single-source machinery, P172)

# Strict window: every prediction past the split-aware GMM fit boundary.
ep.MIN_TRAIN = 7200

COST_RT_BPS = 6.0          # coinbase taker 3bps x 2 legs (same as edge_probe)
MARGIN = 2.0               # P166 gate margin — the bar is 12bps of edge
HORIZONS = (4, 12, 24, 42)  # 16h anchor + 48h / 96h / 168h
ANCHOR_H = 4
ANCHOR_REF_IC = 0.088      # recorded strict BTC 16h ALL/ridge (p281 report)
ANCHOR_TOL = 0.010
GROUP_NAMES = ("ALL", "external")

REPORT = _HERE.parent / "reports" / "edge_probe_longhorizon_p288.json"


def edge_bps(ic: float, sigma_bps: float) -> float:
    """P166 exact form: E|z| * pearson(rho) * sigma, pearson = 2 sin(pi rho/6)."""
    return 0.7979 * 2.0 * math.sin(math.pi * ic / 6.0) * sigma_bps


def required_ic_exact(sigma_bps: float) -> float:
    """Smallest IC whose edge clears MARGIN * COST_RT_BPS (exact inverse)."""
    if sigma_bps <= 0:
        return float("inf")
    s = (MARGIN * COST_RT_BPS) / (0.7979 * 2.0 * sigma_bps)
    if s >= 1.0:
        return float("inf")
    return (6.0 / math.pi) * math.asin(s)


def probe_cell(df: pd.DataFrame, feats: list, close: np.ndarray,
               h: int, gname: str) -> dict:
    sel = ep.GROUPS[gname]
    cols = feats if sel is None else [c for c in feats if sel(c)]
    fwd = np.full(len(close), np.nan)
    fwd[:-h] = close[h:] / close[:-h] - 1.0

    X = df[cols].to_numpy(dtype=float)
    preds = np.full(len(fwd), np.nan)
    for idx, p in ep.walk_forward(X, fwd, "ridge"):
        preds[idx] = p
    m = ~(np.isnan(preds) | np.isnan(fwd))
    n_oos = int(m.sum())
    ic = ep.spearman(preds[m], fwd[m])

    # sigma from the SAME joined pairs the IC is computed on
    sigma_bps = float(np.nanstd(fwd[m])) * 1e4 if n_oos else float("nan")
    n_eff = n_oos / h
    t = (ic * math.sqrt(max(n_eff - 1.0, 1.0))
         if not math.isnan(ic) else float("nan"))
    req = required_ic_exact(sigma_bps)
    detect = 2.0 / math.sqrt(max(n_eff - 1.0, 1.0))
    cell_edge = edge_bps(ic, sigma_bps) if not math.isnan(ic) else float("nan")

    passes = (not math.isnan(ic) and ic > 0 and abs(t) >= 2.0
              and cell_edge >= MARGIN * COST_RT_BPS)
    if passes:
        verdict = "EDGE_CANDIDATE"
    elif detect > req:
        verdict = "UNDECIDABLE"
    else:
        verdict = "FAIL"

    return dict(
        horizon_bars=h, horizon_hours=h * 4, group=gname, model="ridge",
        n_oos=n_oos, n_eff=round(n_eff, 1),
        ic=None if math.isnan(ic) else round(ic, 4),
        t_overlap_corrected=None if math.isnan(t) else round(t, 2),
        sigma_fwd_bps=round(sigma_bps, 1),
        edge_bps=None if math.isnan(cell_edge) else round(cell_edge, 2),
        required_ic=round(req, 4), detectable_ic=round(detect, 4),
        point_above_bar=bool(not math.isnan(ic) and ic >= req),
        verdict=verdict,
    )


def main() -> int:
    reports = []
    anchor_checked = False
    for asset in ("BTC", "ETH", "SOL"):
        df = pd.read_parquet(ep.DATA_DIR / f"{asset}_4H_full.parquet")
        manifest = json.loads(ep.MANIFEST.read_text(encoding="utf-8"))
        feats = [c for c in manifest["all_features"] if c in df.columns]
        feats += [c for c in df.columns if c.startswith("fv2_")]
        close = df["close"].to_numpy(dtype=float)
        out = {"asset": asset, "n_bars": len(df), "min_train": ep.MIN_TRAIN,
               "cells": []}
        print(f"\n===== {asset}: {len(df)} bars, {len(feats)} features, "
              f"min_train={ep.MIN_TRAIN} =====")
        print(f"{'h':>5}{'group':>10}{'n_oos':>7}{'n_eff':>8}{'IC':>8}"
              f"{'t_corr':>8}{'sigma':>7}{'edge':>7}{'reqIC':>7}"
              f"{'detIC':>7}  verdict")
        for h in HORIZONS:
            for gname in GROUP_NAMES:
                cell = probe_cell(df, feats, close, h, gname)
                out["cells"].append(cell)
                print(f"{cell['horizon_hours']:>4}h{cell['group']:>10}"
                      f"{cell['n_oos']:>7}{cell['n_eff']:>8.0f}"
                      f"{(cell['ic'] if cell['ic'] is not None else float('nan')):>8.3f}"
                      f"{(cell['t_overlap_corrected'] if cell['t_overlap_corrected'] is not None else float('nan')):>8.2f}"
                      f"{cell['sigma_fwd_bps']:>7.0f}"
                      f"{(cell['edge_bps'] if cell['edge_bps'] is not None else float('nan')):>7.1f}"
                      f"{cell['required_ic']:>7.3f}{cell['detectable_ic']:>7.3f}"
                      f"  {cell['verdict']}"
                      + ("  [point>=bar]" if cell["point_above_bar"]
                         and cell["verdict"] == "UNDECIDABLE" else ""))
                # Reproduction anchor: strict BTC 16h ALL/ridge must match
                # the recorded probe before anything else is trusted.
                if (asset == "BTC" and h == ANCHOR_H and gname == "ALL"):
                    got = cell["ic"] if cell["ic"] is not None else float("nan")
                    if math.isnan(got) or abs(got - ANCHOR_REF_IC) > ANCHOR_TOL:
                        print(f"\nANCHOR FAILED: BTC 16h ALL/ridge IC={got} "
                              f"vs recorded {ANCHOR_REF_IC} (tol {ANCHOR_TOL})."
                              f" STOPPING — machinery drift, diagnose before "
                              f"trusting any long-horizon number.")
                        return 2
                    anchor_checked = True
                    print(f"      [anchor OK: reproduces recorded strict "
                          f"IC {ANCHOR_REF_IC} within {ANCHOR_TOL}]")
        reports.append(out)

    if not anchor_checked:
        print("ANCHOR NEVER RAN — refusing to report.")
        return 2

    new_cells = [c for r in reports for c in r["cells"]
                 if c["horizon_bars"] != ANCHOR_H]
    n_pass = sum(1 for c in new_cells if c["verdict"] == "EDGE_CANDIDATE")
    n_und = sum(1 for c in new_cells if c["verdict"] == "UNDECIDABLE")
    n_fail = sum(1 for c in new_cells if c["verdict"] == "FAIL")
    overall = ("EDGE_CANDIDATE" if n_pass else
               ("POWER_LIMITED" if n_und > n_fail else "NO_EDGE"))
    print(f"\n========== LONG-HORIZON OVERALL: {overall} "
          f"(new cells: {n_pass} pass / {n_und} undecidable / {n_fail} fail "
          f"of {len(new_cells)}) ==========")

    payload = {
        "probe": "edge_probe_longhorizon_p288",
        "pre_registered": {
            "horizons_bars": list(HORIZONS), "anchor_h": ANCHOR_H,
            "groups": list(GROUP_NAMES), "model": "ridge",
            "min_train": ep.MIN_TRAIN, "cost_rt_bps": COST_RT_BPS,
            "margin": MARGIN,
            "t_correction": "n_eff = n_oos / h (P231 overlap)",
        },
        "caveats": [
            "windows many-times-read (P260): a PASS is a Rung-0 hypothesis "
            "earning at most a forward ledger; P282: EDGE_CANDIDATE carries "
            "near-zero positive information",
            "a FAIL is decisive only for this cheap tier (ridge, these "
            "features)",
            "UNDECIDABLE is a statement about statistical power, not the "
            "market",
        ],
        "overall": overall,
        "assets": reports,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"Report: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

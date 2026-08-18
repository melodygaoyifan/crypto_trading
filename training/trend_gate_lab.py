"""[P306] trend_regime_gate lab — does zeroing the trend seat in the gated
regimes actually IMPROVE the trend book's after-cost PnL, per asset?

WHY THIS EXISTS
    `trend_regime_gate` has sat in "shadow" since P198 with a promotion rule
    nobody could execute, because the two pieces of evidence disagree and
    neither is the right measurement:

      * P198's split is IN-SAMPLE (measured on the very loss window that
        motivated it) and says WEAK_CONSOLIDATION / NEUTRAL_DRIFT lose;
      * the forward reading was INVERTED (blocked ticks +9.2bps vs kept -2.5),
        on n small enough to be noise.

    Both are per-TICK signed-return averages. The decision is not about mean
    returns in a regime - it is whether the GATE, applied to the trend book,
    leaves more money after costs. A regime can have a negative mean return
    and still be one the trend book is SHORT in, in which case gating it
    destroys the profit. That distinction is what nobody had measured.

WHAT IT MEASURES
    pos_base  = sign(close - SMA200)          the live trend seat: +/-1, and
                                              the sleeve sizes by SIGN, so a
                                              magnitude model would answer a
                                              question nobody asked
    pos_gated = pos_base, forced to 0 in the gated regimes

    Both are executed with the chassis's 1-bar delay and per-side costs
    (training.mechanism_lab.pnl_after_cost - imported, not restated, P172),
    and reported in the DESIGN era [3000, 9100) and the PRE-DESIGN era
    [800, 3000) separately. A mechanism that only wins in one era is
    era-fragile and does not promote (P243/P244).

VOCABULARY NOTE, AND IT MATTERS
    The live gate set is {WEAK_CONSOLIDATION, NEUTRAL_DRIFT}. P267 deployed
    the clean split-aware GMMs, and NEUTRAL_DRIFT does not exist in ANY of
    the k=6/7/7 vocabularies - so in production the gate is
    WEAK_CONSOLIDATION only. The lab reports both the live set and each
    single regime, so the recommendation is about the gate that actually
    runs rather than the one the config text describes.

PRE-COMMITTED VERDICT (fixed before the numbers were read)
    ENFORCE for an asset iff the gated book beats the ungated one in BOTH
    eras, after costs. Anything else is BASE STANDS. Deliberately strict:
    the gate is already live-shadow and doing nothing, so the burden is on
    the change. No promotion happens here - a PASS earns the config flip
    being proposed with its numbers attached, nothing more.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if REPO.name == "training":
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

from training.mechanism_lab import pnl_after_cost              # noqa: E402
from training.regime_model_lab import _ctx, DESIGN             # noqa: E402
from training.train_supervised_full import COST_BPS, DATA_DIR  # noqa: E402

DS, DE = DESIGN
PRE = (800, 3000)
SMA_N = 200
LIVE_GATE = ("WEAK_CONSOLIDATION", "NEUTRAL_DRIFT")
REPORT = REPO / "training" / "reports" / "trend_gate_lab_p306.json"


def _regimes(asset: str, n: int):
    """(per-bar cluster id, id->name). Reads the same artifact pair the
    runtime loads, so a rename cannot silently desync the two."""
    import pandas as pd
    df = pd.read_parquet(Path(DATA_DIR) / f"{asset}_4H_full.parquet")
    ids = df["regime"].to_numpy()[:n].astype(int)
    cfg = json.loads(
        (REPO / "training" / "training_data" / "gmm_models" / asset
         / "gmm_config.json").read_text(encoding="utf-8"))
    names = list(cfg.get("regime_names") or [])
    return ids, {i: (names[i] if i < len(names) else f"REGIME_{i}")
                 for i in range(int(cfg.get("n_components") or len(names)))}


def _sma_sign(close: np.ndarray) -> np.ndarray:
    pos = np.zeros(len(close))
    for i in range(SMA_N, len(close)):
        m = close[i - SMA_N:i].mean()
        pos[i] = 1.0 if close[i] > m else -1.0
    return pos


def _era(pos, close, cost, lo, hi):
    """Net after-cost return over [lo, hi). The chassis takes the FULL
    series and the window - slicing first would re-establish the window's
    opening position cost-free and quietly flatter every candidate."""
    return float(pnl_after_cost(close, pos, cost, lo, hi)["net"])


def run(assets=("BTC", "ETH", "SOL")) -> dict:
    out = {"generated": datetime.now(timezone.utc).isoformat(),
           "sma": SMA_N, "eras": {"design": [DS, DE], "pre_design": list(PRE)},
           "live_gate_set": list(LIVE_GATE), "assets": {}}
    for a in assets:
        ctx = _ctx(a)
        close, n = ctx["close"], ctx["n"]
        ids, names = _regimes(a, n)
        cost = COST_BPS[a]
        base = _sma_sign(close)
        rec = {"cost_rt_bps": cost,
               "vocabulary": sorted(set(names.values())),
               "neutral_drift_exists":
                   "NEUTRAL_DRIFT" in set(names.values()),
               "base": {"design": _era(base, close, cost, DS, DE),
                        "pre_design": _era(base, close, cost, *PRE)},
               "gates": {}}
        candidates = {"LIVE_SET": [r for r in LIVE_GATE
                                   if r in set(names.values())]}
        for nm in sorted(set(names.values())):
            candidates[nm] = [nm]
        for label, regs in candidates.items():
            mask = np.isin(ids, [i for i, v in names.items() if v in regs])
            if not mask.any():
                rec["gates"][label] = {"share": 0.0, "note": "regime absent"}
                continue
            gated = base.copy()
            gated[mask] = 0.0
            d = _era(gated, close, cost, DS, DE)
            p = _era(gated, close, cost, *PRE)
            rec["gates"][label] = {
                "regimes": regs,
                "share_of_bars": round(float(mask.mean()), 4),
                "design": d, "pre_design": p,
                "design_delta": round(d - rec["base"]["design"], 4),
                "pre_design_delta": round(p - rec["base"]["pre_design"], 4),
                "verdict": ("ENFORCE" if (d > rec["base"]["design"]
                                          and p > rec["base"]["pre_design"])
                            else "BASE STANDS"),
            }
        out["assets"][a] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="BTC,ETH,SOL")
    args = ap.parse_args()
    rep = run(tuple(x.strip().upper() for x in args.assets.split(",") if x.strip()))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    for a, rec in rep["assets"].items():
        print(f"\n=== {a}   base design={rec['base']['design']:+.3f} "
              f"pre={rec['base']['pre_design']:+.3f}  "
              f"(NEUTRAL_DRIFT exists: {rec['neutral_drift_exists']})")
        for label, g in sorted(rec["gates"].items(),
                               key=lambda kv: -(kv[1].get("design_delta") or -9)):
            if "design" not in g:
                continue
            print(f"  {label:<22} share={g['share_of_bars']:.3f} "
                  f"design={g['design']:+.3f} ({g['design_delta']:+.3f})  "
                  f"pre={g['pre_design']:+.3f} ({g['pre_design_delta']:+.3f})  "
                  f"{g['verdict']}")
    print(f"\nreport -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

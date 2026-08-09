"""[P248-GP2] Measure the RUNTIME-SAFE variant of SOL's bear-cell ridge.

The p247 SOL perp bear cell (ridge_defensive, alpha=30, CV +5.5%) trains on
the pruned 135-feature parquet set. The shadow harness can only faithfully
compute price-derived features from its ~720-bar OHLC frame plus the causal
funding z. Shadowing an unmeasured variant would be dishonest — so this
probe re-runs the bear cell restricted to the runtime-safe subset and
reports the delta. If the variant holds up, IT (not the full-feature
model) is what the shadow deploys; its export carries the exact feature
names + scaler + coefficients.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from training.regime_model_lab import (  # noqa: E402
    _ctx, cell_series, REGIME_ID, DESIGN_ERA,
)
from training.splits import purged_folds  # noqa: E402
from training.eval_report import seg_metrics  # noqa: E402

# Runtime-safe = computable from a 720-bar 4H OHLC frame (price transforms)
# plus the causal funding z the harness derives itself. Excludes: denoised
# (deque warmup), external/Coinglass (separate feeds), fv2 (flow feed),
# regime_proba (GMM artifacts), sentiment/onchain.
RT_SAFE_RE = re.compile(
    r"^(ret_|vol_|atr_|rsi|macd|bb_|dist_|trend_|mom|sma|ema|high_|low_|"
    r"close_|range_|vol_ratio|vol_regime|vol_expansion|vol_trend)"
)


def main():
    for asset in ("SOL", "BTC"):
        ctx = _ctx(asset); ctx["asset"] = asset
        feats = ctx["feats"]
        rt_idx = [i for i, f in enumerate(feats)
                  if RT_SAFE_RE.match(f) and "_denoised" not in f]
        print(f"\n{asset}: runtime-safe features {len(rt_idx)}/{len(feats)}")
        s, e = DESIGN_ERA

        for label, X_use in (("full_pruned", ctx["X"]),
                             ("rt_safe", ctx["X"][:, rt_idx])):
            ctx2 = dict(ctx); ctx2["X"] = X_use
            cv_pnl, cv_sh = [], []
            for tr, va in purged_folds(s, e):
                seg = cell_series("ridge_defensive", {"alpha": 30.0}, ctx2,
                                  "bear", int(va[0]), int(va[-1] + 1),
                                  fit_lt=int(va[0]), instrument="perp")
                m = seg_metrics(seg)
                cv_pnl.append(m["pnl_pct"]); cv_sh.append(m["sharpe"])
            # validation-era read is NOT taken here — design-era CV only;
            # the variant's validation number comes from the next full
            # assembly run if it replaces the cell.
            print(f"  {label:<12} bear-cell CV pnl={np.mean(cv_pnl):+.2f}% "
                  f"sharpe={np.mean(cv_sh):+.3f}")
    print("\nDONE")


if __name__ == "__main__":
    sys.exit(main() or 0)

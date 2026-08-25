"""[P407j] Forward shadow of the skew+ETF ensemble (observation-only, Iron Law 7).

P407i measured that skew (contrarian options) and ETF flow (momentum spot demand)
are COMPLEMENTARY -- on the ~1.7y overlap, requiring their AGREEMENT is positive
OOS on BOTH assets, rescuing exactly the ETH case where skew alone is negative OOS,
and the current live precedence (skew runs last, OVERRIDING ETF on disagreement) is
measured suboptimal. But 1.7y / ~15-30 trades is P348-thin and it is the recent
ETF-era regime, so flipping the LIVE combination rule on it would be the P243/P198
overfit mistake. This shadow accrues the FORWARD comparison instead:

  each tick, for BTC/ETH, it logs three directions to data/strategy_shadow/
  skewetf_{ASSET}.jsonl so the existing compute_shadow_ic scorer can A/B them on
  forward returns:
    skewetf_skew   = skew solo   (== the current live override behaviour)
    skewetf_etf    = ETF solo
    skewetf_agree  = agree-gated (skew iff skew == etf != 0, else flat)

It changes NO live position -- it reads the two already-live feeds and writes a
ledger. A live combination change is a P141 decision, taken only if the forward
read confirms agree-gating beats skew-override (the P200 ladder). confidence =
|direction| so a flat row contributes zero to the IC, never a saturated claim
(P224/P236). A not-fresh feed contributes flat (P2: absence is never a position).
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

logger = logging.getLogger("HMATS.SkewEtfCombo")

SHADOW_STRATEGY_NAMES = ("skewetf_skew", "skewetf_etf", "skewetf_agree")


def combo_directions(skew_dir: float, skew_fresh: bool,
                     etf_dir: float, etf_fresh: bool):
    """Pure: the three shadow directions from the two feed readings.

    Not-fresh -> that leg is flat (0.0). agree-gated fires only when both are
    fresh, directional, and AGREE. Returns {name: direction}.
    """
    skew = float(skew_dir) if skew_fresh else 0.0
    etf = float(etf_dir) if etf_fresh else 0.0
    agree = skew if (skew != 0.0 and skew == etf) else 0.0
    return {"skewetf_skew": skew, "skewetf_etf": etf, "skewetf_agree": agree}


class SkewEtfComboShadow:
    """Self-contained (P248): writes its own ledger, touches no live position."""

    def __init__(self, data_dir: str = "data"):
        self._dir = os.path.join(data_dir, "strategy_shadow")

    def record_tick(self, asset: str, skew_dir: float, skew_fresh: bool,
                    etf_dir: float, etf_fresh: bool) -> None:
        dirs = combo_directions(skew_dir, skew_fresh, etf_dir, etf_fresh)
        now = time.time()
        iso = datetime.now(timezone.utc).isoformat()
        diag = {"skew_fresh": bool(skew_fresh), "etf_fresh": bool(etf_fresh),
                "skew_dir": float(skew_dir), "etf_dir": float(etf_dir)}
        try:
            os.makedirs(self._dir, exist_ok=True)
            path = os.path.join(self._dir, f"skewetf_{asset}.jsonl")
            with open(path, "a", encoding="utf-8") as fh:
                for strat in SHADOW_STRATEGY_NAMES:
                    d = dirs[strat]
                    fh.write(json.dumps({
                        "ts": now, "iso": iso, "strategy": strat, "asset": asset,
                        "direction": float(d), "confidence": abs(float(d)),
                        "diagnostics": diag,
                    }) + "\n")
        except Exception as e:  # noqa: silent-swallow — a ledger write must never kill the tick (Iron Law 7); next tick retries
            logger.warning("[SKEWETF-SHADOW] %s: write failed (%s: %s)",
                           asset, type(e).__name__, e)

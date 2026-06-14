"""
Trend-following decision layer — flag-gated, default OFF.

Lets trend-following (strategies/trend_following.py; backtested 3-asset Sharpe ~0.4,
PBO 0.42) become the live decision layer via one config flag, with three modes:

  off     (DEFAULT): no-op. Engine behaves byte-identically to today.
  shadow:  computes the trend decision each tick from the engine's own OHLCV buffer
           and LOGS it vs the engine's intent. Never changes a trade.
  enforce: OVERRIDES the intent's direction + target_exposure with trend-following.

WHY this is OFF by default and must stay so until forward-confirmed: trend-following
is validated on BACKTEST only (PBO 0.42 = MARGINAL, forward clock barely started).
Flipping to `enforce` before the forward validation confirms would repeat the exact
backtest-vs-live trap that caused the -25% (DRL: backtest Sharpe 9-10 -> live -2.62).
Promotion path: keep `shadow` running -> confirm forward Sharpe>0 via the cron's
`score` -> THEN flip to `enforce`. The flip is a one-line config change; the code
is ready so there is zero friction when the evidence arrives.

Composition notes for when `enforce` is actually turned on (review before flipping):
  - The NET cap (P144) still applies downstream in execute_intent_v2 -> good.
  - The alpha gate is calibrated to the quant signal; a trend-overridden intent may
    be re-gated by it. Decide then whether trend intents bypass/own the gate.
Fail-open everywhere: any error -> no override, WARN, engine decision stands.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List

from strategies.trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)

_VALID_MODES = ("off", "shadow", "enforce")


class TrendDecisionLayer:
    def __init__(self, mode: str = "off", max_exposure: float = 0.20,
                 min_abs_pos: float = 0.05) -> None:
        self._strat = TrendFollowingStrategy()
        self.mode = mode if mode in _VALID_MODES else "off"
        self.max_exposure = float(max_exposure)   # cap on exposure when enforcing
        self.min_abs_pos = float(min_abs_pos)      # |target_position| below this -> flat
        self._closes: Dict[str, List[float]] = {}

    def set_mode(self, mode: str) -> None:
        self.mode = mode if mode in _VALID_MODES else "off"

    def cache(self, asset: str, ohlcv_df: Any) -> None:
        """Stash the asset's close history from the engine's OHLCV df (>=721 bars
        live; trend needs ~421). Only when not OFF. Fail-open."""
        if self.mode == "off" or ohlcv_df is None:
            return
        try:
            if len(ohlcv_df) >= self._strat.min_history():
                self._closes[asset] = [float(x) for x in ohlcv_df["close"].tolist()]
        except Exception as e:  # noqa: silent-swallow — bad df shape, skip caching (engine decision stands)
            logger.debug(f"[TREND-LAYER] {asset} cache skip: {type(e).__name__}: {e}")

    def apply(self, asset: str, intent: Any) -> Optional[Dict[str, Any]]:
        """off -> None. shadow -> log trend vs engine, return info (no change).
        enforce -> override intent.direction/target_exposure. Fail-open -> None."""
        if self.mode == "off":
            return None
        closes = self._closes.get(asset)
        if not closes:
            return None
        try:
            res = self._strat.compute(closes)
        except Exception as e:
            logger.warning(f"[TREND-LAYER] {asset} compute error: {type(e).__name__}: {e}")
            return None
        if res is None:
            return None
        tp = float(res["target_position"])
        tdir = 1.0 if tp > self.min_abs_pos else (-1.0 if tp < -self.min_abs_pos else 0.0)
        texp = min(self.max_exposure, abs(tp) * self.max_exposure)
        eng_dir = float(getattr(intent, "direction", 0.0) or 0.0)

        if self.mode == "shadow":
            logger.info(
                f"[TREND-LAYER][SHADOW] {asset}: trend dir={tdir:+.0f} exp={texp:.3f} "
                f"(pos={tp:+.2f}) | engine dir={eng_dir:+.2f}"
            )
            return {"mode": "shadow", "trend_dir": tdir, "trend_exp": texp,
                    "engine_dir": eng_dir, "target_position": tp}

        # enforce: trend-following becomes the decision
        logger.warning(
            f"[TREND-LAYER][ENFORCE] {asset}: engine dir={eng_dir:+.2f} -> "
            f"trend dir={tdir:+.0f} exp={texp:.3f} (pos={tp:+.2f})"
        )
        try:
            intent.direction = tdir
            intent.target_exposure = texp
            intent.is_actionable = bool(tdir != 0.0)
            intent.veto_active = False
            intent.quant_strategy_id = "trend_following"
        except Exception as e:
            logger.warning(f"[TREND-LAYER] {asset} enforce failed, engine decision stands: "
                           f"{type(e).__name__}: {e}")
            return None
        return {"mode": "enforce", "trend_dir": tdir, "trend_exp": texp, "target_position": tp}


_singleton: Optional[TrendDecisionLayer] = None


def get_trend_decision_layer(mode: str = "off") -> TrendDecisionLayer:
    """Process singleton. `mode` sets/updates the mode each call (cheap)."""
    global _singleton
    if _singleton is None:
        _singleton = TrendDecisionLayer(mode=mode)
    else:
        _singleton.set_mode(mode)
    return _singleton

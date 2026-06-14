"""
Trend-following decision layer — flag-gated, default OFF.

Lets trend-following (strategies/trend_following.py; backtest 3-asset Sharpe ~0.4,
PBO 0.42, +5.4%/yr) become the live decision by REPLACING the quant signal — the
current weak DECIDE signal (live IC -0.018). Three modes via `trend_following_mode`:

  off     (DEFAULT): no-op. Engine byte-identical to today.
  shadow:  computes the trend signal each tick and LOGS it vs the live quant_direction.
           Never changes a trade.
  enforce: INJECTS the trend signal AS the quant signal (quant_direction /
           quant_confidence / signal_edge_bps) BEFORE fusion, so the whole
           battle-tested pipeline (fusion -> alpha gate -> sizing -> NET cap P144 ->
           execution) trades it correctly. No late-override / gate-fighting.

WHY default off and why enforce must be paper-tested before live: trend-following
is BACKTEST-validated only (PBO 0.42 = MARGINAL, forward clock just started).
Flipping to enforce before forward confirmation repeats the backtest-vs-live trap
that caused -25% (DRL: backtest Sharpe 9-10 -> live -2.62). The harness exists so
the flip is one config line; the discipline is to flip only on evidence.

Fail-open everywhere: any error -> no change, engine signal stands.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List

from strategies.trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)

_VALID_MODES = ("off", "shadow", "enforce")


class TrendDecisionLayer:
    def __init__(self, mode: str = "off", base_edge_bps: float = 40.0,
                 min_abs_signal: float = 0.30) -> None:
        self._strat = TrendFollowingStrategy()
        self.mode = mode if mode in _VALID_MODES else "off"
        # signal_edge_bps injected for the gate = base_edge_bps * |signal|. base 40
        # clears the ~5-14bps live threshold at moderate-or-stronger conviction.
        self.base_edge_bps = float(base_edge_bps)
        # |signal| below this -> treat as no trend (flat); avoids trading weak chop.
        self.min_abs_signal = float(min_abs_signal)
        self._closes: Dict[str, List[float]] = {}

    def set_mode(self, mode: str) -> None:
        self.mode = mode if mode in _VALID_MODES else "off"

    def cache(self, asset: str, ohlcv_df: Any) -> None:
        if self.mode == "off" or ohlcv_df is None:
            return
        try:
            if len(ohlcv_df) >= self._strat.min_history():
                self._closes[asset] = [float(x) for x in ohlcv_df["close"].tolist()]
        except Exception as e:  # noqa: silent-swallow — bad df shape, skip (engine signal stands)
            logger.debug(f"[TREND-LAYER] {asset} cache skip: {type(e).__name__}: {e}")

    def process(self, asset: str, ohlcv_df: Any, agent_signals: Dict[str, Any],
                market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """One call, before fusion. off -> None. shadow -> log. enforce -> inject
        trend as the quant signal into agent_signals + market_data. Fail-open."""
        if self.mode == "off":
            return None
        self.cache(asset, ohlcv_df)
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
        sig = float(res["signal"])                 # [-1, 1] conviction
        if abs(sig) < self.min_abs_signal:
            sig = 0.0                               # weak trend -> flat
        eng_q = float(market_data.get("quant_direction", 0.0) or 0.0)

        if self.mode == "shadow":
            logger.info(f"[TREND-LAYER][SHADOW] {asset}: trend sig={sig:+.2f} "
                        f"(pos={res['target_position']:+.2f}) vs quant_dir={eng_q:+.2f}")
            return {"mode": "shadow", "trend_signal": sig, "quant_dir": eng_q}

        # enforce: trend BECOMES the quant DECIDE signal
        edge = self.base_edge_bps * abs(sig)
        try:
            market_data["quant_direction"] = sig
            market_data["quant_confidence"] = abs(sig)
            market_data["signal_edge_bps"] = max(
                float(market_data.get("signal_edge_bps", 0.0) or 0.0), edge)
            market_data["quant_data_quality"] = 1.0
            market_data["primary_strategy"] = "trend_following"
            agent_signals["quant_direction"] = sig
            agent_signals["quant_confidence"] = abs(sig)
            agent_signals["signal_edge_bps"] = market_data["signal_edge_bps"]
            agent_signals["primary_strategy"] = "trend_following"
        except Exception as e:
            logger.warning(f"[TREND-LAYER] {asset} enforce inject failed, engine signal "
                           f"stands: {type(e).__name__}: {e}")
            return None
        logger.warning(f"[TREND-LAYER][ENFORCE] {asset}: quant_dir {eng_q:+.2f} -> "
                       f"trend {sig:+.2f} edge={edge:.0f}bps")
        return {"mode": "enforce", "trend_signal": sig, "edge_bps": edge,
                "replaced_quant_dir": eng_q}


_singleton: Optional[TrendDecisionLayer] = None


def get_trend_decision_layer(mode: str = "off") -> TrendDecisionLayer:
    global _singleton
    if _singleton is None:
        _singleton = TrendDecisionLayer(mode=mode)
    else:
        _singleton.set_mode(mode)
    return _singleton

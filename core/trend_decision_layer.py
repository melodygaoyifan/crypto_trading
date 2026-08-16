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

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

from strategies.trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)

_VALID_MODES = ("off", "shadow", "enforce")
_VALID_GATE_MODES = ("off", "shadow", "enforce")

# [P198 2026-08-07] Regimes in which the trend signal is gated (zeroed when the
# gate is in enforce; logged-only in shadow). Chosen from the 2026-06-14 ->
# 2026-08-06 live window: trend's next-4H-bar signed return by GMM regime was
#   WEAK_CONSOLIDATION  n=364  hit 45.3%  -9.1 bps/tick  (all 3 assets negative)
#   NEUTRAL_DRIFT       n=11   hit 27.3%  -68.9 bps/tick
#   QUIET_ACCUMULATION  n=663  hit 52.5%  +2.6 bps/tick  (NOT gated — mildly
#     positive, and gating it too would block 93% of all ticks = system off)
# That window is IN-SAMPLE (it is the window that showed the loss), which is
# exactly why the gate DEFAULTS TO SHADOW: it logs what it would have done to
# data/trend_regime_shadow.jsonl, and promotion to enforce is a config flip
# (`trend_regime_gate: "enforce"`) to be made only on FORWARD evidence — the
# same discipline that gates trend enforce itself (see module docstring).
_DEFAULT_GATE_REGIMES = ("WEAK_CONSOLIDATION", "NEUTRAL_DRIFT")


class TrendDecisionLayer:
    def __init__(self, mode: str = "off", base_edge_bps: float = 40.0,
                 min_abs_signal: float = 0.30,
                 regime_gate_mode: str = "shadow",
                 gate_regimes=_DEFAULT_GATE_REGIMES) -> None:
        self._strat = TrendFollowingStrategy()
        self.mode = mode if mode in _VALID_MODES else "off"
        # signal_edge_bps injected for the gate = base_edge_bps * |signal|. base 40
        # clears the ~5-14bps live threshold at moderate-or-stronger conviction.
        self.base_edge_bps = float(base_edge_bps)
        # |signal| below this -> treat as no trend (flat); avoids trading weak chop.
        self.min_abs_signal = float(min_abs_signal)
        self._closes: Dict[str, List[float]] = {}
        self._closes_cached_at: Dict[str, float] = {}  # [P265] staleness bound
        # [P198] regime gate: off | shadow (log only, DEFAULT) | enforce (zero
        # the trend signal in gated regimes -> the sleeve flattens there).
        self.regime_gate_mode = (regime_gate_mode
                                 if regime_gate_mode in _VALID_GATE_MODES
                                 else "shadow")
        self.gate_regimes = tuple(gate_regimes or ())
        self._shadow_log_fail_logged = False

    def set_mode(self, mode: str) -> None:
        self.mode = mode if mode in _VALID_MODES else "off"

    def set_regime_gate_mode(self, mode: str) -> None:
        if mode in _VALID_GATE_MODES:
            self.regime_gate_mode = mode

    # ----- [P198] regime-gate shadow measurement ---------------------------

    def _shadow_log_path(self) -> str:
        return os.path.join(os.environ.get("HMATS_DATA_DIR", "data"),
                            "trend_regime_shadow.jsonl")

    def _log_regime_shadow(self, asset: str, sig: float, regime: str,
                           gated: bool) -> None:
        """Append one measurement point. Best-effort, never breaks the tick —
        but a write failure is WARNED (once), not debug-swallowed (P160)."""
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "asset": asset,
            "trend_sig": round(float(sig), 4),
            "regime": regime,
            "gated": bool(gated),
            "gate_mode": self.regime_gate_mode,
        }
        try:
            p = self._shadow_log_path()
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            self._shadow_log_fail_logged = False
        except Exception as e:
            if not self._shadow_log_fail_logged:
                logger.warning(
                    f"[TREND-REGIME-GATE] shadow log write failed "
                    f"({type(e).__name__}: {e}) — trend_regime_shadow.jsonl is "
                    f"NOT accumulating; the forward evidence for the gate "
                    f"decision is being lost")
                self._shadow_log_fail_logged = True

    # [P265] Staleness bound on the cached closes. The sole live decision
    # signal used to be computed from whatever series was cached on SOME
    # earlier tick — no age bound, no log — so under `enforce` a TA-cache
    # outage kept asserting a frozen trend whose 40bps constant cleared the
    # gate indefinitely (P156's rule, unapplied to the book's only driver).
    # 8h = two 4H ticks: one missed refresh is tolerated, two are not.
    CLOSES_MAX_AGE_S = 8 * 3600.0

    def cache(self, asset: str, ohlcv_df: Any) -> None:
        if self.mode == "off" or ohlcv_df is None:
            return
        try:
            if len(ohlcv_df) >= self._strat.min_history():
                self._closes[asset] = [float(x) for x in ohlcv_df["close"].tolist()]
                self._closes_cached_at[asset] = time.time()
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
        _cage = time.time() - self._closes_cached_at.get(asset, 0.0)
        if _cage > self.CLOSES_MAX_AGE_S:
            logger.warning(
                f"[TREND-LAYER] {asset}: cached closes are {_cage/3600:.1f}h "
                f"old (> {self.CLOSES_MAX_AGE_S/3600:.0f}h) — refusing to "
                f"assert a trend from a frozen series; engine signal stands "
                f"(P265)")
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

        # [P198] Regime gate. Log the RAW signal + regime + would-gate verdict
        # every tick (that JSONL is the forward evidence for promoting the
        # gate); only in gate enforce does it change the signal, by zeroing it
        # in gated regimes — flat, so the sleeve flattens rather than trades
        # chop. regime_state is written by the regime stage earlier in the tick.
        if self.regime_gate_mode != "off":
            _regime = str(market_data.get("regime_state", "UNKNOWN") or "UNKNOWN")
            _gated = _regime in self.gate_regimes
            self._log_regime_shadow(asset, sig, _regime, _gated)
            if self.regime_gate_mode == "enforce" and _gated and sig != 0.0:
                logger.warning(
                    f"[TREND-REGIME-GATE][ENFORCE] {asset}: trend sig "
                    f"{sig:+.2f} -> 0.00 (regime={_regime})")
                sig = 0.0

        if self.mode == "shadow":
            logger.info(f"[TREND-LAYER][SHADOW] {asset}: trend sig={sig:+.2f} "
                        f"(pos={res['target_position']:+.2f}) vs quant_dir={eng_q:+.2f}")
            return {"mode": "shadow", "trend_signal": sig, "quant_dir": eng_q}

        # enforce: trend BECOMES the quant DECIDE signal
        edge = self.base_edge_bps * abs(sig)
        try:
            market_data["quant_direction"] = sig
            market_data["quant_confidence"] = abs(sig)
            # [P287] The edge claim is the TREND's own edge, full stop. This
            # used to be max(engine_edge, trend_edge): whenever the pipeline's
            # 65*|quant_dir| exceeded 40*|trend_sig| (e.g. engine dir +/-0.9 vs
            # trend +/-0.30), the alpha gate was judged on an edge computed for
            # a signal this inject just THREW AWAY — the engine's direction may
            # even be opposite. That both loosened the live gate and broke the
            # recorded calibration (P231/P237: live alpha = 40x|trend|x0.75)
            # that the September tripwire arithmetic keys on. Replacing the
            # signal replaces its edge claim — this is a live TIGHTENING.
            market_data["signal_edge_bps"] = edge
            market_data["quant_data_quality"] = 1.0
            market_data["primary_strategy"] = "trend_following"
            agent_signals["quant_direction"] = sig
            agent_signals["quant_confidence"] = abs(sig)
            agent_signals["signal_edge_bps"] = market_data["signal_edge_bps"]
            # [P265] The dq stamp must reach the dict FUSION reads. Fusion's
            # P170 guard reads agent_signals["quant_data_quality"] (copied
            # from market_data BEFORE this inject runs), and the pipeline
            # leaves it 0.0 on every early-return path — so on any
            # pipeline-degraded tick under enforce, the freshly computed,
            # independently valid trend signal was silently EXCLUDED from
            # fusion and the [P126] warning misattributed it as "strategy
            # selector failed". The market_data write above was a dead write
            # whose entire purpose was to prevent exactly this.
            agent_signals["quant_data_quality"] = 1.0
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


def get_trend_decision_layer(mode: str = "off",
                             regime_gate_mode: Optional[str] = None) -> TrendDecisionLayer:
    global _singleton
    if _singleton is None:
        _singleton = TrendDecisionLayer(
            mode=mode,
            regime_gate_mode=(regime_gate_mode or "shadow"))
    else:
        _singleton.set_mode(mode)
        if regime_gate_mode is not None:
            _singleton.set_regime_gate_mode(regime_gate_mode)
    return _singleton

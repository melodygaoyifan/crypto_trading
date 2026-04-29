"""
HMATS v5.1 Phase 8 - Liquidation Cascade Sleeve Expansion (SHADOW MODE)
========================================================================

Two cascade strategies wrapping existing Coinglass liquidation primitives:
  - CascadeAnticipationStrategy : long-side asymmetry, anticipate next leg
  - StopHuntDefenseStrategy     : recommend stop-zone-avoidance offsets

NOT greenfield — LiquidationCascadeHunter (kraken_quant_agent strategy_id=1)
already runs in production. Phase 8 ADDS:
  - shadow-mode anticipation timing (when LCH fires, we predict whether the
    cascade has more legs left rather than re-fading the same exhaustion)
  - defensive stop-hunt avoidance (advisory only — Phase 10 wires into
    place_stop_loss after 30d shadow validates)

Iron Laws honored:
  1. obs_dim=126 untouched (consumes existing market_data fields).
  4. Fail-closed: missing field / NaN -> Signal.NEUTRAL.
  6. Quant strategies untouched (these are NEW shadow observers).
  7. Shadow >=30d before any promotion.

Consumed market_data fields (all already populated):
  - liquidation_imbalance      : float in [-1, +1], +1 = all-long-liquidating
  - long_liquidations_24h      : float USD
  - short_liquidations_24h     : float USD
  - total_liquidations_24h     : float USD
  - largest_single_liquidation : float USD
  - current_price              : float
  - price_change_4h_pct        : float
  - price_change_1h_pct        : float
  - vpin_anomaly               : 'HIGH' | 'NORMAL' | 'LOW'
  - spread_bps                 : float

Research basis:
  In Sep 2025, $1.6B of $1.7B daily liquidations were LONG-side. Short-bias
  HMATS exploits this asymmetry: when liquidation_imbalance > +0.6 AND
  price_change_4h is moderately negative (NOT yet capitulated), expect
  another leg. Capitulation (price already -10%+) means the cascade has
  exhausted; no more anticipation alpha.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared types — re-import shape from microstructure module to keep ledger
# schema stable across phases. Local-define to avoid cross-phase coupling.
# ---------------------------------------------------------------------------

class ShadowSignal(NamedTuple):
    strategy_name: str
    asset: str
    direction: float          # -1.0 / 0.0 / +1.0
    confidence: float         # [0, 1]
    reason: str
    diagnostics: Dict[str, Any]


def NEUTRAL(strategy_name: str, asset: str, reason: str = "insufficient_data") -> ShadowSignal:
    return ShadowSignal(
        strategy_name=strategy_name,
        asset=asset,
        direction=0.0,
        confidence=0.0,
        reason=reason,
        diagnostics={},
    )


def _get_float(d: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    v = d.get(key)
    if v is None:
        return default
    try:
        f = float(v)
        if f != f:  # NaN
            return default
        return f
    except (TypeError, ValueError):  # noqa: silent-swallow
        # Per-field type coercion helper; default IS the documented fallback.
        return default


# ---------------------------------------------------------------------------
# 1. Cascade anticipation — short-bias asymmetry
# ---------------------------------------------------------------------------

@dataclass
class CascadeAnticipationStrategy:
    """Anticipate the NEXT leg of a long-liquidation cascade.

    Logic (short-bias):
        - Cascade in progress: liquidation_imbalance > +0.6 (long-dominant)
        - Cascade NOT yet exhausted: price_change_4h_pct in (-cap_floor, -trigger_drop)
          i.e. there's been a real drop (-trigger_drop or worse), but not yet
          capitulation (more negative than -cap_floor means the move is done).
        - Total volume large: total_liquidations_24h > min_total_usd
        - Optional confirm: vpin_anomaly == HIGH (toxic flow + cascade align)

    Output: SHORT signal with confidence scaling on (a) imbalance excess,
    (b) how much room there is between current drawdown and cap_floor.

    Symmetric LONG branch (short squeeze): liquidation_imbalance < -0.6 AND
    price has had a meaningful UP move not yet capitulated.

    Confidence cap is conservative — cascade timing is noisy.
    """
    imbalance_threshold: float = 0.60
    trigger_drop_pct: float = 0.015          # need at least -1.5% in 4h to confirm cascade started
    cap_floor_pct: float = 0.10              # price already -10%+ in 4h => exhausted, NEUTRAL
    min_total_usd: float = 1.0e8             # min $100M cumulative 24h liquidations to consider real
    confidence_cap: float = 0.60
    require_vpin_confirm: bool = False       # default off; vpin not always 'computed'

    def evaluate(self, asset: str, market_data: Dict[str, Any]) -> ShadowSignal:
        liq_imb = _get_float(market_data, "liquidation_imbalance")
        total_liq = _get_float(market_data, "total_liquidations_24h", 0.0)
        ret_4h = _get_float(market_data, "price_change_4h_pct")
        if liq_imb is None or ret_4h is None:
            return NEUTRAL("cascade_anticipation", asset, "missing_liq_imb_or_ret_4h")
        if total_liq is None or total_liq < self.min_total_usd:
            return NEUTRAL("cascade_anticipation", asset, f"total_liq_too_small(${total_liq or 0:,.0f})")

        # Direction selection
        if liq_imb >= self.imbalance_threshold:
            # Long-side cascade: confirm price has dropped but not capitulated
            if ret_4h > -self.trigger_drop_pct:
                return NEUTRAL("cascade_anticipation", asset,
                               f"long_liq_dominant_but_no_drop_yet(ret4h={ret_4h*100:+.2f}%)")
            if ret_4h < -self.cap_floor_pct:
                return NEUTRAL("cascade_anticipation", asset,
                               f"likely_capitulation(ret4h={ret_4h*100:+.2f}%)")
            direction = -1.0
            side = "long_liq"
        elif liq_imb <= -self.imbalance_threshold:
            # Short squeeze: confirm price has risen but not capitulated up
            if ret_4h < self.trigger_drop_pct:
                return NEUTRAL("cascade_anticipation", asset,
                               f"short_liq_dominant_but_no_rally_yet(ret4h={ret_4h*100:+.2f}%)")
            if ret_4h > self.cap_floor_pct:
                return NEUTRAL("cascade_anticipation", asset,
                               f"likely_short_capitulation(ret4h={ret_4h*100:+.2f}%)")
            direction = 1.0
            side = "short_liq"
        else:
            return NEUTRAL("cascade_anticipation", asset,
                           f"imbalance_below_threshold({liq_imb:+.2f})")

        if self.require_vpin_confirm:
            if market_data.get("vpin_anomaly") != "HIGH":
                return NEUTRAL("cascade_anticipation", asset,
                               f"awaiting_vpin_confirm(anomaly={market_data.get('vpin_anomaly')})")

        # Confidence: imbalance excess (saturate at 1.0) + cascade-room headroom
        imb_excess = (abs(liq_imb) - self.imbalance_threshold) / max(0.05, 1.0 - self.imbalance_threshold)
        imb_excess = max(0.0, min(1.0, imb_excess))
        # Headroom = how far current drawdown is from cap_floor (more room = stronger)
        room = (self.cap_floor_pct - abs(ret_4h)) / max(0.01, self.cap_floor_pct - self.trigger_drop_pct)
        room = max(0.0, min(1.0, room))
        confidence = min(self.confidence_cap, 0.30 + 0.40 * imb_excess + 0.20 * room)

        return ShadowSignal(
            strategy_name="cascade_anticipation",
            asset=asset,
            direction=direction,
            confidence=confidence,
            reason=f"{side}_cascade_anticipation(imb={liq_imb:+.2f},ret4h={ret_4h*100:+.2f}%)",
            diagnostics={
                "liquidation_imbalance": liq_imb,
                "total_liquidations_24h": total_liq,
                "price_change_4h_pct": ret_4h,
                "vpin_anomaly": market_data.get("vpin_anomaly"),
                "side": side,
            },
        )


# ---------------------------------------------------------------------------
# 2. Stop-hunt defense — advisory stop-zone-avoidance offset
# ---------------------------------------------------------------------------

@dataclass
class StopHuntDefenseStrategy:
    """Recommend stop-placement offset when stop-hunting risk is elevated.

    This is NOT a directional strategy in the conventional sense — it produces
    advisory output: how much to widen a stop's distance-from-price to dodge
    cluster zones. We encode the recommendation as a non-directional signal
    (direction=0) with the actual recommendation in diagnostics.

    Risk score components (each in [0,1]):
        a) cascade_in_progress  : abs(liquidation_imbalance) above threshold
        b) toxic_flow_on        : vpin_anomaly == HIGH (computed only)
        c) wide_spread          : spread_bps > min_spread_for_alarm
        d) recent_extreme_move  : abs(price_change_1h_pct) > 1%

    Recommended offset (bps) scales with composite_risk:
        recommended_stop_offset_bps = base_offset + risk * scale_offset

    Phase 10 will wire this advisory into execution_manager.place_stop_loss
    via a Phase 10-specific callback. For Phase 8 it's pure observation —
    we log what we WOULD recommend and let operator validate offline.

    Iron Law 7: zero side-effect on production stop placement.
    """
    imbalance_threshold: float = 0.50
    min_spread_for_alarm: float = 8.0
    extreme_move_1h_pct: float = 0.010
    base_offset_bps: float = 25.0           # baseline buffer over current price level
    scale_offset_bps: float = 75.0          # max additional offset at composite_risk=1.0

    def evaluate(self, asset: str, market_data: Dict[str, Any]) -> ShadowSignal:
        liq_imb = _get_float(market_data, "liquidation_imbalance", 0.0)
        spread_bps = _get_float(market_data, "spread_bps")
        ret_1h = _get_float(market_data, "price_change_1h_pct", 0.0)
        vpin_anom = market_data.get("vpin_anomaly")
        vpin_source = market_data.get("vpin_source")

        if spread_bps is None:
            return NEUTRAL("stop_hunt_defense", asset, "missing_spread")

        # Each component in [0,1]
        cascade_score = max(0.0, min(1.0,
            (abs(liq_imb) - self.imbalance_threshold) / max(0.05, 1.0 - self.imbalance_threshold)
        )) if abs(liq_imb) >= self.imbalance_threshold else 0.0

        toxic_score = 1.0 if (vpin_anom == "HIGH" and vpin_source == "computed") else 0.0

        spread_score = max(0.0, min(1.0,
            (spread_bps - self.min_spread_for_alarm) / max(2.0, self.min_spread_for_alarm)
        )) if spread_bps >= self.min_spread_for_alarm else 0.0

        move_score = max(0.0, min(1.0,
            (abs(ret_1h) - self.extreme_move_1h_pct) / max(0.005, self.extreme_move_1h_pct)
        )) if abs(ret_1h) >= self.extreme_move_1h_pct else 0.0

        # Equal-weighted composite
        composite_risk = (cascade_score + toxic_score + spread_score + move_score) / 4.0

        # Always emit (defensive advisory should be visible per tick) but
        # confidence-zero when composite_risk is negligible.
        recommended_offset = self.base_offset_bps + composite_risk * self.scale_offset_bps

        if composite_risk < 0.10:
            # Quiet regime — recommend baseline only, low confidence
            return ShadowSignal(
                strategy_name="stop_hunt_defense",
                asset=asset,
                direction=0.0,
                confidence=0.10,
                reason=f"quiet(composite={composite_risk:.2f})",
                diagnostics={
                    "composite_risk": composite_risk,
                    "recommended_stop_offset_bps": recommended_offset,
                    "cascade_score": cascade_score,
                    "toxic_score": toxic_score,
                    "spread_score": spread_score,
                    "move_score": move_score,
                    "advisory_only": True,
                },
            )

        return ShadowSignal(
            strategy_name="stop_hunt_defense",
            asset=asset,
            direction=0.0,  # non-directional advisory
            confidence=min(0.85, 0.20 + 0.65 * composite_risk),
            reason=f"elevated_stop_hunt_risk(composite={composite_risk:.2f})",
            diagnostics={
                "composite_risk": composite_risk,
                "recommended_stop_offset_bps": recommended_offset,
                "cascade_score": cascade_score,
                "toxic_score": toxic_score,
                "spread_score": spread_score,
                "move_score": move_score,
                "liquidation_imbalance": liq_imb,
                "spread_bps": spread_bps,
                "price_change_1h_pct": ret_1h,
                "vpin_anomaly": vpin_anom,
                "advisory_only": True,
            },
        )


# ---------------------------------------------------------------------------
# Public registry — Phase 8 cascade strategies
# ---------------------------------------------------------------------------

def build_phase8_cascade_strategies():
    """Return the 2 Phase-8 cascade strategies as a tuple.

    Used by defense/strategy_shadow_v5_1.py to register a second harness
    instance (with prefix='cascade') alongside the Phase 4 microstructure
    harness. Construction is parameterless and side-effect-free.
    """
    return (
        CascadeAnticipationStrategy(),
        StopHuntDefenseStrategy(),
    )

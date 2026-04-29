"""
HMATS v5.1 Phase 4 - Microstructure Layer Expansion (SHADOW MODE)
==================================================================

Three microstructure strategies wrapping existing primitives:
  - OrderFlowImbalanceStrategy : OFI z-score sustained > threshold -> directional
  - VPINSpikeStrategy          : VPIN > 0.7 -> toxic flow -> contrarian reversal
  - KyleLambdaStrategy         : implied price-impact spike -> liquidity crisis -> fade

Iron Laws honored:
  1. obs_dim=126 untouched (consumes existing market_data fields, no new features).
  4. Fail-closed: insufficient data / missing field -> Signal.NEUTRAL.
  6. Quant strategies untouched (these are NEW, ADVISE-track only after promotion).
  7. Shadow >=30d before any promotion (StrategyShadow harness, no fusion wire).

Consumed market_data fields (all already populated by data_mgmt/market_data_pipeline.py):
  - vpin                : float in [0,1], 0.35 synthetic default
  - vpin_source         : 'computed' | 'synthetic'
  - vpin_vs_median      : float ~ 1.0 normal, >1.30 HIGH, <0.70 LOW
  - vpin_anomaly        : 'HIGH' | 'NORMAL' | 'LOW'
  - ofi_zscore          : float in [-5,+5], rolling z-score of order_book_imbalance
  - order_book_imbalance: float in [-1,+1]
  - spread_bps          : float
  - current_price       : float

Output: ShadowSignal (NamedTuple) — direction in {-1,0,+1}, confidence in [0,1],
        plus diagnostics. Signals are written to a JSONL ledger by
        defense/strategy_shadow_v5_1.py; they NEVER enter fusion until Phase 10
        promotion gate clears.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal type
# ---------------------------------------------------------------------------

class ShadowSignal(NamedTuple):
    strategy_name: str
    asset: str
    direction: float          # -1.0 / 0.0 / +1.0
    confidence: float         # [0, 1]
    reason: str               # human-readable trigger
    diagnostics: Dict[str, Any]


def NEUTRAL(strategy_name: str, asset: str, reason: str = "insufficient_data") -> ShadowSignal:
    """Iron Law 4 fail-closed: missing/invalid input -> NEUTRAL with reason."""
    return ShadowSignal(
        strategy_name=strategy_name,
        asset=asset,
        direction=0.0,
        confidence=0.0,
        reason=reason,
        diagnostics={},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_float(d: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    """Safe float extract. Returns default if missing or non-numeric."""
    v = d.get(key)
    if v is None:
        return default
    try:
        f = float(v)
        if f != f:  # NaN check
            return default
        return f
    except (TypeError, ValueError):  # noqa: silent-swallow
        # Per-field type coercion helper; default IS the documented fallback.
        return default


# ---------------------------------------------------------------------------
# 1. Order Flow Imbalance — sustained directional pressure
# ---------------------------------------------------------------------------

@dataclass
class OrderFlowImbalanceStrategy:
    """OFI sustained direction → directional signal.

    Trigger:
        |ofi_zscore| > entry_threshold AND consecutive same-sign streak >= persistence
        AND |order_book_imbalance| > min_imbalance

    Exit (handled by upstream): signal flips or persistence break.

    Sub-hourly IC 0.05-0.10 documented in microstructure literature.
    Honors fail-closed when ofi_zscore field is missing.
    """
    entry_threshold: float = 1.5      # z-score abs threshold
    persistence: int = 2               # consecutive same-sign ticks required
    min_imbalance: float = 0.10        # |raw imbalance| floor
    confidence_cap: float = 0.65       # max confidence (Iron Law 6 — leave room for primary signals)

    def __post_init__(self) -> None:
        # Per-asset z-score sign streak history
        self._streak: Dict[str, int] = {}

    def evaluate(self, asset: str, market_data: Dict[str, Any]) -> ShadowSignal:
        ofi_z = _get_float(market_data, "ofi_zscore")
        ob_imb = _get_float(market_data, "order_book_imbalance",
                            _get_float(market_data, "orderbook_imbalance"))
        if ofi_z is None or ob_imb is None:
            return NEUTRAL("ofi", asset, "missing_ofi_or_imbalance_field")

        # Update streak
        sign = 1 if ofi_z > 0 else (-1 if ofi_z < 0 else 0)
        prev = self._streak.get(asset, 0)
        if sign == 0:
            self._streak[asset] = 0
        elif (prev > 0 and sign > 0) or (prev < 0 and sign < 0):
            self._streak[asset] = prev + sign
        else:
            self._streak[asset] = sign

        streak = self._streak[asset]

        if abs(ofi_z) < self.entry_threshold:
            return NEUTRAL("ofi", asset, f"z_below_threshold({ofi_z:+.2f})")
        if abs(streak) < self.persistence:
            return NEUTRAL("ofi", asset, f"streak_too_short({streak})")
        if abs(ob_imb) < self.min_imbalance:
            return NEUTRAL("ofi", asset, f"raw_imbalance_too_small({ob_imb:+.3f})")

        direction = 1.0 if streak > 0 else -1.0
        # Confidence: scale z-score above threshold up to cap
        excess_z = (abs(ofi_z) - self.entry_threshold) / max(0.5, 5.0 - self.entry_threshold)
        confidence = min(self.confidence_cap, 0.30 + 0.70 * max(0.0, min(1.0, excess_z)))

        return ShadowSignal(
            strategy_name="ofi",
            asset=asset,
            direction=direction,
            confidence=confidence,
            reason=f"ofi_sustained_{streak:+d}",
            diagnostics={"ofi_zscore": ofi_z, "ob_imbalance": ob_imb, "streak": streak},
        )


# ---------------------------------------------------------------------------
# 2. VPIN spike — toxic flow contrarian reversal
# ---------------------------------------------------------------------------

@dataclass
class VPINSpikeStrategy:
    """VPIN > spike threshold + supporting context → contrarian reversal signal.

    Logic:
        Toxic flow (VPIN spike) historically precedes mean-reversion as
        informed-trader pressure exhausts. We fade in the OPPOSITE direction
        of the recent OFI z-score (i.e. if OFI was net-buy and VPIN spiked,
        the contrarian view is SHORT).

    Trigger:
        vpin > spike_threshold OR vpin_anomaly == 'HIGH'
        AND vpin_source == 'computed' (synthetic 0.35 default → useless)
        AND |ofi_zscore| > min_ofi_z (need a direction to fade)

    Confidence is dampened heavily — VPIN signals are rare and noisy.
    """
    spike_threshold: float = 0.70
    min_ofi_z: float = 1.0
    confidence_cap: float = 0.55      # lower than OFI; VPIN historically noisier

    def evaluate(self, asset: str, market_data: Dict[str, Any]) -> ShadowSignal:
        vpin = _get_float(market_data, "vpin")
        vpin_source = market_data.get("vpin_source")
        vpin_anomaly = market_data.get("vpin_anomaly")
        ofi_z = _get_float(market_data, "ofi_zscore")

        if vpin is None or ofi_z is None:
            return NEUTRAL("vpin_spike", asset, "missing_vpin_or_ofi")
        if vpin_source != "computed":
            return NEUTRAL("vpin_spike", asset, f"vpin_source_unreliable({vpin_source})")

        spike = (vpin > self.spike_threshold) or (vpin_anomaly == "HIGH")
        if not spike:
            return NEUTRAL("vpin_spike", asset, f"no_spike(vpin={vpin:.2f})")
        if abs(ofi_z) < self.min_ofi_z:
            return NEUTRAL("vpin_spike", asset, f"ofi_too_weak({ofi_z:+.2f})_to_fade")

        # Contrarian: opposite sign of OFI
        direction = -1.0 if ofi_z > 0 else 1.0
        # Confidence scales with how far above threshold + how strong OFI is
        spike_excess = max(0.0, (vpin - self.spike_threshold) / max(0.01, 1.0 - self.spike_threshold))
        ofi_strength = min(1.0, (abs(ofi_z) - self.min_ofi_z) / max(0.5, 5.0 - self.min_ofi_z))
        confidence = min(self.confidence_cap, 0.30 + 0.50 * spike_excess + 0.20 * ofi_strength)

        return ShadowSignal(
            strategy_name="vpin_spike",
            asset=asset,
            direction=direction,
            confidence=confidence,
            reason=f"vpin_spike_fade({vpin:.2f},ofi_z={ofi_z:+.2f})",
            diagnostics={
                "vpin": vpin,
                "vpin_source": vpin_source,
                "vpin_anomaly": vpin_anomaly,
                "ofi_zscore": ofi_z,
            },
        )


# ---------------------------------------------------------------------------
# 3. Kyle's lambda — liquidity-crisis fade
# ---------------------------------------------------------------------------

@dataclass
class KyleLambdaStrategy:
    """Implied price-impact spike → liquidity crisis → fade extremes.

    Kyle's lambda ≈ Δprice / signed_volume. Without trade-level signed flow,
    we proxy with: lambda_proxy = |return_4h| / max(|order_book_imbalance|, eps).
    A spike in this ratio signals widening price impact per unit imbalance —
    typically pre-reversal when one side runs out of liquidity.

    Trigger:
        lambda_proxy > rolling_pX percentile over self._lookback bars
        AND spread_bps > min_spread (confirms liquidity stress)
        AND |return_4h| > min_move (avoid div-by-near-zero noise)

    Direction: FADE the recent move (return sign).
    """
    lookback: int = 30                 # rolling history per asset
    pct_threshold: float = 0.85        # 85th percentile = spike
    min_spread_bps: float = 5.0
    min_abs_return_4h: float = 0.005   # 0.5%
    eps: float = 1e-4
    confidence_cap: float = 0.50

    def __post_init__(self) -> None:
        self._lambda_history: Dict[str, deque] = {}
        self._last_price: Dict[str, float] = {}

    def evaluate(self, asset: str, market_data: Dict[str, Any]) -> ShadowSignal:
        price = _get_float(market_data, "current_price")
        ob_imb = _get_float(market_data, "order_book_imbalance",
                            _get_float(market_data, "orderbook_imbalance"))
        spread_bps = _get_float(market_data, "spread_bps")
        if price is None or ob_imb is None or spread_bps is None:
            return NEUTRAL("kyle_lambda", asset, "missing_price_imb_or_spread")

        # Compute return since last call
        prev_price = self._last_price.get(asset)
        self._last_price[asset] = price
        if prev_price is None or prev_price <= 0:
            return NEUTRAL("kyle_lambda", asset, "no_prev_price")
        ret = (price - prev_price) / prev_price

        # Lambda proxy
        denom = max(abs(ob_imb), self.eps)
        lambda_proxy = abs(ret) / denom

        # Update rolling history
        if asset not in self._lambda_history:
            self._lambda_history[asset] = deque(maxlen=self.lookback)
        hist = self._lambda_history[asset]
        hist.append(lambda_proxy)

        if len(hist) < max(8, int(self.lookback * 0.3)):
            return NEUTRAL("kyle_lambda", asset, f"history_warmup({len(hist)}/{self.lookback})")

        # Percentile check (no numpy dependency — sort)
        sorted_h = sorted(hist)
        idx = int(self.pct_threshold * (len(sorted_h) - 1))
        threshold = sorted_h[idx]

        if lambda_proxy <= threshold:
            return NEUTRAL("kyle_lambda", asset, f"below_p{int(self.pct_threshold*100)}")
        if spread_bps < self.min_spread_bps:
            return NEUTRAL("kyle_lambda", asset, f"spread_too_tight({spread_bps:.1f}bps)")
        if abs(ret) < self.min_abs_return_4h:
            return NEUTRAL("kyle_lambda", asset, f"move_too_small({ret*100:+.2f}%)")

        # Fade the recent move
        direction = -1.0 if ret > 0 else 1.0

        # Confidence: scale spike magnitude over threshold + spread stress
        spike_mag = min(1.0, (lambda_proxy / threshold - 1.0) / 0.5)  # 50% over = saturation
        spread_stress = min(1.0, (spread_bps - self.min_spread_bps) / max(5.0, self.min_spread_bps))
        confidence = min(self.confidence_cap, 0.25 + 0.40 * spike_mag + 0.15 * spread_stress)

        return ShadowSignal(
            strategy_name="kyle_lambda",
            asset=asset,
            direction=direction,
            confidence=confidence,
            reason=f"kyle_lambda_spike(p{int(self.pct_threshold*100)})_fade",
            diagnostics={
                "lambda_proxy": lambda_proxy,
                "p_threshold": threshold,
                "ret_4h": ret,
                "spread_bps": spread_bps,
                "ob_imbalance": ob_imb,
                "history_n": len(hist),
            },
        )


# ---------------------------------------------------------------------------
# Public registry — the 3 strategies Phase 4 ships in shadow mode
# ---------------------------------------------------------------------------

def build_phase4_microstructure_strategies():
    """Return the 3 Phase-4 microstructure strategies as a tuple.

    Used by defense/strategy_shadow_v5_1.py to register shadow observations.
    Construction is parameterless and side-effect-free — safe to call from
    main.py at startup. Per-strategy state is held in the returned instances.
    """
    return (
        OrderFlowImbalanceStrategy(),
        VPINSpikeStrategy(),
        KyleLambdaStrategy(),
    )

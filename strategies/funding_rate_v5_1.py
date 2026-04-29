"""
HMATS v5.1 Phase 3 - Funding-Rate Strategy Layer (SHADOW MODE)
================================================================

Three funding-rate strategies wrapping existing market_data['funding_rate']
populated by data_mgmt/feeds/kraken_futures_feed.py (per-hour cadence on
Kraken PF_*; 8h prediction also available). Strategies are venue-agnostic —
they consume `funding_rate` and `funding_rate_8h` from market_data, not
from the venue adapter directly. Phase 2 swap from Kraken Futures to
Coinbase derivatives changes the data SOURCE but no strategy code change.

Iron Laws honored:
  1. obs_dim=126 untouched (no new features).
  4. Fail-closed: missing funding fields -> Signal.NEUTRAL.
  6. Quant strategies untouched.
  7. Shadow >=30d before promotion via StrategyShadow harness.

Consumed market_data fields (already populated):
  - funding_rate          : per-period funding (Kraken hourly; per V14
                            Coinbase post-Phase-2 hourly or 8h depending
                            on contract)
  - funding_rate_8h       : per-8h normalization (kraken_futures_feed
                            converts hourly -> 8h via FUNDING_RATE_MULTIPLIER)
  - funding_rate_prediction_8h
  - funding_sustained_hours (computed by upstream — hours of same-sign streak)

Research basis (BIS 2024 retail data):
  - Funding-rate-extreme strategy: Sharpe 1.8 documented when |funding_8h|
    > 0.03% sustained 24h+
  - Funding mean-reversion: Z-score(funding) > 2 reverses next period
    with high probability
  - Post-ETF regime (CFB 2025): post-Jan 2024 ETF flows changed BTC funding
    dynamics; funding capped by cash-carry arbs. Detect regime + adjust
    thresholds.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared types
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
        return default


# ---------------------------------------------------------------------------
# 1. Funding-rate extreme — directional momentum at sustained extremes
# ---------------------------------------------------------------------------

@dataclass
class FundingRateExtremeStrategy:
    """Sustained funding extreme signals one-sided crowding → SHORT the
    pay side / LONG the receive side.

    Logic:
        - funding_8h > +entry_threshold AND sustained_hours >= 24 → SHORT
          (longs are crowded paying funding; reversal expected)
        - funding_8h < -entry_threshold AND sustained_hours >= 24 → LONG
          (shorts are crowded paying; reversal expected)

    Scaling: confidence ~ |funding_8h| / extreme_threshold, capped.
    Per BIS 2024 retail: documented Sharpe 1.8 in this regime.
    """
    entry_threshold: float = 0.0003       # 0.03% per 8h ≈ 32% APR (well above noise)
    extreme_threshold: float = 0.0005     # 0.05% per 8h ≈ 54% APR (saturation)
    min_sustained_hours: float = 24.0
    confidence_cap: float = 0.65

    def evaluate(self, asset: str, market_data: Dict[str, Any]) -> ShadowSignal:
        funding_8h = _get_float(market_data, "funding_rate_8h")
        if funding_8h is None:
            # Fall back: derive from per-period funding_rate × 8 if hourly
            hourly = _get_float(market_data, "funding_rate_hourly")
            if hourly is None:
                hourly = _get_float(market_data, "funding_rate")
            if hourly is None:
                return NEUTRAL("funding_extreme", asset, "missing_funding_rate")
            funding_8h = hourly * 8.0  # crude proxy

        sustained = _get_float(market_data, "funding_sustained_hours", 0.0) or 0.0
        if sustained < self.min_sustained_hours:
            return NEUTRAL("funding_extreme", asset,
                           f"not_sustained({sustained:.1f}h < {self.min_sustained_hours}h)")

        if funding_8h > self.entry_threshold:
            direction = -1.0
            side = "long_crowded"
        elif funding_8h < -self.entry_threshold:
            direction = 1.0
            side = "short_crowded"
        else:
            return NEUTRAL("funding_extreme", asset,
                           f"funding_8h={funding_8h*100:.4f}%_below_threshold")

        # Confidence scales with how far past entry threshold
        excess_ratio = (abs(funding_8h) - self.entry_threshold) / max(
            1e-6, self.extreme_threshold - self.entry_threshold
        )
        confidence = min(self.confidence_cap, 0.30 + 0.50 * max(0.0, min(1.0, excess_ratio)))

        return ShadowSignal(
            strategy_name="funding_extreme",
            asset=asset,
            direction=direction,
            confidence=confidence,
            reason=f"funding_extreme_{side}(funding_8h={funding_8h*100:+.4f}%,h={sustained:.1f})",
            diagnostics={
                "funding_8h": funding_8h,
                "sustained_hours": sustained,
                "side": side,
            },
        )


# ---------------------------------------------------------------------------
# 2. Funding mean-reversion — Z-score reversion signal
# ---------------------------------------------------------------------------

@dataclass
class FundingRateMeanReversionStrategy:
    """Z-score(funding) > zentry on rolling 14-day history → expect reversion
    in next funding period; fade the current side.

    Logic:
        Maintain rolling 14-day buffer of funding_rate_8h per asset.
        compute z = (current - mean) / stdev.
        if z > +zentry  → SHORT (long pays excessive funding; mean-revert down)
        if z < -zentry  → LONG  (short pays excessive funding; mean-revert up)

    Less reliable than funding_extreme — funding can stay extreme for weeks
    in trending regimes. Conservative confidence cap.
    """
    lookback_days: int = 14
    zentry: float = 2.0
    confidence_cap: float = 0.45
    min_history: int = 12   # need at least 12 obs (×3/day = 4 days) for stable std

    def __post_init__(self) -> None:
        # Per-asset rolling history — assume ~3 funding obs/day for 8h cadence
        self._history: Dict[str, deque] = {}
        # 14 days × 3 obs/day
        self._maxlen = self.lookback_days * 3

    def evaluate(self, asset: str, market_data: Dict[str, Any]) -> ShadowSignal:
        funding_8h = _get_float(market_data, "funding_rate_8h")
        if funding_8h is None:
            return NEUTRAL("funding_mean_reversion", asset, "missing_funding_rate_8h")

        if asset not in self._history:
            self._history[asset] = deque(maxlen=self._maxlen)
        hist = self._history[asset]
        hist.append(funding_8h)

        if len(hist) < self.min_history:
            return NEUTRAL("funding_mean_reversion", asset,
                           f"history_warmup({len(hist)}/{self.min_history})")

        # Z-score against own rolling history (excluding current point would be
        # ideal but with deque we approximate)
        n = len(hist)
        mean = sum(hist) / n
        variance = sum((x - mean) ** 2 for x in hist) / max(1, n - 1)
        stdev = variance ** 0.5
        if stdev <= 1e-9:
            return NEUTRAL("funding_mean_reversion", asset, "zero_stdev")
        z = (funding_8h - mean) / stdev

        if z > self.zentry:
            direction = -1.0
        elif z < -self.zentry:
            direction = 1.0
        else:
            return NEUTRAL("funding_mean_reversion", asset, f"z={z:+.2f}_below_entry")

        excess = (abs(z) - self.zentry) / max(0.5, 5.0 - self.zentry)
        confidence = min(self.confidence_cap, 0.20 + 0.45 * max(0.0, min(1.0, excess)))

        return ShadowSignal(
            strategy_name="funding_mean_reversion",
            asset=asset,
            direction=direction,
            confidence=confidence,
            reason=f"funding_zscore_revert(z={z:+.2f})",
            diagnostics={
                "funding_8h": funding_8h,
                "rolling_mean": mean,
                "rolling_stdev": stdev,
                "zscore": z,
                "history_n": n,
            },
        )


# ---------------------------------------------------------------------------
# 3. Post-ETF regime — adaptive threshold based on regime detection
# ---------------------------------------------------------------------------

@dataclass
class FundingRatePostETFRegimeStrategy:
    """Per CFB 2025: post-Jan 2024 BTC ETF flows changed funding dynamics.

    Cash-and-carry arbs cap funding extremes ~0.04%/8h on BTC.
    Pre-ETF: funding could spike to 0.10%+ in BTC.
    Post-ETF: funding ceiling effectively ~0.04% on BTC.

    Strategy: detect regime via long-window funding ceiling, then trade the
    NORMALIZED extreme (funding_8h / regime_ceiling). When normalized > 0.85,
    fade the crowded side. This is a calibrated version of
    FundingRateExtremeStrategy that auto-adjusts thresholds per regime.

    Less aggressive than FundingRateExtreme — confidence scales with how
    close to the regime ceiling (and how stable that ceiling is).
    """
    regime_window_days: int = 60          # rolling window for ceiling detection
    fire_threshold: float = 0.85          # |funding_8h| / regime_ceiling threshold
    confidence_cap: float = 0.50
    min_regime_obs: int = 30              # minimum obs to trust the ceiling

    def __post_init__(self) -> None:
        self._history: Dict[str, deque] = {}
        # 60d × 3 obs/day
        self._maxlen = self.regime_window_days * 3

    def evaluate(self, asset: str, market_data: Dict[str, Any]) -> ShadowSignal:
        funding_8h = _get_float(market_data, "funding_rate_8h")
        if funding_8h is None:
            return NEUTRAL("funding_post_etf_regime", asset, "missing_funding_rate_8h")

        if asset not in self._history:
            self._history[asset] = deque(maxlen=self._maxlen)
        hist = self._history[asset]
        hist.append(funding_8h)

        if len(hist) < self.min_regime_obs:
            return NEUTRAL("funding_post_etf_regime", asset,
                           f"regime_warmup({len(hist)}/{self.min_regime_obs})")

        # Regime ceiling = 95th percentile of |funding_8h| over window
        abs_hist = sorted(abs(x) for x in hist)
        idx = int(0.95 * (len(abs_hist) - 1))
        regime_ceiling = abs_hist[idx]
        if regime_ceiling <= 1e-9:
            return NEUTRAL("funding_post_etf_regime", asset, "regime_ceiling_zero")

        normalized = funding_8h / regime_ceiling   # signed
        if abs(normalized) < self.fire_threshold:
            return NEUTRAL("funding_post_etf_regime", asset,
                           f"normalized={normalized:+.2f}_below_threshold")

        direction = -1.0 if normalized > 0 else 1.0
        excess = (abs(normalized) - self.fire_threshold) / max(0.05, 1.0 - self.fire_threshold)
        confidence = min(self.confidence_cap, 0.25 + 0.40 * max(0.0, min(1.0, excess)))

        return ShadowSignal(
            strategy_name="funding_post_etf_regime",
            asset=asset,
            direction=direction,
            confidence=confidence,
            reason=f"funding_post_etf(normalized={normalized:+.2f},ceiling={regime_ceiling*100:.4f}%)",
            diagnostics={
                "funding_8h": funding_8h,
                "regime_ceiling_p95": regime_ceiling,
                "normalized": normalized,
                "history_n": len(hist),
            },
        )


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

def build_phase3_funding_strategies():
    """Return the 3 Phase-3 funding-rate strategies as a tuple."""
    return (
        FundingRateExtremeStrategy(),
        FundingRateMeanReversionStrategy(),
        FundingRatePostETFRegimeStrategy(),
    )

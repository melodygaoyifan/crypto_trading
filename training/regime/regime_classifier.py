"""Runtime-friendly regime classifier used by strategic coordinator.

This module intentionally provides a lightweight heuristic implementation that
does not require model artifacts, so runtime wiring remains available in
paper/live mode when heavy training dependencies are absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Supported market regimes for near-online segmentation."""

    SLOW_BEAR = "slow_bear"
    CRASH_CASCADE = "crash_cascade"
    BEAR_RALLY = "bear_rally"
    SIDEWAYS = "sideways"
    BULL_RECOVERY = "bull_recovery"


@dataclass
class RegimeFeatures:
    """Input features for regime classification.

    Includes both training-era names (`return_*`) and runtime aliases
    (`price_return_*`) so callers from different modules can share one type.
    """

    timestamp: datetime
    asset: str = "UNKNOWN"

    # Returns
    return_1h: float = 0.0
    return_4h: float = 0.0
    return_24h: float = 0.0
    price_return_1h: float = 0.0
    price_return_4h: float = 0.0
    price_return_24h: float = 0.0

    # Volatility
    volatility_1h: float = 0.02
    volatility_4h: float = 0.04
    volatility_24h: float = 0.08
    vol_percentile: float = 50.0

    # Sentiment / risk
    fear_index: float = 50.0
    fear_greed_index: float = 50.0
    social_sentiment: float = 0.0
    cross_asset_correlation: float = 0.5

    # Momentum / micro
    momentum_4h: float = 0.0
    momentum_24h: float = 0.0
    rsi_14: float = 50.0
    macd_histogram: float = 0.0
    bb_position: float = 0.5
    volume_ratio: float = 1.0
    cvd: float = 0.0
    funding_rate: float = 0.0
    open_interest_change: float = 0.0
    basis: float = 0.0

    def r1h(self) -> float:
        return self.return_1h if self.return_1h != 0.0 else self.price_return_1h

    def r4h(self) -> float:
        return self.return_4h if self.return_4h != 0.0 else self.price_return_4h

    def r24h(self) -> float:
        return self.return_24h if self.return_24h != 0.0 else self.price_return_24h

    def normalized_fear(self) -> float:
        # Prefer explicit fear_index. If unavailable, map greed index inversely.
        if self.fear_index != 50.0:
            return max(0.0, min(100.0, self.fear_index))
        return max(0.0, min(100.0, 100.0 - self.fear_greed_index))


@dataclass
class RegimeClassification:
    """Classifier output consumed by strategic coordinator."""

    regime: MarketRegime
    confidence: float
    bearish_score: float
    bullish_score: float
    reason: str = ""


@dataclass
class RegimeClassifierConfig:
    """Tunable thresholds for heuristic regime inference."""

    crash_return_24h: float = -0.12
    crash_return_4h: float = -0.05
    slow_bear_return_24h: float = -0.05
    bear_rally_bounce_4h: float = 0.02
    bear_rally_prior_24h: float = -0.08
    crash_volatility_24h: float = 0.06
    panic_fear_threshold: float = 20.0
    high_vol_percentile: float = 85.0


class EnsembleRegimeClassifier:
    """Heuristic ensemble-style classifier with deterministic output."""

    def __init__(self, config: Optional[RegimeClassifierConfig] = None):
        self.config = config or RegimeClassifierConfig()

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def classify(self, features: Optional[RegimeFeatures]) -> RegimeClassification:
        """Classify market regime from features."""
        if features is None:
            return RegimeClassification(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.2,
                bearish_score=0.35,
                bullish_score=0.35,
                reason="missing_features",
            )

        r4 = features.r4h()
        r24 = features.r24h()
        fear = features.normalized_fear()
        vol24 = features.volatility_24h
        vol_pct = features.vol_percentile
        corr = features.cross_asset_correlation
        m4 = features.momentum_4h if features.momentum_4h != 0.0 else r4
        m24 = features.momentum_24h if features.momentum_24h != 0.0 else r24

        bear = 0.0
        bull = 0.0

        # Bear pressure components
        if r24 <= self.config.crash_return_24h:
            bear += 0.55
        elif r24 <= self.config.slow_bear_return_24h:
            bear += 0.35
        if r4 <= self.config.crash_return_4h:
            bear += 0.25
        if vol24 >= self.config.crash_volatility_24h or vol_pct >= self.config.high_vol_percentile:
            bear += 0.20
        if fear <= self.config.panic_fear_threshold:
            bear += 0.20
        if corr >= 0.80:
            bear += 0.10

        # Bull pressure components (for bearish_score balancing)
        if r24 >= 0.06:
            bull += 0.45
        elif r24 >= 0.02:
            bull += 0.25
        if r4 >= 0.02:
            bull += 0.20
        if fear >= 65.0:
            bull += 0.15
        if m24 > 0 and m4 > 0:
            bull += 0.10

        bear = self._clamp01(bear)
        bull = self._clamp01(bull)

        # Regime inference
        if (
            r24 <= self.config.crash_return_24h
            and r4 <= self.config.crash_return_4h
            and (fear <= self.config.panic_fear_threshold or vol24 >= self.config.crash_volatility_24h)
        ):
            regime = MarketRegime.CRASH_CASCADE
            confidence = self._clamp01(0.65 + 0.30 * bear)
            reason = "deep_negative_returns_with_stress"
        elif (
            r24 <= self.config.bear_rally_prior_24h
            and (r4 >= self.config.bear_rally_bounce_4h or m4 > 0.0)
        ):
            regime = MarketRegime.BEAR_RALLY
            confidence = self._clamp01(0.45 + 0.25 * abs(r4) + 0.15 * bear)
            reason = "short_term_bounce_with_negative_context"
        elif bear >= 0.45 and r24 < 0:
            regime = MarketRegime.SLOW_BEAR
            confidence = self._clamp01(0.45 + 0.35 * bear)
            reason = "persistent_bearish_pressure"
        elif bull >= 0.55 and fear >= 45.0:
            regime = MarketRegime.BULL_RECOVERY
            confidence = self._clamp01(0.45 + 0.35 * bull)
            reason = "positive_returns_with_recovery_tone"
        else:
            regime = MarketRegime.SIDEWAYS
            spread = abs(r4) + abs(r24)
            confidence = self._clamp01(max(0.25, 0.70 - spread * 3.0))
            reason = "mixed_or_low_signal_context"

        # Keep bearish_score conservative in uncertain contexts so short filter
        # does not over-block by default.
        if regime == MarketRegime.SIDEWAYS and bear < 0.30:
            bear = 0.30

        return RegimeClassification(
            regime=regime,
            confidence=confidence,
            bearish_score=bear,
            bullish_score=bull,
            reason=reason,
        )


_regime_classifier: Optional[EnsembleRegimeClassifier] = None


def get_regime_classifier(
    config: Optional[RegimeClassifierConfig] = None,
) -> EnsembleRegimeClassifier:
    """Get singleton regime classifier."""
    global _regime_classifier
    if _regime_classifier is None:
        _regime_classifier = EnsembleRegimeClassifier(config=config)
        logger.info(
            "[REGIME_CLASSIFIER] Initialized at %s",
            datetime.now(timezone.utc).isoformat(),
        )
    return _regime_classifier


def reset_regime_classifier() -> None:
    """Reset singleton classifier (test utility)."""
    global _regime_classifier
    _regime_classifier = None


"""
================================================================================
HMATS v6.7 - P4-02 Gambler Sizing (Heavy-Bet Mode)
================================================================================
Profile-gated sizing boost for OPPORTUNITY windows with strong regimes.

When all gates pass:
  - Boosts intent.target_exposure up to max_bet_pct_nav (e.g. 0.50)
  - Only boosts, never reduces existing exposure

Gates (ALL must pass):
  1. system_mode in allowed_modes (e.g. OPPORTUNITY)
  2. regime in allowed_regimes (e.g. STRONG_TREND, IGNITION, EXPANSION)
  3. confidence >= confidence_threshold (e.g. 0.70)
  4. intent has direction (abs > 0.1)
  5. cooldown: min_seconds_between_bets (e.g. 14400 = 4h)
  6. weekly limit: max_bets_per_week (e.g. 3)

Does NOT bypass Authority/RiskGovernor/LeverageGuard - runs BEFORE risk chain.
Only active when config.enabled=True (ULTRA_AGGRESSIVE_5Y profile).
================================================================================
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GamblerSizingConfig:
    """Configuration for gambler sizing."""
    enabled: bool = False
    max_bet_pct_nav: float = 0.50
    confidence_threshold: float = 0.70
    allowed_modes: List[str] = field(default_factory=lambda: ["OPPORTUNITY"])
    allowed_regimes: List[str] = field(default_factory=lambda: ["STRONG_TREND", "IGNITION", "EXPANSION"])
    min_seconds_between_bets: int = 14400   # 4 hours
    max_bets_per_week: int = 3

    @classmethod
    def from_dict(cls, d: dict) -> "GamblerSizingConfig":
        return cls(
            enabled=d.get("enabled", False),
            max_bet_pct_nav=d.get("max_bet_pct_nav", 0.50),
            confidence_threshold=d.get("confidence_threshold", 0.70),
            allowed_modes=d.get("allowed_modes", ["OPPORTUNITY"]),
            allowed_regimes=d.get("allowed_regimes", ["STRONG_TREND", "IGNITION", "EXPANSION"]),
            min_seconds_between_bets=d.get("min_seconds_between_bets", 14400),
            max_bets_per_week=d.get("max_bets_per_week", 3),
        )


@dataclass
class GamblerSizingResult:
    """Result of gambler sizing evaluation."""
    triggered: bool = False
    reason: str = ""
    old_exposure: float = 0.0
    new_exposure: float = 0.0
    cooldown_ok: bool = True
    weekly_count: int = 0
    confidence: float = 0.0
    regime: str = ""
    mode: str = ""


# =============================================================================
# GAMBLER SIZING ENGINE
# =============================================================================

_WEEK_SECONDS = 7 * 86400


class GamblerSizing:
    """
    Heavy-bet sizing boost for OPPORTUNITY windows.

    Boosts intent.target_exposure up to max_bet_pct_nav when all gates pass.
    In-memory cooldown state (resets on process restart - fail-safe).

    Usage:
        gambler = GamblerSizing(config)
        result = gambler.evaluate(
            intent_direction=intent.direction,
            target_exposure=intent.target_exposure,
            system_mode=intent.system_mode,
            regime=market_data.get('regime_state', ''),
            confidence=intent.quant_confidence,
        )
        if result.triggered:
            intent.target_exposure = result.new_exposure
    """

    def __init__(self, config: GamblerSizingConfig):
        self.config = config
        self._bet_timestamps: List[float] = []  # epoch timestamps of past bets

    def evaluate(
        self,
        *,
        intent_direction: float,
        target_exposure: float,
        system_mode: str = "NORMAL",
        regime: str = "",
        confidence: float = 0.0,
        now_ts: Optional[float] = None,
    ) -> GamblerSizingResult:
        """
        Evaluate whether gambler sizing boost should trigger.

        Args:
            intent_direction: Decided direction from engine (-1 to +1)
            target_exposure: Current target_exposure from engine
            system_mode: "NORMAL", "OPPORTUNITY", "NO_TRADE"
            regime: Current regime name (e.g. "STRONG_TREND")
            confidence: quant_confidence
            now_ts: Override timestamp (for testing)

        Returns:
            GamblerSizingResult with triggered flag and new exposure
        """
        now = now_ts if now_ts is not None else time.time()
        regime_upper = regime.upper() if regime else ""

        result = GamblerSizingResult(
            old_exposure=target_exposure,
            confidence=confidence,
            regime=regime_upper,
            mode=system_mode,
        )

        # Gate 1: system_mode
        if system_mode not in self.config.allowed_modes:
            result.reason = "mode_not_allowed"
            return result

        # Gate 2: regime
        if regime_upper not in [r.upper() for r in self.config.allowed_regimes]:
            result.reason = "regime_not_allowed"
            return result

        # Gate 3: confidence
        if confidence < self.config.confidence_threshold:
            result.reason = "conf_too_low"
            return result

        # Gate 4: must have direction
        if abs(intent_direction) < 0.1:
            result.reason = "no_direction"
            return result

        # Gate 5: cooldown
        self._prune_old_bets(now)
        if self._bet_timestamps:
            elapsed = now - self._bet_timestamps[-1]
            if elapsed < self.config.min_seconds_between_bets:
                remaining = self.config.min_seconds_between_bets - elapsed
                result.reason = f"cooldown:{remaining:.0f}s_remaining"
                result.cooldown_ok = False
                return result

        # Gate 6: weekly limit
        week_count = len(self._bet_timestamps)
        result.weekly_count = week_count
        if week_count >= self.config.max_bets_per_week:
            result.reason = f"weekly_limit:{week_count}/{self.config.max_bets_per_week}"
            result.cooldown_ok = False
            return result

        # Gate 7: only boost, never reduce
        desired = self.config.max_bet_pct_nav
        if target_exposure >= desired:
            result.reason = "already_at_max"
            return result

        # --- TRIGGERED ---
        result.triggered = True
        result.new_exposure = desired
        result.reason = "gambler_triggered"
        result.weekly_count = week_count + 1

        # Record bet
        self._bet_timestamps.append(now)

        logger.warning(
            f"[GAMBLER_TRIGGER] mode={system_mode} regime={regime_upper} "
            f"conf={confidence:.3f} exposure={target_exposure:.4f}->{desired:.4f} "
            f"weekly={week_count + 1}/{self.config.max_bets_per_week}"
        )

        return result

    def _prune_old_bets(self, now: float):
        """Remove bets older than 7 days."""
        cutoff = now - _WEEK_SECONDS
        self._bet_timestamps = [ts for ts in self._bet_timestamps if ts >= cutoff]

    def get_weekly_count(self, now_ts: Optional[float] = None) -> int:
        """Get current weekly bet count (for proof log)."""
        now = now_ts if now_ts is not None else time.time()
        self._prune_old_bets(now)
        return len(self._bet_timestamps)

    def to_dict(self) -> dict:
        """Serialize state for persistence (T23)."""
        self._prune_old_bets(time.time())
        return {"bet_timestamps": list(self._bet_timestamps)}

    def from_dict(self, d: dict) -> None:
        """Restore state from persisted dict (T23)."""
        self._bet_timestamps = list(d.get("bet_timestamps", []))
        self._prune_old_bets(time.time())

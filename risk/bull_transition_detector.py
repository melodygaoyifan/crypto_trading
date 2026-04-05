"""
BullTransitionDetector - Mild Bull Market Regime Detector.

Audit S23: System blind spot. In a mild bull market:
  - Regime != STRONG_BULL -> existing protections don't trigger
  - Short strategies bleed slowly without detection
  - 8 weeks can lose 20-25%

Detection: 4 conditions, 4-stage state machine:
  INACTIVE -> POTENTIAL (1 cond) -> ACTIVE (2+ cond) -> CONFIRMED (5+ days)

Actions:
  ACTIVE:    Halve new short exposure, flag in proof log
  CONFIRMED: Block naked shorts (hedged shorts still allowed)

Fail-open: All exceptions caught, returns INACTIVE on error.
"""
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class BullTransitionState(Enum):
    INACTIVE = "INACTIVE"
    POTENTIAL = "POTENTIAL"
    ACTIVE = "ACTIVE"
    CONFIRMED = "CONFIRMED"


@dataclass
class BullTransitionSignal:
    state: BullTransitionState = BullTransitionState.INACTIVE
    conditions_met: int = 0
    conditions_detail: Dict[str, bool] = field(default_factory=dict)
    days_in_state: int = 0
    action: str = "NONE"  # NONE | REDUCE_SHORT | BLOCK_NAKED_SHORT


class BullTransitionDetector:
    """Detect mild bull market transitions that bleed short strategies.

    4 detection conditions:
      1. BTC price above 50-bar MA (weekly approx via 4H bars)
      2. Relative strength of SOL vs BTC positive over 14 days
      3. Funding rate positive for 7+ consecutive days
      4. OI rising while liquidations declining (supply/demand imbalance)
    """

    ACTIVE_THRESHOLD = 2       # 2+ conditions -> ACTIVE
    CONFIRMED_DAYS = 5         # 5 consecutive days -> CONFIRMED
    DEACTIVATE_BELOW = 0       # 0 conditions -> back to INACTIVE

    def __init__(self):
        self._state = BullTransitionState.INACTIVE
        self._state_entry_time: Optional[datetime] = None

    def evaluate(
        self,
        btc_price: float = 0.0,
        btc_ma50: float = 0.0,
        sol_btc_relative_strength: float = 0.0,
        funding_positive_streak_days: int = 0,
        oi_rising: bool = False,
        liquidations_declining: bool = False,
    ) -> BullTransitionSignal:
        """Evaluate bull transition conditions.

        Args:
            btc_price: Current BTC price.
            btc_ma50: 50-bar moving average of BTC (from 4H bars).
            sol_btc_relative_strength: SOL/BTC 14-day relative strength (>0 = SOL outperforming).
            funding_positive_streak_days: Days of consecutive positive funding.
            oi_rising: Whether open interest is increasing.
            liquidations_declining: Whether long liquidations are declining.

        Returns:
            BullTransitionSignal with state, conditions, and recommended action.
        """
        try:
            conditions = {
                "btc_above_ma50": btc_price > btc_ma50 > 0,
                "sol_relative_strength": sol_btc_relative_strength > 0,
                "funding_streak_7d": funding_positive_streak_days >= 7,
                "oi_liq_divergence": oi_rising and liquidations_declining,
            }
            conditions_met = sum(conditions.values())

            prev_state = self._state
            now = datetime.now(timezone.utc)

            # State transitions
            if conditions_met >= self.ACTIVE_THRESHOLD:
                if self._state in (BullTransitionState.INACTIVE, BullTransitionState.POTENTIAL):
                    self._state = BullTransitionState.ACTIVE
                    self._state_entry_time = now
                elif self._state == BullTransitionState.ACTIVE:
                    days = (now - self._state_entry_time).days if self._state_entry_time else 0
                    if days >= self.CONFIRMED_DAYS:
                        self._state = BullTransitionState.CONFIRMED
                # CONFIRMED stays CONFIRMED while conditions hold

            elif conditions_met >= 1:
                if self._state == BullTransitionState.INACTIVE:
                    self._state = BullTransitionState.POTENTIAL
                    self._state_entry_time = now
                elif self._state in (BullTransitionState.ACTIVE, BullTransitionState.CONFIRMED):
                    # Downgrade
                    self._state = BullTransitionState.POTENTIAL
                    self._state_entry_time = now

            else:
                self._state = BullTransitionState.INACTIVE
                self._state_entry_time = None

            days_in_state = (now - self._state_entry_time).days if self._state_entry_time else 0

            # Log transitions
            if self._state != prev_state:
                level = logging.WARNING if self._state in (BullTransitionState.ACTIVE, BullTransitionState.CONFIRMED) else logging.INFO
                logger.log(
                    level,
                    f"[BULL-TRANSITION] {prev_state.value} -> {self._state.value} "
                    f"({conditions_met} conditions: {conditions})"
                )

            action = self._determine_action()

            return BullTransitionSignal(
                state=self._state,
                conditions_met=conditions_met,
                conditions_detail=conditions,
                days_in_state=days_in_state,
                action=action,
            )

        except Exception as e:
            logger.error(f"[BULL-TRANSITION] Error: {e} -> fail-open INACTIVE")
            return BullTransitionSignal()

    def _determine_action(self) -> str:
        if self._state == BullTransitionState.CONFIRMED:
            return "BLOCK_NAKED_SHORT"
        elif self._state == BullTransitionState.ACTIVE:
            return "REDUCE_SHORT"
        elif self._state == BullTransitionState.POTENTIAL:
            return "REDUCE_SHORT_LIGHT"  # [FIX-L2-02] Early warning: mild reduction
        return "NONE"

    @property
    def is_bull_transition(self) -> bool:
        return self._state in (BullTransitionState.ACTIVE, BullTransitionState.CONFIRMED)

    @property
    def naked_short_blocked(self) -> bool:
        return self._state == BullTransitionState.CONFIRMED

    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            "state": self._state.value,
            "state_entry_time": self._state_entry_time.isoformat() if self._state_entry_time else None,
        }

    def from_dict(self, data: Dict) -> None:
        """Restore from persisted state."""
        try:
            self._state = BullTransitionState(data.get("state", "INACTIVE"))
            entry = data.get("state_entry_time")
            self._state_entry_time = datetime.fromisoformat(entry) if entry else None
        except Exception as e:
            logger.warning(f"[BULL-TRANSITION] Failed to restore state: {e}")
            self._state = BullTransitionState.INACTIVE
            self._state_entry_time = None


# Module-level singleton
_detector: Optional[BullTransitionDetector] = None


def get_bull_transition_detector() -> BullTransitionDetector:
    """Get or create the singleton BullTransitionDetector."""
    global _detector
    if _detector is None:
        _detector = BullTransitionDetector()
    return _detector
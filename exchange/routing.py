"""
HMATS v5.1 Phase 2 - Dual-Venue Routing Policy
================================================

During Phase 2 dual-venue cutover, RoutingPolicy decides which venue
handles each order. Default policy implements the v5.1 prompt's
[PARAMETER 3] = dual_venue 4-week schedule:

  Week 1-2: Coinbase shadow only (read-only data; no Coinbase orders).
            All orders → Kraken. Adapter still queries Coinbase
            funding/orderbook for parity comparison.
  Week 3:   50/50 split per asset. Routing decided by per-asset
            preference (configurable; default: alphabetical lower-half →
            Coinbase, upper-half → Kraken so SOL → Coinbase, BTC+ETH → Kraken).
  Week 4+:  100% Coinbase. Kraken adapter standby for rollback.

Iron Laws:
  5. DRL ACTIVE during cutover: routing does NOT touch DRL inference.
  8. (NEW) DRL ACTIVE invariant during cutover: validated continuously
     via cutover.validate_drl_active() — refuses to advance phase if
     DRL has demoted to SHADOW.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional


class CutoverPhase(Enum):
    PRE_PHASE_2 = "PRE_PHASE_2"        # Kraken only; Coinbase not active
    SHADOW = "SHADOW"                   # Week 1-2: Coinbase read-only
    DUAL_VENUE = "DUAL_VENUE"           # Week 3: 50/50 split
    COINBASE_PRIMARY = "COINBASE_PRIMARY"  # Week 4+: 100% Coinbase
    ROLLBACK = "ROLLBACK"               # emergency: forced back to Kraken


@dataclass
class RoutingPolicy:
    """Stateful routing policy. Operator drives phase transitions via
    advance_phase() — validation enforces dependency on Iron Law 8."""
    phase: CutoverPhase = CutoverPhase.PRE_PHASE_2
    phase_started_at: Optional[datetime] = None
    coinbase_assets: List[str] = field(default_factory=lambda: ["SOL"])
    kraken_assets: List[str] = field(default_factory=lambda: ["BTC", "ETH"])

    def venue_for(self, asset: str) -> str:
        """Return 'kraken' or 'coinbase' for this order."""
        asset = asset.upper()
        if self.phase == CutoverPhase.PRE_PHASE_2:
            return "kraken"
        if self.phase == CutoverPhase.SHADOW:
            return "kraken"  # Coinbase shadow is read-only; orders → Kraken
        if self.phase == CutoverPhase.DUAL_VENUE:
            if asset in self.coinbase_assets:
                return "coinbase"
            return "kraken"
        if self.phase == CutoverPhase.COINBASE_PRIMARY:
            return "coinbase"
        if self.phase == CutoverPhase.ROLLBACK:
            return "kraken"
        return "kraken"  # safe default

    def advance_phase(
        self,
        new_phase: CutoverPhase,
        drl_authority_level: str = "ACTIVE",
    ) -> tuple[bool, str]:
        """Transition to a new phase. Iron Law 8 enforced: refuses to
        advance away from PRE_PHASE_2 if DRL is not ACTIVE.

        ROLLBACK is always permitted. Operator-driven.

        Returns (success, reason).
        """
        if new_phase == CutoverPhase.ROLLBACK:
            self.phase = new_phase
            self.phase_started_at = datetime.now(timezone.utc)
            return True, "rollback always permitted"

        if str(drl_authority_level).upper() != "ACTIVE":
            return False, (
                f"Iron Law 8 violation: DRL authority is {drl_authority_level}, "
                f"not ACTIVE. Cutover refuses to advance."
            )

        # Sanity: enforce monotonic progression except for ROLLBACK
        valid_progressions = {
            CutoverPhase.PRE_PHASE_2: [CutoverPhase.SHADOW, CutoverPhase.ROLLBACK],
            CutoverPhase.SHADOW: [CutoverPhase.DUAL_VENUE, CutoverPhase.PRE_PHASE_2,
                                  CutoverPhase.ROLLBACK],
            CutoverPhase.DUAL_VENUE: [CutoverPhase.COINBASE_PRIMARY, CutoverPhase.SHADOW,
                                      CutoverPhase.ROLLBACK],
            CutoverPhase.COINBASE_PRIMARY: [CutoverPhase.DUAL_VENUE, CutoverPhase.ROLLBACK],
            CutoverPhase.ROLLBACK: [CutoverPhase.PRE_PHASE_2, CutoverPhase.SHADOW],
        }
        allowed = valid_progressions.get(self.phase, [])
        if new_phase not in allowed:
            return False, (
                f"invalid transition: {self.phase.value} → {new_phase.value}. "
                f"Allowed: {[p.value for p in allowed]}"
            )

        self.phase = new_phase
        self.phase_started_at = datetime.now(timezone.utc)
        return True, f"transitioned to {new_phase.value}"

    def is_dual_venue_active(self) -> bool:
        return self.phase in (CutoverPhase.SHADOW, CutoverPhase.DUAL_VENUE)

    def to_dict(self) -> Dict[str, str]:
        return {
            "phase": self.phase.value,
            "phase_started_at": self.phase_started_at.isoformat() if self.phase_started_at else "",
            "coinbase_assets": ",".join(self.coinbase_assets),
            "kraken_assets": ",".join(self.kraken_assets),
        }

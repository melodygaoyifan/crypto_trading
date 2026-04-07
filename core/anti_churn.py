"""
AntiChurnManager - rate-limiting and fill budget for trade execution.

Extracted from main.py (Phase 4B) to reduce God Object attributes.
Encapsulates AC-1 (min hold), AC-2 (per-asset rate limit), AC-5 (daily budget).
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AntiChurnManager:
    """Anti-churn controls: min hold time, per-asset rate limit, daily fill budget."""

    def __init__(
        self,
        ac1_min_hold_ticks: int = 2,        # [PRE-LAUNCH] was 1 (4h). 2 ticks = 8h min hold to reduce churn
        ac1_flip_min_hold_ticks: int = 3,  # [PRE-LAUNCH] was 2. Flips need 12h to avoid whipsaw
        ac2_max_per_asset: int = 2,
        ac2_max_global: int = 4,       # [CALIBRATION] was 6. 4 global = fewer concurrent entries
        ac2_window_ticks: int = 6,
        ac5_max_per_day: int = 8,      # [UTIL-4] was 4. 8 fills × 15bps = 120bps/day max fee drag. Needed for higher utilization
    ):
        # AC-1: Minimum hold time
        self.ac1_min_hold_ticks = ac1_min_hold_ticks
        self.ac1_flip_min_hold_ticks = ac1_flip_min_hold_ticks

        # AC-2: Per-asset fill rate limiter
        self.ac2_max_per_asset = ac2_max_per_asset
        self.ac2_max_global = ac2_max_global
        self.ac2_window_ticks = ac2_window_ticks
        self._fill_ticks: Dict[str, List[int]] = {}

        # AC-5: Daily fill budget (persisted)
        self.ac5_max_per_day = ac5_max_per_day
        self._fills_today: int = 0
        self._fills_date: str = ""

    # ------------------------------------------------------------------
    # AC-1: Minimum hold time
    # ------------------------------------------------------------------
    _SAFETY_EXITS = frozenset({
        "stop_loss", "drawdown_halt", "p0_safety",
        "max_hold_timeout", "dead_man_switch", "leverage_guard",
        "FRICTION_EXCEEDS_EDGE",
    })

    def check_min_hold(
        self,
        asset: str,
        entry_tick: int,
        current_tick: int,
        exit_reason: str,
        target_exposure: float,
        current_exposure: float,
    ) -> Optional[Dict]:
        """Check AC-1 min hold. Returns block dict or None if OK.

        L4-15: Direction flips (long->short, short->long) require a higher
        hold threshold to prevent churn from rapid regime oscillation.
        Same-direction adds and reduces use the standard threshold.
        """
        ticks_held = current_tick - entry_tick
        is_safety = any(s in exit_reason for s in self._SAFETY_EXITS)

        if entry_tick <= 0 or is_safety:
            return None

        # L4-15: Detect flip (target sign differs from current sign)
        is_flip = (target_exposure * current_exposure < 0)  # opposite signs
        min_hold = self.ac1_flip_min_hold_ticks if is_flip else self.ac1_min_hold_ticks

        if ticks_held < min_hold:
            if abs(target_exposure) < abs(current_exposure) * 0.95 or is_flip:
                tag = "FLIP" if is_flip else "EXIT"
                logger.info(
                    f"[AC-1] {asset}: {tag} BLOCKED - held {ticks_held}/"
                    f"{min_hold} ticks"
                )
                return {
                    "status": "AC1_MIN_HOLD_BLOCKED",
                    "reason": f"min hold ({tag}): {ticks_held}/{min_hold} ticks",
                }
        return None

    # ------------------------------------------------------------------
    # AC-5 + AC-2: Fill budget and rate limiter
    # ------------------------------------------------------------------
    def check_fill_budget(
        self,
        asset: str,
        current_tick: int,
        is_new_entry: bool,
    ) -> Optional[Dict]:
        """Check AC-5 daily budget + AC-2 rate limit. Returns block dict or None."""
        # AC-5: Daily fill budget
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._fills_date != today:
            self._fills_today = 0
            self._fills_date = today

        if self._fills_today >= self.ac5_max_per_day:
            logger.warning(
                f"[AC-5] {asset}: FILL BUDGET EXHAUSTED - "
                f"{self._fills_today}/{self.ac5_max_per_day} today"
            )
            return {
                "status": "AC5_BUDGET_EXHAUSTED",
                "reason": f"daily fill budget: {self._fills_today}/{self.ac5_max_per_day}",
            }

        # AC-2: Per-asset + global rate limit (new entries only)
        if is_new_entry:
            recent = self._fill_ticks.get(asset, [])
            in_window = sum(
                1 for t in recent if current_tick - t <= self.ac2_window_ticks
            )
            if in_window >= self.ac2_max_per_asset:
                logger.info(
                    f"[AC-2] {asset}: ENTRY BLOCKED - {in_window} fills "
                    f"in last {self.ac2_window_ticks} ticks "
                    f"(limit={self.ac2_max_per_asset})"
                )
                return {
                    "status": "AC2_RATE_LIMITED",
                    "reason": f"per-asset rate limit: {in_window}/{self.ac2_max_per_asset}",
                }

            global_count = sum(
                sum(1 for t in fills if current_tick - t <= self.ac2_window_ticks)
                for fills in self._fill_ticks.values()
            )
            if global_count >= self.ac2_max_global:
                logger.info(
                    f"[AC-2] {asset}: ENTRY BLOCKED - global rate limit "
                    f"{global_count}/{self.ac2_max_global}"
                )
                return {
                    "status": "AC2_GLOBAL_RATE_LIMITED",
                    "reason": f"global rate limit: {global_count}/{self.ac2_max_global}",
                }

        return None

    # ------------------------------------------------------------------
    # Recording fills
    # ------------------------------------------------------------------
    def record_fill(self, asset: str, current_tick: int) -> None:
        """Record a fill for AC-2 tracking + AC-5 daily budget."""
        if asset not in self._fill_ticks:
            self._fill_ticks[asset] = []
        self._fill_ticks[asset].append(current_tick)
        # Prune old entries (keep last 20)
        if len(self._fill_ticks[asset]) > 20:
            self._fill_ticks[asset] = self._fill_ticks[asset][-20:]
        self._fills_today += 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "ac5_fills_today": self._fills_today,
            "ac5_fills_date": self._fills_date,
            "ac2_fill_ticks": dict(self._fill_ticks),  # [FIX-M3] persist AC-2 rate limit
        }

    def from_dict(self, data: Dict) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        saved_date = data.get("ac5_fills_date", "")
        if saved_date == today:
            self._fills_today = data.get("ac5_fills_today", 0)
            self._fills_date = saved_date
            logger.info(
                f"[AC-5] Restored fill budget: "
                f"{self._fills_today}/{self.ac5_max_per_day} today"
            )
        else:
            logger.info(
                f"[AC-5] Fill budget reset (new day: was {saved_date}, now {today})"
            )
        # [FIX-M3] Restore AC-2 per-asset rate limit
        _saved_ticks = data.get("ac2_fill_ticks", {})
        if _saved_ticks:
            self._fill_ticks = {k: list(v) for k, v in _saved_ticks.items()}
            logger.info(f"[AC-2] Restored fill ticks: {list(self._fill_ticks.keys())}")

"""
risk/exit_drl_kill_switch.py — Per-asset auto-demote monitor for Exit DRL.

Required by CLAUDE.md P28's "Land a kill switch in the same commit" promotion
prerequisite. Tripped by any of four conditions, all per-asset:

  1. CONSECUTIVE_LOSSES   — 5 consecutive Exit-DRL-driven closes underwater.
  2. ROLLING_PNL_NEGATIVE — 7-day rolling realized PnL since promotion < 0.
  3. HOLD_RATIO_DRIFT     — HOLD share over last 50 ticks outside [50%, 90%].
  4. EXIT_ALL_HIGH        — EXIT_ALL share over last 50 closed exits > 30%.

The kill switch is **read-only** from the agent's perspective: it returns a
demotion *recommendation* via `should_demote(asset)`. The runner is responsible
for actually flipping `agents/exit_drl_agent.ExitDRLMode.EXIT_ONLY` →
`ExitDRLMode.SHADOW` for that asset and writing an audit record.

State is in-memory + JSONL-persisted to `data/exit_drl_kill_switch_state.jsonl`
so per-asset rolling counters survive restart (otherwise a crash-loop would
reset the consecutive-loss counter and let a sick model keep firing).
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Trip thresholds — defaults in CLAUDE.md P28; per-asset overrides via __init__.
# ────────────────────────────────────────────────────────────────
CONSECUTIVE_LOSSES_LIMIT = 5
ROLLING_PNL_WINDOW_DAYS = 7.0
ROLLING_PNL_MIN_BPS = 0.0
HOLD_RATIO_WINDOW = 50
HOLD_RATIO_MIN = 0.50
HOLD_RATIO_MAX = 0.90
EXIT_ALL_WINDOW = 50
EXIT_ALL_RATIO_MAX = 0.30


@dataclass
class _AssetState:
    asset: str
    consecutive_losses: int = 0
    promoted_at: Optional[float] = None
    pnl_history: List[Dict[str, float]] = field(default_factory=list)  # [{ts, pnl_bps}]
    recent_step_actions: Deque[str] = field(default_factory=lambda: deque(maxlen=HOLD_RATIO_WINDOW))
    recent_close_actions: Deque[str] = field(default_factory=lambda: deque(maxlen=EXIT_ALL_WINDOW))
    last_demote_reason: Optional[str] = None


class ExitDRLKillSwitch:
    """In-memory + on-disk per-asset trip monitor. See module docstring."""

    def __init__(
        self,
        state_path: str = "data/exit_drl_kill_switch_state.jsonl",
        consecutive_losses_limit: int = CONSECUTIVE_LOSSES_LIMIT,
        rolling_pnl_window_days: float = ROLLING_PNL_WINDOW_DAYS,
        rolling_pnl_min_bps: float = ROLLING_PNL_MIN_BPS,
        hold_ratio_window: int = HOLD_RATIO_WINDOW,
        hold_ratio_min: float = HOLD_RATIO_MIN,
        hold_ratio_max: float = HOLD_RATIO_MAX,
        exit_all_window: int = EXIT_ALL_WINDOW,
        exit_all_ratio_max: float = EXIT_ALL_RATIO_MAX,
    ):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.consecutive_losses_limit = int(consecutive_losses_limit)
        self.rolling_pnl_window_days = float(rolling_pnl_window_days)
        self.rolling_pnl_min_bps = float(rolling_pnl_min_bps)
        self.hold_ratio_window = int(hold_ratio_window)
        self.hold_ratio_min = float(hold_ratio_min)
        self.hold_ratio_max = float(hold_ratio_max)
        self.exit_all_window = int(exit_all_window)
        self.exit_all_ratio_max = float(exit_all_ratio_max)
        self._states: Dict[str, _AssetState] = {}

    # ------------------------------------------------------------------
    # Promotion lifecycle (called by runner when an asset goes EXIT_ONLY)
    # ------------------------------------------------------------------
    def record_promotion(self, asset: str, ts: Optional[float] = None) -> None:
        st = self._get_state(asset)
        st.promoted_at = ts or time.time()
        st.consecutive_losses = 0
        st.pnl_history.clear()
        st.recent_step_actions.clear()
        st.recent_close_actions.clear()
        st.last_demote_reason = None
        self._persist()
        logger.warning(
            f"[EXIT_DRL_KILLSWITCH] {asset}: promotion timestamp recorded "
            f"(rolling-PnL window starts now)"
        )

    def record_demotion(self, asset: str, reason: str, ts: Optional[float] = None) -> None:
        st = self._get_state(asset)
        st.last_demote_reason = reason
        st.promoted_at = None
        self._persist()
        logger.critical(
            f"[EXIT_DRL_KILLSWITCH] {asset}: DEMOTED to SHADOW — {reason}"
        )

    # ------------------------------------------------------------------
    # Per-tick / per-close updates
    # ------------------------------------------------------------------
    def record_step_action(self, asset: str, action_name: str) -> None:
        """One per asset per tick that ran an Exit-SAC prediction."""
        self._get_state(asset).recent_step_actions.append(str(action_name))

    def record_close(
        self,
        asset: str,
        realized_pnl_bps: float,
        terminal_action: str,
        was_drl_driven: bool,
        ts: Optional[float] = None,
    ) -> None:
        """
        Called from the close path. `was_drl_driven` = True iff the close was
        triggered (or co-triggered) by an Exit-SAC PARTIAL_EXIT firing through
        TRIGGER 4 — only those count for the consecutive-loss + rolling-PnL
        guards. Distribution guards count every close.
        """
        st = self._get_state(asset)
        now = ts or time.time()
        st.recent_close_actions.append(str(terminal_action))

        if was_drl_driven:
            # Consecutive-loss counter
            if realized_pnl_bps < 0:
                st.consecutive_losses += 1
            else:
                st.consecutive_losses = 0
            # Rolling PnL window
            st.pnl_history.append({"ts": now, "pnl_bps": float(realized_pnl_bps)})
            self._prune_pnl_history(st, now)
        self._persist()

    # ------------------------------------------------------------------
    # Trip evaluation
    # ------------------------------------------------------------------
    def should_demote(self, asset: str) -> Optional[str]:
        """Return a reason string if any trip condition fires, else None."""
        st = self._states.get(asset)
        if st is None or st.promoted_at is None:
            return None  # not promoted → not subject to demotion
        now = time.time()
        # 1. Consecutive losses
        if st.consecutive_losses >= self.consecutive_losses_limit:
            return f"CONSECUTIVE_LOSSES: {st.consecutive_losses} >= {self.consecutive_losses_limit}"
        # 2. Rolling PnL (only meaningful after the window has had time to fill)
        days_since_promotion = (now - st.promoted_at) / 86400.0
        if days_since_promotion >= self.rolling_pnl_window_days:
            self._prune_pnl_history(st, now)
            window_pnl = sum(r["pnl_bps"] for r in st.pnl_history)
            if window_pnl < self.rolling_pnl_min_bps:
                return (
                    f"ROLLING_PNL_NEGATIVE: {self.rolling_pnl_window_days:.1f}d window "
                    f"sum={window_pnl:+.1f}bps < {self.rolling_pnl_min_bps:+.1f}bps"
                )
        # 3. HOLD ratio drift (only if window is full)
        if len(st.recent_step_actions) >= self.hold_ratio_window:
            hold_n = sum(1 for a in st.recent_step_actions if a == "HOLD")
            hold_ratio = hold_n / len(st.recent_step_actions)
            if hold_ratio < self.hold_ratio_min:
                return (
                    f"HOLD_RATIO_LOW: {hold_ratio:.2%} < {self.hold_ratio_min:.0%} "
                    f"over last {len(st.recent_step_actions)} ticks (over-active)"
                )
            if hold_ratio > self.hold_ratio_max:
                return (
                    f"HOLD_RATIO_HIGH: {hold_ratio:.2%} > {self.hold_ratio_max:.0%} "
                    f"over last {len(st.recent_step_actions)} ticks (asleep)"
                )
        # 4. EXIT_ALL ratio (closes only)
        if len(st.recent_close_actions) >= self.exit_all_window:
            exit_n = sum(1 for a in st.recent_close_actions if a == "EXIT_ALL")
            exit_ratio = exit_n / len(st.recent_close_actions)
            if exit_ratio > self.exit_all_ratio_max:
                return (
                    f"EXIT_ALL_HIGH: {exit_ratio:.2%} > {self.exit_all_ratio_max:.0%} "
                    f"over last {len(st.recent_close_actions)} closes (panic-exiting)"
                )
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_state(self, asset: str) -> _AssetState:
        if asset not in self._states:
            self._states[asset] = _AssetState(asset=asset)
        return self._states[asset]

    def _prune_pnl_history(self, st: _AssetState, now: float) -> None:
        cutoff = now - self.rolling_pnl_window_days * 86400.0
        st.pnl_history = [r for r in st.pnl_history if r["ts"] >= cutoff]

    def _persist(self) -> None:
        """One JSONL line per asset, latest snapshot. Best-effort."""
        try:
            payload = {
                "ts": time.time(),
                "states": {
                    a: {
                        "consecutive_losses": s.consecutive_losses,
                        "promoted_at": s.promoted_at,
                        "pnl_history_n": len(s.pnl_history),
                        "recent_step_actions_n": len(s.recent_step_actions),
                        "recent_close_actions_n": len(s.recent_close_actions),
                        "last_demote_reason": s.last_demote_reason,
                    }
                    for a, s in self._states.items()
                },
            }
            with self.state_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logger.debug(f"[EXIT_DRL_KILLSWITCH] persist failed: {e}")

    def status(self) -> Dict[str, Any]:
        return {
            "state_path": str(self.state_path),
            "thresholds": {
                "consecutive_losses_limit": self.consecutive_losses_limit,
                "rolling_pnl_window_days": self.rolling_pnl_window_days,
                "hold_ratio_min": self.hold_ratio_min,
                "hold_ratio_max": self.hold_ratio_max,
                "exit_all_ratio_max": self.exit_all_ratio_max,
            },
            "states": {
                a: {
                    "promoted_at": s.promoted_at,
                    "consecutive_losses": s.consecutive_losses,
                    "n_pnl_records": len(s.pnl_history),
                    "n_step_actions": len(s.recent_step_actions),
                    "n_close_actions": len(s.recent_close_actions),
                    "last_demote_reason": s.last_demote_reason,
                }
                for a, s in self._states.items()
            },
        }


# Module-level singleton
_singleton: Optional[ExitDRLKillSwitch] = None


def get_kill_switch(state_path: str = "data/exit_drl_kill_switch_state.jsonl") -> ExitDRLKillSwitch:
    global _singleton
    if _singleton is None:
        _singleton = ExitDRLKillSwitch(state_path=state_path)
    return _singleton


def reset_singleton() -> None:
    global _singleton
    _singleton = None

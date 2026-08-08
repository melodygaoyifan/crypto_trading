"""
DRL Promotion Gate with Auto-Demotion.

Lifecycle: DISABLED -> SHADOW -> EXIT_ONLY -> ACTIVE
Auto-demotion: ACTIVE -> EXIT_ONLY when performance degrades.
Auto-recovery: EXIT_ONLY -> ACTIVE after recovery period.
"""

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field

logger = logging.getLogger("HMATS.DRLGate")


def _strip_tz(dt: Optional[datetime]) -> Optional[datetime]:
    """Strip tzinfo to match the gate's naive datetime.now() convention.

    [P39 2026-04-24] datetime.fromisoformat() returns aware OR naive
    depending on the persisted ISO string. The gate uses naive
    datetime.now() everywhere, so loaded timestamps must be normalized
    on read to avoid TypeError on naive-vs-aware comparison.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


@dataclass
class DemotionConfig:
    """Auto-demotion configuration. Permanently disabled per operator directive 2026-04-30."""
    enable: bool = False
    consecutive_loss_threshold: int = 5
    drawdown_trigger_pct: float = 0.15
    demote_to: str = "EXIT_ONLY"
    recovery_period_days: int = 3
    max_demotions_before_disable: int = 3
    demotion_window_days: int = 14


@dataclass
class DRLTradeRecord:
    """Single DRL-participated trade record."""
    timestamp: str
    pnl: float
    drl_contributed: bool
    authority_level: str


class DRLPromotionGate:
    """
    DRL lifecycle manager with auto-demotion safety net.

    State is persisted to JSON file, survives restarts.
    """

    VALID_LEVELS = {"DISABLED", "SHADOW", "EXIT_ONLY", "ACTIVE"}

    def __init__(self, config: Optional[dict] = None, state_file: str = "data/drl_promotion_state.json"):
        self.demotion_config = DemotionConfig(**(config or {}))
        self.state_file = Path(state_file)

        # State
        self._authority_level: str = "DISABLED"
        self._trade_history: List[DRLTradeRecord] = []
        self._demotion_history: List[dict] = []
        self._demoted_at: Optional[datetime] = None
        self._peak_equity_contribution: float = 0.0
        self._current_equity_contribution: float = 0.0

        self._load_state()

        logger.info(
            f"[DRL_GATE] Initialized: level={self._authority_level}, "
            f"demotion={self.demotion_config.enable}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_authority_level(self) -> str:
        """Get current DRL authority level. The PERSISTED level is authoritative.

        [P227b] The 2026-04-30 "restore to ACTIVE on first read" branch is
        RETIRED. It silently re-promoted whenever `_demoted_at` was set — so
        any demotion performed through this class's own `_demote`
        (which sets `_demoted_at` at the end) would have been UNDONE by the
        very next read: a self-reversing demotion mechanism, the exact shape
        P198 removed from main.py ([DRL_FORCE_ACTIVE]). It never fired live
        only because the P198 manual demote wrote `demoted_at: null`. Under
        P200-LADDER Rung 4a the gate's persisted level is authoritative and
        promotion is a deliberate operator write, never a read side-effect.
        """
        if self._demoted_at and self._authority_level != "ACTIVE":
            # Legacy state observed — REPORT it, never act on it.
            if not getattr(self, "_p227b_demoted_at_logged", False):
                self._p227b_demoted_at_logged = True
                logger.info(
                    f"[DRL_GATE] persisted level={self._authority_level} with "
                    f"demoted_at={self._demoted_at} — honoring the persisted "
                    f"level (the pre-P227b code would have silently restored "
                    f"ACTIVE here)"
                )
        return self._authority_level

    def get_authority(self):
        """Compatibility shim for StubPromotionGate interface."""
        from integration.integration_v36 import DRLAuthorityLevel
        level_str = self.get_authority_level()
        try:
            return DRLAuthorityLevel(level_str)
        except ValueError:
            return DRLAuthorityLevel.DISABLED

    def has_exit_authority(self) -> bool:
        """Compatibility: does DRL have at least exit authority?"""
        return self._authority_level in ("EXIT_ONLY", "ACTIVE")

    def is_shadow_mode(self) -> bool:
        """Compatibility: is DRL in shadow mode?"""
        return self._authority_level == "SHADOW"

    def check_promotion(self) -> dict:
        """Compatibility: return promotion status."""
        return {
            "authority": self.get_authority_level(),
            "can_promote": False,
            "reason": "Managed by DRLPromotionGate",
        }

    def promote(self, level: str):
        """Manually set DRL authority level."""
        if level not in self.VALID_LEVELS:
            logger.error(f"[DRL_GATE] Invalid level: {level}")
            return
        old = self._authority_level
        self._authority_level = level
        self._demoted_at = None
        self._save_state()
        logger.info(f"[DRL_GATE] PROMOTED: {old} -> {level}")

    def record_trade(self, pnl: float, drl_contributed: bool = True):
        """Record a DRL-participated trade, check demotion."""
        record = DRLTradeRecord(
            timestamp=datetime.now().isoformat(),
            pnl=pnl,
            drl_contributed=drl_contributed,
            authority_level=self._authority_level,
        )
        self._trade_history.append(record)

        if drl_contributed:
            self._current_equity_contribution += pnl
            if self._current_equity_contribution > self._peak_equity_contribution:
                self._peak_equity_contribution = self._current_equity_contribution

        # [FIX-DA3] Was ACTIVE-only. EXIT_ONLY can also degrade → should demote to SHADOW.
        if self._authority_level in ("ACTIVE", "EXIT_ONLY") and self.demotion_config.enable:
            self._check_demotion()

        self._save_state()

    def get_status(self) -> dict:
        """Full status for dashboard/logging."""
        peak = max(self._peak_equity_contribution, 1.0)
        dd = (self._peak_equity_contribution - self._current_equity_contribution) / peak
        return {
            "authority_level": self.get_authority_level(),
            "demoted_at": self._demoted_at.isoformat() if self._demoted_at else None,
            "recovery_at": (
                (self._demoted_at + timedelta(days=self.demotion_config.recovery_period_days)).isoformat()
                if self._demoted_at
                else None
            ),
            "drl_equity_contribution": self._current_equity_contribution,
            "drl_peak_equity": self._peak_equity_contribution,
            "drl_drawdown": dd,
            "total_trades": len(self._trade_history),
            # [P39 2026-04-24] Strip tzinfo from loaded timestamps to keep
            # them comparable to naive datetime.now().
            "recent_demotions": len(
                [
                    d
                    for d in self._demotion_history
                    if _strip_tz(datetime.fromisoformat(d["timestamp"]))
                    > datetime.now() - timedelta(days=14)
                ]
            ),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_demotion(self):
        """Auto-demotion permanently disabled per operator directive 2026-04-30.

        Previously demoted ACTIVE→EXIT_ONLY on (a) 5 consecutive DRL losses or
        (b) 15% drawdown of DRL equity contribution. Both paths spuriously fired
        during Kraken outages (post_only mode → forced exits booked as DRL
        losses) — see CLAUDE.md HEALTH_T2 incident 2026-04-30. Manual promote()
        remains the only path to change authority level.
        """
        return

    def _demote(self, reason: str):
        """Execute demotion."""
        old = self._authority_level

        # Check for repeated demotions -> DISABLED
        # [P39 2026-04-24] Strip tzinfo from loaded ISO timestamps to keep
        # comparable with naive `cutoff` from datetime.now().
        cutoff = datetime.now() - timedelta(days=self.demotion_config.demotion_window_days)
        recent_demotions = [
            d
            for d in self._demotion_history
            if _strip_tz(datetime.fromisoformat(d["timestamp"])) > cutoff
        ]

        if len(recent_demotions) >= self.demotion_config.max_demotions_before_disable:
            self._authority_level = "DISABLED"
            logger.warning(
                f"[DRL_GATE] DISABLED: {len(recent_demotions)+1} demotions in "
                f"{self.demotion_config.demotion_window_days} days. Manual intervention required."
            )
        else:
            self._authority_level = self.demotion_config.demote_to

        self._demoted_at = datetime.now()
        self._demotion_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "from": old,
                "to": self._authority_level,
                "reason": reason,
            }
        )

        # Reset equity tracking
        self._peak_equity_contribution = 0.0
        self._current_equity_contribution = 0.0

        self._save_state()
        logger.warning(
            f"[DRL_GATE] DEMOTED: {old} -> {self._authority_level} | reason: {reason}"
        )

    def _auto_recover(self):
        """Auto-recover from demotion to ACTIVE."""
        old = self._authority_level
        if old == "DISABLED":
            return  # DISABLED requires manual recovery

        self._authority_level = "ACTIVE"
        self._demoted_at = None
        self._peak_equity_contribution = 0.0
        self._current_equity_contribution = 0.0

        self._save_state()
        logger.info(
            f"[DRL_GATE] AUTO-RECOVERED: {old} -> ACTIVE "
            f"(after {self.demotion_config.recovery_period_days} day cooldown)"
        )

    def _save_state(self):
        """Persist state to JSON (atomic write — see core/state_persistence.py)."""
        state = {
            "authority_level": self._authority_level,
            "demoted_at": self._demoted_at.isoformat() if self._demoted_at else None,
            "peak_equity": self._peak_equity_contribution,
            "current_equity": self._current_equity_contribution,
            "demotion_history": self._demotion_history[-20:],
            "trade_count": len(self._trade_history),
            "updated_at": datetime.now().isoformat(),
        }
        # [P37 2026-04-24] Was open(w)+json.dump → corrupt on crash mid-write.
        # save_state() uses tempfile + os.replace for atomicity.
        from core.state_persistence import save_state
        if not save_state(self.state_file, state):
            logger.error(f"[DRL_GATE] Failed to save state to {self.state_file}")

    def _load_state(self):
        """Load state from JSON.

        [P39 2026-04-24] datetime.fromisoformat() returns aware OR naive
        depending on whether the persisted ISO string carried a tz marker
        (`+00:00`). The runtime code uses naive `datetime.now()` everywhere,
        so we strip tzinfo on load to keep all comparisons type-consistent.
        Without this, an aware-loaded `_demoted_at` compared to naive
        `datetime.now()` (line 78, recovery deadline check) raises
        TypeError on every tick after restart, blocking auto-recovery.
        """
        try:
            if self.state_file.exists():
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                self._authority_level = state.get("authority_level", "DISABLED")
                if self._authority_level not in self.VALID_LEVELS:
                    self._authority_level = "DISABLED"
                demoted_str = state.get("demoted_at")
                if demoted_str:
                    _dt = datetime.fromisoformat(demoted_str)
                    if _dt.tzinfo is not None:
                        _dt = _dt.replace(tzinfo=None)
                    self._demoted_at = _dt
                else:
                    self._demoted_at = None
                self._peak_equity_contribution = state.get("peak_equity", 0.0)
                self._current_equity_contribution = state.get("current_equity", 0.0)
                # demotion_history timestamps are also stored as ISO strings;
                # they're parsed back via fromisoformat() in _demote/get_status.
                # Strip tz on those during read sites in those methods.
                self._demotion_history = state.get("demotion_history", [])
                logger.info(
                    f"[DRL_GATE] Loaded state: level={self._authority_level}, "
                    f"trades={state.get('trade_count', 0)}"
                )
        except Exception as e:
            logger.warning(f"[DRL_GATE] Failed to load state: {e}. Starting fresh.")

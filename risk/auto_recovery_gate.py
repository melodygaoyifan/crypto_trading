"""
risk/auto_recovery_gate.py - P3-03 AutoRecoveryGate

Cautious auto-recovery from non-drawdown HALT reasons.
Profile-gated: only active when config.auto_recovery is provided and enabled.

Design:
- ADDS latching to the NO_TRADE state machine (currently recalculated every tick)
- When engine.decide() returns NO_TRADE, records halt start (reason + timestamp)
- When engine says "ok to trade again", gate checks observation period + conditions
- Only CORRELATION_CRISIS / CORRELATION_COLLAPSE allow auto-recovery
- DRAWDOWN_HALT NEVER auto-recovers (requires manual intervention)
- Observation period (default 48h) + correlation < recovery_threshold + data integrity ok
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AutoRecoveryConfig:
    """Configuration for auto-recovery gate. Disabled by default."""
    enabled: bool = False
    allowed_reasons: List[str] = field(default_factory=list)
    never_auto_recover: List[str] = field(
        default_factory=lambda: ["DRAWDOWN_HALT"]
    )
    observation_hours: float = 48.0
    correlation_recovery_threshold: float = 0.90
    require_data_integrity_ok: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> AutoRecoveryConfig:
        if not d:
            return cls()
        return cls(
            enabled=d.get("enabled", False),
            allowed_reasons=d.get("allowed_reasons", []),
            never_auto_recover=d.get("never_auto_recover", ["DRAWDOWN_HALT"]),
            observation_hours=d.get("observation_hours", 48.0),
            correlation_recovery_threshold=d.get(
                "correlation_recovery_threshold", 0.90
            ),
            require_data_integrity_ok=d.get("require_data_integrity_ok", True),
        )


# ---------------------------------------------------------------------------
# Persisted halt state
# ---------------------------------------------------------------------------

@dataclass
class HaltState:
    """Tracks the current halt latch: reason, start time, last recovery."""
    halt_reason: str = ""
    halt_since_ts: float = 0.0
    last_auto_recovery_ts: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.halt_reason != "" and self.halt_since_ts > 0

    def to_dict(self) -> Dict:
        return {
            "halt_reason": self.halt_reason,
            "halt_since_ts": self.halt_since_ts,
            "last_auto_recovery_ts": self.last_auto_recovery_ts,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> HaltState:
        return cls(
            halt_reason=d.get("halt_reason", ""),
            halt_since_ts=d.get("halt_since_ts", 0.0),
            last_auto_recovery_ts=d.get("last_auto_recovery_ts", 0.0),
        )


# ---------------------------------------------------------------------------
# AutoRecoveryGate
# ---------------------------------------------------------------------------

class AutoRecoveryGate:
    """
    Latching gate for NO_TRADE / HALT recovery.

    Usage from main.py tick loop (AFTER engine.decide()):

        if intent.veto_active and gate.has_active_halt is False:
            gate.record_halt(intent.veto_reason, now_ts)

        if not intent.veto_active and gate.has_active_halt:
            ok, reason = gate.should_auto_recover(corr, integrity, now_ts)
            if ok:
                gate.clear_halt(now_ts)
                # allow intent to pass
            else:
                # override intent back to NO_TRADE (latch)
                intent.veto_active = True
                intent.veto_reason = f"[AUTO_RECOVERY_LATCH] {reason}"
    """

    def __init__(
        self,
        config: AutoRecoveryConfig,
        state_path: Optional[Path] = None,
    ):
        self.config = config
        self.state_path = state_path or Path("data/auto_recovery_state.json")
        self._state = self._load_state()

    # -- Public properties ---------------------------------------------------

    @property
    def has_active_halt(self) -> bool:
        return self._state.is_active

    @property
    def halt_state(self) -> HaltState:
        return self._state

    # -- Halt lifecycle ------------------------------------------------------

    def record_halt(self, veto_reason: str, timestamp: float) -> None:
        """
        Record the start of a halt.  Idempotent - only records the first
        occurrence (so the halt_since_ts stays at the *start* of the halt).
        """
        if self._state.is_active:
            return
        canonical = self._classify_reason(veto_reason)
        self._state.halt_reason = canonical
        self._state.halt_since_ts = timestamp
        self._persist_state()
        logger.info(
            f"[AUTO_RECOVERY] Halt recorded: reason={canonical}, "
            f"ts={timestamp:.0f}"
        )

    def should_auto_recover(
        self,
        current_correlation: float,
        data_integrity_ok: bool,
        now_ts: float,
    ) -> Tuple[bool, str]:
        """
        Check whether the current halt should be auto-recovered.

        Returns (should_recover, reason_string).
        """
        if not self.config.enabled:
            return False, "disabled"

        if not self._state.is_active:
            return False, "no_active_halt"

        reason = self._state.halt_reason

        # DRAWDOWN_HALT (and other never-recover reasons) NEVER auto-recover
        if reason in self.config.never_auto_recover:
            return False, f"never_auto_recover:{reason}"

        # Must be in allowed_reasons list
        if reason not in self.config.allowed_reasons:
            return False, f"reason_not_allowed:{reason}"

        # Check observation period
        hours_since = (now_ts - self._state.halt_since_ts) / 3600.0
        if hours_since < self.config.observation_hours:
            return (
                False,
                f"observation_period:{hours_since:.1f}h"
                f"<{self.config.observation_hours}h",
            )

        # Check correlation recovery (must drop BELOW threshold)
        if current_correlation >= self.config.correlation_recovery_threshold:
            return (
                False,
                f"corr_still_high:{current_correlation:.3f}"
                f">={self.config.correlation_recovery_threshold}",
            )

        # Check data integrity
        if self.config.require_data_integrity_ok and not data_integrity_ok:
            return False, "data_integrity_not_ok"

        return True, "auto_recover_ok"

    def clear_halt(self, now_ts: float) -> None:
        """Clear the halt latch after successful auto-recovery."""
        prev_reason = self._state.halt_reason
        self._state.last_auto_recovery_ts = now_ts
        self._state.halt_reason = ""
        self._state.halt_since_ts = 0.0
        self._persist_state()
        logger.warning(
            f"[AUTO_RECOVERY] Halt CLEARED - previous reason={prev_reason}, "
            f"recovery_ts={now_ts:.0f}"
        )

    # -- Proof log -----------------------------------------------------------

    def get_proof_snapshot(
        self,
        current_correlation: float,
        data_integrity_ok: bool,
        now_ts: float,
    ) -> Dict:
        """Return a snapshot dict for proof logging."""
        hours = 0.0
        if self._state.is_active:
            hours = (now_ts - self._state.halt_since_ts) / 3600.0

        recover_ok, recover_reason = self.should_auto_recover(
            current_correlation, data_integrity_ok, now_ts,
        )
        return {
            "halt_active": self._state.is_active,
            "halt_reason": self._state.halt_reason,
            "halt_since_ts": self._state.halt_since_ts,
            "hours_in_halt": round(hours, 1),
            "current_correlation": round(current_correlation, 4),
            "data_integrity_ok": data_integrity_ok,
            "recovery_check": recover_reason,
            "would_recover": recover_ok,
            "last_recovery_ts": self._state.last_auto_recovery_ts,
        }

    # -- Reason classification -----------------------------------------------

    @staticmethod
    def _classify_reason(veto_reason: str) -> str:
        """
        Map raw veto_reason string to canonical halt category.

        Sources:
        - "[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE"
        - "[HARD_VETO] correlation ..."
        - "[v3.6.1] NO_TRADE: DATA_INTEGRITY_FAIL"
        - drawdown-based veto reasons
        """
        upper = veto_reason.upper()
        if "CORRELATION_COLLAPSE" in upper:
            return "CORRELATION_COLLAPSE"
        if "CORRELATION_CRISIS" in upper:
            return "CORRELATION_CRISIS"
        # Generic correlation mention (hard veto path)
        if "CORRELATION" in upper and "DRAWDOWN" not in upper:
            return "CORRELATION_CRISIS"
        if "DRAWDOWN" in upper:
            return "DRAWDOWN_HALT"
        # Unknown - won't match allowed_reasons -> won't auto-recover
        return veto_reason

    # -- Persistence ---------------------------------------------------------

    def _load_state(self) -> HaltState:
        """Load persisted halt state from disk.

        [P92 2026-04-26] Distinguish "no file" (fresh start, OK) from
        "file exists but unreadable" (corruption, fail-closed). The
        previous behavior treated both cases the same — silently returning
        empty HaltState. If a P0 abort wrote the halt state, then the file
        got corrupted (truncated by SIGKILL, disk full, etc.), the gate
        would forget the halt and resume trading on next restart.
        """
        if not self.state_path.exists():
            return HaltState()
        try:
            with open(self.state_path) as f:
                data = json.load(f)
            state = HaltState.from_dict(data)
            if state.is_active:
                logger.info(
                    f"[AUTO_RECOVERY] Restored halt state: "
                    f"reason={state.halt_reason}, "
                    f"since={state.halt_since_ts:.0f}"
                )
            return state
        except Exception as e:
            # File exists but unreadable — fail closed: synthesize a halt
            # so operator must explicitly clear via manual recovery.
            # Better to over-halt than to silently resume during a P0
            # situation that may still be active.
            logger.critical(
                f"[AUTO_RECOVERY] State file {self.state_path} EXISTS but "
                f"is unreadable ({type(e).__name__}: {e}). Synthesizing a "
                f"halt with reason=STATE_CORRUPTION_DETECTED. Operator "
                f"action: investigate file contents, then manually clear "
                f"via clear_halt() if recovery is genuinely safe."
            )
            corrupted = HaltState()
            corrupted.halt_reason = "STATE_CORRUPTION_DETECTED"
            corrupted.halt_since_ts = time.time()
            return corrupted

    def _persist_state(self) -> None:
        """Atomic persist of halt state (tmp -> rename)."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(self._state.to_dict(), f, indent=2)
            tmp.replace(self.state_path)
        except Exception as e:
            logger.warning(f"[AUTO_RECOVERY] Failed to persist state: {e}")

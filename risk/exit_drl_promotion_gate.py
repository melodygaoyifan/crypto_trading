"""
risk/exit_drl_promotion_gate.py — Authority gate for the Exit DRL (System 3).

DELIBERATELY separate from `drl/promotion_gate.DRLPromotionGate` (System 1)
per CLAUDE.md P28: a TQC outage must not demote Exit-SAC, and vice versa.

Stage-3 v1: read-only. Tracks evidence (per-prediction outcomes via the ledger
in `data/exit_drl_outcome_ledger.jsonl`) and emits a "would_promote" recommendation
when ALL of the following hold:

  1. Shadow runtime ≥ MIN_SHADOW_DAYS (default 30).
  2. Closed exit events with logged Exit-SAC predictions ≥ MIN_EXIT_EVENTS (30).
  3. Offline validator (`training/exit_drl/validate_against_baseline.py`) reports
     Sharpe lift ≥ MIN_SHARPE_LIFT_PCT (default +10%) for the asset.
  4. Action distribution sanity (HOLD ∈ [50%, 90%], EXIT_ALL ≤ 30%).

The gate emits the recommendation but does NOT flip
`agents/exit_drl_agent.ExitDRLAgent.mode` automatically. Promotion remains a
human-in-the-loop action — the gate exists so the operator can verify the
preconditions in one place rather than eyeballing logs.

Usage:
  from risk.exit_drl_promotion_gate import ExitDRLPromotionGate
  gate = ExitDRLPromotionGate()
  recommendation = gate.evaluate(asset="BTC")
  # → {"would_promote": False, "blockers": ["min_shadow_days_unmet:5"], ...}
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Default thresholds — per spec Stage 3 + the Exit DRL training spec
# ────────────────────────────────────────────────────────────────
MIN_SHADOW_DAYS = 30
MIN_EXIT_EVENTS = 30
MIN_SHARPE_LIFT_PCT = 10.0    # DRL must beat rule-based by 10% in Sharpe
HOLD_RATIO_MIN = 0.50         # at least 50% HOLD (else over-active)
HOLD_RATIO_MAX = 0.90         # at most 90% HOLD (else asleep)
EXIT_ALL_RATIO_MAX = 0.30     # >30% EXIT_ALL = panic-exit pattern


@dataclass
class PromotionEvidence:
    asset: str
    shadow_days: float
    n_exit_events: int
    sharpe_lift_pct: Optional[float]
    mean_pnl_lift_pct: Optional[float]
    action_distribution: Dict[str, int]
    validator_run_at: Optional[float]


class ExitDRLPromotionGate:
    """Read-only gate. See module docstring for design notes."""

    def __init__(
        self,
        outcome_ledger_path: str = "data/exit_drl_outcome_ledger.jsonl",
        validator_dir: str = "data/exit_drl_validation",
        shadow_log_path: str = "data/exit_drl_shadow.jsonl",
        min_shadow_days: float = MIN_SHADOW_DAYS,
        min_exit_events: int = MIN_EXIT_EVENTS,
        min_sharpe_lift_pct: float = MIN_SHARPE_LIFT_PCT,
    ):
        self.outcome_ledger_path = Path(outcome_ledger_path)
        self.validator_dir = Path(validator_dir)
        self.shadow_log_path = Path(shadow_log_path)
        self.min_shadow_days = float(min_shadow_days)
        self.min_exit_events = int(min_exit_events)
        self.min_sharpe_lift_pct = float(min_sharpe_lift_pct)

    # ------------------------------------------------------------------
    # Evidence collection
    # ------------------------------------------------------------------
    def _shadow_days(self) -> float:
        """Return wall-clock days since the first line in the shadow log."""
        if not self.shadow_log_path.exists():
            return 0.0
        try:
            with self.shadow_log_path.open("r", encoding="utf-8") as f:
                first_line = f.readline()
            if not first_line.strip():
                return 0.0
            first_ts = json.loads(first_line).get("ts", time.time())
            return (time.time() - float(first_ts)) / 86400.0
        except Exception as e:
            logger.debug(f"[ExitDRLGate] shadow log read failed: {e}")
            return 0.0

    def _exit_events_for(self, asset: str) -> Dict:
        """Count closed exit events for `asset` in the outcome ledger."""
        if not self.outcome_ledger_path.exists():
            return {"n_events": 0, "action_distribution": {}}
        action_counts: Dict[str, int] = {}
        n_events = 0
        try:
            with self.outcome_ledger_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("asset") != asset:
                        continue
                    if not rec.get("closed", False):
                        continue
                    n_events += 1
                    a = rec.get("predicted_action_at_close", "UNKNOWN")
                    action_counts[a] = action_counts.get(a, 0) + 1
        except Exception as e:
            logger.debug(f"[ExitDRLGate] outcome ledger read failed: {e}")
        return {"n_events": n_events, "action_distribution": action_counts}

    def _validator_metrics(self, asset: str) -> Dict:
        """Load the most recent validator run for `asset`, if any."""
        path = self.validator_dir / f"{asset}_validation.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["_loaded_at"] = path.stat().st_mtime
            return data
        except Exception as e:
            logger.debug(f"[ExitDRLGate] validator read failed: {e}")
            return {}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, asset: str) -> Dict:
        """Return a dict with the promotion recommendation + raw evidence."""
        shadow_days = self._shadow_days()
        exit_evidence = self._exit_events_for(asset)
        validator = self._validator_metrics(asset)

        sharpe_lift = None
        mean_pnl_lift = None
        if validator:
            sharpe_lift = (validator.get("lift_pct") or {}).get("sharpe")
            mean_pnl_lift = (validator.get("lift_pct") or {}).get("mean_pnl_bps")

        evidence = PromotionEvidence(
            asset=asset,
            shadow_days=round(shadow_days, 2),
            n_exit_events=exit_evidence["n_events"],
            sharpe_lift_pct=sharpe_lift,
            mean_pnl_lift_pct=mean_pnl_lift,
            action_distribution=exit_evidence["action_distribution"],
            validator_run_at=validator.get("_loaded_at"),
        )

        blockers: List[str] = []

        # 1. Shadow days
        if shadow_days < self.min_shadow_days:
            blockers.append(
                f"min_shadow_days_unmet: {shadow_days:.1f} < {self.min_shadow_days}"
            )

        # 2. Min closed exit events
        if exit_evidence["n_events"] < self.min_exit_events:
            blockers.append(
                f"min_exit_events_unmet: {exit_evidence['n_events']} < {self.min_exit_events}"
            )

        # 3. Validator sharpe lift
        if sharpe_lift is None:
            blockers.append("validator_not_run: run training/exit_drl/validate_against_baseline.py")
        elif sharpe_lift < self.min_sharpe_lift_pct:
            blockers.append(
                f"sharpe_lift_unmet: {sharpe_lift:+.1f}% < +{self.min_sharpe_lift_pct}%"
            )

        # 4. Action distribution sanity (only meaningful with enough events)
        if exit_evidence["n_events"] >= self.min_exit_events:
            total = exit_evidence["n_events"]
            hold_n = exit_evidence["action_distribution"].get("HOLD", 0)
            exit_all_n = exit_evidence["action_distribution"].get("EXIT_ALL", 0)
            hold_ratio = hold_n / total
            exit_all_ratio = exit_all_n / total
            if hold_ratio < HOLD_RATIO_MIN:
                blockers.append(f"hold_ratio_low: {hold_ratio:.2%} < {HOLD_RATIO_MIN:.0%}")
            if hold_ratio > HOLD_RATIO_MAX:
                blockers.append(f"hold_ratio_high: {hold_ratio:.2%} > {HOLD_RATIO_MAX:.0%}")
            if exit_all_ratio > EXIT_ALL_RATIO_MAX:
                blockers.append(
                    f"exit_all_ratio_high: {exit_all_ratio:.2%} > {EXIT_ALL_RATIO_MAX:.0%}"
                )

        return {
            "asset": asset,
            "would_promote": len(blockers) == 0,
            "blockers": blockers,
            "evidence": {
                "shadow_days": evidence.shadow_days,
                "n_exit_events": evidence.n_exit_events,
                "sharpe_lift_pct": evidence.sharpe_lift_pct,
                "mean_pnl_lift_pct": evidence.mean_pnl_lift_pct,
                "action_distribution": evidence.action_distribution,
                "validator_run_at": evidence.validator_run_at,
            },
            "thresholds": {
                "min_shadow_days": self.min_shadow_days,
                "min_exit_events": self.min_exit_events,
                "min_sharpe_lift_pct": self.min_sharpe_lift_pct,
            },
        }

    def evaluate_all(self, assets: Optional[List[str]] = None) -> Dict:
        assets = assets or ["BTC", "ETH", "SOL"]
        return {a: self.evaluate(a) for a in assets}

    # ------------------------------------------------------------------
    # Manual override audit trail
    # ------------------------------------------------------------------
    def record_override(
        self,
        asset: str,
        reason: str,
        ts: Optional[float] = None,
        state_path: str = "data/exit_drl_promotion_state.json",
    ) -> None:
        """
        Record an explicit override that promoted an asset to EXIT_ONLY without
        clearing all gate blockers. Required by CLAUDE.md P29 — every
        accelerated-path promotion must leave an auditable stamp.
        """
        from pathlib import Path as _P
        path = _P(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
        record = {
            "force_promote_at": float(ts or time.time()),
            "reason": str(reason),
            "blockers_at_override": self.evaluate(asset).get("blockers", []),
        }
        existing.setdefault("overrides", {})[asset] = record
        try:
            # [P92 2026-04-26] Was path.write_text() (no flush/fsync) +
            # except: logger.debug() (operator-invisible). The override
            # is the audit-trail SOURCE OF TRUTH — silent loss means the
            # operator believes BTC was promoted but the kill switch sees
            # no override record on next restart. Use atomic save_state
            # helper (P83-verified atomic + fsync) and surface failures
            # at WARNING so operator knows the override didn't persist.
            try:
                from core.state_persistence import save_state
                save_state(str(path), existing)
            except Exception:
                # Fallback to write_text + fsync if save_state import
                # fails (preserves test-mode and bootstrapping paths).
                import os
                with open(path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(existing, indent=2))
                    f.flush()
                    os.fsync(f.fileno())
            logger.warning(
                f"[EXIT_DRL_GATE] {asset}: override recorded — "
                f"blockers at override = {record['blockers_at_override']}"
            )
        except Exception as e:
            logger.warning(
                f"[EXIT_DRL_GATE] {asset}: override write FAILED "
                f"({type(e).__name__}: {e}). Audit trail NOT persisted; "
                f"on restart the kill switch will not see this override. "
                f"Operator action: re-run the override after fixing the "
                f"underlying I/O issue (disk full / permissions / "
                f"symlink)."
            )

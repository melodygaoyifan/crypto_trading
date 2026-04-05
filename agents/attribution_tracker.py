"""
AttributionTracker - Per-agent alpha attribution scoring.

Records each agent's directional signals per tick and resolves them
against realized price moves to compute rolling hit-rates and
confidence calibration metrics.

Usage:
    tracker = get_attribution_tracker()
    tracker.record_signals(tick_id, asset, envelopes)
    # ... next tick ...
    tracker.resolve_outcome(prev_tick_id, asset, price_change_pct)

    # Decay alerts
    for name in tracker.get_decay_alerts(threshold=0.45):
        logger.warning(f"Agent {name} rolling hit-rate below threshold")
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.signal_envelope import AgentSignalEnvelope

logger = logging.getLogger("hmats.attribution")


# ---------------------------------------------------------------------------
# AgentScorecard
# ---------------------------------------------------------------------------

@dataclass
class AgentScorecard:
    """Running performance record for a single agent."""

    agent_name: str
    total_signals: int = 0
    correct_signals: int = 0
    hit_rate: float = 0.0
    avg_confidence_when_correct: float = 0.0
    avg_confidence_when_wrong: float = 0.0
    hit_rate_by_regime: Dict[str, float] = field(default_factory=dict)
    last_30_trades: deque = field(default_factory=lambda: deque(maxlen=30))
    rolling_hit_rate: float = 0.0

    # Internal accumulators (not exposed in to_dict)
    _correct_conf_sum: float = field(default=0.0, repr=False)
    _wrong_conf_sum: float = field(default=0.0, repr=False)
    _regime_totals: Dict[str, int] = field(default_factory=dict, repr=False)
    _regime_corrects: Dict[str, int] = field(default_factory=dict, repr=False)

    def update(self, correct: bool, confidence: float, regime: str = "unknown") -> None:
        """Record one signal outcome."""
        self.total_signals += 1
        self.last_30_trades.append(1 if correct else 0)

        if correct:
            self.correct_signals += 1
            self._correct_conf_sum += confidence
        else:
            self._wrong_conf_sum += confidence

        # Overall hit rate
        self.hit_rate = self.correct_signals / self.total_signals if self.total_signals else 0.0

        # Rolling hit rate
        if self.last_30_trades:
            self.rolling_hit_rate = sum(self.last_30_trades) / len(self.last_30_trades)

        # Avg confidence
        if self.correct_signals > 0:
            self.avg_confidence_when_correct = self._correct_conf_sum / self.correct_signals
        wrong_count = self.total_signals - self.correct_signals
        if wrong_count > 0:
            self.avg_confidence_when_wrong = self._wrong_conf_sum / wrong_count

        # Per-regime hit rate
        self._regime_totals[regime] = self._regime_totals.get(regime, 0) + 1
        if correct:
            self._regime_corrects[regime] = self._regime_corrects.get(regime, 0) + 1
        rt = self._regime_totals[regime]
        rc = self._regime_corrects.get(regime, 0)
        self.hit_rate_by_regime[regime] = rc / rt if rt else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "total_signals": self.total_signals,
            "correct_signals": self.correct_signals,
            "hit_rate": round(self.hit_rate, 4),
            "avg_confidence_when_correct": round(self.avg_confidence_when_correct, 4),
            "avg_confidence_when_wrong": round(self.avg_confidence_when_wrong, 4),
            "hit_rate_by_regime": {k: round(v, 4) for k, v in self.hit_rate_by_regime.items()},
            "rolling_hit_rate": round(self.rolling_hit_rate, 4),
            "rolling_window_size": len(self.last_30_trades),
        }


# ---------------------------------------------------------------------------
# AttributionTracker
# ---------------------------------------------------------------------------

class AttributionTracker:
    """Records agent signals per tick and resolves against price outcomes."""

    def __init__(self, log_dir: str = "logs/attribution") -> None:
        self._log_dir = log_dir
        self._pending: Dict[str, List[AgentSignalEnvelope]] = {}
        self._scorecard: Dict[str, AgentScorecard] = {}
        self._lock = threading.Lock()

        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as exc:
            logger.warning("attribution log_dir creation failed: %s", exc)

    # ---- Record ----

    def record_signals(
        self,
        tick_id: str,
        asset: str,
        envelopes: List[AgentSignalEnvelope],
    ) -> None:
        """Store envelopes for later outcome resolution and write JSONL."""
        key = f"{asset}_{tick_id}"
        with self._lock:
            self._pending[key] = list(envelopes)

        # Write signal log
        try:
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            path = os.path.join(self._log_dir, f"signals_{date_str}.jsonl")
            records = [env.to_audit_record() for env in envelopes]
            entry = {
                "tick_id": tick_id,
                "asset": asset,
                "ts": datetime.now(timezone.utc).isoformat(),
                "signals": records,
            }
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fh.flush()
        except Exception as exc:
            logger.warning("attribution signal write failed: %s", exc)

    # ---- Resolve ----

    def resolve_outcome(
        self,
        tick_id: str,
        asset: str,
        price_change_pct: float,
        regime: str = "unknown",
    ) -> None:
        """Compare stored signals against realized price change.

        A signal is *correct* if ``direction * price_change_pct > 0``.
        Signals with ``direction == 0`` are skipped (no directional claim).
        """
        key = f"{asset}_{tick_id}"
        with self._lock:
            envelopes = self._pending.pop(key, None)

        if not envelopes:
            return

        outcomes: List[Dict[str, Any]] = []

        for env in envelopes:
            if env.direction == 0.0:
                continue  # no directional claim -> skip

            correct = (env.direction * price_change_pct) > 0

            # Update scorecard
            with self._lock:
                sc = self._scorecard.get(env.agent_name)
                if sc is None:
                    sc = AgentScorecard(agent_name=env.agent_name)
                    self._scorecard[env.agent_name] = sc
                sc.update(correct, env.confidence, regime)

            outcomes.append({
                "agent_name": env.agent_name,
                "direction": env.direction,
                "confidence": env.confidence,
                "price_change_pct": round(price_change_pct, 6),
                "correct": correct,
                "regime": regime,
            })

        # Write outcome log
        if outcomes:
            try:
                date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
                path = os.path.join(self._log_dir, f"outcomes_{date_str}.jsonl")
                entry = {
                    "tick_id": tick_id,
                    "asset": asset,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "price_change_pct": round(price_change_pct, 6),
                    "regime": regime,
                    "outcomes": outcomes,
                }
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    fh.flush()
            except Exception as exc:
                logger.warning("attribution outcome write failed: %s", exc)

    # ---- Queries ----

    def get_scorecard(self, agent_name: str) -> Optional[AgentScorecard]:
        with self._lock:
            return self._scorecard.get(agent_name)

    def get_all_scorecards(self) -> Dict[str, Dict]:
        with self._lock:
            return {name: sc.to_dict() for name, sc in self._scorecard.items()}

    def get_decay_alerts(self, threshold: float = 0.45) -> List[str]:
        """Return agent names whose rolling hit-rate is below *threshold*.

        Only considers agents with at least 10 resolved signals to avoid
        noisy alerts during cold start.
        """
        alerts: List[str] = []
        with self._lock:
            for name, sc in self._scorecard.items():
                if len(sc.last_30_trades) >= 10 and sc.rolling_hit_rate < threshold:
                    alerts.append(name)
        return alerts

    def pending_count(self) -> int:
        """Number of tick-keys awaiting outcome resolution."""
        with self._lock:
            return len(self._pending)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_tracker_instance: Optional[AttributionTracker] = None
_tracker_lock = threading.Lock()


def get_attribution_tracker(log_dir: str = "logs/attribution") -> AttributionTracker:
    """Return the module-level singleton AttributionTracker."""
    global _tracker_instance
    if _tracker_instance is None:
        with _tracker_lock:
            if _tracker_instance is None:
                _tracker_instance = AttributionTracker(log_dir=log_dir)
    elif getattr(_tracker_instance, "_log_dir", None) != log_dir:
        with _tracker_lock:
            if getattr(_tracker_instance, "_log_dir", None) != log_dir:
                _tracker_instance = AttributionTracker(log_dir=log_dir)
    return _tracker_instance


def reset_attribution_tracker() -> None:
    """Reset the singleton (for testing)."""
    global _tracker_instance
    with _tracker_lock:
        _tracker_instance = None

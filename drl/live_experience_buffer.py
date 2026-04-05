"""
LiveExperienceBuffer - passive experience collection for DRL shadow mode.

Follows ShadowLedgerWriter JSONL pattern: thread-safe buffer,
periodic flush, date-rotated files.

Records DRL observations + actions during shadow mode for
future offline fine-tuning.
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("HMATS.ExperienceBuffer")


class LiveExperienceBuffer:
    """Thread-safe JSONL writer for live DRL experiences."""

    BUFFER_SIZE = 50
    FLUSH_INTERVAL = 60  # seconds

    def __init__(
        self,
        output_dir: str = "data/live_experiences",
        buffer_size: int = BUFFER_SIZE,
        flush_interval: int = FLUSH_INTERVAL,
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval

        self._buffer: list = []
        self._lock = threading.Lock()
        self._last_flush = time.time()
        self._total_records = 0

        logger.info(
            f"[ExperienceBuffer] Ready: dir={self._output_dir}, "
            f"buffer={buffer_size}, flush_interval={flush_interval}s"
        )

    def _append_entry(self, entry: dict[str, Any], *, flush_on_interval: bool = True) -> None:
        """Append an experience entry and flush when buffer/interval thresholds are hit."""
        with self._lock:
            self._buffer.append(entry)
            self._total_records += 1

            if (
                len(self._buffer) >= self._buffer_size
                or (flush_on_interval and time.time() - self._last_flush > self._flush_interval)
            ):
                self._flush_locked()

    def record(
        self,
        asset: str,
        timestamp: str,
        obs_126: list,
        action: float,
        regime: str,
        regime_confidence: float,
        market_snapshot: dict,
        tqc_uncertainty: float = 0.0,
    ):
        """Record a DRL observation + action pair.

        Args:
            asset: "BTC", "ETH", or "SOL"
            timestamp: ISO format timestamp
            obs_126: 126-dim observation as list (JSON-serializable)
            action: TQC action [-1, 1]
            regime: Current regime name
            regime_confidence: GMM confidence
            market_snapshot: dict with close price, volume, etc.
            tqc_uncertainty: TQC quantile spread ratio
        """
        entry = {
            "type": "observation",
            "asset": asset,
            "timestamp": timestamp,
            "obs": obs_126,
            "action": round(action, 6),
            "regime": regime,
            "regime_confidence": round(regime_confidence, 4),
            "tqc_uncertainty": round(tqc_uncertainty, 4),
            "close": market_snapshot.get("close", 0.0),
        }
        self._append_entry(entry)

    def record_decision(
        self,
        asset: str,
        timestamp: str,
        action: float,
        regime: str,
        mode: str,
        direction: int,
        position_size_pct: float,
        veto_active: bool,
        signal_strength: float = 0.0,
        sentiment_score: float = 0.0,
        state: Optional[list] = None,
        strategy: str = "",
        alpha_bps: Optional[float] = None,
        alpha_threshold_bps: Optional[float] = None,
        alpha_gate_passed: Optional[bool] = None,
        verdict: str = "",
        veto_reason: str = "",
    ):
        """Record a finalized runtime decision for Step 15 paper-run evidence."""
        entry = {
            "type": "decision",
            "asset": asset,
            "timestamp": timestamp,
            "action": round(action, 6),
            "regime": regime,
            "mode": mode,
            "direction": int(direction),
            "position_size_pct": round(position_size_pct, 6),
            "signal_strength": round(signal_strength, 6),
            "veto_active": bool(veto_active),
            "sentiment_score": round(sentiment_score, 6),
            "strategy": strategy or "",
            "verdict": verdict or "",
            "veto_reason": veto_reason or "",
        }
        if state is not None:
            entry["state"] = state
        if alpha_bps is not None:
            entry["alpha_bps"] = round(alpha_bps, 4)
        if alpha_threshold_bps is not None:
            entry["alpha_threshold_bps"] = round(alpha_threshold_bps, 4)
        if alpha_gate_passed is not None:
            entry["alpha_gate_passed"] = bool(alpha_gate_passed)
        self._append_entry(entry)

    def record_outcome(
        self,
        asset: str,
        timestamp: str,
        realized_pnl: float,
        bars_held: int,
        realized_pnl_bps: Optional[float] = None,
        exit_reason: str = "",
    ):
        """Record trade outcome for linking to prior observation.

        Args:
            asset: "BTC", "ETH", or "SOL"
            timestamp: ISO format timestamp of the trade close
            realized_pnl: Realized PnL from the trade
            bars_held: Number of 4H bars the position was held
        """
        entry = {
            "type": "outcome",
            "asset": asset,
            "timestamp": timestamp,
            "realized_pnl": round(realized_pnl, 4),
            "bars_held": bars_held,
        }
        if realized_pnl_bps is not None:
            entry["pnl_bps"] = round(realized_pnl_bps, 4)
        if exit_reason:
            entry["exit_reason"] = exit_reason
        self._append_entry(entry, flush_on_interval=False)

    def flush(self):
        """Force flush buffer to disk."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        """Write buffer to date-rotated JSONL file. Must hold self._lock."""
        if not self._buffer:
            return

        date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = self._output_dir / f"experiences_{date_str}.jsonl"

        try:
            with open(filepath, "a", encoding="utf-8") as f:
                for entry in self._buffer:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
            logger.debug(
                f"[ExperienceBuffer] Flushed {len(self._buffer)} records to {filepath}"
            )
        except Exception as e:
            logger.warning(f"[ExperienceBuffer] Flush failed: {e}")

        self._buffer.clear()
        self._last_flush = time.time()

    @property
    def total_records(self) -> int:
        return self._total_records

    def get_status(self) -> dict:
        """Status for dashboard/logging."""
        return {
            "total_records": self._total_records,
            "buffer_pending": len(self._buffer),
            "output_dir": str(self._output_dir),
        }

"""
IC Signal Logger — per-4H-bar snapshot of every agent's signal for offline
Information Coefficient computation.

Usage (from fusion engine, after all agents have produced their outputs):

    from analytics.ic.ic_logger import log_bar_safe

    log_bar_safe(
        asset='BTC',
        timestamp=datetime.now(timezone.utc),
        current_price=67450.0,
        agent_signals={
            'quant': {'direction': 0.7, 'confidence': 0.5, ...},
            'drl':   {'direction': -0.3, 'confidence': 0.4, ...},
            ...
        },
        regime={'phase': 'DISTRIBUTION', 'regime_id': 3},
        fusion_output={'final_direction': -0.41, 'final_confidence': 0.67},
    )

Properties:
- Async write via background thread (non-blocking on hot path).
- Drops records (with periodic warning log) when queue full.
- Daily rollover (logs/ic_signals/ic_signals_YYYY-MM-DD.jsonl).
- log_bar_safe NEVER raises — failures are swallowed; trading is never affected.

IRON LAW (per CLAUDE.md): IC logging is analytics-only. It must not:
- Modify any fusion decision (read-only consumption of agent outputs)
- Change obs_dim, constitution.py, or any decision threshold
- Block the hot path under any circumstance (including disk-full / OOM)
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_IC_LOG_DIR = Path("logs/ic_signals")


class _ICLoggerSingleton:
    """Singleton IC logger with async background flush."""

    _instance: Optional["_ICLoggerSingleton"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        try:
            _IC_LOG_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"[IC_LOGGER] Failed to create log dir: {e}")
        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="ic_logger_writer"
        )
        self._thread.start()
        self._dropped = 0
        self._written = 0

    def _current_log_file(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return _IC_LOG_DIR / f"ic_signals_{date_str}.jsonl"

    def _writer_loop(self):
        while not self._stop.is_set():
            try:
                record = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if record is None:
                break
            try:
                logfile = self._current_log_file()
                with logfile.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")
                self._written += 1
            except Exception as e:
                # Best-effort. Even if disk is full or path missing, don't crash.
                logger.debug(f"[IC_LOGGER] write failed: {e}")

    def log_bar(
        self,
        asset: str,
        timestamp: datetime,
        current_price: float,
        agent_signals: Dict[str, Any],
        regime: Dict[str, Any],
        fusion_output: Dict[str, Any],
    ) -> None:
        """Enqueue one bar snapshot. Non-blocking."""
        ts_str = (
            timestamp.isoformat()
            if isinstance(timestamp, datetime)
            else str(timestamp)
        )
        bar_id = (
            f"{asset}_{timestamp.strftime('%Y-%m-%dT%H:%M')}"
            if isinstance(timestamp, datetime)
            else f"{asset}_{timestamp}"
        )
        record = {
            "timestamp": ts_str,
            "asset": asset,
            "bar_id": bar_id,
            "current_price": float(current_price),
            "agent_signals": agent_signals,
            "regime": regime,
            "fusion_output": fusion_output,
            "future_returns": None,  # populated by offline analyzer
        }
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 100 == 1:
                logger.warning(
                    f"[IC_LOGGER] queue full, dropped {self._dropped} records "
                    f"(written so far: {self._written})"
                )

    def status(self) -> Dict[str, Any]:
        return {
            "queue_size": self._queue.qsize(),
            "queue_capacity": self._queue.maxsize,
            "written": self._written,
            "dropped": self._dropped,
            "log_dir": str(_IC_LOG_DIR),
            "writer_alive": self._thread.is_alive() if self._thread else False,
        }

    def shutdown(self, timeout: float = 5.0):
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)


def get_ic_logger() -> _ICLoggerSingleton:
    return _ICLoggerSingleton()


def log_bar_safe(**kwargs) -> None:
    """
    Hot-path-safe IC logger entrypoint. Swallows every exception.

    Use this in fusion engine; calling it must NEVER affect trading state.
    """
    try:
        get_ic_logger().log_bar(**kwargs)
    except Exception as e:
        # Never propagate. IC logging is observability, not critical.
        logger.debug(f"[IC_LOGGER] log_bar_safe ignored {type(e).__name__}: {e}")

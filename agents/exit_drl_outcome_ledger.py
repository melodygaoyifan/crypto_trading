"""
agents/exit_drl_outcome_ledger.py — Per-prediction outcome ledger writer.

Combines the SHADOW-mode predictions (`data/exit_drl_shadow.jsonl`) with the
realized close outcomes from the runner. Produces
`data/exit_drl_outcome_ledger.jsonl` — one record per closed position with:

  {
    "asset", "open_ts", "close_ts",
    "entry_price", "exit_price", "direction",
    "bars_held", "realized_pnl_bps", "exit_reason",
    "predicted_action_at_close": <last Exit-SAC prediction before close>,
    "predicted_action_at_peak":  <last Exit-SAC prediction at peak unrealized PnL>,
    "predicted_actions_history": <full trajectory of predictions during the trade>,
    "closed": True,
  }

Used by `risk/exit_drl_promotion_gate.py` to count exit events + measure how
often Exit-SAC's prediction matched the rule-based exit reason.

Wiring (separate task — this file is just the writer):
- Runner calls `record_open(asset, ts, entry_price, direction)` when a position opens.
- Each tick that produces a prediction, runner calls `record_prediction(asset, prediction)`.
- When position closes, runner calls `record_close(asset, ts, exit_price, exit_reason, realized_pnl_bps)`.
- The ledger flushes on every close — at-most-one open trade per asset at a time
  (matches HMATS's single-position-per-asset model).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OpenTrade:
    asset: str
    open_ts: float
    entry_price: float
    direction: int
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    peak_unrealized_pnl: float = 0.0
    pred_at_peak: Optional[Dict[str, Any]] = None


class ExitDRLOutcomeLedger:
    """In-memory tracker + JSONL flusher. One open trade per asset max."""

    def __init__(self, ledger_path: str = "data/exit_drl_outcome_ledger.jsonl"):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._open: Dict[str, OpenTrade] = {}

    # ------------------------------------------------------------------
    # Open / track / close
    # ------------------------------------------------------------------
    def record_open(
        self,
        asset: str,
        entry_price: float,
        direction: int,
        ts: Optional[float] = None,
    ) -> None:
        if asset in self._open:
            # Defensive: a previous close was missed; abandon the old open.
            logger.debug(
                f"[EXIT_DRL_LEDGER] {asset}: stale open trade abandoned "
                f"(no close observed; predictions={len(self._open[asset].predictions)})"
            )
        self._open[asset] = OpenTrade(
            asset=asset, open_ts=ts or time.time(),
            entry_price=float(entry_price),
            direction=int(direction),
        )

    def record_prediction(
        self,
        asset: str,
        action_name: str,
        confidence: float,
        unrealized_pnl: float,
        bars_held: int,
        ts: Optional[float] = None,
    ) -> None:
        """Append one prediction snapshot. No-op if no open trade for this asset."""
        trade = self._open.get(asset)
        if trade is None:
            return
        snapshot = {
            "ts": ts or time.time(),
            "action": action_name,
            "confidence": round(float(confidence), 4),
            "unrealized_pnl": round(float(unrealized_pnl), 6),
            "bars_held": int(bars_held),
        }
        trade.predictions.append(snapshot)
        # Track peak unrealized + the prediction sitting on it
        if float(unrealized_pnl) > trade.peak_unrealized_pnl:
            trade.peak_unrealized_pnl = float(unrealized_pnl)
            trade.pred_at_peak = snapshot

    def record_close(
        self,
        asset: str,
        exit_price: float,
        exit_reason: str,
        realized_pnl_bps: float,
        ts: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Flush one closed-trade record to the ledger. Returns the record."""
        trade = self._open.pop(asset, None)
        if trade is None:
            return None
        close_ts = ts or time.time()
        bars_held = trade.predictions[-1]["bars_held"] if trade.predictions else 0
        last_pred = trade.predictions[-1] if trade.predictions else None
        record = {
            "asset": asset,
            "open_ts": round(trade.open_ts, 3),
            "close_ts": round(close_ts, 3),
            "entry_price": trade.entry_price,
            "exit_price": float(exit_price),
            "direction": trade.direction,
            "bars_held": int(bars_held),
            "realized_pnl_bps": round(float(realized_pnl_bps), 3),
            "exit_reason": str(exit_reason),
            "predicted_action_at_close": last_pred["action"] if last_pred else None,
            "predicted_action_at_peak": (
                trade.pred_at_peak["action"] if trade.pred_at_peak else None
            ),
            "n_predictions": len(trade.predictions),
            "peak_unrealized_pnl": round(trade.peak_unrealized_pnl, 6),
            "closed": True,
        }
        try:
            with self.ledger_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug(f"[EXIT_DRL_LEDGER] {asset}: flush failed: {e}")
        return record

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def open_assets(self) -> List[str]:
        return list(self._open.keys())

    def status(self) -> Dict[str, Any]:
        return {
            "ledger_path": str(self.ledger_path),
            "open_assets": self.open_assets(),
            "open_predictions": {a: len(t.predictions) for a, t in self._open.items()},
        }


# Module-level singleton
_singleton: Optional[ExitDRLOutcomeLedger] = None


def get_outcome_ledger(
    ledger_path: str = "data/exit_drl_outcome_ledger.jsonl",
) -> ExitDRLOutcomeLedger:
    global _singleton
    if _singleton is None:
        _singleton = ExitDRLOutcomeLedger(ledger_path=ledger_path)
    return _singleton


def reset_singleton() -> None:
    global _singleton
    _singleton = None

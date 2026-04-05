"""
Paper Position Update Logic - extracted from main.py _execute_intent Phase 5.

Contains the four-branch paper position update logic:
  BRANCH A: Full exit (target_exposure==0 or direction==0)
  BRANCH B: Partial exit (same direction, exposure shrinking)
  BRANCH C: Entry or Scale-in (new position / increasing)
  BRANCH D: Noise rebalance (position unchanged, skip)

This module provides pure functions that compute position changes
without modifying any global state. The caller (main.py) is responsible
for persisting the results.

NOTE: This is a PARTIAL extraction. The full _execute_intent method
(~3100 lines) remains in main.py but delegates branch logic here.
Future work will extract the entire execution pipeline.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def classify_position_branch(
    *,
    intent_target_exposure: float,
    intent_direction: float,
    old_pos: Optional[Dict[str, Any]],
    notional_usd: float,
    noise_threshold_usd: float = 50.0,
) -> str:
    """
    Classify which branch to execute for paper position update.

    Returns one of: 'FULL_EXIT', 'PARTIAL_EXIT', 'ENTRY', 'NOISE', 'NONE'
    """
    is_full_exit = (intent_target_exposure == 0 or intent_direction == 0)
    direction_sign = 1 if intent_direction > 0 else (-1 if intent_direction < 0 else 0)

    if is_full_exit:
        return 'FULL_EXIT'

    if old_pos and old_pos.get("direction", 0) != 0:
        old_notional = float(old_pos.get("notional", 0) or 0)
        same_direction = old_pos.get("direction", 0) * direction_sign > 0

        if same_direction and old_notional > 0:
            # Check shrinking FIRST (partial exit), then noise
            if notional_usd < old_notional:
                return 'PARTIAL_EXIT'
            if abs(notional_usd - old_notional) < noise_threshold_usd:
                return 'NOISE'

    return 'ENTRY'


def compute_realized_pnl(
    *,
    old_pos: Dict[str, Any],
    fill_price: float,
    exit_notional: float,
    entry_fee_usd: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute realized PnL for a closed/partially-closed position.

    Returns dict with:
        pnl_usd: float - absolute PnL in USD
        pnl_bps: float - PnL in basis points
        exit_fee_usd: float - estimated exit fee
    """
    entry_price = float(old_pos.get("entry_price", fill_price) or fill_price)
    direction = int(old_pos.get("direction", 1) or 1)
    notional = float(old_pos.get("notional", exit_notional) or exit_notional)

    if entry_price <= 0 or notional <= 0:
        return {"pnl_usd": 0.0, "pnl_bps": 0.0, "exit_fee_usd": 0.0}

    if direction > 0:  # Long
        price_change_pct = (fill_price - entry_price) / entry_price
    else:  # Short
        price_change_pct = (entry_price - fill_price) / entry_price

    pnl_usd = notional * price_change_pct
    pnl_bps = price_change_pct * 10000

    return {
        "pnl_usd": pnl_usd,
        "pnl_bps": pnl_bps,
        "exit_fee_usd": 0.0,  # caller computes actual fee
    }


def build_new_position(
    *,
    asset: str,
    direction_sign: int,
    exposure_fraction: float,
    notional_usd: float,
    fill_price: float,
    strategy_id: str = "",
    tranche_level: int = 1,
) -> Dict[str, Any]:
    """Build a new paper position dict for BRANCH C (entry)."""
    return {
        "exposure": exposure_fraction,
        "direction": direction_sign,
        "notional": notional_usd,
        "entry_price": fill_price,
        "avg_price": fill_price,
        "tranche": tranche_level,
        "strategy": strategy_id,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "last_update": datetime.now(timezone.utc).isoformat(),
        "unrealized_pnl_usd": 0.0,
        "unrealized_pnl_bps": 0.0,
        "realized_pnl_usd": 0.0,
    }


def build_partial_exit_position(
    *,
    old_pos: Dict[str, Any],
    new_exposure: float,
    new_notional: float,
    fill_price: float,
) -> Dict[str, Any]:
    """Build updated position dict for BRANCH B (partial exit)."""
    pos = dict(old_pos)
    pos["exposure"] = new_exposure
    pos["notional"] = new_notional
    pos["last_update"] = datetime.now(timezone.utc).isoformat()
    # Keep entry_price and avg_price from original
    return pos

"""
================================================================================
CONSTANTS - Shared Constants Extracted from main.py
================================================================================
Version: 1.0.0
Purpose: Single location for schema keys, rule tables, emergency thresholds,
         and other configuration constants used across the tick pipeline.

Extracted from main.py lines 1340-1516 (Phase 1 of God Object decomposition).
================================================================================
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Import canonical config values (with fallbacks)
try:
    from configs.canonical_config import (
        MODE_TTL_MAX_BARS, MODE_TTL_CONFIDENCE_DECAY,
        MODE_TTL_CONFIDENCE_FLOOR, MODE_TTL_BLEND_BARS,
        DEADLOCK_T1_FORCE_BARS, DEADLOCK_T2_ABORT_BARS,
        DEADLOCK_EDGE_DECAY_PER_BAR,
        SOL_VPIN_IMMEDIATE, SOL_CORRELATION_IMMEDIATE, SOL_FLASH_MOVE_PCT,
        OPEN_FEE_PCT, ROLLOVER_FEE_PCT,
    )
except ImportError:
    MODE_TTL_MAX_BARS = 4
    MODE_TTL_CONFIDENCE_DECAY = 0.15
    MODE_TTL_CONFIDENCE_FLOOR = 0.20
    MODE_TTL_BLEND_BARS = 1
    DEADLOCK_T1_FORCE_BARS = 2
    DEADLOCK_T2_ABORT_BARS = 3
    DEADLOCK_EDGE_DECAY_PER_BAR = 0.15
    SOL_VPIN_IMMEDIATE = 0.90
    SOL_CORRELATION_IMMEDIATE = 0.92
    SOL_FLASH_MOVE_PCT = 0.03
    OPEN_FEE_PCT = 0.0002
    ROLLOVER_FEE_PCT = 0.0002


# =============================================================================
# [RULETABLE] OPPORTUNITY Mode Rule Table
# =============================================================================
#
# Consolidates ALL OPPORTUNITY-mode parameter relaxations into one table.
# Any module calls get_rule(name, is_opportunity) instead of local if/else.
#
# Status legend:
#   HARD     = Same value in both modes (safety floor, never relaxed)
#   SOFTENED = Relaxed in OPPORTUNITY mode (releases alpha)
#   REF      = Enforced in external module (documented here for audit)
#
# Principle: HARD protects capital. SOFTENED releases alpha. REF = external.

OPPORTUNITY_RULE_TABLE = {
    # --- HARD: never relaxed (safety floor) ---
    "hard_drawdown_halt":       {"status": "HARD",     "normal": 0.20,   "opportunity": 0.20,   "note": "DD >=0% ->halt [UL-5]"},
    "reduce_at_drawdown":       {"status": "HARD",     "normal": 0.08,   "opportunity": 0.08,   "note": "DD >=% ->reduce [UL-5]"},
    "correlation_crisis":       {"status": "HARD",     "normal": 0.98,   "opportunity": 0.98,   "note": ">=.98 ->flatten [UL-4]"},
    "data_integrity":           {"status": "HARD",     "normal": True,   "opportunity": True,   "note": "CRC32/schema/shadow_ledger"},
    "no_trade_enforcement":     {"status": "HARD",     "normal": True,   "opportunity": True,   "note": "NO_TRADE always overrides"},

    # --- SOFTENED: OPPORTUNITY relaxes (main.py) ---
    "max_hold_hours":           {"status": "SOFTENED", "normal": 72,     "opportunity": 336,    "note": "Position max duration [UL-8e]"},
    "philosophy_multiplier":    {"status": "SOFTENED", "normal": 0.50,   "opportunity": 0.70,   "note": "Aggressiveness at 5% DD [V33HR]"},

    # --- REF: Enforced in external modules (documented for audit) ---
    "alpha_gate_multiplier":    {"status": "REF",      "normal": 1.25,   "opportunity": 1.0,    "note": "friction x mult [constitution.py L1020] (was 1.5, too aggressive)"},
    "max_single_position_pct":  {"status": "REF",      "normal": 0.80,   "opportunity": 0.95,   "note": "gambler+opp [high_risk_mode.py L108-111]"},
    "max_gross_exposure":       {"status": "REF",      "normal": 1.50,   "opportunity": 2.00,   "note": "gambler+opp [high_risk_mode.py L114-117]"},
    "exposure_cap_sol":         {"status": "REF",      "normal": 0.40,   "opportunity": 0.70,   "note": "SOL cap [integrated_manager.py L612-613]"},
    "exposure_cap_other":       {"status": "REF",      "normal": 0.35,   "opportunity": 0.50,   "note": "BTC/ETH cap [integrated_manager.py L612-613]"},
}


def get_rule(rule_name: str, is_opportunity: bool) -> Optional[Any]:
    """
    Query rule table for current mode's parameter value.

    Usage:
        max_h = get_rule("max_hold_hours", _is_opportunity)  # 72 or 336
    """
    rule = OPPORTUNITY_RULE_TABLE.get(rule_name)
    if rule is None:
        logger.warning(f"[RULETABLE] Unknown rule: {rule_name}")
        return None
    if rule["status"] == "HARD":
        return rule["normal"]  # HARD never changes
    return rule["opportunity"] if is_opportunity else rule["normal"]


def _validate_rule_table():
    """Startup assertion: HARD rules must have identical normal/opportunity values."""
    for name, rule in OPPORTUNITY_RULE_TABLE.items():
        if rule["status"] == "HARD":
            assert rule["normal"] == rule["opportunity"], (
                f"[RULETABLE] HARD rule '{name}' has mismatched values: "
                f"{rule['normal']} vs {rule['opportunity']}"
            )


_validate_rule_table()


# =============================================================================
# [MODE-TTL] OPPORTUNITY Mode TTL Auto-Expiry
# =============================================================================

MODE_TTL_CONFIG = {
    'max_bars': MODE_TTL_MAX_BARS,
    'confidence_decay_per_bar': MODE_TTL_CONFIDENCE_DECAY,
    'confidence_floor': MODE_TTL_CONFIDENCE_FLOOR,
    'blend_bars': MODE_TTL_BLEND_BARS,
}


# =============================================================================
# [REGIME-LEV] Margin Cost Tracker
# =============================================================================

class MarginCostTracker:
    """Track Kraken margin costs for leveraged trades."""

    def __init__(self):
        self._total_margin_cost = 0.0
        self._total_leveraged_pnl = 0.0
        self._leveraged_trades = 0

    def record_trade(self, notional_usd: float, leverage: float,
                     holding_bars: int, pnl_usd: float):
        """Record a completed leveraged trade's costs and PnL."""
        if leverage <= 1.0:
            return
        margin_cost = notional_usd * OPEN_FEE_PCT
        margin_cost += notional_usd * ROLLOVER_FEE_PCT * holding_bars
        self._total_margin_cost += margin_cost
        self._total_leveraged_pnl += pnl_usd
        self._leveraged_trades += 1
        net = pnl_usd - margin_cost
        logger.info(
            f"[REGIME-LEV] Margin trade closed: pnl=${pnl_usd:.2f} "
            f"margin_cost=${margin_cost:.2f} net=${net:.2f} "
            f"lev={leverage:.0f}x hold={holding_bars}bars"
        )

    def get_summary(self) -> dict:
        net = self._total_leveraged_pnl - self._total_margin_cost
        return {
            "leveraged_trades": self._leveraged_trades,
            "total_pnl": self._total_leveraged_pnl,
            "total_margin_cost": self._total_margin_cost,
            "net_after_cost": net,
            "leverage_profitable": net > 0,
        }


# =============================================================================
# [PATCH-1] Schema Required-Key Validation
# =============================================================================

SCHEMA_REQUIRED_KEYS = {
    'current_price': {'type': float, 'min': 0.0001},
    'volume_24h':    {'type': float, 'min': 0},
}

SCHEMA_CRITICAL_KEYS = {
    'dvol_zscore':              {'default': 0.0},
    'vpin':                     {'default': 0.5},
    'correlation_btc_eth_sol':  {'default': 0.0},
    'regime_confidence':        {'default': 0.5},
}

SCHEMA_OPTIONAL_KEYS = {
    'spread_bps':               {'default': 10.0},
    'orderbook_imbalance':      {'default': 0.0},
    'volume_ratio':             {'default': 1.0},
    'volume_ratio_effective':   {'default': 1.0},
    'estimated_friction_bps':   {'default': 15.0},
    'orderbook_depth_1pct_usd': {'default': 500_000.0},
}


# =============================================================================
# [PATCH-5] Deadlock Resolution Config
# =============================================================================

DEADLOCK_CONFIG = {
    'T1_force_bars': DEADLOCK_T1_FORCE_BARS,
    'T2_abort_bars': DEADLOCK_T2_ABORT_BARS,
    'edge_decay_per_bar': DEADLOCK_EDGE_DECAY_PER_BAR,
}


# =============================================================================
# [PATCH-6] SOL Emergency Exit Thresholds
# =============================================================================

SOL_EMERGENCY_THRESHOLDS = {
    'vpin_immediate':       SOL_VPIN_IMMEDIATE,
    'correlation_immediate': SOL_CORRELATION_IMMEDIATE,
    'flash_move_pct':       SOL_FLASH_MOVE_PCT,
}

SOL_DANGER_PHASES = frozenset({'SATURATION', 'EXHAUSTION', 'CRASH', 'CAPITULATION'})
SOL_SAFE_PHASES = frozenset({'IGNITION', 'EXPANSION', 'ACCUMULATION', 'RECOVERY'})

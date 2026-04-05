"""
================================================================================
HMATS v6.7 - CANONICAL CONFIGURATION (Single Source of Truth)
================================================================================
Purpose: Eliminate config sprawl by defining authoritative values in ONE place.

All modules should import from here instead of hardcoding values.
If a value appears in a JSON config file, the JSON value wins at runtime
(loaded via ProductionConfig.from_file). This module defines the DEFAULTS
that JSON configs fall back to.

DRAWDOWN GRADIENT (authoritative):
    reduce_at_drawdown  = 8%   -> reduce position sizes
    hard_drawdown_halt  = 20%  -> halt new trades
    kill_switch_dd      = 35%  -> system halt, manual restart
    daily_loss_halt     = 10%  -> daily loss kill-switch

AUTHORITY HIERARCHY:
    1. Environment variables (highest)
    2. JSON config file (cloud_production.json + profile overlay)
    3. This module (canonical defaults)
    4. Code-level fallbacks (lowest - should match this module)
================================================================================
"""

# =============================================================================
# PROJECT PATHS - [FIX-37] single source for data/log directories
# =============================================================================

from pathlib import Path

DATA_DIR = Path("data")
LOG_DIR = Path("logs")

# =============================================================================
# ACCOUNT
# =============================================================================

INITIAL_CAPITAL = 10_000.0          # Actual account size ($10K)
# env override: HMATS_INITIAL_CAPITAL

# =============================================================================
# DRAWDOWN GRADIENT (strictly ordered: reduce < halt < kill_switch)
# =============================================================================

REDUCE_AT_DRAWDOWN = 0.08           # 8% - first response: reduce sizes
HARD_DRAWDOWN_HALT = 0.20           # 20% - halt new trades
CRITICAL_DRAWDOWN = 0.15            # 15% - second safety wall (risk_agent)
KILL_SWITCH_DRAWDOWN = 0.35         # 35% - system halt, manual restart
DAILY_LOSS_HALT = 0.08              # 8% - daily loss reduction trigger (SOTARiskConfig layer)
DAILY_LOSS_KILL = 0.10              # 10% - daily loss kill-switch (full halt)

# Note: CRITICAL_DRAWDOWN < HARD_DRAWDOWN_HALT is intentional.
# critical_drawdown (15%) triggers risk_agent position reduction.
# hard_drawdown_halt (20%) halts all new trades. They are different layers.

# Multi-step drawdown gradient lookup table:
# Each tuple is (drawdown_threshold, size_multiplier).
# Applied top-down: first match (dd >= threshold) wins.
DRAWDOWN_GRADIENT = [
    (0.35, 0.00),   # ≥35% - KILL SWITCH, no trades
    (0.25, 0.10),   # ≥25% - emergency: 10% sizing
    (0.20, 0.25),   # ≥20% - halt new trades, exits only at 25% size
    (0.15, 0.50),   # ≥15% - critical: half size
    (0.08, 0.75),   # ≥8%  - reduce: 75% size
    (0.00, 1.00),   # <8%  - normal: full size
]


def get_drawdown_multiplier(drawdown_pct: float) -> float:
    """Return position-size multiplier for the current drawdown level.

    Scans DRAWDOWN_GRADIENT top-down; first threshold exceeded wins.
    """
    for threshold, mult in DRAWDOWN_GRADIENT:
        if drawdown_pct >= threshold:
            return mult
    return 1.0

# =============================================================================
# CORRELATION
# =============================================================================

CORRELATION_CRISIS = 0.98           # ≥0.98 -> flatten (UL-4)
CORRELATION_COLLAPSE = 0.30         # Collapse detection threshold

# =============================================================================
# EXPOSURE CAPS
# =============================================================================

MAX_GROSS_EXPOSURE = 1.50           # 150% total gross
MAX_SINGLE_ASSET_PCT = 0.80         # 80% single asset (NORMAL)
MAX_LEVERAGE = 3.0                  # 3x max

# Per-asset pre-leverage caps
EXPOSURE_CAPS = {
    "BTC": 0.40,
    "ETH": 0.40,
    "SOL": 0.50,
}

# Per-asset post-leverage caps
POST_LEVERAGE_CAPS = {
    "BTC": 0.25,
    "ETH": 0.25,
    "SOL": 0.20,
}

# Leverage guards
MAX_POSITION_NOTIONAL_USD = 50_000.0
MAX_TOTAL_NOTIONAL_USD = 100_000.0

# [FIX-33] Kraken fee tiers - single source of truth
KRAKEN_FREE_TIER_USD = 10_000.0  # No fees below this monthly volume
KRAKEN_TAKER_BPS = 26.0
KRAKEN_MAKER_BPS = 16.0

# =============================================================================
# REGIME LEVERAGE
# =============================================================================

REGIME_LEVERAGE_MAP = {
    "VOLATILE_CHOP": 3.0,
    "MOMENTUM_RALLY": 2.0,
    "PANIC_SELLOFF": 2.0,
    "WEAK_CONSOLIDATION": 1.0,
    "QUIET_ACCUMULATION": 1.0,
    "EXTREME_VOLATILITY": 1.0,
    # Legacy regime names
    "STRONG_TREND": 2.0,
    "TREND": 1.5,
    "BULL_TREND": 1.5,
    "BEAR_TREND": 1.0,
    "UNCERTAIN": 1.0,
    "MEAN_REVERT": 1.0,
    "TRANSITION": 1.0,
    "SIDEWAYS": 1.0,
    "UNKNOWN": 1.0,
}

# =============================================================================
# TIMING
# =============================================================================

MASTER_TICK_SECONDS = 14_400        # 4H = 14400s
EXECUTION_LOOP_MS = 200             # 200ms execution modifiers
DMS_TIMEOUT_SECONDS = 60            # Dead-man switch timeout

# =============================================================================
# MARGIN COSTS (Kraken)
# =============================================================================

OPEN_FEE_PCT = 0.0002              # 0.02% open fee
ROLLOVER_FEE_PCT = 0.0002          # 0.02% per 4H rollover

# [FIX-35] Regime confidence floor - below this, reduce exposure
MIN_REGIME_CONFIDENCE = 0.3

# =============================================================================
# THESIS BUDGET
# =============================================================================

THESIS_BUDGET_LOSS_PCT_NAV = 0.02   # 2% NAV (UL-6b)
THESIS_BUDGET_COOLDOWN_BARS = 2     # 8h cooldown (UL-6b)
THESIS_BUDGET_LOSS_STREAK_LIMIT = 6 # (UL-6b)

# =============================================================================
# TRANCHE SIZES (cumulative, must match tranche_manager + position_sizer)
# =============================================================================

TRANCHE_CUMULATIVE = {
    "TRANCHE_1": 0.35,
    "TRANCHE_2": 0.65,
    "TRANCHE_3": 0.90,
    "TRANCHE_4": 1.00,
}

# =============================================================================
# MODE-TTL (OPPORTUNITY auto-expiry)
# =============================================================================

MODE_TTL_MAX_BARS = 4               # 16H absolute max
MODE_TTL_CONFIDENCE_DECAY = 0.15    # 15% decay per 4H bar
MODE_TTL_CONFIDENCE_FLOOR = 0.20    # Below this -> expire
MODE_TTL_BLEND_BARS = 1             # Bars to blend on expiry

# =============================================================================
# EXIT ALPHA
# =============================================================================

SCALE_OUT_PCT = 0.10                # 10% first exit
SCALE_OUT_MIN_PROFIT_BPS = 500.0    # 5% min profit before scale
SCALE_OUT_MIN_BARS = 3              # Min bars held before scale
RUNNER_INITIAL_TRAIL_PCT = 0.03     # 3% initial trail
RUNNER_TIGHT_TRAIL_PCT = 0.015      # 1.5% tight trail

# =============================================================================
# STOP MANAGEMENT
# =============================================================================

STOP_ATR_MULTIPLIER = 3.5           # ATR multiplier for initial stop
STOP_MIN_PCT = 0.01                 # 1% minimum stop
MAX_HOLDING_HOURS = 336             # 14 days (OPPORTUNITY mode)

# =============================================================================
# DEADLOCK RESOLUTION
# =============================================================================

DEADLOCK_T1_FORCE_BARS = 2          # Small tranche -> force after 2 bars
DEADLOCK_T2_ABORT_BARS = 3          # Large tranche -> abort after 3 bars
DEADLOCK_EDGE_DECAY_PER_BAR = 0.15  # 15% edge decay per bar

# =============================================================================
# SOL EMERGENCY THRESHOLDS
# =============================================================================

SOL_VPIN_IMMEDIATE = 0.90
SOL_CORRELATION_IMMEDIATE = 0.92
SOL_FLASH_MOVE_PCT = 0.03           # 3% move

# =============================================================================
# REBUILD COOLDOWN
# =============================================================================

REBUILD_COOLDOWN_EXEMPT_ADDON = True  # [CFG-9] Pyramid add-on exempt from cooldown

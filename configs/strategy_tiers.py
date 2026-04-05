"""
SOTA-ACT GA-4: Strategy tier budget configuration.

Core:    trend/momentum  - stable income, lower leverage
Vol:     mean_reversion  - medium frequency, medium leverage
Convex:  breakout/tail   - low frequency high payoff
Event:   sentiment/chain - event-driven, lower leverage
"""

STRATEGY_TIER_MAP = {
    # Core (50%, max 2x) - trend/momentum, stable income
    'trend_following': 'CORE',
    'momentum': 'CORE',
    'ma_crossover': 'CORE',
    'macd_trend': 'CORE',
    'RelativeStrengthStrategy': 'CORE',     # [AUDIT M1] kraken_quant_agent
    'ETFSpotCointegration': 'CORE',         # [AUDIT M1] kraken_quant_agent

    # Vol (20%, max 3x) - mean-reversion, medium frequency
    'mean_reversion': 'VOL',
    'mean_revert': 'VOL',
    'bb_revert': 'VOL',
    'rsi_mean_revert': 'VOL',
    'OrnsteinUhlenbeckStrategy': 'VOL',     # [AUDIT M1] kraken_quant_agent (OU = mean-revert)
    'HurstExponentStrategy': 'VOL',         # [AUDIT M1] kraken_quant_agent (Hurst = regime filter)
    'ShannonEntropyStrategy': 'VOL',        # [AUDIT M1] kraken_quant_agent (entropy = vol regime)

    # Convex (15%, max 3x) - breakout/tail, low frequency high payoff
    'range_breakout': 'CONVEX',
    'volume_breakout': 'CONVEX',
    'vol_breakout': 'CONVEX',
    'short_bias': 'CONVEX',
    'crack_break': 'CONVEX',
    'vrp': 'CONVEX',
    'VarianceRiskPremiumStrategy': 'CONVEX', # [AUDIT M1] kraken_quant_agent (VRP = vol premium)
    'OrderBookImbalance': 'CONVEX',          # [AUDIT M1] kraken_quant_agent (OBI = microstructure)
    'DarkPoolVolumeStrategy': 'CONVEX',      # [AUDIT M1] kraken_quant_agent (dark pool = tail)

    # Event (15%, max 2x) - sentiment/chain, event-driven
    'sentiment': 'EVENT',
    'onchain': 'EVENT',
    'funding_carry': 'EVENT',
    'liquidation_cascade': 'EVENT',
    'LiquidationCascadeHunter': 'EVENT',     # [AUDIT M1] kraken_quant_agent
    'FundingDivergenceStrategy': 'EVENT',    # [AUDIT M1] kraken_quant_agent (funding = event)
    'DeltaNeutralFundingStrategy': 'EVENT',  # [AUDIT M1] kraken_quant_agent (delta-neutral = event)
}

TIER_BUDGETS = {
    # [FIX-P0-5] CORE raised 45%->70%: Best-of-N selects momentum ~80% of ticks,
    # 3 assets x 20-25% MAX_EXPOSURE need >=60% headroom. Old 45% blocked 90%+ trades.
    # [FIX-P0-6] CONVEX raised 10%->30%: Best-of-N selects volume_breakout simultaneously
    # for multiple correlated crypto assets. Old 10% budget ($997) was exhausted by a single
    # ETH position ($2,144), blocking BTC entries entirely (same root cause as CORE fix).
    # 3 assets x ~15% avg exposure = 45% needed; 30% allows partial entry via clamp logic.
    # [FIX-VOL-TIER] VOL raised 25%->70%: mean_revert selected in WEAK_CONSOLIDATION regime,
    # same root cause as CORE/CONVEX fixes. 25% budget ($2,441) was exhausted by 2 assets at T1
    # (~$1,991), blocking all T2 escalations and 3rd asset (ETH) entry. Per-asset limits
    # (BTC/ETH=25%, SOL=20%) sum to exactly 70% of NAV — the per-asset limit IS the real cap.
    'CORE':   {'pct': 0.70, 'max_leverage': 2.0, 'max_drawdown': 0.05},
    'VOL':    {'pct': 0.70, 'max_leverage': 3.0, 'max_drawdown': 0.03},
    'CONVEX': {'pct': 0.30, 'max_leverage': 3.0, 'max_drawdown': 0.04},
    'EVENT':  {'pct': 0.10, 'max_leverage': 2.0, 'max_drawdown': 0.03},
    # NOTE: tiers can overlap (sum > 100%) — each tier is an independent cap.
    # Total portfolio exposure is still bounded by MAX_EXPOSURE_FRACTION per asset.
}

DEFAULT_TIER = 'CORE'


def get_tier(strategy_name: str) -> str:
    """Get tier for a strategy name."""
    return STRATEGY_TIER_MAP.get(strategy_name, DEFAULT_TIER)


def get_budget(tier: str) -> dict:
    """Get budget config for a tier."""
    return TIER_BUDGETS.get(tier, TIER_BUDGETS[DEFAULT_TIER])
"""
HMATS v6.2.2 - Fee Management Module
====================================

This module contains fee calculation and blending logic.

Components:
- KrakenPlusFeeBlender: Kraken+ fee blending based on monthly volume
- KrakenSpotVolumeTracker: Monthly volume tracking
- KrakenPlusFeeCalculator: Integrated fee calculation

Usage:
    from execution.fees import on_trade_fill, get_monthly_volume
    
    # On trade fill
    result = on_trade_fill(
        notional_usd=5000,
        fee_std=0.0026,
        symbol="BTC/USD",
        is_spot=True,
        exchange="kraken"
    )
"""

from execution.fees.kraken_plus_fee_blender import (
    # Core classes
    KrakenPlusFeeBlender,
    KrakenPlusFeeConfig,
    KrakenPlusFeeCalculator,
    KrakenSpotVolumeTracker,
    MonthlyVolumeRecord,
    
    # Singleton access
    get_fee_calculator,
    reset_fee_calculator,
    
    # Convenience functions
    compute_weight,
    apply_fee_blending,
    on_trade_fill,
    get_monthly_volume,
    is_fee_blending_enabled,
    get_fee_config_from_env,
)

__all__ = [
    # Classes
    'KrakenPlusFeeBlender',
    'KrakenPlusFeeConfig',
    'KrakenPlusFeeCalculator',
    'KrakenSpotVolumeTracker',
    'MonthlyVolumeRecord',
    
    # Singleton
    'get_fee_calculator',
    'reset_fee_calculator',
    
    # Functions
    'compute_weight',
    'apply_fee_blending',
    'on_trade_fill',
    'get_monthly_volume',
    'is_fee_blending_enabled',
    'get_fee_config_from_env',
]

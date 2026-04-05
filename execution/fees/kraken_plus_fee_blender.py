"""
================================================================================
HMATS v6.2.2 - Kraken+ Fee Blending Model
================================================================================

Purpose: Implement Kraken+ spot trading fee blending based on monthly volume.

Fee Blending Rule:
    V = Monthly executed notional (USD) for Kraken Spot
    F_std = Standard fee from existing fee model
    
    fee_effective = w(V) * F_std
    
    where:
        w(V) = 0                      when V ≤ FREE_TIER (default 10,000)
        w(V) = (V - FREE_TIER) / BLEND_BAND   when FREE_TIER < V < FREE_TIER + BLEND_BAND
        w(V) = 1                      when V ≥ FREE_TIER + BLEND_BAND
    
    w(V) is clamped to [0, 1]

Scope:
    - Only applies to Kraken Spot trades
    - Margin/Futures continue to use original fee model
    - Volume V is based on EXECUTED notional, not intent

Author: HMATS v6.2.2 Fee Blending
Version: 6.2.2
================================================================================
"""

import os
import json
import time
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Optional, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class KrakenPlusFeeConfig:
    """Configuration for Kraken+ Fee Blending."""
    
    # Enable/disable flag
    enabled: bool = True
    
    # Free tier threshold (USD)
    free_tier_usd: float = 10_000.0
    
    # Blend band width (USD) - transition from free_tier to full fees
    blend_band_usd: float = 2_000.0
    
    # Calendar month basis (UTC)
    use_utc: bool = True
    
    # Persistence file for monthly volume
    volume_file_path: str = "data/kraken_plus_monthly_volume.json"
    
    @property
    def full_fee_threshold(self) -> float:
        """Threshold above which full fees apply."""
        return self.free_tier_usd + self.blend_band_usd


def _env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    val = os.environ.get(f"HMATS_{key}", "")
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    return default


def _env_bool(key: str, default: bool) -> bool:
    """Get bool from environment variable."""
    val = os.environ.get(f"HMATS_{key}", "").lower()
    if val in ("true", "1", "yes", "on"):
        return True
    elif val in ("false", "0", "no", "off"):
        return False
    return default


def get_fee_config_from_env() -> KrakenPlusFeeConfig:
    """Get fee config with environment overrides."""
    return KrakenPlusFeeConfig(
        enabled=_env_bool("ENABLE_KRAKEN_PLUS_FEE_BLENDING", True),
        free_tier_usd=_env_float("KRAKEN_PLUS_FREE_TIER_USD", 10_000.0),
        blend_band_usd=_env_float("KRAKEN_PLUS_BLEND_BAND_USD", 2_000.0),
    )


# =============================================================================
# FEE BLENDING CORE
# =============================================================================

class KrakenPlusFeeBlender:
    """
    Fee blending calculator for Kraken+ spot trading.
    
    Implements the weight function w(V) for fee blending:
        w(V) = 0 when V ≤ free_tier
        w(V) = (V - free_tier) / blend_band when free_tier < V < free_tier + blend_band
        w(V) = 1 when V ≥ free_tier + blend_band
    
    Usage:
        blender = KrakenPlusFeeBlender(config)
        fee_effective = blender.apply(fee_std=0.0026, monthly_volume_usd=9500)
    """
    
    def __init__(self, config: Optional[KrakenPlusFeeConfig] = None):
        self.config = config or get_fee_config_from_env()
        logger.info(
            f"KrakenPlusFeeBlender initialized: enabled={self.config.enabled}, "
            f"free_tier=${self.config.free_tier_usd:,.0f}, "
            f"blend_band=${self.config.blend_band_usd:,.0f}"
        )
    
    def compute_weight(self, monthly_volume_usd: float) -> float:
        """
        Compute the blending weight w(V).
        
        Args:
            monthly_volume_usd: Total executed notional this calendar month (USD)
        
        Returns:
            Weight in [0, 1]
                0 = no fee (within free tier)
                1 = full standard fee
                0-1 = blended (in transition zone)
        """
        if not self.config.enabled:
            return 1.0  # Full fees when disabled
        
        V = monthly_volume_usd
        free_tier = self.config.free_tier_usd
        blend_band = self.config.blend_band_usd
        
        if V <= free_tier:
            return 0.0
        elif V >= free_tier + blend_band:
            return 1.0
        else:
            # Linear interpolation in blend band
            w = (V - free_tier) / blend_band
            # Clamp to [0, 1] for safety
            return max(0.0, min(1.0, w))
    
    def apply(self, fee_std: float, monthly_volume_usd: float) -> float:
        """
        Apply fee blending to get effective fee.
        
        Args:
            fee_std: Standard fee from original fee model (as decimal, e.g., 0.0026 for 0.26%)
            monthly_volume_usd: Total executed notional this calendar month (USD)
        
        Returns:
            Effective fee after blending
        """
        if not self.config.enabled:
            return fee_std
        
        weight = self.compute_weight(monthly_volume_usd)
        fee_effective = weight * fee_std
        
        return fee_effective
    
    def is_in_free_tier(self, monthly_volume_usd: float) -> bool:
        """Check if volume is within free tier (zero fees)."""
        return self.config.enabled and monthly_volume_usd <= self.config.free_tier_usd
    
    def is_in_blend_zone(self, monthly_volume_usd: float) -> bool:
        """Check if volume is in blend zone (partial fees)."""
        if not self.config.enabled:
            return False
        V = monthly_volume_usd
        return self.config.free_tier_usd < V < self.config.full_fee_threshold
    
    def get_status(self, monthly_volume_usd: float, fee_std: float = 0.0) -> Dict[str, Any]:
        """Get detailed status for logging/dashboard."""
        weight = self.compute_weight(monthly_volume_usd)
        fee_effective = self.apply(fee_std, monthly_volume_usd)
        
        if monthly_volume_usd <= self.config.free_tier_usd:
            zone = "FREE_TIER"
        elif monthly_volume_usd >= self.config.full_fee_threshold:
            zone = "FULL_FEE"
        else:
            zone = "BLEND_ZONE"
        
        return {
            "enabled": self.config.enabled,
            "monthly_volume_usd": monthly_volume_usd,
            "free_tier_usd": self.config.free_tier_usd,
            "blend_band_usd": self.config.blend_band_usd,
            "full_fee_threshold": self.config.full_fee_threshold,
            "weight": weight,
            "zone": zone,
            "fee_std": fee_std,
            "fee_effective": fee_effective,
            "fee_savings_pct": (1 - weight) * 100 if fee_std > 0 else 0,
        }


# =============================================================================
# MONTHLY VOLUME TRACKER
# =============================================================================

@dataclass
class MonthlyVolumeRecord:
    """Record of monthly executed volume."""
    year: int
    month: int
    executed_notional_usd: float
    last_updated: float  # Unix timestamp
    trade_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "year": self.year,
            "month": self.month,
            "executed_notional_usd": self.executed_notional_usd,
            "last_updated": self.last_updated,
            "trade_count": self.trade_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MonthlyVolumeRecord':
        return cls(
            year=data["year"],
            month=data["month"],
            executed_notional_usd=data["executed_notional_usd"],
            last_updated=data["last_updated"],
            trade_count=data.get("trade_count", 0),
        )


class KrakenSpotVolumeTracker:
    """
    Tracks monthly executed volume for Kraken Spot trades.
    
    Uses calendar month (UTC) for volume reset.
    Persists to JSON file for hot restart recovery.
    """
    
    def __init__(self, config: Optional[KrakenPlusFeeConfig] = None):
        self.config = config or get_fee_config_from_env()
        self._lock = threading.Lock()
        self._current_record: Optional[MonthlyVolumeRecord] = None
        self._file_path = Path(self.config.volume_file_path)
        
        # Load persisted data
        self._load()
        
        logger.info(f"KrakenSpotVolumeTracker initialized, current volume: ${self.get_monthly_volume():,.2f}")
    
    def _get_current_month(self) -> tuple:
        """Get current (year, month) in UTC."""
        now = datetime.now(timezone.utc) if self.config.use_utc else datetime.now()
        return (now.year, now.month)
    
    def _load(self):
        """Load persisted volume data."""
        try:
            if self._file_path.exists():
                with open(self._file_path) as f:
                    data = json.load(f)
                self._current_record = MonthlyVolumeRecord.from_dict(data)
                
                # Check if we need to reset for new month
                current_year, current_month = self._get_current_month()
                if (self._current_record.year != current_year or 
                    self._current_record.month != current_month):
                    logger.info(
                        f"New month detected, resetting volume "
                        f"(was {self._current_record.year}-{self._current_record.month:02d}, "
                        f"now {current_year}-{current_month:02d})"
                    )
                    self._current_record = None
        except Exception as e:
            logger.warning(f"Failed to load volume data: {e}")
            self._current_record = None
    
    def _save(self):
        """Persist volume data to file."""
        if self._current_record is None:
            return
        
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._file_path, 'w') as f:
                json.dump(self._current_record.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save volume data: {e}")
    
    def _ensure_current_month(self):
        """Ensure we have a record for the current month."""
        current_year, current_month = self._get_current_month()
        
        if (self._current_record is None or
            self._current_record.year != current_year or
            self._current_record.month != current_month):
            # New month - reset
            self._current_record = MonthlyVolumeRecord(
                year=current_year,
                month=current_month,
                executed_notional_usd=0.0,
                last_updated=time.time(),
                trade_count=0,
            )
            self._save()
    
    def record_fill(
        self,
        notional_usd: float,
        symbol: str = "",
        is_spot: bool = True,
        exchange: str = "kraken",
    ) -> Dict[str, Any]:
        """
        Record an executed trade fill.
        
        Args:
            notional_usd: Executed notional value in USD
            symbol: Trading pair symbol
            is_spot: True if spot trade (not margin/futures)
            exchange: Exchange name
        
        Returns:
            Dict with updated volume info for logging
        """
        # Only track Kraken Spot
        if exchange.lower() != "kraken" or not is_spot:
            return {
                "tracked": False,
                "reason": f"Not Kraken Spot (exchange={exchange}, is_spot={is_spot})",
            }
        
        with self._lock:
            self._ensure_current_month()
            
            old_volume = self._current_record.executed_notional_usd
            self._current_record.executed_notional_usd += notional_usd
            self._current_record.trade_count += 1
            self._current_record.last_updated = time.time()
            
            self._save()
            
            new_volume = self._current_record.executed_notional_usd
            
            return {
                "tracked": True,
                "symbol": symbol,
                "notional_usd": notional_usd,
                "old_monthly_volume": old_volume,
                "new_monthly_volume": new_volume,
                "trade_count": self._current_record.trade_count,
                "month": f"{self._current_record.year}-{self._current_record.month:02d}",
            }
    
    def get_monthly_volume(self) -> float:
        """Get current month's executed volume."""
        with self._lock:
            self._ensure_current_month()
            return self._current_record.executed_notional_usd
    
    def get_status(self) -> Dict[str, Any]:
        """Get tracker status for logging/dashboard."""
        with self._lock:
            self._ensure_current_month()
            return {
                "year": self._current_record.year,
                "month": self._current_record.month,
                "monthly_volume_usd": self._current_record.executed_notional_usd,
                "trade_count": self._current_record.trade_count,
                "last_updated": self._current_record.last_updated,
            }
    
    def reset_for_testing(self):
        """Reset volume for testing purposes."""
        with self._lock:
            self._current_record = None
            self._ensure_current_month()
            logger.warning("Volume tracker reset for testing")


# =============================================================================
# INTEGRATED FEE CALCULATOR
# =============================================================================

class KrakenPlusFeeCalculator:
    """
    Integrated fee calculator with blending and volume tracking.
    
    This is the main interface for the fee system.
    
    Usage:
        calc = KrakenPlusFeeCalculator()
        
        # On fill:
        result = calc.on_fill(notional_usd=5000, fee_std=0.0026, symbol="BTC/USD")
        print(f"Fee effective: {result['fee_effective']}")
    """
    
    def __init__(self, config: Optional[KrakenPlusFeeConfig] = None):
        self.config = config or get_fee_config_from_env()
        self.blender = KrakenPlusFeeBlender(self.config)
        self.tracker = KrakenSpotVolumeTracker(self.config)
        
        logger.info("KrakenPlusFeeCalculator initialized")
    
    def calculate_fee(
        self,
        notional_usd: float,
        fee_std: float,
        is_spot: bool = True,
        exchange: str = "kraken",
    ) -> Dict[str, Any]:
        """
        Calculate effective fee (without recording - for pre-trade estimation).
        
        Args:
            notional_usd: Trade notional (USD)
            fee_std: Standard fee rate (decimal, e.g., 0.0026)
            is_spot: True if spot trade
            exchange: Exchange name
        
        Returns:
            Dict with fee calculation details
        """
        # Check if blending applies
        if not self.config.enabled or exchange.lower() != "kraken" or not is_spot:
            fee_usd = notional_usd * fee_std
            return {
                "blending_applied": False,
                "fee_std": fee_std,
                "fee_effective": fee_std,
                "fee_usd": fee_usd,
                "weight": 1.0,
                "reason": "Blending not applicable" if self.config.enabled else "Blending disabled",
            }
        
        # Get current monthly volume
        monthly_volume = self.tracker.get_monthly_volume()
        
        # Calculate blended fee
        weight = self.blender.compute_weight(monthly_volume)
        fee_effective = self.blender.apply(fee_std, monthly_volume)
        fee_usd = notional_usd * fee_effective
        fee_std_usd = notional_usd * fee_std
        
        return {
            "blending_applied": True,
            "monthly_volume_usd": monthly_volume,
            "fee_std": fee_std,
            "fee_effective": fee_effective,
            "fee_usd": fee_usd,
            "fee_std_usd": fee_std_usd,
            "fee_savings_usd": fee_std_usd - fee_usd,
            "weight": weight,
            "in_free_tier": self.blender.is_in_free_tier(monthly_volume),
            "in_blend_zone": self.blender.is_in_blend_zone(monthly_volume),
        }
    
    def on_fill(
        self,
        notional_usd: float,
        fee_std: float,
        symbol: str = "",
        is_spot: bool = True,
        exchange: str = "kraken",
        is_maker: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a trade fill and calculate effective fee.
        
        This records the fill to the volume tracker AND calculates the fee.
        
        Args:
            notional_usd: Executed notional (USD)
            fee_std: Standard fee rate (decimal)
            symbol: Trading pair
            is_spot: True if spot trade
            exchange: Exchange name
            is_maker: True if maker order
        
        Returns:
            Dict with fee details and volume update info
        """
        # [AC-3] Calculate fee BEFORE recording volume - fee applies to
        # the volume tier the trader was in at fill time, not after
        fee_result = self.calculate_fee(
            notional_usd=notional_usd,
            fee_std=fee_std,
            is_spot=is_spot,
            exchange=exchange,
        )

        # Now record the fill (updates volume for next trade's tier)
        fill_result = self.tracker.record_fill(
            notional_usd=notional_usd,
            symbol=symbol,
            is_spot=is_spot,
            exchange=exchange,
        )
        
        # Combine results
        result = {
            **fee_result,
            "fill_tracked": fill_result.get("tracked", False),
            "symbol": symbol,
            "notional_usd": notional_usd,
            "is_maker": is_maker,
        }
        
        # Log to ShadowLedger if available
        self._log_to_shadow_ledger(result)
        
        return result
    
    def _log_to_shadow_ledger(self, result: Dict[str, Any]):
        """Log fee calculation to ShadowLedger for audit."""
        try:
            from data_mgmt.shadow_ledger_manager import get_shadow_ledger
            ledger = get_shadow_ledger()
            
            ledger.log_veto(
                source="KRAKEN_PLUS_FEE_BLENDER",
                reason="FEE_MODEL_UPDATE",
                details={
                    "monthly_volume_usd": result.get("monthly_volume_usd", 0),
                    "weight": result.get("weight", 1),
                    "fee_std": result.get("fee_std", 0),
                    "fee_effective": result.get("fee_effective", 0),
                    "fee_usd": result.get("fee_usd", 0),
                    "in_free_tier": result.get("in_free_tier", False),
                    "blending_applied": result.get("blending_applied", False),
                }
            )
        except ImportError:
            pass  # ShadowLedger not available
        except Exception as e:
            logger.debug(f"Failed to log to ShadowLedger: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get full status for dashboard/logging."""
        volume = self.tracker.get_monthly_volume()
        return {
            **self.blender.get_status(volume),
            **self.tracker.get_status(),
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_fee_calculator: Optional[KrakenPlusFeeCalculator] = None
_fee_calculator_lock = threading.Lock()


def get_fee_calculator(config: Optional[KrakenPlusFeeConfig] = None) -> KrakenPlusFeeCalculator:
    """Get singleton KrakenPlusFeeCalculator instance."""
    global _fee_calculator
    if _fee_calculator is None:
        with _fee_calculator_lock:
            if _fee_calculator is None:
                _fee_calculator = KrakenPlusFeeCalculator(config)
    elif config is not None and getattr(_fee_calculator, "config", None) != config:
        with _fee_calculator_lock:
            if getattr(_fee_calculator, "config", None) != config:
                logger.info("KrakenPlusFeeCalculator config changed; refreshing singleton instance")
                _fee_calculator = KrakenPlusFeeCalculator(config)
    return _fee_calculator


def reset_fee_calculator():
    """Reset singleton (for testing)."""
    global _fee_calculator
    with _fee_calculator_lock:
        _fee_calculator = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_weight(monthly_volume_usd: float) -> float:
    """Compute blending weight for given volume."""
    return get_fee_calculator().blender.compute_weight(monthly_volume_usd)


def apply_fee_blending(fee_std: float, monthly_volume_usd: float) -> float:
    """Apply fee blending to get effective fee."""
    return get_fee_calculator().blender.apply(fee_std, monthly_volume_usd)


def on_trade_fill(
    notional_usd: float,
    fee_std: float,
    symbol: str = "",
    is_spot: bool = True,
    exchange: str = "kraken",
    is_maker: bool = False,
) -> Dict[str, Any]:
    """
    Record a trade fill and calculate effective fee.
    
    Main entry point for integration with execution layer.
    """
    return get_fee_calculator().on_fill(
        notional_usd=notional_usd,
        fee_std=fee_std,
        symbol=symbol,
        is_spot=is_spot,
        exchange=exchange,
        is_maker=is_maker,
    )


def get_monthly_volume() -> float:
    """Get current month's executed volume."""
    return get_fee_calculator().tracker.get_monthly_volume()


def is_fee_blending_enabled() -> bool:
    """Check if fee blending is enabled."""
    try:
        from configs.sota_flags import get_sota_flags
        flags = get_sota_flags()
        return getattr(flags, 'ENABLE_KRAKEN_PLUS_FEE_BLENDING', True)
    except ImportError:
        return get_fee_config_from_env().enabled


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("KRAKEN+ FEE BLENDER TEST")
    print("=" * 60)
    
    # Create fresh calculator for testing
    config = KrakenPlusFeeConfig(
        enabled=True,
        free_tier_usd=10000,
        blend_band_usd=2000,
        volume_file_path="/tmp/test_volume.json",
    )
    
    blender = KrakenPlusFeeBlender(config)
    
    # Test weight computation
    test_volumes = [0, 5000, 9000, 10000, 10500, 11000, 11500, 12000, 15000, 20000]
    print("\nWeight computation tests:")
    print("-" * 50)
    for v in test_volumes:
        w = blender.compute_weight(v)
        fee_std = 0.0026
        fee_eff = blender.apply(fee_std, v)
        zone = "FREE" if v <= 10000 else ("BLEND" if v < 12000 else "FULL")
        print(f"V=${v:>6,}  w={w:.4f}  fee_eff={fee_eff:.6f}  zone={zone}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

#!/usr/bin/env python3
"""
================================================================================
HMATS v5.1 - Data Validation & Fallback Layer
================================================================================
Purpose: Remove random simulation fallbacks and implement proper data handling

P3 SOTA Requirement:
- "去掉'缺数据就随机模拟'的链上特征路径"
- "缺数据时要么置为 NaN + 降权，要么走明确的 fallback"

Problem (Before):
    if not onchain_data:
        return random.uniform(0, 1)  # OFF Pollutes signals!
        
Solution (After):
    if not onchain_data:
        return DataFallback(
            value=0.5,           # Neutral default
            confidence=0.0,      # Zero confidence
            is_missing=True,     # Flag for downstream
            source="default"     # Document source
        )

Architecture:
    Raw Data -> Validator -> Valid? ──Yes──-> Use data (conf=1.0)
                            │
                            No
                            │
                            ▼
                         Fallback Strategy
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
           Default      Historical     Cross-ref
           (conf=0)     (conf=0.5)    (conf=0.7)

Author: HMATS v5.1 P3 Fix
Version: 5.1.0
================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum, auto
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# FALLBACK STRATEGIES
# =============================================================================

class FallbackStrategy(Enum):
    """Data fallback strategies when primary source unavailable."""
    REJECT = "reject"           # Reject signal entirely
    DEFAULT_NEUTRAL = "default" # Use neutral default value
    HISTORICAL_AVG = "historical"  # Use historical average
    CROSS_REFERENCE = "cross_ref"  # Use correlated data source
    LAST_KNOWN = "last_known"   # Use last valid value
    DEGRADED = "degraded"       # Continue with reduced confidence


class DataQuality(Enum):
    """Data quality classification."""
    VALID = "valid"             # Fresh, complete data
    STALE = "stale"             # Valid but old
    PARTIAL = "partial"         # Some fields missing
    FALLBACK = "fallback"       # Using fallback value
    MISSING = "missing"         # No data available
    INVALID = "invalid"         # Data failed validation


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DataFallback:
    """
    Wrapper for data values with confidence and source tracking.
    
    All data that might be missing should return DataFallback instead
    of raw values to enable proper downstream handling.
    """
    value: Any                           # The actual value
    confidence: float = 1.0              # 0.0 = no confidence, 1.0 = full
    quality: DataQuality = DataQuality.VALID
    source: str = "primary"              # Where the value came from
    is_missing: bool = False             # True if using fallback
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stale_seconds: float = 0.0           # How stale the data is
    original_value: Optional[Any] = None # Pre-fallback value if modified
    
    def __float__(self) -> float:
        """Allow using as float in calculations."""
        if isinstance(self.value, (int, float)):
            return float(self.value)
        return 0.0
        
    def __bool__(self) -> bool:
        """Check if data is usable."""
        return self.confidence > 0 and not self.is_missing
        
    @property
    def weight(self) -> float:
        """Get weight for signal fusion (confidence * quality factor)."""
        quality_weights = {
            DataQuality.VALID: 1.0,
            DataQuality.STALE: 0.7,
            DataQuality.PARTIAL: 0.5,
            DataQuality.FALLBACK: 0.2,
            DataQuality.MISSING: 0.0,
            DataQuality.INVALID: 0.0
        }
        return self.confidence * quality_weights.get(self.quality, 0.5)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    quality: DataQuality
    issues: List[str] = field(default_factory=list)
    suggested_fallback: FallbackStrategy = FallbackStrategy.DEFAULT_NEUTRAL


@dataclass 
class OnchainFeatures:
    """
    Onchain features with proper fallback handling.
    
    CRITICAL: No random values allowed! All missing data returns
    DataFallback with confidence=0.
    """
    # DEX metrics
    dex_volume_24h: DataFallback = field(default_factory=lambda: DataFallback(
        value=0.0, confidence=0.0, quality=DataQuality.MISSING,
        source="default", is_missing=True
    ))
    dex_tvl: DataFallback = field(default_factory=lambda: DataFallback(
        value=0.0, confidence=0.0, quality=DataQuality.MISSING,
        source="default", is_missing=True
    ))
    
    # Flow metrics
    exchange_inflow: DataFallback = field(default_factory=lambda: DataFallback(
        value=0.0, confidence=0.0, quality=DataQuality.MISSING,
        source="default", is_missing=True
    ))
    exchange_outflow: DataFallback = field(default_factory=lambda: DataFallback(
        value=0.0, confidence=0.0, quality=DataQuality.MISSING,
        source="default", is_missing=True
    ))
    
    # Address metrics
    active_addresses: DataFallback = field(default_factory=lambda: DataFallback(
        value=0.0, confidence=0.0, quality=DataQuality.MISSING,
        source="default", is_missing=True
    ))
    whale_transactions: DataFallback = field(default_factory=lambda: DataFallback(
        value=0.0, confidence=0.0, quality=DataQuality.MISSING,
        source="default", is_missing=True
    ))
    
    def overall_confidence(self) -> float:
        """Get average confidence across all features."""
        features = [
            self.dex_volume_24h, self.dex_tvl,
            self.exchange_inflow, self.exchange_outflow,
            self.active_addresses, self.whale_transactions
        ]
        return sum(f.confidence for f in features) / len(features)
        
    def usable_features(self) -> int:
        """Count features with usable data."""
        features = [
            self.dex_volume_24h, self.dex_tvl,
            self.exchange_inflow, self.exchange_outflow,
            self.active_addresses, self.whale_transactions
        ]
        return sum(1 for f in features if f.confidence > 0)


# =============================================================================
# DATA VALIDATOR
# =============================================================================

class DataValidator:
    """
    Validates incoming data and handles missing/invalid values.
    
    Usage:
        validator = DataValidator()
        
        # Validate single value
        result = validator.validate_price(price=150.0, symbol="SOL/USD")
        
        # Validate with fallback
        fallback = validator.get_with_fallback(
            value=raw_onchain_data,
            field_name="dex_volume",
            fallback_strategy=FallbackStrategy.HISTORICAL_AVG
        )
    """
    
    def __init__(
        self,
        max_stale_seconds: float = 60.0,
        enable_cross_reference: bool = True
    ):
        self.max_stale_seconds = max_stale_seconds
        self.enable_cross_reference = enable_cross_reference
        
        # Historical averages for fallback
        self._historical: Dict[str, deque] = {}
        self._last_valid: Dict[str, DataFallback] = {}
        
        # Validation rules
        self._rules: Dict[str, Callable] = {}
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """Setup default validation rules."""
        self._rules["price"] = lambda x: x is not None and x > 0
        self._rules["volume"] = lambda x: x is not None and x >= 0
        self._rules["ratio"] = lambda x: x is not None and 0 <= x <= 1
        self._rules["percentage"] = lambda x: x is not None and -100 <= x <= 100
        self._rules["count"] = lambda x: x is not None and x >= 0
        
    def validate(
        self,
        value: Any,
        field_type: str = "generic",
        field_name: str = "unknown",
        timestamp: Optional[datetime] = None
    ) -> ValidationResult:
        """
        Validate a data value.
        
        Args:
            value: Value to validate
            field_type: Type of field (price, volume, ratio, etc.)
            field_name: Name for logging/tracking
            timestamp: Data timestamp for staleness check
            
        Returns:
            ValidationResult with quality assessment
        """
        issues = []
        
        # Check for None/NaN
        if value is None:
            return ValidationResult(
                is_valid=False,
                quality=DataQuality.MISSING,
                issues=["Value is None"],
                suggested_fallback=FallbackStrategy.DEFAULT_NEUTRAL
            )
            
        if isinstance(value, float) and np.isnan(value):
            return ValidationResult(
                is_valid=False,
                quality=DataQuality.INVALID,
                issues=["Value is NaN"],
                suggested_fallback=FallbackStrategy.LAST_KNOWN
            )
            
        # Check staleness
        if timestamp:
            age = (datetime.now(timezone.utc) - timestamp).total_seconds()
            if age > self.max_stale_seconds:
                issues.append(f"Data is stale ({age:.1f}s old)")
                return ValidationResult(
                    is_valid=True,
                    quality=DataQuality.STALE,
                    issues=issues,
                    suggested_fallback=FallbackStrategy.LAST_KNOWN
                )
                
        # Run type-specific validation
        rule = self._rules.get(field_type)
        if rule and not rule(value):
            issues.append(f"Failed {field_type} validation")
            return ValidationResult(
                is_valid=False,
                quality=DataQuality.INVALID,
                issues=issues,
                suggested_fallback=FallbackStrategy.DEFAULT_NEUTRAL
            )
            
        return ValidationResult(
            is_valid=True,
            quality=DataQuality.VALID,
            issues=[]
        )
        
    def get_with_fallback(
        self,
        value: Any,
        field_name: str,
        field_type: str = "generic",
        fallback_strategy: FallbackStrategy = FallbackStrategy.DEFAULT_NEUTRAL,
        default_value: Any = 0.0,
        timestamp: Optional[datetime] = None
    ) -> DataFallback:
        """
        Get value with proper fallback handling.
        
        CRITICAL: This is the main entry point for data that might be missing.
        Never use random values - always return DataFallback with confidence.
        
        Args:
            value: Raw value to validate
            field_name: Field identifier
            field_type: Type for validation
            fallback_strategy: Strategy if validation fails
            default_value: Default if using DEFAULT_NEUTRAL
            timestamp: Data timestamp
            
        Returns:
            DataFallback with value and confidence
        """
        # Validate
        result = self.validate(value, field_type, field_name, timestamp)
        
        # Valid data - return with full confidence
        if result.is_valid and result.quality == DataQuality.VALID:
            fallback = DataFallback(
                value=value,
                confidence=1.0,
                quality=DataQuality.VALID,
                source="primary",
                is_missing=False,
                timestamp=timestamp or datetime.now(timezone.utc)
            )
            self._update_history(field_name, fallback)
            return fallback
            
        # Stale data - reduced confidence
        if result.quality == DataQuality.STALE:
            age = (datetime.now(timezone.utc) - timestamp).total_seconds() if timestamp else 0
            confidence = max(0.3, 1.0 - age / (self.max_stale_seconds * 3))
            return DataFallback(
                value=value,
                confidence=confidence,
                quality=DataQuality.STALE,
                source="primary_stale",
                is_missing=False,
                timestamp=timestamp or datetime.now(timezone.utc),
                stale_seconds=age
            )
            
        # Invalid/missing - use fallback strategy
        return self._apply_fallback(
            field_name=field_name,
            strategy=fallback_strategy,
            default_value=default_value
        )
        
    def _apply_fallback(
        self,
        field_name: str,
        strategy: FallbackStrategy,
        default_value: Any
    ) -> DataFallback:
        """Apply fallback strategy."""
        
        if strategy == FallbackStrategy.REJECT:
            logger.warning(f"[DataValidator] Rejecting {field_name} - no fallback")
            return DataFallback(
                value=None,
                confidence=0.0,
                quality=DataQuality.MISSING,
                source="rejected",
                is_missing=True
            )
            
        elif strategy == FallbackStrategy.LAST_KNOWN:
            last = self._last_valid.get(field_name)
            if last and last.confidence > 0:
                age = (datetime.now(timezone.utc) - last.timestamp).total_seconds()
                decay = max(0.1, 1.0 - age / 3600)  # Decay over 1 hour
                return DataFallback(
                    value=last.value,
                    confidence=last.confidence * decay * 0.5,
                    quality=DataQuality.FALLBACK,
                    source="last_known",
                    is_missing=True,
                    original_value=None
                )
                
        elif strategy == FallbackStrategy.HISTORICAL_AVG:
            history = self._historical.get(field_name)
            if history and len(history) > 0:
                avg = np.mean([f.value for f in history if isinstance(f.value, (int, float))])
                return DataFallback(
                    value=avg,
                    confidence=0.3,
                    quality=DataQuality.FALLBACK,
                    source="historical_avg",
                    is_missing=True
                )
                
        # Default neutral fallback
        return DataFallback(
            value=default_value,
            confidence=0.0,  # Zero confidence for default
            quality=DataQuality.FALLBACK,
            source="default",
            is_missing=True
        )
        
    def _update_history(self, field_name: str, fallback: DataFallback):
        """Update historical tracking."""
        if field_name not in self._historical:
            self._historical[field_name] = deque(maxlen=100)
        self._historical[field_name].append(fallback)
        self._last_valid[field_name] = fallback


# =============================================================================
# ONCHAIN DATA PROCESSOR
# =============================================================================

class OnchainDataProcessor:
    """
    Process onchain data with proper validation and fallbacks.
    
    CRITICAL: Replaces random simulation with proper fallback handling.
    """
    
    def __init__(self):
        self.validator = DataValidator()
        self._cache: Dict[str, OnchainFeatures] = {}
        
    def process_raw_data(
        self,
        symbol: str,
        raw_data: Optional[Dict[str, Any]]
    ) -> OnchainFeatures:
        """
        Process raw onchain data into validated features.
        
        Args:
            symbol: Asset symbol
            raw_data: Raw data dict (may be None or incomplete)
            
        Returns:
            OnchainFeatures with proper fallback handling
        """
        if raw_data is None:
            logger.warning(f"[Onchain] No data for {symbol} - using full fallback")
            return OnchainFeatures()  # All defaults with confidence=0
            
        features = OnchainFeatures()
        
        # Process each field with validation
        features.dex_volume_24h = self.validator.get_with_fallback(
            value=raw_data.get("dex_volume_24h"),
            field_name=f"{symbol}_dex_volume",
            field_type="volume",
            fallback_strategy=FallbackStrategy.HISTORICAL_AVG,
            default_value=0.0
        )
        
        features.dex_tvl = self.validator.get_with_fallback(
            value=raw_data.get("dex_tvl"),
            field_name=f"{symbol}_dex_tvl",
            field_type="volume",
            fallback_strategy=FallbackStrategy.LAST_KNOWN,
            default_value=0.0
        )
        
        features.exchange_inflow = self.validator.get_with_fallback(
            value=raw_data.get("exchange_inflow"),
            field_name=f"{symbol}_inflow",
            field_type="volume",
            fallback_strategy=FallbackStrategy.DEFAULT_NEUTRAL,
            default_value=0.0
        )
        
        features.exchange_outflow = self.validator.get_with_fallback(
            value=raw_data.get("exchange_outflow"),
            field_name=f"{symbol}_outflow",
            field_type="volume",
            fallback_strategy=FallbackStrategy.DEFAULT_NEUTRAL,
            default_value=0.0
        )
        
        features.active_addresses = self.validator.get_with_fallback(
            value=raw_data.get("active_addresses"),
            field_name=f"{symbol}_addresses",
            field_type="count",
            fallback_strategy=FallbackStrategy.HISTORICAL_AVG,
            default_value=0
        )
        
        features.whale_transactions = self.validator.get_with_fallback(
            value=raw_data.get("whale_transactions"),
            field_name=f"{symbol}_whales",
            field_type="count",
            fallback_strategy=FallbackStrategy.DEFAULT_NEUTRAL,
            default_value=0
        )
        
        # Cache for future reference
        self._cache[symbol] = features
        
        # Log quality
        logger.info(
            f"[Onchain] {symbol}: {features.usable_features()}/6 features usable, "
            f"avg confidence={features.overall_confidence():.2f}"
        )
        
        return features


# =============================================================================
# SIGNAL WEIGHT ADJUSTER
# =============================================================================

class SignalWeightAdjuster:
    """
    Adjust signal weights based on data quality.
    
    Ensures signals with missing data are properly downweighted.
    """
    
    @staticmethod
    def adjust_signal_weight(
        base_weight: float,
        data_quality: DataQuality,
        confidence: float
    ) -> float:
        """
        Adjust signal weight based on data quality.
        
        Args:
            base_weight: Original signal weight
            data_quality: Quality of underlying data
            confidence: Data confidence score
            
        Returns:
            Adjusted weight (always <= base_weight)
        """
        quality_multipliers = {
            DataQuality.VALID: 1.0,
            DataQuality.STALE: 0.7,
            DataQuality.PARTIAL: 0.5,
            DataQuality.FALLBACK: 0.1,  # Heavily penalize fallback
            DataQuality.MISSING: 0.0,    # Zero weight if missing
            DataQuality.INVALID: 0.0
        }
        
        quality_mult = quality_multipliers.get(data_quality, 0.5)
        return base_weight * quality_mult * confidence
        
    @staticmethod
    def should_include_signal(
        data_quality: DataQuality,
        confidence: float,
        min_confidence: float = 0.1
    ) -> bool:
        """
        Determine if signal should be included in fusion.
        
        Returns:
            True if signal has enough quality to include
        """
        if data_quality in [DataQuality.MISSING, DataQuality.INVALID]:
            return False
        return confidence >= min_confidence


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_validator: Optional[DataValidator] = None
_onchain_processor: Optional[OnchainDataProcessor] = None


def get_data_validator() -> DataValidator:
    """Get singleton DataValidator instance."""
    global _validator
    if _validator is None:
        _validator = DataValidator()
    return _validator


def get_onchain_processor() -> OnchainDataProcessor:
    """Get singleton OnchainDataProcessor instance."""
    global _onchain_processor
    if _onchain_processor is None:
        _onchain_processor = OnchainDataProcessor()
    return _onchain_processor


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'DataValidator',
    'DataFallback',
    'ValidationResult',
    'FallbackStrategy',
    'DataQuality',
    'OnchainFeatures',
    'OnchainDataProcessor',
    'SignalWeightAdjuster',
    'get_data_validator',
    'get_onchain_processor',
]

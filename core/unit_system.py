"""
================================================================================
HMATS Unit System - Canonical Unit Definitions
================================================================================
Version: 5.4.1-P0FIX
Purpose: Enforce strict unit typing to prevent exposure fraction / quantity mixing

CRITICAL DESIGN RULE:
    - Strategy / Fusion layers output EXPOSURE_FRACTION (-1.0 to +1.0)
    - Execution layer receives BASE_QUANTITY (e.g., 0.5 BTC)
    - Conversion happens ONLY in this module, NOWHERE else

Usage:
    from core.unit_system import exposure_to_quantity, TypedValue, UnitType
    
    # Convert exposure to tradeable quantity
    qty = exposure_to_quantity(
        exposure_fraction=intent.target_exposure,
        account_equity=await get_account_equity(),
        price=current_price,
        asset="SOL"
    )

================================================================================
"""

import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from enum import Enum, auto
from typing import Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# UNIT TYPES
# =============================================================================

class UnitType(Enum):
    """
    Explicit unit types to prevent mixing.
    
    EXPOSURE_FRACTION: -1.0 to +1.0 (账户净值的风险暴露比例)
        - Used by: Strategy, Fusion, TradeIntent
        - Example: 0.3 means 30% of account equity
        
    USD_NOTIONAL: Dollar value of position
        - Used by: Risk calculations, P&L
        - Example: $30,000
        
    BASE_QUANTITY: Units of base currency
        - Used by: Execution, Exchange API
        - Example: 162.16 SOL
        
    QUOTE_QUANTITY: Units of quote currency (usually USD)
        - Used by: Margin calculations
        - Example: $30,000 USD
    """
    EXPOSURE_FRACTION = auto()
    USD_NOTIONAL = auto()
    BASE_QUANTITY = auto()
    QUOTE_QUANTITY = auto()


# =============================================================================
# TYPED VALUE - PREVENTS IMPLICIT UNIT MIXING
# =============================================================================

@dataclass(frozen=True)
class TypedValue:
    """
    A numeric value with explicit unit type.
    
    Frozen (immutable) to prevent accidental modification.
    Conversion methods are explicit and type-checked.
    
    Example:
        exposure = TypedValue(Decimal("0.3"), UnitType.EXPOSURE_FRACTION, "SOL")
        usd = exposure.to_usd_notional(account_equity=Decimal("100000"))
        qty = usd.to_base_quantity(price=Decimal("185"))
        
        # qty.value = Decimal("162.16...")
        # qty.unit = UnitType.BASE_QUANTITY
    """
    value: Decimal
    unit: UnitType
    asset: str = ""
    
    def __post_init__(self):
        """Validate value ranges based on unit type."""
        if self.unit == UnitType.EXPOSURE_FRACTION:
            if not (-1.0 <= float(self.value) <= 1.0):
                raise ValueError(f"EXPOSURE_FRACTION must be in [-1.0, 1.0], got {self.value}")
        elif self.unit in (UnitType.USD_NOTIONAL, UnitType.BASE_QUANTITY, UnitType.QUOTE_QUANTITY):
            if float(self.value) < 0:
                raise ValueError(f"{self.unit.name} cannot be negative, got {self.value}")
    
    def to_usd_notional(self, account_equity: Union[Decimal, float]) -> 'TypedValue':
        """
        Convert EXPOSURE_FRACTION to USD_NOTIONAL.
        
        Args:
            account_equity: Current account equity in USD
            
        Returns:
            TypedValue with unit=USD_NOTIONAL
            
        Raises:
            ValueError: If current unit is not EXPOSURE_FRACTION
        """
        if self.unit != UnitType.EXPOSURE_FRACTION:
            raise ValueError(
                f"Cannot convert {self.unit.name} to USD_NOTIONAL. "
                f"Only EXPOSURE_FRACTION -> USD_NOTIONAL is allowed."
            )
        
        equity = Decimal(str(account_equity))
        usd_value = abs(self.value) * equity
        
        return TypedValue(
            value=usd_value.quantize(Decimal("0.01"), rounding=ROUND_DOWN),
            unit=UnitType.USD_NOTIONAL,
            asset=self.asset
        )
    
    def to_base_quantity(self, price: Union[Decimal, float]) -> 'TypedValue':
        """
        Convert USD_NOTIONAL to BASE_QUANTITY.
        
        Args:
            price: Current asset price in USD per unit
            
        Returns:
            TypedValue with unit=BASE_QUANTITY
            
        Raises:
            ValueError: If current unit is not USD_NOTIONAL
        """
        if self.unit != UnitType.USD_NOTIONAL:
            raise ValueError(
                f"Cannot convert {self.unit.name} to BASE_QUANTITY. "
                f"Only USD_NOTIONAL -> BASE_QUANTITY is allowed."
            )
        
        p = Decimal(str(price))
        if p <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        
        qty = self.value / p
        
        return TypedValue(
            value=qty.quantize(Decimal("0.00001"), rounding=ROUND_DOWN),
            unit=UnitType.BASE_QUANTITY,
            asset=self.asset
        )
    
    def __float__(self) -> float:
        """Allow float() conversion for compatibility."""
        return float(self.value)
    
    def __str__(self) -> str:
        return f"{self.value} {self.unit.name} ({self.asset})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# L4-06: Kraken minimum step sizes per asset
KRAKEN_STEP_SIZE = {
    'BTC': 0.0001,   # ~$6 @ $60K
    'ETH': 0.001,    # ~$3 @ $3K
    'SOL': 0.1,      # ~$15 @ $150
}


def quantize_to_step(quantity: float, asset: str) -> float:
    """Round down to Kraken's minimum step size (floor = conservative)."""
    import math
    step = KRAKEN_STEP_SIZE.get(asset.upper(), 0.001)
    return math.floor(quantity / step) * step


def exposure_to_quantity(
    exposure_fraction: float,
    account_equity: float,
    price: float,
    asset: str
) -> float:
    """
    THE canonical conversion from exposure fraction to base quantity.
    
    This is the ONLY function that should be used to convert strategy
    output (exposure fraction) to execution input (base quantity).
    
    Args:
        exposure_fraction: Target exposure as fraction of account (-1.0 to 1.0)
        account_equity: Current account equity in USD
        price: Current asset price in USD per unit
        asset: Asset symbol (for logging)
        
    Returns:
        Base quantity to trade (e.g., number of SOL)
        
    Example:
        # Account: $100,000
        # Target: 30% long SOL
        # SOL price: $185
        
        qty = exposure_to_quantity(
            exposure_fraction=0.3,
            account_equity=100000.0,
            price=185.0,
            asset="SOL"
        )
        # Returns: 162.16 (i.e., buy 162.16 SOL)
    """
    # Validation
    if not (-1.0 <= exposure_fraction <= 1.0):
        raise ValueError(f"exposure_fraction must be in [-1.0, 1.0], got {exposure_fraction}")
    
    if account_equity <= 0:
        raise ValueError(f"account_equity must be positive, got {account_equity}")
    
    # P1 FIX: Check for NaN and invalid prices
    import math
    if price <= 0 or math.isnan(price) or math.isinf(price):
        raise ValueError(f"price must be positive and finite, got {price}")
    
    # Calculate USD notional
    usd_notional = abs(exposure_fraction) * account_equity

    # Convert to base quantity
    base_quantity = usd_notional / price

    # L4-06: Quantize to Kraken step size (floor = conservative)
    base_quantity = quantize_to_step(base_quantity, asset)

    logger.info(
        f"[UNIT_CONVERT] {asset}: "
        f"exposure={exposure_fraction:.2%} -> "
        f"USD=${usd_notional:,.2f} -> "
        f"qty={base_quantity:.6f}"
    )

    return base_quantity


def quantity_to_exposure(
    base_quantity: float,
    price: float,
    account_equity: float,
    is_short: bool = False
) -> float:
    """
    Reverse conversion: base quantity to exposure fraction.
    
    Useful for position reconciliation and display.
    
    Args:
        base_quantity: Position size in base currency
        price: Current asset price
        account_equity: Current account equity
        is_short: If True, return negative exposure
        
    Returns:
        Exposure fraction (-1.0 to 1.0)
    """
    if account_equity <= 0:
        return 0.0
    
    usd_notional = abs(base_quantity) * price
    exposure = usd_notional / account_equity
    exposure = min(exposure, 1.0)  # Cap at 100%
    
    if is_short:
        exposure = -exposure
    
    return exposure


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_exposure_fraction(value: float, context: str = "") -> bool:
    """
    Validate that a value is a valid exposure fraction.
    
    Args:
        value: Value to validate
        context: Context for error message
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If invalid
    """
    if not (-1.0 <= value <= 1.0):
        raise ValueError(
            f"[{context}] Invalid exposure fraction: {value}. "
            f"Must be in [-1.0, 1.0]"
        )
    return True


def validate_base_quantity(value: float, asset: str, context: str = "") -> bool:
    """
    Validate that a value is a valid base quantity.
    
    Args:
        value: Value to validate
        asset: Asset symbol
        context: Context for error message
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If invalid
    """
    if value < 0:
        raise ValueError(
            f"[{context}] Invalid base quantity for {asset}: {value}. "
            f"Cannot be negative."
        )
    
    # Asset-specific limits (rough sanity checks)
    MAX_QUANTITIES = {
        "BTC": 100.0,
        "ETH": 1000.0,
        "SOL": 10000.0,
    }
    
    max_qty = MAX_QUANTITIES.get(asset.upper(), 1000000.0)
    if value > max_qty:
        raise ValueError(
            f"[{context}] Suspiciously large quantity for {asset}: {value}. "
            f"Max expected: {max_qty}"
        )
    
    return True


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Unit System Self-Test")
    print("=" * 60)
    
    # Test 1: Basic conversion
    print("\n[Test 1] Basic exposure -> quantity conversion")
    qty = exposure_to_quantity(
        exposure_fraction=0.3,
        account_equity=100000.0,
        price=185.0,
        asset="SOL"
    )
    expected = 30000.0 / 185.0  # ~162.16
    assert abs(qty - expected) < 0.01, f"Expected ~{expected}, got {qty}"
    print(f"  ✓ 30% of $100k at $185/SOL = {qty:.4f} SOL")
    
    # Test 2: TypedValue chain
    print("\n[Test 2] TypedValue conversion chain")
    exposure = TypedValue(Decimal("0.25"), UnitType.EXPOSURE_FRACTION, "BTC")
    usd = exposure.to_usd_notional(account_equity=200000.0)
    qty_typed = usd.to_base_quantity(price=95000.0)
    print(f"  Exposure: {exposure}")
    print(f"  USD: {usd}")
    print(f"  Quantity: {qty_typed}")
    assert qty_typed.unit == UnitType.BASE_QUANTITY
    
    # Test 3: Invalid conversion rejection
    print("\n[Test 3] Invalid conversion rejection")
    try:
        qty_typed.to_usd_notional(100000)  # Can't convert BASE_QUANTITY to USD
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Correctly rejected: {e}")
    
    # Test 4: Range validation
    print("\n[Test 4] Range validation")
    try:
        TypedValue(Decimal("1.5"), UnitType.EXPOSURE_FRACTION, "SOL")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Correctly rejected out-of-range: {e}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

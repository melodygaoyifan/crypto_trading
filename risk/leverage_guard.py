"""
================================================================================
HMATS Leverage Guard - Margin & Leverage Control
================================================================================
Version: 5.4.1-P0FIX
Purpose: Enforce ISOLATED MARGIN ONLY and MAX LEVERAGE limits

CRITICAL CONSTRAINTS (from design doc):
    1. Isolated margin ONLY - Cross margin is FORBIDDEN
    2. Max leverage: 1-3x (configurable, default 3x)
    3. Pre-liquidation buffer stop: 5% before exchange liquidation
    4. No martingale (no adding to losing positions)

Usage:
    from risk.leverage_guard import LeverageGuard, LeverageConfig
    
    guard = LeverageGuard()
    
    allowed, reason = guard.validate_order(
        position_notional_usd=50000.0,
        account_equity=20000.0,
        current_price=185.0,
        is_short=True,
    )
    
    if not allowed:
        raise OrderRejected(reason)

================================================================================
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# MARGIN MODE ENUM
# =============================================================================

class MarginMode(Enum):
    """Margin mode for derivatives trading."""
    ISOLATED = "isolated"  # Each position has its own margin
    CROSS = "cross"        # FORBIDDEN - shares margin across positions


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LeverageConfig:
    """
    Leverage and margin configuration.
    
    These are NON-NEGOTIABLE safety constraints for derivatives trading.
    """
    # Leverage limits
    max_leverage: float = 3.0           # Maximum allowed leverage
    min_leverage: float = 1.0           # Minimum (1x = no leverage)
    
    # Margin mode (CRITICAL: Only ISOLATED is allowed)
    margin_mode: MarginMode = MarginMode.ISOLATED
    
    # Liquidation protection
    liquidation_buffer_pct: float = 0.05  # 5% buffer before exchange liquidation
    pre_liq_stop_pct: float = 0.03        # 3% stop before our buffer
    
    # Position limits
    max_position_notional_usd: float = 50000.0   # Max single position
    max_total_notional_usd: float = 100000.0     # Max total exposure
    # [P1-FIX] Percentage-based limits relative to current equity (most restrictive wins)
    max_single_position_pct: float = 5.0         # 5x equity = max single position
    max_total_notional_pct: float = 10.0         # 10x equity = max total exposure
    
    # Anti-martingale
    allow_adding_to_losers: bool = False  # FORBIDDEN by default
    
    def __post_init__(self):
        """Validate configuration constraints."""
        # CRITICAL: Enforce isolated margin only
        if self.margin_mode != MarginMode.ISOLATED:
            raise ValueError(
                "CRITICAL SAFETY VIOLATION: Only ISOLATED margin mode is allowed! "
                "Cross margin can lead to total account wipeout."
            )
        
        # Validate leverage range
        if not (1.0 <= self.max_leverage <= 10.0):
            raise ValueError(f"max_leverage must be in [1.0, 10.0], got {self.max_leverage}")
        
        # Warn if leverage > 3x
        if self.max_leverage > 3.0:
            logger.warning(
                f"[LEVERAGE_GUARD] max_leverage={self.max_leverage}x is above recommended 3x. "
                f"High leverage significantly increases liquidation risk."
            )
        
        # Validate buffer
        if not (0.01 <= self.liquidation_buffer_pct <= 0.20):
            raise ValueError(f"liquidation_buffer_pct must be in [0.01, 0.20]")


# =============================================================================
# LEVERAGE GUARD
# =============================================================================

class LeverageGuard:
    """
    Leverage and margin safety guard.
    
    This guard runs BEFORE execution to ensure:
    1. Position would not exceed max leverage
    2. Position size is within limits
    3. We have buffer before liquidation
    4. We're not adding to losing positions (martingale)
    
    FAIL-CLOSED: Any error or exception results in order rejection.
    """
    
    def __init__(self, config: Optional[LeverageConfig] = None):
        """
        Initialize leverage guard.
        
        Args:
            config: Leverage configuration (uses safe defaults if None)
        """
        self.config = config or LeverageConfig()
        
        # Position tracking for anti-martingale
        self._position_pnls: Dict[str, float] = {}
        
        # Statistics
        self._stats = {
            "orders_checked": 0,
            "orders_approved": 0,
            "orders_rejected": 0,
            "rejection_reasons": {},
        }
        
        logger.info(
            f"[LEVERAGE_GUARD] Initialized: "
            f"max_leverage={self.config.max_leverage}x, "
            f"margin_mode={self.config.margin_mode.value}, "
            f"liq_buffer={self.config.liquidation_buffer_pct:.1%}"
        )
    
    def validate_order(
        self,
        position_notional_usd: float,
        account_equity: float,
        current_price: float = 0.0,
        entry_price: Optional[float] = None,
        is_short: bool = False,
        asset: str = "",
        existing_position_usd: float = 0.0,
        existing_pnl: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Validate an order against leverage constraints.
        
        Args:
            position_notional_usd: USD value of the position after this order
            account_equity: Current account equity in USD
            current_price: Current asset price (for liquidation calc)
            entry_price: Entry price (for liquidation calc)
            is_short: True if this is a short position
            asset: Asset symbol (for tracking)
            existing_position_usd: Current position size in USD
            existing_pnl: Current unrealized P&L
            
        Returns:
            (approved, reason) tuple
        """
        self._stats["orders_checked"] += 1

        try:
            # Check 0: Margin health breach — block risk-increasing, allow reduce-only
            if self.is_margin_breach():
                _is_reducing = position_notional_usd < existing_position_usd
                if not _is_reducing:
                    return self._reject(
                        f"MARGIN_BREACH: risk-increasing order blocked. "
                        f"Only reduce-risk orders allowed until margin health recovers."
                    )
                logger.info("[LEVERAGE_GUARD] Margin breach active but order is risk-reducing — allowing")

            # Check 1: Leverage limit
            approved, reason = self._check_leverage_limit(
                position_notional_usd, account_equity
            )
            if not approved:
                return self._reject(reason)
            
            # Check 2: Position size limit
            approved, reason = self._check_position_limit(position_notional_usd, account_equity)
            if not approved:
                return self._reject(reason)
            
            # Check 3: Total exposure limit
            approved, reason = self._check_total_exposure(
                position_notional_usd, account_equity, existing_position_usd
            )
            if not approved:
                return self._reject(reason)
            
            # Check 4: Liquidation buffer (if prices provided)
            if current_price > 0 and entry_price:
                approved, reason = self._check_liquidation_buffer(
                    position_notional_usd, account_equity, 
                    entry_price, current_price, is_short
                )
                if not approved:
                    return self._reject(reason)
            
            # Check 5: Anti-martingale (if adding to position with loss)
            if existing_position_usd > 0 and existing_pnl < 0:
                approved, reason = self._check_anti_martingale(
                    existing_pnl, asset
                )
                if not approved:
                    return self._reject(reason)
            
            # All checks passed
            self._stats["orders_approved"] += 1
            return True, "Approved"
            
        except Exception as e:
            # FAIL-CLOSED: Any exception = reject
            logger.error(f"[LEVERAGE_GUARD] Exception during validation: {e}")
            return self._reject(f"Validation error: {e}")
    
    def _check_leverage_limit(
        self, 
        position_notional_usd: float, 
        account_equity: float
    ) -> Tuple[bool, str]:
        """Check if position would exceed leverage limit."""
        if account_equity <= 0:
            return False, "Account equity is zero or negative"
        
        implied_leverage = position_notional_usd / account_equity
        
        if implied_leverage > self.config.max_leverage:
            return False, (
                f"Leverage {implied_leverage:.1f}x exceeds maximum "
                f"{self.config.max_leverage}x "
                f"(position=${position_notional_usd:,.0f}, equity=${account_equity:,.0f})"
            )
        
        return True, ""
    
    def _check_position_limit(
        self,
        position_notional_usd: float,
        account_equity: float = 0.0,
    ) -> Tuple[bool, str]:
        """Check if position exceeds size limit."""
        if position_notional_usd > self.config.max_position_notional_usd:
            return False, (
                f"Position ${position_notional_usd:,.0f} exceeds maximum "
                f"${self.config.max_position_notional_usd:,.0f}"
            )

        # [P1-FIX] Percentage-based cap relative to current equity
        if account_equity > 0:
            pct_limit = account_equity * self.config.max_single_position_pct
            if position_notional_usd > pct_limit:
                return False, (
                    f"Position ${position_notional_usd:,.0f} exceeds "
                    f"{self.config.max_single_position_pct}x equity cap "
                    f"(${pct_limit:,.0f} on ${account_equity:,.0f} equity)"
                )

        return True, ""
    
    def _check_total_exposure(
        self,
        new_position_usd: float,
        account_equity: float,
        existing_position_usd: float,
    ) -> Tuple[bool, str]:
        """Check if total exposure would exceed limit."""
        # Simple addition for now (proper implementation would sum all positions)
        total = existing_position_usd + new_position_usd

        if total > self.config.max_total_notional_usd:
            return False, (
                f"Total exposure ${total:,.0f} would exceed maximum "
                f"${self.config.max_total_notional_usd:,.0f}"
            )

        # [P1-FIX] Percentage-based total exposure cap relative to current equity
        if account_equity > 0:
            pct_limit = account_equity * self.config.max_total_notional_pct
            if total > pct_limit:
                return False, (
                    f"Total exposure ${total:,.0f} exceeds "
                    f"{self.config.max_total_notional_pct}x equity cap "
                    f"(${pct_limit:,.0f} on ${account_equity:,.0f} equity)"
                )

        return True, ""
    
    def _check_liquidation_buffer(
        self,
        position_notional_usd: float,
        account_equity: float,
        entry_price: float,
        current_price: float,
        is_short: bool,
    ) -> Tuple[bool, str]:
        """Check if position has adequate liquidation buffer."""
        # Calculate implied leverage
        leverage = position_notional_usd / max(account_equity, 1.0)
        
        # Calculate liquidation price
        # For isolated margin:
        # Long: liq_price = entry_price * (1 - 1/leverage + margin_rate)
        # Short: liq_price = entry_price * (1 + 1/leverage - margin_rate)
        margin_rate = 0.02  # Typical maintenance margin
        
        if is_short:
            liq_price = entry_price * (1 + (1 / max(leverage, 1.0)) - margin_rate)
            # For shorts, liquidation happens when price goes UP
            buffer_to_liq = (liq_price - current_price) / current_price
        else:
            liq_price = entry_price * (1 - (1 / max(leverage, 1.0)) + margin_rate)
            # For longs, liquidation happens when price goes DOWN
            buffer_to_liq = (current_price - liq_price) / current_price
        
        if buffer_to_liq < self.config.liquidation_buffer_pct:
            return False, (
                f"Insufficient liquidation buffer: {buffer_to_liq:.1%} < "
                f"required {self.config.liquidation_buffer_pct:.1%} "
                f"(liq_price=${liq_price:,.2f}, current=${current_price:,.2f})"
            )
        
        return True, ""
    
    def _check_anti_martingale(
        self, 
        existing_pnl: float, 
        asset: str
    ) -> Tuple[bool, str]:
        """Check if we're trying to add to a losing position."""
        if not self.config.allow_adding_to_losers and existing_pnl < 0:
            return False, (
                f"MARTINGALE BLOCKED: Cannot add to losing position in {asset} "
                f"(current P&L: ${existing_pnl:,.2f})"
            )
        
        return True, ""
    
    def _reject(self, reason: str) -> Tuple[bool, str]:
        """Record rejection and return result."""
        self._stats["orders_rejected"] += 1
        
        # Track reason
        reason_key = reason.split(":")[0] if ":" in reason else reason[:30]
        self._stats["rejection_reasons"][reason_key] = \
            self._stats["rejection_reasons"].get(reason_key, 0) + 1
        
        logger.warning(f"[LEVERAGE_GUARD] REJECTED: {reason}")
        return False, reason
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def calculate_max_position(self, account_equity: float) -> float:
        """
        Calculate maximum allowed position size.
        
        Returns:
            Maximum position in USD
        """
        leverage_limit = account_equity * self.config.max_leverage
        absolute_limit = self.config.max_position_notional_usd
        return min(leverage_limit, absolute_limit)
    
    def calculate_liquidation_price(
        self,
        entry_price: float,
        leverage: float,
        is_short: bool,
        margin_rate: float = 0.02
    ) -> float:
        """
        Calculate liquidation price for a position.
        
        Args:
            entry_price: Entry price
            leverage: Position leverage
            is_short: True if short position
            margin_rate: Maintenance margin rate
            
        Returns:
            Liquidation price
        """
        if is_short:
            return entry_price * (1 + (1 / max(leverage, 1.0)) - margin_rate)
        else:
            return entry_price * (1 - (1 / max(leverage, 1.0)) + margin_rate)
    
    def calculate_safe_stop_price(
        self,
        entry_price: float,
        leverage: float,
        is_short: bool,
    ) -> float:
        """
        Calculate a safe stop-loss price with buffer before liquidation.
        
        Args:
            entry_price: Entry price
            leverage: Position leverage
            is_short: True if short position
            
        Returns:
            Recommended stop-loss price
        """
        liq_price = self.calculate_liquidation_price(entry_price, leverage, is_short)
        
        # Add buffer
        buffer = self.config.liquidation_buffer_pct + self.config.pre_liq_stop_pct
        
        if is_short:
            # For shorts, stop should be below liq price (lower price = more profit)
            return liq_price * (1 - buffer)
        else:
            # For longs, stop should be above liq price
            return liq_price * (1 + buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get guard statistics."""
        return {
            **self._stats,
            "approval_rate": (
                self._stats["orders_approved"] / max(self._stats["orders_checked"], 1)
            ),
            "config": {
                "max_leverage": self.config.max_leverage,
                "margin_mode": self.config.margin_mode.value,
                "liquidation_buffer": self.config.liquidation_buffer_pct,
            }
        }
    
    def update_position_pnl(self, asset: str, pnl: float):
        """Update tracked P&L for anti-martingale checking."""
        self._position_pnls[asset] = pnl

    # =========================================================================
    # MARGIN HEALTH AWARENESS (Phase 6)
    # =========================================================================

    def update_margin_health(
        self,
        effective_leverage: float,
        initial_margin_pct: float,
        maintenance_margin_pct: float,
        unrealized_funding: float = 0.0,
        equity: float = 0.0,
    ):
        """Ingest margin health fields from derivatives account state.

        Can be called each tick with data from account_sync or derivatives client.
        Works in data_only mode (no execution needed to track margin).
        """
        self._margin_health = {
            "effective_leverage": effective_leverage,
            "initial_margin_pct": initial_margin_pct,
            "maintenance_margin_pct": maintenance_margin_pct,
            "unrealized_funding": unrealized_funding,
            "equity": equity,
            "timestamp": time.time(),
        }

        # Check configurable guards
        breaches = []

        if effective_leverage > self.config.max_leverage:
            breaches.append(
                f"EFFECTIVE_LEVERAGE {effective_leverage:.1f}x > max {self.config.max_leverage:.1f}x"
            )

        # Maintenance margin buffer: warn if IM used > 80% of equity
        if equity > 0 and initial_margin_pct > 0:
            im_ratio = initial_margin_pct / 100.0
            if im_ratio > 0.80:
                breaches.append(
                    f"MARGIN_UTILIZATION {im_ratio:.0%} > 80% — approaching liquidation zone"
                )

        # Liquidation proximity: warn if MM buffer < configured threshold
        if equity > 0 and maintenance_margin_pct > 0:
            mm_buffer = 1.0 - (maintenance_margin_pct / 100.0)
            if mm_buffer < self.config.liquidation_buffer_pct:
                breaches.append(
                    f"LIQUIDATION_PROXIMITY buffer={mm_buffer:.1%} < "
                    f"required={self.config.liquidation_buffer_pct:.1%}"
                )

        if breaches:
            self._margin_breach_active = True
            for b in breaches:
                logger.warning(f"[LEVERAGE_GUARD] MARGIN BREACH: {b}")
        else:
            self._margin_breach_active = False

    def is_margin_breach(self) -> bool:
        """True if any margin health guard is breached.

        When breached:
        - Block risk-increasing orders (new entries, scale-in)
        - Allow reduce-risk only (partial close, full close)
        - Escalate to kill-switch if configured
        """
        return getattr(self, "_margin_breach_active", False)

    def get_margin_health(self) -> Dict[str, Any]:
        """Get current margin health state for monitoring."""
        return getattr(self, "_margin_health", {})


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_leverage_guard: Optional[LeverageGuard] = None


def get_leverage_guard(config: Optional[LeverageConfig] = None) -> LeverageGuard:
    """Get or create the leverage guard singleton."""
    global _leverage_guard
    if _leverage_guard is None:
        _leverage_guard = LeverageGuard(config)
    elif config is not None and _leverage_guard.config != config:
        logger.info("[LEVERAGE_GUARD] Config changed; refreshing singleton instance")
        _leverage_guard = LeverageGuard(config)
    return _leverage_guard


def reset_leverage_guard():
    """Reset the leverage guard singleton (for testing)."""
    global _leverage_guard
    _leverage_guard = None


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Leverage Guard Self-Test")
    print("=" * 60)
    
    guard = LeverageGuard(LeverageConfig(max_leverage=3.0))
    
    # Test 1: Normal order (within limits)
    print("\n[Test 1] Normal order within limits")
    approved, reason = guard.validate_order(
        position_notional_usd=20000.0,
        account_equity=10000.0,  # 2x leverage
        asset="SOL"
    )
    print(f"  Approved: {approved}, Reason: {reason}")
    assert approved, "Should be approved"
    
    # Test 2: Excessive leverage
    print("\n[Test 2] Excessive leverage (5x)")
    approved, reason = guard.validate_order(
        position_notional_usd=50000.0,
        account_equity=10000.0,  # 5x leverage
        asset="SOL"
    )
    print(f"  Approved: {approved}, Reason: {reason}")
    assert not approved, "Should be rejected"
    assert "exceeds" in reason.lower()
    
    # Test 3: Position size limit
    print("\n[Test 3] Position size limit")
    approved, reason = guard.validate_order(
        position_notional_usd=100000.0,  # Above $50k limit
        account_equity=100000.0,  # 1x leverage (fine)
        asset="BTC"
    )
    print(f"  Approved: {approved}, Reason: {reason}")
    assert not approved, "Should be rejected due to size"
    
    # Test 4: Liquidation buffer
    print("\n[Test 4] Liquidation buffer check")
    approved, reason = guard.validate_order(
        position_notional_usd=27000.0,
        account_equity=10000.0,  # 2.7x leverage
        current_price=185.0,
        entry_price=180.0,
        is_short=True,
        asset="SOL"
    )
    print(f"  Approved: {approved}, Reason: {reason}")
    
    # Test 5: Martingale block
    print("\n[Test 5] Anti-martingale")
    approved, reason = guard.validate_order(
        position_notional_usd=15000.0,
        account_equity=10000.0,
        existing_position_usd=10000.0,
        existing_pnl=-500.0,  # Losing position
        asset="ETH"
    )
    print(f"  Approved: {approved}, Reason: {reason}")
    assert not approved, "Should be rejected due to martingale"
    
    # Test 6: Cross margin rejection
    print("\n[Test 6] Cross margin rejection")
    try:
        LeverageConfig(margin_mode=MarginMode.CROSS)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Correctly rejected cross margin: {e}")
    
    # Print stats
    print("\n" + "=" * 60)
    print("Guard Statistics:")
    for k, v in guard.get_stats().items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

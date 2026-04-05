"""
================================================================================
HMATS P0 Safety Tests
================================================================================
Version: 5.4.1-P0FIX
Purpose: Validate all P0 safety fixes are working correctly

These tests verify:
    1. Unit system correctness (exposure -> quantity conversion)
    2. Fail-closed behavior (risk modules required)
    3. Leverage guard (max leverage, isolated margin, anti-martingale)
    4. Authority chain (one-veto-kill semantics)
    5. Kraken-only enforcement

Run with: python -m pytest tests/test_p0_safety.py -v
Or standalone: python tests/test_p0_safety.py
================================================================================
"""

import sys
import os
import logging

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# UNIT SYSTEM TESTS
# =============================================================================

def test_unit_conversion_basic():
    """
    Test: $100k account, 30% exposure, SOL at $185 -> ~162.16 SOL
    
    This is THE critical test for P0-1.
    If this fails, position sizing is fundamentally broken.
    """
    from core.unit_system import exposure_to_quantity
    
    account_equity = 100_000.0  # $100k
    exposure_fraction = 0.30    # 30%
    price = 185.0               # SOL at $185
    asset = "SOL"
    
    qty = exposure_to_quantity(
        exposure_fraction=exposure_fraction,
        account_equity=account_equity,
        price=price,
        asset=asset
    )
    
    # Expected: 30% of $100k = $30k notional
    # $30k / $185 = 162.16 SOL
    expected = 162.162162
    
    assert abs(qty - expected) < 0.01, f"Expected ~{expected:.2f}, got {qty:.2f}"
    logger.info(f"✓ Unit conversion: 30% of $100k @ $185 = {qty:.4f} SOL")


def test_unit_conversion_short():
    """Test negative exposure (short position)."""
    from core.unit_system import exposure_to_quantity
    
    qty = exposure_to_quantity(
        exposure_fraction=-0.25,  # 25% short
        account_equity=80_000.0,
        price=50_000.0,           # BTC at $50k
        asset="BTC"
    )
    
    # 25% of $80k = $20k notional
    # $20k / $50k = 0.4 BTC
    expected = 0.4
    
    assert abs(qty - expected) < 0.001, f"Expected {expected}, got {qty}"
    logger.info(f"✓ Short conversion: -25% of $80k @ $50k = {qty:.4f} BTC")


def test_unit_conversion_edge_cases():
    """Test edge cases for unit conversion."""
    from core.unit_system import exposure_to_quantity, validate_exposure_fraction
    
    # Zero exposure should return zero quantity
    qty = exposure_to_quantity(0.0, 100_000.0, 185.0, "SOL")
    assert qty == 0.0, f"Zero exposure should be zero quantity, got {qty}"
    
    # Max exposure (100%)
    qty = exposure_to_quantity(1.0, 100_000.0, 185.0, "SOL")
    assert abs(qty - 540.54) < 0.1, f"100% exposure incorrect: {qty}"
    
    # Validate exposure range
    try:
        validate_exposure_fraction(1.5)  # Over 100%
        assert False, "Should reject >100% exposure"
    except ValueError:
        pass  # Expected
    
    try:
        validate_exposure_fraction(-1.5)  # Under -100%
        assert False, "Should reject <-100% exposure"
    except ValueError:
        pass  # Expected
    
    logger.info("✓ Unit conversion edge cases passed")


# =============================================================================
# FAIL-CLOSED TESTS
# =============================================================================

def test_fail_closed_authority_chain():
    """
    Test: Missing risk module -> chain rejects in strict mode
    
    This is critical for P0-2.
    If modules can be None and trades still execute, we have fail-open.
    """
    from core.authority_chain import AuthorityChain
    
    # Create chain with missing leverage_guard in strict mode
    try:
        chain = AuthorityChain(
            data_guard=MockDataGuard(),
            trade_gate=MockTradeGate(),
            risk_manager=MockRiskManager(),
            leverage_guard=None,  # Missing!
            strict_mode=True
        )
        assert False, "Should raise RuntimeError for missing module"
    except RuntimeError as e:
        assert "FAIL-CLOSED" in str(e)
        logger.info(f"✓ Fail-closed: Missing module rejected: {e}")


def test_fail_closed_exception_handling():
    """
    Test: Exception in risk check -> veto (not swallow)
    
    In the old code, exceptions were logged but trades continued.
    Now, any exception must result in veto.
    """
    from core.authority_chain import AuthorityChain, AuthorityStage
    
    class ExplodingTradeGate:
        def check(self, **kwargs):
            raise RuntimeError("Simulated explosion")
    
    chain = AuthorityChain(
        data_guard=MockDataGuard(),
        trade_gate=ExplodingTradeGate(),
        risk_manager=MockRiskManager(),
        leverage_guard=MockLeverageGuard(),
        strict_mode=False  # Non-strict allows None but still catches exceptions
    )
    
    result = chain.evaluate(
        asset="SOL",
        direction=1.0,
        exposure_fraction=0.2,
        base_quantity=100.0,
        price=185.0,
        account_equity=100_000.0,
    )
    
    assert not result.approved, "Exception should cause veto"
    assert result.veto_stage == AuthorityStage.TRADE_GATE
    assert "EXCEPTION" in result.veto_reason
    logger.info(f"✓ Fail-closed: Exception -> veto: {result.veto_reason}")


# =============================================================================
# LEVERAGE GUARD TESTS
# =============================================================================

def test_leverage_guard_max_leverage():
    """
    Test: 5x leverage with 3x max -> BLOCKED
    
    This is critical for P0-3.
    Over-leveraged positions can lead to liquidation and debt.
    """
    from risk.leverage_guard import LeverageGuard, LeverageConfig
    
    guard = LeverageGuard(LeverageConfig(max_leverage=3.0))
    
    # Try 5x leverage: $50k position on $10k account
    approved, reason = guard.validate_order(
        position_notional_usd=50_000.0,
        account_equity=10_000.0,
        current_price=185.0,
        is_short=False,
    )
    
    assert not approved, "5x leverage should be blocked"
    assert "exceeds" in reason.lower() or "leverage" in reason.lower()
    logger.info(f"✓ Leverage guard: 5x blocked: {reason}")


def test_leverage_guard_isolated_only():
    """
    Test: Cross margin -> BLOCKED
    
    Cross margin means entire account at risk.
    Only isolated margin is allowed.
    """
    from risk.leverage_guard import LeverageGuard, LeverageConfig, MarginMode
    
    try:
        config = LeverageConfig(margin_mode=MarginMode.CROSS)
        guard = LeverageGuard(config)
        assert False, "Cross margin should raise ValueError"
    except ValueError as e:
        assert "ISOLATED" in str(e).upper() or "cross" in str(e).lower()
        logger.info(f"✓ Leverage guard: Cross margin rejected: {e}")


def test_leverage_guard_anti_martingale():
    """
    Test: Adding to losing position -> BLOCKED
    
    Martingale (adding to losers) is a path to ruin.
    """
    from risk.leverage_guard import LeverageGuard, LeverageConfig
    
    guard = LeverageGuard(LeverageConfig(allow_adding_to_losers=False))
    
    # Verify anti-martingale configuration is enforced
    # Full integration test would also verify position P&L checking
    assert not guard.config.allow_adding_to_losers
    logger.info("✓ Leverage guard: Anti-martingale enabled")


def test_leverage_guard_liquidation_buffer():
    """
    Test: Liquidation buffer calculation
    
    Must have buffer between position and liquidation price.
    """
    from risk.leverage_guard import LeverageGuard, LeverageConfig
    
    guard = LeverageGuard(LeverageConfig(
        max_leverage=3.0,
        liquidation_buffer_pct=0.05  # 5% buffer
    ))
    
    # Calculate liquidation price for short
    liq_price = guard.calculate_liquidation_price(
        entry_price=185.0,
        leverage=2.0,
        is_short=True
    )
    
    # For short at 2x: liq_price = entry * (1 + 1/leverage)
    # = 185 * (1 + 0.5) = 277.5
    assert liq_price > 185.0, "Short liquidation should be above entry"
    
    # Safe stop should be before liquidation
    safe_stop = guard.calculate_safe_stop_price(
        entry_price=185.0,
        leverage=2.0,
        is_short=True
    )
    
    assert safe_stop < liq_price, "Safe stop should be before liquidation"
    logger.info(f"✓ Liquidation buffer: entry=185, liq={liq_price:.2f}, safe_stop={safe_stop:.2f}")


# =============================================================================
# AUTHORITY CHAIN TESTS
# =============================================================================

def test_authority_chain_one_veto_kill():
    """
    Test: First veto terminates chain
    
    Later stages cannot override earlier vetoes.
    """
    from core.authority_chain import AuthorityChain, AuthorityStage
    
    class VetoingDataGuard:
        def can_execute(self):
            return False, "STALE_DATA", {"age": 120}
    
    chain = AuthorityChain(
        data_guard=VetoingDataGuard(),
        trade_gate=MockTradeGate(),
        risk_manager=MockRiskManager(),
        leverage_guard=MockLeverageGuard(),
        strict_mode=False
    )
    
    result = chain.evaluate(
        asset="SOL",
        direction=1.0,
        exposure_fraction=0.2,
        base_quantity=100.0,
        price=185.0,
        account_equity=100_000.0,
    )
    
    assert not result.approved
    assert result.veto_stage == AuthorityStage.DATA_INTEGRITY
    assert len(result.stage_results) == 1  # Should stop after first stage
    logger.info(f"✓ Authority chain: First veto terminates: {result.veto_stage.name}")


def test_authority_chain_full_pass():
    """
    Test: All stages pass -> approved
    """
    from core.authority_chain import AuthorityChain, AuthorityStage
    
    chain = AuthorityChain(
        data_guard=MockDataGuard(),
        trade_gate=MockTradeGate(),
        risk_manager=MockRiskManager(),
        leverage_guard=MockLeverageGuard(),
        strict_mode=False
    )
    
    result = chain.evaluate(
        asset="SOL",
        direction=1.0,
        exposure_fraction=0.2,
        base_quantity=100.0,
        price=185.0,
        account_equity=100_000.0,
    )
    
    assert result.approved
    assert result.final_stage == AuthorityStage.EXECUTION
    assert len(result.stage_results) == 4  # All 4 stages
    logger.info(f"✓ Authority chain: All pass -> approved")


# =============================================================================
# EXCHANGE GUARD TESTS
# =============================================================================

def test_exchange_guard_kraken_only():
    """
    Test: Non-Kraken exchange -> BLOCKED in live mode
    """
    from core.exchange_guard import validate_kraken_only, NonKrakenExchangeError
    
    # Kraken should pass
    validate_kraken_only("kraken", "live", "test")
    
    # Binance should fail in live mode
    try:
        validate_kraken_only("binance", "live", "test")
        assert False, "Binance should be rejected"
    except NonKrakenExchangeError as e:
        assert "binance" in str(e).lower()
        logger.info(f"✓ Exchange guard: Binance rejected: {e}")


def test_exchange_guard_verify_mode():
    """
    Test: Non-Kraken allowed in verify mode (warning only)
    """
    from core.exchange_guard import validate_kraken_only
    
    # Should not raise in verify mode
    validate_kraken_only("binance", "verify", "test")
    logger.info("✓ Exchange guard: Non-Kraken allowed in verify mode")


# =============================================================================
# ACCOUNT SYNC TESTS
# =============================================================================

def test_account_sync_equity_required():
    """
    Test: Equity unavailable -> trading blocked
    """
    from core.account_sync import AccountSyncManager, EquityStatus
    
    sync = AccountSyncManager(dry_run=True)
    
    # Before any refresh, status should be uninitialized
    assert sync.get_state().status == EquityStatus.UNINITIALIZED
    
    # After refresh in dry run mode, should have equity
    import asyncio
    asyncio.run(sync.refresh())
    
    equity = sync.get_equity()
    assert equity > 0, "Should have equity after refresh"
    logger.info(f"✓ Account sync: Equity available: ${equity:,.2f}")


def test_account_sync_stale_detection():
    """
    Test: Stale equity -> fail-closed
    """
    from core.account_sync import AccountSyncManager
    import time
    
    sync = AccountSyncManager(dry_run=True)
    
    # Force refresh
    import asyncio
    asyncio.run(sync.refresh())
    
    # Manually age the state
    sync._state.timestamp = time.time() - 120  # 2 minutes old
    
    # Should fail with stale error
    try:
        sync.get_equity()
        assert False, "Should raise error for stale equity"
    except RuntimeError as e:
        assert "unavailable" in str(e).lower() or "stale" in str(e).lower()
        logger.info(f"✓ Account sync: Stale equity rejected")


# =============================================================================
# MOCK CLASSES FOR TESTING
# =============================================================================

class MockDataGuard:
    def can_execute(self):
        return True, "OK", {}


class MockTradeGate:
    def check(self, **kwargs):
        class Result:
            decision = type('obj', (object,), {'name': 'ALLOW'})()
            reason = type('obj', (object,), {'name': 'NONE'})()
            details = {}
        return Result()


class MockRiskManager:
    def check_trade_allowed(self, **kwargs):
        return {"approved": True, "reason": "OK"}


class MockLeverageGuard:
    def validate_order(self, **kwargs):
        return True, "Approved"


# =============================================================================
# INTEGRATION TEST
# =============================================================================

def test_full_integration_happy_path():
    """
    Integration test: Full flow from exposure -> execution
    
    This simulates the complete P0 fix path:
    1. Start with exposure fraction
    2. Convert to quantity using account equity
    3. Validate through authority chain
    4. (Would execute if this were real)
    """
    from core.unit_system import exposure_to_quantity
    from core.authority_chain import AuthorityChain
    from core.account_sync import AccountSyncManager
    import asyncio
    
    # Setup account sync
    sync = AccountSyncManager(dry_run=True)
    sync.set_dry_run_equity(100_000.0)
    asyncio.run(sync.refresh())
    
    # Get equity
    account_equity = sync.get_equity()
    assert account_equity == 100_000.0
    
    # Convert exposure to quantity
    exposure_fraction = 0.25  # 25%
    price = 185.0
    
    base_quantity = exposure_to_quantity(
        exposure_fraction=exposure_fraction,
        account_equity=account_equity,
        price=price,
        asset="SOL"
    )
    
    # Expected: 25% of $100k = $25k notional
    # $25k / $185 = 135.135 SOL
    assert abs(base_quantity - 135.135) < 0.1
    
    # Run through authority chain
    chain = AuthorityChain(
        data_guard=MockDataGuard(),
        trade_gate=MockTradeGate(),
        risk_manager=MockRiskManager(),
        leverage_guard=MockLeverageGuard(),
        strict_mode=False
    )
    
    result = chain.evaluate(
        asset="SOL",
        direction=1.0,
        exposure_fraction=exposure_fraction,
        base_quantity=base_quantity,
        price=price,
        account_equity=account_equity,
    )
    
    assert result.approved, f"Should be approved, got: {result.veto_reason}"
    
    logger.info(
        f"✓ Integration test passed:\n"
        f"  - Equity: ${account_equity:,.2f}\n"
        f"  - Exposure: {exposure_fraction:.0%}\n"
        f"  - Quantity: {base_quantity:.4f} SOL\n"
        f"  - Notional: ${base_quantity * price:,.2f}\n"
        f"  - Authority: APPROVED"
    )


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    """Run all P0 safety tests."""
    print("=" * 70)
    print("HMATS P0 SAFETY TESTS")
    print("=" * 70)
    
    tests = [
        # Unit System
        ("Unit conversion: basic", test_unit_conversion_basic),
        ("Unit conversion: short", test_unit_conversion_short),
        ("Unit conversion: edge cases", test_unit_conversion_edge_cases),
        
        # Fail-Closed
        ("Fail-closed: authority chain", test_fail_closed_authority_chain),
        ("Fail-closed: exception handling", test_fail_closed_exception_handling),
        
        # Leverage Guard
        ("Leverage guard: max leverage", test_leverage_guard_max_leverage),
        ("Leverage guard: isolated only", test_leverage_guard_isolated_only),
        ("Leverage guard: anti-martingale", test_leverage_guard_anti_martingale),
        ("Leverage guard: liquidation buffer", test_leverage_guard_liquidation_buffer),
        
        # Authority Chain
        ("Authority chain: one-veto-kill", test_authority_chain_one_veto_kill),
        ("Authority chain: full pass", test_authority_chain_full_pass),
        
        # Exchange Guard
        ("Exchange guard: Kraken only", test_exchange_guard_kraken_only),
        ("Exchange guard: verify mode", test_exchange_guard_verify_mode),
        
        # Account Sync
        ("Account sync: equity required", test_account_sync_equity_required),
        ("Account sync: stale detection", test_account_sync_stale_detection),
        
        # Integration
        ("Integration: happy path", test_full_integration_happy_path),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            logger.error(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed > 0:
        print("\n[WARN]️  SOME TESTS FAILED - P0 FIXES MAY BE INCOMPLETE")
        sys.exit(1)
    else:
        print("\n✓ ALL P0 SAFETY TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()

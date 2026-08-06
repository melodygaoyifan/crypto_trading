#!/usr/bin/env python3
"""
================================================================================
HMATS P0 Safety Tests - Standalone Version
================================================================================
Version: 5.4.1-P0FIX
Purpose: Validate P0 safety fixes in a clean environment

This test file has MINIMAL dependencies and can run standalone.
It only imports P0 modules, not the full HMATS system.

Usage:
    python tests/test_p0_safety_standalone.py

Requirements:
    - Python 3.8+
    - No external packages required (stdlib only)
================================================================================
"""

import sys
import os
import logging
import time

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# IMPORT TESTS - Verify P0 modules can be imported standalone
# =============================================================================

def test_import_unit_system():
    """Test: core.unit_system can be imported standalone."""
    from core.unit_system import exposure_to_quantity, validate_exposure_fraction
    logger.info("✓ core.unit_system imported")
    return True


def test_import_account_sync():
    """Test: core.account_sync can be imported standalone."""
    from core.account_sync import AccountSyncManager, get_account_sync
    logger.info("✓ core.account_sync imported")
    return True


def test_import_exchange_guard():
    """Test: core.exchange_guard can be imported standalone."""
    from core.exchange_guard import validate_kraken_only, ExchangeGuard
    logger.info("✓ core.exchange_guard imported")
    return True


def test_import_authority_chain():
    """Test: core.authority_chain can be imported standalone."""
    from core.authority_chain import AuthorityChain, get_authority_chain
    logger.info("✓ core.authority_chain imported")
    return True


def test_import_leverage_guard():
    """Test: risk.leverage_guard can be imported standalone."""
    from risk.leverage_guard import LeverageGuard, LeverageConfig
    logger.info("✓ risk.leverage_guard imported")
    return True


def test_import_core_package():
    """Test: core package can be imported (with lazy loading)."""
    import core
    assert hasattr(core, '__version__')
    assert core.__version__ == "5.4.1-P0FIX"
    logger.info(f"✓ core package imported (v{core.__version__})")
    return True


def test_import_risk_package():
    """Test: risk package can be imported (with lazy loading)."""
    import risk
    assert hasattr(risk, '__version__')
    assert risk.__version__ == "5.4.1-P0FIX"
    logger.info(f"✓ risk package imported (v{risk.__version__})")
    return True


# =============================================================================
# UNIT SYSTEM TESTS
# =============================================================================

def test_unit_conversion_basic():
    """Test: $100k account, 30% exposure, SOL at $185 -> ~162.16 SOL"""
    from core.unit_system import exposure_to_quantity
    
    qty = exposure_to_quantity(
        exposure_fraction=0.30,
        account_equity=100_000.0,
        price=185.0,
        asset="SOL"
    )
    
    # [P165] $30k / $185 = 162.162162, then L4-06 FLOORS to the Kraken step
    # size (SOL = 0.1) -> 162.1. The old tolerance of 0.01 was tighter than
    # the step itself, so it could only ever pass before quantize_to_step
    # existed. Assert the quantization, not a pre-quantization ideal:
    # rounding UP here would size an order the balance cannot cover.
    raw = 0.3 * 100_000.0 / 185.0
    assert abs(qty - 162.1) < 1e-9, f"expected 162.1 (floored), got {qty}"
    assert qty <= raw, "quantization must floor, never round up"
    assert raw - qty < 0.1, "must be within one step of the ideal size"
    logger.info(f"✓ Unit conversion: 30% of $100k @ $185 = {qty:.4f} SOL")
    return True


def test_unit_conversion_short():
    """Test: Negative exposure (short position)."""
    from core.unit_system import exposure_to_quantity
    
    qty = exposure_to_quantity(
        exposure_fraction=-0.25,
        account_equity=80_000.0,
        price=50_000.0,
        asset="BTC"
    )
    
    expected = 0.4
    assert abs(qty - expected) < 0.001, f"Expected {expected}, got {qty}"
    logger.info(f"✓ Short conversion: -25% of $80k @ $50k = {qty:.4f} BTC")
    return True


def test_unit_validation():
    """Test: Exposure fraction validation."""
    from core.unit_system import validate_exposure_fraction
    
    # Valid values
    validate_exposure_fraction(0.0)
    validate_exposure_fraction(0.5)
    validate_exposure_fraction(-0.5)
    validate_exposure_fraction(1.0)
    validate_exposure_fraction(-1.0)
    
    # Invalid values
    try:
        validate_exposure_fraction(1.5)
        assert False, "Should reject >100%"
    except ValueError:
        pass
    
    try:
        validate_exposure_fraction(-1.5)
        assert False, "Should reject <-100%"
    except ValueError:
        pass
    
    logger.info("✓ Exposure validation works correctly")
    return True


# =============================================================================
# LEVERAGE GUARD TESTS
# =============================================================================

def test_leverage_guard_max_leverage():
    """Test: 5x leverage with 3x max -> BLOCKED"""
    from risk.leverage_guard import LeverageGuard, LeverageConfig
    
    guard = LeverageGuard(LeverageConfig(max_leverage=3.0))
    
    approved, reason = guard.validate_order(
        position_notional_usd=50_000.0,
        account_equity=10_000.0,
        current_price=185.0,
        is_short=False,
    )
    
    assert not approved, "5x leverage should be blocked"
    logger.info(f"✓ Leverage guard: 5x blocked: {reason}")
    return True


def test_leverage_guard_isolated_only():
    """Test: Cross margin -> BLOCKED"""
    from risk.leverage_guard import LeverageGuard, LeverageConfig, MarginMode
    
    try:
        config = LeverageConfig(margin_mode=MarginMode.CROSS)
        guard = LeverageGuard(config)
        assert False, "Cross margin should raise ValueError"
    except ValueError as e:
        logger.info(f"✓ Cross margin rejected: {e}")
    return True


def test_leverage_guard_approved():
    """Test: Valid leverage -> APPROVED"""
    from risk.leverage_guard import LeverageGuard, LeverageConfig
    
    guard = LeverageGuard(LeverageConfig(max_leverage=3.0))
    
    approved, reason = guard.validate_order(
        position_notional_usd=20_000.0,
        account_equity=10_000.0,  # 2x leverage
        current_price=185.0,
        is_short=False,
    )
    
    assert approved, f"2x leverage should be approved, got: {reason}"
    logger.info("✓ Leverage guard: 2x approved")
    return True


# =============================================================================
# EXCHANGE GUARD TESTS
# =============================================================================

def test_exchange_guard_kraken_only():
    """Test: Non-Kraken exchange -> BLOCKED in live mode"""
    from core.exchange_guard import validate_kraken_only, NonKrakenExchangeError
    
    # Kraken should pass
    validate_kraken_only("kraken", "live", "test")
    
    # Binance should fail
    try:
        validate_kraken_only("binance", "live", "test")
        assert False, "Binance should be rejected"
    except NonKrakenExchangeError:
        pass
    
    logger.info("✓ Exchange guard: Kraken-only enforced")
    return True


def test_exchange_guard_verify_mode():
    """Test: Non-Kraken allowed in verify mode."""
    from core.exchange_guard import validate_kraken_only
    
    # Should not raise in verify mode
    validate_kraken_only("binance", "verify", "test")
    logger.info("✓ Exchange guard: Non-Kraken allowed in verify mode")
    return True


# =============================================================================
# ACCOUNT SYNC TESTS
# =============================================================================

def test_account_sync_dry_run():
    """Test: Dry run mode provides equity."""
    from core.account_sync import AccountSyncManager
    import asyncio
    
    sync = AccountSyncManager(dry_run=True)
    asyncio.run(sync.refresh())
    
    equity = sync.get_equity()
    assert equity > 0, "Should have equity after refresh"
    logger.info(f"✓ Account sync: Dry run equity ${equity:,.2f}")
    return True


def test_account_sync_stale_detection():
    """Test: Stale equity -> rejected."""
    from core.account_sync import AccountSyncManager, MAX_EQUITY_AGE_SECONDS
    import asyncio

    sync = AccountSyncManager(dry_run=True)
    asyncio.run(sync.refresh())

    # [P188] Age past the threshold, not exactly to it, and read the threshold
    # from the module instead of restating it.
    #
    # This was `time.time() - 120`, and MAX_EQUITY_AGE_SECONDS is 120.0. The
    # predicate is `age > MAX`, so age was 120.0 + (the microseconds between
    # this line and the get_equity() call below) and the test passed only
    # because that gap happened to survive float64 rounding at epoch magnitude
    # (~240ns of resolution near 1.7e9). It failed roughly one full-suite run
    # in two on this machine. A test that passes on the strength of how long
    # the interpreter took to reach the next statement is not testing the
    # staleness rule; it is racing it.
    #
    # The literal was also wrong twice over: it read as "2 minutes old, well
    # past a 60s limit", which is what core/account_sync.py:299 still claims
    # the limit is. The limit is 120s.
    sync._state.timestamp = time.time() - (MAX_EQUITY_AGE_SECONDS * 2)

    try:
        sync.get_equity()
        assert False, "Should raise error for stale equity"
    except RuntimeError:
        pass
    
    logger.info("✓ Account sync: Stale equity rejected")
    return True


# =============================================================================
# AUTHORITY CHAIN TESTS
# =============================================================================

def test_authority_chain_missing_module():
    """Test: Missing module in strict mode -> rejection."""
    from core.authority_chain import AuthorityChain
    
    class MockModule:
        pass
    
    try:
        chain = AuthorityChain(
            data_guard=MockModule(),
            trade_gate=MockModule(),
            risk_manager=MockModule(),
            leverage_guard=None,  # Missing!
            strict_mode=True
        )
        assert False, "Should raise RuntimeError"
    except RuntimeError as e:
        assert "FAIL-CLOSED" in str(e)
        logger.info(f"✓ Authority chain: Missing module rejected")
    return True


def test_authority_chain_full_pass():
    """Test: All stages pass -> approved."""
    from core.authority_chain import AuthorityChain, AuthorityStage
    
    class MockDataGuard:
        def can_execute(self):
            return True, "OK", {}
    
    class MockTradeGate:
        def check(self, **kwargs):
            class Result:
                decision = type('obj', (object,), {'name': 'ALLOW'})()
                reason = type('obj', (object,), {'name': 'NONE'})()
            return Result()
    
    class MockRiskManager:
        def check_trade_allowed(self, **kwargs):
            return {"approved": True}
    
    class MockLeverageGuard:
        def validate_order(self, **kwargs):
            return True, "Approved"
    
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
    
    assert result.approved, f"Should be approved, got: {result.veto_reason}"
    assert result.final_stage == AuthorityStage.EXECUTION
    logger.info("✓ Authority chain: All pass -> approved")
    return True


# =============================================================================
# MODULE STATUS TESTS
# =============================================================================

def test_core_module_status():
    """Test: Core module status shows P0 modules available."""
    import core
    status = core.get_module_status()
    
    # P0 modules must be True
    assert status["unit_system"] == True
    assert status["account_sync"] == True
    assert status["exchange_guard"] == True
    assert status["authority_chain"] == True
    
    logger.info("✓ Core module status: P0 modules available")
    return True


def test_risk_module_status():
    """Test: Risk module status shows P0 modules available."""
    import risk
    status = risk.get_module_status()
    
    # P0 modules must be True
    assert status["leverage_guard"] == True
    
    logger.info("✓ Risk module status: P0 modules available")
    return True


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    """Run all P0 safety tests."""
    print("=" * 70)
    print("HMATS P0 SAFETY TESTS - STANDALONE")
    print("=" * 70)
    
    tests = [
        # Import tests
        ("Import: core.unit_system", test_import_unit_system),
        ("Import: core.account_sync", test_import_account_sync),
        ("Import: core.exchange_guard", test_import_exchange_guard),
        ("Import: core.authority_chain", test_import_authority_chain),
        ("Import: risk.leverage_guard", test_import_leverage_guard),
        ("Import: core package", test_import_core_package),
        ("Import: risk package", test_import_risk_package),
        
        # Unit system tests
        ("Unit: basic conversion", test_unit_conversion_basic),
        ("Unit: short conversion", test_unit_conversion_short),
        ("Unit: validation", test_unit_validation),
        
        # Leverage guard tests
        ("Leverage: max leverage", test_leverage_guard_max_leverage),
        ("Leverage: isolated only", test_leverage_guard_isolated_only),
        ("Leverage: approved", test_leverage_guard_approved),
        
        # Exchange guard tests
        ("Exchange: Kraken only", test_exchange_guard_kraken_only),
        ("Exchange: verify mode", test_exchange_guard_verify_mode),
        
        # Account sync tests
        ("Account: dry run", test_account_sync_dry_run),
        ("Account: stale detection", test_account_sync_stale_detection),
        
        # Authority chain tests
        ("Authority: missing module", test_authority_chain_missing_module),
        ("Authority: full pass", test_authority_chain_full_pass),
        
        # Module status tests
        ("Status: core modules", test_core_module_status),
        ("Status: risk modules", test_risk_module_status),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                logger.error(f"✗ FAILED: returned False")
        except Exception as e:
            failed += 1
            logger.error(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed > 0:
        print("\n[WARN]️  SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\n✓ ALL P0 SAFETY TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()

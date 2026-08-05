#!/usr/bin/env python3
"""
================================================================================
HMATS v3.6.1 - PRODUCTION RELIABILITY PATCHES TEST
================================================================================
Verifies all 6 patches are working correctly.
================================================================================
"""

import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Add parent to path for imports
sys.path.insert(0, '/home/claude/hmats_v36')

from integration.core.production_reliability_patches import (
    # Patch 1
    get_schema_validator,
    SchemaValidationLevel,
    CANONICAL_SCHEMA_SPEC,
    
    # Patch 2
    EntrypointLock,
    canonical_entrypoint_patch,
    legacy_redirect_patch,
    
    # Patch 3
    TickProofLogBuilder,
    ComponentStatusPatch,
    
    # Patch 4
    get_veto_classifier,
    HardVetoConditionPatch,
    SoftVetoConditionPatch,
    
    # Patch 5
    get_deadlock_resolver_patch,
    TrancheBias,
    
    # Patch 6
    get_sol_forced_exit_patch,
    ExitUrgencyPatch,
    ExecutionLoopHook,
    
    # Apply
    apply_all_patches,
    PATCH_VERSION,
)


def test_patch1_schema_validation():
    """Test Patch 1: Canonical schema key mapping + required-key validation."""
    print("\n" + "=" * 80)
    print("TEST 1: CANONICAL SCHEMA + REQUIRED-KEY VALIDATION (Patch 1)")
    print("=" * 80)
    
    validator = get_schema_validator()
    
    # Test 1a: Complete valid data
    print("\n[Test 1a] Complete valid data in NORMAL mode:")
    valid_data = {
        "current_price": 180.50,
        "volume_24h": 1000000.0,
        "data_age_seconds": 0.5,
        "dvol_zscore": 2.5,
        "vpin": 0.4,
        "correlation_btc_eth_sol": 0.75,
        "orderbook_depth_1pct_usd": 500000.0,
    }
    canonical, proof = validator.validate(valid_data, mode="NORMAL")
    print(f"  Proof: {proof.to_proof_log()}")
    assert proof.is_valid, "Valid data should pass"
    assert not proof.triggers_no_trade, "Valid data should not trigger NO_TRADE"
    print("  ✓ PASSED: Valid data accepted")
    
    # Test 1b: Missing required key
    print("\n[Test 1b] Missing required key (current_price):")
    missing_required = {
        "volume_24h": 1000000.0,
        "data_age_seconds": 0.5,
    }
    canonical, proof = validator.validate(missing_required, mode="NORMAL")
    print(f"  Proof: {proof.to_proof_log()}")
    assert not proof.is_valid, "Missing required should fail"
    assert proof.triggers_no_trade, "Missing required should trigger NO_TRADE"
    assert "current_price" in proof.missing_required
    print("  ✓ PASSED: Missing required triggers NO_TRADE")
    
    # Test 1c: Missing critical in NORMAL mode
    print("\n[Test 1c] Missing critical key in NORMAL mode:")
    missing_critical = {
        "current_price": 180.50,
        "volume_24h": 1000000.0,
        "data_age_seconds": 0.5,
    }
    canonical, proof = validator.validate(missing_critical, mode="NORMAL")
    print(f"  Proof: {proof.to_proof_log()}")
    assert not proof.is_valid, "Missing critical in NORMAL should fail"
    assert proof.triggers_no_trade, "Missing critical in NORMAL should trigger NO_TRADE"
    print("  ✓ PASSED: Missing critical in NORMAL triggers NO_TRADE")
    
    # Test 1d: Missing critical in OPPORTUNITY mode (should NOT trigger NO_TRADE)
    print("\n[Test 1d] Missing critical key in OPPORTUNITY mode:")
    canonical, proof = validator.validate(missing_critical, mode="OPPORTUNITY")
    print(f"  Proof: {proof.to_proof_log()}")
    assert proof.is_valid, "Missing critical in OPPORTUNITY should pass with defaults"
    assert not proof.triggers_no_trade, "Missing critical in OPPORTUNITY should NOT trigger NO_TRADE"
    assert len(proof.defaults_applied) > 0, "Defaults should be applied"
    print("  ✓ PASSED: Missing critical in OPPORTUNITY uses defaults, no NO_TRADE")
    
    # Test 1e: Alias resolution
    print("\n[Test 1e] Alias resolution (dvol -> dvol_zscore):")
    aliased_data = {
        "current_price": 180.50,
        "volume_24h": 1000000.0,
        "age_seconds": 0.5,  # Alias for data_age_seconds
        "dvol": 2.5,  # Alias for dvol_zscore
    }
    canonical, proof = validator.validate(aliased_data, mode="OPPORTUNITY")
    print(f"  Proof: {proof.to_proof_log()}")
    assert "data_age_seconds" in canonical, "age_seconds should resolve to data_age_seconds"
    assert "dvol_zscore" in canonical, "dvol should resolve to dvol_zscore"
    assert "age_seconds" in proof.aliases_resolved
    print("  ✓ PASSED: Aliases resolved correctly")
    
    return True


def test_patch2_canonical_entrypoint():
    """Test Patch 2: Single canonical entrypoint + version banner."""
    print("\n" + "=" * 80)
    print("TEST 2: CANONICAL ENTRYPOINT + VERSION BANNER (Patch 2)")
    print("=" * 80)
    
    # Reset for clean test
    EntrypointLock.reset()
    
    # Test 2a: Lock canonical entrypoint
    print("\n[Test 2a] Lock canonical entrypoint:")
    
    @canonical_entrypoint_patch("3.6.1-TEST")
    def my_canonical_decide(data):
        EntrypointLock.record_call("DECIDE", "processing")
        return {"result": "success", "data": data}
    
    # Call it
    result = my_canonical_decide({"test": True})
    print(f"  Result: {result}")
    assert result["result"] == "success"
    assert EntrypointLock._canonical == "my_canonical_decide"
    print("  ✓ PASSED: Canonical entrypoint locked")
    
    # Test 2b: Call chain tracking
    print("\n[Test 2b] Call chain tracking:")
    chain_proof = EntrypointLock.get_call_chain_proof()
    print(f"  Call chain: {chain_proof}")
    assert "ENTRYPOINT" in chain_proof
    assert "DECIDE" in chain_proof
    print("  ✓ PASSED: Call chain tracked")
    
    # Test 2c: Attempt to re-lock should fail
    print("\n[Test 2c] Re-lock attempt (should fail):")
    try:
        EntrypointLock.set_canonical("different_name")
        print("  ✗ FAILED: Should have raised error")
        return False
    except RuntimeError as e:
        print(f"  Caught expected error: {e}")
        print("  ✓ PASSED: Re-lock prevented")
    
    return True


def test_patch3_structured_proof_log():
    """Test Patch 3: Structured [PROOF] log per 4H tick."""
    print("\n" + "=" * 80)
    print("TEST 3: STRUCTURED [PROOF] LOG (Patch 3)")
    print("=" * 80)
    
    # Build a complete proof log
    print("\n[Test 3a] Build structured proof log:")
    builder = TickProofLogBuilder()
    
    proof = (
        builder
        .set_components(
            data=ComponentStatusPatch.REAL,
            quant=ComponentStatusPatch.REAL,
            drl=ComponentStatusPatch.DISABLED,
            dvol=ComponentStatusPatch.REAL,
            sentiment=ComponentStatusPatch.DISABLED,
            macro=ComponentStatusPatch.DISABLED,
        )
        .set_decision(mode="OPPORTUNITY", direction=0.72, exposure=0.25)
        .set_execution(mode="PASSIVE_PREFERRED", tranche_action="ENTER_T1", tranche_level=1)
        .set_veto(veto_type="NONE", reason="", cap=1.0)
        .set_alpha(passed=True, est_bps=85.0, thresh_bps=50.0)
        .set_deadlock(bars=0, resolution="NONE", bias="NEUTRAL")
        .set_sol_exit(forced=False, urgency="NONE")
        .set_schema(valid=True, missing_count=0)
        .build()
    )
    
    log_line = proof.to_log_line()
    print(f"  {log_line}")
    
    # Verify components
    assert "Data=REAL" in log_line
    assert "Quant=REAL" in log_line
    assert "DRL=DISABLED" in log_line
    assert "Mode=OPPORTUNITY" in log_line
    assert "dir=+0.72" in log_line
    assert "Tranche={action=ENTER_T1,level=1" in log_line
    print("  ✓ PASSED: All components in proof log")
    
    return True


def test_patch4_hard_soft_veto():
    """Test Patch 4: HARD vs SOFT risk veto split."""
    print("\n" + "=" * 80)
    print("TEST 4: HARD vs SOFT RISK VETO (Patch 4)")
    print("=" * 80)
    
    classifier = get_veto_classifier()
    
    # Test 4a: HARD veto (drawdown)
    print("\n[Test 4a] HARD veto (drawdown >= 10%):")
    result = classifier.classify(
        mode="OPPORTUNITY",
        drawdown=0.12,  # 12% - triggers HARD
    )
    print(f"  Result: {result.to_proof_str()}")
    assert result.veto_type == "HARD"
    assert not result.allows_trade
    assert HardVetoConditionPatch.DRAWDOWN_HALT in result.hard_conditions
    print("  ✓ PASSED: HARD veto blocks even in OPPORTUNITY")
    
    # Test 4b: SOFT veto in OPPORTUNITY (caps but doesn't block)
    print("\n[Test 4b] SOFT veto in OPPORTUNITY mode (caps but allows):")
    result = classifier.classify(
        mode="OPPORTUNITY",
        dvol_zscore=3.5,  # Elevated but not extreme
    )
    print(f"  Result: {result.to_proof_str()}")
    assert result.veto_type == "SOFT"
    assert result.allows_trade, "SOFT in OPPORTUNITY should ALLOW"
    assert result.exposure_cap < 1.0, "SOFT should cap exposure"
    print(f"  ✓ PASSED: SOFT in OPPORTUNITY caps to {result.exposure_cap:.0%} but allows trade")
    
    # Test 4c: SOFT veto in NORMAL mode (blocks)
    print("\n[Test 4c] SOFT veto in NORMAL mode (blocks):")
    result = classifier.classify(
        mode="NORMAL",
        dvol_zscore=3.5,  # Same condition
    )
    print(f"  Result: {result.to_proof_str()}")
    assert result.veto_type == "SOFT"
    assert not result.allows_trade, "SOFT in NORMAL should BLOCK"
    print("  ✓ PASSED: SOFT in NORMAL blocks")
    
    # Test 4d: No veto
    print("\n[Test 4d] No veto (clean conditions):")
    result = classifier.classify(
        mode="NORMAL",
        drawdown=0.02,
        dvol_zscore=1.5,
        correlation=0.5,
    )
    print(f"  Result: {result.to_proof_str()}")
    assert result.veto_type == "NONE"
    assert result.allows_trade
    print("  ✓ PASSED: No veto when conditions clean")
    
    return True


def test_patch5_tranche_deadlock():
    """Test Patch 5: Execution deadlock + tranche coupling."""
    print("\n" + "=" * 80)
    print("TEST 5: TRANCHE-COUPLED DEADLOCK RESOLUTION (Patch 5)")
    print("=" * 80)
    
    resolver = get_deadlock_resolver_patch()
    resolver.reset("SOL")
    
    # Test 5a: T1 deadlock - bias toward FORCE
    print("\n[Test 5a] T1 deadlock (bias toward FORCE):")
    
    # Simulate 2 bars of being stuck at T1
    for i in range(2):
        result = resolver.resolve(
            asset="SOL",
            tranche_level=1,  # T1
            current_edge_bps=50.0,
            friction_bps=15.0,
            is_stuck=True,
        )
        print(f"  Bar {i+1}: resolution={result.resolution}, bias={result.bias.value}")
    
    assert result.resolution == "FORCE_AGGRESSIVE", "T1 should FORCE after 2 bars"
    assert result.bias == TrancheBias.FORCE
    print("  ✓ PASSED: T1 resolves to FORCE_AGGRESSIVE")
    
    # Test 5b: T2+ deadlock - bias toward ABORT
    print("\n[Test 5b] T2+ deadlock (bias toward ABORT):")
    resolver.reset("ETH")
    
    # Simulate 3 bars of being stuck at T2
    for i in range(3):
        result = resolver.resolve(
            asset="ETH",
            tranche_level=2,  # T2
            current_edge_bps=50.0,
            friction_bps=15.0,
            is_stuck=True,
        )
        print(f"  Bar {i+1}: resolution={result.resolution}, bias={result.bias.value}")
    
    assert result.resolution == "ABORT_OPPORTUNITY", "T2+ should ABORT after 3 bars"
    assert result.bias == TrancheBias.ABORT
    print("  ✓ PASSED: T2+ resolves to ABORT_OPPORTUNITY")
    
    # Test 5c: T1 with degraded edge - should abort
    print("\n[Test 5c] T1 with degraded edge (should abort despite bias):")
    resolver.reset("BTC")
    
    for i in range(2):
        result = resolver.resolve(
            asset="BTC",
            tranche_level=1,
            current_edge_bps=10.0,  # Low edge
            friction_bps=15.0,
            is_stuck=True,
        )
    
    print(f"  Result: {result.resolution}, edge_decay={result.edge_decay_pct:.0f}%")
    assert result.resolution == "ABORT_OPPORTUNITY", "T1 should ABORT when edge degraded"
    print("  ✓ PASSED: T1 aborts when edge insufficient")
    
    return True


def test_patch6_sol_forced_exit():
    """Test Patch 6: SOL dominance forced exit via execution loop."""
    print("\n" + "=" * 80)
    print("TEST 6: SOL FORCED EXIT VIA EXECUTION LOOP (Patch 6)")
    print("=" * 80)
    
    exit_handler = get_sol_forced_exit_patch()
    exit_handler.reset("SOL")
    
    # Test 6a: IMMEDIATE exit on VPIN spike
    print("\n[Test 6a] IMMEDIATE exit on VPIN spike:")
    signal = exit_handler.check_forced_exit(
        asset="SOL",
        dominance_active=True,
        dominance_ttl=2,
        current_phase="EXPANSION",
        vpin=0.92,  # >= 0.90 threshold
        correlation=0.75,
        price_move_pct=0.01,
        current_exposure=0.50,
    )
    print(f"  Signal: urgency={signal.urgency.value}, reason={signal.reason}")
    assert signal.should_exit
    assert signal.urgency == ExitUrgencyPatch.IMMEDIATE
    assert signal.target_exposure == 0.0
    print("  ✓ PASSED: VPIN spike triggers IMMEDIATE exit")
    
    # Check execution loop has signal
    exec_signals = exit_handler.get_execution_loop_signals()
    print(f"  Execution loop signals: {len(exec_signals)}")
    assert len(exec_signals) > 0, "Should have execution loop signal"
    assert exec_signals[0]["urgency"] == "IMMEDIATE"
    print("  ✓ PASSED: Signal added to execution loop")
    
    # Test 6b: IMMEDIATE exit on phase transition
    print("\n[Test 6b] IMMEDIATE exit on phase transition to CRASH:")
    exit_handler.reset("SOL")
    exit_handler._last_phase["SOL"] = "EXPANSION"  # Set previous phase
    
    signal = exit_handler.check_forced_exit(
        asset="SOL",
        dominance_active=True,
        dominance_ttl=2,
        current_phase="CRASH",  # Danger phase
        vpin=0.5,
        correlation=0.75,
        price_move_pct=0.01,
        current_exposure=0.50,
    )
    print(f"  Signal: urgency={signal.urgency.value}, reason={signal.reason}")
    assert signal.should_exit
    assert signal.urgency == ExitUrgencyPatch.IMMEDIATE
    print("  ✓ PASSED: Phase transition triggers IMMEDIATE exit")
    
    # Test 6c: SCHEDULED exit on TTL approaching
    print("\n[Test 6c] SCHEDULED exit on TTL approaching:")
    exit_handler.reset("ETH")
    
    signal = exit_handler.check_forced_exit(
        asset="ETH",
        dominance_active=True,
        dominance_ttl=1,  # 1 bar remaining
        current_phase="EXPANSION",
        vpin=0.5,
        correlation=0.75,
        price_move_pct=0.01,
        current_exposure=0.50,
    )
    print(f"  Signal: urgency={signal.urgency.value}, reason={signal.reason}")
    assert signal.should_exit
    assert signal.urgency == ExitUrgencyPatch.SCHEDULED
    assert signal.target_exposure == 0.25, "Should reduce by 50%"
    print("  ✓ PASSED: TTL=1 triggers SCHEDULED exit (reduce 50%)")
    
    # Test 6d: No exit when no position
    print("\n[Test 6d] No exit when no position:")
    signal = exit_handler.check_forced_exit(
        asset="BTC",
        dominance_active=True,
        dominance_ttl=0,
        current_phase="CRASH",
        vpin=0.95,
        correlation=0.95,
        price_move_pct=0.10,
        current_exposure=0.0,  # No position
    )
    print(f"  Signal: should_exit={signal.should_exit}")
    assert not signal.should_exit, "No exit when no position"
    print("  ✓ PASSED: No exit when exposure=0")
    
    return True


def run_all_tests():
    """Run all patch tests."""
    print("\n" + "=" * 80)
    print("HMATS v3.6.1 PRODUCTION RELIABILITY PATCHES TEST SUITE")
    print(f"Patch Version: {PATCH_VERSION}")
    print("=" * 80)
    
    # Apply patches first
    print("\nApplying patches...")
    result = apply_all_patches()
    print(f"Applied: {result['version']}")
    for patch in result['patches']:
        print(f"  - {patch}")
    
    # Run tests
    tests = [
        ("Patch 1: Schema Validation", test_patch1_schema_validation),
        ("Patch 2: Canonical Entrypoint", test_patch2_canonical_entrypoint),
        ("Patch 3: Structured Proof Log", test_patch3_structured_proof_log),
        ("Patch 4: HARD vs SOFT Veto", test_patch4_hard_soft_veto),
        ("Patch 5: Tranche Deadlock", test_patch5_tranche_deadlock),
        ("Patch 6: SOL Forced Exit", test_patch6_sol_forced_exit),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, passed, error in results:
        status = "✓ PASSED" if passed else f"✗ FAILED: {error}"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("Production reliability patches verified.")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

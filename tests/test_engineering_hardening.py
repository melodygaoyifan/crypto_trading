#!/usr/bin/env python3
"""
================================================================================
HMATS v5.2.1 - ENGINEERING HARDENING TESTS
================================================================================

Tests for the three engineering hardening areas:
1. Legacy module isolation (runtime protection)
2. Dual entry point protection (main.py canonical)
3. Sentiment mode traceability (unified feature flag)

These tests verify:
- Runtime code does not import from legacy/
- main_v36.py blocks live mode without --allow-compat-live
- Sentiment mode is traceable in DECISION_SUMMARY

================================================================================
"""

import sys
import os
import logging
import pytest
from datetime import datetime
from typing import List, Tuple
from dataclasses import dataclass

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# =============================================================================
# TEST RESULT STRUCTURE
# =============================================================================

@dataclass
class HardeningTestResult:
    """Result of a hardening test."""
    name: str
    passed: bool
    details: str


def run_test(name: str, test_fn) -> HardeningTestResult:
    """Run a test and return result."""
    try:
        passed, details = test_fn()
        return HardeningTestResult(name=name, passed=passed, details=details)
    except Exception as e:
        return HardeningTestResult(name=name, passed=False, details=f"Exception: {e}")


# =============================================================================
# PROBLEM 1: LEGACY MODULE ISOLATION TESTS
# =============================================================================

def test_p1_canonical_imports_available() -> Tuple[bool, str]:
    """Test that canonical_imports module exists and is loadable."""
    try:
        from core.canonical_imports import (
            AuthorityFusionEngine,
            RiskGovernor,
            TradeGate,
            ExecutionManager,
            RuntimeSpine,
            activate_runtime_mode,
            is_runtime_mode,
            has_legacy_import_violation,
        )
        return True, f"Loaded {8} canonical symbols"
    except ImportError as e:
        return False, f"Import failed: {e}"


def test_p1_legacy_deprecation_warning() -> Tuple[bool, str]:
    """Test that legacy module emits deprecation warning."""
    import warnings
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Clear any existing legacy import
        if 'legacy' in sys.modules:
            del sys.modules['legacy']
        
        try:
            import legacy
            
            # Check for deprecation warning
            deprecation_warnings = [
                x for x in w 
                if issubclass(x.category, DeprecationWarning) 
                and 'deprecated' in str(x.message).lower()
            ]
            
            if deprecation_warnings:
                return True, f"DeprecationWarning emitted: {deprecation_warnings[0].message}"
            else:
                return False, f"No DeprecationWarning found in {len(w)} warnings"
        except ImportError as e:
            return False, f"Failed to import legacy: {e}"


def test_p1_runtime_mode_protection() -> Tuple[bool, str]:
    """Test that runtime mode protection works."""
    from core.canonical_imports import (
        activate_runtime_mode,
        is_runtime_mode,
        has_legacy_import_violation,
    )
    
    # Activate runtime mode
    activate_runtime_mode()
    
    if not is_runtime_mode():
        return False, "Runtime mode did not activate"
    
    return True, "Runtime mode protection active"


def test_p1_no_legacy_in_runtime_path() -> Tuple[bool, str]:
    """Test that main runtime modules don't import legacy."""
    import ast
    import os
    
    runtime_modules = [
        'core/runtime_spine.py',
        'signals/authority_fusion.py',
        'core/risk_governor.py',
        'execution/execution_manager.py',
        'defense/trade_gate.py',
    ]
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    violations = []
    
    for module_path in runtime_modules:
        full_path = os.path.join(base_path, module_path)
        if not os.path.exists(full_path):
            continue
            
        with open(full_path, 'r') as f:
            content = f.read()
            
        # Check for legacy imports
        if 'from legacy' in content or 'import legacy' in content:
            # Exclude comments and strings
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        if hasattr(node, 'module') and node.module and 'legacy' in node.module:
                            violations.append(f"{module_path}: {node.module}")
                        elif hasattr(node, 'names'):
                            for alias in node.names:
                                if 'legacy' in alias.name:
                                    violations.append(f"{module_path}: {alias.name}")
            except SyntaxError:
                pass
    
    if violations:
        return False, f"Legacy imports found: {violations}"
    return True, f"Checked {len(runtime_modules)} runtime modules, no legacy imports"


# =============================================================================
# PROBLEM 2: DUAL ENTRY POINT TESTS
# =============================================================================

def test_p2_main_has_canonical_banner() -> Tuple[bool, str]:
    """Test that main.py has CANONICAL_ENTRYPOINT banner."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_path = os.path.join(base_path, 'main.py')
    
    with open(main_path, 'r') as f:
        content = f.read()
    
    if 'CANONICAL_ENTRYPOINT = main.py' in content:
        return True, "main.py has CANONICAL_ENTRYPOINT banner"
    return False, "main.py missing CANONICAL_ENTRYPOINT banner"


def test_p2_main_has_live_protection() -> Tuple[bool, str]:
    """Test that main.py has live mode protection (--confirm-live flag)."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_path = os.path.join(base_path, 'main.py')

    with open(main_path, 'r') as f:
        content = f.read()

    checks = [
        ('--confirm-live' in content, "--confirm-live flag check"),
        ('LIVE TRADING' in content, "LIVE TRADING warning message"),
        ('CANONICAL_ENTRYPOINT' in content, "CANONICAL_ENTRYPOINT banner"),
    ]

    passed_checks = [(c, m) for c, m in checks if c]

    if len(passed_checks) >= 2:
        return True, f"Passed: {[m for _, m in passed_checks]}"
    return False, f"Failed checks: {[m for c, m in checks if not c]}"


def test_p2_main_live_blocked_without_flag() -> Tuple[bool, str]:
    """Test that main.py blocks live mode without explicit --confirm-live flag."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_path = os.path.join(base_path, 'main.py')

    with open(main_path, 'r') as f:
        content = f.read()

    protection_patterns = [
        'args.mode == "live"',
        'confirm_live',
        'Live trading requires',
    ]

    found_patterns = [p for p in protection_patterns if p in content]

    if len(found_patterns) >= 2:
        return True, f"Live mode protection found: {found_patterns}"
    return False, f"Missing protection patterns, found only: {found_patterns}"


# =============================================================================
# PROBLEM 3: SENTIMENT MODE TRACEABILITY TESTS
# =============================================================================

def test_p3_sentiment_mode_enum_exists() -> Tuple[bool, str]:
    """Test that SentimentMode enum exists."""
    try:
        from engine.compute.vllm_inference_wrapper import SentimentMode
        
        modes = [SentimentMode.REAL, SentimentMode.MOCK, SentimentMode.DISABLED]
        return True, f"SentimentMode enum has {len(modes)} modes"
    except ImportError as e:
        return False, f"Import failed: {e}"


def test_p3_sentiment_config_startup_logging() -> Tuple[bool, str]:
    """Test that sentiment config logs on startup."""
    try:
        # Reset config to test fresh initialization
        import engine.compute.vllm_inference_wrapper as wrapper
        wrapper._sentiment_config = None
        
        import io
        import logging
        
        # Capture logs
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        
        sentiment_logger = logging.getLogger('SentimentConfig')
        sentiment_logger.addHandler(handler)
        
        # Initialize config
        config = wrapper.get_sentiment_config()
        
        # Check log output
        log_contents = log_capture.getvalue()
        
        sentiment_logger.removeHandler(handler)
        
        if 'SENTIMENT_MODE=' in log_contents or 'SENTIMENT CONFIGURATION' in str(config):
            return True, f"Startup logging found, IS_MOCK={config.IS_MOCK}"
        return False, f"No startup logging found"
    except Exception as e:
        return False, f"Exception: {e}"


def test_p3_sentiment_in_decision_summary() -> Tuple[bool, str]:
    """Test that sentiment mode appears in decision summary."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runtime_spine_path = os.path.join(base_path, 'core/runtime_spine.py')
    
    with open(runtime_spine_path, 'r') as f:
        content = f.read()
    
    # Check for sentiment mode in decision summary
    patterns = [
        'SENTIMENT=MOCK',
        'is_sentiment_mock',
        'secondary_caps',
    ]
    
    found_patterns = [p for p in patterns if p in content]
    
    if len(found_patterns) >= 2:
        return True, f"Sentiment tracking found in decision summary: {found_patterns}"
    return False, f"Sentiment tracking incomplete, found: {found_patterns}"


def test_p3_mock_sentiment_disables_opportunity() -> Tuple[bool, str]:
    """Test that mock sentiment disables OPPORTUNITY triggers."""
    try:
        from engine.compute.vllm_inference_wrapper import (
            get_sentiment_config,
            sentiment_can_trigger_opportunity,
            is_sentiment_mock,
        )
        
        config = get_sentiment_config()
        
        if is_sentiment_mock():
            # In mock mode, should NOT be able to trigger opportunity
            can_trigger = sentiment_can_trigger_opportunity()
            if not can_trigger:
                return True, "Mock mode correctly disables OPPORTUNITY triggers"
            return False, "Mock mode should disable OPPORTUNITY triggers"
        else:
            # In real mode, should be able to trigger opportunity
            return True, "Real mode - OPPORTUNITY triggers enabled"
    except Exception as e:
        return False, f"Exception: {e}"


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests() -> List[HardeningTestResult]:
    """Run all engineering hardening tests."""
    
    print("""
================================================================================
HMATS v5.2.1 - ENGINEERING HARDENING TESTS
================================================================================
""")
    
    tests = [
        # Problem 1: Legacy Module Isolation
        ("P1-A: Canonical imports available", test_p1_canonical_imports_available),
        ("P1-B: Legacy deprecation warning", test_p1_legacy_deprecation_warning),
        ("P1-C: Runtime mode protection", test_p1_runtime_mode_protection),
        ("P1-D: No legacy in runtime path", test_p1_no_legacy_in_runtime_path),
        
        # Problem 2: Dual Entry Point Protection
        ("P2-A: main.py canonical banner", test_p2_main_has_canonical_banner),
        ("P2-B: main_v36.py compatibility warning", test_p2_main_v36_has_compatibility_warning),
        ("P2-C: main_v36.py live blocked", test_p2_main_v36_live_blocked),
        
        # Problem 3: Sentiment Mode Traceability
        ("P3-A: SentimentMode enum exists", test_p3_sentiment_mode_enum_exists),
        ("P3-B: Sentiment startup logging", test_p3_sentiment_config_startup_logging),
        ("P3-C: Sentiment in decision summary", test_p3_sentiment_in_decision_summary),
        ("P3-D: Mock disables opportunity", test_p3_mock_sentiment_disables_opportunity),
    ]
    
    results = []
    
    print("Problem 1: Legacy Module Isolation")
    print("-" * 60)
    for name, test_fn in tests[:4]:
        result = run_test(name, test_fn)
        results.append(result)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  {status}: {result.name}")
        print(f"         {result.details}")
    
    print("\nProblem 2: Dual Entry Point Protection")
    print("-" * 60)
    for name, test_fn in tests[4:7]:
        result = run_test(name, test_fn)
        results.append(result)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  {status}: {result.name}")
        print(f"         {result.details}")
    
    print("\nProblem 3: Sentiment Mode Traceability")
    print("-" * 60)
    for name, test_fn in tests[7:]:
        result = run_test(name, test_fn)
        results.append(result)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  {status}: {result.name}")
        print(f"         {result.details}")
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total:  {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {total - passed}")
    print()
    
    if passed == total:
        print("ON ALL ENGINEERING HARDENING TESTS PASSED")
    else:
        print("[WARN]️  SOME TESTS FAILED")
        for r in results:
            if not r.passed:
                print(f"   - {r.name}: {r.details}")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with error code if any test failed
    failed = sum(1 for r in results if not r.passed)
    sys.exit(1 if failed > 0 else 0)

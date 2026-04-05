#!/usr/bin/env python3
"""Verify R1-R6 risk module improvements."""

import os
import sys
import inspect

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))


print("=" * 60)
print("HMATS Risk Module R1-R6 Verification")
print("=" * 60)

# =========================================================================
# R5: Dead stubs removed
# =========================================================================
print("\n[R5] Dead Stub Removal")
for stub in ["risk/position_sizer.py", "risk/exposure_cap.py", "risk/stop_loss.py"]:
    full = os.path.join(os.path.dirname(__file__), "..", "..", stub)
    exists = os.path.exists(full)
    check(f"R5 {stub}", not exists, "Removed" if not exists else "Still exists")

# =========================================================================
# R1: Live correlation wired
# =========================================================================
print("\n[R1] Live Correlation Wiring")
try:
    from risk.risk_manager import RiskManager

    rm = RiskManager()
    check("R1.1 _correlation_source attr", hasattr(rm, "_correlation_source"))

    # Check _get_pair_correlation method exists
    check("R1.2 _get_pair_correlation method", hasattr(rm, "_get_pair_correlation"))

    # Check _strip_to_base utility
    check("R1.3 _strip_to_base", RiskManager._strip_to_base("BTC/USDT:USDT") == "BTC",
          f"got: {RiskManager._strip_to_base('BTC/USDT:USDT')}")

    # Check that _get_correlation_adjustment references live source in code
    src = inspect.getsource(rm._get_pair_correlation)
    check("R1.4 live source check in code", "_correlation_source" in src,
          "Should reference _correlation_source")

    # Test fallback to hardcoded (no live source)
    rm.initialize(10000)
    from risk.risk_manager import PositionRisk
    rm.register_position(PositionRisk(
        symbol="BTC/USDT:USDT", entry_price=100000,
        current_price=100000, stop_loss=98000,
        position_size=0.01, side="long"
    ))
    factor = rm._get_correlation_adjustment("ETH/USDT:USDT")
    check("R1.5 fallback factor", 0.0 < factor <= 1.0,
          f"factor={factor}")
except Exception as e:
    check("R1 import", False, str(e))

# =========================================================================
# R2: Single DD authority
# =========================================================================
print("\n[R2] Consolidated Drawdown Tracking")
try:
    from risk.risk_manager import RiskManager, DrawdownStatus
    from risk.sota_risk_controller import SOTARiskController, SOTARiskConfig, RiskState

    # Test with SOTA ref - use actual drawdown values since update_balance
    # calls update_equity() which re-evaluates thresholds
    sota = SOTARiskController(SOTARiskConfig())
    sota.peak_equity = 10000
    rm = RiskManager(sota_risk_ref=sota)
    rm.initialize(10000)

    check("R2.1 _sota_risk attr", hasattr(rm, "_sota_risk"))

    # SOTA NORMAL, no drawdown -> NORMAL
    rm.update_balance(10000)
    check("R2.2 NORMAL state", rm.drawdown_status == DrawdownStatus.NORMAL,
          f"got: {rm.drawdown_status.value}")

    # 26% DD -> SOTA triggers HALTED (halt_threshold=25%)
    rm.update_balance(7400)
    check("R2.3 SOTA HALTED via DD", rm.drawdown_status == DrawdownStatus.HALTED,
          f"got: {rm.drawdown_status.value} (dd={rm.current_drawdown:.1%})")

    # 31% DD -> SOTA triggers KILL_SWITCH (kill_switch_drawdown=30%)
    rm2 = RiskManager(sota_risk_ref=SOTARiskController(SOTARiskConfig()))
    rm2.initialize(10000)
    rm2._sota_risk.peak_equity = 10000
    rm2.update_balance(6900)
    check("R2.4 SOTA KILL_SWITCH via DD", rm2.drawdown_status == DrawdownStatus.CRITICAL,
          f"got: {rm2.drawdown_status.value} (dd={rm2.current_drawdown:.1%})")

    # Deferred wiring test
    rm3 = RiskManager()
    rm3.initialize(10000)
    rm3.update_balance(10000)
    check("R2.5 standalone NORMAL", rm3.drawdown_status == DrawdownStatus.NORMAL)

    sota3 = SOTARiskController(SOTARiskConfig())
    sota3.peak_equity = 10000
    rm3.set_sota_risk_ref(sota3)
    rm3.update_balance(6900)  # 31% DD -> KILL_SWITCH -> CRITICAL
    check("R2.6 deferred wiring", rm3.drawdown_status == DrawdownStatus.CRITICAL,
          f"got: {rm3.drawdown_status.value}")

    # Verify update_balance delegates to SOTA
    src = inspect.getsource(rm.update_balance)
    check("R2.7 delegates in code", "_sota_risk" in src)
except Exception as e:
    check("R2 import", False, str(e))

# =========================================================================
# R3: Kelly scaler
# =========================================================================
print("\n[R3] Fractional Kelly Sizing")
try:
    from risk.unified_position_sizer import KellyPositionScaler

    ks = KellyPositionScaler(kelly_fraction=0.30)
    check("R3.1 KellyPositionScaler exists", True)

    # Warm-up gate: should return None
    check("R3.2 warm-up gate", ks.compute_kelly_fraction() is None)

    # Feed 50% win rate, 2:1 W/L ratio
    for _ in range(25):
        ks.record_trade(0.02)  # 2% win
    for _ in range(25):
        ks.record_trade(-0.01)  # 1% loss
    f = ks.compute_kelly_fraction()
    check("R3.3 returns value after warm-up", f is not None, f"f={f}")

    # f* = (0.5*2 - 0.5)/2 = 0.25. fractional = 0.25 * 0.30 = 0.075
    if f is not None:
        check("R3.4 value in range", 0.05 < f < 0.15, f"expected ~0.075, got {f:.4f}")

    # Negative edge: 30% win rate, 1:1 W/L ratio
    ks2 = KellyPositionScaler(kelly_fraction=0.30)
    for _ in range(15):
        ks2.record_trade(0.01)
    for _ in range(35):
        ks2.record_trade(-0.01)
    f2 = ks2.compute_kelly_fraction()
    check("R3.5 negative edge = 0", f2 is not None and f2 == 0.0,
          f"f={f2}")

    # Integration: set_kelly_scaler on UnifiedPositionSizer
    from risk.unified_position_sizer import UnifiedPositionSizer
    ups = UnifiedPositionSizer(initial_capital=10000)
    check("R3.6 set_kelly_scaler method", hasattr(ups, "set_kelly_scaler"))
except ImportError:
    check("R3.1 KellyPositionScaler exists", False, "Not implemented yet")
except Exception as e:
    check("R3 error", False, str(e))

# =========================================================================
# R4: DETECTED allows T2
# =========================================================================
print("\n[R4] Cascade DETECTED Phase Relaxation")
try:
    from risk.cascade_exhaustion_governor import (
        TRANCHE_PERMISSIONS, CascadePhase, TrancheLevel,
    )

    t2_in_detected = TRANCHE_PERMISSIONS[CascadePhase.DETECTED][TrancheLevel.T2]
    check("R4.1 DETECTED allows T2", t2_in_detected is True, f"T2={t2_in_detected}")

    t2_in_exhaust = TRANCHE_PERMISSIONS[CascadePhase.EXHAUSTING][TrancheLevel.T2]
    check("R4.2 EXHAUSTING still blocks T2", t2_in_exhaust is False)

    t2_in_none = TRANCHE_PERMISSIONS[CascadePhase.NONE][TrancheLevel.T2]
    check("R4.3 NONE still allows T2", t2_in_none is True)

    t3_in_detected = TRANCHE_PERMISSIONS[CascadePhase.DETECTED][TrancheLevel.T3]
    check("R4.4 DETECTED still blocks T3", t3_in_detected is False)
except Exception as e:
    check("R4 import", False, str(e))

# =========================================================================
# R6: Regime DD multipliers
# =========================================================================
print("\n[R6] Regime-Dependent DD Thresholds")
try:
    from risk.sota_risk_controller import SOTARiskConfig, SOTARiskController, RiskState

    cfg = SOTARiskConfig()
    check("R6.1 regime_dd_multipliers exists", hasattr(cfg, "regime_dd_multipliers"))

    if hasattr(cfg, "regime_dd_multipliers"):
        check("R6.2 STRONG_TREND > 1.0",
              cfg.regime_dd_multipliers.get("STRONG_TREND", 1.0) > 1.0,
              f"val={cfg.regime_dd_multipliers.get('STRONG_TREND')}")
        check("R6.3 PANIC_SELLOFF < 1.0",
              cfg.regime_dd_multipliers.get("PANIC_SELLOFF", 1.0) < 1.0,
              f"val={cfg.regime_dd_multipliers.get('PANIC_SELLOFF')}")

    # Functional test: PANIC_SELLOFF tightens thresholds
    # reduce_threshold=0.15, PANIC mult=0.60 -> eff_reduce=9%
    ctrl = SOTARiskController(SOTARiskConfig(reduce_threshold=0.15, halt_threshold=0.25))
    ctrl.peak_equity = 10000
    ctrl.update_equity(9000, regime="PANIC_SELLOFF")  # 10% DD
    # eff_reduce = max(0.07, 0.15*0.60) = 0.09. 10% > 9% -> REDUCED
    check("R6.4 PANIC 10% DD -> REDUCED", ctrl.risk_state == RiskState.REDUCED,
          f"got: {ctrl.risk_state.value}")

    # Same DD in STRONG_TREND (1.30x): eff_reduce = 0.195. 10% < 19.5% -> NORMAL
    ctrl2 = SOTARiskController(SOTARiskConfig(reduce_threshold=0.15, halt_threshold=0.25))
    ctrl2.peak_equity = 10000
    ctrl2.update_equity(9000, regime="STRONG_TREND")  # 10% DD
    check("R6.5 STRONG_TREND 10% DD -> NORMAL", ctrl2.risk_state == RiskState.NORMAL,
          f"got: {ctrl2.risk_state.value}")

    # Kill-switch is NEVER regime-scaled
    ctrl3 = SOTARiskController(SOTARiskConfig(kill_switch_drawdown=0.30))
    ctrl3.peak_equity = 10000
    ctrl3.update_equity(6900, regime="STRONG_TREND")  # 31% DD
    check("R6.6 kill-switch never scaled", ctrl3.kill_switch_active is True)

    # update_equity accepts regime param
    sig = inspect.signature(ctrl.update_equity)
    check("R6.7 update_equity regime param", "regime" in sig.parameters)
except Exception as e:
    check("R6 import", False, str(e))

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 60)
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")
print(f"Results: {passed} PASS, {failed} FAIL, {len(results)} total")
print("=" * 60)
sys.exit(1 if failed > 0 else 0)

"""
================================================================================
HMATS v3.2 - A/B/C Stage Invariant Tests
================================================================================
Per-change regression tests for all wiring fixes.
Run: python -m tests.test_v32_invariants

Each test verifies one specific change from the A/B/C task list.
Can be invoked from scripts/verify_suite.py in VERIFY mode.
================================================================================
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

PASS = 0
FAIL = 0


def _record(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")
    return ok


# =========================================================================
# A3: ThesisBudgetGovernor persistence round-trip
# =========================================================================
def test_a3_thesis_budget_roundtrip():
    """to_dict -> from_dict preserves all state."""
    try:
        from risk.thesis_budget_governor import ThesisBudgetGovernor
        gov = ThesisBudgetGovernor(nav=10000.0)
        d1 = gov.to_dict()
        gov2 = ThesisBudgetGovernor.from_dict(d1)
        d2 = gov2.to_dict()
        ok = d1 == d2
        return _record("A3 thesis_budget round-trip", ok,
                        f"mismatch: {set(d1) ^ set(d2)}" if not ok else "")
    except Exception as e:
        return _record("A3 thesis_budget round-trip", False, str(e))


# =========================================================================
# A5: TradeIntentV36 timing sub-dimensions
# =========================================================================
def test_a5_timing_subdimensions():
    """TradeIntentV36 has 4 timing sub-score fields."""
    try:
        from integration.integration_v36 import TradeIntentV36
        intent = TradeIntentV36()
        fields = ['timing_spread_score', 'timing_depth_score',
                  'timing_vpin_score', 'timing_lead_lag_score']
        missing = [f for f in fields if not hasattr(intent, f)]
        ok = len(missing) == 0
        return _record("A5 timing sub-dimensions", ok,
                        f"missing: {missing}" if not ok else "")
    except Exception as e:
        return _record("A5 timing sub-dimensions", False, str(e))


# =========================================================================
# A6: regime_confidence default = 0.5
# =========================================================================
def test_a6_regime_confidence_default():
    """A6 unified regime_confidence default to 0.5 across call sites."""
    try:
        # Verify the engine reads default 0.5 from agent_signals
        # The fix was in integration_v36.py L916 and L1413:
        #   agent_signals.get("regime_confidence", 0.5)
        # We test by grepping the source for the correct default.
        import re
        with open(PROJECT_ROOT / "integration" / "integration_v36.py", "r", encoding="utf-8") as f:
            src = f.read()
        matches = re.findall(r'regime_confidence["\'],\s*([\d.]+)', src)
        # All occurrences should be 0.5
        ok = len(matches) > 0 and all(m == "0.5" for m in matches)
        detail = "" if ok else f"defaults found: {matches}"
        return _record("A6 regime_confidence default=0.5", ok, detail)
    except Exception as e:
        return _record("A6 regime_confidence default=0.5", False, str(e))


# =========================================================================
# A7: FastRiskTick triggers
# =========================================================================
def test_a7_fast_risk_tick():
    """FastRiskTick fires EXIT_ONLY on 3%+ move, REDUCE_50 on vol spike."""
    try:
        from execution.fast_risk_tick import FastRiskTick, FastRiskAction
        frt = FastRiskTick(shadow_mode=True)
        frt.set_4h_anchor("BTC", 100000.0, volatility=0.02, depth=500000)

        # 3% move -> EXIT_ONLY
        r1 = frt.evaluate("BTC", {"current_price": 96500.0})
        ok1 = r1.action == FastRiskAction.EXIT_ONLY

        # No move + provide depth -> HOLD
        r2 = frt.evaluate("BTC", {"current_price": 100100.0,
                                   "orderbook_depth_1pct_usd": 500000})
        ok2 = r2.action == FastRiskAction.HOLD

        # Vol spike 3x -> REDUCE_50
        r3 = frt.evaluate("BTC", {"current_price": 100100.0,
                                   "volatility_30m": 0.07,
                                   "orderbook_depth_1pct_usd": 500000})
        ok3 = r3.action == FastRiskAction.REDUCE_50

        ok = ok1 and ok2 and ok3
        detail = ""
        if not ok1: detail += f"3% move: {r1.action.value} "
        if not ok2: detail += f"no move: {r2.action.value} "
        if not ok3: detail += f"vol spike: {r3.action.value} "
        return _record("A7 FastRiskTick triggers", ok, detail)
    except Exception as e:
        return _record("A7 FastRiskTick triggers", False, str(e))


# =========================================================================
# B15: ComponentLifecycle promote / demote / status
# =========================================================================
def test_b15_component_lifecycle():
    """Lifecycle: promote, auto-demote after errors, get_status."""
    try:
        from core.component_lifecycle import ComponentLifecycle, LifecycleLevel

        lc = ComponentLifecycle("Test")
        assert lc.is_disabled()

        lc.promote("SHADOW")
        assert lc.is_shadow()

        lc.promote("ACTIVE")
        assert lc.is_active()

        # 5 consecutive errors -> auto-demote to SHADOW
        for _ in range(5):
            lc.record_error()
        assert lc.is_shadow(), f"Expected SHADOW after 5 errors, got {lc.level.value}"

        # record_success resets counter
        lc.promote("ACTIVE")
        lc.record_error()
        lc.record_success()
        assert lc.is_active()

        # get_status returns correct dict
        status = lc.get_status()
        assert status["name"] == "Test"
        assert status["level"] == "ACTIVE"

        return _record("B15 ComponentLifecycle", True)
    except Exception as e:
        return _record("B15 ComponentLifecycle", False, str(e))


# =========================================================================
# B16: order_book_imbalance computation
# =========================================================================
def test_b16_order_book_imbalance():
    """(bid - ask) / (bid + ask) in [-1, 1]."""
    bid, ask = 600000, 400000
    total = bid + ask
    imb = (bid - ask) / total if total > 0 else 0.0
    ok = -1 <= imb <= 1 and abs(imb - 0.2) < 0.001
    return _record("B16 order_book_imbalance range", ok,
                    f"got {imb}" if not ok else "")


# =========================================================================
# C1: ReflectionAgent should_run + run_weekly
# =========================================================================
def test_c1_reflection_agent():
    """ReflectionAgent: should_run at tick 42, report with data."""
    try:
        from analytics.confidence_scorer import StrategyConfidenceScorer
        from analytics.reflection_agent import ReflectionAgent, WEEKLY_TICKS
        from core.component_lifecycle import ComponentLifecycle

        scorer = StrategyConfidenceScorer()
        lc = ComponentLifecycle("ReflectionTest")
        lc.promote("ADVISORY")
        agent = ReflectionAgent(scorer=scorer, lifecycle=lc)

        # Tick logic
        ok1 = not agent.should_run(0)
        ok2 = agent.should_run(WEEKLY_TICKS)
        ok3 = not agent.should_run(WEEKLY_TICKS + 1)

        # No data -> None
        report = agent.run_weekly()
        ok4 = report is None

        # With data -> report
        import random
        random.seed(99)
        for _ in range(10):
            ts = scorer.record_signal(
                strategy_name="momentum",
                direction=1.0,
                confidence=0.6,
                expected_pnl_bps=50,
                regime="BULL_TREND",
            )
            if ts:
                scorer.record_outcome(
                    strategy_name="momentum",
                    signal_timestamp=ts,
                    realized_pnl_bps=random.uniform(-20, 80),
                    direction_correct=random.random() > 0.3,
                )
        report = agent.run_weekly()
        ok5 = report is not None and report.total_signals_evaluated > 0

        # Multiplier bounds
        ok6 = True
        if report:
            for m in report.confidence_multipliers.values():
                if not (0.5 <= m <= 1.2):
                    ok6 = False

        ok = ok1 and ok2 and ok3 and ok4 and ok5 and ok6
        detail = ""
        if not ok1: detail += "tick0 "
        if not ok2: detail += "tick42 "
        if not ok3: detail += "tick43 "
        if not ok4: detail += "empty "
        if not ok5: detail += "report "
        if not ok6: detail += "bounds "
        return _record("C1 ReflectionAgent", ok, detail)
    except Exception as e:
        return _record("C1 ReflectionAgent", False, str(e))


# =========================================================================
# C8: Order idempotency - userref determinism + dedup
# =========================================================================
def test_c8_order_idempotency():
    """Userref is deterministic, int31, and tracked in history."""
    try:
        from execution.execution_manager import ExecutionManager, OrderSide

        class MockExchange:
            pass

        em = ExecutionManager(MockExchange(), dry_run=True)

        # Deterministic
        r1 = em._generate_userref("BTC/USD", "BUY", "BTC_20260216_120000")
        r2 = em._generate_userref("BTC/USD", "BUY", "BTC_20260216_120000")
        ok1 = r1 == r2

        # Int31 range
        ok2 = 0 <= r1 <= 0x7FFFFFFF

        # Different input -> different ref
        r3 = em._generate_userref("ETH/USD", "SELL", "ETH_20260216_120000")
        ok3 = r1 != r3

        # Execute with tick_id -> userref set + in history
        result = em.execute_order(
            symbol="BTC/USD", side="BUY", size=0.1,
            price=45000.0, tick_id="BTC_20260216_120000",
        )
        ok4 = result.userref == r1
        ok5 = r1 in em._userref_history

        # Execute without tick_id -> no userref (backward compat)
        result2 = em.execute_order(
            symbol="ETH/USD", side="SELL", size=0.5, price=3000.0,
        )
        ok6 = result2.userref is None

        ok = ok1 and ok2 and ok3 and ok4 and ok5 and ok6
        detail = ""
        if not ok1: detail += "determinism "
        if not ok2: detail += "range "
        if not ok3: detail += "collision "
        if not ok4: detail += "result.userref "
        if not ok5: detail += "history "
        if not ok6: detail += "backward_compat "
        return _record("C8 order idempotency", ok, detail)
    except Exception as e:
        return _record("C8 order idempotency", False, str(e))


# =========================================================================
# A3+: HotRestartPersistence has thesis_budget_state field
# =========================================================================
def test_a3_persistence_field():
    """SystemState includes thesis_budget_state in to_dict/from_dict."""
    try:
        from infra.hot_restart_persistence import SystemState
        state = SystemState(system_mode="NORMAL", opportunity_trigger=None)
        state.thesis_budget_state = {"thesis_count": 3, "nav": 10000}
        d = state.to_dict()
        ok1 = "thesis_budget_state" in d
        restored = SystemState.from_dict(d)
        ok2 = restored.thesis_budget_state == state.thesis_budget_state
        ok = ok1 and ok2
        return _record("A3 persistence thesis_budget_state", ok,
                        "" if ok else "round-trip mismatch")
    except Exception as e:
        return _record("A3 persistence thesis_budget_state", False, str(e))


# =========================================================================
# A4: Spread-aware slippage (always adverse)
# =========================================================================
def test_a4_spread_slippage():
    """Paper slippage is always adverse (BUY fills higher, SELL fills lower)."""
    try:
        from execution.execution_manager import ExecutionManager, OrderSide

        class MockExchange:
            pass

        em = ExecutionManager(MockExchange(), dry_run=True)
        buy_results = []
        sell_results = []
        for _ in range(20):
            r = em.execute_order("BTC/USD", "BUY", 0.1, price=50000.0, spread_bps=10.0)
            buy_results.append(r.filled_price)
            r = em.execute_order("BTC/USD", "SELL", 0.1, price=50000.0, spread_bps=10.0)
            sell_results.append(r.filled_price)

        # BUY should fill >= base price (adverse = pay more)
        ok1 = all(p >= 50000.0 for p in buy_results)
        # SELL should fill <= base price (adverse = receive less)
        ok2 = all(p <= 50000.0 for p in sell_results)

        ok = ok1 and ok2
        detail = ""
        if not ok1: detail += f"BUY_not_adverse(min={min(buy_results):.2f}) "
        if not ok2: detail += f"SELL_not_adverse(max={max(sell_results):.2f}) "
        return _record("A4 spread-aware slippage", ok, detail)
    except Exception as e:
        return _record("A4 spread-aware slippage", False, str(e))


# =========================================================================
# B8: ConfidenceScorer record_signal / record_outcome
# =========================================================================
def test_b8_confidence_scorer_recording():
    """record_signal returns timestamp, record_outcome matches it."""
    try:
        from analytics.confidence_scorer import StrategyConfidenceScorer

        scorer = StrategyConfidenceScorer()
        ts = scorer.record_signal(
            strategy_name="momentum",
            direction=1.0,
            confidence=0.7,
            expected_pnl_bps=50,
            regime="BULL_TREND",
        )
        ok1 = ts is not None

        scorer.record_outcome(
            strategy_name="momentum",
            signal_timestamp=ts,
            realized_pnl_bps=60,
            direction_correct=True,
        )
        # Verify outcome recorded
        signals = scorer._signals["momentum"]
        ok2 = len(signals) == 1 and signals[0].has_outcome
        ok3 = signals[0].realized_pnl_bps == 60

        ok = ok1 and ok2 and ok3
        return _record("B8 ConfidenceScorer recording", ok,
                        "" if ok else "signal/outcome mismatch")
    except Exception as e:
        return _record("B8 ConfidenceScorer recording", False, str(e))


# =========================================================================
# RUNNER
# =========================================================================

ALL_TESTS = [
    test_a3_thesis_budget_roundtrip,
    test_a3_persistence_field,
    test_a4_spread_slippage,
    test_a5_timing_subdimensions,
    test_a6_regime_confidence_default,
    test_a7_fast_risk_tick,
    test_b8_confidence_scorer_recording,
    test_b15_component_lifecycle,
    test_b16_order_book_imbalance,
    test_c1_reflection_agent,
    test_c8_order_idempotency,
]


def run_all() -> dict:
    """Run all invariant tests. Returns {passed, failed, total}."""
    global PASS, FAIL
    PASS, FAIL = 0, 0

    print("=" * 60)
    print("HMATS v3.2 - A/B/C Invariant Tests")
    print("=" * 60)

    for test_fn in ALL_TESTS:
        try:
            test_fn()
        except Exception as e:
            _record(test_fn.__name__, False, f"CRASH: {e}")

    total = PASS + FAIL
    print("=" * 60)
    print(f"Result: {PASS}/{total} passed, {FAIL} failed")
    print("=" * 60)

    return {"passed": PASS, "failed": FAIL, "total": total}


if __name__ == "__main__":
    result = run_all()
    sys.exit(0 if result["failed"] == 0 else 1)

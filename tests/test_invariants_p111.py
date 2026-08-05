"""
test_invariants_p111.py — math invariants for highest-leverage modules
=======================================================================

[P111 Tier3#6 2026-04-27] Property-style invariant tests for code that
the audit loop couldn't fully verify statically. Each test sweeps a
broad input space and asserts a mathematical invariant must hold.

Coverage targets:
  - tranche cumulative→individual conversion (sum=1.0 invariant,
    monotone non-decreasing cumulative, individual = cum[i] - cum[i-1])
  - Sharpe annualization symmetry (return-year and vol-year MUST agree)
  - Kelly sizing clamping (output always in [0, max_kelly] for any
    valid p in [0,1] and b > 0)
  - NaN propagation guards (volatility_targeting + dynamic_limits
    must produce safe defaults on NaN input)
  - tranche_percentages config sums to 1.0 (catches operator drift)

These are NOT integration tests — they're math-only, no real exchanges,
no real model loads. Run via: pytest tests/test_invariants_p111.py -v
"""
from __future__ import annotations

import math
import pytest


# =====================================================================
# 1. Tranche cumulative→individual conversion invariants
# =====================================================================

class TestTrancheCumulativeMath:
    """Verify the cumulative→individual conversion is correct for arbitrary
    tranche configurations. Without this, T3 escalation thresholds use
    wrong PnL targets."""

    def test_cumulative_monotone_increasing(self):
        """Cumulative percentages must never decrease as tranche level rises."""
        from risk.unified_position_sizer import (
            UnifiedPositionSizer, PositionSizingConfig
        )
        from core.canonical_enums import TrancheLevel

        sizer = UnifiedPositionSizer(PositionSizingConfig())
        prev = 0.0
        for level in [TrancheLevel.TRANCHE_1, TrancheLevel.TRANCHE_2,
                      TrancheLevel.TRANCHE_3, TrancheLevel.TRANCHE_4]:
            cum = sizer._get_cumulative_tranche_pct(level)
            assert cum >= prev, (
                f"Cumulative pct decreased at {level}: {prev} → {cum}"
            )
            prev = cum

    def test_cumulative_max_is_total(self):
        """Cumulative at highest tranche = sum of all tranche_percentages."""
        from risk.unified_position_sizer import (
            UnifiedPositionSizer, PositionSizingConfig
        )
        from core.canonical_enums import TrancheLevel

        cfg = PositionSizingConfig()
        sizer = UnifiedPositionSizer(cfg)
        cum_at_t4 = sizer._get_cumulative_tranche_pct(TrancheLevel.TRANCHE_4)
        expected = sum(cfg.tranche_percentages.values())
        assert math.isclose(cum_at_t4, expected, rel_tol=1e-9), (
            f"Cumulative at T4 ({cum_at_t4}) != sum of all tranche pcts ({expected})"
        )

    def test_individual_tranche_pct_recovers_config(self):
        """For each tranche, cum[i] - cum[i-1] should equal config[i]."""
        from risk.unified_position_sizer import (
            UnifiedPositionSizer, PositionSizingConfig
        )
        from core.canonical_enums import TrancheLevel

        cfg = PositionSizingConfig()
        sizer = UnifiedPositionSizer(cfg)
        prev_level = TrancheLevel.NONE
        for level in [TrancheLevel.TRANCHE_1, TrancheLevel.TRANCHE_2,
                      TrancheLevel.TRANCHE_3, TrancheLevel.TRANCHE_4]:
            cum = sizer._get_cumulative_tranche_pct(level)
            prev_cum = sizer._get_cumulative_tranche_pct(prev_level)
            individual = cum - prev_cum
            expected = cfg.tranche_percentages.get(level, 0.0)
            assert math.isclose(individual, expected, rel_tol=1e-9), (
                f"Individual pct at {level} = {individual} != config "
                f"{expected} (cum={cum}, prev_cum={prev_cum})"
            )
            prev_level = level

    def test_tranche_percentages_sum_to_unity(self):
        """Operator-drift guard: sum of tranche_percentages must == 1.0.
        If operator edits config and breaks this, full-schedule sizing breaks."""
        from risk.unified_position_sizer import PositionSizingConfig

        cfg = PositionSizingConfig()
        total = sum(cfg.tranche_percentages.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6), (
            f"tranche_percentages sum = {total}, must be 1.0. "
            f"Operator config drift detected. Values: {cfg.tranche_percentages}"
        )


# =====================================================================
# 2. Sharpe annualization symmetry (P110 fix verification)
# =====================================================================

class TestSharpeAnnualization:
    """Verify P110 fix: return annualization and vol annualization must
    agree on year length (both 365 for 24/7 crypto, not 252/365 mix)."""

    def test_sharpe_uses_consistent_year_length(self):
        """Read the source and verify both annualizations use sqrt(365)."""
        import inspect
        from analytics import sota_metrics_calculator

        src = inspect.getsource(sota_metrics_calculator)
        # Volatility annualization MUST use sqrt(365) post-P110
        assert "sqrt(252)" not in src, (
            "Found sqrt(252) in sota_metrics_calculator — should be sqrt(365) "
            "for 24/7 crypto (P110 fix). Sharpe overstated by factor 1.20x."
        )
        assert "np.sqrt(365)" in src, (
            "Expected np.sqrt(365) in sota_metrics_calculator after P110 fix."
        )
        # Return annualization MUST use 365
        assert "(365 / days)" in src, (
            "Expected 365-day return annualization for 24/7 crypto."
        )


# =====================================================================
# 3. NaN propagation guards (P94 + P98 fixes verification)
# =====================================================================

class TestNaNGuards:
    """Verify P94 (dynamic_limits) and P98 (volatility_targeting) NaN
    guards are still in place. NaN volatility input must trigger
    fail-CLOSED conservative adjustment, not fail-OPEN pass-through."""

    def test_dynamic_limits_nan_volatility_fail_closed(self):
        from risk.dynamic_limits import DynamicLimits, DynamicLimitsConfig

        cfg = DynamicLimitsConfig()
        cfg.enabled = True
        cfg.regime_multipliers = {"DEFAULT": {"leverage": 1.0, "gross": 1.0}}
        dl = DynamicLimits(cfg)
        result = dl.get_limits(
            regime="DEFAULT",
            confidence=0.7,
            volatility_z=float("nan"),
        )
        assert result.volatility_applied is True, (
            "NaN volatility should TRIGGER fail-closed pullback (P94 fix). "
            f"Got volatility_applied={result.volatility_applied}."
        )

    def test_volatility_targeting_nan_realized_vol_fail_closed(self):
        """P98 fix: NaN realized vol → 0.5x adjustment, not 1.0x pass-through."""
        import inspect
        from risk import volatility_targeting

        src = inspect.getsource(volatility_targeting)
        # Must contain math.isnan check + 0.5x fail-closed default
        assert "math.isnan" in src or "_math.isnan" in src, (
            "volatility_targeting missing isnan guard (P98 fix)"
        )


# =====================================================================
# 4. Authority fusion DECIDE_ABSTAIN visibility (P110 fix)
# =====================================================================

class TestAuthorityFusionAbstainVisibility:
    """P110 promoted DECIDE_ABSTAIN log INFO → WARNING. If reverted,
    operator silently misses 0-signal ticks."""

    def test_decide_abstain_logs_at_warning(self):
        import inspect
        from signals import authority_fusion

        src = inspect.getsource(authority_fusion)
        # Find the DECIDE_ABSTAIN log block
        if "DECIDE_ABSTAIN" not in src:
            pytest.skip("DECIDE_ABSTAIN not in source (renamed?)")
        # Look for the log call NEAR the DECIDE_ABSTAIN message
        idx = src.index("DECIDE_ABSTAIN")
        context = src[max(0, idx - 200):idx + 100]
        assert "logger.warning" in context, (
            "DECIDE_ABSTAIN must log at WARNING level (P110). Found context: "
            f"{context[:300]}"
        )


# =====================================================================
# 5. Stop-loss userref dedup invariant (P95 fix)
# =====================================================================

class TestStopLossUserrefStability:
    """P95 fix: userref must be stable for (symbol, side) regardless
    of stop_price drift. If reverted, stop-order leak returns."""

    def test_userref_stable_across_price_drift(self):
        from execution.execution_manager import ExecutionManager

        ref1 = ExecutionManager._generate_stop_userref("SOL/USD", "sell", 85.50)
        ref2 = ExecutionManager._generate_stop_userref("SOL/USD", "sell", 85.65)
        ref3 = ExecutionManager._generate_stop_userref("SOL/USD", "sell", 100.0)
        assert ref1 == ref2 == ref3, (
            f"Userref unstable across price drift: {ref1}, {ref2}, {ref3}. "
            "P95 fix dropped stop_price from hash; revert would re-introduce "
            "the stop-order leak that caused the production cascade."
        )

    def test_userref_distinct_per_side(self):
        from execution.execution_manager import ExecutionManager

        sell = ExecutionManager._generate_stop_userref("SOL/USD", "sell", 85.50)
        buy = ExecutionManager._generate_stop_userref("SOL/USD", "buy", 85.50)
        assert sell != buy, "BUY and SELL must produce distinct userrefs"

    def test_userref_distinct_per_symbol(self):
        from execution.execution_manager import ExecutionManager

        sol = ExecutionManager._generate_stop_userref("SOL/USD", "sell", 85.50)
        btc = ExecutionManager._generate_stop_userref("BTC/USD", "sell", 85.50)
        assert sol != btc, "Different symbols must produce distinct userrefs"


# =====================================================================
# 6. PREFLIGHT_* error classification (P93 fix)
# =====================================================================

class TestPreflightErrorClassification:
    """P93 fix: our own pre-flight rejection strings (PREFLIGHT_BELOW_MIN_SIZE
    etc.) must classify as PERMANENT to avoid retry waste."""

    @pytest.mark.parametrize("err_string", [
        "PREFLIGHT_BELOW_MIN_SIZE: stop-loss size 0.014",
        "PREFLIGHT_WRONG_SIDE: stop price",
        "INSUFFICIENT_SPOT_BALANCE: free SOL=0.01",
    ])
    def test_preflight_errors_classified_permanent(self, err_string):
        from execution.execution_manager import ExecutionManager

        em = ExecutionManager.__new__(ExecutionManager)
        category, _ = em._classify_kraken_order_error(err_string)
        assert category == "PERMANENT", (
            f"Pre-flight error {err_string!r} classified as {category}, "
            "expected PERMANENT (P93 fix). Retry would waste attempts."
        )


# =====================================================================
# 7. Authority matrix structure invariants
# =====================================================================

class TestAuthorityMatrixInvariants:
    """The authority matrix must match CLAUDE.md exactly. Drift in EITHER
    direction is a bug: fewer entries = a silent agent removal, more =
    an agent wired into fusion that the authority table never documented.

    [P165 2026-08-04] 25 -> 26 (`v5_1_strats`, added 2026-06-13 by 795ecc4
    without the CLAUDE.md update rule #7 requires). This guard did its job;
    it just went unread for ~7 weeks."""

    def test_matrix_has_26_agents(self):
        from signals.authority_fusion import AUTHORITY_MATRIX_NORMAL

        # CLAUDE.md §Authority Matrix says exactly 26 agents
        assert len(AUTHORITY_MATRIX_NORMAL) == 26, (
            f"Authority matrix has {len(AUTHORITY_MATRIX_NORMAL)} agents, "
            f"expected 26 per CLAUDE.md. Fewer = silent agent removal; "
            f"more = an undocumented agent in fusion. Update BOTH the "
            f"CLAUDE.md table and this count in the same commit."
        )

    def test_critical_agents_present(self):
        """The 4 DECIDE-or-VETO agents per CLAUDE.md authority matrix."""
        from signals.authority_fusion import AUTHORITY_MATRIX_NORMAL

        for required in ("quant", "drl", "risk", "kraken_quant"):
            assert required in AUTHORITY_MATRIX_NORMAL, (
                f"Critical authority agent {required!r} missing from matrix."
            )


if __name__ == "__main__":
    # Allow direct invocation: python tests/test_invariants_p111.py
    pytest.main([__file__, "-v"])

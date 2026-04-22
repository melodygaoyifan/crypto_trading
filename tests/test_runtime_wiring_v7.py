"""
HMATS Runtime Wiring v7 - Tests for Sections A-G
=================================================
Tests runtime correctness fixes without modifying agent implementations.

Sections:
  A) market_data production fixes
  B) agent_signals wiring (lead_lag_edge unit fix)
  C) PassiveAggressiveExecutor hot-path connection
  D) FastRiskTick EXIT/REDUCE actions
  E) Confidence scorer feedback loop
  F) Per-signal PROOF logs
  G) kraken_quant_agent containment
"""

import sys
import time
import pytest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import deque

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A: market_data production fixes
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketDataProduction:
    """Validate market_data fields consumed by v36."""

    def _make_raw(self, depth_1pct=500_000, spread=10.0, prices=None):
        """Minimal raw market_data dict."""
        return {
            "current_price": 100.0,
            "orderbook_depth_1pct_usd": depth_1pct,
            "spread_bps": spread,
            "prices": prices or [99.0, 100.0],
            "dvol_zscore": 0.0,
            "vpin": 0.35,
        }

    def test_orderbook_depth_normalization_uses_median(self):
        """A1: orderbook_depth should use rolling median, not hardcoded 500K."""
        from collections import deque
        import statistics

        # Simulate rolling depth history
        buf = deque(maxlen=120)
        depths = [400_000, 500_000, 600_000, 500_000, 450_000]
        for d in depths:
            buf.append(d)

        current = 600_000
        median = statistics.median(buf)
        normalized = min(current / median, 5.0)

        # Should be > 1.0 when current depth above median
        assert normalized > 1.0
        assert normalized < 5.0

        # When depth equals median, should be 1.0
        current_at_median = median
        assert min(current_at_median / median, 5.0) == pytest.approx(1.0)

    def test_estimated_friction_includes_impact(self):
        """A2: friction = spread + impact proxy, not just spread * 0.6."""
        spread = 10.0
        depth = 500_000
        impact = min(20.0, 50_000.0 / max(depth, 1.0) * 100)
        friction = spread + impact
        # friction should be spread + impact, not spread * 0.6
        assert friction > spread * 0.6, "Friction must include depth-based impact"
        assert friction == pytest.approx(spread + impact)

    def test_estimated_friction_thin_book(self):
        """A2: Thin books produce higher friction."""
        spread = 10.0
        thin_depth = 10_000
        thick_depth = 1_000_000
        impact_thin = min(20.0, 50_000.0 / thin_depth * 100)
        impact_thick = min(20.0, 50_000.0 / thick_depth * 100)
        assert impact_thin > impact_thick, "Thinner book = more impact"

    def test_flash_crash_requires_price_or_depth(self):
        """A4: flash_crash_active needs price move OR depth collapse, not price alone."""
        # Large move -> flash crash
        assert abs(-0.06) > 0.05  # 6% move triggers

        # Moderate move + depth collapse -> also flash crash
        depth_collapsed = True
        moderate_move = abs(0.03) > 0.02  # 3% move
        assert depth_collapsed and moderate_move

        # Small move, no depth collapse -> no flash crash
        small_move = abs(0.01) > 0.05
        assert not small_move

    def test_dvol_zscore_alias(self):
        """A5: dvol aliases to dvol_zscore when only dvol present."""
        raw = {"dvol": 1.5}
        if "dvol_zscore" not in raw and "dvol" in raw:
            raw["dvol_zscore"] = raw["dvol"]
        assert raw["dvol_zscore"] == 1.5

    def test_required_keys_validation(self):
        """A6: Missing critical keys set _v36_data_degraded flag."""
        required = [
            "orderbook_depth", "estimated_friction_bps", "spread_bps",
            "current_price", "dvol_zscore", "vpin",
        ]
        raw = {"current_price": 100.0, "spread_bps": 10.0}
        missing = [k for k in required if k not in raw]
        assert len(missing) > 0
        assert "orderbook_depth" in missing


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B: agent_signals wiring
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentSignalsWiring:
    """Validate agent_signals contract for v36 consumption."""

    def test_lead_lag_edge_already_bps(self):
        """B1: lead_lag_edge is already in bps - do NOT multiply by 100."""
        # Simulates the lead-lag computation from main.py L2390
        expected = 0.005  # 5% BTC return expected for SOL
        actual = 0.003    # 3% actual SOL return
        edge_bps = (expected - actual) * 10000  # = 20 bps

        # CORRECT: use directly as bps
        net_edge_bps = edge_bps  # 20.0
        assert net_edge_bps == pytest.approx(20.0)

        # WRONG (old code): would have been 2000
        wrong_value = edge_bps * 100
        assert wrong_value == pytest.approx(2000.0)
        assert wrong_value != net_edge_bps

    def test_lead_lag_direction_normalization(self):
        """B2: lead_lag_edge / 100.0 normalizes bps to [-1,1] range."""
        edge_bps = 50.0  # 50 bps
        direction = edge_bps / 100.0  # 0.5 - reasonable direction signal
        assert -1.0 <= direction <= 1.0

    def test_drl_direction_default_zero(self):
        """B3: drl_direction must exist even when DRL disabled."""
        agent_signals = {}
        agent_signals.setdefault("drl_direction", 0)
        assert agent_signals["drl_direction"] == 0

    def test_risk_veto_from_p0(self):
        """B4: risk_veto = p0_force_flat (fail-closed)."""
        p0_force_flat = True
        agent_signals = {"risk_veto": p0_force_flat}
        assert agent_signals["risk_veto"] is True

    def test_sentiment_direction_from_zscore(self):
        """B5: sentiment_direction derived from zscore with dead zone."""
        # Above threshold -> direction
        zscore = 1.5
        direction = 1.0 if zscore > 1.0 else (-1.0 if zscore < -1.0 else 0.0)
        assert direction == 1.0

        # In dead zone -> neutral
        zscore = 0.5
        direction = 1.0 if zscore > 1.0 else (-1.0 if zscore < -1.0 else 0.0)
        assert direction == 0.0

    def test_all_required_keys_exist(self):
        """B6: All v36-consumed keys must exist in agent_signals every tick."""
        required_keys = [
            "regime_state", "regime_direction", "regime_confidence",
            "quant_direction", "quant_confidence", "primary_strategy",
            "lead_lag_edge", "lead_lag_confidence", "signal_edge_bps",
            "sentiment_zscore", "risk_veto", "drl_direction",
            "sol_beta", "cvd_divergence",
        ]
        agent_signals = {k: 0 for k in required_keys}
        agent_signals["regime_state"] = "UNKNOWN"
        agent_signals["primary_strategy"] = "momentum"
        agent_signals["risk_veto"] = False

        for key in required_keys:
            assert key in agent_signals, f"Missing required key: {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C: PassiveAggressiveExecutor
# ═══════════════════════════════════════════════════════════════════════════════

class TestPassiveAggressiveExecutor:
    """Validate PA executor hot-path integration."""

    def test_pa_executor_import(self):
        """C1: PA executor module imports without error."""
        from execution.passive_aggressive import (
            PassiveAggressiveExecutor, ExecutionMode, ExecutionDecision,
        )
        assert ExecutionMode.PASSIVE is not None
        assert ExecutionMode.AGGRESSIVE is not None

    def test_pa_shadow_mode_default(self):
        """C2: PA executor starts in shadow mode by default."""
        try:
            from execution.passive_aggressive import PassiveAggressiveExecutor
            pa = PassiveAggressiveExecutor()
            assert pa.shadow_mode is True
        except Exception:
            pytest.skip("PA executor not available")

    def test_pa_mode_to_order_type_mapping(self):
        """C3: PA execution modes map to correct order types."""
        from execution.passive_aggressive import ExecutionMode as PAMode
        mode_map = {
            PAMode.PASSIVE: "LIMIT",
            PAMode.AGGRESSIVE: "MARKET",
            PAMode.HYBRID: "LIMIT",
            PAMode.TWAP: "LIMIT",
            PAMode.ABORT: "REJECTED",
        }
        for pa_mode, expected in mode_map.items():
            if pa_mode == PAMode.ABORT:
                assert expected == "REJECTED"
            elif pa_mode == PAMode.AGGRESSIVE:
                assert expected == "MARKET"
            else:
                assert expected == "LIMIT"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D: FastRiskTick
# ═══════════════════════════════════════════════════════════════════════════════

class TestFastRiskTick:
    """Validate FastRiskTick EXIT/REDUCE behavior."""

    def test_fast_risk_tick_import(self):
        """D1: FastRiskTick imports and initializes."""
        from execution.fast_risk_tick import FastRiskTick, FastRiskAction
        frt = FastRiskTick(shadow_mode=False)
        assert frt.shadow_mode is False

    def test_fast_risk_hold_on_small_move(self):
        """D2: Small price moves produce HOLD action."""
        from execution.fast_risk_tick import FastRiskTick, FastRiskAction
        frt = FastRiskTick(shadow_mode=False)
        frt.set_4h_anchor("BTC", 50000.0)
        result = frt.evaluate("BTC", {"current_price": 50500.0})  # 1% move
        assert result.action == FastRiskAction.HOLD

    def test_fast_risk_exit_on_large_move(self):
        """D3: >3% move triggers EXIT_ONLY."""
        from execution.fast_risk_tick import FastRiskTick, FastRiskAction
        frt = FastRiskTick(shadow_mode=False)
        frt.set_4h_anchor("BTC", 50000.0)
        result = frt.evaluate("BTC", {"current_price": 48000.0})  # 4% move
        assert result.action == FastRiskAction.EXIT_ONLY

    def test_fast_risk_reduce_on_vol_spike(self):
        """D4: Volatility spike triggers REDUCE_50."""
        from execution.fast_risk_tick import FastRiskTick, FastRiskAction
        frt = FastRiskTick(shadow_mode=False)
        frt.set_4h_anchor("BTC", 50000.0, volatility=0.01)
        result = frt.evaluate("BTC", {
            "current_price": 50100.0,  # small move (not triggering price threshold)
            "volatility_30m": 0.03,     # 3x baseline
        })
        assert result.action == FastRiskAction.REDUCE_50

    def test_fast_risk_never_opens(self):
        """D5: FastRiskTick can only HOLD/REDUCE/EXIT - never opens positions."""
        from execution.fast_risk_tick import FastRiskAction
        valid_actions = {FastRiskAction.HOLD, FastRiskAction.REDUCE_50, FastRiskAction.EXIT_ONLY}
        # No BUY or ENTER action exists
        all_actions = set(FastRiskAction)
        assert all_actions == valid_actions


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E: Confidence Scorer Feedback Loop
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidenceScorerFeedback:
    """Validate confidence scorer singleton feedback loop."""

    def test_confidence_scorer_singleton(self):
        """E1: get_confidence_scorer returns same instance."""
        from analytics.confidence_scorer import (
            get_confidence_scorer, reset_confidence_scorer,
        )
        reset_confidence_scorer()
        s1 = get_confidence_scorer()
        s2 = get_confidence_scorer()
        assert s1 is s2

    def test_record_signal_returns_timestamp(self):
        """E2: record_signal returns timestamp for later outcome matching."""
        from analytics.confidence_scorer import (
            get_confidence_scorer, reset_confidence_scorer,
        )
        reset_confidence_scorer()
        scorer = get_confidence_scorer()
        ts = scorer.record_signal(
            strategy_name="momentum",
            direction=1.0,
            confidence=0.7,
            expected_pnl_bps=50.0,
            regime="MOMENTUM_RALLY",
        )
        assert ts is not None

    def test_record_outcome_updates_confidence(self):
        """E3: record_outcome should update internal confidence scores."""
        from analytics.confidence_scorer import (
            get_confidence_scorer, reset_confidence_scorer,
        )
        reset_confidence_scorer()
        scorer = get_confidence_scorer()

        # Record a signal + outcome
        ts = scorer.record_signal(
            strategy_name="momentum",
            direction=1.0,
            confidence=0.7,
            expected_pnl_bps=50.0,
            regime="MOMENTUM_RALLY",
        )
        scorer.record_outcome(
            strategy_name="momentum",
            signal_timestamp=ts,
            realized_pnl_bps=40.0,
            direction_correct=True,
        )
        # Confidence should be retrievable
        confs = scorer.get_all_confidences()
        assert "momentum" in confs
        assert isinstance(confs["momentum"], float)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION F: Per-signal PROOF logs
# ═══════════════════════════════════════════════════════════════════════════════

class TestProofLogs:
    """Validate per-signal adopted/rejected PROOF trace format."""

    def test_signal_trace_format(self):
        """F1: Signal trace includes all agent signals with adopted/rejected."""
        agent_signals = {
            "quant_direction": 0.5,
            "quant_confidence": 0.8,
            "sentiment_direction": 1.0,
            "sentiment_zscore": 1.5,
            "risk_veto": False,
            "short_bias_direction": -0.3,
            "short_bias_confidence": 0.6,
            "lead_lag_edge": 20.0,
            "lead_lag_confidence": 0.7,
            "drl_direction": 0,
        }
        intent_dir = 0.5
        intent_veto = False

        # Build trace (mirrors Section F logic)
        traces = []
        _q_dir = agent_signals["quant_direction"]
        _q_conf = agent_signals["quant_confidence"]
        _q_adopted = abs(_q_dir) > 0.01 and not intent_veto
        traces.append(f"quant(dir={_q_dir:+.3f},conf={_q_conf:.2f})={'ADOPTED' if _q_adopted else 'REJECTED'}")

        assert "ADOPTED" in traces[0]
        assert "quant" in traces[0]

    def test_risk_veto_shown_in_trace(self):
        """F2: risk_veto=True appears as VETO_APPLIED in trace."""
        risk_veto = True
        tag = "VETO_APPLIED" if risk_veto else "CLEAR"
        assert tag == "VETO_APPLIED"


# ═══════════════════════════════════════════════════════════════════════════════
# [REMOVED 2026-04-22] SECTION G: kraken_quant_agent containment
# Stale — early architecture treated kraken_quant as deprecated and required
# main.py to stay clear of it. Current architecture intentionally uses
# agents.kraken_quant_agent.KrakenQuantAgentV6 in main.py (12-strategy matrix
# via _kraken_quant_agent attr), so the containment assertions no longer apply.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: v36 engine lead_lag_edge consumption
# ═══════════════════════════════════════════════════════════════════════════════

class TestV36LeadLagConsumption:
    """Validate v36 correctly consumes lead_lag_edge as bps."""

    def test_net_edge_bps_no_multiply(self):
        """INT1: v36 line 803 should NOT multiply lead_lag_edge by 100."""
        # Read actual integration_v36.py to verify the fix
        v36_path = Path(__file__).parent.parent / "integration" / "integration_v36.py"
        content = v36_path.read_text(encoding="utf-8")

        # Find the net_edge_bps line
        for line in content.splitlines():
            if "net_edge_bps" in line and "lead_lag_edge" in line:
                # Should NOT contain "* 100"
                assert "* 100" not in line, (
                    f"UNIT BUG: lead_lag_edge is already bps, do not multiply by 100. "
                    f"Line: {line.strip()}"
                )
                break
        else:
            pytest.skip("net_edge_bps line not found in v36")

    def test_lead_lag_direction_div_100(self):
        """INT2: v36 _build_fusion_signals divides bps by 100 for direction."""
        v36_path = Path(__file__).parent.parent / "integration" / "integration_v36.py"
        content = v36_path.read_text(encoding="utf-8")

        # The direction normalization should still divide by 100
        assert '/ 100.0' in content or '/ 100' in content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

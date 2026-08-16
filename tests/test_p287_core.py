"""
[P287] Core/integration fix batch — pins for the 2026-08-16 read-through's
core-owned findings.

Each pin either exercises the fixed behavior directly or reads the source at
the fix site (for logic inlined in execute_intent_v2, which cannot be called
in isolation without a full runner). Source pins were falsification-probed:
surgically reverting the fix makes the pin red (probes recorded in the P287
campaign report, not committed).

The two LOOSENINGS (FLAT_ONLY exit pass-through, AC-5 exit exemption) each
carry a companion pin proving the block still fires on entries — an
exemption must never become a bypass.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

from tests._source_scan import code_only

REPO = Path(__file__).resolve().parent.parent
EXEC_SVC = (REPO / "core" / "execution_service.py").read_text(encoding="utf-8-sig")
# comment-stripped views for "statement X is GONE" assertions — the fixes'
# own explanatory comments quote the removed statements (P177 trap).
EXEC_SVC_CODE = code_only(REPO / "core" / "execution_service.py")
INTEG_CODE = code_only(REPO / "integration" / "integration_v36.py")
TICK_EXIT = (REPO / "core" / "tick_exit_triggers.py").read_text(encoding="utf-8-sig")
INTEG = (REPO / "integration" / "integration_v36.py").read_text(encoding="utf-8-sig")
CONSTITUTION = (REPO / "defense" / "constitution.py").read_text(encoding="utf-8-sig")
TREND = (REPO / "core" / "trend_decision_layer.py").read_text(encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 1. Trend-enforce edge claim (LIVE tightening)
# ---------------------------------------------------------------------------
class TestTrendEdgeClaim:
    def _layer(self, sig: float):
        from core.trend_decision_layer import TrendDecisionLayer

        layer = TrendDecisionLayer(mode="enforce", base_edge_bps=40.0,
                                   min_abs_signal=0.10,
                                   regime_gate_mode="off")

        class _FakeStrat:
            def compute(self, closes):
                return {"signal": sig, "target_position": sig}

            def min_history(self):
                return 1

        layer._strat = _FakeStrat()
        layer._closes["BTC"] = [100.0, 101.0]
        layer._closes_cached_at["BTC"] = time.time()
        return layer

    def test_engine_edge_never_rides_the_trend_trade(self):
        """The exact audit scenario: engine dir 0.9 (edge 58.5) vs trend 0.30
        (edge 12.0). The gate must see 12.0 — the signal that will trade —
        never the discarded engine's larger claim."""
        layer = self._layer(0.30)
        market_data = {"signal_edge_bps": 58.5, "quant_direction": 0.90}
        agent_signals = {}
        res = layer.process("BTC", None, agent_signals, market_data)
        assert res is not None and res["mode"] == "enforce"
        assert market_data["signal_edge_bps"] == pytest.approx(12.0)
        assert agent_signals["signal_edge_bps"] == pytest.approx(12.0)

    def test_trend_edge_still_asserted_when_engine_was_weaker(self):
        layer = self._layer(0.80)
        market_data = {"signal_edge_bps": 5.0, "quant_direction": 0.05}
        agent_signals = {}
        layer.process("BTC", None, agent_signals, market_data)
        assert market_data["signal_edge_bps"] == pytest.approx(32.0)

    def test_no_max_composition_survives_in_the_inject(self):
        # The bug was a max(...) between the engine's claim and the trend's.
        inject = TREND[TREND.index("# enforce: trend BECOMES"):]
        inject = inject[:inject.index("def ") if "def " in inject else len(inject)]
        assert "max(" not in inject.split("signal_edge_bps")[1].split("\n")[0], (
            "signal_edge_bps under enforce must be the trend's OWN edge — "
            "max() with the discarded engine signal's claim is the P287 bug")


# ---------------------------------------------------------------------------
# 2. ExecutionGuard FLAT_ONLY — exits pass, entries still blocked
# ---------------------------------------------------------------------------
class TestFlatOnlyExitPassThrough:
    def _guard_block(self) -> str:
        start = EXEC_SVC.index("T19: Pre-execution safety gate")
        end = EXEC_SVC.index("P0 FIX STEP 1")
        return EXEC_SVC[start:end]

    def test_flat_only_allows_close_and_reduce(self):
        blk = self._guard_block()
        assert "FLAT_ONLY" in blk
        assert "_is_full_exit_request or _is_reduce_request" in blk, (
            "FLAT_ONLY must let a close/reduce of a real position through "
            "(P195/P265h: a data-degraded tick must not trap the position)")

    def test_entries_still_rejected(self):
        """Companion pin: the exemption is not a bypass — the REJECTED return
        survives in the else branch for everything that is not a
        FLAT_ONLY close/reduce."""
        blk = self._guard_block()
        assert '"status": "REJECTED", "reason": f"ExecutionGuard:' in blk

    def test_close_never_wears_an_entry_side_label(self):
        blk = self._guard_block()
        assert '_side_str = "exit"' in blk
        assert "_execution_direction > 0" in blk, (
            "side must come from the CORRECTED direction, not intent.direction")
        # the old expression must be gone
        assert '"long" if intent.direction > 0 else "short"' not in blk

    def test_guard_module_flat_only_semantics_unchanged(self):
        # The fix is at the CALLER; the guard itself still returns FLAT_ONLY
        # for stale data (that contract is what the caller now honors).
        from defense.execution_guards import ExecutionMode
        assert hasattr(ExecutionMode, "FLAT_ONLY")

    # -- behavioral: drive the REAL execute_intent_v2 up to (and past) the
    #    guard. A source pin proves the code was written; this proves the
    #    branch runs (P234's lesson). The account_sync fake raises a sentinel
    #    so "passed the guard" is observable as the P0_FAIL_CLOSED reject.

    def _ctx(self, with_position: bool):
        import asyncio
        from types import SimpleNamespace
        from defense.execution_guards import ExecutionMode as _GM

        class _Guard:
            def __init__(self):
                self.block_reasons = ["Stale Data: test"]

            def check_execution(self, **kwargs):
                return False, _GM.FLAT_ONLY, {}

        class _Sync:
            dry_run = False

            async def refresh(self):
                raise RuntimeError("SENTINEL_EQUITY_UNAVAILABLE")

            def get_equity(self):  # pragma: no cover — refresh raises first
                raise RuntimeError("SENTINEL_EQUITY_UNAVAILABLE")

        from core.canonical_enums import RunMode
        positions = {}
        if with_position:
            positions["BTC"] = {"exposure": 0.20, "direction": 1.0}
        return SimpleNamespace(
            config=SimpleNamespace(mode=RunMode.PAPER, initial_capital=10_000.0),
            paper_positions=positions,
            fn_is_active_paper_position=lambda p: bool(p) and abs(
                p.get("exposure", 0.0)) > 1e-9,
            dead_man_switch=None,
            execution_guard=_Guard(),
            fn_get_drl_weight=None,
            account_sync=_Sync(),
        )

    def _run(self, ctx, intent):
        import asyncio
        from core.execution_service import execute_intent_v2
        return asyncio.run(execute_intent_v2(
            ctx, "BTC", intent, {"current_price": 100.0}, {}))

    def test_behavioral_full_exit_passes_flat_only_guard(self):
        from types import SimpleNamespace
        ctx = self._ctx(with_position=True)
        intent = SimpleNamespace(direction=-1.0, target_exposure=0.0)
        res = self._run(ctx, intent)
        # It must get PAST the guard and die on the sentinel equity instead.
        assert "ExecutionGuard" not in str(res.get("reason", "")), (
            f"full exit was rejected AT the FLAT_ONLY guard: {res}")
        assert "P0_FAIL_CLOSED" in str(res.get("reason", ""))

    def test_behavioral_entry_still_blocked_by_flat_only_guard(self):
        from types import SimpleNamespace
        ctx = self._ctx(with_position=False)
        intent = SimpleNamespace(direction=1.0, target_exposure=0.25)
        res = self._run(ctx, intent)
        assert res.get("status") == "REJECTED"
        assert "ExecutionGuard" in str(res.get("reason", "")), (
            f"entry must still be blocked under FLAT_ONLY: {res}")


# ---------------------------------------------------------------------------
# 3. AC-5 — exits exempt, entries still budget-blocked
# ---------------------------------------------------------------------------
class TestAC5ExitExemption:
    def _ac5_block(self) -> str:
        start = EXEC_SVC.index("[AC-5] Fill budget hard cap")
        end = EXEC_SVC.index("[AC-2] Anti-churn rate limiter")
        return EXEC_SVC[start:end]

    def test_gate_is_conditioned_on_entry_or_add(self):
        blk = self._ac5_block()
        assert re.search(r"if\s*\(\s*\(is_new_entry or is_adding\)", blk), (
            "AC-5 must gate entries/adds only — an unconditional budget gate "
            "blocks the 9th-fill forced exit and holds a loser overnight "
            "(P195 doctrine)")

    def test_entries_still_blocked_when_budget_exhausted(self):
        """Companion pin: the AC5_BUDGET_EXHAUSTED return still exists —
        the exemption did not delete the control."""
        blk = self._ac5_block()
        assert "AC5_BUDGET_EXHAUSTED" in blk


# ---------------------------------------------------------------------------
# 4. EXIT_ALPHA cannot downgrade a fired stop
# ---------------------------------------------------------------------------
class TestExitAlphaRespectsForceExecution:
    def test_block_guarded_on_force_execution(self):
        start = TICK_EXIT.index("EXIT ALPHA: Phase-aware scale-out")
        blk = TICK_EXIT[start:start + 1200]
        assert "not getattr(intent, 'force_execution', False)" in blk, (
            "a same-tick stop (target=0, force_execution=True) must never be "
            "overwritten into a partial scale-out")

    def test_the_overwrite_lines_still_exist_for_the_unforced_case(self):
        # The scale-out itself is legitimate when no stop fired.
        assert "intent.force_execution = False" in TICK_EXIT
        assert "intent.target_exposure = _new_exposure" in TICK_EXIT


# ---------------------------------------------------------------------------
# 5. [PNL_PROMO] retired — signal logged, authority never changed
# ---------------------------------------------------------------------------
class TestPnlPromoRetired:
    def test_no_promote_call_survives(self):
        # The only remaining `.promote(` in execution_service must be inside
        # a comment (the retirement record), never a call statement.
        for m in re.finditer(r"^(?P<line>.*\.promote\(.*)$", EXEC_SVC, re.M):
            line = m.group("line").lstrip()
            assert line.startswith("#"), (
                f"live promotion call found: {line!r} — [PNL_PROMO] must never "
                f"auto-promote (Rule 4; P287)")

    def test_ready_signal_logs_not_auto_promoting(self):
        assert "NOT " in EXEC_SVC and "auto-promoting (Rule 4; P287)" in EXEC_SVC

    def test_no_authority_sync_in_the_promo_block(self):
        start = EXEC_SVC_CODE.index("_c11_signal = ctx.pnl_attribution.get_promotion_signal()")
        blk = EXEC_SVC_CODE[start:start + 2000]
        assert "fn_sync_drl_authority" not in blk


# ---------------------------------------------------------------------------
# 6. Post-fill fuse equity reads guarded
# ---------------------------------------------------------------------------
class TestFuseEquityGuards:
    def test_all_three_sites_guarded(self):
        assert EXEC_SVC.count("[P287-FUSE]") == 3, (
            "each of the three post-fill fuse feeds (full exit, partial, flip) "
            "must have its own guard — a staleness RuntimeError must never "
            "abort execute_intent_v2 after a real venue fill")

    def test_guard_is_error_level_not_debug(self):
        # a skipped fuse record is a loss-forgiveness event; it must be loud
        idx = EXEC_SVC.index("[P287-FUSE]")
        preceding = EXEC_SVC[max(0, idx - 200):idx]
        assert "logger.error" in preceding


# ---------------------------------------------------------------------------
# 7. MAX_HOLD close definition aligned
# ---------------------------------------------------------------------------
class TestMaxHoldCloseDefinition:
    def test_close_definition_matches_branch_tree(self):
        assert ("_mh_is_close = (intent.target_exposure == 0 "
                "or intent.direction == 0)") in EXEC_SVC

    def test_both_branches_use_it(self):
        start = EXEC_SVC.index("_mh_is_close = ")
        blk = EXEC_SVC[start:start + 1600]
        assert "if not _mh_is_close and asset not in ctx.position_entry_times" in blk
        assert "elif _mh_is_close and asset in ctx.position_entry_times" in blk


# ---------------------------------------------------------------------------
# 8. EXEC-PREGATE balance probe off the event loop
# ---------------------------------------------------------------------------
class TestPregateToThread:
    def test_fetch_balance_wrapped(self):
        start = EXEC_SVC.index("[EXEC-PREGATE]") - 3000
        blk = EXEC_SVC[max(0, start):EXEC_SVC.index("[EXEC-PREGATE]") + 500]
        assert "to_thread" in blk
        assert re.search(r"^\s*_bal = ctx\.execution_manager\.exchange\.fetch_balance\(\)",
                         EXEC_SVC, re.M) is None, "the blocking inline call must be gone"


# ---------------------------------------------------------------------------
# 10/11/12 — behavioral: account state field, sentiment default, ctx dict
# ---------------------------------------------------------------------------
class TestAccountStateRename:
    def test_wrong_name_gone_and_honest_name_present(self):
        from core.account_sync import AccountState
        st = AccountState()
        assert hasattr(st, "non_quote_holdings_value")
        assert not hasattr(st, "used_margin"), (
            "`used_margin` was equity - available on a SPOT account — the "
            "value of non-quote holdings wearing a margin field's name")


class TestSentimentMockDefault:
    def test_bare_constructor_cannot_claim_real_without_a_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from core.sentiment_config import SentimentConfig
        cfg = SentimentConfig()
        assert cfg.IS_MOCK is True, (
            "no API key + bare constructor must resolve MOCK — the old "
            "`IS_MOCK: bool = False` default claimed REAL and could let a "
            "synthetic signal trigger OPPORTUNITY")

    def test_key_resolves_real(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-key")
        from core.sentiment_config import SentimentConfig
        assert SentimentConfig().IS_MOCK is False

    def test_explicit_is_mock_still_wins(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from core.sentiment_config import SentimentConfig
        assert SentimentConfig(IS_MOCK=False).IS_MOCK is False


class TestAdaptiveStopRegimeMultIsDict:
    def test_dataclass_default_supports_get(self):
        from core.execution_context import ExecutionContext
        ctx = ExecutionContext()
        assert ctx.adaptive_stop_regime_mult.get("ANY_REGIME", 1.0) == 1.0

    def test_build_fallback_is_a_dict_expression(self):
        src = (REPO / "core" / "execution_context.py").read_text(encoding="utf-8-sig")
        assert "getattr(runner, '_adaptive_stop_regime_mult', None) or {}" in src
        assert "getattr(runner, '_adaptive_stop_regime_mult', 1.0)" not in src


# ---------------------------------------------------------------------------
# 13 — dead fg_risk assignment deleted (exactly one live assignment)
# ---------------------------------------------------------------------------
class TestSmartBetaDeadAssignment:
    def test_single_fg_risk_assignment(self):
        src = (REPO / "core" / "smart_beta_controller.py").read_text(encoding="utf-8-sig")
        assigns = [l for l in src.splitlines()
                   if re.match(r"\s*fg_risk\s*=", l) and not l.lstrip().startswith("#")]
        assert len(assigns) == 1, f"expected exactly 1 fg_risk assignment, got {assigns}"


# ---------------------------------------------------------------------------
# 14 — [PROOF] fields can vary
# ---------------------------------------------------------------------------
class TestProofLogFieldsCanVary:
    def test_quant_real_uses_class_not_metaclass(self):
        assert "isinstance(self.fusion_engine, StubFusionEngine)" in INTEG_CODE
        assert "type(StubFusionEngine)" not in INTEG_CODE, (
            "isinstance(x, type(Cls)) tests against the METACLASS — always "
            "False for instances, so quant_real was the constant True")

    def test_quant_real_expression_semantics(self):
        from integration.integration_v36 import StubFusionEngine
        stub = StubFusionEngine()
        assert (not isinstance(stub, StubFusionEngine)) is False  # stub -> quant_real False
        real = object()
        assert (not isinstance(real, StubFusionEngine)) is True   # real -> quant_real True

    def test_dvol_real_has_a_producer(self):
        assert "self._last_dvol_real = market_data.get(\"dvol_zscore\") is not None" in INTEG
        assert "dvol_real=getattr(self, '_last_dvol_real', False)" in INTEG
        assert "dvol_real=hasattr(intent, 'dvol_zscore')" not in INTEG, (
            "no writer of intent.dvol_zscore exists — that expression was the "
            "constant False")


# ---------------------------------------------------------------------------
# 15 — SOL forced-exit enum coercion
# ---------------------------------------------------------------------------
class TestSolExitEnumCoercion:
    def test_coercion_present(self):
        start = INTEG.index("SOL IMMEDIATE EXIT") - 1500
        blk = INTEG[max(0, start):INTEG.index("SOL IMMEDIATE EXIT") + 200]
        assert "ExecutionMode(" in blk
        assert re.search(r"^\s*intent\.execution_mode = sol_exit_signal\.execution_mode\s*$",
                         INTEG, re.M) is None

    def test_string_values_coerce_to_members(self):
        from core.canonical_enums import ExecutionMode
        assert ExecutionMode("AGGRESSIVE") is ExecutionMode.AGGRESSIVE
        assert ExecutionMode("PASSIVE_PREFERRED") is ExecutionMode.PASSIVE_PREFERRED
        with pytest.raises(ValueError):
            ExecutionMode("NOT_A_MODE")


# ---------------------------------------------------------------------------
# 16 — FLASH_CRASH explicit disable + real protection still live
# ---------------------------------------------------------------------------
class TestFlashCrashExplicitDisable:
    def test_returns_zero_even_with_fabricated_in_window_history(self):
        """Old code with a 50% move inside the window scored 1.0; the honest
        state is a hard 0.0 — the checker is unreachable at the 4H cadence
        (each sample evicts the last; len(history) < 2 forever)."""
        from defense.constitution import NoTradeTriggerChecker
        c = NoTradeTriggerChecker()
        c._price_history["BTC"] = [
            (datetime.now(timezone.utc) - timedelta(seconds=10), 200.0)]
        assert c._check_flash_crash(100.0, "BTC", volume_ratio=2.0) == 0.0

    def test_disable_is_documented_at_the_site(self):
        assert "unreachable-by-cadence, disabled" in CONSTITUTION

    def test_real_flash_protection_path_still_exists(self):
        """The disable is only honest because the LIVE protection is
        elsewhere: the pipeline's bar-over-bar producer and the
        integration-level veto. If either disappears, this disable becomes
        a protection gap and must be revisited."""
        pipeline = (REPO / "data_mgmt" / "market_data_pipeline.py").read_text(
            encoding="utf-8-sig")
        assert "raw['flash_crash_active']" in pipeline
        assert 'market_data.get("flash_crash_active"' in INTEG

    def test_feed_disagreement_debounce_annotated(self):
        assert "DELIBERATELY UNUSED" in CONSTITUTION


# ---------------------------------------------------------------------------
# 17 — twin-implementation banners
# ---------------------------------------------------------------------------
class TestSignalsTwinBanners:
    @pytest.mark.parametrize("fname", ["no_trade_triggers.py", "opportunity_triggers.py"])
    def test_banner_present(self, fname):
        src = (REPO / "signals" / fname).read_text(encoding="utf-8-sig")
        assert "PARALLEL IMPLEMENTATION" in src
        assert "NOT the live decide path" in src

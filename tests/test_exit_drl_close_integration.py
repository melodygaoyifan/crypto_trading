"""
Tests that the per-trade outcome ledger gets a record_close() call on every
real position-close path through execution_service.py.

The bug we want to prevent: ledger.record_close has no callers (P15-style
half-wired feedback loop where the read side exists but the write side is
never fed). Each close path in execution_service.py / main.py needs to
call record_close so the promotion gate can count exit events.

Strategy: don't actually run execution_service end-to-end (too much surface
area to mock). Instead, statically verify the call sites exist by reading
the file and looking for the explicit `record_close(` invocation in the
expected branches.
"""
from pathlib import Path
import re


def test_full_exit_branch_calls_record_close():
    """BRANCH A (full exit) must flush the ledger record."""
    src = Path("core/execution_service.py").read_text(encoding="utf-8")
    # The full-exit branch has the marker `[FIX-P0-3] Feed real PnL into fuse rolling window`
    # then a `ctx.existence_fuse.record_pnl(... )` block. Right after that block
    # we expect the ledger close.
    branch_a = re.search(
        r"# \[FIX-P0-3\] Feed real PnL into fuse rolling window.*?"
        r"# \[EXIT-DRL\].*?_edrl_ledger\.record_close\(",
        src, re.DOTALL,
    )
    assert branch_a is not None, "BRANCH A (full exit) is missing the ledger.record_close() call"


def test_flip_close_branch_calls_record_close():
    """BRANCH C (flip close) must flush the ledger record for the closing trade."""
    src = Path("core/execution_service.py").read_text(encoding="utf-8")
    branch_c = re.search(
        r"_flip_net_pnl_usd.*?# \[EXIT-DRL\].*?_edrl_ledger_flip\.record_close\(",
        src, re.DOTALL,
    )
    assert branch_c is not None, "BRANCH C (flip close) is missing the ledger.record_close() call"


def test_partial_exit_branch_does_not_call_record_close():
    """BRANCH B (partial) must NOT close the ledger — position stays open."""
    src = Path("core/execution_service.py").read_text(encoding="utf-8")
    # BRANCH B is in the partial-exit block bounded by `_partial_net_pnl_usd`.
    # Find the partial-exit block (~50 lines after `_partial_net_pnl_usd =`).
    partial_match = re.search(
        r"_partial_net_pnl_usd = .*?(?=\n            ctx\.fn_record_realized_pnl_breakdown\()",
        src, re.DOTALL,
    )
    assert partial_match is not None, "couldn't locate partial-exit branch"
    partial_block = partial_match.group(0)
    # No record_close in the partial block
    assert "record_close(" not in partial_block, \
        "BRANCH B (partial exit) should not call record_close — position stays open"


def test_emergency_flatten_calls_record_close_in_paper_mode():
    """Paper-mode FORCE_FLAT clears positions; ledger needs to flush opens."""
    src = Path("main.py").read_text(encoding="utf-8-sig")
    # The paper-mode emergency flatten block:
    #   else: closed_assets = list(self._paper_positions.keys())
    #   ... record_close(...) ...
    #   self._paper_positions.clear()
    emerg_match = re.search(
        r"closed_assets = list\(self\._paper_positions\.keys\(\)\).*?"
        r"_edrl_emerg\.record_close\(.*?"
        r"self\._paper_positions\.clear\(\)",
        src, re.DOTALL,
    )
    assert emerg_match is not None, \
        "EMERGENCY_FLATTEN paper mode should flush ledger opens before clearing positions"


# ──────────────────────────────────────────────────────────────────────
# Validator histogram fix — per-step distribution, not just terminal
# ──────────────────────────────────────────────────────────────────────
def test_validator_outcome_carries_action_counts():
    """TradeOutcome must expose per-step action counts."""
    from training.exit_drl.validate_against_baseline import TradeOutcome
    out = TradeOutcome(
        entry_idx=0, exit_idx=10, direction=+1, pnl_bps=50.0,
        bars_held=10, final_action=3,
    )
    # Default factory produces 4-slot zero list
    assert out.action_counts == [0, 0, 0, 0]


def test_validator_aggregate_reports_step_action_pct():
    """Aggregate must include per-step histogram in addition to terminal."""
    from training.exit_drl.validate_against_baseline import TradeOutcome, aggregate
    # Two trades: one that HELD 8x then EXIT_ALL once, one that HELD 4x +
    # PARTIAL_EXIT 1x + HOLD 3x + EXIT_ALL 1x
    o1 = TradeOutcome(
        entry_idx=0, exit_idx=8, direction=+1, pnl_bps=20.0,
        bars_held=8, final_action=3, action_counts=[8, 0, 0, 1],
    )
    o2 = TradeOutcome(
        entry_idx=10, exit_idx=18, direction=-1, pnl_bps=-10.0,
        bars_held=8, final_action=3, action_counts=[7, 1, 0, 1],
    )
    metrics = aggregate([o1, o2])
    assert "step_action_pct" in metrics
    assert "total_steps" in metrics
    # 8+0+0+1 + 7+1+0+1 = 18 steps total
    assert metrics["total_steps"] == 18
    # HOLD: (8+7)/18 = 83.33%
    assert abs(metrics["step_action_pct"]["HOLD"] - 83.33) < 0.1
    # EXIT_ALL: 2/18 = 11.11%
    assert abs(metrics["step_action_pct"]["EXIT_ALL"] - 11.11) < 0.1
    # final_action_dist still present (terminal action only)
    assert metrics["final_action_dist"]["EXIT_ALL"] == 2
    assert metrics["final_action_dist"]["HOLD"] == 0

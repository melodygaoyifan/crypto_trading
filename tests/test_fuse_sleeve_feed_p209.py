"""[P209] The existence fuse, fed from the book that actually holds risk.

Non-Negotiable Rule #3 (28d rolling window, cumulative loss -> halt, manual
recovery) has been inert since Phase B. All three `record_pnl()` call sites live
in `core/execution_service.py`, PAST the P152 early return, so for a
Coinbase-routed asset they never execute. Live evidence: the fuse's persisted
`pnl_history` was a single June-12 record while the sleeve carried 100% of the
directional risk.

Two halves had to be true for this to be a real control, and only one was:

  * OUTPUT (already wired): a suspended fuse sets
    `veto_reason=[STRATEGY_SUSPENDED]`, which P206's translator classifies as
    neither a HOLD veto nor a venue-inapplicable one, so it falls through to
    `veto_flat` and the sleeve flattens. Pinned below, because P209's value
    depends entirely on it.
  * INPUT (this change): feed the sleeve's per-tick equity delta.

And a third thing that made both moot: `run_live()` never calls
`_save_paper_positions()`. The per-tick calls are all inside `run_paper()`; the
live-reachable ones each need a Kraken `_paper_positions` entry, and that dict
has been `{}` since 2026-06-13 — which is exactly the date the live file stopped
being written. Without persistence the 28d window resets on every deploy and the
fuse can never accumulate 28 days of anything. Same in-memory-baseline class as
P150 and P148.

The test that carries the most weight here is
`test_a_sustained_loss_actually_suspends`: a risk control that cannot fire is
the recurring defect this whole area keeps producing (P174/P175/P176/P192/P196),
so "it can fire" is asserted directly rather than inferred from wiring.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from defense.strategy_existence_fuse import (
    ExistenceFuseConfig,
    FuseState,
    StrategyExistenceFuse,
)

_MAIN = Path(__file__).resolve().parents[1] / "main.py"
_SRC = _MAIN.read_text(encoding="utf-8", errors="replace")


def _feed_block() -> str:
    """The [P209] fuse-feed + persist region of run_live().

    Delimited by its own end marker rather than a character count — a fixed
    window silently truncates when the comments grow and the assertions below
    then fail for a reason that has nothing to do with the contract.
    """
    i = _SRC.index("[P209] FEED THE EXISTENCE FUSE")
    j = _SRC.index("[P209] state persist failed", i)
    return _SRC[i:j]


def _feed_code() -> str:
    """`_feed_block()` with comment lines removed.

    A "this call must not appear" assertion has to read CODE. Checking the raw
    block matched the comment explaining why the call is absent — the assertion
    passed judgement on prose, so it would have failed on a correct
    implementation and passed on one that merely deleted the explanation.
    """
    return _code_only(_feed_block())


def _code_only(text: str) -> str:
    """Drop whole-line comments. See `_feed_code` for why this matters."""
    return "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )


def _fuse(**over) -> StrategyExistenceFuse:
    cfg = ExistenceFuseConfig(
        evaluation_interval_seconds=0.0,  # evaluate on every record
        min_data_points=3,
        **over,
    )
    return StrategyExistenceFuse(cfg)


# ---------------------------------------------------------------------------
# the control can actually fire
# ---------------------------------------------------------------------------

class TestTheFuseCanActuallyFire:

    def test_a_sustained_loss_actually_suspends(self):
        """The whole point. Deltas summing past `pnl_threshold_pct` over the
        window must SUSPEND — not merely be recorded."""
        f = _fuse()
        eq = 4000.0
        f.record_pnl(realized_pnl=0.0, current_equity=eq)  # anchor point
        for _ in range(6):
            eq -= 150.0                                    # -900 total = -22.5%
            f.record_pnl(realized_pnl=-150.0, current_equity=eq)
        assert f.get_status().state is FuseState.SUSPENDED, (
            "fed a 22% loss and the fuse stayed ACTIVE — the control cannot fire"
        )

    def test_a_flat_series_does_not_suspend(self):
        """The complement: it must not fire on nothing, or it is just noise."""
        f = _fuse()
        for _ in range(8):
            f.record_pnl(realized_pnl=0.0, current_equity=4000.0)
        assert f.get_status().state is FuseState.ACTIVE

    def test_a_profitable_series_does_not_suspend(self):
        f = _fuse()
        eq = 4000.0
        f.record_pnl(realized_pnl=0.0, current_equity=eq)
        for _ in range(6):
            eq += 100.0
            f.record_pnl(realized_pnl=+100.0, current_equity=eq)
        assert f.get_status().state is FuseState.ACTIVE


# ---------------------------------------------------------------------------
# delta semantics
# ---------------------------------------------------------------------------

class TestDeltaSemantics:

    def test_window_pnl_is_the_sum_of_deltas(self):
        """`_calculate_window_metrics` SUMS `realized_pnl`. So the feed must pass
        a per-tick DELTA. Passing cumulative PnL each tick would compound
        quadratically and suspend almost immediately on a tiny real loss."""
        f = _fuse()
        f.record_pnl(realized_pnl=0.0, current_equity=4000.0)
        f.record_pnl(realized_pnl=-50.0, current_equity=3950.0)
        f.record_pnl(realized_pnl=-50.0, current_equity=3900.0)
        st = f.get_status()
        assert st.window_pnl == pytest.approx(-100.0)
        assert st.window_pnl_pct == pytest.approx(-0.025)

    def test_the_feed_passes_a_delta_not_a_cumulative(self):
        blk = _feed_block()
        assert "_fz_delta = (" in blk and "_fz_eq - float(_fz_anchor)" in blk, (
            "feed must compute equity - anchor (a delta)"
        )
        assert "realized_pnl=_fz_delta" in blk

    def test_the_first_point_is_zero_not_retroactive(self):
        """Seeding the window with inception-to-date PnL would suspend on tick
        one for losses the fuse could never have acted on."""
        blk = _feed_block()
        assert "if _fz_anchor else 0.0" in blk
        f = _fuse()
        f.record_pnl(realized_pnl=0.0, current_equity=3772.0)
        assert f.get_status().window_pnl == pytest.approx(0.0)

    def test_the_anchor_advances_every_tick(self):
        blk = _feed_block()
        assert "self._fuse_sleeve_anchor_equity = _fz_eq" in blk, (
            "anchor never advances -> every delta measured from inception, so "
            "the same loss is re-counted on every tick"
        )


# ---------------------------------------------------------------------------
# a tick is not a trade
# ---------------------------------------------------------------------------

class TestTicksAreNotTrades:

    def test_the_feed_never_calls_on_trade_close(self):
        """`on_trade_close` counts a CONSECUTIVE-LOSS streak and suspends at 10.
        A 4H mark-to-market tick is not a trade: ten red ticks (~1.7 days of
        ordinary drift) would halt the system for no reason."""
        assert "on_trade_close" not in _feed_code(), (
            "feed calls on_trade_close — 10 consecutive red TICKS would suspend"
        )

    def test_ten_red_ticks_would_indeed_have_suspended_via_that_path(self):
        """Demonstrates the trap is real, not hypothetical."""
        f = _fuse()
        for _ in range(10):
            f.on_trade_close(-1.0)
        assert f.get_status().state is FuseState.SUSPENDED

    def test_trade_count_zero_is_passed(self):
        assert "trade_count=0" in _feed_block()


# ---------------------------------------------------------------------------
# stale data must not read as "no loss"
# ---------------------------------------------------------------------------

class TestStaleEquityIsSkipped:

    def test_feed_requires_a_fresh_reconcile(self):
        """`sleeve_equity_usd()` returns the last KNOWN value on API failure, so
        an unguarded delta would be 0.0 and enter the window as 'no loss'."""
        blk = _feed_block()
        assert "_reconcile_ok" in blk
        assert "_fz_eq <= 0" in blk
        i_guard = blk.index("if not _fz_fresh or _fz_eq <= 0:")
        i_rec = blk.index("_fz.record_pnl(")
        assert i_guard < i_rec, "freshness guard must precede record_pnl"

    def test_a_skip_records_nothing(self):
        """A gap in the series is honest; a fabricated zero is not."""
        blk = _feed_block()
        head = blk[blk.index("if not _fz_fresh"):blk.index("else:")]
        assert "record_pnl" not in head


# ---------------------------------------------------------------------------
# persistence — without it the window can never fill
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_run_live_persists_state_after_the_feed(self):
        blk = _feed_block()
        assert "_save_paper_positions(force=True)" in blk, (
            "run_live never persists: the 28d window resets on every deploy"
        )
        assert blk.index("_fz.record_pnl(") < blk.index("_save_paper_positions"), (
            "persist must follow the mutation it is persisting"
        )

    def test_the_bundle_carries_the_anchor_and_the_basis(self):
        assert '"fuse_sleeve_anchor_equity"' in _SRC
        assert '"existence_fuse_equity_basis"' in _SRC

    def test_the_restore_reads_both_back(self):
        i = _SRC.index("Restore existence fuse state")
        w = _SRC[i:_SRC.index("Restore cascade governor", i)]
        assert "self._fuse_sleeve_anchor_equity = (" in w
        assert 'data.get("fuse_sleeve_anchor_equity")' in w

    def test_fuse_history_survives_a_to_dict_from_dict_round_trip(self):
        f = _fuse()
        f.record_pnl(realized_pnl=0.0, current_equity=4000.0)
        f.record_pnl(realized_pnl=-25.0, current_equity=3975.0)
        restored = _fuse()
        restored.from_dict(f.to_dict())
        assert restored.get_status().window_pnl == pytest.approx(-25.0)

    def test_a_suspension_survives_the_round_trip(self):
        """A halt that forgets itself on restart is not a halt (P150's lesson)."""
        f = _fuse()
        f.force_suspend("test")
        restored = _fuse()
        restored.from_dict(f.to_dict())
        assert restored.is_suspended()


class TestLiveRestore:
    """Found by reading the live log after deploying the feed: the first record
    came back with `cumulative_pnl: 0.0` and `history: 1`, i.e. the previous
    state had NOT been read back. `_load_paper_positions()` is called only from
    run_paper(), so live persistence was write-only — the fuse started empty on
    every deploy and the 28d window could never fill. Persisting without
    restoring is the same non-control as not persisting.
    """

    def _live_restore_block(self) -> str:
        # Align to the start of the line so `_code_only` can recognise the
        # leading comment lines as comments (a mid-line slice makes the first
        # line look like code, which is how this test first failed).
        m = _SRC.index("[P209] RESTORE the existence-fuse series")
        i = _SRC.rindex("\n", 0, m) + 1
        return _SRC[i:_SRC.index("[P209] live fuse restore failed", m)]

    def test_run_live_restores_the_fuse(self):
        i = _SRC.index("async def run_live")
        j = _SRC.index("[P209] RESTORE the existence-fuse series")
        assert i < j, "restore must be inside run_live"
        blk = self._live_restore_block()
        assert "_fz0.from_dict(_fs0)" in blk

    def test_it_restores_the_anchor_with_the_history(self):
        """Restoring history without the anchor would measure the next delta
        against a fresh reading and silently drop everything that happened while
        the process was down."""
        blk = self._live_restore_block()
        assert '_d0.get("fuse_sleeve_anchor_equity")' in blk
        assert "self._fuse_sleeve_anchor_equity = float(_a0) if _a0 else None" in blk

    def test_it_does_not_repopulate_kraken_positions(self):
        """`_load_paper_positions()` would also restore `_paper_positions`, and
        resurrecting a Kraken book from a file in live is the P139/P140 failure
        shape. Scope is deliberately the fuse only."""
        code = _code_only(self._live_restore_block())
        assert "_load_paper_positions" not in code
        assert "self._paper_positions" not in code

    def test_a_restore_failure_is_non_fatal(self):
        """Live trading must not fail to start because a state file is corrupt
        (the P85 10-restart-loop lesson)."""
        i = _SRC.index("[P209] RESTORE the existence-fuse series")
        w = _SRC[i:i + 3000]
        assert "except Exception as _fz0_err:" in w
        assert "starting fresh" in w


# ---------------------------------------------------------------------------
# the capital-regime guard must not eat the history
# ---------------------------------------------------------------------------

class TestCapitalRegimeGuardCarveOut:

    def test_guard_is_skipped_for_a_sleeve_denominated_series(self):
        """sleeve ~$3.8k vs initial_capital $10k => ratio 2.65 > 2.0, so the
        guard would discard the history on EVERY restart and the window could
        never fill: an armed-looking fuse that can never fire."""
        i = _SRC.index("[P209] Fuse capital-regime guard skipped")
        w = _SRC[i - 1200:i + 900]
        assert '_basis == "coinbase_sleeve"' in w
        assert "_ratio = 1.0" in w

    def test_the_ratio_that_made_this_necessary(self):
        assert (10000.0 / 3772.0) > 2.0

    def test_the_guard_still_applies_to_other_bases(self):
        i = _SRC.index("[P209] Fuse capital-regime guard skipped")
        w = _SRC[i:i + 900]
        assert "if _ratio > 2.0:" in w, "guard removed outright rather than scoped"


# ---------------------------------------------------------------------------
# the output half this depends on
# ---------------------------------------------------------------------------

class TestSuspensionReachesTheSleeve:

    def test_strategy_suspended_translates_to_flat(self):
        """P209 is worthless unless suspension actually flattens the book."""
        import types

        from main import sleeve_direction_from_intent

        d, why = sleeve_direction_from_intent(
            types.SimpleNamespace(
                direction=+0.9, target_exposure=0.8, veto_active=True,
                veto_reason="[STRATEGY_SUSPENDED] Rolling 28d performance negative"),
            fallback_dir=+1.0)
        assert d == 0.0, "a suspended fuse did not flatten the sleeve"
        assert why.startswith("veto_flat:")

    def test_it_is_not_swallowed_as_a_hold_veto(self):
        from main import _SLEEVE_HOLD_VETOES, _SLEEVE_VENUE_NA_VETOES
        assert not any("SUSPEND" in v.upper() for v in _SLEEVE_HOLD_VETOES)
        assert not any("SUSPEND" in v.upper() for v in _SLEEVE_VENUE_NA_VETOES)


# ---------------------------------------------------------------------------
# window scoping
# ---------------------------------------------------------------------------

class TestWindowScoping:

    def test_records_older_than_the_window_drop_out(self):
        """Legacy Kraken-era records (equity ~$7.1k) must not contaminate a
        sleeve-denominated window (~$3.8k)."""
        f = _fuse()
        old = datetime.now(timezone.utc) - timedelta(days=40)
        f.record_pnl(realized_pnl=-20.0, current_equity=7148.0, timestamp=old)
        f.record_pnl(realized_pnl=0.0, current_equity=3772.0)
        st = f.get_status()
        assert st.window_start_equity == pytest.approx(3772.0), (
            "a 40-day-old record leaked into the 28d window"
        )

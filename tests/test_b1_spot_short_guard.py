"""
B1 / B1-EXT spot-short guard regression tests.

P140 (2026-06-12): a short-biased strategy ran at spot leverage (regime_leverage
=1). On a spot account a "short" intent degenerates into selling held coin, so
the bot churned dust-rejects and silently accumulated real spot LONGS while the
tracker recorded phantom SHORTS (-25% equity). B1 (main.py) converts such a short
intent to a HOLD.

B1-EXT (2026-06-13): the original B1 only fired when the asset was EXACTLY flat,
so a short ADD on an already-short position at spot leverage slipped through to
the order layer and dust-rejected. B1-EXT broadens the block to fire whenever we
are NOT reducing an existing LONG (flat OR already short), while preserving the
critical carve-out: a SELL that reduces a real LONG sells held coin and MUST stay
executable.

These tests lock both the source-level guard (so a refactor can't silently revert
to flat-only) and the intended truth table.
"""

from pathlib import Path

import pytest


# --- the B1-EXT decision predicate, mirrored from main.py for truth-table
# documentation. block == this short intent CANNOT be expressed on spot. ---
def b1_blocks(*, flag_on: bool, is_actionable: bool, veto_active: bool,
              intent_direction: float, regime_leverage: float,
              cur_dir: float, cur_exp: float) -> bool:
    if not (flag_on and is_actionable and not veto_active and intent_direction < 0):
        return False
    reducing_long = (cur_dir > 0.0 and cur_exp > 1e-9)
    return regime_leverage <= 1.0 and not reducing_long


class TestB1ExtTruthTable:
    def test_flat_spot_short_blocked(self):
        # original B1 case: flat + spot leverage + short -> HOLD
        assert b1_blocks(flag_on=True, is_actionable=True, veto_active=False,
                         intent_direction=-1.0, regime_leverage=1.0,
                         cur_dir=0.0, cur_exp=0.0)

    def test_already_short_add_on_spot_blocked(self):
        # B1-EXT: short ADD on an existing short at spot leverage -> HOLD
        assert b1_blocks(flag_on=True, is_actionable=True, veto_active=False,
                         intent_direction=-1.0, regime_leverage=1.0,
                         cur_dir=-1.0, cur_exp=0.05)

    def test_reducing_long_on_spot_allowed(self):
        # critical carve-out: SELL that reduces a real LONG must execute
        assert not b1_blocks(flag_on=True, is_actionable=True, veto_active=False,
                             intent_direction=-1.0, regime_leverage=1.0,
                             cur_dir=1.0, cur_exp=0.10)

    def test_margin_short_allowed(self):
        # leverage >= 2 (real margin) -> short is expressible -> never blocked
        assert not b1_blocks(flag_on=True, is_actionable=True, veto_active=False,
                             intent_direction=-1.0, regime_leverage=2.0,
                             cur_dir=0.0, cur_exp=0.0)

    def test_long_intent_never_blocked(self):
        assert not b1_blocks(flag_on=True, is_actionable=True, veto_active=False,
                             intent_direction=+1.0, regime_leverage=1.0,
                             cur_dir=0.0, cur_exp=0.0)

    def test_flag_off_disables_guard(self):
        assert not b1_blocks(flag_on=False, is_actionable=True, veto_active=False,
                             intent_direction=-1.0, regime_leverage=1.0,
                             cur_dir=0.0, cur_exp=0.0)

    def test_non_actionable_or_vetoed_skipped(self):
        assert not b1_blocks(flag_on=True, is_actionable=False, veto_active=False,
                             intent_direction=-1.0, regime_leverage=1.0,
                             cur_dir=0.0, cur_exp=0.0)
        assert not b1_blocks(flag_on=True, is_actionable=True, veto_active=True,
                             intent_direction=-1.0, regime_leverage=1.0,
                             cur_dir=0.0, cur_exp=0.0)


class TestB1ExtSourceGuard:
    """Source-level guard: main.py's B1 block must key off the
    not-reducing-long predicate, not the old flat-only check."""

    def _main_src(self) -> str:
        p = Path(__file__).resolve().parents[1] / "main.py"
        if not p.exists():
            pytest.skip("main.py not at expected location")
        return p.read_text(encoding="utf-8-sig")

    def test_block_uses_not_reducing_long(self):
        src = self._main_src()
        assert "_b1_reducing_long" in src, (
            "B1-EXT regression: main.py no longer computes _b1_reducing_long. "
            "A revert to flat-only lets short ADDs on spot dust-reject again."
        )
        assert "not _b1_reducing_long" in src, (
            "B1-EXT regression: the block condition must fire when NOT reducing "
            "a long (covers flat AND already-short)."
        )

    def test_long_reduce_carveout_preserved(self):
        # the carve-out must read BOTH direction>0 and exposure>0
        src = self._main_src()
        assert "_b1_cur_dir > 0.0 and _b1_cur_exp > 1e-9" in src, (
            "B1 safety carve-out missing: a SELL reducing a real LONG must stay "
            "executable (P140 invariant — never block reduces/exits)."
        )

    def test_block_reason_tag_present(self):
        assert "B1_SPOT_SHORT_BLOCK" in self._main_src()

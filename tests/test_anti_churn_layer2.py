"""
Layer 2 churn-control regression tests (2026-06-13).

Two changes, both targeting the ~75% of the Apr-Jun -25% loss that was execution
churn (median 12h holds on a 4H clock, ~45 direction-flips per asset):

  Change 1 — tightened anti-churn thresholds (main.py:~2105 constructor literals):
    ac1_min_hold_ticks 1->3 (4h->12h), ac2_max_global 6->3, ac5_max_per_day 8->4.
    These propagate AntiChurnManager -> runner._AC*_ mirrors -> ExecutionContext ->
    enforcement in core/execution_service.py (the manager's own check_* methods are
    dead; enforcement reads ctx.AC1_MIN_HOLD_TICKS etc.).

  Change 2 — flip-persistence whipsaw guard (main.py, intent level, near B1):
    a direction FLIP is suppressed until the opposing signal persists N consecutive
    ticks. Estimate-independent (the existing FLIP_GATE trusts the unreliable alpha
    estimate). Single-tick reversals just hold the position. Sim: persist=2 cut
    historical flips ~59%.
"""

from pathlib import Path

import pytest


# --- flip-persistence state machine, mirrored from main.py for truth-table + the
# historical-flip regression. Returns (suppress_flip: bool, new_state). ---
def flip_step(state, new_dir, cur_pos_dir, cur_pos_exp, need):
    """state = {"dir": int, "consec": int}; mutated and returned."""
    nd = 1 if new_dir > 0 else (-1 if new_dir < 0 else 0)
    if nd != 0:
        if nd == state["dir"]:
            state["consec"] += 1
        else:
            state["dir"] = nd
            state["consec"] = 1
    is_flip = (nd != 0 and cur_pos_dir != 0 and nd * cur_pos_dir < 0 and abs(cur_pos_exp) > 1e-9)
    suppress = is_flip and state["consec"] < max(1, need)
    return suppress, state


class TestFlipPersistenceTruthTable:
    def test_single_opposing_tick_suppressed(self):
        st = {"dir": 1, "consec": 3}  # was long-signalling
        # position is LONG, signal flips SHORT for the first time
        sup, st = flip_step(st, -1, cur_pos_dir=1, cur_pos_exp=0.1, need=2)
        assert sup is True and st["consec"] == 1

    def test_second_consecutive_opposing_allows_flip(self):
        st = {"dir": 1, "consec": 3}
        sup1, st = flip_step(st, -1, 1, 0.1, 2)   # tick 1: suppress
        sup2, st = flip_step(st, -1, 1, 0.1, 2)   # tick 2: opposing persisted -> allow
        assert sup1 is True and sup2 is False and st["consec"] == 2

    def test_intervening_same_dir_resets_counter(self):
        st = {"dir": 1, "consec": 3}
        flip_step(st, -1, 1, 0.1, 2)              # short once (consec=1)
        sup, st = flip_step(st, 1, 1, 0.1, 2)     # back to long -> reset
        assert st["dir"] == 1 and st["consec"] == 1
        # now a fresh short again is only 1 consecutive -> suppressed
        sup2, st = flip_step(st, -1, 1, 0.1, 2)
        assert sup2 is True

    def test_flat_position_never_a_flip(self):
        st = {"dir": 0, "consec": 0}
        sup, st = flip_step(st, -1, cur_pos_dir=0, cur_pos_exp=0.0, need=2)
        assert sup is False  # opening from flat is an entry, not a flip

    def test_same_direction_add_not_suppressed(self):
        st = {"dir": 1, "consec": 5}
        sup, st = flip_step(st, 1, cur_pos_dir=1, cur_pos_exp=0.1, need=2)
        assert sup is False

    def test_neutral_signal_not_suppressed(self):
        st = {"dir": 1, "consec": 5}
        sup, st = flip_step(st, 0, cur_pos_dir=1, cur_pos_exp=0.1, need=2)
        assert sup is False  # exit/neutral is not a flip


class TestLayer2SourceGuard:
    def _main_src(self):
        p = Path(__file__).resolve().parents[1] / "main.py"
        if not p.exists():
            pytest.skip("main.py not found")
        return p.read_text(encoding="utf-8-sig")

    def test_thresholds_tightened(self):
        src = self._main_src()
        assert "ac1_min_hold_ticks=3" in src, "Change 1 reverted: min-hold not 3 ticks"
        assert "ac2_max_global=3" in src, "Change 1 reverted: global cap not 3"
        assert "ac5_max_per_day=4" in src, "Change 1 reverted: daily budget not 4"

    def test_flip_persist_guard_present(self):
        src = self._main_src()
        assert "FLIP_PERSIST_HOLD" in src
        assert "flip_persistence_enabled" in src
        assert "_flip_persist" in src

    def test_flip_guard_only_suppresses_real_flips(self):
        # the guard's flip predicate must require an opposing live position
        src = self._main_src()
        assert "_fp_newdir * _fp_curdir < 0 and _fp_curexp > 1e-9" in src


class TestConfigPlumbing:
    def test_production_config_has_fields(self):
        import main
        cfg = main.ProductionConfig
        # dataclass defaults present
        assert "flip_persistence_enabled" in cfg.__dataclass_fields__
        assert "flip_persist_ticks" in cfg.__dataclass_fields__
        assert cfg.__dataclass_fields__["flip_persist_ticks"].default == 2


class TestHistoricalFlipReduction:
    """Replay a representative whipsaw sequence and assert persist=2 reduces
    committed flips vs persist=1 (locks the mechanism to the forensic finding)."""

    def _commit_flips(self, dirs, need):
        """Simulate position evolution: count committed flips under the guard."""
        state = {"dir": 0, "consec": 0}
        pos_dir = 0
        flips = 0
        for d in dirs:
            sup, state = flip_step(state, d, pos_dir, 1.0 if pos_dir != 0 else 0.0, need)
            nd = 1 if d > 0 else (-1 if d < 0 else 0)
            if nd == 0:
                continue
            if pos_dir == 0:
                pos_dir = nd          # entry from flat
            elif nd != pos_dir and not sup:
                flips += 1            # committed flip
                pos_dir = nd
            # suppressed flip or same-dir add: position direction unchanged
        return flips

    def test_persist2_reduces_flips_vs_persist1(self):
        # whipsaw: alternating with occasional 2-in-a-row genuine reversals
        seq = [1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1]
        f1 = self._commit_flips(seq, need=1)
        f2 = self._commit_flips(seq, need=2)
        assert f2 < f1, f"persist=2 ({f2}) should commit fewer flips than persist=1 ({f1})"

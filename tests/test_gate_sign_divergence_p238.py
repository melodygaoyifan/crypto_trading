"""[P238] Gate-vs-trade sign divergence instrumentation (task-#5 step b).

The alpha gate (STEP 7) judges ``_alpha_input_direction`` (effective_alpha/
quant), while the traded direction is assigned from fusion 165 lines later
(STEP 8). Every sign-dependent gate branch — short x0.80 discount,
``min_alpha_bps_short``, quiet-accum short floor, short epsilon — can
therefore run for the opposite side of what trades. Before designing any
gate re-check, the live rate of that divergence must be MEASURED — this
ships the measurement and nothing else.

Pinned contracts:
  - the predicate is pure and behaviorally tested (P234 lesson);
  - the counter block is LOG-ONLY: it may read the intent, never write it;
  - it sits AFTER fusion assigns intent.direction (else it re-creates the
    P234 bug — comparing against a not-yet-assigned field);
  - new self attributes are getattr-defended (P85).
"""

import re
from pathlib import Path

from integration.integration_v36 import alpha_gate_sign_diverges as diverges

REPO = Path(__file__).resolve().parents[1]
V36 = (REPO / "integration" / "integration_v36.py").read_text(
    encoding="utf-8-sig", errors="replace")


class TestPredicate:
    def test_opposite_signs_diverge(self):
        assert diverges(0.4, -0.2)
        assert diverges(-0.4, 0.2)

    def test_same_sign_does_not(self):
        assert not diverges(0.4, 0.2)
        assert not diverges(-0.4, -0.2)

    def test_flat_on_either_side_is_not_a_divergence(self):
        """A flat signal has no sign to disagree with — counting it would
        inflate the rate with ticks where no sign-dependent branch mattered."""
        assert not diverges(0.0, -0.5)
        assert not diverges(0.5, 0.0)
        assert not diverges(0.0, 0.0)
        assert not diverges(1e-12, -0.5)  # numerical dust = flat

    def test_none_is_flat_not_a_crash(self):
        assert not diverges(None, -0.5)
        assert not diverges(0.5, None)


class TestWiring:
    def _block(self):
        i = V36.find("[P238] Gate-vs-trade sign divergence counter")
        assert i > 0, "the counter block is gone"
        return V36[i:V36.find("failure memory aggressiveness", i)]

    def test_counter_runs_after_fusion_assigns_direction(self):
        """The whole point: the comparison must read the ASSIGNED traded
        direction. Placing it before the assignment re-creates P234."""
        assign = V36.find("intent.direction = fusion_result.direction")
        block = V36.find("[P238] Gate-vs-trade sign divergence counter")
        assert 0 < assign < block

    def test_block_is_log_only(self):
        """No intent writes inside the block — this is instrumentation, not
        a control."""
        blk = self._block()
        assert not re.search(r"intent\.\w+\s*=", blk), (
            "P238 block writes to the intent — it must stay log-only until "
            "the measured rate justifies a designed re-check"
        )
        assert "alpha_gate_sign_diverges(" in blk
        assert "_alpha_input_direction" in blk

    def test_counters_are_getattr_defended(self):
        """P85: a new attribute read on the tick path must default safely —
        the first tick of every process has neither counter."""
        blk = self._block()
        assert blk.count('getattr(') >= 2
        assert '"_gate_sign_decide_count", 0)' in blk
        assert '"_gate_sign_diverge_count", 0)' in blk

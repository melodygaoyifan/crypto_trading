"""[P236] model_alpha disagreement filter on the sleeve driver.

Evidence (2026-08-08 live counterfactual, 30d attribution logs, 16h horizon,
overlap-corrected t): quant earns +24.9bps/tick when model_alpha agrees and
-78.9bps/tick (t=-3.42) when it disagrees; model_alpha standalone
+49.9bps/tick (t=2.91), positive on all three assets. This ships the split as
a SHADOW ledger (data/strategy_shadow/ma_filter_*.jsonl, scored by the P166
gate) plus a default-OFF enforcement flag (`coinbase_ma_filter_enforce`).

Contracts pinned here:
  - the decision is pure and behaviorally tested (the P234 lesson: a
    source-substring pin passes on dead code);
  - exits/reduces are NEVER filtered (P195) and a silent model_alpha always
    allows (P208: a dead agent must not become a trading stop);
  - the ledger's confidence never saturates on a zero direction (P224);
  - the prefix is registered in compute_shadow_ic at BOTH default sites
    (P192: an allowlist spanning two files needs a gate reading both);
  - the flag is declared AND parsed (P201) and absent from the live profile.
"""

import json
import re
from pathlib import Path

import pytest

from main import sleeve_ma_filter_decision as decide

REPO = Path(__file__).resolve().parents[1]
MAIN = (REPO / "main.py").read_text(encoding="utf-8-sig", errors="replace")
SHADOW_IC = (REPO / "analytics" / "shadow_ic" / "compute_shadow_ic.py"
             ).read_text(encoding="utf-8-sig", errors="replace")


class TestPureDecision:
    def test_agreement_allows(self):
        assert decide(0, 1, 0.3) == (1, "", "ma_agrees")
        assert decide(-1, -1, -0.2) == (-1, "", "ma_agrees")

    def test_silence_allows_fail_open(self):
        """A model_alpha of 0.0 — absent, aborted, or genuinely neutral — must
        never block (P208)."""
        led, act, why = decide(0, 1, 0.0)
        assert (led, act) == (1, "") and why == "ma_silent"
        led, act, why = decide(0, -1, 1e-12)  # numerical dust = silence
        assert (led, act) == (-1, "") and why == "ma_silent"

    def test_exits_and_holds_are_never_filtered(self):
        """raw_target == 0 is an exit/flatten/hold intent — the filter must
        not touch it whatever model_alpha says (P195)."""
        for pos in (-1, 0, 1):
            for ma in (-0.9, 0.0, 0.9):
                led, act, why = decide(pos, 0, ma)
                assert led == 0 and act == "" and why == "no_target"

    def test_disagreeing_entry_from_flat_is_blocked(self):
        led, act, why = decide(0, 1, -0.3)
        assert led == 0 and act == "block_entry"
        led, act, why = decide(0, -1, 0.3)  # short side symmetric
        assert led == 0 and act == "block_entry"

    def test_disagreeing_flip_demotes_to_flatten(self):
        """The closing leg is a reduce and always free; only the OPENING leg
        of the flip is suppressed."""
        led, act, why = decide(1, -1, 0.3)
        assert led == 0 and act == "flip_to_flat"
        led, act, why = decide(-1, 1, -0.3)
        assert led == 0 and act == "flip_to_flat"

    def test_aligned_held_position_is_kept_but_ledger_zeroes(self):
        """v1 scope: the filter is an ENTRY filter — it never force-exits an
        aligned held position (that would be its own churn engine). The
        shadow claim still zeroes, so the IC gate scores the full strategy."""
        led, act, why = decide(1, 1, -0.5)
        assert led == 0 and act == "" and why == "ma_disagrees_hold_kept"
        led, act, why = decide(-1, -1, 0.5)
        assert led == 0 and act == "" and why == "ma_disagrees_hold_kept"

    def test_no_action_grid_without_disagreement(self):
        """Enforcement may only ever fire on a genuine disagreement."""
        for pos in (-1, 0, 1):
            for raw in (-1, 0, 1):
                for ma in (0.0,) + ((0.4,) if raw >= 0 else (-0.4,)):
                    _, act, _ = decide(pos, raw, ma)
                    if raw == 0 or ma == 0.0 or ma * raw > 0:
                        assert act == "", (pos, raw, ma)


class TestConfigContract:
    def test_declared_default_off(self):
        assert re.search(
            r"^\s+coinbase_ma_filter_enforce: bool = False", MAIN, re.M)

    def test_parsed_in_from_file(self):
        """P201: a flag read via getattr but never parsed silently no-ops."""
        assert 'data.get("coinbase_ma_filter_enforce"' in MAIN

    def test_explicit_false_in_live_profile(self):
        # [P253] was "absent from the profile" — the flag's state was
        # invisible from the production config (the one sleeve flag with no
        # key), so it is now declared EXPLICITLY at its live value. The
        # guard's intent is unchanged and stronger: enforcement must not
        # silently flip on.
        live = json.loads((REPO / "configs" / "live_high_risk.json"
                           ).read_text(encoding="utf-8-sig"))
        assert live.get("coinbase_ma_filter_enforce") is False, (
            "coinbase_ma_filter_enforce is not explicitly false in the live "
            "profile — flipping it on is an operator decision requiring P166 "
            "forward evidence from the ma_filter ledger + its own P-entry"
        )


class TestHarness:
    def _observe(self, tmp_path, payload):
        from defense.strategy_shadow_v5_1 import build_ma_filter_shadow_harness
        h = build_ma_filter_shadow_harness(log_dir=tmp_path)
        h.observe("BTC", payload)
        files = list(tmp_path.glob("ma_filter_*.jsonl"))
        assert len(files) == 1
        return json.loads(files[0].read_text(encoding="utf-8").strip())

    def test_record_echoes_the_driver_decision(self, tmp_path):
        rec = self._observe(tmp_path, {
            "_maf_ledger_dir": 0.0, "_maf_ma_dir": -0.42,
            "_maf_raw_target": 1, "_maf_sleeve_dir": 0.87,
            "_maf_pos": 0, "_maf_action": "block_entry",
            "_maf_reason": "ma_disagrees_entry", "_maf_enforce": False,
        })
        assert rec["strategy"] == "ma_filtered"
        assert rec["asset"] == "BTC"
        assert rec["direction"] == 0.0
        assert rec["reason"] == "ma_disagrees_entry"
        assert rec["diagnostics"]["action"] == "block_entry"
        assert rec["diagnostics"]["pos_contracts"] == 0
        # a flat claim carries 0 confidence (P224); the vetoing opinion's
        # strength lives in diagnostics.ma_dir
        assert rec["confidence"] == 0.0
        assert rec["diagnostics"]["ma_dir"] == pytest.approx(-0.42)

    def test_directional_claim_carries_full_confidence(self, tmp_path):
        """compute_shadow_ic scores x = direction * confidence. A pass-through
        claim (ma silent or agreeing) scored at conf < 1 is silently
        down-weighted out of the IC — the scored series then measures 'quant
        when model_alpha agrees', not the filtered book. Full confidence on
        every directional claim keeps x == the claim."""
        rec = self._observe(tmp_path, {
            "_maf_ledger_dir": 1.0, "_maf_ma_dir": 0.0,
            "_maf_raw_target": 1, "_maf_sleeve_dir": 0.9,
            "_maf_pos": 1, "_maf_action": "",
            "_maf_reason": "ma_silent", "_maf_enforce": False,
        })
        assert rec["direction"] == 1.0
        assert rec["confidence"] == 1.0

    def test_confidence_never_saturates_on_a_flat_non_veto_claim(self, tmp_path):
        """P224: a confident non-signal is worse than a missing one. A flat
        claim with no model_alpha opinion behind it must carry 0."""
        rec = self._observe(tmp_path, {
            "_maf_ledger_dir": 0.0, "_maf_ma_dir": 0.0,
            "_maf_raw_target": 0, "_maf_sleeve_dir": 0.02,
            "_maf_pos": 0, "_maf_action": "",
            "_maf_reason": "no_target", "_maf_enforce": False,
        })
        assert rec["direction"] == 0.0
        assert rec["confidence"] == 0.0


class TestRegistration:
    def test_prefix_registered_at_both_default_sites(self):
        """P192: the loader default and the argparse default are two files'
        worth of allowlist in one file — both must carry the prefix or the
        CLI run silently skips the ledger."""
        assert SHADOW_IC.count('"ma_filter"') >= 1
        assert re.search(
            r'prefixes: Tuple\[str, \.\.\.\] = \([^)]*"ma_filter"', SHADOW_IC)
        assert re.search(
            r'default="[^"]*,ma_filter"', SHADOW_IC)


class TestDriverWiring:
    """Source pins on the sleeve loop — the pure function is behaviorally
    tested above; these pin that the loop actually consults it (the P234
    lesson applied in the direction that is checkable: wiring, not logic)."""

    def _loop_block(self):
        i = MAIN.find("[P236] model_alpha disagreement\n")
        start = MAIN.find("sleeve_ma_filter_decision(", i)
        assert start > 0, "the sleeve loop no longer calls the filter"
        return MAIN[i:MAIN.find("Re-entry cooldown (P168's", i)]

    def test_filter_runs_before_the_cooldown(self):
        blk = self._loop_block()
        assert "sleeve_ma_filter_decision(" in blk
        assert "_ma_filter_shadow.observe" in blk

    def test_blocked_entry_still_reconciles_the_stop(self):
        """The skip path must pass intended_target=0 (P207: intent beats a
        snapshot taken after an order) before continuing."""
        blk = self._loop_block()
        i = blk.find('"block_entry"')
        assert i > 0
        tail = blk[i:]
        assert "ensure_protective_stop" in tail
        assert "intended_target=0" in tail
        assert "continue" in tail

    def test_flip_demotion_flattens_not_reverses(self):
        blk = self._loop_block()
        i = blk.find('"flip_to_flat"')
        assert i > 0
        assert "_m_dir = 0.0" in blk[i:]

    def test_stash_resets_before_the_agent_block_and_writes_after(self):
        """The stash must be a CURRENT reading (P155-L5, not a high-water
        mark) and a dead agent must read as 0.0 (P208 fail-open)."""
        reset = MAIN.find('self._last_model_alpha_directions[asset] = 0.0')
        write = MAIN.find(
            'self._last_model_alpha_directions[asset] = float(')
        agent_block = MAIN.find("if self.model_alpha is not None "
                                "and not p0_abort_tick:")
        assert 0 < reset < agent_block < write, (
            "stash ordering broken: reset must precede the model_alpha "
            "block, the real write must sit inside/after it"
        )

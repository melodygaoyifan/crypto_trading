"""[P422] Two controls that stopped a certified book from trading, re-set.

1. The SOTA risk controller's EXIT-ONLY mode ("no NEW entries") reached the
   sleeve as "[P0_SAFETY] Exit-only mode: correlation=..." and that prefix
   sits in the FLATTEN roster — so it LIQUIDATED held positions (the ETH
   2026-09-25 long was sold the next tick on it). It is now ENTRY-BLOCKED:
   a flat book stays flat, a held agreeing position survives, a flip
   closes the abandoned leg (the P341 sentinel). The other [P0_SAFETY]
   reasons (kill switch, HALTED drawdown) still flatten.
2. The BitBeast SNR/Volume/Structure entry guard gained a mode. "enforce"
   is byte-identical to before; "shadow" logs WOULD-BLOCK and vetoes
   nothing. The live profile is pinned to the DECIDED value.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

import main as m

REPO = Path(__file__).resolve().parents[1]


def _intent(direction=0.0, target_exposure=0.0, veto_active=False, veto_reason=""):
    return types.SimpleNamespace(direction=direction, target_exposure=target_exposure,
                                 veto_active=veto_active, veto_reason=veto_reason)


EXIT_ONLY = "[P0_SAFETY] Exit-only mode: correlation=RECOVERING"
CRISIS_NO_NEW = "[P0_SAFETY] Correlation crisis detected - no new positions"


class TestExitOnlyIsEntryBlockedNotFlatten:
    @pytest.mark.parametrize("reason", [EXIT_ONLY,
                                        "[P0_SAFETY] Exit-only mode: correlation=DECOUPLING",
                                        CRISIS_NO_NEW])
    def test_translator_classifies_it_entry_blocked(self, reason):
        tgt, why = m.sleeve_direction_from_intent(
            _intent(direction=0.5, target_exposure=0.015, veto_active=True, veto_reason=reason),
            fallback_dir=1.0)
        assert tgt is m.SLEEVE_ENTRY_BLOCKED, why
        assert why.startswith("entry_quality_veto:")

    def test_the_eth_0926_case_holds_the_position(self):
        # the live record: ETH held 6ct long, book still +0.5, gate passed,
        # veto = exit-only mode. Pre-P422 this resolved to a 6ct SELL.
        tgt, _ = m.sleeve_direction_from_intent(
            _intent(direction=0.5, target_exposure=0.015, veto_active=True, veto_reason=EXIT_ONLY),
            fallback_dir=1.0)
        resolved, why = m.sleeve_entry_blocked_resolve(6.0, 0.5, "x")
        assert tgt is m.SLEEVE_ENTRY_BLOCKED and resolved is m.SLEEVE_HOLD, why

    def test_from_flat_it_still_blocks_the_entry(self):
        resolved, why = m.sleeve_entry_blocked_resolve(0.0, 1.0, "x")
        assert resolved is m.SLEEVE_HOLD and "entry_blocked_flat" in why

    def test_a_flip_still_closes_the_abandoned_leg(self):
        resolved, _ = m.sleeve_entry_blocked_resolve(6.0, -1.0, "x")
        assert resolved == 0.0

    @pytest.mark.parametrize("reason", ["[P0_SAFETY] Kill-switch active: correlation_crisis",
                                        "[P0_SAFETY] Trading halted: drawdown=26.00%",
                                        "[P0_SAFETY] System in KILL_SWITCH state"])
    def test_the_other_p0_reasons_still_flatten(self, reason):
        tgt, why = m.sleeve_direction_from_intent(
            _intent(direction=0.5, target_exposure=0.015, veto_active=True, veto_reason=reason),
            fallback_dir=1.0)
        assert tgt == 0.0 and why.startswith("veto_flat"), why

    def test_roster_members_are_the_controllers_own_literals(self):
        src = (REPO / "risk" / "sota_risk_controller.py").read_text(encoding="utf-8")
        assert "Exit-only mode: correlation=" in src
        assert "Correlation crisis detected - no new positions" in src


class TestBitBeastVetoMode:
    def test_enforce_vetoes_a_refusal(self):
        assert m.bitbeast_veto_apply("enforce", allowed=False) is True

    def test_shadow_never_vetoes(self):
        assert m.bitbeast_veto_apply("shadow", allowed=False) is False

    def test_an_allowed_entry_is_never_vetoed_in_any_mode(self):
        assert m.bitbeast_veto_apply("enforce", allowed=True) is False
        assert m.bitbeast_veto_apply("shadow", allowed=True) is False

    @pytest.mark.parametrize("mode", ["", None, "SHADOWW", "off", "observe"])
    def test_an_unrecognised_mode_reads_as_enforce(self, mode):
        # a typo must not disarm a guard
        assert m.bitbeast_veto_apply(mode, allowed=False) is True

    def test_config_default_is_enforce_and_parse_honours_shadow(self, tmp_path):
        assert m.ProductionConfig.__dataclass_fields__["bitbeast_veto_mode"].default == "enforce"
        base = json.loads((REPO / "configs" / "live_high_risk.json").read_text(encoding="utf-8-sig"))
        base["bitbeast_veto_mode"] = "ShAdOw"
        p = tmp_path / "c.json"
        p.write_text(json.dumps(base), encoding="utf-8")
        assert m.ProductionConfig.from_file(p).bitbeast_veto_mode == "shadow"
        base["bitbeast_veto_mode"] = "nonsense"
        p.write_text(json.dumps(base), encoding="utf-8")
        assert m.ProductionConfig.from_file(p).bitbeast_veto_mode == "enforce"
        del base["bitbeast_veto_mode"]
        p.write_text(json.dumps(base), encoding="utf-8")
        assert m.ProductionConfig.from_file(p).bitbeast_veto_mode == "enforce"

    def test_live_profile_is_the_decided_value(self):
        # [P237] a silent revert to enforce (and a silent deletion) both fail:
        # either direction is a live behaviour change that must be argued for.
        cfg = json.loads((REPO / "configs" / "live_high_risk.json").read_text(encoding="utf-8-sig"))
        assert cfg.get("bitbeast_veto_mode") == "shadow"
        assert "_bitbeast_veto_mode_note" in cfg

    def test_the_block_routes_through_the_helper(self):
        import inspect
        src = inspect.getsource(m.HMATSProductionRunner._process_4h_tick_inner)
        i = src.index("_vg_allowed, _vg_reasons = _veto_guard.check_entry(")
        blk = src[i:i + 1500]
        assert "bitbeast_veto_apply(" in blk and "[VC-9-SHADOW]" in blk

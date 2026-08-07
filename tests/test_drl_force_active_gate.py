"""[P198] The DRL promotion gate's persisted level must be able to stick.

Since 2026-04-11, a [DRL_FORCE_ACTIVE] block in main.py re-promoted ANY
persisted authority level to ACTIVE on every boot whenever models loaded —
so the promotion gate's demotion machinery (auto-demotion, manual demote,
rule #4) was cosmetic across restarts. P198 gates that block on
drl.force_active (default True = historical behavior), and the live config
sets it false so the 2026-08-07 ACTIVE -> SHADOW demotion actually holds.

These are wiring/pin tests (P152 shape: a config knob that nothing reads is
not a knob).
"""

import io
import json
import re

import pytest


def _main_src():
    return io.open("main.py", encoding="utf-8").read()


def test_force_active_is_gated_on_config():
    src = _main_src()
    assert "_drl_force_active = bool(_drl_cfg.get('force_active', True))" in src, (
        "the force_active read disappeared from main.py — [DRL_FORCE_ACTIVE] "
        "is unconditional again and any demotion is silently re-promoted to "
        "ACTIVE on the next boot"
    )
    # the promote("ACTIVE") inside the force block must be reachable only
    # under the flag
    m = re.search(
        r"if \(self\._drl_models_ready > 0\s*\n\s*and self\._drl_authority_level != \"ACTIVE\"\s*\n\s*and _drl_force_active\):",
        src,
    )
    assert m, (
        "the [DRL_FORCE_ACTIVE] branch is no longer conditioned on "
        "_drl_force_active"
    )


def test_default_preserves_historical_behavior():
    """force_active must DEFAULT to True: a config with no `drl` section
    (every paper/verify profile) keeps the 2026-04-11 behavior. Only an
    explicit false changes anything."""
    src = _main_src()
    assert ".get('force_active', True)" in src


def test_live_config_pins_the_demotion():
    cfg = json.load(io.open("configs/live_high_risk.json", encoding="utf-8"))
    drl = cfg.get("drl") or {}
    assert drl.get("force_active") is False, (
        "configs/live_high_risk.json no longer sets drl.force_active=false — "
        "the next deploy will force the demoted DRL back to ACTIVE at boot"
    )


def test_fusion_admits_drl_only_at_exit_only_or_active():
    """The demotion's semantics rest on fusion excluding a SHADOW-level DRL.
    If this admission condition ever loosens, SHADOW stops meaning 'runs
    inference, does not vote' and the P198 demotion silently stops demoting."""
    src = io.open("integration/integration_v36.py", encoding="utf-8").read()
    m = re.search(
        r"_drl_admitted = \(\s*\n\s*_drl_level in \(\"EXIT_ONLY\", \"ACTIVE\"\)",
        src,
    )
    assert m, (
        "integration_v36's _drl_admitted no longer requires "
        "EXIT_ONLY/ACTIVE — verify what SHADOW now means in fusion before "
        "trusting the P198 demotion"
    )


def test_gate_shadow_is_a_valid_level():
    from drl.promotion_gate import DRLPromotionGate
    assert "SHADOW" in DRLPromotionGate.VALID_LEVELS

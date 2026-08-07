"""
Tests for the accelerated-promotion components: kill switch, per-asset mode,
override audit stamp, and the static guards that wire the System-3 bridge into
core/tick_exit_triggers.py.

This is the regression suite for CLAUDE.md P29.
"""
import json
import re
import time
from pathlib import Path

import pytest

from agents.exit_drl_agent import ExitDRLMode
from risk.exit_drl_kill_switch import (
    CONSECUTIVE_LOSSES_LIMIT, EXIT_ALL_RATIO_MAX, HOLD_RATIO_MAX,
    HOLD_RATIO_MIN, ROLLING_PNL_WINDOW_DAYS,
    ExitDRLKillSwitch, get_kill_switch, reset_singleton,
)

# [TEST-CLEANUP 2026-06-09] ExitDRLKillSwitch.should_demote() permanently
# returns None per operator directive 2026-04-30 (risk/exit_drl_kill_switch.py:226).
# Trip conditions still record_close/record_step_action for diagnostics
# but never recommend demotion (Kraken outages produce spurious trips).
# Below applied to the 4 trip-condition tests; the recording-only tests
# (state persistence, record_close mechanics) still pass and stay live.
_KILLSWITCH_DEMOTION_DISABLED = pytest.mark.skip(
    reason="[TEST-CLEANUP] should_demote() permanently returns None per "
           "operator directive 2026-04-30; record_close still tracks state."
)


# ────────────────────────────────────────────────────────────────
# Kill switch
# ────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _reset():
    reset_singleton()
    yield
    reset_singleton()


def test_killswitch_no_demote_before_promotion(tmp_path):
    ks = ExitDRLKillSwitch(state_path=str(tmp_path / "ks.jsonl"))
    # Without record_promotion → no demotion ever (asset isn't promoted)
    for _ in range(20):
        ks.record_close("BTC", -100.0, "EXIT_ALL", was_drl_driven=True)
    assert ks.should_demote("BTC") is None


@_KILLSWITCH_DEMOTION_DISABLED
def test_killswitch_consecutive_losses_trip(tmp_path):
    ks = ExitDRLKillSwitch(state_path=str(tmp_path / "ks.jsonl"))
    ks.record_promotion("BTC")
    for _ in range(CONSECUTIVE_LOSSES_LIMIT - 1):
        ks.record_close("BTC", -50.0, "EXIT_ALL", was_drl_driven=True)
    assert ks.should_demote("BTC") is None  # one short
    ks.record_close("BTC", -50.0, "EXIT_ALL", was_drl_driven=True)
    reason = ks.should_demote("BTC")
    assert reason is not None
    assert "CONSECUTIVE_LOSSES" in reason


def test_killswitch_winning_close_resets_consecutive(tmp_path):
    ks = ExitDRLKillSwitch(state_path=str(tmp_path / "ks.jsonl"))
    ks.record_promotion("BTC")
    for _ in range(4):
        ks.record_close("BTC", -50.0, "EXIT_ALL", was_drl_driven=True)
    # One winner resets
    ks.record_close("BTC", +30.0, "EXIT_ALL", was_drl_driven=True)
    assert ks.should_demote("BTC") is None


def test_killswitch_non_drl_driven_close_does_not_count(tmp_path):
    ks = ExitDRLKillSwitch(state_path=str(tmp_path / "ks.jsonl"))
    ks.record_promotion("BTC")
    # Stop-loss / phase / momentum closes are NOT DRL-driven
    for _ in range(20):
        ks.record_close("BTC", -100.0, "EXIT_ALL", was_drl_driven=False)
    # Consecutive-loss + rolling-PnL guards never tripped
    reason = ks.should_demote("BTC")
    assert reason is None or "CONSECUTIVE_LOSSES" not in reason


@_KILLSWITCH_DEMOTION_DISABLED
def test_killswitch_hold_ratio_low_trip(tmp_path):
    ks = ExitDRLKillSwitch(state_path=str(tmp_path / "ks.jsonl"))
    ks.record_promotion("BTC")
    # Force HOLD ratio below 50% by feeding 50 PARTIAL_EXIT actions
    for _ in range(50):
        ks.record_step_action("BTC", "PARTIAL_EXIT")
    reason = ks.should_demote("BTC")
    assert reason is not None
    assert "HOLD_RATIO_LOW" in reason


@_KILLSWITCH_DEMOTION_DISABLED
def test_killswitch_hold_ratio_high_trip(tmp_path):
    ks = ExitDRLKillSwitch(state_path=str(tmp_path / "ks.jsonl"))
    ks.record_promotion("BTC")
    for _ in range(50):
        ks.record_step_action("BTC", "HOLD")
    reason = ks.should_demote("BTC")
    assert reason is not None
    assert "HOLD_RATIO_HIGH" in reason


@_KILLSWITCH_DEMOTION_DISABLED
def test_killswitch_exit_all_high_trip(tmp_path):
    ks = ExitDRLKillSwitch(state_path=str(tmp_path / "ks.jsonl"))
    ks.record_promotion("BTC")
    # 50 closes, 35 EXIT_ALL = 70% > 30% threshold
    for _ in range(35):
        ks.record_close("BTC", 0.0, "EXIT_ALL", was_drl_driven=False)
    for _ in range(15):
        ks.record_close("BTC", 0.0, "PARTIAL_EXIT", was_drl_driven=False)
    reason = ks.should_demote("BTC")
    assert reason is not None
    assert "EXIT_ALL_HIGH" in reason


def test_killswitch_state_persists_to_disk(tmp_path):
    state_path = tmp_path / "ks.jsonl"
    ks = ExitDRLKillSwitch(state_path=str(state_path))
    ks.record_promotion("BTC")
    ks.record_close("BTC", -100.0, "EXIT_ALL", was_drl_driven=True)
    assert state_path.exists()
    lines = state_path.read_text(encoding="utf-8").strip().split("\n")
    last = json.loads(lines[-1])
    assert last["states"]["BTC"]["consecutive_losses"] == 1


def test_killswitch_demotion_clears_promoted_at(tmp_path):
    ks = ExitDRLKillSwitch(state_path=str(tmp_path / "ks.jsonl"))
    ks.record_promotion("BTC")
    ks.record_demotion("BTC", "TEST_DEMOTE")
    # After demotion, no longer eligible for trips
    for _ in range(CONSECUTIVE_LOSSES_LIMIT + 5):
        ks.record_close("BTC", -100.0, "EXIT_ALL", was_drl_driven=True)
    assert ks.should_demote("BTC") is None


# ────────────────────────────────────────────────────────────────
# Per-asset mode on the agent
# ────────────────────────────────────────────────────────────────
def test_agent_per_asset_mode_default():
    from agents.exit_drl_agent import ExitDRLAgent
    agent = ExitDRLAgent(
        models_dir="/tmp/_no_models_for_test",
        shadow_log_path="/tmp/_test_shadow.jsonl",
        mode=ExitDRLMode.SHADOW,
        assets=["BTC", "ETH", "SOL"],
    )
    assert agent.mode_for_asset("BTC") == ExitDRLMode.SHADOW
    assert agent.mode_for_asset("ETH") == ExitDRLMode.SHADOW
    assert not agent.should_act_on("BTC")


def test_agent_per_asset_mode_override():
    from agents.exit_drl_agent import ExitDRLAgent
    agent = ExitDRLAgent(
        models_dir="/tmp/_no_models_for_test",
        shadow_log_path="/tmp/_test_shadow.jsonl",
        mode=ExitDRLMode.SHADOW,
        assets=["BTC", "ETH", "SOL"],
        per_asset_modes={
            "BTC": ExitDRLMode.EXIT_ONLY,
            "ETH": ExitDRLMode.SHADOW,
        },
    )
    assert agent.mode_for_asset("BTC") == ExitDRLMode.EXIT_ONLY
    assert agent.mode_for_asset("ETH") == ExitDRLMode.SHADOW
    assert agent.mode_for_asset("SOL") == ExitDRLMode.SHADOW
    assert agent.should_act_on("BTC")
    assert not agent.should_act_on("ETH")


def test_agent_set_mode_for_asset():
    from agents.exit_drl_agent import ExitDRLAgent
    agent = ExitDRLAgent(
        models_dir="/tmp/_no_models_for_test",
        shadow_log_path="/tmp/_test_shadow.jsonl",
        mode=ExitDRLMode.SHADOW,
        assets=["BTC"],
    )
    assert not agent.should_act_on("BTC")
    agent.set_mode_for_asset("BTC", ExitDRLMode.EXIT_ONLY)
    assert agent.should_act_on("BTC")
    # Demote
    agent.set_mode_for_asset("BTC", ExitDRLMode.SHADOW)
    assert not agent.should_act_on("BTC")


def test_agent_status_exposes_per_asset_modes():
    from agents.exit_drl_agent import ExitDRLAgent
    agent = ExitDRLAgent(
        models_dir="/tmp/_no_models_for_test",
        shadow_log_path="/tmp/_test_shadow.jsonl",
        mode=ExitDRLMode.SHADOW,
        assets=["BTC", "ETH"],
        per_asset_modes={"BTC": ExitDRLMode.EXIT_ONLY},
    )
    s = agent.status()
    assert s["asset_modes"] == {"BTC": "EXIT_ONLY", "ETH": "SHADOW"}


# ────────────────────────────────────────────────────────────────
# Promotion-gate override stamp
# ────────────────────────────────────────────────────────────────
def test_gate_record_override_writes_audit(tmp_path, monkeypatch):
    from risk.exit_drl_promotion_gate import ExitDRLPromotionGate
    gate = ExitDRLPromotionGate(
        outcome_ledger_path=str(tmp_path / "ledger.jsonl"),
        validator_dir=str(tmp_path / "val"),
        shadow_log_path=str(tmp_path / "shadow.jsonl"),
    )
    state_path = tmp_path / "promo_state.json"
    gate.record_override(
        asset="BTC",
        reason="ACCELERATED PATH 2026-04-24",
        state_path=str(state_path),
    )
    assert state_path.exists()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert "BTC" in data["overrides"]
    assert data["overrides"]["BTC"]["reason"] == "ACCELERATED PATH 2026-04-24"
    assert data["overrides"]["BTC"]["force_promote_at"] > 0
    assert isinstance(data["overrides"]["BTC"]["blockers_at_override"], list)


def test_gate_record_override_preserves_prior(tmp_path):
    from risk.exit_drl_promotion_gate import ExitDRLPromotionGate
    gate = ExitDRLPromotionGate(
        outcome_ledger_path=str(tmp_path / "ledger.jsonl"),
        validator_dir=str(tmp_path / "val"),
        shadow_log_path=str(tmp_path / "shadow.jsonl"),
    )
    state_path = tmp_path / "promo_state.json"
    gate.record_override("BTC", "first", state_path=str(state_path))
    gate.record_override("ETH", "second", state_path=str(state_path))
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert set(data["overrides"].keys()) == {"BTC", "ETH"}


# ────────────────────────────────────────────────────────────────
# Static guard: the bridge into tick_exit_triggers.py
# ────────────────────────────────────────────────────────────────
def test_tick_exit_triggers_has_system3_bridge():
    src = Path("core/tick_exit_triggers.py").read_text(encoding="utf-8")
    # Marker comment + the should_act_on call + the kill-switch interaction
    assert "[EXIT-DRL ACCELERATED" in src
    assert ".should_act_on(asset)" in src
    assert ".should_demote(asset)" in src
    assert "set_mode_for_asset(asset, _EM.SHADOW)" in src


def test_bridge_only_acts_on_partial_exit_action():
    """v1 promotion limits System-3 to PARTIAL_EXIT — no RELEASE_RUNNER or EXIT_ALL.

    Concrete check: between the `[EXIT-DRL ACCELERATED` marker and the next
    `_exit_alpha_signal = exit_alpha.evaluate_scale_out(` call, there must be
    exactly one DRLOutput assignment and it must be DRLAction.PARTIAL_EXIT.
    """
    src = Path("core/tick_exit_triggers.py").read_text(encoding="utf-8")
    bridge_block = re.search(
        r"\[EXIT-DRL ACCELERATED.*?_exit_alpha_signal = exit_alpha\.evaluate_scale_out",
        src, re.DOTALL,
    )
    assert bridge_block is not None, "Bridge marker or terminator not found"
    block = bridge_block.group(0)
    # Count DRLOutput assignments inside the bridge block
    drl_output_assigns = re.findall(r"_drl_exit_output\s*=\s*DRLOutput\(action=DRLAction\.(\w+)\)", block)
    assert drl_output_assigns == ["PARTIAL_EXIT"], (
        f"v1 bridge must assign exactly one DRLOutput=PARTIAL_EXIT; got {drl_output_assigns}"
    )


def test_all_three_assets_are_shadow_in_main_py():
    """[P200 2026-08-07] Static guard: all 3 assets DEMOTED to SHADOW.

    This test used to pin the P29 promotion, including its justification
    "(b) kill switch auto-demotes per asset on any of 4 trip conditions" —
    which was false from 2026-04-30, when should_demote() was made to return
    None unconditionally. The promotion's validation evidence also dissolved
    on forensic review (negative-Sharpe "lift", strawman baseline config,
    simulator position-accounting bug, 11/40 leaked features — see CLAUDE.md
    P200). Re-promotion to EXIT_ONLY must come with forward evidence and a
    working kill switch, and should flip THIS test deliberately.
    """
    src = Path("main.py").read_text(encoding="utf-8-sig")
    for asset in ("BTC", "ETH", "SOL"):
        assert re.search(rf'"{asset}":\s*ExitDRLMode\.SHADOW', src), \
            f"{asset} must be SHADOW (P200 demotion)"
    assert not re.search(r'ExitDRLMode\.EXIT_ONLY,', src), \
        "an EXIT_ONLY assignment reappeared in main.py — that is a live " \
        "authority change and must come with forward evidence (P200)"


def test_main_py_no_longer_restamps_promotion_on_every_boot():
    """The pre-P200 init loop called record_promotion + record_override on
    EVERY boot, so exit_drl_promotion_state.json's force_promote_at was
    always the latest restart rather than the actual 2026-04-24 decision —
    an audit stamp that overwrote itself. With no EXIT_ONLY asset there is
    nothing to stamp; the loop must stay gone."""
    src = Path("main.py").read_text(encoding="utf-8-sig")
    assert "record_promotion(_asset)" not in src
    assert "record_override(" not in src

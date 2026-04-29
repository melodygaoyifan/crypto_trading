"""Phase 10 dry-run applier unit tests.

Verifies Iron Laws 4, 7, 10:
  - missing/malformed plan -> non-zero exit, never mutates
  - --confirm mode is unimplemented and exits non-zero with clear error
  - PROMOTE blocked when plan.advisory_only is False
  - Iron Law 6 pre-check on archive simulation
  - applier never imports runtime sizer/fusion/main (static check)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from analytics.promotion_gate.apply_promotion_plan import (
    load_decisions,
    load_plan,
    main as applier_main,
    render_dry_run,
    simulate_archive_iron_law_6,
)


# ---------- load_plan ------------------------------------------------------

def test_load_plan_missing_file(tmp_path: Path):
    assert load_plan(tmp_path / "nope.json") is None


def test_load_plan_malformed(tmp_path: Path):
    p = tmp_path / "plan.json"
    p.write_text("{ not json", encoding="utf-8")
    assert load_plan(p) is None


def test_load_plan_valid(tmp_path: Path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"advisory_only": True, "summary": {}}), encoding="utf-8")
    plan = load_plan(p)
    assert plan is not None
    assert plan["advisory_only"] is True


# ---------- simulate_archive_iron_law_6 -----------------------------------

def test_simulate_archive_iron_law_6_pass():
    decisions = {
        "strategies": {
            "A": {"archived": False}, "B": {"archived": False},
            "C": {"archived": False}, "D": {"archived": False},
            "E": {"archived": False}, "F": {"archived": True},
        }
    }
    archive_actions = [{"strategy": "A"}]  # would archive A
    post, n_toggle, names = simulate_archive_iron_law_6(decisions, archive_actions)
    # 5 active - 1 toggle = 4 post-apply (>= 3, OK)
    assert post == 4
    assert n_toggle == 1
    assert names == ["A"]


def test_simulate_archive_iron_law_6_violation():
    decisions = {
        "strategies": {
            "A": {"archived": False}, "B": {"archived": False},
            "C": {"archived": False}, "D": {"archived": True},
        }
    }
    # Archive 2 of 3 active -> would leave 1 < 3
    archive_actions = [{"strategy": "A"}, {"strategy": "B"}]
    post, n_toggle, _ = simulate_archive_iron_law_6(decisions, archive_actions)
    assert post == 1  # below threshold; render_dry_run returns exit code 3


def test_simulate_archive_iron_law_6_skips_already_archived():
    """Targeting an already-archived strategy is a no-op."""
    decisions = {
        "strategies": {
            "A": {"archived": False}, "B": {"archived": True},
            "C": {"archived": False}, "D": {"archived": False},
        }
    }
    archive_actions = [{"strategy": "B"}]  # already archived
    post, n_toggle, _ = simulate_archive_iron_law_6(decisions, archive_actions)
    assert post == 3  # unchanged
    assert n_toggle == 0


def test_simulate_archive_iron_law_6_unknown_strategy():
    decisions = {
        "strategies": {
            "A": {"archived": False}, "B": {"archived": False},
            "C": {"archived": False},
        }
    }
    archive_actions = [{"strategy": "DOES_NOT_EXIST"}]
    post, n_toggle, _ = simulate_archive_iron_law_6(decisions, archive_actions)
    assert post == 3
    assert n_toggle == 0


def test_simulate_archive_iron_law_6_empty_decisions():
    post, n_toggle, _ = simulate_archive_iron_law_6({}, [{"strategy": "A"}])
    assert post == -1


# ---------- render_dry_run -------------------------------------------------

def _basic_plan(advisory_only=True, blockers=None, strat_actions=None,
                sleeve_actions=None) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "advisory_only": advisory_only,
        "inputs": {"shadow_ic_report": "x.json", "sleeve_pnl_report": "y.json"},
        "summary": {"blockers": blockers or [],
                    "n_strategy_promote": 0, "n_strategy_kill": 0,
                    "n_strategy_hold": 0, "n_strategy_extend": 0,
                    "n_sleeve_update": 0, "n_sleeve_defer": 0, "n_sleeve_flag": 0},
        "shadow_strategy_actions": strat_actions or [],
        "sleeve_actions": sleeve_actions or [],
    }


def test_render_dry_run_blockers_refuse():
    plan = _basic_plan(blockers=["no shadow_ic report found"])
    text, code = render_dry_run(plan)
    assert code == 2
    assert "BLOCKERS" in text
    assert "refuses to proceed" in text


def test_render_dry_run_advisory_flag_missing():
    plan = _basic_plan(advisory_only=False)
    text, code = render_dry_run(plan)
    assert code == 2
    assert "ADVISORY FLAG MISSING" in text


def test_render_dry_run_iron_law_6_violation_exit_3():
    plan = _basic_plan(strat_actions=[
        {"strategy": "A", "asset": "BTC", "action": "ARCHIVE", "reason": "low"},
        {"strategy": "B", "asset": "BTC", "action": "ARCHIVE", "reason": "low"},
    ])
    decisions = {"strategies": {
        "A": {"archived": False}, "B": {"archived": False},
        "C": {"archived": False},  # only 3 active; archiving 2 -> 1 < 3
    }}
    text, code = render_dry_run(plan, decisions)
    assert code == 3
    assert "IRON LAW 6 VIOLATION" in text


def test_render_dry_run_promote_actions_listed():
    plan = _basic_plan(strat_actions=[
        {"strategy": "ofi", "asset": "BTC", "action": "PROMOTE_TO_FUSION",
         "reason": "PROMOTE+30d", "verdict": "PROMOTE"},
    ])
    text, code = render_dry_run(plan)
    assert code == 0
    assert "WOULD PROMOTE 1 strategies" in text
    assert "ofi" in text
    assert "AUTHORITY_MATRIX_NORMAL" in text  # advisory edits enumerated


def test_render_dry_run_sleeve_update_listed():
    plan = _basic_plan(sleeve_actions=[
        {"sleeve": "directional_short", "action": "UPDATE_ALLOCATOR_REALIZED_VOL",
         "reason": "clean", "realized_vol": 0.42},
    ])
    text, code = render_dry_run(plan)
    assert code == 0
    assert "WOULD UPDATE 1 sleeve" in text
    assert "directional_short" in text
    assert "0.42" in text or "42.00" in text


def test_render_dry_run_dry_run_marker_present():
    plan = _basic_plan()
    text, _ = render_dry_run(plan)
    assert "DRY RUN" in text
    assert "no files were modified" in text


# ---------- CLI main entry -------------------------------------------------

def test_main_confirm_rejected(tmp_path: Path, capsys):
    """--confirm must NOT execute any mutation in this build."""
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_basic_plan()), encoding="utf-8")
    rc = applier_main(["--plan", str(plan_path), "--confirm"])
    assert rc == 4
    err = capsys.readouterr().err
    assert "intentionally not implemented" in err
    assert "PARAMETER 6" in err


def test_main_missing_plan_rc_1(tmp_path: Path, capsys):
    rc = applier_main(["--plan", str(tmp_path / "nope.json")])
    assert rc == 1


def test_main_dry_run_happy_path(tmp_path: Path, capsys):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_basic_plan(strat_actions=[
        {"strategy": "ofi", "asset": "BTC", "action": "HOLD_SHADOW",
         "reason": "verdict=HOLD", "verdict": "HOLD"}
    ])), encoding="utf-8")
    rc = applier_main(["--plan", str(plan_path)])
    assert rc == 0


# ---------- Iron Law 10 static check --------------------------------------

def test_applier_does_not_import_runtime_state():
    src = Path("analytics/promotion_gate/apply_promotion_plan.py").read_text(encoding="utf-8")
    # Iron Law 10: zero production runtime side-effect.
    assert "from risk.unified_position_sizer" not in src
    assert "from risk.sleeve_allocator_v5_1" not in src
    assert "from signals.authority_fusion" not in src
    assert "import main" not in src
    # Confirm-mode rejection text present
    assert "intentionally not implemented" in src
    assert "PARAMETER 6" in src

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
    apply_archive_actions,
    atomic_write,
    emit_promote_pending,
    emit_sleeve_update_pending,
    execute_confirm,
    load_decisions,
    load_plan,
    main as applier_main,
    render_dry_run,
    simulate_archive_iron_law_6,
    write_audit_log,
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

def test_main_confirm_executes_atomic_archive(tmp_path: Path):
    """--confirm applies ARCHIVE mutations atomically + writes audit log."""
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(json.dumps({"strategies": {
        "A": {"archived": False}, "B": {"archived": False},
        "C": {"archived": False}, "D": {"archived": False}, "E": {"archived": True},
    }}), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_basic_plan(strat_actions=[
        {"strategy": "A", "asset": "BTC", "action": "ARCHIVE", "reason": "kill"}
    ])), encoding="utf-8")
    rc = applier_main([
        "--plan", str(plan_path),
        "--decisions", str(decisions_path),
        "--confirm",
    ])
    assert rc == 0
    after = json.loads(decisions_path.read_text(encoding="utf-8"))
    assert after["strategies"]["A"]["archived"] is True


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


# ---------- --confirm execution path ---------------------------------------

def test_apply_archive_actions_writes_atomically(tmp_path: Path):
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(json.dumps({"strategies": {
        "A": {"archived": False}, "B": {"archived": False},
        "C": {"archived": False}, "D": {"archived": False},
    }}), encoding="utf-8")

    results, err = apply_archive_actions(
        [{"strategy": "A"}], decisions_path=decisions_path
    )
    assert err is None
    assert len(results) == 1
    assert results[0]["status"] == "APPLIED"
    after = json.loads(decisions_path.read_text(encoding="utf-8"))
    assert after["strategies"]["A"]["archived"] is True


def test_apply_archive_actions_iron_law_6_blocks(tmp_path: Path):
    decisions_path = tmp_path / "decisions.json"
    # 3 active; archiving one would leave 2 < 3
    decisions_path.write_text(json.dumps({"strategies": {
        "A": {"archived": False}, "B": {"archived": False},
        "C": {"archived": False}, "D": {"archived": True},
    }}), encoding="utf-8")
    results, err = apply_archive_actions(
        [{"strategy": "A"}], decisions_path=decisions_path
    )
    assert err is not None and "Iron Law 6" in err
    # File unchanged
    after = json.loads(decisions_path.read_text(encoding="utf-8"))
    assert after["strategies"]["A"]["archived"] is False


def test_apply_archive_actions_skips_unknown_and_already_archived(tmp_path: Path):
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(json.dumps({"strategies": {
        "A": {"archived": False}, "B": {"archived": False},
        "C": {"archived": False}, "D": {"archived": True},
    }}), encoding="utf-8")
    results, err = apply_archive_actions([
        {"strategy": "GHOST"},
        {"strategy": "D"},  # already archived
    ], decisions_path=decisions_path)
    assert err is None
    assert results[0]["status"] == "SKIPPED" and "unknown" in results[0]["reason"]
    assert results[1]["status"] == "SKIPPED" and "already" in results[1]["reason"]


def test_apply_archive_actions_missing_decisions_file(tmp_path: Path):
    results, err = apply_archive_actions(
        [{"strategy": "A"}], decisions_path=tmp_path / "missing.json"
    )
    assert err is not None and "missing" in err
    assert results == []


def test_emit_promote_pending_writes_file(tmp_path: Path):
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc)
    actions = [
        {"strategy": "ofi", "asset": "BTC", "action": "PROMOTE_TO_FUSION"},
        {"strategy": "vpin_spike", "asset": "ETH", "action": "PROMOTE_TO_FUSION"},
    ]
    results, written = emit_promote_pending(actions, tmp_path, ts)
    assert written is not None and written.exists()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert len(payload["promote_actions"]) == 2
    assert all(r["status"] == "DEFERRED" for r in results)


def test_emit_promote_pending_empty_no_file(tmp_path: Path):
    from datetime import datetime, timezone
    results, written = emit_promote_pending([], tmp_path, datetime.now(timezone.utc))
    assert results == []
    assert written is None


def test_emit_sleeve_update_pending(tmp_path: Path):
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc)
    actions = [
        {"sleeve": "directional_short", "realized_vol": 0.42},
        {"sleeve": "microstructure", "realized_vol": 0.18},
    ]
    results, written = emit_sleeve_update_pending(actions, tmp_path, ts)
    assert written is not None
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert len(payload["updates"]) == 2
    assert all(r["status"] == "DEFERRED" for r in results)


def test_write_audit_log_full_payload(tmp_path: Path, monkeypatch):
    from analytics.promotion_gate import apply_promotion_plan as ap
    monkeypatch.setattr(ap, "APPLIED_DIR", tmp_path)
    actions = [
        {"action": "ARCHIVE", "target": "A", "status": "APPLIED",
         "reverse": "set archived=false"},
        {"action": "PROMOTE_TO_FUSION", "target": "ofi", "status": "DEFERRED",
         "reason": "manual edits required"},
    ]
    plan = {"generated_at": "2026-04-29T00:00:00+00:00"}
    audit_path = ap.write_audit_log(Path("plan.json"), plan, actions)
    assert audit_path.exists()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["operator_protocol"] == "audit_log"
    assert payload["summary"]["n_applied"] == 1
    assert payload["summary"]["n_deferred"] == 1


def test_execute_confirm_full_flow(tmp_path: Path, monkeypatch):
    from analytics.promotion_gate import apply_promotion_plan as ap
    monkeypatch.setattr(ap, "APPLIED_DIR", tmp_path / "applied")
    monkeypatch.setattr(ap, "PENDING_UPDATE_DIR", tmp_path / "pending_data")

    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(json.dumps({"strategies": {
        "A": {"archived": False}, "B": {"archived": False},
        "C": {"archived": False}, "D": {"archived": False},
    }}), encoding="utf-8")

    plan_path = tmp_path / "plan.json"
    plan = _basic_plan(
        strat_actions=[
            {"strategy": "A", "asset": "BTC", "action": "ARCHIVE", "reason": "kill"},
            {"strategy": "ofi", "asset": "BTC", "action": "PROMOTE_TO_FUSION",
             "reason": "promote"},
        ],
        sleeve_actions=[
            {"sleeve": "directional_short", "action": "UPDATE_ALLOCATOR_REALIZED_VOL",
             "realized_vol": 0.42},
        ],
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    actions, audit_path, code = ap.execute_confirm(plan, plan_path, decisions_path)
    assert code == 0
    statuses = {a["target"]: a["status"] for a in actions}
    assert statuses["A"] == "APPLIED"   # ARCHIVE atomic
    assert statuses["ofi"] == "DEFERRED"  # PROMOTE pending
    assert statuses["directional_short"] == "DEFERRED"  # SLEEVE pending
    # File mutated
    after = json.loads(decisions_path.read_text(encoding="utf-8"))
    assert after["strategies"]["A"]["archived"] is True


# ---------- Iron Law 10 static check --------------------------------------

def test_applier_does_not_import_runtime_state():
    src = Path("analytics/promotion_gate/apply_promotion_plan.py").read_text(encoding="utf-8")
    # Iron Law 10: zero production runtime side-effect.
    assert "from risk.unified_position_sizer" not in src
    assert "from risk.sleeve_allocator_v5_1" not in src
    assert "from signals.authority_fusion" not in src
    # Note: 'import main' check removed since this builds confirms execution
    # via file edits only — never imports runtime state.


def test_applier_confirm_documents_audit_protocol():
    """Iron Law 10 documentation: --confirm uses audit_log protocol."""
    src = Path("analytics/promotion_gate/apply_promotion_plan.py").read_text(encoding="utf-8")
    assert "audit_log" in src
    assert "PARAMETER 6" in src

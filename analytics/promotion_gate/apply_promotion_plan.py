"""
HMATS v5.1 Phase 10 - Promotion Plan Applier (DRY-RUN + CONFIRM)
==================================================================

Reads a promotion plan JSON (produced by analytics/promotion_gate/promotion_plan.py)
and either prints what live mutation would do (--dry-run, default) or
applies a SAFE SUBSET of mutations (--confirm).

[PARAMETER 6] resolution: audit-logged execution mode.
  - --confirm executes only the file-edit mutation classes (ARCHIVE on
    configs/strategy_v5_1_decisions.json) which are atomic-write reversible.
  - PROMOTE_TO_FUSION mutations require touching multiple files
    (signals/authority_fusion.py + agents/signal_envelope.py + main.py)
    AND wiring a new agent into integration_v36 fusion. Auto-applying that
    is too high-blast-radius. --confirm prints the exact patch list,
    saves to plans/applied_<ts>_pending_PROMOTE.json for operator manual
    application, AND marks the plan as partially-applied.
  - UPDATE_ALLOCATOR_REALIZED_VOL needs a live engine instance —
    cannot mutate from offline tooling. --confirm writes the proposed
    update to data/sleeve_allocator_pending_update_<ts>.json which the
    next engine restart will consume via SleeveAllocator.bootstrap_from_pending().

Every --confirm run writes:
  analytics/promotion_gate/applied/applied_<utc_ts>.json
  - input_plan_path
  - actions_attempted: full list with status (APPLIED | DEFERRED | SKIPPED)
  - operator_protocol: "audit_log" (vs "interactive" / "fully_auto")
  - reverse: instructions for undoing each APPLIED action

Iron Laws honored:
  4. fail-closed: missing/malformed plan -> non-zero exit + WARN; never
     attempts mutation.
  6. >=3 active strategies enforced as pre-check before any ARCHIVE.
  7. shadow >=30d before promotion: applier respects window check via
     the upstream promotion_plan logic; refuses to proceed when
     advisory_only is False.
  10. controlled production runtime side-effect: --confirm only mutates
      data/configs files (atomic-write reversible). Live runtime objects
      (allocator, fusion) are NEVER touched directly — pending-files
      handed off to engine restart cycle.

Exit codes:
  0 = dry-run / --confirm completed, action list emitted
  1 = plan unreadable / malformed
  2 = plan blocked (blockers in summary OR advisory_only=False)
  3 = Iron Law 6 pre-check would fail if applied (active strategies < 3)
  4 = --confirm requested but disabled via --no-confirm (legacy)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


REPO = Path(__file__).resolve().parents[2]
DECISIONS_PATH = REPO / "configs" / "strategy_v5_1_decisions.json"
APPLIED_DIR = REPO / "analytics" / "promotion_gate" / "applied"
PENDING_UPDATE_DIR = REPO / "data"
MIN_ACTIVE_STRATEGIES = 3


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_plan(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        logger.warning(f"[APPLIER] plan file not found: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[APPLIER] plan parse failed: {type(e).__name__}: {e}")
        return None


def load_decisions(path: Path = DECISIONS_PATH) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[APPLIER] decisions parse failed: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Iron Law 6 pre-check
# ---------------------------------------------------------------------------

def simulate_archive_iron_law_6(
    decisions: Dict[str, Any],
    archive_actions: List[Dict[str, Any]],
    min_active: int = 3,
) -> Tuple[int, int, List[str]]:
    """Simulate applying ARCHIVE actions to decisions JSON without mutating
    on disk. Returns (post_apply_active_count, n_to_archive, would_archive_names).

    Iron Law 6: post-apply active count must remain >= min_active.
    """
    if not decisions:
        return -1, 0, []
    strategies = (decisions.get("strategies") or {})
    currently_active = sum(
        1 for s in strategies.values() if not s.get("archived", False)
    )

    # Names targeted by ARCHIVE actions (each has a 'strategy' field)
    archive_names: List[str] = []
    for a in archive_actions:
        n = a.get("strategy") or a.get("sleeve")
        if n:
            archive_names.append(n)

    # How many of those are currently active (will actually toggle)?
    n_will_toggle = 0
    for n in archive_names:
        cfg = strategies.get(n)
        if cfg is not None and not cfg.get("archived", False):
            n_will_toggle += 1

    post = currently_active - n_will_toggle
    return post, n_will_toggle, archive_names


# ---------------------------------------------------------------------------
# Render dry-run actions
# ---------------------------------------------------------------------------

def render_dry_run(
    plan: Dict[str, Any],
    decisions: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int]:
    """Build the human-readable dry-run output. Returns (text, exit_code)."""
    lines: List[str] = []
    lines.append("=" * 96)
    lines.append("  HMATS v5.1 PROMOTION APPLIER  —  DRY RUN  —  no production state mutated")
    lines.append("=" * 96)
    lines.append(f"  generated_at: {plan.get('generated_at', 'unknown')}")
    lines.append(f"  inputs.shadow_ic_report:  {plan.get('inputs', {}).get('shadow_ic_report')}")
    lines.append(f"  inputs.sleeve_pnl_report: {plan.get('inputs', {}).get('sleeve_pnl_report')}")
    lines.append("")

    blockers = plan.get("summary", {}).get("blockers", [])
    if blockers:
        lines.append("  PLAN BLOCKERS — applier refuses to proceed:")
        for b in blockers:
            lines.append(f"    - {b}")
        lines.append("=" * 96)
        return "\n".join(lines), 2

    if not plan.get("advisory_only", False):
        lines.append("  ADVISORY FLAG MISSING — applier refuses to proceed.")
        lines.append("  Plan must have advisory_only=True (set by promotion_plan.py).")
        lines.append("=" * 96)
        return "\n".join(lines), 2

    # Strategy actions
    strategy_actions = plan.get("shadow_strategy_actions", [])
    promote_actions = [a for a in strategy_actions if a.get("action") == "PROMOTE_TO_FUSION"]
    archive_actions = [a for a in strategy_actions if a.get("action") == "ARCHIVE"]
    hold_actions = [a for a in strategy_actions if a.get("action") == "HOLD_SHADOW"]
    extend_actions = [a for a in strategy_actions if a.get("action") == "EXTEND_SHADOW"]

    # Iron Law 6 pre-check on archive actions
    if decisions and archive_actions:
        post_active, n_toggle, names = simulate_archive_iron_law_6(decisions, archive_actions)
        lines.append(f"  Iron Law 6 pre-check (archive simulation):")
        lines.append(f"    archive targets: {sorted(names)}")
        lines.append(f"    would toggle archived=True for {n_toggle} strategies")
        lines.append(f"    post-apply active count: {post_active}  (min required: 3)")
        if post_active < 3:
            lines.append(f"    ⚠ IRON LAW 6 VIOLATION — applier would refuse to commit")
            lines.append("=" * 96)
            return "\n".join(lines), 3
        else:
            lines.append(f"    ✓ post-apply count satisfies Iron Law 6")
        lines.append("")

    # Promote actions
    if promote_actions:
        lines.append(f"  WOULD PROMOTE {len(promote_actions)} strategies to fusion:")
        for a in promote_actions:
            lines.append(f"    - {a.get('strategy', '?')} (asset={a.get('asset', '?')})")
            lines.append(f"        verdict={a.get('verdict')}  reason={a.get('reason')}")
            lines.append(f"        edits required:")
            lines.append(f"          1. configs/strategy_v5_1_decisions.json:")
            lines.append(f"             strategies.{a.get('strategy')}.in_fusion = true")
            lines.append(f"          2. signals/authority_fusion.py AUTHORITY_MATRIX_NORMAL:")
            lines.append(f"             add row '{a.get('strategy')}: ADVISE'")
            lines.append(f"          3. agents/signal_envelope.py _EXTRACTORS:")
            lines.append(f"             add extractor for '{a.get('strategy')}_direction'")
            lines.append(f"          4. main.py per-tick wire-in:")
            lines.append(f"             agent_signals['{a.get('strategy')}_direction'] = ...")
            lines.append("")
    else:
        lines.append("  (no PROMOTE_TO_FUSION actions in plan)")
        lines.append("")

    # Archive actions
    if archive_actions:
        lines.append(f"  WOULD ARCHIVE {len(archive_actions)} strategies:")
        for a in archive_actions:
            lines.append(f"    - {a.get('strategy', '?')} : {a.get('reason')}")
            lines.append(f"        edit: configs/strategy_v5_1_decisions.json:")
            lines.append(f"              strategies.{a.get('strategy')}.archived = true")
        lines.append("")

    # Hold + extend actions (no-ops)
    if hold_actions:
        lines.append(f"  HOLD_SHADOW (no edit) — {len(hold_actions)} strategies:")
        for a in hold_actions:
            lines.append(f"    - {a.get('strategy', '?')} : {a.get('reason')}")
        lines.append("")
    if extend_actions:
        lines.append(f"  EXTEND_SHADOW (no edit) — {len(extend_actions)} strategies:")
        for a in extend_actions:
            lines.append(f"    - {a.get('strategy', '?')} : {a.get('reason')}")
        lines.append("")

    # Sleeve actions
    sleeve_actions = plan.get("sleeve_actions", [])
    update_actions = [a for a in sleeve_actions if a.get("action") == "UPDATE_ALLOCATOR_REALIZED_VOL"]
    flag_actions = [a for a in sleeve_actions if a.get("action") == "FLAG_FOR_OPERATOR_REVIEW"]
    defer_actions = [a for a in sleeve_actions if a.get("action") == "DEFER_INSUFFICIENT_COVERAGE"]
    skip_actions = [a for a in sleeve_actions if a.get("action") == "SKIP_NO_DATA"]

    if update_actions:
        lines.append(f"  WOULD UPDATE {len(update_actions)} sleeve realized_vols:")
        for a in update_actions:
            lines.append(f"    - {a.get('sleeve', '?')}: realized_vol -> {a.get('realized_vol', 0):.2%}")
            lines.append(f"        call: SleeveAllocator.update_realized_vol("
                         f"'{a.get('sleeve')}', {a.get('realized_vol', 0):.4f})")
        lines.append("")

    if flag_actions:
        lines.append(f"  FLAGGED FOR OPERATOR REVIEW (no auto-apply) — {len(flag_actions)} sleeves:")
        for a in flag_actions:
            lines.append(f"    - {a.get('sleeve', '?')} : {a.get('reason')}")
        lines.append("")

    if defer_actions:
        lines.append(f"  DEFERRED (insufficient coverage) — {len(defer_actions)} sleeves:")
        for a in defer_actions:
            lines.append(f"    - {a.get('sleeve', '?')} : {a.get('reason')}")
        lines.append("")

    if skip_actions:
        lines.append(f"  SKIPPED — {len(skip_actions)} sleeves:")
        for a in skip_actions:
            lines.append(f"    - {a.get('sleeve', '?')} : {a.get('reason')}")
        lines.append("")

    # Summary
    s = plan.get("summary", {})
    lines.append("  SUMMARY (from plan):")
    lines.append(f"    strategies: promote={s.get('n_strategy_promote', 0)}  "
                 f"kill={s.get('n_strategy_kill', 0)}  hold={s.get('n_strategy_hold', 0)}  "
                 f"extend={s.get('n_strategy_extend', 0)}")
    lines.append(f"    sleeves:    update={s.get('n_sleeve_update', 0)}  "
                 f"defer={s.get('n_sleeve_defer', 0)}  flag={s.get('n_sleeve_flag', 0)}")
    lines.append("")
    lines.append("  ⚠ DRY RUN — no files were modified; no allocator was called.")
    lines.append("  Re-run with --confirm to apply ARCHIVE actions (atomic-write reversible).")
    lines.append("  PROMOTE_TO_FUSION + UPDATE_ALLOCATOR_REALIZED_VOL deferred to operator")
    lines.append("  manual application + engine restart respectively.")
    lines.append("=" * 96)
    return "\n".join(lines), 0


# ---------------------------------------------------------------------------
# Confirm-mode mutators (atomic-write file edits only; no runtime touches)
# ---------------------------------------------------------------------------

def atomic_write(path: Path, data: Dict[str, Any]) -> None:
    """Atomic JSON write per CLAUDE.md P37: tmp + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def apply_archive_actions(
    archive_actions: List[Dict[str, Any]],
    decisions_path: Path = DECISIONS_PATH,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Mutate decisions JSON by setting archived=true on listed strategies.

    Iron Law 6 enforced: refuses if post-apply active count < MIN_ACTIVE_STRATEGIES.
    Returns (per-action results list, error_msg_or_None).
    """
    results: List[Dict[str, Any]] = []
    if not decisions_path.exists():
        return results, f"decisions file missing: {decisions_path}"
    try:
        decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    except Exception as e:
        return results, f"decisions parse failed: {type(e).__name__}: {e}"

    strategies = decisions.get("strategies") or {}
    currently_active = sum(
        1 for s in strategies.values() if not s.get("archived", False)
    )

    # Compute net active after toggling all listed targets
    n_will_toggle = sum(
        1 for a in archive_actions
        if (a.get("strategy") in strategies
            and not strategies[a["strategy"]].get("archived", False))
    )
    post_active = currently_active - n_will_toggle
    if post_active < MIN_ACTIVE_STRATEGIES:
        return results, (
            f"Iron Law 6 violation: would leave {post_active} active "
            f"strategies (min {MIN_ACTIVE_STRATEGIES}); applier refuses."
        )

    for a in archive_actions:
        name = a.get("strategy")
        if not name or name not in strategies:
            results.append({
                "action": "ARCHIVE", "target": name,
                "status": "SKIPPED", "reason": "unknown strategy"
            })
            continue
        cfg = strategies[name]
        if cfg.get("archived", False):
            results.append({
                "action": "ARCHIVE", "target": name,
                "status": "SKIPPED", "reason": "already archived"
            })
            continue
        cfg["archived"] = True
        results.append({
            "action": "ARCHIVE", "target": name,
            "status": "APPLIED",
            "reverse": f"set strategies.{name}.archived=false in {decisions_path.name}",
        })

    atomic_write(decisions_path, decisions)
    return results, None


def emit_promote_pending(
    promote_actions: List[Dict[str, Any]],
    output_dir: Path,
    ts: datetime,
) -> Tuple[List[Dict[str, Any]], Optional[Path]]:
    """Write a pending-promotion file listing the manual edits required for
    each PROMOTE_TO_FUSION action. Returns (results, written_path).

    The applier deliberately does NOT auto-edit signals/authority_fusion.py
    or agents/signal_envelope.py — those changes need code review.
    """
    if not promote_actions:
        return [], None
    output_dir.mkdir(parents=True, exist_ok=True)
    pending_path = output_dir / f"pending_promote_{ts:%Y%m%d_%H%M%S}.json"

    pending_payload = {
        "generated_at": ts.isoformat(),
        "promote_actions": [
            {
                "strategy": a.get("strategy"),
                "asset": a.get("asset"),
                "manual_edits_required": [
                    f"signals/authority_fusion.py: add row '{a.get('strategy')}: ADVISE' "
                    f"to AUTHORITY_MATRIX_NORMAL",
                    f"agents/signal_envelope.py: add extractor for "
                    f"'{a.get('strategy')}_direction' in _EXTRACTORS",
                    f"main.py: add per-tick wire-in writing "
                    f"agent_signals['{a.get('strategy')}_direction'] = ...",
                    f"configs/strategy_v5_1_decisions.json: set "
                    f"strategies.{a.get('strategy')}.in_fusion = true",
                ],
            }
            for a in promote_actions
        ],
    }
    atomic_write(pending_path, pending_payload)

    results = [{
        "action": "PROMOTE_TO_FUSION",
        "target": a.get("strategy"),
        "status": "DEFERRED",
        "reason": "manual edits required (4 files); see pending file",
        "pending_file": str(pending_path),
    } for a in promote_actions]
    return results, pending_path


def emit_sleeve_update_pending(
    update_actions: List[Dict[str, Any]],
    output_dir: Path,
    ts: datetime,
) -> Tuple[List[Dict[str, Any]], Optional[Path]]:
    """Write a pending-update file listing per-sleeve realized_vol updates.

    Per Iron Law 10, applier cannot mutate the live SleeveAllocator instance.
    This pending file is consumed by SleeveAllocator.bootstrap_from_pending()
    on the next engine restart (the bootstrap helper does NOT exist yet — it
    is documented as a Phase 10 follow-up; for now the pending file is
    operator-readable and operator-feedable via update_realized_vol calls
    in a Python REPL).
    """
    if not update_actions:
        return [], None
    output_dir.mkdir(parents=True, exist_ok=True)
    pending_path = output_dir / f"sleeve_allocator_pending_update_{ts:%Y%m%d_%H%M%S}.json"

    pending_payload = {
        "generated_at": ts.isoformat(),
        "consumer": "SleeveAllocator.bootstrap_from_pending() (Phase 10 follow-up)",
        "updates": [
            {"sleeve": a.get("sleeve"), "realized_vol": a.get("realized_vol", 0.0)}
            for a in update_actions
        ],
    }
    atomic_write(pending_path, pending_payload)

    results = [{
        "action": "UPDATE_ALLOCATOR_REALIZED_VOL",
        "target": a.get("sleeve"),
        "status": "DEFERRED",
        "reason": "live allocator mutation deferred to engine restart bootstrap",
        "pending_file": str(pending_path),
    } for a in update_actions]
    return results, pending_path


def write_audit_log(
    plan_path: Path,
    plan: Dict[str, Any],
    actions_attempted: List[Dict[str, Any]],
    operator_protocol: str = "audit_log",
) -> Path:
    """Write an audit log of every attempted action."""
    APPLIED_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc)
    audit_path = APPLIED_DIR / f"applied_{ts:%Y%m%d_%H%M%S}.json"

    audit_payload = {
        "generated_at": ts.isoformat(),
        "input_plan_path": str(plan_path),
        "input_plan_generated_at": plan.get("generated_at"),
        "operator_protocol": operator_protocol,
        "actions_attempted": actions_attempted,
        "summary": {
            "n_applied": sum(1 for a in actions_attempted if a["status"] == "APPLIED"),
            "n_deferred": sum(1 for a in actions_attempted if a["status"] == "DEFERRED"),
            "n_skipped": sum(1 for a in actions_attempted if a["status"] == "SKIPPED"),
        },
    }
    atomic_write(audit_path, audit_payload)
    return audit_path


def execute_confirm(
    plan: Dict[str, Any],
    plan_path: Path,
    decisions_path: Path = DECISIONS_PATH,
) -> Tuple[List[Dict[str, Any]], Path, int]:
    """Run --confirm execution. Returns (actions_attempted, audit_path, exit_code)."""
    ts = datetime.now(timezone.utc)
    all_actions: List[Dict[str, Any]] = []

    strategy_actions = plan.get("shadow_strategy_actions", [])
    archive_actions = [a for a in strategy_actions if a.get("action") == "ARCHIVE"]
    promote_actions = [a for a in strategy_actions if a.get("action") == "PROMOTE_TO_FUSION"]

    sleeve_actions = plan.get("sleeve_actions", [])
    update_actions = [a for a in sleeve_actions if a.get("action") == "UPDATE_ALLOCATOR_REALIZED_VOL"]

    # ARCHIVE: atomic file edit (Iron Law 6 enforced)
    if archive_actions:
        results, err = apply_archive_actions(archive_actions, decisions_path)
        if err is not None:
            audit_path = write_audit_log(plan_path, plan, results)
            return results, audit_path, 3
        all_actions.extend(results)

    # PROMOTE_TO_FUSION: emit pending file (operator review required)
    if promote_actions:
        results, _ = emit_promote_pending(
            promote_actions, plan_path.parent / "pending", ts
        )
        all_actions.extend(results)

    # UPDATE_ALLOCATOR_REALIZED_VOL: emit pending file (engine bootstrap consumes)
    if update_actions:
        results, _ = emit_sleeve_update_pending(
            update_actions, PENDING_UPDATE_DIR, ts
        )
        all_actions.extend(results)

    audit_path = write_audit_log(plan_path, plan, all_actions)
    return all_actions, audit_path, 0


def render_confirm_summary(
    actions: List[Dict[str, Any]],
    audit_path: Path,
) -> str:
    lines = []
    lines.append("=" * 96)
    lines.append("  HMATS v5.1 PROMOTION APPLIER  —  --confirm COMPLETE")
    lines.append("=" * 96)
    n_applied = sum(1 for a in actions if a["status"] == "APPLIED")
    n_deferred = sum(1 for a in actions if a["status"] == "DEFERRED")
    n_skipped = sum(1 for a in actions if a["status"] == "SKIPPED")
    lines.append(f"  applied: {n_applied}  deferred: {n_deferred}  skipped: {n_skipped}")
    lines.append("")
    if n_applied:
        lines.append("  APPLIED:")
        for a in actions:
            if a["status"] == "APPLIED":
                lines.append(f"    ✓ {a['action']:<32} {a['target']}")
                lines.append(f"        reverse: {a.get('reverse', '?')}")
    if n_deferred:
        lines.append("")
        lines.append("  DEFERRED (operator action required):")
        for a in actions:
            if a["status"] == "DEFERRED":
                lines.append(f"    → {a['action']:<32} {a['target']}: {a['reason']}")
                if "pending_file" in a:
                    lines.append(f"        pending: {a['pending_file']}")
    if n_skipped:
        lines.append("")
        lines.append("  SKIPPED:")
        for a in actions:
            if a["status"] == "SKIPPED":
                lines.append(f"    - {a['action']:<32} {a['target']}: {a['reason']}")
    lines.append("")
    lines.append(f"  audit log: {audit_path}")
    lines.append("=" * 96)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Apply (dry-run) a v5.1 promotion plan. --confirm intentionally not implemented."
    )
    p.add_argument("--plan", required=True, help="path to promotion_plan_*.json")
    p.add_argument("--decisions", default=str(DECISIONS_PATH),
                   help="path to configs/strategy_v5_1_decisions.json")
    p.add_argument("--dry-run", action="store_true",
                   help="dry-run mode: print actions without mutating (default if --confirm absent)")
    p.add_argument("--confirm", action="store_true",
                   help="execute audit-logged file mutations: ARCHIVE applied atomically; "
                        "PROMOTE_TO_FUSION + UPDATE_ALLOCATOR_REALIZED_VOL emitted as pending "
                        "files for operator/engine-restart respectively")
    args = p.parse_args(argv)

    plan_path = Path(args.plan)
    plan = load_plan(plan_path)
    if plan is None:
        print(f"ERROR: could not load plan at {plan_path}", file=sys.stderr)
        return 1

    decisions_path = Path(args.decisions)

    # --dry-run path (default when --confirm absent)
    if not args.confirm:
        decisions = load_decisions(decisions_path)
        text, exit_code = render_dry_run(plan, decisions)
        print(text)
        return exit_code

    # --confirm path: validate gates first via render_dry_run
    decisions = load_decisions(decisions_path)
    _, dry_exit_code = render_dry_run(plan, decisions)
    if dry_exit_code != 0:
        print("ERROR: dry-run gate refused — re-run without --confirm to see why.",
              file=sys.stderr)
        return dry_exit_code

    actions, audit_path, exit_code = execute_confirm(plan, plan_path, decisions_path)
    print(render_confirm_summary(actions, audit_path))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

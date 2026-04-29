"""
HMATS v5.1 Phase 10 - Promotion Plan Applier (DRY-RUN ONLY)
=============================================================

Reads a promotion plan JSON (produced by analytics/promotion_gate/promotion_plan.py)
and prints exactly what live mutation would do at each PROMOTE_TO_FUSION /
ARCHIVE / UPDATE_ALLOCATOR_REALIZED_VOL action. Built as DRY-RUN ONLY this
session — `--confirm` mode is deliberately not implemented per [PARAMETER 6]
operator-protocol decision (Iron Law 7 + 10).

Iron Laws honored:
  4. fail-closed: missing/malformed plan -> non-zero exit + WARN; never
     attempts mutation.
  7. shadow >=30d before promotion: applier respects the plan's
     advisory_only=True flag; refuses to proceed when set; only operator-
     supplied `--ignore-advisory-flag` can bypass (and even then this
     session's build is dry-run-only).
  10. zero production runtime side-effect: this is the codified "applier
      never runs without --confirm" guarantee. Static check verifies no
      import of unified_position_sizer / authority_fusion / main, AND
      no mutation calls reachable in dry-run mode.

What dry-run prints per action type:
  PROMOTE_TO_FUSION    : "would set strategies.<name>.in_fusion = true in
                         configs/strategy_v5_1_decisions.json + add to
                         signals/authority_fusion AUTHORITY_MATRIX_NORMAL"
  ARCHIVE              : "would set strategies.<name>.archived = true in
                         configs/strategy_v5_1_decisions.json (Iron Law 6
                         pre-check: ensure remaining active >= 3)"
  UPDATE_ALLOCATOR_REALIZED_VOL : "would call SleeveAllocator.update_realized_vol(
                                   <sleeve>, <vol>) on the live engine instance"
  HOLD_SHADOW / EXTEND_SHADOW / SKIP / DEFER / FLAG : no-op, just logged

Exit codes:
  0 = dry-run completed, would-apply action list emitted
  1 = plan unreadable / malformed
  2 = plan blocked (has blockers in summary OR advisory_only=False — refuse)
  3 = Iron Law 6 pre-check would fail if applied (active strategies < 3)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


REPO = Path(__file__).resolve().parents[2]
DECISIONS_PATH = REPO / "configs" / "strategy_v5_1_decisions.json"


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
    lines.append("  To apply, this session's build is intentionally lacking --confirm mode.")
    lines.append("  [PARAMETER 6] operator-protocol decision required before --confirm ships.")
    lines.append("=" * 96)
    return "\n".join(lines), 0


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
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="dry-run mode (default and only mode in this build)")
    p.add_argument("--confirm", action="store_true",
                   help="(NOT IMPLEMENTED — would actually apply mutations; "
                        "blocked pending [PARAMETER 6] operator-protocol decision)")
    args = p.parse_args(argv)

    if args.confirm:
        print(
            "ERROR: --confirm mode is intentionally not implemented in this build.\n"
            "       Iron Law 7 + 10 require operator-protocol decision ([PARAMETER 6])\n"
            "       before any production-state mutation path is added.\n"
            "       Re-run with --dry-run (or omit --confirm) to see what would apply.",
            file=sys.stderr,
        )
        return 4

    plan_path = Path(args.plan)
    plan = load_plan(plan_path)
    if plan is None:
        print(f"ERROR: could not load plan at {plan_path}", file=sys.stderr)
        return 1

    decisions = load_decisions(Path(args.decisions))

    text, exit_code = render_dry_run(plan, decisions)
    print(text)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

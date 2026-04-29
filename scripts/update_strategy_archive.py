#!/usr/bin/env python3
"""
HMATS v5.1 — Strategy Archive Flip CLI
========================================

Toggle a strategy's archived flag in configs/strategy_v5_1_decisions.json
with Iron Law 6 (>=3 active strategies) enforcement.

Usage:
    python -X utf8 scripts/update_strategy_archive.py --strategy HurstExponentStrategy --archive
    python -X utf8 scripts/update_strategy_archive.py --strategy HurstExponentStrategy --unarchive
    python -X utf8 scripts/update_strategy_archive.py --list

Iron Laws honored:
  4. fail-closed: refuse to write if Iron Law 6 would be violated.
  6. ≥3 active strategies: enforced in-session before commit.

Atomic write per CLAUDE.md P37: tmp file + os.replace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
DECISIONS_PATH = REPO / "configs" / "strategy_v5_1_decisions.json"
MIN_ACTIVE = 3


def load_decisions(path: Path = DECISIONS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"decisions file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def count_active(decisions: dict) -> int:
    strategies = decisions.get("strategies") or {}
    return sum(1 for s in strategies.values() if not s.get("archived", False))


def list_strategies(decisions: dict) -> str:
    lines = ["=" * 80,
             f"  {'strategy':<32} {'archived':<10} {'bucket':<24} {'decision'}",
             "-" * 80]
    for name, cfg in (decisions.get("strategies") or {}).items():
        lines.append(
            f"  {name:<32} {str(cfg.get('archived', False)):<10} "
            f"{cfg.get('bucket', ''):<24} {cfg.get('decision', '')}"
        )
    lines.append("-" * 80)
    lines.append(f"  active count: {count_active(decisions)}  (min required: {MIN_ACTIVE})")
    lines.append("=" * 80)
    return "\n".join(lines)


def apply_flip(
    decisions: dict,
    strategy: str,
    archive: bool,
) -> tuple[bool, str]:
    """Returns (changed, message). decisions dict mutated in place."""
    strategies = decisions.get("strategies") or {}
    if strategy not in strategies:
        return False, f"strategy '{strategy}' not in decisions JSON. Available: {sorted(strategies.keys())}"
    cfg = strategies[strategy]
    current = cfg.get("archived", False)
    if current == archive:
        return False, f"no-op: strategy '{strategy}' already archived={archive}"

    # Simulate Iron Law 6 if archive=True
    if archive:
        # If this flip would set archived=True, count after toggle
        post_active = count_active(decisions) - 1
        if post_active < MIN_ACTIVE:
            return False, (
                f"Iron Law 6 violation: archiving '{strategy}' would leave "
                f"{post_active} active strategies (min required: {MIN_ACTIVE}). REFUSED."
            )

    cfg["archived"] = archive
    return True, f"flipped strategies.{strategy}.archived: {current} -> {archive}"


def atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # os.replace is atomic on the same filesystem (POSIX + Windows)
    os.replace(tmp, path)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Flip strategy archive flag with Iron Law 6 enforcement.")
    p.add_argument("--decisions", default=str(DECISIONS_PATH))
    p.add_argument("--strategy", help="strategy name to flip")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--archive", action="store_true", help="set archived=true")
    grp.add_argument("--unarchive", action="store_true", help="set archived=false")
    p.add_argument("--list", action="store_true", help="print current state and exit")
    p.add_argument("--dry-run", action="store_true", help="don't write; just print what would change")
    args = p.parse_args(argv)

    path = Path(args.decisions)
    try:
        decisions = load_decisions(path)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if args.list:
        print(list_strategies(decisions))
        return 0

    if not args.strategy:
        print("ERROR: --strategy required (unless --list)", file=sys.stderr)
        return 1

    if not args.archive and not args.unarchive:
        print("ERROR: must specify --archive or --unarchive", file=sys.stderr)
        return 1

    archive = bool(args.archive)
    changed, msg = apply_flip(decisions, args.strategy, archive)

    print(msg)
    if not changed:
        return 0 if "no-op" in msg else 2

    if args.dry_run:
        print("(--dry-run: no write performed)")
        return 0

    atomic_write(path, decisions)
    print(f"wrote {path}; active count now: {count_active(decisions)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

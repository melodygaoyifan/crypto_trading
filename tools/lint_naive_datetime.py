#!/usr/bin/env python3
"""
lint_naive_datetime.py — flags naive datetime construction
================================================================

[P111 Tier2#4 2026-04-27] Static scanner for the most common bug class
caught during the full-codebase audit (P90→P110): ~25 sites used
`datetime.utcnow()` or bare `datetime.now()` in dataclass defaults
or comparisons against tz-aware datetimes — silently TypeError'd
inside try blocks or produced wrong-by-N-hours arithmetic.

Same shape as `tools/lint_silent_swallow.py` (P72): AST scan,
prints findings + counts, integrates with `ci_check_invariants.py`
for baseline drift detection.

Patterns detected:

  A) `datetime.utcnow()` — deprecated in Python 3.12+, returns naive.
     The fix is `datetime.now(timezone.utc)` (or
     `lambda: datetime.now(timezone.utc)` in dataclass defaults).

  B) `datetime.now()` with NO ARGS in dataclass `default_factory=` or
     in code that compares with tz-aware datetime. The bare form is
     local-tz-naive; should be `datetime.now(timezone.utc)`.

  C) `field(default_factory=datetime.utcnow)` and
     `field(default_factory=datetime.now)` — these silently produce
     naive defaults forever.

Per-line opt-out: `# noqa: naive-datetime` on the same line.
File-level opt-out: `# lint_naive_datetime: skip` anywhere in file.

CLI:
    python tools/lint_naive_datetime.py                    # scan default LIVE_DIRS
    python tools/lint_naive_datetime.py path/to/file.py    # scan specific path(s)
    python tools/lint_naive_datetime.py --staged           # only git-staged files
    python tools/lint_naive_datetime.py --json             # machine-readable output
    python tools/lint_naive_datetime.py --baseline-format  # for ci_check_invariants
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# Same scope as lint_silent_swallow.py for consistency.
LIVE_DIRS = [
    "agents", "analytics", "core", "data_mgmt", "defense", "drl",
    "execution", "infra", "integration", "liquidity", "market",
    "orchestration", "risk", "signals", "strategies", "tools",
]


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str  # "utcnow_call" | "naive_now" | "default_factory_naive"
    snippet: str


def _file_skips(path: Path) -> bool:
    """File-level opt-out via `# lint_naive_datetime: skip` anywhere."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if "# lint_naive_datetime: skip" in line:
                    return True
                # cheap early-out — directive should be near top
                if not line.strip().startswith("#") and "import" not in line:
                    return False
    except OSError:
        return True
    return False


def _line_has_noqa(source_lines: List[str], lineno: int) -> bool:
    """Per-line opt-out check."""
    if 0 < lineno <= len(source_lines):
        return "# noqa: naive-datetime" in source_lines[lineno - 1]
    return False


def _walk_ast(path: Path) -> List[Finding]:
    """Walk AST of a single file looking for the 3 bug shapes."""
    if _file_skips(path):
        return []
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    source_lines = source.splitlines()
    findings: List[Finding] = []

    rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")

    for node in ast.walk(tree):
        # Pattern A + B: datetime.utcnow() and datetime.now() with NO args
        if isinstance(node, ast.Call):
            func = node.func
            # Match `datetime.utcnow(...)` or `datetime.now(...)` etc.
            if isinstance(func, ast.Attribute):
                attr = func.attr
                # Only catch when the receiver is `datetime` (not e.g. `time`)
                base = func.value
                base_name = (
                    base.id if isinstance(base, ast.Name) else
                    (getattr(base, "attr", None) if isinstance(base, ast.Attribute) else None)
                )
                if base_name != "datetime":
                    continue
                if attr == "utcnow":
                    line = node.lineno
                    if not _line_has_noqa(source_lines, line):
                        findings.append(Finding(
                            path=rel, line=line, kind="utcnow_call",
                            snippet=source_lines[line - 1].strip()[:120]
                        ))
                elif attr == "now":
                    # Bare `datetime.now()` (no args) only.
                    if len(node.args) == 0 and len(node.keywords) == 0:
                        line = node.lineno
                        if not _line_has_noqa(source_lines, line):
                            findings.append(Finding(
                                path=rel, line=line, kind="naive_now",
                                snippet=source_lines[line - 1].strip()[:120]
                            ))
        # Pattern C: field(default_factory=datetime.utcnow) — note no parens
        # i.e. passing the function reference itself, not calling it
        if isinstance(node, ast.keyword) and node.arg == "default_factory":
            val = node.value
            if isinstance(val, ast.Attribute):
                base = val.value
                base_name = (
                    base.id if isinstance(base, ast.Name) else
                    (getattr(base, "attr", None) if isinstance(base, ast.Attribute) else None)
                )
                if base_name == "datetime" and val.attr in ("utcnow", "now"):
                    line = node.value.lineno
                    if not _line_has_noqa(source_lines, line):
                        findings.append(Finding(
                            path=rel, line=line, kind="default_factory_naive",
                            snippet=source_lines[line - 1].strip()[:120]
                        ))
    return findings


def _staged_paths() -> List[Path]:
    """Files in git index (staged for commit)."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=REPO_ROOT, text=True
        )
        return [REPO_ROOT / p for p in out.splitlines() if p.endswith(".py")]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _gather_paths(args) -> List[Path]:
    if args.staged:
        return _staged_paths()
    if args.paths:
        out = []
        for p in args.paths:
            pp = Path(p)
            if pp.is_dir():
                out.extend(pp.rglob("*.py"))
            elif pp.exists():
                out.append(pp)
        return out
    paths: List[Path] = []
    for d in LIVE_DIRS:
        dp = REPO_ROOT / d
        if dp.exists():
            paths.extend(dp.rglob("*.py"))
    # Plus root-level main.py
    main_py = REPO_ROOT / "main.py"
    if main_py.exists():
        paths.append(main_py)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="Specific paths/dirs to scan")
    ap.add_argument("--staged", action="store_true", help="Only git-staged files")
    ap.add_argument("--json", action="store_true", help="Machine-readable output")
    ap.add_argument("--baseline-format", action="store_true",
                    help="Emit count-only summary for ci_check_invariants")
    args = ap.parse_args()

    paths = _gather_paths(args)
    all_findings: List[Finding] = []
    for p in paths:
        # Skip __pycache__, venv, archive
        rel_str = str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
        if any(skip in rel_str for skip in ("__pycache__", "venv", "archive", "tests")):
            continue
        all_findings.extend(_walk_ast(p))

    if args.baseline_format:
        by_kind: Dict[str, int] = {"utcnow_call": 0, "naive_now": 0, "default_factory_naive": 0}
        files_with: Set[str] = set()
        for f in all_findings:
            by_kind[f.kind] += 1
            files_with.add(f.path)
        out = {
            "by_kind": by_kind,
            "files_with_findings": len(files_with),
            "total_count": len(all_findings),
        }
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    if args.json:
        print(json.dumps([f.__dict__ for f in all_findings], indent=2))
        return 0

    if not all_findings:
        print("[lint_naive_datetime] 0 findings — clean.")
        return 0

    # Human readable
    print(f"[lint_naive_datetime] {len(all_findings)} findings:")
    by_kind: Dict[str, List[Finding]] = {}
    for f in all_findings:
        by_kind.setdefault(f.kind, []).append(f)
    for kind, items in sorted(by_kind.items()):
        print(f"\n  === {kind} ({len(items)}) ===")
        for f in items[:50]:
            print(f"    {f.path}:{f.line}  {f.snippet}")
        if len(items) > 50:
            print(f"    ... +{len(items) - 50} more")
    print(
        f"\nFix: replace `datetime.utcnow()` → `datetime.now(timezone.utc)`; "
        f"replace `datetime.now()` (no args) → `datetime.now(timezone.utc)`; "
        f"replace `default_factory=datetime.utcnow` → "
        f"`default_factory=lambda: datetime.now(timezone.utc)`. "
        f"Per-line opt-out: append `# noqa: naive-datetime`."
    )
    return 1 if all_findings else 0


if __name__ == "__main__":
    sys.exit(main())

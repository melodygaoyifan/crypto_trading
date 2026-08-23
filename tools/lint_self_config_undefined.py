#!/usr/bin/env python3
"""
lint_self_config_undefined.py — flag classes that READ self.config but never SET it
====================================================================================

[P111 Tier2#5 2026-04-27] AST-based scanner for the P101 bug shape:
classes that have `self.config.X` or `getattr(self.config, ...)`
ANYWHERE in their methods, but their `__init__` does NOT contain
`self.config = ...` (no fallback either via `or {}`, `or SomeConfig()`, etc).

Production impact (real bugs P101 caught):
  - data_mgmt/feeds/onchain_feed.py: silent fall-through to mock for
    EVERY Helius/Solana RPC call because self.config raised AttributeError
  - data_mgmt/feeds/sentiment_feed.py: same shape for LunarCrush

The bare `self.config` access raises AttributeError → caught by
outer try/except in caller → silently degraded behavior.

Usage:
    python tools/lint_self_config_undefined.py                # default LIVE_DIRS
    python tools/lint_self_config_undefined.py --json
    python tools/lint_self_config_undefined.py --baseline-format

Per-class opt-out: `# noqa: self-config-undefined` on the class def line
(useful for classes intentionally inheriting config from a mixin/parent).
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

LIVE_DIRS = [
    "agents", "analytics", "core", "data_mgmt", "defense", "drl",
    "execution", "infra", "integration", "liquidity", "market",
    "orchestration", "risk", "signals", "strategies", "tools",
    # [P382] the venue layer + API + portfolio packages were never scanned
    # (P366's LIVE_DIRS finding); coverage, not regression.
    "exchange", "api", "portfolio",
]


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    class_name: str
    reads: int  # how many times the class reads self.config
    snippet: str


def _is_self_config_read(node: ast.AST) -> bool:
    """Match `self.config` or `self.config.X` or `getattr(self.config, ...)`."""
    if isinstance(node, ast.Attribute):
        # self.config.foo → node.value is `self.config` Attribute
        v = node.value
        if isinstance(v, ast.Attribute):
            if (isinstance(v.value, ast.Name) and v.value.id == "self"
                    and v.attr == "config"):
                return True
        # bare self.config (top-level access)
        if (isinstance(v, ast.Name) and v.id == "self"
                and node.attr == "config"):
            # This is "self.config" being read but it's actually node IS self.config
            return False  # caught at upper-attribute level above
    if isinstance(node, ast.Call):
        if (isinstance(node.func, ast.Name) and node.func.id == "getattr"
                and len(node.args) >= 1):
            arg0 = node.args[0]
            if (isinstance(arg0, ast.Attribute)
                    and isinstance(arg0.value, ast.Name)
                    and arg0.value.id == "self"
                    and arg0.attr == "config"):
                return True
    return False


def _has_self_config_assign(method_body: List[ast.stmt]) -> bool:
    """True if method body has `self.config = ...` (anywhere)."""
    for node in ast.walk(ast.Module(body=method_body, type_ignores=[])):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == "config"):
                    return True
    return False


def _scan_class(cls: ast.ClassDef, source_lines: List[str]) -> Optional[Finding]:
    """Check one class. Returns Finding if it reads self.config but never sets it."""
    # File-level/class-level opt-out via noqa on class def line
    cls_line = cls.lineno
    if 0 < cls_line <= len(source_lines):
        if "# noqa: self-config-undefined" in source_lines[cls_line - 1]:
            return None

    # Find __init__ body (or any method that sets self.config)
    has_assign = False
    init_node = None
    for item in cls.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name == "__init__":
                init_node = item
            if _has_self_config_assign(item.body):
                has_assign = True
                break
    if has_assign:
        return None

    # Count self.config reads across all methods
    read_count = 0
    first_read_line = None
    for item in cls.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(item):
                if _is_self_config_read(sub):
                    read_count += 1
                    if first_read_line is None:
                        first_read_line = getattr(sub, "lineno", cls_line)
    if read_count == 0:
        return None

    snippet = source_lines[first_read_line - 1].strip()[:120] if first_read_line else ""
    return Finding(
        path="",  # filled by caller
        line=first_read_line or cls_line,
        class_name=cls.name,
        reads=read_count,
        snippet=snippet,
    )


def _scan_file(path: Path) -> List[Finding]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    source_lines = source.splitlines()
    rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            f = _scan_class(node, source_lines)
            if f is not None:
                out.append(Finding(
                    path=rel, line=f.line, class_name=f.class_name,
                    reads=f.reads, snippet=f.snippet,
                ))
    return out


def _gather_paths(args) -> List[Path]:
    if args.paths:
        out = []
        for p in args.paths:
            # [P382] resolve() first: a RELATIVE CLI path (e.g. `exchange`)
            # crashed in the per-file scan's `path.relative_to(REPO_ROOT)`.
            pp = Path(p).resolve()
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
    main_py = REPO_ROOT / "main.py"
    if main_py.exists():
        paths.append(main_py)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="Specific paths/dirs to scan")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--baseline-format", action="store_true")
    args = ap.parse_args()

    paths = _gather_paths(args)
    findings: List[Finding] = []
    for p in paths:
        rel = str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
        if any(skip in rel for skip in ("__pycache__", "venv", "archive", "tests")):
            continue
        findings.extend(_scan_file(p))

    if args.baseline_format:
        files_with: Set[str] = {f.path for f in findings}
        out = {
            "total_count": len(findings),
            "files_with_findings": len(files_with),
        }
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    if args.json:
        print(json.dumps([f.__dict__ for f in findings], indent=2))
        return 0

    if not findings:
        print("[lint_self_config_undefined] 0 findings — clean.")
        return 0

    print(f"[lint_self_config_undefined] {len(findings)} class(es) read "
          f"`self.config` but never set it:")
    for f in findings:
        print(f"  {f.path}:{f.line}  class {f.class_name}  "
              f"({f.reads} read sites)  {f.snippet}")
    print(
        f"\nFix: add `self.config = config or YourConfig()` (or similar) to "
        f"__init__. Per-class opt-out: append `# noqa: self-config-undefined` "
        f"on the class def line (e.g. for classes inheriting config from a "
        f"mixin/parent)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

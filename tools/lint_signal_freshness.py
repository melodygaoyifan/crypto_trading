#!/usr/bin/env python3
"""
lint_signal_freshness.py — agent_signals[X] writer freshness audit (P120)
================================================================================

Inventories every `agent_signals["<key>"] = <value>` write site in the
codebase and classifies its freshness-guard level. Three categories:

  GUARDED   — write is gated by an `if <freshness check>:` (e.g. timestamp
              age, data_quality > 0, signal.is_valid, market_data.fresh).
              Pre-write check would skip stale data.

  TIMESTAMPED — write is unconditional but the WRITER attaches a per-signal
                timestamp via `*_data_quality` or `*_timestamp` key. Reader
                CAN check freshness even though writer doesn't.

  BLIND     — write is unconditional + no per-signal timestamp. Reader has
              no way to detect stale data; the value persists across ticks
              and silently propagates.

CLAUDE.md P68 D3 documents the gap: "Fusion can't distinguish fresh vs stale
signals. Current `agent_signals['_signal_timestamp']` is global. Per-key
marker would be a design change with downstream consumer impact."

This scanner makes the gap MEASURABLE — we have a count of BLIND writers
that can only decrease over time. CI gate forces deliberate decisions.

Usage:
    python -X utf8 tools/lint_signal_freshness.py
    python -X utf8 tools/lint_signal_freshness.py --json
    python -X utf8 tools/lint_signal_freshness.py paths main.py integration/
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Files / dirs to scan by default — match the live tree
DEFAULT_SCAN = [
    "main.py",
    "integration",
    "agents",
    "core",
    "data_mgmt",
    "risk",
    "signals",
]


# Heuristics: phrases that indicate a freshness check in surrounding code
FRESHNESS_GUARD_HINTS = (
    "_signal_timestamp",
    "_timestamp",
    "asof_timestamp",
    "data_quality",
    "is_valid",
    "is_fresh",
    "max_age",
    "MAX_DATA_AGE",
    "stale",
    "freshness",
    "_data_quality",
    "fresh=True",
    "data_age_seconds",
)

# Keys that ARE timestamp/quality markers themselves — writing these is
# documenting freshness, not a bare value write
META_KEY_SUFFIXES = (
    "_timestamp",
    "_data_quality",
    "_age",
    "_is_valid",
    "_freshness",
    "_signal_timestamp",
    "_reconnect_grace",
)


class SignalWriteVisitor(ast.NodeVisitor):
    """Find `agent_signals[<key>] = <expr>` and `agent_signals.update(...)`
    write sites + their surrounding control-flow context."""

    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.writes: List[Dict] = []
        self.context_stack: List[ast.AST] = []

    def generic_visit(self, node):
        # Track the enclosing FunctionDef / If / Try / For / With for context
        is_block = isinstance(node, (
            ast.FunctionDef, ast.AsyncFunctionDef,
            ast.If, ast.Try, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith,
        ))
        if is_block:
            self.context_stack.append(node)
        super().generic_visit(node)
        if is_block:
            self.context_stack.pop()

    def visit_Assign(self, node: ast.Assign):
        for tgt in node.targets:
            if not isinstance(tgt, ast.Subscript):
                continue
            # Match: agent_signals[<key>] = <expr>
            if not (isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "agent_signals"):
                continue
            key = self._extract_key(tgt.slice)
            if key is None:
                key = "<dynamic>"
            if any(key.endswith(s) for s in META_KEY_SUFFIXES):
                continue
            category, evidence = self._classify_freshness(node)
            self.writes.append({
                "key": key, "line": node.lineno,
                "category": category, "evidence": evidence,
            })
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Catch agent_signals.update({...}) and agent_signals.setdefault(...)."""
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "agent_signals"):
            method = node.func.attr
            if method in ("update", "setdefault"):
                category, evidence = self._classify_freshness(node)
                if method == "update" and node.args and isinstance(node.args[0], ast.Dict):
                    # Extract literal keys from the dict
                    for k_node in node.args[0].keys:
                        if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
                            key = k_node.value
                            if any(key.endswith(s) for s in META_KEY_SUFFIXES):
                                continue
                            self.writes.append({
                                "key": key, "line": node.lineno,
                                "category": category,
                                "evidence": f"via .update() — {evidence}",
                            })
                elif method == "setdefault" and node.args:
                    k_node = node.args[0]
                    if isinstance(k_node, ast.Constant) and isinstance(k_node.value, str):
                        key = k_node.value
                        if not any(key.endswith(s) for s in META_KEY_SUFFIXES):
                            self.writes.append({
                                "key": key, "line": node.lineno,
                                "category": category,
                                "evidence": f"via .setdefault() — {evidence}",
                            })
                else:
                    # Dynamic .update(some_dict) — record without key
                    self.writes.append({
                        "key": "<dynamic-update>", "line": node.lineno,
                        "category": category,
                        "evidence": f"via .{method}() — {evidence}",
                    })
        self.generic_visit(node)

    def _extract_key(self, slice_node) -> str | None:
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return slice_node.value
        # Python <3.9 used ast.Index wrapper
        if hasattr(slice_node, "value") and isinstance(getattr(slice_node, "value", None), ast.Constant):
            v = slice_node.value
            if isinstance(v.value, str):
                return v.value
        return None

    def _classify_freshness(self, node: ast.Assign) -> Tuple[str, str]:
        """Look at:
          (a) the enclosing If statement test for freshness hints
          (b) the assignment value for `*.is_valid`, `*.fresh`, etc.
          (c) any nearby write of a *_timestamp or *_data_quality key
        """
        # (a) Check enclosing control-flow for guard hints
        for ctx in reversed(self.context_stack):
            if isinstance(ctx, ast.If):
                test_src = ast.unparse(ctx.test)
                for hint in FRESHNESS_GUARD_HINTS:
                    if hint in test_src:
                        return "GUARDED", f"if-test contains '{hint}'"
            if isinstance(ctx, ast.Try):
                # Try body — check if there's a freshness test inside
                pass

        # (b) Check the RHS for object.is_valid / object.fresh / etc.
        try:
            rhs_src = ast.unparse(node.value)
        except Exception:
            rhs_src = ""
        for hint in FRESHNESS_GUARD_HINTS:
            if hint in rhs_src:
                return "GUARDED", f"RHS contains '{hint}'"

        # (c) Check sibling lines (within ~10 lines above) for a sibling
        #     *_data_quality or *_timestamp write — indicates writer is
        #     timestamping its output even if write itself is unconditional
        line_no = node.lineno
        start = max(0, line_no - 10)
        window = "\n".join(self.source_lines[start:line_no + 5])
        for suffix in META_KEY_SUFFIXES:
            if f'agent_signals["' in window or f"agent_signals['" in window:
                # Look for a line like: agent_signals["X_data_quality"] = ...
                for line in window.splitlines():
                    if "agent_signals" in line and any(
                        f'{s}"]' in line or f"{s}']" in line
                        for s in META_KEY_SUFFIXES
                    ):
                        return "TIMESTAMPED", "sibling *_timestamp/*_data_quality write nearby"

        return "BLIND", "no freshness guard or per-signal timestamp"


def scan_path(root: Path) -> List[Dict]:
    """Walk a path (file or dir) and return all writes."""
    results: List[Dict] = []
    paths: List[Path] = []
    if root.is_file() and root.suffix == ".py":
        paths.append(root)
    elif root.is_dir():
        paths.extend(root.rglob("*.py"))

    for path in paths:
        # Skip __pycache__, archive, tests
        sp = path.as_posix()
        if any(skip in sp for skip in ("__pycache__", "/archive/", "\\archive\\",
                                        "/tests/", "\\tests\\", "/scripts/",
                                        "\\scripts\\", "/tools/", "\\tools\\")):
            continue
        try:
            # utf-8-sig strips a leading BOM if present (main.py has one).
            # Without this the file silently failed to parse and was skipped.
            src = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(src, filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"[skip] {path}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        visitor = SignalWriteVisitor(src.splitlines())
        visitor.visit(tree)
        for w in visitor.writes:
            w["file"] = str(path.relative_to(Path.cwd())) if path.is_absolute() else str(path)
            results.append(w)
    return results


def summarize(writes: List[Dict]) -> Dict:
    """Group writes by category + key for reporting."""
    by_cat = defaultdict(int)
    by_key_cat: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for w in writes:
        by_cat[w["category"]] += 1
        by_key_cat[w["key"]][w["category"]] += 1

    # Surface BLIND keys for operator review
    blind_keys = sorted([
        k for k, cats in by_key_cat.items()
        if cats.get("BLIND", 0) > 0 and cats.get("GUARDED", 0) == 0 and cats.get("TIMESTAMPED", 0) == 0
    ])

    return {
        "total_writes": len(writes),
        "by_category": dict(by_cat),
        "blind_only_keys_count": len(blind_keys),
        "blind_only_keys": blind_keys,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="Files/dirs to scan (default: live tree)")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--baseline-format", action="store_true",
                        help="Emit only the count summary for ci_check_invariants.py diff")
    parser.add_argument("--show-all", action="store_true", help="List every write site")
    args = parser.parse_args()

    paths = args.paths or DEFAULT_SCAN
    all_writes: List[Dict] = []
    for p in paths:
        all_writes.extend(scan_path(Path(p)))

    summary = summarize(all_writes)

    if args.baseline_format:
        # Stable shape for baseline diff — counts only, no per-key churn
        out = {
            "total_writes": summary["total_writes"],
            "by_category": summary["by_category"],
            "blind_only_keys_count": summary["blind_only_keys_count"],
        }
        print(json.dumps(out, indent=2, sort_keys=True))
        sys.exit(0)

    if args.json:
        print(json.dumps({"summary": summary, "writes": all_writes if args.show_all else None},
                         indent=2))
        sys.exit(0)

    print(f"agent_signals[X] = ... write-site freshness audit")
    print(f"=" * 70)
    print(f"Total writes:        {summary['total_writes']}")
    for cat in ("GUARDED", "TIMESTAMPED", "BLIND"):
        print(f"  {cat:12s}: {summary['by_category'].get(cat, 0)}")
    print()
    print(f"Blind-only keys (no freshness guard ANYWHERE in writers): "
          f"{summary['blind_only_keys_count']}")
    if summary["blind_only_keys"]:
        for k in summary["blind_only_keys"][:30]:
            sites = [w for w in all_writes if w["key"] == k]
            print(f"  {k:35s} ({len(sites)} write site{'s' if len(sites) > 1 else ''})")
            if args.show_all:
                for s in sites[:3]:
                    print(f"      {s['file']}:{s['line']} — {s['evidence']}")

    sys.exit(0)


if __name__ == "__main__":
    main()

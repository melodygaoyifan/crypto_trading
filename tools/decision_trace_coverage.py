#!/usr/bin/env python3
"""
decision_trace_coverage.py — gate/branch test-coverage gap report (P121)
================================================================================

Walks the static call graph from the live decision pipeline and reports which
GATE outcomes are reachable in source but NEVER asserted-upon in tests.

Currently covers:
  - defense/trade_gate.py — every RejectReason firing site
  - signals/authority_fusion.py — every veto label written to vetoes_active

For each gate outcome:
  REACHABLE  — the source contains code that produces this outcome
  COVERED    — at least one test file references the outcome (string or enum)
  GAP        — REACHABLE but no test references it

Output is a coverage matrix + GAP list. CI gate semantics: gap count can
DECREASE freely; INCREASE blocks (a refactor that adds a new reject without
a covering test should fail CI).

Usage:
    python -X utf8 tools/decision_trace_coverage.py
    python -X utf8 tools/decision_trace_coverage.py --json
    python -X utf8 tools/decision_trace_coverage.py --baseline-format
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent

# (module_relative_path, enum_class_name, scan_call_pattern)
GATE_SOURCES = [
    {
        "name": "trade_gate",
        "source_file": "defense/trade_gate.py",
        "enum_class": "RejectReason",
        "call_attr": "_reject",
        "fail_tier": "REJECT",
    },
]

# For authority_fusion, the "outcomes" are veto LABELS appended to a list,
# not enum values. We scan for string literals appearing in
# `vetoes_active.append(...)` or `vetoes_active=[...]` constructions.
FUSION_VETO_FILE = "signals/authority_fusion.py"

# Test files to scan for coverage
TEST_DIRS = ["tests"]


def _read(path: Path) -> str:
    """Read with BOM-safe encoding (P120 lesson)."""
    return path.read_text(encoding="utf-8-sig")


def _enum_members(source_path: Path, enum_name: str) -> Set[str]:
    """Parse the source and extract enum member names."""
    try:
        tree = ast.parse(_read(source_path))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    members: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == enum_name:
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for tgt in item.targets:
                        if isinstance(tgt, ast.Name):
                            members.add(tgt.id)
    members.discard("NONE")  # NONE is the not-rejected sentinel
    return members


def _reject_call_sites(source_path: Path, call_attr: str, enum_name: str) -> Set[str]:
    """Find every `self._reject(RejectReason.X, ...)` and return the set of X."""
    try:
        tree = ast.parse(_read(source_path))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    fired: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match: <anything>._reject(RejectReason.X, ...) OR self._reject(...)
        is_reject_call = (
            isinstance(node.func, ast.Attribute) and node.func.attr == call_attr
        )
        if not is_reject_call:
            continue
        if not node.args:
            continue
        arg0 = node.args[0]
        # arg0 is RejectReason.X
        if (isinstance(arg0, ast.Attribute)
                and isinstance(arg0.value, ast.Name)
                and arg0.value.id == enum_name):
            fired.add(arg0.attr)
        # arg0 might be a variable (`reject_reason`) — we can't statically
        # resolve which enum value it'll be. Mark as "<dynamic>".
        elif isinstance(arg0, ast.Name):
            fired.add("<dynamic>")
    return fired


def _veto_strings_in_fusion(source_path: Path) -> Set[str]:
    """Extract every string literal appearing in vetoes_active assignments
    or appends. Scans for both:
        vetoes_active=["X"]
        vetoes_active.append("X")
    """
    try:
        src = _read(source_path)
        tree = ast.parse(src)
    except (SyntaxError, UnicodeDecodeError):
        return set()
    found: Set[str] = set()

    for node in ast.walk(tree):
        # vetoes_active=[...] kwarg in FusionResult constructor
        if isinstance(node, ast.keyword) and node.arg == "vetoes_active":
            if isinstance(node.value, ast.List):
                for el in node.value.elts:
                    if isinstance(el, ast.Constant) and isinstance(el.value, str):
                        found.add(el.value)
        # vetoes_active.append("X")
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Attribute) and f.attr == "append"
                    and isinstance(f.value, ast.Attribute)
                    and f.value.attr == "vetoes_active"):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        found.add(arg.value)
    return found


def _test_corpus() -> str:
    """Concatenate every test file's text — test references can be string or
    enum-name; we only need substring search."""
    chunks: List[str] = []
    for d in TEST_DIRS:
        root = REPO_ROOT / d
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                chunks.append(_read(path))
            except (UnicodeDecodeError, OSError):
                continue
    return "\n".join(chunks)


def coverage_report() -> Dict:
    test_text = _test_corpus()

    report = {
        "trade_gate": {"reasons": {}, "summary": {}},
        "authority_fusion": {"veto_labels": {}, "summary": {}},
    }

    # ---- trade_gate ----
    for spec in GATE_SOURCES:
        src_path = REPO_ROOT / spec["source_file"]
        all_members = _enum_members(src_path, spec["enum_class"])
        fired = _reject_call_sites(src_path, spec["call_attr"], spec["enum_class"])
        # Heuristic for "<dynamic>" — assume covers everything (we can't
        # statically resolve, so don't overreport gaps)
        dynamic_match = "<dynamic>" in fired
        fired_concrete = fired - {"<dynamic>"}

        per_reason: Dict[str, Dict] = {}
        for member in sorted(all_members):
            is_reachable = member in fired_concrete or dynamic_match
            # Coverage: search test corpus for the member name
            is_covered = (
                f".{member}" in test_text
                or f'"{member}"' in test_text
                or f"'{member}'" in test_text
            )
            per_reason[member] = {
                "reachable": is_reachable,
                "concretely_fired": member in fired_concrete,
                "covered": is_covered,
                "gap": is_reachable and not is_covered,
            }
        report["trade_gate"]["reasons"] = per_reason
        report["trade_gate"]["summary"] = {
            "total_members": len(all_members),
            "reachable": sum(1 for r in per_reason.values() if r["reachable"]),
            "concretely_fired": sum(1 for r in per_reason.values() if r["concretely_fired"]),
            "covered_by_tests": sum(1 for r in per_reason.values() if r["covered"]),
            "coverage_gaps": sum(1 for r in per_reason.values() if r["gap"]),
            "dynamic_dispatch_present": dynamic_match,
        }

    # ---- authority_fusion ----
    fusion_path = REPO_ROOT / FUSION_VETO_FILE
    veto_labels = _veto_strings_in_fusion(fusion_path)
    per_label: Dict[str, Dict] = {}
    for label in sorted(veto_labels):
        is_covered = (
            f'"{label}"' in test_text
            or f"'{label}'" in test_text
        )
        per_label[label] = {
            "reachable": True,  # if it appears in source, it's reachable
            "covered": is_covered,
            "gap": not is_covered,
        }
    report["authority_fusion"]["veto_labels"] = per_label
    report["authority_fusion"]["summary"] = {
        "total_labels": len(veto_labels),
        "covered_by_tests": sum(1 for r in per_label.values() if r["covered"]),
        "coverage_gaps": sum(1 for r in per_label.values() if r["gap"]),
    }

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--baseline-format", action="store_true",
                        help="Stable counts-only output for ci_check_invariants.py")
    args = parser.parse_args()

    report = coverage_report()

    if args.baseline_format:
        out = {
            "trade_gate_gaps": report["trade_gate"]["summary"]["coverage_gaps"],
            "fusion_veto_gaps": report["authority_fusion"]["summary"]["coverage_gaps"],
        }
        print(json.dumps(out, indent=2, sort_keys=True))
        sys.exit(0)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        sys.exit(0)

    print("decision-trace coverage report")
    print("=" * 70)
    tg = report["trade_gate"]["summary"]
    print(f"\n[trade_gate.RejectReason]")
    print(f"  Total enum members:     {tg['total_members']}")
    print(f"  Reachable in source:    {tg['reachable']}")
    print(f"  Concretely fired:       {tg['concretely_fired']}")
    print(f"  Covered by tests:       {tg['covered_by_tests']}")
    print(f"  COVERAGE GAPS:          {tg['coverage_gaps']}")
    if tg["dynamic_dispatch_present"]:
        print(f"  (dynamic dispatch present — concrete fires under-reported)")

    gap_reasons = sorted([
        name for name, info in report["trade_gate"]["reasons"].items()
        if info["gap"]
    ])
    if gap_reasons:
        print(f"\n  Reasons reachable but NOT tested:")
        for r in gap_reasons:
            info = report["trade_gate"]["reasons"][r]
            mark = "*" if info["concretely_fired"] else " "
            print(f"    {mark} {r}")
        print(f"  (* = concrete fire site found in source)")

    af = report["authority_fusion"]["summary"]
    print(f"\n[authority_fusion vetoes_active labels]")
    print(f"  Total labels:           {af['total_labels']}")
    print(f"  Covered by tests:       {af['covered_by_tests']}")
    print(f"  COVERAGE GAPS:          {af['coverage_gaps']}")
    gap_labels = sorted([
        name for name, info in report["authority_fusion"]["veto_labels"].items()
        if info["gap"]
    ])
    if gap_labels:
        print(f"\n  Veto labels reachable but NOT tested:")
        for l in gap_labels:
            print(f"    {l}")

    sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════
HMATS V8 - Full Dead Code Audit
════════════════════════════════════════════════════════════════════

审查所有非 __pycache__ 的 .py 文件，输出:
  1. 每个文件的 live/dead 状态（基于 import graph）
  2. Dead 文件的内容摘要（docstring, classes, functions）
  3. 是否包含未接入但有价值的功能
  4. 建议: DELETE / KEEP(future) / WIRE(应接入)

Usage:
    cd C:\\Users\\melod\\Downloads\\hmats
    python -X utf8 scripts/audit_dead_code.py
    python -X utf8 scripts/audit_dead_code.py --verbose
    python -X utf8 scripts/audit_dead_code.py --output audit_report.json
"""

import ast
import json
import os
import sys
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# ═══════════════════════════════════════════════════════════════
# STEP 1: Build complete import graph from ALL entry points
# ═══════════════════════════════════════════════════════════════

ENTRY_POINTS = [
    "main.py",
    "integration/integration_v36.py",
]

# 独立入口（不被 main.py import，但各自独立运行）
INDEPENDENT_ENTRIES = [
    "training/train_drl_full.py",
    "training/train_drl_per_regime.py",
    "training/scripts/rebuild_pipeline.py",
    "training/scripts/compare_architectures.py",
    "training/scripts/generate_split_manifest.py",
    "training/scripts/train_per_asset_gmm.py",
    "training/scripts/fetch_coinglass_history.py",
    "training/scripts/wavelet_denoise.py",
    "training/scripts/run_stage7_comparison.py",
    "training/scripts/run_stage7_nav_retest.py",
    "training/scripts/stage7_diagnostic_eval.py",
    "training/run_training.py",
    "scripts/launch_paper.py",
    "scripts/paper_run_monitor.py",
    "scripts/pre_flight_check.py",
]

# Dashboard（streamlit 独立启动）
DASHBOARD_ENTRIES = []
dashboard_dir = ROOT / "dashboard"
if dashboard_dir.exists():
    DASHBOARD_ENTRIES = [str(p.relative_to(ROOT)) for p in dashboard_dir.glob("*.py")
                         if p.name != "__init__.py"]

# Tests（pytest 独立运行）
TEST_DIR = ROOT / "tests"

SKIP_DIRS = {"venv", "venv_cuda", "pytorch_build", "__pycache__", ".git", "node_modules"}
STDLIB_PREFIXES = (
    "os", "sys", "time", "json", "math", "random", "copy", "re", "io",
    "abc", "ast", "csv", "enum", "glob", "shutil", "struct", "socket",
    "hashlib", "hmac", "base64", "urllib", "http", "email", "html",
    "logging", "pathlib", "datetime", "typing", "collections",
    "functools", "itertools", "operator", "contextlib", "dataclasses",
    "threading", "multiprocessing", "queue", "subprocess", "signal",
    "traceback", "inspect", "importlib", "pkgutil", "warnings",
    "unittest", "pytest", "argparse", "textwrap", "string", "decimal",
    "fractions", "statistics", "bisect", "heapq", "array",
    "pprint", "tempfile", "pickle", "shelve", "sqlite3",
    "concurrent", "asyncio",
)
THIRDPARTY_PREFIXES = (
    "numpy", "np", "pandas", "pd", "torch", "tensorflow", "tf",
    "stable_baselines3", "sb3_contrib", "sklearn", "scipy", "optuna",
    "gymnasium", "gym", "streamlit", "plotly", "matplotlib",
    "requests", "websocket", "websockets", "aiohttp",
    "joblib", "tqdm", "yaml", "toml", "dotenv",
    "ta", "talib", "ccxt", "krakenex", "pywt",
    "psutil", "GPUtil", "colorama", "rich",
)


def get_imports(filepath):
    """Extract local imports from a Python file."""
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
            # Handle relative imports
            if node.level and node.level > 0 and node.module:
                imports.append(node.module)

    # Filter to local imports only
    local = []
    for imp in imports:
        top = imp.split(".")[0]
        if top.startswith(STDLIB_PREFIXES) or top.startswith(THIRDPARTY_PREFIXES):
            continue
        local.append(imp)
    return local


def resolve_import_to_files(imp, search_root=ROOT):
    """Resolve a dotted import to possible file paths."""
    parts = imp.split(".")
    candidates = []

    # Try as module: a.b.c -> a/b/c.py
    path = search_root / "/".join(parts)
    candidates.append(path.with_suffix(".py"))

    # Try as package: a.b.c -> a/b/c/__init__.py
    candidates.append(path / "__init__.py")

    # Try partial: a.b -> a/b.py (importing a.b.SomeClass)
    for i in range(len(parts) - 1, 0, -1):
        partial = search_root / "/".join(parts[:i])
        candidates.append(partial.with_suffix(".py"))
        candidates.append(partial / "__init__.py")

    return [c for c in candidates if c.exists()]


def _normalize(rel_path):
    """Normalize path separators to forward slashes for consistent comparison."""
    return str(rel_path).replace("\\", "/")


def build_reachability(entry_points):
    """BFS from entry points, return set of reachable file paths (relative, forward-slash)."""
    visited = set()
    queue = []

    for ep in entry_points:
        ep_path = ROOT / ep
        if ep_path.exists():
            queue.append(ep_path)

    while queue:
        current = queue.pop(0)
        rel = _normalize(current.relative_to(ROOT))
        if rel in visited:
            continue
        visited.add(rel)

        imports = get_imports(current)
        for imp in imports:
            resolved = resolve_import_to_files(imp)
            for r in resolved:
                r_rel = _normalize(r.relative_to(ROOT))
                if r_rel not in visited:
                    queue.append(r)

    return visited


# ═══════════════════════════════════════════════════════════════
# STEP 2: Analyze each file
# ═══════════════════════════════════════════════════════════════

def analyze_file(filepath):
    """Deep analysis of a single Python file."""
    info = {
        "path": str(filepath),
        "lines": 0,
        "modified": "unknown",
        "docstring": "",
        "classes": [],
        "functions": [],
        "markers": {
            "deprecated": False,
            "todo": False,
            "fixme": False,
            "v3_only": False,
            "placeholder": False,
            "pass_only": False,
        },
        "imports_from": [],  # what this file imports
        "imported_by": [],   # who imports this file (filled later)
        "keywords": [],      # domain keywords for value assessment
    }

    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")
        info["lines"] = len(lines)

        # Modified time
        mtime = os.path.getmtime(filepath)
        info["modified"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

        # Docstring (module-level)
        try:
            tree = ast.parse(content)
            if (tree.body and isinstance(tree.body[0], ast.Expr)
                    and isinstance(tree.body[0].value, (ast.Constant, ast.Str))):
                doc = tree.body[0].value
                info["docstring"] = (doc.value if isinstance(doc, ast.Constant)
                                     else doc.s)[:300]

            # Classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    info["classes"].append(node.name)
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    info["functions"].append(node.name)
            info["functions"] = info["functions"][:15]  # cap
        except:
            pass

        # Markers
        upper = content.upper()
        info["markers"]["deprecated"] = "DEPRECATED" in upper or "DEPRECATE" in upper
        info["markers"]["todo"] = "TODO" in upper
        info["markers"]["fixme"] = "FIXME" in upper
        info["markers"]["placeholder"] = upper.count("PASS") > len(lines) * 0.1
        info["markers"]["pass_only"] = (
            len([l for l in lines if l.strip() and l.strip() != "pass"
                 and not l.strip().startswith("#")
                 and not l.strip().startswith("\"\"\"")
                 and not l.strip().startswith("'''")]) < 5
        )

        # v3-only reference (no v5/v6/v7/v8 references)
        lower = content.lower()
        has_v3 = any(x in lower for x in ["v3.", "v3_", "v31", "v32", "v33", "v34", "v35", "v36"])
        has_newer = any(x in lower for x in ["v5", "v6", "v7", "v8"])
        info["markers"]["v3_only"] = has_v3 and not has_newer

        # Domain keywords for value assessment
        VALUABLE_KEYWORDS = {
            "twap": "高级执行算法",
            "vwap": "高级执行算法",
            "iceberg": "高级执行算法",
            "weekend": "周末风险管理",
            "holiday": "节假日风险管理",
            "ensemble": "模型集成 (Stage 11)",
            "regime_router": "Per-regime routing",
            "shadow": "Shadow mode 观察",
            "promotion": "DRL promotion",
            "lead_lag": "Lead-lag 信号",
            "market_impact": "市场冲击模型",
            "correlation": "相关性监控",
            "funding_rate": "资金费率",
            "liquidation": "清算监控",
            "volatility_target": "波动率目标",
        }

        for kw, desc in VALUABLE_KEYWORDS.items():
            if kw in lower:
                info["keywords"].append(f"{kw} ({desc})")

        # Local imports
        info["imports_from"] = get_imports(filepath)

    except Exception as e:
        info["error"] = str(e)

    return info


# ═══════════════════════════════════════════════════════════════
# STEP 3: Generate verdicts
# ═══════════════════════════════════════════════════════════════

def verdict(info, is_live, is_independent, is_test, is_dashboard):
    """Generate action recommendation."""

    if is_live:
        return "ON LIVE", "活代码 - 从 main.py 可达"
    if is_independent:
        return "🔧 INDEPENDENT", "独立入口 - training/scripts/dashboard"
    if is_test:
        return "🧪 TEST", "测试文件"
    if is_dashboard:
        return "📊 DASHBOARD", "Streamlit UI - 独立启动"

    # Dead code analysis
    markers = info.get("markers", {})
    lines = info.get("lines", 0)
    keywords = info.get("keywords", [])
    classes = info.get("classes", [])

    # Definitely delete
    if markers.get("pass_only") or lines < 5:
        return "🗑️ DELETE", "空文件/stub"
    if markers.get("deprecated"):
        return "🗑️ DELETE", "标记 DEPRECATED"

    # Has valuable functionality
    if keywords:
        kw_str = ", ".join(keywords[:3])
        return "🔌 WIRE?", f"有价值功能未接入: {kw_str}"

    # v3-only, likely outdated
    if markers.get("v3_only"):
        return "🗑️ LIKELY DELETE", "仅引用 v3，可能过时"

    # Has substantial code but no clear value signal
    if lines > 200 and classes:
        return "🔍 REVIEW", f"{lines} lines, {len(classes)} classes - 需要人工判断"
    if lines > 100:
        return "🔍 REVIEW", f"{lines} lines - 需要人工判断"

    # Small dead files
    if lines < 50:
        return "🗑️ LIKELY DELETE", f"小文件 ({lines} lines), 0 imports"

    return "🔍 REVIEW", f"{lines} lines - 需要人工判断"


# ═══════════════════════════════════════════════════════════════
# STEP 4: Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HMATS Dead Code Audit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full details for each file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save full report as JSON")
    parser.add_argument("--dir", "-d", type=str, default=None,
                        help="Only audit specific directory")
    args = parser.parse_args()

    print("═" * 70)
    print("  HMATS V8 - Full Dead Code Audit")
    print("═" * 70)

    # 1. Build reachability from main runtime
    print("\n[1/4] Building import graph from main.py...")
    live_files = build_reachability(ENTRY_POINTS)
    print(f"  Runtime reachable: {len(live_files)} files")

    # 2. Build reachability from independent entries
    print("[2/4] Building import graph from independent entries...")
    independent_files = build_reachability(INDEPENDENT_ENTRIES)
    # Remove overlap with live
    independent_only = independent_files - live_files
    print(f"  Independent-only: {len(independent_only)} files")

    # 3. Dashboard files
    dashboard_files = set()
    for dp in DASHBOARD_ENTRIES:
        dashboard_files.add(dp)
        dashboard_files |= build_reachability([dp])
    dashboard_only = dashboard_files - live_files - independent_only
    print(f"  Dashboard-only: {len(dashboard_only)} files")

    # 4. Test files
    test_files = set()
    if TEST_DIR.exists():
        for f in TEST_DIR.rglob("*.py"):
            test_files.add(_normalize(f.relative_to(ROOT)))
    print(f"  Test files: {len(test_files)} files")

    # 5. Collect ALL .py files
    print("[3/4] Scanning all Python files...")
    all_files = []
    for f in ROOT.rglob("*.py"):
        rel = f.relative_to(ROOT)
        # Skip
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        all_files.append(rel)

    all_files.sort()
    print(f"  Total Python files: {len(all_files)}")

    # 6. Analyze and classify
    print("[4/4] Analyzing files...\n")

    results_by_dir = defaultdict(list)
    summary = {"live": 0, "independent": 0, "test": 0, "dashboard": 0,
               "delete": 0, "likely_delete": 0, "review": 0, "wire": 0}
    all_results = []

    for rel in all_files:
        filepath = ROOT / rel
        rel_str = str(rel).replace("\\", "/")
        dir_name = rel.parts[0] if len(rel.parts) > 1 else "(root)"

        if args.dir and dir_name != args.dir:
            continue

        is_live = rel_str in live_files
        is_independent = rel_str in independent_only
        is_test = rel_str in test_files
        is_dashboard = rel_str in dashboard_only

        info = analyze_file(filepath)
        v, reason = verdict(info, is_live, is_independent, is_test, is_dashboard)

        info["verdict"] = v
        info["reason"] = reason
        info["category"] = (
            "live" if is_live
            else "independent" if is_independent
            else "test" if is_test
            else "dashboard" if is_dashboard
            else "dead"
        )

        results_by_dir[dir_name].append(info)
        all_results.append(info)

        # Count
        if "LIVE" in v:
            summary["live"] += 1
        elif "INDEPENDENT" in v:
            summary["independent"] += 1
        elif "TEST" in v:
            summary["test"] += 1
        elif "DASHBOARD" in v:
            summary["dashboard"] += 1
        elif "DELETE" in v and "LIKELY" not in v:
            summary["delete"] += 1
        elif "LIKELY DELETE" in v:
            summary["likely_delete"] += 1
        elif "REVIEW" in v:
            summary["review"] += 1
        elif "WIRE" in v:
            summary["wire"] += 1

    # ═══ Output ═══

    for dir_name in sorted(results_by_dir.keys()):
        files = results_by_dir[dir_name]
        live_count = sum(1 for f in files if f["category"] == "live")
        dead_count = sum(1 for f in files if f["category"] == "dead")
        total_lines = sum(f.get("lines", 0) for f in files)

        print(f"\n{'─'*70}")
        print(f"  {dir_name}/  ({len(files)} files, {total_lines:,} lines, "
              f"{live_count} live, {dead_count} dead)")
        print(f"{'─'*70}")

        for f in files:
            rel = f["path"].replace("\\", "/")
            v = f["verdict"]
            lines = f.get("lines", 0)
            reason = f["reason"]

            # Compact display
            print(f"  {v:20s} {rel}")
            print(f"  {'':20s} {lines:>5} lines | {reason}")

            if args.verbose and f["category"] == "dead":
                if f.get("docstring"):
                    doc_short = f["docstring"][:120].replace("\n", " ")
                    print(f"  {'':20s} Doc: {doc_short}")
                if f.get("classes"):
                    print(f"  {'':20s} Classes: {', '.join(f['classes'][:5])}")
                if f.get("functions"):
                    print(f"  {'':20s} Functions: {', '.join(f['functions'][:8])}")
                if f.get("keywords"):
                    print(f"  {'':20s} Keywords: {', '.join(f['keywords'])}")
                if f.get("markers", {}).get("deprecated"):
                    print(f"  {'':20s} [WARN] DEPRECATED marker found")
                print(f"  {'':20s} Modified: {f.get('modified', '?')}")

    # ═══ Summary ═══

    print(f"\n{'═'*70}")
    print(f"  SUMMARY")
    print(f"{'═'*70}")
    print(f"  ON LIVE (runtime reachable):     {summary['live']:>4}")
    print(f"  🔧 INDEPENDENT (training/scripts): {summary['independent']:>4}")
    print(f"  📊 DASHBOARD (streamlit):        {summary['dashboard']:>4}")
    print(f"  🧪 TEST:                         {summary['test']:>4}")
    print(f"  {'─'*50}")
    print(f"  🗑️  DELETE (confirmed):           {summary['delete']:>4}")
    print(f"  🗑️  LIKELY DELETE:                {summary['likely_delete']:>4}")
    print(f"  🔌 WIRE? (有价值未接入):          {summary['wire']:>4}")
    print(f"  🔍 REVIEW (需人工判断):           {summary['review']:>4}")

    total_dead_lines = sum(f.get("lines", 0) for f in all_results
                           if "DELETE" in f.get("verdict", ""))
    total_review_lines = sum(f.get("lines", 0) for f in all_results
                             if "REVIEW" in f.get("verdict", "")
                             or "WIRE" in f.get("verdict", ""))

    print(f"\n  可直接删除: ~{total_dead_lines:,} lines")
    print(f"  需审查决定: ~{total_review_lines:,} lines")

    # ═══ Action Items ═══

    print(f"\n{'═'*70}")
    print(f"  ACTION ITEMS")
    print(f"{'═'*70}")

    wire_files = [f for f in all_results if "WIRE" in f.get("verdict", "")]
    if wire_files:
        print(f"\n  🔌 有价值但未接入 ({len(wire_files)} files):")
        for f in wire_files:
            print(f"     {f['path']}")
            for kw in f.get("keywords", []):
                print(f"       -> {kw}")

    review_files = [f for f in all_results if "REVIEW" in f.get("verdict", "")]
    if review_files:
        print(f"\n  🔍 需要人工审查 ({len(review_files)} files):")
        for f in review_files:
            print(f"     {f['path']} ({f.get('lines', 0)} lines)")
            if f.get("classes"):
                print(f"       Classes: {', '.join(f['classes'][:5])}")

    # ═══ Save JSON ═══

    if args.output:
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "total_files": len(all_files),
            "total_dead_lines": total_dead_lines,
            "total_review_lines": total_review_lines,
            "files": all_results,
        }
        output_path = ROOT / args.output
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        print(f"\n  Full report saved: {output_path}")

    print(f"\n{'═'*70}")
    print(f"  Next: 审查 🔌 WIRE? 和 🔍 REVIEW 文件，决策后执行清理")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()

"""HMATS Completeness Audit — full inventory / wiring / 5-hop analysis.

Stages 0-11 from HMATS_COMPLETENESS_AUDIT.md condensed into one Python script.
Produces /tmp/hmats_audit/{inventory,wiring_analysis,final_report}.json + stdout.

Usage: python -X utf8 scripts/completeness_audit.py
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "scripts" else Path.cwd()
OUT = Path("/tmp/hmats_audit") if Path("/tmp").exists() else ROOT / "audit_output"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_DIRS = [
    "agents", "signals", "strategies", "core", "risk", "defense",
    "execution", "market", "integration", "orchestration",
    "analytics", "data_mgmt", "drl", "infra", "liquidity", "v36",
]

SKIP_PATTERNS = ("__pycache__", "archive", "legacy")


def enumerate_files() -> List[Path]:
    out = []
    for d in TARGET_DIRS:
        p = ROOT / d
        if not p.exists():
            continue
        for f in p.rglob("*.py"):
            if any(s in f.parts for s in SKIP_PATTERNS):
                continue
            if f.name == "__init__.py":
                continue
            out.append(f)
    return sorted(out)


def ast_inventory(files: List[Path]) -> Dict[str, Dict]:
    inv = {}
    for f in files:
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        try:
            src = f.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception as e:
            inv[rel] = {"error": str(e), "lines": 0, "classes": [], "functions": []}
            continue
        classes, functions = [], []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                methods = [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                classes.append({"name": node.name, "methods": methods, "line": node.lineno})
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append({"name": node.name, "line": node.lineno})
        inv[rel] = {
            "lines": len(src.splitlines()),
            "classes": classes,
            "functions": functions,
        }
    return inv


def load_hot_path() -> str:
    parts = []
    for name in ("main.py",):
        p = ROOT / name
        if p.exists():
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    for extra in ("integration/integration_v36.py", "v36/integration_v36.py"):
        p = ROOT / extra
        if p.exists():
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


DECISION_KW = [
    "agent_signals", "trade_intent", "TradeIntent", "veto", "fusion",
    "authority", "direction", "confidence", "order", "intent",
    "execute", "scale_out", "target_exposure",
]
IMPACT_KW = ["veto", "authority", "intent", "execute"]


def hop_analysis(inv: Dict[str, Dict], hot: str) -> Dict[str, Dict]:
    results = {}
    hot_lines = hot.splitlines()
    for rel, info in inv.items():
        module_path = rel.replace("/", ".").replace(".py", "")
        basename = Path(rel).stem
        file_h1 = (
            f"from {module_path} import" in hot
            or f"import {module_path}" in hot
            or f"from .{basename} import" in hot
        )
        for cls in info.get("classes", []):
            name = cls["name"]
            key = f"{rel}::{name}"
            h1 = file_h1 or (name in hot)
            h2 = bool(re.search(rf"\b{re.escape(name)}\s*\(", hot))
            h3 = False
            for m in re.finditer(rf"self\.(\w+)\s*=\s*{re.escape(name)}\s*\(", hot):
                attr = m.group(1)
                if re.search(rf"self\.{re.escape(attr)}\.\w+\s*\(", hot):
                    h3 = True
                    break
            # Also detect factory: get_<snake>()
            if not h3:
                snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
                for prefix in ("get_", "create_"):
                    fn = prefix + snake
                    for m in re.finditer(rf"self\.(\w+)\s*=\s*{re.escape(fn)}\s*\(", hot):
                        attr = m.group(1)
                        if re.search(rf"self\.{re.escape(attr)}\.\w+\s*\(", hot):
                            h3 = True
                            break
                    if h3:
                        break
            h4 = h5 = False
            if h3:
                for i, line in enumerate(hot_lines):
                    if name in line:
                        ctx = "\n".join(hot_lines[max(0, i - 5):min(len(hot_lines), i + 6)])
                        if any(k in ctx for k in DECISION_KW):
                            h4 = True
                        if any(k in ctx for k in IMPACT_KW):
                            h5 = True
                        if h4 and h5:
                            break
            score = sum([h1, h2, h3, h4, h5])
            status = (
                "ACTIVE" if score == 5
                else "SEMI-ACTIVE" if score >= 3
                else "SHADOW" if score >= 1
                else "DEAD"
            )
            results[key] = {
                "file": rel, "class": name,
                "methods_count": len(cls.get("methods", [])),
                "lines": info.get("lines", 0),
                "hop1": h1, "hop2": h2, "hop3": h3, "hop4": h4, "hop5": h5,
                "score": score, "status": status,
            }
    return results


def agent_signals_flow() -> Dict[str, Any]:
    mp = ROOT / "main.py"
    if not mp.exists():
        return {"writers": [], "readers": [], "orphans": [], "dead_reads": []}
    src = mp.read_text(encoding="utf-8", errors="ignore")
    writers = set(re.findall(r'agent_signals\[[\'"]([^\'"]+)[\'"]\]\s*=', src))
    readers = set(re.findall(r'agent_signals\.get\([\'"]([^\'"]+)[\'"]', src))
    readers |= set(re.findall(r'agent_signals\[[\'"]([^\'"]+)[\'"]\](?!\s*=)', src))
    return {
        "writers": sorted(writers),
        "readers": sorted(readers),
        "orphans": sorted(writers - readers),
        "dead_reads": sorted(readers - writers),
    }


def config_flags() -> Dict[str, Any]:
    declared = set()
    for p in ("configs", "config"):
        cdir = ROOT / p
        if cdir.exists():
            for cf in cdir.rglob("*.py"):
                try:
                    src = cf.read_text(encoding="utf-8", errors="ignore")
                    declared.update(re.findall(r"^\s*([A-Z][A-Z0-9_]{2,})\s*[:=]", src, re.MULTILINE))
                except Exception:
                    pass
    env = ROOT / ".env"
    if env.exists():
        try:
            for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
                m = re.match(r"^([A-Z][A-Z0-9_]{2,})\s*=", line.strip())
                if m:
                    declared.add(m.group(1))
        except Exception:
            pass
    consumers = set()
    check = [ROOT / "main.py"] if (ROOT / "main.py").exists() else []
    for d in ("core", "defense", "risk", "execution", "v36", "integration"):
        p = ROOT / d
        if p.exists():
            check.extend(p.rglob("*.py"))
    for sf in check:
        if not sf.exists():
            continue
        try:
            src = sf.read_text(encoding="utf-8", errors="ignore")
            for f in declared:
                if f in src:
                    consumers.add(f)
        except Exception:
            pass
    return {
        "declared": len(declared),
        "consumed": len(consumers),
        "orphan": sorted(declared - consumers)[:40],
    }


def summarize(results: Dict[str, Dict]) -> Dict:
    dir_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, v in results.items():
        d = v["file"].split("/")[0] + "/"
        dir_stats[d][v["status"]] += 1
        dir_stats[d]["TOTAL"] += 1
    summary = Counter()
    for s in dir_stats.values():
        for k, v in s.items():
            if k != "TOTAL":
                summary[k] += v
    return {
        "per_directory": {k: dict(v) for k, v in sorted(dir_stats.items())},
        "overall": dict(summary),
    }


def print_report(summary: Dict, results: Dict, flows: Dict, flags: Dict):
    total = sum(v for k, v in summary["overall"].items())
    active = summary["overall"].get("ACTIVE", 0)
    semi = summary["overall"].get("SEMI-ACTIVE", 0)
    shadow = summary["overall"].get("SHADOW", 0)
    dead = summary["overall"].get("DEAD", 0)

    print("\n" + "=" * 70)
    print("  HMATS Completeness Audit")
    print("=" * 70)

    print("\nPer-directory:")
    print(f"  {'Dir':<18} {'Total':>6} {'ACTIVE':>8} {'SEMI':>6} {'SHADOW':>8} {'DEAD':>6}")
    for d, s in summary["per_directory"].items():
        print(f"  {d:<18} {s.get('TOTAL', 0):>6} {s.get('ACTIVE', 0):>8} "
              f"{s.get('SEMI-ACTIVE', 0):>6} {s.get('SHADOW', 0):>8} {s.get('DEAD', 0):>6}")

    print(f"\nOverall: {total} classes")
    if total > 0:
        print(f"  ACTIVE      {active:>4}  ({100 * active / total:.1f}%)")
        print(f"  SEMI-ACTIVE {semi:>4}  ({100 * semi / total:.1f}%)")
        print(f"  SHADOW      {shadow:>4}  ({100 * shadow / total:.1f}%)")
        print(f"  DEAD        {dead:>4}  ({100 * dead / total:.1f}%)")

    print("\nagent_signals flow:")
    print(f"  writers: {len(flows['writers'])}, readers: {len(flows['readers'])}")
    print(f"  orphan (write-no-read): {len(flows['orphans'])}")
    for w in flows["orphans"][:15]:
        print(f"    ◦ {w}")
    print(f"  dead reads (read-no-write): {len(flows['dead_reads'])}")
    for r in flows["dead_reads"][:10]:
        print(f"    ✗ {r}")

    print("\nconfig flags:")
    print(f"  declared: {flags['declared']}, consumed: {flags['consumed']}, "
          f"orphan: {len(flags['orphan'])}")
    for f in flags["orphan"][:20]:
        print(f"    ◦ {f}")

    critical = ("agents/", "signals/", "defense/", "risk/", "core/", "strategies/")

    print("\nHIGH-PRIORITY: SEMI-ACTIVE in critical paths (top 15):")
    semi_crit = [
        (k, v) for k, v in results.items()
        if v["status"] == "SEMI-ACTIVE" and any(v["file"].startswith(d) for d in critical)
    ]
    for k, v in sorted(semi_crit, key=lambda x: -x[1]["lines"])[:15]:
        missing = [h.upper() for h in ("hop1", "hop2", "hop3", "hop4", "hop5") if not v[h]]
        print(f"  {v['file']}::{v['class']:<30} {v['lines']:>5}L  missing={','.join(missing)}")

    print("\nDEAD in critical paths (top 15):")
    dead_crit = [
        (k, v) for k, v in results.items()
        if v["status"] == "DEAD" and any(v["file"].startswith(d) for d in critical)
    ]
    for k, v in sorted(dead_crit, key=lambda x: -x[1]["lines"])[:15]:
        print(f"  {v['file']}::{v['class']:<30} {v['lines']:>5}L  0/5 hops")


def main():
    print(f"Root: {ROOT}")
    print(f"Output: {OUT}")

    files = enumerate_files()
    print(f"Stage 0: {len(files)} files enumerated")

    inv = ast_inventory(files)
    total_cls = sum(len(v.get("classes", [])) for v in inv.values())
    total_fn = sum(len(v.get("functions", [])) for v in inv.values())
    print(f"Stage 1: {total_cls} classes, {total_fn} top-level functions")
    (OUT / "inventory.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")

    hot = load_hot_path()
    print(f"Stage 2: hot-path source {len(hot):,} chars")
    results = hop_analysis(inv, hot)
    (OUT / "wiring_analysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    flows = agent_signals_flow()
    flags = config_flags()

    summary = summarize(results)
    final = {
        "summary": {
            "total": sum(v for v in summary["overall"].values()),
            **summary["overall"],
        },
        "per_directory": summary["per_directory"],
        "agent_signals_flow": flows,
        "config_flags": flags,
    }
    (OUT / "final_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    print_report(summary, results, flows, flags)
    print(f"\nReports: {OUT}/inventory.json, wiring_analysis.json, final_report.json")


if __name__ == "__main__":
    main()

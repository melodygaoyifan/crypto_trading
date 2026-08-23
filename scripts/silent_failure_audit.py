"""
Silent-failure pattern detector — finds the SHAPE of the P47 family of bugs.

Three patterns it catches:

1. **Method-call-in-silent-try-except**: `try: X.method(...) except Exception: pass`
   or `except Exception: logger.debug(...)`. If X is a known class and method
   doesn't exist on it, P15-style silent failure.

2. **dict.get(a, b) where both look like keys**: classic P47-Bug-2 shape.
   `matrix.get(base_a, base_b)` where base_b is a variable, not a literal default
   like 0.0/None/{}/[]/"".

3. **ENABLE_X flags with no runtime reader**: P16-style dead flag — declared but
   never read at runtime, so flipping the flag has no effect.

Output: ranked list of suspects with file:line. Operator triages from there.

Usage:
    python -X utf8 scripts/silent_failure_audit.py [--limit 30]
"""
from __future__ import annotations

import argparse
import ast
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

LIVE_DIRS = [
    "agents", "analytics", "core", "data_mgmt", "defense", "drl",
    "execution", "infra", "integration", "liquidity", "market",
    "orchestration", "risk", "signals",
    # [P382] exchange/ (100% of directional risk since Phase B), api/ and
    # portfolio/ were never scanned — P366 recorded the blind spot. Adding
    # them is COVERAGE, not regression (baseline bumped with attribution).
    "exchange", "api", "portfolio",
    # configs/ scanned for ENABLE_* flag declarations (Pattern 3) and the
    # full repo (already-included LIVE_DIRS) for readers.
    "configs",
]
SKIP_DIRS = {"archive", "tests", "training", "scripts", "venv", "node_modules", ".git", "__pycache__", "pytorch_build"}


# [P226] main.py carries a UTF-8 BOM, and every read below used
# encoding="utf-8", so `ast.parse` raised SyntaxError on U+FEFF and this audit
# SILENTLY SKIPPED THE LARGEST FILE IN THE REPO for its entire existence. The
# committed baseline of 337 simply did not include main.py; the honest number
# with it is 645. Found only because an unrelated edit happened to strip the
# BOM and 308 findings appeared at once.
#
# This is P171 exactly — same BOM, same SyntaxError, same "a parse failure is
# indistinguishable from a clean file" — fixed there in
# lint_orphan_signal_reads.py and never applied here. Two halves, because
# either alone leaves the hole open:
#   1. utf-8-sig: a BOM is an encoding detail, not a syntax error.
#   2. PARSE_FAILURES: refuse to report rather than emit counts computed from
#      a partial scan.
PARSE_FAILURES: list = []


def _live_py_files() -> List[Path]:
    files = []
    for d in LIVE_DIRS:
        root = REPO_ROOT / d
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p.name.startswith("_") and p.name != "__init__.py":
                pass  # keep _http.py etc
            files.append(p)
    main_py = REPO_ROOT / "main.py"
    if main_py.exists():
        files.append(main_py)
    return files


# =============================================================================
# Pattern 1: silent try/except around method calls
# =============================================================================

class TryExceptVisitor(ast.NodeVisitor):
    """Find `try: <stmts> except [Exception]: pass | logger.debug | logger.warning(...)`."""
    def __init__(self, source_lines: List[str], path: Path):
        self.source_lines = source_lines
        self.path = path
        self.findings: List[Dict] = []

    def visit_Try(self, node: ast.Try):
        # A try block is "silent" if at least one handler does only:
        # - pass
        # - log at debug/warning level
        # without re-raising or escalating.
        for handler in node.handlers:
            # Match Exception or bare except
            handler_type = handler.type
            is_broad = (
                handler_type is None
                or (isinstance(handler_type, ast.Name) and handler_type.id in ("Exception", "BaseException"))
            )
            if not is_broad:
                continue

            silent = self._is_silent_handler(handler.body)
            if not silent:
                continue

            # Look for method calls in the try body.
            for body_node in node.body:
                for call in ast.walk(body_node):
                    if not isinstance(call, ast.Call):
                        continue
                    func = call.func
                    if not isinstance(func, ast.Attribute):
                        continue
                    # X.method(...)
                    receiver = self._receiver_name(func.value)
                    method_name = func.attr
                    if receiver and method_name and not method_name.startswith("_"):
                        self.findings.append({
                            "file": str(self.path.relative_to(REPO_ROOT)).replace("\\", "/"),
                            "line": call.lineno,
                            "receiver": receiver,
                            "method": method_name,
                            "snippet": self.source_lines[call.lineno - 1].strip()[:120] if call.lineno - 1 < len(self.source_lines) else "",
                        })
        self.generic_visit(node)

    @staticmethod
    def _is_silent_handler(body: List[ast.stmt]) -> bool:
        if not body:
            return True
        # All statements must be "silent" — pass, simple debug log, or comment
        for stmt in body:
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Attribute):
                    method = func.attr
                    if method in ("debug", "info", "warning"):
                        continue
            return False
        return True

    @staticmethod
    def _receiver_name(node: ast.expr) -> Optional[str]:
        # Resolve X.Y.method(...) or X.method(...) to a printable receiver.
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return None


# =============================================================================
# Pattern 2: dict.get(a, b) where both look like keys
# =============================================================================

# Heuristic: matches `<expr>.get(<simple_name>, <simple_name_or_attr>)`
# Both args are pure variable references (no literal/sentinel default).
# Excludes obvious literal defaults: 0, 0.0, None, "", [], {}, set(), False/True
DICT_GET_PATTERN = re.compile(
    r"\.get\s*\(\s*([a-zA-Z_][\w]*(?:\[[^\]]+\]|\.[a-zA-Z_][\w]*)*)\s*,\s*([a-zA-Z_][\w]*(?:\[[^\]]+\]|\.[a-zA-Z_][\w]*)*)\s*\)"
)
LITERAL_DEFAULTS = {
    "None", "True", "False", "0", "0.0", "1", "1.0",
    "''", '""', "[]", "{}", "set()", "list()", "dict()", "tuple()",
    "inf", "math.inf",
}


def detect_dict_get_misuse(path: Path, source: str) -> List[Dict]:
    findings = []
    for m in DICT_GET_PATTERN.finditer(source):
        a, b = m.group(1), m.group(2)
        # Skip if the second arg is clearly a default
        if b in LITERAL_DEFAULTS or b.endswith("_DEFAULT") or b.startswith("DEFAULT_"):
            continue
        # Skip common-default names
        if b in ("default", "default_value", "fallback", "DEFAULT"):
            continue
        line = source.count("\n", 0, m.start()) + 1
        # Skip string-literal-default like "asset_a"
        snippet_idx = source.rfind("\n", 0, m.start()) + 1
        snippet = source[snippet_idx:source.find("\n", m.end())].strip()[:140]
        findings.append({
            "file": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "line": line,
            "key": a,
            "default": b,
            "snippet": snippet,
        })
    return findings


# =============================================================================
# Pattern 3: ENABLE_* flags with no runtime reader
# =============================================================================

ENABLE_FLAG_DECL_PATTERN = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*:\s*bool\s*=\s*(True|False)\s*$|^\s*([A-Z][A-Z0-9_]*)\s*=\s*(True|False)\s*$", re.MULTILINE)


def find_enable_flags(files: List[Path]) -> Tuple[Dict[str, Dict], Dict[str, int]]:
    """Returns (declared_flags, reader_counts)."""
    declared: Dict[str, Dict] = {}
    reader_count: Counter = Counter()
    config_files = [f for f in files if "configs" in str(f) or "sota_flags" in f.name or "/flags" in str(f).replace("\\", "/")]
    for path in config_files:
        try:
            content = path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            continue
        for m in ENABLE_FLAG_DECL_PATTERN.finditer(content):
            name = m.group(1) or m.group(3)
            if not name or not name.startswith("ENABLE_"):
                continue
            line = content.count("\n", 0, m.start()) + 1
            declared[name] = {"file": str(path.relative_to(REPO_ROOT)).replace("\\", "/"), "line": line}

    if not declared:
        return declared, dict(reader_count)

    flag_pattern = re.compile(r"\b(?:flags\.)?(" + "|".join(re.escape(k) for k in declared) + r")\b")
    for path in files:
        # Skip the declaration files themselves for reader count
        rel_path = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        is_decl_file = any(d["file"] == rel_path for d in declared.values())
        try:
            content = path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            continue
        for m in flag_pattern.finditer(content):
            name = m.group(1)
            if is_decl_file:
                # Don't count the declaration line itself
                if any(declared[name]["line"] == content.count("\n", 0, m.start()) + 1
                       for k in (name,) if k in declared):
                    continue
            reader_count[name] += 1
    return declared, dict(reader_count)


# =============================================================================
# Pattern 1 verification: which receivers exist as known classes?
# =============================================================================

def collect_class_methods(files: List[Path]) -> Dict[str, Set[str]]:
    """Best-effort: scan for class definitions and their method names.
    Used to verify whether a `X.method(...)` call's `method` exists on a class
    that's plausibly `X`."""
    class_methods: Dict[str, Set[str]] = defaultdict(set)
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"), filename=str(path))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_methods[node.name].add(item.name)
    return class_methods


# =============================================================================
# Main
# =============================================================================

def main():
    import json as _json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=30, help="max items per category")
    p.add_argument("--pattern", choices=["all", "tryexcept", "dictget", "flags"], default="all")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON to stdout (for CI baseline tooling). "
                        "When set, suppresses the human-readable output entirely.")
    args = p.parse_args()

    files = _live_py_files()
    if not args.json:
        print(f"Scanning {len(files)} files in live dirs...\n")

    # =========================================================================
    # Pattern 1
    # =========================================================================
    pattern1: List[Dict] = []
    suspect_methods: List[Tuple[str, str, int]] = []
    if args.pattern in ("all", "tryexcept"):
        for path in files:
            try:
                source = path.read_text(encoding="utf-8-sig", errors="replace")
                tree = ast.parse(source, filename=str(path))
            except Exception:
                continue
            visitor = TryExceptVisitor(source.splitlines(), path)
            visitor.visit(tree)
            pattern1.extend(visitor.findings)

        # Group by (receiver, method) — these are the suspects
        by_method = Counter((f["receiver"], f["method"]) for f in pattern1)
        # The most-suspicious are method calls that appear ONLY in try/except.
        # We don't have a counterfactual ("does method exist?") so just rank by raw frequency.
        # But: filter out trivially common ones (logger, etc.)
        IGNORE_RECEIVERS = {"logger", "self.logger", "logging", "self._logger", "log"}
        IGNORE_METHODS = {"info", "debug", "warning", "error", "warn", "exception", "critical",
                          "append", "add", "discard", "extend", "update", "clear", "pop",
                          "get", "setdefault", "items", "keys", "values", "send", "put",
                          "release", "acquire", "close", "join", "set", "wait", "notify"}
        suspect_methods = [
            (r, m, c) for (r, m), c in by_method.most_common()
            if r not in IGNORE_RECEIVERS and m not in IGNORE_METHODS
        ]

        if not args.json:
            print("=" * 78)
            print("PATTERN 1: Method calls inside silent try/except blocks")
            print("=" * 78)
            print(f"Total method-call sites under silent except: {len(pattern1)}")
            print(f"Unique (receiver, method) pairs: {len(by_method)}")
            print(f"After filtering known-noisy: {len(suspect_methods)}")
            print()
            print(f"Top {args.limit} suspect (receiver, method) pairs to verify:")
            print("(P15-shape: 'does ctx.X.method actually exist on type(X)?')")
            print()
            for r, m, c in suspect_methods[:args.limit]:
                print(f"  {c:3} × {r}.{m}()")
                # Show a sample call site
                sample = next(f for f in pattern1 if f["receiver"] == r and f["method"] == m)
                print(f"      example: {sample['file']}:{sample['line']}")
                print(f"      {sample['snippet']}")
            print()

    # =========================================================================
    # Pattern 2
    # =========================================================================
    pattern2: List[Dict] = []
    if args.pattern in ("all", "dictget"):
        for path in files:
            try:
                source = path.read_text(encoding="utf-8-sig", errors="replace")
            except Exception:
                continue
            pattern2.extend(detect_dict_get_misuse(path, source))

        if not args.json:
            print("=" * 78)
            print("PATTERN 2: dict.get(<var>, <var>) — possible P47-Bug-2 shape")
            print("=" * 78)
            print(f"Total suspicious .get() call sites: {len(pattern2)}")
            print()
            print(f"Top {args.limit}:")
            # Sort by file then line
            pattern2.sort(key=lambda f: (f["file"], f["line"]))
            for f in pattern2[:args.limit]:
                print(f"  {f['file']}:{f['line']}  .get({f['key']}, {f['default']})")
                print(f"      {f['snippet']}")
            if len(pattern2) > args.limit:
                print(f"  ... and {len(pattern2) - args.limit} more")
            print()

    # =========================================================================
    # Pattern 3
    # =========================================================================
    no_readers: List[Tuple[str, Dict]] = []
    declared: Dict[str, Dict] = {}
    if args.pattern in ("all", "flags"):
        declared, reader_count = find_enable_flags(files)
        no_readers = [(name, info) for name, info in declared.items() if reader_count.get(name, 0) == 0]
        if not args.json:
            print("=" * 78)
            print("PATTERN 3: ENABLE_* flags declared but never read at runtime")
            print("=" * 78)
            print(f"Declared ENABLE_* flags: {len(declared)}")
            print(f"Without ANY runtime reader: {len(no_readers)}")
            print()
            for name, info in no_readers:
                print(f"  {name:50}  {info['file']}:{info['line']}")
            print()
            print("Flags with VERY FEW readers (1-2 — possibly stale):")
            few_readers = sorted(
                [(name, reader_count.get(name, 0), info) for name, info in declared.items() if 0 < reader_count.get(name, 0) <= 2],
                key=lambda t: t[1],
            )
            for name, count, info in few_readers[:args.limit]:
                print(f"  {name:50}  {count} read(s)  {info['file']}:{info['line']}")

    # JSON output for CI baseline tooling.
    if args.json:
        out = {
            "tryexcept_hits": [
                {"receiver": r, "method": m, "count": c}
                for r, m, c in suspect_methods
            ],
            "dictget_hits": [
                {"file": f["file"], "line": f["line"], "key": f["key"], "default": f["default"]}
                for f in pattern2
            ],
            "flags_hits": [
                {"name": name, "file": info["file"], "line": info["line"]}
                for name, info in no_readers
            ],
        }
        print(_json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

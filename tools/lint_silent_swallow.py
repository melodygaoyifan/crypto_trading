"""
lint_silent_swallow.py — block silent-swallow patterns at write time.

Detects three sub-patterns of the P15/P25/P47 family:

  A) `try: ...; except ...: pass`           — exception fully discarded.
  B) `try: ...; except ...: logger.debug(...)` — exception logged below
     OPERATOR_VISIBILITY threshold (debug = noise floor in this codebase).
  C) `try: ...; except ...: continue/return None/return False` immediately
     followed by no log call at all.

Opt-out: any line in the offending except block carrying
`# noqa: silent-swallow` (or one carried on the `except` line itself)
is treated as intentional.

Why both forms exist
--------------------
Many `try/except: pass` are intentional (e.g. swallowing
ImportError for optional deps, ignoring missing-key on dict.pop).
The `# noqa: silent-swallow` opt-out forces a deliberate decision
at write time: every silent block is either acknowledged or fixed,
no third option.

Usage
-----
    python -X utf8 tools/lint_silent_swallow.py             # all live dirs
    python -X utf8 tools/lint_silent_swallow.py path/file.py path/dir/
    python -X utf8 tools/lint_silent_swallow.py --staged    # only staged files (pre-commit)
    python -X utf8 tools/lint_silent_swallow.py --json      # machine-readable output

Exit codes
----------
    0 = clean (zero unannotated silent swallows)
    1 = found unannotated silent swallows (printed to stdout)
    2 = invocation error
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

REPO = Path(__file__).resolve().parents[1]

LIVE_DIRS = (
    "agents", "analytics", "core", "data_mgmt", "defense", "drl",
    "execution", "infra", "integration", "liquidity", "market",
    "orchestration", "risk", "signals",
)
SKIP_DIRS = {
    "archive", "tests", "training", "scripts", "tools", "venv",
    "node_modules", ".git", "__pycache__", "pytorch_build", "models",
    "data", "logs", ".github",
}

NOQA_TAG = "noqa: silent-swallow"


def _live_py_files(custom_paths: Optional[Iterable[Path]] = None) -> List[Path]:
    """Yield .py files to scan. If custom_paths given, walk those (file or dir).
    Otherwise walk LIVE_DIRS + main.py."""
    if custom_paths:
        out: List[Path] = []
        for p in custom_paths:
            p = (REPO / p).resolve() if not p.is_absolute() else p
            if not p.exists():
                continue
            if p.is_file() and p.suffix == ".py":
                out.append(p)
            elif p.is_dir():
                for q in p.rglob("*.py"):
                    if any(part in SKIP_DIRS for part in q.parts):
                        continue
                    out.append(q)
        return out

    out = []
    for d in LIVE_DIRS:
        root = REPO / d
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            out.append(p)
    main_py = REPO / "main.py"
    if main_py.exists():
        out.append(main_py)
    return out


def _staged_py_files() -> List[Path]:
    """Files staged for commit. For use as a pre-commit hook."""
    try:
        r = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True, text=True, cwd=REPO,
        )
    except FileNotFoundError:
        return []
    if r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line.endswith(".py"):
            continue
        p = REPO / line
        if not p.exists():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return out


def _is_pure_pass(body: List[ast.stmt]) -> bool:
    return len(body) == 1 and isinstance(body[0], ast.Pass)


def _is_only_logger_debug(body: List[ast.stmt]) -> bool:
    if len(body) != 1:
        return False
    stmt = body[0]
    if not isinstance(stmt, ast.Expr):
        return False
    call = stmt.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr != "debug":
        return False
    # logger.debug, self.logger.debug, log.debug, etc.
    return True


def _no_log_with_early_exit(body: List[ast.stmt]) -> bool:
    """Pattern C: `except ...: <no log> ; return/continue/break`.

    Specifically: body has zero log calls at any visibility level AND
    contains a Return/Continue/Break (or just falls through). Excludes
    blocks that re-raise (those propagate the error correctly)."""
    if not body:
        return False  # bare empty body — caught by syntax error usually
    has_log = False
    has_reraise = False
    for stmt in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(stmt, ast.Raise):
            has_reraise = True
        if isinstance(stmt, ast.Call):
            f = stmt.func
            if isinstance(f, ast.Attribute) and f.attr in (
                "info", "warning", "warn", "error", "exception", "critical",
            ):
                has_log = True
            elif isinstance(f, ast.Name) and f.id in ("print",):
                has_log = True
    if has_reraise:
        return False  # re-raise is correct propagation, not a swallow
    if has_log:
        return False
    # If body is JUST `pass` we already caught that in pattern A.
    if _is_pure_pass(body):
        return False
    if _is_only_logger_debug(body):
        return False
    # Otherwise: any block with no logging that doesn't re-raise is suspect.
    # We only flag if there's an early-exit (return/continue/break) — that's
    # the actual "swallow and move on" pattern. Falling-through with side-effects
    # (e.g. setting a flag) is sometimes legitimate.
    for stmt in body:
        if isinstance(stmt, (ast.Return, ast.Continue, ast.Break)):
            return True
    return False


class SilentSwallowVisitor(ast.NodeVisitor):
    def __init__(self, source_lines: List[str], path: Path):
        self.source_lines = source_lines
        self.path = path
        self.findings: List[dict] = []

    def visit_Try(self, node: ast.Try) -> None:
        for handler in node.handlers:
            self._check_handler(handler)
        self.generic_visit(node)

    def _line(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1]
        return ""

    def _is_noqa(self, handler: ast.ExceptHandler) -> bool:
        # Check the `except ...:` line itself + every line in the body.
        ranges = [handler.lineno]
        for stmt in handler.body:
            if hasattr(stmt, "lineno"):
                ranges.append(stmt.lineno)
            if hasattr(stmt, "end_lineno") and stmt.end_lineno:
                ranges.extend(range(stmt.lineno, stmt.end_lineno + 1))
        for ln in ranges:
            if NOQA_TAG in self._line(ln):
                return True
        return False

    def _check_handler(self, handler: ast.ExceptHandler) -> None:
        body = handler.body
        if _is_pure_pass(body):
            kind = "pass"
        elif _is_only_logger_debug(body):
            kind = "debug"
        elif _no_log_with_early_exit(body):
            kind = "no-log-early-exit"
        else:
            return  # not a swallow shape

        if self._is_noqa(handler):
            return

        # Render the except clause for the report.
        exc_name = "Exception"
        if handler.type is not None:
            try:
                exc_name = ast.unparse(handler.type)
            except Exception:
                pass

        self.findings.append({
            "file": str(self.path.relative_to(REPO)).replace("\\", "/"),
            "line": handler.lineno,
            "kind": kind,
            "exception": exc_name,
            "snippet": self._line(handler.lineno).rstrip(),
            "remediation": (
                "Either: (a) raise/log at WARN+ level with type(e).__name__ "
                "and the relevant context; OR (b) add `# noqa: silent-swallow` "
                "on the except line if the swallow is intentional."
            ),
        })


def scan_file(path: Path) -> List[dict]:
    try:
        source = path.read_text(encoding="utf-8-sig", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []
    v = SilentSwallowVisitor(source.splitlines(), path)
    v.visit(tree)
    return v.findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to scan. Default: all LIVE_DIRS + main.py.",
    )
    ap.add_argument(
        "--staged",
        action="store_true",
        help="Scan only files staged in the index (for pre-commit hooks).",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="JSON output to stdout (for CI tooling).",
    )
    args = ap.parse_args()

    if args.staged:
        files = _staged_py_files()
    elif args.paths:
        files = _live_py_files(custom_paths=args.paths)
    else:
        files = _live_py_files()

    all_findings: List[dict] = []
    for path in files:
        all_findings.extend(scan_file(path))

    if args.json:
        # JSON mode is for tooling consumption (ci_check_invariants).
        # Exit 0 unconditionally — caller decides what to do with the
        # findings list. Non-zero would have to mean "scanner crashed".
        print(json.dumps({"findings": all_findings, "n_files": len(files)}, indent=2))
        return 0

    if not all_findings:
        print(f"[lint_silent_swallow] OK — scanned {len(files)} files, "
              f"zero unannotated silent swallows.")
        return 0

    # Group by file for human-readable output.
    by_file: dict = {}
    for f in all_findings:
        by_file.setdefault(f["file"], []).append(f)

    print("=" * 70)
    print(f"[lint_silent_swallow] {len(all_findings)} unannotated silent "
          f"swallows in {len(by_file)} file(s):")
    print("=" * 70)
    for fpath, hits in sorted(by_file.items()):
        for h in hits:
            print(f"  {fpath}:{h['line']}  ({h['kind']}, except {h['exception']})")
            print(f"      {h['snippet']}")
    print("=" * 70)
    print(
        "Each finding is one of:\n"
        "  pass              — exception fully discarded\n"
        "  debug             — only logger.debug, below operator visibility\n"
        "  no-log-early-exit — early return/continue with no log + no re-raise\n\n"
        "Fix options per site:\n"
        "  (a) Promote to logger.warning/error with type(e).__name__ + context.\n"
        "  (b) Re-raise (or `raise X from e`) if the caller should handle it.\n"
        "  (c) Add `# noqa: silent-swallow` on the `except` line if the\n"
        "      swallow is genuinely intentional (optional dep, expected-miss\n"
        "      on dict pop, etc.). The opt-out is per-block, deliberate.\n\n"
        "First-time roll-out: bulk-annotate existing silent swallows with\n"
        "  python -X utf8 tools/lint_silent_swallow.py --json | python -X utf8 \\\n"
        "      -c \"import json,sys,...\" \n"
        "(or just iterate file-by-file). New code is then guarded going forward."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

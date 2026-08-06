"""Source scanning helpers for tests that assert on code, not behaviour.

[P177] The first draft of tests/test_dead_risk_controller.py failed against
its own fix: the fix replaced the offending code with a comment block that
quoted the removed log line, and the regexes matched the explanation. A
scanner that cannot tell code from prose about code is not measuring what it
claims to.

[P179] The same thing happened again one file over. The P179 fix documents the
line it removed *inside a docstring* ("it previously hardcoded fee_bps = 0.0"),
so a scanner that strips comments but keeps string literals still matches.
Hence `strip_docstrings`.

Which mode you want depends on what you are pinning:

  * asserting a *log string* or other literal is gone -> keep literals
    (`strip_docstrings=False`), or the test cannot fail.
  * asserting a *statement* is gone -> strip docstrings too, or the test
    fails on the comment explaining why it is gone.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path
from typing import List


def read_source(path: Path) -> str:
    """Read a repo file. utf-8-sig because main.py carries a BOM."""
    return Path(path).read_text(encoding="utf-8-sig", errors="replace")


def _blank_span(lines: List[str], srow: int, scol: int,
                erow: int, ecol: int) -> None:
    """Overwrite a 1-indexed source span with spaces, preserving positions.

    In place, and preserving columns, deliberately. An earlier version joined
    tokens with a separator, which turned `self._short_control.evaluate(` into
    `self . _short_control . evaluate (` and broke every regex downstream.
    """
    for row in range(srow - 1, erow):
        line = lines[row]
        a = scol if row == srow - 1 else 0
        b = ecol if row == erow - 1 else len(line.rstrip("\n"))
        b = min(b, len(line))
        if b > a:
            lines[row] = line[:a] + " " * (b - a) + line[b:]


def code_only(path: Path, strip_docstrings: bool = False) -> str:
    """Source with `#` comments (and optionally docstrings) blanked out.

    Returns the raw source unchanged if it cannot be tokenized or parsed —
    a scanner that silently returns "" on a syntax error would make every
    "X is absent" assertion pass vacuously.
    """
    src = read_source(path)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src

    lines = src.splitlines(keepends=True)
    for t in toks:
        if t.type == tokenize.COMMENT:
            _blank_span(lines, t.start[0], t.start[1], t.end[0], t.end[1])

    if not strip_docstrings:
        return "".join(lines)

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "".join(lines)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            v = first.value
            _blank_span(lines, v.lineno, v.col_offset,
                        v.end_lineno, v.end_col_offset)

    return "".join(lines)

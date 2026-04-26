"""
touch_for_audit.py — prepend a no-op marker comment to every .py file under
the given directories so they show up in `git diff` for an `/ultrareview`
slice.

Why this exists
---------------
`/ultrareview` is diff-based — it only reviews files that changed vs main.
For the codebase-coverage slice plan (audit/safety-defense, audit/execution,
audit/signals-fusion, ...), we need a way to put every file in scope into
the diff WITHOUT making behavioral changes.

This script:
  1. Walks each given directory recursively (.py only by default).
  2. Inserts a single comment line at the top of each file (above any
     existing top-of-file comment, after the shebang/encoding line if
     present). The line carries the slice tag + today's date so future
     audits can find/clean these markers.
  3. Skips files that already carry the same marker (idempotent — safe to
     re-run).

Usage
-----
    git checkout -b audit/safety-defense main
    python -X utf8 scripts/touch_for_audit.py defense risk \
        --tag audit/safety-defense
    git add defense/ risk/
    git commit -m "audit slice marker: defense + risk (no behavior change)"
    git push origin audit/safety-defense
    # then: /ultrareview audit/safety-defense

Cleanup after the slice ships (or is discarded):
    python -X utf8 scripts/touch_for_audit.py defense risk \
        --tag audit/safety-defense --remove
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    "venv",
    "node_modules",
    "archive",
    "legacy",
    "models",
    "data",
}

MARKER_PREFIX = "# [AUDIT-SLICE:"


def _build_marker(tag: str, today: str) -> str:
    return (
        f"{MARKER_PREFIX} {tag} {today}] no-op marker for /ultrareview "
        f"diff inclusion. Safe to remove after review.\n"
    )


def _has_marker(text: str, tag: str) -> bool:
    return f"{MARKER_PREFIX} {tag}" in text.splitlines()[0:5].__str__()


def _insert_marker(path: Path, marker: str) -> bool:
    """Insert marker at the top of the file (after shebang/encoding line).
    Returns True if the file was modified."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    lines = text.splitlines(keepends=True)
    # Already marked?
    head = "".join(lines[:5])
    if marker.strip() in head or MARKER_PREFIX in head and "audit/" in head:
        # If a marker for ANY slice already exists at the top, don't stack
        # another. Replace if it's the same tag, skip otherwise.
        if marker.strip() in head:
            return False
        # Different slice tag — let it stack (operator's choice).
    # Skip insertion point past shebang and `# -*- coding -*-` lines.
    insert_idx = 0
    if lines and lines[0].startswith("#!"):
        insert_idx = 1
    if (
        insert_idx < len(lines)
        and lines[insert_idx].startswith("#")
        and "coding" in lines[insert_idx]
    ):
        insert_idx += 1
    lines.insert(insert_idx, marker)
    path.write_text("".join(lines), encoding="utf-8")
    return True


def _remove_marker(path: Path, tag: str) -> bool:
    """Remove the marker line for the given tag if present."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    lines = text.splitlines(keepends=True)
    needle = f"{MARKER_PREFIX} {tag}"
    new_lines = [ln for ln in lines if needle not in ln]
    if len(new_lines) == len(lines):
        return False
    path.write_text("".join(new_lines), encoding="utf-8")
    return True


def _iter_py_files(root: Path, exclude: set[str]):
    for p in root.rglob("*.py"):
        if any(part in exclude for part in p.parts):
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "paths",
        nargs="+",
        help="One or more directories under repo root (e.g. defense risk).",
    )
    ap.add_argument(
        "--tag",
        required=True,
        help="Slice tag to embed in the marker (e.g. audit/safety-defense).",
    )
    ap.add_argument(
        "--remove",
        action="store_true",
        help="Remove markers with this tag instead of inserting.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing.",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Additional directory names to skip (joined with defaults).",
    )
    args = ap.parse_args()

    exclude = set(DEFAULT_EXCLUDE_DIRS) | set(args.exclude)
    today = _dt.date.today().isoformat()
    marker = _build_marker(args.tag, today)

    total_visited = 0
    total_changed = 0

    for raw_path in args.paths:
        root = (REPO_ROOT / raw_path).resolve()
        if not root.exists():
            print(f"[skip] not found: {raw_path}", file=sys.stderr)
            continue
        if not root.is_dir():
            print(f"[skip] not a directory: {raw_path}", file=sys.stderr)
            continue
        # Safety: refuse to walk outside the repo.
        try:
            root.relative_to(REPO_ROOT)
        except ValueError:
            print(f"[skip] outside repo: {raw_path}", file=sys.stderr)
            continue

        for p in _iter_py_files(root, exclude):
            total_visited += 1
            if args.dry_run:
                print(f"[dry] would touch: {p.relative_to(REPO_ROOT)}")
                continue
            if args.remove:
                changed = _remove_marker(p, args.tag)
                if changed:
                    total_changed += 1
                    print(f"[remove] {p.relative_to(REPO_ROOT)}")
            else:
                changed = _insert_marker(p, marker)
                if changed:
                    total_changed += 1
                    print(f"[touch] {p.relative_to(REPO_ROOT)}")

    print(
        f"\nVisited {total_visited} .py files; "
        f"{'would change' if args.dry_run else 'changed'} {total_changed}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

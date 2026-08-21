"""[P360] Run a repo CLI IN-PROCESS, keeping its exit code.

THE FINDING, and it is about what a test can even ASK, not about a bug:

P358b drove `kq_strategy_diagnostic.py` through
`subprocess.run([sys.executable, ...])`, which is the obvious way to test a
CLI. **A subprocess cannot be patched, neutralised or inspected from the
parent.** So that test was structurally incapable of asking anything stronger
than "the output contains X" — and when P359 built `assert_drives_output`
(which proves a table is CONSUMED by removing it and requiring the output to
change), the strongest available check simply could not be applied. Choosing
subprocess had silently chosen the weakest assertion, and nothing said so.

Measured: **24 test files drive a repo script with `sys.executable`.** Every
one of them is capped the same way.

WHY SUBPROCESS WAS THE OBVIOUS CHOICE, AND WHY THAT IS THE POINT
    Because exit codes matter here. This repo's tools deliberately carry
    distinct ones — `ci_status` 0/1/2/3/4, `compute_shadow_ic` 2 vs 3,
    `seat_check` 2 vs 3 — and P185 records testing "both refusal paths at
    real exit-code level". `main()` returning an int or raising SystemExit is
    easy to lose when you call it directly, so subprocess is the safe way to
    get the code, and taking the weak assertion with it is invisible.

    So this is not "stop using subprocess". It is that the cheap path should
    not also be the weak one: `run_cli` returns the exit code AND leaves the
    module patchable, which removes the reason to trade one for the other.

WHEN SUBPROCESS IS STILL RIGHT — say so at the call site:
  * the test is ABOUT process-level behaviour: an import-time SystemExit, a
    hard crash, a signal, or `-X utf8` / encoding behaviour (P294);
  * the script must run against a DIFFERENT tree (a worktree or scratch
    checkout, P328/P344), where importing the parent's copy measures the
    wrong subject;
  * global-state isolation is load-bearing — in-process execution shares
    `sys.modules`, logging handlers and `os.environ` with the test session,
    so a script that mutates any of those can leak into later tests.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib
import sys
from typing import Any, List, NamedTuple, Optional, Sequence


class CliResult(NamedTuple):
    exit_code: int
    stdout: str
    stderr: str
    module: Any          # the loaded module — patch it, then call again


def load_cli(path) -> Any:
    """Import a script by path WITHOUT running its main().

    Scripts guard execution behind `if __name__ == "__main__":`, and this
    loads under a private name, so the module body runs and main() does not.
    Module-level side effects (a constant computed at import, a logger
    configured) DO happen — that is inherent to importing, and is one of the
    reasons the docstring above names global-state isolation as a case where
    subprocess is still correct.
    """
    p = pathlib.Path(path)
    name = "_cli_" + p.stem
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {p} as a module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_cli(target, argv: Optional[Sequence[str]] = None,
            entry: str = "main") -> CliResult:
    """Run a CLI in-process and return (exit_code, stdout, stderr, module).

    `target` is a path or an already-loaded module — pass the module when you
    want to patch it between runs, which is the whole point of being
    in-process (see `assert_drives_output`, P359).

    The exit code is resolved the way a shell would: `SystemExit` wins,
    otherwise the entry point's return value, otherwise 0. `None` counts as 0,
    matching `sys.exit(None)`. Without this the in-process route would be a
    downgrade from subprocess and nobody would take it.
    """
    mod = target if hasattr(target, "__dict__") and not isinstance(
        target, (str, pathlib.Path)) else load_cli(target)
    fn = getattr(mod, entry, None)
    if fn is None:
        raise AssertionError(
            f"{getattr(mod, '__name__', mod)} has no entry point {entry!r}")

    out, err = io.StringIO(), io.StringIO()
    saved: List[str] = sys.argv
    sys.argv = [getattr(mod, "__file__", "cli")] + list(argv or [])
    code = 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rv = fn()
            except SystemExit as e:          # the shell's own rule
                rv = e.code
        code = 0 if rv is None else (rv if isinstance(rv, int) else 1)
    finally:
        sys.argv = saved
    return CliResult(code, out.getvalue(), err.getvalue(), mod)

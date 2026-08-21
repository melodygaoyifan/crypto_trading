"""[P360] Subprocess silently caps a test at presence-only assertions.

P358b drove a CLI through `subprocess.run([sys.executable, ...])` — the
obvious way — and was therefore structurally unable to neutralise anything
inside it. When P359 built `assert_drives_output` (prove a table is CONSUMED
by removing it and requiring the output to change), the strongest available
check could not be applied. **The invocation had chosen the assertion, and
nothing said so.** 24 test files drive a repo script the same way.

The remedy is not "stop using subprocess" — its exit codes are exactly why it
was chosen, and this repo's tools carry distinct ones on purpose (P185). It is
that the cheap path should not also be the weak one. `run_cli` returns the
exit code AND leaves the module patchable, so the trade disappears.
"""

import json
import pathlib

import pytest

import main
from tests._cli_harness import load_cli, run_cli

REPO = pathlib.Path(main.__file__).parent


# ==========================================================================
# The exit code must survive, or nobody would take the in-process route
# ==========================================================================
def _script(tmp_path, body):
    p = tmp_path / "cli.py"
    p.write_text(body, encoding="utf-8")
    return p


def test_a_sys_exit_code_is_reported(tmp_path):
    r = run_cli(_script(tmp_path,
                        "import sys\ndef main():\n    sys.exit(3)\n"))
    assert r.exit_code == 3


def test_a_returned_code_is_reported(tmp_path):
    r = run_cli(_script(tmp_path, "def main():\n    return 2\n"))
    assert r.exit_code == 2


def test_returning_None_is_success_like_the_shell(tmp_path):
    r = run_cli(_script(tmp_path, "def main():\n    return None\n"))
    assert r.exit_code == 0


def test_stdout_and_stderr_are_captured_separately(tmp_path):
    r = run_cli(_script(tmp_path,
                        "import sys\ndef main():\n"
                        "    print('out')\n"
                        "    print('err', file=sys.stderr)\n"))
    assert "out" in r.stdout and "err" in r.stderr
    assert "err" not in r.stdout


def test_argv_reaches_argparse_and_is_restored(tmp_path):
    body = ("import argparse\n"
            "def main():\n"
            "    ap = argparse.ArgumentParser()\n"
            "    ap.add_argument('--path')\n"
            "    print('got', ap.parse_args().path)\n")
    before = list(__import__("sys").argv)
    r = run_cli(_script(tmp_path, body), ["--path", "X.json"])
    assert "got X.json" in r.stdout
    assert __import__("sys").argv == before, "sys.argv was not restored"


def test_argv_is_restored_even_when_the_cli_raises(tmp_path):
    import sys as _sys
    before = list(_sys.argv)
    with pytest.raises(RuntimeError):
        run_cli(_script(tmp_path, "def main():\n    raise RuntimeError('x')\n"))
    assert _sys.argv == before, (
        "a crashing CLI left sys.argv rewritten — that leaks into every later "
        "test in the session (P186)"
    )


def test_importing_does_not_run_main(tmp_path):
    """Scripts guard on __name__; loading under a private name must not fire
    the entry point, or merely inspecting a CLI would execute it."""
    marker = tmp_path / "ran.txt"
    body = (f"import pathlib\n"
            f"def main():\n"
            f"    pathlib.Path(r'{marker}').write_text('ran')\n"
            f"if __name__ == '__main__':\n    main()\n")
    load_cli(_script(tmp_path, body))
    assert not marker.exists(), "importing the script executed its main()"


def test_a_missing_entry_point_is_a_loud_refusal(tmp_path):
    with pytest.raises(AssertionError) as e:
        run_cli(_script(tmp_path, "x = 1\n"))
    assert "entry point" in str(e.value)


# ==========================================================================
# The capability that subprocess cannot offer at all
# ==========================================================================
def test_the_module_comes_back_so_it_can_be_NEUTRALISED(tmp_path):
    """This is the whole finding. Under subprocess the table lives in another
    process and cannot be touched, so the test can only ever assert that a
    label APPEARS — never that it came from the table (P359)."""
    body = ("TABLE = {'k': 'LABEL'}\n"
            "def main():\n"
            "    print('row:', TABLE.get('k', 'plain'))\n")
    mod = load_cli(_script(tmp_path, body))
    assert "LABEL" in run_cli(mod).stdout
    mod.TABLE = {}
    assert "LABEL" not in run_cli(mod).stdout, (
        "the label survived the table being emptied — presence cannot "
        "distinguish consumed from coincidental (P359)"
    )


def test_it_composes_with_assert_drives_output(tmp_path):
    """The two helpers exist for one job: run_cli makes the consumer
    patchable, assert_drives_output uses that to prove consumption."""
    from tests._guard_pins import assert_drives_output

    body = ("TABLE = {'k': 'LABEL'}\n"
            "def main():\n"
            "    print('row:', TABLE.get('k', 'plain'))\n")
    mod = load_cli(_script(tmp_path, body))

    def _disable():
        original = mod.TABLE
        mod.TABLE = {}
        return lambda: setattr(mod, "TABLE", original)

    assert_drives_output(lambda: run_cli(mod).stdout, "LABEL", _disable)


# ==========================================================================
# Adoption — a helper nothing calls is decoration (P170/P345)
# ==========================================================================
def test_it_runs_a_REAL_repo_cli_and_keeps_its_exit_code(tmp_path):
    """The diagnostic P358b drove through a subprocess. Same output, same
    exit code, and now patchable."""
    stats = {
        "regime_ticks": {"SIDEWAYS": 4},
        "by_regime": {"SIDEWAYS": [
            {"name": "DarkPoolVolumeStrategy", "attempts": 4, "fires": 0}]},
        "archived": [],
    }
    f = tmp_path / "stats.json"
    f.write_text(json.dumps(stats), encoding="utf-8")
    r = run_cli(REPO / "scripts" / "kq_strategy_diagnostic.py",
                ["--path", str(f)])
    assert r.exit_code == 0
    assert "[STRUCTURAL]" in r.stdout


def test_the_harness_has_a_caller_outside_its_own_test():
    """P170 applied to itself (P345: a second uncalled helper would be the
    joke writing itself)."""
    root = pathlib.Path(__file__).parent
    callers = [p.name for p in root.glob("test_*.py")
               if "_cli_harness" in p.read_text(encoding="utf-8")
               and p.name != "test_p360_cli_harness.py"]
    assert callers, (
        "_cli_harness has no caller outside its own test — adopt it at a real "
        "site or it is the decoration P359 was built to detect"
    )


def test_the_returned_module_lets_a_PATH_caller_patch_without_a_second_load(
        tmp_path):
    """[P360] `CliResult.module` was decoration until this test — a field
    nothing read, in the helper built one hour after the detector for exactly
    that (P170/P359). A falsification probe defaulting it to None stayed
    green, which is the harness reporting a gap in the guard.

    It earns its place for the PATH caller: you get the module back and can
    neutralise it without a separate `load_cli`, which is the difference
    between this and subprocess."""
    body = ("TABLE = {'k': 'LABEL'}\n"
            "def main():\n"
            "    print('row:', TABLE.get('k', 'plain'))\n")
    r = run_cli(_script(tmp_path, body))
    assert "LABEL" in r.stdout
    assert r.module is not None, "no handle came back, so nothing is patchable"
    r.module.TABLE = {}
    assert "LABEL" not in run_cli(r.module).stdout, (
        "the handle does not refer to the module that produced the output"
    )

"""[P189] training/run_training.py invoked two scripts that do not exist.

P186 found `make drl` pointing at a trainer that was never in the tree. This is
the same defect one layer up, in the orchestrator that `make all`, `make quick`
and `make gmm` all delegate to:

    run_gmm -> root_dir/scripts/retrain_gmm.py   (the only copy of that file is
               in archive/gmm_research/, and it trains the GLOBAL 6-component
               model that main.py:3552 treats as the legacy fallback)
    run_drl -> root_dir/train_drl_full.py        (off by one directory; the
               file is in training/)

So the documented full pipeline could not complete. The failure was a bare
subprocess returncode 2 arriving after however many hours the preceding steps
had already burned, and `main()` discarded every return value and returned
None, so the process exited 0 either way — `make all` reported success whether
it trained anything or not.

These tests are structural: nothing here runs a training job. They check that
every script the orchestrator can invoke exists, that every flag it passes is
one the target script defines, that preflight refuses to start when a path is
wrong, and that a failed stage reaches the shell as a nonzero exit.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING = REPO_ROOT / "training"

sys.path.insert(0, str(TRAINING))
import run_training  # noqa: E402


@pytest.fixture
def orch(tmp_path):
    """Instantiated with a temp output dir — __init__ mkdirs it."""
    return run_training.TrainingOrchestrator(
        data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "models")
    )


def test_the_orchestrator_is_where_this_expects_it():
    """A moved module would make every test below vacuously pass."""
    assert (TRAINING / "run_training.py").is_file()
    assert run_training.TrainingOrchestrator.SCRIPTS, "SCRIPTS map is empty"


def test_every_script_the_orchestrator_can_invoke_exists(orch):
    missing = {k: str(orch._script(k)) for k in orch.SCRIPTS
               if not orch._script(k).exists()}
    assert not missing, (
        f"training/run_training.py would invoke files that do not exist: "
        f"{missing}. This is how `make all` spent its life dying in step 1 on "
        f"scripts/retrain_gmm.py, and `make quick` in step 3 on a "
        f"train_drl_full.py resolved one directory too high."
    )


def test_preflight_passes_on_the_real_tree(orch):
    assert orch.preflight() is True


def test_preflight_fails_when_a_script_is_missing(orch, monkeypatch, caplog):
    """The check must be able to fail, or it reports nothing.

    Falsification for the test above: with a bad path injected, preflight has
    to return False and name the key and the resolved path.
    """
    monkeypatch.setitem(orch.SCRIPTS, "gmm", ("script_dir", "no/such/trainer.py"))
    with caplog.at_level("ERROR"):
        assert orch.preflight() is False
    joined = caplog.text
    assert "gmm" in joined and "no/such/trainer.py" in joined, (
        f"preflight failed without saying which script or where it looked: "
        f"{joined!r}"
    )


def test_main_exits_nonzero_when_preflight_fails(monkeypatch, tmp_path):
    """main() used to discard every return value and exit 0 regardless."""
    monkeypatch.setattr(sys, "argv", [
        "run_training.py", "--gmm",
        "--data-dir", str(tmp_path / "d"), "--output-dir", str(tmp_path / "m"),
    ])
    monkeypatch.setitem(run_training.TrainingOrchestrator.SCRIPTS,
                        "drl", ("script_dir", "no/such/trainer.py"))
    assert run_training.main() == 1


def test_main_exits_nonzero_when_a_stage_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "run_training.py", "--gmm",
        "--data-dir", str(tmp_path / "d"), "--output-dir", str(tmp_path / "m"),
    ])
    monkeypatch.setattr(run_training.TrainingOrchestrator, "_run",
                        lambda self, cmd, name: False)
    assert run_training.main() == 1


def test_main_exits_zero_when_the_stage_succeeds(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "run_training.py", "--gmm",
        "--data-dir", str(tmp_path / "d"), "--output-dir", str(tmp_path / "m"),
    ])
    monkeypatch.setattr(run_training.TrainingOrchestrator, "_run",
                        lambda self, cmd, name: True)
    assert run_training.main() == 0


def test_the_gmm_step_trains_the_per_asset_models_the_runtime_loads(orch):
    """main.py:3520 tries per-asset first and calls the global one legacy.

    The orchestrator pointed at the archived global trainer. Even had that path
    existed, a `make all` would have refreshed the fallback and left the models
    the runtime actually reads untouched.
    """
    gmm = orch._script("gmm")
    assert gmm.name == "train_per_asset_gmm.py", (
        f"the GMM step trains {gmm.name}. The runtime loads "
        f"models/gmm/<ASSET>/gmm_model.pkl first (main.py 'Try per-asset "
        f"models first (v7)') and only falls back to the global model."
    )
    assert "archive" not in gmm.parts, (
        f"the GMM step points into archive/: {gmm}"
    )


def _captured_commands(orch, monkeypatch):
    """Run every stage with the subprocess layer stubbed out; collect argv."""
    seen = []
    monkeypatch.setattr(run_training.TrainingOrchestrator, "_run",
                        lambda self, cmd, name: seen.append(list(cmd)) or True)
    orch.run_gmm()
    orch.run_dt(epochs=1, assets=["BTC"])
    orch.run_drl(assets=["BTC"])
    orch.run_sentiment(epochs=1)
    return seen


def test_the_drl_command_states_its_venue(orch, monkeypatch):
    """After P179 the env charges real fees; which venue must be in the command."""
    drl = [c for c in _captured_commands(orch, monkeypatch)
           if any("train_drl_full" in a for a in c)]
    assert len(drl) == 1, f"expected one DRL command, got {drl}"
    assert "--venue" in drl[0] and "--fee-side" in drl[0], (
        f"the orchestrator's DRL step does not state the venue it prices: "
        f"{drl[0]}. A model trained at Kraken's 26bps taker and one trained "
        f"for the Coinbase nano sleeve at 3bps are not interchangeable, and "
        f"nothing in the command would say which this was."
    )


def _argparse_flags(path: Path) -> set:
    """Flags a script defines, by parsing add_argument — no import.

    These modules import torch/gymnasium at scope, which is not installed
    everywhere.
    """
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    flags = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flags.add(arg.value)
    return flags


def test_every_flag_the_orchestrator_passes_is_defined_by_its_target(
        orch, monkeypatch):
    """argparse exits 2 on an unknown flag — after the job has been queued."""
    unknown = []
    for cmd in _captured_commands(orch, monkeypatch):
        script = next((a for a in cmd if a.endswith(".py")), None)
        assert script, f"no script in command: {cmd}"
        known = _argparse_flags(Path(script))
        assert known, f"no argparse flags parsed out of {script}"
        for arg in cmd:
            if arg.startswith("--") and arg not in known:
                unknown.append((Path(script).name, arg))
    assert not unknown, (
        f"run_training.py passes flags the target script does not define: "
        f"{unknown}."
    )


def test_the_makefile_pipeline_targets_thread_the_venue_through():
    """`make all DRL_VENUE=x` must mean what `make drl DRL_VENUE=x` means."""
    text = (TRAINING / "Makefile").read_text(encoding="utf-8")
    orchestrator_flags = _argparse_flags(TRAINING / "run_training.py")
    for target in ("all", "quick"):
        block = re.search(rf"^{target}:\n((?:\t.*\n)+)", text, re.M)
        assert block, f"the `{target}` target is gone"
        recipe = block.group(1)
        assert "--venue" in recipe and "--fee-side" in recipe, (
            f"`make {target}` runs the DRL step without stating a venue, so it "
            f"silently takes run_training.py's own default while `make drl` "
            f"honours DRL_VENUE. Two ways to run the same trainer, charging "
            f"different fees."
        )
        for flag in re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", recipe):
            assert flag in orchestrator_flags, (
                f"`make {target}` passes {flag}, which run_training.py does "
                f"not define; argparse exits 2."
            )

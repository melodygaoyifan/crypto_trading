"""[P186] `make drl` invoked a script that does not exist.

training/Makefile advertised `make drl   DRL v5.5 (~6-12h)` and ran
`drl/train_drl_v55.py`. That file is not in the tree. The target had been
failing with "can't open file" for as long as anyone ran it, and the help text
gave no hint — an operator following the documented path to retrain gets a
Python error about a path, not "this target is stale".

It also passed only `SOL_60m.parquet` while v5.5 is documented as Cross-Asset
BTC/ETH/SOL, so even had the script existed, `make drl` would have retrained
one asset and read as three.

These tests parse the Makefile and check that every script a recipe invokes is
present, and that the DRL targets cover the asset set `make check` verifies
data for. They are deliberately structural: nothing here runs a training job.
"""

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING = REPO_ROOT / "training"
MAKEFILE = TRAINING / "Makefile"

# `$(PYTHON) [-X utf8] [-u] <script.py>` — the script is what must exist.
_INVOKE = re.compile(r"\$\(PYTHON\)((?:\s+-\S+(?:\s+\S+)?)*)\s+(\S+\.py)")


def _recipe_lines():
    """Every line of every recipe (tab-indented), with $(VAR) left intact."""
    for raw in MAKEFILE.read_text(encoding="utf-8").splitlines():
        if raw.startswith("\t"):
            yield raw.strip()


def test_the_makefile_is_where_this_expects_it():
    """A moved Makefile would make every test below vacuously pass."""
    assert MAKEFILE.is_file(), (
        f"{MAKEFILE} is gone. These tests would otherwise report nothing "
        f"wrong with a Makefile they never read."
    )


def test_every_invoked_script_exists():
    missing = []
    for line in _recipe_lines():
        for m in _INVOKE.finditer(line):
            script = m.group(2)
            if "$(" in script:  # variable-built path, not resolvable here
                continue
            if not (TRAINING / script).is_file():
                missing.append(script)
    assert not missing, (
        f"training/Makefile invokes scripts that do not exist: "
        f"{sorted(set(missing))}. This is how `make drl` spent its life "
        f"failing on drl/train_drl_v55.py while `make help` advertised it as "
        f"the way to retrain DRL v5.5."
    )


class TestTheDrlTargetsCoverTheDocumentedAssets:
    def test_drl_assets_matches_what_check_verifies(self):
        """`make check` tests for three parquets; `make drl` used one asset."""
        text = MAKEFILE.read_text(encoding="utf-8")
        m = re.search(r"^DRL_ASSETS\s*\?=\s*(.+)$", text, re.M)
        assert m, (
            "DRL_ASSETS is gone. The DRL target is back to a hardcoded asset "
            "list, which is how it came to train SOL alone."
        )
        assets = set(m.group(1).split())
        checked = set(re.findall(r"\$\(DATA_DIR\)/raw/(\w+)_60m\.parquet", text))
        assert assets == checked, (
            f"`make drl` trains {sorted(assets)} but `make check` verifies "
            f"data for {sorted(checked)}. Whichever is right, an operator who "
            f"runs check-then-drl gets a different answer from each."
        )

    def test_the_drl_recipe_uses_the_trainer_that_exists(self):
        text = MAKEFILE.read_text(encoding="utf-8")
        drl_block = re.search(r"^drl:\n((?:\t.*\n)+)", text, re.M)
        assert drl_block, "the `drl` target is gone"
        assert "train_drl_full.py" in drl_block.group(1), (
            "the drl target points at something other than train_drl_full.py "
            "— the only DRL trainer in the tree, and the one carrying the "
            "P179-P184 cost and selection fixes."
        )

    @pytest.mark.parametrize("target", ["drl", "drl-fast"])
    def test_the_venue_is_explicit(self, target):
        """After P179 the env charges real fees; which venue must be visible."""
        text = MAKEFILE.read_text(encoding="utf-8")
        block = re.search(rf"^{re.escape(target)}:\n((?:\t.*\n)+)", text, re.M)
        assert block, f"the `{target}` target is gone"
        assert "--venue" in block.group(1) and "--fee-side" in block.group(1), (
            f"`make {target}` does not state the venue it prices. The default "
            f"is Kraken at 26bps taker; a model trained for the Coinbase nano "
            f"sleeve at 3bps and one trained for Kraken are not "
            f"interchangeable, and nothing in the command would say which "
            f"this was."
        )


def test_the_documented_flags_are_accepted_by_the_trainer():
    """A recipe flag the trainer rejects fails only after the job starts.

    Parses train_drl_full.py's argparse calls rather than importing it — the
    module imports gymnasium at scope, which is not installed everywhere.
    """
    import ast

    trainer = TRAINING / "train_drl_full.py"
    tree = ast.parse(trainer.read_text(encoding="utf-8-sig"))
    known = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    known.add(arg.value)
    assert known, "no argparse flags found — the parse above is not working"

    text = MAKEFILE.read_text(encoding="utf-8")
    unknown = []
    for target in ("drl", "drl-fast"):
        block = re.search(rf"^{re.escape(target)}:\n((?:\t.*\n)+)", text, re.M)
        if not block:
            continue
        for flag in re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", block.group(1)):
            if flag not in known:
                unknown.append((target, flag))
    assert not unknown, (
        f"training/Makefile passes flags train_drl_full.py does not define: "
        f"{unknown}. argparse exits 2 on these, so the failure arrives after "
        f"the operator has queued a 6-12h job."
    )

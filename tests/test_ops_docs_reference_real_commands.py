"""[P190] The operations runbook documented 14 scripts that never existed.

`docs/HMATS_Architecture_Part5_Operations_v10.md` — the daily checklist, the
weekly and monthly procedures, and 紧急程序 1 (emergency flatten) — told the
operator to run 14 files under `/opt/hmats/scripts/`. None of them are in the
tree, and `git log --all --diff-filter=A` finds no commit that ever added one.
The first command in the emergency-flatten procedure was
`python /opt/hmats/scripts/emergency_flatten.py --confirm`.

Second, independent defect: `scripts/` is not COPYed into `Dockerfile.engine`,
yet CLAUDE.md documents three diagnostics as
`docker exec hmats-engine python -X utf8 scripts/<x>.py`. CLAUDE.md:1118 had
already noticed this in passing ("scp the script in first — scripts/ isn't
baked into the image") without the other call sites being corrected.

Same class as P186 (`make drl` -> a trainer that never existed) and P189
(`run_training.py` -> two paths that never existed). The difference is where it
lands: a stale build target wastes an afternoon; a stale runbook is discovered
during an incident.

These tests read only the docs and the Dockerfile. Nothing here runs a command.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((REPO_ROOT / "docs").glob("*.md"))
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
DOCKERFILE_ENGINE = REPO_ROOT / "Dockerfile.engine"

# A path token that looks like a script this repo owns.
_SCRIPT = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*(?:scripts|tools)/[\w.-]+\.py)")
_FENCE = re.compile(r"^```(\w*)\s*$")


def _shell_blocks(path: Path):
    """Yield (line_no, line) for lines inside ```bash / ```sh / ```shell fences.

    Prose is deliberately excluded: a P-entry that *describes* a broken path
    ("run_gmm pointed at scripts/retrain_gmm.py") is history, not an
    instruction. What this gate covers is the commands a reader is told to run.
    """
    lang = None
    for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        m = _FENCE.match(line.rstrip())
        if m:
            lang = None if lang is not None else (m.group(1) or "").lower()
            continue
        if lang in ("bash", "sh", "shell", "console"):
            yield i, line


def _documented_files():
    return [CLAUDE_MD] + DOCS


def test_the_docs_this_reads_are_where_it_expects():
    """A moved runbook would make every test below vacuously pass."""
    assert CLAUDE_MD.is_file()
    assert DOCS, "docs/*.md matched nothing"
    assert DOCKERFILE_ENGINE.is_file()


@pytest.mark.parametrize("doc", _documented_files(), ids=lambda p: p.name)
def test_every_script_a_documented_command_runs_exists(doc):
    missing = []
    for lineno, line in _shell_blocks(doc):
        for rel in _SCRIPT.findall(line):
            if not (REPO_ROOT / rel).is_file():
                missing.append(f"{doc.name}:{lineno} -> {rel}")
    assert not missing, (
        "documented commands invoke scripts that do not exist:\n  "
        + "\n  ".join(missing)
        + "\n\nAn operator following the runbook gets \"can't open file\". If "
        "the capability is genuinely absent, say so in the doc ([未实现]) "
        "rather than leaving a command that reads as working."
    )


def test_no_doc_still_points_at_the_phantom_ops_script_directory():
    """The 14 /opt/hmats/scripts/*.py files were never in any commit."""
    offenders = []
    for doc in _documented_files():
        for lineno, line in _shell_blocks(doc):
            if "/opt/hmats/scripts/" in line:
                offenders.append(f"{doc.name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "commands still reference /opt/hmats/scripts/, which does not exist "
        "in the image (scripts/ is copied by allowlist, see Dockerfile.engine "
        "[P190]):\n  " + "\n  ".join(offenders)
    )


def _engine_copied_scripts():
    """Script basenames the engine image actually contains."""
    copied = set()
    for line in DOCKERFILE_ENGINE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("COPY"):
            continue
        for tok in line.split():
            if tok.startswith("scripts/") and tok.endswith(".py"):
                copied.add(Path(tok).name)
    return copied


def test_the_engine_image_copies_the_diagnostics_the_docs_exec_into_it():
    """`docker exec hmats-engine python scripts/x.py` needs x.py in the image."""
    copied = _engine_copied_scripts()
    assert copied, (
        "Dockerfile.engine copies no scripts at all, so every documented "
        "`docker exec hmats-engine python ... scripts/<x>.py` fails with "
        "\"can't open file\" — which is the state P190 found."
    )

    exec_re = re.compile(
        r"docker\s+(?:compose\s+)?exec\s+\S+\s+.*?(?<![\w/.-])scripts/([\w.-]+\.py)")
    referenced = {}
    for doc in _documented_files():
        for lineno, line in _shell_blocks(doc):
            for name in exec_re.findall(line):
                referenced.setdefault(name, f"{doc.name}:{lineno}")

    assert referenced, (
        "no `docker exec ... scripts/<x>.py` command found in any doc — the "
        "regex above has stopped matching, so this test now checks nothing."
    )
    not_in_image = {n: where for n, where in referenced.items() if n not in copied}
    assert not not_in_image, (
        f"documented `docker exec` commands name scripts that are not in the "
        f"engine image: {not_in_image}. Either add the file to the allowlist "
        f"COPY in Dockerfile.engine, or change the doc to the scp-it-in "
        f"procedure. Do not leave a command that cannot run."
    )


def test_the_allowlist_stays_an_allowlist():
    """`COPY scripts/ ./scripts/` would put order-placing code in the image.

    scripts/ holds launch_live.py, coinbase_test_order.py and
    coinbase_flatten.py. P141 keeps live order placement out of the automated
    path; a blanket copy quietly undoes that.
    """
    text = DOCKERFILE_ENGINE.read_text(encoding="utf-8")
    assert not re.search(r"^COPY\s+.*\bscripts/\s+\./scripts/\s*$", text, re.M), (
        "Dockerfile.engine copies the whole scripts/ directory. That includes "
        "launch_live.py and coinbase_test_order.py. Copy the specific "
        "diagnostics instead."
    )
    forbidden = {"launch_live.py", "coinbase_test_order.py", "coinbase_flatten.py"}
    inside = forbidden & _engine_copied_scripts()
    assert not inside, (
        f"order-placing scripts baked into the live trading image: {inside}"
    )


def test_the_operations_runbook_does_not_drive_the_system_with_systemd():
    """The live deployment is docker-compose.hetzner.yml, not a systemd unit.

    deploy/systemd/hmats.service is a v5.1.0 artifact that still launches
    `main.py --mode paper` from a venv. `sudo systemctl stop hmats` in an
    incident stops nothing.
    """
    runbook = REPO_ROOT / "docs" / "HMATS_Architecture_Part5_Operations_v10.md"
    offenders = [f"{n}: {l.strip()}" for n, l in _shell_blocks(runbook)
                 if re.search(r"systemctl\s+\w+\s+hmats", l)]
    assert not offenders, (
        "the operations runbook drives hmats through systemd:\n  "
        + "\n  ".join(offenders)
        + "\nThe engine runs as the `hmats-engine` container from "
          "docker-compose.hetzner.yml."
    )

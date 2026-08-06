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


def _all_fenced_lines(path: Path):
    """Yield (line_no, line) for every line inside any ``` fence.

    Wider than _shell_blocks: docs here put runnable commands in untagged
    fences too (Part4's deploy checklist sits next to a directory tree).
    """
    inside = False
    for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if _FENCE.match(line.rstrip()):
            inside = not inside
            continue
        if inside:
            yield i, line


def _documented_files():
    # [P191] Root-level markdown too. README_DEPLOY_HETZNER.md is a deployment
    # doc — the exact class of file P190/P191 found rotted — and scanning only
    # docs/ + CLAUDE.md left it uncovered. It is clean today; the point is that
    # it stays that way.
    return sorted(set([CLAUDE_MD] + DOCS + list(REPO_ROOT.glob("*.md"))))


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


DOCKERIGNORE = REPO_ROOT / ".dockerignore"


def _dockerignore_patterns():
    return [
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_every_allowlisted_script_survives_dockerignore():
    """[P192] Being named in a COPY is not the same as being buildable.

    The test above asserts the diagnostics appear in Dockerfile.engine. They
    did — and the image still could not build, because `.dockerignore` excludes
    `scripts/`, so the COPY resolved against an empty build context and
    `docker build` died with `"/scripts/why_no_trade.py": not found`. The gate
    that was supposed to guarantee "the documented command works" was satisfied
    by a Dockerfile line that could never run.

    Checking membership in one file while the neighbouring file silently
    removes it is the same shape as P170/P176: a check whose subject was
    already gone.
    """
    patterns = _dockerignore_patterns()
    copied = _engine_copied_scripts()
    assert copied, (
        "Dockerfile.engine copies no scripts at all — the allowlist this test "
        "guards has vanished; see P190."
    )

    dir_excluded = [p for p in patterns
                    if p.rstrip("/") == "scripts" or p.startswith("scripts/**")]
    # Asserted, not branched on: the exclusion is itself a safety property.
    # Without it the whole directory enters the build context, which is what
    # P141 (no order-placing code in the live image) exists to prevent. If this
    # ever fails, do not delete the negations — restore the exclusion.
    assert dir_excluded, (
        "`scripts/` is no longer excluded in .dockerignore, so the entire "
        "directory — including launch_live.py, coinbase_test_order.py and "
        "coinbase_flatten.py — is now in the engine build context. P141 keeps "
        "order-placing code out of the live trading image. Restore the "
        "`scripts/` exclusion along with the `!scripts/<file>` re-includes."
    )

    negated = {p[len("!scripts/"):] for p in patterns if p.startswith("!scripts/")}
    missing = sorted(name for name in copied if name not in negated)
    assert not missing, (
        f".dockerignore excludes scripts/ via {dir_excluded[0]!r} but never "
        f"re-includes {missing}. `docker build` fails on the COPY at "
        f"Dockerfile.engine with \"not found\" — the engine image cannot be "
        f"built at all, which is the P192 breakage. Add a `!scripts/<name>` "
        f"line for each, placed AFTER the `scripts/` line (Docker takes the "
        f"last matching pattern)."
    )

    # A negation naming a file that does not exist is dead config: it re-includes
    # nothing, and the failure only shows up in a build log nobody reads.
    ghosts = sorted(n for n in negated if not (REPO_ROOT / "scripts" / n).is_file())
    assert not ghosts, (
        f".dockerignore re-includes scripts that do not exist on disk: {ghosts}. "
        f"Either restore the file or drop the stale `!scripts/` line."
    )


DEPLOY_GUIDE = REPO_ROOT / "docs" / "hetzner_deployment_guide.md"
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "hetzner_deploy.sh"


def test_the_deployment_guide_documents_the_deployment_that_actually_happens():
    """The guide's production path must be the one hetzner_deploy.sh performs.

    It previously walked the operator through `docker build -t hmats:6.8.0 .`
    — the root Dockerfile (v5.1.0 layout) — and a hand-rolled single container
    named `hmats-paper`, mounting host dirs at /var/log/hmats and
    /var/lib/hmats. The deploy script builds Dockerfile.engine via
    docker-compose.hetzner.yml and brings up hmats-engine + hmats-api, whose
    state lives in /opt/hmats/data and /opt/hmats/logs. Every operational doc
    says `docker exec hmats-engine`; the build guide never named that container.
    """
    guide = DEPLOY_GUIDE.read_text(encoding="utf-8")
    for token in ("docker-compose.hetzner.yml", "hmats-engine",
                  "scripts/hetzner_deploy.sh"):
        assert token in guide, (
            f"docs/hetzner_deployment_guide.md never mentions {token}, which is "
            f"what scripts/hetzner_deploy.sh actually deploys. A build guide "
            f"that produces a different container than the deploy script is a "
            f"guide for a system nobody runs."
        )

    # The APP_DIR the deploy script hardcodes must be the one the guide clones into.
    m = re.search(r'^APP_DIR="([^"]+)"', DEPLOY_SCRIPT.read_text(encoding="utf-8"), re.M)
    assert m, "APP_DIR is gone from scripts/hetzner_deploy.sh"
    app_dir = m.group(1).replace("${REMOTE_USER}", "hmats")
    assert app_dir == "/home/hmats/hmats/app", app_dir
    assert "~/hmats/app" in guide, (
        f"the guide does not clone into {app_dir}; hetzner_deploy.sh cds there "
        f"unconditionally and will fail on the first `git pull`."
    )


def _legacy_systemd_section_lines():
    """(first, last) line numbers of deployment-guide section 7, or None.

    [P191] Only that section is exempt — not the whole file. The first cut of
    this test skipped DEPLOY_GUIDE entirely, and that hid four more places in
    the same file (9.1 更新代码, 9.2 更新模型, 快速参考, Paper→Live) that still
    drove the engine through systemd while reading as current instructions.
    An exemption the width of a file is not a carve-out, it is a blind spot.
    """
    lines = DEPLOY_GUIDE.read_text(encoding="utf-8").splitlines()
    start = next((i for i, l in enumerate(lines, 1)
                  if l.startswith("## 7. Systemd")), None)
    if start is None:
        return None
    end = next((i for i, l in enumerate(lines, 1)
                if i > start and l.startswith("## ")), len(lines))
    return start, end


def test_systemd_instructions_are_confined_to_the_guide_that_labels_them_legacy():
    """`systemctl stop hmats` stops nothing — there is no such unit in prod.

    docs/hetzner_deployment_guide.md section 7 keeps the systemd recipe for the
    non-Docker install, prefixed with an explicit legacy banner and a table of
    docker equivalents. Anywhere else — including elsewhere in that same guide
    — it reads as a live instruction.
    """
    # Every fenced block, not just ```bash. The instance this was written for
    # — Part4's deploy checklist, `6. systemctl restart hmats` — sat in an
    # untagged fence alongside a directory tree. Scoping this to ```bash would
    # have exempted the one line that prompted the check.
    legacy = _legacy_systemd_section_lines()
    assert legacy, "section 7 of the deployment guide is gone or was renumbered"
    offenders = []
    for doc in _documented_files():
        for lineno, line in _all_fenced_lines(doc):
            if doc == DEPLOY_GUIDE and legacy[0] <= lineno <= legacy[1]:
                continue
            if re.search(r"systemctl\s+(?:is-active\s+)?[\w-]*\s*hmats\b", line):
                offenders.append(f"{doc.name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "docs outside the deployment guide drive hmats through systemd:\n  "
        + "\n  ".join(offenders)
        + "\nProduction is the `hmats-engine` container from "
          "docker-compose.hetzner.yml."
    )


def test_the_legacy_systemd_section_says_it_is_legacy():
    """Falsification guard for the exemption above.

    The test before this one exempts the deployment guide. That exemption is
    only safe while the section carries the banner — otherwise the exemption
    is a hole in the gate rather than a considered carve-out.
    """
    guide = DEPLOY_GUIDE.read_text(encoding="utf-8")
    m = re.search(r"^## 7\. Systemd[^\n]*\n(.*?)^## 8\.", guide, re.M | re.S)
    assert m, "section 7 of the deployment guide is gone or was renumbered"
    head = m.group(1)[:1500]
    assert "P190" in head and "docker compose" in head, (
        "the systemd section no longer carries the legacy banner + the table of "
        "docker equivalents, but test_systemd_instructions_are_confined_to_"
        "the_guide_that_labels_them_legacy still exempts this file. Either "
        "restore the banner or drop the exemption."
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

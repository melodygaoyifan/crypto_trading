"""[P362] A refusal that asserted one cause, and the cause had rotted.

`tripwire_check` printed, whenever it found no reports:

    "no slope_*.json under <dir> — the weekly calibrator has not written
     there yet (first cron run 2026-08-11)"

True when written. **False since 2026-08-10**, because the calibrator has
written every Monday since. Run from a laptop it sends the reader to
investigate a cron that is working — the real cause is that this tool reads
the CONTAINER's data volume (P213), which the message never mentions.

I hit this myself while sweeping for outstanding work: the tool told me the
calibrator had not run, and the server had two reports sitting in that exact
directory. **An alert must name its cause from the DATA, not from a guess
baked into the format string (P155)** — and a hardcoded date inside a refusal
is a claim that rots silently, since the refusal path is the one nobody reads
until something is wrong.

The split is the same missing-vs-neutral distinction this file keeps making
(P2/P199): *I could not look* and *I looked and found nothing* are different
states with different fixes, and collapsing them costs a diagnosis.
"""

import pathlib

import main
from tests._cli_harness import run_cli

REPO = pathlib.Path(main.__file__).parent
TOOL = REPO / "analytics" / "calibration" / "tripwire_check.py"


def _run(reports_dir):
    return run_cli(TOOL, ["--reports-dir", str(reports_dir)])


def test_a_missing_directory_says_WRONG_ENVIRONMENT(tmp_path):
    """The case that misled me. It must not blame the calibrator."""
    r = _run(tmp_path / "nope")
    assert r.exit_code == 2, "a refusal must not read as a verdict (P199)"
    msg = r.stderr
    assert "does not exist here" in msg
    assert "docker exec" in msg, "the message does not name the fix"
    assert "calibrator has not written" not in msg, (
        "it still blames the weekly calibrator for what is an environment "
        "problem — the diagnosis this entry exists to stop"
    )


def test_an_empty_directory_DOES_blame_the_cron(tmp_path):
    """The other half. Quieting the wrong cause must not quiet the right one
    (P248): when the directory really is there and empty, the cron IS the
    thing to check."""
    d = tmp_path / "evidence"
    d.mkdir()
    r = _run(d)
    assert r.exit_code == 2
    assert "holds no slope_*.json" in r.stderr
    assert "cron" in r.stderr


def test_neither_refusal_reads_as_NOT_FIRED():
    """P199's rule, which the original message already had right and which
    the split must preserve on both branches."""
    import inspect
    src = TOOL.read_text(encoding="utf-8")
    i = src.index("TRIPWIRE CANNOT BE EVALUATED")
    block = src[i:i + 1800]
    assert block.count("is NOT") >= 2, (
        "a refusal branch lost the 'this is not a verdict' clause"
    )
    assert inspect  # (import kept meaningful for readers)


def test_no_hardcoded_date_survives_in_the_refusal():
    """The rot itself: a date literal inside a message nobody reads until
    something is wrong will be wrong by the time it is read."""
    src = TOOL.read_text(encoding="utf-8")
    i = src.index("TRIPWIRE CANNOT BE EVALUATED")
    block = src[i:i + 1800]
    import re
    dates = re.findall(r"20\d\d-\d\d-\d\d", block)
    assert not dates, (
        f"the refusal hardcodes {dates} — that is what went stale"
    )


def test_the_real_reports_directory_still_evaluates(tmp_path):
    """Anti-vacuity (P174): with reports present it must produce a VERDICT,
    not a refusal — otherwise the split above could be satisfied by a tool
    that never works at all."""
    d = tmp_path / "evidence"
    d.mkdir()
    (d / "slope_20260810_060000.json").write_text(
        '{"per_asset": {"BTC": {"verdict": "GATE-CLOSED"}}}', encoding="utf-8")
    (d / "slope_20260817_060000.json").write_text(
        '{"per_asset": {"BTC": {"verdict": "GATE-CLOSED"}}}', encoding="utf-8")
    r = _run(d)
    assert r.exit_code != 2, (
        f"real reports still produced a refusal: {r.stderr[:200]}"
    )
    assert "report day" in r.stdout

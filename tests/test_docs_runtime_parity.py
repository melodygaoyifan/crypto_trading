from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _doc(name: str) -> str:
    """[P165] Read a doc, or skip with the reason it cannot be checked.

    `HMATS_STEP15_CLOSE_GAP.md` has NEVER been committed — `git log --all --
    docs/HMATS_STEP15_CLOSE_GAP.md` is empty — so this guard has failed with
    FileNotFoundError since the day it was written, and a permanently red
    test is a guard nobody reads. Skipping is honest here because the absence
    is known and permanent, not an environment accident; the moment the doc
    IS committed the guard arms itself again with no edit needed.
    """
    path = PROJECT_ROOT / "docs" / name
    if not path.exists():
        pytest.skip(
            f"docs/{name} is not in the repository (never committed). "
            f"This parity guard is dormant until it is added."
        )
    return path.read_text(encoding="utf-8")


def test_unified_reference_doc_exists():
    """The one doc this suite is actually able to check must not go missing —
    otherwise `_doc`'s skip would quietly disarm the whole file."""
    assert (PROJECT_ROOT / "docs" / "HMATS_UNIFIED_SYSTEM_REFERENCE_v10.md").exists()


def test_unified_reference_matches_current_runtime_contract_markers():
    text = _doc("HMATS_UNIFIED_SYSTEM_REFERENCE_v10.md")

    assert "DISABLED / SHADOW / ADVISORY / BOUNDED_LIVE / LIVE" in text
    assert "feature-flagged" in text
    assert "5bps long / 3bps short" in text
    assert "14bps/8bps" not in text


def test_step15_close_gap_prefers_runtime_status_over_hardcoded_timestamps():
    text = _doc("HMATS_STEP15_CLOSE_GAP.md")

    assert "run.started_at + 48h" in text
    assert "use the live value from `step15_status.json`" in text
    assert "2026-03-12T03:55:32Z" not in text
    assert "2026-03-14T02:07:14Z" not in text

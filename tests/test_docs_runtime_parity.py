from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_unified_reference_matches_current_runtime_contract_markers():
    text = (PROJECT_ROOT / "docs" / "HMATS_UNIFIED_SYSTEM_REFERENCE_v10.md").read_text(
        encoding="utf-8"
    )

    assert "DISABLED / SHADOW / ADVISORY / BOUNDED_LIVE / LIVE" in text
    assert "feature-flagged" in text
    assert "5bps long / 3bps short" in text
    assert "14bps/8bps" not in text


def test_step15_close_gap_prefers_runtime_status_over_hardcoded_timestamps():
    text = (PROJECT_ROOT / "docs" / "HMATS_STEP15_CLOSE_GAP.md").read_text(
        encoding="utf-8"
    )

    assert "run.started_at + 48h" in text
    assert "use the live value from `step15_status.json`" in text
    assert "2026-03-12T03:55:32Z" not in text
    assert "2026-03-14T02:07:14Z" not in text

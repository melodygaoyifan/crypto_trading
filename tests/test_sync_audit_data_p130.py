"""
test_sync_audit_data_p130.py — sync script smoke + structure (P130, v3 1.3)
================================================================================

v3 Track A item 1.3 (P1-7). Verifies the sync_audit_data.sh script's
shape — required steps present, validation logic intact. End-to-end
sync requires SSH access to production and is exercised manually
(documented in script header).
"""
from __future__ import annotations

from pathlib import Path

import pytest


SYNC_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sync_audit_data.sh"


class TestSyncScriptStructure:
    def test_script_exists(self):
        assert SYNC_SCRIPT.exists(), f"P130 regression: {SYNC_SCRIPT.name} missing"

    def test_script_has_5_data_sources(self):
        """All 5 data sources documented in v3 prompt + audit pipeline
        must be pulled."""
        src = SYNC_SCRIPT.read_text(encoding="utf-8-sig")
        for source in (
            "equity_history.jsonl",
            "kq_firing_stats.json",
            "kq_firing_stats.jsonl",
            "ic_signals",
            "outcomes_",
            "proof_log_",
        ):
            assert source in src, (
                f"P130 regression: sync script no longer pulls {source}. "
                f"Audit pipeline will lose this data source."
            )

    def test_script_uses_set_minus_e(self):
        """set -euo pipefail must be present so failures don't silently
        return success."""
        src = SYNC_SCRIPT.read_text(encoding="utf-8-sig")
        assert "set -euo pipefail" in src, (
            "P130 regression: sync script missing 'set -euo pipefail'. "
            "Errors during scp will silently exit 0 and dataset will be "
            "stale without operator notice."
        )

    def test_script_validates_required_files_nonempty(self):
        """Required-files validation block must check size > 0."""
        src = SYNC_SCRIPT.read_text(encoding="utf-8-sig")
        assert "REQUIRED_FILES" in src and '"$f"' in src, (
            "P130 regression: required-files validation block changed. "
            "Empty file from broken scp would pass silently."
        )

    def test_script_validates_json_parseability(self):
        """Validation must json.loads at least one record per .jsonl."""
        src = SYNC_SCRIPT.read_text(encoding="utf-8-sig")
        assert "json.loads" in src, (
            "P130 regression: JSON parseability check removed. Cloud "
            "writer corruption would not be detected."
        )

    def test_script_writes_dated_directory(self):
        """Output directory must be date-stamped so historical syncs
        accumulate (don't overwrite)."""
        src = SYNC_SCRIPT.read_text(encoding="utf-8-sig")
        assert "%Y-%m-%d" in src or "+%Y-%m-%d" in src, (
            "P130 regression: sync script overwrites a single directory "
            "instead of date-stamping. Loses historical sync snapshots."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

"""
[P323] `positions_fresh` was permanently False — a malformed timestamp, and the
same bug that was already fixed once for the sibling file.

Symptom: /health reports `"positions_fresh": false` on every call, regardless
of how recently paper_positions.json was written (measured live: the file's
mtime and its own saved_at were seconds old, and the flag was still false).

Cause: main.py wrote

    "saved_at": datetime.now(timezone.utc).isoformat() + "Z"

but `isoformat()` on an aware datetime ALREADY emits "+00:00", so the value is
"...+00:00Z" — a double timezone marker. api/server._is_fresh then does
`ts.replace("Z", "+00:00")`, producing "...+00:00+00:00", which
`datetime.fromisoformat` rejects. The exception is swallowed and the function
returns False, so the field could never be True for any age.

THE PART THAT MAKES IT MORE THAN A COSMETIC FLAG: /health is
`healthy if (engine_state_fresh or positions_fresh)`. That OR exists so a
dashboard-export failure cannot alone mark the container unhealthy — exactly
the P160 scenario, where the dashboard writer can fail silently. With
positions_fresh stuck False the redundancy was gone: /health depended on
dashboard_state.json alone.

AND IT WAS ALREADY FIXED ONCE. main.py:~20068 carries a 2026-04-22 comment
describing this identical failure ("malformed ...+00:00Z (double tz marker).
API /health datetime.fromisoformat then failed silently -> fresh=False") for
dashboard_state.json. The sibling writer was left alone — a mitigation applied
to one instance of a class rather than to the class (P171, P226).
"""
from __future__ import annotations

import ast
import io
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_is_fresh():
    """Import _is_fresh WITHOUT importing api.server (which needs fastapi,
    absent in CI). Compiling the single function keeps this test runnable
    everywhere — a guard that skips on the machine where it matters is not a
    guard (P194)."""
    tree = ast.parse(io.open(REPO / "api" / "server.py", encoding="utf-8").read())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_is_fresh")
    ns = {"datetime": datetime, "timezone": timezone, "Dict": dict}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<f>", "exec"), ns)
    return ns["_is_fresh"]


class TestFreshnessParsesEveryTimestampShape:

    @pytest.mark.parametrize("shape", ["legacy_double_tz", "fixed_z", "offset"])
    def test_a_fresh_stamp_reads_fresh(self, shape):
        now = datetime.now(timezone.utc)
        ts = {
            # what the writer emitted before this fix — still on the volume
            # until each file is next rewritten, so it MUST keep parsing
            "legacy_double_tz": now.isoformat() + "Z",
            "fixed_z": now.isoformat().replace("+00:00", "Z"),
            "offset": now.isoformat(),
        }[shape]
        assert _load_is_fresh()({"saved_at": ts}, 15000) is True, (
            f"{shape} ({ts}) did not parse — positions_fresh would be stuck "
            f"False regardless of the file's real age")

    def test_the_check_still_fails_on_genuinely_stale(self):
        """The fix must not neuter the check — that would trade a false
        'stale' for a false 'fresh', which is the worse direction."""
        old = (datetime.now(timezone.utc) - timedelta(days=30))
        f = _load_is_fresh()
        assert f({"saved_at": old.isoformat() + "Z"}, 15000) is False
        assert f({"saved_at": old.isoformat().replace("+00:00", "Z")}, 15000) is False

    def test_missing_or_garbage_is_not_fresh(self):
        f = _load_is_fresh()
        for ts in ("", None, "not-a-date", 12345):
            assert f({"saved_at": ts}, 15000) is False


class TestTheWriterEmitsAParseableStamp:

    def test_saved_at_has_no_double_timezone_marker(self):
        """THE BUG, at the source. `isoformat() + "Z"` on an aware datetime
        yields '+00:00Z'."""
        src = io.open(REPO / "main.py", encoding="utf-8").read()
        i = src.index('"saved_at": datetime.now(timezone.utc)')
        stmt = src[i:i + 200]
        assert 'isoformat() + "Z"' not in stmt, (
            "paper_positions.json's saved_at is being written with a double "
            "timezone marker again — /health's positions_fresh will be "
            "permanently False (P323)")
        assert '.replace(' in stmt and '"+00:00", "Z"' in stmt

    def test_the_written_stamp_round_trips_through_the_reader(self):
        """End-to-end: whatever the writer produces must satisfy the reader.
        The two halves live in different files and had drifted (P2)."""
        written = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        assert _load_is_fresh()({"saved_at": written}, 15000) is True

    def test_the_sibling_writer_that_was_fixed_first_stays_fixed(self):
        """dashboard_state.json's stamp was corrected on 2026-04-22 for this
        exact reason. Pinned so the pair cannot diverge again — fixing one
        instance of a class and not the class is what produced P323."""
        src = io.open(REPO / "main.py", encoding="utf-8").read()
        i = src.index("_runtime_ts = datetime.now(timezone.utc)")
        assert '.replace("+00:00", "Z")' in src[i:i + 160]

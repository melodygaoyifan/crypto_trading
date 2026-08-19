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


class TestTheClassIsClosedNotJustTheInstance:
    """[P323b] P323 fixed paper_positions.json; the same `isoformat() + "Z"`
    shape survived at seven further sites. That is exactly how P323 happened —
    the 2026-04-22 fix for dashboard_state.json was applied to one instance and
    not to the class (P171, P226). This guard closes it with a MECHANISM rather
    than another lesson (P280).
    """

    # Directories that actually run in production. tests/ and archive/ are
    # excluded: a test may legitimately construct the malformed shape to prove
    # the reader tolerates it (this file does).
    _DIRS = ("agents", "api", "core", "defense", "risk", "signals", "execution",
             "data_mgmt", "exchange", "infra", "analytics", "scripts",
             "integration", "market", "orchestration", "strategies")

    def _offenders(self):
        from tests._source_scan import code_only
        bad = []
        for d in self._DIRS:
            for f in (REPO / d).rglob("*.py"):
                if "archive" in f.parts or f.name.startswith("test_"):
                    continue
                # strip comments AND docstrings: every remaining mention of the
                # pattern in this repo is an explanation of the fix, and a
                # scanner that matches its own explanation is worthless (P177).
                src = code_only(f, strip_docstrings=True)
                if 'isoformat() + "Z"' in src or "isoformat() + 'Z'" in src:
                    bad.append(str(f.relative_to(REPO)))
        for f in (REPO / "main.py",):
            src = code_only(f, strip_docstrings=True)
            if 'isoformat() + "Z"' in src:
                bad.append("main.py")
        return sorted(bad)

    def test_no_production_site_appends_Z_to_an_isoformat(self):
        assert self._offenders() == [], (
            f"these append \"Z\" to isoformat(), producing the malformed "
            f"\"+00:00Z\" on an aware datetime (or mislabelling local time as "
            f"UTC on a naive one): {self._offenders()}. Use "
            f".isoformat().replace(\"+00:00\", \"Z\") or an _iso_utc() helper.")

    def test_the_scanner_can_actually_see_the_pattern(self):
        """ANTI-VACUITY (P174): if the stripper removed too much, the guard
        above would pass forever. Prove it fires on a constructed offender."""
        import tempfile
        from tests._source_scan import code_only
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "offender.py"
            p.write_text(
                'from datetime import datetime, timezone\n'
                'x = datetime.now(timezone.utc).isoformat() + "Z"\n',
                encoding="utf-8")
            assert 'isoformat() + "Z"' in code_only(p, strip_docstrings=True)

    def test_the_reconcile_script_cannot_reintroduce_the_broken_stamp(self):
        """That script writes the SAME `saved_at` field api/server._is_fresh
        parses, into the SAME paper_positions.json — so the malformed form
        there would re-break positions_fresh the next time an operator ran it.
        It is the one of the seven that was NOT merely latent."""
        src = io.open(REPO / "scripts" / "reconcile_flatten_2026_06_12.py",
                      encoding="utf-8").read()
        i = src.index('state["saved_at"]')
        assert '.replace("+00:00", "Z")' in src[i:i + 220]


class TestTheHelperNormalisesRatherThanRelabels:

    @pytest.mark.parametrize("mod", [
        "agents.microstructure_agent", "agents.model_alpha_agent",
        "agents.onchain_graph_alpha", "agents.onchain_sentiment_fusion",
        "agents.risk_agent",
    ])
    def test_iso_utc_emits_one_marker_and_fixes_naive(self, mod):
        import importlib
        f = importlib.import_module(mod)._iso_utc
        aware = datetime(2026, 8, 19, 7, 0, 0, tzinfo=timezone.utc)
        naive = datetime(2026, 8, 19, 7, 0, 0)
        assert f(aware) == "2026-08-19T07:00:00Z"
        # naive is NORMALISED to UTC, not merely reformatted: naive + "Z"
        # labels local time as UTC (P40/P97) — wrong, not just malformed.
        assert f(naive) == "2026-08-19T07:00:00Z"
        assert "+00:00" not in f(aware) and f(aware).endswith("Z")

    def test_the_envelope_contract_still_holds(self):
        """Existing agent tests assert asof_ts endswith("Z"). The fix must keep
        that contract while removing the double marker."""
        import importlib
        for mod in ("agents.risk_agent", "agents.model_alpha_agent"):
            assert importlib.import_module(mod)._iso_utc().endswith("Z")

"""[P336] An expected absence must not read as a fault.

`orchestration.sota_integration` safe_imports 21 modules; exactly two live only
under `archive/`, so every boot logged two WARNINGs about modules nobody
intends to have. An alert whose only resolutions are theatre or ignoring it is
how a real CRITICAL stops being read (P202/P303).
"""
from __future__ import annotations

import importlib
import io
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MOD = REPO / "orchestration" / "sota_integration.py"


def _src() -> str:
    return io.open(MOD, encoding="utf-8").read()


class TestExpectedAbsencesAreNotFaults:

    def test_an_archived_module_logs_INFO_naming_where_it_went(self, caplog):
        from orchestration.sota_integration import safe_import
        with caplog.at_level(logging.INFO):
            got = safe_import("infra.event_replay", "EventReplayManager",
                              archived="archive/infra/event_replay.py")
        assert got is None
        rec = [r for r in caplog.records if "event_replay" in r.message]
        assert rec, "nothing logged at all"
        assert all(r.levelno < logging.WARNING for r in rec)
        assert any("archive/infra/event_replay.py" in r.message for r in rec)

    def test_an_UNEXPECTED_absence_still_WARNs(self, caplog):
        """The half that stops this becoming a blanket mute: a guard weakened
        to admit a known case stops catching the unknown ones (P248)."""
        from orchestration.sota_integration import safe_import
        with caplog.at_level(logging.INFO):
            got = safe_import("totally.not.a.module", "Nope")
        assert got is None
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_only_the_two_known_archived_targets_are_marked(self):
        """Anti-rot: `archived=` must name a real archive path, and must not
        spread to modules that are simply missing."""
        import re
        src = _src()
        marked = re.findall(r'archived="([^"]+)"', src)
        assert sorted(marked) == ["archive/core/plugin_registry.py",
                                  "archive/infra/event_replay.py"]
        for rel in marked:
            assert (REPO / rel).exists(), rel

    def test_every_other_safe_import_target_still_resolves(self):
        """If a THIRD module goes missing it must surface as a WARNING, not be
        pre-absolved — so the rest of the roster is asserted importable."""
        import re
        src = _src()
        mods = sorted(set(re.findall(r'safe_import\("([^"]+)"', src)))
        archived = {"infra.event_replay", "core.plugin_registry"}
        missing = []
        for m in mods:
            if m in archived:
                continue
            try:
                importlib.import_module(m)
            except Exception:
                missing.append(m)
        assert not missing, f"newly-absent safe_import targets: {missing}"

    def test_the_inert_flag_is_annotated(self):
        """`enable_event_replay` reads as enabled and cannot take effect."""
        src = _src()
        i = src.index("enable_event_replay: bool = True")
        assert "INERT" in src[max(0, i - 500):i]

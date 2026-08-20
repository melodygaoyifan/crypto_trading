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


# ---------------------------------------------------------------------------
# [P343] The THIRD expected absence, left behind by P336.
# ---------------------------------------------------------------------------

class TestTheRegimeClassifierAbsenceIsExpected:
    """P336 fixed two of three expected-absence WARNINGs and left this one --
    on the very module the same session had just characterised as a deliberate
    non-ship (P214). A mitigation applied to one instance of a class is not
    applied to the class (P171/P226/P323).

    The module IS in the repo, so this cannot be tested by deleting it: what
    makes the absence expected is that `Dockerfile.engine` does not copy it,
    which is a property of the IMAGE, not of the checkout.
    """

    COORD = REPO / "orchestration" / "strategic_coordinator.py"
    TARGET = "training.regime.regime_classifier"

    def _coord_src(self) -> str:
        return io.open(self.COORD, encoding="utf-8").read()

    def test_an_ImportError_logs_INFO_naming_the_decision(self, caplog):
        """Behavioural, not a source pin: a comment proves the code was
        written, not that it runs (P234)."""
        import orchestration.strategic_coordinator as sc
        # setting a sys.modules entry to None makes `import` raise ImportError
        with caplog.at_level(logging.INFO):
            saved = sys.modules.get(self.TARGET, "absent")
            sys.modules[self.TARGET] = None  # type: ignore[assignment]
            try:
                c = sc.StrategicCoordinator()
            finally:
                if saved == "absent":
                    sys.modules.pop(self.TARGET, None)
                else:
                    sys.modules[self.TARGET] = saved
        assert c._regime_classifier is None
        rec = [r for r in caplog.records if "Regime Classifier" in r.message]
        assert rec, "the absence was not reported at all"
        assert all(r.levelno < logging.WARNING for r in rec), (
            [f"{r.levelname}:{r.message}" for r in rec])
        msg = " ".join(r.message for r in rec)
        assert "Dockerfile.engine" in msg, msg
        assert "operator" in msg, msg

    def test_a_REAL_init_failure_still_WARNs(self, caplog):
        """The half that stops this becoming a blanket mute (P248): if the
        module ever IS shipped and then fails to construct, that is a fault."""
        import types
        import orchestration.strategic_coordinator as sc

        class _Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("ctor exploded")

        fake = types.ModuleType(self.TARGET)
        fake.EnsembleRegimeClassifier = _Boom
        fake.RegimeClassifierConfig = object
        fake.RegimeFeatures = object
        fake.MarketRegime = object
        with caplog.at_level(logging.INFO):
            saved = sys.modules.get(self.TARGET, "absent")
            sys.modules[self.TARGET] = fake
            try:
                c = sc.StrategicCoordinator()
            finally:
                if saved == "absent":
                    sys.modules.pop(self.TARGET, None)
                else:
                    sys.modules[self.TARGET] = saved
        assert c._regime_classifier is None
        assert any(r.levelno >= logging.WARNING and "Regime Classifier" in r.message
                   for r in caplog.records), (
            "a module that imported and then failed to construct is a real "
            "fault and must stay loud")

    def test_the_message_stays_true__it_is_NOT_in_the_image(self):
        """Anti-rot. The INFO says 'not shipped by decision'; if someone ships
        it, that message becomes a lie and the absence stops being expected --
        so this must go red and send the author back to the arming decision
        (the P318 anti-rot pattern)."""
        for f in ("Dockerfile.engine", ".dockerignore"):
            src = io.open(REPO / f, encoding="utf-8").read()
            assert "training/regime" not in src, (
                f"{f} now references training/regime -- if the Ensemble Regime "
                f"Classifier is being shipped, its consumer at "
                f"strategic_coordinator.py:~595 becomes LIVE for the first "
                f"time. Re-open the P141 arming decision; do not just update "
                f"this test.")

    def test_the_consumer_is_still_gated_on_the_import(self):
        """If the absence stopped gating the consumer, an unshipped module
        would raise on the live path instead of staying inert."""
        src = self._coord_src()
        assert "if self._regime_classifier is not None:" in src

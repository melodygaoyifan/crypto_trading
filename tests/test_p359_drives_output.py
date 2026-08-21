"""[P359] The mechanism for "I checked the table and not its use".

Four sightings, three in one day, each found by a falsification probe and
then re-pinned BY HAND a different way:

  P312   a producer's report-building seam that nothing called
  P324   main() no longer calling decide_verdict
  P357   a `done()` check the test could not reach at all
  P358b  KNOWN_SILENT_CAUSES verified for shape while `_cause = None`
         collapsed every label and left ALL tests green

P170 named the class in 2026-04 ("a mechanism nothing calls is decoration")
and every fix since has been rewritten from scratch. That is the signature of
a missing affordance rather than of carelessness — the same reading that
produced `Probe.near` in P357 after four rounds of sibling ambiguity.

**The insight the helper encodes: presence cannot distinguish CONSUMED from
COINCIDENTAL.** A test that looks for a value in the output passes whether the
consumer read the table or produced that value some other way — which is
exactly what happened in P358b, where the label came from a literal fallback.
Only VARYING the input and watching the output change settles it. So
`assert_drives_output` runs the consumer twice, once with the data neutralised,
and requires the marker to appear and then disappear: a falsification probe
executed inline, on every run, instead of one a human has to think to run.
"""

import pytest

from tests._guard_pins import assert_drives_output


class _Consumer:
    """A table and a consumer, so both directions can be exercised."""

    def __init__(self, reads_table=True):
        self.table = {"k": "LABEL"}
        self._reads = reads_table

    def render(self):
        if self._reads:
            return "row: " + self.table.get("k", "plain")
        return "row: LABEL"      # decoration: same output, table unread

    def disable(self):
        original = self.table
        self.table = {}
        return lambda: setattr(self, "table", original)


def test_a_real_consumer_passes():
    c = _Consumer(reads_table=True)
    assert_drives_output(c.render, "LABEL", c.disable)


def test_a_DECORATION_consumer_is_caught():
    """The case the helper exists for: the marker is in the output, the table
    is well-formed, and the consumer never reads it. Every presence-based
    test passes here; this one must not."""
    c = _Consumer(reads_table=False)
    with pytest.raises(AssertionError) as e:
        assert_drives_output(c.render, "LABEL", c.disable)
    assert "does not read it" in str(e.value)


def test_the_OTHER_vacuity_is_caught_too():
    """If the marker is absent even with the data in place, the test would
    pass for a consumer that never produced it — vacuous in the direction
    nobody checks (P174). Must fail loudly rather than silently succeed."""
    c = _Consumer(reads_table=True)
    with pytest.raises(AssertionError) as e:
        assert_drives_output(c.render, "NOT_IN_OUTPUT", c.disable)
    assert "vacuous" in str(e.value)


def test_the_data_is_restored_even_when_the_assertion_fires():
    """A guard that leaves the table neutralised poisons every later test in
    the session — the P186 shape (a test mutating shared state)."""
    c = _Consumer(reads_table=False)
    with pytest.raises(AssertionError):
        assert_drives_output(c.render, "LABEL", c.disable)
    assert c.table == {"k": "LABEL"}, "the table was left neutralised"


def test_it_restores_when_the_consumer_itself_raises():
    """Restoration must sit in a `finally`, or one exploding consumer leaves
    the neutralised table behind for everything after it."""
    c = _Consumer(reads_table=True)
    calls = {"n": 0}

    def _boom():
        calls["n"] += 1
        if calls["n"] == 1:
            return "row: LABEL"
        raise RuntimeError("consumer blew up")

    with pytest.raises(RuntimeError):
        assert_drives_output(_boom, "LABEL", c.disable)
    assert c.table == {"k": "LABEL"}, "the table was left neutralised"


def test_a_disable_that_restores_itself_is_accepted():
    """Not every caller can hand back a restorer — a monkeypatch fixture
    restores on its own. Returning None must be legal, not a crash."""
    c = _Consumer(reads_table=True)
    state = {}

    def _disable_no_restore():
        state["saved"] = c.table
        c.table = {}
        return None

    assert_drives_output(c.render, "LABEL", _disable_no_restore)
    c.table = state["saved"]


def test_the_helper_has_a_real_caller():
    """P170 applied to itself: a helper nothing calls is decoration, and
    shipping one to fix decoration would be the joke writing itself (P345)."""
    import pathlib
    import tests._guard_pins as gp

    root = pathlib.Path(gp.__file__).parent
    callers = [p.name for p in root.glob("test_*.py")
               if "assert_drives_output" in p.read_text(encoding="utf-8")
               and p.name != "test_p359_drives_output.py"]
    assert callers, (
        "assert_drives_output has no caller outside its own test — adopt it "
        "at a real site or it is exactly the decoration it detects"
    )

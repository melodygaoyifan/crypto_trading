"""
[P317] The strategy-aging tracker did not know the name of the strategy
driving the book.

Found in a routine live-log sweep: `analytics.strategy_aging | WARNING |
Unknown strategy: regimebook`, three times per tick.

`main.py:10294` writes `primary_strategy = "regimebook"` whenever the
regimebook seat is enforced (P298 armed it), and the tracker's vocabulary is
still the v3.2 twelve. So `record_signal` took its unknown-name branch and
returned "" — **the one strategy actually deciding was the one strategy this
tracker recorded nothing about**, and it said so 18 times a day.

Impact is bounded and stated rather than dramatised: the only external caller
of `get_weight_modifiers` is a WEEKLY logging block in execution_service, and
nothing multiplies a weight by the result — so no order changed. What was lost
is the weekly degraded-strategy report's ability to say anything about the
live decider, and what was gained was recurring noise.

Three separate defects, three fixes:
  1. the vocabulary did not include the live decider
  2. the docstring claimed a wiring ("Quant Agent: Applies to strategy
     weights") that does not exist anywhere in the tree — P177
  3. the unknown-name warning fired per tick, not per name — P202
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analytics.strategy_aging import (  # noqa: E402
    ALL_STRATEGIES, STRATEGY_GROUPS, StrategyAgingManager)


class TestTheLiveDeciderIsTrackable:

    def test_regimebook_is_in_the_vocabulary(self):
        """It is the name main.py:10294 emits under the enforced seat."""
        assert "regimebook" in ALL_STRATEGIES

    def test_it_is_classified_regime_agnostic(self):
        """A regime book picks its leg FROM the regime, so bucketing it as
        bull/bear/volatile would assert something false about it."""
        groups = [g for g, v in STRATEGY_GROUPS.items() if "regimebook" in v]
        assert groups == ["universal"]

    def test_recording_a_regimebook_signal_now_returns_an_id(self):
        """The behavioural half: the unknown branch returned "" and dropped
        the record entirely."""
        m = StrategyAgingManager()
        sid = m.record_signal(strategy_name="regimebook", direction=1.0,
                              confidence=0.9, regime="peace")
        assert sid != "", "a recorded signal must yield an id"

    def test_nothing_was_added_speculatively(self):
        """A vocabulary entry no producer emits is the inverse defect (P310).

        [P378] `whale` MOVED from the forbidden list to the required one, and
        the reason is that its premise changed rather than that the rule did:
        when P317 wrote this, the whale seat did not overwrite
        `primary_strategy` at all, so the entry would have been speculative.
        P376 gave it a producer (`main.py` now writes
        `primary_strategy = "whale"` at the seat), so the entry is now
        CORRECT and its absence would be the defect. The invariant this test
        exists for is unchanged in both directions — every name here must
        have a live producer, and every live producer must be named — so the
        assertion is re-pointed at the decided value rather than relaxed
        (P237/P270). `_the_producer_still_emits_the_name_this_pins` below is
        the other half.
        """
        assert "whale" in ALL_STRATEGIES, (
            "the whale seat writes primary_strategy='whale' (P376), so the "
            "aging tracker must know the name or it silently records nothing "
            "for a live decider — the exact P317 defect"
        )
        assert "trend" not in ALL_STRATEGIES, "trend arrives as trend_following"
        assert "trend_following" in ALL_STRATEGIES

    def test_the_producer_still_emits_the_name_this_pins(self):
        """The other half of P310's lesson: pin the CONSUMER against what the
        PRODUCER actually writes, or this test drifts into fiction."""
        from tests._source_scan import code_only
        src = code_only(REPO / "main.py")
        assert '"primary_strategy"] = "regimebook"' in src, (
            "main.py no longer emits this name — the vocabulary entry is now "
            "the speculative kind this file forbids")
        # [P378] The half that was missing, and its absence is why the sibling
        # test above could go stale unnoticed: `whale` was pinned ABSENT on the
        # premise that no producer wrote it, and when P376 added one, only the
        # forbidding half failed. Pinning BOTH directions means either change
        # — a name gaining a producer, or losing one — fails loudly instead of
        # letting the vocabulary and the writers drift apart (P310).
        assert '"primary_strategy"] = "whale"' in src, (
            "main.py no longer emits 'whale' — remove it from the aging "
            "vocabulary in the same change, or the entry becomes the "
            "speculative kind this file forbids")


class TestUnknownNamesAreSaidOnce:

    def test_a_repeated_unknown_name_warns_only_once(self):
        """It fired once per asset per tick — ~18 identical lines a day that
        say nothing new after the first (P202)."""
        m = StrategyAgingManager()
        with_caplog = logging.getLogger("analytics.strategy_aging")
        recs = []

        class _H(logging.Handler):
            def emit(self, r):
                recs.append(r.getMessage())

        h = _H(level=logging.WARNING)
        with_caplog.addHandler(h)
        try:
            for _ in range(10):
                m.record_signal(strategy_name="not_a_strategy", direction=1.0,
                                confidence=0.5, regime="peace")
        finally:
            with_caplog.removeHandler(h)
        hits = [r for r in recs if "not_a_strategy" in r]
        assert len(hits) == 1, f"expected one warning, got {len(hits)}"

    def test_a_second_distinct_unknown_name_still_warns(self):
        """Latching per name, not globally — otherwise the first unknown
        silences every later one, which is the P193 latch bug."""
        m = StrategyAgingManager()
        lg = logging.getLogger("analytics.strategy_aging")
        recs = []

        class _H(logging.Handler):
            def emit(self, r):
                recs.append(r.getMessage())

        h = _H(level=logging.WARNING)
        lg.addHandler(h)
        try:
            m.record_signal("unknown_one", 1.0, 0.5, "peace")
            m.record_signal("unknown_two", 1.0, 0.5, "peace")
        finally:
            lg.removeHandler(h)
        assert any("unknown_one" in r for r in recs)
        assert any("unknown_two" in r for r in recs)

    def test_the_message_states_the_consequence(self):
        """"Unknown strategy: X" tells the reader nothing about what it costs
        or how to fix it (P240)."""
        m = StrategyAgingManager()
        lg = logging.getLogger("analytics.strategy_aging")
        recs = []

        class _H(logging.Handler):
            def emit(self, r):
                recs.append(r.getMessage())

        h = _H(level=logging.WARNING)
        lg.addHandler(h)
        try:
            m.record_signal("some_new_decider", 1.0, 0.5, "peace")
        finally:
            lg.removeHandler(h)
        msg = " ".join(recs)
        # Fragments chosen to avoid the words the P307 condition-pin guard
        # scans for (==/!=/and/or/not/is). "NOTHING is being recorded" trips
        # it on the bare word "is" — a false positive, but that guard errs
        # toward catching too much, which is the right direction for it.
        assert "NOTHING" in msg and "recorded for it" in msg
        assert "ALL_STRATEGIES" in msg


class TestTheDocstringNoLongerAssertsAWiringThatDoesNotExist:

    def test_the_false_consumption_claim_is_gone(self):
        """It said "Quant Agent: Applies to strategy weights /
        effective_weight = base_regime_weight * aging_modifier". Nothing in
        the tree multiplies a weight by this (P177)."""
        import inspect
        d = inspect.getdoc(StrategyAgingManager.get_weight_modifiers) or ""
        assert "Quant Agent: Applies" not in d
        assert "execution_service" in d, "name the caller that DOES exist"

    def test_the_only_external_caller_is_still_the_weekly_logger(self):
        """If something ever does apply the modifier, this docstring becomes
        wrong in the other direction — so the claim is pinned to the code."""
        import re
        hits = []
        for sub in ("core", "agents", "defense", "risk", "signals"):
            for f in (REPO / sub).rglob("*.py"):
                if "archive" in f.parts:
                    continue
                try:
                    src = f.read_text(encoding="utf-8-sig")
                except Exception:
                    continue
                if re.search(r"get_weight_modifiers\s*\(", src):
                    hits.append(f.name)
        assert hits == ["execution_service.py"], (
            f"the docstring names execution_service as the only consumer; "
            f"found {hits}")

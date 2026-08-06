"""[P168] The rebuild cooldown exempted direction flips — the churn it cost the
most to allow, on the path it fired most often.

`execute_intent_v2` set an 8h cooldown after every close (full or partial) and
then blocked new entries during it — except that an entry opposite to the
closed position was waved through, on this reasoning:

    # Cooldown is designed to prevent same-direction re-entry churn.
    # Opposite-direction entry is a signal-aligned reversal, allow it.

Two things are wrong with that.

**It has the cost backwards.** A reversal pays a full round trip to close and
commits another to open (P167). It is the *most* expensive thing that can
happen inside a cooldown window, not an exception to be carved out.

**It swallows its own dominant case.** A close is usually *caused* by the
signal turning, so the next entry the system proposes is opposite by
construction — that is what "reversal" means. The exemption therefore fired on
the common path and left the cooldown binding only when the signal reversed and
then reversed back inside 8h.

Measured over the 52 timestamped closed trades in `data/trade_attribution.jsonl`
(assets BTC/ETH/SOL, total net -$540.07):

    re-entry after a close        inside 8h window   outside
    ------------------------------------------------------------
    FLIP  (was exempt)                  10              17
    SAME-DIR (was blocked)               5              17

    -> 10/15 = 67% of in-window re-entries took the exemption
    -> those 10 flips: 8 losers, net -$94.45, mean -$9.45
    -> 6 of the 10 opened at 0.0h — exit and reversal in the same 4H bar

Splitting by whether the *closed* trade won does not rescue a narrower carve-out
(after-loser: 7 trades, 7 losers, -$67.75; after-winner: 3 trades, -$26.70), so
the exemption is off entirely rather than conditioned. One ETH sequence ran
+$8.46 -> flip +$2.18 -> flip -$31.54, three round trips inside one window.

The decision was extracted to `rebuild_cooldown_decision` to be testable at all:
it lived inside a ~2000-line async function needing a full runner, positions,
market data and an event loop, which is why this branch went unexercised for its
entire life.
"""

import datetime as dt

import pytest

from core.execution_service import (
    REBUILD_COOLDOWN_EXEMPT_ADDON,
    REBUILD_COOLDOWN_EXEMPT_FLIP,
    rebuild_cooldown_decision,
)

NOW = dt.datetime(2026, 8, 5, 12, 0, tzinfo=dt.timezone.utc)
LONG, SHORT = 1, -1


def _entry(hours_left=6.0, closed_dir=LONG, reason="full_close_net=$-8.48"):
    """A `ctx.rebuild_cooldown[asset]` tuple as the close paths write it."""
    return (NOW + dt.timedelta(hours=hours_left), reason, closed_dir)


def _decide(entry, direction, is_addon=False, now=NOW, **flags):
    return rebuild_cooldown_decision(entry, direction, is_addon, now, **flags)


# --------------------------------------------------------------------------

class TestDefaults:
    def test_flip_exemption_is_off(self):
        assert REBUILD_COOLDOWN_EXEMPT_FLIP is False

    def test_addon_exemption_is_untouched(self):
        """P168 changes the flip carve-out only. The pyramid carve-out has a
        real justification — it adds to a position that is already winning, so
        it is not a re-entry at all."""
        assert REBUILD_COOLDOWN_EXEMPT_ADDON is True


class TestFlipIsNoLongerExempt:
    @pytest.mark.parametrize("closed_dir,new_dir", [(LONG, SHORT), (SHORT, LONG)])
    def test_flip_inside_window_is_blocked(self, closed_dir, new_dir):
        action, kind, remaining = _decide(_entry(closed_dir=closed_dir), new_dir)
        assert action == "BLOCK"
        assert kind == "FLIP"
        assert remaining == pytest.approx(6.0)

    def test_same_tick_flip_is_blocked(self):
        """6 of the 10 measured flips opened 0.0h after the close — the exit
        and its reversal landed in the same 4H bar."""
        action, kind, _ = _decide(_entry(hours_left=8.0), SHORT, now=NOW)
        assert (action, kind) == ("BLOCK", "FLIP")

    def test_flip_is_allowed_once_the_window_expires(self):
        """The cooldown delays a reversal; it does not forbid one. 17 of the 27
        observed flips were already outside the window and are unaffected."""
        action, kind, remaining = _decide(_entry(hours_left=-0.1), SHORT)
        assert action == "EXPIRED"
        assert kind == "FLIP"
        assert remaining == 0.0

    def test_same_direction_still_blocked(self):
        """The case that always worked keeps working."""
        action, kind, _ = _decide(_entry(closed_dir=LONG), LONG)
        assert (action, kind) == ("BLOCK", "SAME-DIR")


class TestUnknownDirectionBlocks:
    """Missing must not read as exempt — the P2/P15/P138/P152 failure mode."""

    def test_legacy_two_tuple_has_no_direction_and_blocks(self):
        """Cooldown entries written before the closed-direction field existed
        are 2-tuples. Absent direction is 0, which cannot be a flip."""
        action, kind, _ = _decide((NOW + dt.timedelta(hours=6), "full_close"), SHORT)
        assert (action, kind) == ("BLOCK", "SAME-DIR")

    def test_zero_closed_direction_blocks(self):
        action, _, _ = _decide(_entry(closed_dir=0), SHORT)
        assert action == "BLOCK"

    def test_zero_intent_direction_blocks(self):
        """A flat intent is not a reversal of anything."""
        action, kind, _ = _decide(_entry(closed_dir=LONG), 0)
        assert (action, kind) == ("BLOCK", "SAME-DIR")

    def test_unknown_direction_does_not_become_exempt_when_the_flag_is_on(self):
        """Even restoring the legacy flag must not let an unknown direction
        through: the exemption requires a *known* opposite direction."""
        action, _, _ = _decide((NOW + dt.timedelta(hours=6), "x"), SHORT,
                               exempt_flip=True)
        assert action == "BLOCK"


class TestAddonPathUnaffected:
    def test_addon_is_still_exempt(self):
        action, _, _ = _decide(_entry(), LONG, is_addon=True)
        assert action == "EXEMPT_ADDON"

    def test_addon_exemption_wins_over_a_flip(self):
        """Ordering is load-bearing: an add-on that happens to look like a flip
        must take the add-on branch, not fall into the blocked chain. An earlier
        draft of this fix broke the if/elif chain and would have started
        blocking pyramids."""
        action, kind, _ = _decide(_entry(closed_dir=LONG), SHORT, is_addon=True)
        assert action == "EXEMPT_ADDON"
        assert kind == "FLIP"

    def test_addon_still_blocked_when_its_own_flag_is_off(self):
        action, _, _ = _decide(_entry(), LONG, is_addon=True, exempt_addon=False)
        assert action == "BLOCK"


class TestLegacyFlagRestoresOldBehaviour:
    def test_flag_on_reproduces_the_exemption(self):
        action, kind, _ = _decide(_entry(closed_dir=LONG), SHORT, exempt_flip=True)
        assert action == "EXEMPT_FLIP"
        assert kind == "FLIP"

    def test_flag_on_does_not_exempt_same_direction(self):
        action, _, _ = _decide(_entry(closed_dir=LONG), LONG, exempt_flip=True)
        assert action == "BLOCK"


class TestOnlyTightens:
    """No entry the old gate blocked may be allowed by the new one.

    The change removes one exemption and adds none, so the new gate's allowed
    set is a subset of the old one's. Checked across the whole input grid rather
    than asserted in prose.
    """

    @pytest.mark.parametrize("closed_dir", [-1, 0, 1])
    @pytest.mark.parametrize("new_dir", [-1, 0, 1])
    @pytest.mark.parametrize("is_addon", [False, True])
    @pytest.mark.parametrize("hours_left", [-1.0, 0.0, 0.5, 8.0])
    def test_new_never_allows_what_old_blocked(self, closed_dir, new_dir,
                                               is_addon, hours_left):
        entry = _entry(hours_left=hours_left, closed_dir=closed_dir)
        old, _, _ = _decide(entry, new_dir, is_addon, exempt_flip=True)
        new, _, _ = _decide(entry, new_dir, is_addon, exempt_flip=False)
        allowed = lambda a: a != "BLOCK"
        if allowed(new):
            assert allowed(old), (old, new)

    @pytest.mark.parametrize("hours_left", [0.5, 8.0])
    def test_the_difference_is_exactly_the_flip_case(self, hours_left):
        """Every input where old and new disagree is an in-window flip."""
        disagreements = []
        for closed_dir in (-1, 0, 1):
            for new_dir in (-1, 0, 1):
                entry = _entry(hours_left=hours_left, closed_dir=closed_dir)
                old, kind, _ = _decide(entry, new_dir, False, exempt_flip=True)
                new, _, _ = _decide(entry, new_dir, False, exempt_flip=False)
                if old != new:
                    disagreements.append((closed_dir, new_dir, kind, old, new))
        assert disagreements, "the flag must actually change something"
        for closed_dir, new_dir, kind, old, new in disagreements:
            assert kind == "FLIP"
            assert closed_dir * new_dir < 0
            assert (old, new) == ("EXEMPT_FLIP", "BLOCK")


class TestRemainingHours:
    def test_remaining_is_reported_for_the_block_message(self):
        _, _, remaining = _decide(_entry(hours_left=3.25), SHORT)
        assert remaining == pytest.approx(3.25)

    def test_remaining_never_goes_negative(self):
        """The value is formatted into an operator-facing reason string; a
        negative 'hours remaining' would read as a bug in the cooldown."""
        _, _, remaining = _decide(_entry(hours_left=-5.0), SHORT)
        assert remaining == 0.0

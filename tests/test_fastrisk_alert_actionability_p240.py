"""[P240] 74 CRITICALs nobody could act on, and a percentage that could not be diagnosed.

LIVE EVIDENCE, 2026-08-08: `[FastRiskTick][LIVE] SOL: REDUCE_50 - depth_drop=66-69%(3x)`
fired **74 times** between 11:10 and 12:01 UTC. SOL was **flat** the whole time,
**zero** reduces executed, and every one of those lines forwarded to Discord.

Two separate defects, one symptom.

1. UNACTIONABLE SEVERITY. A REDUCE_50 on a flat asset is a no-op by
   construction — P227's sleeve handler already returns `FLAT / "no sleeve
   position"` and does nothing. Escalating that to CRITICAL is the P202 pattern:
   an alert whose only possible resolutions are theatre or ignoring it, and a
   standing CRITICAL teaches everyone to ignore the channel (the same mechanism
   that let P192's broken image build hide for weeks).

2. UNDIAGNOSABLE CONTENT. "depth_drop=69%" cannot distinguish a genuine
   liquidity collapse from a degraded feed reading. That is precisely the
   question the burst left open and which nobody could settle from the log
   afterwards — so the raw depth and baseline now travel with it.

THE INVARIANT THAT MATTERS MOST HERE. This is a SAFETY control, so the change is
strictly to alert severity: the returned action is byte-identical in every case,
and only an EXPLICIT `has_position=False` downgrades. `None` — the default, and
what any caller that forgets to pass it gets — keeps full CRITICAL severity. A
downgrade must never be the default, or "quieter" silently becomes "blind".
"""

import logging

import pytest

from execution.fast_risk_tick import FastRiskAction, FastRiskTick

_DEPTH_BASE = 10_000_000.0
_DEPTH_NOW = 3_000_000.0        # 70% drop, past DEPTH_DROP_THRESHOLD
_MD = {"current_price": 100.0, "volatility_30m": 0.01,
       "orderbook_depth_1pct_usd": _DEPTH_NOW}


def _armed():
    f = FastRiskTick(shadow_mode=False)
    f.set_4h_anchor("SOL", 100.0, volatility=0.01, depth=_DEPTH_BASE)
    return f


def _fire(f, **kw):
    """Three evaluations — DEPTH_DROP_CONFIRM_STREAK is 3."""
    for _ in range(FastRiskTick.DEPTH_DROP_CONFIRM_STREAK):
        r = f.evaluate("SOL", _MD, **kw)
    return r


class TestTheActionIsNeverChanged:
    """The whole change must be invisible to the risk decision."""

    @pytest.mark.parametrize("kw", [{}, {"has_position": None},
                                    {"has_position": False},
                                    {"has_position": True}])
    def test_reduce_is_returned_regardless_of_position_hint(self, kw):
        assert _fire(_armed(), **kw).action is FastRiskAction.REDUCE_50

    def test_the_reason_is_the_same_decision_either_way(self):
        a = _fire(_armed(), has_position=False).reason
        b = _fire(_armed(), has_position=True).reason
        assert a == b, "the hint changed the decision, not just the log"

    def test_a_healthy_book_still_holds(self):
        f = _armed()
        r = f.evaluate("SOL", dict(_MD, orderbook_depth_1pct_usd=_DEPTH_BASE),
                       has_position=True)
        assert r.action is FastRiskAction.HOLD


class TestSeverityIsGatedOnActionability:

    def test_flat_downgrades_to_info(self, caplog):
        with caplog.at_level(logging.INFO):
            _fire(_armed(), has_position=False)
        crit = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        info = [r for r in caplog.records if "FastRiskTick][LIVE" in r.message]
        assert not crit, "still CRITICAL on a flat asset"
        assert info and "FLAT" in info[-1].message

    def test_a_held_position_still_shouts(self, caplog):
        """The control must not have been quietened generally."""
        with caplog.at_level(logging.INFO):
            _fire(_armed(), has_position=True)
        assert [r for r in caplog.records if r.levelno >= logging.CRITICAL], (
            "a real emergency on a held position no longer alerts"
        )

    def test_UNKNOWN_position_still_shouts(self, caplog):
        """The load-bearing one. `None` is the default, so a caller that simply
        forgets to pass the hint must NOT get a silenced emergency."""
        with caplog.at_level(logging.INFO):
            _fire(_armed(), has_position=None)
        assert [r for r in caplog.records if r.levelno >= logging.CRITICAL]

    def test_the_default_is_unknown_not_false(self, caplog):
        import inspect
        sig = inspect.signature(FastRiskTick.evaluate)
        assert sig.parameters["has_position"].default is None, (
            "defaulting to False would make every un-updated caller silent"
        )
        with caplog.at_level(logging.INFO):
            _fire(_armed())
        assert [r for r in caplog.records if r.levelno >= logging.CRITICAL]

    def test_shadow_mode_is_untouched(self, caplog):
        f = FastRiskTick(shadow_mode=True)
        f.set_4h_anchor("SOL", 100.0, volatility=0.01, depth=_DEPTH_BASE)
        with caplog.at_level(logging.INFO):
            for _ in range(FastRiskTick.DEPTH_DROP_CONFIRM_STREAK):
                f.evaluate("SOL", _MD, has_position=False)
        assert any("SHADOW" in r.message for r in caplog.records)


class TestRawDepthTravelsWithThePercentage:

    def test_reason_carries_current_and_baseline(self):
        r = _fire(_armed(), has_position=True)
        assert "depth=$3,000,000" in r.reason, r.reason
        assert "baseline=$10,000,000" in r.reason, r.reason

    def test_the_percentage_is_still_there(self):
        """Don't break the existing greppable shape."""
        assert "depth_drop=70%" in _fire(_armed(), has_position=True).reason

    def test_a_degraded_feed_is_now_distinguishable(self):
        """The question the 2026-08-08 burst left unanswered: a plausible-looking
        69% is very different at $3M vs at $300. Both now appear in the line."""
        f = FastRiskTick(shadow_mode=False)
        f.set_4h_anchor("SOL", 100.0, volatility=0.01, depth=1_000_000.0)
        r = None
        for _ in range(FastRiskTick.DEPTH_DROP_CONFIRM_STREAK):
            r = f.evaluate("SOL", dict(_MD, orderbook_depth_1pct_usd=300_000.0),
                           has_position=True)
        assert "depth=$300,000" in r.reason and "baseline=$1,000,000" in r.reason


class TestWiring:

    def test_both_call_sites_pass_the_hint(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "main.py").read_text(
            encoding="utf-8-sig", errors="replace")
        assert src.count("has_position=_frt_has_pos") == 2, (
            "a call site still omits the hint — it will keep alerting at "
            "CRITICAL on flat assets"
        )

    def test_the_caller_defaults_to_unknown_on_any_doubt(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "main.py").read_text(
            encoding="utf-8-sig", errors="replace")
        i = src.index("_frt_has_pos = None")
        w = src[i:i + 900]
        assert '_reconcile_ok' in w, (
            "must not infer 'flat' from a stale snapshot — that would silence a "
            "real emergency on a position we simply could not see"
        )
        assert "except Exception:" in w and "_frt_has_pos = None" in w

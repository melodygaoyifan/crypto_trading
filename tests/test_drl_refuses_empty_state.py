"""[P178] The DRL scored a zeroed state vector and returned it as a signal.

`DRLAgent.generate_signal` takes eight arguments; the three that carry state
(position_state, market_data, agent_signals) all default to None. A caller
passing only asset/price/regime reached build_state() with three empty dicts.
build_state() does not object — it fills the vector with zeros and defaults,
get_action() scores it, and the result came back with is_valid=True and a real
direction and confidence. The only trace was data_quality=0.20 on a field with
no consumer outside to_dict().

Measured before the fix, with mode=SHADOW and no state arguments:

    is_valid=True  data_quality=0.20
    issues=['position_state_empty', 'regime_result_none',
            'market_data_empty', 'agent_signals_empty']

core/runtime_spine.py:878 is that call, verbatim: three of eight arguments.
It is not live — RuntimeSpine has no production constructor (only its own
factory and tests/test_runtime_singleton_refresh_advanced.py), main.py owns
the tick, and production DRL is the TQC ensemble in main.py `_drl_ensembles`,
not this legacy wrapper. So this closes a latent path, not a bleeding one.

It is still the P170/P174 shape — a zeroed measurement indistinguishable from
a real one downstream — and the cost of leaving it is that reviving the spine
looks safe. These tests pin the refusal, and pin that it refuses ONLY the
all-absent case, so a degraded-but-real tick still gets a signal.
"""

import re
from pathlib import Path

import pytest

from agents.drl_agent import DRLAgent

REPO_ROOT = Path(__file__).resolve().parents[1]
SPINE = REPO_ROOT / "core" / "runtime_spine.py"
MAIN = REPO_ROOT / "main.py"


@pytest.fixture
def agent():
    # DISABLED short-circuits before the guard, so it would pass vacuously.
    return DRLAgent(mode="SHADOW")


class TestNoStateMeansNoSignal:
    def test_the_runtime_spine_call_shape_is_refused(self, agent):
        p = agent.generate_signal(asset="BTC", price=50000.0, regime="BULL")
        assert p.is_valid is False, (
            "generate_signal with no state inputs still reports a valid "
            "signal. The state vector it scored was all zeros."
        )
        assert p.direction == 0.0, (
            "is_valid=False is not enough — nothing downstream reads it. The "
            "direction must be zero or fusion still receives the fabrication."
        )
        assert p.confidence == 0.0
        assert p.meta.get("reason") == "no_state_inputs"

    def test_the_issues_survive_for_the_operator(self, agent):
        p = agent.generate_signal(asset="BTC", price=50000.0, regime="BULL")
        assert set(p.meta.get("issues", [])) >= {
            "position_state_empty", "market_data_empty", "agent_signals_empty",
        }, "the refusal must say which inputs were missing, not just refuse"
        assert p.data_quality == pytest.approx(0.20, abs=0.01), (
            "data_quality drifted from the 0.20 measured for this exact "
            "shape; the scoring changed and the docstring evidence is stale"
        )


class TestPartialInputIsStillAllowed:
    """The refusal must not become a general 'degraded means silent'."""

    @pytest.mark.parametrize("kwargs", [
        {"market_data": {"close": 50000.0, "atr_pct": 1.2}},
        {"position_state": {"has_position": True, "tranche": 1}},
        {"agent_signals": {"quant_direction": 0.4}},
    ])
    def test_any_one_state_dict_is_enough_to_proceed(self, agent, kwargs):
        p = agent.generate_signal(
            asset="BTC", price=50000.0, regime="BULL", **kwargs)
        assert p.is_valid is True, (
            f"a real tick carrying {list(kwargs)} was refused. The guard is "
            f"meant to catch a caller that forgot the arguments, not to "
            f"silence the DRL whenever data is imperfect."
        )
        assert p.meta.get("reason") != "no_state_inputs"

    def test_degraded_input_lowers_quality_rather_than_refusing(self, agent):
        p = agent.generate_signal(
            asset="BTC", price=50000.0, regime="BULL",
            market_data={"close": 50000.0})
        assert p.is_valid is True
        assert 0.0 < p.data_quality < 1.0, (
            "partial input should still be scored down; if it reports 1.0 the "
            "quality signal has stopped measuring anything"
        )


class TestTheGuardIsReachableNotDecorative:
    def test_disabled_mode_does_not_hide_the_guard_from_the_suite(self):
        # DRLAgent() defaults to DISABLED, which returns before the guard.
        # If a future default flips these tests to vacuous, catch it here.
        p = DRLAgent().generate_signal(asset="BTC", price=1.0, regime="BULL")
        assert p.meta.get("reason") == "neutral_default", (
            "the DISABLED short-circuit changed; re-check that the SHADOW "
            "fixture above is still exercising the P178 branch at all"
        )

    def test_neutral_payload_can_express_invalid(self):
        from agents.drl_agent import _neutral_payload
        assert _neutral_payload("BTC", is_valid=False).is_valid is False, (
            "is_valid is hardcoded True again, so the neutral payload claims "
            "'I evaluated and have no view' when it means 'I got nothing'"
        )


class TestTheDeprecationContradictionIsResolved:
    """main.py named the deprecated spine as CANONICAL_MAIN_LOOP."""

    def test_main_no_longer_names_runtime_spine_canonical(self):
        src = MAIN.read_text(encoding="utf-8-sig")
        header = src[:src.find("import ")] if "import " in src else src[:4000]
        assert not re.search(
            r"CANONICAL_(MAIN_LOOP|SPINE)\s*[:=]\s*core/runtime_spine\.py",
            header + src[-40000:]), (
            "main.py still advertises core/runtime_spine.py as the canonical "
            "main loop while that file's own docstring says it is DEPRECATED "
            "and not used. One of the two is wrong and a reader cannot tell "
            "which."
        )

    def test_the_spine_still_declares_itself_deprecated(self):
        head = SPINE.read_text(encoding="utf-8")[:600]
        assert "_DEPRECATED" in head and "NOT used" in head, (
            "runtime_spine dropped its deprecation banner. If it was revived "
            "deliberately, the generate_signal call at ~:878 passes 3 of 8 "
            "arguments and must be fixed before this file processes a tick."
        )

    def test_the_spine_has_no_production_constructor(self):
        """The reason P178 is latent rather than live. Pin it."""
        hits = []
        for p in REPO_ROOT.rglob("*.py"):
            s = "/" + str(p.relative_to(REPO_ROOT)).replace("\\", "/")
            if any(x in s for x in ("/.git/", "/archive/", "/legacy/",
                                    "/tests/", "/docs/")):
                continue
            if p == SPINE or p.name in ("__init__.py", "canonical_imports.py"):
                continue
            if re.search(r"\bget_runtime_spine\s*\(|\bRuntimeSpine\s*\(",
                         p.read_text(encoding="utf-8-sig", errors="replace")):
                hits.append(str(p.relative_to(REPO_ROOT)))
        assert not hits, (
            f"RuntimeSpine is now constructed in production ({hits}). The DRL "
            f"call at runtime_spine.py:~878 passes 3 of 8 arguments and is no "
            f"longer latent — it now runs. Fix the call site."
        )

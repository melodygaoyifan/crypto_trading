"""[P338] Entry-quality vetoes were liquidating held sleeve positions, the
execution layer's whole rejection surface defaulted to LIQUIDATE, and three
switches controlled nothing.

The through-line of findings #1 and #2 is one fact about the post-cutover
system: `position_state["current_exposure"]` is KRAKEN-SHAPED. In LIVE,
`_get_effective_position_state` reads it out of `market_data`, and
`data_mgmt/market_data_pipeline.py` fills it from `_paper_positions` -- the
Kraken book, {} since the 2026-06-13 flatten. So for a Coinbase-routed asset
it is structurally 0.0 whatever the sleeve holds.

P275 traced that to the EXIT stack and correctly called the result dormant.
Nobody traced it to the ENTRY gates, where it is not dormant but wrong in
both directions:

  * BitBeast / OP_BUDGET / GAMBLER_GATE read "no position" and treat every
    actionable tick as a NEW ENTRY even while the sleeve is long, and their
    rejections were classified veto_flat -- so an SNR/volume/structure ENTRY
    filter market-flattened a held perp. The P287 category error, three more
    times.
  * The STRUCTURE fractal-break veto reads "no position" and always takes the
    T1-bypass, so it can never fire and a permanent x0.75 haircut stands in
    its place.

Only the CLASSIFICATION is changed here. Nothing feeds sleeve state into
position_state -- that would arm the whole dormant exit stack on a live
account, which is a P141 activation and not a bug fix.

[P341] And the classification is THREE-valued, not two. A binary
HOLD/FLATTEN is right on a flat book and on a maintain tick, and WRONG on a
flip: plain HOLD would keep a long while the book says short for as long as
an entry filter keeps rejecting the short. The repo already had the right
truth table one layer down (`sleeve_agent_filter_decision`, P236/P293d:
block from flat, demote a flip to a flatten, never touch exits), so these
vetoes now return a SLEEVE_ENTRY_BLOCKED sentinel that the driver resolves
against the venue-reconciled book.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import main as m  # noqa: E402


class _Intent:
    def __init__(self, reason, direction=0.9, target_exposure=0.2):
        self.direction = direction
        self.target_exposure = target_exposure
        self.veto_active = True
        self.veto_reason = reason


def _verdict(reason):
    tgt, why = m.sleeve_direction_from_intent(_Intent(reason), 0.9)
    return ("HOLD" if tgt is m.SLEEVE_HOLD else "FLATTEN"), why


# ---------------------------------------------------------------------------
# Finding #1 -- entry-quality gates must not liquidate a held position
# ---------------------------------------------------------------------------

class TestEntryQualityVetoesHold:

    ENTRY_QUALITY = [
        "[VETO] SNR too low: 0.81 < 1.20; Volume 0.6x < 1.0x",
        "[VETO] Structure: no breakout confirmation",
        "[OP_BUDGET] global budget exhausted (3/3 active)",
        "[GAMBLER_GATE] cooldown: 2 bars remaining",
        "[FLIP_GATE] alpha=12.0 < 30.0",
        "[REBUILD_COOLDOWN] FLIP cooldown, 3.2h remaining",
    ]

    @pytest.mark.parametrize("reason", ENTRY_QUALITY)
    def test_entry_filters_are_resolved_against_the_book(self, reason):
        """[P341] NOT a plain HOLD. The answer depends on what is held and
        which way the signal points, so the translator defers."""
        tgt, why = m.sleeve_direction_from_intent(_Intent(reason), 0.9)
        assert tgt is m.SLEEVE_ENTRY_BLOCKED, (
            f"{reason!r} did not defer to the book. Plain FLATTEN makes an "
            f"entry filter into an exit engine (the P287 category error); "
            f"plain HOLD keeps a position the signal has abandoned on a FLIP")
        assert why.startswith("entry_quality_veto"), why

    @pytest.mark.parametrize("reason", ENTRY_QUALITY)
    @pytest.mark.parametrize("pos,direction,expect", [
        (0, 1.0, "HOLD"),      # flat: the refusal IS the whole effect
        (0, -1.0, "HOLD"),
        (1, 1.0, "HOLD"),      # maintain: the P338 bug -- must not liquidate
        (-1, -1.0, "HOLD"),
        (1, -1.0, "FLATTEN"),  # FLIP: close what the signal abandoned
        (-1, 1.0, "FLATTEN"),
        (1, 0.0, "HOLD"),      # no direction: absence is not an instruction
    ])
    def test_the_full_truth_table(self, reason, pos, direction, expect):
        tgt, why = m.sleeve_direction_from_intent(
            _Intent(reason, direction=direction), 0.9)
        tgt, why = m.sleeve_entry_blocked_resolve(pos, direction, why)
        got = "HOLD" if tgt is m.SLEEVE_HOLD else "FLATTEN"
        assert got == expect, f"pos={pos} dir={direction} {reason!r} -> {why}"

    def test_it_can_never_open_a_position(self):
        """The load-bearing safety property: an ENTRY-quality refusal may
        decline to open, or close what was abandoned -- never open anything.
        A directional return here would turn a veto into an entry."""
        for pos in (-2, -1, 0, 1, 2):
            for d in (-1.0, -0.2, 0.0, 0.2, 1.0):
                tgt, _ = m.sleeve_entry_blocked_resolve(pos, d, "x")
                assert tgt is m.SLEEVE_HOLD or tgt == 0.0, (pos, d, tgt)

    def test_the_guard_still_blocks_the_entry_it_exists_to_block(self):
        """Not a loosening: from FLAT the book stays flat, so the rejected
        entry is still rejected. Without this the fix would read as 'we
        disabled BitBeast'."""
        tgt, why = m.sleeve_direction_from_intent(
            _Intent("[VETO] SNR too low", direction=1.0), fallback_dir=1.0)
        tgt, why = m.sleeve_entry_blocked_resolve(0, 1.0, why)
        assert tgt is m.SLEEVE_HOLD and "flat" in why

    def test_the_driver_resolves_the_sentinel(self):
        """A sentinel nothing resolves would leave `float(SLEEVE_ENTRY_
        BLOCKED)` to raise into the order path's handler -- a seam nothing
        calls is decoration (P170), and here it would be worse than that."""
        from tests._source_scan import code_only
        src = code_only(REPO / "main.py", strip_docstrings=True)
        assert "if _m_tgt is SLEEVE_ENTRY_BLOCKED:" in src
        assert "sleeve_entry_blocked_resolve(" in src
        i_resolve = src.index("_m_tgt is SLEEVE_ENTRY_BLOCKED")
        i_hold = src.index("if _m_tgt is SLEEVE_HOLD:", i_resolve - 2000)
        assert i_resolve < i_hold, (
            "the sentinel must be resolved BEFORE the HOLD branch, or it "
            "falls through to float(_m_tgt) and raises")

    @pytest.mark.parametrize("reason", [
        "[P0_SAFETY] force flat",
        "[BLACK_SWAN_SENTINEL] halt",
        "[v3.6.1] Alpha gate: 10bps < 30bps",
        "[v3.6.1] NO_TRADE: FLASH_CRASH",
        "[STRATEGY_SUSPENDED] existence fuse",
        "[BEST_OF_N_HOLD]",
        "[PROD] HARD VETO: correlation",
    ])
    def test_real_risk_and_signal_vetoes_still_flatten(self, reason):
        """The reclassification must be surgical -- if it moved a market-risk
        or signal-says-flat veto, the sleeve would stop exiting."""
        got, _ = _verdict(reason)
        assert got == "FLATTEN", f"{reason!r} became {got}"

    def test_the_premise_current_exposure_is_kraken_shaped(self):
        """The whole finding rests on this. If a producer ever feeds sleeve
        state into current_exposure, the entry gates become position-aware
        and this classification should be REVISITED rather than inherited."""
        from tests._source_scan import code_only
        src = code_only(REPO / "data_mgmt" / "market_data_pipeline.py",
                        strip_docstrings=True)
        assert '"current_exposure": self._paper_positions.get(' in src, (
            "market_data['current_exposure'] no longer comes straight from "
            "_paper_positions -- re-derive whether the entry gates can now "
            "see the sleeve, and re-open the P338 classification")

    def test_structure_veto_is_recorded_as_unreachable_not_deleted(self):
        """Its roster entries stay so the classification is auditable, and
        the reason it cannot fire is stated at the roster."""
        assert "[STRUCTURE] LONG blocked" in m._SLEEVE_FLATTEN_INTENDED_VETOES
        assert "[STRUCTURE] SHORT blocked" in m._SLEEVE_FLATTEN_INTENDED_VETOES


# ---------------------------------------------------------------------------
# Finding #2 -- execute_intent_v2's rejection surface
# ---------------------------------------------------------------------------

class TestExecutionLayerReasonsAreClassified:

    HOLD = [
        "[EXECUTION] Verify mode",
        "[EXECUTION] Invalid price",
        "[EXECUTION] EXPOSURE_BELOW_MINIMUM_VIABLE",
        "[EXECUTION] No active position quantity available",
        "[BUGFIX H3] Invalid quantity: 0.0",
        "[EXECUTION] min hold: 1/2 ticks",
        "[EXECUTION] per-asset rate limit: 3/3",
        "[EXECUTION] global rate limit: 9/9",
        "[EXECUTION] ADDON_BLOCKED",
        # [P341] [REBUILD_COOLDOWN] and [FLIP_GATE] moved to the
        # entry-quality class above: both can arrive on a FLIP, where
        # the abandoned leg must still close.
        "[AUDIT M3] Below Kraken min: 0.4",
        "[V10S] DynamicSlicer: ATR=1.2",
        "[PA_ABORT] Insufficient edge",
    ]
    FLATTEN = [
        "[P0_FAIL_CLOSED] Equity unavailable: timeout",
        "[DYN_GROSS_CAP] gross limit 0.5",
        "[V10] GlobalExposureCap: net 0.6",
        "[V10A] CascadeExhaustion: phase=2",
        "[DD_GRADIENT] drawdown 0.2",
        "[LEVERAGE_GUARD] 3.0x > 2.0x",
        "[P0_UNIT_SYSTEM] bad units",
        "[EXECUTION] ExecutionGuard: stale data",
        "[AUTHORITY_CHAIN] RISK_MANAGER: not approved",
    ]

    @pytest.mark.parametrize("reason", HOLD)
    def test_wait_smaller_or_not_this_order_holds(self, reason):
        got, _ = _verdict(reason)
        assert got == "HOLD", (
            f"{reason!r} classified {got}. These mean 'wait', 'smaller' or "
            f"'not THIS order' on the KRAKEN path -- none of them is an "
            f"instruction to liquidate the Coinbase book")

    @pytest.mark.parametrize("reason", FLATTEN)
    def test_risk_postures_still_flatten(self, reason):
        got, _ = _verdict(reason)
        assert got == "FLATTEN", f"{reason!r} became {got}"

    def test_invalid_price_is_reachable_before_the_p152_skip(self):
        """The one reason of the 23 that is NOT gated behind the empty Kraken
        book: the price check runs BEFORE the routed-asset skip, so an
        unreadable current_price on an actionable tick reached the stamping
        site and liquidated the perp book (the P265b data class)."""
        from tests._source_scan import code_only
        src = code_only(REPO / "core" / "execution_service.py",
                        strip_docstrings=True)
        i_price = src.index('"reason": "Invalid price"')
        i_p152 = src.index('"reason": "coinbase_routed_no_kraken_entry"')
        assert i_price < i_p152, (
            "the Invalid-price rejection no longer precedes the P152 routed "
            "skip -- re-derive its reachability before trusting this "
            "classification either way")
        assert _verdict("[EXECUTION] Invalid price")[0] == "HOLD"

    def test_the_guard_now_reads_the_execution_layer(self):
        """Anti-vacuity for the widened P276 corpus: the blind spot was that
        execute_intent_v2's reasons were invisible to it, so they defaulted
        to liquidate with nothing going red."""
        from tests.test_sleeve_gated_intent_p206 import (
            _execution_service_stamped_reasons)
        got = _execution_service_stamped_reasons()
        assert len(got) >= 20, len(got)
        assert "[EXECUTION] min hold:" in got
        assert "[EXECUTION] Invalid price" in got


# ---------------------------------------------------------------------------
# Finding #3 -- the EXHAUSTION HTF branch could never take its other side
# ---------------------------------------------------------------------------

def _adapter():
    from signals.profit_max_adapter import ProfitMaxAdapter, ProfitMaxConfig
    c = ProfitMaxConfig()
    # isolate the phase filter: everything else off, so the assertions are
    # about THIS branch and not about a product of six multipliers
    c.regime_quality_enabled = False
    c.loss_streak_enabled = False
    c.funding_bias_enabled = False
    c.false_breakout_enabled = False
    c.signal_quality_enabled = False
    c.transition_risk_enabled = False
    c.phase_filter_enabled = True
    return ProfitMaxAdapter(config=c)


class TestExhaustionHtfAlignment:

    def test_no_htf_reading_is_recorded_as_unknown_not_as_misaligned(self):
        r = _adapter().evaluate(phase="EXHAUSTION", direction=1,
                                htf_trend_direction=0.0)
        pf = r.breakdown["phase_filter"]
        assert pf["htf_aligned"] is None, (
            "an absent htf_trend_direction is still being reported as "
            "misaligned -- that key has NO producer in the tree (P174), so "
            "the old two-branch form asserted a counter-trend position on a "
            "feature that does not exist (P2/P223)")
        assert pf["htf_unknown"] is True
        assert r.conviction_multiplier == pytest.approx(0.9)

    def test_a_real_counter_reading_still_takes_the_heavy_penalty(self):
        """The branch is preserved, not removed -- it becomes reachable the
        moment a real HTF producer exists."""
        r = _adapter().evaluate(phase="EXHAUSTION", direction=1,
                                htf_trend_direction=-1.0)
        pf = r.breakdown["phase_filter"]
        assert pf["htf_aligned"] is False
        assert r.conviction_multiplier == pytest.approx(0.7)
        assert r.execution_urgency < 1.0

    def test_a_real_aligned_reading_takes_the_small_penalty(self):
        r = _adapter().evaluate(phase="EXHAUSTION", direction=1,
                                htf_trend_direction=1.0)
        assert r.breakdown["phase_filter"]["htf_aligned"] is True
        assert r.conviction_multiplier == pytest.approx(0.9)

    def test_the_switch_disarms_the_heavy_penalty(self):
        """Finding #4's point: this switch was unreachable from config, so
        the heavy penalty could not be turned off by any profile."""
        a = _adapter()
        a.config.exhaustion_requires_htf_alignment = False
        r = a.evaluate(phase="EXHAUSTION", direction=1,
                       htf_trend_direction=-1.0)
        assert r.conviction_multiplier == pytest.approx(0.9)

    def test_htf_trend_direction_still_has_no_producer(self):
        """Pins the premise. If someone wires a producer, the unknown branch
        stops firing and the heavy penalty becomes live -- which is a
        behaviour change that should be argued for, not inherited.

        A RELAY is not a producer, and the distinction is the whole point:
        main.py:~12497 does
            agent_signals["htf_trend_direction"] = market_data.get(
                "htf_trend_direction", 0)
        which copies a key nothing writes (that is exactly why the repo's own
        orphan scanner classes it COPY_ONLY). A first version of this guard
        matched any write and fired on that relay -- a detector that cannot
        tell a copy from a computation would have to be silenced, and a
        silenced guard is worse than none. So: flag a subscript write whose
        VALUE does not itself read the same key.
        """
        KEY = "htf_trend_direction"

        def _reads_same_key(node):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Constant)
                        and isinstance(sub.value, str) and sub.value == KEY):
                    return True
            return False

        producers = []
        for rel in ("main.py", "data_mgmt/market_data_pipeline.py",
                    "integration/integration_v36.py"):
            tree = ast.parse((REPO / rel).read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                for tgt in node.targets:
                    if (isinstance(tgt, ast.Subscript)
                            and isinstance(tgt.slice, ast.Constant)
                            and tgt.slice.value == KEY
                            and not _reads_same_key(node.value)):
                        producers.append(f"{rel}:{node.lineno}")

        assert not producers, (
            f"{KEY} now has a real PRODUCER at {producers} (a write whose "
            f"value is computed rather than relayed from the same key). "
            f"Re-open the P338 unknown branch: the EXHAUSTION heavy penalty "
            f"it suppresses is now measurable and should be argued for on "
            f"the data rather than inherited from this fix.")

    def test_that_producer_guard_can_actually_fire(self):
        """Anti-vacuity (P174): a detector built to ignore relays must still
        see a genuine computation, or it passes forever."""
        KEY = "htf_trend_direction"
        tree = ast.parse(
            'agent_signals["htf_trend_direction"] = 1 if sma_fast > '
            'sma_slow else -1\n')
        hits = [n for n in ast.walk(tree)
                if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Subscript)
                and getattr(n.targets[0].slice, "value", None) == KEY
                and not any(isinstance(s, ast.Constant)
                            and s.value == KEY for s in ast.walk(n.value))]
        assert hits, "the producer detector cannot see a real producer"


# ---------------------------------------------------------------------------
# Finding #4 -- the switch is wired at BOTH ends
# ---------------------------------------------------------------------------

class TestProfitMaxSwitchIsWired:

    KEY = "profit_max_exhaustion_requires_htf_alignment"

    def test_the_nested_config_key_takes_effect(self, tmp_path):
        base = json.loads(
            (REPO / "configs" / "live_high_risk.json").read_text(
                encoding="utf-8"))
        assert getattr(m.ProductionConfig.from_file(
            REPO / "configs" / "live_high_risk.json"), self.KEY) is True
        base.setdefault("profit_max", {})[
            "exhaustion_requires_htf_alignment"] = False
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(base), encoding="utf-8")
        assert getattr(m.ProductionConfig.from_file(p), self.KEY) is False, (
            "the config key is parsed but does not reach the field -- it was "
            "in neither _LATE_CONFIG_KEYS nor from_file (P313 covered the "
            "other 12 profit_max keys and missed this one)")

    def test_the_constructor_passes_it_through(self):
        """Structural, not a substring: find the ProfitMaxConfig(...) call in
        main.py and require the kwarg. Parsing the call means a renamed
        kwarg or a deleted line goes red, which `"x" in src` cannot do."""
        tree = ast.parse((REPO / "main.py").read_text(encoding="utf-8-sig"))
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id == "ProfitMaxConfig"]
        assert calls, "ProfitMaxConfig is no longer constructed in main.py"
        assert any(kw.arg == "exhaustion_requires_htf_alignment"
                   for c in calls for kw in c.keywords), (
            "ProfitMaxConfig(...) does not pass "
            "exhaustion_requires_htf_alignment -- the dataclass default "
            "governs and the config field is decorative again")


# ---------------------------------------------------------------------------
# Findings #5 and #6 -- claims and knobs
# ---------------------------------------------------------------------------

class TestHygiene:

    def test_the_exchange_banner_does_not_claim_kraken_only_execution(self):
        src = (REPO / "exchange" / "__init__.py").read_text(
            encoding="utf-8-sig")
        fn = src[src.index("def print_exchange_banner"):]
        body = fn[:fn.index("logger.info")]
        assert "KRAKEN ONLY" not in body.upper(), (
            "the package that CONTAINS coinbase_sleeve.py is printing "
            "'Execution Venue: KRAKEN ONLY' on every import -- falsified "
            "2026-06-13 (P239's header class)")
        assert "COINBASE" in body.upper()

    def test_the_unread_exchange_constants_are_marked_unread(self):
        src = (REPO / "exchange" / "__init__.py").read_text(
            encoding="utf-8-sig")
        i = src.index("SINGLE_EXCHANGE_MODE = True")
        assert "unread" in src[i:i + 400], (
            "SINGLE_EXCHANGE_MODE / ACTIVE_EXCHANGE / FROZEN_EXCHANGES have "
            "no readers anywhere; they must not read as live configuration")

    def test_governor_default_matches_the_declared_flag(self):
        """A getattr default is a policy statement about the ABSENT case; if
        it disagrees with the declaration it silently overrides it."""
        from tests._source_scan import code_only
        from configs.sota_flags import get_sota_flags
        declared = getattr(get_sota_flags(),
                           "ENABLE_REGIME_TRANSITION_BUFFER")
        src = code_only(REPO / "defense" / "governor_integration.py",
                        strip_docstrings=True)
        i = src.index("'ENABLE_REGIME_TRANSITION_BUFFER'")
        tail = src[i:i + 120]
        assert str(declared) in tail, (
            f"governor_integration defaults ENABLE_REGIME_TRANSITION_BUFFER "
            f"to the opposite of the declared {declared}")

    def test_the_dead_order_type_key_is_gone_from_the_live_profile(self):
        cfg = json.loads((REPO / "configs" / "live_high_risk.json").read_text(
            encoding="utf-8"))
        assert "default_order_type" not in cfg["execution"], (
            "a key nothing reads is back in the execution section, beside "
            "keys that ARE read -- it reads as configuring order type")
        assert any(k.startswith("_p338") for k in cfg["execution"]), (
            "the removal lost its provenance note")

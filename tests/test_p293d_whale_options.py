"""
[P293d] The three whale-influence options, built after the measured finding
that adding an ADVISE weight is a NO-OP on live orders.

The finding these tests exist to protect (verified in code, 2026-08-17):
  * fusion LAYER 3 is the ONLY place `result.direction` is set, and it reads
    DECIDE agents alone;
  * every other layer (CONFIRM, HTF, PARTIAL CONSENSUS, ADVISE, CAP) modifies
    `base_exposure`;
  * that exposure is overwritten at integration_v36.py:1678 by the tranche;
  * and the sleeve sizes by SIGN, discarding magnitude.
So the ADVISE channel is severed twice, and 22 of 26 agents cannot reach an
order through it.

  A. sleeve entry filter          — works (acts on the driver, not fusion)
  B. direction seat               — works (quant-slot injection)
  C. conviction to sleeve sizing  — the only reconnection of the exposure
                                    channel, and near-inert at 1-3 contracts
"""

import json
import re
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAIN = REPO / "main.py"


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig", errors="ignore")


def _strip_comments(src: str) -> str:
    out = []
    for line in src.splitlines():
        in_s, quote, cut = False, "", None
        for i, ch in enumerate(line):
            if in_s:
                if ch == quote and (i == 0 or line[i - 1] != "\\"):
                    in_s = False
            elif ch in "\"'":
                in_s, quote = True, ch
            elif ch == "#":
                cut = i
                break
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


# =============================================================================
# The finding itself — if the architecture changes, these must be revisited
# =============================================================================

class TestAdviseChannelIsSevered:
    """Pins WHY options A/B exist. If either of these ever fails, an ADVISE
    weight may have become meaningful and the whole design should be
    reconsidered rather than silently kept."""

    def test_direction_comes_only_from_the_decider(self):
        raw = _src(REPO / "signals" / "authority_fusion.py")
        assert "result.direction = decider_signal.direction" in raw
        # The LAYER markers are comments, so locate on RAW source and only
        # then strip comments inside the located block.
        m = re.search(r"LAYER 4: CONFIRM CHECK.*?LAYER 4\.25", raw, re.S)
        assert m, "CONFIRM layer not found"
        assert "result.direction =" not in _strip_comments(m.group(0)), (
            "CONFIRM now sets direction — the P293d premise has changed"
        )

    def test_advise_layer_only_scales_exposure(self):
        raw = _src(REPO / "signals" / "authority_fusion.py")
        m = re.search(r"LAYER 4\.75: ADVISE INFLUENCE.*?LAYER 5:", raw, re.S)
        assert m, "ADVISE layer not found"
        blk = _strip_comments(m.group(0))
        assert "base_exposure *= (1.0 + influence)" in blk
        assert "result.direction" not in blk, (
            "ADVISE now touches direction — revisit options A/B"
        )

    def test_tranche_overwrites_fusion_exposure(self):
        iv = _strip_comments(_src(REPO / "integration" / "integration_v36.py"))
        assert "intent.target_exposure = tranche_decision.target_exposure" in iv, (
            "the Bug #44 overwrite is what severs the exposure channel; if it "
            "is gone, option C's premise changed"
        )


# =============================================================================
# A — sleeve entry filter
# =============================================================================

class TestOptionAWhaleFilter:
    def test_ma_wrapper_delegates_to_the_generic(self):
        """One implementation, two entry points (P172) — the wrapper must be
        byte-equivalent to the generic across the whole truth table, or the
        two filters could drift apart."""
        from main import sleeve_agent_filter_decision, sleeve_ma_filter_decision
        for pos in (-2, -1, 0, 1, 2):
            for raw in (-2, -1, 0, 1, 2):
                for d in (-1.0, -0.4, 0.0, 0.4, 1.0):
                    assert sleeve_ma_filter_decision(pos, raw, d) == \
                        sleeve_agent_filter_decision(pos, raw, d, "ma")

    def test_reason_strings_are_agent_tagged(self):
        from main import sleeve_agent_filter_decision
        assert sleeve_agent_filter_decision(0, 1, -1.0, "whale")[2] == \
            "whale_disagrees_entry"
        assert sleeve_agent_filter_decision(0, 1, -1.0, "ma")[2] == \
            "ma_disagrees_entry"

    def test_silent_agent_fails_open(self):
        """A dead agent must not become a standing entry veto (P208)."""
        from main import sleeve_agent_filter_decision
        led, act, why = sleeve_agent_filter_decision(0, 1, 0.0, "whale")
        assert (led, act) == (1, "") and why == "whale_silent"

    def test_exits_are_never_filtered(self):
        from main import sleeve_agent_filter_decision
        assert sleeve_agent_filter_decision(3, 0, -1.0, "whale") == \
            (0, "", "no_target")

    def test_held_aligned_position_is_not_force_exited(self):
        """v1 scope: an entry filter must not become an exit engine."""
        from main import sleeve_agent_filter_decision
        led, act, why = sleeve_agent_filter_decision(1, 1, -1.0, "whale")
        assert act == "", "a held position must not be force-exited"
        assert why == "whale_disagrees_hold_kept"
        assert led == 0, "the LEDGER still records the disagreement"

    def test_flip_is_demoted_to_flat_not_reversed(self):
        from main import sleeve_agent_filter_decision
        assert sleeve_agent_filter_decision(1, -1, 1.0, "whale") == \
            (0, "flip_to_flat", "whale_disagrees_flip")

    def test_harness_and_prefix_registered_at_both_sites(self):
        from defense.strategy_shadow_v5_1 import build_whale_filter_shadow_harness
        h = build_whale_filter_shadow_harness()
        # [P294] This assertion used to read
        #     assert getattr(h, "log_prefix", None) == "whale_filter" or True
        # The attribute is `_log_prefix`, so the comparison was ALREADY False
        # and only the trailing `or True` kept it green — a check that could
        # not fail, in the test written to guard the new ledger (P174).
        assert h._log_prefix == "whale_filter"
        s = _src(REPO / "analytics" / "shadow_ic" / "compute_shadow_ic.py")
        assert re.search(
            r'prefixes: Tuple\[str, \.\.\.\] = \([^)]*"whale_filter"', s), \
            "not in the tuple default"
        cli = next(l for l in s.splitlines() if "microstructure,cascade" in l)
        assert "whale_filter" in cli, "not in the CLI default"

    def test_ledger_does_not_pool_into_the_ma_filter_exam(self):
        """[P294] The scorer groups by the record's `strategy` field, NOT by
        prefix or filename (compute_per_strategy_ic: `strat = r.get(...)`).

        Both harnesses reuse MAFilterEchoStrategy, so a shared strategy name
        pooled two different claims into one series — making the whale claim
        unmeasurable AND contaminating the ma_filter exam, whose PASS drives
        a live config flip. The inflated n also loosens the gate's own |t|
        requirement on a merge that means nothing.
        """
        from defense.strategy_shadow_v5_1 import (
            build_whale_filter_shadow_harness, build_ma_filter_shadow_harness)
        obs = {"_maf_ledger_dir": 1.0, "_maf_ma_dir": 1.0,
               "_maf_reason": "agrees"}
        whale = build_whale_filter_shadow_harness()._strategies[0]
        ma = build_ma_filter_shadow_harness()._strategies[0]
        w_name = whale.evaluate("BTC", obs).strategy_name
        m_name = ma.evaluate("BTC", obs).strategy_name
        assert w_name != m_name, (
            f"both ledgers group under {w_name!r} — the scorer cannot tell "
            f"them apart"
        )
        # The incumbent's name is load-bearing: changing it would orphan
        # every P236 row already on the server.
        assert m_name == "ma_filtered"
        assert w_name == "whale_filtered"

    def test_scorer_groups_by_strategy_field_not_by_prefix(self):
        """[P294] Pins the PREMISE of the test above. If the scorer ever
        grouped by filename instead, the distinctness requirement would be
        cosmetic — and this test says so rather than leaving the reason
        implicit in a comment."""
        s = _src(REPO / "analytics" / "shadow_ic" / "compute_shadow_ic.py")
        assert 'strat = r.get("strategy")' in s, (
            "compute_per_strategy_ic no longer groups by the record's "
            "strategy field — re-derive whether ledger names still matter"
        )

    def test_registration_did_not_weaken_the_p236_guard(self):
        """P248's lesson: the P236 guard pins ma_filter at the END of the CLI
        default. Appending after it would break that guard, and weakening
        another workstream's guard to admit a change is never the fix."""
        s = _src(REPO / "analytics" / "shadow_ic" / "compute_shadow_ic.py")
        assert re.search(r'default="[^"]*,ma_filter"', s), (
            "ma_filter must remain LAST in the CLI default"
        )

    def test_whale_stash_is_reset_before_any_early_return(self):
        """[P294] The stash must be CLEARED at the top of the tick, not only
        written at the bridge ~4000 lines down.

        P293d's comment claimed "reset-then-set on EVERY tick"; the code was
        set-only. A tick that returns early (P253's crash / disconnect
        returns, a prefetch failure, any earlier exception) left the previous
        tick's whale direction for the sleeve driver — which runs later in
        run_live regardless — to filter on, and wrote that stale claim into
        the whale_filter ledger. P155-L5, and the same shape P287 fixed for
        the eventfilter stash.
        """
        src = _strip_comments(_src(MAIN))
        reset = "self._last_whale_directions[asset] = 0.0"
        assert reset in src, "no reset site — the stash is set-only again"
        reset_at = src.index(reset)
        bridge = "agent_signals.get('whale_flow_direction', 0.0) or 0.0)"
        assert bridge in src, "the bridge write moved; re-derive this pin"
        assert reset_at < src.index(bridge), (
            "the reset must precede the bridge write, or it is not a reset"
        )
        # It must also precede the tick's own early returns, or the whole
        # point is lost. The DIAG init block is the first thing the tick
        # does, so anchoring there is what makes the reset unconditional.
        # (A comment marker cannot be the anchor here — _strip_comments
        # removes it, which is how the first cut of this test failed.)
        anchor = "_diag = {'tick_time'"
        assert anchor in src, "the tick's opening block moved"
        assert src.index(anchor) < reset_at < src.index(anchor) + 3000, (
            "the reset drifted away from the top of the tick"
        )

    def test_the_reset_is_the_fail_open_value(self):
        """0.0 must mean 'no opinion', i.e. the filter allows. A reset to a
        DIRECTION would turn a skipped tick into a standing entry veto
        (P208)."""
        from main import sleeve_agent_filter_decision
        # entry from flat, sleeve wants long, whale silent -> allowed
        assert sleeve_agent_filter_decision(0, 1, 0.0, "whale") ==             (1, "", "whale_silent")

    def test_driver_wiring_present_and_gated(self):
        src = _strip_comments(_src(MAIN))
        assert 'agent_tag="whale"' in src
        assert '"coinbase_whale_filter_enforce"' in src
        assert "_whale_filter_shadow" in src


# =============================================================================
# B — direction seat
# =============================================================================

class TestOptionBWhaleSeat:
    def test_threshold_is_single_sourced(self):
        """The +/-0.3 deadband must have ONE definition — the seat runs
        before the bridge writes the stash, and a second copy is how two
        consumers of 'the whale signal' start disagreeing (P172)."""
        from main import whale_direction_from_pressure
        assert whale_direction_from_pressure(0.31) == 1.0
        assert whale_direction_from_pressure(-0.31) == -1.0
        assert whale_direction_from_pressure(0.3) == 0.0
        assert whale_direction_from_pressure(0.0) == 0.0

    @pytest.mark.parametrize("bad", [None, "x", float("nan")])
    def test_unusable_pressure_is_no_opinion(self, bad):
        from main import whale_direction_from_pressure
        assert whale_direction_from_pressure(bad) == 0.0

    def test_bridge_uses_the_helper(self):
        src = _strip_comments(_src(MAIN))
        assert "_wh_dir = whale_direction_from_pressure(_wh_net)" in src
        # the old inline thresholds must be gone from the bridge
        assert "if _wh_net > 0.3:" not in src, (
            "the inline threshold copy is back"
        )

    def test_seat_is_gated_and_runs_after_the_other_seats(self):
        src = _strip_comments(_src(MAIN))
        i_mlp = src.find("MLP-SEAT")
        i_whale = src.find("WHALE-SEAT")
        assert i_mlp > 0 and i_whale > i_mlp, (
            "the whale seat must run last so precedence is deterministic"
        )
        assert '"whale_seat_mode"' in src

    def test_seat_asserts_the_same_alpha_bar(self):
        """A seat swap changes the DIRECTION source, never the alpha bar
        (P231/P237 govern that constant)."""
        src = _strip_comments(_src(MAIN))
        m = re.search(
            r"_ws_mode = str\(getattr.*?incumbent signal stands\"\)", src, re.S)
        assert m, "whale seat block not found"
        assert "30.0 * abs(_ws_dir)" in m.group(0), (
            "the seat must assert the same 30bps x |dir| the other seats use"
        )

    def test_missing_producer_does_not_seat(self):
        """Absence must never read as flat (P2) — an absent producer leaves
        the incumbent signal standing, it does not seat a 0.0."""
        src = _strip_comments(_src(MAIN))
        m = re.search(
            r"_ws_mode = str\(getattr.*?incumbent signal stands\"\)", src, re.S)
        assert m, "whale seat block not found"
        assert "_ws_press is None" in m.group(0), (
            "the absent-producer branch is gone"
        )
        assert "seat NOT taken" in _src(MAIN)


# =============================================================================
# C — conviction to sleeve sizing
# =============================================================================

class TestOptionCConviction:
    def test_disabled_is_byte_identical(self):
        from main import sleeve_conviction_contracts
        for raw in (-3, -1, 0, 1, 3):
            assert sleeve_conviction_contracts(raw, 0.1, False) == raw

    def test_scales_magnitude_only(self):
        from main import sleeve_conviction_contracts
        assert sleeve_conviction_contracts(4, 0.5, True) == 2
        assert sleeve_conviction_contracts(-4, 0.5, True) == -2

    def test_never_flips_sign(self):
        from main import sleeve_conviction_contracts
        for raw in (-3, -1, 1, 3):
            for c in (0.0, 0.01, 0.5, 1.0):
                out = sleeve_conviction_contracts(raw, c, True)
                assert out == 0 or (out > 0) == (raw > 0)

    def test_never_increases_magnitude(self):
        """Advisors may express doubt, not leverage.

        NOTE this is protected TWICE (the min(1.0, c) clamp and the
        min(scaled, abs(raw)) clamp), so a probe removing only one does not
        make it fail — the redundancy is deliberate, and the test asserts
        the OUTCOME rather than either mechanism. Both the driver helper and
        the sleeve method are checked, since they clamp independently.
        """
        from main import sleeve_conviction_contracts
        from exchange.coinbase_sleeve import CoinbaseSleeve
        for c in (1.0, 1.5, 2.0, 99.0, float("1e9")):
            assert abs(sleeve_conviction_contracts(2, c, True)) <= 2
            assert CoinbaseSleeve._apply_conviction(2, c) <= 2

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, "x", None])
    def test_unusable_conviction_leaves_size_unchanged(self, bad):
        """A fabricated 0 would read as 'every advisor said flat' — a very
        different claim from 'the ratio was unreadable' (P2)."""
        from main import sleeve_conviction_contracts
        assert sleeve_conviction_contracts(3, bad, True) == 3

    def test_the_honest_rounding_limit_is_real(self):
        """Documented limit: at 1 contract the channel is near-binary."""
        from main import sleeve_conviction_contracts
        assert sleeve_conviction_contracts(1, 0.9, True) == 1
        assert sleeve_conviction_contracts(1, 0.6, True) == 1
        assert sleeve_conviction_contracts(1, 0.3, True) == 0

    def test_sleeve_apply_conviction_matches_the_driver_helper(self):
        """Two implementations exist (driver-side pure helper and sleeve
        method); they must agree or sizing depends on which one ran."""
        from main import sleeve_conviction_contracts
        from exchange.coinbase_sleeve import CoinbaseSleeve
        # Valid values AND the unusable ones. The refusal branch is where the
        # two could most easily disagree (one returning 0, the other the size
        # unchanged) — and an earlier version of this test fed only valid
        # convictions, so a probe that made the sleeve return 0 on NaN left
        # it GREEN. Caught by falsification; the loop now reaches the branch.
        for size in (1, 2, 3, 8):
            for c in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5,
                      float("nan"), float("inf"), -1.0):
                assert CoinbaseSleeve._apply_conviction(size, c) == \
                    sleeve_conviction_contracts(size, c, True), (size, c)

    def test_fusion_records_conviction_ratio(self):
        from signals.authority_fusion import FusionResult
        assert FusionResult().fusion_conviction == 1.0, (
            "default must be 1.0 = 'no modulation information'"
        )

    def test_intent_field_is_declared_not_dynamic(self):
        """P239: a runtime-only attribute is invisible to serialization and
        absent on early-return intents (the P85 contract gap)."""
        from integration.integration_v36 import TradeIntentV36
        assert TradeIntentV36().fusion_conviction == 1.0

    def test_target_for_default_is_unchanged(self):
        """conviction defaults to 1.0 so every existing caller is
        byte-identical."""
        import inspect
        from exchange.coinbase_sleeve import CoinbaseSleeve
        sig = inspect.signature(CoinbaseSleeve.target_for)
        assert sig.parameters["conviction"].default == 1.0
        sig2 = inspect.signature(CoinbaseSleeve.manage_to_signal)
        assert sig2.parameters["conviction"].default == 1.0

    def test_driver_reads_the_local_intents_not_an_attribute(self):
        """`_live_intents` is a LOCAL of run_live; reading self._live_intents
        would getattr-default to 1.0 and make the flag silently inert — the
        exact failure mode this batch exists to remove."""
        src = _strip_comments(_src(MAIN))
        assert "(_live_intents or {}).get(_m_a)" in src
        assert "self._live_intents" not in src


# =============================================================================
# Config trio — all three OFF and absent from the live profile
# =============================================================================

P293D_FLAGS = [
    "coinbase_whale_filter_enforce",
    "whale_seat_mode",
    "whale_seat_assets",
    "fusion_conviction_to_sleeve",
]


class TestConfigTrio:
    @pytest.mark.parametrize("flag", P293D_FLAGS)
    def test_declared_and_parsed(self, flag):
        src = _src(MAIN)
        assert re.search(rf"^\s+{flag}:\s", src, re.M), f"{flag} not declared"
        assert f'data.get("{flag}"' in src, f"{flag} not parsed"

    def test_defaults_are_inert(self):
        from main import ProductionConfig
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text("{}", encoding="utf-8")
            c = ProductionConfig.from_file(p)
        assert c.coinbase_whale_filter_enforce is False
        assert c.whale_seat_mode == "off"
        assert c.whale_seat_assets is None
        assert c.fusion_conviction_to_sleeve is False

    def test_unknown_seat_mode_falls_back_to_off(self):
        """An unrecognised mode must never silently seat an agent on the
        decider slot."""
        from main import ProductionConfig
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({"whale_seat_mode": "banana"}), encoding="utf-8")
            assert ProductionConfig.from_file(p).whale_seat_mode == "off"
            p.write_text(json.dumps({"whale_seat_mode": "ENFORCE"}), encoding="utf-8")
            assert ProductionConfig.from_file(p).whale_seat_mode == "enforce"

    @pytest.mark.parametrize("flag", [
        "coinbase_whale_filter_enforce", "fusion_conviction_to_sleeve"])
    def test_undecided_flags_stay_out_of_the_live_profile(self, flag):
        """A and C remain operator decisions with their own P-entry."""
        prof = REPO / "configs" / "live_high_risk.json"
        if not prof.exists():
            pytest.skip("live profile not present")
        data = json.loads(prof.read_text(encoding="utf-8-sig"))
        assert flag not in data, (
            f"{flag} is in the live profile — that is an activation decision"
        )

    def test_whale_seat_is_the_DECIDED_value(self):
        """[P293j] The seat was turned ON by explicit operator instruction.

        P237 pattern: once a live-money value is decided, the pin becomes the
        DECIDED value rather than 'must be absent' — so a silent REVERT fails
        just as loudly as a silent enable. Both directions are live changes.
        """
        prof = REPO / "configs" / "live_high_risk.json"
        if not prof.exists():
            pytest.skip("live profile not present")
        data = json.loads(prof.read_text(encoding="utf-8-sig"))
        assert data.get("whale_seat_mode") == "enforce"
        assert set(data.get("whale_seat_assets") or []) == {"BTC", "ETH", "SOL"}

    def test_silent_whale_does_not_take_the_seat(self):
        """[P293j] whale is SPARSE (directional on 54/43/12% of ticks). Its
        deadband means NO OPINION, not flat — seating a 0.0 would force the
        book flat on the majority of ticks (a churn engine AND the P2 error)
        and would destroy the fallback that drives those ticks.

        This is where whale differs from the mlp seat it was modelled on:
        mlp emits continuously, whale does not. Copying the pattern without
        checking would have shipped a flattener.
        """
        src = _strip_comments(_src(MAIN))
        m = re.search(
            r"_ws_mode = str\(getattr.*?incumbent signal stands\"\)", src, re.S)
        assert m, "whale seat block not found"
        blk = m.group(0)
        assert "elif abs(whale_direction_from_pressure(_ws_press)) <= 1e-9:" in blk, (
            "a silent whale must be skipped, not seated as 0.0"
        )
        # and the seat write must come AFTER that guard
        assert blk.index("elif abs(whale_direction_from_pressure") <             blk.index('market_data["quant_direction"] = float(_ws_dir)')

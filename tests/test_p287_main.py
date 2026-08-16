"""[P287] The main.py half of the fix-all campaign — behavioral tests for the
pure decision functions the fixes were extracted into (P206/P234 pattern:
blocks that spent their lives dead need behavioral tests, not source pins),
plus wiring/ordering pins for the inline halves.

Findings covered (audit 2026-08-16):
  1.  ProfitMax vetoes were unclassified sleeve-flatten doors
  2.  v9-PATCH-2 dead since birth (undefined `order_type` NameError)
  3.  BullTransitionDetector never evaluated (missing _price_history +
      two producer-less inputs)
  4.  EC panic amplifier misroute (nested crowd.panic_score)
  5.  large_transaction_* had no market_data injection site
  6.  PartialConsensus triggers matched nothing real (P54 class)
  8.  P0/SOTA fed one-bar-stale sleeve equity
  9.  fuse feed: anchor advanced before record_pnl + freshness gate
  10. equity_valid certified only the Kraken half
  11. eventfilter gated-dir high-water dict
  12. mlpshadow shallow-copy mutated the obs stash
"""

import re
import types
from pathlib import Path

import pytest

import main as m
from main import (
    SLEEVE_HOLD,
    bt_funding_streak_update,
    bt_oi_change_4h,
    cc_onchain_injection,
    fuse_feed_freshness,
    partial_consensus_trigger,
    resolve_panic_score,
    sleeve_direction_from_intent as translate,
    sleeve_equity_freshness,
    v9p2_gate_decision,
)

REPO = Path(__file__).resolve().parents[1]
MAIN_SRC = (REPO / "main.py").read_text(encoding="utf-8")


def _intent(direction=0.0, target_exposure=0.0, veto_active=False,
            veto_reason=""):
    return types.SimpleNamespace(
        direction=direction, target_exposure=target_exposure,
        veto_active=veto_active, veto_reason=veto_reason)


def _region(start_marker, end_marker, src=MAIN_SRC):
    """Extract the source between two unique markers (raw text, so the
    markers — comments — survive)."""
    i = src.index(start_marker)
    j = src.index(end_marker, i)
    return src[i:j]


def _strip_comment_lines(block):
    return "\n".join(l for l in block.splitlines()
                     if not l.strip().startswith("#"))


# ---------------------------------------------------------------------------
# 1. ProfitMax vetoes HOLD the sleeve (behavioral, through the real
#    translator — the audit's top live finding)
# ---------------------------------------------------------------------------

class TestProfitMaxVetoesHold:
    """[FALSE_BREAKOUT_VETO] is an ENTRY-quality veto and
    [LOSS_STREAK_HALT] a trading halt; both were unclassified, so the
    translator's default LIQUIDATED every routed asset on them. The real
    strings below are the adapter's own f-string shapes."""

    def test_false_breakout_veto_holds(self):
        d, why = translate(_intent(
            direction=+0.62, target_exposure=0.30, veto_active=True,
            veto_reason="[FALSE_BREAKOUT_VETO] confidence=0.81, "
                        "entry_type=breakout, reasons=['volume_fade']"), 0.0)
        assert d is SLEEVE_HOLD
        assert why.startswith("hold_veto:")

    def test_loss_streak_halt_holds(self):
        d, why = translate(_intent(
            direction=-0.40, target_exposure=0.25, veto_active=True,
            veto_reason="[LOSS_STREAK_HALT] streak=4 >= 4: Trading halted"),
            0.0)
        assert d is SLEEVE_HOLD
        assert why.startswith("hold_veto:")

    def test_a_real_flatten_veto_still_flattens(self):
        """The fix must not have widened HOLD into a blanket: the alpha
        gate's veto still flattens."""
        d, why = translate(_intent(
            direction=0.0, target_exposure=0.0, veto_active=True,
            veto_reason="[v3.6.1] Alpha gate: Alpha 10bps < threshold 59bps"),
            0.0)
        assert d == 0.0
        assert why.startswith("veto_flat:")


# ---------------------------------------------------------------------------
# 2. v9-PATCH-2: pure decision + config trio + the block is really repaired
# ---------------------------------------------------------------------------

class TestV9P2GateDecision:

    def test_alpha_clears_friction_no_action(self):
        halve, action = v9p2_gate_decision(50.0, 20.0, False, False)
        assert action == "none"
        assert halve is False  # 50 >= 30

    def test_marginal_alpha_halves_urgency_without_veto(self):
        halve, action = v9p2_gate_decision(25.0, 20.0, False, False)
        assert halve is True   # 25 < 30
        assert action == "none"  # but 25 >= 20: no veto question

    def test_shadow_is_the_default_failing_direction(self):
        """enforce=False (the default, pinned absent from the live profile)
        must NOT produce a veto — the block was dead its whole life and
        arming FRICTION_EXCEEDS_EDGE on repair would flatten the standing
        BTC long (P141)."""
        halve, action = v9p2_gate_decision(15.0, 20.0, False, False)
        assert action == "shadow_would_veto"
        assert halve is True

    def test_enforce_produces_the_veto(self):
        _, action = v9p2_gate_decision(15.0, 20.0, False, True)
        assert action == "veto"

    def test_hold_band_beats_the_veto_even_enforced(self):
        """[P232] the hysteresis hold band is never overridden."""
        _, action = v9p2_gate_decision(15.0, 20.0, True, True)
        assert action == "hold_exempt"

    def test_live_numbers_would_shadow_veto_btc(self):
        """The rationale for shadow-first, as arithmetic: BTC alpha ~30bps
        vs friction ~23bps -> no veto, but alpha < friction x1.5 halves
        urgency; at friction 34.5bps the veto would fire."""
        halve, action = v9p2_gate_decision(30.0, 23.0, False, False)
        assert action == "none" and halve is True
        _, action2 = v9p2_gate_decision(30.0, 34.5, False, False)
        assert action2 == "shadow_would_veto"


class TestV9P2ConfigAndWiring:

    def test_config_field_declared_default_false(self):
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(m.ProductionConfig)}
        assert "dynamic_alpha_gate_enforce" in fields
        assert fields["dynamic_alpha_gate_enforce"].default is False

    def test_from_file_parses_the_key(self, tmp_path):
        import json
        cfg = {"dynamic_alpha_gate_enforce": True}
        p = tmp_path / "c.json"
        p.write_text(json.dumps(cfg))
        parsed = m.ProductionConfig.from_file(p)
        assert parsed.dynamic_alpha_gate_enforce is True

    def test_live_profile_does_not_carry_the_key(self):
        """Adding the key to the live profile IS the arming action and
        needs its own recorded decision (P237 pattern)."""
        import json
        live = json.loads((REPO / "configs" / "live_high_risk.json")
                          .read_text(encoding="utf-8"))
        assert "dynamic_alpha_gate_enforce" not in live

    def _block(self):
        return _region("# --- [v9-PATCH-2] Dynamic Alpha Gate",
                       "# --- end [v9-PATCH-2] ---")

    def test_the_undefined_name_is_gone(self):
        """The defect itself: `order_type` was an undefined name — every
        pass raised NameError into a DEBUG swallow. It must not appear in
        the repaired block's CODE (comments documenting the history may
        name it)."""
        code = _strip_comment_lines(self._block())
        assert not re.search(r"\border_type\b", code), (
            "order_type is back in the v9-PATCH-2 block — the NameError "
            "class that kept this control dead for its entire life")

    def test_decision_flows_through_the_pure_function(self):
        """P251 rule: bypassing v9p2_gate_decision would make the
        behavioral tests above test nothing about the live block."""
        assert "v9p2_gate_decision(" in self._block()

    def test_veto_only_under_enforce_and_shadow_logs(self):
        blk = self._block()
        assert "[V9P2-SHADOW] WOULD-VETO" in blk
        assert "dynamic_alpha_gate_enforce" in blk

    def test_diag_no_longer_reports_a_constant(self):
        """The old diag line was output={'active': True}, consumed=True
        unconditionally — a constant wearing a health check's name while
        the block raised on every pass (P174 class)."""
        assert "output={'active': True}, consumed=True)  # [FIX-31]" \
            not in MAIN_SRC
        i = MAIN_SRC.index("_diag_record('dynamic_alpha_gate'")
        seg = MAIN_SRC[i:i + 400]
        assert "_v9p2_health" in seg


# ---------------------------------------------------------------------------
# 3. Bull-transition inputs
# ---------------------------------------------------------------------------

class TestBtOiChange:

    def test_normal_delta(self):
        assert bt_oi_change_4h(100.0, 110.0) == pytest.approx(10.0)
        assert bt_oi_change_4h(110.0, 100.0) == pytest.approx(-10.0)

    def test_absent_inputs_yield_none_never_zero(self):
        """P2: absent stays absent — a fabricated 0 was exactly the dead
        input being replaced."""
        assert bt_oi_change_4h(None, 100.0) is None
        assert bt_oi_change_4h(100.0, None) is None
        assert bt_oi_change_4h(0.0, 100.0) is None
        assert bt_oi_change_4h("x", 100.0) is None

    def test_scale_mixed_jump_rejected(self):
        """open_interest has two scale-mixed producers (CoinGlass global vs
        Kraken venue, P253c recorded decision) — an inter-tick source
        switch must not read as a market move."""
        assert bt_oi_change_4h(1_000_000.0, 60_000_000.0) is None
        assert bt_oi_change_4h(60_000_000.0, 1_000_000.0) is None
        # 4.9x is within bound (violent but plausible), 5.1x is not
        assert bt_oi_change_4h(100.0, 490.0) is not None
        assert bt_oi_change_4h(100.0, 510.0) is None


class TestBtFundingStreak:

    def test_streak_advances_once_per_utc_day(self):
        st = {"day": None, "streak": 0}
        bt_funding_streak_update(st, "2026-08-16", 0.0001)
        assert st == {"day": "2026-08-16", "streak": 1}
        # same day, second tick: unchanged
        bt_funding_streak_update(st, "2026-08-16", 0.0002)
        assert st["streak"] == 1
        bt_funding_streak_update(st, "2026-08-17", 0.0001)
        assert st["streak"] == 2

    def test_negative_funding_resets(self):
        st = {"day": "2026-08-16", "streak": 5}
        bt_funding_streak_update(st, "2026-08-17", -0.0001)
        assert st["streak"] == 0

    def test_absent_funding_leaves_state_untouched(self):
        st = {"day": "2026-08-16", "streak": 3}
        bt_funding_streak_update(st, "2026-08-17", None)
        assert st == {"day": "2026-08-16", "streak": 3}


class TestBullTransitionWiring:
    """The detector NEVER EVALUATED ONCE: its first statement read
    `self._price_history`, an attribute with no initializer, and the
    AttributeError was swallowed at DEBUG — which also made P227b's
    persistence of its state machine vacuous."""

    def test_price_history_now_has_a_writer(self):
        writers = re.findall(r"self\._price_history\.setdefault", MAIN_SRC)
        assert len(writers) >= 1, (
            "no writer of self._price_history — the MOD-1 block is back to "
            "raising AttributeError on every tick (dead detector)")

    def test_maintenance_precedes_the_mod1_read(self):
        maint = MAIN_SRC.index("[P287] BULL-TRANSITION DETECTOR INPUTS")
        read = MAIN_SRC.index("_btc_prices = self._price_history.get")
        assert maint < read, (
            "the input-maintenance block must run BEFORE the MOD-1 read — "
            "ordering inverted, the detector reads an empty/missing dict")

    def test_the_debug_swallow_is_gone(self):
        assert "logger.debug(f'[BULL-TRANSITION] Error: {e}')" not in MAIN_SRC

    def test_detector_reaches_block_naked_short_on_two_conditions(self):
        """End-to-end through the REAL detector with inputs of the exact
        shapes main.py now produces: price>ma50 + 7d funding streak =
        2 conditions = ACTIVE (threshold 2)."""
        from risk.bull_transition_detector import BullTransitionDetector
        det = BullTransitionDetector()
        sig = det.evaluate(
            btc_price=65_000.0, btc_ma50=60_000.0,
            sol_btc_relative_strength=0.0,
            funding_positive_streak_days=7,
            oi_rising=False, liquidations_declining=False)
        assert sig.conditions_met == 2
        assert sig.action in ("REDUCE_SHORT", "BLOCK_NAKED_SHORT",
                              "REDUCE_SHORT_LIGHT")


# ---------------------------------------------------------------------------
# 4. EC panic amplifier
# ---------------------------------------------------------------------------

class TestResolvePanicScore:

    def test_nested_crowd_value_wins(self):
        assert resolve_panic_score(
            {"crowd": {"panic_score": 0.9},
             "_ssc_panic": 0.1}) == pytest.approx(0.9)

    def test_legacy_flat_fallbacks_still_read(self):
        assert resolve_panic_score({"_ssc_panic": 0.4}) == pytest.approx(0.4)
        assert resolve_panic_score({"panic_score": 0.3}) == pytest.approx(0.3)

    def test_absent_everywhere_is_zero(self):
        assert resolve_panic_score({}) == 0.0
        assert resolve_panic_score({"crowd": {}}) == 0.0

    def test_extreme_branch_is_now_arithmetically_reachable(self):
        """The defect: panic term constant 0 -> max amp = vix*0.3 + liq*0.2
        = 0.5 < 0.75, PANIC_EXTREME unreachable. With the real panic read,
        panic=1 + vix=1 clears the threshold."""
        panic, vix, liq = 1.0, 1.0, 0.0
        amp = panic * 0.5 + vix * 0.3 + liq * 0.2
        assert amp > 0.75
        # and the old dead-read world could never clear it:
        assert 0.0 * 0.5 + 1.0 * 0.3 + 1.0 * 0.2 <= 0.75

    def test_block_uses_the_resolver(self):
        assert "resolve_panic_score(market_data)" in MAIN_SRC


# ---------------------------------------------------------------------------
# 5. CC on-chain injection
# ---------------------------------------------------------------------------

class TestCcOnchainInjection:

    def _entry(self, **kw):
        d = dict(is_mock=False, large_transaction_count=1200,
                 average_transaction_value=2.5)
        d.update(kw)
        return types.SimpleNamespace(**d)

    def test_real_entry_injects_both(self):
        out = cc_onchain_injection(self._entry())
        assert out == {"large_transaction_count": 1200.0,
                       "average_transaction_value": 2.5}

    def test_mock_entry_injects_nothing(self):
        assert cc_onchain_injection(self._entry(is_mock=True)) == {}

    def test_absent_entry_injects_nothing(self):
        assert cc_onchain_injection(None) == {}

    def test_zero_count_stays_absent_not_zero(self):
        """P2: 'the feed produced nothing' must be indistinguishable from
        'key absent', never from 'zero whales observed'."""
        assert cc_onchain_injection(self._entry(
            large_transaction_count=0)) == {}

    def test_injection_site_exists_in_main(self):
        assert "cc_onchain_injection(" in MAIN_SRC
        # explicit-key writes, not dict.update (dynamic_site_count is not
        # re-baselineable — a computed write makes market_data unprovable)
        assert 'market_data["large_transaction_count"] = (' in MAIN_SRC
        assert 'market_data["average_transaction_value"] = (' in MAIN_SRC
        assert "market_data.update(cc_onchain_injection(" not in MAIN_SRC


# ---------------------------------------------------------------------------
# 6. PartialConsensus triggers
# ---------------------------------------------------------------------------

class TestPartialConsensusTrigger:

    def test_real_conflict_string_is_live(self):
        assert partial_consensus_trigger(
            "[v3.6.1] NO_TRADE: ALL_CONFLICT_FLAT") == "live"

    def test_real_alpha_gate_string_is_shadow(self):
        assert partial_consensus_trigger(
            "[v3.6.1] Alpha gate: Alpha 30bps < threshold 47bps") == "shadow"

    def test_the_old_dead_tags_match_nothing(self):
        """The P54-class defect: the old tags appear in NO real
        veto_reason — and the classifier no longer honors them."""
        assert partial_consensus_trigger("tag ALPHA_GATE tag") is None
        assert partial_consensus_trigger("NO_CONSENSUS") is None
        assert partial_consensus_trigger("[WEEKEND] alpha 10bps") is None
        assert partial_consensus_trigger(None) is None
        assert partial_consensus_trigger("") is None

    def test_conflict_beats_shadow_when_both_present(self):
        assert partial_consensus_trigger(
            "CONFLICT | [v3.6.1] Alpha gate: x") == "live"

    def test_wiring_shadow_never_clears_the_veto(self):
        """In the [FIX-L1-05] block, the veto-clearing assignments must sit
        under the non-shadow branch only."""
        blk = _region("[FIX-L1-05] PartialConsensusChecker",
                      "# [PATCH-5] Tranche-Aware Deadlock")
        assert "partial_consensus_trigger(" in blk
        assert "[PARTIAL-CONSENSUS-SHADOW]" in blk
        # the shadow branch comes FIRST and contains no intent mutation
        shadow_i = blk.index('_pc_trigger == "shadow"')
        enforce_i = blk.index("intent.veto_active = False")
        assert shadow_i < enforce_i
        shadow_branch = blk[shadow_i:blk.index("elif pc_result")]
        assert "intent.veto_active" not in shadow_branch
        assert "intent.target_exposure" not in shadow_branch


# ---------------------------------------------------------------------------
# 8. P0/SOTA sleeve equity is read live
# ---------------------------------------------------------------------------

class TestP0SleeveLiveRead:

    def test_p0_feed_calls_sleeve_equity_usd(self):
        seg = _region("[P287] Read the sleeve equity LIVE",
                      "combined_p0_equity(")
        assert "sleeve_equity_usd()" in seg, (
            "the P0/SOTA feed is back on the cached _last_equity_usd — the "
            "35% kill switch judges drawdowns one 4H bar late again")

    def test_cached_read_survives_only_as_fallback(self):
        seg = _strip_comment_lines(_region(
            "[P287] Read the sleeve equity LIVE", "combined_p0_equity("))
        # the cached read exists exactly once, inside the except fallback
        assert seg.count("_last_equity_usd") == 1
        assert "except Exception" in seg


# ---------------------------------------------------------------------------
# 9. fuse feed: freshness + ordering
# ---------------------------------------------------------------------------

class TestFuseFeedFreshness:

    def test_reconcile_not_ok_refuses(self):
        ok, why = fuse_feed_freshness(False, 4000.0, 60.0)
        assert not ok and why == "reconcile_not_ok"

    def test_nonpositive_equity_refuses(self):
        ok, _ = fuse_feed_freshness(True, 0.0, 60.0)
        assert not ok

    def test_fresh_equity_passes(self):
        ok, why = fuse_feed_freshness(True, 4000.0, 3600.0)
        assert ok and why == "ok"

    def test_stale_age_refuses_with_named_reason(self):
        """The finding: a portfolio-endpoint outage passed the old gate
        (which only saw _reconcile_ok, a POSITIONS-endpoint flag) and fed
        delta=0 'no loss' every tick while positions bled."""
        ok, why = fuse_feed_freshness(True, 4000.0, 100_000.0)
        assert not ok and why.startswith("equity_stale")

    def test_unknown_age_preserves_old_behavior(self):
        """P85: an older sleeve build without the accessor must not silence
        Rule #3's only input."""
        ok, _ = fuse_feed_freshness(True, 4000.0, None)
        assert ok

    def test_unreadable_age_refuses(self):
        ok, why = fuse_feed_freshness(True, 4000.0, "garbage")
        assert not ok and why == "equity_age_unreadable"

    def test_boundary_exactly_8h_passes(self):
        ok, _ = fuse_feed_freshness(True, 4000.0, 28800.0)
        assert ok


class TestFuseFeedOrderingAndWiring:

    def _block(self):
        return _region("[P209] FEED THE EXISTENCE FUSE",
                       "[P209] PERSIST IT")

    def test_record_pnl_precedes_the_anchor_advance(self):
        """The loss-forgiveness ordering: the old block advanced
        _fuse_sleeve_anchor_equity BEFORE record_pnl, so an exception there
        (swallowed by the block's handler) moved the anchor past the
        interval and permanently dropped its loss from the 28d window."""
        blk = self._block()
        rec = blk.index("_fz.record_pnl(")
        anchor = blk.index("self._fuse_sleeve_anchor_equity = _fz_eq")
        assert rec < anchor, (
            "anchor advanced before record_pnl — an exception in record_pnl "
            "forgives the interval's loss (Rule #3 corruption)")

    def test_gate_flows_through_the_pure_function(self):
        assert "fuse_feed_freshness(" in self._block()

    def test_age_accessor_is_getattr_defended(self):
        blk = self._block()
        assert 'getattr(\n                                        _fz_sleeve, "sleeve_equity_age_sec",' \
            in blk or '"sleeve_equity_age_sec"' in blk


# ---------------------------------------------------------------------------
# 10. combined equity validity halves
# ---------------------------------------------------------------------------

class TestSleeveEquityFreshness:

    def test_no_sleeve_is_vacuously_fresh(self):
        assert sleeve_equity_freshness(False, None) is True
        assert sleeve_equity_freshness(False, 999_999.0) is True

    def test_unknown_age_preserves_old_field_semantics(self):
        assert sleeve_equity_freshness(True, None) is True

    def test_stale_age_is_not_fresh(self):
        assert sleeve_equity_freshness(True, 100_000.0) is False
        assert sleeve_equity_freshness(True, float("inf")) is False

    def test_fresh_age_is_fresh(self):
        assert sleeve_equity_freshness(True, 60.0) is True

    def test_unreadable_age_is_not_fresh(self):
        assert sleeve_equity_freshness(True, "x") is False


class TestEquityRecordHalves:

    def test_record_carries_both_halves_and_the_and(self):
        i = MAIN_SRC.index('"kraken_equity": round(_hb_kraken_eq, 4)')
        seg = MAIN_SRC[i:i + 1200]
        assert '"kraken_valid"' in seg
        assert '"sleeve_fresh"' in seg
        assert "_hb_eq_valid\n" in seg or "and _hb_sleeve_fresh" in seg, (
            "equity_valid is no longer the AND of the halves")

    def test_heartbeat_computes_sleeve_freshness(self):
        assert "sleeve_equity_freshness(" in MAIN_SRC


# ---------------------------------------------------------------------------
# 11. eventfilter gated-dir written on every branch
# ---------------------------------------------------------------------------

class TestEventfilterBranchWrites:

    def _driver(self):
        return _region("[P287] eventfilter-claim dict",
                       "self._coinbase_manage_last = _m_summary")

    def test_hold_branch_writes_a_claim(self):
        blk = self._driver()
        hold = blk[blk.index('f"HOLD({_m_why},"'):blk.index(
            "[P236] model_alpha disagreement")]
        assert "self._enh_gated_dirs[_m_a]" in hold, (
            "the HOLD branch no longer writes the eventfilter claim — the "
            "dict is a high-water mark again (P155-L5) and the September "
            "eventfilter ledger records directions the book does not hold")

    def test_ma_veto_branch_writes_a_claim(self):
        blk = self._driver()
        mav = blk[blk.index('f"MA_VETO(entry'):blk.index('"flip_to_flat"')]
        assert "self._enh_gated_dirs[_m_a]" in mav

    def test_cooldown_branch_writes_flat(self):
        blk = self._driver()
        cd = blk[blk.index('f"COOLDOWN({_cd_age}'):blk.index(
            "[P277] stash the FINAL gated dir")]
        assert "self._enh_gated_dirs[_m_a] = 0.0" in cd

    def test_managed_path_still_writes(self):
        assert "self._enh_gated_dirs[_m_a] = _m_dir" in MAIN_SRC


# ---------------------------------------------------------------------------
# 12. mlpshadow stash copy
# ---------------------------------------------------------------------------

class TestMlpStashCopy:

    def test_per_asset_dicts_are_copied(self):
        i = MAIN_SRC.index("[P284] mlp_small Rung-3 shadow tick")
        seg = MAIN_SRC[i:i + 2500]
        assert "_mls_feats = {a: dict(f) for a, f in" in seg, (
            "the shallow copy is back — .update(_fv) mutates the obs "
            "builder's single-source stash in place")

    def test_the_copy_semantics_hold(self):
        """The idiom itself, behaviorally: outer dict() shares inner dicts;
        the comprehension does not."""
        stash = {"BTC": {"a": 1.0}}
        shallow = dict(stash)
        shallow["BTC"].update({"fv2_x": 9.0})
        assert "fv2_x" in stash["BTC"]  # the defect
        stash2 = {"BTC": {"a": 1.0}}
        deep = {a: dict(f) for a, f in stash2.items()}
        deep["BTC"].update({"fv2_x": 9.0})
        assert "fv2_x" not in stash2["BTC"]  # the fix


# ---------------------------------------------------------------------------
# smaller wiring pins: governor wraps, REDUCE_NOOP, funding_bias, B6
# ---------------------------------------------------------------------------

class TestSmallerWiringPins:

    def test_anti_churn_restore_is_wrapped(self):
        i = MAIN_SRC.index("self._anti_churn.from_dict(data)")
        pre = MAIN_SRC[max(0, i - 120):i]
        assert "try:" in pre, (
            "the AC-5 restore lost its own guard — one malformed field "
            "aborts the whole restore after most governors already landed")

    def test_smoother_restore_is_wrapped(self):
        i = MAIN_SRC.index("self._regime_smoother_state.update(rs_data)")
        pre = MAIN_SRC[max(0, i - 300):i]
        assert "try:" in pre

    def test_reduce_noop_refreshes_fast_risk(self):
        assert '_frs_st in ("EXITED", "REDUCED", "REDUCE_NOOP")' in MAIN_SRC

    def test_funding_bias_uses_the_effective_rate(self):
        i = MAIN_SRC.index('market_data["funding_bias"] = float(')
        seg = MAIN_SRC[i:i + 300]
        assert 'market_data.get("funding_rate"' in seg, (
            "funding_bias recomputed from the Kraken ticker again — with "
            "venue-aware funding ON the two keys disagree in SIGN on the "
            "same tick (P218's measured table)")
        assert "_kf_ticker.funding_rate_8h" not in seg

    def test_b6_no_longer_overwrites_lead_lag_edge(self):
        """The two-producer collision: B6 wrote a DIFFERENT estimator's
        numbers into lead_lag_edge/confidence, so consumers gating on
        lead_lag_authority consumed B6's edge certified by the other
        estimator's flag."""
        assert "agent_signals['b6_lead_lag_edge']" in MAIN_SRC
        # the only remaining lead_lag_edge writers: the setdefault default
        # and no direct assignment
        assert "agent_signals['lead_lag_edge'] = round(" not in MAIN_SRC
        assert "market_data['lead_lag_edge'] = agent_signals" not in MAIN_SRC

    def test_run_live_restore_message_is_honest(self):
        assert "LIVE starting with FRESH governors. The existence fuse's" \
            not in MAIN_SRC
        assert "FRESH or PARTIALLY restored" in MAIN_SRC

"""[P206] Translating a gated intent into a Coinbase sleeve position target.

Since Phase B the sleeve is the only venue that places orders, and it is driven
from `_last_quant_directions` — written at main.py:6480/:7834, both BEFORE
`engine.decide()`. So it trades a PRE-GATE snapshot and no risk control binds on
the venue holding the risk (P201). This wires it to the gated intent instead.

THE TRAP THIS FILE EXISTS TO PIN. The literal edit is two lines — read
`_live_intents` instead of the dict. That version is wrong and would liquidate
the book, because the gate stack speaks ORDER semantics ("should I send an
order?") and the sleeve speaks POSITION-TARGET semantics ("what position should
exist?"). Five specific ways it goes wrong, one test class each:

  1. a veto does NOT imply `direction` was zeroed;
  2. some vetoes mean HOLD, not FLAT (anti-churn: "already at target");
  3. some vetoes are Kraken-spot-only and do not apply to a perp venue;
  4. `direction` is overloaded as a CLOSE instruction (`-current`, exposure 0);
  5. a missing intent must mean HOLD, not 0 — else a data outage liquidates.
"""

import types

import pytest

from main import (
    SLEEVE_HOLD,
    sleeve_direction_from_intent as translate,
)


def _intent(direction=0.0, target_exposure=0.0, veto_active=False, veto_reason=""):
    return types.SimpleNamespace(
        direction=direction, target_exposure=target_exposure,
        veto_active=veto_active, veto_reason=veto_reason)


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------

class TestUnvetoedIntentIsUsedDirectly:

    def test_long_intent_passes_through(self):
        d, why = translate(_intent(direction=+0.9998, target_exposure=0.80), 0.0)
        assert d == pytest.approx(0.9998)
        assert why == "gated_direction"

    def test_short_intent_passes_through(self):
        """The sleeve trades perps and CAN hold a short — do not suppress it."""
        d, _ = translate(_intent(direction=-0.75, target_exposure=0.40), 0.0)
        assert d == pytest.approx(-0.75)

    def test_it_ignores_the_pre_gate_value_entirely(self):
        """The whole point: the ungated signal must not leak through."""
        d, _ = translate(_intent(direction=+0.20, target_exposure=0.10),
                         fallback_dir=-1.0)
        assert d == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# 1 + 2 + 3. vetoes
# ---------------------------------------------------------------------------

class TestVetoHandling:

    def test_alpha_gate_veto_flattens(self):
        """Today's live case, and the reason to do this at all: the alpha gate
        (Non-Negotiable Rule #1) refuses ETH/SOL on friction and zeroes both
        fields. Reconnecting makes that bind on the venue that trades."""
        d, why = translate(_intent(
            direction=0.0, target_exposure=0.0, veto_active=True,
            veto_reason="[v3.6.1] Alpha gate: Alpha 10bps < threshold 59bps"), 0.0)
        assert d == 0.0
        assert why.startswith("veto_flat:")

    def test_a_veto_that_did_NOT_zero_direction_still_flattens(self):
        """Landmine 1, from a real emitted record:
        direction=-0.3327 target_exposure=0.2495 veto_active=True
        ([WEEKEND] alpha 10bps < min 20bps). Passing `direction` through would
        OPEN a short the gate just refused."""
        d, why = translate(_intent(
            direction=-0.3327, target_exposure=0.2495, veto_active=True,
            veto_reason="[WEEKEND] [AP-5] Weekend alpha 10bps < min 20bps"), 0.0)
        assert d == 0.0, "opened a position the gate vetoed"
        assert why.startswith("veto_flat:")

    def test_anti_churn_veto_means_HOLD_not_flatten(self):
        """Landmine 2 — the one that would liquidate on every stable tick.
        EXPOSURE_DELTA_BELOW_THRESHOLD means 'already at target, send nothing'.
        Under position-target semantics, mapping it to 0 means FLATTEN."""
        d, why = translate(_intent(
            direction=+0.9, target_exposure=0.8, veto_active=True,
            veto_reason="EXPOSURE_DELTA_BELOW_THRESHOLD"), 0.0)
        assert d is SLEEVE_HOLD, (
            "anti-churn veto mapped to a position change — this flattens the "
            "book on every tick where the position is already correct"
        )
        assert why.startswith("hold_veto:")

    def test_b1_spot_short_block_does_not_apply_to_a_perp_venue(self):
        """Landmine 3. B1 blocks short ENTRIES because Kraken SPOT cannot hold a
        short. The sleeve trades perps and can. B1 also zeroes direction, so the
        signal is unrecoverable from the intent — fall back to the pre-gate
        value, which is the trend signal B1 deleted."""
        d, why = translate(_intent(
            direction=0.0, target_exposure=0.0, veto_active=True,
            veto_reason="B1_SPOT_SHORT_BLOCK"), fallback_dir=-1.0)
        assert d == pytest.approx(-1.0), "a spot-only constraint suppressed a perp short"
        assert why.startswith("venue_na_veto:")

    def test_b1_combined_with_a_real_veto_still_flattens(self):
        """If a genuine risk veto also fired, that one governs."""
        d, why = translate(_intent(
            direction=0.0, target_exposure=0.0, veto_active=True,
            veto_reason="[v3.6.1] Alpha gate: ... | B1_SPOT_SHORT_BLOCK"),
            fallback_dir=-1.0)
        assert d == 0.0
        assert why.startswith("veto_flat:")


# ---------------------------------------------------------------------------
# 4. direction overloaded as a close instruction
# ---------------------------------------------------------------------------

class TestCloseEncoding:

    def test_close_encoding_flattens_rather_than_opening_the_opposite_side(self):
        """Landmine 4. The existence fuse and the deadlock abort encode 'close'
        as direction=-current, target_exposure=0, veto_active=False. Fed to
        target_for_signal that OPENS the opposite side. target_exposure is the
        discriminator and manage_to_signal never sees it."""
        d, why = translate(_intent(
            direction=-1.0, target_exposure=0.0, veto_active=False), 0.0)
        assert d == 0.0, "close instruction opened a short instead of flattening"
        assert why == "zero_target_exposure"

    def test_the_mirror_case_long(self):
        d, _ = translate(_intent(direction=+1.0, target_exposure=0.0), 0.0)
        assert d == 0.0


# ---------------------------------------------------------------------------
# 5. missing intent
# ---------------------------------------------------------------------------

class TestMissingIntentHolds:

    def test_no_intent_this_tick_is_HOLD_not_zero(self):
        """Landmine 5. An asset skipped by a prefetch failure has no intent.
        Defaulting to 0.0 turns a data outage into an unintended liquidation."""
        d, why = translate(None, fallback_dir=+1.0)
        assert d is SLEEVE_HOLD
        assert why == "no_intent_this_tick"

    def test_it_does_not_fall_back_to_the_ungated_signal(self):
        """A silent fallback to the pre-gate value is the gap being closed."""
        d, _ = translate(None, fallback_dir=+1.0)
        assert d is not 1.0 and d is SLEEVE_HOLD  # noqa: F632 — identity is the point


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

class TestWiring:

    def test_the_flag_is_a_real_config_field(self):
        """P201 just fixed two flags read by getattr and never parsed."""
        from pathlib import Path
        from main import ProductionConfig
        c = ProductionConfig.from_file(
            Path(__file__).resolve().parents[1] / "configs" / "live_high_risk.json")
        assert hasattr(c, "coinbase_use_gated_intent")

    def test_the_code_default_is_off(self):
        """The invariant is that the CODE ships OFF — enabling changes live order
        behaviour, so it must be a deliberate profile edit (P141).

        Deliberately asserts the dataclass default, NOT the live profile's value.
        An earlier version asserted the latter and failed the moment the operator
        enabled it, which is a test asserting a decision rather than a contract.
        """
        import dataclasses
        from main import ProductionConfig
        default = {f.name: f.default
                   for f in dataclasses.fields(ProductionConfig)}["coinbase_use_gated_intent"]
        assert default is False

    def test_the_json_key_actually_takes_effect(self):
        import json, os, tempfile
        from pathlib import Path
        from main import ProductionConfig
        p = Path(__file__).resolve().parents[1] / "configs" / "live_high_risk.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        d["coinbase_use_gated_intent"] = True
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                        encoding="utf-8")
        json.dump(d, f); f.close()
        try:
            assert ProductionConfig.from_file(Path(f.name)).coinbase_use_gated_intent is True
        finally:
            os.unlink(f.name)

    def test_the_driver_consults_the_translator_and_honours_HOLD(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "main.py").read_text(
            encoding="utf-8", errors="replace")
        # Anchor on the DRIVER's getattr, not the from_file parse site.
        i = src.index('getattr(self.config, "coinbase_use_gated_intent"')
        window = src[i:i + 1800]
        assert "sleeve_direction_from_intent" in window
        assert "SLEEVE_HOLD" in window, "HOLD is not honoured at the call site"
        assert "_live_intents.get" in window


class TestFlipPersistHoldClassification:
    """[P231] The L2-CHURN flip-persistence guard (main.py ~12520) stamps
    veto_reason FLIP_PERSIST_HOLD with documented semantics 'hold the current
    position — no close, no reverse'. It was missing from _SLEEVE_HOLD_VETOES,
    so the translator classified it veto_flat: the sleeve would LIQUIDATE the
    position the guard exists to hold. Latent (the guard reads the empty
    Kraken book today) — defused before _paper_positions can ever repopulate."""

    def test_flip_persist_hold_is_a_hold_not_a_flatten(self):
        import main as m
        tgt, why = m.sleeve_direction_from_intent(
            _intent(direction=0.0, target_exposure=0.0, veto_active=True,
                    veto_reason="[L2-CHURN] FLIP_PERSIST_HOLD ETH 1/2"),
            fallback_dir=0.7)
        assert tgt is m.SLEEVE_HOLD, (
            "P231 regression: FLIP_PERSIST_HOLD translates to a flatten — "
            "the hold guard would liquidate the book when it first fires."
        )

    def test_ordinary_vetoes_still_flatten(self):
        import main as m
        tgt, why = m.sleeve_direction_from_intent(
            _intent(direction=0.0, veto_active=True,
                    veto_reason="[v3.6.1] Alpha gate: Alpha 10bps < 55bps"),
            fallback_dir=0.7)
        assert tgt == 0.0 and why.startswith("veto_flat")


class TestVetoStringCouplingDriftGuard:
    """[P240] The translator classifies vetoes by SUBSTRING match against
    free-text veto_reason strings written at ~36 independent sites. The
    failure mode is silent and severe: rename a veto at its write site and
    the translator reclassifies "hold, no order" as veto_flat — the sleeve
    liquidates the position the guard exists to hold (exactly how the P231
    FLIP_PERSIST_HOLD bug arose: the set and the write site disagreed).

    Guard: every member of both classification sets must still appear in
    COMMENT-STRIPPED source OUTSIDE the set definitions — i.e. at a real
    write site. Renaming either half alone goes red; renaming both
    consistently passes. Comments are stripped so a stale comment quoting
    the old name cannot satisfy the check (the P177 trap, inverted)."""

    def _write_site_corpus(self):
        import re
        from pathlib import Path
        from tests._source_scan import code_only
        repo = Path(__file__).resolve().parents[1]
        # [P265] The corpus spans every file with sleeve-classified veto write
        # sites: main.py AND integration_v36.py (the data-integrity vetoes —
        # "Data validation failed", "[DATA_INVALID]" — are written in decide()).
        src = code_only(repo / "main.py", strip_docstrings=True)
        src += code_only(repo / "integration" / "integration_v36.py",
                         strip_docstrings=True)
        # [P265] DATA_HEALTH_* reasons surface via "[TRADE_GATE] {reason.name}"
        # — the enum members live in trade_gate.py.
        src += code_only(repo / "defense" / "trade_gate.py",
                         strip_docstrings=True)
        # Excise the definition statements themselves.
        src = re.sub(r"_SLEEVE_HOLD_VETOES\s*=\s*\([^)]*\)", "", src)
        src = re.sub(r"_SLEEVE_VENUE_NA_VETOES\s*=\s*\([^)]*\)", "", src)
        return src

    def test_every_hold_veto_has_a_live_write_site(self):
        import main as m
        corpus = self._write_site_corpus()
        for member in m._SLEEVE_HOLD_VETOES:
            assert member in corpus, (
                f"{member!r} is in _SLEEVE_HOLD_VETOES but no write site "
                f"emits it — the hold classification is dead and the next "
                f"firing of that guard becomes a veto_flat LIQUIDATION "
                f"(the P231 failure class)"
            )

    def test_every_venue_na_veto_has_a_live_write_site(self):
        import main as m
        corpus = self._write_site_corpus()
        for member in m._SLEEVE_VENUE_NA_VETOES:
            assert member in corpus, (
                f"{member!r} is in _SLEEVE_VENUE_NA_VETOES but no write "
                f"site emits it — the venue-inapplicable carve-out is dead"
            )

    def test_sets_are_nonempty(self):
        """The guards above iterate the sets — empty sets would pass
        vacuously (P174)."""
        import main as m
        assert len(m._SLEEVE_HOLD_VETOES) >= 2
        assert len(m._SLEEVE_VENUE_NA_VETOES) >= 1


# ---------------------------------------------------------------------------
# [P265] Data-integrity vetoes mean HOLD, not flatten
# ---------------------------------------------------------------------------

class TestP265DataIntegrityVetoesHold:
    """A degraded-data tick used to LIQUIDATE the routed book: the three
    in-decide data-integrity veto writers ("[v3.6.1] Data validation failed",
    "[v3.6.1] NO_TRADE: STALE_DATA/FEED_DISAGREEMENT", DP-24's
    "[DATA_INVALID] synthetic fallback") produced reasons outside
    _SLEEVE_HOLD_VETOES, so sleeve_direction_from_intent classified them as
    veto_flat -> manage_to_signal(asset, 0.0) closed every Coinbase perp at
    market and re-entered on recovery — a forced round trip exactly during an
    outage window. "State unknown must never be read as no position wanted"
    (P253's own rule), through the three doors P253 did not cover."""

    def test_data_validation_failure_holds(self):
        d, r = translate(
            _intent(direction=-0.33, target_exposure=0.25, veto_active=True,
                    veto_reason="[v3.6.1] Data validation failed: data_age_seconds 74.2 > 60.0"),
            fallback_dir=0.9)
        assert d is SLEEVE_HOLD, (
            f"got {d!r} — a stale-data tick flattens the sleeve (the P265 "
            "liquidation-on-outage class)")
        assert "hold_veto" in r

    def test_synthetic_fallback_dp24_holds(self):
        d, _r = translate(
            _intent(veto_active=True,
                    veto_reason="[DATA_INVALID] synthetic fallback data, new entries blocked"),
            fallback_dir=0.9)
        assert d is SLEEVE_HOLD, (
            "DP-24's own text says 'new entries blocked' — the semantic is "
            "HOLD, the classification was flatten")

    @pytest.mark.parametrize("trigger", [
        "DATA_INTEGRITY_FAIL", "STALE_DATA", "FEED_DISAGREEMENT"])
    def test_data_integrity_no_trade_subtypes_hold(self, trigger):
        d, r = translate(
            _intent(veto_active=True,
                    veto_reason=f"[v3.6.1] NO_TRADE: {trigger}"),
            fallback_dir=0.9)
        assert d is SLEEVE_HOLD
        # STALE_DATA matches the direct hold set (it also covers the trade
        # gate's freshness veto); the others via the NO_TRADE trigger tuple.
        assert "hold_veto" in r

    @pytest.mark.parametrize("trigger", [
        "FLASH_CRASH", "EXTREME_DVOL", "LIQUIDITY_CRITICAL",
        "CORRELATION_COLLAPSE", "ALL_CONFLICT_FLAT"])
    def test_market_risk_no_trade_subtypes_still_flatten(self, trigger):
        # The carve-out must not swallow the risk responses — those NO_TRADE
        # subtypes mean "get out", and flattening is the intended action.
        d, r = translate(
            _intent(veto_active=True,
                    veto_reason=f"[v3.6.1] NO_TRADE: {trigger}"),
            fallback_dir=0.9)
        assert d == 0.0, (
            f"NO_TRADE: {trigger} no longer flattens — the data-integrity "
            "carve-out over-reached into the market-risk responses")
        assert r.startswith("veto_flat")

    def test_alpha_gate_veto_still_flattens(self):
        # Rule #1 binding on the venue that trades (P206's activation
        # behavior) must survive this change.
        d, r = translate(
            _intent(direction=0.9, target_exposure=0.3, veto_active=True,
                    veto_reason="[v3.6.1] Alpha gate: Alpha 10bps < threshold 53bps"),
            fallback_dir=0.9)
        assert d == 0.0
        assert r.startswith("veto_flat")

    def test_hold_triggers_are_pinned_against_the_enum(self):
        # The composed reason is f"NO_TRADE: {NoTradeTriggerType.<name>.name}"
        # — a source-text drift guard cannot see an f-string write site, so
        # the pin is against the ENUM itself: rename a member and this test
        # names the break before the classifier silently stops matching.
        import main as m
        from defense.constitution import NoTradeTriggerType
        names = {t.name for t in NoTradeTriggerType}
        for t in m._SLEEVE_HOLD_NO_TRADE_TRIGGERS:
            assert t in names, (
                f"{t!r} is in _SLEEVE_HOLD_NO_TRADE_TRIGGERS but is not a "
                f"NoTradeTriggerType member — the hold classification is dead "
                f"and that trigger now FLATTENS the sleeve")

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

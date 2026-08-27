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

    def test_a_veto_that_did_NOT_zero_direction_still_opens_nothing(self):
        """Landmine 1, from a real emitted record:
        direction=-0.3327 target_exposure=0.2495 veto_active=True
        ([WEEKEND] alpha 10bps < min 20bps). Passing `direction` through would
        OPEN a short the gate just refused.

        [P416] classification moved: "[WEEKEND]" is an ENTRY-QUALITY veto
        (its write sites are entry gates), so from FLAT it still blocks the
        entry (0.0 -- the original landmine holds) but a HELD position now
        survives the weekend bar instead of paying a scheduled weekly
        round trip."""
        import main as _main
        d, why = translate(_intent(
            direction=-0.3327, target_exposure=0.2495, veto_active=True,
            veto_reason="[WEEKEND] [AP-5] Weekend alpha 10bps < min 20bps"), 0.0)
        assert d is _main.SLEEVE_ENTRY_BLOCKED, (
            "WEEKEND must classify entry-quality (P416)")
        assert why.startswith("entry_quality_veto:")
        # FLAT book: the refusal IS the effect -- nothing opens (landmine 1)
        rd, rwhy = _main.sleeve_entry_blocked_resolve(0.0, -0.3327, why)
        assert rd is _main.SLEEVE_HOLD and "entry_blocked_flat" in rwhy
        # HELD book, signal still agrees: the position SURVIVES the weekend
        # bar -- the pre-P416 flatten here was a scheduled weekly round trip
        hd, hwhy = _main.sleeve_entry_blocked_resolve(2.0, +0.5, why)
        assert hd is _main.SLEEVE_HOLD and "entry_blocked_maintain" in hwhy
        # HELD book, signal FLIPPED: demoted to flatten (closing leg is free)
        fd, fwhy = _main.sleeve_entry_blocked_resolve(2.0, -0.3327, why)
        assert fd == 0.0 and "entry_blocked_flip_to_flat" in fwhy

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


def _execution_service_stamped_reasons():
    """[P338] The reasons `execute_intent_v2` returns, as the SLEEVE SEES THEM.

    The P276 guard's corpus was main.py + integration_v36 + trade_gate, and
    P287 widened it to profit_max_adapter + auto_recovery_gate. It never
    included `core/execution_service.py` — the one file that MUTATES
    `intent.veto_reason` in-tick (line ~1080) and the source of 23 further
    reasons that reach the intent indirectly: main.py:~17053 stamps any
    non-benign execution result onto `intent.veto_reason`, prefixing
    "[EXECUTION] " when the reason is not already bracketed, and that is the
    same object `_live_intents` hands the translator later in the loop.

    So this models the ACTUAL path by which an execution reason becomes a
    veto, rather than only the `veto_reason = "..."` assignment form the
    original regex could see. The benign roster is IMPORTED from the producer
    (P172) — restating it here is how the two would drift.
    """
    import re
    from pathlib import Path
    from tests._source_scan import code_only
    from core.execution_service import BENIGN_EXEC_SKIP_REASONS

    repo = Path(__file__).resolve().parents[1]
    src = code_only(repo / "core" / "execution_service.py",
                    strip_docstrings=True)
    out = set()
    _pat = '"reason"\\s*:\\s*f?"([^"]{2,})"'
    for m in re.finditer(_pat, src):
        lit = m.group(1).split("{")[0].strip()
        if len(lit) < 4 or lit in BENIGN_EXEC_SKIP_REASONS:
            continue
        out.add(lit if lit.startswith("[") else f"[EXECUTION] {lit}")
    assert len(out) >= 15, (
        f"only {len(out)} execution reasons extracted — the regex stopped "
        f"matching execute_intent_v2's returns (a guard that reads nothing "
        f"passes on anything, P174)")
    return out


class TestVetoRosterCompleteness:
    """[P276] Every veto_reason WRITE SITE must be explicitly classified.

    The translator's default for an unclassified veto is veto_flat =
    liquidate every routed asset. P275 recorded the standing hazard: a NEW
    veto writer anywhere upstream silently inherits flatten semantics (the
    P239 guard below catches RENAMES of classified strings, not new
    writers). This guard closes that: extract every veto_reason string
    literal from the write-site corpus and require each to match one of the
    three classification tuples — a new writer goes RED here until its
    author decides HOLD vs FLATTEN (the P253/P265b analysis).

    Its very first enumeration found DISCONNECTED_MID_TICK: the second
    Kraken-disconnect door (main.py:~14747), unclassified since the
    cutover — a Kraken API disconnect liquidated the Coinbase book. Now in
    _SLEEVE_HOLD_VETOES.

    Known limitation (inherited from P265b): a veto composed ENTIRELY at
    runtime (no string literal prefix) is invisible to this scan — the
    NO_TRADE subtype family is the one known case and is enum-pinned
    separately below.
    """

    def _extract_literals(self):
        import re
        from pathlib import Path
        from tests._source_scan import code_only
        repo = Path(__file__).resolve().parents[1]
        src = code_only(repo / "main.py", strip_docstrings=True)
        src += code_only(repo / "integration" / "integration_v36.py",
                         strip_docstrings=True)
        src += code_only(repo / "defense" / "trade_gate.py",
                         strip_docstrings=True)
        # [P287] The guard's first blind spot, found live: ProfitMax's
        # [FALSE_BREAKOUT_VETO]/[LOSS_STREAK_HALT] literals live in
        # signals/profit_max_adapter.py and reach the intent through a
        # runtime-composed write in main.py — invisible to the original
        # three-file corpus, so an ACTIVE-by-default component's vetoes
        # inherited liquidate-the-sleeve semantics unclassified. The corpus
        # now spans every file that authors a veto_reason string the sleeve
        # can see. auto_recovery_gate.py included for its future writers
        # (its current [AUTO_RECOVERY_LATCH] example is docstring-only; the
        # live write site is main.py).
        src += code_only(repo / "signals" / "profit_max_adapter.py",
                         strip_docstrings=True)
        src += code_only(repo / "risk" / "auto_recovery_gate.py",
                         strip_docstrings=True)
        # [P338] The blind spot that let 23 reasons default to LIQUIDATE:
        # execute_intent_v2 mutates intent.veto_reason directly
        # (EXPOSURE_BELOW_MINIMUM_VIABLE) AND supplies every reason main.py
        # stamps onto the intent. Both forms are covered — this file for the
        # assignment form, _execution_service_stamped_reasons() below for the
        # returned-dict form.
        src += code_only(repo / "core" / "execution_service.py",
                         strip_docstrings=True)
        # excise the classification tuples themselves — a roster entry must
        # be justified by a REAL write site, not by its own definition
        src = re.sub(r"_SLEEVE_HOLD_VETOES\s*=\s*\([^)]*\)", "", src)
        src = re.sub(r"_SLEEVE_VENUE_NA_VETOES\s*=\s*\([^)]*\)", "", src)
        src = re.sub(r"_SLEEVE_FLATTEN_INTENDED_VETOES\s*=\s*\([^)]*\)", "",
                     src)
        lits = set()
        # [P382] `p0_abort_reason = "..."` is the SECOND assignment form that
        # reaches `intent.veto_reason` (main.py stamps it verbatim at the P0
        # abort return): "[INTEGRITY] ..." and f"CORRELATION_CRISIS: ..."
        # were runtime-composed and therefore invisible to the veto_reason-
        # only scan — P276's own recorded blind spot. Both prefixes are now
        # extracted, so their roster entries are justified by a REAL writer.
        for m in re.finditer(
                r'(?:veto_reason|p0_abort_reason)\s*=\s*\(?\s*f?"([^"\n]{2,})"',
                src):
            prefix = m.group(1).split("{")[0].strip()
            if len(prefix) >= 4:
                lits.add(prefix)
        assert len(lits) > 30, (
            f"only {len(lits)} veto literals extracted — the scan regex "
            f"stopped matching the corpus (a guard that reads nothing "
            f"passes on anything, P174)")
        # [P338] the returned-dict form, stamped exactly as main.py stamps it
        lits |= _execution_service_stamped_reasons()
        return lits

    def test_every_veto_write_site_is_classified(self):
        import main as m
        rosters = (tuple(m._SLEEVE_HOLD_VETOES)
                   + tuple(m._SLEEVE_VENUE_NA_VETOES)
                   + tuple(m._SLEEVE_FLATTEN_INTENDED_VETOES)
                   # [P341] the third disposition: entry refused, resolved
                   # against the book (flat/maintain -> hold, flip -> close)
                   + tuple(m._SLEEVE_ENTRY_QUALITY_VETOES))
        unclassified = []
        for lit in sorted(self._extract_literals()):
            if not any(r in lit or lit in r for r in rosters):
                unclassified.append(lit)
        assert not unclassified, (
            f"UNCLASSIFIED veto write site(s): {unclassified}. The sleeve "
            f"translator will FLATTEN every routed asset on these by "
            f"default. Decide per the P253/P265b test — does the veto mean "
            f"'state unknown / wait / smaller' (add to _SLEEVE_HOLD_VETOES) "
            f"or 'flat is the intended posture' (add to "
            f"_SLEEVE_FLATTEN_INTENDED_VETOES with a category comment)?")

    def test_flatten_roster_entries_have_live_write_sites(self):
        # the anti-rot mirror: a roster entry with no writer is a stale
        # classification that will silently absorb a future unrelated veto
        import main as m
        lits = self._extract_literals()
        dead = [r for r in m._SLEEVE_FLATTEN_INTENDED_VETOES
                if not any(r in lit or lit in r for lit in lits)]
        assert not dead, (
            f"flatten-roster entries with no live write site: {dead} — "
            f"remove them or they mask the next new writer that happens "
            f"to contain the stale string")

    def test_disconnect_mid_tick_holds(self):
        # the finding itself, pinned behaviorally: both Kraken-disconnect
        # doors must classify as HOLD
        import main as m
        for door in ("EXCHANGE_DISCONNECTED_HOLD", "DISCONNECTED_MID_TICK"):
            assert any(h in door or door in h
                       for h in m._SLEEVE_HOLD_VETOES), (
                f"{door} is not HOLD-classified — a KRAKEN API disconnect "
                f"would liquidate the COINBASE book (the P253 #1 class)")

    def test_short_control_protects_the_alpha_gate(self):
        # [P276/P275 #3] the override must never clear the economics veto
        from tests._source_scan import code_only
        from pathlib import Path
        import re
        repo = Path(__file__).resolve().parents[1]
        src = code_only(repo / "main.py", strip_docstrings=True)
        m = re.search(r"_SC_PROTECTED_VETOES\s*=\s*\{(.*?)\}", src, re.S)
        assert m and '"Alpha gate"' in m.group(1), (
            "the alpha-gate veto string left _SC_PROTECTED_VETOES — "
            "short_control's override could admit a short whose edge "
            "failed the gate (P275 finding #3)")


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
        # [P287] ProfitMax's veto literals live in signals/ (see the
        # completeness guard above); without this the new HOLD members
        # would read as having no write site.
        src += code_only(repo / "signals" / "profit_max_adapter.py",
                         strip_docstrings=True)
        src += code_only(repo / "risk" / "auto_recovery_gate.py",
                         strip_docstrings=True)
        # [P338] execute_intent_v2's reasons reach the intent via main.py's
        # stamping site, so its source is a real write-site corpus for every
        # roster member drawn from it.
        src += code_only(repo / "core" / "execution_service.py",
                         strip_docstrings=True)
        # Excise the definition statements themselves.
        src = re.sub(r"_SLEEVE_HOLD_VETOES\s*=\s*\([^)]*\)", "", src)
        src = re.sub(r"_SLEEVE_VENUE_NA_VETOES\s*=\s*\([^)]*\)", "", src)
        src = re.sub(r"_SLEEVE_ENTRY_QUALITY_VETOES\s*=\s*\([^)]*\)", "",
                     src)
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

    def test_every_entry_quality_veto_has_a_live_write_site(self):
        import main as m
        corpus = self._write_site_corpus()
        for member in m._SLEEVE_ENTRY_QUALITY_VETOES:
            assert member in corpus, (
                f"{member!r} is in _SLEEVE_ENTRY_QUALITY_VETOES but no write "
                f"site emits it -- a stale entry there is worse than in the "
                f"other rosters, because it silently absorbs the next veto "
                f"whose text happens to contain it and hands it "
                f"flip-to-flat semantics")

    def test_sets_are_nonempty(self):
        """The guards above iterate the sets — empty sets would pass
        vacuously (P174)."""
        import main as m
        assert len(m._SLEEVE_HOLD_VETOES) >= 2
        assert len(m._SLEEVE_VENUE_NA_VETOES) >= 1
        assert len(m._SLEEVE_ENTRY_QUALITY_VETOES) >= 3


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

    def test_correlation_collapse_holds_rather_than_liquidates(self):
        # [P382] MOVED from the flatten list above to HOLD, and the reason is
        # a premise correction, not a loosening preference: the LIVE checker
        # (defense/constitution.py ~:443) fires CORRELATION_COLLAPSE on
        # correlation_btc_eth_sol >= 0.92 ALONE. P253d armed the real
        # correlation believing it "ALSO requires all-three-same-direction
        # AND no validated edge" — those conjuncts live in
        # signals/no_trade_triggers.py, which is NOT the live path. By
        # P253d's own measurement the measure is >= 0.92 on 7.8% of bars
        # (17% in the last year), and it fired live 2026-08-19 16:02/20:02
        # on all three assets. A routine correlation reading must not
        # LIQUIDATE a held book (the P364 shape); NO_TRADE still refuses
        # NEW risk by vetoing the intent, and the venue stop + halts keep
        # guarding what is held.
        d, r = translate(
            _intent(veto_active=True,
                    veto_reason="[v3.6.1] NO_TRADE: CORRELATION_COLLAPSE"),
            fallback_dir=0.9)
        assert d is SLEEVE_HOLD
        assert "hold_veto" in r

    @pytest.mark.parametrize("trigger", [
        "FLASH_CRASH", "EXTREME_DVOL", "LIQUIDITY_CRITICAL",
        "ALL_CONFLICT_FLAT"])
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

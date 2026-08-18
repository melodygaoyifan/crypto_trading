"""
[P294] Seat controller — the replacement for the P237 tripwire's shape.

The tripwire is a kill switch: one variable, one threshold, one direction.
Its whole action space is "trade less", which cannot serve a goal of trading.
This controller asks the well-posed version — of the candidates that can hold
the DECIDE slot, which should? — and the tripwire becomes one outcome ("flat
wins") rather than the only one.

Also pins the two INSPECTION findings that the controller encodes:
  * regimebook/SOL is UNAVAILABLE (model deleted in P250), not "flat"
  * trend is a 3-lookback vote quantized to {+-1/3, +-1}
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analytics.seat.seat_controller import (  # noqa: E402
    Candidate, decide_seat, render, FLAT, SWITCH_MARGIN, MIN_SWITCH_T, MIN_N)


def _c(name, ic4, ic16, t4=3.0, t16=3.0, n=500, **kw):
    return Candidate(name=name, ic_4h=ic4, ic_16h=ic16, t_4h=t4, t_16h=t16,
                     n=n, **kw)


# =============================================================================
# Scoring
# =============================================================================

class TestScoring:
    def test_score_is_the_weakest_horizon(self):
        """A candidate must be non-negative on BOTH horizons. This system's
        signals have repeatedly looked fine at 4h and gone negative at 16h
        (P198, P293k), so the weakest horizon is the honest summary."""
        assert _c("x", 0.09, 0.01).score() == pytest.approx(0.01)
        assert _c("x", 0.01, 0.09).score() == pytest.approx(0.01)
        assert _c("x", 0.09, -0.02).score() == pytest.approx(-0.02)

    def test_thin_evidence_scores_None_not_zero(self):
        """'not enough evidence' must never read as 'measured flat' (P199)."""
        assert _c("x", 0.05, 0.05, n=MIN_N - 1).score() is None
        assert _c("x", 0.05, 0.05, n=MIN_N).score() is not None

    def test_missing_horizon_scores_None(self):
        assert _c("x", 0.05, None).score() is None
        assert _c("x", None, 0.05).score() is None

    def test_unavailable_scores_None_not_flat(self):
        """[P294] The SOL regimebook case: its bear-leg model was DELETED in
        P250, so it emits flat forever. Scoring that as a flat OPINION would
        let a structurally broken candidate win by default (P2)."""
        c = _c("regimebook", 0.5, 0.5, available=False, note="model deleted")
        assert c.score() is None
        assert any("UNAVAILABLE" in r for r in c.reasons())

    def test_decisive_t_follows_the_weakest_horizon(self):
        assert _c("x", 0.09, 0.01, t4=5.0, t16=1.1).decisive_t() == pytest.approx(1.1)
        assert _c("x", 0.01, 0.09, t4=1.2, t16=5.0).decisive_t() == pytest.approx(1.2)


# =============================================================================
# The decision
# =============================================================================

class TestDecision:
    def test_flat_wins_when_nothing_is_positive(self):
        """This IS the tripwire's verdict — reached by comparison rather than
        by a single-variable threshold."""
        d = decide_seat([_c("trend", -0.01, -0.05), _c("whale", 0.02, -0.01)],
                        incumbent="trend")
        assert d.winner == FLAT and d.switch is True
        assert "flat wins" in d.reason

    def test_already_flat_does_not_thrash(self):
        d = decide_seat([_c("trend", -0.01, -0.05)], incumbent=FLAT)
        assert d.winner == FLAT and d.switch is False

    def test_incumbent_best_holds(self):
        d = decide_seat([_c("whale", 0.05, 0.04), _c("trend", 0.01, 0.01)],
                        incumbent="whale")
        assert d.winner == "whale" and d.switch is False

    def test_hysteresis_blocks_a_marginal_challenger(self):
        """Without a margin the seat thrashes on noise — the same reason
        flip-persistence exists on the order path."""
        d = decide_seat([_c("trend", 0.030, 0.030), _c("whale", 0.040, 0.035)],
                        incumbent="trend")
        assert d.switch is False
        assert "margin" in d.reason

    def test_margin_is_actually_binding(self):
        """A challenger just OVER the margin must switch — otherwise the
        hysteresis test above would pass for the wrong reason."""
        over = 0.030 + SWITCH_MARGIN + 0.001
        d = decide_seat([_c("trend", 0.030, 0.030), _c("whale", over, over)],
                        incumbent="trend")
        assert d.switch is True and d.winner == "whale"

    def test_significance_floor_blocks_a_noisy_point_estimate(self):
        """Every IC in this system sits inside noise; without this the
        controller would chase random ordering every week."""
        d = decide_seat([_c("trend", 0.01, 0.01),
                         _c("whale", 0.09, 0.08, t4=0.4, t16=0.3)],
                        incumbent="trend")
        assert d.switch is False
        assert "|t|" in d.reason

    def test_a_genuine_switch(self):
        d = decide_seat([_c("trend", 0.010, 0.005, t4=0.5, t16=0.2),
                         _c("whale", 0.090, 0.080, t4=2.6, t16=2.3)],
                        incumbent="trend")
        assert d.switch is True and d.winner == "whale"

    def test_unscoreable_incumbent_is_not_unbeatable(self):
        """An incumbent whose evidence is missing must not keep the seat by
        default — it is compared at 0.0."""
        d = decide_seat([_c("trend", None, None),
                         _c("whale", 0.09, 0.08, t4=2.6, t16=2.3)],
                        incumbent="trend")
        assert d.switch is True and d.winner == "whale"

    def test_unavailable_candidate_can_never_win(self):
        d = decide_seat([_c("regimebook", 0.9, 0.9, available=False),
                         _c("whale", 0.01, 0.01)],
                        incumbent="whale")
        assert d.winner == "whale"
        assert d.scores["regimebook"] is None

    def test_reproduces_the_live_p293j_decision(self):
        """Real 90d numbers: whale is the only candidate positive on BOTH
        horizons, which is the decision made by hand in P293j."""
        d = decide_seat([
            _c("trend", 0.007, -0.046, t4=0.27, t16=-0.89, n=1483),
            _c("whale", 0.040, 0.011, t4=0.98, t16=0.14, n=605),
            _c("regimebook", None, None, n=71, available=True),
        ], incumbent="whale")
        assert d.winner == "whale" and d.switch is False
        assert d.scores["trend"] == pytest.approx(-0.046)


# =============================================================================
# It must never edit config
# =============================================================================

class TestNeverEditsConfig:
    def test_module_has_no_write_calls(self):
        """P141: changing what drives live money stays a human step. The
        controller prints the edit and exits with a code."""
        src = (REPO / "analytics" / "seat" / "seat_controller.py").read_text(
            encoding="utf-8-sig")
        for forbidden in ("write_text", "json.dump(", "open(", "os.replace"):
            assert forbidden not in src, f"controller must not write: {forbidden}"

    def test_cli_writes_nothing_to_config(self):
        src = (REPO / "scripts" / "seat_check.py").read_text(encoding="utf-8-sig")
        assert "write_text" not in src and "json.dump(" not in src

    def test_every_winner_has_a_stated_config_edit(self):
        from analytics.seat.seat_controller import SEAT_CONFIG_EDIT
        for seat in ("trend", "whale", "regimebook", FLAT):
            assert seat in SEAT_CONFIG_EDIT and SEAT_CONFIG_EDIT[seat]


# =============================================================================
# CLI contract
# =============================================================================

def _run(*args):
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(REPO / "scripts" / "seat_check.py"),
         *args], capture_output=True, text=True, cwd=str(REPO),
        timeout=120, encoding="utf-8")


class TestCli:
    def test_switch_exits_3(self):
        r = _run("--incumbent", "trend", "--stats",
                 json.dumps({"trend": {"ic_4h": -0.01, "ic_16h": -0.05,
                                       "t_4h": -0.3, "t_16h": -1.2, "n": 900}}))
        assert r.returncode == 3, r.stdout + r.stderr

    def test_hold_exits_0(self):
        r = _run("--incumbent", "whale", "--stats",
                 json.dumps({"whale": {"ic_4h": 0.04, "ic_16h": 0.011,
                                       "t_4h": 0.98, "t_16h": 0.14, "n": 605}}))
        assert r.returncode == 0, r.stdout + r.stderr

    @pytest.mark.parametrize("args", [
        ("--incumbent", "trend", "--stats", "not json"),
        ("--incumbent", "trend"),                       # no evidence source
        ("--incumbent", "trend", "--ic-report", "/no/such/report.json"),
    ])
    def test_missing_or_bad_evidence_refuses(self, args):
        """A recommendation from missing evidence is a guess wearing a
        measurement's name (P199)."""
        r = _run(*args)
        assert r.returncode == 2
        assert "REFUSING" in r.stderr

    def test_incumbent_is_read_from_the_live_config(self):
        """Seat precedence must mirror main.py's ordering (last seat wins:
        whale > regimebook > trend)."""
        r = _run("--stats", json.dumps(
            {"whale": {"ic_4h": 0.04, "ic_16h": 0.011,
                       "t_4h": 0.98, "t_16h": 0.14, "n": 605}}))
        assert r.returncode == 0
        assert re.search(r"incumbent\s*:\s*whale", r.stdout), r.stdout

    def test_caveats_are_surfaced_with_the_winner(self):
        r = _run("--incumbent", "whale", "--stats", json.dumps(
            {"whale": {"ic_4h": 0.04, "ic_16h": 0.011,
                       "t_4h": 0.98, "t_16h": 0.14, "n": 605}}))
        assert "CAVEAT on 'whale'" in r.stdout
        assert "BINARY" in r.stdout, (
            "the winner's structural caveat must travel with the "
            "recommendation, not live only in a doc"
        )

    def test_output_refuses_to_claim_profitability(self):
        r = _run("--incumbent", "whale", "--stats", json.dumps(
            {"whale": {"ic_4h": 0.04, "ic_16h": 0.011,
                       "t_4h": 0.98, "t_16h": 0.14, "n": 605}}))
        assert "NOT a claim of profitability" in r.stdout


# =============================================================================
# The inspection findings this controller encodes
# =============================================================================

class TestInspectionFindings:
    def test_trend_signal_is_quantized_to_three_votes(self):
        """[P294 inspection] `sig` is a mean of sign(momentum) over 3
        lookbacks, so it can ONLY be +-1/3 or +-1 — and 40*|sig|*0.75 maps
        those to 10bps (below every live threshold) or 30bps. Trend therefore
        trades only on unanimity, which is the mechanism behind 'the system
        does no trade'."""
        from strategies.trend_following import TrendFollowingStrategy
        s = TrendFollowingStrategy()
        assert len(s.lookbacks) == 3, (
            "the reachable signal set depends on the lookback COUNT; if this "
            "changes, the 10bps/30bps arithmetic changes with it"
        )
        reachable = {round(v / 3, 4) for v in (-3, -1, 1, 3)}
        assert reachable == {-1.0, -0.3333, 0.3333, 1.0}

    def test_min_abs_signal_can_never_fire(self):
        """[P294 inspection] `min_abs_signal=0.30` gates |sig| < 0.30, but the
        smallest reachable |sig| is 1/3 = 0.3333. The 'weak trend -> flat'
        branch is unreachable — a P174-class control that cannot act."""
        from core.trend_decision_layer import TrendDecisionLayer
        import inspect as _i
        sig = _i.signature(TrendDecisionLayer.__init__)
        floor = sig.parameters["min_abs_signal"].default
        assert floor == pytest.approx(0.30)
        assert floor < 1.0 / 3.0, (
            "if the floor ever rises above 1/3 it starts zeroing the 2-of-3 "
            "vote, which is a live behaviour change, not a tidy-up"
        )

    def test_sol_regimebook_has_no_model(self):
        """[P294 inspection] P250 deleted configs/regimebook/SOL_bear_ridge.json
        (it was the leak artifact), so SOL's book is structurally inert and
        must be treated as UNAVAILABLE rather than as a flat opinion."""
        assert not (REPO / "configs" / "regimebook" / "SOL_bear_ridge.json").exists()
        src = (REPO / "defense" / "regime_book_shadow.py").read_text(
            encoding="utf-8-sig")
        assert "v1_degraded_no_bear_leg" in src

    def test_cli_encodes_the_regimebook_and_trend_caveats(self):
        src = (REPO / "scripts" / "seat_check.py").read_text(encoding="utf-8-sig")
        assert "UNCERTIFIED" in src, "the P262 BTC funding-leg caveat"
        assert "P250" in src, "SOL's deleted model"
        assert "unanimity" in src, "trend's quantization"


# =============================================================================
# [P295] The three fixes
# =============================================================================

class TestWeakVoteFloorNowActs:
    def test_unreachable_floor_warns(self, caplog):
        """The control could not act for the life of the class. It now says
        so instead of waiting to be rediscovered."""
        import logging
        from core.trend_decision_layer import TrendDecisionLayer
        with caplog.at_level(logging.WARNING):
            TrendDecisionLayer(mode="shadow", min_abs_signal=0.30)
        assert any("UNREACHABLE" in r.message for r in caplog.records), caplog.text

    def test_reachable_floor_is_silent(self, caplog):
        import logging
        from core.trend_decision_layer import TrendDecisionLayer
        with caplog.at_level(logging.WARNING):
            TrendDecisionLayer(mode="shadow", min_abs_signal=0.50)
        assert not any("UNREACHABLE" in r.message for r in caplog.records)

    def test_floor_sits_between_the_two_reachable_values(self):
        """0.50 zeroes the 2-of-3 vote and keeps the unanimous one. Any value
        <= 1/3 is dead; any value > 1 would zero everything."""
        from main import ProductionConfig
        import tempfile, json as _j
        from pathlib import Path as _P
        with tempfile.TemporaryDirectory() as d:
            p = _P(d) / "c.json"
            p.write_text("{}", encoding="utf-8")
            assert ProductionConfig.from_file(p).trend_min_abs_signal == pytest.approx(0.30), (
                "absent key must preserve the historical default"
            )
        prof = _j.loads((REPO / "configs" / "live_high_risk.json").read_text(
            encoding="utf-8-sig"))
        live = prof["trend_min_abs_signal"]
        assert 1.0 / 3.0 < live <= 1.0, (
            f"live floor {live} must sit strictly above 1/3 (else dead) and "
            f"at or below 1.0 (else it zeroes even a unanimous vote)"
        )

    def test_did_not_raise_the_asserted_alpha_instead(self):
        """The rejected fix: making 2-of-3 clear the bar by asserting more
        alpha would be manufacturing tradeability on a signal whose realized
        slope measured -0.74..+3.01 with every |t| < 0.8 (P293k) — the P231
        error. base_edge_bps must be untouched."""
        import inspect as _i
        from core.trend_decision_layer import TrendDecisionLayer
        sig = _i.signature(TrendDecisionLayer.__init__)
        assert sig.parameters["base_edge_bps"].default == pytest.approx(40.0)
        prof = json.loads((REPO / "configs" / "live_high_risk.json").read_text(
            encoding="utf-8-sig"))
        assert "trend_base_edge_bps" not in prof

    def test_layer_is_constructed_with_the_configured_floor(self):
        """P201/P234: a config key read by nobody does nothing. Pin the WIRING."""
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        assert 'self.config, "trend_min_abs_signal", 0.30' in src
        assert "min_abs_signal=float(getattr(" in src

    def test_the_factory_actually_accepts_the_kwarg(self):
        """[P295b] THE BUG THIS FILE MISSED, now covered behaviourally.

        The source-string pin above proves the argument is PASSED; it says
        nothing about whether the callee accepts it. It did not: main.py
        passed `min_abs_signal` to the singleton FACTORY, which had no such
        parameter, so every tick raised TypeError, the handler logged
        "[TREND-LAYER] process skip", and the trend layer produced NOTHING.

        That is P234 verbatim — "a test that asserts a substring exists proves
        the code was written, not that it runs" — committed by the author of
        that lesson's own citation. This test CALLS the factory.
        """
        import core.trend_decision_layer as tdl
        tdl._singleton = None
        try:
            layer = tdl.get_trend_decision_layer(
                mode="shadow", regime_gate_mode="shadow", min_abs_signal=0.50)
            assert layer.min_abs_signal == pytest.approx(0.50)
            # and it must reach an ALREADY-BUILT singleton, or a config change
            # applies only on a cold process
            again = tdl.get_trend_decision_layer(
                mode="shadow", regime_gate_mode="shadow", min_abs_signal=0.60)
            assert again is layer
            assert again.min_abs_signal == pytest.approx(0.60)
            # omitting it must not disturb the existing value
            same = tdl.get_trend_decision_layer(mode="shadow")
            assert same.min_abs_signal == pytest.approx(0.60)
        finally:
            tdl._singleton = None

    def test_main_calls_the_factory_with_a_signature_that_accepts_it(self):
        """Belt-and-braces on the same class of defect: whatever main.py
        passes must exist in the factory's signature."""
        import inspect as _i
        import re as _re
        from core.trend_decision_layer import get_trend_decision_layer
        params = set(_i.signature(get_trend_decision_layer).parameters)
        src = (REPO / "main.py").read_text(encoding="utf-8-sig")
        m = _re.search(r"get_trend_decision_layer\((.*?)\)\.process", src, _re.S)
        assert m, "the trend factory call site was not found"
        passed = set(_re.findall(r"(\w+)\s*=", m.group(1)))
        # only keyword args at the call site count; drop nested kwargs
        unknown = {k for k in passed if k in
                   {"mode", "regime_gate_mode", "min_abs_signal"}} - params
        assert not unknown, f"main.py passes kwargs the factory rejects: {unknown}"


class TestRegimebookAvailability:
    def test_degraded_versions_report_unavailable(self):
        """SOL's `direction: 0.0` and ETH's `direction: 0.0` mean completely
        different things; a consumer reading only `direction` cannot tell a
        broken book from a book that is correctly flat (P2)."""
        src = (REPO / "defense" / "regime_book_shadow.py").read_text(
            encoding="utf-8-sig")
        assert '"available": not str(version).startswith("v1_degraded")' in src

    def test_sol_version_is_a_degraded_one(self):
        from defense.regime_book_shadow import BOOKS_VERSION
        assert BOOKS_VERSION["SOL"].startswith("v1_degraded")
        assert not BOOKS_VERSION["BTC"].startswith("v1_degraded")
        assert not BOOKS_VERSION["ETH"].startswith("v1_degraded")

    def test_seat_controller_reads_the_producers_own_flag(self, tmp_path):
        """Not a hand-maintained list here, which would drift the moment a
        model is restored."""
        sys.path.insert(0, str(REPO / "scripts"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "seat_check", REPO / "scripts" / "seat_check.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        (tmp_path / "regimebook_SOL.jsonl").write_text(
            json.dumps({"available": False, "direction": 0.0}) + "\n",
            encoding="utf-8")
        (tmp_path / "regimebook_ETH.jsonl").write_text(
            json.dumps({"available": True, "direction": 0.0}) + "\n",
            encoding="utf-8")
        assert mod.availability_from_ledger(tmp_path, "SOL") is False
        assert mod.availability_from_ledger(tmp_path, "ETH") is True

    def test_missing_or_legacy_rows_are_unknown_not_available(self, tmp_path):
        """A row predating the flag must read as UNKNOWN — treating it as
        available would let a broken book back into contention silently."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "seat_check2", REPO / "scripts" / "seat_check.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.availability_from_ledger(tmp_path, "NOPE") is None
        (tmp_path / "regimebook_OLD.jsonl").write_text(
            json.dumps({"direction": 0.0}) + "\n", encoding="utf-8")
        assert mod.availability_from_ledger(tmp_path, "OLD") is None


# =============================================================================
# [P295c] The two defects the first LIVE run exposed
# =============================================================================

class TestP295cRefusalIsNotAFlatVerdict:
    """"Nothing measurable" must never be recommended as "go flat".

    The first live run recommended vacating every seat on n=0 evidence — the
    P199 shape (no data reading as a verdict), produced by this tool's own
    parser bug. Flat may only win by COMPARISON.
    """

    def test_no_scoreable_candidate_refuses_and_recommends_nothing(self):
        from analytics.seat.seat_controller import Candidate, decide_seat
        # Exactly the live failure: both candidates below the evidence floor.
        cands = [Candidate(name="trend", n=0), Candidate(name="whale", n=0)]
        d = decide_seat(cands, incumbent="whale")
        assert d.refused is True
        assert d.switch is False, "a refusal must never recommend an action"
        assert d.winner == "whale", "the incumbent holds through a refusal"
        assert "REFUSING" in d.reason

    def test_flat_still_wins_when_candidates_are_measured_and_nonpositive(self):
        """The live numbers: both scoreable, both negative -> flat, NOT refusal."""
        from analytics.seat.seat_controller import Candidate, decide_seat, FLAT
        cands = [
            Candidate(name="trend", ic_4h=-0.0326, ic_16h=-0.086,
                      t_4h=-0.87, t_16h=-1.14, n=704),
            Candidate(name="whale", ic_4h=-0.0698, ic_16h=-0.0651,
                      t_4h=-1.05, t_16h=-0.48, n=225),
        ]
        d = decide_seat(cands, incumbent="whale")
        assert d.refused is False, "measured evidence is not a refusal"
        assert d.winner == FLAT and d.switch is True

    def test_one_scoreable_candidate_is_enough_to_reach_a_verdict(self):
        from analytics.seat.seat_controller import Candidate, decide_seat, FLAT
        cands = [Candidate(name="trend", ic_4h=-0.05, ic_16h=-0.05,
                           t_4h=-1.0, t_16h=-1.0, n=700),
                 Candidate(name="whale", n=0)]
        d = decide_seat(cands, incumbent="trend")
        assert d.refused is False and d.winner == FLAT

    def test_render_of_a_refusal_prints_no_config_edit(self):
        from analytics.seat.seat_controller import Candidate, decide_seat, render
        txt = render(decide_seat([Candidate(name="trend", n=0)], incumbent="trend"))
        assert "CONFIG EDIT IMPLIED" not in txt
        assert "NO RECOMMENDATION" in txt


class TestP295cReportParserMatchesTheRealShape:
    """The parser read agents.<a>.<h>; the emitter writes agents.<a>.horizons.<h>."""

    # Verbatim excerpt of the live 2026-08-17 report on the data volume.
    REAL = {
        "generated": "2026-08-17T06:10:02+00:00",
        "window_days": 30,
        "agents": {
            "quant": {"horizons": {
                "1": {"n": 713, "ic": -0.0326, "t": -0.87},
                "4": {"n": 704, "ic": -0.086, "t": -1.14}},
                "verdict": "HOLD"},
            "whale": {"horizons": {
                "1": {"n": 226, "ic": -0.0698, "t": -1.05},
                "4": {"n": 225, "ic": -0.0651, "t": -0.48}},
                "verdict": "HOLD"},
        },
    }

    def _write(self, tmp_path, payload):
        import json
        p = tmp_path / "agent_ic.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    def test_the_real_nested_shape_yields_real_n(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "seat_check_p295c", REPO / "scripts" / "seat_check.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        parsed = mod._from_ic_report(self._write(tmp_path, self.REAL))
        assert parsed["quant"][4] == (-0.086, -1.14, 704), (
            "the live report's cells must parse; n=0 here is the bug that made "
            "the controller recommend flat from nothing")
        assert parsed["whale"][1] == (-0.0698, -1.05, 226)

    def test_end_to_end_on_the_real_shape_reaches_a_measured_verdict(self, tmp_path):
        """The whole CLI, on the real report, must NOT exit 2."""
        import subprocess, sys as _s
        p = self._write(tmp_path, self.REAL)
        r = subprocess.run(
            [_s.executable, "-X", "utf8", str(REPO / "scripts" / "seat_check.py"),
             "--ic-report", str(p), "--incumbent", "whale"],
            capture_output=True, text=True, encoding="utf-8", cwd=str(REPO))
        assert r.returncode == 3, (
            f"expected a measured SWITCH-to-flat verdict, got rc={r.returncode}\n"
            f"{r.stdout}\n{r.stderr}")
        assert "n=0" not in r.stdout

    def test_the_flat_shape_still_parses(self, tmp_path):
        """Hand-built stats files must keep working."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "seat_check_p295c_flat", REPO / "scripts" / "seat_check.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        parsed = mod._from_ic_report(self._write(
            tmp_path, {"agents": {"quant": {"1": {"n": 99, "ic": 0.1, "t": 2.0}}}}))
        assert parsed["quant"][1] == (0.1, 2.0, 99)

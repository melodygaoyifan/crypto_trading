"""[P287] Evidence-path fix batch — the September decision instruments.

Pins, per finding of the 2026-08-16 read-through:
  1. mlp_seat_check reads the scorer's SUCCESS key (`ic_per_horizon`) — the
     pre-P287 `ic_per_h` read matched only the error record, so the 16h
     kill-screen was structurally UNEVALUATED (P174-class check that could
     not fire, P2 reader/writer mismatch). Both directions (pass AND kill)
     must be reachable, and the WIRING is pinned (P234 lesson: a pin on
     decide() alone proves nothing about the key path).
  2. mlp_seat_check defaults = the *_pulled dirs september_check populates;
     missing dirs refuse naming september_check (P255/P199).
  3. promotion_plan: ARCHIVE gets the same >=30d window guard as PROMOTE —
     a KILL off a 10-day trajectory read is informational, never archival.
  4. maker_fill_review counts the error-path taker legs and refuses on an
     unreadable log instead of reading it as "zero maker attempts".
  5. september_check's countdown covers every accruing candidate, and the
     tripwire's FIRED exit code (3) survives into the script's own rc.
  6. slope_calibrator stamps threshold provenance and warns on staleness.
  7. regimebook ledger rows carry a real per-bar funding carry (P245
     convention: stored 8h event rate / 2) when history is fresh.
  8. mlp_shadow's held position persists across restarts; cold-start rows
     are stamped restart_transient instead of claiming flat as a position.
  9. trend_regime_review counts price-unjoinable records and labels the
     effective joined window.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. mlp_seat_check — the IC key
# ---------------------------------------------------------------------------

class TestMlpSeatCheckIcKey:
    @pytest.fixture(scope="class")
    def msc(self):
        return _load_script("mlp_seat_check")

    def test_success_shape_is_read(self, msc):
        # the scorer's SUCCESS record (compute_shadow_ic.py:478) — the shape
        # the pre-P287 code could never read
        st = {"ic_per_horizon": {4: 0.12}, "n_per_horizon": {4: 100}}
        assert msc._ic_from_stats(st, 4) == pytest.approx(0.12)

    def test_negative_ic_reaches_the_kill_screen(self, msc):
        st = {"ic_per_horizon": {4: -0.08}}
        ic16 = msc._ic_from_stats(st, 4)
        fire, conds = msc.decide(date(2026, 8, 28), 20.0, 60, ic16, True)
        assert fire is False
        ic_cond = next(c for c in conds if c[0] == "ic16_not_negative")
        assert ic_cond[1] is False
        assert "UNEVALUATED" not in ic_cond[2]  # it EVALUATED and killed

    def test_nonnegative_ic_passes_the_screen(self, msc):
        st = {"ic_per_horizon": {4: 0.0}}
        fire, _ = msc.decide(date(2026, 8, 28), 20.0, 60,
                             msc._ic_from_stats(st, 4), True)
        assert fire is True

    def test_legacy_error_shape_still_read(self, msc):
        assert msc._ic_from_stats({"ic_per_h": {4: 0.1}}, 4) == pytest.approx(0.1)

    def test_string_keys_from_json_round_trip(self, msc):
        assert msc._ic_from_stats({"ic_per_horizon": {"4": -0.2}}, 4) == pytest.approx(-0.2)

    def test_absent_or_null_stays_unevaluated(self, msc):
        assert msc._ic_from_stats({}, 4) is None
        assert msc._ic_from_stats({"ic_per_horizon": {}}, 4) is None
        assert msc._ic_from_stats({"ic_per_horizon": {4: None}}, 4) is None
        # and decide() treats None as failing (P199), never as passing
        fire, conds = msc.decide(date(2026, 8, 28), 20.0, 60, None, True)
        assert fire is False

    def test_main_is_wired_through_the_helper(self):
        # P234 lesson: pin the WIRING, not just the pure function. The old
        # dead read must be gone and the helper must be the main-path read.
        src = (REPO / "scripts" / "mlp_seat_check.py").read_text(encoding="utf-8")
        assert 'st.get("ic_per_h", {}).get(HORIZON_BARS)' not in src
        assert "_ic_from_stats(st, HORIZON_BARS)" in src


# ---------------------------------------------------------------------------
# 2. mlp_seat_check — default dirs + refusals
# ---------------------------------------------------------------------------

class TestMlpSeatCheckDirs:
    def test_defaults_are_the_pulled_dirs(self):
        msc = _load_script("mlp_seat_check")
        assert msc.DEFAULT_LEDGER_DIR.name == "strategy_shadow_pulled"
        assert msc.DEFAULT_REPORTS_DIR.name == "evidence_reports_pulled"

    def test_argparse_uses_the_constants(self):
        src = (REPO / "scripts" / "mlp_seat_check.py").read_text(encoding="utf-8")
        assert "default=str(DEFAULT_LEDGER_DIR)" in src
        assert "default=str(DEFAULT_REPORTS_DIR)" in src
        # the old local-residue defaults must be gone from the parser
        assert 'default=str(REPO / "data" / "strategy_shadow")' not in src

    def test_missing_pulled_dirs_refuse_naming_september_check(self):
        src = (REPO / "scripts" / "mlp_seat_check.py").read_text(encoding="utf-8")
        assert "september_check.py first" in src

    def test_suspension_gate_is_still_first_and_binding(self):
        # The P285b/c suspension must remain the binding refusal — running
        # the real script against the real repo reports (FRAGILE probe, no
        # ensemble pass) exits 2 SUSPENDED before any dir is touched.
        out = subprocess.run(
            [sys.executable, "-X", "utf8",
             str(REPO / "scripts" / "mlp_seat_check.py"),
             "--today", "2026-08-28"],
            capture_output=True, text=True, cwd=str(REPO), timeout=120)
        assert out.returncode == 2
        assert "SUSPENDED" in out.stderr


# ---------------------------------------------------------------------------
# 3. promotion_plan — the ARCHIVE window guard
# ---------------------------------------------------------------------------

class TestPromotionPlanWindowGuard:
    @pytest.fixture(scope="class")
    def pp(self):
        from analytics.promotion_gate import promotion_plan
        return promotion_plan

    def test_short_window_kill_becomes_hold(self, pp):
        action, reason = pp.decide_strategy_action({"verdict": "KILL"}, 10)
        assert action is pp.StrategyAction.HOLD_SHADOW
        assert "informational" in reason

    def test_full_window_kill_still_archives(self, pp):
        action, _ = pp.decide_strategy_action({"verdict": "KILL"}, 30)
        assert action is pp.StrategyAction.ARCHIVE

    def test_absent_window_never_archives(self, pp):
        action, _ = pp.decide_strategy_action({"verdict": "KILL"}, 0)
        assert action is pp.StrategyAction.HOLD_SHADOW

    def test_promote_guard_unchanged(self, pp):
        a10, _ = pp.decide_strategy_action({"verdict": "PROMOTE"}, 10)
        a30, _ = pp.decide_strategy_action({"verdict": "PROMOTE"}, 30)
        assert a10 is pp.StrategyAction.HOLD_SHADOW
        assert a30 is pp.StrategyAction.PROMOTE_TO_FUSION

    def test_end_to_end_through_a_ten_day_report(self, pp):
        # the exact scenario: september_check's Monday 10-day trajectory
        # report, newest by mtime, holding an aggressive short-window KILL
        report = {"window_days": 10, "per_strategy": [
            {"strategy": "regimebook", "asset": "BTC", "verdict": "KILL"}]}
        actions = pp.build_strategy_actions(report)
        assert actions[0]["action"] == pp.StrategyAction.HOLD_SHADOW.value


# ---------------------------------------------------------------------------
# 4. maker_fill_review — error-path taker legs + refusal semantics
# ---------------------------------------------------------------------------

def _maker_lines(maker=10, timeout=5, immediate=3, error=4, other=1,
                 cancel_failed=1, no_id=1):
    L = []
    L += ["[COINBASE-MAKER] BTC: post-only left the book within the window "
          "(filled at 0bps maker)"] * maker
    L += ["[COINBASE-MAKER] BTC: unfilled after 45s — cancelled, crossing "
          "the remainder"] * timeout
    L += ["[COINBASE-MAKER] BTC: post-only rejected "
          "(PREVIEW_INVALID_LIMIT_PRICE_POST_ONLY) — taker fallback"] * immediate
    L += ["[COINBASE-MAKER] BTC: attempt error (RuntimeError: boom) — "
          "taker fallback"] * error
    L += ["[COINBASE-MAKER] BTC: no best_bid_ask (TimeoutError) — "
          "taker fallback"] * other
    L += ["[COINBASE-MAKER] BTC: timeout AND cancel FAILED — order x may be "
          "live; refusing to place the cross (double-order risk, P265)."] * cancel_failed
    L += ["[COINBASE-MAKER] BTC: accepted post-only carried no order_id — "
          "resolving via reconcile only"] * no_id
    return L


class TestMakerFillReview:
    def _run(self, tmp_path, lines):
        f = tmp_path / "log.txt"
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return subprocess.run(
            [sys.executable, "-X", "utf8",
             str(REPO / "scripts" / "maker_fill_review.py"),
             "--log-file", str(f)],
            capture_output=True, text=True, cwd=str(REPO), timeout=120)

    def test_error_path_legs_are_counted_as_taker(self, tmp_path):
        # DESIGNED so the verdict is only reachable BECAUSE the error-path
        # legs count: without them n = 18 < MIN_N and the run refuses —
        # i.e. this test is red under the pre-P287 patterns by construction.
        out = self._run(tmp_path, _maker_lines())
        assert out.returncode == 0, out.stdout + out.stderr
        assert "error-path 4" in out.stdout
        assert "taker-fallback=13" in out.stdout
        # leg = (10*0.5 + 13*3.0)/23 = 1.913; RT = 3.826 > 3.0
        assert "NOT unlocked" in out.stdout

    def test_ordering_immediate_not_swallowed_by_generic_fallback(self, tmp_path):
        # "post-only rejected ... — taker fallback" must classify as
        # immediate, not as the generic catch-all
        out = self._run(tmp_path, _maker_lines())
        assert "immediate 3" in out.stdout
        assert "other 1" in out.stdout

    def test_undercount_caveat_is_printed(self, tmp_path):
        out = self._run(tmp_path, _maker_lines())
        assert "UNDERCOUNT" in out.stdout

    def test_below_min_n_still_refuses(self, tmp_path):
        out = self._run(tmp_path, _maker_lines(maker=2, timeout=1, immediate=0,
                                               error=1, other=0,
                                               cancel_failed=0, no_id=0))
        assert out.returncode == 2
        assert "REFUSING a verdict" in out.stdout

    def test_unreadable_local_log_refuses(self, tmp_path):
        out = subprocess.run(
            [sys.executable, "-X", "utf8",
             str(REPO / "scripts" / "maker_fill_review.py"),
             "--log-file", str(tmp_path / "nope.txt")],
            capture_output=True, text=True, cwd=str(REPO), timeout=120)
        assert out.returncode == 2
        assert "REFUSING" in out.stdout

    def test_remote_grep_error_is_distinct_from_no_matches(self):
        # the || true conflation must be gone: grep rc>=2 maps to a distinct
        # remote exit the script refuses on, rc 1 (no matches) exits 0
        # P177 lesson: assert the COMMAND form, not the bare substring — the
        # fix's own comment legitimately quotes the retired `|| true`.
        src = (REPO / "scripts" / "maker_fill_review.py").read_text(encoding="utf-8")
        assert "hmats.log || true" not in src
        assert "if [ $ec -ge 2 ]; then exit 4; fi" in src
        assert "returncode == 4" in src

    def test_patterns_match_the_real_emitters(self):
        # drift guard: every load-bearing literal the patterns key on must
        # still exist in the sleeve's emitter source
        sleeve = (REPO / "exchange" / "coinbase_sleeve.py").read_text(encoding="utf-8")
        for lit in ("post-only left ", "crossing the remainder",
                    "post-only rejected", "attempt error ",
                    "taker fallback", "timeout AND cancel",
                    "carried no order_id"):
            assert lit in sleeve, f"emitter literal {lit!r} gone — patterns drift"


# ---------------------------------------------------------------------------
# 5. september_check — roster + exit codes
# ---------------------------------------------------------------------------

class TestSeptemberCheck:
    @pytest.fixture(scope="class")
    def sc(self):
        return _load_script("september_check")

    def test_countdown_covers_every_accruing_candidate(self, sc):
        for name in ("stablecoinflow", "oidiv_confirm", "oidiv_fade",
                     "calbasis", "xsmom", "eventfilter", "mlpshadow"):
            assert name in sc.CANDIDATES, f"{name} missing from countdown"
            assert sc.CANDIDATES[name][1] == "2026-08-16"

    def test_prior_roster_untouched(self, sc):
        for name in ("regimebook", "regimebook_adj", "derivflow", "ma_filter",
                     "volskip", "etfflow", "breadth books"):
            assert name in sc.CANDIDATES

    def test_tripwire_fired_outranks_everything(self, sc):
        assert sc._final_rc(0, 3) == 3
        assert sc._final_rc(2, 3) == 3
        assert sc._final_rc(1, 3) == 3

    def test_scorer_rc_passes_through_otherwise(self, sc):
        assert sc._final_rc(0, 0) == 0
        assert sc._final_rc(1, 0) == 1
        assert sc._final_rc(2, 0) == 2
        # a tripwire REFUSAL does not fail the run (surfaced in RESULT line)
        assert sc._final_rc(0, 2) == 0

    def test_exit_contract_documented(self):
        src = (REPO / "scripts" / "september_check.py").read_text(encoding="utf-8")
        assert "3 = the P237 tripwire FIRED" in src
        assert "RESULT:" in src


# ---------------------------------------------------------------------------
# 6. slope_calibrator — threshold provenance
# ---------------------------------------------------------------------------

class TestSlopeCalibratorProvenance:
    def test_stamp_and_age_arithmetic(self):
        from analytics.calibration import slope_calibrator as scal
        assert scal.THRESHOLDS_STAMPED == "2026-08-08"
        assert scal.default_thresholds_age_days(date(2026, 9, 10)) == 33
        assert scal.default_thresholds_age_days(date(2026, 8, 20)) == 12
        assert scal.THRESHOLDS_STALE_AFTER_DAYS == 30

    def test_report_carries_provenance_and_warning_exists(self):
        src = (REPO / "analytics" / "calibration" /
               "slope_calibrator.py").read_text(encoding="utf-8")
        assert "thresholds_provenance" in src
        assert "THRESHOLD STALENESS" in src


# ---------------------------------------------------------------------------
# 7. regimebook — per-bar carry on ledger rows
# ---------------------------------------------------------------------------

class TestRegimebookCarry:
    def _harness(self, tmp_path):
        from defense.regime_book_shadow import RegimeBookShadow
        return RegimeBookShadow(data_dir=str(tmp_path))

    def test_fresh_history_yields_rate_over_two(self, tmp_path):
        rbs = self._harness(tmp_path)
        yday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        rbs._fund_hist["BTC"] = {yday: 0.0003}
        # the stored value is the day's LAST 8h EVENT rate; an 8h event
        # spans two 4H bars -> per-bar carry = rate/2 (P245 convention)
        assert rbs.carry_rate_bar("BTC") == pytest.approx(0.00015)

    def test_stale_history_yields_none(self, tmp_path):
        rbs = self._harness(tmp_path)
        old = (datetime.now(timezone.utc).date() - timedelta(days=10)).isoformat()
        rbs._fund_hist["BTC"] = {old: 0.0003}
        assert rbs.carry_rate_bar("BTC") is None

    def test_absent_history_yields_none(self, tmp_path):
        rbs = self._harness(tmp_path)
        assert rbs.carry_rate_bar("BTC") is None

    def test_ledger_row_carries_the_derived_carry(self, tmp_path):
        rbs = self._harness(tmp_path)
        yday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        rbs._fund_hist["BTC"] = {yday: 0.0003}
        closes = [100.0 + 0.05 * i for i in range(600)]
        rec = rbs.record_tick("BTC", closes, price=closes[-1])
        assert rec is not None
        assert rec["carry_rate_bar"] == pytest.approx(0.00015)

    def test_explicit_caller_value_wins(self, tmp_path):
        rbs = self._harness(tmp_path)
        closes = [100.0 + 0.05 * i for i in range(600)]
        rec = rbs.record_tick("BTC", closes, price=closes[-1],
                              carry_rate_bar=0.42)
        assert rec["carry_rate_bar"] == pytest.approx(0.42)

    def test_no_history_row_stays_honestly_none(self, tmp_path):
        rbs = self._harness(tmp_path)
        closes = [100.0 + 0.05 * i for i in range(600)]
        rec = rbs.record_tick("BTC", closes, price=closes[-1])
        assert rec["carry_rate_bar"] is None


# ---------------------------------------------------------------------------
# 8. mlp_shadow — held-position persistence
# ---------------------------------------------------------------------------

def _mlp(tmp_path):
    from defense.mlp_shadow import MlpShadow
    ms = MlpShadow(data_dir=str(tmp_path / "data"),
                   repo_root=tmp_path / "empty_repo")
    ms._models["BTC"] = {"decision_interval": 4, "deadband": 0.25}
    return ms


class TestMlpShadowPersistence:
    DECISION_TS = 16 * 4 * 3600          # bin 16, 16 % 4 == 0
    HOLD_TS = 17 * 4 * 3600              # bin 17 — a hold bin

    def test_decision_persists_and_restart_restores_the_held_position(self, tmp_path):
        ms = _mlp(tmp_path)
        assert ms.decide("BTC", 0.8, self.DECISION_TS) == pytest.approx(0.8)
        state_file = tmp_path / "data" / "mlpshadow_state.json"
        assert state_file.exists()
        # RESTART: a fresh instance over the same data dir
        ms2 = _mlp(tmp_path)
        assert ms2._state["BTC"]["cur"] == pytest.approx(0.8)
        assert ms2._transient.get("BTC") is False
        # a hold bin after restart returns the HELD direction, not a
        # fabricated flat — the pre-P287 phantom-flat window
        assert ms2.decide("BTC", -0.9, self.HOLD_TS) == pytest.approx(0.8)

    def test_cold_start_rows_are_stamped_transient(self, tmp_path):
        ms = _mlp(tmp_path)
        rec = ms._write("BTC", 0.0, None)
        assert rec["restart_transient"] is True
        # after the first decision bin the stamp clears
        ms.decide("BTC", 0.5, self.DECISION_TS)
        rec2 = ms._write("BTC", 1.0, 0.5)
        assert rec2["restart_transient"] is False

    def test_restored_rows_are_not_transient(self, tmp_path):
        ms = _mlp(tmp_path)
        ms.decide("BTC", 0.8, self.DECISION_TS)
        ms2 = _mlp(tmp_path)
        rec = ms2._write("BTC", 1.0, 0.3)
        assert rec["restart_transient"] is False

    def test_corrupt_state_degrades_to_cold_start(self, tmp_path):
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "mlpshadow_state.json").write_text(
            "{corrupt", encoding="utf-8")
        ms = _mlp(tmp_path)          # must not raise
        assert ms._state == {} or "BTC" not in ms._state
        assert ms._write("BTC", 0.0, None)["restart_transient"] is True

    def test_version_mismatch_degrades_to_cold_start(self, tmp_path):
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "mlpshadow_state.json").write_text(
            json.dumps({"v": "other", "assets": {"BTC": {"cur": 0.7,
                                                         "last_bin": 3}}}),
            encoding="utf-8")
        ms = _mlp(tmp_path)
        assert "BTC" not in ms._state

    def test_no_tmp_file_left_behind(self, tmp_path):
        ms = _mlp(tmp_path)
        ms.decide("BTC", 0.8, self.DECISION_TS)
        assert not (tmp_path / "data" / "mlpshadow_state.tmp").exists()


# ---------------------------------------------------------------------------
# 9. trend_regime_review — unjoinable accounting
# ---------------------------------------------------------------------------

class TestTrendRegimeReviewUnjoinable:
    def test_unjoinable_records_are_counted_and_labeled(self, tmp_path,
                                                       monkeypatch, capsys):
        trr = _load_script("trend_regime_review")
        bar = 4 * 3600
        base = 1_755_000_000 - (1_755_000_000 % bar)
        ts_list = [base + i * bar for i in range(100)]
        closes = [100.0 + i for i in range(100)]
        monkeypatch.setattr(trr, "fetch_ohlc",
                            lambda pair: (ts_list, closes))

        def _iso(ep):
            return datetime.fromtimestamp(ep, tz=timezone.utc).isoformat()

        recs = [
            # joinable directional record mid-window
            {"ts": _iso(ts_list[50] + 10), "asset": "BTC", "trend_sig": 0.5,
             "regime": "X", "gated": False},
            # OLDER than the price window -> unjoinable (the ~Dec-2026 case)
            {"ts": _iso(ts_list[0] - 10 * bar), "asset": "BTC",
             "trend_sig": -0.5, "regime": "X", "gated": True},
            # at the price tail -> no forward bar -> unjoinable
            {"ts": _iso(ts_list[-1] + 10), "asset": "BTC", "trend_sig": 0.5,
             "regime": "X", "gated": False},
        ]
        f = tmp_path / "trend_regime_shadow.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in recs) + "\n",
                     encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["trend_regime_review",
                                          "--file", str(f)])
        assert trr.main() == 0
        out = capsys.readouterr().out
        assert "price-unjoinable: 2" in out
        assert "directional: 1" in out
        assert "EXCLUDED" in out
        assert "JOINED window only" in out

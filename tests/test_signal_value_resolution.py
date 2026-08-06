"""[P173] Three more reads of the wrong dict, found by triaging P171's output.

P171's scanner reported ORPHAN=0 but MISROUTED=34, and MISROUTED was left
ungated because name-based write-tracking under-counts producers (the pipeline
fills `raw` and returns it as market_data). "Too noisy to gate" is not "all
false positives", so the list was triaged by hand. Most were the expected
naming gap. Three were real, and all three had the correct read sitting a few
lines away in the same file:

  1. `core/execution_service.py:541` — `market_data.get("drl_confidence", 0.5)`.
     The producer is `agent_signals['drl_confidence']` (main.py:7817), which
     the SAME function reads correctly ~3000 lines below when it stamps
     `latest_drl_confidence` onto the position. So this was the constant 0.5.
     `ExecutionGuard.can_drl_trade` compares it against
     `min_confidence_volatile = 0.7`, so in every VOLATILE regime it failed and
     stamped `drl_blocked_reason` onto every execution. That branch records
     rather than blocks — the cost is a diagnostic that ALWAYS fires, which is
     exactly as uninformative as P170's guard that never fired.

  2. `core/execution_service.py:3606` — `phase_at_entry` read from market_data,
     which nobody writes `phase` into. Every position ever opened recorded
     "UNKNOWN", so phase-attribution analysis was reading a constant.

  3. `core/smart_beta_controller.py:144` — `market_data.get("phase",
     agent_signals.get("_phase", "UNKNOWN"))`. NEITHER key exists; `_phase`
     with the leading underscore appears nowhere else in the tree. `phase` was
     always "UNKNOWN", so the TREND_STRONG branch was unreachable and the
     controller never applied its bullish-trend boost. smart_beta is
     `enabled: true` in configs/live_high_risk.json, so this one has real
     behavioural effect — and fixing it LOOSENS the gate.

Also checked and deliberately NOT changed: `system_state["is_4h_bar_close"]`
defaults to True at main.py:6759 with nothing writing the key. That default is
load-bearing and documented — `_process_4h_tick_inner` is only reached from
loops that sleep to the 4H candle boundary, so the tick IS a bar close by
construction. See TestIs4hBarCloseIsADocumentedAssumption.
"""

import io
from pathlib import Path

import pytest

from core.market_data_helpers import signal_value

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestSignalValuePrefersTheProducer:
    def test_agent_signals_wins(self):
        assert signal_value("k", {"k": 1}, {"k": 2}, 9) == 1

    def test_falls_through_to_market_data(self):
        assert signal_value("k", {}, {"k": 2}, 9) == 2

    def test_default_when_neither_has_it(self):
        assert signal_value("k", {}, {}, 9) == 9

    def test_explicit_none_is_treated_as_absent(self):
        # A producer that wrote None did not make a measurement.
        assert signal_value("k", {"k": None}, {"k": 2}, 9) == 2

    def test_none_everywhere_falls_to_default(self):
        assert signal_value("k", {"k": None}, {"k": None}, 9) == 9

    @pytest.mark.parametrize("bad", [None, "not a dict", 5, [], object()])
    def test_non_mapping_sources_are_skipped_not_fatal(self, bad):
        assert signal_value("k", bad, {"k": 2}, 9) == 2
        assert signal_value("k", {"k": 1}, bad, 9) == 1

    def test_both_sources_unusable(self):
        assert signal_value("k", None, None, 9) == 9

    def test_falsy_measurements_are_returned_not_skipped(self):
        # 0.0 is a measurement. Only None means "nobody wrote it".
        for falsy in (0, 0.0, False, "", [], {}):
            assert signal_value("k", {"k": falsy}, {"k": "wrong"}, "dflt") == falsy

    def test_default_defaults_to_none(self):
        assert signal_value("k", {}, {}) is None


class TestDrlConfidenceReadsTheProducer:
    """Finding 1. The guard's input was a constant."""

    def _src(self):
        return io.open(REPO_ROOT / "core" / "execution_service.py",
                       encoding="utf-8").read()

    def test_no_longer_reads_market_data_directly(self):
        code = [ln for ln in self._src().splitlines()
                if not ln.lstrip().startswith("#")
                and 'market_data.get("drl_confidence"' in ln]
        assert not code, f"the single-dict read is back: {code}"

    def test_resolves_through_the_helper(self):
        assert 'signal_value("drl_confidence", agent_signals, market_data' in self._src()

    def test_absent_drl_gets_no_confidence_not_half(self):
        # Matches the [BUGFIX M7] decision on the adjacent _drl_weight line:
        # absent DRL gets zero weight, so it must get zero confidence too.
        assert signal_value("drl_confidence", {}, {}, 0.0) == 0.0

    def test_a_real_confidence_now_reaches_the_guard(self):
        assert signal_value("drl_confidence", {"drl_confidence": 0.82}, {}, 0.0) == 0.82

    def test_the_old_constant_always_failed_the_volatile_bar(self):
        # Documents what was at stake: min_confidence_volatile is 0.7.
        from defense.execution_guards import DRLConstraintConfig  # noqa: F401
        assert 0.5 < 0.7

    def test_min_confidence_volatile_is_still_above_the_old_constant(self):
        # If this ever drops to <= 0.5 the old bug becomes invisible rather
        # than fixed; pin the number the finding depends on.
        src = io.open(REPO_ROOT / "defense" / "execution_guards.py",
                      encoding="utf-8").read()
        assert "min_confidence_volatile: float = 0.7" in src


class TestPhaseAtEntryReadsTheProducer:
    """Finding 2. Every position recorded UNKNOWN."""

    def _src(self):
        return io.open(REPO_ROOT / "core" / "execution_service.py",
                       encoding="utf-8").read()

    def test_no_longer_reads_market_data_directly(self):
        assert "\"phase_at_entry\": market_data.get('phase'" not in self._src()

    def test_resolves_through_the_helper(self):
        assert '"phase_at_entry": signal_value("phase", agent_signals, market_data' in self._src()

    def test_a_real_phase_is_now_recorded(self):
        assert signal_value("phase", {"phase": "IGNITION"}, {}, "UNKNOWN") == "IGNITION"

    def test_absence_still_records_unknown(self):
        # Absence must stay named. "UNKNOWN" is not a phase.
        assert signal_value("phase", {}, {}, "UNKNOWN") == "UNKNOWN"


class TestSmartBetaPhaseTypo:
    """Finding 3. `_phase` with a leading underscore exists nowhere."""

    def _src(self):
        return io.open(REPO_ROOT / "core" / "smart_beta_controller.py",
                       encoding="utf-8").read()

    def test_underscore_phase_is_gone_from_the_read(self):
        src = self._src()
        code = [ln for ln in src.splitlines()
                if not ln.lstrip().startswith("#") and 'agent_signals.get("_phase"' in ln]
        assert not code, f"the _phase typo is back: {code}"

    def test_resolves_through_the_helper(self):
        assert 'signal_value("phase", agent_signals, market_data' in self._src()

    def test_underscore_phase_still_has_no_producer(self):
        # If someone ever DOES write agent_signals["_phase"], this fix needs
        # revisiting rather than silently preferring the wrong key.
        import pathlib
        excluded = ("archive", ".git", "__pycache__", "node_modules",
                    "training_data", "venv", ".venv", "tests")
        writers = []
        for f in REPO_ROOT.rglob("*.py"):
            if any(p in excluded for p in f.parts):
                continue
            try:
                src = f.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
            for i, ln in enumerate(src.splitlines(), 1):
                if ln.lstrip().startswith("#"):
                    continue  # P173's own comments quote the old code
                if '"_phase"' in ln or "'_phase'" in ln:
                    writers.append(f"{f.name}:{i}")
        assert not writers, f"_phase now exists somewhere: {writers}"

    def test_trend_strong_branch_is_reachable_with_a_real_phase(self):
        # The behavioural consequence: with phase resolved, a bullish regime in
        # IGNITION/EXPANSION can finally tag TREND_STRONG.
        phase = str(signal_value("phase", {"phase": "expansion"}, {}, "UNKNOWN")).upper()
        assert phase in ("IGNITION", "EXPANSION")

    def test_the_old_constant_could_never_reach_it(self):
        phase = str(signal_value("phase", {}, {}, "UNKNOWN")).upper()
        assert phase not in ("IGNITION", "EXPANSION")


class TestFallbackStubMatchesTheRealHelper:
    """The import-failure path is the one nobody tests. It must not regress."""

    def test_stub_reads_both_dicts(self):
        src = io.open(REPO_ROOT / "core" / "execution_service.py",
                      encoding="utf-8").read()
        start = src.index("def signal_value(key, agent_signals, market_data")
        body = src[start:start + 400]
        assert "for _src in (agent_signals, market_data)" in body, (
            "the ImportError fallback stub reads only one dict — it would "
            "restore the exact bug the helper prevents, on the one path that "
            "is never exercised"
        )


class TestIs4hBarCloseIsADocumentedAssumption:
    """Triaged and deliberately left alone. Recorded so it is not 'fixed'."""

    def test_nothing_writes_the_key_and_that_is_intended(self):
        src = io.open(REPO_ROOT / "main.py", encoding="utf-8").read()
        assert 'bool(market_data.get("is_4h_bar_close", True))' in src

    def test_the_reasoning_is_written_down_beside_it(self):
        src = io.open(REPO_ROOT / "main.py", encoding="utf-8").read()
        i = src.index('"is_4h_bar_close": bool(market_data.get')
        assert "only invoked on the scheduled 4H decision loop" in src[i - 500:i], (
            "the True default for is_4h_bar_close permanently satisfies the "
            "T1->T2 tranche escalation gate; it is only safe because every "
            "caller sleeps to the 4H candle boundary. Keep that reasoning "
            "next to the default."
        )

    def test_callers_still_go_through_process_4h_tick(self):
        # If a faster loop ever calls this path, the assumption breaks and the
        # bar-close gate silently unlocks.
        src = io.open(REPO_ROOT / "main.py", encoding="utf-8").read()
        assert src.count("await self.process_4h_tick(") == 4, (
            "a new caller of process_4h_tick appeared — verify it only runs on "
            "4H candle boundaries before trusting is_4h_bar_close=True"
        )

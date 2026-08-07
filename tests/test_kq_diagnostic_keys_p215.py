"""[P215] The kraken_quant diagnostic could only ever report "everything is dead".

`KrakenQuantAgent.get_firing_stats()` keyed its telemetry by the Regime enum's
VALUE (`'chop'`, `'bear'`, `'bull'`); `scripts/kq_strategy_diagnostic.py` looked
it up by the enum's NAME (`'SIDEWAYS'`, `'BEAR'`, `'BULL'`). Every lookup missed:

  * `by_regime.get('SIDEWAYS')` -> `[]`, so every strategy row was empty and the
    table printed `attempts=0 fires=0` for ALL TWELVE strategies **regardless of
    what actually happened**;
  * `regime_ticks.get('SIDEWAYS')` -> `0`, so the status column read
    "never-active (regime not seen)" for SIDEWAYS while the header of the same
    report showed `chop=3 (100%)`.

The method's own docstring declared the contract as
`{'BEAR': int, 'BULL': int, 'SIDEWAYS': int}` — names — so the writer violated a
contract written five lines above it.

Two reasons this matters more than a cosmetic bug. First, a diagnostic whose
output is structurally constant is the P174 class: it cannot fail, so it cannot
inform, and "0 fires" reads as a finding about the strategies. Second, the false
"regime not seen" line sends an operator to fix the regime mapping — which is
CORRECT (`QUIET_ACCUMULATION`/`WEAK_CONSOLIDATION` -> `Regime.SIDEWAYS`). That is
P155's lesson: an alert that names a subsystem from a guess rather than from the
data is worse than silence, because the named subsystem is innocent.

Archived strategies were also indistinguishable from dead ones — both showed
`attempts=0`. Being archived is a deliberate P157 decision and must not read as a
fault.
"""

import json
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_AGENT = _REPO / "agents" / "kraken_quant_agent.py"
_SCRIPT = _REPO / "scripts" / "kq_strategy_diagnostic.py"
_ASRC = _AGENT.read_text(encoding="utf-8", errors="replace")
_SSRC = _SCRIPT.read_text(encoding="utf-8", errors="replace")


class TestWriterAndReaderAgree:

    def test_writer_keys_by_enum_name(self):
        assert "self._regime_ticks[regime.name] += 1" in _ASRC, (
            "regime_ticks keyed by .value again — the reader looks up .name"
        )
        assert "by_regime[regime.name] = rows" in _ASRC, (
            "by_regime keyed by .value again — every strategy row reads empty "
            "and the whole table prints fabricated zeros"
        )

    def test_the_value_keyed_writes_are_gone(self):
        code = "\n".join(l for l in _ASRC.splitlines()
                         if not l.lstrip().startswith("#"))
        assert "self._regime_ticks[regime.value]" not in code
        assert "by_regime[regime.value]" not in code

    def test_the_documented_contract_is_names(self):
        """The docstring was right all along; the code disagreed with it."""
        i = _ASRC.index("'regime_ticks': {'BEAR': int")
        assert i > 0, "the declared contract disappeared — re-derive it"

    def test_round_trip_writer_keys_are_what_the_reader_looks_up(self):
        """The real invariant, independent of spelling: every key the writer
        emits must be one the reader's CANONICAL map contains."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("_kqdiag", _SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from agents.kraken_quant_agent import Regime
        for r in Regime:
            assert r.name in mod.CANONICAL, (
                f"agent can emit bucket {r.name!r} but the diagnostic's "
                f"CANONICAL map has {sorted(mod.CANONICAL)}"
            )


class TestBackCompat:

    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("_kqdiag2", _SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_old_value_keyed_snapshots_still_read(self):
        """A stats file written before the fix must report what it recorded,
        not silently read as 'everything dead' — which IS the bug."""
        mod = self._mod()
        assert mod._by_name({"chop": 7})["SIDEWAYS"] == 7
        assert mod._by_name({"bear": 1, "bull": 2}) == {"BEAR": 1, "BULL": 2}

    def test_new_name_keyed_snapshots_pass_through(self):
        mod = self._mod()
        assert mod._by_name({"SIDEWAYS": 5}) == {"SIDEWAYS": 5}

    def test_unknown_keys_are_preserved_not_dropped(self):
        """Dropping an unrecognised bucket would hide a new regime."""
        mod = self._mod()
        assert mod._by_name({"WAT": 3}) == {"WAT": 3}


class TestArchivedIsNotDead:

    def test_stats_expose_the_archived_set(self):
        assert "'archived': sorted(self._archived_strategies)" in _ASRC

    def test_the_report_labels_archived_distinctly(self):
        assert "ARCHIVED (P157 decision" in _SSRC
        i = _SSRC.index("if s_name in archived:")
        j = _SSRC.index("elif att == 0:", i)
        assert i < j, "the archived branch must precede the attempts==0 branch"

    def test_summary_does_not_score_against_a_flat_twelve(self):
        """With 4 archived and one regime bucket ever active, 'x/12' counts
        strategies excluded by design or unreachable by regime."""
        assert "reachable in the regimes actually seen" in _SSRC
        assert "{alive}/12 alive" not in _SSRC


class TestEndToEndRender:

    def test_a_value_keyed_snapshot_reports_real_numbers(self, tmp_path, capsys):
        """The regression, end to end: feed the OLD on-disk shape and assert the
        table shows the attempts it recorded rather than zeros."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("_kqdiag3", _SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        stats = {
            "ts": "2026-08-07T00:00:00Z", "tick": 10, "uptime_sec": 3600,
            "regime_ticks": {"chop": 21},
            # The agent builds a row for EVERY strategy in the bucket, archived
            # ones included (they are skipped at invoke time, not at report
            # time), so the fixture mirrors that.
            "by_regime": {"chop": [
                {"name": "KalmanCointegrationStrategy", "attempts": 21, "fires": 3},
                {"name": "OrnsteinUhlenbeckStrategy", "attempts": 0, "fires": 0},
                {"name": "DarkPoolVolumeStrategy", "attempts": 21, "fires": 0},
                {"name": "DeltaNeutralFundingStrategy", "attempts": 0, "fires": 0},
            ]},
            "never_fired": ["DarkPoolVolumeStrategy"],
            "archived": ["OrnsteinUhlenbeckStrategy", "DeltaNeutralFundingStrategy"],
        }
        mod.render(stats)
        out = capsys.readouterr().out
        assert "regime_ticks=21" in out, (
            "SIDEWAYS bucket still reads 0 — the key mismatch is back"
        )
        assert re.search(r"KalmanCointegrationStrategy\s+21\s+3", out), out
        # Scope to the SIDEWAYS section: BEAR/BULL genuinely were not seen in
        # this snapshot, so "regime not seen" is CORRECT for them. Asserting
        # over the whole report failed on the tool telling the truth.
        _sw = out[out.index("[SIDEWAYS]"):]
        _sw = _sw[:_sw.index("-" * 40)] if "-" * 40 in _sw else _sw
        assert "never-active (regime not seen)" not in _sw, (
            f"false diagnosis: SIDEWAYS WAS seen 21 times\n{_sw}"
        )
        assert "ARCHIVED" in out
        # ...and the buckets that really were absent must still say so.
        assert "never-active (regime not seen)" in out[:out.index("[SIDEWAYS]")]


class TestStrategyNamesComeFromTheAgent:
    """[P215, second pass] The same mismatch one level down. Three runtime
    `strategy.name` values differ from the script's hardcoded CANONICAL list
    (`KalmanCointegration_SOL_ETH` vs `KalmanCointegrationStrategy`,
    `ETFSpotCointegration`, `OrderBookImbalance`), so the row lookup missed and
    the report printed "[!] not invoked despite regime active" for a strategy
    that HAD been invoked — a fresh false diagnosis, produced by the fix to the
    previous one. A hardcoded mirror of a runtime list drifts; iterate what the
    agent reported.
    """

    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("_kqdiag4", _SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_rows_drive_the_table_not_the_hardcoded_list(self, capsys):
        mod = self._mod()
        mod.render({
            "ts": "x", "tick": 1, "uptime_sec": 60,
            "regime_ticks": {"SIDEWAYS": 5},
            "by_regime": {"SIDEWAYS": [
                # runtime name, deliberately NOT the CANONICAL spelling
                {"name": "KalmanCointegration_SOL_ETH", "attempts": 5, "fires": 2},
            ]},
            "never_fired": [], "archived": [],
        })
        out = capsys.readouterr().out
        assert re.search(r"KalmanCointegration_SOL_ETH\s+5\s+2", out), out
        assert "not invoked despite regime active" not in out, (
            "a strategy with 5 attempts was reported as never invoked"
        )

    def test_reachable_uses_runtime_names(self, capsys):
        mod = self._mod()
        mod.render({
            "ts": "x", "tick": 1, "uptime_sec": 60,
            "regime_ticks": {"SIDEWAYS": 5},
            "by_regime": {"SIDEWAYS": [
                {"name": "KalmanCointegration_SOL_ETH", "attempts": 5, "fires": 0},
                {"name": "OrnsteinUhlenbeckStrategy", "attempts": 0, "fires": 0},
            ]},
            "never_fired": [], "archived": ["OrnsteinUhlenbeckStrategy"],
        })
        out = capsys.readouterr().out
        assert "KalmanCointegration_SOL_ETH" in out.split("Reachable now:")[1]
        assert "1 reachable" in out

    def test_a_bucket_that_lost_a_strategy_is_flagged(self):
        """CANONICAL still earns its place: it must notice a bucket the agent
        stopped reporting a strategy for."""
        assert "absent from the agent's" in _SCRIPT.read_text(encoding="utf-8")

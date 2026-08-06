"""[P171] The scanner that catches reader/writer drift, and the drift it caught.

P170 was the twelfth sighting of one bug: a consumer reads a key off one signal
dict, the producer writes it into a different one, and the `.get()` default —
chosen to look reassuring — becomes the only value that key will ever hold. The
guard does not fail. It cannot fail, which from the logs is indistinguishable
from passing.

`lint_signal_freshness.py` (P120) inventories agent_signals WRITES, so a key
with no writer at all is structurally invisible to it. This scanner is the
complement, and these tests pin the properties that make it trustworthy:

  * a file it cannot parse must make it REFUSE to report, never report clean —
    the first version of this scanner reproduced the exact bug class it was
    written to detect (see TestAParseFailureIsNotACleanScan);
  * a key written to a *different* signal dict is MISROUTED, not ORPHAN, since
    that distinction is the difference between a typo and a real absence;
  * a non-falsy default is HOT, because that is the shape that asserts
    something positive nobody measured.

Plus a regression test for the live bug the scanner surfaced in
agents/model_alpha_agent.py.
"""

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.lint_orphan_signal_reads import (  # noqa: E402
    PARSE_FAILURES,
    _is_falsy_default,
    run,
    scan_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNER = REPO_ROOT / "tools" / "lint_orphan_signal_reads.py"


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return p


def _scan(tmp_path):
    return run([str(tmp_path)])


class TestAParseFailureIsNotACleanScan:
    """The scanner's own original bug. This is the load-bearing test."""

    def test_unparseable_file_is_recorded_not_swallowed(self, tmp_path):
        _write(tmp_path, "broken.py", "def f(:\n")
        result = _scan(tmp_path)
        assert result["parse_failures"], (
            "a file that failed to parse was silently treated as contributing "
            "no writes — every key it produces would look orphaned"
        )
        assert "broken.py" in result["parse_failures"][0]

    def test_scanner_refuses_to_report_when_a_file_failed_to_parse(self, tmp_path):
        _write(tmp_path, "broken.py", "def f(:\n")
        _write(tmp_path, "reader.py", 'x = agent_signals.get("whatever", 1.0)\n')
        r = subprocess.run(
            [sys.executable, "-X", "utf8", str(SCANNER), "--paths", str(tmp_path)],
            capture_output=True, text=True,
        )
        assert r.returncode == 2, "a scan that could not read the code exited clean"
        assert "REFUSING TO REPORT" in r.stdout

    def test_utf8_bom_is_an_encoding_detail_not_a_syntax_error(self, tmp_path):
        # main.py starts with U+FEFF. Reading it as plain utf-8 raises
        # SyntaxError on the BOM, which deleted the tree's biggest producer
        # from the scan and manufactured 36 confident false positives.
        p = tmp_path / "bommed.py"
        p.write_bytes("﻿agent_signals = {'k': 1}\n".encode("utf-8"))
        PARSE_FAILURES.clear()
        assert ("agent_signals", "k") in scan_file(p).writes
        assert not PARSE_FAILURES

    def test_the_real_main_py_is_actually_parsed(self):
        # The regression that mattered: main.py is the dominant producer, so if
        # it drops out of the scan the whole result inverts.
        head = (REPO_ROOT / "main.py").read_bytes()[:3]
        assert head == b"\xef\xbb\xbf", (
            "main.py no longer starts with a BOM — good, but this test is the "
            "record of why the scanner reads utf-8-sig; keep it reading that way"
        )
        writes = scan_file(REPO_ROOT / "main.py").writes
        assert len(writes) > 50, f"main.py contributed only {len(writes)} writes"

    def test_parse_failures_do_not_leak_between_runs(self, tmp_path):
        _write(tmp_path, "broken.py", "def f(:\n")
        assert _scan(tmp_path)["parse_failures"]
        clean = tmp_path / "clean"
        clean.mkdir()
        _write(clean, "ok.py", "agent_signals = {'k': 1}\n")
        assert _scan(clean)["parse_failures"] == []

    def test_unreadable_file_is_a_parse_failure_too(self, tmp_path):
        p = _write(tmp_path, "gone.py", "x = 1\n")
        p.chmod(0o000)
        try:
            result = _scan(tmp_path)
        finally:
            p.chmod(0o644)
        # Root can read anything; only assert when the chmod actually bit.
        if result["parse_failures"]:
            assert "gone.py" in result["parse_failures"][0]


class TestOrphanDetection:
    def test_read_with_no_writer_anywhere_is_an_orphan(self, tmp_path):
        _write(tmp_path, "reader.py", 'q = agent_signals.get("nobody_writes_me", 1.0)\n')
        kinds = {(f["key"], f["kind"]) for f in _scan(tmp_path)["findings"]}
        assert ("nobody_writes_me", "ORPHAN") in kinds

    def test_a_key_with_a_writer_is_not_reported(self, tmp_path):
        _write(tmp_path, "producer.py", 'agent_signals = {"present": 0.5}\n')
        _write(tmp_path, "reader.py", 'q = agent_signals.get("present", 1.0)\n')
        assert not [f for f in _scan(tmp_path)["findings"] if f["key"] == "present"]

    def test_subscript_write_counts_as_a_writer(self, tmp_path):
        _write(tmp_path, "producer.py", 'agent_signals["via_subscript"] = 0.5\n')
        _write(tmp_path, "reader.py", 'q = agent_signals.get("via_subscript", 1.0)\n')
        assert not _scan(tmp_path)["findings"]

    def test_setdefault_counts_as_a_writer(self, tmp_path):
        _write(tmp_path, "producer.py", 'market_data.setdefault("dq", 0.0)\n')
        _write(tmp_path, "reader.py", 'q = market_data.get("dq", 1.0)\n')
        assert not _scan(tmp_path)["findings"]

    def test_update_with_a_literal_counts_as_a_writer(self, tmp_path):
        _write(tmp_path, "producer.py", 'agent_signals.update({"via_update": 1})\n')
        _write(tmp_path, "reader.py", 'q = agent_signals.get("via_update", 1.0)\n')
        assert not _scan(tmp_path)["findings"]

    def test_a_read_with_no_default_is_still_inventoried(self, tmp_path):
        _write(tmp_path, "reader.py", 'q = agent_signals.get("bare")\n')
        f = _scan(tmp_path)["findings"]
        assert [x for x in f if x["key"] == "bare"]

    def test_untracked_dicts_are_ignored(self, tmp_path):
        _write(tmp_path, "reader.py", 'q = some_other_dict.get("whatever", 1.0)\n')
        assert _scan(tmp_path)["findings"] == []


class TestMisroutedIsNotOrphan:
    """P170's exact shape: written to market_data, read from agent_signals."""

    def test_cross_dict_write_is_misrouted(self, tmp_path):
        _write(tmp_path, "producer.py", 'market_data["quant_data_quality"] = 1.0\n')
        _write(tmp_path, "reader.py",
               'q = agent_signals.get("quant_data_quality", 1.0)\n')
        found = [f for f in _scan(tmp_path)["findings"]
                 if f["key"] == "quant_data_quality"]
        assert found and found[0]["kind"] == "MISROUTED"

    def test_misrouted_is_not_counted_as_orphan(self, tmp_path):
        _write(tmp_path, "producer.py", 'market_data["k"] = 1.0\n')
        _write(tmp_path, "reader.py", 'q = agent_signals.get("k", 1.0)\n')
        r = _scan(tmp_path)
        assert r["by_kind"]["MISROUTED"] == 1
        assert r["by_kind"]["ORPHAN"] == 0


class TestHotVsCold:
    """A non-falsy default asserts something positive that nobody measured."""

    @pytest.mark.parametrize("default", ["1.0", "50.0", "True", '"healthy"', "-1"])
    def test_non_falsy_defaults_are_hot(self, tmp_path, default):
        _write(tmp_path, "reader.py", f'q = agent_signals.get("k", {default})\n')
        assert _scan(tmp_path)["findings"][0]["severity"] == "HOT"

    @pytest.mark.parametrize("default", ["0", "0.0", "False", '""', "None", "[]", "{}"])
    def test_falsy_defaults_are_cold(self, tmp_path, default):
        _write(tmp_path, "reader.py", f'q = agent_signals.get("k", {default})\n')
        assert _scan(tmp_path)["findings"][0]["severity"] == "COLD"

    def test_the_two_p170_defaults_would_both_have_been_hot(self, tmp_path):
        _write(tmp_path, "reader.py",
               'a = agent_signals.get("quant_data_quality", 1.0)\n'
               'b = agent_signals.get("signal_edge_bps", 50.0)\n')
        r = _scan(tmp_path)
        assert r["hot_count"] == 2

    def test_negative_default_is_not_treated_as_falsy(self):
        import ast
        assert not _is_falsy_default(ast.parse("-1", mode="eval").body)


class TestUnprovableIsNamedNotHidden:
    """A blind spot must shrink the confidence, not the finding count."""

    def test_dynamic_write_marks_the_dict_unprovable(self, tmp_path):
        _write(tmp_path, "producer.py",
               "for k, v in stuff:\n    agent_signals[k] = v\n")
        _write(tmp_path, "reader.py", 'q = agent_signals.get("maybe", 1.0)\n')
        r = _scan(tmp_path)
        assert r["by_kind"]["UNPROVABLE"] == 1
        assert r["by_kind"]["ORPHAN"] == 0

    def test_dynamic_sites_are_reported(self, tmp_path):
        _write(tmp_path, "producer.py",
               "for k, v in stuff:\n    agent_signals[k] = v\n")
        assert any("agent_signals" in s for s in _scan(tmp_path)["dynamic_write_sites"])

    def test_splat_literal_is_dynamic(self, tmp_path):
        _write(tmp_path, "producer.py", "agent_signals = {**other}\n")
        _write(tmp_path, "reader.py", 'q = agent_signals.get("maybe", 1.0)\n')
        assert _scan(tmp_path)["by_kind"]["UNPROVABLE"] == 1

    def test_aliased_assignment_is_dynamic(self, tmp_path):
        _write(tmp_path, "producer.py", "agent_signals = build_them()\n")
        _write(tmp_path, "reader.py", 'q = agent_signals.get("maybe", 1.0)\n')
        assert _scan(tmp_path)["by_kind"]["UNPROVABLE"] == 1

    def test_unprovable_is_excluded_from_the_gated_count(self, tmp_path):
        _write(tmp_path, "producer.py", "for k in ks:\n    agent_signals[k] = 1\n")
        _write(tmp_path, "reader.py", 'q = agent_signals.get("maybe", 1.0)\n')
        assert _scan(tmp_path)["orphan_count"] == 0


class TestBaselineFormat:
    """What the CI gate scores, and what it deliberately does not."""

    def _baseline(self, paths):
        r = subprocess.run(
            [sys.executable, "-X", "utf8", str(SCANNER),
             "--baseline-format", "--paths", *paths],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout)

    def test_emits_only_the_provable_metrics(self, tmp_path):
        _write(tmp_path, "reader.py", 'q = agent_signals.get("k", 1.0)\n')
        assert set(self._baseline([str(tmp_path)])) == {
            "copy_only_count", "misrouted_count", "misrouted_hot_count",
            "dynamic_site_count", "orphan_count", "orphan_coverage_lost",
            "parse_failure_count",
        }

    def test_misrouted_is_scored_now(self, tmp_path):
        # [P174] Reversal of P171's decision. MISROUTED was left ungated on the
        # theory that producers build these dicts under other variable names —
        # a guess, and the wrong one. Crediting those producers explicitly
        # (PRODUCED_ELSEWHERE) shrank the list to something hand-triaged, and it
        # is the only metric here that has ever caught a real bug: P170 and all
        # three P173 sites. `_diff` fails on rises, so a new suspect trips it.
        _write(tmp_path, "producer.py", 'market_data["k"] = 1.0\n')
        _write(tmp_path, "reader.py", 'q = agent_signals.get("k", 1.0)\n')
        b = self._baseline([str(tmp_path)])
        assert b["orphan_count"] == 0
        assert b["misrouted_count"] == 1
        assert b["misrouted_hot_count"] == 1

    def test_orphans_are_scored(self, tmp_path):
        _write(tmp_path, "reader.py", 'q = agent_signals.get("nobody", 1.0)\n')
        b = self._baseline([str(tmp_path)])
        assert b["orphan_count"] == 1

    def test_parse_failures_are_visible_to_the_gate(self, tmp_path):
        _write(tmp_path, "broken.py", "def f(:\n")
        assert self._baseline([str(tmp_path)])["parse_failure_count"] == 1

    def test_the_repo_is_currently_clean(self):
        b = json.loads(
            (REPO_ROOT / "tools" / "scanner_baselines"
             / "orphan_signal_reads_baseline.json").read_text(encoding="utf-8")
        )
        assert b["parse_failure_count"] == 0, (
            "the committed baseline was taken from a scan that could not read "
            "part of the tree; re-seed it after fixing the parse failure"
        )
        assert b["orphan_count"] == 0


class TestModelAlphaAgentReadsTheRightDict:
    """The live bug this scanner surfaced (P171)."""

    def _src(self):
        return io.open(REPO_ROOT / "agents" / "model_alpha_agent.py",
                       encoding="utf-8").read()

    def test_micro_keys_no_longer_read_agent_signals_directly(self):
        src = self._src()
        for key in ("order_book_imbalance", "spread_bps"):
            assert f'agent_signals.get("{key}"' not in src, (
                f"{key} is read straight off agent_signals again — nothing "
                f"writes it there, so it resolves to 0.0 (balanced book / zero "
                f"spread, i.e. free trading) on every call"
            )

    def test_micro_keys_go_through_the_fallback_helper(self):
        src = self._src()
        assert '_get_either("order_book_imbalance"' in src
        assert '_get_either("spread_bps"' in src

    def test_the_helper_records_a_miss(self):
        # The second half of the bug: bypassing _get meant these keys never
        # appeared in `missing`, so the coverage instrumentation built to catch
        # exactly this reported full coverage.
        src = self._src()
        start = src.index("def _get_either")
        body = src[start:start + 2000]
        assert "missing.append(key)" in body, (
            "_get_either no longer records a miss; the coverage tracker will "
            "report full coverage over keys that never arrive"
        )

    def test_market_data_is_preferred_over_agent_signals(self):
        src = self._src()
        start = src.index("def _get_either")
        body = src[start:start + 2000]
        assert body.index("md.get(key") < body.index("agent_signals.get(key")

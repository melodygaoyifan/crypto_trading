"""[P176] The MISROUTED metric was 100% false positives, and blind to P170.

P174 shipped `misrouted_hot_count: 10` as the gate's headline finding — the
number a reviewer is supposed to look at first. All ten were wrong. They were
four reads of `market_data.get("data_valid", True)` and six of
`market_data.get("vpin_source", "synthetic")`, both keys genuinely produced by
`data_mgmt/market_data_pipeline.py` and returned into `market_data`. They were
flagged only because the classifier tested `written_other` (MISROUTED) before
`produced_elsewhere`, so a key that is both correctly produced AND copied into
`system_state` (main.py:6755) landed in the wrong bucket.

Worse, the same investigation found the metric could not detect the bug it was
named for. Reconstructed synthetically, P170's exact shape —
`agent_signals.get("quant_data_quality", 1.0)` where the pipeline produces the
key into *market_data* and nothing copies it across — classified as
PRODUCED_ELSEWHERE, which is reported but never gated. So MISROUTED was
simultaneously 100% noise and 0% recall.

The fix needs the destination, not just the fact of production, which is why
PRODUCER_MODULES exists. The naive repair — "produced beats misrouted" — makes
the false positives go away and silences P170, P173's `drl_confidence` and
`phase`, and every bug this scanner was built for. These tests pin the
distinction so that repair cannot be reintroduced as a simplification.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNER = REPO_ROOT / "tools" / "lint_orphan_signal_reads.py"
BASELINE = (REPO_ROOT / "tools" / "scanner_baselines"
            / "orphan_signal_reads_baseline.json")

sys.path.insert(0, str(REPO_ROOT / "tools"))
import lint_orphan_signal_reads as scanner  # noqa: E402


def _scan(path):
    r = subprocess.run(
        [sys.executable, "-X", "utf8", str(SCANNER), "--json", "--paths", str(path)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert r.returncode in (0, 1), r.stderr
    return json.loads(r.stdout)


def _kinds(result):
    return {(f["dict"], f["key"]): f["kind"] for f in result["findings"]}


@pytest.fixture
def tree(tmp_path):
    """A miniature of the real architecture: a declared producer + consumers."""
    (tmp_path / "data_mgmt").mkdir()
    (tmp_path / "data_mgmt" / "market_data_pipeline.py").write_text(
        'def fetch_and_prepare(asset):\n'
        '    raw = {"quant_data_quality": 1.0, "data_valid": True}\n'
        '    raw["vpin_source"] = "computed"\n'
        '    return raw\n', encoding="utf-8")
    return tmp_path


class TestTheTenFalseHotsAreGone:
    def test_reading_a_produced_key_off_its_own_dict_is_not_a_defect(self, tree):
        (tree / "consumer.py").write_text(
            'def f(market_data):\n'
            '    return market_data.get("data_valid", True)\n', encoding="utf-8")
        assert _kinds(_scan(tree))[("market_data", "data_valid")] == "PRODUCED_HERE"

    def test_a_copy_into_another_dict_does_not_make_the_correct_read_a_misroute(self, tree):
        # main.py:6755 verbatim in shape: system_state relays the key, and the
        # market_data read beside it is correct. This pairing produced all ten
        # false HOTs.
        (tree / "consumer.py").write_text(
            'def f(market_data):\n'
            '    system_state = {"data_valid": market_data.get("data_valid", True)}\n'
            '    return system_state\n', encoding="utf-8")
        kinds = _kinds(_scan(tree))
        assert kinds.get(("market_data", "data_valid")) == "PRODUCED_HERE", kinds


class TestP170IsStillDetected:
    """The load-bearing half. If these pass vacuously the gate is decoration."""

    def test_the_p170_shape_is_misrouted(self, tree):
        (tree / "consumer.py").write_text(
            'def decide(agent_signals):\n'
            '    return agent_signals.get("quant_data_quality", 1.0)\n',
            encoding="utf-8")
        kinds = _kinds(_scan(tree))
        assert kinds[("agent_signals", "quant_data_quality")] == "MISROUTED", (
            "P170's exact bug is no longer flagged. The scanner is named for "
            "this shape; if it does not fire here it protects nothing."
        )

    def test_a_real_copy_across_clears_it(self, tree):
        # If somebody DOES copy the key across, the read is fine and must not
        # be reported — otherwise the fix for P170 would itself trip the gate.
        (tree / "consumer.py").write_text(
            'def wire(agent_signals, market_data):\n'
            '    agent_signals["quant_data_quality"] = market_data.get(\n'
            '        "quant_data_quality", 0.0)\n'
            'def decide(agent_signals):\n'
            '    return agent_signals.get("quant_data_quality", 1.0)\n',
            encoding="utf-8")
        kinds = _kinds(_scan(tree))
        assert kinds.get(("agent_signals", "quant_data_quality")) != "MISROUTED"

    def test_destination_is_what_distinguishes_the_two(self, tree):
        # Both reads below are of pipeline-produced keys with no static write.
        # The ONLY difference is which dict is read. If a future simplification
        # drops the dname check, these collapse to the same verdict.
        (tree / "consumer.py").write_text(
            'def f(market_data, agent_signals):\n'
            '    a = market_data.get("vpin_source", "synthetic")\n'
            '    b = agent_signals.get("vpin_source", "synthetic")\n'
            '    return a, b\n', encoding="utf-8")
        kinds = _kinds(_scan(tree))
        assert kinds[("market_data", "vpin_source")] == "PRODUCED_HERE"
        assert kinds[("agent_signals", "vpin_source")] == "MISROUTED", (
            "reading a market_data-produced key off agent_signals is P170. "
            "Same key, same default, different dict — the verdicts must differ."
        )


class TestTheDeclaredProducerMapIsHonest:
    """PRODUCER_MODULES is asserted architecture, so it can rot silently."""

    def test_every_declared_module_exists_and_still_produces(self):
        for (mod, func), dest in scanner.PRODUCER_MODULES.items():
            p = REPO_ROOT / mod
            assert p.exists(), f"PRODUCER_MODULES names {mod}, which is gone"
            tree = ast.parse(p.read_text(encoding="utf-8-sig"))
            if func == "*":
                keys = scanner.collect_produced_keys(tree)
            else:
                keys = scanner.collect_produced_by_function(tree).get(func, set())
                assert keys, (
                    f"{mod}::{func} no longer builds and returns a dict. The "
                    f"entry now credits nothing to '{dest}', which silently "
                    f"turns correct reads back into MISROUTED noise."
                )
            assert len(keys) > 3, f"{mod}::{func} produces suspiciously few keys"

    def test_the_declared_destinations_are_real_signal_dicts(self):
        for _k, dest in scanner.PRODUCER_MODULES.items():
            assert dest in scanner.TARGET_DICTS, (
                f"'{dest}' is not a tracked signal dict, so crediting keys to "
                f"it has no effect on any read"
            )

    def test_the_position_state_entry_still_covers_current_exposure(self):
        # 11 of the 14 findings in the intermediate run were
        # position_state.get("current_exposure"), all correct, all cleared by
        # this one entry. Losing it re-creates them.
        tree = ast.parse((REPO_ROOT / "main.py").read_text(encoding="utf-8-sig"))
        keys = scanner.collect_produced_by_function(tree).get(
            "_get_effective_position_state", set())
        assert "current_exposure" in keys


class TestTheCommittedBaselineMatchesReality:
    def test_baseline_equals_a_fresh_scan(self):
        r = subprocess.run(
            [sys.executable, "-X", "utf8", str(SCANNER), "--baseline-format"],
            capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout) == json.loads(
            BASELINE.read_text(encoding="utf-8")), (
            "the committed baseline is stale; CI is diffing against a number "
            "nobody reproduced"
        )

    def test_hot_is_zero_because_it_was_triaged_not_because_it_cannot_rise(self):
        b = json.loads(BASELINE.read_text(encoding="utf-8"))
        assert b["misrouted_hot_count"] == 0
        # and prove a HOT can still be produced — otherwise the 0 is P174 again
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "data_mgmt").mkdir()
            (d / "data_mgmt" / "market_data_pipeline.py").write_text(
                'def f():\n    raw = {"k": 1}\n    return raw\n', encoding="utf-8")
            (d / "c.py").write_text(
                'def g(agent_signals):\n'
                '    return agent_signals.get("k", 1.0)\n', encoding="utf-8")
            res = _scan(d)
        assert res["misrouted_hot_count"] >= 1, (
            "no input produces a HOT misroute, so the committed 0 is vacuous"
        )

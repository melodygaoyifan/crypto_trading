"""[P174] The P171 gate could not fail. These tests keep the replacement honest.

P171 shipped `orphan_count: 0` as a CI baseline and read it as a clean bill of
health. It was arithmetically forced. `main.py` copies signal keys in loops
(`for k, v in ...: agent_signals[k] = v`), which marks agent_signals,
market_data and position_state permanently dynamic, and every unmatched read of
those three is downgraded to UNPROVABLE before it can be counted. Measured: the
ORPHAN check adjudicated 0 of 458 unmatched reads.

That is the defect the scanner exists to find — a check that cannot fail is
indistinguishable from a check that passed (P155-L5, P156, P158, P159, P160,
P164, P166, P169, P170, P171) — shipped inside the tool built to find it.

The fix is not "make ORPHAN work": with a dynamic copy in the tree, absence is
genuinely unprovable, and pretending otherwise would trade a vacuous check for
an unsound one. The fix is to score what the scanner can prove, and to make the
vacuity visible where it remains:

  * `_is_null_coalesce` — `x = x or {}` writes no keys. Eight sites used it and
    each one poisoned a whole dict. Removing that false dynamism is what took
    UNPROVABLE from 427 to 77.
  * `collect_produced_keys` — the real market_data producer builds 51/55/89
    keys into a local and returns it. Crediting those killed the noise P171 had
    hand-waved away as "the pipeline fills `raw`" (which was a guess, and wrong
    about the mechanism).
  * `COPY_ONLY` — a key that only ever gets relayed between signal dicts, with
    no producer anywhere. It independently rediscovered `is_4h_bar_close`,
    which P173 had triaged by hand. That agreement is the evidence it works.
  * `orphan_coverage_lost` is emitted but INERT at coverage 0, and is asserted
    to be inert below rather than described as protection.
"""

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNER = REPO_ROOT / "tools" / "lint_orphan_signal_reads.py"

_spec = importlib.util.spec_from_file_location("_orphan_scanner", SCANNER)
scanner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scanner)


def _write(d: Path, name: str, body: str) -> Path:
    p = d / name
    p.write_text(body, encoding="utf-8")
    return p


def _scan(d: Path):
    return scanner.run([str(d)])


def _baseline(paths):
    r = subprocess.run(
        [sys.executable, "-X", "utf8", str(SCANNER), "--baseline-format",
         "--paths", *paths],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


class TestTheGateCanActuallyFail:
    """The whole point. Each gated metric must be reachable from clean."""

    def test_misrouted_rises_on_a_new_wrong_dict_read(self, tmp_path):
        _write(tmp_path, "producer.py", 'market_data["edge_bps"] = 12.0\n')
        before = _baseline([str(tmp_path)])["misrouted_count"]
        _write(tmp_path, "consumer.py", 'e = agent_signals.get("edge_bps", 50.0)\n')
        assert _baseline([str(tmp_path)])["misrouted_count"] == before + 1

    def test_copy_only_rises_on_a_relayed_key_with_no_producer(self, tmp_path):
        before = _baseline([str(tmp_path)])["copy_only_count"]
        _write(tmp_path, "relay.py",
               'agent_signals["ghost"] = market_data.get("ghost", 0.0)\n')
        assert _baseline([str(tmp_path)])["copy_only_count"] > before

    def test_dynamic_site_count_rises_when_the_tree_gets_less_analyzable(self, tmp_path):
        _write(tmp_path, "a.py", 'agent_signals["k"] = 1\n')
        before = _baseline([str(tmp_path)])["dynamic_site_count"]
        _write(tmp_path, "b.py", "for k, v in src.items():\n    agent_signals[k] = v\n")
        assert _baseline([str(tmp_path)])["dynamic_site_count"] > before, (
            "the blind spot must be scored, or every other metric can be "
            "driven to zero by making the code unanalyzable"
        )

    def test_orphan_still_rises_where_it_is_provable(self, tmp_path):
        _write(tmp_path, "r.py", 'q = system_state.get("nobody_writes_this", 1.0)\n')
        assert _baseline([str(tmp_path)])["orphan_count"] == 1

    def test_parse_failure_is_reachable(self, tmp_path):
        _write(tmp_path, "broken.py", "def f(:\n")
        assert _baseline([str(tmp_path)])["parse_failure_count"] == 1


class TestTheVacuityIsMeasuredNotAssumed:
    """P171's mistake was never measuring coverage. So measure it."""

    def test_a_dynamic_write_makes_orphan_unadjudicable(self, tmp_path):
        _write(tmp_path, "r.py", 'q = agent_signals.get("nobody", 1.0)\n')
        assert _scan(tmp_path)["orphan_adjudicable"] == 1
        _write(tmp_path, "dyn.py", "for k, v in s.items():\n    agent_signals[k] = v\n")
        res = _scan(tmp_path)
        assert res["orphan_adjudicable"] == 0
        assert res["orphan_count"] == 0
        assert res["by_kind"]["UNPROVABLE"] == 1, (
            "the finding must survive as UNPROVABLE, not vanish — a downgrade "
            "that also deletes the record is how the blind spot hid"
        )

    def test_the_real_repo_reports_its_own_vacuity_out_loud(self):
        r = subprocess.run([sys.executable, "-X", "utf8", str(SCANNER)],
                           capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert r.returncode == 0, r.stderr
        assert "ORPHAN coverage:" in r.stdout, (
            "coverage must be printed with the findings. P171 buried the blind "
            "spot in a footnote under the list, which is where it was ignored."
        )
        adjudicable = scanner.run([
            p.name for p in REPO_ROOT.iterdir()
            if (p.is_dir() and p.name not in scanner.EXCLUDED_DIRS
                and not p.name.startswith("."))
            or (p.is_file() and p.suffix == ".py")
        ])["orphan_adjudicable"]
        if adjudicable == 0:
            assert "VACUOUS" in r.stdout, (
                "ORPHAN adjudicates nothing in this tree and the output does "
                "not say so — that is exactly the P171 failure"
            )

    def test_coverage_lost_is_inert_at_zero_and_labelled_as_such(self):
        # Honest bookkeeping: this metric is at its floor, so it cannot rise
        # and cannot trip the gate. It is recorded, not relied upon. If someone
        # later reduces the dynamic writes, it arms itself.
        src = SCANNER.read_text(encoding="utf-8")
        i = src.index("orphan_coverage_lost —")
        assert "inert, not protective" in src[i:i + 700], (
            "orphan_coverage_lost must be documented as inert while coverage "
            "is 0; calling it a guard repeats P171's error one level up"
        )


class TestNullCoalesceIsNotAWrite:
    def test_x_or_empty_dict_does_not_mark_dynamic(self, tmp_path):
        _write(tmp_path, "r.py", 'q = market_data.get("nobody", 1.0)\n')
        _write(tmp_path, "norm.py", "def f(market_data=None):\n"
                                    "    market_data = market_data or {}\n"
                                    "    return market_data\n")
        assert _scan(tmp_path)["orphan_adjudicable"] == 1, (
            "`x = x or {}` adds no key and must not cost the dict its coverage"
        )

    def test_a_genuine_alias_still_marks_dynamic(self, tmp_path):
        _write(tmp_path, "r.py", 'q = market_data.get("nobody", 1.0)\n')
        _write(tmp_path, "alias.py", "market_data = fetch_it()\n")
        assert _scan(tmp_path)["orphan_adjudicable"] == 0, (
            "aliasing from a call CAN introduce keys; that blind spot is real "
            "and must not be optimised away with the false one"
        )

    @pytest.mark.parametrize("body", [
        "market_data = market_data or {'seeded': 1}",  # non-empty fallback
        "market_data = other or {}",                   # different name
        "market_data = market_data and {}",            # not `or`
    ])
    def test_lookalikes_are_not_exempted(self, tmp_path, body):
        _write(tmp_path, "r.py", 'q = market_data.get("nobody", 1.0)\n')
        _write(tmp_path, "x.py", body + "\n")
        assert _scan(tmp_path)["orphan_adjudicable"] == 0, (
            f"{body!r} is not the null-coalesce idiom and must stay dynamic"
        )


class TestHiddenProducersAreCredited:
    def test_a_returned_local_counts_as_a_producer(self, tmp_path):
        _write(tmp_path, "prod.py",
               "def build():\n"
               "    out = {}\n"
               "    out['spread_bps'] = 3.0\n"
               "    return out\n")
        _write(tmp_path, "cons.py", 'q = market_data.get("spread_bps", 0.0)\n')
        f = _scan(tmp_path)["findings"]
        assert [x["kind"] for x in f] == ["PRODUCED_ELSEWHERE"]

    def test_a_returned_dict_literal_counts(self, tmp_path):
        _write(tmp_path, "prod.py", "def build():\n    return {'vpin': 0.3}\n")
        _write(tmp_path, "cons.py", 'q = market_data.get("vpin", 0.35)\n')
        assert _scan(tmp_path)["findings"][0]["kind"] == "PRODUCED_ELSEWHERE"

    def test_a_local_that_is_never_returned_is_not_a_producer(self, tmp_path):
        _write(tmp_path, "prod.py",
               "def build():\n"
               "    scratch = {'spread_bps': 3.0}\n"
               "    log(scratch)\n")
        _write(tmp_path, "cons.py", 'q = system_state.get("spread_bps", 0.0)\n')
        assert _scan(tmp_path)["findings"][0]["kind"] == "ORPHAN", (
            "a dict that never leaves the function produces nothing for anyone"
        )

    def test_the_real_pipeline_is_recognised(self):
        # The concrete case P171 got wrong. If this file stops being credited,
        # 345 reads go back to looking unproduced.
        pipeline = REPO_ROOT / "data_mgmt" / "market_data_pipeline.py"
        assert pipeline.exists()
        produced = scanner.collect_produced_keys(
            ast.parse(pipeline.read_text(encoding="utf-8-sig", errors="replace"))
        )
        assert len(produced) > 100, (
            f"market_data_pipeline produced only {len(produced)} keys — the "
            "producer-detection regressed and MISROUTED will flood with noise"
        )
        for key in ("vpin_source", "data_valid"):
            assert key in produced, (
                f"{key} is read as market_data across the tree; if the pipeline "
                "no longer visibly produces it, that read is now a real suspect"
            )


class TestCopyOnlyFindsRelayedGhosts:
    def test_a_relay_is_not_a_producer(self, tmp_path):
        _write(tmp_path, "relay.py",
               'agent_signals["ghost"] = market_data.get("ghost", 0.0)\n')
        _write(tmp_path, "cons.py", 'q = agent_signals.get("ghost", 0.0)\n')
        kinds = {f["kind"] for f in _scan(tmp_path)["findings"]}
        assert kinds == {"COPY_ONLY"}, (
            "the relay write must not be counted as evidence the key exists; "
            "a copy is downstream of a producer, not a substitute for one"
        )

    def test_a_coercion_wrapper_is_transparent(self, tmp_path):
        # main.py:9374 writes `int(market_data.get("htf_trend_direction", 0))`.
        # Missing the cast cost this detector both of its findings.
        _write(tmp_path, "relay.py",
               'agent_signals["ghost"] = int(market_data.get("ghost", 0))\n')
        _write(tmp_path, "cons.py", 'q = agent_signals.get("ghost", 0)\n')
        assert {f["kind"] for f in _scan(tmp_path)["findings"]} == {"COPY_ONLY"}

    def test_a_real_computation_is_not_a_copy(self, tmp_path):
        _write(tmp_path, "calc.py",
               'agent_signals["squeeze_risk"] = max(a, b)\n')
        _write(tmp_path, "cons.py", 'q = market_data.get("squeeze_risk", 0.0)\n')
        assert {f["kind"] for f in _scan(tmp_path)["findings"]} == {"MISROUTED"}, (
            "`max(a, b)` measures something. This is the real squeeze_risk "
            "shape: a genuine producer on agent_signals, read off market_data."
        )

    def test_a_relay_with_a_real_producer_is_not_copy_only(self, tmp_path):
        _write(tmp_path, "prod.py", "def build():\n    return {'real': 1.0}\n")
        _write(tmp_path, "relay.py",
               'agent_signals["real"] = market_data.get("real", 0.0)\n')
        _write(tmp_path, "cons.py", 'q = agent_signals.get("real", 0.0)\n')
        assert "COPY_ONLY" not in {f["kind"] for f in _scan(tmp_path)["findings"]}

    def test_the_repo_copy_only_set_is_the_vetted_one(self):
        res = scanner.run([
            p.name for p in REPO_ROOT.iterdir()
            if (p.is_dir() and p.name not in scanner.EXCLUDED_DIRS
                and not p.name.startswith("."))
            or (p.is_file() and p.suffix == ".py")
        ])
        keys = {f["key"] for f in res["findings"] if f["kind"] == "COPY_ONLY"}
        assert keys == {"is_4h_bar_close", "htf_trend_direction"}, (
            f"COPY_ONLY membership changed: {sorted(keys)}. Both current "
            "members are triaged: is_4h_bar_close's True default is a "
            "documented assumption (P173, every caller sleeps to the 4H "
            "boundary), and htf_trend_direction has no producer anywhere so "
            "the [S11] authority-fusion input is permanently 0 — which "
            "authority_fusion.py documents as 'no data', the fail-safe value. "
            "A new member is a new suspect: triage it, do not re-baseline it."
        )


class TestFallbackChainsAreNotDefects:
    def test_a_chain_covering_the_producer_is_clean(self, tmp_path):
        _write(tmp_path, "prod.py", 'agent_signals["phase"] = "IGNITION"\n')
        _write(tmp_path, "cons.py",
               'p = market_data.get("phase", agent_signals.get("phase", "UNKNOWN"))\n')
        kinds = [f["kind"] for f in _scan(tmp_path)["findings"]]
        assert kinds == ["FALLBACK_CHAIN"], (
            "main.py:12086 is this shape and is correct; flagging it would "
            "have sent a reviewer to 'fix' working code"
        )

    def test_the_chain_is_reported_once_not_twice(self, tmp_path):
        _write(tmp_path, "prod.py", 'agent_signals["phase"] = "IGNITION"\n')
        _write(tmp_path, "cons.py",
               'p = market_data.get("phase", agent_signals.get("phase", "UNKNOWN"))\n')
        assert len(_scan(tmp_path)["findings"]) == 1

    def test_a_chain_that_misses_the_producer_is_still_misrouted(self, tmp_path):
        _write(tmp_path, "prod.py", 'position_state["phase"] = "IGNITION"\n')
        _write(tmp_path, "cons.py",
               'p = market_data.get("phase", agent_signals.get("phase", "UNKNOWN"))\n')
        assert [f["kind"] for f in _scan(tmp_path)["findings"]] == ["MISROUTED"], (
            "covering two wrong dicts is not better than covering one"
        )

    def test_hotness_comes_from_the_terminal_default(self, tmp_path):
        # The value that actually lands when every arm misses.
        _write(tmp_path, "c.py",
               'p = market_data.get("k", agent_signals.get("k", 1.0))\n')
        assert _scan(tmp_path)["findings"][0]["severity"] == "HOT"
        _write(tmp_path, "c.py",
               'p = market_data.get("k", agent_signals.get("k", 0.0))\n')
        assert _scan(tmp_path)["findings"][0]["severity"] == "COLD"


class TestTheCommittedBaselineMatchesReality:
    def test_baseline_is_current(self):
        committed = json.loads(
            (REPO_ROOT / "tools" / "scanner_baselines"
             / "orphan_signal_reads_baseline.json").read_text(encoding="utf-8")
        )
        r = subprocess.run(
            [sys.executable, "-X", "utf8", str(SCANNER), "--baseline-format"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout) == committed

    def test_the_gate_no_longer_rests_on_a_single_forced_zero(self):
        committed = json.loads(
            (REPO_ROOT / "tools" / "scanner_baselines"
             / "orphan_signal_reads_baseline.json").read_text(encoding="utf-8")
        )
        movable = [k for k, v in committed.items()
                   if k != "orphan_coverage_lost" and v > 0]
        assert len(movable) >= 2, (
            f"only {movable} are non-zero — a baseline of all zeros is what "
            "P171 looked like right before it turned out to mean nothing"
        )

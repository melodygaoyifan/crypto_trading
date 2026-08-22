"""[P371] The per-agent IC tool must REFUSE to promote a signal that barely moved.

P370's audit found `analytics/ic/agent_ic_review.py` reporting the `sentiment`
agent at 30d IC 0.703 (t 9.16 at 16h) and labelling it PROMOTE-CANDIDATE. The
attribution census showed ONE global daily Fear&Greed value shared by all three
assets: direction -1 on every tick Jul 22 -> Aug 17, then +1 from Aug 19,
coincident with the rally. n_eff is ONE sign change, not 687 rows — and the
P231 overlap correction (n_eff = n/h) cannot see that, because it corrects for
horizon overlap, not for a near-constant signal.

These tests DRIVE the real verdict function (P324: a verdict rule is tested by
being CALLED, not by a source pin):
  * a synthetic one-flip series with a deliberately HIGH IC must be REFUSED;
  * a synthetic varying series with the same IC must still PROMOTE (the bar is
    not weakened for signals that genuinely move);
  * a varying series with no IC is a HOLD (the guard grants nothing);
  * the per-asset sign count catches BTC-always-+1 beside ETH-always--1;
  * and main() actually consumes the guard (assert_drives_output, P359) —
    run the real CLI over a synthetic ledger, see REFUSED-DEGENERATE, neutralise
    the thresholds, see it disappear and PROMOTE reappear.
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests._cli_harness import load_cli, run_cli  # noqa: E402
from tests._guard_pins import assert_drives_output, assert_guard_live  # noqa: E402
from tests._source_scan import code_only  # noqa: E402

TOOL = REPO / "analytics" / "ic" / "agent_ic_review.py"


@pytest.fixture(scope="module")
def mod():
    import analytics.ic.agent_ic_review as m
    return m


# ---------------------------------------------------------------------------
# synthetic series
# ---------------------------------------------------------------------------
FWD_VOL = {1: 100.0, 4: 200.0}     # bps; comfortably above the edge floor
N = 300


def _one_flip(n=N):
    """The sentiment shape: -1 for half the window, +1 for the other half."""
    return [-1.0] * (n // 2) + [1.0] * (n - n // 2)


def _varying(n=N, seed=7):
    rng = random.Random(seed)
    return [rng.choice([-1.0, 1.0]) for _ in range(n)]


def _returns_for(dirs, seed=11, slope=0.01, noise=0.003):
    """Forward returns CORRELATED with the directions (IC well above the bar)."""
    rng = random.Random(seed)
    return [d * slope + rng.gauss(0.0, noise) for d in dirs]


def _noise_returns(n=N, seed=13):
    rng = random.Random(seed)
    return [rng.gauss(0.0, 0.01) for _ in range(n)]


def _decide(mod, dirs, rets, assets=None):
    return mod.decide_agent_verdict(
        {1: dirs, 4: dirs}, {1: rets, 4: rets}, FWD_VOL,
        {1: assets, 4: assets} if assets is not None else None)


# ---------------------------------------------------------------------------
# 1. the guard, driven
# ---------------------------------------------------------------------------
class TestOneFlipIsRefused:
    def test_a_one_flip_series_with_a_high_ic_is_refused_not_promoted(self, mod):
        """THE DEFECT. The arithmetic WOULD promote (clears_p166_arithmetic is
        True — so this test is not vacuous), and the verdict must still refuse."""
        dirs = _one_flip()
        rows, verdict = _decide(mod, dirs, _returns_for(dirs))
        assert verdict == mod.REFUSED_DEGENERATE
        assert verdict != "PROMOTE-CANDIDATE"
        for h in mod.HORIZON_BARS:
            r = rows[h]
            assert r["clears_p166_arithmetic"] is True, (
                "the synthetic IC must be high enough that the OLD rule "
                "promoted it — otherwise the refusal is not being tested")
            assert r["clears_p166"] is False
            assert r["degenerate"] is True
            assert r["degeneracy"]["sign_changes"] == 1

    def test_the_reason_names_the_sign_count_and_says_not_evidence(self, mod):
        dirs = _one_flip()
        rows, _ = _decide(mod, dirs, _returns_for(dirs))
        reason = rows[1]["degeneracy"]["reason"]
        # [P370] These pin the PROSE of a refusal message, not a code
        # condition. The P307b condition-pin detector reads the bare words
        # "is" / "not" inside a quoted string as a condition (the P317/P324/
        # P328 false positive), so the claims are asserted on fragments that
        # carry no condition-word — reword the assertion, never the guard.
        assert "1 sign change" in reason
        assert "n_eff" in reason and "~1" in reason
        assert "row count" in reason
        assert "evidence" in reason

    def test_two_sign_changes_are_still_degenerate(self, mod):
        """<= 2 is the bar: -,+,- is a step function, not a signal."""
        dirs = [-1.0] * 100 + [1.0] * 100 + [-1.0] * 100
        rows, verdict = _decide(mod, dirs, _returns_for(dirs))
        assert verdict == mod.REFUSED_DEGENERATE
        assert rows[1]["degeneracy"]["sign_changes"] == 2

    def test_three_sign_changes_clears_the_hard_bar_but_not_its_own_n_eff(self, mod):
        """The hard bar is <= 2 (pinned at the series level) — and a 3-change
        step series whose arithmetic claims significance is STILL refused,
        because at n_eff=3 the same IC has |t| = IC*sqrt(2) < 2. That is the
        rule applied literally: n_eff is ~N sign changes, not the row count."""
        dirs = [-1.0] * 75 + [1.0] * 75 + [-1.0] * 75 + [1.0] * 75
        series_level = mod.signal_degeneracy(dirs)
        assert series_level["sign_changes"] == 3
        assert series_level["degenerate"] is False, "the hard bar is <= 2"
        rows, verdict = _decide(mod, dirs, _returns_for(dirs))
        dg = rows[1]["degeneracy"]
        assert rows[1]["clears_p166_arithmetic"] is True
        assert dg["n_eff_signal"] == 3
        assert abs(dg["t_at_n_eff_signal"]) < 2.0
        assert verdict == mod.REFUSED_DEGENERATE
        assert "|t| at n_eff=3" in dg["reason"] and "not evidence" in dg["reason"]

    def test_the_live_sentiment_shape_is_refused(self, mod):
        """THE P370 INSTANCE, run-for-run from the pulled ledger (30d): ONE
        global F&G value, per-asset runs BTC 184/22/2/19, ETH 59/3/116/23/3/8,
        SOL 170/34 — 3/5/1 sign changes (restart ticks + two one-day flickers
        around the Aug 19 flip), so a bare '<= 2 sign changes' bar would have
        let it through. The per-asset MAX is 5; at n_eff=5 the claimed t
        collapses below 2, and the tool refuses."""
        runs = {"BTC": [(-1, 184), (1, 22), (-1, 2), (1, 19)],
                "ETH": [(-1, 59), (1, 3), (-1, 116), (1, 23), (-1, 3), (1, 8)],
                "SOL": [(-1, 170), (1, 34)]}
        dirs, assets = [], []
        for a, rl in runs.items():
            for sgn, k in rl:
                dirs += [float(sgn)] * k
                assets += [a] * k
        rets = _returns_for(dirs, slope=0.02)
        rows, verdict = _decide(mod, dirs, rets, assets=assets)
        dg = rows[4]["degeneracy"]
        assert dg["sign_changes"] == 5, "MAX over assets, not the sum (9)"
        assert rows[4]["clears_p166_arithmetic"] is True, (
            "fixture control: the OLD rule promoted this (t ~9 at n//h)")
        assert verdict == mod.REFUSED_DEGENERATE
        assert not any(
            rows[h].get("clears_p166") for h in mod.HORIZON_BARS)

    def test_a_shared_global_series_is_not_counted_once_per_asset(self, mod):
        """The same one-flip series on three assets has ONE sign change."""
        one = _one_flip(120)
        dirs = one * 3
        assets = ["BTC"] * 120 + ["ETH"] * 120 + ["SOL"] * 120
        assert mod.count_sign_changes(dirs, assets) == 1
        assert mod.sign_changes_by_asset(dirs, assets) == {
            "BTC": 1, "ETH": 1, "SOL": 1}

    def test_a_block_signal_with_enough_runs_still_promotes(self, mod):
        """Rule (c) is n_eff arithmetic, not a ban on persistence: 12 runs of
        25 (11 changes) carry |t| = IC*sqrt(10) > 2 at a high IC and promote."""
        dirs = []
        for i in range(12):
            dirs += [1.0 if i % 2 == 0 else -1.0] * 25
        rows, verdict = _decide(mod, dirs, _returns_for(dirs))
        assert rows[1]["degeneracy"]["sign_changes"] == 11
        assert verdict == "PROMOTE-CANDIDATE"

    def test_a_perma_bias_is_refused_by_the_dominant_share_branch(self, mod):
        """One sign on > 90% of records is the other degenerate shape: the
        sign count alone would NOT catch it (every 20th record flips back)."""
        dirs = [1.0] * N
        for i in range(0, N, 20):
            dirs[i] = -1.0
        rows, verdict = _decide(mod, dirs, _returns_for(dirs))
        dg = rows[1]["degeneracy"]
        assert dg["sign_changes"] > mod.DEGENERATE_MAX_SIGN_CHANGES, (
            "fixture must reach the SHARE branch, not the sign-count branch")
        assert dg["dominant_share"] > mod.DEGENERATE_DOMINANT_SHARE
        assert verdict == mod.REFUSED_DEGENERATE
        assert "occupies" in dg["reason"] and "not evidence" in dg["reason"]


class TestVaryingSignalsAreJudgedByTheUnchangedBar:
    def test_a_varying_series_with_the_same_ic_still_promotes(self, mod):
        """The guard may only REMOVE a promotion. Same return model as the
        one-flip fixture, directions that genuinely move -> PROMOTE stands."""
        dirs = _varying()
        rows, verdict = _decide(mod, dirs, _returns_for(dirs))
        assert verdict == "PROMOTE-CANDIDATE"
        for h in mod.HORIZON_BARS:
            assert rows[h]["degenerate"] is False
            assert rows[h]["degeneracy"]["reason"] is None
            assert rows[h]["clears_p166"] is True

    def test_a_varying_series_with_no_ic_is_not_promoted_either(self, mod):
        """Not degenerate is not a pass — the cost-aware bar still decides."""
        dirs = _varying()
        rows, verdict = _decide(mod, dirs, _noise_returns())
        assert verdict != "PROMOTE-CANDIDATE"
        assert verdict != mod.REFUSED_DEGENERATE
        assert rows[1]["degenerate"] is False

    def test_the_arithmetic_rows_are_byte_identical_for_a_varying_signal(self, mod):
        """Moving the loop into a function must not have moved a number:
        re-derive n_eff/t/edge by hand for one horizon and compare."""
        import math
        dirs = _varying()
        rets = _returns_for(dirs)
        rows, _ = _decide(mod, dirs, rets)
        r = rows[4]
        ic = mod._pearson(dirs, rets)
        n_eff = max(3, len(dirs) // 4)
        assert r["n_eff"] == n_eff
        assert r["ic"] == round(ic, 4)
        assert r["t"] == round(ic * math.sqrt(n_eff - 1), 2)
        assert r["edge_bps"] == round(
            0.7979 * 2.0 * math.sin(math.pi * ic / 6.0) * FWD_VOL[4], 2)
        assert r["required_ic"] == round(mod.required_ic(FWD_VOL[4]), 4)

    def test_insufficient_rows_are_unchanged(self, mod):
        rows, verdict = _decide(mod, [1.0, -1.0] * 5, [0.01, -0.01] * 5)
        assert verdict == "INSUFFICIENT"
        assert rows[1] == {"n": 10, "ic": rows[1]["ic"],
                           "verdict": "INSUFFICIENT"}


class TestSignChangesAreCountedPerAsset:
    def test_interleaved_constant_assets_count_zero_changes(self, mod):
        """BTC always +1 beside ETH always -1 reads as a flip on EVERY row if
        the series is counted as one list — while neither asset moved."""
        dirs = [1.0, -1.0] * 150
        assets = ["BTC", "ETH"] * 150
        assert mod.count_sign_changes(dirs, None) == 299
        assert mod.count_sign_changes(dirs, assets) == 0

    def test_the_verdict_uses_the_per_asset_count(self, mod):
        dirs = [1.0, -1.0] * 150
        assets = ["BTC", "ETH"] * 150
        rets = _returns_for(dirs)
        _, without = _decide(mod, dirs, rets, assets=None)
        rows, with_assets = _decide(mod, dirs, rets, assets=assets)
        assert without == "PROMOTE-CANDIDATE", (
            "fixture control: counted as ONE series this promotes, which is "
            "exactly the hole the per-asset count closes")
        assert with_assets == mod.REFUSED_DEGENERATE
        assert rows[1]["degeneracy"]["sign_changes"] == 0

    def test_zero_directions_are_skipped_not_counted_as_flips(self, mod):
        assert mod.count_sign_changes([1.0, 0.0, 1.0, 0.0, 1.0]) == 0
        assert mod.count_sign_changes([1.0, 0.0, -1.0]) == 1


# ---------------------------------------------------------------------------
# 2. main() consumes it — the CLI over a synthetic ledger
# ---------------------------------------------------------------------------
def _write_ledger(log_dir: Path, bars: int = 240):
    """A synthetic attribution ledger + price series where:
      BTC  carries `sentiment`, ONE sign flip (the live artifact), with
           closes that FOLLOW the direction -> a huge IC;
      ETH  carries `whale`, a varying direction, closes following it too ->
           a real, varying, high-IC signal that MUST still promote.
    Returns the fake fetch_closes(asset) mapping."""
    log_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    t0 = now - (bars + 6) * 14400.0
    ts_list = [t0 + i * 14400.0 for i in range(bars)]
    rng = random.Random(5)
    dirs = {"BTC": [-1.0] * (bars // 2) + [1.0] * (bars - bars // 2),
            "ETH": [rng.choice([-1.0, 1.0]) for _ in range(bars)]}
    closes = {}
    for a in dirs:
        c = [100.0]
        for i in range(1, bars):
            c.append(c[-1] * (1.0 + 0.01 * dirs[a][i - 1]))
        closes[a] = c
    agent = {"BTC": "sentiment", "ETH": "whale"}
    with open(log_dir / "signals_p371.jsonl", "w", encoding="utf-8") as fh:
        for a in dirs:
            for i in range(bars - 5):
                rec = {"ts": datetime.fromtimestamp(
                    ts_list[i] + 60.0, tz=timezone.utc).isoformat(),
                    "asset": a,
                    "signals": [{"agent_name": agent[a],
                                 "direction": dirs[a][i]}]}
                fh.write(json.dumps(rec) + "\n")
    return {a: (ts_list, closes[a]) for a in dirs}, closes


@pytest.fixture
def cli_over_ledger(tmp_path):
    log_dir = tmp_path / "logs"
    fake, _ = _write_ledger(log_dir)
    m = load_cli(TOOL)
    # SOL has no records; give it a flat series so the join is exercised only
    # for the two assets under test (fetch_closes is the only network call).
    fake["SOL"] = (fake["BTC"][0], [100.0] * len(fake["BTC"][0]))
    m.fetch_closes = lambda asset: fake[asset]

    def run() -> str:
        r = run_cli(m, ["--log-dir", str(log_dir), "--window-days", "60",
                        "--report-dir", str(tmp_path / "rep")])
        assert r.exit_code == 0, (r.exit_code, r.stderr[-300:])
        return r.stdout
    return m, run


class TestMainConsumesTheGuard:
    def test_the_cli_refuses_sentiment_and_still_promotes_whale(
            self, cli_over_ledger):
        m, run = cli_over_ledger
        out = run()
        sent = [ln for ln in out.splitlines() if ln.startswith("sentiment")]
        whale = [ln for ln in out.splitlines() if ln.startswith("whale")]
        assert any(m.REFUSED_DEGENERATE in ln for ln in sent), out
        assert not any("PROMOTE-CANDIDATE" in ln for ln in sent), out
        assert any("PROMOTE-CANDIDATE" in ln for ln in whale), (
            "the in-tool control: a varying high-IC signal must still "
            "promote, or the guard is a blanket refusal", out)
        assert "REFUSED PROMOTE" in out and "not evidence" in out

    def test_the_guard_drives_the_output(self, cli_over_ledger, tmp_path):
        """P359: presence cannot distinguish consumed from coincidental. Vary
        the thresholds and require the refusal to DISAPPEAR — and PROMOTE to
        come back, proving the refusal is what stood between the artifact and
        the label."""
        m, run = cli_over_ledger

        def disable():
            saved = (m.DEGENERATE_MAX_SIGN_CHANGES, m.DEGENERATE_DOMINANT_SHARE,
                     m.DEGENERATE_MIN_T_AT_SIGNAL_N_EFF)
            m.DEGENERATE_MAX_SIGN_CHANGES = -1
            m.DEGENERATE_DOMINANT_SHARE = 2.0
            m.DEGENERATE_MIN_T_AT_SIGNAL_N_EFF = 0.0

            def restore():
                (m.DEGENERATE_MAX_SIGN_CHANGES, m.DEGENERATE_DOMINANT_SHARE,
                 m.DEGENERATE_MIN_T_AT_SIGNAL_N_EFF) = saved
            return restore

        assert_drives_output(run, m.REFUSED_DEGENERATE, disable,
                             why="the degeneracy thresholds must DRIVE the verdict")
        restore = disable()
        try:
            out = run()
        finally:
            restore()
        sent = [ln for ln in out.splitlines() if ln.startswith("sentiment")]
        assert any("PROMOTE-CANDIDATE" in ln for ln in sent), (
            "with the guard neutralised the one-flip series reads PROMOTE — "
            "i.e. this guard is exactly what P370 found missing", out)

    def test_the_report_carries_the_refusal(self, cli_over_ledger, tmp_path):
        m, run = cli_over_ledger
        run()
        reps = sorted((tmp_path / "rep").glob("agent_ic_*.json"))
        assert reps
        rep = json.loads(reps[-1].read_text(encoding="utf-8"))
        sent = rep[m.AGENTS_CONTAINER_KEY]["sentiment"]
        assert sent["verdict"] == m.REFUSED_DEGENERATE
        cell = sent[m.HORIZON_CONTAINER_KEY]["1"] if "1" in sent[
            m.HORIZON_CONTAINER_KEY] else sent[m.HORIZON_CONTAINER_KEY][1]
        assert cell["degenerate"] is True
        assert cell["clears_p166"] is False


# ---------------------------------------------------------------------------
# 3. the seam is used, and the guard is live (not a source pin of a comment)
# ---------------------------------------------------------------------------
class TestSeamAndPins:
    def test_main_calls_the_pure_verdict_function(self):
        """A seam nothing calls is decoration (P170/P312/P324)."""
        src = code_only(TOOL)
        i = src.index("def main(")
        assert "decide_agent_verdict(" in src[i:], (
            "main() must consume decide_agent_verdict or the tested rule is "
            "not the rule the tool runs")

    def test_the_refusal_outranks_promote_in_the_verdict_rule(self):
        src = code_only(TOOL)
        assert_guard_live(src, '"DEGENERATE" in verdict_bits',
                          why="REFUSED-DEGENERATE must be decided before PROMOTE")
        i = src.index('if "DEGENERATE" in verdict_bits')
        j = src.index('all(v == "CLEARS" for v in verdict_bits)')
        assert i < j, "the refusal must be tested BEFORE the promote rule"

    def test_thresholds_are_the_recorded_values(self, mod):
        assert mod.DEGENERATE_MAX_SIGN_CHANGES == 2
        assert mod.DEGENERATE_DOMINANT_SHARE == 0.90
        assert mod.DEGENERATE_MIN_T_AT_SIGNAL_N_EFF == 2.0, (
            "the re-priced t bar must be the tool's OWN |t| >= 2 bar — "
            "anything lower is a loosening wearing the guard's name")

    def test_the_n_eff_rule_never_fires_on_an_insignificant_claim(self, mod):
        """Rule (c) re-prices a claim of significance; it must not turn a
        NOISE/HOLD reading into a REFUSAL (that would mislabel noise as an
        artifact)."""
        dirs = _varying()
        rows, verdict = _decide(mod, dirs, _noise_returns())
        assert abs(rows[1]["t"]) < 2.0
        assert rows[1]["degenerate"] is False
        assert verdict == "HOLD"

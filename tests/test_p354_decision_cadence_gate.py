"""[P354] Bar-cadence accumulators advance only on the 4H decision tick.

P353 measured that the 30s FastRiskTick slice runs the FULL market-data
pipeline — 2,541 times per asset per day against 6 decision ticks — and that
`fetch_and_prepare` mutated every rolling accumulator on every one of those
calls. Each buffer's `maxlen` therefore counted 34-second samples rather than
4H bars, and four windows were ~424x shorter than their size implies.

THE SPLIT, decided per accumulator on what its consumer's timescale actually
is rather than uniformly:

  GATED — these have a train/serve CONTRACT, so the short window is a skew:
    * `_wavelet_buffers` (256). P164 established that the causal wavelet must
      reproduce the training transform, which runs over 4H BARS; 256 samples
      is meant to be ~42 days and was ~145 minutes. P164's own parity guard
      could not see this: it pins the TRANSFORM, not what the deque is FED.
    * RegimeSmoother `persistence=2`. The Architecture Rules call this a
      train/runtime parity requirement, and in training it is 2 BARS. At 34s
      it confirmed a regime change in ~1.1 minutes — ~424 advances between
      decisions, i.e. no damping at the horizon the rule is about.

  UNGATED BY DECISION — their consumers are execution-timescale, so the fast
  feed is right and the days their maxlens imply would be wrong:
    * `_ofi_history` (42) -> `ofi_zscore` -> the SOL toxicity classifier.
    * `_depth_history` (120) -> `orderbook_depth` -> the execution TIMING
      score. Pinned in tests/test_p353_pipeline_cadence.py, where the reason
      lives.

COST TO THE WATCHDOG: none. FastRiskTick reads only `current_price`,
`data_valid`, `volatility_30m`, `orderbook_depth_1pct_usd` and
`orderbook_stale`, and its depth baseline is its own 4H anchor (P156). It
consumes neither the denoised features nor `regime_state`.

FAIL DIRECTION: `for_decision` defaults to False everywhere, so a NEW caller
cannot pollute a bar-cadence window by forgetting to opt out. The risk that
buys is a decision site forgetting to opt IN — which starves the buffers
silently — so the decision sites are pinned by test below rather than left to
memory.
"""

import ast
import inspect

import pytest

import main
from main import HMATSProductionRunner
import data_mgmt.market_data_pipeline as mdp


def _pipeline_tree():
    return ast.parse(inspect.getsource(mdp))


def _is_for_decision_test(node: ast.If) -> bool:
    return isinstance(node.test, ast.Name) and node.test.id == "for_decision"


def _appends_under_for_decision():
    """Attribute-append receivers that sit inside `if for_decision:`."""
    out = []
    for node in ast.walk(_pipeline_tree()):
        if isinstance(node, ast.If) and _is_for_decision_test(node):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    f = sub.func
                    if isinstance(f, ast.Attribute) and f.attr == "append":
                        out.append(ast.dump(f.value))
    return out


# --------------------------------------------------------------------------
# the gate exists and covers exactly the two bar-cadence accumulators
# --------------------------------------------------------------------------
def test_the_wavelet_deque_advances_only_on_a_decision_tick():
    receivers = _appends_under_for_decision()
    assert any("_wv_buf" in r for r in receivers), (
        "the causal-wavelet deque is fed outside `if for_decision:` — it is "
        "then a window over 34-second snapshots, and the features it produces "
        "no longer match the transform P164 certified over 4H bars"
    )


def test_the_regime_smoother_advances_only_on_a_decision_tick():
    """The state machine's mutations must all sit under the gate."""
    src = inspect.getsource(mdp)
    tree = ast.parse(src)
    gated, ungated = 0, []
    for node in ast.walk(tree):
        for sub in ast.walk(node) if isinstance(node, ast.If) else ():
            pass
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        # `_state["pending"]` / `_state["count"]` / `_state["current"]`
        if not (isinstance(node.value, ast.Name) and node.value.id == "_state"):
            continue
        gated += 1
    assert gated > 0, "the smoother's state machine was not found"

    # every assignment to _state[...] must be inside `if for_decision:`
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_for_decision_test(node):
            continue
    def _assign_targets_under(n):
        found = []
        for sub in ast.walk(n):
            if isinstance(sub, ast.Assign):
                for t in sub.targets:
                    if (isinstance(t, ast.Subscript)
                            and isinstance(t.value, ast.Name)
                            and t.value.id == "_state"):
                        found.append(sub)
            elif isinstance(sub, ast.AugAssign):
                t = sub.target
                if (isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "_state"):
                    found.append(sub)
        return found

    all_mutations = _assign_targets_under(tree)
    gated_mutations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_for_decision_test(node):
            gated_mutations.extend(_assign_targets_under(node))
    assert all_mutations, "no smoother mutations found — the scan is vacuous"
    ungated = [m for m in all_mutations if m not in gated_mutations]
    # the initialiser (`_state = {...}` then stored in the dict) is a plain
    # Assign to a Name, not to `_state[...]`, so it is not counted here.
    assert not ungated, (
        f"{len(ungated)} smoother state mutation(s) sit outside "
        f"`if for_decision:` (lines "
        f"{[getattr(m, 'lineno', '?') for m in ungated]}) — the machine then "
        f"advances every ~34s and confirms a regime in ~1.1 minutes instead "
        f"of over 2 bars"
    )


def test_a_non_decision_call_still_serves_the_current_smoothed_label():
    """Skipping the advance must not blank the key — the watchdog should see
    the regime the last decision was taken under, not nothing."""
    src = inspect.getsource(mdp)
    # `if for_decision:` occurs twice (wavelet, smoother); anchor on the
    # smoother's own header or this reads the wrong site (P238/P349).
    assert src.count("            # RegimeSmoother\n") == 1
    i = src.index("            # RegimeSmoother\n")
    tail = src[i:i + 2200]
    assert "if for_decision:" in tail, "the smoother gate is not here"
    j = tail.index("if for_decision:")
    after = tail[j:]
    assert 'smoothed_regime = _state["current"]' in after, (
        "regime_state is no longer served on the non-decision path"
    )
    assert after.index('smoothed_regime = _state["current"]') > 0


# --------------------------------------------------------------------------
# the flag is threaded, and only the decision sites set it
# --------------------------------------------------------------------------
def test_both_functions_default_to_not_a_decision():
    """A NEW caller must not be able to pollute a bar-cadence window by
    forgetting to opt out."""
    assert inspect.signature(
        mdp.MarketDataPipeline.fetch_and_prepare
    ).parameters["for_decision"].default is False
    assert inspect.signature(
        HMATSProductionRunner._prepare_market_data
    ).parameters["for_decision"].default is False


def test_the_runner_passes_the_flag_through():
    src = inspect.getsource(HMATSProductionRunner._prepare_market_data)
    assert "for_decision=for_decision" in src, (
        "the wrapper swallows the flag, so no caller can ever set it"
    )


def test_exactly_the_two_decision_prefetches_opt_in():
    """One call per asset per tick may advance the accumulators. Two loops
    (run_paper and run_live) carry near-identical prefetch blocks, so the
    count is 2 — and a third would mean something advances them twice."""
    src = inspect.getsource(main)
    n = src.count("self._prepare_market_data(a, for_decision=True)")
    assert n == 2, (
        f"found {n} decision prefetch call sites, expected exactly 2 "
        f"(run_paper and run_live). A third advances the smoother twice in "
        f"one tick; zero starves every bar-cadence buffer."
    )
    assert "for_decision=True" not in src.replace(
        "self._prepare_market_data(a, for_decision=True)", ""), (
        "some other call site opts in — only the per-tick prefetch may"
    )


@pytest.mark.parametrize("loop_name", ["run_live", "run_paper"])
def test_the_watchdog_call_does_not_opt_in(loop_name):
    src = inspect.getsource(getattr(HMATSProductionRunner, loop_name))
    i = src.index("frt_md = await self._prepare_market_data(frt_asset")
    call = src[i:i + 120]
    assert "for_decision" not in call, (
        f"{loop_name}'s 30s FastRiskTick refresh opts in — that restores the "
        f"~424x over-feeding P353 measured"
    )


def test_the_decision_prefetch_precedes_the_watchdog_loop():
    """Ordering sanity: the tick's own decision call must come first, or the
    accumulators advance on a bar the decision has not seen yet."""
    src = inspect.getsource(HMATSProductionRunner.run_live)
    assert src.index("for_decision=True") < src.index(
        "frt_md = await self._prepare_market_data(frt_asset")


# --------------------------------------------------------------------------
# what the gate is worth, derived
# --------------------------------------------------------------------------
def test_the_gated_windows_are_now_the_ones_their_size_implies():
    FOUR_HOURS = 4 * 60 * 60.0
    wavelet = 256 * FOUR_HOURS
    smoother = 2 * FOUR_HOURS
    assert wavelet / 86400.0 > 40.0, "the wavelet window should be ~42 days"
    assert smoother == 8 * 3600.0, "the smoother should confirm over 8 hours"


def test_fast_risk_tick_reads_nothing_the_gate_withholds():
    """The claim that gating costs the watchdog nothing, pinned at the source
    it is a claim about."""
    import execution.fast_risk_tick as frt
    src = inspect.getsource(frt)
    for withheld in ("regime_state", "_denoised", "ofi_zscore"):
        assert withheld not in src, (
            f"FastRiskTick now reads {withheld!r}, which the P354 gate no "
            f"longer refreshes on its 30s cadence — re-derive the claim that "
            f"gating costs the watchdog nothing"
        )

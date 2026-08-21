"""[P353] Every rolling window in the market-data pipeline is ~424x shorter
than the bars its size implies, because the 30s watchdog drives the FULL
pipeline.

`run_live` sleeps to the next 4H candle in 30-second chunks so FastRiskTick
can act between decisions (P110/P227). On every chunk it calls
`self._prepare_market_data(frt_asset)` for all three assets — the WHOLE
pipeline: Kraken OHLC, 1000 trades for VPIN, Best-of-N selection, GMM
inference. Measured over the retained logs:

    asset      n   median gap   per day
    BTC    14773        34.0s      2541
    ETH    14771        34.0s      2541
    SOL    14774        34.0s      2541

2,541 full pipeline runs per asset per day, against 6 decision ticks.

`fetch_and_prepare` mutates its accumulators on EVERY call — there is no
tick_count gate anywhere — so each buffer's maxlen counts 34-second samples,
not 4H bars:

    buffer                      maxlen   actual      implied by "bars"
    _ofi_history                    42   23.8 min    7 days
    _depth_history                 120   68.0 min    20 days
    _wavelet_buffers               256  145.1 min    42.7 days
    RegimeSmoother persistence       2    1.1 min    8 hours

Verified live, not inferred: the persisted ofi_history buffer's contents
changed for all three assets within 80 seconds, between two 4H ticks.

WHAT THIS FALSIFIES, in the repo's own record:

  * P316 reasoned that `_ofi_history` is "a per-process deque needing >= 5
    samples at ONE sample per 4H tick = ~20 HOURS of uninterrupted uptime",
    and P301/P302's persistence work was motivated by the same premise. It
    fills in about 2.5 MINUTES.
  * The Architecture Rules say "RegimeSmoother persistence=2 must match
    training + runtime (prevents regime flip ping-pong)". In training the
    persistence is 2 BARS; at serve it is 2 x 34s, so by the time a 4H tick
    reads `regime_state` the smoother has advanced ~424 times and provides no
    damping at the decision horizon at all. Measured: 52 suppressions across
    ~44,000 pipeline calls, while BTC's GMM label spans five distinct regimes.
  * P164's parity guard (`test_causal_matches_the_live_recurrence_exactly`)
    proves the causal wavelet TRANSFORM reproduces the training one bar for
    bar. It says nothing about what the runtime deque is FED, and the deque is
    fed 34-second snapshots — the P234 shape inside a leak guard.

BLAST RADIUS, traced rather than assumed: the five `*_denoised` features reach
only the DRL observation builder and the Exit-SAC agent, both SHADOW, so no
live order depends on them — but both shadow IC streams are confounded.
`ofi_zscore` reaches the SOL toxicity classifier; `_depth_history` feeds the
depth median and collapse detector that FastRiskTick itself alerts on; and
`regime_state` is consumed broadly and live (ADVISE weights, smart beta,
kraken_quant buckets, the trend gate).

NOT FIXED HERE, deliberately. The four accumulators serve two consumers with
genuinely different cadence needs — a 30-second watchdog and a 4-hourly
decision — so choosing a window per accumulator is a live-behaviour decision
with effects in both directions, not a bugfix. These tests pin the facts the
arithmetic rests on so the next reader re-derives rather than inherits.
"""

import ast
import inspect
import re

import pytest

import main
from main import HMATSProductionRunner
import data_mgmt.market_data_pipeline as mdp
from tests._guard_pins import assert_text_pin

FOUR_HOURS = 4 * 60 * 60.0
# The measured per-asset gap: the 30s sleep chunk plus three sequential
# full-pipeline fetches.
MEASURED_GAP_SEC = 34.0


# --------------------------------------------------------------------------
# the premise
# --------------------------------------------------------------------------
def test_the_inter_tick_watchdog_drives_the_full_pipeline():
    """If this call ever becomes a light price-only read, every number in the
    P353 entry changes and the entry must be re-derived."""
    src = inspect.getsource(HMATSProductionRunner.run_live)
    assert_text_pin(
        src, "frt_md = await self._prepare_market_data(frt_asset)",
        why="the 30s FastRiskTick slice runs the WHOLE market-data pipeline, "
            "which is what makes every rolling window a window over 34-second "
            "samples instead of over 4H bars",
    )


def test_the_sleep_chunk_is_thirty_seconds():
    src = inspect.getsource(HMATSProductionRunner.run_live)
    assert_text_pin(
        src, "sleep_chunk = min(30.0, remaining)",
        why="the cadence the P353 arithmetic rests on",
    )


def _accumulator_appends():
    """(name, ast node) for each rolling-buffer mutation in fetch_and_prepare."""
    tree = ast.parse(inspect.getsource(mdp))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "append"):
            continue
        seg = ast.dump(f.value)
        for name in ("_ofi_history", "_depth_history", "_wavelet_buffers"):
            if name in seg:
                found.append((name, node))
    return found


def test_the_accumulators_are_not_gated_to_decision_ticks():
    """The fact that makes the windows short — pinned so the FIX moves the record.

    This is NOT a test that forbids the fix. If the appends are ever gated to
    real decision ticks (a `tick_count` / `for_decision` guard), that IS the
    P353 fix — and every window in the entry above becomes days again, so this
    test and that entry must be updated together rather than one drifting from
    the other (the P318 anti-rot pattern).
    """
    src = mdp.__file__ and inspect.getsource(mdp)
    tree = ast.parse(src)
    gated = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test_src = ast.dump(node.test)
            if "tick_count" in test_src or "for_decision" in test_src:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        f = sub.func
                        if isinstance(f, ast.Attribute) and f.attr == "append":
                            seg = ast.dump(f.value)
                            if any(n in seg for n in ("_ofi_history",
                                                      "_depth_history",
                                                      "_wavelet_buffers")):
                                gated.append(seg)
    assert not gated, (
        "a rolling-buffer append is now gated on the decision tick — that is "
        "the P353 fix. Re-derive every window in the P353 entry (they become "
        "days rather than minutes) and update this test, or the record and the "
        "code have silently disagreed again."
    )


def test_the_accumulator_scan_actually_finds_appends():
    """Anti-vacuity: the ungated-check above must have something to check.

    `_depth_history` is appended through a local alias (`_depth_buf`), so an
    AST scan by attribute name cannot see it — which is why the ungated-check
    covers the ones it CAN resolve and this test pins the third by name.
    """
    names = {n for n, _ in _accumulator_appends()}
    assert "_ofi_history" in names, (
        f"the accumulator scan found {names or 'nothing'} — if it stops "
        f"finding them the ungated-check above passes for the wrong reason"
    )
    assert "_depth_buf.append(" in inspect.getsource(mdp), (
        "the depth buffer's append moved; the P353 turnover table covers it"
    )


# --------------------------------------------------------------------------
# the arithmetic, derived rather than typed
# --------------------------------------------------------------------------
def _maxlen_of(name: str) -> int:
    src = inspect.getsource(mdp)
    # the declarations span two lines (a dict comprehension), so search a
    # window after the attribute name rather than the same line
    i = src.index("self." + name)
    m = re.search(r"deque\(maxlen=(\d+)\)", src[i:i + 400])
    assert m, f"could not read the maxlen for {name}"
    return int(m.group(1))


@pytest.mark.parametrize("name,expected_maxlen", [
    ("_ofi_history", 42),
    ("_depth_history", 120),
])
def test_the_buffer_sizes_the_entry_quotes_are_the_ones_in_the_code(
        name, expected_maxlen):
    assert _maxlen_of(name) == expected_maxlen, (
        f"{name}'s maxlen moved; the P353 turnover table is now wrong"
    )


@pytest.mark.parametrize("name,maxlen", [
    ("_ofi_history", 42),
    ("_depth_history", 120),
    ("_wavelet_buffers", 256),
])
def test_every_window_is_minutes_rather_than_the_days_its_size_implies(
        name, maxlen):
    """The finding, computed from the cadence and the size, not quoted."""
    actual_sec = maxlen * MEASURED_GAP_SEC
    implied_sec = maxlen * FOUR_HOURS
    assert actual_sec < FOUR_HOURS, (
        f"{name} spans {actual_sec/60:.1f} min — less than ONE decision bar, "
        f"while its size implies {implied_sec/86400:.1f} days"
    )
    assert implied_sec / actual_sec > 400


def test_the_regime_smoother_confirms_in_about_a_minute_not_eight_hours():
    """The Architecture Rules call persistence=2 a train/runtime parity
    requirement. In training it is 2 BARS; here it is 2 x 34s."""
    sig = inspect.signature(mdp.MarketDataPipeline.__init__)
    persistence = sig.parameters["regime_smoother_persistence"].default
    assert persistence == 2
    assert persistence * MEASURED_GAP_SEC < 120.0, (
        "the smoother confirms a regime change in about a minute, so by the "
        "time a 4H tick reads regime_state it has advanced ~424 times and "
        "provides no damping at the decision horizon"
    )


def test_p316s_arithmetic_is_recorded_as_falsified():
    """P316 sized the OFI warmup at '~20 HOURS of uninterrupted uptime' from
    one sample per 4H tick. At the real cadence five samples take minutes."""
    five_samples_sec = 5 * MEASURED_GAP_SEC
    assert five_samples_sec < 5 * 60, (
        "the OFI warmup P316 measured in HOURS completes in "
        f"{five_samples_sec:.0f} seconds"
    )


def test_the_denoised_features_reach_no_live_order_path():
    """Bounds the blast radius: the wavelet skew confounds two SHADOW streams
    and nothing that places an order. If a live consumer appears, the P353
    entry's 'no live order depends on them' stops being true."""
    import pathlib
    # Every known reader, each with the reason it cannot move money. A NEW
    # file appearing here fails, which is the point — an allowlist with
    # reasons rather than a blanket exclusion (P214).
    KNOWN = {
        "data_mgmt/market_data_pipeline.py": "the producer",
        "drl/runtime_obs_builder.py": "DRL observation, SHADOW",
        "agents/exit_drl_agent.py": "Exit-SAC, SHADOW and Kraken-only",
        "main.py": "_STEP15_WAVELET_FEATURES, a status-file roster",
        "scripts/runtime_parity_check.py": "offline diagnostic",
        "scripts/step15_gate_report.py": "offline report",
    }
    root = pathlib.Path(main.__file__).parent
    seen = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(("tests/", "archive/", "training/", "venv/")):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "_denoised" in txt:
            seen.append(rel)
    unknown = sorted(set(seen) - set(KNOWN))
    assert unknown == [], (
        f"a new consumer of the denoised features appeared in {unknown} — "
        f"check whether it is on a live order path before trusting P353's "
        f"bound that the wavelet cadence skew confounds only SHADOW streams"
    )

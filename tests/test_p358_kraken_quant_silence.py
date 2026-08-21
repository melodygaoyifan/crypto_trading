"""[P358] Why the second DECIDE agent is silent — measured, and it is neither
of the two options I offered.

I closed P357 by writing that `kraken_quant` "holds DECIDE authority and
contributes nothing … retiring the authority or repairing its reachable
surface is a live-behaviour change, so it stays recorded." The operator
quoted that back. Measuring it corrected me twice.

--------------------------------------------------------------------------
1. THE COST OF THE AUTHORITY IS UNDER 0.1%, so "retire it" was the wrong lever
--------------------------------------------------------------------------
The live DECIDE pool is exactly `['quant', 'kraken_quant']`, so fusion takes
its MULTI-decider branch every tick. LAYER 3's abstention rule (FIX-1/P71)
drops abstainers from the AGREEMENT ratio — but the weighting loop below it
iterates every DECIDE agent and adds `max(conf, 0.01)**2` to `total_weight`
regardless. A silent agent therefore contributes `1e-4` of weight:

    final_direction = D * C**2 / (C**2 + 1e-4)
    avg_conf        = sqrt(C**2 + 1e-4)

Measured over every retained `[AGENT-TRACE]`, quant's confidence takes three
values and none is near the 0.01 floor: **0.90 (131), 0.33 (87), 1.00 (21)**.
At 0.90 and 1.00 the SOLO-conviction branch fires (`_n_active == 1 and
best_conf >= 0.5`) and passes `best_conf` through untouched, so confidence is
exactly clean; at 0.33 the `else` branch inflates it by **+0.046%**. Direction
is diluted by **at most 0.09%** — and the sleeve sizes by the SIGN (P293d),
which makes that a literal no-op on live orders.

So the authority costs essentially nothing, and removing it would be churn on
a live account for a measured ~0.1% of a magnitude nothing reads. **Anchoring
a recommendation to the loud fact (a DECIDE agent is silent!) rather than to
the priced one is the mistake this file exists to catch.**

--------------------------------------------------------------------------
2. THE SILENCE IS NOT THE MARKET BEING QUIET — and "2 of 12 reachable" was stale
--------------------------------------------------------------------------
I quoted P215's "only 2 of 12 strategies are reachable" without checking
whether a later entry had moved it. **P217 added `STEADY_UPTREND` and
`NEUTRAL_DRIFT` to `_REGIME_MAP`**, and with `MOMENTUM_RALLY` now live the
BULL bucket is active too: the in-container diagnostic reports **BULL 33% /
SIDEWAYS 67% and 6 of 12 reachable**. That is P228 — a claim reused without
re-checking its premise — committed by me one entry after citing P228.

And the six are **ATTEMPTING and DECLINING** (attempts 2-4, fires 0), which is
a different diagnosis from unreachable. Two of them cannot fire at all:

  * **OrderBookImbalance is arithmetically pinned to zero.** main.py builds
    the cross-asset snapshot with `bid_depth = ask_depth =
    orderbook_depth_1pct_usd / 2`, so the strategy's
    `(bid - ask) / (bid + ask)` is **identically 0.0 for every asset on every
    tick**, its buffer fills with zeros, and `sol_obm > threshold` can never
    be true at ANY threshold. A check that cannot fire (P174), created by two
    inputs derived from one number.
  * **DarkPoolVolume reads a shape the converter never builds.** It asks for
    `market_data[asset]['volume'/'close']`, but `_convert_market_data`
    returns a FIXED key set (`prices`, `bid_depth`, `ask_depth`, …) with no
    per-asset sub-dict — so both reads are 0 and it `continue`s past all
    three assets. The P2 reader/writer mismatch.

Two more (`RelativeStrength`, `KalmanCointegration`) need >= 50 samples in an
in-memory buffer fed ~3 times per 4H tick — days of uninterrupted uptime, on
a system that deploys often, with no persistence. That is the P301/P316
warmup class, for which `strategies/_warmup_state.py` is the built precedent.

--------------------------------------------------------------------------
WHAT THIS FILE DOES, AND DELIBERATELY DOES NOT
--------------------------------------------------------------------------
It PINS the two structural defects as known-dead, with the arithmetic, so
that repairing either is a decision somebody takes on purpose rather than a
side effect (P318's anti-rot pattern). Repairing them would let a DECIDE-
authority agent start emitting directions on a live account — a P141
activation, not a bugfix, and not something to slip in behind a diagnosis.
"""

import ast
import inspect
import json
from contextlib import redirect_stdout as _redirect
from io import StringIO as _StringIO
import pathlib

import pytest

import main
from agents.kraken_quant_agent import KrakenQuantAgentV6

REPO = pathlib.Path(main.__file__).parent


# ==========================================================================
# 1. The measured cost of the silent DECIDE agent
# ==========================================================================
def _fuse_pair(quant_conf, quant_dir=1.0, other_conf=0.0):
    """Reproduce LAYER 3's multi-decider arithmetic for (quant, kraken_quant)."""
    w_q = max(quant_conf, 0.01) ** 2
    w_k = max(other_conf, 0.01) ** 2
    total = w_q + w_k
    direction = (quant_dir * w_q) / total
    n_active = 1 if abs(quant_dir) > 0.01 else 0
    avg_conf = (total / max(n_active, 1)) ** 0.5
    solo = n_active == 1 and quant_conf >= 0.5
    return direction, (quant_conf if solo else avg_conf)


@pytest.mark.parametrize("conf", [0.90, 1.00, 0.33])
def test_the_silent_decider_costs_under_one_tenth_of_a_percent(conf):
    """Every confidence quant has ever been observed at. If this ever exceeds
    the bound, the 'leave it alone' disposition has to be re-derived."""
    direction, merged = _fuse_pair(conf)
    assert abs(direction - 1.0) < 1e-3, (
        f"direction diluted by {(1 - direction) * 100:.3f}% at conf={conf}"
    )
    assert abs(merged - conf) / conf < 1e-3, (
        f"confidence moved {abs(merged - conf) / conf * 100:.3f}% at conf={conf}"
    )


def test_the_high_confidence_path_is_EXACTLY_clean():
    """At conf >= 0.5 the SOLO branch passes best_conf through untouched, so
    the abstainer's floor weight cannot touch it at all."""
    _, merged = _fuse_pair(0.90)
    assert merged == 0.90


def test_the_cost_would_NOT_be_negligible_near_the_floor():
    """The bound above is a property of quant's OBSERVED confidence, not of
    the arithmetic. Pinning the counter-case keeps the disposition honest: if
    a decider ever publishes confidence near 0.01, the abstainer's floor
    weight becomes material and this entry's conclusion does not carry."""
    direction, _ = _fuse_pair(0.01, quant_dir=1.0)
    assert direction == pytest.approx(0.5), (
        "at the floor the abstainer takes half the weight — the reason the "
        "measurement had to be of quant's real confidence, not of the formula"
    )


# ==========================================================================
# 2. The two structural defects, pinned as KNOWN-DEAD
# ==========================================================================
def test_orderbook_imbalance_is_pinned_to_zero_by_construction():
    """bid_depth and ask_depth are both `orderbook_depth_1pct_usd / 2`, so the
    imbalance is identically zero — the strategy cannot fire at ANY threshold.

    Pinned at the PRODUCER: if the split ever becomes a real bid/ask, this
    strategy goes live at DECIDE authority, which is a P141 decision."""
    src = inspect.getsource(main.HMATSProductionRunner._process_4h_tick_inner)
    i = src.index('"bid_depth"')
    window = src[i:i + 300]
    assert 'orderbook_depth_1pct_usd", 0) / 2' in window, (
        "the bid/ask split changed — OrderBookImbalanceStrategy may now be "
        "able to fire, which is a live behaviour change at DECIDE authority "
        "and needs an operator decision (P141), not a silent repair"
    )
    assert window.count("/ 2") >= 2, "bid and ask are no longer the same value"


def test_the_converter_builds_no_per_asset_subdict():
    """DarkPoolVolumeStrategy reads `market_data[asset]['volume'/'close']`.
    The converter returns a fixed key set and never keys by asset name, so
    both reads are 0 and it skips every asset. Pinned so that adding such a
    sub-dict is recognised as arming a strategy, not as plumbing."""
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(
        KrakenQuantAgentV6._convert_market_data)))
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
    assert returns, "the converter no longer returns a literal"
    keys = set()
    for r in returns:
        if isinstance(r.value, ast.Dict):
            keys |= {k.value for k in r.value.keys
                     if isinstance(k, ast.Constant)}
    assert keys, "could not read the converter's key set"
    for asset in ("BTC", "ETH", "SOL"):
        assert asset not in keys, (
            f"the converter now emits a per-asset sub-dict for {asset} — "
            f"DarkPoolVolumeStrategy would start reading real volume and can "
            f"fire at DECIDE authority (P141 decision)"
        )


def test_dark_pool_still_reads_the_shape_that_is_not_produced():
    """The other half of the mismatch. If the STRATEGY is rewritten to read
    the produced shape, that is equally an arming."""
    from agents.kraken_quant_agent import DarkPoolVolumeStrategy
    src = inspect.getsource(DarkPoolVolumeStrategy)
    assert "market_data.get(asset, {})" in src, (
        "DarkPoolVolumeStrategy no longer reads the absent per-asset "
        "sub-dict — it may now be live at DECIDE authority (P141)"
    )


# ==========================================================================
# 3. The premise this entry rests on
# ==========================================================================
def test_both_deciders_are_still_in_the_pool():
    """The whole cost analysis assumes exactly these two. A third DECIDE
    agent changes the arithmetic and the disposition."""
    from signals.authority_fusion import AUTHORITY_MATRIX_NORMAL, Authority
    deciders = sorted(a for a, auth in AUTHORITY_MATRIX_NORMAL.items()
                      if auth == Authority.DECIDE)
    assert deciders == ["kraken_quant", "quant"], (
        f"the DECIDE pool changed to {deciders} — re-derive P358's cost "
        f"arithmetic before relying on it"
    )


def test_the_regime_map_still_covers_what_P217_added():
    """'2 of 12 reachable' was stale because P217 widened the map. Pin it, so
    the reachable count in this entry does not rot the same way."""
    from agents.kraken_quant_agent import _REGIME_MAP
    for name in ("STEADY_UPTREND", "NEUTRAL_DRIFT", "MOMENTUM_RALLY"):
        assert name in _REGIME_MAP, (
            f"{name} dropped out of the regime map — the reachable-strategy "
            f"count in P358 was measured with it present"
        )


# ==========================================================================
# 4. The diagnostic's cause table — both directions (P310)
# ==========================================================================
def _cause_table():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_kqdiag", REPO / "scripts" / "kq_strategy_diagnostic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.KNOWN_SILENT_CAUSES


def _reported_names():
    """The names the AGENT emits, read from the PRODUCER.

    Every strategy passes its own `name` to BaseStrategy.__init__, and that is
    the string the firing stats are keyed by — so instantiating the classes is
    the only honest source. A hardcoded mirror here would be the very drift
    this test exists to catch (P310/P172)."""
    import agents.kraken_quant_agent as kq
    out = set()
    for obj in vars(kq).values():
        if (isinstance(obj, type) and issubclass(obj, kq.BaseStrategy)
                and obj is not kq.BaseStrategy):
            try:
                out.add(obj().name)
            except Exception:      # abstract or ctor needs args — skip
                continue
    return out


def test_every_cause_names_a_strategy_the_agent_actually_REPORTS():
    """[P358] The producer/consumer contract, in the direction that bit me.

    My first cut keyed the table on CLASS names (`OrderBookImbalanceStrategy`)
    while the agent reports `OrderBookImbalance` — so two of four entries
    matched nothing and the tool printed 'cause UNKNOWN' for defects it had
    been told about. Silent, and only visible by running the consumer against
    real producer output (P264/P309). A table entry that matches nothing is
    worse than an absent one: it reads as coverage."""
    reported = _reported_names()
    assert reported, "could not read the agent's strategy roster"
    unmatched = sorted(set(_cause_table()) - reported)
    assert not unmatched, (
        f"cause-table keys match no reported strategy: {unmatched} — the "
        f"diagnostic will print 'cause UNKNOWN' for these (P310)"
    )


def test_every_cause_carries_a_class_and_a_reason():
    """A label without the arithmetic sends the next reader back to re-derive
    it; and STRUCTURAL entries must say that repairing them is an arming, or
    the table reads as a to-do list against a live DECIDE-authority agent."""
    for name, (cls, reason) in _cause_table().items():
        assert cls in ("STRUCTURAL", "WARMUP"), f"{name}: unknown class {cls}"
        assert len(reason) > 40, f"{name}: reason is not a reason"
        assert "P358" in reason, f"{name}: no provenance"
        if cls == "STRUCTURAL":
            assert "P141" in reason, (
                f"{name}: a structural cause must state that repairing it "
                f"ARMS a DECIDE-authority agent on a live account"
            )


def test_the_diagnostic_ACTUALLY_USES_the_cause_table(tmp_path):
    """[P358b/P359] A falsification probe setting `_cause = None` left every
    test GREEN — the table was checked and its USE was not (P170).

    [P359] Re-expressed through `assert_drives_output`, which runs the real
    consumer twice: once as-is and once with the table neutralised, requiring
    the label to appear and then DISAPPEAR. Presence alone cannot distinguish
    consumed from coincidental; only varying the input can. That turns the
    probe I had to run by hand into something this test does on every run."""
    from tests._cli_harness import load_cli, run_cli
    from tests._guard_pins import assert_drives_output

    # [P360] Through the harness rather than hand-rolled importlib +
    # redirect_stdout. The subprocess version this replaced could not
    # neutralise the table at all, which is what made the check impossible.
    mod = load_cli(REPO / "scripts" / "kq_strategy_diagnostic.py")

    stats = {
        "regime_ticks": {"SIDEWAYS": 4},
        "by_regime": {
            "SIDEWAYS": [
                {"name": "DarkPoolVolumeStrategy", "attempts": 4, "fires": 0},
                {"name": "KalmanCointegration_SOL_ETH", "attempts": 4,
                 "fires": 0},
            ]
        },
        "archived": [],
    }

    def _run():
        buf = _StringIO()
        with _redirect(buf):
            mod.render(stats)
        return buf.getvalue()

    def _disable():
        original = mod.KNOWN_SILENT_CAUSES
        mod.KNOWN_SILENT_CAUSES = {}
        return lambda: setattr(mod, "KNOWN_SILENT_CAUSES", original)

    for marker in ("[STRUCTURAL]", "[WARMUP]", "P141"):
        assert_drives_output(
            _run, marker, _disable,
            why=f"{marker} does not come from KNOWN_SILENT_CAUSES")

    # ...and the residue must stay visible, or naming the known causes would
    # hide the set actually worth tracing.
    stats["by_regime"]["SIDEWAYS"].append(
        {"name": "ETFSpotCointegration", "attempts": 4, "fires": 0})
    assert "cause UNKNOWN" in _run(), (
        "a strategy with no measured cause no longer reads as unexplained"
    )

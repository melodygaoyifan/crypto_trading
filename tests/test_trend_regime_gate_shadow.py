"""[P198] Trend regime gate: shadow measures, enforce zeroes, off is silent.

The gate exists because trend-following's live loss concentrates in
WEAK_CONSOLIDATION (-9.1bps/4H-tick, 45.3% hit, negative on all 3 assets,
2026-06-14 -> 08-06) while QUIET_ACCUMULATION is mildly positive — but that
window is IN-SAMPLE, so the gate ships in SHADOW and is promoted only on the
forward evidence accumulating in data/trend_regime_shadow.jsonl
(scripts/trend_regime_review.py).
"""

import json
from datetime import datetime

import pytest

import core.trend_decision_layer as tdl_mod
from core.trend_decision_layer import (
    TrendDecisionLayer,
    get_trend_decision_layer,
    _DEFAULT_GATE_REGIMES,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Fresh singleton per test; shadow log redirected to tmp."""
    monkeypatch.setattr(tdl_mod, "_singleton", None)
    monkeypatch.setenv("HMATS_DATA_DIR", str(tmp_path))
    yield tmp_path


def _layer(mode="enforce", gate="shadow", sig=0.8):
    import time as _time
    layer = TrendDecisionLayer(mode=mode, regime_gate_mode=gate)
    layer._strat.compute = lambda closes: {"signal": sig, "target_position": sig}
    layer._closes["BTC"] = [1.0] * 300
    layer._closes_cached_at["BTC"] = _time.time()  # [P265] fresh-closes stamp
    return layer


def _run(layer, regime="WEAK_CONSOLIDATION"):
    agent_signals, market_data = {}, {"regime_state": regime}
    res = layer.process("BTC", None, agent_signals, market_data)
    return res, agent_signals, market_data


def _log_lines(tmp_path):
    p = tmp_path / "trend_regime_shadow.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# SHADOW (the default): measures, never changes the trade
# ---------------------------------------------------------------------------

def test_shadow_gate_logs_but_does_not_change_the_injected_signal(_isolate):
    layer = _layer(gate="shadow")
    _, agent_signals, market_data = _run(layer, "WEAK_CONSOLIDATION")
    assert market_data["quant_direction"] == pytest.approx(0.8), (
        "SHADOW gate altered the live signal — shadow must be measurement only"
    )
    recs = _log_lines(_isolate)
    assert len(recs) == 1
    assert recs[0]["gated"] is True and recs[0]["regime"] == "WEAK_CONSOLIDATION"
    assert recs[0]["trend_sig"] == pytest.approx(0.8)


def test_shadow_log_record_carries_tz_aware_timestamp(_isolate):
    layer = _layer(gate="shadow")
    _run(layer)
    ts = datetime.fromisoformat(_log_lines(_isolate)[0]["ts"])
    assert ts.tzinfo is not None, "naive datetime in a persisted record (P40/P97)"


def test_ungated_regime_logs_gated_false(_isolate):
    layer = _layer(gate="shadow")
    _run(layer, "QUIET_ACCUMULATION")
    assert _log_lines(_isolate)[0]["gated"] is False


def test_unknown_regime_is_not_gated(_isolate):
    layer = _layer(gate="enforce")
    _, _, market_data = _run(layer, "SOME_FUTURE_REGIME")
    assert market_data["quant_direction"] == pytest.approx(0.8), (
        "an unrecognized regime must fail toward TRADING (gate is an "
        "in-sample hypothesis; absence of evidence must not block)"
    )


# ---------------------------------------------------------------------------
# ENFORCE: zeroes the signal only in gated regimes
# ---------------------------------------------------------------------------

def test_enforce_is_retired_and_cannot_zero_the_signal(_isolate, caplog):
    """[P307b] This test used to pin that enforce ZEROES the signal in a
    gated regime. training/trend_gate_lab.py then measured the gate on the
    question that decides it — does zeroing the trend seat leave more money
    after costs — and the answer is NO in BOTH eras on ALL THREE assets
    (BTC -0.699/-0.289, ETH -0.243/-0.228, SOL -0.623/-0.258 net). Six reads
    out of six, and it reproduced either side of the P307 GMM refit.

    So "enforce" is retired rather than left as a flag someone can flip, and
    what is pinned now is that asking for it changes nothing AND says so.
    """
    import logging
    with caplog.at_level(logging.ERROR):
        layer = _layer(gate="enforce")
    assert layer.regime_gate_mode == "shadow"
    assert any("RETIRED" in r.message for r in caplog.records), (
        "a retired mode was coerced SILENTLY — a config that says enforce "
        "and behaves as shadow is the P16 shape")
    _, agent_signals, market_data = _run(layer, "WEAK_CONSOLIDATION")
    assert market_data["quant_direction"] == pytest.approx(0.8)
    assert agent_signals["quant_direction"] == pytest.approx(0.8)
    # the ledger keeps accruing: retiring a PRESCRIPTION is not a reason to
    # retire the DETECTION beside it (P299)
    assert _log_lines(_isolate)[0]["trend_sig"] == pytest.approx(0.8)


def test_shadow_still_leaves_an_ungated_regime_alone(_isolate):
    layer = _layer(gate="shadow")
    _, _, market_data = _run(layer, "QUIET_ACCUMULATION")
    assert market_data["quant_direction"] == pytest.approx(0.8)


def test_gate_off_writes_no_log_and_changes_nothing(_isolate):
    layer = _layer(gate="off")
    _, _, market_data = _run(layer, "WEAK_CONSOLIDATION")
    assert market_data["quant_direction"] == pytest.approx(0.8)
    assert _log_lines(_isolate) == []


# ---------------------------------------------------------------------------
# Configuration pins
# ---------------------------------------------------------------------------

def test_default_gate_regimes_are_the_measured_losers():
    """Changing the gated set must be a conscious act — it was chosen from
    measured live data (WEAK_CONSOLIDATION -9.1bps, NEUTRAL_DRIFT -68.9bps),
    and QUIET_ACCUMULATION must NOT be in it (mildly positive, and gating it
    would block 93% of all ticks = system off)."""
    assert set(_DEFAULT_GATE_REGIMES) == {"WEAK_CONSOLIDATION", "NEUTRAL_DRIFT"}
    assert "QUIET_ACCUMULATION" not in _DEFAULT_GATE_REGIMES


def test_default_gate_mode_is_shadow():
    assert TrendDecisionLayer(mode="enforce").regime_gate_mode == "shadow"


def test_singleton_updates_gate_mode():
    layer = get_trend_decision_layer("enforce", regime_gate_mode="shadow")
    layer2 = get_trend_decision_layer("enforce", regime_gate_mode="off")
    assert layer is layer2 and layer2.regime_gate_mode == "off"


def test_singleton_refuses_the_retired_mode_and_keeps_the_current_one():
    """[P307b] The setter must not silently downgrade either."""
    get_trend_decision_layer("enforce", regime_gate_mode="off")
    layer = get_trend_decision_layer("enforce", regime_gate_mode="enforce")
    assert layer.regime_gate_mode == "off", (
        "a retired mode must leave the live mode untouched, not reset it")


def test_singleton_gate_mode_untouched_when_not_passed():
    get_trend_decision_layer("enforce", regime_gate_mode="off")
    layer = get_trend_decision_layer("enforce")
    assert layer.regime_gate_mode == "off"


def test_invalid_gate_mode_falls_back_to_shadow():
    assert TrendDecisionLayer(regime_gate_mode="banana").regime_gate_mode == "shadow"


def test_main_passes_trend_regime_gate_config():
    """P152 lesson: the gate must actually be wired at the call site."""
    import io
    src = io.open("main.py", encoding="utf-8").read()
    assert "trend_regime_gate" in src and "regime_gate_mode=" in src


def test_shadow_write_failure_is_warned_not_debug_swallowed(_isolate, monkeypatch, caplog):
    """P160: a writer that fails silently leaves the reader unable to tell
    'not accumulating' from 'accumulated the same value'."""
    import logging
    layer = _layer(gate="shadow")
    monkeypatch.setattr(layer, "_shadow_log_path",
                        lambda: str(_isolate / "no_such_dir\0bad" / "x.jsonl"))
    with caplog.at_level(logging.WARNING):
        _run(layer)
    assert any("shadow log write failed" in r.message for r in caplog.records)


def test_re_arming_enforce_requires_editing_two_constants_and_fails_both():
    """[P307b] Defence in depth, pinned. "enforce" must be absent from the
    VALID set AND present in the RETIRED set: removing it from one alone
    leaves it either refused-with-a-reason or coerced-as-invalid, so no
    single edit can silently re-arm a gate measured negative 6 reads out of
    6. A falsification probe that only put it back in the VALID set stayed
    GREEN, which is what this pin exists to make impossible."""
    from core import trend_decision_layer as M
    assert "enforce" in M._RETIRED_GATE_MODES
    assert "enforce" not in M._VALID_GATE_MODES

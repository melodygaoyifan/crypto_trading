from types import SimpleNamespace

from main import HMATSProductionRunner


def _make_runner(alpha_gate_config=None):
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner.config = SimpleNamespace(alpha_gate_config=alpha_gate_config or {})
    return runner


def test_suppresses_bullish_model_alpha_context_boost_in_quiet_for_high_risk():
    runner = _make_runner(
        {
            "suppress_model_alpha_long_context_in_quiet": True,
        }
    )

    result = runner._compute_effective_alpha_direction(
        "BTC",
        {
            "quant_direction": 0.041,
            "regime_state": "QUIET_ACCUMULATION",
            "model_alpha_authority": "CONTEXT",
            "model_alpha_direction": 1.0,
            "model_alpha_confidence": 0.49,
            "model_alpha_data_quality": 1.0,
        },
        {"regime_state": "QUIET_ACCUMULATION"},
    )

    assert result["direction"] == 0.041
    assert result["context_boost"] == 0.0
    assert result["sources"] == []


def test_keeps_bearish_model_alpha_context_boost_available_in_quiet():
    runner = _make_runner(
        {
            "suppress_model_alpha_long_context_in_quiet": True,
        }
    )

    result = runner._compute_effective_alpha_direction(
        "BTC",
        {
            "quant_direction": -0.041,
            "regime_state": "QUIET_ACCUMULATION",
            "model_alpha_authority": "CONTEXT",
            "model_alpha_direction": -1.0,
            "model_alpha_confidence": 0.49,
            "model_alpha_data_quality": 1.0,
        },
        {"regime_state": "QUIET_ACCUMULATION"},
    )

    assert result["direction"] < -0.041
    assert result["context_boost"] > 0.0
    assert result["sources"] == ["model_alpha:+0.100"]


def test_allows_bullish_model_alpha_context_boost_when_flag_disabled():
    runner = _make_runner({})

    result = runner._compute_effective_alpha_direction(
        "BTC",
        {
            "quant_direction": 0.041,
            "regime_state": "QUIET_ACCUMULATION",
            "model_alpha_authority": "CONTEXT",
            "model_alpha_direction": 1.0,
            "model_alpha_confidence": 0.49,
            "model_alpha_data_quality": 1.0,
        },
        {"regime_state": "QUIET_ACCUMULATION"},
    )

    assert result["direction"] > 0.041
    assert result["context_boost"] > 0.0
    assert result["sources"] == ["model_alpha:+0.100"]

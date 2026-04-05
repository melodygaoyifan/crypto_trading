from types import SimpleNamespace

from core.runtime_authority import ALLOWED_RUNTIME_AUTHORITY_STATES
from core.runtime_control_service import (
    build_aggressive_allocator_runtime_settings,
    build_asset_runtime_authority_snapshot,
    build_fast_market_execution_decision,
    build_g6_runtime_settings,
    build_runtime_authority_matrix,
)


def test_build_g6_runtime_settings_prefers_profile_and_respects_env_override():
    settings = build_g6_runtime_settings(
        {
            "enabled": True,
            "shadow_mode": False,
            "cold_start_prior_enabled": True,
            "cold_start_prior_floor": 0.15,
            "cold_start_prior_max_multiplier": 1.40,
            "min_multiplier": 0.75,
            "max_multiplier": 1.30,
        },
        env_shadow="true",
    )

    assert settings["enabled"] is True
    assert settings["shadow_mode"] is True
    assert settings["cold_start_prior_enabled"] is True
    assert settings["cold_start_prior_floor"] == 0.15
    assert settings["cold_start_prior_max_multiplier"] == 1.40
    assert settings["min_multiplier"] == 0.75
    assert settings["max_multiplier"] == 1.30


def test_build_aggressive_allocator_runtime_settings_prefers_env_when_present():
    settings = build_aggressive_allocator_runtime_settings(
        {"live_enabled": True, "min_session_fills": 2},
        env_enabled="false",
        env_min_fills="4",
    )

    assert settings["live_enabled"] is False
    assert settings["min_session_fills"] == 4


def test_build_fast_market_execution_decision_handles_opportunity_and_quality_short():
    opp_intent = SimpleNamespace(
        system_mode="OPPORTUNITY",
        execution_mode=SimpleNamespace(value="AGGRESSIVE"),
        direction=0.18,
        expected_net_alpha_bps=6.0,
    )
    short_intent = SimpleNamespace(
        system_mode="NORMAL",
        execution_mode=SimpleNamespace(value="AGGRESSIVE"),
        direction=-0.22,
        expected_net_alpha_bps=12.5,
    )

    opp = build_fast_market_execution_decision(
        opp_intent,
        enabled=True,
        modes={"OPPORTUNITY"},
        require_aggressive_mode=True,
        min_short_net_alpha_bps=10.0,
        min_short_abs_direction=0.15,
    )
    short = build_fast_market_execution_decision(
        short_intent,
        enabled=True,
        modes={"OPPORTUNITY"},
        require_aggressive_mode=True,
        min_short_net_alpha_bps=10.0,
        min_short_abs_direction=0.15,
    )

    assert opp["apply"] is True
    assert "mode=OPPORTUNITY" in opp["reason"]
    assert short["apply"] is True
    assert "short_quality" in short["reason"]


def test_build_runtime_authority_matrix_uses_only_allowed_states():
    matrix = build_runtime_authority_matrix(
        asset_runtime_map={
            "BTC": {
                "composite_toxicity_authority": "EXECUTION",
                "learned_exec_authority": "EXECUTION",
                "learned_exec_effective_mode": "ADVISORY_HEURISTIC",
                "g6_authority": "SIZING",
                "g6_mode": "LIVE",
                "aggressive_allocator_authority": "SIZING_DEFERRED",
                "lead_lag_authority": "CONTEXT",
                "model_alpha_authority": "CONTEXT",
            }
        },
        drl_runtime={"gate_level": "EXIT_ONLY", "execution_authority": "ON"},
        assets=["BTC"],
    )

    allowed = set(ALLOWED_RUNTIME_AUTHORITY_STATES)
    assert set(matrix["allowed_states"]) == allowed
    assert set(matrix["by_asset"]["BTC"].values()).issubset(allowed)
    assert matrix["by_asset"]["BTC"]["g6"] == "BOUNDED_LIVE"
    assert matrix["by_asset"]["BTC"]["aggressive_allocator"] == "SHADOW"
    assert matrix["by_asset"]["BTC"]["drl"] == "BOUNDED_LIVE"


def test_build_asset_runtime_authority_snapshot_matches_matrix_shape():
    snap = build_asset_runtime_authority_snapshot(
        asset_runtime={
            "composite_toxicity_authority": "EXECUTION",
            "learned_exec_authority": "EXECUTION",
            "learned_exec_effective_mode": "ADVISORY_HEURISTIC",
            "g6_authority": "SIZING",
            "g6_mode": "LIVE",
        },
        drl_runtime={"gate_level": "SHADOW", "execution_authority": "OFF"},
    )

    assert set(snap["allowed_states"]) == set(ALLOWED_RUNTIME_AUTHORITY_STATES)
    assert snap["by_module"]["composite_toxicity"] == "ADVISORY"
    assert snap["by_module"]["learned_execution_policy"] == "ADVISORY"
    assert snap["by_module"]["g6"] == "BOUNDED_LIVE"
    assert snap["by_module"]["drl"] == "SHADOW"

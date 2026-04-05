import json
from types import SimpleNamespace

from main import HMATSProductionRunner, ProductionConfig


def test_g6_runtime_settings_follow_profile_when_env_absent(monkeypatch):
    monkeypatch.delenv("HMATS_G6_SHADOW", raising=False)
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner.config = SimpleNamespace(
        g6_config={
            "enabled": True,
            "shadow_mode": False,
            "cold_start_prior_enabled": True,
            "cold_start_prior_floor": 0.15,
            "cold_start_prior_max_multiplier": 1.40,
            "min_multiplier": 0.75,
            "max_multiplier": 1.30,
        }
    )

    settings = runner._get_g6_runtime_settings()

    assert settings["enabled"] is True
    assert settings["shadow_mode"] is False
    assert settings["cold_start_prior_enabled"] is True
    assert settings["cold_start_prior_floor"] == 0.15
    assert settings["cold_start_prior_max_multiplier"] == 1.40
    assert settings["min_multiplier"] == 0.75
    assert settings["max_multiplier"] == 1.30


def test_aggressive_allocator_runtime_settings_allow_env_override(monkeypatch):
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner.config = SimpleNamespace(
        allocation_config={
            "enabled": True,
            "live_enabled": True,
            "min_session_fills": 2,
        }
    )

    monkeypatch.setenv("HMATS_ENABLE_AGGRESSIVE_ALLOCATOR", "false")
    monkeypatch.setenv("HMATS_AGGRESSIVE_ALLOCATOR_MIN_FILLS", "4")

    settings = runner._get_aggressive_allocator_runtime_settings()

    assert settings["live_enabled"] is False
    assert settings["min_session_fills"] == 4


def test_quiet_accum_short_direction_floor_uses_asset_override():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner._quiet_accum_min_direction_short = 0.05
    runner._quiet_accum_min_direction_short_by_asset = {"BTC": 0.03, "ETH": 0.015}

    assert runner._get_quiet_accum_short_direction_floor("BTC") == 0.03
    assert runner._get_quiet_accum_short_direction_floor("ETH") == 0.015
    assert runner._get_quiet_accum_short_direction_floor("SOL") == 0.05


def test_opportunity_actionable_short_direction_floor_uses_asset_override():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner._opportunity_actionable_min_direction_short = 0.045
    runner._opportunity_actionable_min_direction_short_by_asset = {"ETH": 0.015}

    assert runner._get_opportunity_actionable_short_direction_floor("ETH") == 0.015
    assert runner._get_opportunity_actionable_short_direction_floor("BTC") == 0.045


def test_opportunity_actionable_min_exposure_uses_asset_override():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner._opportunity_actionable_min_exposure = 0.01
    runner._opportunity_actionable_min_exposure_by_asset = {"ETH": 0.005}

    assert runner._get_opportunity_actionable_min_exposure("ETH") == 0.005
    assert runner._get_opportunity_actionable_min_exposure("BTC") == 0.01


def test_quiet_accum_long_direction_floor_uses_asset_override():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner._quiet_accum_min_direction = 0.15
    runner._quiet_accum_min_direction_by_asset = {"BTC": 0.04, "ETH": 0.025}

    assert runner._get_quiet_accum_direction_floor("BTC") == 0.04
    assert runner._get_quiet_accum_direction_floor("ETH") == 0.025
    assert runner._get_quiet_accum_direction_floor("SOL") == 0.15


def test_borderline_alpha_epsilon_uses_asset_override():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner._borderline_alpha_pass_epsilon_bps = 0.0
    runner._borderline_alpha_pass_epsilon_bps_by_asset = {"BTC": 1.0, "ETH": 4.5}

    assert runner._get_borderline_alpha_epsilon_bps("BTC") == 1.0
    assert runner._get_borderline_alpha_epsilon_bps("ETH") == 4.5
    assert runner._get_borderline_alpha_epsilon_bps("SOL") == 0.0


def test_short_borderline_alpha_epsilon_uses_asset_override():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner._borderline_alpha_pass_epsilon_bps_short = 0.25
    runner._borderline_alpha_pass_epsilon_bps_short_by_asset = {"BTC": 1.0, "ETH": 4.5}

    assert runner._get_short_borderline_alpha_epsilon_bps("BTC") == 1.0
    assert runner._get_short_borderline_alpha_epsilon_bps("ETH") == 4.5
    assert runner._get_short_borderline_alpha_epsilon_bps("SOL") == 0.25


def test_legacy_soft_veto_caps_only_correlation_in_normal():
    assert HMATSProductionRunner._should_cap_legacy_soft_veto_in_normal(["corr=0.90"]) is True
    assert HMATSProductionRunner._should_cap_legacy_soft_veto_in_normal(["corr=0.90", "vpin=0.72"]) is False
    assert HMATSProductionRunner._should_cap_legacy_soft_veto_in_normal(["dvol=3.2"]) is False


def test_structure_short_min_edge_bps_uses_asset_override():
    cfg = {
        "min_net_edge_bps_short": 5.0,
        "min_net_edge_bps_short_by_asset": {"ETH": 1.5, "SOL": -2.0},
    }

    assert HMATSProductionRunner._get_structure_short_min_edge_bps(cfg, "ETH") == 1.5
    assert HMATSProductionRunner._get_structure_short_min_edge_bps(cfg, "SOL") == -2.0
    assert HMATSProductionRunner._get_structure_short_min_edge_bps(cfg, "BTC") == 5.0


def test_effective_structure_short_min_edge_preserves_negative_asset_floor():
    assert HMATSProductionRunner._get_effective_structure_short_min_edge_bps(-2.0, 0.0) == -2.0
    assert HMATSProductionRunner._get_effective_structure_short_min_edge_bps(-2.0, 0.5) == -1.5
    assert HMATSProductionRunner._get_effective_structure_short_min_edge_bps(1.5, -3.0) == 0.0


def test_production_config_reads_signal_quality_block(tmp_path):
    cfg_path = tmp_path / "risk.json"
    cfg_path.write_text(
        json.dumps(
            {
                "signal_quality": {
                    "min_score_for_add_by_asset_side": {
                        "BTC:LONG": 55.0
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = ProductionConfig.from_file(cfg_path)

    assert cfg.signal_quality_config["min_score_for_add_by_asset_side"]["BTC:LONG"] == 55.0


def test_structure_long_min_edge_bps_uses_asset_override():
    cfg = {
        "min_net_edge_bps_long": 8.0,
        "min_net_edge_bps_long_by_asset": {"BTC": 2.0, "ETH": -3.0},
    }

    assert HMATSProductionRunner._get_structure_min_edge_bps(cfg, "BTC", is_short=False) == 2.0
    assert HMATSProductionRunner._get_structure_min_edge_bps(cfg, "ETH", is_short=False) == -3.0
    assert HMATSProductionRunner._get_structure_min_edge_bps(cfg, "SOL", is_short=False) == 8.0


def test_structure_short_min_abs_direction_uses_asset_override():
    cfg = {
        "min_abs_direction": 0.10,
        "min_abs_direction_short_by_asset": {"ETH": 0.085},
    }

    assert HMATSProductionRunner._get_structure_min_abs_direction(cfg, "ETH", is_short=True) == 0.085
    assert HMATSProductionRunner._get_structure_min_abs_direction(cfg, "BTC", is_short=True) == 0.10
    assert HMATSProductionRunner._get_structure_min_abs_direction(cfg, "BTC", is_short=False) == 0.10


def test_structure_long_min_abs_direction_uses_asset_override():
    cfg = {
        "min_abs_direction": 0.10,
        "min_abs_direction_long_by_asset": {"BTC": 0.08, "ETH": 0.05},
    }

    assert HMATSProductionRunner._get_structure_min_abs_direction(cfg, "BTC", is_short=False) == 0.08
    assert HMATSProductionRunner._get_structure_min_abs_direction(cfg, "ETH", is_short=False) == 0.05
    assert HMATSProductionRunner._get_structure_min_abs_direction(cfg, "SOL", is_short=False) == 0.10


def test_profit_calibration_live_adjustments_apply_bounded_short_relax_and_long_tighten():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner._profit_calibration_shadow_enabled = True
    runner._profit_calibration_shadow_min_closed_trades_per_side = 2
    runner._profit_calibration_shadow_tighten_long_avg_bps_below = -10.0
    runner._profit_calibration_shadow_tighten_long_total_pnl_usd_below = -20.0
    runner._profit_calibration_shadow_keep_long_avg_bps_above = 10.0
    runner._profit_calibration_shadow_relax_short_avg_bps_above = 15.0
    runner._profit_calibration_shadow_relax_short_total_pnl_usd_above = 20.0
    runner._profit_calibration_live_enabled = True
    runner._profit_calibration_live_min_closed_trades_per_side = 5
    runner._profit_calibration_live_short_alpha_bps_relax = 1.0
    runner._profit_calibration_live_short_structure_gap_bonus_bps = 20.0
    runner._profit_calibration_live_short_structure_min_edge_relax_bps = 1.0
    runner._profit_calibration_live_quiet_long_direction_tighten = 0.02
    runner._realized_pnl_breakdown = runner._new_realized_pnl_breakdown()
    runner._realized_pnl_breakdown["closed_trade_count"] = 10
    runner._realized_pnl_breakdown["by_side"] = {
        "long": {"count": 5, "total_pnl_usd": -24.0, "avg_pnl_bps": -12.0},
        "short": {"count": 5, "total_pnl_usd": 28.0, "avg_pnl_bps": 18.0},
    }

    adjustments = runner._get_profit_calibration_live_adjustments()

    assert adjustments["enabled"] is True
    assert adjustments["min_closed_trades_per_side"] == 5
    assert adjustments["live_sample_ready_by_side"]["long"] is True
    assert adjustments["live_sample_ready_by_side"]["short"] is True
    assert adjustments["short_alpha_bps_delta"] == -1.0
    assert adjustments["short_structure_gap_bonus_bps"] == 20.0
    assert adjustments["short_structure_min_edge_delta_bps"] == -1.0
    assert adjustments["quiet_long_direction_delta"] == 0.02
    assert "TIGHTEN_QUIET_LONG" in adjustments["reasons"]
    assert "RELAX_SHORT_STRUCTURE_CANDIDATE" in adjustments["reasons"]


def test_profit_calibration_live_adjustments_wait_for_higher_live_sample():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner._profit_calibration_shadow_enabled = True
    runner._profit_calibration_shadow_min_closed_trades_per_side = 2
    runner._profit_calibration_shadow_tighten_long_avg_bps_below = -10.0
    runner._profit_calibration_shadow_tighten_long_total_pnl_usd_below = -20.0
    runner._profit_calibration_shadow_keep_long_avg_bps_above = 10.0
    runner._profit_calibration_shadow_relax_short_avg_bps_above = 15.0
    runner._profit_calibration_shadow_relax_short_total_pnl_usd_above = 20.0
    runner._profit_calibration_live_enabled = True
    runner._profit_calibration_live_min_closed_trades_per_side = 5
    runner._profit_calibration_live_short_alpha_bps_relax = 1.0
    runner._profit_calibration_live_short_structure_gap_bonus_bps = 20.0
    runner._profit_calibration_live_short_structure_min_edge_relax_bps = 1.0
    runner._profit_calibration_live_quiet_long_direction_tighten = 0.02
    runner._realized_pnl_breakdown = runner._new_realized_pnl_breakdown()
    runner._realized_pnl_breakdown["closed_trade_count"] = 4
    runner._realized_pnl_breakdown["by_side"] = {
        "long": {"count": 2, "total_pnl_usd": -24.0, "avg_pnl_bps": -12.0},
        "short": {"count": 2, "total_pnl_usd": 28.0, "avg_pnl_bps": 18.0},
    }

    adjustments = runner._get_profit_calibration_live_adjustments()

    assert adjustments["quiet_long_direction_delta"] == 0.0
    assert adjustments["short_alpha_bps_delta"] == 0.0
    assert adjustments["live_sample_ready_by_side"]["long"] is False
    assert adjustments["live_sample_ready_by_side"]["short"] is False
    assert "LONG_LIVE_SAMPLE_2/5" in adjustments["reasons"]
    assert "SHORT_LIVE_SAMPLE_2/5" in adjustments["reasons"]


def test_fast_market_execution_decision_applies_to_opportunity_and_quality_short():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    runner._fast_market_execution_enabled = True
    runner._fast_market_execution_modes = {"OPPORTUNITY"}
    runner._fast_market_execution_require_aggressive_mode = True
    runner._fast_market_execution_min_short_net_alpha_bps = 10.0
    runner._fast_market_execution_min_short_abs_direction = 0.15

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

    opp = runner._get_fast_market_execution_decision(opp_intent)
    short = runner._get_fast_market_execution_decision(short_intent)

    assert opp["apply"] is True
    assert "mode=OPPORTUNITY" in opp["reason"]
    assert short["apply"] is True
    assert "short_quality" in short["reason"]

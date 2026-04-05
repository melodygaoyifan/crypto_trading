from main import HMATSProductionRunner


def test_pre_structure_hold_short_triggers_for_weak_short_near_support():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    cfg = {
        "pre_veto_hold_short_enabled": True,
        "apply_in_system_modes": ["OPPORTUNITY"],
        "apply_in_system_modes_short": ["NORMAL", "OPPORTUNITY"],
        "apply_in_regimes_short": ["QUIET_ACCUMULATION", "WEAK_CONSOLIDATION"],
        "pre_veto_hold_max_support_gap_bps_short": 30,
        "pre_veto_hold_max_net_edge_bps_short": 1.5,
        "short_mean_revert_strategies": ["mean_revert", "mean_reversion"],
    }

    assert runner._should_pre_structure_hold_short(
        cfg,
        system_mode="NORMAL",
        regime="QUIET_ACCUMULATION",
        strategy_id="volume_breakout",
        gap_to_support_bps=20.0,
        edge_bps=0.8,
        can_short=False,
    ) is True


def test_pre_structure_hold_short_does_not_trigger_for_strong_edge():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    cfg = {
        "pre_veto_hold_short_enabled": True,
        "apply_in_system_modes_short": ["NORMAL", "OPPORTUNITY"],
        "apply_in_regimes_short": ["QUIET_ACCUMULATION", "WEAK_CONSOLIDATION"],
        "pre_veto_hold_max_support_gap_bps_short": 30,
        "pre_veto_hold_max_net_edge_bps_short": 1.5,
    }

    assert runner._should_pre_structure_hold_short(
        cfg,
        system_mode="NORMAL",
        regime="QUIET_ACCUMULATION",
        strategy_id="volume_breakout",
        gap_to_support_bps=20.0,
        edge_bps=4.0,
        can_short=False,
    ) is False


def test_pre_structure_hold_short_skips_mean_revert_strategies():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    cfg = {
        "pre_veto_hold_short_enabled": True,
        "apply_in_system_modes_short": ["NORMAL", "OPPORTUNITY"],
        "apply_in_regimes_short": ["QUIET_ACCUMULATION", "WEAK_CONSOLIDATION"],
        "pre_veto_hold_max_support_gap_bps_short": 30,
        "pre_veto_hold_max_net_edge_bps_short": 1.5,
        "short_mean_revert_strategies": ["mean_revert", "mean_reversion"],
    }

    assert runner._should_pre_structure_hold_short(
        cfg,
        system_mode="NORMAL",
        regime="QUIET_ACCUMULATION",
        strategy_id="mean_revert",
        gap_to_support_bps=20.0,
        edge_bps=0.8,
        can_short=False,
    ) is False


def test_pre_structure_hold_short_does_not_trigger_when_gap_too_wide():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)
    cfg = {
        "pre_veto_hold_short_enabled": True,
        "apply_in_system_modes_short": ["NORMAL", "OPPORTUNITY"],
        "apply_in_regimes_short": ["QUIET_ACCUMULATION", "WEAK_CONSOLIDATION"],
        "pre_veto_hold_max_support_gap_bps_short": 30,
        "pre_veto_hold_max_net_edge_bps_short": 1.5,
    }

    assert runner._should_pre_structure_hold_short(
        cfg,
        system_mode="NORMAL",
        regime="QUIET_ACCUMULATION",
        strategy_id="volume_breakout",
        gap_to_support_bps=65.0,
        edge_bps=0.8,
        can_short=False,
    ) is False

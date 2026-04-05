from main import HMATSProductionRunner


def _base_cfg(enabled: bool) -> dict:
    return {
        "btc_short_shadow_relax_enabled": enabled,
        "btc_short_shadow_relax_apply_in_regimes": ["QUIET_ACCUMULATION"],
        "btc_short_shadow_relax_max_support_gap_bps": 45,
        "btc_short_shadow_relax_min_net_edge_bps": 2.0,
        "btc_short_shadow_relax_max_vpin": 0.55,
        "btc_short_shadow_relax_exposure_scale": 0.35,
        "btc_short_shadow_relax_block_on_buy_whale": True,
        "apply_in_system_modes_short": ["NORMAL", "OPPORTUNITY"],
        "short_mean_revert_strategies": ["mean_revert", "mean_reversion"],
    }


def test_btc_short_shadow_relax_candidate_stays_shadow_by_default():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)

    decision = runner._get_btc_short_shadow_relax_decision(
        _base_cfg(enabled=False),
        asset="BTC",
        system_mode="NORMAL",
        regime="QUIET_ACCUMULATION",
        strategy_id="momentum",
        gap_to_support_bps=32.0,
        edge_bps=2.8,
        vpin=0.42,
        whale_veto_short=False,
        can_short=False,
    )

    assert decision["candidate"] is True
    assert decision["applied"] is False
    assert decision["exposure_scale"] == 0.35
    assert "enabled=False" in decision["reason"]


def test_btc_short_shadow_relax_blocks_on_buy_whale_conflict():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)

    decision = runner._get_btc_short_shadow_relax_decision(
        _base_cfg(enabled=False),
        asset="BTC",
        system_mode="NORMAL",
        regime="QUIET_ACCUMULATION",
        strategy_id="momentum",
        gap_to_support_bps=28.0,
        edge_bps=2.6,
        vpin=0.44,
        whale_veto_short=True,
        can_short=False,
    )

    assert decision["candidate"] is False
    assert decision["applied"] is False


def test_btc_short_shadow_relax_can_be_explicitly_enabled():
    runner = HMATSProductionRunner.__new__(HMATSProductionRunner)

    decision = runner._get_btc_short_shadow_relax_decision(
        _base_cfg(enabled=True),
        asset="BTC",
        system_mode="NORMAL",
        regime="QUIET_ACCUMULATION",
        strategy_id="momentum",
        gap_to_support_bps=26.0,
        edge_bps=3.1,
        vpin=0.40,
        whale_veto_short=False,
        can_short=False,
    )

    assert decision["candidate"] is True
    assert decision["applied"] is True
    assert decision["exposure_scale"] == 0.35
    assert "enabled=True" in decision["reason"]

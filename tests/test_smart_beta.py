"""Tests for Smart Beta Controller V1 — noop, observe-only, bounded influence."""
import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.smart_beta_controller import SmartBetaController, SmartBetaConfig, SmartBetaState


@pytest.fixture
def market_data():
    return {
        "regime_state": "QUIET_ACCUMULATION",
        "hurst_exponent": 0.5,
        "mtf_momentum": 0.0,
        "adx": 15.0,
        "vol_regime": 1.0,
        "vol_z_score": 0.0,
        "vpin": 0.35,
        "kurtosis": 3.0,
        "black_swan_sentinel": 3,
        "funding_rate": 0.0001,
        "oi_change_24h_pct": 1.0,
        "liquidation_imbalance": 0.0,
        "_fear_greed_value": 50,
        "_ssc_crowding": 0.2,
    }


@pytest.fixture
def agent_signals():
    return {
        "_regime_alpha_gate_mult": 1.0,
        "_regime_alpha_gate_mult_short": 1.0,
        "_regime_position_size_mult": 1.0,
    }


class TestNoopWhenDisabled:
    def test_disabled_returns_neutral(self, market_data, agent_signals):
        ctrl = SmartBetaController(SmartBetaConfig(enabled=False))
        state = ctrl.compute("BTC", market_data, agent_signals)
        assert state.recommended_alpha_gate_mult == 1.0
        assert state.recommended_size_mult == 1.0
        assert state.recommended_confidence_mult == 1.0

    def test_disabled_apply_is_noop(self, market_data, agent_signals):
        ctrl = SmartBetaController(SmartBetaConfig(enabled=False))
        state = ctrl.compute("BTC", market_data, agent_signals)
        original_gate = agent_signals["_regime_alpha_gate_mult"]
        applied = ctrl.apply_to_agent_signals(state, agent_signals, "BTC")
        assert applied is False
        assert agent_signals["_regime_alpha_gate_mult"] == original_gate


class TestObserveOnly:
    def test_observe_only_computes_state(self, market_data, agent_signals):
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True, mode="observe_only"))
        state = ctrl.compute("BTC", market_data, agent_signals)
        # Should compute non-default values
        assert isinstance(state.explanation_tags, list)

    def test_observe_only_no_decision_change(self, market_data, agent_signals):
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True, mode="observe_only"))
        state = ctrl.compute("BTC", market_data, agent_signals)
        original_gate = agent_signals["_regime_alpha_gate_mult"]
        original_size = agent_signals["_regime_position_size_mult"]
        applied = ctrl.apply_to_agent_signals(state, agent_signals, "BTC")
        assert applied is False
        assert agent_signals["_regime_alpha_gate_mult"] == original_gate
        assert agent_signals["_regime_position_size_mult"] == original_size


class TestBoundedInfluence:
    def test_bounded_influence_modifies_signals(self, market_data, agent_signals):
        """Neutral drift + compressed vol should tighten gate and reduce size."""
        market_data["regime_state"] = "QUIET_ACCUMULATION"
        market_data["vol_regime"] = 0.6  # compressed
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True, mode="bounded_influence"))
        state = ctrl.compute("BTC", market_data, agent_signals)
        applied = ctrl.apply_to_agent_signals(state, agent_signals, "BTC")
        assert applied is True
        # Gate should be raised (tighter)
        assert agent_signals["_regime_alpha_gate_mult"] > 1.0
        # Size should be reduced
        assert agent_signals["_regime_position_size_mult"] < 1.0

    def test_all_clamps_respected(self, market_data, agent_signals):
        """Even extreme inputs should stay within config bounds."""
        market_data["regime_state"] = "PANIC_SELLOFF"
        market_data["vol_regime"] = 3.0
        market_data["vpin"] = 0.95
        market_data["kurtosis"] = 20.0
        market_data["funding_rate"] = 0.01
        market_data["oi_change_24h_pct"] = 15.0
        market_data["liquidation_imbalance"] = 0.9
        market_data["black_swan_sentinel"] = 0
        market_data["_ssc_crowding"] = 0.9

        cfg = SmartBetaConfig(enabled=True, mode="bounded_influence")
        ctrl = SmartBetaController(cfg)
        state = ctrl.compute("BTC", market_data, agent_signals)

        assert cfg.alpha_gate_mult_min <= state.recommended_alpha_gate_mult <= cfg.alpha_gate_mult_max
        assert cfg.size_mult_min <= state.recommended_size_mult <= cfg.size_mult_max
        assert cfg.confidence_mult_min <= state.recommended_confidence_mult <= cfg.confidence_mult_max
        assert cfg.short_restriction_mult_min <= state.short_restriction_mult <= cfg.short_restriction_mult_max


class TestScaleInRestriction:
    def test_neutral_drift_blocks_scale_in(self, market_data, agent_signals):
        market_data["regime_state"] = "NEUTRAL_DRIFT"
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True, mode="bounded_influence"))
        state = ctrl.compute("BTC", market_data, agent_signals)
        assert state.allow_scale_in is False

    def test_strong_trend_allows_scale_in(self, market_data, agent_signals):
        market_data["regime_state"] = "MOMENTUM_RALLY"
        market_data["adx"] = 35.0
        market_data["hurst_exponent"] = 0.7
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True, mode="bounded_influence"))
        state = ctrl.compute("BTC", market_data, agent_signals)
        assert state.allow_scale_in is True


class TestNoAuthorityEscalation:
    def test_no_direction_override(self):
        cfg = SmartBetaConfig()
        assert cfg.allow_direction_override is False

    def test_no_veto(self):
        cfg = SmartBetaConfig()
        assert cfg.allow_veto is False


class TestTrendContext:
    def test_bullish_regime_positive_trend(self, market_data, agent_signals):
        market_data["regime_state"] = "MOMENTUM_RALLY"
        market_data["mtf_momentum"] = 0.05
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True))
        state = ctrl.compute("BTC", market_data, agent_signals)
        assert state.trend_dir > 0

    def test_bearish_regime_negative_trend(self, market_data, agent_signals):
        market_data["regime_state"] = "PANIC_SELLOFF"
        market_data["mtf_momentum"] = -0.05
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True))
        state = ctrl.compute("BTC", market_data, agent_signals)
        assert state.trend_dir < 0


class TestVolatilityContext:
    def test_compressed_vol(self, market_data, agent_signals):
        market_data["vol_regime"] = 0.5
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True))
        state = ctrl.compute("BTC", market_data, agent_signals)
        assert "VOL_COMPRESSED" in state.explanation_tags

    def test_expanded_vol(self, market_data, agent_signals):
        market_data["vol_regime"] = 2.0
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True))
        state = ctrl.compute("BTC", market_data, agent_signals)
        assert "VOL_EXPANDED" in state.explanation_tags


class TestLiquidityContext:
    def test_extreme_funding(self, market_data, agent_signals):
        market_data["funding_rate"] = 0.005  # 0.5% per 8h
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True))
        state = ctrl.compute("BTC", market_data, agent_signals)
        assert "FUNDING_EXTREME" in state.explanation_tags

    def test_high_crowding(self, market_data, agent_signals):
        market_data["_ssc_crowding"] = 0.8
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True))
        state = ctrl.compute("BTC", market_data, agent_signals)
        assert "HIGH_CROWDING" in state.explanation_tags


class TestCarryOpportunityDeleted:
    """[P228] The CARRY_OPPORTUNITY_* tags are DELETED, and these tests pin
    the deletion. The old tests here validated a regime that does not occur
    (they used funding_rate=-0.001 — 2.6x the largest real reading; the tag
    never fired once on live data) against an implementation that was
    directionally broken (the "LONG-only" relaxation loosened BOTH gate keys
    while the compensating short mult was never wired) keyed off the wrong
    venue's funding sign (P218). See core/smart_beta_controller.py [P228]."""

    @pytest.mark.parametrize("funding", [-0.01, -0.001, 0.001, 0.01])
    def test_no_carry_tag_at_any_funding_level(
            self, market_data, agent_signals, funding):
        """Even at extreme funding the tags must not resurface — if a carry
        strategy returns, it comes through shadow + the P166 gate, not a
        resurrected tag."""
        market_data["funding_rate"] = funding
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True))
        state = ctrl.compute("ETH", market_data, agent_signals)
        assert "CARRY_OPPORTUNITY_LONG" not in state.explanation_tags
        assert "CARRY_OPPORTUNITY_SHORT" not in state.explanation_tags

    def test_funding_heat_is_still_computed(self, market_data, agent_signals):
        """Deleting the tag must not delete the diagnostic — funding_heat
        stays observable so future evidence can accumulate."""
        market_data["funding_rate"] = -0.001
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True))
        state = ctrl.compute("ETH", market_data, agent_signals)
        assert state.funding_heat < 0

    def test_unwired_channels_are_recorded_as_deliberate(self):
        """short_restriction_mult / recommended_confidence_mult are
        diagnostics-only BY DECISION — the source must say so."""
        import inspect
        from core import smart_beta_controller
        src = inspect.getsource(smart_beta_controller)
        assert "DIAGNOSTICS" in src and "[P228]" in src


class TestExistingDataOnlyGuard:
    def test_no_external_api_dependency(self):
        """SmartBetaController only reads from market_data dict — no API calls."""
        import inspect
        from core.smart_beta_controller import SmartBetaController
        source = inspect.getsource(SmartBetaController.compute)
        # Must NOT contain any API/fetch/request calls
        forbidden = ["requests.get", "aiohttp", "fetch(", "api_call", "httpx"]
        for f in forbidden:
            assert f not in source, f"SmartBetaController.compute contains {f}"

    def test_only_reads_market_data_dict(self, market_data, agent_signals):
        """Compute only reads from market_data and agent_signals — no side effects."""
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True))
        md_copy = dict(market_data)
        as_copy = dict(agent_signals)
        ctrl.compute("BTC", market_data, agent_signals)
        # market_data should not be modified by compute()
        assert market_data == md_copy


class TestNoObsContractChange:
    def test_obs_dim_unchanged(self):
        """Smart Beta does NOT modify obs_dim or feature_manifest."""
        import json
        from pathlib import Path
        manifest = Path("configs/feature_manifest.json")
        if manifest.exists():
            with open(manifest, encoding="utf-8") as f:
                m = json.load(f)
            # obs_dim should still be 122 features + 4 env = 126
            total_features = m.get("total_feature_count", m.get("total_features", 0))
            assert total_features == 122, f"feature_manifest changed! expected 122, got {total_features}"

    def test_smart_beta_does_not_import_training(self):
        """SmartBetaController must not import training modules."""
        import inspect
        from core.smart_beta_controller import SmartBetaController
        source = inspect.getsource(SmartBetaController)
        assert "train_drl" not in source
        assert "feature_manifest" not in source
        assert "obs_dim" not in source


class TestNeutralDriftScore:
    def test_quiet_accum_has_high_drift(self, market_data, agent_signals):
        market_data["regime_state"] = "QUIET_ACCUMULATION"
        market_data["hurst_exponent"] = 0.5
        market_data["adx"] = 12.0
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True))
        state = ctrl.compute("BTC", market_data, agent_signals)
        assert state.neutral_drift_score > 0.3

    def test_momentum_rally_has_low_drift(self, market_data, agent_signals):
        market_data["regime_state"] = "MOMENTUM_RALLY"
        market_data["adx"] = 35.0
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True))
        state = ctrl.compute("BTC", market_data, agent_signals)
        assert state.neutral_drift_score == 0.0


class TestDownsideBeta:
    def test_downside_beta_computed(self):
        """BetaExposureAnalyzer should compute downside_beta."""
        import numpy as np
        from analytics.beta_exposure import BetaExposureAnalyzer
        analyzer = BetaExposureAnalyzer()
        np.random.seed(42)
        s = np.random.normal(0, 0.01, 100)
        b = np.random.normal(0, 0.01, 100)
        db = analyzer.compute_downside_beta(s, b)
        assert isinstance(db, float)
        assert abs(db) < 5.0  # sanity bound


class TestExistingContractsUntouched:
    def test_sentiment_not_modified(self, market_data, agent_signals):
        agent_signals["sentiment_zscore"] = -2.0
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True, mode="bounded_influence"))
        state = ctrl.compute("BTC", market_data, agent_signals)
        ctrl.apply_to_agent_signals(state, agent_signals, "BTC")
        assert agent_signals["sentiment_zscore"] == -2.0  # untouched

    def test_drl_direction_not_modified(self, market_data, agent_signals):
        agent_signals["drl_direction"] = 0.8
        ctrl = SmartBetaController(SmartBetaConfig(enabled=True, mode="bounded_influence"))
        state = ctrl.compute("BTC", market_data, agent_signals)
        ctrl.apply_to_agent_signals(state, agent_signals, "BTC")
        assert agent_signals["drl_direction"] == 0.8  # untouched

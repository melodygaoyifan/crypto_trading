from pathlib import Path


def test_strategic_coordinator_refreshes_on_config_change(monkeypatch):
    import orchestration.strategic_coordinator as mod

    class FakeCoordinator:
        def __init__(self, config=None):
            self.config = config or {}

    mod.reset_coordinator()
    monkeypatch.setattr(mod, "StrategicCoordinator", FakeCoordinator)

    first = mod.get_strategic_coordinator({"mode": "a"})
    second = mod.get_strategic_coordinator({"mode": "a"})
    third = mod.get_strategic_coordinator({"mode": "b"})

    assert first is second
    assert third is not first
    assert third.config == {"mode": "b"}


def test_correlation_controllers_refresh_on_config_change():
    from risk.correlation_realtime_controller import (
        CorrelationControllerConfig,
        get_correlation_controller,
        get_model_alpha_weight_controller,
        reset_correlation_controller,
    )

    reset_correlation_controller()
    first_cfg = CorrelationControllerConfig(short_window=12)
    second_cfg = CorrelationControllerConfig(short_window=24)

    first = get_correlation_controller(first_cfg)
    second = get_correlation_controller(first_cfg)
    third = get_correlation_controller(second_cfg)

    assert first is second
    assert third is not first

    weight_first = get_model_alpha_weight_controller(first)
    weight_second = get_model_alpha_weight_controller(third)

    assert weight_second is not weight_first
    assert weight_second.corr_controller is third


def test_cloud_config_refreshes_on_path_change(tmp_path):
    import json

    from core.cloud_config import get_cloud_config, reset_cloud_config

    reset_cloud_config()
    first_path = tmp_path / "a.json"
    second_path = tmp_path / "b.json"
    first_path.write_text(json.dumps({"mode": "paper"}), encoding="utf-8")
    second_path.write_text(json.dumps({"mode": "live"}), encoding="utf-8")

    first = get_cloud_config(first_path)
    second = get_cloud_config(first_path)
    third = get_cloud_config(second_path)

    assert first is second
    assert third is not first


def test_account_sync_refreshes_on_constructor_input_change():
    from core.account_sync import get_account_sync, reset_account_sync

    reset_account_sync()
    client_a = object()
    client_b = object()

    first = get_account_sync(exchange_client=client_a, exchange_name="kraken", dry_run=True)
    second = get_account_sync(exchange_client=client_a, exchange_name="kraken", dry_run=True)
    third = get_account_sync(exchange_client=client_b, exchange_name="kraken", dry_run=True)

    assert first is second
    assert third is not first


def test_exchange_guard_refreshes_on_mode_change():
    from core.exchange_guard import get_exchange_guard, reset_exchange_guard

    reset_exchange_guard()
    first = get_exchange_guard("verify")
    second = get_exchange_guard("verify")
    third = get_exchange_guard("observe")

    assert first is second
    assert third is not first
    assert third.mode == "observe"


def test_attribution_tracker_refreshes_on_log_dir_change(tmp_path):
    from agents.attribution_tracker import get_attribution_tracker, reset_attribution_tracker

    reset_attribution_tracker()
    first = get_attribution_tracker(str(tmp_path / "a"))
    second = get_attribution_tracker(str(tmp_path / "a"))
    third = get_attribution_tracker(str(tmp_path / "b"))

    assert first is second
    assert third is not first


def test_fee_calculator_refreshes_on_config_change():
    from execution.fees.kraken_plus_fee_blender import (
        KrakenPlusFeeConfig,
        get_fee_calculator,
        reset_fee_calculator,
    )

    reset_fee_calculator()
    first_cfg = KrakenPlusFeeConfig(free_tier_usd=10_000.0)
    second_cfg = KrakenPlusFeeConfig(free_tier_usd=25_000.0)

    first = get_fee_calculator(first_cfg)
    second = get_fee_calculator(first_cfg)
    third = get_fee_calculator(second_cfg)

    assert first is second
    assert third is not first
    assert third.config.free_tier_usd == 25_000.0


def test_fringe_panic_detector_refreshes_on_monitor_change(monkeypatch):
    import signals.fringe_panic_detector as mod

    class FakeFusion:
        def __init__(self, x_monitor=None, trends_monitor=None, reddit_monitor=None):
            self._x = x_monitor
            self._trends = trends_monitor
            self._reddit = reddit_monitor

    mod.reset_fringe_panic_detector()
    monkeypatch.setattr(mod, "FringePanicFusion", FakeFusion)

    x_a = object()
    x_b = object()
    first = mod.get_fringe_panic_detector(x_monitor=x_a)
    second = mod.get_fringe_panic_detector(x_monitor=x_a)
    third = mod.get_fringe_panic_detector(x_monitor=x_b)

    assert first is second
    assert third is not first
    assert third._x is x_b


def test_adaptive_weight_manager_refreshes_on_config_change():
    from signals.adaptive_weight_v521 import (
        AdaptiveWeightConfigV521,
        get_adaptive_weight_manager_v521,
        reset_adaptive_weight_manager_v521,
    )

    reset_adaptive_weight_manager_v521()
    first_cfg = AdaptiveWeightConfigV521(evaluation_interval_hours=4)
    second_cfg = AdaptiveWeightConfigV521(evaluation_interval_hours=8)

    first = get_adaptive_weight_manager_v521(first_cfg)
    second = get_adaptive_weight_manager_v521(first_cfg)
    third = get_adaptive_weight_manager_v521(second_cfg)

    assert first is second
    assert third is not first
    assert third.config.evaluation_interval_hours == 8


def test_macro_crowd_adapter_refreshes_on_feed_change(monkeypatch):
    import signals.macro_crowd_adapter as mod

    class FakeAdapter:
        def __init__(
            self,
            fred_feed=None,
            coinglass_feed=None,
            lunarcrush_feed=None,
            cryptopanic_feed=None,
            sentiment_engine=None,
        ):
            self._fred_feed = fred_feed
            self._coinglass_feed = coinglass_feed
            self._lunarcrush_feed = lunarcrush_feed
            self._cryptopanic_feed = cryptopanic_feed
            self._sentiment_engine = sentiment_engine

    mod.reset_macro_crowd_adapter()
    monkeypatch.setattr(mod, "MacroCrowdAdapter", FakeAdapter)

    fred_a = object()
    fred_b = object()
    first = mod.get_macro_crowd_adapter(fred_feed=fred_a)
    second = mod.get_macro_crowd_adapter(fred_feed=fred_a)
    third = mod.get_macro_crowd_adapter(fred_feed=fred_b)

    assert first is second
    assert third is not first
    assert third._fred_feed is fred_b

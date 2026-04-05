from integration.integration_v36 import DRLAuthorityLevel, HMATSv36Engine


class _DisabledDRLGate:
    def get_authority(self):
        return DRLAuthorityLevel.DISABLED


def _make_engine():
    engine = HMATSv36Engine.__new__(HMATSv36Engine)
    engine.drl_gate = _DisabledDRLGate()
    engine._last_sentiment_active = False
    engine._last_macro_active = False
    engine._last_phase = ""
    return engine


def test_extreme_greed_sentiment_is_modulation_only():
    engine = _make_engine()

    signals = engine._build_fusion_signals({"sentiment_zscore": 4.0}, {})
    sentiment = signals["sentiment"]

    assert sentiment.direction == 1.0
    assert sentiment.confidence == 0.5
    assert sentiment.veto_active is False
    assert sentiment.veto_direction is None


def test_extreme_fear_sentiment_is_modulation_only():
    engine = _make_engine()

    signals = engine._build_fusion_signals({"sentiment_zscore": -4.0}, {})
    sentiment = signals["sentiment"]

    assert sentiment.direction == -1.0
    assert sentiment.confidence == 0.5
    assert sentiment.veto_active is False
    assert sentiment.veto_direction is None

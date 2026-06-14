"""Trend decision layer tests — flag-gated, default OFF (the ready-to-flip integration)."""

from core.trend_decision_layer import TrendDecisionLayer, get_trend_decision_layer


class _Intent:
    def __init__(self, direction=0.0, target_exposure=0.0):
        self.direction = direction
        self.target_exposure = target_exposure
        self.is_actionable = True
        self.veto_active = False
        self.quant_strategy_id = "quant"


class _DF:
    """Minimal stand-in for the engine OHLCV df: len() + ['close'].tolist()."""
    def __init__(self, closes):
        self._c = closes

    def __len__(self):
        return len(self._c)

    def __getitem__(self, k):
        assert k == "close"
        class _Col:
            def __init__(s, c): s._c = c
            def tolist(s): return list(s._c)
        return _Col(self._c)


def _uptrend(n):
    c = [100.0]
    for _ in range(n - 1):
        c.append(c[-1] * 1.001)
    return c


def _downtrend(n):
    c = [100.0]
    for _ in range(n - 1):
        c.append(c[-1] * 0.999)
    return c


class TestTrendDecisionLayer:
    def test_default_off_is_noop(self):
        tl = TrendDecisionLayer()  # default off
        assert tl.mode == "off"
        tl.cache("BTC", _DF(_uptrend(500)))
        intent = _Intent(direction=0.5)
        assert tl.apply("BTC", intent) is None
        assert intent.direction == 0.5  # unchanged

    def test_shadow_logs_does_not_change_intent(self):
        tl = TrendDecisionLayer(mode="shadow")
        tl.cache("BTC", _DF(_uptrend(500)))
        intent = _Intent(direction=-0.7)
        out = tl.apply("BTC", intent)
        assert out["mode"] == "shadow" and out["trend_dir"] > 0  # uptrend -> long
        assert intent.direction == -0.7  # NOT changed in shadow

    def test_enforce_overrides_intent_long_in_uptrend(self):
        tl = TrendDecisionLayer(mode="enforce")
        tl.cache("BTC", _DF(_uptrend(500)))
        intent = _Intent(direction=-0.7)  # engine said short
        out = tl.apply("BTC", intent)
        assert out["mode"] == "enforce"
        assert intent.direction > 0 and intent.target_exposure > 0  # trend overrode -> long
        assert intent.quant_strategy_id == "trend_following"

    def test_enforce_shorts_in_downtrend(self):
        tl = TrendDecisionLayer(mode="enforce")
        tl.cache("ETH", _DF(_downtrend(500)))
        intent = _Intent(direction=0.7)
        tl.apply("ETH", intent)
        assert intent.direction < 0  # downtrend -> trend goes short

    def test_insufficient_history_no_decision(self):
        tl = TrendDecisionLayer(mode="enforce")
        tl.cache("BTC", _DF(_uptrend(50)))  # < min_history -> not cached
        intent = _Intent(direction=0.3)
        assert tl.apply("BTC", intent) is None
        assert intent.direction == 0.3

    def test_exposure_capped(self):
        tl = TrendDecisionLayer(mode="enforce", max_exposure=0.20)
        tl.cache("SOL", _DF(_uptrend(500)))
        intent = _Intent()
        tl.apply("SOL", intent)
        assert abs(intent.target_exposure) <= 0.20 + 1e-9

    def test_fail_open_on_bad_df(self):
        tl = TrendDecisionLayer(mode="enforce")
        tl.cache("BTC", None)  # no df
        intent = _Intent(direction=0.4)
        assert tl.apply("BTC", intent) is None
        assert intent.direction == 0.4

    def test_singleton_mode_update(self):
        a = get_trend_decision_layer("shadow")
        b = get_trend_decision_layer("off")
        assert a is b and b.mode == "off"

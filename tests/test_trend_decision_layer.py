"""Trend decision layer tests — flag-gated, default OFF; enforce injects trend AS quant signal."""

from core.trend_decision_layer import TrendDecisionLayer, get_trend_decision_layer


class _DF:
    def __init__(self, closes): self._c = closes
    def __len__(self): return len(self._c)
    def __getitem__(self, k):
        assert k == "close"
        class _Col:
            def __init__(s, c): s._c = c
            def tolist(s): return list(s._c)
        return _Col(self._c)


def _trend(n, step):
    c = [100.0]
    for _ in range(n - 1):
        c.append(c[-1] * (1 + step))
    return c


def _md(qd=0.0):
    return {"quant_direction": qd, "quant_confidence": 0.0, "signal_edge_bps": 0.0}


class TestProcess:
    def test_default_off_noop(self):
        tl = TrendDecisionLayer()
        assert tl.mode == "off"
        md, ags = _md(0.5), {"quant_direction": 0.5}
        assert tl.process("BTC", _DF(_trend(500, 0.001)), ags, md) is None
        assert md["quant_direction"] == 0.5  # unchanged

    def test_shadow_logs_no_change(self):
        tl = TrendDecisionLayer(mode="shadow")
        md, ags = _md(-0.7), {"quant_direction": -0.7}
        out = tl.process("BTC", _DF(_trend(500, 0.001)), ags, md)
        assert out["mode"] == "shadow" and out["trend_signal"] > 0  # uptrend
        assert md["quant_direction"] == -0.7  # NOT changed in shadow

    def test_enforce_injects_trend_as_quant_uptrend(self):
        tl = TrendDecisionLayer(mode="enforce")
        md, ags = _md(-0.7), {"quant_direction": -0.7}  # engine said short
        out = tl.process("BTC", _DF(_trend(500, 0.001)), ags, md)
        assert out["mode"] == "enforce"
        assert md["quant_direction"] > 0 and ags["quant_direction"] > 0  # now LONG (trend)
        assert md["signal_edge_bps"] > 0  # gate-passing edge injected
        assert ags["primary_strategy"] == "trend_following"

    def test_enforce_shorts_in_downtrend(self):
        tl = TrendDecisionLayer(mode="enforce")
        md, ags = _md(0.7), {"quant_direction": 0.7}
        tl.process("ETH", _DF(_trend(500, -0.001)), ags, md)
        assert md["quant_direction"] < 0  # downtrend -> trend shorts (the regime fix)

    def test_weak_trend_goes_flat(self):
        tl = TrendDecisionLayer(mode="enforce", min_abs_signal=0.5)
        # a flat-ish series -> |signal| below threshold -> injected direction 0
        md, ags = _md(0.6), {"quant_direction": 0.6}
        # mostly-flat with tiny noise so signal averages near 0
        import random
        rng = random.Random(0)
        c = [100.0]
        for _ in range(499):
            c.append(c[-1] * (1 + 0.0001 * rng.gauss(0, 1)))
        out = tl.process("SOL", _DF(c), ags, md)
        if out:  # if it produced a decision, a weak one must be flat
            assert md["quant_direction"] == 0.0

    def test_insufficient_history_noop(self):
        tl = TrendDecisionLayer(mode="enforce")
        md, ags = _md(0.3), {"quant_direction": 0.3}
        assert tl.process("BTC", _DF(_trend(50, 0.001)), ags, md) is None
        assert md["quant_direction"] == 0.3

    def test_fail_open_bad_df(self):
        tl = TrendDecisionLayer(mode="enforce")
        md, ags = _md(0.4), {"quant_direction": 0.4}
        assert tl.process("BTC", None, ags, md) is None
        assert md["quant_direction"] == 0.4

    def test_singleton_mode_update(self):
        a = get_trend_decision_layer("shadow")
        b = get_trend_decision_layer("off")
        assert a is b and b.mode == "off"

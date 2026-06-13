"""
RegimeICFusion module tests (Layers 1+2 shadow tool).

This module is SHADOW-ONLY: it never decides a live trade. These tests lock the
mechanics (regime-conditional sign, IC-weighting, stable-sign carve-out, warmup
gating, fail-open, persistence). The empirical question of whether it BEATS the
current path is answered by walk-forward OOS on live data — and as of 2026-06-13
it does NOT clearly win (it stays shadow). See
docs/HMATS_IMPLEMENTATION_PROMPT_ALPHA_BETA_ENDTOEND.md.
"""

from signals.regime_ic_fusion import RegimeICFusion


def _train(fus, regime, agent, direction, fwd_ret, n):
    for _ in range(n):
        fus.record_outcome(regime, {agent: direction}, fwd_ret)


class TestRegimeICFusion:
    def test_warmup_returns_none(self):
        fus = RegimeICFusion(min_samples=20)
        # no history -> fail-open None
        assert fus.shadow_fuse("QUIET", {"quant": 1.0}) is None

    def test_ic_edge_and_count(self):
        fus = RegimeICFusion(window=50, min_samples=5)
        # long signals that were followed by +returns -> positive edge
        _train(fus, "QUIET", "quant", 1.0, 0.01, 10)
        edge, n = fus.ic("quant", "QUIET")
        assert n == 10 and edge > 0

    def test_regime_conditional_sign_flip(self):
        # a flipping agent whose +signal was followed by NEGATIVE returns in this
        # regime -> the regime sign should INVERT the contribution.
        fus = RegimeICFusion(window=50, min_samples=5)
        _train(fus, "UPTREND", "quant", 1.0, -0.01, 12)  # long -> down: negative edge
        out = fus.shadow_fuse("UPTREND", {"quant": 1.0})
        assert out is not None
        # signal is +1 but regime IC is negative -> contribution flips to short
        assert out["direction"] < 0

    def test_stable_sign_agent_not_flipped(self):
        # whale is in the stable-sign allowlist: even with negative measured edge,
        # it is used BLIND (sign follows the signal, not the regime IC).
        fus = RegimeICFusion(window=50, min_samples=5)
        _train(fus, "UPTREND", "whale", 1.0, -0.01, 12)
        out = fus.shadow_fuse("UPTREND", {"whale": 1.0})
        assert out is not None
        assert out["direction"] > 0  # blind: follows the +1 signal, not flipped

    def test_zero_ic_agent_gets_no_weight(self):
        # an agent with ~zero edge contributes ~zero weight (Layer 2 auto-demote)
        fus = RegimeICFusion(window=50, min_samples=5)
        # alternating returns -> edge ~ 0
        for i in range(20):
            fus.record_outcome("QUIET", {"noise": 1.0}, 0.01 if i % 2 else -0.01)
        edge, n = fus.ic("noise", "QUIET")
        assert n >= 5 and abs(edge) < 2e-3

    def test_below_min_samples_excluded(self):
        fus = RegimeICFusion(window=50, min_samples=20)
        _train(fus, "QUIET", "quant", 1.0, 0.01, 5)  # only 5 < 20
        assert fus.shadow_fuse("QUIET", {"quant": 1.0}) is None

    def test_fail_open_on_bad_input(self):
        fus = RegimeICFusion(min_samples=1)
        # non-numeric direction must not raise
        out = fus.shadow_fuse("QUIET", {"quant": "not_a_number"})
        assert out is None

    def test_persistence_roundtrip(self):
        fus = RegimeICFusion(window=50, min_samples=5)
        _train(fus, "QUIET", "quant", 1.0, 0.01, 10)
        blob = fus.to_dict()
        fus2 = RegimeICFusion(window=50, min_samples=5)
        fus2.from_dict(blob)
        e1, n1 = fus.ic("quant", "QUIET")
        e2, n2 = fus2.ic("quant", "QUIET")
        assert n1 == n2 and abs(e1 - e2) < 1e-9

    def test_never_decides__shadow_only_contract(self):
        # structural guarantee: the module exposes no enforce/decide method
        assert not hasattr(RegimeICFusion, "decide")
        assert not hasattr(RegimeICFusion, "enforce")

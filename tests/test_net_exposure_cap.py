"""
NET-CAP (2026-06-14) regression tests.

Forensic: the book ran +0.54 net-long into a -23% market = ~half the Apr-Jun loss.
Gross caps don't constrain net DIRECTION (BTC+ETH+SOL all-long passes a gross cap),
and `net_exposure` was computed but never enforced. This cap clamps |net signed
exposure| to a budget — but ONLY ever reduces a request that INCREASES |net|;
de-risking (reducing |net|) is always free.
"""

from risk.global_exposure_cap import GlobalExposureCapManager, ExposureCapConfig


def _mgr(net_cap):
    return GlobalExposureCapManager(ExposureCapConfig(max_net_exposure=net_cap))


class TestNetExposureCap:
    def test_off_by_default(self):
        m = _mgr(None)  # OFF
        m.update_positions(btc_exposure=0.40, eth_exposure=0.0, sol_exposure=0.0)
        r = m.validate_new_exposure("BTC", 0.30)  # net would be 0.70
        assert abs(r.adjusted_size - 0.30) < 1e-9  # unconstrained by net

    def test_clamps_same_asset_add_beyond_budget(self):
        m = _mgr(0.50)
        m.update_positions(btc_exposure=0.40, eth_exposure=0.0, sol_exposure=0.0)
        r = m.validate_new_exposure("BTC", 0.30)  # net 0.40 -> would be 0.70
        assert abs(r.adjusted_size - 0.10) < 1e-9  # clamped to 0.50-0.40

    def test_clamps_cross_asset_same_direction(self):
        # BTC already +0.40 net-long; adding ETH long is blocked by the NET budget
        m = _mgr(0.50)
        m.update_positions(btc_exposure=0.40, eth_exposure=0.0, sol_exposure=0.0)
        r = m.validate_new_exposure("ETH", 0.30)
        assert abs(r.adjusted_size - 0.10) < 1e-9
        assert "NET" in r.adjustment_reason

    def test_derisking_always_free(self):
        # net 0.60 (over budget) — reducing it must NOT be clamped
        m = _mgr(0.50)
        m.update_positions(btc_exposure=0.60, eth_exposure=0.0, sol_exposure=0.0)
        r = m.validate_new_exposure("BTC", -0.30)  # net -> 0.30
        assert abs(r.adjusted_size - (-0.30)) < 1e-9

    def test_opposite_direction_within_budget_allowed(self):
        m = _mgr(0.50)
        m.update_positions(btc_exposure=0.0, eth_exposure=0.0, sol_exposure=0.0)
        r = m.validate_new_exposure("ETH", -0.40)  # net -> -0.40, |.|<0.50
        assert abs(r.adjusted_size - (-0.40)) < 1e-9

    def test_hedging_reduces_net_allowed(self):
        # long BTC, opening a SHORT ETH reduces net -> allowed (market-neutralizing)
        m = _mgr(0.50)
        m.update_positions(btc_exposure=0.50, eth_exposure=0.0, sol_exposure=0.0)
        r = m.validate_new_exposure("ETH", -0.30)  # net 0.50 -> 0.20
        assert abs(r.adjusted_size - (-0.30)) < 1e-9

    def test_net_short_beyond_budget_clamped(self):
        m = _mgr(0.50)
        m.update_positions(btc_exposure=-0.40, eth_exposure=0.0, sol_exposure=0.0)
        r = m.validate_new_exposure("BTC", -0.30)  # net -0.40 -> -0.70
        assert abs(r.adjusted_size - (-0.10)) < 1e-9  # clamped to budget

"""Layer-3 Sharpe validation harness tests (analytics-only; never touches trading)."""

import math
import random

from analytics.validation.sharpe_validation import (
    annualized_sharpe, probabilistic_sharpe_ratio, backtest_vs_live,
    deflated_sharpe_ratio, _ncdf, _ppf,
)


class TestNormalHelpers:
    def test_ncdf(self):
        assert abs(_ncdf(0.0) - 0.5) < 1e-9
        assert _ncdf(3.0) > 0.99

    def test_ppf_inverse(self):
        for p in (0.1, 0.5, 0.9, 0.975):
            assert abs(_ncdf(_ppf(p)) - p) < 1e-3


class TestPSR:
    def test_strong_positive_returns_high_psr(self):
        rng = random.Random(42)
        rets = [0.01 + 0.005 * rng.gauss(0, 1) for _ in range(200)]  # SR ~ 2/period
        psr = probabilistic_sharpe_ratio(rets)
        assert psr["psr"] > 0.95 and psr["sr"] > 0

    def test_noise_returns_low_psr(self):
        rng = random.Random(7)
        rets = [0.0001 + 0.02 * rng.gauss(0, 1) for _ in range(40)]  # tiny edge, short
        psr = probabilistic_sharpe_ratio(rets)
        assert psr["psr"] < 0.95

    def test_short_sample_not_confirmed(self):
        psr = probabilistic_sharpe_ratio([0.01, 0.02], 0.0)
        assert psr["psr"] == 0.0  # n<3 degenerate


class TestBacktestVsLive:
    def test_implausible_backtest_flagged_overfit(self):
        rng = random.Random(1)
        live = [-0.001 + 0.03 * rng.gauss(0, 1) for _ in range(50)]  # negative-ish, noisy
        r = backtest_vs_live(9.22, live, 365)
        assert r["backtest_implausible_flag"] is True
        assert r["verdict"] == "OVERFIT_SUSPECT"
        assert r["gap"] > 3.0

    def test_modest_backtest_matching_live_confirmed(self):
        rng = random.Random(2)
        live = [0.01 + 0.004 * rng.gauss(0, 1) for _ in range(250)]  # real edge, long
        r = backtest_vs_live(1.5, live, 365)
        assert r["verdict"] == "LIVE_CONFIRMED"
        assert r["backtest_implausible_flag"] is False


class TestDSRCaveat:
    def test_dsr_carries_refutation_caveat(self):
        rng = random.Random(3)
        rets = [0.005 + 0.01 * rng.gauss(0, 1) for _ in range(100)]
        out = deflated_sharpe_ratio(rets, n_trials=30, sr_variance_across_trials=0.02)
        assert "dsr" in out and "_caveat" in out
        assert "refuted" in out["_caveat"]  # must not present as a precise gate


class TestAnnualization:
    def test_annualization_scales_by_sqrt(self):
        rets = [0.001] * 100  # zero variance -> sharpe 0 (degenerate guard)
        assert annualized_sharpe(rets, 365) == 0.0
        rng = random.Random(9)
        r2 = [0.001 + 0.01 * rng.gauss(0, 1) for _ in range(100)]
        from analytics.validation.sharpe_validation import sharpe
        assert abs(annualized_sharpe(r2, 365) - sharpe(r2) * math.sqrt(365)) < 1e-9

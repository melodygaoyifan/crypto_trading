"""
Tests for exit trigger logic — strategy-differentiated exits, time stops,
alpha fade, gambler soft exit thresholds.
"""
import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.high_risk_mode import HighRiskModeConfig


@pytest.fixture
def config():
    return HighRiskModeConfig()


# =============================================================================
# STRATEGY-DIFFERENTIATED EXIT PARAMETERS
# =============================================================================

class TestStrategyExitOverrides:
    def test_mean_revert_has_wider_pnl_threshold(self, config):
        """Mean_revert: -65bps (wider than default -42bps)."""
        mr_pnl = config.get_soft_exit_pnl_threshold(gambler_mode=False, strategy="mean_revert")
        default_pnl = config.get_soft_exit_pnl_threshold(gambler_mode=False, strategy="")
        assert mr_pnl < default_pnl  # -65 < -42 (wider = more negative)

    def test_momentum_has_tighter_pnl_threshold(self, config):
        """Momentum: -30bps (tighter than default -42bps)."""
        mom_pnl = config.get_soft_exit_pnl_threshold(gambler_mode=False, strategy="momentum")
        default_pnl = config.get_soft_exit_pnl_threshold(gambler_mode=False, strategy="")
        assert mom_pnl > default_pnl  # -30 > -42 (tighter = less negative)

    def test_mean_revert_more_structure_bars(self, config):
        """Mean_revert: 5 bars vs default 3."""
        mr_bars = config.get_structure_failure_bars(gambler_mode=False, strategy="mean_revert")
        default_bars = config.get_structure_failure_bars(gambler_mode=False, strategy="")
        assert mr_bars > default_bars
        assert mr_bars == 5

    def test_momentum_fewer_structure_bars(self, config):
        """Momentum: 2 bars (confirms fast)."""
        mom_bars = config.get_structure_failure_bars(gambler_mode=False, strategy="momentum")
        assert mom_bars == 2

    def test_mean_revert_longer_timeout(self, config):
        """Mean_revert: 180min vs default 120min."""
        mr_timeout = config.get_no_confirmation_timeout(gambler_mode=False, strategy="mean_revert")
        default_timeout = config.get_no_confirmation_timeout(gambler_mode=False, strategy="")
        assert mr_timeout > default_timeout
        assert mr_timeout == 180.0

    def test_unknown_strategy_uses_default(self, config):
        """Unknown strategy name falls back to mode-based default."""
        unknown = config.get_soft_exit_pnl_threshold(gambler_mode=False, strategy="unknown_strategy")
        default = config.get_soft_exit_pnl_threshold(gambler_mode=False, strategy="")
        assert unknown == default

    def test_gambler_mode_overridden_by_strategy(self, config):
        """Strategy override takes precedence over gambler mode."""
        mr_normal = config.get_soft_exit_pnl_threshold(gambler_mode=False, strategy="mean_revert")
        mr_gambler = config.get_soft_exit_pnl_threshold(gambler_mode=True, strategy="mean_revert")
        # Both should return -65.0 (strategy override, not gambler)
        assert mr_normal == mr_gambler == -65.0


# =============================================================================
# TIME STOP BARS (Strategy-Aware)
# =============================================================================

class TestTimeStopBars:
    def test_mean_revert_losing_gets_4_bars(self, config):
        bars = config.get_time_stop_bars(is_profitable=False, strategy="mean_revert")
        assert bars == 4

    def test_mean_revert_winning_gets_6_bars(self, config):
        bars = config.get_time_stop_bars(is_profitable=True, strategy="mean_revert")
        assert bars == 6

    def test_momentum_losing_gets_2_bars(self, config):
        bars = config.get_time_stop_bars(is_profitable=False, strategy="momentum")
        assert bars == 2

    def test_momentum_winning_gets_4_bars(self, config):
        bars = config.get_time_stop_bars(is_profitable=True, strategy="momentum")
        assert bars == 4

    def test_default_losing_is_2(self, config):
        bars = config.get_time_stop_bars(is_profitable=False, strategy="")
        assert bars == 2

    def test_default_winning_is_4(self, config):
        bars = config.get_time_stop_bars(is_profitable=True, strategy="")
        assert bars == 4


# =============================================================================
# RUNNER TRAIL PERCENTAGES (Strategy-Aware)
# =============================================================================

class TestRegimeAwareTrail:
    def test_trending_regime_widens_trail(self, config):
        """MOMENTUM_RALLY should widen trail by 50%."""
        normal = config.get_runner_trail_pct(tight=False, strategy="momentum", regime="")
        trending = config.get_runner_trail_pct(tight=False, strategy="momentum", regime="MOMENTUM_RALLY")
        assert trending == pytest.approx(normal * 1.5, rel=0.01)

    def test_chop_regime_no_widening(self, config):
        """QUIET_ACCUMULATION should NOT widen trail."""
        normal = config.get_runner_trail_pct(tight=False, strategy="momentum", regime="")
        chop = config.get_runner_trail_pct(tight=False, strategy="momentum", regime="QUIET_ACCUMULATION")
        assert chop == normal

    def test_panic_selloff_widens_trail(self, config):
        """PANIC_SELLOFF is also a trending regime — should widen."""
        normal = config.get_runner_trail_pct(tight=False, strategy="momentum", regime="")
        panic = config.get_runner_trail_pct(tight=False, strategy="momentum", regime="PANIC_SELLOFF")
        assert panic > normal

    def test_tight_trail_also_regime_widened(self, config):
        """Tight trail in trending regime should also be widened."""
        normal_tight = config.get_runner_trail_pct(tight=True, strategy="momentum", regime="")
        trend_tight = config.get_runner_trail_pct(tight=True, strategy="momentum", regime="STEADY_UPTREND")
        assert trend_tight == pytest.approx(normal_tight * 1.5, rel=0.01)


class TestRunnerTrail:
    def test_mean_revert_wider_initial_trail(self, config):
        mr = config.get_runner_trail_pct(tight=False, strategy="mean_revert")
        mom = config.get_runner_trail_pct(tight=False, strategy="momentum")
        assert mr > mom  # 4% > 2.5%
        assert mr == pytest.approx(0.04)

    def test_momentum_tighter_trail(self, config):
        mom = config.get_runner_trail_pct(tight=False, strategy="momentum")
        assert mom == pytest.approx(0.025)

    def test_tight_trail_smaller_than_initial(self, config):
        for strategy in ["mean_revert", "momentum", "volume_breakout", ""]:
            initial = config.get_runner_trail_pct(tight=False, strategy=strategy)
            tight = config.get_runner_trail_pct(tight=True, strategy=strategy)
            assert tight < initial, f"{strategy}: tight {tight} >= initial {initial}"

    def test_default_trail_matches_config(self, config):
        default_init = config.get_runner_trail_pct(tight=False, strategy="")
        default_tight = config.get_runner_trail_pct(tight=True, strategy="")
        assert default_init == pytest.approx(0.03)
        assert default_tight == pytest.approx(0.015)

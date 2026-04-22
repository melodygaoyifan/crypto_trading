"""
Tests for Black Swan Sentinel, HOLD strategy integration, and confidence bucketing.
"""
import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# BLACK SWAN SENTINEL SCORING
# =============================================================================

class TestBlackSwanSentinel:
    """Test BSS score computation logic (replicated from pipeline for unit testing)."""

    @staticmethod
    def _compute_bss(fear_greed: int, vol_spike: float, mtf_momentum: float = 0.0):
        """Replicate BSS logic from market_data_pipeline.py for isolated testing."""
        bss = 3  # default calm
        is_extreme_fear = fear_greed < 20
        is_fear = fear_greed < 35
        if is_extreme_fear and vol_spike > 2.0:
            bss = 0
        elif is_extreme_fear or vol_spike > 2.5:
            bss = 1
        elif is_fear and abs(mtf_momentum) > 0.10:
            bss = 1
        elif not is_fear and vol_spike < 1.5:
            bss = 3
        contrarian = False
        if is_extreme_fear and vol_spike < 1.3:
            bss = 4
            contrarian = True
        elif is_fear and vol_spike < 1.5:
            contrarian = True
        return bss, contrarian

    def test_extreme_fear_plus_vol_spike_is_crisis(self):
        """F&G < 20 AND vol_spike > 2.0 → BSS = 0 (black swan)."""
        bss, _ = self._compute_bss(fear_greed=10, vol_spike=2.5)
        assert bss == 0

    def test_extreme_fear_no_vol_spike_is_contrarian(self):
        """F&G < 20 AND vol_spike < 1.3 → BSS = 4 (contrarian opportunity)."""
        bss, contrarian = self._compute_bss(fear_greed=15, vol_spike=1.0)
        assert bss == 4
        assert contrarian is True

    def test_moderate_fear_low_vol_is_contrarian(self):
        """F&G 20-35 AND vol_spike < 1.5 → contrarian = True."""
        bss, contrarian = self._compute_bss(fear_greed=30, vol_spike=1.2)
        assert contrarian is True

    def test_neutral_calm_is_normal(self):
        """F&G = 50, vol_spike = 1.0 → BSS = 3 (calm)."""
        bss, _ = self._compute_bss(fear_greed=50, vol_spike=1.0)
        assert bss == 3

    def test_extreme_vol_without_fear_is_warning(self):
        """F&G = 50 but vol_spike > 2.5 → BSS = 1."""
        bss, _ = self._compute_bss(fear_greed=50, vol_spike=3.0)
        assert bss == 1

    def test_fear_with_momentum_is_warning(self):
        """F&G 20-35 with strong momentum → BSS = 1."""
        bss, _ = self._compute_bss(fear_greed=30, vol_spike=1.2, mtf_momentum=0.15)
        assert bss == 1


# =============================================================================
# CONFIDENCE BUCKETING
# =============================================================================

class TestConfidenceBucketing:
    """Test N1: symmetric confidence clamping by signal direction.

    [UPDATED 2026-04-22] Was asymmetric (LONG [0.30,1.00], SHORT [0.10,0.60]).
    14d paper run showed the asymmetry forced 88% LONG bias and handicapped
    SHORT confidence; symmetrized to [0.30, 1.00] for both directions.
    """

    @staticmethod
    def _bucket(quant_dir: float, quant_conf: float) -> float:
        """Replicate bucketing logic from pipeline (post-symmetrization)."""
        if abs(quant_dir) > 0.05:  # directional (LONG or SHORT)
            return max(0.30, min(1.00, quant_conf))
        else:                      # neutral
            return max(0.40, min(0.70, quant_conf))

    def test_bullish_floor_is_030(self):
        """Bullish signal can't have confidence < 0.30."""
        assert self._bucket(0.5, 0.10) == 0.30

    def test_bullish_ceiling_is_100(self):
        """Bullish signal can reach confidence = 1.00."""
        assert self._bucket(0.5, 1.50) == 1.00

    def test_bearish_ceiling_is_100(self):
        """Bearish signal can also reach confidence = 1.00 (symmetric)."""
        assert self._bucket(-0.5, 0.90) == 0.90  # no clamp at 0.60 anymore
        assert self._bucket(-0.5, 1.50) == 1.00

    def test_bearish_floor_is_030(self):
        """Bearish signal floored at 0.30 (symmetric with bullish)."""
        assert self._bucket(-0.5, 0.05) == 0.30

    def test_neutral_clamped_to_040_070(self):
        """Neutral signal clamped to [0.40, 0.70]."""
        assert self._bucket(0.0, 0.10) == 0.40
        assert self._bucket(0.0, 0.90) == 0.70

    def test_slight_bullish_uses_directional_range(self):
        """Direction = 0.06 (just above 0.05 threshold) → directional bucketing."""
        assert self._bucket(0.06, 0.20) == 0.30  # clamped to floor

    def test_slight_bearish_uses_directional_range(self):
        """Direction = -0.06 → directional bucketing, no asymmetric cap."""
        assert self._bucket(-0.06, 0.80) == 0.80  # no longer clamped to 0.60


# =============================================================================
# SORTINO FIX (P3-10)
# =============================================================================

class TestSortinoFix:
    def test_correct_denominator_formula(self):
        """Sortino denominator should be sqrt(mean(min(r,0)^2)), NOT std(negatives)."""
        import numpy as np
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from core.asset_alpha_tilt import AssetAlphaTilt

        tilt = AssetAlphaTilt()
        pnls = [10.0, -5.0, 15.0, -3.0, 8.0, -7.0, 20.0, -2.0]

        # Correct: sqrt(mean(min(r,0)^2)) over ALL returns
        arr = np.array(pnls)
        downside_diff = np.minimum(arr, 0.0)
        expected_dev = np.sqrt(np.mean(downside_diff ** 2))
        expected_sortino = np.mean(arr) / expected_dev

        actual_sortino = tilt._compute_sortino(pnls)
        assert actual_sortino == pytest.approx(expected_sortino, rel=0.01)

    def test_all_positive_returns_high_sortino(self):
        """All positive PnL → very high Sortino (downside_dev ≈ 0)."""
        from core.asset_alpha_tilt import AssetAlphaTilt
        tilt = AssetAlphaTilt()
        result = tilt._compute_sortino([10.0, 20.0, 15.0, 25.0])
        assert result == 3.0  # returns cap value when no downside

    def test_all_negative_returns_low_sortino(self):
        """All negative PnL → Sortino < 0."""
        from core.asset_alpha_tilt import AssetAlphaTilt
        tilt = AssetAlphaTilt()
        result = tilt._compute_sortino([-10.0, -20.0, -15.0, -5.0])
        assert result < 0

    def test_mixed_returns_reasonable(self):
        """Mixed PnL → Sortino between -3 and +3."""
        from core.asset_alpha_tilt import AssetAlphaTilt
        tilt = AssetAlphaTilt()
        result = tilt._compute_sortino([5, -3, 8, -2, 10, -6, 3, -1])
        assert -3.0 < result < 3.0


# =============================================================================
# CANONICAL ENUMS (str,Enum)
# =============================================================================

class TestStrEnumPattern:
    def test_system_mode_string_comparison(self):
        """SystemMode.NORMAL == 'NORMAL' should be True (str,Enum)."""
        from core.canonical_enums import SystemMode
        assert SystemMode.NORMAL == "NORMAL"
        assert SystemMode.OPPORTUNITY == "OPPORTUNITY"
        assert SystemMode.NO_TRADE == "NO_TRADE"

    def test_execution_mode_string_comparison(self):
        from core.canonical_enums import ExecutionMode
        assert ExecutionMode.AGGRESSIVE == "AGGRESSIVE"
        assert ExecutionMode.PASSIVE_PREFERRED == "PASSIVE_PREFERRED"

    def test_deadlock_resolution_string_comparison(self):
        from core.canonical_enums import DeadlockResolution
        assert DeadlockResolution.NONE == "NONE"

    def test_authority_string_comparison(self):
        from core.canonical_enums import Authority
        assert Authority.DECIDE == "DECIDE"
        assert Authority.VETO == "VETO"

    def test_enum_in_dict_key(self):
        """str,Enum should work as dict key matching string."""
        from core.canonical_enums import SystemMode
        d = {"NORMAL": 1, "OPPORTUNITY": 2}
        assert d[SystemMode.NORMAL] == 1
        assert d[SystemMode.OPPORTUNITY] == 2

    def test_enum_json_serializable(self):
        """str,Enum should be JSON serializable without .value."""
        import json
        from core.canonical_enums import SystemMode
        data = {"mode": SystemMode.NORMAL}
        serialized = json.dumps(data)
        assert '"NORMAL"' in serialized

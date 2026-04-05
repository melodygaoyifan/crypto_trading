"""
HMATS v3.2 - SOL Exposure Manager
==================================
Purpose: SOL-specific exposure rules with aggressive all-in capability.

Philosophy:
- NORMAL mode: SOL ~50% baseline, BTC/ETH anchors
- OPPORTUNITY + CRACK + high beta + flow alignment:
  -> SOL can go up to ~1.0 (near all-in)
  -> Auto-cap BTC/ETH to prevent correlation collapse

SOL is the HIGH-RISK, HIGH-REWARD asset.
User wants profit maximization, so we allow aggressive SOL.

SOTA alignment:
- Mode hysteresis: fast de-risk, slow upgrade (defensive shell)
- NO_TRADE: fail-closed, no exposure increase beyond current state
- Correlation collapse: configurable threshold/floor
- SOL_DOMINANT: priority-based gross scaling preserves SOL
- Structured decision metadata for audit/proof logs
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Tuple, Optional, Any, List
from datetime import datetime, timezone
import logging
import math
import time as _time

from market.regime_detector import EnhancedRegimeState, MarketRegimeState, OpportunityMode

logger = logging.getLogger(__name__)


class ExposureMode(Enum):
    """Exposure mode for portfolio."""
    CONSERVATIVE = auto()  # Reduced exposure all around
    NORMAL = auto()        # Standard allocation
    AGGRESSIVE = auto()    # Allow aggressive positions
    SOL_DOMINANT = auto()  # SOL can go all-in


# Risk ordering for hysteresis: lower number = less risk
_MODE_RISK_ORDER = {
    ExposureMode.CONSERVATIVE: 0,
    ExposureMode.NORMAL: 1,
    ExposureMode.AGGRESSIVE: 2,
    ExposureMode.SOL_DOMINANT: 3,
}


@dataclass
class SOLExposureConfig:
    """Configuration for SOLExposureManager.

    All defaults preserve safe, defensive behavior.  Set flags to
    ``False`` to revert to pre-overhaul behavior where noted.
    """

    enable_mode_hysteresis: bool = True
    min_mode_hold_seconds: float = 300.0
    sol_dominant_confirmations_required: int = 1
    force_no_trade_no_increase: bool = True
    use_baseline_when_no_intent: bool = False
    apply_sol_scaling_factor: bool = False
    correlation_collapse_threshold: float = 0.85
    correlation_collapse_floor: float = 0.60
    prefer_sol_in_gross_scaling: bool = True  # only affects SOL_DOMINANT mode


@dataclass
class AssetExposureLimits:
    """Exposure limits per asset."""
    min_exposure: float = -1.0
    max_exposure: float = 1.0
    baseline: float = 0.0

    def clamp(self, exposure: float) -> float:
        return max(self.min_exposure, min(self.max_exposure, exposure))


@dataclass
class PortfolioExposureState:
    """Current portfolio exposure state."""
    btc_exposure: float = 0.0
    eth_exposure: float = 0.0
    sol_exposure: float = 0.0

    total_long: float = 0.0
    total_short: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0

    exposure_mode: ExposureMode = ExposureMode.NORMAL

    def update_totals(self):
        """Update derived totals."""
        self.total_long = max(0, self.btc_exposure) + max(0, self.eth_exposure) + max(0, self.sol_exposure)
        self.total_short = abs(min(0, self.btc_exposure)) + abs(min(0, self.eth_exposure)) + abs(min(0, self.sol_exposure))
        self.net_exposure = self.btc_exposure + self.eth_exposure + self.sol_exposure
        self.gross_exposure = abs(self.btc_exposure) + abs(self.eth_exposure) + abs(self.sol_exposure)

    def to_dict(self) -> Dict:
        self.update_totals()
        return {
            "btc_exposure": self.btc_exposure,
            "eth_exposure": self.eth_exposure,
            "sol_exposure": self.sol_exposure,
            "total_long": self.total_long,
            "total_short": self.total_short,
            "net_exposure": self.net_exposure,
            "gross_exposure": self.gross_exposure,
            "exposure_mode": self.exposure_mode.name
        }


class SOLExposureManager:
    """
    Manages SOL-specific exposure with aggressive capability.

    Key Features:
    - SOL baseline ~50% in NORMAL mode
    - SOL can reach ~100% in OPPORTUNITY+CRACK with alignment
    - Auto-caps BTC/ETH when SOL is dominant
    - Prevents correlation collapse
    - Mode hysteresis: fast de-risk, slow upgrade (defensive shell)
    - NO_TRADE: fail-closed, no exposure increase
    - Structured decision metadata for auditing
    """

    # Baseline allocations (NORMAL mode).
    # Applied only when ``config.use_baseline_when_no_intent`` is True
    # and all intents are zero.  In CONSERVATIVE/NO_TRADE the baseline
    # is forced to 0.0 (fail-closed).
    BASELINE_ALLOCATION = {
        "BTC": 0.25,
        "ETH": 0.25,
        "SOL": 0.50
    }

    # Maximum allocations per mode
    MAX_EXPOSURE_BY_MODE = {
        ExposureMode.CONSERVATIVE: {
            "BTC": 0.5, "ETH": 0.5, "SOL": 0.5,
            "total_gross": 1.0
        },
        ExposureMode.NORMAL: {
            "BTC": 0.8, "ETH": 0.8, "SOL": 0.8,
            "total_gross": 1.5
        },
        ExposureMode.AGGRESSIVE: {
            "BTC": 1.0, "ETH": 1.0, "SOL": 1.0,
            "total_gross": 2.0
        },
        ExposureMode.SOL_DOMINANT: {
            "BTC": 0.3, "ETH": 0.3, "SOL": 1.0,  # SOL can go all-in
            "total_gross": 1.3  # SOL dominant, BTC/ETH capped
        }
    }

    # Thresholds for SOL_DOMINANT mode
    SOL_DOMINANT_THRESHOLDS = {
        "beta_sol_btc_min": 1.3,      # SOL must be high beta
        "flow_alignment_min": 0.5,    # Flow must be aligned
        "regime_confidence_min": 0.6  # Regime must be confident
    }

    def __init__(self, config: Optional[SOLExposureConfig] = None):
        self.config = config or SOLExposureConfig()
        self.state = PortfolioExposureState()
        self.last_mode_change = datetime.now(timezone.utc)

        # Hysteresis tracking
        self._last_mode_change_mono: float = _time.monotonic()
        self._sol_dominant_consecutive: int = 0

        # Decision metadata (G)
        self._last_decision: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Input validation (H)
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_inputs(
        regime_state, flow_alignment: float,
    ) -> Tuple[bool, float, bool, List[str]]:
        """Validate and sanitise inputs.

        Returns ``(regime_valid, sanitised_flow, correlation_valid, issues)``.
        """
        issues: List[str] = []

        # --- regime_state ---
        regime_valid = True
        if regime_state is None:
            regime_valid = False
            issues.append("regime_state_is_none")
        else:
            for attr in (
                "is_no_trade", "is_opportunity_active", "beta_sol_btc",
                "volatility_regime", "correlation_btc_eth_sol",
                "crack_window", "confidence",
            ):
                if not hasattr(regime_state, attr):
                    regime_valid = False
                    issues.append(f"regime_missing_{attr}")
                    break

        # --- flow_alignment ---
        if flow_alignment is None or not isinstance(flow_alignment, (int, float)):
            flow_alignment = 0.0
            issues.append("flow_alignment_invalid")
        elif isinstance(flow_alignment, float) and math.isnan(flow_alignment):
            flow_alignment = 0.0
            issues.append("flow_alignment_nan")
        elif abs(flow_alignment) > 1.0:
            flow_alignment = max(-1.0, min(1.0, flow_alignment))
            issues.append("flow_alignment_clamped")

        # --- correlation ---
        correlation_valid = True
        if regime_valid:
            try:
                corr = regime_state.correlation_btc_eth_sol
                if corr is None:
                    correlation_valid = False
                    issues.append("correlation_is_none")
                elif isinstance(corr, float) and math.isnan(corr):
                    correlation_valid = False
                    issues.append("correlation_is_nan")
                elif not (-1.0 <= corr <= 1.0):
                    correlation_valid = False
                    issues.append("correlation_out_of_bounds")
            except Exception:
                correlation_valid = False
                issues.append("correlation_access_error")
        else:
            correlation_valid = False

        return regime_valid, flow_alignment, correlation_valid, issues

    @staticmethod
    def _safe_exposure(intent) -> float:
        """Extract ``target_exposure`` from intent, defaulting to 0.0."""
        if intent is None:
            return 0.0
        try:
            val = intent.target_exposure
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return 0.0
            return float(val)
        except (AttributeError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def _cap_no_trade(target: float, current: float) -> float:
        """In NO_TRADE: do not increase absolute exposure."""
        if abs(target) <= abs(current):
            return target
        return current

    # ------------------------------------------------------------------
    # Mode hysteresis (B)
    # ------------------------------------------------------------------
    def _apply_hysteresis(self, desired_mode: ExposureMode) -> ExposureMode:
        """Apply mode hysteresis: fast de-risk, slow upgrade.

        De-risk transitions happen immediately (defensive shell).
        Upgrades require ``min_mode_hold_seconds`` elapsed since last
        change, plus confirmation count for SOL_DOMINANT.
        """
        if not self.config.enable_mode_hysteresis:
            if desired_mode != self.state.exposure_mode:
                self._last_mode_change_mono = _time.monotonic()
                self.last_mode_change = datetime.now(timezone.utc)
            return desired_mode

        current = self.state.exposure_mode
        desired_risk = _MODE_RISK_ORDER[desired_mode]
        current_risk = _MODE_RISK_ORDER[current]

        # Track SOL_DOMINANT confirmations
        if desired_mode == ExposureMode.SOL_DOMINANT:
            self._sol_dominant_consecutive += 1
        else:
            self._sol_dominant_consecutive = 0

        # De-risk: immediate (defensive shell)
        if desired_risk < current_risk:
            self._last_mode_change_mono = _time.monotonic()
            self.last_mode_change = datetime.now(timezone.utc)
            self._sol_dominant_consecutive = 0
            return desired_mode

        # Same risk level: no change
        if desired_risk == current_risk:
            return current

        # Upgrade: requires hold time
        elapsed = _time.monotonic() - self._last_mode_change_mono
        if elapsed < self.config.min_mode_hold_seconds:
            return current

        # SOL_DOMINANT additionally requires consecutive confirmations
        if desired_mode == ExposureMode.SOL_DOMINANT:
            if self._sol_dominant_consecutive < self.config.sol_dominant_confirmations_required:
                return current

        # Upgrade approved
        self._last_mode_change_mono = _time.monotonic()
        self.last_mode_change = datetime.now(timezone.utc)
        return desired_mode

    # ------------------------------------------------------------------
    # Mode determination
    # ------------------------------------------------------------------
    def determine_exposure_mode(
        self,
        regime_state: EnhancedRegimeState,
        flow_alignment: float = 0.0
    ) -> ExposureMode:
        """
        Determine exposure mode based on regime and conditions.

        Returns the appropriate ExposureMode.
        """

        # NO_TRADE or defensive -> CONSERVATIVE
        if regime_state.is_no_trade():
            return ExposureMode.CONSERVATIVE

        # Check for SOL_DOMINANT conditions
        if self._should_sol_dominant(regime_state, flow_alignment):
            return ExposureMode.SOL_DOMINANT

        # CRACK window or OPPORTUNITY -> AGGRESSIVE
        if regime_state.crack_window.is_valid() or regime_state.is_opportunity_active():
            return ExposureMode.AGGRESSIVE

        # High volatility -> CONSERVATIVE
        if regime_state.volatility_regime == "extreme":
            return ExposureMode.CONSERVATIVE

        # Default -> NORMAL
        return ExposureMode.NORMAL

    def _should_sol_dominant(
        self,
        regime_state: EnhancedRegimeState,
        flow_alignment: float
    ) -> bool:
        """
        Check if SOL should be allowed to dominate.

        Requires:
        - OPPORTUNITY or CRACK active
        - High SOL beta
        - Aligned flow
        - Good regime confidence
        - CRACK direction matches SOL opportunity
        """

        if not regime_state.is_opportunity_active():
            return False

        # Must have high beta
        if regime_state.beta_sol_btc < self.SOL_DOMINANT_THRESHOLDS["beta_sol_btc_min"]:
            return False

        # Flow must be aligned
        if abs(flow_alignment) < self.SOL_DOMINANT_THRESHOLDS["flow_alignment_min"]:
            return False

        # Confidence check
        if regime_state.confidence.overall() < self.SOL_DOMINANT_THRESHOLDS["regime_confidence_min"]:
            return False

        # CRACK window check
        if regime_state.crack_window.is_valid():
            crack_dir = regime_state.crack_window.direction
            # Flow should align with CRACK direction
            if crack_dir == "long" and flow_alignment < 0:
                return False
            if crack_dir == "short" and flow_alignment > 0:
                return False

        logger.debug(
            "SOL_DOMINANT conditions met: beta=%.2f, flow=%.2f",
            regime_state.beta_sol_btc, flow_alignment,
        )
        return True

    # ------------------------------------------------------------------
    # Core exposure calculation
    # ------------------------------------------------------------------
    def calculate_target_exposures(
        self,
        btc_intent: Optional[Any],
        eth_intent: Optional[Any],
        sol_intent: Optional[Any],
        regime_state: EnhancedRegimeState,
        flow_alignment: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate final target exposures respecting mode limits.

        Returns dict of asset -> target_exposure.
        """

        # 1. Validate inputs (H)
        regime_valid, flow_alignment, corr_valid, issues = self._validate_inputs(
            regime_state, flow_alignment,
        )

        # 2. Determine desired mode
        if regime_valid:
            try:
                desired_mode = self.determine_exposure_mode(regime_state, flow_alignment)
            except Exception:
                desired_mode = ExposureMode.NORMAL
                issues.append("mode_determination_error")
        else:
            desired_mode = ExposureMode.NORMAL
            issues.append("regime_invalid_fallback_normal")

        # 3. Apply hysteresis (B)
        mode = self._apply_hysteresis(desired_mode)
        if desired_mode != mode:
            issues.append("hysteresis_held")
        self.state.exposure_mode = mode

        # 4. Get limits for this mode
        limits = self.MAX_EXPOSURE_BY_MODE[mode]

        # 5. Extract raw intents (fail-closed)
        btc_raw = self._safe_exposure(btc_intent)
        eth_raw = self._safe_exposure(eth_intent)
        sol_raw = self._safe_exposure(sol_intent)

        # 5b. Baseline allocation (E)
        if self.config.use_baseline_when_no_intent:
            if btc_raw == 0.0 and eth_raw == 0.0 and sol_raw == 0.0:
                # Baseline only in NORMAL or above, never in CONSERVATIVE
                if mode not in (ExposureMode.CONSERVATIVE,):
                    btc_raw = self.BASELINE_ALLOCATION["BTC"]
                    eth_raw = self.BASELINE_ALLOCATION["ETH"]
                    sol_raw = self.BASELINE_ALLOCATION["SOL"]
                    issues.append("baseline_applied")

        # 6. SOL scaling factor (F)
        sol_scaling = 1.0
        if self.config.apply_sol_scaling_factor and regime_valid:
            try:
                sol_scaling = self.get_sol_scaling_factor(regime_state, flow_alignment)
                sol_raw *= sol_scaling
            except Exception:
                issues.append("sol_scaling_error")

        # 7. Clamp per-asset limits
        btc_target = max(-limits["BTC"], min(limits["BTC"], btc_raw))
        eth_target = max(-limits["ETH"], min(limits["ETH"], eth_raw))
        sol_target = max(-limits["SOL"], min(limits["SOL"], sol_raw))

        # 8. NO_TRADE cap (C)
        no_trade_capped = False
        is_no_trade = False
        if regime_valid:
            try:
                is_no_trade = regime_state.is_no_trade()
            except Exception:
                pass

        if self.config.force_no_trade_no_increase and is_no_trade:
            btc_before, eth_before, sol_before = btc_target, eth_target, sol_target
            btc_target = self._cap_no_trade(btc_target, self.state.btc_exposure)
            eth_target = self._cap_no_trade(eth_target, self.state.eth_exposure)
            sol_target = self._cap_no_trade(sol_target, self.state.sol_exposure)
            if (btc_target, eth_target, sol_target) != (btc_before, eth_before, sol_before):
                no_trade_capped = True
                issues.append("no_trade_no_increase")

        # 9. Gross scaling (D)
        gross_before = abs(btc_target) + abs(eth_target) + abs(sol_target)
        max_gross = limits["total_gross"]

        if gross_before > max_gross:
            if mode == ExposureMode.SOL_DOMINANT and self.config.prefer_sol_in_gross_scaling:
                # SOL gets priority: keep SOL, scale BTC/ETH to fit
                remaining = max_gross - abs(sol_target)
                if remaining > 0:
                    btc_eth_gross = abs(btc_target) + abs(eth_target)
                    if btc_eth_gross > 0:
                        scale = min(1.0, remaining / btc_eth_gross)
                        btc_target *= scale
                        eth_target *= scale
                else:
                    btc_target = 0.0
                    eth_target = 0.0
                    sol_target = max(-max_gross, min(max_gross, sol_target))
                issues.append("gross_scaled_sol_priority")
            else:
                scale = max_gross / gross_before
                btc_target *= scale
                eth_target *= scale
                sol_target *= scale
                issues.append("gross_scaled_proportional")

        # 10. SOL_DOMINANT special handling (existing)
        if mode == ExposureMode.SOL_DOMINANT and abs(sol_target) > 0.7:
            btc_cap = 0.2
            eth_cap = 0.2
            btc_target = max(-btc_cap, min(btc_cap, btc_target))
            eth_target = max(-eth_cap, min(eth_cap, eth_target))
            issues.append("sol_dominant_caps")

        # 11. Correlation collapse prevention
        corr_reduction = 1.0
        if corr_valid and regime_valid:
            targets, corr_reduction = self._prevent_correlation_collapse(
                btc_target, eth_target, sol_target, regime_state,
            )
            if corr_reduction < 1.0:
                issues.append("correlation_collapse_reduction")
        else:
            targets = {"BTC": btc_target, "ETH": eth_target, "SOL": sol_target}
            if not corr_valid:
                issues.append("correlation_invalid_skip_check")

        # 12. Update state
        self.state.btc_exposure = targets["BTC"]
        self.state.eth_exposure = targets["ETH"]
        self.state.sol_exposure = targets["SOL"]
        self.state.update_totals()

        # 13. Decision metadata (G)
        self._last_decision = {
            "mode_requested": desired_mode.name,
            "mode_applied": mode.name,
            "hysteresis_applied": desired_mode != mode,
            "no_trade_capped": no_trade_capped,
            "reasons": list(issues),
            "inputs": {
                "correlation": (
                    getattr(regime_state, "correlation_btc_eth_sol", None)
                    if regime_valid else None
                ),
                "beta_sol_btc": (
                    getattr(regime_state, "beta_sol_btc", None)
                    if regime_valid else None
                ),
                "flow_alignment": flow_alignment,
                "regime_valid": regime_valid,
                "correlation_valid": corr_valid,
            },
            "scaling": {
                "gross_before": gross_before,
                "gross_after": self.state.gross_exposure,
                "correlation_reduction_factor": corr_reduction,
                "sol_scaling_factor": sol_scaling,
            },
            "final_targets": dict(targets),
        }

        # 14. Logging (INFO only on mode changes / major reductions)
        if desired_mode != mode:
            logger.info(
                "[SOLExposure] Hysteresis: requested %s, applied %s",
                desired_mode.name, mode.name,
            )
        if no_trade_capped:
            logger.info("[SOLExposure] NO_TRADE: capped exposure increase")
        if corr_reduction < 1.0:
            logger.warning(
                "[SOLExposure] Correlation collapse reduction: %.2f", corr_reduction,
            )

        logger.debug(
            "[SOLExposure] targets(%s): BTC=%.3f ETH=%.3f SOL=%.3f | reasons=%s",
            mode.name, targets["BTC"], targets["ETH"], targets["SOL"], issues,
        )

        return targets

    def _prevent_correlation_collapse(
        self,
        btc: float,
        eth: float,
        sol: float,
        regime_state: EnhancedRegimeState
    ) -> Tuple[Dict[str, float], float]:
        """
        Prevent correlation collapse (all assets same direction at max).

        Returns ``(targets_dict, reduction_factor)``.
        """
        threshold = self.config.correlation_collapse_threshold
        floor = self.config.correlation_collapse_floor

        # Check if all same direction
        all_long = btc > 0.3 and eth > 0.3 and sol > 0.3
        all_short = btc < -0.3 and eth < -0.3 and sol < -0.3

        # High correlation + same direction = risk
        corr = regime_state.correlation_btc_eth_sol
        if corr > threshold and (all_long or all_short):
            # Reduce exposure by correlation factor
            reduction = 1 - ((corr - threshold) * 2)
            reduction = max(floor, reduction)

            btc *= reduction
            eth *= reduction
            sol *= reduction

            return {"BTC": btc, "ETH": eth, "SOL": sol}, reduction

        return {"BTC": btc, "ETH": eth, "SOL": sol}, 1.0

    def get_sol_scaling_factor(
        self,
        regime_state: EnhancedRegimeState,
        flow_alignment: float
    ) -> float:
        """
        Get SOL-specific scaling factor.

        Advisory helper for upstream components.  Only applied internally
        when ``config.apply_sol_scaling_factor`` is True.

        Returns multiplier (0.5 to 1.5).
        """

        base = 1.0

        # High beta bonus
        if regime_state.beta_sol_btc > 1.5:
            base += 0.2
        elif regime_state.beta_sol_btc > 1.2:
            base += 0.1

        # Flow alignment bonus
        if abs(flow_alignment) > 0.7:
            base += 0.15

        # CRACK window bonus
        if regime_state.crack_window.is_valid():
            base += 0.15 * regime_state.crack_window.remaining_strength

        # Volatility penalty
        if regime_state.volatility_regime == "extreme":
            base *= 0.7
        elif regime_state.volatility_regime == "high":
            base *= 0.85

        # Clamp to 0.5 - 1.5
        return max(0.5, min(1.5, base))

    def get_state(self) -> PortfolioExposureState:
        """Get current portfolio exposure state."""
        return self.state

    def get_last_decision(self) -> Dict[str, Any]:
        """Return structured decision metadata from the last calculation.

        Keys: mode_requested, mode_applied, hysteresis_applied,
        no_trade_capped, reasons, inputs, scaling, final_targets.
        """
        return dict(self._last_decision)


# Singleton instance
_sol_exposure_manager: Optional[SOLExposureManager] = None


def get_sol_exposure_manager(
    config: Optional[SOLExposureConfig] = None,
) -> SOLExposureManager:
    """Get singleton SOLExposureManager instance."""
    global _sol_exposure_manager
    if _sol_exposure_manager is None:
        _sol_exposure_manager = SOLExposureManager(config)
    elif config is not None and _sol_exposure_manager.config != config:
        logger.info("[SOL_EXPOSURE] Config changed; refreshing singleton instance")
        _sol_exposure_manager = SOLExposureManager(config)
    return _sol_exposure_manager


def reset_sol_exposure_manager():
    """Reset singleton SOLExposureManager (for testing)."""
    global _sol_exposure_manager
    _sol_exposure_manager = None

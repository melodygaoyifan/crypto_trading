"""
================================================================================
UNIFIED POSITION SIZER - Single Source of Truth for Position Sizing
================================================================================
Version: 3.3.0 (v6.2.1: High-Risk Gambler Mode support)
Purpose: Resolves HIGH impact item #1 from CTO Audit

PROBLEM SOLVED:
- Risk-based sizing (1% per trade) and Tranche percentages (20/30/30/20)
  were operating independently, potentially producing conflicting sizes.

SOLUTION:
- This module is the ONLY authority for final position size
- Risk-based sizing sets the MAXIMUM position for the ENTIRE trade
- Tranche percentages determine HOW MUCH of that maximum to deploy at each stage

MATH:
    max_position_value = account_balance × risk_per_trade / stop_loss_pct
    tranche_1_size = max_position_value × 0.20
    tranche_2_size = max_position_value × 0.30 (cumulative 0.50)
    tranche_3_size = max_position_value × 0.30 (cumulative 0.80)
    tranche_4_size = max_position_value × 0.20 (cumulative 1.00)

CRITICAL RULES:
1. Final size NEVER exceeds risk-based maximum
2. Final size NEVER exceeds global gross exposure cap
3. Final size NEVER exceeds asset-specific cap (SOL dominance rules)
4. Drawdown status can reduce all sizes

v6.2.1 HIGH-RISK GAMBLER MODE:
- In OPPORTUNITY mode: Higher single position cap (95% vs 80%)
- In OPPORTUNITY mode: Higher gross exposure cap (200% vs 150%)
- All changes gated by ENABLE_HIGH_RISK_GAMBLER_MODE flag
- Still respects: Governors, RiskController, Existence Fuse
================================================================================
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


from core.canonical_enums import TrancheLevel  # Single source of truth


@dataclass
class PositionSizingConfig:
    """Configuration for unified position sizing."""
    
    # Risk-based sizing
    risk_per_trade: float = 0.01  # 1% account risk per full position
    
    # Tranche percentages (must sum to 1.0)
    # [UNLEASH] UL-8d: Heavier T1 for faster initial entry
    # Original: 20/30/30/20. New: 35/30/25/10
    tranche_percentages: Dict[TrancheLevel, float] = field(default_factory=lambda: {
        TrancheLevel.TRANCHE_1: 0.35,  # [UNLEASH] was 0.20 -> 0.35
        TrancheLevel.TRANCHE_2: 0.30,  # Individual, not cumulative
        TrancheLevel.TRANCHE_3: 0.25,  # [UNLEASH] was 0.30 -> 0.25
        TrancheLevel.TRANCHE_4: 0.10,  # [UNLEASH] was 0.20 -> 0.10
    })
    
    # Global caps (NORMAL mode)
    max_single_position_pct: float = 0.80  # Max 80% in any single asset (NORMAL)
    max_gross_exposure: float = 1.50       # Max 150% gross exposure (NORMAL)
    min_position_value_usd: float = 50.0   # Minimum position size
    
    # Drawdown adjustments
    drawdown_size_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "NORMAL": 1.0,
        "REDUCED": 0.50,
        "HALTED": 0.0,
        "CRITICAL": 0.0,
    })
    
    # Asset-specific caps (default, can be overridden by SOL dominance)
    default_asset_caps: Dict[str, float] = field(default_factory=lambda: {
        "BTC": 0.80,
        "ETH": 0.80,
        "SOL": 0.80,
    })


@dataclass
class PositionSizingResult:
    """Result from unified position sizer."""
    
    # Core output
    final_size_usd: float
    final_exposure_pct: float  # As percentage of account
    
    # Breakdown
    risk_based_max_usd: float
    tranche_requested_usd: float
    cap_applied: str  # Which cap was binding: "none", "risk", "gross", "asset", "drawdown"
    
    # Metadata
    asset: str
    tranche_level: TrancheLevel
    drawdown_status: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Validation
    is_valid: bool = True
    rejection_reason: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "final_size_usd": self.final_size_usd,
            "final_exposure_pct": self.final_exposure_pct,
            "risk_based_max_usd": self.risk_based_max_usd,
            "tranche_requested_usd": self.tranche_requested_usd,
            "cap_applied": self.cap_applied,
            "asset": self.asset,
            "tranche_level": self.tranche_level.name,
            "drawdown_status": self.drawdown_status,
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason,
        }


class KellyPositionScaler:
    """R3: Fractional Kelly criterion for crypto position sizing.

    Tracks trade results and computes optimal bet size as a fraction
    of account equity. Requires MIN_TRADES_FOR_KELLY warm-up trades
    before producing a signal. Returns None during warm-up.
    """

    MIN_TRADES_FOR_KELLY = 20  # Warm-up requirement

    def __init__(self, kelly_fraction: float = 0.30):
        self.kelly_fraction = kelly_fraction
        self._trade_results: deque = deque(maxlen=100)

    def record_trade(self, pnl_pct: float):
        """Record trade result as percentage return (e.g. 0.02 = +2%)."""
        self._trade_results.append(pnl_pct)

    def compute_kelly_fraction(self) -> Optional[float]:
        """Compute fractional Kelly bet size as fraction of account.

        Returns None if insufficient trades for reliable estimate.
        """
        if len(self._trade_results) < self.MIN_TRADES_FOR_KELLY:
            return None  # Not enough data

        wins = [r for r in self._trade_results if r > 0]
        losses = [abs(r) for r in self._trade_results if r < 0]

        if not wins or not losses:
            return None

        win_rate = len(wins) / len(self._trade_results)
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        if win_loss_ratio <= 0:
            return 0.0

        # Kelly formula: f* = (p * b - q) / b
        p = win_rate
        q = 1 - p
        b = win_loss_ratio
        full_kelly = (p * b - q) / b

        # Apply fraction and clamp
        fractional = max(0.0, full_kelly * self.kelly_fraction)
        fractional = min(fractional, 0.50)  # Never exceed 50% of account

        logger.debug(
            f"[KELLY] wr={p:.2f} wlr={b:.2f} full={full_kelly:.3f} "
            f"frac={fractional:.3f}"
        )
        return fractional


class UnifiedPositionSizer:
    """
    Single source of truth for all position sizing decisions.
    
    This class resolves the conflict between:
    - Risk-based sizing (defines MAXIMUM total position)
    - Tranche percentages (defines HOW MUCH of maximum to deploy)
    
    v6.2.1: High-Risk Gambler Mode support
    - In OPPORTUNITY mode with gambler flag: Higher position caps
    - Still respects: Governors, RiskController, hard limits
    
    USAGE:
        sizer = UnifiedPositionSizer(config)
        result = sizer.calculate_position_size(
            asset="SOL",
            account_balance=100000,
            stop_loss_pct=0.02,
            tranche_level=TrancheLevel.TRANCHE_1,
            current_gross_exposure=0.0,
            current_asset_exposure=0.0,
            drawdown_status="NORMAL",
            asset_cap_override=None,  # SOL dominance can override
            is_opportunity=False      # v6.2.1: OPPORTUNITY mode flag
        )
    """
    
    # Safety cap: prevent equity growth from creating dangerously large positions
    MAX_SIZING_EQUITY_MULTIPLIER = 3.0

    def __init__(self, config: Optional[PositionSizingConfig] = None,
                 initial_capital: Optional[float] = None):
        self.config = config or PositionSizingConfig()
        self._initial_capital = initial_capital
        self._kelly_scaler: Optional[KellyPositionScaler] = None
        self._validate_config()
        logger.info("UnifiedPositionSizer initialized"
                    f" (initial_capital=${initial_capital:,.0f})"
                    if initial_capital else "")

    def set_kelly_scaler(self, scaler: KellyPositionScaler) -> None:
        """Set Kelly position scaler for optional fractional Kelly sizing."""
        self._kelly_scaler = scaler
        logger.info(f"[KELLY] Scaler attached (fraction={scaler.kelly_fraction})")
    
    def _validate_config(self):
        """Validate configuration consistency."""
        tranche_sum = sum(self.config.tranche_percentages.values())
        if abs(tranche_sum - 1.0) > 0.001:
            raise ValueError(f"Tranche percentages must sum to 1.0, got {tranche_sum}")
    
    def _is_gambler_mode(self) -> bool:
        """Check if high-risk gambler mode is active."""
        try:
            from configs.high_risk_mode import is_gambler_mode_active
            return is_gambler_mode_active()
        except ImportError:
            return False
    
    def _get_gambler_config(self):
        """Get gambler mode configuration."""
        try:
            from configs.high_risk_mode import get_high_risk_config
            return get_high_risk_config()
        except ImportError:
            return None
    
    def _log_gambler_action(self, details: Dict = None):
        """Log gambler mode position boost for audit."""
        try:
            from configs.high_risk_mode import log_gambler_action, GamblerModeReason
            log_gambler_action(GamblerModeReason.OPPORTUNITY_MAX_BOOST, details)
        except ImportError:
            pass
    
    def calculate_position_size(
        self,
        asset: str,
        account_balance: float,
        stop_loss_pct: float,
        tranche_level: TrancheLevel,
        current_gross_exposure: float,
        current_asset_exposure: float,
        drawdown_status: str = "NORMAL",
        asset_cap_override: Optional[float] = None,
        price_usd: Optional[float] = None,
        is_opportunity: bool = False,  # v6.2.1: OPPORTUNITY mode flag
        confidence: float = 0.5,      # [2026-04-08] Confidence-based sizing
        cross_asset_correlation: float = 0.5,  # [2026-04-11] Correlation-adjusted sizing
        annualized_vol: float = 0.6,           # [2026-04-11] Dynamic vol-adjusted limit
        beta_to_btc: float = 1.0,             # [2026-04-12] Beta-adjusted sizing
    ) -> PositionSizingResult:
        """
        Calculate final position size with all constraints applied.
        
        Args:
            asset: Asset symbol (BTC, ETH, SOL)
            account_balance: Total account value in USD
            stop_loss_pct: Stop loss distance as decimal (0.02 = 2%)
            tranche_level: Which tranche we're entering/adding
            current_gross_exposure: Current total gross exposure (0.0 to 1.5+)
            current_asset_exposure: Current exposure in this specific asset
            drawdown_status: NORMAL, REDUCED, HALTED, CRITICAL
            asset_cap_override: Override default asset cap (for SOL dominance)
            price_usd: Current price (for calculating quantity)
            is_opportunity: v6.2.1 - True if in OPPORTUNITY mode
        
        Returns:
            PositionSizingResult with final size and metadata
        """
        
        # Step 0a: Clamp equity for sizing (prevent runaway compounding)
        if self._initial_capital and self._initial_capital > 0:
            cap = self._initial_capital * self.MAX_SIZING_EQUITY_MULTIPLIER
            if account_balance > cap:
                logger.info(
                    f"[COMPOUND_CAP] Clamping sizing equity ${account_balance:,.0f} "
                    f"-> ${cap:,.0f} ({self.MAX_SIZING_EQUITY_MULTIPLIER}× initial)"
                )
                account_balance = cap

        # Step 0: Check if trading allowed
        if drawdown_status in ["HALTED", "CRITICAL"]:
            return PositionSizingResult(
                final_size_usd=0.0,
                final_exposure_pct=0.0,
                risk_based_max_usd=0.0,
                tranche_requested_usd=0.0,
                cap_applied="drawdown",
                asset=asset,
                tranche_level=tranche_level,
                drawdown_status=drawdown_status,
                is_valid=False,
                rejection_reason=f"Trading blocked due to {drawdown_status} drawdown status"
            )
        
        if tranche_level == TrancheLevel.NONE:
            return PositionSizingResult(
                final_size_usd=0.0,
                final_exposure_pct=0.0,
                risk_based_max_usd=0.0,
                tranche_requested_usd=0.0,
                cap_applied="none",
                asset=asset,
                tranche_level=tranche_level,
                drawdown_status=drawdown_status,
                is_valid=True,
                rejection_reason=""
            )
        
        # v6.2.1: Determine caps based on gambler mode
        gambler_mode = self._is_gambler_mode()
        gambler_config = self._get_gambler_config()
        
        if gambler_mode and gambler_config and is_opportunity:
            # GAMBLER MODE + OPPORTUNITY: Use boosted caps
            max_single_pct = gambler_config.get_max_single_position(True, True)
            max_gross = gambler_config.get_max_gross_exposure(True, True)
            
            self._log_gambler_action({
                "asset": asset,
                "tranche": tranche_level.name,
                "max_single_pct": max_single_pct,
                "max_gross": max_gross
            })
        else:
            # NORMAL mode: Use standard caps
            max_single_pct = self.config.max_single_position_pct
            max_gross = self.config.max_gross_exposure
        
        # Step 1: Calculate risk-based maximum
        # Formula: max_position = (account × risk_per_trade) / stop_loss_pct
        if stop_loss_pct <= 0:
            stop_loss_pct = 0.02  # Default 2% stop
        
        # [2026-04-08] Confidence-based risk scaling: low conviction = smaller bets,
        # high conviction = larger bets. Multiplier range [0.5, 1.5] applied to
        # base risk_per_trade. At conf=0.3 → 0.8×, conf=0.5 → 1.0×, conf=0.9 → 1.4×.
        _conf_mult = max(0.5, min(1.5, 0.5 + float(confidence)))
        _effective_risk = self.config.risk_per_trade * _conf_mult
        risk_amount = account_balance * _effective_risk
        risk_based_max_usd = risk_amount / stop_loss_pct

        # Step 1b: R3 - Apply Kelly cap if available
        if self._kelly_scaler:
            kelly_frac = self._kelly_scaler.compute_kelly_fraction()
            if kelly_frac is not None:
                kelly_max = account_balance * kelly_frac
                if kelly_max < risk_based_max_usd:
                    risk_based_max_usd = kelly_max
                    logger.info(
                        f"[KELLY] Capping {asset} to ${kelly_max:,.0f} "
                        f"(f={kelly_frac:.3f})"
                    )

        # Step 2: Calculate tranche-requested size
        cumulative_pct = self._get_cumulative_tranche_pct(tranche_level)
        previous_pct = self._get_cumulative_tranche_pct(
            TrancheLevel(tranche_level.value - 1) if tranche_level.value > 0 else TrancheLevel.NONE
        )
        tranche_pct = cumulative_pct - previous_pct
        tranche_requested_usd = risk_based_max_usd * tranche_pct
        
        # Step 3: Apply drawdown multiplier
        dd_multiplier = self.config.drawdown_size_multipliers.get(drawdown_status, 1.0)
        size_after_drawdown = tranche_requested_usd * dd_multiplier

        # Step 3b: [2026-04-11] Correlation-adjusted sizing (ai-hedge-fund risk_manager pattern)
        # BTC/ETH/SOL are typically 0.7-0.9 correlated. When correlation is high,
        # reduce per-asset size to limit concentrated risk.
        # corr >= 0.80 → ×0.70, corr >= 0.60 → ×0.85, corr >= 0.40 → ×1.00,
        # corr >= 0.20 → ×1.05, corr < 0.20 → ×1.10
        _corr = float(cross_asset_correlation)
        if _corr >= 0.80:
            _corr_mult = 0.70
        elif _corr >= 0.60:
            _corr_mult = 0.85
        elif _corr >= 0.40:
            _corr_mult = 1.00
        elif _corr >= 0.20:
            _corr_mult = 1.05
        else:
            _corr_mult = 1.10
        size_after_drawdown *= _corr_mult

        # Step 3c: [2026-04-12] Beta-adjusted sizing
        # Higher-beta assets (ETH beta=1.22, SOL beta=1.35 to BTC) should get
        # proportionally smaller positions to equalize risk contribution.
        # Formula: size *= 1/beta (clamped to [0.65, 1.10])
        # BTC (beta=1.0) → ×1.0, ETH (beta=1.2) → ×0.83, SOL (beta=1.35) → ×0.74
        _beta = max(0.5, min(2.5, float(beta_to_btc)))
        if _beta > 0.01:
            _beta_mult = max(0.65, min(1.10, 1.0 / _beta))
        else:
            _beta_mult = 1.0
        size_after_drawdown *= _beta_mult

        # Step 4: Apply global gross exposure cap (may be boosted in gambler mode)
        remaining_gross_room = (max_gross - current_gross_exposure) * account_balance
        size_after_gross_cap = min(size_after_drawdown, max(0, remaining_gross_room))
        
        # Step 4b: [P2-6] Dynamic vol-adjusted asset cap (linear interpolation)
        # Crypto-adjusted: BTC ~60-80%, ETH ~80-100%, SOL ~100-150% annualized vol.
        # Higher vol → lower effective cap. Smooth linear decay within tiers.
        _ann_vol = float(annualized_vol)
        if _ann_vol < 0.40:
            _vol_cap_mult = 1.20
        elif _ann_vol < 0.80:
            _vol_cap_mult = 1.00 - (_ann_vol - 0.40) * 0.75  # 1.00 → 0.70
        elif _ann_vol < 1.20:
            _vol_cap_mult = 0.70 - (_ann_vol - 0.80) * 0.50  # 0.70 → 0.50
        else:
            _vol_cap_mult = 0.50
        _vol_cap_mult = max(0.30, min(1.20, _vol_cap_mult))

        # Step 5: Apply asset-specific cap (now dynamic)
        _base_asset_cap = asset_cap_override or self.config.default_asset_caps.get(asset, 0.80)
        asset_cap = _base_asset_cap * _vol_cap_mult
        remaining_asset_room = (asset_cap - current_asset_exposure) * account_balance
        size_after_asset_cap = min(size_after_gross_cap, max(0, remaining_asset_room))
        
        # Step 6: Apply single position maximum (may be boosted in gambler mode)
        max_single = max_single_pct * account_balance
        size_after_single_cap = min(size_after_asset_cap, max_single)
        
        # Step 7: Apply minimum position size
        final_size_usd = size_after_single_cap
        if 0 < final_size_usd < self.config.min_position_value_usd:
            final_size_usd = 0.0  # Below minimum, don't trade
        
        # Determine which cap was binding
        cap_applied = "none"
        if final_size_usd < tranche_requested_usd * 0.999:
            if size_after_drawdown < tranche_requested_usd * 0.999:
                cap_applied = "drawdown"
            elif size_after_gross_cap < size_after_drawdown * 0.999:
                cap_applied = "gross"
            elif size_after_asset_cap < size_after_gross_cap * 0.999:
                cap_applied = "asset"
            elif size_after_single_cap < size_after_asset_cap * 0.999:
                cap_applied = "single"
            elif final_size_usd == 0 and size_after_single_cap > 0:
                cap_applied = "minimum"
        
        # Calculate exposure percentage
        final_exposure_pct = final_size_usd / account_balance if account_balance > 0 else 0.0
        
        result = PositionSizingResult(
            final_size_usd=final_size_usd,
            final_exposure_pct=final_exposure_pct,
            risk_based_max_usd=risk_based_max_usd,
            tranche_requested_usd=tranche_requested_usd,
            cap_applied=cap_applied,
            asset=asset,
            tranche_level=tranche_level,
            drawdown_status=drawdown_status,
            is_valid=final_size_usd > 0 or tranche_level == TrancheLevel.NONE,
            rejection_reason="" if final_size_usd > 0 else f"Position size zero after {cap_applied} cap"
        )
        
        logger.debug(
            f"Position sizing: {asset} T{tranche_level.value} -> "
            f"${final_size_usd:.2f} ({final_exposure_pct:.1%}), cap={cap_applied}"
            f"{' [GAMBLER]' if gambler_mode and is_opportunity else ''}"
        )
        
        return result
    
    def _get_cumulative_tranche_pct(self, level: TrancheLevel) -> float:
        """Get cumulative percentage up to and including this tranche."""
        if level == TrancheLevel.NONE:
            return 0.0
        
        cumulative = 0.0
        for lvl in TrancheLevel:
            if lvl == TrancheLevel.NONE:
                continue
            cumulative += self.config.tranche_percentages.get(lvl, 0.0)
            if lvl == level:
                break
        return cumulative
    
    def calculate_full_position_schedule(
        self,
        asset: str,
        account_balance: float,
        stop_loss_pct: float,
        current_gross_exposure: float = 0.0,
        current_asset_exposure: float = 0.0,
        drawdown_status: str = "NORMAL",
        asset_cap_override: Optional[float] = None,
    ) -> Dict[TrancheLevel, PositionSizingResult]:
        """
        Calculate position sizes for all tranche levels.
        
        Useful for planning full trade execution.
        """
        results = {}
        running_gross = current_gross_exposure
        running_asset = current_asset_exposure
        
        for level in [TrancheLevel.TRANCHE_1, TrancheLevel.TRANCHE_2, 
                      TrancheLevel.TRANCHE_3, TrancheLevel.TRANCHE_4]:
            result = self.calculate_position_size(
                asset=asset,
                account_balance=account_balance,
                stop_loss_pct=stop_loss_pct,
                tranche_level=level,
                current_gross_exposure=running_gross,
                current_asset_exposure=running_asset,
                drawdown_status=drawdown_status,
                asset_cap_override=asset_cap_override,
            )
            results[level] = result
            
            # Update running exposures for next tranche calculation
            running_gross += result.final_exposure_pct
            running_asset += result.final_exposure_pct
        
        return results
    
    def get_max_position_for_risk(
        self,
        account_balance: float,
        stop_loss_pct: float,
    ) -> float:
        """
        Get the maximum position size based purely on risk parameters.
        
        This is the theoretical maximum before any caps are applied.
        """
        if stop_loss_pct <= 0:
            return 0.0
        return (account_balance * self.config.risk_per_trade) / stop_loss_pct


# Singleton instance for global access
_unified_sizer: Optional[UnifiedPositionSizer] = None


def get_unified_position_sizer(config: Optional[PositionSizingConfig] = None,
                               initial_capital: Optional[float] = None) -> UnifiedPositionSizer:
    """Get or create the singleton UnifiedPositionSizer instance."""
    global _unified_sizer
    if _unified_sizer is None:
        _unified_sizer = UnifiedPositionSizer(config, initial_capital=initial_capital)
    elif (
        (config is not None and _unified_sizer.config != config)
        or _unified_sizer._initial_capital != initial_capital
    ):
        logger.info("[UNIFIED_SIZER] Constructor inputs changed; refreshing singleton instance")
        _unified_sizer = UnifiedPositionSizer(config, initial_capital=initial_capital)
    return _unified_sizer


def reset_unified_position_sizer():
    """Reset the singleton (for testing)."""
    global _unified_sizer
    _unified_sizer = None

"""
================================================================================
HMATS v6.2.1 - HIGH-RISK GAMBLER MODE CONFIGURATION
================================================================================

[WARN]️ WARNING: This module contains aggressive risk parameters.

PURPOSE:
    - Maximize odds and "big win probability"
    - Accept lower win rate for higher payoff ratio
    - Get in earlier, scale faster, fail faster

CONSTRAINTS (NEVER VIOLATED):
    OFF Does NOT modify: core signal definitions, indicator calculations, integration agents
    OFF Does NOT relax: final stop loss, liquidation stop, hard limits
    OFF Does NOT remove: any governor, RiskController, TradeGate
    OFF Does NOT introduce: new dependencies or infrastructure

WHAT IT MODIFIES:
    ON Entry timing thresholds (earlier T1 entry)
    ON Anti-Martingale escalation speed (faster T2/T3)
    ON OPPORTUNITY mode position limits (higher caps)
    ON Soft exit triggers (earlier recognition of failed trades)

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum

from config.sota_flags import get_sota_flags

logger = logging.getLogger(__name__)


class GamblerModeReason(Enum):
    """Reasons for gambler mode decisions (for ShadowLedger audit)."""
    EARLY_ENTRY_T1 = "GAMBLER_EARLY_ENTRY_T1"
    FAST_ESCALATE_T2 = "GAMBLER_FAST_ESCALATE_T2"
    FAST_ESCALATE_T3 = "GAMBLER_FAST_ESCALATE_T3"
    FAST_ESCALATE_T4 = "GAMBLER_FAST_ESCALATE_T4"
    OPPORTUNITY_MAX_BOOST = "GAMBLER_OP_MAX_BOOST"
    SOFT_EXIT_EARLY = "GAMBLER_SOFT_EXIT_EARLY"


@dataclass
class HighRiskModeConfig:
    """
    High-Risk Gambler Mode Configuration.
    
    All parameters are compared against NORMAL mode defaults.
    Only active when ENABLE_HIGH_RISK_GAMBLER_MODE = True.
    """
    
    # =========================================================================
    # [A] EARLY ENTRY BIAS - Lower thresholds for T1 entry
    # =========================================================================
    # NORMAL: entry_confidence >= 0.75
    # GAMBLER: entry_confidence >= 0.55
    
    # Entry confidence threshold (signal must exceed this)
    entry_confidence_threshold_normal: float = 0.75
    entry_confidence_threshold_gambler: float = 0.55
    
    # Entry agreement threshold (strategies agreeing)
    entry_agreement_threshold_normal: float = 0.70
    entry_agreement_threshold_gambler: float = 0.50
    
    # Entry score minimum (composite signal score)
    entry_score_min_normal: float = 0.65
    entry_score_min_gambler: float = 0.45
    
    # =========================================================================
    # [B] AGGRESSIVE ANTI-MARTINGALE - Faster escalation
    # =========================================================================
    # NORMAL: T2 requires +50bps, T3 requires +100bps, T4 requires +150bps
    # GAMBLER: T2 requires +15bps, T3 requires +40bps, T4 requires +80bps
    
    # T2 profit threshold (bps)
    t2_profit_threshold_normal: float = 0.0    # Original: break-even (0 bps)
    t2_profit_threshold_gambler: float = -20.0  # Allow slight loss (-20 bps)
    
    # T3 profit threshold (bps)  
    t3_profit_threshold_normal: float = 0.0     # Original: break-even (0 bps)
    t3_profit_threshold_gambler: float = 15.0   # Just 15 bps profit
    
    # T4 profit threshold (bps)
    t4_profit_threshold_normal: float = 50.0    # Original: 50 bps profit
    t4_profit_threshold_gambler: float = 30.0   # Only 30 bps profit
    
    # T4 time requirement (hours)
    t4_time_requirement_normal: float = 2.0     # Original: 2 hours
    t4_time_requirement_gambler: float = 0.5    # Only 30 minutes
    
    # Allow escalation during CASCADE=ACCELERATING
    allow_escalate_in_accelerating: bool = True  # Gambler: YES
    # Still blocked in EXHAUSTING - this is a HARD constraint
    
    # =========================================================================
    # [C] OPPORTUNITY ALL-IN MULTIPLIER - Higher position limits
    # =========================================================================
    # NORMAL: max_single_position = 0.80 (80%)
    # GAMBLER in OPPORTUNITY: max_single_position = 0.95 (95%)
    
    # Single asset position cap in NORMAL mode
    max_single_position_normal: float = 0.80
    
    # Single asset position cap in OPPORTUNITY mode (gambler boost)
    max_single_position_opportunity_gambler: float = 0.95
    
    # Gross exposure cap in NORMAL mode
    max_gross_exposure_normal: float = 1.50
    
    # Gross exposure cap in OPPORTUNITY mode (gambler boost)
    max_gross_exposure_opportunity_gambler: float = 2.00
    
    # Concurrent same-direction strategies allowed
    max_concurrent_same_direction_normal: int = 2
    max_concurrent_same_direction_gambler: int = 4
    
    # =========================================================================
    # [D] FAST-FAIL BIAS - Earlier soft exit (NOT changing hard stop)
    # =========================================================================
    # NORMAL: soft exit at -30bps unrealized with adverse momentum
    # GAMBLER: soft exit at -15bps unrealized with adverse momentum
    
    # Soft exit P&L threshold (bps) - triggers early consideration
    soft_exit_pnl_threshold_normal: float = -30.0
    soft_exit_pnl_threshold_gambler: float = -15.0
    
    # Time without confirmation before soft exit (minutes)
    no_confirmation_timeout_normal: float = 120.0   # 2 hours
    no_confirmation_timeout_gambler: float = 45.0   # 45 minutes
    
    # Structure break failure recognition (faster in gambler mode)
    structure_failure_bars_normal: int = 3   # 3 bars to confirm failure
    structure_failure_bars_gambler: int = 1  # 1 bar = immediate failure
    
    # VPIN soft exit threshold (earlier exit in toxic flow)
    vpin_soft_exit_normal: float = 0.80
    vpin_soft_exit_gambler: float = 0.70
    
    def get_entry_confidence_threshold(self, gambler_mode: bool) -> float:
        """Get entry confidence threshold based on mode."""
        return self.entry_confidence_threshold_gambler if gambler_mode else self.entry_confidence_threshold_normal
    
    def get_entry_agreement_threshold(self, gambler_mode: bool) -> float:
        """Get entry agreement threshold based on mode."""
        return self.entry_agreement_threshold_gambler if gambler_mode else self.entry_agreement_threshold_normal
    
    def get_entry_score_min(self, gambler_mode: bool) -> float:
        """Get minimum entry score based on mode."""
        return self.entry_score_min_gambler if gambler_mode else self.entry_score_min_normal
    
    def get_t2_profit_threshold(self, gambler_mode: bool) -> float:
        """Get T2 escalation profit threshold (bps)."""
        return self.t2_profit_threshold_gambler if gambler_mode else self.t2_profit_threshold_normal
    
    def get_t3_profit_threshold(self, gambler_mode: bool) -> float:
        """Get T3 escalation profit threshold (bps)."""
        return self.t3_profit_threshold_gambler if gambler_mode else self.t3_profit_threshold_normal
    
    def get_t4_profit_threshold(self, gambler_mode: bool) -> float:
        """Get T4 escalation profit threshold (bps)."""
        return self.t4_profit_threshold_gambler if gambler_mode else self.t4_profit_threshold_normal
    
    def get_t4_time_requirement(self, gambler_mode: bool) -> float:
        """Get T4 time requirement (hours)."""
        return self.t4_time_requirement_gambler if gambler_mode else self.t4_time_requirement_normal
    
    def get_max_single_position(self, gambler_mode: bool, is_opportunity: bool) -> float:
        """Get max single position cap."""
        if gambler_mode and is_opportunity:
            return self.max_single_position_opportunity_gambler
        return self.max_single_position_normal
    
    def get_max_gross_exposure(self, gambler_mode: bool, is_opportunity: bool) -> float:
        """Get max gross exposure cap."""
        if gambler_mode and is_opportunity:
            return self.max_gross_exposure_opportunity_gambler
        return self.max_gross_exposure_normal
    
    def get_max_concurrent_same_direction(self, gambler_mode: bool) -> int:
        """Get max concurrent same-direction strategies."""
        return self.max_concurrent_same_direction_gambler if gambler_mode else self.max_concurrent_same_direction_normal
    
    def get_soft_exit_pnl_threshold(self, gambler_mode: bool) -> float:
        """Get soft exit P&L threshold (bps)."""
        return self.soft_exit_pnl_threshold_gambler if gambler_mode else self.soft_exit_pnl_threshold_normal
    
    def get_no_confirmation_timeout(self, gambler_mode: bool) -> float:
        """Get no-confirmation timeout (minutes)."""
        return self.no_confirmation_timeout_gambler if gambler_mode else self.no_confirmation_timeout_normal
    
    def get_structure_failure_bars(self, gambler_mode: bool) -> int:
        """Get structure failure bar count."""
        return self.structure_failure_bars_gambler if gambler_mode else self.structure_failure_bars_normal
    
    def get_vpin_soft_exit(self, gambler_mode: bool) -> float:
        """Get VPIN soft exit threshold."""
        return self.vpin_soft_exit_gambler if gambler_mode else self.vpin_soft_exit_normal


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_high_risk_config: Optional[HighRiskModeConfig] = None


def get_high_risk_config() -> HighRiskModeConfig:
    """Get singleton high-risk mode config."""
    global _high_risk_config
    if _high_risk_config is None:
        _high_risk_config = HighRiskModeConfig()
    return _high_risk_config


def is_gambler_mode_active() -> bool:
    """Check if gambler mode is currently active."""
    flags = get_sota_flags()
    return flags.ENABLE_HIGH_RISK_GAMBLER_MODE


def get_gambler_mode_status() -> Dict:
    """Get current gambler mode status for dashboard/logging."""
    active = is_gambler_mode_active()
    config = get_high_risk_config()
    
    return {
        "active": active,
        "mode_name": "HIGH_RISK_GAMBLER" if active else "NORMAL",
        "entry_confidence": config.get_entry_confidence_threshold(active),
        "t2_threshold_bps": config.get_t2_profit_threshold(active),
        "t3_threshold_bps": config.get_t3_profit_threshold(active),
        "t4_threshold_bps": config.get_t4_profit_threshold(active),
        "max_single_position_op": config.get_max_single_position(active, True),
        "soft_exit_threshold_bps": config.get_soft_exit_pnl_threshold(active),
    }


def log_gambler_action(action: GamblerModeReason, details: Dict = None):
    """
    Log a gambler mode action for ShadowLedger audit.
    
    Should be called when gambler-specific logic is triggered.
    """
    if not is_gambler_mode_active():
        return
    
    logger.info(f"[GAMBLER_MODE] {action.value}: {details or {}}")
    
    # Integration point for ShadowLedger
    try:
        from data_mgmt.shadow_ledger_manager import get_shadow_ledger
        ledger = get_shadow_ledger()
        if ledger:
            ledger.log_veto(
                source="GAMBLER_MODE",
                reason=action.value,
                details=details or {}
            )
    except ImportError:
        pass  # ShadowLedger not available


if __name__ == "__main__":
    # Test: Print current configuration
    print("=" * 60)
    print("HIGH-RISK GAMBLER MODE CONFIGURATION")
    print("=" * 60)
    
    config = get_high_risk_config()
    status = get_gambler_mode_status()
    
    print(f"\nActive: {status['active']}")
    print(f"Mode: {status['mode_name']}")
    print(f"\n[A] Entry Confidence: {status['entry_confidence']}")
    print(f"[B] T2 Threshold: {status['t2_threshold_bps']} bps")
    print(f"[B] T3 Threshold: {status['t3_threshold_bps']} bps")
    print(f"[B] T4 Threshold: {status['t4_threshold_bps']} bps")
    print(f"[C] Max Single Position (OP): {status['max_single_position_op']:.0%}")
    print(f"[D] Soft Exit Threshold: {status['soft_exit_threshold_bps']} bps")

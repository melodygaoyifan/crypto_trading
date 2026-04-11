"""
================================================================================
HMATS v6.2.1 - GAMBLER MODE ENTRY/EXIT THRESHOLDS
================================================================================

This module provides the entry and exit threshold adjustments for High-Risk 
Gambler Mode. It does NOT modify any signal definitions or calculations.

WHAT IT MODIFIES:
    [A] Early Entry Bias - Lower confidence/agreement thresholds for T1
    [D] Fast-Fail Bias - Earlier soft exit recognition

WHAT IT DOES NOT MODIFY:
    OFF Signal definitions
    OFF Indicator calculations  
    OFF Final stop loss
    OFF Hard limits

================================================================================
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


@dataclass
class EntryCheckResult:
    """Result of entry threshold check."""
    allowed: bool
    reason: str
    confidence: float
    threshold_used: float
    is_gambler_mode: bool = False


@dataclass
class SoftExitCheckResult:
    """Result of soft exit check."""
    should_exit: bool
    reason: str
    urgency: str  # "immediate", "soon", "monitor"
    is_gambler_mode: bool = False


class GamblerEntryChecker:
    """
    Entry threshold checker with Gambler Mode support.
    
    This does NOT change signal values - only the thresholds for entry.
    """
    
    def __init__(self):
        self._entry_count = 0
    
    def _is_gambler_mode(self) -> bool:
        """Check if gambler mode is active."""
        try:
            from configs.high_risk_mode import is_gambler_mode_active
            return is_gambler_mode_active()
        except ImportError:
            return False
    
    def _get_config(self):
        """Get gambler config."""
        try:
            from configs.high_risk_mode import get_high_risk_config
            return get_high_risk_config()
        except ImportError:
            return None
    
    def _log_gambler_action(self, details: Dict):
        """Log gambler entry for audit."""
        try:
            from configs.high_risk_mode import log_gambler_action, GamblerModeReason
            log_gambler_action(GamblerModeReason.EARLY_ENTRY_T1, details)
        except ImportError:
            pass
    
    def check_entry_allowed(
        self,
        signal_confidence: float,
        strategy_agreement: float,
        composite_score: float,
        tranche_level: int = 1,
        is_opportunity: bool = False,
    ) -> EntryCheckResult:
        """
        Check if entry is allowed based on thresholds.
        
        Args:
            signal_confidence: Confidence from signal (0-1)
            strategy_agreement: Agreement between strategies (0-1)
            composite_score: Composite signal score (0-1)
            tranche_level: Which tranche (1-4)
            is_opportunity: True if in OPPORTUNITY mode
        
        Returns:
            EntryCheckResult with decision and reasoning
        """
        
        gambler_mode = self._is_gambler_mode()
        config = self._get_config()
        
        # Only apply gambler thresholds for T1 entry
        use_gambler_thresholds = gambler_mode and tranche_level == 1
        
        if use_gambler_thresholds and config:
            conf_threshold = config.get_entry_confidence_threshold(True)
            agree_threshold = config.get_entry_agreement_threshold(True)
            score_threshold = config.get_entry_score_min(True)
        else:
            # Normal thresholds
            if config:
                conf_threshold = config.get_entry_confidence_threshold(False)
                agree_threshold = config.get_entry_agreement_threshold(False)
                score_threshold = config.get_entry_score_min(False)
            else:
                # Fallback defaults
                conf_threshold = 0.75
                agree_threshold = 0.70
                score_threshold = 0.65
        
        # Check all thresholds
        conf_ok = signal_confidence >= conf_threshold
        agree_ok = strategy_agreement >= agree_threshold
        score_ok = composite_score >= score_threshold
        
        allowed = conf_ok and agree_ok and score_ok
        
        if not allowed:
            reasons = []
            if not conf_ok:
                reasons.append(f"confidence {signal_confidence:.2f} < {conf_threshold:.2f}")
            if not agree_ok:
                reasons.append(f"agreement {strategy_agreement:.2f} < {agree_threshold:.2f}")
            if not score_ok:
                reasons.append(f"score {composite_score:.2f} < {score_threshold:.2f}")
            reason = "; ".join(reasons)
        else:
            reason = "thresholds met"
            if use_gambler_thresholds:
                reason = "GAMBLER: early entry thresholds met"
                self._log_gambler_action({
                    "signal_confidence": signal_confidence,
                    "strategy_agreement": strategy_agreement,
                    "composite_score": composite_score,
                    "conf_threshold": conf_threshold,
                    "agree_threshold": agree_threshold,
                    "score_threshold": score_threshold,
                })
        
        return EntryCheckResult(
            allowed=allowed,
            reason=reason,
            confidence=signal_confidence,
            threshold_used=conf_threshold,
            is_gambler_mode=use_gambler_thresholds
        )


class GamblerSoftExitChecker:
    """
    Soft exit checker with Gambler Mode support.
    
    Gambler Mode triggers earlier soft exit to "fail fast".
    This does NOT modify the hard stop loss.
    """
    
    def __init__(self):
        self._positions_entered: Dict[str, datetime] = {}
    
    def _is_gambler_mode(self) -> bool:
        """Check if gambler mode is active."""
        try:
            from configs.high_risk_mode import is_gambler_mode_active
            return is_gambler_mode_active()
        except ImportError:
            return False
    
    def _get_config(self):
        """Get gambler config."""
        try:
            from configs.high_risk_mode import get_high_risk_config
            return get_high_risk_config()
        except ImportError:
            return None
    
    def _log_gambler_action(self, details: Dict):
        """Log gambler soft exit for audit."""
        try:
            from configs.high_risk_mode import log_gambler_action, GamblerModeReason
            log_gambler_action(GamblerModeReason.SOFT_EXIT_EARLY, details)
        except ImportError:
            pass
    
    def record_entry(self, asset: str):
        """Record when a position was entered."""
        self._positions_entered[asset] = datetime.now(timezone.utc)
    
    def check_soft_exit(
        self,
        asset: str,
        unrealized_pnl_bps: float,
        vpin: float,
        structure_confirmed: bool,
        adverse_momentum: bool,
        bars_since_entry: int = 0,
        strategy: str = "",  # [2026-04-08] Strategy-differentiated exit thresholds
    ) -> SoftExitCheckResult:
        """
        Check if soft exit should be triggered.
        
        Args:
            asset: Asset symbol
            unrealized_pnl_bps: Current unrealized P&L in bps
            vpin: Current VPIN value
            structure_confirmed: Whether structure break was confirmed
            adverse_momentum: Whether momentum is against position
            bars_since_entry: Number of bars since entry
        
        Returns:
            SoftExitCheckResult with decision and reasoning
        """
        
        gambler_mode = self._is_gambler_mode()
        config = self._get_config()
        
        # Get thresholds based on mode + strategy (strategy overrides take precedence)
        if gambler_mode and config:
            pnl_threshold = config.get_soft_exit_pnl_threshold(True, strategy)
            vpin_threshold = config.get_vpin_soft_exit(True)
            structure_bars = config.get_structure_failure_bars(True, strategy)
            timeout_mins = config.get_no_confirmation_timeout(True, strategy)
        else:
            if config:
                pnl_threshold = config.get_soft_exit_pnl_threshold(False, strategy)
                vpin_threshold = config.get_vpin_soft_exit(False)
                structure_bars = config.get_structure_failure_bars(False, strategy)
                timeout_mins = config.get_no_confirmation_timeout(False, strategy)
            else:
                # Fallback defaults
                pnl_threshold = -30.0
                vpin_threshold = 0.80
                structure_bars = 3
                timeout_mins = 120.0
        
        # Check conditions
        pnl_trigger = unrealized_pnl_bps <= pnl_threshold
        vpin_trigger = vpin >= vpin_threshold
        structure_failure = not structure_confirmed and bars_since_entry >= structure_bars
        
        # Check time-based trigger
        time_trigger = False
        if asset in self._positions_entered:
            mins_in_position = (datetime.now(timezone.utc) - self._positions_entered[asset]).total_seconds() / 60
            time_trigger = mins_in_position >= timeout_mins and not structure_confirmed
        
        # Determine urgency and exit decision
        should_exit = False
        urgency = "monitor"
        reasons = []
        
        if pnl_trigger and adverse_momentum:
            should_exit = True
            urgency = "immediate"
            reasons.append(f"P&L {unrealized_pnl_bps:.0f}bps <= {pnl_threshold:.0f}bps + adverse momentum")
        
        if vpin_trigger:
            should_exit = True
            urgency = "immediate" if urgency == "immediate" else "soon"
            reasons.append(f"VPIN {vpin:.2f} >= {vpin_threshold:.2f}")
        
        if structure_failure:
            should_exit = True
            urgency = "soon" if urgency == "monitor" else urgency
            reasons.append(f"structure unconfirmed after {bars_since_entry} bars")
        
        if time_trigger:
            should_exit = True
            urgency = "soon" if urgency == "monitor" else urgency
            reasons.append(f"timeout: no confirmation after {timeout_mins:.0f}min")
        
        reason = "; ".join(reasons) if reasons else "no soft exit triggers"
        
        if should_exit and gambler_mode:
            self._log_gambler_action({
                "asset": asset,
                "pnl_bps": unrealized_pnl_bps,
                "vpin": vpin,
                "structure_confirmed": structure_confirmed,
                "urgency": urgency,
                "reasons": reasons
            })
        
        return SoftExitCheckResult(
            should_exit=should_exit,
            reason=reason,
            urgency=urgency,
            is_gambler_mode=gambler_mode
        )
    
    def clear_entry(self, asset: str):
        """Clear entry record when position closed."""
        self._positions_entered.pop(asset, None)


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_entry_checker: Optional[GamblerEntryChecker] = None
_exit_checker: Optional[GamblerSoftExitChecker] = None


def get_entry_checker() -> GamblerEntryChecker:
    """Get singleton entry checker."""
    global _entry_checker
    if _entry_checker is None:
        _entry_checker = GamblerEntryChecker()
    return _entry_checker


def get_soft_exit_checker() -> GamblerSoftExitChecker:
    """Get singleton soft exit checker."""
    global _exit_checker
    if _exit_checker is None:
        _exit_checker = GamblerSoftExitChecker()
    return _exit_checker


def reset_checkers():
    """Reset singletons (for testing)."""
    global _entry_checker, _exit_checker
    _entry_checker = None
    _exit_checker = None


if __name__ == "__main__":
    # Test: Check entry thresholds
    print("=" * 60)
    print("GAMBLER ENTRY/EXIT CHECKER TEST")
    print("=" * 60)
    
    entry = get_entry_checker()
    exit_checker = get_soft_exit_checker()
    
    # Test entry
    result = entry.check_entry_allowed(
        signal_confidence=0.60,
        strategy_agreement=0.55,
        composite_score=0.50,
        tranche_level=1,
        is_opportunity=True
    )
    print(f"\nEntry Check:")
    print(f"  Allowed: {result.allowed}")
    print(f"  Reason: {result.reason}")
    print(f"  Gambler Mode: {result.is_gambler_mode}")
    
    # Test soft exit
    exit_checker.record_entry("SOL")
    exit_result = exit_checker.check_soft_exit(
        asset="SOL",
        unrealized_pnl_bps=-20,
        vpin=0.72,
        structure_confirmed=False,
        adverse_momentum=True,
        bars_since_entry=2
    )
    print(f"\nSoft Exit Check:")
    print(f"  Should Exit: {exit_result.should_exit}")
    print(f"  Urgency: {exit_result.urgency}")
    print(f"  Reason: {exit_result.reason}")
    print(f"  Gambler Mode: {exit_result.is_gambler_mode}")

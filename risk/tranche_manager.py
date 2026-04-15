"""
HMATS v3.2 - Tranche Manager (Pyramiding)
==========================================
Implements Strategy Constitution Section 5.

v6.2.1: Added High-Risk Gambler Mode support for aggressive escalation.

Tranche Schedule:
- Tranche-1: 20% (early entry in OPPORTUNITY)
- Tranche-2: 30% (4H close OR structure confirmation)
- Tranche-3: 30% (profit + regime aligned)
- Tranche-4: 20% (full confirmation + momentum)

Abort Conditions:
- VPIN spike > 0.85
- DVOL shock (increase > 1.5 in 1H)
- Signal conflict emerges
- Price reversal > 1%
- Volume collapse < 0.3x average
- NO_TRADE triggered
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
import logging

from core.market_data_helpers import effective_volume_ratio
from orchestration.system_mode import SystemMode

logger = logging.getLogger(__name__)


from core.canonical_enums import TrancheLevel  # Single source of truth


class TrancheAction(Enum):
    """Actions for tranche management."""
    HOLD = auto()           # Hold current position
    ENTER_TRANCHE_1 = auto()
    ESCALATE_TO_2 = auto()
    ESCALATE_TO_3 = auto()
    ESCALATE_TO_4 = auto()
    REDUCE_TO_1 = auto()    # Abort: reduce to T1 size
    FLATTEN = auto()        # Abort: close all


@dataclass
class TranchePosition:
    """Current tranche position state."""
    asset: str
    level: TrancheLevel = TrancheLevel.NONE
    direction: str = "none"  # "long", "short", "none"
    target_exposure: float = 0.0
    current_exposure: float = 0.0
    
    entry_price: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl_bps: float = 0.0
    
    entered_at: Optional[datetime] = None
    last_escalation: Optional[datetime] = None
    
    def cumulative_size(self) -> float:
        """Get cumulative size for current level."""
        # [FIX-13] Synced with unified_position_sizer.py tranche_percentages
        sizes = {
            TrancheLevel.NONE: 0.0,
            TrancheLevel.TRANCHE_1: 0.35,
            TrancheLevel.TRANCHE_2: 0.65,
            TrancheLevel.TRANCHE_3: 0.90,
            TrancheLevel.TRANCHE_4: 1.00
        }
        return sizes.get(self.level, 0.0)

    def current_tranche_size(self) -> float:
        """Get size of just the current tranche."""
        # [FIX-13] Synced: T1=35%, T2=30%, T3=25%, T4=10%
        sizes = {
            TrancheLevel.NONE: 0.0,
            TrancheLevel.TRANCHE_1: 0.35,
            TrancheLevel.TRANCHE_2: 0.30,
            TrancheLevel.TRANCHE_3: 0.25,
            TrancheLevel.TRANCHE_4: 0.10
        }
        return sizes.get(self.level, 0.0)


@dataclass
class TrancheDecision:
    """Decision from tranche manager."""
    action: TrancheAction
    new_exposure: float
    exposure_delta: float
    reason: str
    
    def to_dict(self) -> Dict:
        return {
            "action": self.action.name,
            "new_exposure": self.new_exposure,
            "exposure_delta": self.exposure_delta,
            "reason": self.reason
        }


@dataclass
class AbortConditions:
    """Current state of abort conditions."""
    vpin_spike: bool = False
    dvol_shock: bool = False
    signal_conflict: bool = False
    price_reversal: bool = False
    volume_collapse: bool = False
    no_trade_triggered: bool = False
    
    def any_triggered(self) -> bool:
        return any([
            self.vpin_spike,
            self.dvol_shock,
            self.signal_conflict,
            self.price_reversal,
            self.volume_collapse,
            self.no_trade_triggered
        ])
    
    def triggered_reasons(self) -> List[str]:
        reasons = []
        if self.vpin_spike:
            reasons.append("VPIN > 0.85")
        if self.dvol_shock:
            reasons.append("DVOL shock")
        if self.signal_conflict:
            reasons.append("Signal conflict")
        if self.price_reversal:
            reasons.append("Price reversal > threshold")
        if self.volume_collapse:
            reasons.append("Volume collapse")
        if self.no_trade_triggered:
            reasons.append("NO_TRADE mode")
        return reasons


class TrancheManager:
    """
    Manages position pyramiding with tranches.

    From Strategy Constitution Section 5.

    v6.2.1: High-Risk Gambler Mode support:
        - Faster T2/T3/T4 escalation with lower profit thresholds
        - Allow escalation during CASCADE=ACCELERATING
        - All changes gated by ENABLE_HIGH_RISK_GAMBLER_MODE flag

    P1-03: Accepts optional tranche_config dict for profile-gated overrides.
    """

    # Default tranche sizing (individual increments)
    _DEFAULT_T1_SIZE = 0.20  # 20%
    _DEFAULT_T2_SIZE = 0.30  # 30%
    _DEFAULT_T3_SIZE = 0.30  # 30%
    _DEFAULT_T4_SIZE = 0.20  # 20%

    # Default cumulative sizes
    # [TRANCHE-CONSISTENCY 2026-04-15] Aligned to canonical_config.TRANCHE_CUMULATIVE
    # (35/65/90/100). Was 20/50/80/100 — silent fallback to a different schedule
    # if JSON config dropped sizes_cumulative key. TranchePosition.cumulative_size()
    # already used 35/65/90/100, so this default was internally inconsistent.
    _DEFAULT_CUMULATIVE = {
        TrancheLevel.NONE: 0.0,
        TrancheLevel.TRANCHE_1: 0.35,
        TrancheLevel.TRANCHE_2: 0.65,
        TrancheLevel.TRANCHE_3: 0.90,
        TrancheLevel.TRANCHE_4: 1.00,
    }

    # Abort thresholds
    VPIN_SPIKE_THRESHOLD = 0.85
    DVOL_SHOCK_THRESHOLD = 1.5  # z-score increase in 1H
    # Bug #43: Per-regime price reversal thresholds (must match constitution.py)
    PRICE_REVERSAL_THRESHOLDS = {
        "QUIET_ACCUMULATION":   0.02,
        "WEAK_CONSOLIDATION":   0.02,
        "MOMENTUM_RALLY":       0.025,
        "VOLATILE_CHOP":        0.03,
        "EXTREME_VOLATILITY":   0.04,
        "PANIC_SELLOFF":        0.05,
    }
    PRICE_REVERSAL_THRESHOLD_DEFAULT = 0.025
    VOLUME_COLLAPSE_THRESHOLD = 0.05  # [VOL-FIX 2026-04-15] 0.10 → 0.05: live data vol_ratio routinely 0.10-0.20 mid-bar; would force-abort fresh positions. Real collapse <0.05 still blocks.

    # Default escalation requirements (NORMAL mode - can be overridden by gambler mode or profile)
    _DEFAULT_T3_MIN_PROFIT_BPS = 0  # At least break-even
    _DEFAULT_T4_MIN_PROFIT_BPS = 50  # At least 50 bps profit
    _DEFAULT_T4_MIN_TIME_HOURS = 2  # At least 2 hours in position

    def __init__(self, tranche_config: Optional[Dict] = None):
        self.positions: Dict[str, TranchePosition] = {}
        self._last_dvol_zscores: List[Tuple[datetime, float]] = []

        # P1-03: Apply profile-gated config overrides
        cfg = tranche_config or {}
        self._tranche_config = copy.deepcopy(cfg)

        # Cumulative sizing from config
        sizes_cfg = cfg.get("sizes_cumulative", {})
        self._cumulative_sizes = dict(self._DEFAULT_CUMULATIVE)
        if sizes_cfg:
            _level_map = {
                "TRANCHE_1": TrancheLevel.TRANCHE_1,
                "TRANCHE_2": TrancheLevel.TRANCHE_2,
                "TRANCHE_3": TrancheLevel.TRANCHE_3,
                "TRANCHE_4": TrancheLevel.TRANCHE_4,
            }
            for name, level in _level_map.items():
                if name in sizes_cfg:
                    self._cumulative_sizes[level] = float(sizes_cfg[name])

        # Derive individual tranche sizes from cumulative
        self.TRANCHE_1_SIZE = self._cumulative_sizes[TrancheLevel.TRANCHE_1]
        self.TRANCHE_2_SIZE = self._cumulative_sizes[TrancheLevel.TRANCHE_2] - self._cumulative_sizes[TrancheLevel.TRANCHE_1]
        self.TRANCHE_3_SIZE = self._cumulative_sizes[TrancheLevel.TRANCHE_3] - self._cumulative_sizes[TrancheLevel.TRANCHE_2]
        self.TRANCHE_4_SIZE = self._cumulative_sizes[TrancheLevel.TRANCHE_4] - self._cumulative_sizes[TrancheLevel.TRANCHE_3]

        # Escalation parameters
        self.TRANCHE_3_MIN_PROFIT_BPS = cfg.get("t3_min_profit_bps", self._DEFAULT_T3_MIN_PROFIT_BPS)
        self.TRANCHE_4_MIN_PROFIT_BPS = cfg.get("t4_min_profit_bps", self._DEFAULT_T4_MIN_PROFIT_BPS)
        self.TRANCHE_4_MIN_TIME_HOURS = cfg.get("t4_min_time_hours", self._DEFAULT_T4_MIN_TIME_HOURS)
    
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
    
    def _log_gambler_action(self, action, details: Dict = None):
        """Log gambler mode action for audit."""
        try:
            from configs.high_risk_mode import log_gambler_action, GamblerModeReason
            if hasattr(GamblerModeReason, action):
                log_gambler_action(getattr(GamblerModeReason, action), details)
        except ImportError:
            pass
    
    def get_position(self, asset: str) -> TranchePosition:
        """Get position for asset, creating if needed."""
        if asset not in self.positions:
            self.positions[asset] = TranchePosition(asset=asset)
        return self.positions[asset]
    
    def decide(
        self,
        asset: str,
        mode: SystemMode,
        target_exposure: float,
        current_price: float,
        market_data: Dict,
        signal_data: Dict,
        is_4h_bar_close: bool = False
    ) -> TrancheDecision:
        """
        Decide tranche action based on current state.
        
        From Strategy Constitution Section 5.5.
        """
        
        position = self.get_position(asset)
        
        # Check abort conditions
        abort = self._check_abort_conditions(position, market_data, signal_data)
        
        # Handle NO_TRADE
        if mode == SystemMode.NO_TRADE:
            abort.no_trade_triggered = True
            return self._handle_abort(position, target_exposure, abort)
        
        # Handle abort conditions
        if abort.any_triggered():
            return self._handle_abort(position, target_exposure, abort)
        
        # Update position P&L
        self._update_pnl(position, current_price)
        
        # NORMAL mode: no early entry, only execute at 4H
        if mode == SystemMode.NORMAL:
            if is_4h_bar_close:
                return self._execute_full_position(position, target_exposure)
            else:
                return TrancheDecision(
                    action=TrancheAction.HOLD,
                    new_exposure=position.current_exposure,
                    exposure_delta=0,
                    reason="NORMAL mode: waiting for 4H bar close"
                )
        
        # OPPORTUNITY mode: pyramiding allowed
        elif mode == SystemMode.OPPORTUNITY:
            return self._manage_opportunity_tranches(
                position, target_exposure, market_data, signal_data, is_4h_bar_close
            )
        
        return TrancheDecision(
            action=TrancheAction.HOLD,
            new_exposure=position.current_exposure,
            exposure_delta=0,
            reason="Default hold"
        )
    
    def _check_abort_conditions(
        self,
        position: TranchePosition,
        market_data: Dict,
        signal_data: Dict
    ) -> AbortConditions:
        """Check all abort conditions."""
        
        abort = AbortConditions()
        
        # VPIN spike
        vpin = market_data.get('vpin', 0)
        if vpin > self.VPIN_SPIKE_THRESHOLD:
            abort.vpin_spike = True
        
        # DVOL shock
        dvol_zscore = market_data.get('dvol_zscore', 0)
        abort.dvol_shock = self._check_dvol_shock(dvol_zscore)
        
        # Signal conflict
        quant_dir = signal_data.get('quant_direction', 0)
        drl_dir = signal_data.get('drl_direction', 0)
        if position.direction == "long" and (quant_dir < -0.3 or drl_dir < -0.3):
            abort.signal_conflict = True
        elif position.direction == "short" and (quant_dir > 0.3 or drl_dir > 0.3):
            abort.signal_conflict = True
        
        # Price reversal - per-regime dynamic threshold (Bug #43)
        regime = market_data.get('regime_state', '')
        reversal_bps_threshold = self.PRICE_REVERSAL_THRESHOLDS.get(
            regime, self.PRICE_REVERSAL_THRESHOLD_DEFAULT
        ) * 10000  # Convert to bps
        if position.unrealized_pnl_bps < -reversal_bps_threshold:
            abort.price_reversal = True
        
        # Volume collapse
        volume_ratio = effective_volume_ratio(market_data)
        if volume_ratio < self.VOLUME_COLLAPSE_THRESHOLD:
            abort.volume_collapse = True
        
        return abort
    
    def _check_dvol_shock(self, current_dvol: float) -> bool:
        """Check for DVOL shock (increase > 1.5 in 1H)."""
        now = datetime.now(timezone.utc)
        
        # Add current reading
        self._last_dvol_zscores.append((now, current_dvol))
        
        # Clean old readings
        cutoff = now - timedelta(hours=1)
        self._last_dvol_zscores = [
            (t, d) for t, d in self._last_dvol_zscores if t > cutoff
        ]
        
        if len(self._last_dvol_zscores) >= 2:
            oldest_dvol = self._last_dvol_zscores[0][1]
            increase = current_dvol - oldest_dvol
            return increase > self.DVOL_SHOCK_THRESHOLD
        
        return False
    
    def _handle_abort(
        self,
        position: TranchePosition,
        target_exposure: float,
        abort: AbortConditions
    ) -> TrancheDecision:
        """Handle abort conditions."""
        
        reasons = abort.triggered_reasons()
        reason_str = ", ".join(reasons)
        
        # NO_TRADE = flatten all
        if abort.no_trade_triggered:
            new_exposure = 0.0
            action = TrancheAction.FLATTEN
            reason = f"NO_TRADE: flatten all ({reason_str})"
        
        # Losing > 100bps = reduce to T1
        elif position.unrealized_pnl_bps < -100:
            new_exposure = target_exposure * self.TRANCHE_1_SIZE
            action = TrancheAction.REDUCE_TO_1
            reason = f"Abort (losing): reduce to T1 ({reason_str})"
        
        # Other abort = hold current
        else:
            new_exposure = position.current_exposure
            action = TrancheAction.HOLD
            reason = f"Abort: stop pyramiding ({reason_str})"
        
        return TrancheDecision(
            action=action,
            new_exposure=new_exposure,
            exposure_delta=new_exposure - position.current_exposure,
            reason=reason
        )
    
    def _update_pnl(self, position: TranchePosition, current_price: float):
        """Update position P&L."""
        if position.avg_price > 0 and current_price > 0:
            if position.direction == "long":
                pnl_pct = (current_price - position.avg_price) / position.avg_price
            else:  # short
                pnl_pct = (position.avg_price - current_price) / position.avg_price
            position.unrealized_pnl_bps = pnl_pct * 10000
    
    def _execute_full_position(
        self,
        position: TranchePosition,
        target_exposure: float
    ) -> TrancheDecision:
        """Execute full position at 4H bar close (NORMAL mode)."""
        
        new_exposure = target_exposure
        delta = new_exposure - position.current_exposure
        
        return TrancheDecision(
            action=TrancheAction.ESCALATE_TO_4 if abs(new_exposure) > 0.8 else TrancheAction.ESCALATE_TO_2,
            new_exposure=new_exposure,
            exposure_delta=delta,
            reason="NORMAL mode: full position at 4H close"
        )
    
    def _manage_opportunity_tranches(
        self,
        position: TranchePosition,
        target_exposure: float,
        market_data: Dict,
        signal_data: Dict,
        is_4h_bar_close: bool
    ) -> TrancheDecision:
        """Manage tranches in OPPORTUNITY mode."""
        
        current_level = position.level
        
        # No position yet - enter Tranche-1
        if current_level == TrancheLevel.NONE:
            new_exposure = target_exposure * self.TRANCHE_1_SIZE
            position.level = TrancheLevel.TRANCHE_1
            position.direction = "long" if target_exposure > 0 else "short"
            position.target_exposure = target_exposure
            position.entered_at = datetime.now(timezone.utc)
            position.last_escalation = datetime.now(timezone.utc)
            
            return TrancheDecision(
                action=TrancheAction.ENTER_TRANCHE_1,
                new_exposure=new_exposure,
                exposure_delta=new_exposure - position.current_exposure,
                reason="OPPORTUNITY: entering Tranche-1 (early entry)"
            )
        
        # Tranche-1 -> Tranche-2
        elif current_level == TrancheLevel.TRANCHE_1:
            can_escalate = self._can_escalate_to_2(position, market_data, is_4h_bar_close)
            if can_escalate:
                new_exposure = target_exposure * (self.TRANCHE_1_SIZE + self.TRANCHE_2_SIZE)
                position.level = TrancheLevel.TRANCHE_2
                position.last_escalation = datetime.now(timezone.utc)
                
                return TrancheDecision(
                    action=TrancheAction.ESCALATE_TO_2,
                    new_exposure=new_exposure,
                    exposure_delta=new_exposure - position.current_exposure,
                    reason="Escalating to Tranche-2"
                )
        
        # Tranche-2 -> Tranche-3
        elif current_level == TrancheLevel.TRANCHE_2:
            can_escalate = self._can_escalate_to_3(position, market_data, signal_data)
            if can_escalate:
                new_exposure = target_exposure * (self.TRANCHE_1_SIZE + self.TRANCHE_2_SIZE + self.TRANCHE_3_SIZE)
                position.level = TrancheLevel.TRANCHE_3
                position.last_escalation = datetime.now(timezone.utc)
                
                return TrancheDecision(
                    action=TrancheAction.ESCALATE_TO_3,
                    new_exposure=new_exposure,
                    exposure_delta=new_exposure - position.current_exposure,
                    reason="Escalating to Tranche-3"
                )
        
        # Tranche-3 -> Tranche-4
        elif current_level == TrancheLevel.TRANCHE_3:
            can_escalate = self._can_escalate_to_4(position, market_data)
            if can_escalate:
                new_exposure = target_exposure  # Full 100%
                position.level = TrancheLevel.TRANCHE_4
                position.last_escalation = datetime.now(timezone.utc)
                
                return TrancheDecision(
                    action=TrancheAction.ESCALATE_TO_4,
                    new_exposure=new_exposure,
                    exposure_delta=new_exposure - position.current_exposure,
                    reason="Escalating to Tranche-4 (full position)"
                )
        
        # Already at Tranche-4 or can't escalate
        return TrancheDecision(
            action=TrancheAction.HOLD,
            new_exposure=position.current_exposure,
            exposure_delta=0,
            reason=f"Holding at {current_level.name}"
        )
    
    def _can_escalate_to_2(
        self,
        position: TranchePosition,
        market_data: Dict,
        is_4h_bar_close: bool
    ) -> bool:
        """
        Check if can escalate to Tranche-2.
        
        Requires:
        - 4H bar close AND direction confirmed, OR
        - Structure break confirmed AND volume sustained
        
        v6.2.1 GAMBLER MODE:
        - Also allow escalation with lower profit threshold
        - Allow during CASCADE=ACCELERATING
        """
        
        gambler_mode = self._is_gambler_mode()
        config = self._get_gambler_config()
        
        if is_4h_bar_close:
            return True  # 4H close is enough
        
        # Structure confirmation
        structure_confirmed = market_data.get('structure_break_confirmed', False)
        volume_sustained = market_data.get('volume_ratio', 0) >= 1.5
        
        if structure_confirmed and volume_sustained:
            return True
        
        # GAMBLER MODE: Allow T2 with minimal profit (or slight loss)
        if gambler_mode and config:
            t2_threshold = config.get_t2_profit_threshold(True)
            if position.unrealized_pnl_bps >= t2_threshold:
                # Check if in ACCELERATING cascade (still allowed in gambler mode)
                cascade_state = market_data.get('cascade_state', 'NONE')
                if cascade_state == 'EXHAUSTING':
                    return False  # HARD CONSTRAINT: Never in EXHAUSTING
                
                self._log_gambler_action("FAST_ESCALATE_T2", {
                    "pnl_bps": position.unrealized_pnl_bps,
                    "threshold": t2_threshold,
                    "cascade_state": cascade_state
                })
                return True
        
        return False
    
    def _can_escalate_to_3(
        self,
        position: TranchePosition,
        market_data: Dict,
        signal_data: Dict
    ) -> bool:
        """
        Check if can escalate to Tranche-3.
        
        Requires:
        - Position in profit OR break-even
        - Regime still aligned
        - No VPIN spike (< 0.75)
        - No opposing sentiment shock
        
        v6.2.1 GAMBLER MODE:
        - Lower profit threshold (just 15 bps)
        - Allow during CASCADE=ACCELERATING
        """
        
        gambler_mode = self._is_gambler_mode()
        config = self._get_gambler_config()
        
        # Get profit threshold based on mode
        if gambler_mode and config:
            min_profit_bps = config.get_t3_profit_threshold(True)
        else:
            min_profit_bps = self.TRANCHE_3_MIN_PROFIT_BPS
        
        # Must meet profit threshold
        if position.unrealized_pnl_bps < min_profit_bps:
            return False
        
        # Regime aligned
        regime_aligned = signal_data.get('regime_aligned', True)
        if not regime_aligned:
            return False
        
        # VPIN not elevated
        vpin = market_data.get('vpin', 0)
        if vpin >= 0.75:
            return False
        
        # No opposing sentiment
        sentiment_dir = signal_data.get('sentiment_direction', 0)
        if position.direction == "long" and sentiment_dir < -0.5:
            return False
        if position.direction == "short" and sentiment_dir > 0.5:
            return False
        
        # GAMBLER MODE: Check CASCADE state
        if gambler_mode:
            cascade_state = market_data.get('cascade_state', 'NONE')
            if cascade_state == 'EXHAUSTING':
                return False  # HARD CONSTRAINT: Never in EXHAUSTING
            
            self._log_gambler_action("FAST_ESCALATE_T3", {
                "pnl_bps": position.unrealized_pnl_bps,
                "threshold": min_profit_bps,
                "cascade_state": cascade_state
            })
        
        return True
    
    def _can_escalate_to_4(
        self,
        position: TranchePosition,
        market_data: Dict
    ) -> bool:
        """
        Check if can escalate to Tranche-4.
        
        Requires:
        - Position profit > 50 bps
        - Momentum indicator positive
        - Time in position > 2 hours
        - No approaching resistance/support
        
        v6.2.1 GAMBLER MODE:
        - Lower profit threshold (30 bps)
        - Shorter time requirement (30 min)
        - Allow during CASCADE=ACCELERATING
        """
        
        gambler_mode = self._is_gambler_mode()
        config = self._get_gambler_config()
        
        # Get thresholds based on mode
        if gambler_mode and config:
            min_profit_bps = config.get_t4_profit_threshold(True)
            min_time_hours = config.get_t4_time_requirement(True)
        else:
            min_profit_bps = self.TRANCHE_4_MIN_PROFIT_BPS
            min_time_hours = self.TRANCHE_4_MIN_TIME_HOURS
        
        # Must be in profit above threshold
        if position.unrealized_pnl_bps < min_profit_bps:
            return False
        
        # Time in position
        if position.entered_at:
            hours_in_position = (datetime.now(timezone.utc) - position.entered_at).total_seconds() / 3600
            if hours_in_position < min_time_hours:
                return False
        
        # Momentum positive
        momentum = market_data.get('momentum_indicator', 0)
        if position.direction == "long" and momentum < 0:
            return False
        if position.direction == "short" and momentum > 0:
            return False
        
        # No nearby resistance/support
        approaching_level = market_data.get('approaching_key_level', False)
        if approaching_level:
            return False
        
        # GAMBLER MODE: Check CASCADE state
        if gambler_mode:
            cascade_state = market_data.get('cascade_state', 'NONE')
            if cascade_state == 'EXHAUSTING':
                return False  # HARD CONSTRAINT: Never in EXHAUSTING
            
            self._log_gambler_action("FAST_ESCALATE_T4", {
                "pnl_bps": position.unrealized_pnl_bps,
                "threshold": min_profit_bps,
                "time_hours": hours_in_position if position.entered_at else 0,
                "cascade_state": cascade_state
            })
        
        return True
    
    def reset_position(self, asset: str):
        """Reset position for asset."""
        self.positions[asset] = TranchePosition(asset=asset)
    
    def update_executed(self, asset: str, new_exposure: float, fill_price: float):
        """Update position after execution."""
        position = self.get_position(asset)
        
        # Update average price
        if position.current_exposure == 0:
            position.avg_price = fill_price
        else:
            # Weighted average
            old_value = position.current_exposure * position.avg_price
            new_value = (new_exposure - position.current_exposure) * fill_price
            position.avg_price = (old_value + new_value) / new_exposure if new_exposure != 0 else 0
        
        position.current_exposure = new_exposure


# Singleton instance
_tranche_manager: Optional[TrancheManager] = None


def get_tranche_manager(tranche_config: Optional[Dict] = None) -> TrancheManager:
    """Get or create singleton TrancheManager. Pass tranche_config on first call."""
    global _tranche_manager
    if _tranche_manager is None:
        _tranche_manager = TrancheManager(tranche_config=tranche_config)
    elif (_tranche_manager._tranche_config if hasattr(_tranche_manager, "_tranche_config") else {}) != (tranche_config or {}):
        logger.info("[TRANCHE_MANAGER] Config changed; refreshing singleton instance")
        _tranche_manager = TrancheManager(tranche_config=tranche_config)
    return _tranche_manager


def reset_tranche_manager():
    """Reset singleton TrancheManager (for testing)."""
    global _tranche_manager
    _tranche_manager = None

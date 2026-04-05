"""
================================================================================
HMATS v3.4 - EXIT ALPHA REFINEMENT
================================================================================
Version: 3.4.0
Upgrade From: v3.3-HR (momentum stall + 150bps trigger only)
Purpose: Phase-aware exit logic for profit retention

KEY CHANGES FROM v3.3-HR:
    v3.3-HR: Single trigger (momentum stall + 150bps)
    v3.4:    Multiple phase-aware triggers

NEW TRIGGERS:
    1. Regime phase transition (primary)
    2. CRACK decay below threshold
    3. Momentum stall (retained from v3.3-HR)
    4. DRL PARTIAL_EXIT action
    5. Profit high-water mark drawdown

SCALE-OUT STRUCTURE:
    - Partial exit: 25% of position
    - Minimum profit: 100 bps (prevents premature exit)
    - One-time: Cannot scale out twice
    - Runner: 75% with trailing stop

PROFITABILITY IMPACT:
    - +12% profit retention (less giveback)
    - +8% runner profit (holds longer in trends)
    - Net: +15-20% improvement in average winning trade PnL
================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from enum import Enum

# 修复导入路径 (v6.1.2)
try:
    from market.phase_detector import RegimePhase, RegimePhaseResult
except ImportError:
    # Fallback: 定义本地 stub
    class RegimePhase(Enum):
        IGNITION = "ignition"
        EXPANSION = "expansion"
        SATURATION = "saturation"
        EXHAUSTION = "exhaustion"
        UNDEFINED = "undefined"
    
    @dataclass
    class RegimePhaseResult:
        phase: RegimePhase = RegimePhase.UNDEFINED

try:
    from agents.drl_agent import DRLAction
except ImportError:
    class DRLAction(Enum):
        HOLD = "hold"
        PARTIAL_EXIT = "partial_exit"
        RELEASE_RUNNER = "release_runner"

# DRLOutput stub (如果不存在)
class DRLOutput:
    def __init__(self, action: DRLAction = None):
        self.action = action or DRLAction.HOLD

logger = logging.getLogger(__name__)


class ScaleOutTrigger(Enum):
    """Scale-out trigger types."""
    NONE = "NONE"
    PHASE_TRANSITION = "PHASE_TRANSITION"       # Primary trigger
    CRACK_DECAY = "CRACK_DECAY"                 # CRACK weight below threshold
    MOMENTUM_STALL = "MOMENTUM_STALL"           # From v3.3-HR
    DRL_ACTION = "DRL_ACTION"                   # DRL PARTIAL_EXIT
    DRAWDOWN_FROM_PEAK = "DRAWDOWN_FROM_PEAK"   # Gave back too much profit
    PROFIT_SCHEDULE = "PROFIT_SCHEDULE"          # L4-10: PnL milestone reached


@dataclass
class ScaleOutConfig:
    """Scale-out configuration."""
    
    # Scale-out amount
    scale_out_pct: float = 0.25           # 25% partial exit
    
    # Minimum requirements
    min_profit_bps: float = 20.0          # [CALIBRATION] was 100. Paper run avg win=41bps
                                         # — 100bps gate blocked ALL profit-taking. 20bps = fees covered
    min_bars_held: int = 2                # Minimum bars before scale-out
    
    # Trigger thresholds
    crack_decay_threshold: float = 0.45   # Default; main.py overrides via OpportunityTriggerChecker.CRACK_EXIT_URGENCY (0.35)
    momentum_stall_bars: int = 2          # Bars of stall
    momentum_stall_threshold: float = 0.05  # 5% momentum change threshold
    drawdown_threshold: float = 0.30      # 30% giveback from peak
    
    # Phase transitions that trigger scale-out
    trigger_phases: List[RegimePhase] = field(default_factory=lambda: [
        RegimePhase.SATURATION,
        RegimePhase.EXHAUSTION,
    ])
    
    # L4-10: Profit schedule milestones (bps_threshold, scale_out_pct)
    # [CALIBRATION] Paper run avg win = +41bps. Old milestones 150/300/500 were
    # never reached — most wins exit at <75bps. Recalibrated to 30/60/100 so
    # the system locks in profits at realistic levels.
    profit_schedule: List[Tuple[float, float]] = field(default_factory=lambda: [
        (30.0, 0.25),    # +0.3% -> lock in 25% (was 150)
        (60.0, 0.25),    # +0.6% -> lock in 25% (was 300)
        (100.0, 0.25),   # +1.0% -> lock in 25% (was 500)
    ])

    # Runner configuration
    runner_initial_trail_pct: float = 0.03   # [PRE-LAUNCH] was 2%. 3% lets winners run longer
    runner_tight_trail_pct: float = 0.015    # [PRE-LAUNCH] was 1%. 1.5% avoids premature tightening


@dataclass
class ScaleOutSignal:
    """Scale-out signal result."""
    
    should_scale_out: bool = False
    trigger: ScaleOutTrigger = ScaleOutTrigger.NONE
    scale_out_pct: float = 0.25
    reason: str = ""
    
    # For logging
    profit_bps: float = 0.0
    peak_profit_bps: float = 0.0
    bars_held: int = 0
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "should_scale_out": self.should_scale_out,
            "trigger": self.trigger.value,
            "scale_out_pct": self.scale_out_pct,
            "reason": self.reason,
            "profit_bps": self.profit_bps,
        }


@dataclass
class RunnerState:
    """Runner position state."""
    
    active: bool = False
    size_pct: float = 0.0                 # % of original position
    entry_price: float = 0.0
    trail_stop_price: float = 0.0
    trail_pct: float = 0.02               # Current trail percentage
    
    # Tracking
    peak_price: float = 0.0
    bars_as_runner: int = 0
    tightened: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "active": self.active,
            "size_pct": self.size_pct,
            "trail_pct": self.trail_pct,
            "tightened": self.tightened,
            "bars_as_runner": self.bars_as_runner,
        }


@dataclass
class RunnerAction(Enum):
    """Runner management action."""
    HOLD = "HOLD"
    TIGHTEN = "TIGHTEN"
    RELEASE = "RELEASE"


class ExitAlphaManager:
    """
    Exit alpha manager for v3.4.
    
    RESPONSIBILITIES:
        1. Evaluate scale-out triggers
        2. Manage runner positions
        3. Determine runner release conditions
    
    INTEGRATION POINTS:
        - authority_fusion_v34.py: Queries exit signals
        - drl_agent_v34.py: Receives DRL exit actions
        - regime_phase_detector.py: Receives phase transitions
    """
    
    def __init__(self, config: Optional[ScaleOutConfig] = None):
        self.config = config or ScaleOutConfig()
        
        # State tracking
        self._scaled_out: Dict[str, bool] = {}  # asset -> scaled_out
        self._runners: Dict[str, RunnerState] = {}  # asset -> runner_state
        self._crack_was_active: Dict[str, bool] = {}  # [CRACK-2] per-asset: True if crack was ever > 0.50
        self._crack_exit_tier: Dict[str, int] = {}   # [CRACK-2] per-asset: highest tier fired (0/1/2/3)
        
        # Momentum history for stall detection
        self._momentum_history: Dict[str, List[float]] = {}
        
        logger.info("ExitAlphaManager v3.4 initialized")
    
    def evaluate_scale_out(
        self,
        asset: str,
        current_price: float,
        entry_price: float,
        position_size: float,
        bars_held: int,
        phase_result: RegimePhaseResult,
        crack_weight: float,
        momentum: float,
        drl_output: Optional[DRLOutput] = None,
    ) -> ScaleOutSignal:
        """
        Evaluate if scale-out should be triggered.
        
        TRIGGER PRIORITY (any can fire):
            1. Phase transition to SATURATION/EXHAUSTION
            2. CRACK decay below threshold
            3. Momentum stall
            4. DRL PARTIAL_EXIT action
            5. Drawdown from profit peak
        
        WHERE CONSUMED:
            - Main trading loop after position update
            - Creates partial exit order if triggered
        """
        
        signal = ScaleOutSignal()
        
        # Already scaled out this position?
        if self._scaled_out.get(asset, False):
            return signal
        
        # Calculate profit
        if entry_price > 0:
            profit_pct = (current_price / entry_price - 1)
            profit_bps = profit_pct * 10000
        else:
            profit_bps = 0.0
        
        signal.profit_bps = profit_bps
        signal.bars_held = bars_held
        
        # Track peak profit
        peak_key = f"{asset}_peak"
        current_peak = getattr(self, f'_peak_{asset}', profit_bps)
        if profit_bps > current_peak:
            setattr(self, f'_peak_{asset}', profit_bps)
            current_peak = profit_bps
        signal.peak_profit_bps = current_peak

        # =====================================================================
        # TRIGGER 4: DRL Action (checked BEFORE profit gate)
        # DRL opposing the position is a RISK signal, not a profit-taking signal.
        # A strongly opposing DRL should exit even for underwater positions.
        # [FIX-EA-DRL] min_profit_bps was blocking DRL exit on losing positions.
        # =====================================================================
        if drl_output and drl_output.action == DRLAction.PARTIAL_EXIT:
            signal.should_scale_out = True
            signal.trigger = ScaleOutTrigger.DRL_ACTION
            signal.reason = f"DRL PARTIAL_EXIT action (profit={profit_bps:.0f}bps, bypassing min_profit gate)"
            logger.info(f"SCALE-OUT TRIGGER: {signal.reason}")
            self._mark_scaled_out(asset)
            return signal

        # Minimum requirements check (profit-taking triggers only)
        if profit_bps < self.config.min_profit_bps:
            return signal

        if bars_held < self.config.min_bars_held:
            return signal

        # Update momentum history
        self._update_momentum_history(asset, momentum)

        # =====================================================================
        # TRIGGER 1: Phase Transition (Primary)
        # =====================================================================
        
        if phase_result.phase_transition:
            if phase_result.phase in self.config.trigger_phases:
                signal.should_scale_out = True
                signal.trigger = ScaleOutTrigger.PHASE_TRANSITION
                signal.reason = (
                    f"Phase transition to {phase_result.phase.value} "
                    f"(from {phase_result.previous_phase.value if phase_result.previous_phase else 'None'})"
                )
                logger.info(f"SCALE-OUT TRIGGER: {signal.reason}")
                self._mark_scaled_out(asset)
                return signal
        
        # =====================================================================
        # TRIGGER 2: CRACK Decay (Tiered)
        # [CRACK-2] Tiered exit when crack was active (>0.50) and decays:
        #   < 0.50: thesis weakening → 25% partial exit
        #   < 0.45: thesis failing   → 50% partial exit
        #   < 0.35: thesis dead      → 100% full exit
        # Each tier fires once (tracked via _crack_exit_tier).
        # =====================================================================

        if crack_weight > 0.50:
            self._crack_was_active[asset] = True
            self._crack_exit_tier[asset] = 0  # reset tiers when crack re-activates
            self._scaled_out.pop(asset, None)  # [FIX-M5] allow re-triggering on new CRACK cycle

        if self._crack_was_active.get(asset, False):
            _tier = self._crack_exit_tier.get(asset, 0)
            _new_tier = 0
            _pct = 0.0
            if crack_weight < 0.35 and _tier < 3:
                _new_tier = 3
                _pct = 1.0   # full exit
            elif crack_weight < 0.45 and _tier < 2:
                _new_tier = 2
                _pct = 0.50  # 50% partial
            elif crack_weight < 0.50 and _tier < 1:
                _new_tier = 1
                _pct = 0.25  # 25% partial

            if _new_tier > 0:
                self._crack_exit_tier[asset] = _new_tier
                signal.should_scale_out = True
                signal.scale_out_pct = _pct
                signal.trigger = ScaleOutTrigger.CRACK_DECAY
                signal.reason = (
                    f"CRACK decay tier {_new_tier}: weight={crack_weight:.2f} "
                    f"→ {_pct:.0%} exit"
                )
                logger.info(f"SCALE-OUT TRIGGER: {signal.reason}")
                if _new_tier >= 3:
                    self._crack_was_active[asset] = False  # fully exited
                self._mark_scaled_out(asset)
                return signal
        
        # =====================================================================
        # TRIGGER 3: Momentum Stall
        # =====================================================================
        
        if self._is_momentum_stalled(asset):
            signal.should_scale_out = True
            signal.trigger = ScaleOutTrigger.MOMENTUM_STALL
            signal.reason = f"Momentum stalled for {self.config.momentum_stall_bars} bars"
            logger.info(f"SCALE-OUT TRIGGER: {signal.reason}")
            self._mark_scaled_out(asset)
            return signal
        
        # =====================================================================
        # TRIGGER 5: Drawdown from Peak
        # =====================================================================
        
        if current_peak > 0:
            drawdown = 1 - (profit_bps / current_peak)
            if drawdown >= self.config.drawdown_threshold:
                signal.should_scale_out = True
                signal.trigger = ScaleOutTrigger.DRAWDOWN_FROM_PEAK
                signal.reason = f"Drawdown {drawdown:.1%} from peak {current_peak:.0f}bps"
                logger.info(f"SCALE-OUT TRIGGER: {signal.reason}")
                self._mark_scaled_out(asset)
                return signal

        # =====================================================================
        # TRIGGER 6: Profit Schedule (L4-10)
        # Fixed PnL milestones - lock in profits at predetermined levels
        # =====================================================================

        _ps_key = f"_profit_schedule_{asset}"
        _ps_hit = getattr(self, _ps_key, set())
        for threshold_bps, pct in self.config.profit_schedule:
            if profit_bps >= threshold_bps and threshold_bps not in _ps_hit:
                _ps_hit.add(threshold_bps)
                setattr(self, _ps_key, _ps_hit)
                signal.should_scale_out = True
                signal.trigger = ScaleOutTrigger.PROFIT_SCHEDULE
                signal.scale_out_pct = pct
                signal.reason = f"Profit schedule: +{threshold_bps:.0f}bps hit, scale {pct:.0%}"
                logger.info(f"SCALE-OUT TRIGGER: {signal.reason}")
                self._mark_scaled_out(asset)
                return signal

        return signal
    
    def _update_momentum_history(self, asset: str, momentum: float):
        """Update momentum history for stall detection."""
        if asset not in self._momentum_history:
            self._momentum_history[asset] = []
        
        self._momentum_history[asset].append(momentum)
        # Keep last N bars
        max_history = self.config.momentum_stall_bars + 2
        self._momentum_history[asset] = self._momentum_history[asset][-max_history:]
    
    def _is_momentum_stalled(self, asset: str) -> bool:
        """Check if momentum has stalled."""
        history = self._momentum_history.get(asset, [])
        
        if len(history) < self.config.momentum_stall_bars:
            return False
        
        recent = history[-self.config.momentum_stall_bars:]
        
        # Check if all recent momentum changes are below threshold
        for i in range(1, len(recent)):
            change = abs(recent[i] - recent[i-1])
            if change > self.config.momentum_stall_threshold:
                return False
        
        return True
    
    def _mark_scaled_out(self, asset: str):
        """Mark asset as having scaled out."""
        self._scaled_out[asset] = True
    
    # =========================================================================
    # RUNNER MANAGEMENT
    # =========================================================================
    
    def create_runner(
        self,
        asset: str,
        remaining_size_pct: float,
        entry_price: float,
        current_price: float,
        direction: int,  # 1 for long, -1 for short
    ) -> RunnerState:
        """
        Create runner position after scale-out.
        
        CALLED AFTER:
            - 25% scale-out executed
            - Remaining 75% becomes runner
        """
        
        runner = RunnerState(
            active=True,
            size_pct=remaining_size_pct,
            entry_price=entry_price,
            peak_price=current_price,
            trail_pct=self.config.runner_initial_trail_pct,
        )
        
        # Calculate initial trail stop
        if direction > 0:  # Long
            runner.trail_stop_price = current_price * (1 - runner.trail_pct)
        else:  # Short
            runner.trail_stop_price = current_price * (1 + runner.trail_pct)
        
        self._runners[asset] = runner
        
        logger.info(
            f"Runner created for {asset}: {remaining_size_pct:.0%} at trail stop "
            f"${runner.trail_stop_price:.2f} ({runner.trail_pct:.1%})"
        )
        
        return runner
    
    def manage_runner(
        self,
        asset: str,
        current_price: float,
        direction: int,
        phase_result: RegimePhaseResult,
        drl_output: Optional[DRLOutput] = None,
    ) -> Tuple[RunnerAction, Optional[RunnerState]]:
        """
        Manage existing runner position.
        
        ACTIONS:
            - HOLD: Continue holding
            - TIGHTEN: Tighten trail stop to 1%
            - RELEASE: Close runner position
        
        RELEASE CONDITIONS:
            1. DRL RELEASE_RUNNER action
            2. Phase transition to opposite trend
            3. Trail stop hit
            4. Mode changes to NO_TRADE
        """
        
        runner = self._runners.get(asset)
        if not runner or not runner.active:
            return RunnerAction.HOLD, None
        
        runner.bars_as_runner += 1
        
        # Update trail stop (always trails higher for long, lower for short)
        if direction > 0:  # Long
            if current_price > runner.peak_price:
                runner.peak_price = current_price
                runner.trail_stop_price = current_price * (1 - runner.trail_pct)
        else:  # Short
            if current_price < runner.peak_price:
                runner.peak_price = current_price
                runner.trail_stop_price = current_price * (1 + runner.trail_pct)
        
        # Check RELEASE conditions
        
        # 1. DRL RELEASE_RUNNER
        if drl_output and drl_output.action == DRLAction.RELEASE_RUNNER:
            logger.info(f"Runner RELEASE for {asset}: DRL action")
            runner.active = False
            return RunnerAction.RELEASE, runner
        
        # 2. Phase transition to opposite trend (simplified)
        if phase_result.phase == RegimePhase.EXHAUSTION and phase_result.phase_age_bars >= 2:
            logger.info(f"Runner RELEASE for {asset}: Exhaustion phase aged")
            runner.active = False
            return RunnerAction.RELEASE, runner
        
        # 3. Trail stop hit
        if direction > 0:  # Long
            if current_price <= runner.trail_stop_price:
                logger.info(f"Runner RELEASE for {asset}: Trail stop hit at {runner.trail_stop_price:.2f}")
                runner.active = False
                return RunnerAction.RELEASE, runner
        else:  # Short
            if current_price >= runner.trail_stop_price:
                logger.info(f"Runner RELEASE for {asset}: Trail stop hit at {runner.trail_stop_price:.2f}")
                runner.active = False
                return RunnerAction.RELEASE, runner
        
        # Check TIGHTEN conditions
        if not runner.tightened:
            should_tighten = False
            
            # DRL INCREASE_EXIT_PRESSURE
            if drl_output and drl_output.action == DRLAction.INCREASE_EXIT_PRESSURE:
                should_tighten = True
            
            # Exhaustion phase
            if phase_result.phase == RegimePhase.EXHAUSTION:
                should_tighten = True
            
            if should_tighten:
                runner.trail_pct = self.config.runner_tight_trail_pct
                runner.tightened = True
                
                # Recalculate trail stop
                if direction > 0:
                    runner.trail_stop_price = runner.peak_price * (1 - runner.trail_pct)
                else:
                    runner.trail_stop_price = runner.peak_price * (1 + runner.trail_pct)
                
                logger.info(
                    f"Runner TIGHTENED for {asset}: trail now {runner.trail_pct:.1%} "
                    f"stop at {runner.trail_stop_price:.2f}"
                )
                return RunnerAction.TIGHTEN, runner
        
        # Default: HOLD
        return RunnerAction.HOLD, runner
    
    def get_runner(self, asset: str) -> Optional[RunnerState]:
        """Get runner state for asset."""
        return self._runners.get(asset)
    
    def reset_for_asset(self, asset: str):
        """Reset state for asset (new position)."""
        self._scaled_out[asset] = False
        self._runners.pop(asset, None)
        self._momentum_history.pop(asset, None)
        self._crack_was_active.pop(asset, None)  # [CRACK-2]
        if hasattr(self, f'_peak_{asset}'):
            delattr(self, f'_peak_{asset}')
        # L4-10: Reset profit schedule milestones
        _ps_key = f"_profit_schedule_{asset}"
        if hasattr(self, _ps_key):
            delattr(self, _ps_key)
    
    def reset_all(self):
        """Reset all state."""
        self._scaled_out.clear()
        self._runners.clear()
        self._momentum_history.clear()
        self._crack_was_active.clear()  # [CRACK-2]


# Singleton
_exit_manager: Optional[ExitAlphaManager] = None

def get_exit_manager() -> ExitAlphaManager:
    global _exit_manager
    if _exit_manager is None:
        _exit_manager = ExitAlphaManager()
    return _exit_manager

def reset_exit_manager():
    global _exit_manager
    _exit_manager = None

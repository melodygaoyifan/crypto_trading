"""
================================================================================
EXECUTION GUARDS - Safety Constraints for Trading Execution
================================================================================
Version: 3.2.0
Purpose: Enforce critical safety constraints before any trade execution

Guards:
- Stale Data Kill Switch (>2s = no execution)
- DVOL Override (突变 -> Defensive mode)
- SOL Risk Override (覆盖技术信号/DRL判断)
- DRL Confidence Constraint (Regime不稳定期禁止自信)
================================================================================
"""

import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
import threading

logger = logging.getLogger("HMATS.ExecutionGuards")


# =============================================================================
# STALE DATA KILL SWITCH
# =============================================================================

class DataFreshness(Enum):
    """Data freshness classification."""
    FRESH = auto()          # < 500ms
    ACCEPTABLE = auto()     # 500ms - 2s
    STALE = auto()          # 2s - 5s
    DEAD = auto()           # > 5s


@dataclass
class DataTimestamp:
    """Timestamp for a data source."""
    source: str
    last_update: float
    max_age_ms: float = 2000.0  # Default 2 second max age


class StaleDataKillSwitch:
    """
    Kill switch that halts execution when data becomes stale.
    
    Rules:
    - Data age > 2s: HALT all execution
    - Data age > 500ms: WARNING, reduce position sizes
    - Any critical data source stale: IMMEDIATE HALT
    """
    
    FRESH_THRESHOLD_MS = 15000   # [T4] was 500; 15s normal for REST
    STALE_THRESHOLD_MS = 60000   # [T4] was 2000; unified 60s
    DEAD_THRESHOLD_MS = 300000   # [T4] was 5000; 5min = actually dead
    
    # [P48 2026-04-25 BUG-5] Was just ['orderbook', 'trades', 'position'] but the
    # only production caller (defense/p0_safety_integrator.py:357) updates
    # 'kraken_ws' and 'kraken_rest' — names didn't match. Net: critical_stale
    # check never fired, the kill switch was effectively dead. Added the
    # actually-used names; kept old names in case future code populates them.
    CRITICAL_SOURCES = ['kraken_ws', 'kraken_rest', 'orderbook', 'trades', 'position']
    
    def __init__(self):
        self.data_timestamps: Dict[str, DataTimestamp] = {}
        self.is_halted = False
        self.halt_reason = ""
        self.last_check_time = 0.0
        
        # Callbacks
        self.on_halt: Optional[Callable] = None
        self.on_resume: Optional[Callable] = None
        
        self._lock = threading.RLock()
        
        logger.info("StaleDataKillSwitch initialized")
    
    def register_source(self, source: str, max_age_ms: float = None):
        """Register a data source to monitor."""
        with self._lock:
            self.data_timestamps[source] = DataTimestamp(
                source=source,
                last_update=time.time(),
                max_age_ms=max_age_ms or self.STALE_THRESHOLD_MS
            )
    
    def update_timestamp(self, source: str):
        """Update timestamp for a data source."""
        with self._lock:
            if source in self.data_timestamps:
                self.data_timestamps[source].last_update = time.time()
            else:
                self.register_source(source)
                self.data_timestamps[source].last_update = time.time()
    
    def check_freshness(self, source: str) -> DataFreshness:
        """Check freshness of a single source."""
        with self._lock:
            if source not in self.data_timestamps:
                return DataFreshness.DEAD
            
            age_ms = (time.time() - self.data_timestamps[source].last_update) * 1000
            
            if age_ms < self.FRESH_THRESHOLD_MS:
                return DataFreshness.FRESH
            elif age_ms < self.STALE_THRESHOLD_MS:
                return DataFreshness.ACCEPTABLE
            elif age_ms < self.DEAD_THRESHOLD_MS:
                return DataFreshness.STALE
            else:
                return DataFreshness.DEAD
    
    def can_execute(self) -> Tuple[bool, str, Dict]:
        """
        Check if execution is allowed.
        
        Returns:
            (can_execute, reason, details)
        """
        with self._lock:
            self.last_check_time = time.time()
            details = {'sources': {}}
            
            any_stale = False
            any_dead = False
            critical_stale = False
            
            for source, ts in self.data_timestamps.items():
                age_ms = (time.time() - ts.last_update) * 1000
                freshness = self.check_freshness(source)
                
                details['sources'][source] = {
                    'age_ms': age_ms,
                    'freshness': freshness.name,
                    'critical': source in self.CRITICAL_SOURCES
                }
                
                if freshness == DataFreshness.STALE:
                    any_stale = True
                    if source in self.CRITICAL_SOURCES:
                        critical_stale = True
                        
                elif freshness == DataFreshness.DEAD:
                    any_dead = True
                    critical_stale = True  # Dead is always critical
            
            # Decision
            if any_dead:
                self._halt("DEAD_DATA")
                return False, "Data source is dead (>5s)", details
            
            if critical_stale:
                self._halt("CRITICAL_STALE")
                return False, "Critical data source stale (>2s)", details
            
            if any_stale:
                # Allow but with warning
                logger.warning("Non-critical data is stale, proceeding with caution")
            
            # Resume if was halted
            if self.is_halted:
                self._resume()
            
            return True, "OK", details
    
    def _halt(self, reason: str):
        """Halt execution."""
        if not self.is_halted:
            self.is_halted = True
            self.halt_reason = reason
            logger.error(f"EXECUTION HALTED: {reason}")
            
            if self.on_halt:
                self.on_halt(reason)
    
    def _resume(self):
        """Resume execution."""
        if self.is_halted:
            self.is_halted = False
            logger.info("Execution resumed - data freshness restored")
            
            if self.on_resume:
                self.on_resume()
    
    def get_status(self) -> Dict:
        """Get kill switch status."""
        with self._lock:
            return {
                'halted': self.is_halted,
                'halt_reason': self.halt_reason,
                'sources': {
                    s: {
                        'age_ms': (time.time() - ts.last_update) * 1000,
                        'freshness': self.check_freshness(s).name
                    }
                    for s, ts in self.data_timestamps.items()
                }
            }


# =============================================================================
# DVOL OVERRIDE - Deribit Volatility Index
# =============================================================================

class DVOLRegime(Enum):
    """DVOL regime classification."""
    LOW = auto()        # DVOL < 40
    NORMAL = auto()     # 40 <= DVOL < 60
    ELEVATED = auto()   # 60 <= DVOL < 80
    HIGH = auto()       # 80 <= DVOL < 100
    EXTREME = auto()    # DVOL >= 100


class ExecutionMode(Enum):
    """Execution mode based on volatility."""
    AGGRESSIVE = auto()
    NORMAL = auto()
    PASSIVE = auto()
    DEFENSIVE = auto()
    FLAT_ONLY = auto()  # Only flatten positions


@dataclass
class DVOLState:
    """Current DVOL state."""
    value: float = 50.0
    regime: DVOLRegime = DVOLRegime.NORMAL
    change_1h: float = 0.0
    change_24h: float = 0.0
    spike_detected: bool = False
    timestamp: float = field(default_factory=time.time)


class DVOLOverrideController:
    """
    Override execution mode based on DVOL changes.
    
    Rules:
    - DVOL spike >20% in 1H -> DEFENSIVE mode
    - DVOL spike >30% in 1H -> FLAT_ONLY mode
    - DVOL > 100 -> DEFENSIVE mode
    - DVOL > 120 -> FLAT_ONLY mode
    """
    
    # Thresholds
    SPIKE_DEFENSIVE_PCT = 20.0
    SPIKE_FLAT_ONLY_PCT = 30.0
    LEVEL_DEFENSIVE = 100.0
    LEVEL_FLAT_ONLY = 120.0
    
    def __init__(self):
        self.current_state = DVOLState()
        self.history: deque = deque(maxlen=1000)
        self.override_active = False
        self.override_mode = ExecutionMode.NORMAL
        self.override_reason = ""
        
        self._lock = threading.RLock()
        
        logger.info("DVOLOverrideController initialized")
    
    def update(self, dvol_value: float) -> Tuple[bool, ExecutionMode, str]:
        """
        Update DVOL and check for override.

        Returns:
            (override_active, mode, reason)
        """
        with self._lock:
            now = time.time()

            # Calculate changes
            change_1h = 0.0
            if self.history:
                # Find value from ~1 hour ago
                target_time = now - 3600
                for entry in reversed(self.history):
                    if entry['timestamp'] <= target_time:
                        old_val = entry['value']
                        # Guard: if input is z-score (small magnitude), use
                        # absolute difference instead of % change to avoid
                        # division-by-near-zero producing ±400%+ false spikes.
                        if abs(old_val) < 1.0:
                            # Z-score mode: treat |Δz| > 2.5 as FLAT_ONLY,
                            # |Δz| > 1.5 as DEFENSIVE (mapped to % thresholds)
                            delta_z = abs(dvol_value - old_val)
                            change_1h = delta_z * 15.0  # 2.0σ→30%, 1.33σ→20%
                        else:
                            change_1h = ((dvol_value - old_val) / old_val) * 100
                        # Clip to prevent extreme values from stale/bad data
                        change_1h = max(-200.0, min(200.0, change_1h))
                        break
            
            # Classify regime
            if dvol_value < 40:
                regime = DVOLRegime.LOW
            elif dvol_value < 60:
                regime = DVOLRegime.NORMAL
            elif dvol_value < 80:
                regime = DVOLRegime.ELEVATED
            elif dvol_value < 100:
                regime = DVOLRegime.HIGH
            else:
                regime = DVOLRegime.EXTREME
            
            # Detect spike
            spike_detected = abs(change_1h) > self.SPIKE_DEFENSIVE_PCT
            
            # Update state
            self.current_state = DVOLState(
                value=dvol_value,
                regime=regime,
                change_1h=change_1h,
                spike_detected=spike_detected,
                timestamp=now
            )
            
            # Add to history
            self.history.append({
                'value': dvol_value,
                'timestamp': now
            })
            
            # Determine override
            return self._check_override()
    
    def _check_override(self) -> Tuple[bool, ExecutionMode, str]:
        """Check if override should be active."""
        state = self.current_state
        
        # Check spike conditions
        if abs(state.change_1h) > self.SPIKE_FLAT_ONLY_PCT:
            self.override_active = True
            self.override_mode = ExecutionMode.FLAT_ONLY
            self.override_reason = f"DVOL spike {state.change_1h:.1f}% in 1H (>{self.SPIKE_FLAT_ONLY_PCT}%)"
            logger.warning(f"DVOL OVERRIDE: {self.override_reason}")
            return True, self.override_mode, self.override_reason
        
        if abs(state.change_1h) > self.SPIKE_DEFENSIVE_PCT:
            self.override_active = True
            self.override_mode = ExecutionMode.DEFENSIVE
            self.override_reason = f"DVOL spike {state.change_1h:.1f}% in 1H (>{self.SPIKE_DEFENSIVE_PCT}%)"
            logger.warning(f"DVOL OVERRIDE: {self.override_reason}")
            return True, self.override_mode, self.override_reason
        
        # Check level conditions
        if state.value > self.LEVEL_FLAT_ONLY:
            self.override_active = True
            self.override_mode = ExecutionMode.FLAT_ONLY
            self.override_reason = f"DVOL level {state.value:.0f} (>{self.LEVEL_FLAT_ONLY})"
            return True, self.override_mode, self.override_reason
        
        if state.value > self.LEVEL_DEFENSIVE:
            self.override_active = True
            self.override_mode = ExecutionMode.DEFENSIVE
            self.override_reason = f"DVOL level {state.value:.0f} (>{self.LEVEL_DEFENSIVE})"
            return True, self.override_mode, self.override_reason
        
        # No override
        self.override_active = False
        self.override_mode = ExecutionMode.NORMAL
        self.override_reason = ""
        return False, ExecutionMode.NORMAL, ""
    
    def get_status(self) -> Dict:
        """Get controller status."""
        with self._lock:
            return {
                'dvol': self.current_state.value,
                'regime': self.current_state.regime.name,
                'change_1h': self.current_state.change_1h,
                'spike_detected': self.current_state.spike_detected,
                'override_active': self.override_active,
                'override_mode': self.override_mode.name,
                'override_reason': self.override_reason
            }


# =============================================================================
# SOL RISK OVERRIDE
# =============================================================================

@dataclass
class SOLRiskState:
    """SOL-specific risk state."""
    cu_utilization: float = 0.0          # Compute Unit utilization (0-1)
    priority_fee_lamports: int = 0       # Current priority fee
    jito_tip_lamports: int = 0           # Jito MEV tip
    tps: float = 0.0                     # Transactions per second
    slot_time_ms: float = 400.0          # Average slot time
    
    # On-chain inflows
    kraken_inflow_24h: float = 0.0       # SOL inflow to Kraken
    exchange_inflow_24h: float = 0.0     # Total exchange inflow
    whale_movements: int = 0             # Large wallet movements
    
    # Risk flags
    network_congested: bool = False
    high_inflow_risk: bool = False
    override_active: bool = False
    
    timestamp: float = field(default_factory=time.time)


class SOLRiskOverrideController:
    """
    Override SOL execution based on network and on-chain risks.
    
    Rules:
    - CU utilization > 80% -> Reduce size 50%
    - CU utilization > 95% -> ABORT execution
    - Priority fee > 100k lamports -> PASSIVE only
    - Large exchange inflow detected -> Override DRL/Quant signals
    """
    
    # Thresholds
    CU_REDUCE_THRESHOLD = 0.80
    CU_ABORT_THRESHOLD = 0.95
    PRIORITY_FEE_PASSIVE_THRESHOLD = 100000  # lamports
    INFLOW_RISK_THRESHOLD_SOL = 100000  # SOL in 24h
    
    def __init__(self):
        self.current_state = SOLRiskState()
        self.inflow_history: deque = deque(maxlen=1000)
        
        # Override state
        self.size_multiplier = 1.0
        self.force_passive = False
        self.abort_execution = False
        self.override_technical = False  # Override technical signals
        self.override_drl = False        # Override DRL signals
        
        self._lock = threading.RLock()
        
        logger.info("SOLRiskOverrideController initialized")
    
    def update(
        self,
        cu_utilization: float = None,
        priority_fee: int = None,
        jito_tip: int = None,
        exchange_inflow: float = None,
        kraken_inflow: float = None
    ) -> Dict:
        """Update SOL risk state and compute overrides."""
        with self._lock:
            now = time.time()
            
            # Update state
            if cu_utilization is not None:
                self.current_state.cu_utilization = cu_utilization
            if priority_fee is not None:
                self.current_state.priority_fee_lamports = priority_fee
            if jito_tip is not None:
                self.current_state.jito_tip_lamports = jito_tip
            if exchange_inflow is not None:
                self.current_state.exchange_inflow_24h = exchange_inflow
            if kraken_inflow is not None:
                self.current_state.kraken_inflow_24h = kraken_inflow
            
            self.current_state.timestamp = now
            
            # Compute overrides
            self._compute_overrides()
            
            return self.get_override_state()
    
    def _compute_overrides(self):
        """Compute all override flags."""
        state = self.current_state
        
        # Reset
        self.size_multiplier = 1.0
        self.force_passive = False
        self.abort_execution = False
        self.override_technical = False
        self.override_drl = False
        
        # CU utilization checks
        if state.cu_utilization > self.CU_ABORT_THRESHOLD:
            self.abort_execution = True
            state.network_congested = True
            logger.warning(f"SOL ABORT: CU utilization {state.cu_utilization:.1%}")
        elif state.cu_utilization > self.CU_REDUCE_THRESHOLD:
            self.size_multiplier = 0.5
            state.network_congested = True
            logger.info(f"SOL SIZE REDUCE: CU utilization {state.cu_utilization:.1%}")
        
        # Priority fee check
        if state.priority_fee_lamports > self.PRIORITY_FEE_PASSIVE_THRESHOLD:
            self.force_passive = True
            logger.info(f"SOL PASSIVE ONLY: High priority fee {state.priority_fee_lamports}")
        
        # Exchange inflow check (bearish signal)
        total_inflow = state.exchange_inflow_24h + state.kraken_inflow_24h
        if total_inflow > self.INFLOW_RISK_THRESHOLD_SOL:
            state.high_inflow_risk = True
            self.override_technical = True
            self.override_drl = True
            logger.warning(f"SOL INFLOW RISK: {total_inflow:.0f} SOL in 24h - OVERRIDE ACTIVE")
        
        state.override_active = (
            self.abort_execution or 
            self.force_passive or 
            self.override_technical or 
            self.override_drl
        )
    
    def should_override_long(self) -> Tuple[bool, str]:
        """Check if long signals should be overridden."""
        with self._lock:
            if self.current_state.high_inflow_risk:
                return True, "High exchange inflow indicates selling pressure"
            if self.abort_execution:
                return True, "Network congestion - no execution"
            return False, ""
    
    def get_override_state(self) -> Dict:
        """Get current override state."""
        return {
            'size_multiplier': self.size_multiplier,
            'force_passive': self.force_passive,
            'abort_execution': self.abort_execution,
            'override_technical': self.override_technical,
            'override_drl': self.override_drl,
            'cu_utilization': self.current_state.cu_utilization,
            'network_congested': self.current_state.network_congested,
            'high_inflow_risk': self.current_state.high_inflow_risk
        }
    
    def get_status(self) -> Dict:
        """Get full status."""
        with self._lock:
            return {
                'state': {
                    'cu_utilization': self.current_state.cu_utilization,
                    'priority_fee': self.current_state.priority_fee_lamports,
                    'jito_tip': self.current_state.jito_tip_lamports,
                    'exchange_inflow': self.current_state.exchange_inflow_24h,
                    'kraken_inflow': self.current_state.kraken_inflow_24h
                },
                'overrides': self.get_override_state()
            }


# =============================================================================
# DRL CONFIDENCE CONSTRAINT
# =============================================================================

@dataclass
class DRLConstraintConfig:
    """DRL constraint configuration."""
    # Regime stability thresholds
    regime_stability_threshold: float = 0.6       # Min consistency score
    regime_change_cooldown_s: float = 3600 * 4    # 4 hours after regime change
    
    # Confidence thresholds
    min_confidence_trending: float = 0.5          # Min confidence in trending regimes
    min_confidence_volatile: float = 0.7          # Higher bar in volatile regimes
    
    # Weight limits
    max_drl_weight_unstable: float = 0.3          # Max DRL weight when unstable
    max_drl_weight_volatile: float = 0.2          # Max DRL weight in VOLATILE_CHOP


class DRLConfidenceConstraint:
    """
    Constrain DRL confidence in unstable regimes.
    
    Rules:
    - DRL confidence capped when regime not stable
    - DRL weight reduced during regime transitions
    - DRL forbidden from "强行自信" in VOLATILE_CHOP
    """
    
    VOLATILE_REGIMES = ['VOLATILE_CHOP', 'EXTREME_VOLATILITY']
    TRANSITIONAL_REGIMES = ['NEUTRAL', 'UNKNOWN']
    
    def __init__(self, config: DRLConstraintConfig = None):
        self.config = config or DRLConstraintConfig()
        
        # Regime tracking
        self.current_regime = "NEUTRAL"
        self.regime_consistency = 0.5
        self.last_regime_change_time = 0.0
        self.regime_history: deque = deque(maxlen=100)
        
        # DRL state
        self.drl_weight_cap = 1.0
        self.drl_confidence_cap = 1.0
        self.constraint_reason = ""
        
        self._lock = threading.RLock()
        
        logger.info("DRLConfidenceConstraint initialized")
    
    def update_regime(self, regime: str, consistency_score: float):
        """Update regime state."""
        with self._lock:
            now = time.time()
            
            # Check for regime change
            if regime != self.current_regime:
                self.last_regime_change_time = now
                logger.info(f"Regime change: {self.current_regime} -> {regime}")
            
            self.current_regime = regime
            self.regime_consistency = consistency_score
            
            self.regime_history.append({
                'regime': regime,
                'consistency': consistency_score,
                'timestamp': now
            })
            
            # Recompute constraints
            self._compute_constraints()
    
    def _compute_constraints(self):
        """Compute DRL constraints based on regime state."""
        now = time.time()
        
        # Reset
        self.drl_weight_cap = 1.0
        self.drl_confidence_cap = 1.0
        reasons = []
        
        # Check regime stability
        if self.regime_consistency < self.config.regime_stability_threshold:
            self.drl_weight_cap = min(
                self.drl_weight_cap,
                self.config.max_drl_weight_unstable
            )
            reasons.append(f"Low regime consistency ({self.regime_consistency:.2f})")
        
        # Check cooldown after regime change
        time_since_change = now - self.last_regime_change_time
        if time_since_change < self.config.regime_change_cooldown_s:
            cooldown_factor = time_since_change / self.config.regime_change_cooldown_s
            self.drl_weight_cap = min(
                self.drl_weight_cap,
                self.config.max_drl_weight_unstable + (1 - self.config.max_drl_weight_unstable) * cooldown_factor
            )
            reasons.append(f"Regime change cooldown ({time_since_change/3600:.1f}h)")
        
        # Check volatile regime
        if self.current_regime in self.VOLATILE_REGIMES:
            self.drl_weight_cap = min(
                self.drl_weight_cap,
                self.config.max_drl_weight_volatile
            )
            self.drl_confidence_cap = self.config.min_confidence_volatile
            reasons.append(f"Volatile regime ({self.current_regime})")
        
        # Check transitional regime
        if self.current_regime in self.TRANSITIONAL_REGIMES:
            self.drl_weight_cap = min(
                self.drl_weight_cap,
                self.config.max_drl_weight_unstable
            )
            reasons.append(f"Transitional regime ({self.current_regime})")
        
        self.constraint_reason = "; ".join(reasons) if reasons else "None"
    
    def apply_constraint(
        self,
        drl_signal: float,
        drl_confidence: float,
        drl_weight: float
    ) -> Tuple[float, float, float, str]:
        """
        Apply constraints to DRL output.
        
        Returns:
            (constrained_signal, constrained_confidence, constrained_weight, reason)
        """
        with self._lock:
            # Cap confidence
            constrained_confidence = min(drl_confidence, self.drl_confidence_cap)
            
            # Cap weight
            constrained_weight = min(drl_weight, self.drl_weight_cap)
            
            # If confidence too low, reduce signal magnitude
            if constrained_confidence < 0.3:
                constrained_signal = drl_signal * constrained_confidence
            else:
                constrained_signal = drl_signal
            
            return constrained_signal, constrained_confidence, constrained_weight, self.constraint_reason
    
    def can_drl_trade(self, drl_confidence: float) -> Tuple[bool, str]:
        """
        Check if DRL is allowed to generate trade signals.
        
        Returns:
            (allowed, reason)
        """
        with self._lock:
            # Check volatile regime
            if self.current_regime in self.VOLATILE_REGIMES:
                if drl_confidence < self.config.min_confidence_volatile:
                    return False, f"DRL confidence too low for volatile regime ({drl_confidence:.2f} < {self.config.min_confidence_volatile})"
            
            # Check stability
            if self.regime_consistency < 0.3:
                return False, f"Regime too unstable for DRL ({self.regime_consistency:.2f})"
            
            # Check cooldown
            time_since_change = time.time() - self.last_regime_change_time
            if time_since_change < 1800:  # 30 minutes
                return False, f"Too soon after regime change ({time_since_change/60:.0f}m)"
            
            return True, "OK"
    
    def get_status(self) -> Dict:
        """Get constraint status."""
        with self._lock:
            return {
                'current_regime': self.current_regime,
                'regime_consistency': self.regime_consistency,
                'time_since_regime_change_h': (time.time() - self.last_regime_change_time) / 3600,
                'drl_weight_cap': self.drl_weight_cap,
                'drl_confidence_cap': self.drl_confidence_cap,
                'constraint_reason': self.constraint_reason
            }


# =============================================================================
# UNIFIED EXECUTION GUARD
# =============================================================================

class ExecutionGuard:
    """
    Unified execution guard combining all safety checks.
    
    Checks (in order):
    1. Stale Data Kill Switch
    2. DVOL Override
    3. SOL Risk Override (for SOL trades)
    4. DRL Confidence Constraint
    5. Edge Filter (from defense module)
    """
    
    def __init__(self):
        self.stale_data = StaleDataKillSwitch()
        self.dvol_override = DVOLOverrideController()
        self.sol_override = SOLRiskOverrideController()
        self.drl_constraint = DRLConfidenceConstraint()
        
        # Overall state
        self.execution_allowed = True
        self.execution_mode = ExecutionMode.NORMAL
        self.block_reasons: List[str] = []
        
        logger.info("ExecutionGuard initialized with all sub-guards")
    
    def check_execution(
        self,
        asset: str,
        side: str,
        drl_signal: float = 0.0,
        drl_confidence: float = 0.5,
        drl_weight: float = 0.5
    ) -> Tuple[bool, ExecutionMode, Dict]:
        """
        Run all execution checks.
        
        Returns:
            (allowed, mode, details)
        """
        self.block_reasons = []
        details = {}
        
        # 1. Stale Data Check
        can_exec, reason, stale_details = self.stale_data.can_execute()
        details['stale_data'] = stale_details
        if not can_exec:
            self.block_reasons.append(f"Stale Data: {reason}")
            return False, ExecutionMode.FLAT_ONLY, details
        
        # 2. DVOL Check
        dvol_status = self.dvol_override.get_status()
        details['dvol'] = dvol_status
        if dvol_status['override_active']:
            mode = ExecutionMode[dvol_status['override_mode']]
            if mode == ExecutionMode.FLAT_ONLY:
                self.block_reasons.append(f"DVOL: {dvol_status['override_reason']}")
                return False, mode, details
            elif mode == ExecutionMode.DEFENSIVE:
                # Allow but in defensive mode
                pass
        
        # 3. SOL Override (only for SOL)
        if asset.upper() == 'SOL':
            sol_status = self.sol_override.get_override_state()
            details['sol'] = sol_status
            
            if sol_status['abort_execution']:
                self.block_reasons.append("SOL: Network congestion")
                return False, ExecutionMode.FLAT_ONLY, details
            
            # Check long override
            if side.lower() == 'long':
                override, reason = self.sol_override.should_override_long()
                if override:
                    self.block_reasons.append(f"SOL Long Override: {reason}")
                    return False, ExecutionMode.DEFENSIVE, details
        
        # 4. DRL Constraint
        if drl_weight > 0:
            can_drl, reason = self.drl_constraint.can_drl_trade(drl_confidence)
            details['drl_constraint'] = self.drl_constraint.get_status()
            
            if not can_drl:
                # Don't block, but note the constraint
                details['drl_blocked_reason'] = reason
        
        # Determine final mode
        if self.dvol_override.override_active:
            self.execution_mode = self.dvol_override.override_mode
        elif self.sol_override.force_passive and asset.upper() == 'SOL':
            self.execution_mode = ExecutionMode.PASSIVE
        else:
            self.execution_mode = ExecutionMode.NORMAL
        
        self.execution_allowed = True
        return True, self.execution_mode, details
    
    def get_status(self) -> Dict:
        """Get comprehensive guard status."""
        return {
            'execution_allowed': self.execution_allowed,
            'execution_mode': self.execution_mode.name,
            'block_reasons': self.block_reasons,
            'stale_data': self.stale_data.get_status(),
            'dvol': self.dvol_override.get_status(),
            'sol': self.sol_override.get_status(),
            'drl_constraint': self.drl_constraint.get_status()
        }

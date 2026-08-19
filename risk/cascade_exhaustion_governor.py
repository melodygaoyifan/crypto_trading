"""
================================================================================
HMATS v6.1.3 - Cascade Exhaustion Governor
================================================================================
Purpose: 清算级联削峰，控制清算策略的 T3/T4 加仓阶段

背景:
    系统已有清算级联检测 + Anti-Martingale T1-T4，
    但缺少清算生命周期状态机，在 V 反转中可能满仓暴露。

功能:
    1. 清算级联状态机 (DETECTED -> ACCELERATING -> EXHAUSTING -> STOP)
    2. 只限制加仓阶段 (T3/T4)
    3. 不改变清算信号定义
    4. 不改变 T1/T2 行为

状态机行为:
    状态           | T1 | T2 | T3 | T4
    DETECTED       | ON | ON | OFF | OFF
    ACCELERATING   | ON | ON | ON | OFF
    EXHAUSTING     | OFF | OFF | OFF | OFF (仅 trailing stop)

接线点:
    在 position sizing / scaling 前
    被阻止的加仓必须写入 Shadow Ledger

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class CascadePhase(Enum):
    """Liquidation cascade lifecycle phases."""
    NONE = "NONE"                   # No cascade detected
    DETECTED = "DETECTED"           # Initial detection
    ACCELERATING = "ACCELERATING"   # Cascade intensifying
    EXHAUSTING = "EXHAUSTING"       # Cascade weakening, possible reversal
    STOP = "STOP"                   # Cascade ended


class TrancheLevel(Enum):
    """Anti-Martingale tranche levels."""
    T1 = 1  # Initial position
    T2 = 2  # First add
    T3 = 3  # Second add
    T4 = 4  # Final add (max position)


# Tranche permission matrix by cascade phase
TRANCHE_PERMISSIONS: Dict[CascadePhase, Dict[TrancheLevel, bool]] = {
    CascadePhase.NONE: {
        TrancheLevel.T1: True,
        TrancheLevel.T2: True,
        TrancheLevel.T3: True,
        TrancheLevel.T4: True,
    },
    CascadePhase.DETECTED: {
        TrancheLevel.T1: True,
        TrancheLevel.T2: True,   # R4: Allow first add during early cascade
        TrancheLevel.T3: False,
        TrancheLevel.T4: False,
    },
    CascadePhase.ACCELERATING: {
        TrancheLevel.T1: True,
        TrancheLevel.T2: True,
        TrancheLevel.T3: True,
        TrancheLevel.T4: False,
    },
    CascadePhase.EXHAUSTING: {
        TrancheLevel.T1: False,
        TrancheLevel.T2: False,
        TrancheLevel.T3: False,
        TrancheLevel.T4: False,
    },
    CascadePhase.STOP: {
        TrancheLevel.T1: True,
        TrancheLevel.T2: True,
        TrancheLevel.T3: True,
        TrancheLevel.T4: True,
    },
}


@dataclass
class CascadeMetrics:
    """Metrics for cascade detection."""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Liquidation metrics
    liquidation_volume_1h: float = 0.0
    liquidation_volume_4h: float = 0.0
    liquidation_acceleration: float = 0.0  # Rate of change
    
    # Price metrics
    price_change_1h_pct: float = 0.0
    price_change_4h_pct: float = 0.0
    price_velocity: float = 0.0
    
    # Volume metrics
    volume_spike_ratio: float = 1.0  # Current vs average
    
    # Derived
    cascade_intensity: float = 0.0  # 0-1 scale
    exhaustion_signal: float = 0.0  # 0-1 scale
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "liquidation_volume_1h": self.liquidation_volume_1h,
            "liquidation_volume_4h": self.liquidation_volume_4h,
            "liquidation_acceleration": self.liquidation_acceleration,
            "price_change_1h_pct": self.price_change_1h_pct,
            "price_change_4h_pct": self.price_change_4h_pct,
            "cascade_intensity": self.cascade_intensity,
            "exhaustion_signal": self.exhaustion_signal,
        }


@dataclass
class TrancheCheckResult:
    """Result of tranche permission check."""
    allowed: bool
    phase: CascadePhase
    tranche: TrancheLevel
    reason: str
    cascade_intensity: float
    recommended_action: str  # "PROCEED", "WAIT", "EXIT"
    
    def to_dict(self) -> Dict:
        return {
            "allowed": self.allowed,
            "phase": self.phase.value,
            "tranche": self.tranche.value,
            "reason": self.reason,
            "cascade_intensity": self.cascade_intensity,
            "recommended_action": self.recommended_action,
        }


@dataclass
class CascadeExhaustionConfig:
    """Configuration for cascade governor."""
    # Detection thresholds
    # [P311] ABSOLUTE fallbacks, used only when no per-asset baseline is
    # supplied. A single dollar figure across assets is wrong in BOTH
    # directions and was measured so over 11 days of the derivflow ledger:
    # median 24h liquidations are BTC $79.6M / ETH $37.8M / SOL $6.4M, so
    # $10M/h is ~3x BTC's normal hourly rate (it fired on 7% of ticks) and
    # ~38x SOL's (its maximum EVER observed is $2.09M/h — structurally
    # unreachable). See the multiples below.
    cascade_detect_liq_threshold: float = 10_000_000
    cascade_accelerate_threshold: float = 25_000_000

    # [P311] SELF-NORMALISING thresholds: a multiple of the asset's OWN
    # recent liquidation rate, which is what a cascade actually is —
    # liquidations running far above this market's normal.
    #
    # One multiplier serves all three because the SHAPE of the burst
    # distribution is comparable once normalised: peak-to-baseline over the
    # replay was BTC 8.1x, ETH 12.0x, SOL 7.9x, while the LEVELS differ by
    # 12.5x. That similarity is what licenses a dimensionless constant, and
    # it is the finding that made this fixable at all.
    #
    # [P316] The level is now calibrated on 1,039 bars per asset rather than
    # 85, and the DETECT itself was replayed against forward returns over the
    # same window. At 5x a spike carries NO information (|t| <= 0.93 on all
    # three vs the unconditional mean); at 20x there is a weak, BTC-ONLY hint
    # (-1.09% over 24h, t=-2.31, n=21) which does not survive being one of 12
    # cells looked at. So the multiple is set where the firing rate is
    # exceptional, and the WINDOW FLAG IS OFF: an exceptional firing rate is
    # a precondition, not evidence that a DETECT helps.
    #
    # The structural fixes stand and are what this constant is for — a single
    # dollar threshold was routine on BTC and unreachable on SOL, and any
    # future arming needs a threshold expressed in each asset's own units.
    # [P316] 5.0 -> 20.0, and the reason is a correction to P311's own
    # calibration basis. P311 derived 5.0 from 11 days of the derivflow
    # ledger (85 rate observations per asset) using the CHANGE in the rolling
    # 24h total between irregular ticks — a noisy estimator whose median is
    # 0. Six months of proper per-4H-bar liquidation history
    # (training_data/coinglass_history, 1,039 bars x 3 assets) says the same
    # multiple fires on 11-14% of bars, not the 1-2% that measurement
    # implied. A ~6x error, and 13% is "routine", which is the exact failure
    # the re-calibration existed to remove.
    #
    # On the 6-month basis the firing rates are:
    #     C      BTC     ETH     SOL
    #     5    13.7%   10.7%   11.1%
    #    10     5.1%    4.8%    4.4%
    #    20     2.1%    1.0%    1.5%   <- exceptional, and p99 is 20-27x on
    #    30     0.9%    0.3%    0.7%      all three, so one multiple still fits
    cascade_detect_liq_multiple: float = 20.0
    # Same 2.5x ratio the absolute pair carried (25M/10M), in the 4h window.
    cascade_accelerate_liq_multiple: float = 50.0
    exhaustion_decel_threshold: float = 0.5  # Liquidation rate drops 50%
    
    # Timing
    phase_min_duration_minutes: int = 15
    max_cascade_duration_hours: int = 8
    
    # Price thresholds
    price_drop_detect_pct: float = 0.03  # 3% drop
    price_drop_accelerate_pct: float = 0.07  # 7% drop
    
    # Exhaustion detection
    price_stabilization_window_minutes: int = 30
    price_stabilization_threshold_pct: float = 0.01  # <1% movement = stabilizing
    
    # Integration
    enable_shadow_ledger: bool = True
    enable_runtime_state: bool = True


class CascadeExhaustionGovernor:
    """
    Cascade Exhaustion Governor - 清算级联削峰

    功能:
    1. 维护清算级联状态机
    2. 控制 T3/T4 加仓权限
    3. 检测 exhaustion (即将反转)
    4. 写入 Shadow Ledger 和 RuntimeState

    P1-03: Accepts optional permissions_override dict for profile-gated relaxation.
    Format: {"DETECTED": {"T1": true, "T2": true, ...}, ...}
    """

    def __init__(self, config: CascadeExhaustionConfig = None, permissions_override: Dict = None):
        self.config = config or CascadeExhaustionConfig()
        self.permissions_override = dict(permissions_override or {})

        # P1-03: Build effective permissions matrix
        self._effective_permissions = dict(TRANCHE_PERMISSIONS)
        if permissions_override:
            _phase_map = {p.value: p for p in CascadePhase}
            _tranche_map = {"T1": TrancheLevel.T1, "T2": TrancheLevel.T2,
                            "T3": TrancheLevel.T3, "T4": TrancheLevel.T4}
            for phase_str, perm_dict in permissions_override.items():
                if phase_str.startswith("_"):
                    continue  # skip meta fields like _description
                phase = _phase_map.get(phase_str)
                if phase and isinstance(perm_dict, dict):
                    self._effective_permissions[phase] = {
                        _tranche_map[k]: bool(v)
                        for k, v in perm_dict.items()
                        if k in _tranche_map
                    }
            logger.info(f"[CASCADE] Permissions override applied for phases: "
                        f"{[p for p in permissions_override if not p.startswith('_')]}")
        
        # State machine
        self._detect_thr = (config or CascadeExhaustionConfig()).cascade_detect_liq_threshold
        self._accel_thr = (config or CascadeExhaustionConfig()).cascade_accelerate_threshold
        self._phase = CascadePhase.NONE
        self._phase_start: Optional[datetime] = None
        self._cascade_start: Optional[datetime] = None
        
        # Metrics history
        self._metrics_history: deque = deque(maxlen=100)
        self._current_metrics = CascadeMetrics()
        
        # Statistics
        self._blocks_count = 0
        self._allows_count = 0
        self._phase_transitions: List[Dict] = []
        self._block_log: List[Dict] = []
        
        # Integrations
        self._shadow_ledger = None
        self._runtime_state = None
        
        self._init_integrations()
        
        logger.info("[CASCADE] Exhaustion Governor initialized")
    
    def _init_integrations(self):
        """Initialize integrations."""
        if self.config.enable_shadow_ledger:
            try:
                from data_mgmt.shadow_ledger_manager import get_shadow_ledger
                self._shadow_ledger = get_shadow_ledger()
            except (ImportError, Exception):
                pass
        
        if self.config.enable_runtime_state:
            try:
                from core.runtime_state import get_runtime_state
                self._runtime_state = get_runtime_state()
            except (ImportError, Exception):
                pass
    
    def effective_liq_thresholds(self, baseline_liq_1h=None):
        """[P311] (detect, accelerate) for THIS asset, in its own units.

        With a baseline the thresholds are a multiple of the asset's own
        recent hourly liquidation rate; without one they fall back to the
        absolute dollar figures, which is exactly today's behaviour. A
        non-finite or non-positive baseline is NOT a baseline — it falls
        back rather than producing a zero threshold, which would make the
        detector fire on every tick (P2: absence must never become a
        permissive value).
        """
        try:
            b = float(baseline_liq_1h)
        except (TypeError, ValueError):
            b = 0.0
        if not (b > 0.0) or b != b or b == float("inf"):
            return (self.config.cascade_detect_liq_threshold,
                    self.config.cascade_accelerate_threshold)
        return (self.config.cascade_detect_liq_multiple * b,
                self.config.cascade_accelerate_liq_multiple * b)

    def update_metrics(
        self,
        liquidation_volume_1h: float = 0.0,
        liquidation_volume_4h: float = 0.0,
        price_change_1h_pct: float = 0.0,
        price_change_4h_pct: float = 0.0,
        volume_spike_ratio: float = 1.0,
        baseline_liq_1h: float = None,
    ):
        """
        Update cascade metrics from market data.
        
        Args:
            liquidation_volume_1h: Liquidation volume in last hour (USD)
            liquidation_volume_4h: Liquidation volume in last 4 hours (USD)
            price_change_1h_pct: Price change % in last hour
            price_change_4h_pct: Price change % in last 4 hours
            volume_spike_ratio: Current volume vs average ratio
        """
        now = datetime.now()
        
        # Calculate acceleration
        prev_metrics = self._current_metrics
        liq_acceleration = 0.0
        if prev_metrics.liquidation_volume_1h > 0:
            liq_acceleration = (liquidation_volume_1h - prev_metrics.liquidation_volume_1h) / prev_metrics.liquidation_volume_1h
        
        # Calculate cascade intensity (0-1)
        # [P311] intensity is a FRACTION OF THE ACCELERATE LEVEL, so it has
        # to use the same (possibly self-normalised) denominator or a
        # small-cap asset reads ~0 intensity during its own cascade.
        _accel_now = self.effective_liq_thresholds(baseline_liq_1h)[1]
        intensity = min(liquidation_volume_4h / _accel_now, 1.0)
        
        # Calculate exhaustion signal
        exhaustion = 0.0
        if liq_acceleration < -self.config.exhaustion_decel_threshold:
            # Liquidations slowing down significantly
            exhaustion = min(abs(liq_acceleration) / 2, 1.0)
        
        # Check price stabilization
        if abs(price_change_1h_pct) < self.config.price_stabilization_threshold_pct:
            exhaustion = max(exhaustion, 0.5)
        
        # Update metrics
        self._current_metrics = CascadeMetrics(
            timestamp=now,
            liquidation_volume_1h=liquidation_volume_1h,
            liquidation_volume_4h=liquidation_volume_4h,
            liquidation_acceleration=liq_acceleration,
            price_change_1h_pct=price_change_1h_pct,
            price_change_4h_pct=price_change_4h_pct,
            price_velocity=price_change_1h_pct - prev_metrics.price_change_1h_pct,
            volume_spike_ratio=volume_spike_ratio,
            cascade_intensity=intensity,
            exhaustion_signal=exhaustion,
        )
        
        self._metrics_history.append(self._current_metrics)

        # [P311] Resolve the thresholds for THIS tick from this asset's own
        # baseline, in update_metrics where the parameter is in scope, and
        # BEFORE the transition reads them. Stored rather than recomputed
        # inside _update_phase so the value that decided the phase is the
        # value get_status() reports.
        self._detect_thr, self._accel_thr = self.effective_liq_thresholds(
            baseline_liq_1h)

        # Update state machine
        self._update_phase()
    
    def _update_phase(self):
        """Update cascade phase based on metrics."""
        metrics = self._current_metrics
        now = datetime.now()
        old_phase = self._phase
        
        # Check minimum phase duration
        phase_duration_ok = True
        if self._phase_start:
            minutes_in_phase = (now - self._phase_start).total_seconds() / 60
            phase_duration_ok = minutes_in_phase >= self.config.phase_min_duration_minutes
        
        # State transitions
        if self._phase == CascadePhase.NONE:
            # Check for cascade detection
            if (metrics.liquidation_volume_1h >= self._detect_thr or
                metrics.price_change_1h_pct <= -self.config.price_drop_detect_pct):
                self._transition_to(CascadePhase.DETECTED)
        
        elif self._phase == CascadePhase.DETECTED:
            if phase_duration_ok:
                # Check for acceleration
                if (metrics.liquidation_volume_4h >= self._accel_thr or
                    metrics.price_change_4h_pct <= -self.config.price_drop_accelerate_pct):
                    self._transition_to(CascadePhase.ACCELERATING)
                # Check for end
                elif metrics.exhaustion_signal > 0.7:
                    self._transition_to(CascadePhase.EXHAUSTING)
                elif metrics.cascade_intensity < 0.2:
                    self._transition_to(CascadePhase.STOP)
        
        elif self._phase == CascadePhase.ACCELERATING:
            if phase_duration_ok:
                # Check for exhaustion
                if metrics.exhaustion_signal > 0.5:
                    self._transition_to(CascadePhase.EXHAUSTING)
                # Check for end
                elif metrics.cascade_intensity < 0.3:
                    self._transition_to(CascadePhase.STOP)
        
        elif self._phase == CascadePhase.EXHAUSTING:
            if phase_duration_ok:
                # Check for cascade end
                if metrics.exhaustion_signal > 0.8 or metrics.cascade_intensity < 0.1:
                    self._transition_to(CascadePhase.STOP)
                # Check for re-acceleration (rare)
                elif metrics.liquidation_acceleration > 0.5:
                    self._transition_to(CascadePhase.ACCELERATING)
        
        elif self._phase == CascadePhase.STOP:
            # Reset to NONE after confirmation
            if phase_duration_ok and metrics.cascade_intensity < 0.1:
                self._transition_to(CascadePhase.NONE)
        
        # Check max cascade duration
        if self._cascade_start:
            hours_active = (now - self._cascade_start).total_seconds() / 3600
            if hours_active >= self.config.max_cascade_duration_hours:
                logger.warning(f"[CASCADE] Max duration reached ({hours_active:.1f}h), forcing STOP")
                self._transition_to(CascadePhase.STOP)
        
        self._update_runtime_state()
    
    def _transition_to(self, new_phase: CascadePhase):
        """Transition to new phase."""
        old_phase = self._phase
        self._phase = new_phase
        self._phase_start = datetime.now()
        
        if new_phase == CascadePhase.DETECTED:
            self._cascade_start = datetime.now()
        elif new_phase == CascadePhase.NONE:
            self._cascade_start = None
        
        transition_record = {
            "timestamp": datetime.now().isoformat(),
            "from_phase": old_phase.value,
            "to_phase": new_phase.value,
            "metrics": self._current_metrics.to_dict(),
        }
        
        self._phase_transitions.append(transition_record)
        if len(self._phase_transitions) > 50:
            self._phase_transitions = self._phase_transitions[-50:]
        
        logger.warning(f"[CASCADE] Phase: {old_phase.value} -> {new_phase.value}")
    
    def check_tranche(
        self,
        symbol: str,
        tranche: int,
        direction: str = "short",
    ) -> TrancheCheckResult:
        """
        Check if a tranche level is allowed in current cascade phase.
        
        Args:
            symbol: Trading symbol
            tranche: Tranche level (1-4)
            direction: Trade direction
            
        Returns:
            TrancheCheckResult with permission and reason
        """
        # Parse tranche level
        try:
            tranche_level = TrancheLevel(tranche)
        except ValueError:
            tranche_level = TrancheLevel.T1
        
        # Get permission from effective matrix (may be overridden by profile)
        permissions = self._effective_permissions.get(
            self._phase, self._effective_permissions.get(CascadePhase.NONE, {})
        )
        allowed = permissions.get(tranche_level, False)
        
        # Determine recommended action
        if self._phase == CascadePhase.EXHAUSTING:
            recommended = "EXIT"
        elif self._phase == CascadePhase.DETECTED and tranche > 1:
            recommended = "WAIT"
        else:
            recommended = "PROCEED" if allowed else "WAIT"
        
        # Build reason
        if allowed:
            reason = f"T{tranche} allowed in {self._phase.value} phase"
        else:
            reason = f"CASCADE_EXHAUSTED: T{tranche} blocked in {self._phase.value} phase"
        
        result = TrancheCheckResult(
            allowed=allowed,
            phase=self._phase,
            tranche=tranche_level,
            reason=reason,
            cascade_intensity=self._current_metrics.cascade_intensity,
            recommended_action=recommended,
        )
        
        if allowed:
            self._allows_count += 1
        else:
            self._handle_block(symbol, tranche_level, direction, result)
        
        return result
    
    def _handle_block(
        self,
        symbol: str,
        tranche: TrancheLevel,
        direction: str,
        result: TrancheCheckResult,
    ):
        """Handle blocked tranche."""
        self._blocks_count += 1
        
        block_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "tranche": tranche.value,
            "direction": direction,
            "phase": self._phase.value,
            "reason": result.reason,
            "cascade_intensity": result.cascade_intensity,
            "recommended_action": result.recommended_action,
        }
        
        self._block_log.append(block_record)
        if len(self._block_log) > 100:
            self._block_log = self._block_log[-100:]
        
        logger.warning(f"[CASCADE] BLOCKED: T{tranche.value} {direction} {symbol} | {result.reason}")
        
        # Write to Shadow Ledger
        if self._shadow_ledger:
            try:
                self._shadow_ledger.log_veto(
                    source="CascadeExhaustionGovernor",
                    reason="CASCADE_EXHAUSTED",
                    details=block_record,
                )
            except Exception:
                pass
    
    def _update_runtime_state(self):
        """Update RuntimeState."""
        if self._runtime_state:
            try:
                self._runtime_state.set("cascade_exhaustion", self.get_status())
            except Exception:
                pass
    
    def get_status(self) -> Dict:
        """Get current status."""
        return {
            "enabled": True,
            "phase": self._phase.value,
            "phase_start": self._phase_start.isoformat() if self._phase_start else None,
            "cascade_start": self._cascade_start.isoformat() if self._cascade_start else None,
            "cascade_duration_hours": (
                (datetime.now() - self._cascade_start).total_seconds() / 3600
                if self._cascade_start else 0
            ),
            "current_metrics": self._current_metrics.to_dict(),
            "tranche_permissions": {
                f"T{t.value}": self._effective_permissions.get(self._phase, {}).get(t, False)
                for t in TrancheLevel
            },
            "statistics": {
                "blocks": self._blocks_count,
                "allows": self._allows_count,
                "phase_transitions": len(self._phase_transitions),
            },
            "recent_transitions": self._phase_transitions[-5:],
            "recent_blocks": self._block_log[-10:],
        }
    
    def reset(self):
        """Reset state."""
        self._phase = CascadePhase.NONE
        self._phase_start = None
        self._cascade_start = None
        self._current_metrics = CascadeMetrics()
        self._metrics_history.clear()
        self._blocks_count = 0
        self._allows_count = 0
        self._phase_transitions.clear()
        self._block_log.clear()
        logger.info("[CASCADE] Governor reset")

    def to_dict(self) -> Dict:
        """Serialize governor state for restart persistence."""
        return {
            "phase": self._phase.value,
            "phase_start": self._phase_start.isoformat() if self._phase_start else None,
            "cascade_start": self._cascade_start.isoformat() if self._cascade_start else None,
            "blocks_count": self._blocks_count,
            "allows_count": self._allows_count,
            "phase_transitions": self._phase_transitions[-20:],
        }

    def from_dict(self, data: Dict):
        """Restore governor state from persistence."""
        phase_str = data.get("phase", "NONE")
        try:
            self._phase = CascadePhase(phase_str)
        except (ValueError, KeyError):
            self._phase = CascadePhase.NONE

        # [P51 2026-04-25] Same naive/aware mismatch as P39 promotion_gate:
        # runtime uses naive datetime.now(), but fromisoformat() returns aware
        # if persisted ISO carries +00:00. After restart, line 507's
        # `datetime.now() - self._cascade_start` would TypeError silently
        # inside try/except. Strip tzinfo on load to match runtime convention.
        def _strip_tz(dt):
            if dt is None or dt.tzinfo is None:
                return dt
            return dt.replace(tzinfo=None)
        if data.get("phase_start"):
            try:
                self._phase_start = _strip_tz(datetime.fromisoformat(data["phase_start"]))
            except (ValueError, TypeError):
                pass
        if data.get("cascade_start"):
            try:
                self._cascade_start = _strip_tz(datetime.fromisoformat(data["cascade_start"]))
            except (ValueError, TypeError):
                pass

        self._blocks_count = data.get("blocks_count", 0)
        self._allows_count = data.get("allows_count", 0)
        self._phase_transitions = data.get("phase_transitions", [])

        logger.info(
            f"[CASCADE] State restored: phase={self._phase.value}, "
            f"blocks={self._blocks_count}, allows={self._allows_count}"
        )


# =============================================================================
# SINGLETON
# =============================================================================

_cascade_exhaustion_governor: Optional[CascadeExhaustionGovernor] = None

# [P306] PER-ASSET INSTANCES. The accessor used to hand every caller ONE
# instance, and main.py feeds it BTC -> ETH -> SOL in sequence every tick. So
# the state machine's `prev_metrics` was always the PREVIOUS ASSET's reading:
#
#   liq_acceleration = (this_asset_vol - prev_asset_vol) / prev_asset_vol
#   price_velocity   = this_asset_1h  - prev_asset_1h
#
# Measured on the live ledger 2026-08-18: BTC feeds ~$6.19M/h against SOL's
# ~$0.16M/h, so the SOL tick computes acceleration -0.97, which clears
# `< -exhaustion_decel_threshold (-0.5)` and raises the exhaustion signal on
# essentially every SOL tick — purely from the order the assets are iterated
# in. `market_data["cascade_phase"]` was likewise read off a machine last
# transitioned by a different asset.
#
# This is the P225 cross-asset leak (`_last_phase_result`) on a live risk
# path. Instances are now keyed by asset; the no-arg call keeps returning the
# same shared instance it always did, so callers that legitimately have no
# asset in scope are byte-identical.
_cascade_governors: Dict[str, CascadeExhaustionGovernor] = {}
_cascade_permissions_override: Optional[Dict] = None


def get_cascade_exhaustion_governor(
    config: CascadeExhaustionConfig = None,
    permissions_override: Dict = None,
    asset: Optional[str] = None,
) -> CascadeExhaustionGovernor:
    """Get or create the governor for `asset` (None = the shared instance).

    A permissions override supplied once (main.py's [P1-03] pre-init) is
    remembered and applied to every instance created afterwards — otherwise
    per-asset instances would silently run on default permissions while the
    operator believed the override was in force.
    """
    global _cascade_exhaustion_governor, _cascade_permissions_override
    if permissions_override is not None:
        _cascade_permissions_override = permissions_override
    effective = (permissions_override
                 if permissions_override is not None
                 else _cascade_permissions_override)

    key = str(asset).upper() if asset else ""
    existing = (_cascade_exhaustion_governor if not key
                else _cascade_governors.get(key))

    def _store(inst):
        global _cascade_exhaustion_governor
        if key:
            _cascade_governors[key] = inst
        else:
            _cascade_exhaustion_governor = inst
        return inst

    if existing is None:
        return _store(CascadeExhaustionGovernor(
            config, permissions_override=effective))
    if ((config is not None and existing.config != config)
            or getattr(existing, "permissions_override", {}) != (effective or {})):
        logger.info(
            "[CASCADE_EXHAUSTION] Constructor inputs changed; refreshing "
            "instance for %s", key or "<shared>")
        return _store(CascadeExhaustionGovernor(
            config, permissions_override=effective))
    return existing


def all_governor_states() -> Dict[str, Dict]:
    """[P306] Every instance's state, keyed by asset ("" = the shared one).

    Per-asset machines were introduced in P306; without this the phase of
    each asset's machine would reset on every deploy while the shared
    instance's kept being persisted — which is exactly the RAM-only-state
    class (P301/P302/P303) that made three other signals look dead.
    """
    out: Dict[str, Dict] = {}
    try:
        if _cascade_exhaustion_governor is not None:
            out[""] = _cascade_exhaustion_governor.to_dict()
        for k, inst in _cascade_governors.items():
            out[k] = inst.to_dict()
    except Exception as e:  # noqa: silent-swallow — logged; persistence only
        logger.warning("[CASCADE_EXHAUSTION] state capture failed (%s: %s)",
                       type(e).__name__, e)
    return out


def restore_governor_states(data: Optional[Dict]) -> int:
    """Restore states written by `all_governor_states`. Returns the count.

    Accepts the PRE-P306 flat shape too (a single governor's own to_dict),
    which restores into the shared instance — a state file written by the
    previous build must not be read as "no state", nor mistaken for a
    per-asset map (its keys are field names, not asset names).
    """
    if not isinstance(data, dict) or not data:
        return 0
    per_asset = all(isinstance(v, dict) for v in data.values())
    n = 0
    try:
        if not per_asset:
            get_cascade_exhaustion_governor().from_dict(data)
            return 1
        for key, state in data.items():
            if not isinstance(state, dict):
                continue
            inst = get_cascade_exhaustion_governor(asset=key or None)
            inst.from_dict(state)
            n += 1
    except Exception as e:  # noqa: silent-swallow — logged; cold start is safe
        logger.warning("[CASCADE_EXHAUSTION] state restore failed (%s: %s) — "
                       "starting from NONE", type(e).__name__, e)
    return n


def reset_cascade_exhaustion_governor():
    """Reset every instance."""
    global _cascade_exhaustion_governor, _cascade_permissions_override
    if _cascade_exhaustion_governor:
        _cascade_exhaustion_governor.reset()
    for _inst in _cascade_governors.values():
        try:
            _inst.reset()
        except Exception:  # noqa: silent-swallow - teardown only
            pass
    _cascade_governors.clear()
    _cascade_exhaustion_governor = None
    _cascade_permissions_override = None


# =============================================================================
# CONVENIENCE FUNCTION FOR CANONICAL FLOW
# =============================================================================

def check_cascade_tranche(
    symbol: str,
    tranche: int,
    direction: str = "short",
) -> Tuple[bool, str, str]:
    """
    Convenience function for canonical flow integration.
    
    Returns:
        Tuple of (allowed, reason, recommended_action)
    """
    # Check if enabled via flags
    try:
        from configs.sota_flags import get_sota_flags
        flags = get_sota_flags()
        if not getattr(flags, 'ENABLE_CASCADE_EXHAUSTION_GOVERNOR', True):
            return True, "Governor disabled", "PROCEED"
    except ImportError:
        pass
    
    governor = get_cascade_exhaustion_governor()
    result = governor.check_tranche(symbol, tranche, direction)
    return result.allowed, result.reason, result.recommended_action


__all__ = [
    "CascadePhase",
    "TrancheLevel",
    "CascadeMetrics",
    "TrancheCheckResult",
    "CascadeExhaustionConfig",
    "CascadeExhaustionGovernor",
    "get_cascade_exhaustion_governor",
    "reset_cascade_exhaustion_governor",
    "check_cascade_tranche",
    "TRANCHE_PERMISSIONS",
]

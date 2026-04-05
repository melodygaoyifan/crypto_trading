"""
================================================================================
TRADE GATE - Signal Validation and Trade Permission
================================================================================
Version: 3.2.0
Purpose: Final gate before trade execution (Part 4, 5, 7 compliance)

Features:
- DVOL突变强制Defensive模式 (Part 4)
- No-trade逻辑 (Part 4)
- Stale-data 2秒 kill switch (Part 5)
- DRL regime约束 (Part 7)
- 结构性约束检查 (Part 11)
================================================================================
"""

import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
import threading

from configs.canonical_config import KRAKEN_FREE_TIER_USD as _CC_FREE_TIER  # [FIX-33]

logger = logging.getLogger("HMATS.TradeGate")


class GateDecision(Enum):
    """Trade gate decision."""
    ALLOW = auto()           # Trade allowed
    REDUCE = auto()          # Trade with reduced size
    DELAY = auto()           # Wait and retry
    REJECT = auto()          # Trade rejected
    EMERGENCY_FLAT = auto()  # Emergency flatten all
    EXIT_ONLY = auto()       # Only exit trades allowed (DRL demotion state)
    CLIP = auto()            # Trade allowed but size clipped


class RejectReason(Enum):
    """Reason for trade rejection."""
    NONE = auto()
    STALE_DATA = auto()
    DVOL_SPIKE = auto()
    INSUFFICIENT_EDGE = auto()
    REGIME_UNSTABLE = auto()
    DRL_OVERCONFIDENT = auto()
    VOLUME_CONTRACTING = auto()
    STRUCTURE_INVALID = auto()
    MACRO_RISK = auto()
    SOL_NETWORK_STRESS = auto()
    POSITION_LIMIT = auto()
    DRAWDOWN_LIMIT = auto()
    # P0: 数据健康相关
    DATA_HEALTH_DEGRADED = auto()    # 数据源健康度降级
    DATA_HEALTH_CRITICAL = auto()    # 关键数据源缺失
    DATA_HEALTH_EXIT_ONLY = auto()   # 仅允许平仓
    # v6.1.3: Long-Term Survival Governors
    OPPORTUNITY_BUDGET_EXCEEDED = auto()  # OP 预算超限
    REGIME_TRANSITION_BLOCK = auto()      # 过渡态阻止
    CASCADE_EXHAUSTED = auto()            # 清算削峰阻止


@dataclass
class TradeProposal:
    """Proposed trade for validation."""
    asset: str
    side: str                      # 'long' or 'short'
    size: Decimal
    expected_alpha_bps: Decimal
    signal_source: str             # 'quant', 'drl', 'fusion'
    confidence: float
    regime: str
    regime_confidence: float
    drl_weight: float = 0.5
    timestamp: float = field(default_factory=time.time)
    
    # Context
    dvol_current: float = 0.0
    dvol_zscore: float = 0.0
    volume_ratio: float = 1.0      # Current vs average volume
    is_structure_breakout: bool = False
    price_direction: str = "flat"  # 'up', 'down', 'flat'


@dataclass
class GateResult:
    """Result from trade gate."""
    decision: GateDecision
    reason: RejectReason = RejectReason.NONE
    adjusted_size: Decimal = Decimal('0')
    delay_seconds: float = 0.0
    details: Dict = field(default_factory=dict)


@dataclass
class TradeGateConfig:
    """Trade gate configuration."""
    # Edge requirements (Part 4)
    min_edge_multiplier: float = 1.5  # Lowered from 3.0: alpha gate at 50bps is primary filter; 3x costs (99bps) too aggressive for 4H system
    min_edge_multiplier_by_asset: Dict[str, float] = field(default_factory=dict)
    min_edge_multiplier_long_by_asset: Dict[str, float] = field(default_factory=dict)
    min_edge_multiplier_short_by_asset: Dict[str, float] = field(default_factory=dict)
    kraken_taker_fee_bps: float = 26.0  # Default; overridden by update_fee_tier()
    kraken_maker_fee_bps: float = 16.0  # Maker fee; used when post_only/PASSIVE_PREFERRED
    use_maker_fee: bool = True  # v6.7: post_only maker orders enabled by default
    slippage_bps: float = 5.0
    latency_cost_bps: float = 2.0
    KRAKEN_FREE_TIER_USD: float = _CC_FREE_TIER  # [FIX-33] from canonical_config
    
    # DVOL thresholds (Part 4)
    dvol_zscore_warning: float = 2.0
    dvol_zscore_defensive: float = 3.0
    dvol_zscore_emergency: float = 5.0
    
    # Data freshness (Part 5)
    max_data_age_seconds: float = 60.0  # [T4] was 30.0; unified 60s for 4H tick cycle
    max_snapshot_age_at_fetch_seconds: float = 5.0
    tick_processing_grace_seconds: float = 120.0
    max_cached_orderbook_age_for_tick_grace_seconds: float = 120.0
    
    # DRL constraints (Part 7)
    min_regime_confidence_for_drl: float = 0.6
    max_drl_weight_unstable_regime: float = 0.3
    drl_confidence_penalty_threshold: float = 0.9  # Too confident = penalize
    
    # Volume constraints (Part 11)
    min_volume_ratio: float = 0.5   # Reject if volume < 50% of average
    
    # Structure constraints (Part 11)
    require_structure_for_trend: bool = True


class DVOLDefenseModule:
    """
    DVOL-based defensive mode control.
    
    Forces defensive posture when DVOL spikes.
    """
    
    def __init__(self, config: TradeGateConfig):
        self.config = config
        
        self.current_dvol = 0.0
        self.dvol_zscore = 0.0
        self.defensive_mode = False
        self.emergency_mode = False
        
        self._history: List[float] = []
        self._lock = threading.RLock()  # [BUGFIX AUDIT-C2] RLock for re-entrant safety
    
    def update(self, dvol: float, zscore: float = None):
        """Update DVOL state."""
        with self._lock:
            self.current_dvol = dvol
            self._history.append(dvol)
            
            # Keep 100 samples
            if len(self._history) > 100:
                self._history = self._history[-100:]
            
            # Calculate z-score if not provided
            if zscore is not None:
                self.dvol_zscore = zscore
            elif len(self._history) >= 20:
                import numpy as np
                mean = np.mean(self._history)
                std = max(np.std(self._history), 0.01)
                self.dvol_zscore = (dvol - mean) / std
            
            # Update modes
            self.emergency_mode = self.dvol_zscore >= self.config.dvol_zscore_emergency
            self.defensive_mode = self.dvol_zscore >= self.config.dvol_zscore_defensive
    
    def check(self) -> Tuple[bool, str]:
        """
        Check if trading should be allowed.
        
        Returns:
            (allowed, mode)
        """
        with self._lock:
            if self.emergency_mode:
                return False, "EMERGENCY_FLAT"
            elif self.defensive_mode:
                return True, "DEFENSIVE"
            elif self.dvol_zscore >= self.config.dvol_zscore_warning:
                return True, "CAUTIOUS"
            else:
                return True, "NORMAL"


class StaleDataGuard:
    """
    Stale data detection and kill switch.
    
    Rejects trades if data is older than threshold.
    """
    
    def __init__(self, max_age_seconds: float = 2.0):
        self.max_age = max_age_seconds
        
        self.last_update_times: Dict[str, float] = {}
        self._lock = threading.RLock()  # [BUGFIX AUDIT-C2] RLock for re-entrant safety
    
    def update_timestamp(self, source: str, timestamp: float = None):
        """Update data timestamp for source."""
        with self._lock:
            self.last_update_times[source] = timestamp or time.time()
    
    def check_freshness(self, required_sources: List[str] = None, log_on_stale: bool = True) -> Tuple[bool, Dict]:
        """
        Check if all required data is fresh.
        
        Returns:
            (is_fresh, details)
        """
        with self._lock:
            now = time.time()
            required = required_sources or ['price', 'orderbook', 'vpin']
            
            stale_sources = []
            details = {}
            
            for source in required:
                last_update = self.last_update_times.get(source, 0)
                age = now - last_update
                details[source] = {'age_seconds': age, 'fresh': age <= self.max_age}
                
                if age > self.max_age:
                    stale_sources.append(source)
            
            is_fresh = len(stale_sources) == 0
            details['stale_sources'] = stale_sources
            
            if not is_fresh and log_on_stale:
                logger.warning(f"Stale data detected: {stale_sources}")
            
            return is_fresh, details


class DRLConstraintEnforcer:
    """
    DRL behavior constraints.
    
    Prevents DRL from being overconfident in unstable regimes.
    """
    
    def __init__(self, config: TradeGateConfig):
        self.config = config
    
    def evaluate(
        self,
        drl_confidence: float,
        drl_weight: float,
        regime: str,
        regime_confidence: float
    ) -> Tuple[bool, float, str]:
        """
        Evaluate DRL behavior constraints.
        
        Returns:
            (allowed, adjusted_weight, reason)
        """
        # Check regime stability requirement
        if regime_confidence < self.config.min_regime_confidence_for_drl:
            if drl_weight > self.config.max_drl_weight_unstable_regime:
                return False, self.config.max_drl_weight_unstable_regime, "REGIME_UNSTABLE"
        
        # Check for overconfidence penalty
        if drl_confidence > self.config.drl_confidence_penalty_threshold:
            # DRL is suspiciously confident - reduce weight
            penalty = (drl_confidence - self.config.drl_confidence_penalty_threshold) * 2
            adjusted_weight = max(0.1, drl_weight - penalty)
            return True, adjusted_weight, "OVERCONFIDENCE_PENALTY"
        
        # Check regime-specific constraints
        unstable_regimes = ['VOLATILE_CHOP', 'REGIME_TRANSITION']
        if regime in unstable_regimes:
            max_weight = 0.3
            if drl_weight > max_weight:
                return True, max_weight, "UNSTABLE_REGIME_CAP"
        
        return True, drl_weight, "OK"


class StructureConstraintChecker:
    """
    BitBeast structure constraints (Part 11).
    
    - 禁止缩量下跌抄底
    - 结构突破才顺势
    """
    
    def __init__(self, config: TradeGateConfig):
        self.config = config
    
    def check_volume_constraint(
        self,
        side: str,
        volume_ratio: float,
        price_direction: str
    ) -> Tuple[bool, str]:
        """
        Check volume constraint.
        
        Rule: 禁止在缩量下跌中抄底
        """
        # If trying to go long while price is falling and volume contracting
        if side == 'long' and price_direction == 'down':
            if volume_ratio < self.config.min_volume_ratio:
                return False, "CONTRACTING_VOLUME_BOTTOM_FISHING"
        
        return True, "OK"
    
    def check_structure_constraint(
        self,
        side: str,
        is_structure_breakout: bool,
        regime: str
    ) -> Tuple[bool, str]:
        """
        Check structure constraint.
        
        Rule: 趋势交易需要结构突破确认
        """
        if not self.config.require_structure_for_trend:
            return True, "DISABLED"
        
        # Trend regimes require structure breakout
        trend_regimes = ['STRONG_BULL', 'STRONG_BEAR', 'MODERATE_BULL', 'MODERATE_BEAR']
        
        if regime in trend_regimes:
            if not is_structure_breakout:
                return False, "NO_STRUCTURE_BREAKOUT"
        
        return True, "OK"


class TradeGate:
    """
    Final gate before trade execution.
    
    All trades must pass through this gate which enforces:
    - DVOL spike defensive mode
    - Edge requirements
    - Data freshness
    - DRL constraints
    - Structure constraints
    """
    
    def __init__(self, config: TradeGateConfig = None):
        self.config = config or TradeGateConfig()
        
        # Sub-modules
        self.dvol_defense = DVOLDefenseModule(self.config)
        self.stale_guard = StaleDataGuard(self.config.max_data_age_seconds)
        self.drl_enforcer = DRLConstraintEnforcer(self.config)
        self.structure_checker = StructureConstraintChecker(self.config)
        
        # P0: 数据健康检查
        self._data_health_snapshot = None
        self._data_health_enabled = True
        
        # v6.1.3: Governor Integration
        self._governor_integration = None
        self._governor_enabled = True
        self._init_governors()
        
        # Statistics
        self.total_proposals = 0
        self.allowed_count = 0
        self.rejected_count = 0
        self.rejection_reasons: Dict[RejectReason, int] = {}
        
        self._lock = threading.RLock()  # [BUGFIX AUDIT-C2] RLock for re-entrant safety
        
        logger.info("TradeGate initialized")

    def _get_min_edge_multiplier(self, asset: str, side: str) -> float:
        """Resolve narrow asset/side overrides before falling back to the global multiplier."""
        asset_key = str(asset or "").replace("/USD", "").upper()
        side_key = "short" if str(side or "").lower() == "short" else "long"

        if side_key == "short" and asset_key in self.config.min_edge_multiplier_short_by_asset:
            return float(self.config.min_edge_multiplier_short_by_asset[asset_key])
        if side_key == "long" and asset_key in self.config.min_edge_multiplier_long_by_asset:
            return float(self.config.min_edge_multiplier_long_by_asset[asset_key])
        if asset_key in self.config.min_edge_multiplier_by_asset:
            return float(self.config.min_edge_multiplier_by_asset[asset_key])
        return float(self.config.min_edge_multiplier)

    def update_fee_tier(self, monthly_volume_usd: float) -> None:
        """[VC-8] Smooth fee transition matching kraken_plus_fee_blender blend zone.

        $0-$10K: free tier (0 fees)
        $8K-$10K: warning zone (still free, log approaching)
        $10K-$12K: blend zone (linear interpolation 0 -> full fees)
        $12K+: full fees (26bps taker / 16bps maker)
        """
        _FREE = self.config.KRAKEN_FREE_TIER_USD       # 10,000
        _BLEND_BAND = 2_000.0
        _FULL = _FREE + _BLEND_BAND                     # 12,000
        _WARN = _FREE - 2_000.0                          # 8,000

        if monthly_volume_usd < _WARN:
            # Deep in free tier
            self.config.kraken_taker_fee_bps = 0.0
            self.config.kraken_maker_fee_bps = 0.0
        elif monthly_volume_usd < _FREE:
            # Warning zone: still free, but approaching limit
            self.config.kraken_taker_fee_bps = 0.0
            self.config.kraken_maker_fee_bps = 0.0
            logger.info(
                f"[VC-8] Fee tier WARNING: ${monthly_volume_usd:,.0f}/${_FREE:,.0f} "
                f"({monthly_volume_usd/_FREE:.0%}) -- approaching paid tier"
            )
        elif monthly_volume_usd < _FULL:
            # Blend zone: linear interpolation
            weight = (monthly_volume_usd - _FREE) / _BLEND_BAND
            self.config.kraken_taker_fee_bps = 26.0 * weight
            self.config.kraken_maker_fee_bps = 16.0 * weight
        else:
            # Full fee tier
            self.config.kraken_taker_fee_bps = 26.0
            self.config.kraken_maker_fee_bps = 16.0

    def _init_governors(self):
        """Initialize v6.1.3 governors."""
        try:
            from configs.sota_flags import get_sota_flags
            flags = get_sota_flags()
            if not any([
                getattr(flags, 'ENABLE_OPPORTUNITY_BUDGET_GOVERNOR', True),
                getattr(flags, 'ENABLE_REGIME_TRANSITION_BUFFER', True),
                getattr(flags, 'ENABLE_CASCADE_EXHAUSTION_GOVERNOR', True),
            ]):
                self._governor_enabled = False
                return
        except ImportError:
            pass
        
        try:
            from defense.governor_integration import get_governor_integration
            self._governor_integration = get_governor_integration()
            logger.info("[TradeGate] v6.1.3 Governors connected")
        except ImportError as e:
            logger.debug(f"[TradeGate] Governor integration not available: {e}")
            self._governor_enabled = False
        except Exception as e:
            logger.warning(f"[TradeGate] Governor init failed: {e}")
            self._governor_enabled = False
    
    def set_data_health_snapshot(self, snapshot):
        """
        P0: 设置数据健康快照
        
        由 data_health 模块调用
        
        Args:
            snapshot: DataHealthSnapshot 对象
        """
        with self._lock:
            self._data_health_snapshot = snapshot
    
    def set_data_health_enabled(self, enabled: bool):
        """启用/禁用数据健康检查"""
        self._data_health_enabled = enabled
    
    def _check_freshness_with_context(self, **kwargs) -> Tuple[bool, Dict]:
        """Allow fresh current-tick snapshots to survive internal processing lag."""
        is_fresh, details = self.stale_guard.check_freshness(log_on_stale=False)
        if is_fresh:
            details["freshness_mode"] = "direct"
            return True, details

        stale_sources = set(details.get("stale_sources", []))
        snapshot_age = kwargs.get("market_data_age_seconds")
        fetched_at_ts = kwargs.get("market_data_fetched_at_ts")
        orderbook_stale = bool(kwargs.get("orderbook_stale", False))
        orderbook_fallback_reason = str(kwargs.get("orderbook_fallback_reason", "") or "")
        orderbook_cache_age = kwargs.get("orderbook_cache_age_seconds")
        decision_lag_seconds = None

        try:
            snapshot_age = None if snapshot_age is None else float(snapshot_age)
        except (TypeError, ValueError):
            snapshot_age = None

        try:
            orderbook_cache_age = None if orderbook_cache_age is None else float(orderbook_cache_age)
        except (TypeError, ValueError):
            orderbook_cache_age = None

        try:
            if fetched_at_ts is not None:
                decision_lag_seconds = max(0.0, time.time() - float(fetched_at_ts))
        except (TypeError, ValueError):
            decision_lag_seconds = None

        cached_depth_grace = (
            orderbook_stale
            and orderbook_fallback_reason == "cached_depth"
            and orderbook_cache_age is not None
            and orderbook_cache_age <= self.config.max_cached_orderbook_age_for_tick_grace_seconds
        )
        orderbook_fresh_enough = (not orderbook_stale) or cached_depth_grace

        if (
            stale_sources
            and stale_sources.issubset({"price", "orderbook", "vpin"})
            and decision_lag_seconds is not None
            and decision_lag_seconds <= self.config.tick_processing_grace_seconds
            and snapshot_age is not None
            and snapshot_age <= self.config.max_snapshot_age_at_fetch_seconds
            and orderbook_fresh_enough
        ):
            for source in stale_sources:
                source_details = details.setdefault(source, {})
                source_details["fresh_via_tick_grace"] = True
                source_details["decision_lag_seconds"] = decision_lag_seconds
                source_details["snapshot_age_at_fetch_seconds"] = snapshot_age
                if source == "orderbook" and cached_depth_grace:
                    source_details["fresh_via_cached_depth"] = True
                    source_details["orderbook_cache_age_seconds"] = orderbook_cache_age

            details["stale_sources"] = []
            details["freshness_mode"] = "tick_grace"
            details["decision_lag_seconds"] = decision_lag_seconds
            details["snapshot_age_at_fetch_seconds"] = snapshot_age
            details["tick_processing_grace_seconds"] = self.config.tick_processing_grace_seconds
            details["max_snapshot_age_at_fetch_seconds"] = self.config.max_snapshot_age_at_fetch_seconds
            details["orderbook_stale"] = orderbook_stale
            details["orderbook_fallback_reason"] = orderbook_fallback_reason or "live"
            details["orderbook_cache_age_seconds"] = orderbook_cache_age
            details["orderbook_cached_depth_grace"] = cached_depth_grace
            logger.info(
                "[TradeGate] Freshness grace applied: "
                f"lag={decision_lag_seconds:.1f}s age_at_fetch={snapshot_age:.2f}s "
                f"sources={sorted(stale_sources)} "
                f"ob={details['orderbook_fallback_reason']}"
            )
            return True, details

        details["orderbook_stale"] = orderbook_stale
        if orderbook_fallback_reason:
            details["orderbook_fallback_reason"] = orderbook_fallback_reason
        if orderbook_cache_age is not None:
            details["orderbook_cache_age_seconds"] = orderbook_cache_age
        if stale_sources:
            logger.warning(f"Stale data detected: {sorted(stale_sources)}")
        return False, details

    def evaluate(self, proposal: TradeProposal, **kwargs) -> GateResult:
        """
        Evaluate trade proposal through all gates.
        
        Returns GateResult with decision and details.
        
        Additional kwargs for v6.1.3 governors:
            is_opportunity: bool - Whether this is an OPPORTUNITY mode trade
            opportunity_type: str - Type of opportunity signal
            is_naked_short: bool - Whether this is a naked short
            strategy_name: str - Name of the strategy
            tranche: int - Tranche level (1-4)
            has_hedge: bool - Whether position has hedge
        """
        with self._lock:
            self.total_proposals += 1
        
        details = {}
        
        # Gate 0: P0 Data Health Check (NEW)
        if self._data_health_enabled and self._data_health_snapshot:
            snapshot = self._data_health_snapshot
            details['data_health'] = {
                'status': snapshot.overall_status.value if hasattr(snapshot.overall_status, 'value') else str(snapshot.overall_status),
                'degradation_level': snapshot.degradation_level.value if hasattr(snapshot.degradation_level, 'value') else str(snapshot.degradation_level),
                'reason': snapshot.degradation_reason,
            }
            
            # 检查是否允许交易
            if hasattr(snapshot, 'degradation_level'):
                from data_mgmt.data_health import DegradationLevel
                
                # NO_TRADE -> 拒绝所有交易
                if snapshot.degradation_level == DegradationLevel.NO_TRADE:
                    logger.warning(f"Trade rejected: DATA_HEALTH_CRITICAL - {snapshot.degradation_reason}")
                    return self._reject(RejectReason.DATA_HEALTH_CRITICAL, details)
                
                # EXIT_ONLY -> 只允许平仓（需要检查 proposal 是否是平仓）
                if snapshot.degradation_level == DegradationLevel.EXIT_ONLY:
                    # 假设新开仓 = entry，简化判断
                    is_entry = not getattr(proposal, 'is_exit', False)
                    if is_entry:
                        logger.warning(f"Trade rejected: DATA_HEALTH_EXIT_ONLY - only exits allowed")
                        return self._reject(RejectReason.DATA_HEALTH_EXIT_ONLY, details)
                
                # REDUCED -> 减少仓位（在后面处理）
                if snapshot.degradation_level == DegradationLevel.REDUCED:
                    details['data_health_reduced'] = True
        
        # Gate 1: Stale Data Check (Part 5)
        is_fresh, freshness_details = self._check_freshness_with_context(**kwargs)
        details['freshness'] = freshness_details
        
        if not is_fresh:
            return self._reject(RejectReason.STALE_DATA, details)
        
        # Gate 2: DVOL Defense (Part 4)
        self.dvol_defense.update(proposal.dvol_current, proposal.dvol_zscore)
        dvol_allowed, dvol_mode = self.dvol_defense.check()
        details['dvol_mode'] = dvol_mode
        
        if not dvol_allowed:
            return GateResult(
                decision=GateDecision.EMERGENCY_FLAT,
                reason=RejectReason.DVOL_SPIKE,
                details=details
            )
        
        # Gate 3: Edge Filter (Part 4)
        # [VC-1] Skip when Constitution alpha_gate already passed (avoids duplicate check)
        _alpha_gate_passed = kwargs.get('alpha_gate_passed', False)
        if not _alpha_gate_passed:
            fee_bps = (self.config.kraken_maker_fee_bps if self.config.use_maker_fee
                       else self.config.kraken_taker_fee_bps)
            edge_multiplier = self._get_min_edge_multiplier(proposal.asset, proposal.side)
            min_edge = edge_multiplier * (
                fee_bps +
                self.config.slippage_bps +
                self.config.latency_cost_bps
            )
            details['min_edge_multiplier'] = edge_multiplier

            if float(proposal.expected_alpha_bps) < min_edge:
                details['min_edge_bps'] = min_edge
                details['expected_alpha_bps'] = float(proposal.expected_alpha_bps)
                return self._reject(RejectReason.INSUFFICIENT_EDGE, details)
        else:
            details['gate3_skipped'] = 'alpha_gate_passed'
        
        # Gate 4: DRL Constraints (Part 7)
        drl_allowed, adjusted_drl_weight, drl_reason = self.drl_enforcer.evaluate(
            drl_confidence=proposal.confidence if proposal.signal_source == 'drl' else 0.5,
            drl_weight=proposal.drl_weight,
            regime=proposal.regime,
            regime_confidence=proposal.regime_confidence
        )
        details['drl_constraint'] = drl_reason
        details['adjusted_drl_weight'] = adjusted_drl_weight
        
        if not drl_allowed:
            return self._reject(RejectReason.DRL_OVERCONFIDENT, details)
        
        # Gate 5: Volume Constraint (Part 11)
        volume_ok, volume_reason = self.structure_checker.check_volume_constraint(
            side=proposal.side,
            volume_ratio=proposal.volume_ratio,
            price_direction=proposal.price_direction,
        )
        details['volume_constraint'] = volume_reason
        details['price_direction'] = proposal.price_direction
        
        if not volume_ok:
            return self._reject(RejectReason.VOLUME_CONTRACTING, details)
        
        # Gate 6: Structure Constraint (Part 11)
        structure_ok, structure_reason = self.structure_checker.check_structure_constraint(
            side=proposal.side,
            is_structure_breakout=proposal.is_structure_breakout,
            regime=proposal.regime
        )
        details['structure_constraint'] = structure_reason
        
        if not structure_ok:
            return self._reject(RejectReason.STRUCTURE_INVALID, details)
        
        # Gate 7: v6.1.3 Governor Checks (Long-Term Survival)
        if self._governor_enabled and self._governor_integration:
            try:
                # Determine if this is an opportunity trade
                is_opportunity = kwargs.get('is_opportunity', False) or \
                                proposal.signal_source == 'opportunity'
                opportunity_type = kwargs.get('opportunity_type', 'generic')
                is_naked_short = kwargs.get('is_naked_short', 
                                           proposal.side == 'short' and not kwargs.get('has_hedge', False))
                strategy_name = kwargs.get('strategy_name', proposal.signal_source)
                tranche = kwargs.get('tranche', 1)
                
                gov_result = self._governor_integration.check_all(
                    symbol=proposal.asset,
                    direction=proposal.side,
                    intent_type='trade',
                    is_opportunity=is_opportunity,
                    opportunity_type=opportunity_type,
                    is_naked_short=is_naked_short,
                    strategy_name=strategy_name,
                    tranche=tranche,
                    confidence=proposal.confidence,
                )
                
                details['governor_result'] = gov_result.to_dict()
                
                if not gov_result.allowed:
                    # Map veto source to reject reason
                    from defense.governor_integration import GovernorVetoSource
                    reason_map = {
                        GovernorVetoSource.OPPORTUNITY_BUDGET: RejectReason.OPPORTUNITY_BUDGET_EXCEEDED,
                        GovernorVetoSource.REGIME_TRANSITION: RejectReason.REGIME_TRANSITION_BLOCK,
                        GovernorVetoSource.CASCADE_EXHAUSTION: RejectReason.CASCADE_EXHAUSTED,
                    }
                    reject_reason = reason_map.get(gov_result.vetoed_by, RejectReason.STRUCTURE_INVALID)
                    details['governor_veto_reason'] = gov_result.reason
                    return self._reject(reject_reason, details)
                
                # Apply governor modifications
                if gov_result.modifications:
                    details['governor_modifications'] = gov_result.modifications
                    
            except Exception as e:
                logger.warning(f"[TradeGate] Governor check error (allowing): {e}")
        
        # All gates passed
        adjusted_size = proposal.size
        
        # Reduce size if in DVOL cautious/defensive mode
        if dvol_mode == 'DEFENSIVE':
            adjusted_size = proposal.size * Decimal('0.5')
            details['size_reduction'] = 'DVOL_DEFENSIVE'
        elif dvol_mode == 'CAUTIOUS':
            adjusted_size = proposal.size * Decimal('0.75')
            details['size_reduction'] = 'DVOL_CAUTIOUS'
        
        with self._lock:
            self.allowed_count += 1
        
        return GateResult(
            decision=GateDecision.ALLOW if adjusted_size == proposal.size else GateDecision.REDUCE,
            reason=RejectReason.NONE,
            adjusted_size=adjusted_size,
            details=details
        )
    
    def _reject(self, reason: RejectReason, details: Dict) -> GateResult:
        """Record rejection and return result."""
        with self._lock:
            self.rejected_count += 1
            self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
        
        logger.info(f"Trade rejected: {reason.name}")
        
        return GateResult(
            decision=GateDecision.REJECT,
            reason=reason,
            details=details
        )
    
    def update_data_timestamp(self, source: str, timestamp: float = None):
        """Update data timestamp for freshness tracking."""
        self.stale_guard.update_timestamp(source, timestamp)
    
    def update_dvol(self, dvol: float, zscore: float = None):
        """Update DVOL for defense module."""
        self.dvol_defense.update(dvol, zscore)
    
    def get_stats(self) -> Dict:
        """Get gate statistics."""
        with self._lock:
            return {
                'total_proposals': self.total_proposals,
                'allowed': self.allowed_count,
                'rejected': self.rejected_count,
                'pass_rate': self.allowed_count / max(1, self.total_proposals),
                'rejection_reasons': {
                    reason.name: count
                    for reason, count in self.rejection_reasons.items()
                },
                'dvol_defensive_mode': self.dvol_defense.defensive_mode,
                'dvol_emergency_mode': self.dvol_defense.emergency_mode
            }
    
    # =========================================================================
    # v5.1.0-HARDENED: check() interface for main.py compatibility
    # =========================================================================
    
    def check(
        self,
        asset: str,
        side: str,
        size: float,
        confidence: float = 0.5,
        regime: str = "UNKNOWN",
        **kwargs
    ) -> GateResult:
        """
        Simplified check interface for main.py compatibility.
        
        v5.1.0-HARDENED: Glue-layer fix - main.py calls check() but
        TradeGate only had evaluate() which requires TradeProposal.
        
        Args:
            asset: Asset symbol
            side: 'long' or 'short'
            size: Position size
            confidence: Signal confidence
            regime: Current regime
            **kwargs: Additional context
            
        Returns:
            GateResult with decision
        """
        # Build TradeProposal from simplified interface
        proposal = TradeProposal(
            asset=asset,
            side=side,
            size=Decimal(str(size)),
            expected_alpha_bps=Decimal(str(kwargs.get('expected_alpha_bps', 60.0))),  # [FIX-28] aligned with callers (was 50.0)
            signal_source=kwargs.get('signal_source', 'fusion'),
            confidence=confidence,
            regime=regime,
            regime_confidence=kwargs.get('regime_confidence', 0.5),
            drl_weight=kwargs.get('drl_weight', 0.5),
            dvol_current=kwargs.get('dvol_current', 0.0),
            dvol_zscore=kwargs.get('dvol_zscore', 0.0),
            volume_ratio=kwargs.get('volume_ratio', 1.0),
            is_structure_breakout=kwargs.get('is_structure_breakout', False),
            price_direction=str(kwargs.get('price_direction', 'flat') or 'flat'),
        )
        
        return self.evaluate(proposal, **kwargs)


# =============================================================================
# SINGLETON
# =============================================================================

_trade_gate_instance: Optional[TradeGate] = None


def get_trade_gate(config: Optional[TradeGateConfig] = None) -> TradeGate:
    """获取 TradeGate 单例。

    Refresh the singleton when a different config is explicitly supplied later
    in the same process so callers do not silently keep the first config.
    """
    global _trade_gate_instance
    if _trade_gate_instance is None:
        _trade_gate_instance = TradeGate(config or TradeGateConfig())
    elif config is not None and _trade_gate_instance.config != config:
        logger.info("TradeGate config changed; refreshing singleton instance")
        _trade_gate_instance = TradeGate(config)
    return _trade_gate_instance


def reset_trade_gate():
    """重置单例"""
    global _trade_gate_instance
    _trade_gate_instance = None

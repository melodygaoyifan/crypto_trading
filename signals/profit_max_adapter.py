"""
================================================================================
HMATS v6.5 - Profit Max Adapter (P0+P1+Macro/Crowd Unified Integration)
================================================================================
Purpose: Unified adapter for all profit-max improvements

This module integrates:
- P0 #1: Adaptive regime quality (not hard skip, but budget/threshold tuning)
- P0 #3: Phase-aware filtering (not hard ban, but confirmation tightening)
- P0 #7: Loss streak degradation curve (tiered reduction before halt)
- P0 #12: Extreme funding bias (not pause, but directional modulation)
- P1 #6: False breakout veto (HARD veto only for breakout/retest)
- P1 #8: Signal quality score (conviction multiplier, execution urgency)
- P1 #9: Regime transition risk (stability for tightening add-on)
- v6.4.2: Deterministic sentiment integration
- v6.5: Macro/Crowd context integration (TASK E3)

PROFIT-MAX PHILOSOPHY:
- Don't kill trades, modulate them
- Small positions can capture rare reversals
- Hard vetos ONLY for:
  1. FALSE_BREAKOUT (breakout/retest entry types only)
  2. LOSS_STREAK_HALT (6+ consecutive losses)
- LOW_SIGNAL_QUALITY must NEVER veto, only apply soft penalties
- Macro/Crowd data NEVER vetoes, only modulates sizing/urgency/addon

PATCH v6.5 CHANGES:
- Added macro context integration (event window, macro_shock)
- Added crowd context integration (extreme crowding, panic)
- Macro event window reduces urgency and blocks addon at high risk
- Macro shock scales size based on alignment with price
- Extreme crowding blocks addon and increases false breakout strictness
- NO NEW HARD GATES - only modulation

Author: HMATS v6.5 Upgrade
Version: 6.5
================================================================================
"""

import logging
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Dict, Optional, Any, List, Tuple
from enum import Enum

logger = logging.getLogger("ProfitMaxAdapter")


# =============================================================================
# VETO SOURCE ENUM (v6.4.1b)
# =============================================================================

class VetoSource(Enum):
    """
    Explicit enumeration of allowed veto sources.
    
    CRITICAL: Only these sources may set result.veto = True:
    - FALSE_BREAKOUT: High-confidence false breakout on breakout/retest entries
    - LOSS_STREAK_HALT: 6+ consecutive losses (existing halt logic)
    
    All other conditions MUST use soft modulation only.
    LOW_SIGNAL_QUALITY is explicitly NOT a veto source.
    """
    NONE = "none"
    FALSE_BREAKOUT = "false_breakout"
    LOSS_STREAK_HALT = "loss_streak_halt"


# =============================================================================
# IMPORTS (with safe fallbacks)
# =============================================================================

def _safe_import(module_path: str, class_name: str):
    """Safely import a class from a module."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name, None)
    except ImportError as e:
        logger.warning(f"Could not import {class_name} from {module_path}: {e}")
        return None


# P1 Modules
FalseBreakoutDetector = _safe_import("signals.false_breakout_detector", "FalseBreakoutDetector")
FalseBreakoutConfig = _safe_import("signals.false_breakout_detector", "FalseBreakoutConfig")
get_false_breakout_detector = _safe_import("signals.false_breakout_detector", "get_false_breakout_detector")

SignalQualityScorer = _safe_import("signals.signal_quality_scorer", "SignalQualityScorer")
SignalQualityScorerConfig = _safe_import("signals.signal_quality_scorer", "SignalQualityScorerConfig")
get_signal_quality_scorer = _safe_import("signals.signal_quality_scorer", "get_signal_quality_scorer")

RegimeTransitionRisk = _safe_import("signals.regime_transition_risk", "RegimeTransitionRisk")
RegimeTransitionConfig = _safe_import("signals.regime_transition_risk", "RegimeTransitionConfig")
get_regime_transition_risk = _safe_import("signals.regime_transition_risk", "get_regime_transition_risk")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ProfitMaxConfig:
    """Configuration for profit-max adapter."""
    
    # Master switch
    enabled: bool = True
    
    # P0 #1: Adaptive Regime Quality
    regime_quality_enabled: bool = True
    # Instead of hard skip, these regimes get reduced conviction
    regime_conviction_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "STRONG_TREND": 1.0,
        "TREND": 0.9,
        "NEUTRAL": 0.7,
        "VOLATILE_CHOP": 0.5,  # Was: skip entirely. Now: 50% conviction
        "MEAN_REVERT": 0.4,   # Was: skip entirely. Now: 40% conviction
        "TRANSITION": 0.6,
    })
    regime_execution_urgency_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "STRONG_TREND": 1.2,
        "TREND": 1.0,
        "NEUTRAL": 0.8,
        "VOLATILE_CHOP": 0.6,  # More passive
        "MEAN_REVERT": 0.5,
        "TRANSITION": 0.7,
    })
    
    # P0 #3: Phase-Aware Filtering
    phase_filter_enabled: bool = True
    # EXHAUSTION phase: don't hard-ban, but require confirmation
    exhaustion_requires_htf_alignment: bool = True
    exhaustion_conviction_penalty: float = 0.3  # Reduce by 30% if counter-HTF
    expansion_conviction_bonus: float = 0.1     # Boost by 10% in expansion
    
    # P0 #7: Loss Streak Degradation Curve
    loss_streak_enabled: bool = True
    # Tiered degradation before full halt
    # [UNLEASH] UL-6c: Halved degradation - short 3-loss streaks are normal in bull phases
    # Original: 2->0.7, 3->0.5, 4->0.4, 5->0.3. New: gentler curve
    loss_streak_tiers: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        # streak_count: {size_mult, min_score_add, regime_filter}
        "2": {"size_mult": 0.85, "min_score_add": 3, "require_good_regime": False},  # [UNLEASH] was 0.7/5
        "3": {"size_mult": 0.75, "min_score_add": 5, "require_good_regime": False},  # [UNLEASH] was 0.5/10
        "4": {"size_mult": 0.65, "min_score_add": 8, "require_good_regime": False},  # [UNLEASH] was 0.4/15
        "5": {"size_mult": 0.55, "min_score_add": 10, "require_good_regime": False}, # [UNLEASH] was 0.3/20
        "6": {"size_mult": 0.45, "min_score_add": 12, "require_good_regime": True},  # [UNLEASH] new tier
        "7": {"size_mult": 0.35, "min_score_add": 15, "require_good_regime": True},  # [UNLEASH] new tier
        "8": {"size_mult": 0.25, "min_score_add": 18, "require_good_regime": True},  # [UNLEASH] new tier
        "9": {"size_mult": 0.20, "min_score_add": 20, "require_good_regime": True},  # [UNLEASH] new tier
        # 10+ handled by halt logic
    })
    good_regimes_for_streak: List[str] = field(default_factory=lambda: [
        "STRONG_TREND", "TREND", "OPPORTUNITY"
    ])
    
    # P0 #12: Extreme Funding Bias
    funding_bias_enabled: bool = True
    extreme_funding_threshold: float = 0.15  # Annualized
    high_funding_threshold: float = 0.08
    # Bias modulation: positive funding + long = penalty, short = bonus
    funding_direction_bonus: float = 0.1    # 10% boost for counter-funding trades
    funding_direction_penalty: float = 0.15  # 15% penalty for crowded-side trades
    extreme_funding_urgency_reduction: float = 0.3  # Reduce urgency by 30%
    
    # P1 #6: False Breakout Detector
    false_breakout_enabled: bool = True
    # P1-2B: FalseBreakout action mode ("VETO" = hard block, "SCALE" = reduce size)
    false_breakout_action: str = "VETO"    # HIGH_RISK default: hard veto
    false_breakout_scale: float = 0.5      # When action=SCALE, multiply size by this
    false_breakout_urgency_penalty: float = 0.15  # When action=SCALE, reduce urgency
    
    # P1 #8: Signal Quality Scorer
    signal_quality_enabled: bool = True
    signal_quality_config: Dict[str, Any] = field(default_factory=dict)
    
    # P1 #9: Regime Transition Risk
    transition_risk_enabled: bool = True
    transition_risk_addon_penalty: float = 0.3  # Reduce add-on permission at high risk
    unstable_regime_urgency_reduction: float = 0.2
    
    # Loss streak halt threshold
    # [UNLEASH] UL-6c: was 6 -> 10 (short systems can easily hit 6 in bull phases)
    loss_streak_halt_threshold: int = 10  # [UNLEASH] was 6 -> 10
    
    # P1 #8: Signal Quality - SOFT penalties only (v6.4.1b)
    signal_quality_min_score_warning: float = 35.0  # Below this: apply soft penalties
    signal_quality_low_urgency_mult: float = 0.8    # Multiply urgency by this
    signal_quality_low_conviction_mult: float = 0.85  # Multiply conviction by this
    
    # Entry type inference (v6.4.1b)
    entry_type_inference_enabled: bool = True
    breakout_atr_threshold: float = 1.2  # Price move > 1.2*ATR from level = breakout
    
    # Anti-floor-drift tracking (v6.4.1b)
    anti_floor_drift_enabled: bool = True
    conviction_floor: float = 0.3       # Minimum conviction multiplier
    floor_drift_window: int = 10        # Track last N evaluations
    floor_drift_threshold: int = 5      # If >= N consecutive floor hits, apply penalty
    floor_drift_urgency_penalty: float = 0.7  # Extra urgency reduction on floor drift
    
    # v6.4.2: Deterministic Sentiment Integration
    sentiment_enabled: bool = True
    sentiment_direction_alignment_bonus: float = 0.1    # +10% when aligned
    sentiment_direction_misalignment_penalty: float = 0.05  # -5% when misaligned (NOT veto)
    sentiment_crowding_urgency_reduction: float = 0.2   # Reduce urgency when crowded
    sentiment_extreme_crowding_size_mult: float = 0.7   # 70% size on extreme crowding
    sentiment_low_strength_threshold: float = 0.3
    sentiment_low_strength_urgency_reduction: float = 0.15
    
    # v6.5: Macro/Crowd Integration (TASK E3)
    # CRITICAL: These are MODULATIONS only, NO new hard gates
    macro_crowd_enabled: bool = True
    
    # Macro event window modulation
    macro_event_window_urgency_mult: float = 0.6      # At macro_risk=1, urgency = 60%
    macro_event_window_addon_threshold: float = 0.6   # Block addon if macro_risk >= 0.6
    
    # Macro shock scaling
    macro_shock_aligned_size_mult: float = 1.15       # +15% size when shock aligned
    macro_shock_misaligned_size_mult: float = 0.85    # -15% size when shock misaligned
    macro_shock_threshold: float = 0.4                # Shock must be >= 0.4 to apply
    
    # Crowd extreme crowding modulation
    crowd_extreme_crowding_addon: bool = False        # No add-ons when extreme
    crowd_extreme_crowding_urgency_mult: float = 0.8
    crowd_extreme_crowding_strictness_add: int = 1    # +1 to false breakout strictness
    
    # Crowd high crowding
    crowd_high_crowding_threshold: float = 0.6
    crowd_high_crowding_urgency_mult: float = 0.9
    
    # Panic modulation
    crowd_panic_threshold: float = 0.7
    crowd_panic_urgency_mult: float = 0.85
    crowd_panic_strictness_add: int = 1
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProfitMaxConfig':
        """Load from config dict."""
        pm = data.get("profit_max", {})
        defaults = cls()
        kwargs = {
            f.name: pm.get(f.name, getattr(defaults, f.name))
            for f in dataclass_fields(cls)
        }
        return cls(**kwargs)


# =============================================================================
# FEE CONFIG (S16)
# =============================================================================

@dataclass
class FeeConfig:
    """Fee-aware signal gating. Supports Spot + Futures dual-market."""
    fee_aware_enabled: bool = True
    asset_market_type: Dict[str, str] = field(default_factory=lambda: {
        "BTC": "futures", "ETH": "futures", "SOL": "spot",
    })
    spot_fee_free_volume: float = 10_000.0
    spot_maker_bps: float = 25.0
    spot_taker_bps: float = 40.0
    futures_maker_bps: float = 2.0
    futures_taker_bps: float = 5.0
    fee_alpha_threshold_mult: float = 1.5
    fee_low_alpha_urgency_mult: float = 0.5


# =============================================================================
# ADAPTER RESULT
# =============================================================================

@dataclass
class ProfitMaxResult:
    """Result from profit-max adapter."""
    
    # Trade decision modifiers
    veto: bool = False
    veto_reason: str = ""
    veto_source: VetoSource = VetoSource.NONE
    
    # Size/conviction modulation
    conviction_multiplier: float = 1.0
    size_multiplier: float = 1.0
    
    # Execution parameters
    execution_urgency: float = 1.0
    prefer_limit: bool = False
    limit_offset_bps: float = 0.0
    
    # Add-on permission
    allow_add: bool = True
    
    # Quality score
    signal_quality_score: float = 50.0
    
    # Anti-floor-drift tracking (v6.4.1b)
    floor_drift_active: bool = False
    consecutive_floor_hits: int = 0
    
    # Entry type inference (v6.4.1b)
    inferred_entry_type: str = ""
    
    # Warnings and breakdowns
    warnings: List[str] = field(default_factory=list)
    breakdown: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "veto": self.veto,
            "veto_reason": self.veto_reason,
            "veto_source": self.veto_source.value,
            "conviction_multiplier": self.conviction_multiplier,
            "size_multiplier": self.size_multiplier,
            "execution_urgency": self.execution_urgency,
            "prefer_limit": self.prefer_limit,
            "limit_offset_bps": self.limit_offset_bps,
            "allow_add": self.allow_add,
            "signal_quality_score": self.signal_quality_score,
            "floor_drift_active": self.floor_drift_active,
            "consecutive_floor_hits": self.consecutive_floor_hits,
            "inferred_entry_type": self.inferred_entry_type,
            "warnings": self.warnings,
            "breakdown": self.breakdown,
        }


# =============================================================================
# ENTRY TYPE INFERENCE (v6.4.1b)
# =============================================================================

def infer_entry_type(
    price_series: Optional[List[float]],
    breakout_level: Optional[float],
    atr: Optional[float],
    direction: int,
    threshold_mult: float = 1.2
) -> str:
    """
    Infer entry_type when not explicitly provided by main.py.
    
    Heuristic:
    1. If breakout_level provided and price moved > threshold*ATR from it -> "breakout"
    2. If price is near breakout_level (within 0.5*ATR) -> "retest"
    3. Otherwise -> "momentum" (default)
    
    Args:
        price_series: Recent prices (most recent last)
        breakout_level: Key level that was/is being broken
        atr: Average True Range
        direction: Trade direction (1=long, -1=short)
        threshold_mult: ATR multiplier for breakout classification
        
    Returns:
        Inferred entry type string: "breakout", "breakdown", "retest", or "momentum"
    """
    if not price_series or len(price_series) < 2:
        return "momentum"
    
    current_price = price_series[-1]
    
    # If no breakout level, can't be breakout/retest
    if breakout_level is None:
        return "momentum"
    
    # Safe ATR default
    if atr is None or atr <= 0:
        atr = abs(current_price) * 0.02  # 2% default
    
    # Distance from breakout level
    distance = current_price - breakout_level
    distance_atr = abs(distance) / atr if atr > 0 else 0
    
    # Check if price has moved significantly past the level
    if direction > 0:  # Long
        # For long, breakout is when price is above level
        if distance > 0 and distance_atr > threshold_mult:
            return "breakout"
        elif abs(distance_atr) < 0.5:
            return "retest"
    else:  # Short
        # For short (breakdown), price should be below level
        if distance < 0 and distance_atr > threshold_mult:
            return "breakdown"
        elif abs(distance_atr) < 0.5:
            return "retest"
    
    return "momentum"


# =============================================================================
# PROFIT MAX ADAPTER IMPLEMENTATION
# =============================================================================

class ProfitMaxAdapter:
    """
    Unified adapter for all profit-max improvements.
    
    This adapter processes market context and returns modulation parameters
    that affect sizing, execution, and add-on decisions.
    
    CRITICAL VETO POLICY (v6.4.1b):
    - ONLY VetoSource.FALSE_BREAKOUT and VetoSource.LOSS_STREAK_HALT may set veto=True
    - LOW_SIGNAL_QUALITY NEVER sets veto=True, only applies soft penalties:
      * allow_add = False
      * urgency_multiplier *= 0.8
      * conviction_multiplier *= 0.85
      * Add warning to result.warnings
    """
    
    def __init__(self, config: ProfitMaxConfig = None, fee_config: FeeConfig = None):
        """Initialize adapter with all sub-modules."""
        self.config = config or ProfitMaxConfig()
        self.fee_config = fee_config or FeeConfig()

        # Initialize P1 modules
        self._false_breakout_detector = None
        self._signal_quality_scorer = None
        self._transition_risk_tracker = None
        
        if self.config.enabled:
            self._init_modules()
        
        # Loss streak state (updated externally)
        self._current_loss_streak = 0
        
        # Anti-floor-drift tracking (v6.4.1b)
        self._floor_hit_history: List[bool] = []
        self._consecutive_floor_hits = 0
        
        logger.info(
            f"[PROFIT_MAX] Initialized v6.4.1b: enabled={self.config.enabled}, "
            f"false_breakout={self.config.false_breakout_enabled}, "
            f"signal_quality={self.config.signal_quality_enabled}, "
            f"anti_floor_drift={self.config.anti_floor_drift_enabled}"
        )
    
    def _init_modules(self):
        """Initialize sub-modules with safe fallbacks."""
        if self.config.false_breakout_enabled and FalseBreakoutDetector:
            try:
                self._false_breakout_detector = FalseBreakoutDetector()
            except Exception as e:
                logger.warning(f"[PROFIT_MAX] FalseBreakoutDetector init failed: {e}")
        
        if self.config.signal_quality_enabled and SignalQualityScorer:
            try:
                sq_config = None
                if SignalQualityScorerConfig:
                    sq_config = SignalQualityScorerConfig.from_dict({
                        "signal_quality": self.config.signal_quality_config or {}
                    })
                self._signal_quality_scorer = SignalQualityScorer(config=sq_config)
            except Exception as e:
                logger.warning(f"[PROFIT_MAX] SignalQualityScorer init failed: {e}")
        
        if self.config.transition_risk_enabled and RegimeTransitionRisk:
            try:
                self._transition_risk_tracker = RegimeTransitionRisk()
            except Exception as e:
                logger.warning(f"[PROFIT_MAX] RegimeTransitionRisk init failed: {e}")

    # -- [S16] Fee estimation --------------------------------------------------

    def _estimate_round_trip_fee_bps(
        self, asset: str, prefer_limit: bool = True,
        monthly_volume_used: float = 0.0,
    ) -> float:
        """Estimate round-trip fee in bps for a given asset and market type."""
        fc = self.fee_config
        if not fc.fee_aware_enabled:
            return 0.0

        market_type = fc.asset_market_type.get(asset, "spot")

        if market_type == "futures":
            one_way = fc.futures_maker_bps if prefer_limit else fc.futures_taker_bps
            return one_way * 2

        # Spot: check free tier
        if monthly_volume_used < fc.spot_fee_free_volume:
            return 0.0
        if prefer_limit:
            return fc.spot_maker_bps * 2
        return fc.spot_taker_bps + fc.spot_maker_bps

    def evaluate(
        self,
        # Market context
        regime: str = "UNKNOWN",
        regime_confidence: float = 0.5,
        phase: str = "UNKNOWN",
        volatility_regime: str = "NORMAL",
        
        # Signal context
        direction: int = 0,  # 1=long, -1=short
        entry_type: str = "momentum",  # breakout, retest, momentum, mean_revert
        signal_direction: float = 0.0,
        
        # HTF context
        htf_trend_direction: float = 0.0,
        
        # Funding/crowding
        funding_rate: float = 0.0,
        
        # Price/volume context
        price_series: Optional[List[float]] = None,
        volume_series: Optional[List[float]] = None,
        atr: Optional[float] = None,
        volume_ratio: float = 1.0,
        breakout_level: Optional[float] = None,
        rsi: Optional[float] = None,
        
        # Loss streak
        loss_streak: int = 0,
        
        # v6.4.2: Deterministic Sentiment
        sentiment: Optional[Dict] = None,
        
        # v6.5: Macro/Crowd Context
        macro: Optional[Dict] = None,
        crowd: Optional[Dict] = None,
        
        **kwargs
    ) -> ProfitMaxResult:
        """
        Evaluate market context and return profit-max modulations.
        
        VETO POLICY (v6.4.1b):
        - Only FALSE_BREAKOUT and LOSS_STREAK_HALT (6+) may veto
        - LOW_SIGNAL_QUALITY applies SOFT penalties only (never veto)
        
        Returns:
            ProfitMaxResult with veto decision and modulation parameters
        """
        if not self.config.enabled:
            return ProfitMaxResult()
        
        result = ProfitMaxResult()
        result.breakdown = {}
        result.warnings = []
        
        # Track loss streak
        self._current_loss_streak = loss_streak
        
        # ==== STEP 0: Entry Type Inference (v6.4.1b) ====
        effective_entry_type = entry_type
        if self.config.entry_type_inference_enabled:
            needs_inference = (
                entry_type is None or
                entry_type == "" or
                entry_type.lower() in ["unknown", "generic", "auto", "momentum"]
            )
            
            if needs_inference and breakout_level is not None:
                inferred = infer_entry_type(
                    price_series=price_series,
                    breakout_level=breakout_level,
                    atr=atr,
                    direction=direction,
                    threshold_mult=self.config.breakout_atr_threshold
                )
                if inferred != "momentum":
                    effective_entry_type = inferred
                    result.inferred_entry_type = inferred
                    result.breakdown["entry_type_inference"] = {
                        "original": entry_type,
                        "inferred": inferred,
                    }
                    logger.debug(f"[PROFIT_MAX] Inferred entry_type: {entry_type} -> {inferred}")
        
        # ==== STEP 1: Check Loss Streak HALT (only allowed veto besides false breakout) ====
        if self.config.loss_streak_enabled and loss_streak >= self.config.loss_streak_halt_threshold:
            result.veto = True
            result.veto_source = VetoSource.LOSS_STREAK_HALT
            result.veto_reason = (
                f"[LOSS_STREAK_HALT] streak={loss_streak} >= "
                f"{self.config.loss_streak_halt_threshold}: Trading halted"
            )
            logger.warning(f"[PROFIT_MAX] {result.veto_reason}")
            return result  # Early exit on veto
        
        # ==== STEP 2: Update regime transition tracker ====
        transition_stability = 0.5
        if self._transition_risk_tracker:
            try:
                trans_result = self._transition_risk_tracker.update(regime)
                transition_stability = trans_result.stability
                result.breakdown["transition_risk"] = trans_result.to_dict()
            except Exception as e:
                logger.debug(f"[PROFIT_MAX] Transition risk update failed: {e}")
        
        # ==== STEP 3: Signal Quality Score ====
        # CRITICAL: Signal quality NEVER vetoes, only applies soft penalties
        if self._signal_quality_scorer:
            try:
                sq_result = self._signal_quality_scorer.score(
                    regime=regime,
                    regime_confidence=regime_confidence,
                    trend_direction=htf_trend_direction,
                    signal_direction=signal_direction,
                    volatility_regime=volatility_regime,
                    volume_ratio=volume_ratio,
                    transition_stability=transition_stability,
                    funding_rate=funding_rate,
                    signal_side=direction,
                    asset=kwargs.get("asset"),
                )
                result.signal_quality_score = sq_result.score
                result.conviction_multiplier *= sq_result.conviction_multiplier
                result.execution_urgency *= sq_result.execution_urgency
                result.allow_add = sq_result.allow_add
                result.breakdown["signal_quality"] = sq_result.to_dict()
                
                # ==== LOW SIGNAL QUALITY SOFT PENALTIES (v6.4.1b) ====
                # IMPORTANT: This is NOT a veto, only soft modulation
                if sq_result.score < self.config.signal_quality_min_score_warning:
                    # Apply soft penalties
                    result.allow_add = False
                    result.execution_urgency *= self.config.signal_quality_low_urgency_mult
                    result.conviction_multiplier *= self.config.signal_quality_low_conviction_mult
                    
                    warning_msg = (
                        f"[LOW_SIGNAL_QUALITY] score={sq_result.score:.1f} < "
                        f"{self.config.signal_quality_min_score_warning:.1f}: "
                        f"Soft penalties applied (NO VETO). "
                        f"conv×{self.config.signal_quality_low_conviction_mult}, "
                        f"urg×{self.config.signal_quality_low_urgency_mult}, "
                        f"allow_add=False"
                    )
                    result.warnings.append(warning_msg)
                    result.breakdown["low_signal_quality_penalty"] = {
                        "score": sq_result.score,
                        "threshold": self.config.signal_quality_min_score_warning,
                        "conviction_penalty": self.config.signal_quality_low_conviction_mult,
                        "urgency_penalty": self.config.signal_quality_low_urgency_mult,
                        "veto": False,  # EXPLICIT: Never veto for signal quality
                    }
                    logger.info(f"[PROFIT_MAX] {warning_msg}")
                    
            except Exception as e:
                logger.debug(f"[PROFIT_MAX] Signal quality scoring failed: {e}")
        
        # ==== M5 FIX: Pre-compute crowd strictness for false breakout detection ====
        _crowd_strictness_add = 0
        if crowd:
            if crowd.get("extreme_crowding", False):
                _crowd_strictness_add += self.config.crowd_extreme_crowding_strictness_add
            if crowd.get("panic_score", 0.0) >= self.config.crowd_panic_threshold:
                _crowd_strictness_add += self.config.crowd_panic_strictness_add

        # ==== STEP 4: False breakout detection (ONLY HARD VETO SOURCE for entries) ====
        if self._false_breakout_detector and self.config.false_breakout_enabled:
            try:
                _fb_context = None
                if _crowd_strictness_add > 0:
                    _fb_context = {"strictness_add": _crowd_strictness_add}

                fb_result = self._false_breakout_detector.detect(
                    price_series=price_series or [],
                    atr=atr,
                    volume_series=volume_series,
                    rsi_or_momentum=rsi,
                    breakout_level=breakout_level,
                    direction=direction,
                    entry_type=effective_entry_type,  # Use inferred type
                    context_tags=_fb_context,
                )
                result.breakdown["false_breakout"] = fb_result.to_dict()
                
                if fb_result.is_false_breakout:
                    # P1-2B: Profile-gated action - VETO (default) or SCALE
                    if self.config.false_breakout_action == "SCALE":
                        # SCALE mode: reduce size instead of hard veto
                        scale = self.config.false_breakout_scale
                        result.size_multiplier *= scale
                        result.execution_urgency -= self.config.false_breakout_urgency_penalty
                        result.execution_urgency = max(result.execution_urgency, 0.1)
                        result.breakdown["false_breakout_action"] = {
                            "action": "SCALE",
                            "scale": scale,
                            "urgency_penalty": self.config.false_breakout_urgency_penalty,
                            "confidence": fb_result.confidence,
                            "reasons": fb_result.reasons,
                        }
                        logger.info(
                            f"[PROFIT_MAX] FALSE_BREAKOUT_SCALE: "
                            f"size*={scale}, urg-={self.config.false_breakout_urgency_penalty}, "
                            f"conf={fb_result.confidence:.2f}, "
                            f"entry_type={effective_entry_type}, "
                            f"reasons={fb_result.reasons}"
                        )
                        # NO early return - continue through soft modulation steps
                    else:
                        # VETO mode (HIGH_RISK default): hard block
                        result.veto = True
                        result.veto_source = VetoSource.FALSE_BREAKOUT
                        result.veto_reason = (
                            f"[FALSE_BREAKOUT_VETO] confidence={fb_result.confidence:.2f}, "
                            f"entry_type={effective_entry_type}, "
                            f"reasons={fb_result.reasons}"
                        )
                        logger.warning(f"[PROFIT_MAX] {result.veto_reason}")
                        return result  # Early exit on veto
            except Exception as e:
                logger.debug(f"[PROFIT_MAX] False breakout detection failed: {e}")
        
        # ==== STEP 5: P0 #1 - Adaptive Regime Quality ====
        if self.config.regime_quality_enabled:
            regime_upper = regime.upper()
            regime_conv_mult = self.config.regime_conviction_multipliers.get(
                regime_upper, 0.7
            )
            regime_urg_mult = self.config.regime_execution_urgency_multipliers.get(
                regime_upper, 0.8
            )
            result.conviction_multiplier *= regime_conv_mult
            result.execution_urgency *= regime_urg_mult
            result.breakdown["regime_quality"] = {
                "regime": regime,
                "conviction_mult": regime_conv_mult,
                "urgency_mult": regime_urg_mult,
            }
        
        # ==== STEP 6: P0 #3 - Phase-Aware Filtering ====
        if self.config.phase_filter_enabled:
            phase_upper = phase.upper()
            
            if phase_upper == "EXHAUSTION":
                # Check HTF alignment
                is_aligned = (
                    (htf_trend_direction > 0 and direction > 0) or
                    (htf_trend_direction < 0 and direction < 0)
                )
                
                if not is_aligned and self.config.exhaustion_requires_htf_alignment:
                    # Counter-HTF in exhaustion: heavy penalty (NOT veto)
                    result.conviction_multiplier *= (1.0 - self.config.exhaustion_conviction_penalty)
                    result.execution_urgency *= 0.7
                    result.breakdown["phase_filter"] = {
                        "phase": phase,
                        "htf_aligned": False,
                        "penalty_applied": self.config.exhaustion_conviction_penalty,
                    }
                else:
                    # Aligned with HTF in exhaustion: small penalty
                    result.conviction_multiplier *= 0.9
                    result.breakdown["phase_filter"] = {
                        "phase": phase,
                        "htf_aligned": True,
                        "penalty_applied": 0.1,
                    }
            
            elif phase_upper == "EXPANSION":
                # Expansion phase: bonus
                result.conviction_multiplier *= (1.0 + self.config.expansion_conviction_bonus)
                result.breakdown["phase_filter"] = {
                    "phase": phase,
                    "bonus_applied": self.config.expansion_conviction_bonus,
                }
        
        # ==== STEP 7: P0 #7 - Loss Streak Degradation (soft tiers, halt handled above) ====
        if self.config.loss_streak_enabled and loss_streak >= 2:
            streak_key = str(min(loss_streak, 9))  # [UNLEASH] UL-6c: was 5 -> 9 (extended tiers)
            tier = self.config.loss_streak_tiers.get(streak_key)
            
            if tier:
                result.size_multiplier *= tier["size_mult"]
                
                # Check regime requirement
                if tier.get("require_good_regime"):
                    if regime.upper() not in self.config.good_regimes_for_streak:
                        result.allow_add = False
                        result.conviction_multiplier *= 0.5
                
                result.breakdown["loss_streak"] = {
                    "streak": loss_streak,
                    "size_mult": tier["size_mult"],
                    "min_score_add": tier["min_score_add"],
                    "required_good_regime": tier.get("require_good_regime", False),
                }
        
        # ==== STEP 8: P0 #12 - Extreme Funding Bias ====
        if self.config.funding_bias_enabled and abs(funding_rate) > 0.01:
            abs_funding = abs(funding_rate)
            
            # Determine if trade is going with or against crowding
            is_crowded_side = (
                (funding_rate > 0 and direction > 0) or  # Long when longs pay
                (funding_rate < 0 and direction < 0)      # Short when shorts pay
            )
            
            if abs_funding >= self.config.extreme_funding_threshold:
                # Extreme funding - soft modulation only
                if is_crowded_side:
                    result.conviction_multiplier *= (1.0 - self.config.funding_direction_penalty * 2)
                    result.execution_urgency *= (1.0 - self.config.extreme_funding_urgency_reduction)
                else:
                    result.conviction_multiplier *= (1.0 + self.config.funding_direction_bonus)
                
                result.prefer_limit = True
                result.limit_offset_bps = 5.0
                
            elif abs_funding >= self.config.high_funding_threshold:
                # High but not extreme
                if is_crowded_side:
                    result.conviction_multiplier *= (1.0 - self.config.funding_direction_penalty)
                else:
                    result.conviction_multiplier *= (1.0 + self.config.funding_direction_bonus * 0.5)
            
            result.breakdown["funding_bias"] = {
                "funding_rate": funding_rate,
                "is_crowded_side": is_crowded_side,
                "extreme": abs_funding >= self.config.extreme_funding_threshold,
            }
        
        # ==== STEP 9: P1 #9 - Transition Risk Modulation ====
        if self.config.transition_risk_enabled and transition_stability < 0.4:
            # Unstable regime - tighten add-on (NOT veto)
            result.allow_add = False
            result.execution_urgency *= (1.0 - self.config.unstable_regime_urgency_reduction)
            result.breakdown["transition_instability"] = {
                "stability": transition_stability,
                "addon_blocked": True,
            }
        
        # ==== STEP 9.5: v6.4.2 Deterministic Sentiment Modulation ====
        # PROFIT-MAX: Sentiment affects sizing/urgency, NEVER vetoes
        if self.config.sentiment_enabled and sentiment:
            sentiment_direction = sentiment.get("direction_score", 0.0)
            sentiment_strength = sentiment.get("strength", 0.5)
            crowding_bias = crowd.get("crowding_bias", 0.0) if crowd else 0.0
            extreme_crowding = crowd.get("extreme_crowding", False) if crowd else False
            
            # 1. Direction alignment effect (NOT veto)
            if direction != 0:
                alignment = sentiment_direction * direction
                if alignment > 0.3:
                    # Strongly aligned - small bonus
                    result.conviction_multiplier *= (1.0 + self.config.sentiment_direction_alignment_bonus)
                elif alignment < -0.3:
                    # Misaligned - small penalty (NOT veto)
                    result.conviction_multiplier *= (1.0 - self.config.sentiment_direction_misalignment_penalty)
                    result.warnings.append(
                        f"[SENTIMENT] Direction misalignment: sent={sentiment_direction:.2f}, trade={direction}"
                    )
            
            # 2. Extreme crowding effect (NOT veto)
            if extreme_crowding:
                result.size_multiplier *= self.config.sentiment_extreme_crowding_size_mult
                result.execution_urgency *= (1.0 - self.config.sentiment_crowding_urgency_reduction)
                result.prefer_limit = True
                result.warnings.append(
                    f"[SENTIMENT] Extreme crowding: bias={crowding_bias:.2f}"
                )
            elif abs(crowding_bias) > 0.5:
                # Moderate crowding
                result.execution_urgency *= (1.0 - self.config.sentiment_crowding_urgency_reduction * 0.5)
            
            # 3. Low strength effect
            if sentiment_strength < self.config.sentiment_low_strength_threshold:
                result.execution_urgency *= (1.0 - self.config.sentiment_low_strength_urgency_reduction)
            
            result.breakdown["sentiment"] = {
                "direction_score": sentiment_direction,
                "strength": sentiment_strength,
                "crowding_bias": crowding_bias,
                "extreme_crowding": extreme_crowding,
            }
        
        # ==== STEP 9.6: v6.5 Macro/Crowd Modulation (TASK E3) ====
        # CRITICAL: These are MODULATIONS only, NO new hard gates
        # Only false-breakout may veto trades
        if self.config.macro_crowd_enabled:
            # Default values if not provided
            macro = macro or {}
            crowd = crowd or {}
            
            # === Macro Event Window Modulation ===
            event_window = macro.get("event_window", {})
            if event_window.get("active", False):
                macro_risk = macro.get("macro_risk", 0.5)
                
                # Reduce urgency: urgency *= (1 - 0.4 * macro_risk)
                urgency_factor = 1.0 - (1.0 - self.config.macro_event_window_urgency_mult) * macro_risk
                result.execution_urgency *= urgency_factor
                
                # Block add-on at high risk
                if macro_risk >= self.config.macro_event_window_addon_threshold:
                    result.allow_add = False
                    result.warnings.append(
                        f"[MACRO] Event window: risk={macro_risk:.2f}, add-on blocked"
                    )
                
                # Prefer limit orders during events
                result.prefer_limit = True
                
                result.breakdown["macro_event"] = {
                    "active": True,
                    "name": event_window.get("name"),
                    "macro_risk": macro_risk,
                    "urgency_factor": urgency_factor,
                }
            
            # === Macro Shock Scaling ===
            macro_shock = macro.get("macro_shock", 0.0)
            if abs(macro_shock) >= self.config.macro_shock_threshold and direction != 0:
                # Check alignment: shock direction vs trade direction
                shock_direction = 1 if macro_shock > 0 else -1
                aligned = (shock_direction == direction)
                
                if aligned:
                    result.size_multiplier *= self.config.macro_shock_aligned_size_mult
                else:
                    result.size_multiplier *= self.config.macro_shock_misaligned_size_mult
                    result.warnings.append(
                        f"[MACRO] Shock misaligned: shock={macro_shock:.2f}, direction={direction}"
                    )
                
                result.breakdown["macro_shock"] = {
                    "shock": macro_shock,
                    "aligned": aligned,
                }
            
            # === Crowd Extreme Crowding Modulation ===
            crowd_extreme = crowd.get("extreme_crowding", False)
            if crowd_extreme:
                result.allow_add = self.config.crowd_extreme_crowding_addon
                result.execution_urgency *= self.config.crowd_extreme_crowding_urgency_mult
                result.prefer_limit = True
                
                # M5 FIX: strictness_add now pre-computed and passed to
                # FalseBreakoutDetector via context_tags in STEP 4 above
                result.breakdown["crowd_extreme"] = {
                    "extreme_crowding": True,
                    "addon_allowed": result.allow_add,
                    "strictness_add": _crowd_strictness_add,
                }
                result.warnings.append("[CROWD] Extreme crowding: strictness increased")
            
            # === Crowd High Crowding Modulation ===
            crowding_bias = crowd.get("crowding_bias", 0.0)
            if abs(crowding_bias) >= self.config.crowd_high_crowding_threshold and not crowd_extreme:
                result.execution_urgency *= self.config.crowd_high_crowding_urgency_mult
                
                result.breakdown["crowd_high"] = {
                    "crowding_bias": crowding_bias,
                }
            
            # === Panic Score Modulation ===
            panic_score = crowd.get("panic_score", 0.0)
            if panic_score >= self.config.crowd_panic_threshold:
                result.execution_urgency *= self.config.crowd_panic_urgency_mult
                
                result.breakdown["crowd_panic"] = {
                    "panic_score": panic_score,
                }
                result.warnings.append(f"[CROWD] High panic: score={panic_score:.2f}")
        
        # ==== STEP 10: Clamp values to allowed ranges ====
        conviction_floor = self.config.conviction_floor
        result.conviction_multiplier = max(conviction_floor, min(2.0, result.conviction_multiplier))
        result.size_multiplier = max(0.2, min(1.5, result.size_multiplier))
        result.execution_urgency = max(0.3, min(2.0, result.execution_urgency))
        
        # ==== STEP 11: Anti-Floor-Drift Tracking (v6.4.1b) ====
        if self.config.anti_floor_drift_enabled:
            hit_floor = (result.conviction_multiplier <= conviction_floor + 0.01)
            self._update_floor_drift_tracking(hit_floor)
            result.consecutive_floor_hits = self._consecutive_floor_hits
            
            if self._consecutive_floor_hits >= self.config.floor_drift_threshold:
                # Apply additional soft penalties for floor drift (NO VETO)
                result.floor_drift_active = True
                result.execution_urgency *= self.config.floor_drift_urgency_penalty
                
                result.breakdown["floor_drift"] = {
                    "consecutive_hits": self._consecutive_floor_hits,
                    "threshold": self.config.floor_drift_threshold,
                    "urgency_penalty": self.config.floor_drift_urgency_penalty,
                }
                
                warning_msg = (
                    f"[FLOOR_DRIFT] {self._consecutive_floor_hits} consecutive floor hits: "
                    f"Additional urgency penalty applied (NO VETO). Consider reducing trade frequency."
                )
                result.warnings.append(warning_msg)
                logger.info(f"[PROFIT_MAX] {warning_msg}")
        
        # ==== STEP 12: Final Logging ====
        if result.conviction_multiplier < 0.7 or result.size_multiplier < 0.7:
            logger.info(
                f"[PROFIT_MAX] Modulation: conv={result.conviction_multiplier:.2f}, "
                f"size={result.size_multiplier:.2f}, urg={result.execution_urgency:.2f}, "
                f"add={result.allow_add}, score={result.signal_quality_score:.1f}, "
                f"floor_drift={result.floor_drift_active}"
            )
        
        return result
    
    def _update_floor_drift_tracking(self, hit_floor: bool):
        """
        Track consecutive times conviction hits the floor.
        
        Used to detect systematic underperformance patterns.
        """
        self._floor_hit_history.append(hit_floor)
        
        # Keep window size
        if len(self._floor_hit_history) > self.config.floor_drift_window:
            self._floor_hit_history.pop(0)
        
        # Count consecutive hits from end
        consecutive = 0
        for hit in reversed(self._floor_hit_history):
            if hit:
                consecutive += 1
            else:
                break
        
        self._consecutive_floor_hits = consecutive
    
    def update_loss_streak(self, streak: int):
        """Update current loss streak count."""
        self._current_loss_streak = streak
    
    def reset_floor_drift_tracking(self):
        """Reset floor drift tracking (e.g., after a winning trade)."""
        self._floor_hit_history.clear()
        self._consecutive_floor_hits = 0
        logger.debug("[PROFIT_MAX] Floor drift tracking reset")
    
    # --- [v9-PATCH-9] ConvexSpikeEngine: pyramid into winners ---
    def _check_convex_spike(
        self, asset: str, unrealized_pnl_pct: float, pyramid_count: int = 0
    ) -> Optional[ProfitMaxResult]:
        """If position is significantly profitable, allow pyramiding."""
        _PATCH_9_ACTIVE = False  # Advisory-first: set True after paper validation

        if unrealized_pnl_pct > 0.05 and pyramid_count < 1:  # >5% profit, max 1 add
            logger.info(
                f"[v9-PATCH-9] CONVEX_SPIKE: {asset} pnl={unrealized_pnl_pct:.1%} -> "
                f"{'allow pyramid' if _PATCH_9_ACTIVE else 'would allow pyramid (advisory)'}"
            )
            if _PATCH_9_ACTIVE:
                return ProfitMaxResult(
                    allow_add=True,
                    size_multiplier=1.5,
                    conviction_multiplier=1.2,
                    prefer_limit=True,
                    execution_urgency=0.7,
                    signal_quality_score=70.0,
                )
        return None
    # --- end [v9-PATCH-9] ---

    def get_stats(self) -> Dict:
        """Get adapter statistics."""
        stats = {
            "enabled": self.config.enabled,
            "current_loss_streak": self._current_loss_streak,
            "consecutive_floor_hits": self._consecutive_floor_hits,
            "floor_drift_active": self._consecutive_floor_hits >= self.config.floor_drift_threshold,
        }
        
        if self._false_breakout_detector:
            stats["false_breakout"] = self._false_breakout_detector.get_stats()
        
        if self._signal_quality_scorer:
            stats["signal_quality"] = self._signal_quality_scorer.get_stats()
        
        if self._transition_risk_tracker:
            stats["transition_risk"] = self._transition_risk_tracker.get_stats()
        
        return stats


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_adapter_instance: Optional[ProfitMaxAdapter] = None


def get_profit_max_adapter(config: ProfitMaxConfig = None) -> ProfitMaxAdapter:
    """Get or create the singleton adapter instance."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = ProfitMaxAdapter(config)
    elif config is not None and _adapter_instance.config != config:
        logger.info("ProfitMaxAdapter config changed; refreshing singleton instance")
        _adapter_instance = ProfitMaxAdapter(config)
    return _adapter_instance


def reset_profit_max_adapter():
    """Reset the singleton instance."""
    global _adapter_instance
    _adapter_instance = None

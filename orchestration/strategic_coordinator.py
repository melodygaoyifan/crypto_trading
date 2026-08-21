"""
================================================================================
HMATS v6.0.0 - Strategic Coordinator (?
================================================================================
Purpose: 

?
1. Two-Stage Intelligence (multi_asset_intelligence.py)
   -> ?/??
   
2. Adaptive Strategy Weights (adaptive_strategy_weight_manager.py)
   -> 
   
3. Ensemble Regime Classifier (regime_classifier.py)
   -> ?

INTEGRATION POINTS:
- ?integration decide() ?Two-Stage  strategic_intent
- ?fusion ?adaptive weights
- ?intent ?regime filter

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# --- [v9-PATCH-5a] Import correlation controller ---
try:
    from risk.correlation_realtime_controller import (
        CorrelationRealtimeController,
        get_correlation_controller,
    )
    CORRELATION_CONTROLLER_AVAILABLE = True
except ImportError:
    CORRELATION_CONTROLLER_AVAILABLE = False
# --- end [v9-PATCH-5a] ---


# =============================================================================
# STRATEGIC INTENT (Two-Stage )
# =============================================================================

class StrategicDirection(Enum):
    """Strategic direction."""
    STRONG_BEARISH = -2
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1
    STRONG_BULLISH = 2


@dataclass
class AssetIntent:
    """Per-asset strategic intent."""
    asset: str
    direction: StrategicDirection
    strength: float  # 0-1
    confidence: float  # 0-1
    recommended_exposure: float  # 0-1
    scenarios: List[Dict] = field(default_factory=list)


@dataclass
class StrategicIntent:
    """Strategic intent produced by the Two-Stage model."""
    timestamp: datetime
    market_regime: str  # "risk_on" | "risk_off" | "mixed"
    
    # ?
    asset_intents: Dict[str, AssetIntent] = field(default_factory=dict)
    
    # ?
    correlations: Dict[str, float] = field(default_factory=dict)
    portfolio_allocation: Dict[str, float] = field(default_factory=dict)
    
    # 
    overall_direction: StrategicDirection = StrategicDirection.NEUTRAL
    overall_confidence: float = 0.0
    
    # 
    key_insight: str = ""
    
    def get_intent_for_asset(self, asset: str) -> Optional[AssetIntent]:
        return self.asset_intents.get(asset)


# =============================================================================
# SHORT FILTER RESULT
# =============================================================================

@dataclass
class ShortFilterResult:
    """Result of short-intent filtering."""
    allow_new_short: bool = True
    allow_add_short: bool = True
    force_reduce_short: bool = False
    max_short_exposure: float = 1.0
    
    reasons: List[str] = field(default_factory=list)
    
    # 
    regime_is_bullish: bool = False
    squeeze_risk_high: bool = False
    funding_rate_extreme: bool = False


# =============================================================================
# STRATEGIC COORDINATOR
# =============================================================================

class StrategicCoordinator:
    """
    ?- ?
    
    Usage:
        coordinator = StrategicCoordinator()
        
        # 
        intent = coordinator.generate_strategic_intent(market_data, prices)
        
        # ?fusion ?
        weights = coordinator.get_adjusted_weights(base_weights)
        
        # 
        filter_result = coordinator.filter_short_intent(asset, direction, market_data)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # ?(pre-init ALL attributes before _init_modules)
        self._two_stage = None
        self._two_stage_light = None
        self._weight_manager = None
        self._fusion_integration = None
        self._regime_classifier = None
        self._MarketRegime = None
        self._RegimeFeatures = None
        
        self._init_modules()
        
        # ?
        self._last_strategic_intent: Optional[StrategicIntent] = None
        self._last_regime_result = None
        self._intent_valid_until: Optional[datetime] = None

        # --- [v9-PATCH-5a-init] Init correlation controller ---
        self._correlation_controller = None
        if CORRELATION_CONTROLLER_AVAILABLE:
            try:
                self._correlation_controller = get_correlation_controller()
                logger.info("StrategicCoordinator: correlation_controller connected")
            except Exception as e:
                logger.warning(f"StrategicCoordinator: correlation_controller init failed: {e}")
        # --- end [v9-PATCH-5a-init] ---

        logger.info("StrategicCoordinator initialized")
    
    def _init_modules(self):
        """"""
        
        # 1. Two-Stage Intelligence
        try:
            from market.multi_asset_intelligence import (
                MultiAssetHeavyModel,
                MultiAssetLightModel,
                MultiAssetConfig,
            )
            config = MultiAssetConfig()
            self._two_stage = MultiAssetHeavyModel(config)
            self._two_stage_light = MultiAssetLightModel(config)
            logger.info("[OK] Two-Stage Intelligence connected")
        except Exception as e:
            logger.warning(f"Two-Stage Intelligence not available: {e}")
            self._two_stage = None
        
        # 2. Adaptive Strategy Weights
        try:
            from signals.adaptive_weight_v521 import (
                AdaptiveWeightManagerV521,
                AuthorityFusionWeightAdapter,
            )
            self._weight_manager = AdaptiveWeightManagerV521()
            self._fusion_integration = AuthorityFusionWeightAdapter(self._weight_manager)
            logger.info("[OK] Adaptive Strategy Weights connected (v521)")
        except Exception as e:
            logger.warning(f"Adaptive Strategy Weights not available: {e}")
            self._weight_manager = None
            self._fusion_integration = None
        
        # 3. Ensemble Regime Classifier
        try:
            from training.regime.regime_classifier import (
                EnsembleRegimeClassifier,
                RegimeClassifierConfig,
                RegimeFeatures,
                MarketRegime,
            )
            self._regime_classifier = EnsembleRegimeClassifier()
            self._MarketRegime = MarketRegime
            self._RegimeFeatures = RegimeFeatures
            logger.info("[OK] Ensemble Regime Classifier connected")
        except ImportError:
            # [P343] EXPECTED in the container, not a fault. The module is in
            # the repo but is deliberately NOT copied into Dockerfile.engine
            # (P214): its consumer at :595 has therefore never executed, and
            # arming a never-run decision path on a live account is an operator
            # call (P141), not a dependency fix. Logged at INFO naming the
            # decision so it stops reading as a broken import -- an expected
            # absence reported as a fault is how a real one stops being read
            # (P202/P240), and P336 gave the same treatment to the two sibling
            # archived imports while leaving this one behind.
            logger.info(
                "Ensemble Regime Classifier NOT SHIPPED by decision "
                "(training/regime/regime_classifier.py is excluded from "
                "Dockerfile.engine, P214) -- the classify() consumer stays "
                "inert; shipping it is an operator decision, not a fix")
            self._regime_classifier = None
        except Exception as e:
            # a real fault: the module imported and then failed to construct
            logger.warning(f"Ensemble Regime Classifier failed to init: {e}")
            self._regime_classifier = None
    
    # =========================================================================
    # TWO-STAGE INTELLIGENCE
    # =========================================================================
    
    def generate_strategic_intent(
        self,
        asset_data: Dict[str, Any],
        asset_features: Dict[str, Dict],
        positions: Dict = None,
        force_refresh: bool = False,
    ) -> StrategicIntent:
        """
         (Stage 1 Heavy Model)
        
        ?4H ?
        
        Returns:
            StrategicIntent :
            - //?
            - ?
            - 
        """
        now = datetime.now(timezone.utc)
        
        # ?
        if not force_refresh and self._last_strategic_intent:
            if self._intent_valid_until and now < self._intent_valid_until:
                logger.debug("Using cached strategic intent")
                return self._last_strategic_intent
        
        #  Two-Stage 
        if self._two_stage is None:
            return self._generate_fallback_intent(asset_features)
        
        try:
            #  Heavy Model
            portfolio_state = self._two_stage.analyze(
                asset_data=asset_data,
                asset_features=asset_features,
                positions=positions,
            )
            
            # ?StrategicIntent
            intent = self._convert_to_strategic_intent(portfolio_state)
            
            #  Light Model
            if self._two_stage_light:
                self._two_stage_light.update_strategy(portfolio_state)
            
            # 
            self._last_strategic_intent = intent
            self._intent_valid_until = now + timedelta(hours=4)
            
            logger.info(
                f"Strategic intent generated: regime={intent.market_regime}, "
                f"direction={intent.overall_direction.name}, "
                f"confidence={intent.overall_confidence:.2f}"
            )
            
            return intent
            
        except Exception as e:
            logger.error(f"Two-Stage analysis failed: {e}")
            return self._generate_fallback_intent(asset_features)
    
    def _convert_to_strategic_intent(self, portfolio_state) -> StrategicIntent:
        """?Two-Stage ?StrategicIntent"""
        regime = getattr(portfolio_state, 'market_regime', 'mixed') if portfolio_state else 'mixed'
        intent = StrategicIntent(timestamp=datetime.now(timezone.utc), market_regime=regime)

        if portfolio_state is None:
            return intent
        
        # ?
        for asset, state in portfolio_state.asset_states.items():
            direction = self._convert_direction(state.directional_bias)
            
            asset_intent = AssetIntent(
                asset=asset,
                direction=direction,
                strength=abs(state.directional_bias),
                confidence=state.conviction,
                recommended_exposure=state.recommended_allocation,
                scenarios=state.scenarios if hasattr(state, 'scenarios') else [],
            )
            intent.asset_intents[asset] = asset_intent
        
        #  - map PortfolioStrategicState fields to StrategicIntent
        intent.correlations = {
            'btc_eth': getattr(portfolio_state, 'correlation_btc_eth', 0.0),
            'btc_sol': getattr(portfolio_state, 'correlation_btc_sol', 0.0),
            'eth_sol': getattr(portfolio_state, 'correlation_eth_sol', 0.0),
        }
        intent.portfolio_allocation = getattr(portfolio_state, 'allocation_weights', {})
        
        # 
        intent.overall_direction, intent.overall_confidence = self._aggregate_direction(
            intent.asset_intents
        )
        
        return intent
    
    def _convert_direction(self, bias: float) -> StrategicDirection:
        """"""
        if bias <= -0.6:
            return StrategicDirection.STRONG_BEARISH
        elif bias <= -0.2:
            return StrategicDirection.BEARISH
        elif bias >= 0.6:
            return StrategicDirection.STRONG_BULLISH
        elif bias >= 0.2:
            return StrategicDirection.BULLISH
        else:
            return StrategicDirection.NEUTRAL
    
    def _aggregate_direction(
        self,
        asset_intents: Dict[str, AssetIntent],
    ) -> Tuple[StrategicDirection, float]:
        """"""
        if not asset_intents:
            return StrategicDirection.NEUTRAL, 0.0
        
        # 
        total_weight = 0
        weighted_direction = 0
        weighted_confidence = 0
        
        for asset, intent in asset_intents.items():
            weight = intent.recommended_exposure
            weighted_direction += intent.direction.value * weight * intent.confidence
            weighted_confidence += intent.confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_direction = weighted_direction / total_weight
            avg_confidence = weighted_confidence / total_weight
        else:
            avg_direction = 0
            avg_confidence = 0
        
        # ?
        if avg_direction <= -1.5:
            direction = StrategicDirection.STRONG_BEARISH
        elif avg_direction <= -0.5:
            direction = StrategicDirection.BEARISH
        elif avg_direction >= 1.5:
            direction = StrategicDirection.STRONG_BULLISH
        elif avg_direction >= 0.5:
            direction = StrategicDirection.BULLISH
        else:
            direction = StrategicDirection.NEUTRAL
        
        return direction, avg_confidence
    
    def _generate_fallback_intent(self, asset_features: Dict) -> StrategicIntent:
        """Fallback: RSI/MACD direction + real-time correlation constraints. [v9-PATCH-5b]"""
        intent = StrategicIntent(timestamp=datetime.now(timezone.utc), market_regime="mixed")

        # --- [v9-PATCH-5b] Correlation-aware fallback ---
        # 1. Real-time correlation matrix
        corr_penalty = 1.0
        asset_adjustments = {}
        if self._correlation_controller:
            try:
                corr_matrix = self._correlation_controller.compute_correlation_matrix("short")
                if corr_matrix:
                    intent.correlations = {
                        'btc_eth': corr_matrix.get(("BTC", "ETH"), 0.0),
                        'btc_sol': corr_matrix.get(("BTC", "SOL"), 0.0),
                        'eth_sol': corr_matrix.get(("ETH", "SOL"), 0.0),
                    }
                _positions = getattr(self, '_corr_portfolio', None)
                adjustment = self._correlation_controller.get_position_adjustment(_positions)
                corr_penalty = getattr(adjustment, 'scale_factor', 1.0)
                asset_adjustments = getattr(adjustment, 'asset_adjustments', {})
            except Exception as e:
                logger.debug(f"[v9-PATCH-5b] Correlation fallback: {e}")
                corr_penalty = 1.0

        # 2. Per-asset direction + exposure
        confidences = {}
        for asset, features in asset_features.items():
            rsi = features.get("rsi", 50)
            macd = features.get("macd_histogram", 0)

            if rsi < 30 and macd < 0:
                direction = StrategicDirection.BEARISH
                strength = 0.6
            elif rsi > 70 and macd > 0:
                direction = StrategicDirection.BULLISH
                strength = 0.6
            elif rsi < 40:
                direction = StrategicDirection.BEARISH
                strength = 0.4
            elif rsi > 60:
                direction = StrategicDirection.BULLISH
                strength = 0.4
            else:
                direction = StrategicDirection.NEUTRAL
                strength = 0.3

            base_exposure = 0.3
            exposure = base_exposure * corr_penalty
            if asset in asset_adjustments:
                exposure *= asset_adjustments[asset]

            conf = max(0.3, min(1.0, strength))
            confidences[asset] = conf

            intent.asset_intents[asset] = AssetIntent(
                asset=asset,
                direction=direction,
                strength=strength,
                confidence=conf,
                recommended_exposure=exposure,
            )

        # 3. Winner-Take-Most allocation (confidence weighting)
        weighted = {a: c ** 2 for a, c in confidences.items()}
        total_w = sum(weighted.values())
        if total_w > 0:
            intent.portfolio_allocation = {a: w / total_w for a, w in weighted.items()}
        else:
            n = max(1, len(confidences))
            intent.portfolio_allocation = {a: 1.0 / n for a in confidences}

        # 4. Same-direction overload check
        directions = [ai.direction for ai in intent.asset_intents.values()]
        dir_names = [d.name for d in directions]
        if len(dir_names) > 1 and len(set(dir_names)) == 1 and dir_names[0] != "NEUTRAL":
            for ai in intent.asset_intents.values():
                ai.recommended_exposure *= 0.7
            logger.info(f"[v9-PATCH-5b] SAME_DIR_OVERLOAD: all {dir_names[0]} -> exposure x 0.7")
        # --- end [v9-PATCH-5b] ---

        intent.overall_direction, intent.overall_confidence = self._aggregate_direction(
            intent.asset_intents
        )

        return intent
    
    # =========================================================================
    # ADAPTIVE STRATEGY WEIGHTS
    # =========================================================================
    
    def get_adjusted_weights(
        self,
        base_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        
        
         authority_fusion 
        
        Args:
            base_weights:  {strategy_name: weight}
        
        Returns:
            
        """
        if self._fusion_integration is None:
            return base_weights
        
        try:
            adjusted = self._fusion_integration.get_adjusted_weights(base_weights)
            
            # 
            changed = {k: (base_weights.get(k, 0), v) 
                      for k, v in adjusted.items() 
                      if abs(base_weights.get(k, 0) - v) > 0.01}
            
            if changed:
                logger.info(f"Weights adjusted: {changed}")
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Weight adjustment failed: {e}")
            return base_weights
    
    def weighted_signal_fusion(
        self,
        signals: Dict[str, Tuple[float, float]],
    ) -> Tuple[float, float]:
        """
        
        
        Args:
            signals: {strategy: (direction, confidence)}
        
        Returns:
            (fused_direction, fused_confidence)
        """
        if self._fusion_integration is None:
            # ?
            if not signals:
                return 0.0, 0.0
            
            total_conf = sum(c for _, c in signals.values())
            if total_conf == 0:
                return 0.0, 0.0
            
            fused_dir = sum(d * c for d, c in signals.values()) / total_conf
            fused_conf = total_conf / len(signals)
            return fused_dir, fused_conf
        
        try:
            return self._fusion_integration.weighted_fusion(signals)
        except Exception as e:
            logger.error(f"Signal fusion failed: {e}")
            return 0.0, 0.0
    
    def record_trade_completed(
        self,
        strategy: str,
        pnl: float,
        pnl_pct: float,
        duration_hours: float,
    ):
        """Record a completed trade for adaptive weight learning.

        [P47 2026-04-25] Renamed from `record_trade_result` to match the
        single caller site at `core/execution_service.py:2171`. The mismatch
        was silent — caller wrapped the call in try/except, swallowing the
        AttributeError at debug level. P15 (FIXED 2026-04-24) said this
        was wired but the call has been failing for every trade since.
        Result: `_weight_manager.record_trade()` never invoked → all
        strategies stayed at compute_weight()=1.0 forever.
        """
        if self._weight_manager is None:
            return
        
        try:
            self._weight_manager.record_trade(
                strategy_name=strategy,
                pnl_pct=pnl_pct,
                is_winner=(pnl >= 0),
            )
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
    
    # =========================================================================
    # REGIME-BASED SHORT FILTER
    # =========================================================================
    
    def filter_short_intent(
        self,
        asset: str,
        intended_direction: int,  # -1 = short
        market_data: Dict[str, float],
        squeeze_risk: float = 0.0,
        funding_rate: float = 0.0,
        # [P355] `cvd` is a real classifier input and it lives in
        # agent_signals, not market_data. Optional and defaulted so every
        # existing caller is unchanged; when it is absent `cvd` simply joins
        # the MISSING list rather than being fabricated as 0.
        agent_signals: Optional[Dict[str, Any]] = None,
    ) -> ShortFilterResult:
        """
        ?
        
        ?intent ?
        
        :
        1. Regime  -> 
        2. ?-> /
        3.  -> 
        
        Args:
            asset: 
            intended_direction:  (-1=short, 1=long)
            market_data: 
            squeeze_risk:  (0-1)
            funding_rate: 
        
        Returns:
            ShortFilterResult
        """
        result = ShortFilterResult()
        
        # ?
        if intended_direction >= 0:
            return result
        
        # 1. Regime ?
        if self._regime_classifier is not None:
            try:
                features, _missing = self._build_regime_features(
                    asset, market_data, agent_signals)
                # [P355] REFUSE ON A MOSTLY-FABRICATED VECTOR. Resolving the
                # names is only half the fix: without this, a feed outage
                # silently restores the exact state P347 found — a classifier
                # scoring defaults and a veto firing on the result. Three
                # inputs have no producer anywhere and are always absent, so
                # the floor is set against the 13 that are obtainable.
                _absent = len(_missing)
                if _absent > self._REGIME_MAX_ABSENT_INPUTS:
                    if not getattr(self, "_regime_absent_warned", False):
                        self._regime_absent_warned = True
                        logger.warning(
                            "[REGIME-CLASSIFIER] %s: %d of %d inputs have no "
                            "value this tick (%s) — REFUSING to classify "
                            "rather than score a fabricated-neutral vector "
                            "(P355). The short filter contributes nothing "
                            "while this holds.",
                            asset, _absent, self._REGIME_TOTAL_INPUTS,
                            ", ".join(sorted(_missing)[:8]))
                    return result
                self._regime_absent_warned = False
                regime_result = self._regime_classifier.classify(features)
                self._last_regime_result = regime_result
                
                #  regime
                if regime_result.regime.value == "bear_rally":
                    # Bear Rally = ?
                    result.max_short_exposure *= 0.7
                    result.reasons.append("Bear rally detected - reduced short exposure")
                
                #  bearish_score 
                if regime_result.bearish_score < 0.3:
                    result.regime_is_bullish = True
                    result.allow_new_short = False
                    result.max_short_exposure = 0.3
                    result.reasons.append(
                        f"Bullish regime (bearish_score={regime_result.bearish_score:.2f}) - "
                        "new shorts blocked"
                    )
                
            except Exception as e:
                logger.error(f"Regime classification failed: {e}")
        
        # 2.  Strategic Intent ?
        if self._last_strategic_intent:
            intent = self._last_strategic_intent.get_intent_for_asset(asset)
            if intent and intent.direction in [
                StrategicDirection.BULLISH,
                StrategicDirection.STRONG_BULLISH,
            ]:
                if intent.confidence > 0.7:
                    result.regime_is_bullish = True
                    result.allow_new_short = False
                    result.reasons.append(
                        f"Two-Stage signals bullish ({intent.direction.name}) with "
                        f"confidence {intent.confidence:.2f} - new shorts blocked"
                    )
        
        # 3. ?
        if squeeze_risk > 0.8:
            result.squeeze_risk_high = True
            result.force_reduce_short = True
            result.allow_new_short = False
            result.allow_add_short = False
            result.max_short_exposure = 0.0
            result.reasons.append(
                f"Extreme squeeze risk ({squeeze_risk:.2f}) - force flatten shorts"
            )
        elif squeeze_risk > 0.6:
            result.squeeze_risk_high = True
            result.allow_add_short = False
            result.max_short_exposure = min(result.max_short_exposure, 0.5)
            result.reasons.append(
                f"High squeeze risk ({squeeze_risk:.2f}) - no new shorts, reduce exposure"
            )
        
        # 4. ?
        # Coinglass returns per-8h funding rate.
        # Normal: 0.01-0.05% (0.0001-0.0005)
        # High:   0.05-0.30% (0.0005-0.003)
        # Extreme: >0.30% (>0.003)
        # Old threshold 0.003 (0.3%) was too low - blocked all shorts
        # in normal high-vol markets. Kraken spot mode doesn't pay
        # funding directly, so these are proxy signals only.
        # [WIRE-6] Funding cost does NOT block trades - advisory signal only.
        # Positive funding = shorts EARN carry, so blocking is counterproductive.
        # Only log extreme rates as informational.
        if funding_rate > 0.01:  # 1.0% per 8h (~456% annualized)
            result.funding_rate_extreme = True
            # DO NOT block: result.allow_new_short = False  [WIRE-6 removed]
            # DO NOT cap: result.max_short_exposure = 0.3   [WIRE-6 removed]
            result.reasons.append(
                f"INFO: Extreme funding rate ({funding_rate:.4%}) - shorts earn carry"
            )
        elif funding_rate > 0.005:  # 0.5% per 8h
            result.reasons.append(
                f"INFO: High funding rate ({funding_rate:.4%}) - shorts earn carry"
            )
        
        # 
        if result.reasons:
            logger.warning(
                f"[SHORT FILTER] {asset}: allow_new={result.allow_new_short}, "
                f"max_exposure={result.max_short_exposure:.2f}, "
                f"reasons={result.reasons}"
            )
        
        return result
    
    # [P355] The names this classifier reads were never the names the pipeline
    # writes. P347 measured 8 of its 10 market/technical inputs with NO
    # producer anywhere, so every `.get(key, default)` returned the default on
    # every tick: the model scored a market in which the price had not moved on
    # any horizon, MACD was flat, Bollinger sat mid-band and CVD was zero — the
    # same fabricated-neutral vector whatever the market was doing (P2/P223).
    # A control fed defaults is not a conservative control, it is a random one
    # with a safety-sounding name.
    #
    # Resolved at the READER, because that is where the mismatch is — not by
    # publishing a second set of aliases into market_data, since a second name
    # for one quantity is how the first one stops being read (P172).
    #
    # NAME-ONLY: the pipeline already computes exactly this quantity.
    _REGIME_INPUT_ALIASES = {
        "price_return_1h": ("price_change_1h_pct", "market_data"),
        "price_return_4h": ("price_change_4h_pct", "market_data"),
        "rsi_14": ("rsi", "market_data"),
        "macd_histogram": ("macd_hist", "market_data"),
        "volume_ratio": ("volume_ratio", "market_data"),
        "volatility_4h": ("volatility_4h", "market_data"),
        "funding_rate": ("funding_rate", "market_data"),
        "open_interest_change": ("oi_change_24h_pct", "market_data"),
        "fear_greed_index": ("fear_greed_value", "market_data"),
        "cvd": ("cvd_divergence", "agent_signals"),
    }
    # DERIVED from real series already present — see _resolve_regime_inputs.
    _REGIME_INPUT_DERIVED = ("price_return_24h", "volatility_24h", "bb_position")
    # NO SOURCE ANYWHERE, deliberately left ABSENT rather than defaulted: an
    # invented value here is indistinguishable from a measured one, which is
    # the whole defect. `volatility_1h` needs an hourly price SERIES and the
    # pipeline keeps only the last hourly CHANGE (P306); `basis` exists only
    # inside the calbasis shadow ledger; `social_sentiment` has no producer.
    _REGIME_INPUT_UNAVAILABLE = ("volatility_1h", "basis", "social_sentiment")
    _REGIME_TOTAL_INPUTS = 16
    # 3 are permanently unavailable, so this admits at most 2 further gaps
    # (a transient feed outage) before the vector stops being a measurement.
    _REGIME_MAX_ABSENT_INPUTS = 5

    @staticmethod
    def _resolve_regime_inputs(market_data, agent_signals=None):
        """[P355] Real values for the classifier's inputs, plus what is MISSING.

        Returns ``(values, missing)``. `values` carries only inputs actually
        resolved from a producer; the caller supplies the dataclass default for
        anything in `missing` and — crucially — can SEE how many it is
        supplying, which the old `.get(key, default)` form could not.
        """
        sig = agent_signals or {}
        out = {}
        missing = []

        def _num(v):
            try:
                f = float(v)
            except (TypeError, ValueError):  # noqa: silent-swallow — a value
                # that will not coerce is "the producer gave us nothing
                # usable", which is precisely what the MISSING list is for.
                # Logging per field per tick would be wallpaper (P202), and
                # the caller already reports the whole missing set once.
                return None
            if f != f or f in (float("inf"), float("-inf")):
                return None
            return f

        for field_name, (src_key, where) in (
                StrategicCoordinator._REGIME_INPUT_ALIASES.items()):
            src_dict = market_data if where == "market_data" else sig
            v = _num(src_dict.get(src_key))
            if v is None:
                missing.append(field_name)
            else:
                out[field_name] = v

        prices = market_data.get("prices") or []
        try:
            prices = [float(x) for x in prices]
        except (TypeError, ValueError):
            prices = []

        # 24h == 6 bars of 4H; 7 closes are needed to form the return.
        if len(prices) >= 7 and prices[-7] > 0:
            out["price_return_24h"] = (prices[-1] - prices[-7]) / prices[-7]
        else:
            missing.append("price_return_24h")

        if len(prices) >= 7:
            rets = [(prices[i] - prices[i - 1]) / prices[i - 1]
                    for i in range(len(prices) - 6, len(prices))
                    if prices[i - 1] > 0]
            if len(rets) >= 2:
                mean = sum(rets) / len(rets)
                var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
                out["volatility_24h"] = var ** 0.5
            else:
                missing.append("volatility_24h")
        else:
            missing.append("volatility_24h")

        _u = _num(market_data.get("bb_upper"))
        _l = _num(market_data.get("bb_lower"))
        _c = _num(market_data.get("current_price"))
        if _u is not None and _l is not None and _c is not None and _u > _l:
            out["bb_position"] = max(0.0, min(1.0, (_c - _l) / (_u - _l)))
        else:
            missing.append("bb_position")

        missing.extend(StrategicCoordinator._REGIME_INPUT_UNAVAILABLE)
        return out, missing

    def _build_regime_features(self, asset, market_data, agent_signals=None):
        """[P355] Build RegimeFeatures from REAL values; report what is absent.

        Returns ``(features, missing)`` so the caller can refuse to act on a
        vector that is mostly defaults.
        """
        if self._RegimeFeatures is None:
            return None, ["RegimeFeatures unavailable"]

        vals, missing = self._resolve_regime_inputs(market_data, agent_signals)
        g = vals.get
        features = self._RegimeFeatures(
            timestamp=datetime.now(timezone.utc),
            asset=asset,
            price_return_1h=g("price_return_1h", 0),
            price_return_4h=g("price_return_4h", 0),
            price_return_24h=g("price_return_24h", 0),
            volatility_1h=g("volatility_1h", 0.02),
            volatility_4h=g("volatility_4h", 0.04),
            volatility_24h=g("volatility_24h", 0.08),
            rsi_14=g("rsi_14", 50),
            macd_histogram=g("macd_histogram", 0),
            bb_position=g("bb_position", 0.5),
            volume_ratio=g("volume_ratio", 1.0),
            cvd=g("cvd", 0),
            funding_rate=g("funding_rate", 0),
            open_interest_change=g("open_interest_change", 0),
            basis=g("basis", 0),
            fear_greed_index=g("fear_greed_index", 50),
            social_sentiment=g("social_sentiment", 0),
        )
        return features, missing

    # =========================================================================
    # COMBINED DECISION WRAPPER
    # =========================================================================
    
    def pre_decision_check(
        self,
        asset: str,
        market_data: Dict[str, Any],
        agent_signals: Dict[str, Any],
        position_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        ?- 
        
        ?integration decide() 
        
        Returns:
            Dict with:
            - strategic_intent: Two-Stage 
            - adjusted_weights: 
            - short_filter:  ()
            - prior_direction: ?
            - prior_confidence: 
        """
        result = {
            'strategic_intent': None,
            'adjusted_weights': {},
            'short_filter': None,
            'prior_direction': 0,
            'prior_confidence': 0.5,
        }
        
        # 1. Two-Stage Intent
        asset_features = {
            asset: {
                'rsi': market_data.get('rsi', 50),
                'macd_histogram': market_data.get('macd_histogram', 0),
                'bb_position': market_data.get('bb_position', 0.5),
                'volume_ratio': market_data.get('volume_ratio', 1.0),
            }
        }
        
        strategic_intent = self.generate_strategic_intent(
            asset_data={},  # 
            asset_features=asset_features,
            positions=position_state,
        )
        result['strategic_intent'] = strategic_intent
        
        # 
        asset_intent = strategic_intent.get_intent_for_asset(asset)
        if asset_intent:
            result['prior_direction'] = asset_intent.direction.value / 2  #  [-1, 1]
            result['prior_confidence'] = asset_intent.confidence
        
        # 2. Adjusted Weights
        base_weights = {
            'quant_agent': 1.0,
            'drl_agent': 0.5,
            'sentiment_agent': 1.0,
            'microstructure_agent': 0.4,
        }
        result['adjusted_weights'] = self.get_adjusted_weights(base_weights)
        
        # 3. Short Filter (?
        intended_direction = agent_signals.get('quant_direction', 0)
        if intended_direction < 0:
            result['short_filter'] = self.filter_short_intent(
                asset=asset,
                intended_direction=intended_direction,
                market_data=market_data,
                squeeze_risk=agent_signals.get('squeeze_risk', 0),
                funding_rate=market_data.get('funding_rate', 0),
                agent_signals=agent_signals,  # [P355] carries `cvd`
            )

        # --- [v9-PATCH-4] Leverage gating ---
        _PATCH_4_ACTIVE = False  # Advisory-first: set True after paper validation
        # [P22 2026-04-24] Was hardcoded 2.0; consolidated to canonical_config.MAX_LEVERAGE (3.0)
        # to prevent silent under-sizing if _PATCH_4_ACTIVE is ever flipped on.
        try:
            from configs.canonical_config import MAX_LEVERAGE
        except Exception:
            MAX_LEVERAGE = 3.0
        try:
            total_exposure = sum(
                abs(pos.get('notional_usd', 0) if isinstance(pos, dict) else getattr(pos, 'notional_usd', 0))
                for pos in (position_state.values() if isinstance(position_state, dict) else [])
            )
            equity = market_data.get('account_equity', 0)
            if equity > 0 and total_exposure / equity > MAX_LEVERAGE * 0.9:
                headroom = max(0, MAX_LEVERAGE * equity - total_exposure)
                max_allowed = headroom / equity if equity > 0 else 0
                asset_intent = strategic_intent.get_intent_for_asset(asset)
                if asset_intent and asset_intent.recommended_exposure > max_allowed:
                    logger.warning(
                        f"[v9-PATCH-4] LEVERAGE_GATE: {asset} exposure "
                        f"{asset_intent.recommended_exposure:.2f} -> capped to "
                        f"{max_allowed:.2f} (leverage={total_exposure/equity:.1f}x)"
                    )
                    if _PATCH_4_ACTIVE:
                        asset_intent.recommended_exposure = max_allowed
        except Exception as e:
            logger.debug(f"[v9-PATCH-4] Leverage gating skipped: {e}")
        # --- end [v9-PATCH-4] ---

        # --- [v9-PATCH-11b] Volatility regime -> mode switch ---
        _PATCH_11_ACTIVE = False  # Advisory-first: set True after paper validation
        try:
            vol_14d = market_data.get('volatility_14d', None)
            if vol_14d is not None:
                HIGH_VOL_THRESHOLD = 0.05  # 5% daily std dev
                LOW_VOL_THRESHOLD = 0.01

                if vol_14d > HIGH_VOL_THRESHOLD:
                    logger.info(
                        f"[v9-PATCH-11b] VOL_REGIME: vol={vol_14d:.3f} > "
                        f"{HIGH_VOL_THRESHOLD} -> HIGH_VOL mode, exposure halved"
                    )
                    if _PATCH_11_ACTIVE:
                        result['vol_regime'] = 'HIGH_VOL'
                        asset_intent = strategic_intent.get_intent_for_asset(asset)
                        if asset_intent:
                            asset_intent.recommended_exposure *= 0.5
                elif vol_14d < LOW_VOL_THRESHOLD:
                    logger.info(
                        f"[v9-PATCH-11b] VOL_REGIME: vol={vol_14d:.3f} < "
                        f"{LOW_VOL_THRESHOLD} -> exposure x 1.3"
                    )
                    if _PATCH_11_ACTIVE:
                        result['vol_regime'] = 'LOW_VOL'
                        asset_intent = strategic_intent.get_intent_for_asset(asset)
                        if asset_intent:
                            asset_intent.recommended_exposure = min(
                                asset_intent.recommended_exposure * 1.3, 0.5
                            )
        except Exception as e:
            logger.debug(f"[v9-PATCH-11b] Vol regime check skipped: {e}")
        # --- end [v9-PATCH-11b] ---

        return result
    
    def get_status(self) -> Dict:
        """Get coordinator status."""
        return {
            'two_stage_available': self._two_stage is not None,
            'weight_manager_available': self._weight_manager is not None,
            'regime_classifier_available': self._regime_classifier is not None,
            'last_intent_timestamp': (
                self._last_strategic_intent.timestamp.isoformat()
                if self._last_strategic_intent else None
            ),
            'last_regime': (
                self._last_regime_result.regime.value
                if self._last_regime_result else None
            ),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_coordinator: Optional[StrategicCoordinator] = None


def get_strategic_coordinator(config: Optional[Dict] = None) -> StrategicCoordinator:
    """Get or create the strategic coordinator singleton."""
    global _coordinator
    normalized_config = dict(config or {})
    if _coordinator is None:
        _coordinator = StrategicCoordinator(normalized_config)
    elif getattr(_coordinator, "config", {}) != normalized_config:
        logger.info("StrategicCoordinator config changed; refreshing singleton instance")
        _coordinator = StrategicCoordinator(normalized_config)
    return _coordinator


def reset_coordinator():
    """Reset coordinator singleton."""
    global _coordinator
    _coordinator = None

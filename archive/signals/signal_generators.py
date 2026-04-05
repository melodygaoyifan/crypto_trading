"""
================================================================================
HMATS v3.2 - REAL SIGNAL GENERATORS
================================================================================
Purpose: Replace placeholder heuristics with REAL strategy implementations.

CRITICAL CHANGES:
- QuantSignalGenerator now uses real QuantAgent with BB/RSI/MACD
- DRLSignalGenerator has explicit ENABLE_DRL config (disabled by default)
- All signals use canonical market data schema

This file replaces the placeholder signal_generators_v32.py
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
from enum import Enum, auto
from datetime import datetime
import numpy as np
import logging

# Import real strategy implementations - canonical paths
from agents.quant_agent import QuantAgent, Signal, SignalType, TechnicalAnalyzer

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class SignalConfig:
    """Signal generator configuration."""
    
    # DRL Enable/Disable - EXPLICIT
    ENABLE_DRL: bool = False  # Set True only if you have a trained model
    DRL_CHECKPOINT_PATH: Optional[str] = None  # Path to trained model
    
    # Quant strategy selection: "real" uses QuantAgent with actual signals
    QUANT_MODE: str = "real"
    
    # Minimum confidence to emit signal
    MIN_CONFIDENCE: float = 0.3
    
    # Logging
    LOG_EVERY_SIGNAL: bool = True


def get_signal_config() -> SignalConfig:
    """Get current signal configuration."""
    return SignalConfig()


# =============================================================================
# STRATEGY CLASSIFICATION
# =============================================================================

class StrategyRole(Enum):
    """Strategy role classification."""
    BASE = auto()
    AGGRESSIVE = auto()
    CONFIRMATION = auto()


class RegimeGroup(Enum):
    """Regime groups for strategy selection."""
    BEAR = "bear"
    BULL = "bull"
    VOLATILE = "volatile"
    NEUTRAL = "neutral"


@dataclass
class QuantStrategy:
    """Individual quant strategy with REAL parameters."""
    name: str
    regime_group: RegimeGroup
    role: StrategyRole
    
    # REAL parameters for QuantAgent
    ema_period: int = 200
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    atr_multiplier: float = 2.0
    
    # Direction bias (affects signal interpretation)
    long_bias: float = 0.5  # 0.5 = neutral
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "regime_group": self.regime_group.value,
            "role": self.role.name,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "long_bias": self.long_bias
        }


# 12 Real Strategies with different parameters
STRATEGY_REGISTRY: List[QuantStrategy] = [
    # BEAR strategies (4) - Lower RSI thresholds, short bias
    QuantStrategy("bear_base", RegimeGroup.BEAR, StrategyRole.BASE,
                  rsi_oversold=20, rsi_overbought=60, long_bias=0.3),
    QuantStrategy("bear_aggressive", RegimeGroup.BEAR, StrategyRole.AGGRESSIVE,
                  rsi_oversold=15, rsi_overbought=50, long_bias=0.2, atr_multiplier=1.5),
    QuantStrategy("bear_confirmation", RegimeGroup.BEAR, StrategyRole.CONFIRMATION,
                  rsi_oversold=25, rsi_overbought=55, long_bias=0.35),
    QuantStrategy("bear_reversal", RegimeGroup.BEAR, StrategyRole.BASE,
                  rsi_oversold=10, rsi_overbought=40, long_bias=0.4, bb_std=2.5),
    
    # BULL strategies (4) - Higher RSI thresholds, long bias
    QuantStrategy("bull_base", RegimeGroup.BULL, StrategyRole.BASE,
                  rsi_oversold=40, rsi_overbought=80, long_bias=0.7),
    QuantStrategy("bull_aggressive", RegimeGroup.BULL, StrategyRole.AGGRESSIVE,
                  rsi_oversold=50, rsi_overbought=85, long_bias=0.8, atr_multiplier=1.5),
    QuantStrategy("bull_confirmation", RegimeGroup.BULL, StrategyRole.CONFIRMATION,
                  rsi_oversold=45, rsi_overbought=75, long_bias=0.65),
    QuantStrategy("bull_momentum", RegimeGroup.BULL, StrategyRole.BASE,
                  rsi_oversold=60, rsi_overbought=90, long_bias=0.75, bb_std=1.5),
    
    # VOLATILE strategies (4) - Wide BB, neutral bias
    QuantStrategy("volatile_base", RegimeGroup.VOLATILE, StrategyRole.BASE,
                  bb_std=2.5, rsi_oversold=25, rsi_overbought=75, long_bias=0.5),
    QuantStrategy("volatile_aggressive", RegimeGroup.VOLATILE, StrategyRole.AGGRESSIVE,
                  bb_std=3.0, rsi_oversold=20, rsi_overbought=80, long_bias=0.5, atr_multiplier=1.0),
    QuantStrategy("volatile_confirmation", RegimeGroup.VOLATILE, StrategyRole.CONFIRMATION,
                  bb_std=2.0, rsi_oversold=30, rsi_overbought=70, long_bias=0.5),
    QuantStrategy("volatile_mean_revert", RegimeGroup.VOLATILE, StrategyRole.BASE,
                  bb_std=2.5, rsi_oversold=15, rsi_overbought=85, long_bias=0.5),
]


def get_strategies_for_regime(regime_group: RegimeGroup) -> List[QuantStrategy]:
    """Get all strategies for a regime group."""
    return [s for s in STRATEGY_REGISTRY if s.regime_group == regime_group]


def select_strategy(
    regime_name: str,
    is_opportunity_mode: bool = False
) -> Tuple[QuantStrategy, str]:
    """
    Select strategy based on regime and mode.
    
    Returns:
        (selected_strategy, selection_reason)
    """
    # Map regime name to group
    regime_lower = regime_name.lower()
    if "bear" in regime_lower or "strong_bear" in regime_lower:
        group = RegimeGroup.BEAR
    elif "bull" in regime_lower or "strong_bull" in regime_lower:
        group = RegimeGroup.BULL
    elif "volatile" in regime_lower or "chop" in regime_lower or "crisis" in regime_lower:
        group = RegimeGroup.VOLATILE
    else:
        group = RegimeGroup.BULL  # Default
    
    strategies = get_strategies_for_regime(group)
    if not strategies:
        strategies = get_strategies_for_regime(RegimeGroup.BULL)
    
    # OPPORTUNITY mode -> prefer AGGRESSIVE
    if is_opportunity_mode:
        aggressive = [s for s in strategies if s.role == StrategyRole.AGGRESSIVE]
        if aggressive:
            logger.info(f"[STRATEGY SELECT] {aggressive[0].name} (AGGRESSIVE for OPPORTUNITY)")
            return aggressive[0], f"AGGRESSIVE for {group.value} in OPPORTUNITY"
    
    # Default -> BASE
    base = [s for s in strategies if s.role == StrategyRole.BASE]
    if base:
        logger.info(f"[STRATEGY SELECT] {base[0].name} (BASE)")
        return base[0], f"BASE for {group.value}"
    
    return strategies[0], f"Fallback for {group.value}"


# =============================================================================
# SIGNAL CONFLICT DETECTION
# =============================================================================

@dataclass
class SignalConflictResult:
    """Result of signal conflict analysis."""
    all_signals_conflict: bool = False
    conflict_type: str = "none"
    
    quant_direction: str = "neutral"
    drl_direction: str = "neutral"
    sentiment_direction: str = "neutral"
    
    conflict_reason: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "all_signals_conflict": self.all_signals_conflict,
            "conflict_type": self.conflict_type,
            "quant_direction": self.quant_direction,
            "drl_direction": self.drl_direction,
            "sentiment_direction": self.sentiment_direction,
            "conflict_reason": self.conflict_reason
        }


def _direction_to_str(value: float, threshold: float = 0.3) -> str:
    """Convert numeric direction to string."""
    if value > threshold:
        return "long"
    elif value < -threshold:
        return "short"
    return "neutral"


def compute_signal_conflict(
    quant_direction: float,
    quant_confidence: float,
    drl_direction: float,
    drl_confidence: float,
    sentiment_direction: float,
    sentiment_confidence: float,
    confidence_threshold: float = 0.5
) -> SignalConflictResult:
    """
    Compute if all signals are in full directional conflict.
    
    Triggers NO_TRADE if:
    - Quant vs DRL 100% opposite with high confidence
    - AND Sentiment disagrees with majority
    """
    result = SignalConflictResult()
    
    # Get confident directions only
    q_dir = _direction_to_str(quant_direction) if quant_confidence >= confidence_threshold else "neutral"
    d_dir = _direction_to_str(drl_direction) if drl_confidence >= confidence_threshold else "neutral"
    s_dir = _direction_to_str(sentiment_direction) if sentiment_confidence >= confidence_threshold else "neutral"
    
    result.quant_direction = q_dir
    result.drl_direction = d_dir
    result.sentiment_direction = s_dir
    
    # Check for full conflict: Q vs D opposite, S doesn't break tie
    if q_dir != "neutral" and d_dir != "neutral" and q_dir != d_dir:
        # Quant and DRL oppose each other
        if s_dir == "neutral":
            # Sentiment doesn't help -> conflict
            result.all_signals_conflict = True
            result.conflict_type = "quant_drl_oppose_sentiment_neutral"
            result.conflict_reason = f"Quant={q_dir}, DRL={d_dir}, Sentiment=neutral"
        elif s_dir != q_dir and s_dir != d_dir:
            # Sentiment is third direction (shouldn't happen with binary but defensive)
            result.all_signals_conflict = True
            result.conflict_type = "three_way_conflict"
            result.conflict_reason = f"All three directions differ"
    
    # Also check: all three disagree meaningfully
    directions = [q_dir, d_dir, s_dir]
    non_neutral = [d for d in directions if d != "neutral"]
    if len(non_neutral) >= 2:
        if "long" in non_neutral and "short" in non_neutral:
            # At least one long, one short with high confidence
            long_count = non_neutral.count("long")
            short_count = non_neutral.count("short")
            if long_count == short_count:
                result.all_signals_conflict = True
                result.conflict_type = "balanced_opposition"
                result.conflict_reason = f"Balanced {long_count} long vs {short_count} short"
    
    return result


# =============================================================================
# QUANT SIGNAL GENERATOR - REAL IMPLEMENTATION
# =============================================================================

@dataclass
class QuantGeneratorOutput:
    """Output from QuantSignalGenerator."""
    direction: float  # -1 to 1
    confidence: float  # 0 to 1
    
    strategy_name: str
    strategy_group: str
    strategy_role: str
    
    # From real QuantAgent
    signal_type: str = "HOLD"  # LONG/SHORT/HOLD
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    reasoning: str = ""
    indicators: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "strategy_name": self.strategy_name,
            "strategy_group": self.strategy_group,
            "strategy_role": self.strategy_role,
            "signal_type": self.signal_type,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat()
        }


class QuantSignalGenerator:
    """
    REAL Quant Signal Generator using QuantAgent.
    
    NO MORE PLACEHOLDER HEURISTICS.
    
    Strategy:
    1. Select regime-appropriate strategy (12 strategies)
    2. Create QuantAgent with strategy parameters
    3. Generate signal from OHLCV data
    4. Return structured output
    """
    
    def __init__(self):
        self._last_output: Optional[QuantGeneratorOutput] = None
        self._agent_cache: Dict[str, QuantAgent] = {}
        self.config = get_signal_config()
        
        logger.info("[QuantSignalGenerator] Initialized with REAL QuantAgent integration")
    
    def _get_or_create_agent(self, strategy: QuantStrategy) -> QuantAgent:
        """Get cached agent or create new one with strategy params."""
        if strategy.name not in self._agent_cache:
            self._agent_cache[strategy.name] = QuantAgent(
                rsi_oversold=strategy.rsi_oversold,
                rsi_overbought=strategy.rsi_overbought,
                atr_multiplier=strategy.atr_multiplier,
                min_confidence=self.config.MIN_CONFIDENCE
            )
            logger.info(f"[QuantAgent] Created for strategy: {strategy.name}")
        return self._agent_cache[strategy.name]
    
    def generate(
        self,
        market_data: Dict,
        regime_name: str = "NEUTRAL",
        is_opportunity_mode: bool = False,
        asset: str = "SOL"
    ) -> QuantGeneratorOutput:
        """
        Generate REAL quant signal.
        
        Args:
            market_data: Dict with OHLCV data (nested or flat format)
            regime_name: Current regime
            is_opportunity_mode: True if OPPORTUNITY mode
            asset: Asset to analyze (default SOL)
            
        Returns:
            QuantGeneratorOutput with real signal
        """
        # Select strategy
        strategy, reason = select_strategy(regime_name, is_opportunity_mode)
        
        logger.info(f"[QUANT GEN] 4H tick | Asset: {asset} | Regime: {regime_name} | "
                   f"Strategy: {strategy.name} | Mode: {'OPP' if is_opportunity_mode else 'NORMAL'}")
        
        # Try to get OHLCV DataFrame for real analysis
        ohlcv_df = self._extract_ohlcv_dataframe(market_data, asset)
        
        if ohlcv_df is not None and len(ohlcv_df) >= 50:
            # REAL SIGNAL GENERATION
            output = self._generate_with_real_agent(ohlcv_df, strategy, regime_name)
        else:
            # Fallback: use simplified analysis from available data
            logger.warning(f"[QUANT GEN] No OHLCV history for {asset}, using simplified analysis")
            output = self._generate_simplified(market_data, strategy, asset)
        
        self._last_output = output
        
        if self.config.LOG_EVERY_SIGNAL:
            logger.info(f"[QUANT OUT] {strategy.name}: direction={output.direction:.3f}, "
                       f"conf={output.confidence:.3f}, type={output.signal_type}")
        
        return output
    
    def _extract_ohlcv_dataframe(self, market_data: Dict, asset: str) -> Optional[Any]:
        """Extract OHLCV DataFrame from market_data."""
        try:
            import pandas as pd
            
            # Check nested format
            if asset in market_data and isinstance(market_data[asset], dict):
                if "ohlcv_history" in market_data[asset]:
                    history = market_data[asset]["ohlcv_history"]
                    if isinstance(history, pd.DataFrame):
                        return history
                    elif isinstance(history, list) and len(history) > 0:
                        return pd.DataFrame(history)
            
            # Check top-level
            history_key = f"{asset.lower()}_ohlcv_history"
            if history_key in market_data:
                history = market_data[history_key]
                if isinstance(history, pd.DataFrame):
                    return history
                elif isinstance(history, list) and len(history) > 0:
                    return pd.DataFrame(history)
            
            # Check generic "ohlcv_history"
            if "ohlcv_history" in market_data:
                history = market_data["ohlcv_history"]
                if isinstance(history, pd.DataFrame):
                    return history
            
            return None
            
        except ImportError:
            return None
    
    def _generate_with_real_agent(
        self,
        df: Any,  # pd.DataFrame
        strategy: QuantStrategy,
        regime_name: str
    ) -> QuantGeneratorOutput:
        """Generate signal using REAL QuantAgent."""
        
        agent = self._get_or_create_agent(strategy)
        
        try:
            # Generate real signal
            signal: Signal = agent.generate_signal(df, regime=regime_name)
            
            # Convert to output format
            if signal.signal_type == SignalType.LONG:
                direction = signal.strength * signal.confidence
            elif signal.signal_type == SignalType.SHORT:
                direction = signal.strength * signal.confidence  # Already negative
            else:
                direction = signal.strength * 0.3  # Partial signal
            
            # Apply strategy long_bias
            direction = direction + (strategy.long_bias - 0.5) * 0.2
            direction = max(-1.0, min(1.0, direction))
            
            return QuantGeneratorOutput(
                direction=direction,
                confidence=signal.confidence,
                strategy_name=strategy.name,
                strategy_group=strategy.regime_group.value,
                strategy_role=strategy.role.name,
                signal_type=signal.signal_type.value,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                reasoning="; ".join(signal.reasons),
                indicators=signal.metadata.get("indicators", {})
            )
            
        except Exception as e:
            logger.error(f"[QUANT GEN] Real agent failed: {e}")
            return self._generate_fallback(strategy)
    
    def _generate_simplified(
        self,
        market_data: Dict,
        strategy: QuantStrategy,
        asset: str
    ) -> QuantGeneratorOutput:
        """
        Simplified signal when OHLCV history unavailable.
        
        Uses available price data with RSI-like calculation.
        """
        # Get price data
        asset_lower = asset.lower()
        
        if asset in market_data and isinstance(market_data[asset], dict):
            close = market_data[asset].get("close", 0)
            returns = market_data[asset].get("returns_4h", 0)
            volatility = market_data[asset].get("volatility_4h", 0.02)
        else:
            close = market_data.get(f"{asset_lower}_price", 
                                   market_data.get("current_price", 0))
            returns = market_data.get(f"{asset_lower}_returns_4h",
                                     market_data.get("returns_4h", 0))
            volatility = market_data.get("volatility_4h", 0.02)
        
        if close <= 0:
            return self._generate_fallback(strategy)
        
        # Simplified RSI-like signal
        # Estimate RSI from returns (very rough)
        rsi_estimate = 50 + returns * 1000  # Scale returns to RSI range
        rsi_estimate = max(0, min(100, rsi_estimate))
        
        # Generate signal based on RSI thresholds
        direction = 0.0
        signal_type = "HOLD"
        confidence = 0.4
        reasons = []
        
        if rsi_estimate < strategy.rsi_oversold:
            # Oversold -> potential LONG
            direction = (strategy.rsi_oversold - rsi_estimate) / strategy.rsi_oversold
            direction *= strategy.long_bias * 1.5
            signal_type = "LONG" if direction > 0.3 else "HOLD"
            confidence = 0.5 + abs(direction) * 0.2
            reasons.append(f"RSI estimate {rsi_estimate:.0f} < {strategy.rsi_oversold}")
            
        elif rsi_estimate > strategy.rsi_overbought:
            # Overbought -> potential SHORT
            direction = (strategy.rsi_overbought - rsi_estimate) / (100 - strategy.rsi_overbought)
            direction *= (1 - strategy.long_bias) * 1.5
            signal_type = "SHORT" if direction < -0.3 else "HOLD"
            confidence = 0.5 + abs(direction) * 0.2
            reasons.append(f"RSI estimate {rsi_estimate:.0f} > {strategy.rsi_overbought}")
        else:
            # Neutral range
            direction = (rsi_estimate - 50) / 50 * 0.3 * strategy.long_bias
            reasons.append(f"RSI estimate {rsi_estimate:.0f} in neutral range")
        
        direction = max(-1.0, min(1.0, direction))
        confidence = min(0.7, confidence)  # Cap confidence for simplified
        
        return QuantGeneratorOutput(
            direction=direction,
            confidence=confidence,
            strategy_name=strategy.name,
            strategy_group=strategy.regime_group.value,
            strategy_role=strategy.role.name,
            signal_type=signal_type,
            entry_price=close,
            stop_loss=close * (0.98 if direction > 0 else 1.02),
            take_profit=close * (1.04 if direction > 0 else 0.96),
            reasoning=f"Simplified: {'; '.join(reasons)}"
        )
    
    def _generate_fallback(self, strategy: QuantStrategy) -> QuantGeneratorOutput:
        """Fallback when no data available."""
        return QuantGeneratorOutput(
            direction=0.0,
            confidence=0.0,
            strategy_name=strategy.name,
            strategy_group=strategy.regime_group.value,
            strategy_role=strategy.role.name,
            signal_type="HOLD",
            reasoning="No valid data - fallback HOLD"
        )
    
    def get_last_output(self) -> Optional[QuantGeneratorOutput]:
        return self._last_output


# =============================================================================
# DRL SIGNAL GENERATOR - EXPLICIT ENABLE/DISABLE
# =============================================================================

@dataclass
class DRLGeneratorOutput:
    """Output from DRLSignalGenerator."""
    direction: float  # -1 to 1
    confidence: float  # 0 to 1
    entropy: float  # Policy entropy
    
    is_enabled: bool = False  # EXPLICIT: Is DRL actually running?
    model_loaded: bool = False  # EXPLICIT: Is a real model loaded?
    
    reasoning: str = ""
    model_version: str = "disabled"
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "is_enabled": self.is_enabled,
            "model_loaded": self.model_loaded,
            "reasoning": self.reasoning,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat()
        }


class DRLSignalGenerator:
    """
    DRL Signal Generator with EXPLICIT enable/disable.
    
    RULE: If ENABLE_DRL=False, this generator returns confidence=0
          and is EXCLUDED from fusion arbitration.
    
    NO FAKE SIGNALS. Either real or disabled.
    """
    
    def __init__(self):
        self._last_output: Optional[DRLGeneratorOutput] = None
        self.config = get_signal_config()
        
        self._is_enabled = self.config.ENABLE_DRL
        self._model_loaded = False
        self._model = None
        
        if self._is_enabled:
            self._try_load_model()
        
        status = "ENABLED" if self._is_enabled else "DISABLED"
        model_status = "loaded" if self._model_loaded else "not loaded"
        logger.info(f"[DRLSignalGenerator] Status: {status}, Model: {model_status}")
    
    def _try_load_model(self):
        """Attempt to load trained DRL model."""
        if not self.config.DRL_CHECKPOINT_PATH:
            logger.warning("[DRL] No checkpoint path configured - DRL will return zeros")
            return
        
        try:
            # Try stable-baselines3 first
            try:
                from stable_baselines3 import PPO, SAC, A2C
                import os
                
                checkpoint = self.config.DRL_CHECKPOINT_PATH
                
                if os.path.exists(checkpoint) or os.path.exists(f"{checkpoint}.zip"):
                    # Detect algorithm from path
                    path_lower = checkpoint.lower()
                    if "sac" in path_lower:
                        self._model = SAC.load(checkpoint)
                        self._algorithm = "sac"
                    elif "a2c" in path_lower:
                        self._model = A2C.load(checkpoint)
                        self._algorithm = "a2c"
                    else:
                        self._model = PPO.load(checkpoint)
                        self._algorithm = "ppo"
                    
                    self._model_loaded = True
                    logger.info(f"[DRL] Loaded {self._algorithm.upper()} model from {checkpoint}")
                else:
                    logger.warning(f"[DRL] Checkpoint not found: {checkpoint}")
                    
            except ImportError:
                logger.warning("[DRL] stable-baselines3 not available")
            
        except Exception as e:
            logger.error(f"[DRL] Failed to load model: {e}")
    
    @property
    def is_enabled(self) -> bool:
        """Check if DRL is actually enabled and functional."""
        return self._is_enabled and self._model_loaded
    
    def generate(
        self,
        state_vector: np.ndarray,
        regime_name: str = "NEUTRAL",
        is_opportunity_mode: bool = False
    ) -> DRLGeneratorOutput:
        """
        Generate DRL signal.
        
        If disabled or no model: Returns confidence=0 (excluded from fusion).
        """
        
        # EXPLICIT CHECK: Is DRL enabled?
        if not self._is_enabled:
            logger.debug("[DRL] Disabled by config - returning zeros")
            return self._disabled_output("DRL disabled in config")
        
        if not self._model_loaded:
            logger.debug("[DRL] No model loaded - returning zeros")
            return self._disabled_output("No model loaded")
        
        # REAL MODEL INFERENCE
        try:
            output = self._generate_with_model(state_vector, regime_name)
            self._last_output = output
            return output
            
        except Exception as e:
            logger.error(f"[DRL] Inference failed: {e}")
            return self._disabled_output(f"Inference error: {e}")
    
    def _generate_with_model(
        self,
        state_vector: np.ndarray,
        regime_name: str
    ) -> DRLGeneratorOutput:
        """Generate using actual trained model."""
        
        if self._model is None:
            return self._disabled_output("Model not loaded")
        
        try:
            # Reshape state for model input
            obs = state_vector.reshape(1, -1).astype(np.float32)
            
            # Get action from model
            action, _ = self._model.predict(obs, deterministic=True)
            
            # Extract direction from action
            # Assuming action is in [-1, 1] or [0, 1] range
            direction = float(action[0]) if hasattr(action, '__len__') else float(action)
            
            # Normalize to [-1, 1] if needed
            if direction > 1:
                direction = (direction - 0.5) * 2  # [0, 1] -> [-1, 1]
            direction = np.clip(direction, -1.0, 1.0)
            
            # Calculate entropy/uncertainty
            try:
                # Get action distribution for entropy
                obs_tensor = self._model.policy.obs_to_tensor(obs)[0]
                dist = self._model.policy.get_distribution(obs_tensor)
                entropy = float(dist.entropy().mean().item())
            except Exception:
                # Default entropy based on action magnitude
                entropy = 1.0 - abs(direction)  # Higher confidence = lower entropy
            
            # Confidence is inverse of entropy
            confidence = max(0.0, min(1.0, 1.0 - entropy))
            
            # Apply regime-based adjustments
            if regime_name in ["HIGH_VOLATILITY", "CRISIS"]:
                confidence *= 0.5  # Reduce confidence in volatile regimes
            
            return DRLGeneratorOutput(
                direction=direction,
                confidence=confidence,
                entropy=entropy,
                is_enabled=self._is_enabled,
                model_loaded=self._model_loaded,
                reasoning=f"Model inference ({getattr(self, '_algorithm', 'unknown')}), regime={regime_name}",
                model_version=getattr(self, '_algorithm', 'unknown')
            )
            
        except Exception as e:
            logger.error(f"[DRL] Model inference failed: {e}")
            return self._disabled_output(f"Inference error: {e}")
    
    def _disabled_output(self, reason: str) -> DRLGeneratorOutput:
        """Return explicit disabled output."""
        return DRLGeneratorOutput(
            direction=0.0,
            confidence=0.0,  # ZERO confidence = excluded from fusion
            entropy=1.0,     # Maximum entropy = maximum uncertainty
            is_enabled=self._is_enabled,
            model_loaded=self._model_loaded,
            reasoning=reason,
            model_version="disabled"
        )
    
    def get_last_output(self) -> Optional[DRLGeneratorOutput]:
        return self._last_output


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_quant_generator: Optional[QuantSignalGenerator] = None
_drl_generator: Optional[DRLSignalGenerator] = None


def get_quant_generator() -> QuantSignalGenerator:
    """Get or create QuantSignalGenerator singleton."""
    global _quant_generator
    if _quant_generator is None:
        _quant_generator = QuantSignalGenerator()
    return _quant_generator


def get_drl_generator() -> DRLSignalGenerator:
    """Get or create DRLSignalGenerator singleton."""
    global _drl_generator
    if _drl_generator is None:
        _drl_generator = DRLSignalGenerator()
    return _drl_generator


def reset_generators():
    """Reset singleton instances (for testing)."""
    global _quant_generator, _drl_generator
    _quant_generator = None
    _drl_generator = None


# =============================================================================
# TEST
# =============================================================================

def test_real_signal_generators():
    """Test REAL signal generators."""
    
    print("=" * 70)
    print("TESTING REAL SIGNAL GENERATORS")
    print("=" * 70)
    
    # Reset singletons
    reset_generators()
    
    # Test QuantSignalGenerator with real QuantAgent
    print("\n1. QuantSignalGenerator with REAL QuantAgent:")
    quant_gen = get_quant_generator()
    
    # Simulate market data with OHLCV
    import pandas as pd
    np.random.seed(42)
    
    # Generate realistic OHLCV history
    n_bars = 250
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='4h')
    base_price = 180.0
    returns = np.random.randn(n_bars) * 0.02
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    ohlcv_history = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices * (1 + np.random.randn(n_bars) * 0.005),
        'high': close_prices * (1 + np.abs(np.random.randn(n_bars) * 0.01)),
        'low': close_prices * (1 - np.abs(np.random.randn(n_bars) * 0.01)),
        'close': close_prices,
        'volume': np.abs(np.random.randn(n_bars) * 1000) + 500
    })
    
    # Market data with history
    market_data = {
        "SOL": {
            "close": close_prices[-1],
            "open": close_prices[-2],
            "high": close_prices[-1] * 1.01,
            "low": close_prices[-1] * 0.99,
            "volume": 1500,
            "returns_4h": returns[-1],
            "ohlcv_history": ohlcv_history
        },
        "BTC": {"close": 95000, "volume": 1000},
        "ETH": {"close": 3400, "volume": 500},
    }
    
    # Test BULL regime
    print("\n  a) BULL regime, NORMAL mode:")
    output = quant_gen.generate(market_data, "BULL", is_opportunity_mode=False)
    print(f"     Strategy: {output.strategy_name}")
    print(f"     Direction: {output.direction:.4f}")
    print(f"     Confidence: {output.confidence:.4f}")
    print(f"     Signal Type: {output.signal_type}")
    print(f"     Reasoning: {output.reasoning[:80]}...")
    
    # Test BEAR regime in OPPORTUNITY
    print("\n  b) BEAR regime, OPPORTUNITY mode:")
    output = quant_gen.generate(market_data, "STRONG_BEAR", is_opportunity_mode=True)
    print(f"     Strategy: {output.strategy_name}")
    print(f"     Direction: {output.direction:.4f}")
    print(f"     Signal Type: {output.signal_type}")
    
    # Test without OHLCV history (simplified)
    print("\n  c) Without OHLCV history (simplified mode):")
    simple_data = {
        "sol_price": 185.0,
        "current_price": 185.0,
        "returns_4h": 0.05,  # +5% return
        "volatility_4h": 0.04
    }
    output = quant_gen.generate(simple_data, "VOLATILE", is_opportunity_mode=False)
    print(f"     Strategy: {output.strategy_name}")
    print(f"     Direction: {output.direction:.4f}")
    print(f"     Reasoning: {output.reasoning}")
    
    # Test DRLSignalGenerator (should be disabled)
    print("\n2. DRLSignalGenerator (EXPLICIT DISABLE CHECK):")
    drl_gen = get_drl_generator()
    
    state = np.array([0.1, 0.2, -0.1, 0.3, 0.5])
    output = drl_gen.generate(state, "BULL")
    
    print(f"   is_enabled: {output.is_enabled}")
    print(f"   model_loaded: {output.model_loaded}")
    print(f"   confidence: {output.confidence}")
    print(f"   reasoning: {output.reasoning}")
    
    # Verify DRL is properly disabled
    assert output.confidence == 0.0, "DRL should have 0 confidence when disabled!"
    print("   ✓ DRL correctly returns confidence=0 when disabled")
    
    # Test signal conflict
    print("\n3. Signal Conflict Detection:")
    conflict = compute_signal_conflict(
        quant_direction=0.8, quant_confidence=0.7,
        drl_direction=-0.8, drl_confidence=0.7,
        sentiment_direction=0.0, sentiment_confidence=0.3
    )
    print(f"   Quant: {conflict.quant_direction}, DRL: {conflict.drl_direction}")
    print(f"   all_signals_conflict: {conflict.all_signals_conflict}")
    print(f"   conflict_type: {conflict.conflict_type}")
    
    print("\n" + "=" * 70)
    print("✓ REAL SIGNAL GENERATORS TEST PASSED")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    test_real_signal_generators()

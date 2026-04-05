"""
================================================================================
HMATS v5.2.0 - RL Execution Fallback System
================================================================================
Purpose: Robust fallback logic when RL execution agent is uncertain or untrained

Features:
1. Drift-aware model confidence scoring
2. Deterministic fallback strategies
3. Automatic mode switching based on conditions
4. Performance tracking for RL vs fallback
5. Graceful degradation path

Author: HMATS v5.2 Upgrade
Version: 5.2.0
================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timezone, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Current execution mode."""
    RL_ACTIVE = "rl_active"
    RL_SHADOW = "rl_shadow"
    FALLBACK_DRIFT = "fallback_drift"
    FALLBACK_UNTRAINED = "fallback_untrained"
    FALLBACK_ERROR = "fallback_error"
    DETERMINISTIC = "deterministic"


class FallbackReason(Enum):
    """Why fallback mode was activated."""
    NONE = "none"
    DRIFT_DETECTED = "drift_detected"
    LOW_CONFIDENCE = "low_confidence"
    UNTRAINED_STATE = "untrained_state"
    MODEL_ERROR = "model_error"
    MANUAL_OVERRIDE = "manual_override"
    HIGH_VOLATILITY = "high_volatility"
    REGIME_MISMATCH = "regime_mismatch"


@dataclass
class DriftMetrics:
    """Metrics for detecting distribution drift."""
    state_mean_shift: float = 0.0
    state_std_shift: float = 0.0
    reward_mean_shift: float = 0.0
    action_distribution_kl: float = 0.0
    overall_drift_score: float = 0.0
    is_drifted: bool = False


@dataclass
class FallbackConfig:
    """Configuration for fallback system."""
    drift_threshold: float = 0.15
    state_shift_threshold: float = 2.0
    kl_divergence_threshold: float = 0.5
    min_rl_confidence: float = 0.6
    min_samples_for_rl: int = 1000
    high_vol_threshold: float = 0.8
    performance_window: int = 50
    underperformance_threshold: float = -0.01
    fallback_aggression: float = 0.5
    max_fallback_impact_bps: float = 20.0


@dataclass
class ExecutionDecision:
    """Result of execution decision."""
    action: str
    mode: ExecutionMode
    order_type: str
    price_offset_bps: float
    size_fraction: float
    confidence: float
    fallback_reason: Optional[FallbackReason]
    rl_action: Optional[Any]
    reasoning: str


class DriftDetector:
    """Detects distribution drift between training and live environments."""
    
    def __init__(self, config: FallbackConfig = None):
        self.config = config or FallbackConfig()
        self.training_state_mean: Optional[np.ndarray] = None
        self.training_state_std: Optional[np.ndarray] = None
        self.training_reward_mean: float = 0.0
        self.training_reward_std: float = 1.0
        self.training_action_dist: Optional[np.ndarray] = None
        self.live_states: deque = deque(maxlen=500)
        self.live_rewards: deque = deque(maxlen=500)
        self.live_actions: deque = deque(maxlen=500)
        self.drift_history: deque = deque(maxlen=100)
        self._initialized = False
    
    def set_training_baseline(
        self,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        reward_mean: float,
        reward_std: float,
        action_dist: np.ndarray = None
    ):
        """Set reference distributions from training data."""
        self.training_state_mean = state_mean
        self.training_state_std = np.maximum(state_std, 1e-6)
        self.training_reward_mean = reward_mean
        self.training_reward_std = max(reward_std, 1e-6)
        self.training_action_dist = action_dist
        self._initialized = True
        logger.info("[DRIFT] Training baseline set")
    
    def update_live(self, state: np.ndarray, reward: float = None, action: Any = None):
        """Update live distributions with new observation."""
        self.live_states.append(state)
        if reward is not None:
            self.live_rewards.append(reward)
        if action is not None:
            self.live_actions.append(action)
    
    def compute_drift(self) -> DriftMetrics:
        """Compute drift metrics between training and live."""
        if not self._initialized or len(self.live_states) < 50:
            return DriftMetrics()
        
        live_states = np.array(list(self.live_states))
        live_mean = np.mean(live_states, axis=0)
        live_std = np.std(live_states, axis=0)
        
        state_mean_shift = np.mean(np.abs(
            (live_mean - self.training_state_mean) / self.training_state_std
        ))
        state_std_shift = np.mean(np.abs(
            (live_std - self.training_state_std) / self.training_state_std
        ))
        
        reward_mean_shift = 0.0
        if self.live_rewards and self.training_reward_std > 0:
            live_reward_mean = np.mean(list(self.live_rewards))
            reward_mean_shift = abs(
                (live_reward_mean - self.training_reward_mean) / self.training_reward_std
            )
        
        action_kl = 0.0
        if self.training_action_dist is not None and self.live_actions:
            live_action_counts = np.bincount(
                [int(a) for a in self.live_actions if isinstance(a, (int, float))],
                minlength=len(self.training_action_dist)
            )
            live_action_dist = live_action_counts / max(np.sum(live_action_counts), 1)
            epsilon = 1e-10
            action_kl = np.sum(
                live_action_dist * np.log(
                    (live_action_dist + epsilon) / (self.training_action_dist + epsilon)
                )
            )
        
        overall = (
            0.4 * state_mean_shift +
            0.2 * state_std_shift +
            0.2 * reward_mean_shift +
            0.2 * action_kl
        )
        
        metrics = DriftMetrics(
            state_mean_shift=state_mean_shift,
            state_std_shift=state_std_shift,
            reward_mean_shift=reward_mean_shift,
            action_distribution_kl=action_kl,
            overall_drift_score=overall,
            is_drifted=overall > self.config.drift_threshold
        )
        
        self.drift_history.append({"timestamp": datetime.now(timezone.utc), "metrics": metrics})
        return metrics


class DeterministicFallbackStrategy:
    """Rule-based execution strategy used when RL is unavailable or unreliable."""
    
    def __init__(self, config: FallbackConfig = None):
        self.config = config or FallbackConfig()
    
    def decide(
        self,
        state: Dict[str, Any],
        order_side: str,
        remaining_size: float,
        urgency: str = "normal"
    ) -> ExecutionDecision:
        """Make deterministic execution decision."""
        spread_bps = state.get("spread_bps", 10.0)
        volatility = state.get("volatility", 0.3)
        book_imbalance = state.get("book_imbalance", 0.0)
        time_to_close = state.get("time_to_close_hours", 24.0)
        
        action = "wait"
        order_type = "limit"
        price_offset = 0.0
        size_fraction = 1.0
        reasoning_parts = []
        
        if urgency == "immediate":
            action = "execute_now"
            order_type = "market"
            size_fraction = 1.0
            reasoning_parts.append("Immediate urgency - market order")
        elif spread_bps > 30:
            action = "wait"
            reasoning_parts.append(f"Wide spread ({spread_bps:.1f}bps) - wait")
        elif volatility > self.config.high_vol_threshold:
            action = "slice"
            size_fraction = 0.25
            order_type = "limit"
            price_offset = -2.0 if order_side == "buy" else 2.0
            reasoning_parts.append(f"High volatility ({volatility:.0%}) - slice")
        elif (order_side == "buy" and book_imbalance > 0.2) or \
             (order_side == "sell" and book_imbalance < -0.2):
            action = "execute_now"
            order_type = "limit"
            price_offset = 1.0
            size_fraction = 0.5
            reasoning_parts.append(f"Favorable imbalance ({book_imbalance:+.2f})")
        elif (order_side == "buy" and book_imbalance < -0.3) or \
             (order_side == "sell" and book_imbalance > 0.3):
            action = "slice"
            order_type = "post_only"
            price_offset = -3.0 if order_side == "buy" else 3.0
            size_fraction = 0.2
            reasoning_parts.append(f"Unfavorable imbalance ({book_imbalance:+.2f})")
        else:
            action = "execute_now"
            order_type = "limit"
            price_offset = 0.0
            size_fraction = 0.5 if remaining_size > 10000 else 1.0
            reasoning_parts.append("Normal conditions - post at touch")
        
        if time_to_close < 1.0 and urgency != "immediate":
            size_fraction = min(1.0, size_fraction * 1.5)
            reasoning_parts.append("Near close - increase urgency")
        
        if order_type == "limit":
            price_offset *= (1 - self.config.fallback_aggression)
        
        return ExecutionDecision(
            action=action,
            mode=ExecutionMode.DETERMINISTIC,
            order_type=order_type,
            price_offset_bps=price_offset,
            size_fraction=size_fraction,
            confidence=0.7,
            fallback_reason=FallbackReason.NONE,
            rl_action=None,
            reasoning="; ".join(reasoning_parts)
        )


class RLExecutionFallbackManager:
    """Manages switching between RL and fallback execution."""
    
    def __init__(self, config: FallbackConfig = None):
        self.config = config or FallbackConfig()
        self.drift_detector = DriftDetector(config)
        self.fallback_strategy = DeterministicFallbackStrategy(config)
        self.current_mode = ExecutionMode.FALLBACK_UNTRAINED
        self.fallback_reason = FallbackReason.UNTRAINED_STATE
        self.rl_model_loaded = False
        self.rl_performance: deque = deque(maxlen=self.config.performance_window)
        self.fallback_performance: deque = deque(maxlen=self.config.performance_window)
        self._rl_predict: Optional[Callable] = None
        self._rl_confidence: Optional[Callable] = None
        self.stats = {
            "rl_decisions": 0,
            "fallback_decisions": 0,
            "mode_switches": 0,
            "drift_events": 0
        }
        logger.info("[FALLBACK] RLExecutionFallbackManager initialized")
    
    def set_rl_model(
        self,
        predict_fn: Callable[[np.ndarray], Any],
        confidence_fn: Callable[[np.ndarray], float] = None,
        training_stats: Dict = None
    ):
        """Connect RL model for execution decisions."""
        self._rl_predict = predict_fn
        self._rl_confidence = confidence_fn
        self.rl_model_loaded = True
        
        if training_stats:
            self.drift_detector.set_training_baseline(
                state_mean=training_stats.get("state_mean", np.zeros(10)),
                state_std=training_stats.get("state_std", np.ones(10)),
                reward_mean=training_stats.get("reward_mean", 0.0),
                reward_std=training_stats.get("reward_std", 1.0),
                action_dist=training_stats.get("action_dist")
            )
        
        self.current_mode = ExecutionMode.RL_SHADOW
        self.fallback_reason = FallbackReason.NONE
        logger.info("[FALLBACK] RL model connected, starting in shadow mode")
    
    def enable_rl(self):
        """Enable RL model for active decisions."""
        if self.rl_model_loaded:
            self.current_mode = ExecutionMode.RL_ACTIVE
            logger.info("[FALLBACK] RL mode enabled")
    
    def disable_rl(self, reason: FallbackReason = FallbackReason.MANUAL_OVERRIDE):
        """Disable RL and use fallback."""
        self.current_mode = ExecutionMode.DETERMINISTIC
        self.fallback_reason = reason
        self.stats["mode_switches"] += 1
        logger.info(f"[FALLBACK] RL disabled, reason: {reason.value}")
    
    def decide(
        self,
        state: np.ndarray,
        state_dict: Dict[str, Any],
        order_side: str,
        remaining_size: float,
        urgency: str = "normal"
    ) -> ExecutionDecision:
        """Make execution decision using appropriate mode."""
        self.drift_detector.update_live(state)
        self._check_mode_switch(state)
        
        if self.current_mode == ExecutionMode.RL_ACTIVE:
            decision = self._rl_decide(state, state_dict, order_side, remaining_size, urgency)
            self.stats["rl_decisions"] += 1
        else:
            decision = self.fallback_strategy.decide(state_dict, order_side, remaining_size, urgency)
            decision.fallback_reason = self.fallback_reason
            decision.mode = self.current_mode
            self.stats["fallback_decisions"] += 1
        
        return decision
    
    def _check_mode_switch(self, state: np.ndarray):
        """Check if mode should be switched based on current conditions."""
        if not self.rl_model_loaded:
            return
        
        old_mode = self.current_mode
        drift = self.drift_detector.compute_drift()
        
        if drift.is_drifted:
            if self.current_mode == ExecutionMode.RL_ACTIVE:
                self.current_mode = ExecutionMode.FALLBACK_DRIFT
                self.fallback_reason = FallbackReason.DRIFT_DETECTED
                self.stats["drift_events"] += 1
                logger.warning(f"[FALLBACK] Drift detected (score={drift.overall_drift_score:.3f})")
        else:
            if self.current_mode == ExecutionMode.FALLBACK_DRIFT:
                self.current_mode = ExecutionMode.RL_ACTIVE
                self.fallback_reason = FallbackReason.NONE
                logger.info("[FALLBACK] Drift recovered, returning to RL mode")
        
        if self.current_mode == ExecutionMode.RL_ACTIVE and self._rl_confidence:
            try:
                confidence = self._rl_confidence(state)
                if confidence < self.config.min_rl_confidence:
                    self.current_mode = ExecutionMode.FALLBACK_DRIFT
                    self.fallback_reason = FallbackReason.LOW_CONFIDENCE
            except Exception as e:
                self.current_mode = ExecutionMode.FALLBACK_ERROR
                self.fallback_reason = FallbackReason.MODEL_ERROR
                logger.error(f"[FALLBACK] RL confidence error: {e}")
        
        if old_mode != self.current_mode:
            self.stats["mode_switches"] += 1
    
    def _rl_decide(self, state, state_dict, order_side, remaining_size, urgency) -> ExecutionDecision:
        """Get decision from RL model."""
        try:
            rl_action = self._rl_predict(state)
            confidence = self._rl_confidence(state) if self._rl_confidence else 0.8
            
            action_map = {
                0: ("wait", "limit", 0.0, 0.0),
                1: ("execute_now", "limit", 0.0, 0.5),
                2: ("execute_now", "limit", 1.0, 0.5),
                3: ("execute_now", "market", 0.0, 1.0),
                4: ("slice", "post_only", -2.0, 0.25),
            }
            
            action_idx = int(rl_action) if isinstance(rl_action, (int, float, np.integer)) else 1
            action_idx = min(action_idx, len(action_map) - 1)
            action, order_type, price_offset, size_fraction = action_map.get(action_idx, ("execute_now", "limit", 0.0, 0.5))
            
            return ExecutionDecision(
                action=action,
                mode=ExecutionMode.RL_ACTIVE,
                order_type=order_type,
                price_offset_bps=price_offset,
                size_fraction=size_fraction,
                confidence=confidence,
                fallback_reason=None,
                rl_action=rl_action,
                reasoning=f"RL action {rl_action} with confidence {confidence:.2f}"
            )
        except Exception as e:
            logger.error(f"[FALLBACK] RL prediction error: {e}")
            self.current_mode = ExecutionMode.FALLBACK_ERROR
            self.fallback_reason = FallbackReason.MODEL_ERROR
            return self.fallback_strategy.decide(state_dict, order_side, remaining_size, urgency)
    
    def record_performance(self, mode: ExecutionMode, result: float):
        """Record execution performance for mode comparison."""
        if mode == ExecutionMode.RL_ACTIVE:
            self.rl_performance.append(result)
        else:
            self.fallback_performance.append(result)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current manager status."""
        drift = self.drift_detector.compute_drift() if self.drift_detector._initialized else None
        return {
            "current_mode": self.current_mode.value,
            "fallback_reason": self.fallback_reason.value if self.fallback_reason else None,
            "rl_model_loaded": self.rl_model_loaded,
            "drift_score": drift.overall_drift_score if drift else None,
            "is_drifted": drift.is_drifted if drift else None,
            "stats": self.stats,
            "rl_avg_performance": np.mean(list(self.rl_performance)) if self.rl_performance else None,
            "fallback_avg_performance": np.mean(list(self.fallback_performance)) if self.fallback_performance else None
        }


_fallback_manager: Optional[RLExecutionFallbackManager] = None

def get_fallback_manager(config: FallbackConfig = None) -> RLExecutionFallbackManager:
    global _fallback_manager
    if _fallback_manager is None:
        _fallback_manager = RLExecutionFallbackManager(config)
    elif config is not None and _fallback_manager.config != config:
        logger.info("[RL_FALLBACK] Config changed; refreshing singleton instance")
        _fallback_manager = RLExecutionFallbackManager(config)
    return _fallback_manager

def reset_fallback_manager():
    global _fallback_manager
    _fallback_manager = None

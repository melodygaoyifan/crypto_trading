#!/usr/bin/env python3
"""
Advanced SignalFusionModule - Dynamic Quant/DRL Weighting
=========================================================
SOTA v3.0 Component for Intelligent Signal Integration

Features:
1. Dynamic weighting based on regime confidence
2. Entropy-based DRL trust adjustment
3. VPIN/OFI toxicity-aware signal dampening
4. API cost-aware position sizing
5. Integrated reward function for training

Hardware: RTX 5090 (32GB VRAM)
Frequency: 4H/1D strategy decisions

Author: Senior Quant Team
Version: SOTA v3.0
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SignalFusionModule')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QuantSignal:
    """Signal from Quant Agent (12 strategies)."""
    timestamp: float
    asset: str
    signal: float
    confidence: float
    active_strategies: int
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    regime: str = 'NEUTRAL'
    information_ratio: float = 0.0


@dataclass
class DRLSignal:
    """Signal from DRL Agent (Decision Transformer)."""
    timestamp: float
    asset: str
    signal: float
    confidence: float
    entropy: float
    q_value: float
    state_features: Optional[np.ndarray] = None


@dataclass
class FusedSignal:
    """Final fused signal for execution."""
    timestamp: float
    asset: str
    signal: float
    magnitude: float
    confidence: float
    quant_weight: float
    drl_weight: float
    quant_contribution: float
    drl_contribution: float
    toxicity_dampening: float
    regime_adjustment: float
    api_cost_factor: float
    execution_mode: str = 'NORMAL'
    reasoning: str = ''


@dataclass
class RewardComponents:
    """Breakdown of reward for RL training."""
    total_reward: float
    sharpe_component: float
    drawdown_penalty: float
    fee_penalty: float
    api_cost_penalty: float
    regime_bonus: float
    risk_adjusted_return: float


# =============================================================================
# DYNAMIC WEIGHT CALCULATOR
# =============================================================================

class DynamicWeightCalculator:
    """Calculate dynamic weights for Quant vs DRL signals."""
    
    BASE_QUANT_WEIGHT = 0.6
    BASE_DRL_WEIGHT = 0.4
    WEIGHT_MIN = 0.15
    WEIGHT_MAX = 0.85
    
    def __init__(self, performance_window: int = 50, entropy_threshold: float = 0.8,
                 confidence_threshold: float = 0.6):
        self.performance_window = performance_window
        self.entropy_threshold = entropy_threshold
        self.confidence_threshold = confidence_threshold
        
        self.quant_returns = deque(maxlen=performance_window)
        self.drl_returns = deque(maxlen=performance_window)
        self.current_quant_weight = self.BASE_QUANT_WEIGHT
        self.current_drl_weight = self.BASE_DRL_WEIGHT
    
    def calculate_weights(self, quant_signal: QuantSignal, drl_signal: DRLSignal,
                         regime_confidence: float, vpin: float,
                         circuit_breaker_active: bool = False) -> Tuple[float, float, str]:
        reasons = []
        w_quant = self.BASE_QUANT_WEIGHT
        w_drl = self.BASE_DRL_WEIGHT
        
        if circuit_breaker_active:
            return 0.80, 0.20, "Circuit breaker: 80% Quant"
        
        # Regime confidence adjustment
        if regime_confidence > self.confidence_threshold:
            regime_boost = (regime_confidence - self.confidence_threshold) * 0.3
            w_quant += regime_boost
            w_drl -= regime_boost
            reasons.append(f"High regime conf (+{regime_boost:.0%} Quant)")
        elif regime_confidence < 0.4:
            regime_shift = (0.4 - regime_confidence) * 0.2
            w_quant -= regime_shift
            w_drl += regime_shift
            reasons.append(f"Low regime conf (+{regime_shift:.0%} DRL)")
        
        # DRL entropy adjustment
        if drl_signal.entropy > self.entropy_threshold:
            entropy_penalty = (drl_signal.entropy - self.entropy_threshold) * 0.4
            w_drl -= entropy_penalty
            w_quant += entropy_penalty
            reasons.append(f"High DRL entropy")
        
        # Strategy agreement bonus
        if quant_signal.active_strategies >= 3:
            agreement_bonus = min(0.15, quant_signal.active_strategies * 0.03)
            w_quant += agreement_bonus
            w_drl -= agreement_bonus
        
        # Toxicity adjustment
        if vpin > 0.5:
            toxicity_shift = (vpin - 0.5) * 0.4
            w_quant += toxicity_shift * 0.5
            w_drl -= toxicity_shift * 0.5
            reasons.append(f"Toxicity (VPIN={vpin:.2f})")
        
        # Normalize
        w_quant = max(self.WEIGHT_MIN, min(self.WEIGHT_MAX, w_quant))
        w_drl = max(self.WEIGHT_MIN, min(self.WEIGHT_MAX, w_drl))
        total = w_quant + w_drl
        w_quant /= total
        w_drl /= total
        
        self.current_quant_weight = w_quant
        self.current_drl_weight = w_drl
        
        return w_quant, w_drl, " | ".join(reasons) if reasons else "Standard"
    
    def record_performance(self, source: str, ret: float, correct: bool):
        if source == 'quant':
            self.quant_returns.append(ret)
        else:
            self.drl_returns.append(ret)


# =============================================================================
# INTEGRATED REWARD FUNCTION
# =============================================================================

class IntegratedRewardFunction:
    """
    R_t = Sharpe_rolling - α×Drawdown² - β×Kraken_Fee - γ×API_Cost + δ×Regime_Bonus
    """
    
    DEFAULT_ALPHA = 2.0
    DEFAULT_BETA = 0.5
    DEFAULT_GAMMA = 0.1
    DEFAULT_DELTA = 0.1
    KRAKEN_TAKER_FEE_BPS = 26
    
    API_COSTS = {'gpt4': 0.03, 'claude': 0.025, 'local_llm': 0.0}
    
    def __init__(self, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA,
                 gamma: float = DEFAULT_GAMMA, delta: float = DEFAULT_DELTA,
                 sharpe_window: int = 24):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.sharpe_window = sharpe_window
        self.returns_history = deque(maxlen=sharpe_window)
        self.equity_peak = 1.0
        self.current_equity = 1.0
        self.total_fees_paid = 0.0
        self.total_api_costs = 0.0
    
    def calculate_reward(self, period_return: float, trade_value_usd: float,
                        is_taker: bool, api_calls: Dict[str, int],
                        regime_correct: bool, current_drawdown: float) -> RewardComponents:
        self.current_equity *= (1 + period_return)
        self.equity_peak = max(self.equity_peak, self.current_equity)
        
        self.returns_history.append(period_return)
        sharpe_component = self._calculate_rolling_sharpe()
        
        drawdown_penalty = self.alpha * (current_drawdown ** 2)
        
        fee_bps = self.KRAKEN_TAKER_FEE_BPS if is_taker else 16
        fee_cost = (fee_bps / 10000) * trade_value_usd
        self.total_fees_paid += fee_cost
        fee_penalty = self.beta * (fee_cost / 100)
        
        api_cost = sum(self.API_COSTS.get(t, 0.01) * c for t, c in api_calls.items())
        self.total_api_costs += api_cost
        api_penalty = self.gamma * api_cost
        
        regime_bonus = self.delta if regime_correct else -self.delta * 0.5
        
        total_reward = sharpe_component - drawdown_penalty - fee_penalty - api_penalty + regime_bonus
        
        volatility = np.std(self.returns_history) if len(self.returns_history) > 1 else 0.01
        risk_adjusted_return = period_return / (volatility + 1e-6)
        
        return RewardComponents(total_reward, sharpe_component, drawdown_penalty,
                               fee_penalty, api_penalty, regime_bonus, risk_adjusted_return)
    
    def _calculate_rolling_sharpe(self) -> float:
        if len(self.returns_history) < 5:
            return 0.0
        returns = np.array(self.returns_history)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return < 1e-6:
            return 0.0
        sharpe = mean_return / std_return * np.sqrt(6 * 365)
        return np.clip(sharpe, -3.0, 3.0)
    
    def get_current_drawdown(self) -> float:
        if self.equity_peak <= 0:
            return 0.0
        return (self.equity_peak - self.current_equity) / self.equity_peak
    
    def reset(self):
        self.returns_history.clear()
        self.equity_peak = 1.0
        self.current_equity = 1.0
        self.total_fees_paid = 0.0
        self.total_api_costs = 0.0
    
    def get_stats(self) -> Dict[str, float]:
        returns = list(self.returns_history)
        return {
            'cumulative_return': self.current_equity - 1.0,
            'max_drawdown': self.get_current_drawdown(),
            'sharpe': self._calculate_rolling_sharpe(),
            'total_fees_usd': self.total_fees_paid,
            'total_api_costs_usd': self.total_api_costs,
        }


# =============================================================================
# SIGNAL FUSION MODULE
# =============================================================================

class AdvancedSignalFusionModule:
    """Advanced Signal Fusion Module - SOTA v3.0"""
    
    MAX_SIGNAL_MAGNITUDE = 1.0
    MIN_SIGNAL_MAGNITUDE = 0.01
    VPIN_DAMPENING_START = 0.4
    VPIN_DAMPENING_FULL = 0.8
    
    def __init__(self, performance_window: int = 50, entropy_threshold: float = 0.8,
                 confidence_threshold: float = 0.6):
        self.weight_calculator = DynamicWeightCalculator(
            performance_window, entropy_threshold, confidence_threshold
        )
        self.reward_function = IntegratedRewardFunction()
        self.fused_signals = deque(maxlen=500)
        self.signal_performance = []
        logger.info("Advanced Signal Fusion Module initialized")
    
    def fuse_signals(self, quant_signal: QuantSignal, drl_signal: DRLSignal,
                    regime_confidence: float, vpin: float, ofi_zscore: float,
                    circuit_breaker_active: bool = False,
                    api_quota_remaining: float = 1.0) -> FusedSignal:
        
        w_quant, w_drl, weight_reasoning = self.weight_calculator.calculate_weights(
            quant_signal, drl_signal, regime_confidence, vpin, circuit_breaker_active
        )
        
        quant_contribution = w_quant * quant_signal.signal * quant_signal.confidence
        drl_contribution = w_drl * drl_signal.signal * drl_signal.confidence
        raw_signal = quant_contribution + drl_contribution
        
        toxicity_factor = self._calculate_toxicity_dampening(vpin, ofi_zscore)
        dampened_signal = raw_signal * toxicity_factor
        
        regime_factor = self._calculate_regime_adjustment(quant_signal.regime, regime_confidence, dampened_signal)
        adjusted_signal = dampened_signal * regime_factor
        
        api_factor = self._calculate_api_cost_factor(api_quota_remaining, drl_signal.confidence)
        
        combined_confidence = w_quant * quant_signal.confidence + w_drl * drl_signal.confidence
        magnitude = abs(adjusted_signal) * combined_confidence * api_factor
        magnitude = np.clip(magnitude, self.MIN_SIGNAL_MAGNITUDE, self.MAX_SIGNAL_MAGNITUDE)
        
        final_signal = np.clip(np.sign(adjusted_signal) * magnitude, -1.0, 1.0)
        execution_mode = self._determine_execution_mode(vpin, regime_confidence, combined_confidence, circuit_breaker_active)
        
        fused = FusedSignal(
            timestamp=time.time(), asset=quant_signal.asset, signal=final_signal,
            magnitude=magnitude, confidence=combined_confidence, quant_weight=w_quant,
            drl_weight=w_drl, quant_contribution=quant_contribution,
            drl_contribution=drl_contribution, toxicity_dampening=toxicity_factor,
            regime_adjustment=regime_factor, api_cost_factor=api_factor,
            execution_mode=execution_mode, reasoning=weight_reasoning
        )
        
        self.fused_signals.append(fused)
        return fused
    
    def _calculate_toxicity_dampening(self, vpin: float, ofi_zscore: float) -> float:
        if vpin <= self.VPIN_DAMPENING_START:
            vpin_factor = 1.0
        elif vpin >= self.VPIN_DAMPENING_FULL:
            vpin_factor = 0.2
        else:
            vpin_factor = 1.0 - 0.8 * (vpin - self.VPIN_DAMPENING_START) / (
                self.VPIN_DAMPENING_FULL - self.VPIN_DAMPENING_START)
        
        ofi_factor = 0.5 if abs(ofi_zscore) > 3.0 else (0.7 if abs(ofi_zscore) > 2.0 else 1.0)
        return vpin_factor * ofi_factor
    
    def _calculate_regime_adjustment(self, regime: str, confidence: float, signal: float) -> float:
        regime_directions = {
            'STRONG_BULL': 1.0, 'MODERATE_BULL': 0.5, 'NEUTRAL': 0.0,
            'VOLATILE_CHOP': 0.0, 'MODERATE_BEAR': -0.5, 'STRONG_BEAR': -1.0,
        }
        expected = regime_directions.get(regime, 0.0)
        if expected == 0.0:
            return 0.7
        if np.sign(signal) == np.sign(expected):
            return 1.0 + confidence * 0.1
        return 1.0 - confidence * 0.3
    
    def _calculate_api_cost_factor(self, quota: float, confidence: float) -> float:
        if quota > 0.5:
            return 1.0
        if quota > 0.2:
            return 0.9 if confidence > 0.7 else 0.6
        return 0.7 if confidence > 0.85 else 0.3
    
    def _determine_execution_mode(self, vpin: float, regime_conf: float,
                                  signal_conf: float, circuit: bool) -> str:
        if circuit:
            return 'EMERGENCY'
        if vpin > 0.7:
            return 'HIDDEN'
        if vpin > 0.5:
            return 'PASSIVE'
        if signal_conf > 0.8 and regime_conf > 0.7:
            return 'AGGRESSIVE'
        return 'NORMAL' if signal_conf > 0.6 else 'CONSERVATIVE'
    
    def calculate_reward(self, period_return: float, trade_value_usd: float,
                        is_taker: bool, api_calls: Dict[str, int],
                        regime_correct: bool) -> RewardComponents:
        return self.reward_function.calculate_reward(
            period_return, trade_value_usd, is_taker, api_calls,
            regime_correct, self.reward_function.get_current_drawdown()
        )
    
    def record_signal_performance(self, fused: FusedSignal, actual_return: float):
        correct = (fused.signal > 0 and actual_return > 0) or (fused.signal < 0 and actual_return < 0)
        total = abs(fused.quant_contribution) + abs(fused.drl_contribution) + 1e-6
        q_share = abs(fused.quant_contribution) / total
        d_share = abs(fused.drl_contribution) / total
        
        self.weight_calculator.record_performance('quant', actual_return * q_share, correct)
        self.weight_calculator.record_performance('drl', actual_return * d_share, correct)
        
        self.signal_performance.append({
            'timestamp': time.time(), 'signal': fused.signal, 'actual_return': actual_return,
            'correct': correct, 'quant_weight': fused.quant_weight, 'drl_weight': fused.drl_weight,
        })
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        if not self.signal_performance:
            return {}
        
        correct = sum(1 for s in self.signal_performance if s['correct'])
        returns = [s['actual_return'] for s in self.signal_performance]
        
        return {
            'accuracy': correct / len(self.signal_performance),
            'total_signals': len(self.signal_performance),
            'avg_quant_weight': np.mean([s['quant_weight'] for s in self.signal_performance]),
            'avg_drl_weight': np.mean([s['drl_weight'] for s in self.signal_performance]),
            'avg_return': np.mean(returns),
            'total_return': np.sum(returns),
            **self.reward_function.get_stats(),
        }
    
    def reset(self):
        self.reward_function.reset()
        self.signal_performance.clear()


def example_usage():
    """Demonstrate Signal Fusion Module."""
    fusion = AdvancedSignalFusionModule()
    np.random.seed(42)
    
    for i in range(50):
        quant = QuantSignal(time.time(), 'SOL', np.random.uniform(-0.5, 0.5),
                           np.random.uniform(0.5, 0.9), np.random.randint(1, 5),
                           regime='MODERATE_BULL' if np.random.random() > 0.3 else 'NEUTRAL')
        
        drl = DRLSignal(time.time(), 'SOL', np.random.uniform(-0.5, 0.5),
                       np.random.uniform(0.4, 0.8), np.random.uniform(0.3, 1.0),
                       np.random.uniform(-1, 1))
        
        fused = fusion.fuse_signals(quant, drl, np.random.uniform(0.4, 0.9),
                                   np.random.uniform(0.2, 0.6), np.random.uniform(-2, 2))
        
        actual_return = np.random.normal(fused.signal * 0.01, 0.02)
        fusion.record_signal_performance(fused, actual_return)
        fusion.calculate_reward(actual_return, 10000, True, {'claude': 1},
                               fused.signal * actual_return > 0)
        
        if i % 10 == 0:
            print(f"Period {i}: Signal={fused.signal:.3f} Mode={fused.execution_mode}")
    
    print("\n=== Statistics ===")
    for k, v in fusion.get_fusion_stats().items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == '__main__':
    example_usage()

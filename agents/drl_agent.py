"""
================================================================================
HMATS v3.4 - DRL AGENT ROLE (SOTA CORRECT USAGE)
================================================================================
Version: 3.4.0
Upgrade From: v3.3-HR (where DRL was disabled)
Purpose: Define correct DRL role for profitability optimization

CRITICAL SCOPE DEFINITION:

    DRL DOES NOT DECIDE:
        × Direction (long/short)
        × Target exposure
        × Entry timing
        × Stop loss levels
    
    DRL ONLY DECIDES:
        ✓ Tranche escalation timing (when to go from T2 -> T3)
        ✓ Exit pressure (urgency of exit)
        ✓ Runner hold/release (when to close final portion)

WHY THIS IS SOTA CORRECT:
    - DRL is excellent at LOCAL OPTIMIZATION (timing, sizing)
    - DRL is POOR at GLOBAL DIRECTION (trend prediction)
    - This design plays to DRL strengths
    - Reward is aligned with realized PnL and profit retention

================================================================================
### DRL TRAINING CONSTRAINT (MANDATORY) ###

If DRL training is proposed:
    - Training MUST be offline first using historical quant-generated trajectories
    - Online learning, if any, MUST be paper-trading only with extremely low
      learning rate
    - DRL must never learn direction or entry timing

================================================================================
"""

import json
import logging
import numpy as np
import os
import sys
import time as _time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timezone
from enum import Enum

try:
    from market.phase_detector import RegimePhase
except ImportError:
    RegimePhase = None

# Stable-Baselines3 import with graceful fallback
try:
    from stable_baselines3 import PPO, SAC, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = SAC = A2C = None
    DummyVecEnv = VecNormalize = None

# TQC (sb3-contrib) - separate import since it's an extension
try:
    from sb3_contrib import TQC
    TQC_AVAILABLE = True
except ImportError:
    TQC_AVAILABLE = False
    TQC = None

logger = logging.getLogger(__name__)


# ============================================================================
# DRL MODE (DISABLED / SHADOW / EXIT_ONLY)
# ============================================================================

class DRLMode(Enum):
    """DRL operating mode. Controls output constraints."""
    DISABLED = "DISABLED"       # Neutral output always
    SHADOW = "SHADOW"           # Log recommendations, no effect
    EXIT_ONLY = "EXIT_ONLY"     # Can recommend exits/reduce only; never entries
    ACTIVE = "ACTIVE"           # Full participation: entries + exits


# ============================================================================
# HMATS PAYLOAD ADAPTER
# ============================================================================

@dataclass
class DRLAgentPayload:
    """
    HMATS-facing DRL output.

    Consumed by runtime_spine.py and Authority Fusion.
    Converts DRLOutput into flat dict with standardized keys.
    """
    asset: str = ""
    direction: float = 0.0          # -1 to 1 (from exit pressure / escalation intent)
    confidence: float = 0.0         # 0 to 1
    action: str = "HOLD"            # DRLAction name
    exit_pressure: float = 0.0      # 0 to 1
    tranche_advice: Optional[str] = None  # "ESCALATE" or None
    exit_signal: Optional[str] = None     # "PARTIAL_EXIT", "RELEASE_RUNNER", or None
    entropy: float = 1.0
    data_quality: float = 1.0       # 0-1 input data completeness
    data_age_seconds: float = 0.0
    is_valid: bool = True
    mode: str = "DISABLED"          # DRLMode.value - transparency field
    leverage_cap: float = 1.0       # Soft leverage cap adjustment (0..1)
    meta: Dict = field(default_factory=dict)

    # V6 spec aliases
    @property
    def drl_direction(self) -> float:
        return self.direction

    @property
    def drl_confidence(self) -> float:
        return self.confidence

    @property
    def drl_exit_pressure(self) -> float:
        return self.exit_pressure

    @property
    def drl_tranche_advice(self) -> Optional[str]:
        return self.tranche_advice

    def to_agent_signal_dict(self) -> Dict:
        """Convert to flat dict for Authority Fusion / runtime_spine consumption."""
        return {
            "drl_direction": self.direction,
            "drl_confidence": self.confidence,
            "drl_exit_pressure": self.exit_pressure,
            "drl_tranche_advice": self.tranche_advice,
            "direction": self.direction,
            "confidence": self.confidence,
            "action": self.action,
            "exit_pressure": self.exit_pressure,
            "tranche_advice": self.tranche_advice,
            "exit_signal": self.exit_signal,
            "entropy": self.entropy,
            "asset": self.asset,
            "data_quality": self.data_quality,
            "data_age_seconds": self.data_age_seconds,
            "is_valid": self.is_valid,
            "mode": self.mode,
            "leverage_cap": self.leverage_cap,
            "asof_timestamp": _time.time(),
            "meta": self.meta,
        }


def _neutral_payload(asset: str = "") -> DRLAgentPayload:
    """Return a safe neutral payload (no recommendation)."""
    return DRLAgentPayload(
        asset=asset,
        direction=0.0,
        confidence=0.0,
        action="HOLD",
        entropy=1.0,
        is_valid=True,
        meta={"reason": "neutral_default"},
    )


# ============================================================================
# DRL ENABLE FLAG
# ============================================================================

# DRL agent wrapper (legacy). Real DRL inference runs via TQC ensemble
# in main.py (_drl_ensembles), controlled by PromotionGate authority level.
# This flag only governs the old DRLAgent class, NOT the ensemble system.
ENABLE_DRL = True


# ============================================================================
# ACTION SPACE
# ============================================================================

class DRLAction(Enum):
    """
    DRL action space (discrete, 5 actions).
    
    CRITICAL: None of these actions decide DIRECTION or ENTRY.
    """
    HOLD = 0                    # No change
    ESCALATE = 1                # Signal tranche escalation
    PARTIAL_EXIT = 2            # Signal 25% scale-out
    INCREASE_EXIT_PRESSURE = 3  # Tighten runner stop
    RELEASE_RUNNER = 4          # Close remaining runner


# Action descriptions for logging
ACTION_DESCRIPTIONS = {
    DRLAction.HOLD: "Maintain current state",
    DRLAction.ESCALATE: "Signal tranche escalation (if conditions allow)",
    DRLAction.PARTIAL_EXIT: "Signal 25% position scale-out",
    DRLAction.INCREASE_EXIT_PRESSURE: "Tighten runner stop to 1%",
    DRLAction.RELEASE_RUNNER: "Close remaining runner position",
}


# ============================================================================
# STATE SPACE
# ============================================================================

@dataclass
class DRLState:
    """
    DRL state vector (48 dimensions).
    
    ORGANIZED BY CATEGORY:
        - Position State (8): Current position info
        - Regime State (12): Market phase info
        - Market State (16): Price/volume/flow info
        - Time State (4): Temporal features
        - Signal State (8): Agent signal info
    """
    
    # Position State (8)
    current_tranche: int = 0
    position_pnl_bps: float = 0.0
    time_in_position_bars: int = 0
    avg_entry_vs_current: float = 0.0
    unrealized_high_watermark: float = 0.0
    drawdown_from_high: float = 0.0
    stop_distance_pct: float = 0.02
    runner_active: bool = False
    
    # Regime State (12)
    regime_phase_ignition: float = 0.0
    regime_phase_expansion: float = 0.0
    regime_phase_saturation: float = 0.0
    regime_phase_exhaustion: float = 0.0
    phase_age_bars: int = 0
    opportunity_density: float = 0.5
    crack_weight: float = 0.0
    sol_phase_early: float = 0.0  # 1 if IGNITION/EXPANSION
    sol_phase_late: float = 0.0   # 1 if SATURATION/EXHAUSTION
    btc_phase_early: float = 0.0
    btc_phase_late: float = 0.0
    eth_phase_early: float = 0.0
    
    # Market State (16)
    momentum_1bar: float = 0.0
    momentum_4bar: float = 0.0
    volume_ratio: float = 1.0
    vpin: float = 0.5
    volatility_zscore: float = 0.0
    correlation_btc_sol: float = 0.5
    correlation_eth_sol: float = 0.5
    orderbook_imbalance: float = 0.0
    funding_rate: float = 0.0
    lead_lag_edge: float = 0.0
    lead_lag_confidence: float = 0.0
    dvol_zscore: float = 0.0
    rsi: float = 50.0
    price_vs_vwap: float = 0.0
    atr_ratio: float = 1.0
    spread_bps: float = 10.0
    
    # Time State (4) - cyclical encoding
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    day_sin: float = 0.0
    day_cos: float = 0.0
    
    # Signal State (8)
    quant_direction: float = 0.0
    quant_confidence: float = 0.0
    sentiment_zscore: float = 0.0
    macro_leverage_cap: float = 1.0
    regime_trend_strength: float = 0.0
    signal_conflict_level: float = 0.0
    authority_clarity: float = 1.0
    mode_numeric: float = 2.0  # 0=NO_TRADE, 1=OPPORTUNITY, 2=NORMAL
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            # Position (8)
            self.current_tranche / 4.0,
            self.position_pnl_bps / 500.0,  # Normalize to [-1, 1] range
            min(self.time_in_position_bars / 20.0, 1.0),
            self.avg_entry_vs_current,
            self.unrealized_high_watermark / 500.0,
            self.drawdown_from_high,
            self.stop_distance_pct * 50,  # 2% -> 1.0
            float(self.runner_active),
            
            # Regime (12)
            self.regime_phase_ignition,
            self.regime_phase_expansion,
            self.regime_phase_saturation,
            self.regime_phase_exhaustion,
            min(self.phase_age_bars / 10.0, 1.0),
            self.opportunity_density,
            self.crack_weight,
            self.sol_phase_early,
            self.sol_phase_late,
            self.btc_phase_early,
            self.btc_phase_late,
            self.eth_phase_early,
            
            # Market (16)
            self.momentum_1bar / 5.0,
            self.momentum_4bar / 10.0,
            min(self.volume_ratio / 3.0, 1.0),
            self.vpin,
            self.volatility_zscore / 3.0,
            self.correlation_btc_sol,
            self.correlation_eth_sol,
            self.orderbook_imbalance,
            self.funding_rate * 100,
            self.lead_lag_edge,
            self.lead_lag_confidence,
            self.dvol_zscore / 3.0,
            (self.rsi - 50) / 50,
            self.price_vs_vwap,
            self.atr_ratio - 1.0,
            min(self.spread_bps / 20.0, 1.0),
            
            # Time (4)
            self.hour_sin,
            self.hour_cos,
            self.day_sin,
            self.day_cos,
            
            # Signal (8)
            self.quant_direction,
            self.quant_confidence,
            self.sentiment_zscore / 3.0,
            self.macro_leverage_cap,
            self.regime_trend_strength,
            self.signal_conflict_level,
            self.authority_clarity,
            self.mode_numeric / 2.0,
        ], dtype=np.float32)
    
    @staticmethod
    def from_phase(phase: RegimePhase) -> Tuple[float, float, float, float]:
        """Convert phase to one-hot encoding."""
        return (
            1.0 if phase == RegimePhase.IGNITION else 0.0,
            1.0 if phase == RegimePhase.EXPANSION else 0.0,
            1.0 if phase == RegimePhase.SATURATION else 0.0,
            1.0 if phase == RegimePhase.EXHAUSTION else 0.0,
        )


# ============================================================================
# REWARD FUNCTION
# ============================================================================

@dataclass
class DRLRewardComponents:
    """Components of DRL reward."""
    pnl_reward: float = 0.0
    retention_reward: float = 0.0
    tranche_reward: float = 0.0
    exit_timing_reward: float = 0.0
    total: float = 0.0


def calculate_drl_reward(
    state: DRLState,
    action: DRLAction,
    next_state: DRLState,
    realized_pnl_bps: float = 0.0,
    peak_unrealized_pnl: float = 0.0,
    next_bar_favorable: bool = False,
    exit_bar: int = 0,
    optimal_exit_bar: int = 0,
) -> DRLRewardComponents:
    """
    Calculate DRL reward aligned with realized PnL and profit retention.
    
    REWARD WEIGHTS:
        - Realized PnL: 0.5 (primary)
        - Profit Retention: 0.25
        - Tranche Efficiency: 0.15
        - Exit Timing: 0.10
    
    WHY THIS ALIGNMENT:
        DRL should optimize for ACTUAL PROFITS, not predicted moves.
        Profit retention rewards locking gains instead of giving back.
    """
    
    components = DRLRewardComponents()
    
    # 1. Realized PnL (Primary, weight=0.5)
    # Directly rewards profitable closes
    components.pnl_reward = realized_pnl_bps / 100.0
    
    # 2. Profit Retention (weight=0.25)
    # Rewards keeping profits vs giving them back
    if realized_pnl_bps > 0 and peak_unrealized_pnl > 0:
        retention = realized_pnl_bps / peak_unrealized_pnl
        components.retention_reward = retention * 0.5
    
    # 3. Tranche Efficiency (weight=0.15)
    # Rewards good escalation decisions
    if action == DRLAction.ESCALATE:
        if next_bar_favorable:
            components.tranche_reward = 0.1
        else:
            components.tranche_reward = -0.15  # Penalize bad escalation
    
    # 4. Exit Timing (weight=0.10)
    # Rewards exiting close to optimal
    if action in [DRLAction.PARTIAL_EXIT, DRLAction.RELEASE_RUNNER]:
        bars_from_optimal = abs(exit_bar - optimal_exit_bar)
        components.exit_timing_reward = max(0, 0.2 - bars_from_optimal * 0.05)
    
    # Calculate weighted total
    components.total = (
        0.50 * components.pnl_reward +
        0.25 * components.retention_reward +
        0.15 * components.tranche_reward +
        0.10 * components.exit_timing_reward
    )
    
    return components


# ============================================================================
# DRL AGENT INTERFACE
# ============================================================================

@dataclass
class DRLOutput:
    """Output from DRL agent."""
    action: DRLAction = DRLAction.HOLD
    action_probabilities: Dict[DRLAction, float] = field(default_factory=dict)
    entropy: float = 1.0  # High entropy = uncertain
    confidence: float = 0.0
    uncertainty_ratio: float = 0.0  # [v6.7-P1] TQC quantile spread / |median_q|

    @property
    def is_confident(self) -> bool:
        """True if model is confident (entropy < 0.7)."""
        return self.entropy < 0.7


class DRLAgent:
    """
    DRL Agent for tranche/exit optimization.
    
    USAGE:
        agent = DRLAgent()
        
        # Build state from current context
        state = agent.build_state(context)
        
        # Get action
        output = agent.get_action(state)
        
        # DRL action is ADVISORY in OPPORTUNITY mode
        # Cannot veto entry!
    
    CRITICAL: DRL silence ≠ veto in OPPORTUNITY
        If DRL outputs HOLD, it does NOT block entry.
        It simply means DRL has no tranche/exit recommendation.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        mode: str = "DISABLED",
        obs_dim: int = 126,
        n_stack: int = 1,
    ):
        # Mode: DISABLED / SHADOW / EXIT_ONLY (read from config or env)
        _mode_str = os.environ.get("HMATS_DRL_MODE", mode).upper()
        try:
            self.mode = DRLMode(_mode_str)
        except ValueError:
            self.mode = DRLMode.DISABLED

        self.enabled = ENABLE_DRL and self.mode != DRLMode.DISABLED
        self.model = None
        self._model_path = model_path or os.environ.get("HMATS_DRL_MODEL_PATH", "")
        self.vec_normalize = None
        self._stage9_config: Dict[str, Any] = {}

        # Observation dimensions from Stage 9 config
        self.obs_dim = obs_dim
        self.n_stack = n_stack

        # Per-asset obs ring buffer for multi-timestep models (LSTM/FiLM)
        self._obs_buffers: Dict[str, deque] = {}

        # Per-asset tracking
        self._last_payloads: Dict[str, DRLAgentPayload] = {}
        self._last_outputs: Dict[str, DRLOutput] = {}

        # Load Stage 9 config if available
        self._load_stage9_config()

        if self.enabled and self._model_path:
            self._load_model(self._model_path)

        logger.info(
            f"DRL Agent: mode={self.mode.value}, enabled={self.enabled}, "
            f"obs_dim={self.obs_dim}, n_stack={self.n_stack}"
        )

    def _load_stage9_config(self):
        """Load Stage 9 deployment config (obs_dim, n_stack, extractor) if available."""
        for cfg_name in ["stage9_config.json", "drl_config.json"]:
            # Check next to model
            for base in [os.path.dirname(self._model_path) if self._model_path else "", "models"]:
                cfg_path = os.path.join(base, cfg_name) if base else cfg_name
                if os.path.exists(cfg_path):
                    try:
                        with open(cfg_path, "r") as f:
                            self._stage9_config = json.load(f)
                        self.obs_dim = self._stage9_config.get("obs_dim", self.obs_dim)
                        self.n_stack = self._stage9_config.get("n_stack", self.n_stack)
                        logger.info(f"[DRL] Stage 9 config loaded: {cfg_path} (obs_dim={self.obs_dim}, n_stack={self.n_stack})")
                        return
                    except Exception as e:
                        logger.debug(f"[DRL] Stage 9 config load failed ({cfg_path}): {e}")

    def _get_obs_buffer(self, asset: str) -> deque:
        """Get or create per-asset observation ring buffer."""
        if asset not in self._obs_buffers:
            self._obs_buffers[asset] = deque(maxlen=max(self.n_stack, 256))
        return self._obs_buffers[asset]
    
    def _load_model(self, path: str):
        """Load trained model checkpoint."""
        if not SB3_AVAILABLE and not TQC_AVAILABLE:
            logger.warning("stable-baselines3/sb3-contrib not available - DRL will use heuristics")
            self.enabled = False
            return

        try:
            # Detect algorithm from path or default to PPO
            algorithm = "ppo"
            if "tqc" in path.lower():
                algorithm = "tqc"
            elif "sac" in path.lower():
                algorithm = "sac"
            elif "a2c" in path.lower():
                algorithm = "a2c"

            logger.info(f"Loading {algorithm.upper()} model from {path}")

            # Strip .zip extension if present (SB3 auto-appends .zip)
            load_path = path[:-4] if path.endswith(".zip") else path

            if algorithm == "tqc":
                if not TQC_AVAILABLE:
                    logger.warning("sb3-contrib not available for TQC model")
                    self.enabled = False
                    return
                # Register module aliases so cloudpickle can deserialize LSTM_FILM_A natively.
                # Do NOT use custom_objects with policy_kwargs - SB3's json_to_data()
                # REPLACES (not updates) the dict, which drops n_quantiles, net_arch, etc.
                import importlib as _il
                if "models" not in sys.modules or sys.modules["models"] is None:
                    try:
                        sys.modules["models"] = _il.import_module("training.models")
                        sys.modules["models.film_extractor"] = _il.import_module("training.models.film_extractor")
                        sys.modules["models.feature_extractors"] = _il.import_module("training.models.feature_extractors")
                        logger.info("[DRL] Registered training.models aliases for cloudpickle")
                    except Exception as _e:
                        logger.debug(f"[DRL] Module alias registration skipped: {_e}")
                self.model = TQC.load(load_path)
            elif algorithm == "ppo":
                self.model = PPO.load(load_path)
            elif algorithm == "sac":
                self.model = SAC.load(load_path)
            elif algorithm == "a2c":
                self.model = A2C.load(load_path)
            
            # Load normalizer if exists
            normalizer_path = f"{path}_normalizer.pkl"
            if os.path.exists(normalizer_path):
                self.vec_normalize = VecNormalize.load(normalizer_path, DummyVecEnv([lambda: None]))
                logger.info(f"Loaded state normalizer from {normalizer_path}")
            else:
                self.vec_normalize = None
            
            logger.info(f"DRL model loaded successfully ({algorithm.upper()})")
            
        except Exception as e:
            logger.error(f"Failed to load DRL model: {e}")
            self.model = None
            self.enabled = False
    
    def build_state(
        self,
        position_state: Dict,
        regime_result,  # RegimePhaseResult
        market_data: Dict,
        agent_signals: Dict,
        system_mode: int,  # 0=NO_TRADE, 1=OPPORTUNITY, 2=NORMAL
    ) -> DRLState:
        """Build DRL state from current context."""
        
        state = DRLState()
        
        # Position state
        state.current_tranche = position_state.get("tranche", 0)
        state.position_pnl_bps = position_state.get("pnl_bps", 0.0)
        state.time_in_position_bars = position_state.get("bars_held", 0)
        state.avg_entry_vs_current = position_state.get("entry_vs_current", 0.0)
        state.unrealized_high_watermark = position_state.get("high_water", 0.0)
        state.drawdown_from_high = position_state.get("drawdown", 0.0)
        state.stop_distance_pct = position_state.get("stop_distance", 0.02)
        state.runner_active = position_state.get("runner_active", False)
        
        # Regime state
        if regime_result:
            phase_encoding = DRLState.from_phase(regime_result.phase)
            state.regime_phase_ignition = phase_encoding[0]
            state.regime_phase_expansion = phase_encoding[1]
            state.regime_phase_saturation = phase_encoding[2]
            state.regime_phase_exhaustion = phase_encoding[3]
            state.phase_age_bars = regime_result.phase_age_bars
            state.opportunity_density = regime_result.opportunity_density
            
            # Asset phases
            state.sol_phase_early = 1.0 if regime_result.phase_sol in [
                RegimePhase.IGNITION, RegimePhase.EXPANSION
            ] else 0.0
            state.sol_phase_late = 1.0 if regime_result.phase_sol in [
                RegimePhase.SATURATION, RegimePhase.EXHAUSTION
            ] else 0.0
        
        # Market state
        state.momentum_1bar = market_data.get("momentum_1bar", 0.0)
        state.momentum_4bar = market_data.get("momentum_4bar", 0.0)
        state.volume_ratio = market_data.get("volume_ratio", 1.0)
        state.vpin = market_data.get("vpin", 0.35)
        state.rsi = market_data.get("rsi", 50.0)
        state.lead_lag_edge = market_data.get("lead_lag_edge", 0.0)
        state.lead_lag_confidence = market_data.get("lead_lag_confidence", 0.0)
        
        # Signal state
        state.quant_direction = agent_signals.get("quant_direction", 0.0)
        state.quant_confidence = agent_signals.get("quant_confidence", 0.0)
        state.mode_numeric = float(system_mode)
        
        # Time encoding
        now = datetime.now(timezone.utc)
        hour = now.hour
        day = now.weekday()
        state.hour_sin = np.sin(2 * np.pi * hour / 24)
        state.hour_cos = np.cos(2 * np.pi * hour / 24)
        state.day_sin = np.sin(2 * np.pi * day / 7)
        state.day_cos = np.cos(2 * np.pi * day / 7)
        
        return state
    
    def get_action(self, state: DRLState) -> DRLOutput:
        """
        Get action from DRL model.
        
        CRITICAL: If DRL is disabled or uncertain, returns HOLD.
        HOLD does NOT veto entry - it means no recommendation.
        """
        
        output = DRLOutput()
        
        if not self.enabled or self.model is None:
            # Return HOLD with high entropy (no recommendation)
            output.action = DRLAction.HOLD
            output.entropy = 1.0
            output.confidence = 0.0
            return output
        
        try:
            # Get action from model
            state_vector = state.to_vector()
            
            # Normalize state if normalizer available
            if hasattr(self, 'vec_normalize') and self.vec_normalize is not None:
                state_vector = self.vec_normalize.normalize_obs(state_vector)
            
            # Real model inference
            action_raw, _ = self.model.predict(state_vector.reshape(1, -1), deterministic=True)
            action_value = float(action_raw[0]) if hasattr(action_raw, '__len__') else float(action_raw)
            
            # Map continuous action to discrete DRLAction
            # Action space: [-1, 1] where:
            #   [-1.0, -0.6] = RELEASE_RUNNER (4)
            #   [-0.6, -0.2] = INCREASE_EXIT_PRESSURE (3)
            #   [-0.2,  0.2] = HOLD (0)
            #   [ 0.2,  0.6] = PARTIAL_EXIT (2)
            #   [ 0.6,  1.0] = ESCALATE (1)
            if action_value < -0.6:
                output.action = DRLAction.RELEASE_RUNNER
            elif action_value < -0.2:
                output.action = DRLAction.INCREASE_EXIT_PRESSURE
            elif action_value < 0.2:
                output.action = DRLAction.HOLD
            elif action_value < 0.6:
                output.action = DRLAction.PARTIAL_EXIT
            else:
                output.action = DRLAction.ESCALATE
            
            # Calculate entropy from action distribution if available
            try:
                obs_tensor = self.model.policy.obs_to_tensor(state_vector.reshape(1, -1))[0]
                dist = self.model.policy.get_distribution(obs_tensor)
                output.entropy = float(dist.entropy().mean().item())
            except Exception:
                output.entropy = 0.3  # Default moderate entropy

            # [v6.7-P1] TQC quantile spread -> uncertainty_ratio
            # Accesses critic networks for IQR of Q-value quantiles.
            # Falls back gracefully if SB3 API changes.
            try:
                import torch
                with torch.no_grad():
                    _obs_t = self.model.policy.obs_to_tensor(state_vector.reshape(1, -1))[0]
                    _act_t = torch.tensor(action_raw, dtype=torch.float32).reshape(1, -1)
                    if hasattr(_act_t, 'to'):
                        _act_t = _act_t.to(_obs_t.device)

                    _q_all = []
                    # SB3-contrib TQC: self.model.critic has .quantiles_total attribute
                    # and forward(obs, action) returns (batch, n_quantiles, n_critics)
                    if hasattr(self.model, 'critic'):
                        _q_out = self.model.critic(_obs_t, _act_t)
                        # _q_out shape varies: could be tuple or tensor
                        if isinstance(_q_out, tuple):
                            for _qt in _q_out:
                                _q_all.append(_qt.flatten())
                        else:
                            _q_all.append(_q_out.flatten())

                    if _q_all:
                        _q_cat = torch.cat(_q_all)
                        _q75 = torch.quantile(_q_cat, 0.75).item()
                        _q25 = torch.quantile(_q_cat, 0.25).item()
                        _med = torch.median(_q_cat).item()
                        _iqr = _q75 - _q25
                        output.uncertainty_ratio = _iqr / (abs(_med) + 1e-8)
                        logger.debug(
                            f"[TQC-Uncertainty] IQR={_iqr:.4f}, median={_med:.4f}, "
                            f"ratio={output.uncertainty_ratio:.3f}"
                        )
            except Exception as _tqc_err:
                logger.debug(f"[TQC-Uncertainty] Quantile extraction failed: {_tqc_err}")
                # Fallback: use entropy as proxy (normalize to similar scale)
                output.uncertainty_ratio = output.entropy * 0.5

            # Confidence based on action magnitude (further from 0 = more confident)
            output.confidence = min(abs(action_value), 1.0)
            
        except Exception as e:
            logger.error(f"DRL inference failed: {e}")
            output.action = DRLAction.HOLD
            output.entropy = 1.0
        
        return output
    
    def interpret_action(self, output: DRLOutput) -> Dict:
        """
        Interpret DRL action for consumption by other modules.
        
        Returns dict with:
            - tranche_advice: ESCALATE | HOLD | None
            - exit_signal: PARTIAL_EXIT | RELEASE_RUNNER | None
            - exit_pressure: float 0-1
        """
        
        result = {
            "tranche_advice": None,
            "exit_signal": None,
            "exit_pressure": 0.0,
        }
        
        if output.action == DRLAction.ESCALATE:
            result["tranche_advice"] = "ESCALATE"
        
        elif output.action == DRLAction.PARTIAL_EXIT:
            result["exit_signal"] = "PARTIAL_EXIT"
            result["exit_pressure"] = 0.7
        
        elif output.action == DRLAction.INCREASE_EXIT_PRESSURE:
            result["exit_pressure"] = 0.8
        
        elif output.action == DRLAction.RELEASE_RUNNER:
            result["exit_signal"] = "RELEASE_RUNNER"
            result["exit_pressure"] = 1.0
        
        return result

    # ====================================================================
    # DATA HEALTH GATING
    # ====================================================================

    @staticmethod
    def _validate_build_state_inputs(
        position_state: Dict,
        regime_result,
        market_data: Dict,
        agent_signals: Dict,
        system_mode: int,
    ) -> Tuple[float, List[str]]:
        """
        Validate inputs to build_state(). Returns (data_quality, issues).
        data_quality: 0-1 reflecting completeness.
        issues: list of human-readable warnings.
        """
        issues: List[str] = []
        score = 1.0

        # position_state checks
        if not position_state:
            issues.append("position_state_empty")
            score -= 0.2
        elif not isinstance(position_state, dict):
            issues.append("position_state_not_dict")
            score -= 0.2

        # regime_result checks
        if regime_result is None:
            issues.append("regime_result_none")
            score -= 0.15

        # market_data checks
        if not market_data:
            issues.append("market_data_empty")
            score -= 0.3
        elif not isinstance(market_data, dict):
            issues.append("market_data_not_dict")
            score -= 0.3

        # agent_signals checks
        if not agent_signals:
            issues.append("agent_signals_empty")
            score -= 0.15
        elif not isinstance(agent_signals, dict):
            issues.append("agent_signals_not_dict")
            score -= 0.15

        # system_mode bounds
        if system_mode not in (0, 1, 2):
            issues.append(f"system_mode_invalid:{system_mode}")
            score -= 0.1

        return max(0.0, score), issues

    # ====================================================================
    # HMATS INTEGRATION - generate_signal / get_signal
    # ====================================================================

    def generate_signal(
        self,
        asset: str = "BTC",
        price: float = 0.0,
        regime: str = "",
        position_state: Optional[Dict] = None,
        market_data: Optional[Dict] = None,
        agent_signals: Optional[Dict] = None,
        system_mode: int = 2,
    ) -> DRLAgentPayload:
        """
        HMATS-facing entry point for runtime_spine.py.

        Always returns DRLAgentPayload (never None, never raises).
        Respects DRLMode: DISABLED -> neutral, SHADOW -> recommend only,
        EXIT_ONLY -> suppress ESCALATE, only exits/reduce.

        Args:
            asset: Asset identifier
            price: Current asset price (for meta only)
            regime: Regime name (for meta only)
            position_state: Current position info dict
            market_data: Market data dict for build_state()
            agent_signals: Agent signals dict for build_state()
            system_mode: 0=NO_TRADE, 1=OPPORTUNITY, 2=NORMAL
        """
        # DISABLED mode -> return neutral immediately (still cache for queries)
        if self.mode == DRLMode.DISABLED:
            p = _neutral_payload(asset)
            p.mode = DRLMode.DISABLED.value
            self._last_payloads[asset] = p
            return p

        try:
            pos = position_state or {}
            mkt = market_data or {}
            sigs = agent_signals or {}

            # Data age tracking
            _mkt_ts = mkt.get("_exchange_timestamp", 0.0) or mkt.get("timestamp", 0.0)
            data_age = (_time.time() - _mkt_ts) if _mkt_ts > 0 else 0.0

            # Validate inputs
            data_quality, issues = self._validate_build_state_inputs(
                pos, None, mkt, sigs, system_mode,
            )

            # Store observation in ring buffer
            obs_buf = self._get_obs_buffer(asset)

            # Build state and get action
            state = self.build_state(
                position_state=pos,
                regime_result=None,
                market_data=mkt,
                agent_signals=sigs,
                system_mode=system_mode,
            )
            output = self.get_action(state)
            interp = self.interpret_action(output)

            # NaN / out-of-range safety clamp
            if np.isnan(output.confidence) or np.isinf(output.confidence):
                output = DRLOutput(
                    action=DRLAction.HOLD, probabilities=output.probabilities,
                    entropy=1.0, confidence=0.0,
                )
                issues.append("confidence_nan_clamped")
                data_quality = max(0.0, data_quality - 0.3)

            # Convert to direction for Authority Fusion:
            # Exit actions -> negative direction (want to reduce)
            # Escalate -> positive direction (want to increase)
            # Hold -> 0 (no recommendation)
            direction = 0.0
            suppressed = False
            if output.action == DRLAction.ESCALATE:
                # EXIT_ONLY mode: suppress escalation
                if self.mode == DRLMode.EXIT_ONLY:
                    direction = 0.0
                    suppressed = True
                else:
                    direction = output.confidence * 0.5  # Mild positive
            elif output.action in (DRLAction.PARTIAL_EXIT, DRLAction.INCREASE_EXIT_PRESSURE):
                direction = -output.confidence * 0.5  # Mild negative
            elif output.action == DRLAction.RELEASE_RUNNER:
                direction = -output.confidence * 0.8  # Strong exit

            # Clamp direction to [-1, 1]
            direction = float(np.clip(direction, -1.0, 1.0))

            # Leverage cap: high exit pressure -> reduce leverage allowance
            leverage_cap = 1.0 - interp["exit_pressure"] * 0.3  # 0.7 at max pressure

            payload = DRLAgentPayload(
                asset=asset,
                direction=direction,
                confidence=float(np.clip(output.confidence, 0.0, 1.0)),
                action=output.action.name,
                exit_pressure=interp["exit_pressure"],
                tranche_advice=interp["tranche_advice"],
                exit_signal=interp["exit_signal"],
                entropy=output.entropy,
                data_quality=data_quality,
                data_age_seconds=round(data_age, 1),
                is_valid=True,
                mode=self.mode.value,
                leverage_cap=leverage_cap,
                meta={
                    "price": price,
                    "regime": regime,
                    "issues": issues,
                    "action_value": output.action.value,
                    "is_confident": output.is_confident,
                    "suppressed_by_mode": suppressed,
                },
            )

            # Cache per-asset
            self._last_payloads[asset] = payload
            self._last_outputs[asset] = output

            return payload

        except Exception as e:
            logger.error(f"[DRL] generate_signal failed for {asset}: {e}", exc_info=True)
            p = _neutral_payload(asset)
            p.mode = self.mode.value
            return p

    def get_signal(
        self,
        asset: str = "BTC",
        market_data: Optional[Dict] = None,
    ) -> Tuple[float, float]:
        """
        Compatibility method for integrated_system.py.

        Returns (direction, confidence) tuple.
        """
        payload = self.generate_signal(asset=asset, market_data=market_data)
        return payload.direction, payload.confidence

    def get_last_payload(self, asset: str = "BTC") -> DRLAgentPayload:
        """Get cached payload for an asset. Returns neutral if none cached."""
        return self._last_payloads.get(asset, _neutral_payload(asset))

    def reset_state(self, asset: Optional[str] = None):
        """Reset per-asset caches. If asset=None, reset all."""
        if asset:
            self._last_payloads.pop(asset, None)
            self._last_outputs.pop(asset, None)
        else:
            self._last_payloads.clear()
            self._last_outputs.clear()


# ============================================================================
# TRAINING SPECIFICATION (FOR FUTURE IMPLEMENTATION)
# ============================================================================

"""
DRL TRAINING SPECIFICATION:

Algorithm: PPO (Proximal Policy Optimization)
    - Stable with discrete actions
    - Good for financial time series
    - Handles partial observability

Data Requirement:
    - Minimum: 10,000 episodes (simulated trades)
    - Recommended: 50,000 episodes
    - Must include various regime phases
    - Must include both long and short positions

Validation Protocol:
    - Walk-forward: 70% train, 15% validate, 15% test
    - Out-of-sample Sharpe must exceed random policy
    - Entropy threshold: < 0.7 (model must be confident)

MANDATORY CONSTRAINTS:
    1. Training MUST be offline first using historical quant-generated trajectories
    2. Online learning MUST be paper-trading only
    3. Online learning rate MUST be extremely low (< 1e-5)
    4. DRL must NEVER learn direction or entry timing

Reward Function:
    - Primary: Realized PnL (50%)
    - Profit retention (25%)
    - Tranche efficiency (15%)
    - Exit timing (10%)

State Normalization:
    - All features normalized to approximately [-1, 1]
    - Use running statistics for online normalization
"""


# Singleton
_drl_agent: Optional[DRLAgent] = None

def get_drl_agent(**kwargs) -> DRLAgent:
    global _drl_agent
    model_path = kwargs.get("model_path")
    obs_dim = kwargs.get("obs_dim", 126)
    n_stack = kwargs.get("n_stack", 1)
    requested_mode = kwargs.get("mode", "DISABLED")
    effective_mode = os.environ.get("HMATS_DRL_MODE", requested_mode).upper()
    effective_model_path = model_path or os.environ.get("HMATS_DRL_MODEL_PATH", "")

    if _drl_agent is None:
        _drl_agent = DRLAgent(**kwargs)
    elif (
        _drl_agent.mode.value != effective_mode
        or _drl_agent._model_path != effective_model_path
        or _drl_agent.obs_dim != obs_dim
        or _drl_agent.n_stack != n_stack
    ):
        logger.info("[DRL] Constructor inputs changed; refreshing singleton instance")
        _drl_agent = DRLAgent(**kwargs)
    return _drl_agent

def reset_drl_agent():
    global _drl_agent
    _drl_agent = None


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    """Dry-run self-test: feed random obs, get valid output."""
    logging.basicConfig(level=logging.INFO)
    agent = DRLAgent(mode="EXIT_ONLY")
    for asset in ["BTC", "ETH", "SOL"]:
        payload = agent.generate_signal(
            asset=asset, price=50000.0, regime="UNKNOWN",
            market_data={"current_price": 50000.0},
            position_state={"tranche": 1, "exposure": 0.05},
        )
        sig = payload.to_agent_signal_dict()
        print(f"{asset}: dir={sig['drl_direction']:+.2f} conf={sig['drl_confidence']:.2f} "
              f"exit_p={sig['drl_exit_pressure']:.2f} mode={sig['mode']} "
              f"advice={sig['drl_tranche_advice']}")
        assert isinstance(sig, dict)
        assert "drl_direction" in sig
        assert "drl_exit_pressure" in sig
        assert sig["mode"] == "EXIT_ONLY"
        # EXIT_ONLY: direction must be <= 0 (never positive entry)
        assert sig["drl_direction"] <= 0, f"EXIT_ONLY must not suggest entry, got {sig['drl_direction']}"
    print("\nAll self-tests passed.")

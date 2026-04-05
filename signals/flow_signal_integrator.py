"""
================================================================================
HMATS v6.1.2 - Flow Signal Integrator
================================================================================
Purpose: 

:
1. OnChainFeed -  (whale, exchange flow)
2. GlobalContextInformer - ETF ?3. SOLDefenseModule - SOL ito Tips

:
- whale_flow:  (-1 to 1)
- exchange_flow:  (-1 to 1, ?=)
- etf_flow: ETF  (-1 to 1)
- network_health: ?(0 to 1)

Author: HMATS Flow Integration
Version: 6.1.2
================================================================================
"""

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FlowSignalType(Enum):
    """Flow signal types."""
    WHALE_FLOW = "whale_flow"
    EXCHANGE_FLOW = "exchange_flow"
    ETF_FLOW = "etf_flow"
    STAKING_FLOW = "staking_flow"
    NETWORK_HEALTH = "network_health"
    JITO_PRESSURE = "jito_pressure"


@dataclass
class FlowSignal:
    """Individual flow signal."""
    signal_type: FlowSignalType
    direction: float = 0.0      # -1 (bearish) to 1 (bullish)
    confidence: float = 0.0     # 0 to 1
    magnitude: float = 0.0      # Absolute magnitude (USD or units)
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "type": self.signal_type.value,
            "direction": self.direction,
            "confidence": self.confidence,
            "magnitude": self.magnitude,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class FlowState:
    """Aggregated flow state."""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Whale signals
    whale_direction: float = 0.0
    whale_confidence: float = 0.0
    whale_net_flow_usd: float = 0.0
    
    # Exchange flow signals
    exchange_direction: float = 0.0
    exchange_confidence: float = 0.0
    exchange_net_flow_usd: float = 0.0
    
    # ETF signals (BTC/ETH)
    btc_etf_direction: float = 0.0
    eth_etf_direction: float = 0.0
    etf_confidence: float = 0.0
    etf_streak_days: int = 0
    
    # Network health (SOL)
    sol_network_health: float = 1.0  # 0-1
    jito_pressure: float = 0.0       # 0-1 (high = congested)
    
    # Combined signals
    combined_direction: float = 0.0
    combined_confidence: float = 0.0
    
    # Flags
    whale_alert: bool = False
    etf_outflow_caution: bool = False
    network_degraded: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "whale": {
                "direction": self.whale_direction,
                "confidence": self.whale_confidence,
                "net_flow_usd": self.whale_net_flow_usd,
                "alert": self.whale_alert,
            },
            "exchange": {
                "direction": self.exchange_direction,
                "confidence": self.exchange_confidence,
                "net_flow_usd": self.exchange_net_flow_usd,
            },
            "etf": {
                "btc_direction": self.btc_etf_direction,
                "eth_direction": self.eth_etf_direction,
                "confidence": self.etf_confidence,
                "streak_days": self.etf_streak_days,
                "outflow_caution": self.etf_outflow_caution,
            },
            "network": {
                "sol_health": self.sol_network_health,
                "jito_pressure": self.jito_pressure,
                "degraded": self.network_degraded,
            },
            "combined": {
                "direction": self.combined_direction,
                "confidence": self.combined_confidence,
            },
        }


@dataclass
class FlowIntegratorConfig:
    """Configuration for flow signal integrator."""
    # Thresholds
    whale_threshold_usd: float = 1_000_000      # $1M = whale
    exchange_flow_threshold: float = 10_000_000  # $10M significant
    etf_outflow_streak_alert: int = 3           # 3 days consecutive outflow
    
    # Weights for combined signal
    whale_weight: float = 0.25
    exchange_weight: float = 0.25
    etf_weight: float = 0.30
    network_weight: float = 0.20
    
    # Confidence decay
    stale_data_seconds: int = 300  # 5 min = stale
    
    # Mock mode
    mock_mode: bool = False  # [FIX-2] Default to real data; mock only if sources unavailable


class FlowSignalIntegrator:
    """
    Integrates all flow signals into unified trading signals.
    
    Sources:
    - OnChainFeed: whale activity, exchange flows
    - GlobalContextInformer: ETF flows, macro indicators
    - SOLDefenseModule: network health, Jito tips
    """
    
    def __init__(self, config: FlowIntegratorConfig = None):
        self.config = config or FlowIntegratorConfig()
        
        # Component references (set externally)
        self._onchain_feed = None
        self._global_context = None
        self._sol_defense = None
        
        # State
        self._current_state = FlowState()
        self._signals: Dict[str, FlowSignal] = {}
        self._last_update = datetime.now()
        
        # Initialize components
        self._init_components()
        
        logger.info(f"FlowSignalIntegrator initialized (mock_mode={self.config.mock_mode})")
    
    def _init_components(self):
        """Initialize data source components."""
        # Try to load OnChainFeed
        try:
            from data_mgmt.feeds.onchain_feed import get_onchain_feed
            self._onchain_feed = get_onchain_feed()
            logger.info("[OK] OnChainFeed connected")
        except ImportError as e:
            logger.warning(f"OnChainFeed not available: {e}")
        except Exception as e:
            logger.warning(f"OnChainFeed init failed: {e}")
        
        # Try to load GlobalContextInformer
        try:
            from data_mgmt.global_context_informer import GlobalContextInformer
            self._global_context = GlobalContextInformer()
            logger.info("[OK] GlobalContextInformer connected")
        except ImportError as e:
            logger.warning(f"GlobalContextInformer not available: {e}")
        except Exception as e:
            logger.warning(f"GlobalContextInformer init failed: {e}")
        
        # Try to load SOLDefenseModule
        try:
            from defense.sol_defense import SOLDefenseModule
            self._sol_defense = SOLDefenseModule()
            logger.info("[OK] SOLDefenseModule connected")
        except ImportError as e:
            logger.warning(f"SOLDefenseModule not available: {e}")
        except Exception as e:
            logger.warning(f"SOLDefenseModule init failed: {e}")
    
    def update(self) -> FlowState:
        """
        Update all flow signals from data sources.
        
        Returns:
            Updated FlowState
        """
        now = datetime.now()
        state = FlowState(timestamp=now)
        
        # 1. Get whale & exchange flow from OnChainFeed
        if self._onchain_feed:
            try:
                tick = self._onchain_feed.get_latest()
                if tick:
                    state.whale_net_flow_usd = tick.net_whale_flow_usd
                    state.exchange_net_flow_usd = tick.net_exchange_flow_usd
                    
                    # Convert to direction signal
                    # Whale: positive flow = buying = bullish
                    if abs(state.whale_net_flow_usd) > self.config.whale_threshold_usd:
                        state.whale_direction = 1.0 if state.whale_net_flow_usd > 0 else -1.0
                        state.whale_confidence = min(abs(state.whale_net_flow_usd) / (self.config.whale_threshold_usd * 5), 1.0)
                        state.whale_alert = True
                    
                    # Exchange: positive flow (inflow) = selling = bearish
                    if abs(state.exchange_net_flow_usd) > self.config.exchange_flow_threshold:
                        state.exchange_direction = -1.0 if state.exchange_net_flow_usd > 0 else 1.0
                        state.exchange_confidence = min(abs(state.exchange_net_flow_usd) / (self.config.exchange_flow_threshold * 5), 1.0)
            except Exception as e:
                logger.debug(f"OnChainFeed update failed: {e}")
        elif self.config.mock_mode:
            # Mock whale/exchange data
            state.whale_direction = 0.0
            state.whale_confidence = 0.3
            state.exchange_direction = 0.0
            state.exchange_confidence = 0.3
        
        # 2. Get ETF flow from GlobalContextInformer
        if self._global_context:
            try:
                ctx_state = self._global_context.get_state()
                
                # BTC ETF flow
                if ctx_state.btc_total_daily_flow != 0:
                    state.btc_etf_direction = 1.0 if ctx_state.btc_total_daily_flow > 0 else -1.0
                
                # ETH ETF flow
                if ctx_state.eth_total_daily_flow != 0:
                    state.eth_etf_direction = 1.0 if ctx_state.eth_total_daily_flow > 0 else -1.0
                
                # Streak
                state.etf_streak_days = max(abs(ctx_state.btc_flow_streak), abs(ctx_state.eth_flow_streak))
                
                # Outflow caution
                if ctx_state.btc_flow_streak <= -self.config.etf_outflow_streak_alert:
                    state.etf_outflow_caution = True
                
                # Confidence based on streak
                state.etf_confidence = min(state.etf_streak_days / 5.0, 1.0)
            except Exception as e:
                logger.debug(f"GlobalContextInformer update failed: {e}")
        elif self.config.mock_mode:
            # Mock ETF data
            state.btc_etf_direction = 0.0
            state.eth_etf_direction = 0.0
            state.etf_confidence = 0.3
        
        # 3. Get network health from SOLDefenseModule
        if self._sol_defense:
            try:
                status = self._sol_defense.get_status()
                
                # Network health
                risk_level = status.get("risk_level", "NORMAL")
                health_map = {
                    "LOW": 1.0,
                    "NORMAL": 0.8,
                    "ELEVATED": 0.6,
                    "HIGH": 0.3,
                    "CRITICAL": 0.1,
                }
                state.sol_network_health = health_map.get(risk_level, 0.5)
                
                # Jito pressure
                jito = status.get("jito_metrics", {})
                tip_90th = jito.get("tip_90th_percentile", 0)
                # Normalize: 100k lamports = moderate pressure
                state.jito_pressure = min(tip_90th / 500_000, 1.0)
                
                # Degraded flag
                if state.sol_network_health < 0.5 or state.jito_pressure > 0.7:
                    state.network_degraded = True
            except Exception as e:
                logger.debug(f"SOLDefenseModule update failed: {e}")
        elif self.config.mock_mode:
            # Mock network data
            state.sol_network_health = 0.9
            state.jito_pressure = 0.1
        
        # 4. Calculate combined signal
        signals_to_combine = []
        weights_to_use = []
        
        if state.whale_confidence > 0:
            signals_to_combine.append(state.whale_direction * state.whale_confidence)
            weights_to_use.append(self.config.whale_weight)
        
        if state.exchange_confidence > 0:
            signals_to_combine.append(state.exchange_direction * state.exchange_confidence)
            weights_to_use.append(self.config.exchange_weight)
        
        if state.etf_confidence > 0:
            # Average BTC and ETH ETF signals
            avg_etf = (state.btc_etf_direction + state.eth_etf_direction) / 2
            signals_to_combine.append(avg_etf * state.etf_confidence)
            weights_to_use.append(self.config.etf_weight)
        
        if signals_to_combine:
            total_weight = sum(weights_to_use)
            if total_weight > 0:
                state.combined_direction = sum(
                    s * w for s, w in zip(signals_to_combine, weights_to_use)
                ) / total_weight
                state.combined_confidence = sum(weights_to_use) / (
                    self.config.whale_weight +
                    self.config.exchange_weight +
                    self.config.etf_weight +
                    self.config.network_weight  # [FIX-DA1] Was missing — confidence inflated 25% when all sources active
                )
        
        # Update internal state
        self._current_state = state
        self._last_update = now
        
        return state
    
    def get_signal_for_symbol(self, symbol: str) -> Tuple[float, float]:
        """
        Get flow-based trading signal for a symbol.
        
        Args:
            symbol: Trading symbol (BTC, ETH, SOL)
            
        Returns:
            Tuple of (direction, confidence)
        """
        state = self._current_state
        
        if symbol == "SOL":
            # SOL uses whale + exchange + network health
            direction = (
                state.whale_direction * 0.3 +
                state.exchange_direction * 0.3 +
                (state.sol_network_health - 0.5) * 0.4  # Health > 0.5 = bullish
            )
            confidence = max(state.whale_confidence, state.exchange_confidence)
            
            # Reduce confidence if network degraded
            if state.network_degraded:
                confidence *= 0.5
                
        elif symbol == "BTC":
            # BTC uses ETF flow primarily
            direction = state.btc_etf_direction * 0.7 + state.whale_direction * 0.3
            confidence = max(state.etf_confidence, state.whale_confidence * 0.5)
            
            # Increase confidence if sustained streak
            if abs(state.etf_streak_days) >= 3:
                confidence = min(confidence * 1.3, 1.0)
                
        elif symbol == "ETH":
            # ETH uses ETF flow
            direction = state.eth_etf_direction * 0.7 + state.btc_etf_direction * 0.3
            confidence = max(state.etf_confidence, state.whale_confidence * 0.5)
            
        else:
            # Unknown symbol - use combined
            direction = state.combined_direction
            confidence = state.combined_confidence
        
        return direction, confidence
    
    def get_risk_adjustment(self, symbol: str) -> float:
        """
        Get risk adjustment factor based on flow signals.
        
        Returns:
            Multiplier 0.5 to 1.0 (lower = reduce size)
        """
        state = self._current_state
        adjustment = 1.0
        
        # ETF outflow caution
        if state.etf_outflow_caution:
            adjustment *= 0.8
        
        # Network degradation (SOL only)
        if symbol == "SOL" and state.network_degraded:
            adjustment *= 0.7
        
        # High Jito pressure (SOL only)
        if symbol == "SOL" and state.jito_pressure > 0.5:
            adjustment *= (1.0 - state.jito_pressure * 0.3)
        
        return max(adjustment, 0.5)
    
    def get_state(self) -> FlowState:
        """Get current flow state."""
        return self._current_state
    
    def get_status(self) -> Dict[str, Any]:
        """Get integrator status."""
        return {
            "last_update": self._last_update.isoformat(),
            "sources": {
                "onchain_feed": self._onchain_feed is not None,
                "global_context": self._global_context is not None,
                "sol_defense": self._sol_defense is not None,
            },
            "mock_mode": self.config.mock_mode,
            "state": self._current_state.to_dict(),
        }


# =============================================================================
# JITO TIP MONITOR - SOL-Specific Alpha Signal (S14)
# =============================================================================
# Tracks Jito tip volume z-score and price-tip divergence for SOL shorts.
# Uses jito_pressure from FlowState as proxy (zero external dependency).

@dataclass
class JitoTipConfig:
    """Configuration for Jito tip alpha monitor."""
    enabled: bool = False         # shadow mode default
    history_length: int = 42      # 7D  6 (4H intervals)
    zscore_threshold: float = 2.0
    divergence_window: int = 6    # last 6 periods for divergence
    max_alpha_adjustment: float = 0.10


@dataclass
class JitoTipSignal:
    """Output of the Jito tip monitor."""
    tip_zscore: float = 0.0
    tip_trend: str = "STABLE"       # RISING / STABLE / FALLING
    price_tip_divergence: float = 0.0  # negative = bearish divergence
    short_alpha_adjustment: float = 0.0
    is_active: bool = False


class JitoTipMonitor:
    """
    SOL-specific alpha signal from MEV activity (Jito tips).

    Signals:
    - tip_zscore > 2  -> MEV overheating -> mean-reversion short
    - tip declining + price rising -> bearish divergence
    - tip_zscore < -1 + price falling -> de-leverage, don't chase short
    """

    def __init__(self, config: Optional[JitoTipConfig] = None):
        self.config = config or JitoTipConfig()
        self._history: list = []  # list of {ts, tip, price}

    def update(self, tip_volume_proxy: float, sol_price: float, timestamp: float = 0.0) -> None:
        """Append a tip volume observation."""
        self._history.append({
            "ts": timestamp or time.time(),
            "tip": tip_volume_proxy,
            "price": sol_price,
        })
        if len(self._history) > self.config.history_length:
            self._history = self._history[-self.config.history_length:]

    def get_signal(self) -> JitoTipSignal:
        """Compute z-score and divergence from history."""
        sig = JitoTipSignal()

        min_samples = max(self.config.history_length // 2, 6)
        if len(self._history) < min_samples:
            return sig  # not enough data

        sig.is_active = True
        tips = [h["tip"] for h in self._history]
        prices = [h["price"] for h in self._history]

        # Z-score of latest tip value
        mean_tip = statistics.mean(tips)
        std_tip = statistics.pstdev(tips)
        if std_tip > 1e-12:
            sig.tip_zscore = (tips[-1] - mean_tip) / std_tip
        else:
            sig.tip_zscore = 0.0

        # Trend (simple linear slope of last N)
        window = min(self.config.divergence_window, len(tips))
        recent_tips = tips[-window:]
        if window >= 3:
            mid = window // 2
            first_half = statistics.mean(recent_tips[:mid])
            second_half = statistics.mean(recent_tips[mid:])
            diff = second_half - first_half
            if diff > std_tip * 0.3:
                sig.tip_trend = "RISING"
            elif diff < -std_tip * 0.3:
                sig.tip_trend = "FALLING"
            else:
                sig.tip_trend = "STABLE"

        # Price-tip divergence: normalized correlation of recent changes
        if window >= 3:
            recent_prices = prices[-window:]
            tip_delta = recent_tips[-1] - recent_tips[0]
            price_delta = recent_prices[-1] - recent_prices[0]
            tip_norm = abs(tip_delta) + 1e-12
            price_norm = abs(price_delta) + 1e-12
            # -1 = bearish divergence (tip down, price up)
            if tip_norm > 0 and price_norm > 0:
                tip_sign = 1.0 if tip_delta > 0 else -1.0
                price_sign = 1.0 if price_delta > 0 else -1.0
                sig.price_tip_divergence = tip_sign * price_sign  # +1 aligned, -1 diverged

        # Alpha adjustment
        adj = 0.0
        if sig.tip_zscore > 3.0:
            adj += 0.10
        elif sig.tip_zscore > self.config.zscore_threshold:
            adj += 0.05

        if sig.price_tip_divergence < -0.5:
            adj += 0.05

        if sig.tip_zscore < -1.0 and prices[-1] < prices[-window] if window >= 2 else False:
            adj -= 0.05  # de-leverage, don't chase short

        sig.short_alpha_adjustment = max(
            -self.config.max_alpha_adjustment,
            min(self.config.max_alpha_adjustment, adj),
        )

        if abs(sig.short_alpha_adjustment) > 0.01:
            logger.info(
                f"[JITO] SOL tip_zscore={sig.tip_zscore:.2f}, "
                f"trend={sig.tip_trend}, div={sig.price_tip_divergence:+.2f}, "
                f"alpha_adj={sig.short_alpha_adjustment:+.2f}"
            )

        return sig


# =============================================================================
# SINGLETON
# =============================================================================

_flow_integrator: Optional[FlowSignalIntegrator] = None


def get_flow_integrator(config: FlowIntegratorConfig = None) -> FlowSignalIntegrator:
    """Get or create flow signal integrator singleton."""
    global _flow_integrator
    if _flow_integrator is None:
        _flow_integrator = FlowSignalIntegrator(config)
    elif config is not None and _flow_integrator.config != config:
        logger.info("[FLOW_INTEGRATOR] Config changed; refreshing singleton instance")
        _flow_integrator = FlowSignalIntegrator(config)
    return _flow_integrator


def reset_flow_integrator():
    """Reset singleton."""
    global _flow_integrator
    _flow_integrator = None


__all__ = [
    "FlowSignalType",
    "FlowSignal",
    "FlowState",
    "FlowIntegratorConfig",
    "FlowSignalIntegrator",
    "get_flow_integrator",
    "reset_flow_integrator",
    # S14: Jito Tip Monitor
    "JitoTipConfig",
    "JitoTipSignal",
    "JitoTipMonitor",
]

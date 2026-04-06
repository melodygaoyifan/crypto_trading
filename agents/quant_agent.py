"""
_DEPRECATED (T26): Not used by main.py or integration_v36.py.
Quant signals now come from main.py _build_agent_signals() + kraken_quant_agent.py.
Kept for archive/ and tests/ imports.

Original docstring:
Institutional Quant Agent - Aggressive Bearish Strategies
==========================================================
NO RETAIL INDICATORS. Microstructure + Derivatives + Stat-Arb ONLY.

Strategies:
1. Liquidation Cascade Hunter (Microstructure)
2. Dynamic Cointegration Stat-Arb (Kalman Filter)
3. Funding Rate / Price Divergence (Derivatives Arbitrage)

Risk Profile: AGGRESSIVE / High Sharpe Target
Market Bias: SHORT-TERM BEARISH
Primary Target: SOL (High Beta Short)
Secondary: ETH, BTC (Relative Value)

Author: Quant Research Team
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import warnings
import logging
import time

# Suppress numpy warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    logger.warning("filterpy not installed: pip install filterpy")

try:
    from scipy import stats
    from scipy.special import erfinv
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# pandas_ta for technical indicators
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    ta = None
    PANDAS_TA_AVAILABLE = False


# =============================================================================
# MATHEMATICAL CONSTANTS & TYPES
# =============================================================================

class SignalType(Enum):
    """Signal types for the quant agent."""
    NO_SIGNAL = 0
    AGGRESSIVE_SHORT = 1
    SHORT = 2
    REDUCE_SHORT = 3
    FLATTEN = 4
    LONG_HEDGE = 5  # For stat-arb leg


class Signal(NamedTuple):
    """Trading signal with metadata."""
    timestamp: float
    asset: str
    signal_type: SignalType
    size_multiplier: float  # 0.0 to 2.0
    confidence: float  # 0.0 to 1.0
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit: float
    metadata: Dict


@dataclass
class QuantAgentIntent:
    """HMATS-facing output from QuantAgent.

    Maps to the AgentSignal contract used by authority_fusion and
    integration_v36.  Execution fields (stop_loss / take_profit) live in
    ``meta`` only - the agent emits *intent*, not orders.
    """

    asset: str
    direction: float  # -1.0 (short) .. 0.0 (neutral) .. +1.0 (long)
    confidence: float  # 0.0 .. 1.0
    strategy: str  # which sub-strategy fired
    signal_type: str  # SignalType.name
    size_multiplier: float = 1.0  # 0.0 .. 2.0
    data_age_seconds: float = 0.0
    meta: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_agent_signals_dict(self) -> Dict:
        """Convert to the flat dict consumed by integration_v36 / fusion."""
        return {
            "quant_direction": self.direction,
            "quant_confidence": self.confidence,
            "quant_strategy": self.strategy,
            "quant_signal_type": self.signal_type,
            "quant_size_multiplier": self.size_multiplier,
            "quant_asset": self.asset,
            "quant_data_age_s": self.data_age_seconds,
            "quant_meta": self.meta,
        }


# Neutral singleton - returned when no signal or data is stale
_NEUTRAL_INTENT_CACHE: Dict[str, "QuantAgentIntent"] = {}


def _neutral_intent(asset: str) -> "QuantAgentIntent":
    if asset not in _NEUTRAL_INTENT_CACHE:
        _NEUTRAL_INTENT_CACHE[asset] = QuantAgentIntent(
            asset=asset,
            direction=0.0,
            confidence=0.0,
            strategy="none",
            signal_type=SignalType.NO_SIGNAL.name,
            size_multiplier=0.0,
        )
    return _NEUTRAL_INTENT_CACHE[asset]


# =============================================================================
# TECHNICAL ANALYZER (Legacy Compat - from integration/core/quant_agent)
# =============================================================================

class TechnicalAnalyzer:
    """
    Calculate technical indicators using pandas_ta.
    
    Migrated from legacy/v36_archive/core/quant_agent.py for canonical import path.
    """
    
    def __init__(self, 
                 ema_period: int = 200,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_period: int = 14,
                 atr_period: int = 14):
        self.ema_period = ema_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.logger = logging.getLogger(__name__)
    
    def calculate_indicators(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """Calculate all technical indicators."""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas_ta required: pip install pandas_ta")
        
        df = df.copy()
        
        # EMA 200 - Trend Filter
        df['ema_200'] = ta.ema(df['close'], length=self.ema_period)
        
        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=self.bb_period, std=self.bb_std)
        if bbands is not None:
            df['bb_upper'] = bbands[f'BBU_{self.bb_period}_{self.bb_std}']
            df['bb_middle'] = bbands[f'BBM_{self.bb_period}_{self.bb_std}']
            df['bb_lower'] = bbands[f'BBL_{self.bb_period}_{self.bb_std}']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=self.rsi_period)
        
        # ATR for dynamic stop loss
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        
        # Additional indicators for enhanced signals
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        
        # MACD for momentum confirmation
        macd = ta.macd(df['close'])
        if macd is not None:
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
        
        # ADX for trend strength
        adx = ta.adx(df['high'], df['low'], df['close'])
        if adx is not None:
            df['adx'] = adx['ADX_14']
            df['di_plus'] = adx['DMP_14']
            df['di_minus'] = adx['DMN_14']
        
        return df
    
    def get_latest_indicators(self, df: 'pd.DataFrame') -> Dict[str, float]:
        """Get the latest indicator values."""
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        
        return {
            'close': latest['close'],
            'ema_200': latest.get('ema_200', np.nan),
            'bb_upper': latest.get('bb_upper', np.nan),
            'bb_middle': latest.get('bb_middle', np.nan),
            'bb_lower': latest.get('bb_lower', np.nan),
            'bb_width': latest.get('bb_width', np.nan),
            'rsi': latest.get('rsi', np.nan),
            'atr': latest.get('atr', np.nan),
            'sma_50': latest.get('sma_50', np.nan),
            'volume': latest.get('volume', np.nan),
            'volume_sma': latest.get('volume_sma', np.nan),
            'macd': latest.get('macd', np.nan),
            'macd_signal': latest.get('macd_signal', np.nan),
            'macd_hist': latest.get('macd_hist', np.nan),
            'adx': latest.get('adx', np.nan),
            'di_plus': latest.get('di_plus', np.nan),
            'di_minus': latest.get('di_minus', np.nan),
        }


@dataclass
class MarketMicrostructure:
    """Real-time microstructure state."""
    timestamp: float
    
    # Order flow
    bid_depth: np.ndarray  # [price, size] pairs
    ask_depth: np.ndarray
    bid_imbalance: float  # (bid_vol - ask_vol) / total
    
    # Volume metrics
    taker_buy_volume: float
    taker_sell_volume: float
    cvd: float  # Cumulative Volume Delta
    cvd_velocity: float  # d(CVD)/dt
    
    # VWAP metrics
    vwap: float
    vwap_deviation: float  # (price - vwap) / vwap
    
    # Trade flow
    trade_count: int
    avg_trade_size: float
    large_trade_ratio: float  # % of volume from large trades


@dataclass
class DerivativesState:
    """Derivatives market state."""
    timestamp: float
    asset: str
    
    # Open Interest
    open_interest: float
    oi_change_1h: float
    oi_change_24h: float
    oi_velocity: float  # d(OI)/dt
    
    # Funding
    funding_rate: float
    predicted_funding: float
    funding_velocity: float  # d(funding)/dt
    
    # Liquidations
    long_liquidations: float
    short_liquidations: float
    liquidation_imbalance: float  # (long - short) / total
    liquidation_volume_zscore: float
    
    # Basis
    spot_price: float
    perp_price: float
    basis: float  # (perp - spot) / spot
    basis_zscore: float


# =============================================================================
# STRATEGY 1: LIQUIDATION CASCADE HUNTER
# =============================================================================

class LiquidationCascadeHunter:
    """
    Microstructure strategy to detect and exploit liquidation cascades.
    
    MATHEMATICAL FRAMEWORK:
    =======================
    
    Let L(t) = Liquidation volume at time t
    Let σ_L = Rolling StdDev of L over window W
    Let μ_L = Rolling Mean of L over window W
    
    Liquidation Z-Score:
        z_L(t) = (L(t) - μ_L) / σ_L
    
    Cascade Detection Conditions:
        1. Price Velocity: dP/dt < -θ_price (negative)
        2. OI Velocity: dOI/dt < -θ_oi (longs closing)
        3. CVD: CVD(t) < -θ_cvd (aggressive selling)
        4. Liquidation Spike: z_L(t) > N (typically N=2.5)
    
    Entry Signal Strength:
        S(t) = w1 * norm(z_L) + w2 * norm(|dP/dt|) + w3 * norm(|CVD|)
        
        where norm(x) = (x - min) / (max - min)
    
    VWAP-Normalized Liquidation Intensity:
        LI(t) = L(t) / VWAP_Volume(t)
        
    This captures liquidation intensity relative to normal volume.
    """
    
    def __init__(
        self,
        lookback_window: int = 100,
        zscore_threshold: float = 2.5,
        price_velocity_threshold: float = -0.005,  # -0.5% per candle
        oi_velocity_threshold: float = -0.02,  # -2% OI drop
        cvd_threshold: float = -1.5,  # CVD z-score
        cascade_confirmation_candles: int = 3,
    ):
        self.lookback = lookback_window
        self.zscore_thresh = zscore_threshold
        self.price_vel_thresh = price_velocity_threshold
        self.oi_vel_thresh = oi_velocity_threshold
        self.cvd_thresh = cvd_threshold
        self.confirm_candles = cascade_confirmation_candles
        
        # Rolling buffers
        self.liquidation_buffer = deque(maxlen=lookback_window)
        self.price_buffer = deque(maxlen=lookback_window)
        self.oi_buffer = deque(maxlen=lookback_window)
        self.cvd_buffer = deque(maxlen=lookback_window)
        self.vwap_vol_buffer = deque(maxlen=lookback_window)
        
        # State
        self.cascade_active = False
        self.cascade_start_price = 0.0
        self.consecutive_cascade_signals = 0
        
    def update(
        self,
        timestamp: float,
        price: float,
        open_interest: float,
        long_liquidations: float,
        short_liquidations: float,
        cvd: float,
        vwap_volume: float,
        asset: str = "SOL",
    ) -> Optional[Signal]:
        """
        Update state and check for liquidation cascade entry.

        Returns Signal if cascade detected, None otherwise.
        """
        # Update buffers
        total_liquidations = long_liquidations + short_liquidations
        self.liquidation_buffer.append(total_liquidations)
        self.price_buffer.append(price)
        self.oi_buffer.append(open_interest)
        self.cvd_buffer.append(cvd)
        self.vwap_vol_buffer.append(vwap_volume)
        
        # Need enough data
        if len(self.liquidation_buffer) < self.lookback // 2:
            return None
        
        # Calculate metrics
        liq_array = np.array(self.liquidation_buffer)
        price_array = np.array(self.price_buffer)
        oi_array = np.array(self.oi_buffer)
        cvd_array = np.array(self.cvd_buffer)
        
        # 1. Liquidation Z-Score
        liq_mean = np.mean(liq_array)
        liq_std = np.std(liq_array) + 1e-10
        liq_zscore = (total_liquidations - liq_mean) / liq_std
        
        # 2. Price Velocity (returns)
        price_velocity = (price - price_array[-2]) / price_array[-2] if len(price_array) > 1 else 0
        
        # 3. OI Velocity
        oi_velocity = (open_interest - oi_array[-2]) / oi_array[-2] if len(oi_array) > 1 else 0
        
        # 4. CVD Z-Score
        cvd_mean = np.mean(cvd_array)
        cvd_std = np.std(cvd_array) + 1e-10
        cvd_zscore = (cvd - cvd_mean) / cvd_std
        
        # 5. VWAP-Normalized Liquidation Intensity
        vwap_vol = np.mean(self.vwap_vol_buffer) + 1e-10
        liquidation_intensity = total_liquidations / vwap_vol
        
        # 6. Long/Short Liquidation Imbalance
        liq_imbalance = (long_liquidations - short_liquidations) / (total_liquidations + 1e-10)
        
        # CASCADE DETECTION
        cascade_conditions = {
            'price_dropping': price_velocity < self.price_vel_thresh,
            'oi_dropping': oi_velocity < self.oi_vel_thresh,
            'cvd_negative': cvd_zscore < self.cvd_thresh,
            'liquidation_spike': liq_zscore > self.zscore_thresh,
            'long_liquidations_dominant': liq_imbalance > 0.3,  # More longs liquidated
        }
        
        conditions_met = sum(cascade_conditions.values())
        
        # Need at least 4/5 conditions
        if conditions_met >= 4:
            self.consecutive_cascade_signals += 1
        else:
            self.consecutive_cascade_signals = 0
        
        # Generate signal after confirmation
        if self.consecutive_cascade_signals >= self.confirm_candles and not self.cascade_active:
            self.cascade_active = True
            self.cascade_start_price = price
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(
                liq_zscore, price_velocity, cvd_zscore, liq_imbalance
            )
            
            # Dynamic stop-loss based on ATR-like volatility
            recent_vol = np.std(np.diff(price_array[-20:])) / np.mean(price_array[-20:])
            stop_distance = max(0.02, recent_vol * 3)  # At least 2%, or 3x recent vol
            
            return Signal(
                timestamp=timestamp,
                asset=asset,
                signal_type=SignalType.AGGRESSIVE_SHORT,
                size_multiplier=min(2.0, signal_strength),
                confidence=min(0.95, 0.6 + signal_strength * 0.2),
                strategy="liquidation_cascade",
                entry_price=price,
                stop_loss=price * (1 + stop_distance),
                take_profit=price * (1 - stop_distance * 2),  # 2:1 R:R minimum
                metadata={
                    'liquidation_zscore': liq_zscore,
                    'price_velocity': price_velocity,
                    'oi_velocity': oi_velocity,
                    'cvd_zscore': cvd_zscore,
                    'liquidation_intensity': liquidation_intensity,
                    'conditions_met': conditions_met,
                    'cascade_conditions': cascade_conditions,
                }
            )
        
        # Check for cascade exhaustion (exit signal)
        if self.cascade_active:
            exhaustion = self._check_cascade_exhaustion(
                price, liq_zscore, cvd_zscore, liq_imbalance
            )
            if exhaustion:
                self.cascade_active = False
                return Signal(
                    timestamp=timestamp,
                    asset=asset,
                    signal_type=SignalType.REDUCE_SHORT,
                    size_multiplier=0.5,
                    confidence=0.7,
                    strategy="liquidation_cascade",
                    entry_price=price,
                    stop_loss=price * 0.98,
                    take_profit=price * 1.01,
                    metadata={'reason': 'cascade_exhaustion'}
                )
        
        return None
    
    def _calculate_signal_strength(
        self,
        liq_zscore: float,
        price_velocity: float,
        cvd_zscore: float,
        liq_imbalance: float,
    ) -> float:
        """Calculate composite signal strength [0, 2]."""
        # Normalize components
        liq_component = min(1.0, (liq_zscore - self.zscore_thresh) / 2)
        vel_component = min(1.0, abs(price_velocity) / 0.02)
        cvd_component = min(1.0, abs(cvd_zscore) / 3)
        imbalance_component = min(1.0, liq_imbalance)
        
        # Weighted sum
        strength = (
            0.35 * liq_component +
            0.25 * vel_component +
            0.25 * cvd_component +
            0.15 * imbalance_component
        ) * 2  # Scale to [0, 2]
        
        return strength
    
    def _check_cascade_exhaustion(
        self,
        current_price: float,
        liq_zscore: float,
        cvd_zscore: float,
        liq_imbalance: float,
    ) -> bool:
        """Check if the liquidation cascade is exhausting."""
        # Exhaustion indicators
        price_drop = (self.cascade_start_price - current_price) / self.cascade_start_price
        
        conditions = [
            price_drop > 0.10,  # Already dropped 10%
            liq_zscore < 1.0,  # Liquidations normalizing
            cvd_zscore > -0.5,  # Selling pressure easing
            liq_imbalance < 0.1,  # Liquidations balanced
        ]
        
        return sum(conditions) >= 3


# =============================================================================
# STRATEGY 2: DYNAMIC COINTEGRATION STAT-ARB (KALMAN FILTER)
# =============================================================================

class KalmanCointegration:
    """
    Dynamic Pairs Trading using Kalman Filter for adaptive hedge ratio.
    
    MATHEMATICAL FRAMEWORK:
    =======================
    
    LINEAR STATE-SPACE MODEL:
    -------------------------
    We model the relationship between log prices:
        
        y_t = β_t * x_t + α_t + ε_t
        
    Where:
        y_t = log(P_SOL)
        x_t = log(P_ETH)  or log(P_BTC)
        β_t = Time-varying hedge ratio (DYNAMIC)
        α_t = Time-varying intercept
        ε_t ~ N(0, R)  Observation noise
    
    State Vector:
        θ_t = [β_t, α_t]'
    
    STATE TRANSITION (Random Walk):
        θ_t = θ_{t-1} + w_t
        w_t ~ N(0, Q)  Process noise
    
    KALMAN FILTER EQUATIONS:
    ------------------------
    
    PREDICT STEP:
        θ̂_t|t-1 = θ̂_{t-1|t-1}           (State prediction)
        P_t|t-1 = P_{t-1|t-1} + Q        (Covariance prediction)
    
    UPDATE STEP:
        Observation matrix: H_t = [x_t, 1]
        Innovation: v_t = y_t - H_t * θ̂_t|t-1
        Innovation covariance: S_t = H_t * P_t|t-1 * H_t' + R
        Kalman Gain: K_t = P_t|t-1 * H_t' * S_t^{-1}
        
        State update: θ̂_t|t = θ̂_t|t-1 + K_t * v_t
        Covariance update: P_t|t = (I - K_t * H_t) * P_t|t-1
    
    SPREAD CALCULATION:
        z_t = y_t - β̂_t * x_t - α̂_t  (Kalman residual)
        
    Z-SCORE:
        z_score_t = (z_t - μ_z) / σ_z
        
        where μ_z, σ_z are rolling mean/std of z_t
    
    TRADING RULES:
        SHORT SOL / LONG ETH when: z_score > +2σ (SOL overpriced)
        LONG SOL / SHORT ETH when: z_score < -2σ (SOL underpriced)
        EXIT when: |z_score| < 0.5σ (mean reversion complete)
    
    BEAR MARKET BIAS:
        In bearish conditions, we ONLY take the SHORT SOL leg when z > +2σ
        because high-beta assets (SOL) fall faster.
    """
    
    def __init__(
        self,
        delta: float = 1e-5,  # Process noise (smaller = slower adaptation)
        obs_noise: float = 1e-3,  # Observation noise
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        spread_lookback: int = 100,
        bear_bias: bool = True,  # Only short SOL in bearish conditions
    ):
        self.delta = delta
        self.obs_noise = obs_noise
        self.entry_z = entry_zscore
        self.exit_z = exit_zscore
        self.spread_lookback = spread_lookback
        self.bear_bias = bear_bias
        
        # Kalman Filter State
        self.kf_initialized = False
        self.state = np.array([1.0, 0.0])  # [beta, alpha]
        self.P = np.eye(2) * 1.0  # State covariance
        self.Q = np.eye(2) * delta  # Process noise
        self.R = obs_noise  # Observation noise (scalar)
        
        # Spread history for z-score
        self.spread_buffer = deque(maxlen=spread_lookback)
        self.price_history = {
            'SOL': deque(maxlen=spread_lookback),
            'ETH': deque(maxlen=spread_lookback),
            'BTC': deque(maxlen=spread_lookback),
        }
        
        # Position tracking
        self.in_position = False
        self.position_side = None  # 'short_sol' or 'long_sol'
        self.entry_zscore = 0.0
        
    def update(
        self,
        timestamp: float,
        sol_price: float,
        eth_price: float,
        btc_price: float = None,
        hedge_asset: str = 'ETH',
        target_asset: str = 'SOL',
    ) -> Optional[Signal]:
        """
        Update Kalman Filter and generate trading signals.
        """
        # Log prices
        y = np.log(sol_price)
        x = np.log(eth_price) if hedge_asset == 'ETH' else np.log(btc_price)
        
        # Store prices
        self.price_history['SOL'].append(sol_price)
        self.price_history['ETH'].append(eth_price)
        if btc_price:
            self.price_history['BTC'].append(btc_price)
        
        # Kalman Filter Update
        beta, alpha, spread = self._kalman_update(y, x)
        
        # Store spread
        self.spread_buffer.append(spread)
        
        # Need enough data for z-score
        if len(self.spread_buffer) < self.spread_lookback // 2:
            return None
        
        # Calculate z-score
        spread_array = np.array(self.spread_buffer)
        spread_mean = np.mean(spread_array)
        spread_std = np.std(spread_array) + 1e-10
        zscore = (spread - spread_mean) / spread_std
        
        # Half-life calculation for mean reversion speed
        half_life = self._calculate_half_life(spread_array)
        
        # Generate signals
        return self._generate_signal(
            timestamp=timestamp,
            sol_price=sol_price,
            eth_price=eth_price,
            beta=beta,
            alpha=alpha,
            zscore=zscore,
            half_life=half_life,
            hedge_asset=hedge_asset,
            target_asset=target_asset,
        )
    
    def _kalman_update(self, y: float, x: float) -> Tuple[float, float, float]:
        """
        Single Kalman Filter update step.
        
        Returns: (beta, alpha, spread)
        """
        # PREDICT
        # State prediction (random walk: state unchanged)
        state_pred = self.state
        P_pred = self.P + self.Q
        
        # UPDATE
        # Observation matrix H = [x, 1]
        H = np.array([x, 1.0])
        
        # Predicted observation
        y_pred = H @ state_pred
        
        # Innovation (residual)
        innovation = y - y_pred
        
        # Innovation covariance S = H * P * H' + R
        S = H @ P_pred @ H.T + self.R
        
        # Kalman Gain K = P * H' * S^{-1}
        K = P_pred @ H.T / S
        
        # State update
        self.state = state_pred + K * innovation
        
        # Covariance update P = (I - K*H) * P
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred
        
        # Enforce positive definiteness
        self.P = (self.P + self.P.T) / 2
        
        # Extract beta and alpha
        beta = self.state[0]
        alpha = self.state[1]
        
        # Spread (Kalman residual)
        spread = y - beta * x - alpha
        
        return beta, alpha, spread
    
    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """
        Calculate half-life of mean reversion using OLS.
        
        Model: Δz_t = λ * z_{t-1} + ε
        Half-life = -ln(2) / λ
        """
        if len(spread) < 20:
            return np.inf
        
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)
        
        # OLS regression
        X = spread_lag.reshape(-1, 1)
        y = spread_diff
        
        # lambda = (X'X)^{-1} X'y
        try:
            lambda_coef = np.linalg.lstsq(X, y, rcond=None)[0][0]
            
            if lambda_coef >= 0:
                return np.inf  # No mean reversion
            
            half_life = -np.log(2) / lambda_coef
            return max(1, min(half_life, 1000))  # Bound between 1 and 1000 periods
        except Exception:
            return np.inf
    
    def _generate_signal(
        self,
        timestamp: float,
        sol_price: float,
        eth_price: float,
        beta: float,
        alpha: float,
        zscore: float,
        half_life: float,
        hedge_asset: str,
        target_asset: str = "SOL",
    ) -> Optional[Signal]:
        """Generate trading signal based on z-score."""
        
        # Entry conditions
        if not self.in_position:
            # SHORT SOL (Long ETH hedge): z-score significantly positive
            # SOL is overpriced relative to ETH
            if zscore > self.entry_z:
                self.in_position = True
                self.position_side = 'short_sol'
                self.entry_zscore = zscore
                
                # Position sizing based on z-score magnitude
                size_mult = min(2.0, 1.0 + (zscore - self.entry_z) * 0.5)
                
                # Stop loss at 3x entry z-score (spread widens further)
                # Take profit when spread reverts to mean
                vol_estimate = np.std(list(self.spread_buffer))
                
                return Signal(
                    timestamp=timestamp,
                    asset=target_asset,
                    signal_type=SignalType.AGGRESSIVE_SHORT,
                    size_multiplier=size_mult,
                    confidence=min(0.9, 0.5 + zscore / 10),
                    strategy="kalman_cointegration",
                    entry_price=sol_price,
                    stop_loss=sol_price * (1 + 0.05),  # 5% stop
                    take_profit=sol_price * (1 - 0.03 * zscore),  # Dynamic TP
                    metadata={
                        'beta': beta,
                        'alpha': alpha,
                        'zscore': zscore,
                        'half_life': half_life,
                        'hedge_asset': hedge_asset,
                        'hedge_price': eth_price,
                        'hedge_ratio': beta,
                        'position_type': f'short_{target_asset.lower()}_long_{hedge_asset.lower()}',
                    }
                )
            
            # LONG SOL (Short ETH hedge): Only if not bearish biased
            elif zscore < -self.entry_z and not self.bear_bias:
                # Skip in bearish conditions
                pass
        
        # Exit conditions
        elif self.in_position:
            exit_signal = False
            
            if self.position_side == 'short_sol':
                # Exit when z-score reverts toward zero
                if zscore < self.exit_z:
                    exit_signal = True
                # Stop loss if z-score expands significantly
                elif zscore > self.entry_zscore * 1.5:
                    exit_signal = True
            
            if exit_signal:
                self.in_position = False
                self.position_side = None
                
                return Signal(
                    timestamp=timestamp,
                    asset=target_asset,
                    signal_type=SignalType.FLATTEN,
                    size_multiplier=1.0,
                    confidence=0.8,
                    strategy="kalman_cointegration",
                    entry_price=sol_price,
                    stop_loss=sol_price,
                    take_profit=sol_price,
                    metadata={
                        'exit_zscore': zscore,
                        'entry_zscore': self.entry_zscore,
                        'pnl_zscore': self.entry_zscore - zscore,
                    }
                )
        
        return None
    
    def get_hedge_ratio(self) -> float:
        """Return current Kalman-estimated hedge ratio."""
        return self.state[0]
    
    def get_state_covariance(self) -> np.ndarray:
        """Return state estimation uncertainty."""
        return self.P


# =============================================================================
# STRATEGY 3: FUNDING RATE / PRICE DIVERGENCE
# =============================================================================

class FundingDivergence:
    """
    Exploit dislocation between spot and perpetual futures markets.
    
    MATHEMATICAL FRAMEWORK:
    =======================
    
    FUNDING RATE MECHANICS:
    -----------------------
    Funding Rate = Premium Index + Clamp(Interest Rate - Premium Index, -0.05%, 0.05%)
    
    Premium Index = (Max(0, Impact Bid Price - Mark Price) - Max(0, Mark Price - Impact Ask Price)) / Spot Price
    
    When Funding > 0: Longs pay Shorts (market bullish, perps overpriced)
    When Funding < 0: Shorts pay Longs (market bearish, perps underpriced)
    
    BEARISH DIVERGENCE DETECTION:
    -----------------------------
    Signal when:
        1. Spot price is FLAT or RISING
        2. Predicted funding rate is FALLING (toward negative)
        3. Open Interest is RISING (smart money building shorts)
    
    Mathematical Formulation:
        
        Price Momentum (normalized):
            M_p = (P_t - P_{t-n}) / (σ_p * √n)
        
        Funding Momentum:
            M_f = (F_t - F_{t-n}) / σ_f
        
        OI Momentum:
            M_oi = (OI_t - OI_{t-n}) / OI_{t-n}
        
        Divergence Score:
            D = w1 * I(M_p > 0) * |M_f| + w2 * I(M_f < 0) * M_oi
            
            where I() is the indicator function
        
        Bearish Signal when:
            D > threshold AND M_f < -1.5 AND M_oi > 0.05
    
    INTERPRETATION:
    ---------------
    - Rising OI + Falling Funding = Smart money building SHORT positions
    - They absorb spot buying, knowing price will dump
    - Spot price eventually collapses to meet perp price
    """
    
    def __init__(
        self,
        funding_lookback: int = 48,  # 48 funding periods (8h funding = 16 days)
        price_lookback: int = 24,  # 24 hourly candles
        oi_lookback: int = 24,
        divergence_threshold: float = 1.5,
        funding_momentum_threshold: float = -1.5,
        oi_growth_threshold: float = 0.05,  # 5% OI growth
    ):
        self.funding_lookback = funding_lookback
        self.price_lookback = price_lookback
        self.oi_lookback = oi_lookback
        self.div_threshold = divergence_threshold
        self.funding_mom_thresh = funding_momentum_threshold
        self.oi_growth_thresh = oi_growth_threshold
        
        # Buffers
        self.funding_buffer = deque(maxlen=funding_lookback)
        self.price_buffer = deque(maxlen=price_lookback)
        self.oi_buffer = deque(maxlen=oi_lookback)
        self.basis_buffer = deque(maxlen=price_lookback)
        
        # State
        self.divergence_active = False
        self.divergence_start_price = 0.0
        
    def update(
        self,
        timestamp: float,
        spot_price: float,
        perp_price: float,
        funding_rate: float,
        predicted_funding: float,
        open_interest: float,
        asset: str = "SOL",
    ) -> Optional[Signal]:
        """
        Update state and check for funding divergence signals.
        """
        # Calculate basis
        basis = (perp_price - spot_price) / spot_price
        
        # Update buffers
        self.funding_buffer.append(funding_rate)
        self.price_buffer.append(spot_price)
        self.oi_buffer.append(open_interest)
        self.basis_buffer.append(basis)
        
        # Need enough data
        if len(self.funding_buffer) < self.funding_lookback // 2:
            return None
        
        # Calculate metrics
        metrics = self._calculate_divergence_metrics(
            spot_price, funding_rate, predicted_funding, open_interest, basis
        )
        
        # Check for bearish divergence
        if self._is_bearish_divergence(metrics):
            if not self.divergence_active:
                self.divergence_active = True
                self.divergence_start_price = spot_price

                return self._generate_short_signal(
                    timestamp, spot_price, metrics, asset=asset
                )
        
        # Check for divergence resolution
        if self.divergence_active:
            if self._is_divergence_resolved(spot_price, metrics):
                self.divergence_active = False
                
                return Signal(
                    timestamp=timestamp,
                    asset=asset,
                    signal_type=SignalType.REDUCE_SHORT,
                    size_multiplier=0.5,
                    confidence=0.7,
                    strategy="funding_divergence",
                    entry_price=spot_price,
                    stop_loss=spot_price * 0.98,
                    take_profit=spot_price * 1.01,
                    metadata={'reason': 'divergence_resolved', **metrics}
                )
        
        return None
    
    def _calculate_divergence_metrics(
        self,
        spot_price: float,
        funding_rate: float,
        predicted_funding: float,
        open_interest: float,
        basis: float,
    ) -> Dict:
        """Calculate all divergence metrics."""
        
        # Price momentum (normalized)
        price_array = np.array(self.price_buffer)
        price_returns = np.diff(np.log(price_array))
        price_vol = np.std(price_returns) + 1e-10
        price_momentum = (spot_price - price_array[0]) / (price_vol * np.sqrt(len(price_array)))
        
        # Funding momentum
        funding_array = np.array(self.funding_buffer)
        funding_std = np.std(funding_array) + 1e-10
        funding_momentum = (funding_rate - funding_array[0]) / funding_std
        
        # Funding velocity (recent trend)
        if len(funding_array) >= 8:
            funding_velocity = (funding_array[-1] - funding_array[-8]) / 8
        else:
            funding_velocity = 0
        
        # OI growth
        oi_array = np.array(self.oi_buffer)
        oi_growth = (open_interest - oi_array[0]) / oi_array[0]
        
        # OI velocity
        if len(oi_array) >= 4:
            oi_velocity = (oi_array[-1] - oi_array[-4]) / oi_array[-4]
        else:
            oi_velocity = 0
        
        # Basis metrics
        basis_array = np.array(self.basis_buffer)
        basis_zscore = (basis - np.mean(basis_array)) / (np.std(basis_array) + 1e-10)
        
        # Composite divergence score
        # High score = bearish divergence (price flat/up, funding down, OI up)
        divergence_score = 0.0
        
        if price_momentum > -0.5:  # Price not crashing yet
            divergence_score += abs(funding_momentum) * 0.4
        
        if funding_momentum < 0:  # Funding falling
            divergence_score += oi_growth * 0.3 if oi_growth > 0 else 0
            divergence_score += abs(funding_velocity) * 100 * 0.3
        
        return {
            'price_momentum': price_momentum,
            'funding_rate': funding_rate,
            'predicted_funding': predicted_funding,
            'funding_momentum': funding_momentum,
            'funding_velocity': funding_velocity,
            'oi_growth': oi_growth,
            'oi_velocity': oi_velocity,
            'basis': basis,
            'basis_zscore': basis_zscore,
            'divergence_score': divergence_score,
        }
    
    def _is_bearish_divergence(self, metrics: Dict) -> bool:
        """Check if bearish divergence conditions are met."""
        
        conditions = [
            # Price not dumping yet (flat or rising)
            metrics['price_momentum'] > -1.0,
            
            # Funding falling significantly
            metrics['funding_momentum'] < self.funding_mom_thresh,
            
            # OI rising (smart money building positions)
            metrics['oi_growth'] > self.oi_growth_thresh,
            
            # Divergence score above threshold
            metrics['divergence_score'] > self.div_threshold,
            
            # Predicted funding falling or negative
            metrics['predicted_funding'] < metrics['funding_rate'],
        ]
        
        return sum(conditions) >= 4
    
    def _is_divergence_resolved(self, current_price: float, metrics: Dict) -> bool:
        """Check if divergence has resolved (price caught up to derivatives signal)."""
        
        price_drop = (self.divergence_start_price - current_price) / self.divergence_start_price
        
        conditions = [
            price_drop > 0.08,  # Price dropped 8%+
            metrics['funding_momentum'] > 0,  # Funding rebounding
            metrics['oi_growth'] < 0,  # OI dropping (positions closing)
        ]
        
        return sum(conditions) >= 2
    
    def _generate_short_signal(
        self,
        timestamp: float,
        spot_price: float,
        metrics: Dict,
        asset: str = "SOL",
    ) -> Signal:
        """Generate short signal based on divergence."""

        # Signal strength based on divergence magnitude
        strength = min(2.0, metrics['divergence_score'])

        # Confidence based on multiple confirmations
        confidence = 0.5 + min(0.4, metrics['oi_growth'] * 2)

        # Dynamic stop based on basis
        stop_distance = max(0.03, abs(metrics['basis']) * 10)

        return Signal(
            timestamp=timestamp,
            asset=asset,
            signal_type=SignalType.AGGRESSIVE_SHORT,
            size_multiplier=strength,
            confidence=confidence,
            strategy="funding_divergence",
            entry_price=spot_price,
            stop_loss=spot_price * (1 + stop_distance),
            take_profit=spot_price * (1 - stop_distance * 2),
            metadata=metrics
        )


# =============================================================================
# RISK CONTROLS - SHORT SQUEEZE PREVENTION
# =============================================================================

class ShortSqueezeMonitor:
    """
    Monitor for short squeeze conditions to protect aggressive short positions.
    
    SHORT SQUEEZE INDICATORS:
    -------------------------
    1. Aggregated OI Spike + Price Rising = Forced covering
    2. Funding Rate Spike Positive = Extreme short crowding
    3. CVD Flip Positive + Volume Surge = Aggressive buying
    4. Large Liquidations (Short) = Cascade starting
    
    PROTECTION MECHANISMS:
    ----------------------
    1. Reduce position when squeeze risk > threshold
    2. Tighten stops dynamically
    3. Hedge with BTC longs (lower beta)
    """
    
    def __init__(
        self,
        oi_spike_threshold: float = 0.10,  # 10% OI jump
        funding_spike_threshold: float = 0.001,  # 0.1% funding spike
        cvd_flip_threshold: float = 2.0,  # CVD z-score flip
        squeeze_risk_threshold: float = 0.7,
    ):
        self.oi_spike_thresh = oi_spike_threshold
        self.funding_spike_thresh = funding_spike_threshold
        self.cvd_flip_thresh = cvd_flip_threshold
        self.risk_thresh = squeeze_risk_threshold
        
        # Buffers
        self.oi_buffer = deque(maxlen=24)
        self.funding_buffer = deque(maxlen=24)
        self.cvd_buffer = deque(maxlen=100)
        self.short_liq_buffer = deque(maxlen=24)
        
    def update(
        self,
        open_interest: float,
        funding_rate: float,
        cvd: float,
        short_liquidations: float,
        price_change: float,
    ) -> Tuple[float, Dict]:
        """
        Calculate squeeze risk score [0, 1].
        
        Returns: (risk_score, risk_components)
        """
        # Update buffers
        self.oi_buffer.append(open_interest)
        self.funding_buffer.append(funding_rate)
        self.cvd_buffer.append(cvd)
        self.short_liq_buffer.append(short_liquidations)
        
        if len(self.oi_buffer) < 10:
            return 0.0, {}
        
        # 1. OI Spike Detection
        oi_array = np.array(self.oi_buffer)
        oi_change = (open_interest - oi_array[-4]) / oi_array[-4]
        oi_spike_risk = min(1.0, max(0, oi_change / self.oi_spike_thresh))
        
        # 2. Funding Spike Detection
        funding_array = np.array(self.funding_buffer)
        funding_change = funding_rate - np.mean(funding_array[-8:])
        funding_spike_risk = min(1.0, max(0, funding_change / self.funding_spike_thresh))
        
        # 3. CVD Flip Detection
        cvd_array = np.array(self.cvd_buffer)
        cvd_mean = np.mean(cvd_array)
        cvd_std = np.std(cvd_array) + 1e-10
        cvd_zscore = (cvd - cvd_mean) / cvd_std
        cvd_flip_risk = min(1.0, max(0, cvd_zscore / self.cvd_flip_thresh))
        
        # 4. Short Liquidation Cascade
        short_liq_array = np.array(self.short_liq_buffer)
        short_liq_mean = np.mean(short_liq_array) + 1e-10
        short_liq_spike = short_liquidations / short_liq_mean
        short_liq_risk = min(1.0, max(0, (short_liq_spike - 1) / 2))
        
        # 5. Price Action Confirmation
        price_risk = min(1.0, max(0, price_change / 0.02))  # Rising price
        
        # Composite risk score
        risk_score = (
            0.25 * oi_spike_risk +
            0.20 * funding_spike_risk +
            0.25 * cvd_flip_risk +
            0.15 * short_liq_risk +
            0.15 * price_risk
        )
        
        risk_components = {
            'oi_spike_risk': oi_spike_risk,
            'funding_spike_risk': funding_spike_risk,
            'cvd_flip_risk': cvd_flip_risk,
            'short_liq_risk': short_liq_risk,
            'price_risk': price_risk,
            'oi_change': oi_change,
            'funding_change': funding_change,
            'cvd_zscore': cvd_zscore,
        }
        
        return risk_score, risk_components
    
    def get_position_adjustment(self, risk_score: float) -> float:
        """
        Get recommended position size multiplier based on squeeze risk.
        
        Returns multiplier [0, 1] where:
            1.0 = full position
            0.5 = half position
            0.0 = flatten
        """
        if risk_score < 0.3:
            return 1.0  # Full position OK
        elif risk_score < 0.5:
            return 0.8  # Slight reduction
        elif risk_score < 0.7:
            return 0.5  # Significant reduction
        elif risk_score < 0.9:
            return 0.25  # Defensive
        else:
            return 0.0  # Flatten immediately


# =============================================================================
# MAIN QUANT AGENT
# =============================================================================

class QuantAgent:
    """
    Main Quant Agent orchestrating all strategies.

    Per-asset state isolation: each asset gets its own
    LiquidationCascadeHunter, FundingDivergence, and ShortSqueezeMonitor
    instances.  KalmanCointegration is inherently cross-asset (pairs
    trade) and remains shared.

    FEATURES REQUIRED:
    ------------------
    Microstructure:
        - bid_depth, ask_depth (order book)
        - taker_buy_volume, taker_sell_volume
        - cvd (cumulative volume delta)
        - trade_count, avg_trade_size

    Derivatives:
        - open_interest
        - funding_rate, predicted_funding_rate
        - liquidations_long, liquidations_short
        - spot_price, perp_price, basis

    Multi-Asset:
        - BTC price (for correlation/hedge)
        - ETH price (for stat-arb)
        - SOL price (primary target)
    """

    def __init__(
        self,
        config: Dict = None,
    ):
        self.config = config or {}
        self._logger = logging.getLogger(f"{__name__}.QuantAgent")

        # -- Freshness config (seconds) --
        self.microstructure_max_age_s: float = self.config.get(
            "microstructure_max_age_s", 120.0
        )
        self.derivatives_max_age_s: float = self.config.get(
            "derivatives_max_age_s", 120.0
        )

        # -- Per-asset strategy instances (lazy-initialised) --
        self._liq_hunters: Dict[str, LiquidationCascadeHunter] = {}
        self._funding_divs: Dict[str, FundingDivergence] = {}
        self._squeeze_monitors: Dict[str, ShortSqueezeMonitor] = {}

        # KalmanCointegration is cross-asset (shared)
        self.kalman_cointegration = KalmanCointegration(
            delta=self.config.get("kalman_delta", 1e-5),
            entry_zscore=self.config.get("cointegration_entry_z", 2.0),
            exit_zscore=self.config.get("cointegration_exit_z", 0.5),
            bear_bias=True,
        )

        # -- Per-asset signal cache (latest tick) --
        self._active_signals: Dict[str, List[Signal]] = {}
        self._last_intents: Dict[str, QuantAgentIntent] = {}
        self.signal_history: deque = deque(maxlen=1000)

        # -- Per-asset price tracking for squeeze monitor --
        self._prev_prices: Dict[str, float] = {}

        # Backward-compat aliases
        self.active_signals: List[Signal] = []

    # ------------------------------------------------------------------
    # Lazy per-asset initialisation
    # ------------------------------------------------------------------
    def _ensure_asset(self, asset: str) -> None:
        """Create per-asset strategy instances on first use."""
        if asset not in self._liq_hunters:
            self._liq_hunters[asset] = LiquidationCascadeHunter(
                lookback_window=self.config.get("liquidation_lookback", 100),
                zscore_threshold=self.config.get("liquidation_zscore", 2.5),
            )
        if asset not in self._funding_divs:
            self._funding_divs[asset] = FundingDivergence(
                funding_lookback=self.config.get("funding_lookback", 48),
                divergence_threshold=self.config.get("divergence_threshold", 1.5),
            )
        if asset not in self._squeeze_monitors:
            self._squeeze_monitors[asset] = ShortSqueezeMonitor(
                squeeze_risk_threshold=self.config.get(
                    "squeeze_risk_threshold", 0.7
                ),
            )

    # ------------------------------------------------------------------
    # Freshness check
    # ------------------------------------------------------------------
    def _check_freshness(
        self, data_timestamp: float, max_age_s: float, now: float
    ) -> float:
        """Return age in seconds.  Negative = stale beyond max_age_s."""
        age = now - data_timestamp
        return age

    # ------------------------------------------------------------------
    # Core per-asset processing
    # ------------------------------------------------------------------
    def _update_for_asset(
        self,
        asset: str,
        timestamp: float,
        microstructure: Optional[MarketMicrostructure],
        deriv: Optional[DerivativesState],
        prices: Dict[str, float],
    ) -> List[Signal]:
        """Run all strategies for a single asset.  Returns raw Signals."""
        self._ensure_asset(asset)
        signals: List[Signal] = []
        now = time.time()
        price = prices.get(asset)
        if price is None or deriv is None:
            return signals

        # -- Freshness gating --
        micro_age = 0.0
        if microstructure is not None:
            micro_age = self._check_freshness(
                microstructure.timestamp, self.microstructure_max_age_s, now
            )
            if micro_age > self.microstructure_max_age_s:
                self._logger.debug(
                    "[%s] microstructure stale (%.1fs > %.1fs) - skipping",
                    asset, micro_age, self.microstructure_max_age_s,
                )
                microstructure = None  # degrade: skip microstructure strategies

        deriv_age = self._check_freshness(
            deriv.timestamp, self.derivatives_max_age_s, now
        )
        if deriv_age > self.derivatives_max_age_s:
            self._logger.debug(
                "[%s] derivatives stale (%.1fs > %.1fs) - skipping all",
                asset, deriv_age, self.derivatives_max_age_s,
            )
            return signals  # no derivatives = no signals

        # -- 1. Squeeze monitor (risk control) --
        price_change = 0.0
        prev = self._prev_prices.get(asset)
        if prev is not None and prev > 0:
            price_change = (price - prev) / prev
        self._prev_prices[asset] = price

        squeeze_monitor = self._squeeze_monitors[asset]
        cvd_val = microstructure.cvd if microstructure else 0.0
        squeeze_risk, squeeze_components = squeeze_monitor.update(
            open_interest=deriv.open_interest,
            funding_rate=deriv.funding_rate,
            cvd=cvd_val,
            short_liquidations=deriv.short_liquidations,
            price_change=price_change,
        )

        position_mult = squeeze_monitor.get_position_adjustment(squeeze_risk)

        if squeeze_risk > 0.8:
            signals.append(Signal(
                timestamp=timestamp,
                asset=asset,
                signal_type=SignalType.FLATTEN,
                size_multiplier=1.0,
                confidence=squeeze_risk,
                strategy="risk_control",
                entry_price=price,
                stop_loss=price,
                take_profit=price,
                metadata={"squeeze_risk": squeeze_risk, **squeeze_components},
            ))
            return signals

        # -- 2. Liquidation Cascade Hunter (needs microstructure) --
        if microstructure is not None:
            liq_signal = self._liq_hunters[asset].update(
                timestamp=timestamp,
                price=price,
                open_interest=deriv.open_interest,
                long_liquidations=deriv.long_liquidations,
                short_liquidations=deriv.short_liquidations,
                cvd=microstructure.cvd,
                vwap_volume=(
                    microstructure.taker_buy_volume
                    + microstructure.taker_sell_volume
                ),
                asset=asset,
            )
            if liq_signal:
                liq_signal = liq_signal._replace(
                    size_multiplier=liq_signal.size_multiplier * position_mult
                )
                signals.append(liq_signal)

        # -- 3. Kalman Cointegration (cross-asset, SOL vs ETH) --
        if asset == "SOL" and "ETH" in prices:
            kalman_signal = self.kalman_cointegration.update(
                timestamp=timestamp,
                sol_price=prices["SOL"],
                eth_price=prices["ETH"],
                btc_price=prices.get("BTC"),
                hedge_asset="ETH",
                target_asset=asset,
            )
            if kalman_signal:
                kalman_signal = kalman_signal._replace(
                    size_multiplier=kalman_signal.size_multiplier * position_mult
                )
                signals.append(kalman_signal)

        # -- 4. Funding Divergence --
        funding_signal = self._funding_divs[asset].update(
            timestamp=timestamp,
            spot_price=deriv.spot_price,
            perp_price=deriv.perp_price,
            funding_rate=deriv.funding_rate,
            predicted_funding=deriv.predicted_funding,
            open_interest=deriv.open_interest,
            asset=asset,
        )
        if funding_signal:
            funding_signal = funding_signal._replace(
                size_multiplier=funding_signal.size_multiplier * position_mult
            )
            signals.append(funding_signal)

        return signals

    # ------------------------------------------------------------------
    # Signal -> QuantAgentIntent conversion
    # ------------------------------------------------------------------
    @staticmethod
    def _signals_to_intent(
        asset: str, signals: List[Signal], data_age: float = 0.0
    ) -> QuantAgentIntent:
        """Convert raw Signal list to a single HMATS-facing QuantAgentIntent."""
        if not signals:
            return _neutral_intent(asset)

        # Filter for directional signals
        short_signals = [
            s for s in signals
            if s.signal_type in (SignalType.AGGRESSIVE_SHORT, SignalType.SHORT)
        ]
        flatten_signals = [
            s for s in signals if s.signal_type == SignalType.FLATTEN
        ]

        # Flatten overrides
        if flatten_signals:
            best = max(flatten_signals, key=lambda s: s.confidence)
            return QuantAgentIntent(
                asset=asset,
                direction=0.0,
                confidence=best.confidence,
                strategy=best.strategy,
                signal_type=best.signal_type.name,
                size_multiplier=0.0,
                data_age_seconds=data_age,
                meta={
                    "reason": "flatten",
                    "component_signals": [s.strategy for s in signals],
                },
            )

        if not short_signals:
            return _neutral_intent(asset)

        # Confidence-weighted aggregation
        total_conf = sum(s.confidence for s in short_signals)
        if total_conf == 0:
            return _neutral_intent(asset)

        weighted_size = (
            sum(s.size_multiplier * s.confidence for s in short_signals)
            / total_conf
        )
        best = max(short_signals, key=lambda s: s.confidence)

        return QuantAgentIntent(
            asset=asset,
            direction=-1.0,  # short bias
            confidence=min(0.95, total_conf / len(short_signals)),
            strategy="composite" if len(short_signals) > 1 else best.strategy,
            signal_type=best.signal_type.name,
            size_multiplier=min(2.0, weighted_size),
            data_age_seconds=data_age,
            meta={
                "component_signals": [s.strategy for s in short_signals],
                "signal_count": len(short_signals),
                "stop_loss": best.stop_loss,
                "take_profit": best.take_profit,
                "entry_price": best.entry_price,
            },
        )

    # ------------------------------------------------------------------
    # Public API: legacy multi-asset update (backward compat)
    # ------------------------------------------------------------------
    def update(
        self,
        timestamp: float,
        microstructure: MarketMicrostructure,
        derivatives: Dict[str, DerivativesState],
        prices: Dict[str, float],
    ) -> List[Signal]:
        """Process market data and generate signals from all strategies.

        Backward-compatible: processes SOL by default (original behaviour).
        For multi-asset, call ``analyze_asset()`` per asset instead.
        """
        asset = "SOL"
        deriv = derivatives.get(asset)
        signals = self._update_for_asset(
            asset=asset,
            timestamp=timestamp,
            microstructure=microstructure,
            deriv=deriv,
            prices=prices,
        )
        # Cache
        self._active_signals[asset] = signals
        self.active_signals = signals  # backward compat
        for sig in signals:
            self.signal_history.append(sig)
        # Build intent
        self._last_intents[asset] = self._signals_to_intent(asset, signals)
        return signals

    # ------------------------------------------------------------------
    # Public API: HMATS per-asset entry point
    # ------------------------------------------------------------------
    def analyze_asset(
        self,
        asset: str,
        timestamp: float,
        microstructure: Optional[MarketMicrostructure] = None,
        derivatives: Optional[Dict[str, DerivativesState]] = None,
        prices: Optional[Dict[str, float]] = None,
    ) -> QuantAgentIntent:
        """Per-asset analysis - primary HMATS integration method.

        Returns a ``QuantAgentIntent`` (never None).
        """
        try:
            deriv = (derivatives or {}).get(asset)
            signals = self._update_for_asset(
                asset=asset,
                timestamp=timestamp,
                microstructure=microstructure,
                deriv=deriv,
                prices=prices or {},
            )
            self._active_signals[asset] = signals
            for sig in signals:
                self.signal_history.append(sig)
            intent = self._signals_to_intent(asset, signals)
            self._last_intents[asset] = intent
            return intent
        except Exception:
            self._logger.exception("[%s] analyze_asset failed - returning neutral", asset)
            return _neutral_intent(asset)

    # ------------------------------------------------------------------
    # Public API: runtime_spine compatibility
    # ------------------------------------------------------------------
    def generate_signal(
        self, asset: str = "SOL", price: float = None, regime: str = None,
        **kwargs,
    ) -> QuantAgentIntent:
        """Called by ``runtime_spine.py``.

        Returns the last cached intent for *asset*, or neutral if none.
        The ``price`` and ``regime`` args are accepted for interface compat
        but the actual signal comes from the most recent ``update()`` or
        ``analyze_asset()`` call.
        """
        return self._last_intents.get(asset, _neutral_intent(asset))

    # ------------------------------------------------------------------
    # Public API: integrated_system compatibility
    # ------------------------------------------------------------------
    def get_signal(
        self, asset: str = "SOL", market_data: Dict = None,
    ) -> Tuple[float, float]:
        """Called by ``integrated_system.py``.

        Returns ``(direction, confidence)`` tuple.
        """
        intent = self._last_intents.get(asset, _neutral_intent(asset))
        return (intent.direction, intent.confidence)

    # ------------------------------------------------------------------
    # Composite signal (backward compat, now asset-aware)
    # ------------------------------------------------------------------
    def get_composite_signal(self, asset: str = "SOL") -> Optional[Signal]:
        """Aggregate active signals for *asset* into a single Signal.

        Uses confidence-weighted averaging.
        """
        signals = self._active_signals.get(asset, self.active_signals)
        if not signals:
            return None

        short_signals = [
            s for s in signals
            if s.signal_type in (SignalType.AGGRESSIVE_SHORT, SignalType.SHORT)
        ]
        if not short_signals:
            return None

        total_confidence = sum(s.confidence for s in short_signals)
        if total_confidence == 0:
            return None

        weighted_size = (
            sum(s.size_multiplier * s.confidence for s in short_signals)
            / total_confidence
        )
        weighted_stop = (
            sum(s.stop_loss * s.confidence for s in short_signals)
            / total_confidence
        )
        weighted_tp = (
            sum(s.take_profit * s.confidence for s in short_signals)
            / total_confidence
        )

        best_signal = max(short_signals, key=lambda s: s.confidence)

        return Signal(
            timestamp=best_signal.timestamp,
            asset=asset,
            signal_type=SignalType.AGGRESSIVE_SHORT,
            size_multiplier=min(2.0, weighted_size),
            confidence=min(0.95, total_confidence / len(short_signals)),
            strategy="composite",
            entry_price=best_signal.entry_price,
            stop_loss=weighted_stop,
            take_profit=weighted_tp,
            metadata={
                "component_signals": [s.strategy for s in short_signals],
                "signal_count": len(short_signals),
            },
        )

    # ------------------------------------------------------------------
    # Hedge recommendation
    # ------------------------------------------------------------------
    def get_hedge_recommendation(self) -> Dict:
        """Return Kalman-estimated hedge ratio for stat-arb."""
        beta = self.kalman_cointegration.get_hedge_ratio()
        P = self.kalman_cointegration.get_state_covariance()

        return {
            "hedge_asset": "ETH",
            "hedge_ratio": beta,
            "hedge_uncertainty": np.sqrt(P[0, 0]),
            "recommendation": (
                f"For every 1 SOL short, long {beta:.3f} ETH equivalent"
            ),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_state(self, asset: str = None) -> None:
        """Reset per-asset state.  If *asset* is None, reset all."""
        if asset:
            self._liq_hunters.pop(asset, None)
            self._funding_divs.pop(asset, None)
            self._squeeze_monitors.pop(asset, None)
            self._active_signals.pop(asset, None)
            self._last_intents.pop(asset, None)
            self._prev_prices.pop(asset, None)
        else:
            self._liq_hunters.clear()
            self._funding_divs.clear()
            self._squeeze_monitors.clear()
            self._active_signals.clear()
            self._last_intents.clear()
            self._prev_prices.clear()
            self.active_signals = []


# =============================================================================
# FACTORY
# =============================================================================

def create_quant_agent(config: Dict = None) -> QuantAgent:
    """Factory function to create configured Quant Agent."""

    default_config = {
        # Liquidation Cascade
        "liquidation_lookback": 100,
        "liquidation_zscore": 2.5,
        # Kalman Cointegration
        "kalman_delta": 1e-5,
        "cointegration_entry_z": 2.0,
        "cointegration_exit_z": 0.5,
        # Funding Divergence
        "funding_lookback": 48,
        "divergence_threshold": 1.5,
        # Risk Control
        "squeeze_risk_threshold": 0.7,
        # Freshness
        "microstructure_max_age_s": 120.0,
        "derivatives_max_age_s": 120.0,
    }

    if config:
        default_config.update(config)

    return QuantAgent(default_config)


# =============================================================================
# V6 ADAPTER - unified analyze(asset, market_data, regime_state) interface
# =============================================================================

class QuantAgentV6Adapter:
    """
    Thin adapter that wraps QuantAgent with the V6 unified interface.

    Usage:
        adapter = QuantAgentV6Adapter()
        result = adapter.analyze("BTC", market_data, regime_state)
        # result is a flat dict for Authority Fusion / integration_v36
    """

    def __init__(self, config: Dict = None):
        self._agent = create_quant_agent(config)

    @property
    def inner(self) -> QuantAgent:
        """Access the underlying QuantAgent for advanced queries."""
        return self._agent

    def analyze(
        self,
        asset: str,
        market_data: Dict[str, Any],
        regime_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        V6 unified entry point.  Extracts microstructure/derivatives from
        market_data dict, runs all strategies, returns flat signal dict.

        Always returns a dict (never None, never raises).
        """
        try:
            now = time.time()

            # Extract derivatives state from market_data
            deriv = self._extract_derivatives(asset, market_data, now)
            micro = self._extract_microstructure(market_data, now)

            # Prices from market_data
            prices: Dict[str, float] = {}
            for a in ("BTC", "ETH", "SOL"):
                p = market_data.get(f"{a.lower()}_price") or market_data.get("current_price") if a == asset else None
                if p:
                    prices[a] = float(p)
            if asset not in prices:
                cp = market_data.get("current_price", 0)
                if cp:
                    prices[asset] = float(cp)

            intent = self._agent.analyze_asset(
                asset=asset,
                timestamp=now,
                microstructure=micro,
                derivatives={asset: deriv} if deriv else None,
                prices=prices,
            )

            result = intent.to_agent_signals_dict()
            result["regime_state"] = regime_state or ""
            result["asof_timestamp"] = now
            result["data_age_seconds"] = intent.data_age_seconds
            return result

        except Exception as e:
            logger.error(f"[QuantV6Adapter] analyze({asset}) failed: {e}", exc_info=True)
            return _neutral_intent(asset).to_agent_signals_dict()

    # ------------------------------------------------------------------
    # Extractors: market_data dict -> typed dataclass
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_derivatives(
        asset: str, md: Dict, now: float
    ) -> Optional[DerivativesState]:
        """Best-effort extraction.  Returns None if no derivative data."""
        oi = md.get("open_interest")
        funding = md.get("funding_rate")
        if oi is None and funding is None:
            return None
        return DerivativesState(
            timestamp=now,
            asset=asset,
            open_interest=float(oi or 0),
            oi_change_1h=float(md.get("oi_change_1h", 0)),
            oi_change_24h=float(md.get("oi_change_24h", 0)),
            oi_velocity=float(md.get("oi_velocity", 0)),
            funding_rate=float(funding or 0),
            predicted_funding=float(md.get("predicted_funding", funding or 0)),
            funding_velocity=float(md.get("funding_velocity", 0)),
            long_liquidations=float(md.get("long_liquidations", 0)),
            short_liquidations=float(md.get("short_liquidations", 0)),
            liquidation_imbalance=float(md.get("liquidation_imbalance", 0)),
            liquidation_volume_zscore=float(md.get("liquidation_volume_zscore", 0)),
            spot_price=float(md.get("current_price", 0)),
            perp_price=float(md.get("perp_price", md.get("current_price", 0))),
            basis=float(md.get("basis", 0)),
            basis_zscore=float(md.get("basis_zscore", 0)),
        )

    @staticmethod
    def _extract_microstructure(
        md: Dict, now: float
    ) -> Optional[MarketMicrostructure]:
        """Best-effort extraction.  Returns None if no microstructure data."""
        cvd = md.get("cvd")
        bid_imb = md.get("bid_imbalance") or md.get("depth_imbalance")
        if cvd is None and bid_imb is None:
            return None
        return MarketMicrostructure(
            timestamp=now,
            bid_depth=np.array([]),
            ask_depth=np.array([]),
            bid_imbalance=float(bid_imb or 0),
            taker_buy_volume=float(md.get("taker_buy_volume", 0)),
            taker_sell_volume=float(md.get("taker_sell_volume", 0)),
            cvd=float(cvd or 0),
            cvd_velocity=float(md.get("cvd_velocity", 0)),
            vwap=float(md.get("vwap", md.get("current_price", 0))),
            vwap_deviation=float(md.get("vwap_deviation", 0)),
            trade_count=int(md.get("trade_count", 0)),
            avg_trade_size=float(md.get("avg_trade_size", 0)),
            large_trade_ratio=float(md.get("large_trade_ratio", 0)),
        )


if __name__ == "__main__":
    # Legacy entry point
    agent = create_quant_agent()
    logger.info("Quant Agent initialized with strategies: "
                "LiquidationCascade, KalmanCointegration, FundingDivergence")

    # V6 adapter smoke test
    adapter = QuantAgentV6Adapter()
    result = adapter.analyze("BTC", {"current_price": 97000.0, "open_interest": 1e9})
    print(f"V6 adapter result: direction={result['quant_direction']}, "
          f"confidence={result['quant_confidence']}, strategy={result['quant_strategy']}")

"""
================================================================================
BITBEAST ENGINE - Advanced Signal Processing
================================================================================
Version: 3.1.2
Purpose: Enhanced signal filtering and position management

Key Features:
1. SNR Filter (Signal-to-Noise Ratio / Ants vs Train)
2. Dynamic Pyramiding (20-15-15 scaling)
3. MTF Tensor Fusion (128-dim multi-timeframe)
4. Circuit Breakers (anomaly detection)
5. Structure Break Detection (4H fractal breakout)
================================================================================
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
from collections import deque


class TrendState(Enum):
    """Trend state classification for SNR filter."""
    STRONG_TREND = "STRONG_TREND"
    WEAK_TREND = "WEAK_TREND"
    CONSOLIDATION = "CONSOLIDATION"
    REVERSAL = "REVERSAL"
    UNKNOWN = "UNKNOWN"


class CircuitBreakerType(Enum):
    """Types of circuit breakers."""
    VOLUME_DIVERGENCE = "VOLUME_DIVERGENCE"
    CROSS_ASSET_LAG = "CROSS_ASSET_LAG"
    LIQUIDATION_PROXIMITY = "LIQUIDATION_PROXIMITY"
    MTF_CONFLICT = "MTF_CONFLICT"


@dataclass
class SNRResult:
    """Signal-to-Noise Ratio analysis result."""
    trend_state: TrendState
    snr_ratio: float  # Signal vs noise ratio
    consolidation_ratio: float  # Consolidation time / Trend time
    pattern_duration: int  # Bars in current pattern
    signal_weight_adjustment: float  # Multiplier for signal strength
    confidence: float
    
    def should_reduce_reversion_signal(self) -> bool:
        """Check if mean reversion signals should be reduced."""
        return self.consolidation_ratio < 0.5


@dataclass
class PyramidTranche:
    """Single tranche of a pyramided position."""
    tranche_number: int
    size_pct: float  # 20%, 15%, 15%
    entry_price: float
    required_confidence: float  # 60%, 70%, 80%
    timestamp: datetime
    status: str = "PENDING"  # PENDING, FILLED, CANCELLED


@dataclass
class MTFTensorResult:
    """Multi-timeframe tensor fusion result."""
    tensor: np.ndarray  # 128-dim combined tensor
    alignment_score: float  # How aligned are the timeframes
    conflict_detected: bool
    dominant_timeframe: str
    confidence: float


@dataclass
class CircuitBreakerAlert:
    """Circuit breaker alert."""
    breaker_type: CircuitBreakerType
    severity: str  # WARNING, CRITICAL
    message: str
    should_halt: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)


class SNRFilter:
    """
    Signal-to-Noise Ratio Filter (Ants vs Train Detection).
    
    Philosophy: In markets, "ants" are small choppy moves during consolidation,
    while "trains" are strong trending moves. We want to ride trains, not ants.
    
    Rule: If consolidation_time < 50% of trend_time, reduce reversion signals
    because we're likely in a strong trend that will continue.
    """
    
    def __init__(self, 
                 lookback_periods: int = 50,
                 trend_threshold: float = 0.02,
                 consolidation_threshold: float = 0.01):
        self.lookback_periods = lookback_periods
        self.trend_threshold = trend_threshold
        self.consolidation_threshold = consolidation_threshold
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.trend_bars: int = 0
        self.consolidation_bars: int = 0
        self.current_state: TrendState = TrendState.UNKNOWN
    
    def analyze(self, 
               closes: np.ndarray,
               highs: np.ndarray,
               lows: np.ndarray) -> SNRResult:
        """
        Analyze price action to determine signal-to-noise ratio.
        
        Args:
            closes: Array of close prices
            highs: Array of high prices
            lows: Array of low prices
            
        Returns:
            SNRResult with trend state and adjustment factors
        """
        if len(closes) < self.lookback_periods:
            return SNRResult(
                trend_state=TrendState.UNKNOWN,
                snr_ratio=1.0,
                consolidation_ratio=0.5,
                pattern_duration=0,
                signal_weight_adjustment=1.0,
                confidence=0.0
            )
        
        # Calculate returns
        returns = np.diff(closes[-self.lookback_periods:]) / closes[-self.lookback_periods:-1]
        
        # Calculate ATR-normalized moves
        atr = np.mean(highs[-self.lookback_periods:] - lows[-self.lookback_periods:])
        normalized_moves = np.abs(returns * closes[-self.lookback_periods:-1]) / atr
        
        # Classify each bar
        trend_bars = 0
        consolidation_bars = 0
        
        for move in normalized_moves:
            if move > self.trend_threshold * 100:
                trend_bars += 1
            elif move < self.consolidation_threshold * 100:
                consolidation_bars += 1
        
        total_bars = len(normalized_moves)
        
        # Calculate ratios
        if trend_bars > 0:
            consolidation_ratio = consolidation_bars / trend_bars
        else:
            consolidation_ratio = float('inf')
        
        # Calculate SNR
        signal_strength = np.abs(closes[-1] - closes[-self.lookback_periods]) / atr
        noise = np.std(returns) * closes[-1] / atr
        snr_ratio = signal_strength / (noise + 1e-6)
        
        # Determine trend state
        if snr_ratio > 2.0 and trend_bars > consolidation_bars:
            trend_state = TrendState.STRONG_TREND
        elif snr_ratio > 1.0:
            trend_state = TrendState.WEAK_TREND
        elif consolidation_bars > trend_bars * 2:
            trend_state = TrendState.CONSOLIDATION
        else:
            trend_state = TrendState.UNKNOWN
        
        # Calculate signal weight adjustment
        if consolidation_ratio < 0.5:
            # Strong trend - reduce reversion signals
            signal_weight_adjustment = 0.5
        elif consolidation_ratio > 2.0:
            # Consolidation - boost reversion signals
            signal_weight_adjustment = 1.3
        else:
            signal_weight_adjustment = 1.0
        
        # Confidence based on clarity
        confidence = min(1.0, snr_ratio / 3.0)
        
        self.trend_bars = trend_bars
        self.consolidation_bars = consolidation_bars
        self.current_state = trend_state
        
        return SNRResult(
            trend_state=trend_state,
            snr_ratio=snr_ratio,
            consolidation_ratio=consolidation_ratio,
            pattern_duration=max(trend_bars, consolidation_bars),
            signal_weight_adjustment=signal_weight_adjustment,
            confidence=confidence
        )


class DynamicPyramiding:
    """
    Dynamic Position Pyramiding (20-15-15 Scaling).
    
    Strategy: Build into winning positions gradually:
    - Tranche 1: 20% at confidence ≥ 60%
    - Tranche 2: 15% at confidence ≥ 70% (after min 3 bars)
    - Tranche 3: 15% at confidence ≥ 80% (after min 3 bars from T2)
    
    Total: 50% of planned position
    Remaining 50% held for scaling out
    """
    
    def __init__(self,
                 tranche_sizes: List[float] = None,
                 confidence_thresholds: List[float] = None,
                 min_bars_between: int = 3):
        self.tranche_sizes = tranche_sizes or [0.20, 0.15, 0.15]
        self.confidence_thresholds = confidence_thresholds or [0.60, 0.70, 0.80]
        self.min_bars_between = min_bars_between
        self.logger = logging.getLogger(__name__)
        
        # Active pyramid tracking
        self.active_pyramids: Dict[str, List[PyramidTranche]] = {}
        self.last_fill_bar: Dict[str, int] = {}
    
    def should_add_tranche(self,
                          symbol: str,
                          confidence: float,
                          current_bar: int,
                          price: float) -> Optional[PyramidTranche]:
        """
        Check if we should add a new tranche to the position.
        
        Args:
            symbol: Trading pair
            confidence: Current signal confidence
            current_bar: Current bar number
            price: Current price
            
        Returns:
            PyramidTranche if we should add, None otherwise
        """
        if symbol not in self.active_pyramids:
            self.active_pyramids[symbol] = []
            self.last_fill_bar[symbol] = -self.min_bars_between
        
        filled_tranches = len([t for t in self.active_pyramids[symbol] if t.status == "FILLED"])
        
        # Maximum pyramid depth reached - no more tranches to add
        if filled_tranches >= len(self.tranche_sizes):
            return None
        
        # Check minimum bars since last fill
        if current_bar - self.last_fill_bar[symbol] < self.min_bars_between:
            return None
        
        # Check confidence threshold
        required_confidence = self.confidence_thresholds[filled_tranches]
        if confidence < required_confidence:
            return None
        
        # Create new tranche
        tranche = PyramidTranche(
            tranche_number=filled_tranches + 1,
            size_pct=self.tranche_sizes[filled_tranches],
            entry_price=price,
            required_confidence=required_confidence,
            timestamp=datetime.utcnow()
        )
        
        return tranche
    
    def fill_tranche(self, symbol: str, tranche: PyramidTranche, bar: int) -> None:
        """Mark a tranche as filled."""
        tranche.status = "FILLED"
        self.active_pyramids[symbol].append(tranche)
        self.last_fill_bar[symbol] = bar
        self.logger.info(f"Filled tranche {tranche.tranche_number} for {symbol} @ ${tranche.entry_price:,.2f}")
    
    def get_position_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of pyramid position."""
        if symbol not in self.active_pyramids:
            return {"filled": 0, "total_size": 0, "avg_price": 0}
        
        filled = [t for t in self.active_pyramids[symbol] if t.status == "FILLED"]
        
        if not filled:
            return {"filled": 0, "total_size": 0, "avg_price": 0}
        
        total_size = sum(t.size_pct for t in filled)
        avg_price = sum(t.entry_price * t.size_pct for t in filled) / total_size
        
        return {
            "filled": len(filled),
            "total_size": total_size,
            "avg_price": avg_price,
            "tranches": [
                {"num": t.tranche_number, "price": t.entry_price, "size": t.size_pct}
                for t in filled
            ]
        }
    
    def clear_pyramid(self, symbol: str) -> None:
        """Clear pyramid for a symbol (on position close)."""
        if symbol in self.active_pyramids:
            del self.active_pyramids[symbol]
        if symbol in self.last_fill_bar:
            del self.last_fill_bar[symbol]


class MTFTensorFusion:
    """
    Multi-Timeframe Tensor Fusion (128-dim).
    
    Combines features from multiple timeframes:
    - 1m: 64 dimensions (micro-risk, immediate signals)
    - 15m: 32 dimensions (confirmation, intermediate trend)
    - 4H: 32 dimensions (macro trend, structure)
    
    Total: 128 dimensions
    
    Features alignment scoring and conflict detection.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Feature dimensions per timeframe
        self.tf_dims = {
            "1m": 64,
            "15m": 32,
            "4H": 32
        }
        
        # Cached tensors
        self.cached_tensors: Dict[str, np.ndarray] = {}
    
    def create_tensor(self,
                     features_1m: np.ndarray,
                     features_15m: np.ndarray,
                     features_4h: np.ndarray) -> MTFTensorResult:
        """
        Create fused 128-dim tensor from multi-timeframe features.
        
        Args:
            features_1m: 64-dim features from 1-minute data
            features_15m: 32-dim features from 15-minute data
            features_4h: 32-dim features from 4-hour data
            
        Returns:
            MTFTensorResult with fused tensor and analysis
        """
        # Validate dimensions
        if len(features_1m) != 64:
            features_1m = self._resize_features(features_1m, 64)
        if len(features_15m) != 32:
            features_15m = self._resize_features(features_15m, 32)
        if len(features_4h) != 32:
            features_4h = self._resize_features(features_4h, 32)
        
        # Concatenate tensors
        tensor = np.concatenate([features_1m, features_15m, features_4h])
        
        # Calculate alignment score
        alignment_score = self._calculate_alignment(features_1m, features_15m, features_4h)
        
        # Detect conflicts
        conflict_detected = self._detect_conflicts(features_1m, features_15m, features_4h)
        
        # Determine dominant timeframe
        dominant_tf = self._get_dominant_timeframe(features_1m, features_15m, features_4h)
        
        # Confidence based on alignment
        confidence = alignment_score if not conflict_detected else alignment_score * 0.5
        
        return MTFTensorResult(
            tensor=tensor,
            alignment_score=alignment_score,
            conflict_detected=conflict_detected,
            dominant_timeframe=dominant_tf,
            confidence=confidence
        )
    
    def _resize_features(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """Resize features to target dimension."""
        if len(features) == target_dim:
            return features
        elif len(features) > target_dim:
            return features[:target_dim]
        else:
            padded = np.zeros(target_dim)
            padded[:len(features)] = features
            return padded
    
    def _calculate_alignment(self,
                            f1m: np.ndarray,
                            f15m: np.ndarray,
                            f4h: np.ndarray) -> float:
        """Calculate alignment score between timeframes."""
        # Extract directional signals (simplified)
        dir_1m = np.sign(np.mean(f1m[:8]))  # First 8 dims assumed directional
        dir_15m = np.sign(np.mean(f15m[:8]))
        dir_4h = np.sign(np.mean(f4h[:8]))
        
        # All aligned = 1.0, mixed = 0.5, all opposed = 0.0
        agreements = (dir_1m == dir_15m) + (dir_15m == dir_4h) + (dir_1m == dir_4h)
        return agreements / 3.0
    
    def _detect_conflicts(self,
                         f1m: np.ndarray,
                         f15m: np.ndarray,
                         f4h: np.ndarray) -> bool:
        """Detect significant conflicts between timeframes."""
        dir_1m = np.sign(np.mean(f1m[:8]))
        dir_4h = np.sign(np.mean(f4h[:8]))
        
        # Conflict if 1m and 4H show opposite directions
        return dir_1m != 0 and dir_4h != 0 and dir_1m != dir_4h
    
    def _get_dominant_timeframe(self,
                               f1m: np.ndarray,
                               f15m: np.ndarray,
                               f4h: np.ndarray) -> str:
        """Determine which timeframe has strongest signal."""
        strength_1m = np.abs(np.mean(f1m[:8]))
        strength_15m = np.abs(np.mean(f15m[:8]))
        strength_4h = np.abs(np.mean(f4h[:8]))
        
        if strength_4h > strength_15m and strength_4h > strength_1m:
            return "4H"
        elif strength_15m > strength_1m:
            return "15m"
        else:
            return "1m"


class CircuitBreaker:
    """
    Circuit Breaker System - Anomaly Detection.
    
    Monitors for dangerous conditions and can halt trading:
    1. Volume Divergence: Price down, volume down (weak sell-off)
    2. Cross-Asset Lag: BTC leads, SOL lags dangerously
    3. Liquidation Proximity: Large liquidation cluster within 1%
    4. MTF Conflict: Timeframes showing opposite signals
    """
    
    def __init__(self,
                 volume_divergence_threshold: float = 0.3,
                 lag_threshold_bars: int = 5,
                 liquidation_proximity_pct: float = 0.01):
        self.volume_divergence_threshold = volume_divergence_threshold
        self.lag_threshold_bars = lag_threshold_bars
        self.liquidation_proximity_pct = liquidation_proximity_pct
        self.logger = logging.getLogger(__name__)
        
        # Alert history
        self.alerts: deque = deque(maxlen=100)
        self.active_breakers: Dict[CircuitBreakerType, bool] = {
            CircuitBreakerType.VOLUME_DIVERGENCE: False,
            CircuitBreakerType.CROSS_ASSET_LAG: False,
            CircuitBreakerType.LIQUIDATION_PROXIMITY: False,
            CircuitBreakerType.MTF_CONFLICT: False,
        }
    
    def check_volume_divergence(self,
                               price_change: float,
                               volume_change: float) -> Optional[CircuitBreakerAlert]:
        """Check for volume divergence (price down, volume down)."""
        if price_change < -0.01 and volume_change < -self.volume_divergence_threshold:
            alert = CircuitBreakerAlert(
                breaker_type=CircuitBreakerType.VOLUME_DIVERGENCE,
                severity="WARNING",
                message=f"Volume divergence: Price {price_change:.1%}, Volume {volume_change:.1%}",
                should_halt=False
            )
            self.alerts.append(alert)
            self.active_breakers[CircuitBreakerType.VOLUME_DIVERGENCE] = True
            return alert
        
        self.active_breakers[CircuitBreakerType.VOLUME_DIVERGENCE] = False
        return None
    
    def check_cross_asset_lag(self,
                             btc_move: float,
                             alt_move: float,
                             lag_bars: int) -> Optional[CircuitBreakerAlert]:
        """Check for dangerous cross-asset lag."""
        if lag_bars > self.lag_threshold_bars and abs(btc_move) > 0.02:
            severity = "CRITICAL" if lag_bars > self.lag_threshold_bars * 2 else "WARNING"
            alert = CircuitBreakerAlert(
                breaker_type=CircuitBreakerType.CROSS_ASSET_LAG,
                severity=severity,
                message=f"BTC moved {btc_move:.1%}, alt lagging by {lag_bars} bars",
                should_halt=severity == "CRITICAL"
            )
            self.alerts.append(alert)
            self.active_breakers[CircuitBreakerType.CROSS_ASSET_LAG] = True
            return alert
        
        self.active_breakers[CircuitBreakerType.CROSS_ASSET_LAG] = False
        return None
    
    def check_liquidation_proximity(self,
                                   current_price: float,
                                   liquidation_levels: List[Tuple[float, float]]) -> Optional[CircuitBreakerAlert]:
        """
        Check for large liquidation clusters nearby.
        
        Args:
            current_price: Current price
            liquidation_levels: List of (price, volume) tuples
        """
        for liq_price, liq_volume in liquidation_levels:
            distance = abs(liq_price - current_price) / current_price
            if distance < self.liquidation_proximity_pct and liq_volume > 1_000_000:
                alert = CircuitBreakerAlert(
                    breaker_type=CircuitBreakerType.LIQUIDATION_PROXIMITY,
                    severity="CRITICAL",
                    message=f"${liq_volume/1e6:.1f}M liquidations at ${liq_price:,.0f} ({distance:.2%} away)",
                    should_halt=True
                )
                self.alerts.append(alert)
                self.active_breakers[CircuitBreakerType.LIQUIDATION_PROXIMITY] = True
                return alert
        
        self.active_breakers[CircuitBreakerType.LIQUIDATION_PROXIMITY] = False
        return None
    
    def check_mtf_conflict(self, mtf_result: MTFTensorResult) -> Optional[CircuitBreakerAlert]:
        """Check for multi-timeframe conflict."""
        if mtf_result.conflict_detected:
            alert = CircuitBreakerAlert(
                breaker_type=CircuitBreakerType.MTF_CONFLICT,
                severity="WARNING",
                message=f"MTF conflict detected, dominant TF: {mtf_result.dominant_timeframe}",
                should_halt=False
            )
            self.alerts.append(alert)
            self.active_breakers[CircuitBreakerType.MTF_CONFLICT] = True
            return alert
        
        self.active_breakers[CircuitBreakerType.MTF_CONFLICT] = False
        return None
    
    def should_halt_trading(self) -> Tuple[bool, List[str]]:
        """Check if any circuit breaker requires trading halt."""
        halt_reasons = []
        
        for alert in list(self.alerts)[-10:]:  # Check recent alerts
            if alert.should_halt:
                halt_reasons.append(alert.message)
        
        return len(halt_reasons) > 0, halt_reasons
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "active_breakers": {k.value: v for k, v in self.active_breakers.items()},
            "recent_alerts": len([a for a in self.alerts if a.timestamp > datetime.utcnow() - timedelta(hours=1)]),
            "should_halt": self.should_halt_trading()[0]
        }


class StructureBreakDetector:
    """
    Structure Break Detection using 4H Fractal Breakout.
    
    Uses Williams Fractals (2-bar) to identify key levels:
    - Fractal High: Higher high surrounded by lower highs
    - Fractal Low: Lower low surrounded by higher lows
    
    Rules:
    - LONG only allowed after 4H fractal high breakout
    - SHORT only allowed after 4H fractal low breakout
    """
    
    def __init__(self, fractal_bars: int = 2):
        self.fractal_bars = fractal_bars
        self.logger = logging.getLogger(__name__)
        
        # Tracked levels
        self.fractal_highs: List[Tuple[datetime, float]] = []
        self.fractal_lows: List[Tuple[datetime, float]] = []
        self.last_break: Optional[Tuple[str, float, datetime]] = None
    
    def detect_fractals(self,
                       timestamps: List[datetime],
                       highs: np.ndarray,
                       lows: np.ndarray) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Detect Williams Fractals in price data.
        
        A fractal high requires `fractal_bars` lower highs on each side.
        A fractal low requires `fractal_bars` higher lows on each side.
        """
        n = self.fractal_bars
        new_highs = []
        new_lows = []
        
        for i in range(n, len(highs) - n):
            # Check for fractal high
            is_fractal_high = True
            for j in range(1, n + 1):
                if highs[i - j] >= highs[i] or highs[i + j] >= highs[i]:
                    is_fractal_high = False
                    break
            if is_fractal_high:
                new_highs.append((timestamps[i], highs[i]))
            
            # Check for fractal low
            is_fractal_low = True
            for j in range(1, n + 1):
                if lows[i - j] <= lows[i] or lows[i + j] <= lows[i]:
                    is_fractal_low = False
                    break
            if is_fractal_low:
                new_lows.append((timestamps[i], lows[i]))
        
        # Update cached levels (keep last 10)
        self.fractal_highs = (self.fractal_highs + new_highs)[-10:]
        self.fractal_lows = (self.fractal_lows + new_lows)[-10:]
        
        return {
            "fractal_highs": new_highs,
            "fractal_lows": new_lows
        }
    
    def check_structure_break(self, current_price: float) -> Dict[str, Any]:
        """
        Check if current price breaks structure.
        
        Returns dict with break direction and validity.
        """
        result = {
            "long_allowed": False,
            "short_allowed": False,
            "break_direction": None,
            "break_level": None
        }
        
        # Check for bullish break (price > recent fractal high)
        if self.fractal_highs:
            recent_high = max(self.fractal_highs, key=lambda x: x[0])
            if current_price > recent_high[1]:
                result["long_allowed"] = True
                result["break_direction"] = "BULLISH"
                result["break_level"] = recent_high[1]
                self.last_break = ("BULLISH", recent_high[1], datetime.utcnow())
        
        # Check for bearish break (price < recent fractal low)
        if self.fractal_lows:
            recent_low = max(self.fractal_lows, key=lambda x: x[0])
            if current_price < recent_low[1]:
                result["short_allowed"] = True
                result["break_direction"] = "BEARISH"
                result["break_level"] = recent_low[1]
                self.last_break = ("BEARISH", recent_low[1], datetime.utcnow())
        
        return result


class BitBeastEngine:
    """
    BitBeast Engine - Main Integration Point.
    
    Combines all BitBeast components:
    1. SNR Filter
    2. Dynamic Pyramiding
    3. MTF Tensor Fusion
    4. Circuit Breakers
    5. Structure Break Detection
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.snr_filter = SNRFilter()
        self.pyramiding = DynamicPyramiding()
        self.mtf_fusion = MTFTensorFusion()
        self.circuit_breaker = CircuitBreaker()
        self.structure_detector = StructureBreakDetector()
        
        self.logger.info("BitBeast Engine initialized")
    
    def analyze(self,
               symbol: str,
               price_data: Dict[str, np.ndarray],
               features: Dict[str, np.ndarray],
               current_price: float,
               confidence: float,
               current_bar: int) -> Dict[str, Any]:
        """
        Run full BitBeast analysis.
        
        Args:
            symbol: Trading pair
            price_data: Dict with 'close', 'high', 'low', 'volume'
            features: Dict with '1m', '15m', '4H' feature arrays
            current_price: Current price
            confidence: Current signal confidence
            current_bar: Current bar number
            
        Returns:
            Comprehensive analysis result
        """
        results = {}
        
        # 1. SNR Analysis
        snr_result = self.snr_filter.analyze(
            price_data.get('close', np.array([])),
            price_data.get('high', np.array([])),
            price_data.get('low', np.array([]))
        )
        results['snr'] = {
            'trend_state': snr_result.trend_state.value,
            'snr_ratio': snr_result.snr_ratio,
            'signal_adjustment': snr_result.signal_weight_adjustment,
            'reduce_reversion': snr_result.should_reduce_reversion_signal()
        }
        
        # 2. MTF Tensor Fusion
        mtf_result = self.mtf_fusion.create_tensor(
            features.get('1m', np.zeros(64)),
            features.get('15m', np.zeros(32)),
            features.get('4H', np.zeros(32))
        )
        results['mtf'] = {
            'alignment': mtf_result.alignment_score,
            'conflict': mtf_result.conflict_detected,
            'dominant_tf': mtf_result.dominant_timeframe,
            'confidence': mtf_result.confidence
        }
        
        # 3. Circuit Breaker Check
        cb_status = self.circuit_breaker.get_status()
        mtf_alert = self.circuit_breaker.check_mtf_conflict(mtf_result)
        results['circuit_breakers'] = {
            'status': cb_status,
            'mtf_alert': mtf_alert.message if mtf_alert else None
        }
        
        # 4. Pyramid Check
        tranche = self.pyramiding.should_add_tranche(symbol, confidence, current_bar, current_price)
        results['pyramiding'] = {
            'should_add': tranche is not None,
            'tranche': tranche.tranche_number if tranche else None,
            'size': tranche.size_pct if tranche else None,
            'position_summary': self.pyramiding.get_position_summary(symbol)
        }
        
        # 5. Structure Break
        structure = self.structure_detector.check_structure_break(current_price)
        results['structure'] = structure
        
        # 6. Final Recommendation
        should_halt, halt_reasons = self.circuit_breaker.should_halt_trading()
        
        results['recommendation'] = {
            'should_trade': not should_halt,
            'halt_reasons': halt_reasons,
            'adjusted_confidence': confidence * snr_result.signal_weight_adjustment * mtf_result.confidence,
            'long_allowed': structure['long_allowed'] and not should_halt,
            'short_allowed': structure['short_allowed'] and not should_halt
        }
        
        return results
    
    def fill_pyramid_tranche(self, symbol: str, tranche: PyramidTranche, bar: int) -> None:
        """Fill a pyramid tranche."""
        self.pyramiding.fill_tranche(symbol, tranche, bar)
    
    def clear_position(self, symbol: str) -> None:
        """Clear position tracking for a symbol."""
        self.pyramiding.clear_pyramid(symbol)


if __name__ == "__main__":
    # Test BitBeast Engine
    engine = BitBeastEngine()
    
    # Create sample data
    np.random.seed(42)
    n = 100
    close = 45000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    volume = np.abs(np.random.randn(n) * 1000) + 500
    
    price_data = {
        'close': close,
        'high': high,
        'low': low,
        'volume': volume
    }
    
    features = {
        '1m': np.random.randn(64),
        '15m': np.random.randn(32),
        '4H': np.random.randn(32)
    }
    
    result = engine.analyze(
        symbol="BTC/USDT:USDT",
        price_data=price_data,
        features=features,
        current_price=close[-1],
        confidence=0.75,
        current_bar=100
    )
    
    print("BitBeast Analysis Result:")
    print(f"  SNR Trend State: {result['snr']['trend_state']}")
    print(f"  SNR Ratio: {result['snr']['snr_ratio']:.2f}")
    print(f"  MTF Alignment: {result['mtf']['alignment']:.2f}")
    print(f"  MTF Conflict: {result['mtf']['conflict']}")
    print(f"  Should Trade: {result['recommendation']['should_trade']}")
    print(f"  Adjusted Confidence: {result['recommendation']['adjusted_confidence']:.2f}")

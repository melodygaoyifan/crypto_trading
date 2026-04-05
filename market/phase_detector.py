"""
================================================================================
HMATS v3.4 - REGIME PHASE DETECTOR
================================================================================
Version: 3.4.0
Upgrade From: v3.3-HR
Purpose: Extend Regime Agent with phase detection for profit-driven decisions

NEW OUTPUTS (v3.4):
    - regime_phase: IGNITION | EXPANSION | SATURATION | EXHAUSTION
    - opportunity_density: 0.0 to 1.0 (quality score)
    - regime_phase_sol: SOL-specific phase
    - regime_phase_btc: BTC-specific phase
    - regime_phase_eth: ETH-specific phase

DOWNSTREAM CONSUMERS:
    - Fusion Engine: Authority assignment based on phase
    - Tranche Manager: Escalation gating (no T4 in SATURATION)
    - DRL Agent: State vector input
    - Exit Manager: Phase transition triggers scale-out
    - Risk Agent: Per-asset exposure adjustment

PROFITABILITY IMPACT:
    - Better entry timing (IGNITION = aggressive T1-T2)
    - Better size management (SATURATION = no T4)
    - Better exits (EXHAUSTION = trigger scale-out)
    - Expected: +15-25% win rate improvement on trend trades
================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RegimePhase(Enum):
    """
    Regime phase within a trend/move.
    
    CRITICAL: This is NOT the same as regime_state (BULL/BEAR/SIDEWAYS).
    This describes WHERE we are within a move.
    """
    IGNITION = "IGNITION"           # Early: Structure break, low participation
    EXPANSION = "EXPANSION"         # Middle: Acceleration, broadening participation
    SATURATION = "SATURATION"       # Late: Plateau, everyone is in
    EXHAUSTION = "EXHAUSTION"       # End: Divergences, reversal signals
    UNDEFINED = "UNDEFINED"         # Insufficient data


# Phase scores for opportunity density calculation
PHASE_OPPORTUNITY_SCORES = {
    RegimePhase.IGNITION: 1.0,      # Best entry zone
    RegimePhase.EXPANSION: 0.8,     # Still good
    RegimePhase.SATURATION: 0.3,    # Caution
    RegimePhase.EXHAUSTION: 0.1,    # Exit zone
    RegimePhase.UNDEFINED: 0.5,     # Neutral
}


@dataclass
class PhaseIndicators:
    """Raw indicators used for phase detection."""
    
    # Trend strength
    adx: float = 0.0
    adx_slope: float = 0.0
    
    # Momentum
    momentum_1bar: float = 0.0
    momentum_4bar: float = 0.0
    rsi: float = 50.0
    rsi_divergence: bool = False
    
    # Volume
    volume_ratio: float = 1.0
    volume_trend: float = 0.0
    
    # Participation
    participation_rate: float = 0.5
    
    # Structure
    price_break_pct: float = 0.0
    bars_since_break: int = 0
    reversal_candle: bool = False
    trend_bars: int = 0


@dataclass
class RegimePhaseResult:
    """Complete regime phase analysis result."""
    
    # Primary output
    phase: RegimePhase = RegimePhase.UNDEFINED
    phase_confidence: float = 0.0
    phase_age_bars: int = 0
    
    # Opportunity quality
    opportunity_density: float = 0.5
    
    # Asset-level phases
    phase_sol: RegimePhase = RegimePhase.UNDEFINED
    phase_btc: RegimePhase = RegimePhase.UNDEFINED
    phase_eth: RegimePhase = RegimePhase.UNDEFINED
    
    # Phase transition detection
    phase_transition: bool = False
    previous_phase: Optional[RegimePhase] = None
    
    # Supporting data
    indicators: Optional[PhaseIndicators] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_early_phase(self) -> bool:
        return self.phase in [RegimePhase.IGNITION, RegimePhase.EXPANSION]
    
    @property
    def is_late_phase(self) -> bool:
        return self.phase in [RegimePhase.SATURATION, RegimePhase.EXHAUSTION]
    
    @property
    def allows_tranche_escalation(self) -> bool:
        """No T4 in SATURATION or EXHAUSTION."""
        return self.phase in [RegimePhase.IGNITION, RegimePhase.EXPANSION]


class RegimePhaseDetector:
    """
    Detects current phase within a market regime.
    
    HOW IT IMPROVES PROFITABILITY:
        1. IGNITION detection -> Enter early in trends
        2. EXPANSION detection -> Build size aggressively  
        3. SATURATION detection -> Stop adding size
        4. EXHAUSTION detection -> Trigger exits
    """
    
    CONFIG = {
        # IGNITION thresholds
        "ignition_price_break_min": 0.02,
        "ignition_volume_min": 2.0,
        "ignition_participation_max": 0.4,
        
        # EXPANSION thresholds
        "expansion_adx_min": 25.0,
        "expansion_volume_sustained_min": 1.5,
        "expansion_participation_min": 0.6,
        
        # SATURATION thresholds
        "saturation_adx_flat_threshold": 2.0,
        "saturation_participation_high": 0.8,
        
        # EXHAUSTION thresholds
        "exhaustion_volume_collapse": 0.7,
        
        # Asset lookbacks
        "sol_lookback_bars": 15,
        "btc_lookback_bars": 25,
        "eth_lookback_bars": 20,
        
        "min_bars_required": 10,
    }
    
    def __init__(self):
        self._last_phase: Optional[RegimePhase] = None
        self._phase_start_bar: int = 0
        self._current_bar: int = 0
        logger.info("RegimePhaseDetector initialized")
    
    def detect(
        self,
        regime_state: str,
        crack_active: bool,
        prices: Dict[str, List[float]],
        volumes: Dict[str, List[float]],
        agent_signals: Optional[Dict] = None,
    ) -> RegimePhaseResult:
        """
        Detect current regime phase.
        
        WHERE THIS IS CONSUMED:
            - authority_fusion_v34.py: Determines authority assignment
            - tranche_manager_v34.py: Gates T4 escalation
            - exit_alpha_v34.py: Triggers scale-out on transition
            - drl_state_v34.py: Input to DRL state vector
        """
        self._current_bar += 1
        
        sol_indicators = self._calculate_indicators(
            prices.get("sol", []),
            volumes.get("sol", []),
            prices,
        )
        
        primary_phase, confidence = self._detect_phase(
            sol_indicators, regime_state, crack_active
        )
        
        phase_sol = self._detect_asset_phase("sol", prices, volumes)
        phase_btc = self._detect_asset_phase("btc", prices, volumes)
        phase_eth = self._detect_asset_phase("eth", prices, volumes)
        
        phase_transition = (
            self._last_phase is not None and 
            primary_phase != self._last_phase and
            primary_phase != RegimePhase.UNDEFINED
        )
        
        previous_phase = self._last_phase if phase_transition else None
        
        if phase_transition or self._last_phase is None:
            self._phase_start_bar = self._current_bar
        
        phase_age_bars = self._current_bar - self._phase_start_bar
        
        opportunity_density = self._calculate_opportunity_density(
            primary_phase, crack_active, agent_signals, sol_indicators
        )
        
        self._last_phase = primary_phase
        
        result = RegimePhaseResult(
            phase=primary_phase,
            phase_confidence=confidence,
            phase_age_bars=phase_age_bars,
            opportunity_density=opportunity_density,
            phase_sol=phase_sol,
            phase_btc=phase_btc,
            phase_eth=phase_eth,
            phase_transition=phase_transition,
            previous_phase=previous_phase,
            indicators=sol_indicators,
        )
        
        if phase_transition:
            logger.info(
                f"PHASE TRANSITION: {previous_phase.value} -> {primary_phase.value} "
                f"(density={opportunity_density:.2f})"
            )
        
        return result
    
    def _calculate_indicators(
        self,
        prices: List[float],
        volumes: List[float],
        all_prices: Dict[str, List[float]],
    ) -> PhaseIndicators:
        """Calculate phase detection indicators."""
        
        indicators = PhaseIndicators()
        
        if len(prices) < self.CONFIG["min_bars_required"]:
            return indicators
        
        prices_arr = np.array(prices)
        volumes_arr = np.array(volumes) if volumes else np.ones_like(prices_arr)
        
        # Momentum
        if len(prices_arr) >= 5:
            indicators.momentum_1bar = (prices_arr[-1] / prices_arr[-2] - 1) * 100
            indicators.momentum_4bar = (prices_arr[-1] / prices_arr[-5] - 1) * 100
        
        # RSI
        if len(prices_arr) >= 14:
            indicators.rsi = self._calculate_rsi(prices_arr, 14)
        
        # Volume ratio
        if len(volumes_arr) >= 20:
            avg_vol = np.mean(volumes_arr[-20:])
            if avg_vol > 0:
                indicators.volume_ratio = volumes_arr[-1] / avg_vol
            recent_vol = np.mean(volumes_arr[-5:])
            prev_vol = np.mean(volumes_arr[-10:-5])
            if prev_vol > 0:
                indicators.volume_trend = (recent_vol / prev_vol) - 1
        
        # ADX proxy
        if len(prices_arr) >= 20:
            returns = np.diff(prices_arr[-20:]) / prices_arr[-21:-1]
            indicators.adx = abs(np.mean(returns)) * 1000
            if len(prices_arr) >= 25:
                returns_prev = np.diff(prices_arr[-25:-5]) / prices_arr[-26:-6]
                adx_prev = abs(np.mean(returns_prev)) * 1000
                indicators.adx_slope = indicators.adx - adx_prev
        
        # Price break
        if len(prices_arr) >= 10:
            recent_high = np.max(prices_arr[-10:-1])
            recent_low = np.min(prices_arr[-10:-1])
            if prices_arr[-1] > recent_high:
                indicators.price_break_pct = (prices_arr[-1] / recent_high - 1)
            elif prices_arr[-1] < recent_low:
                indicators.price_break_pct = (prices_arr[-1] / recent_low - 1)
        
        # RSI divergence
        if len(prices_arr) >= 20:
            price_making_high = prices_arr[-1] >= np.max(prices_arr[-20:-1])
            recent_rsi = indicators.rsi
            prev_rsi = self._calculate_rsi(prices_arr[:-5], 14)
            indicators.rsi_divergence = price_making_high and recent_rsi < prev_rsi
        
        # Participation
        indicators.participation_rate = self._calculate_participation(all_prices)
        
        return indicators
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_participation(self, all_prices: Dict[str, List[float]]) -> float:
        if len(all_prices) < 2:
            return 0.5
        directions = []
        for asset, prices in all_prices.items():
            if len(prices) >= 5:
                change = (prices[-1] / prices[-5] - 1)
                directions.append(1 if change > 0.005 else (-1 if change < -0.005 else 0))
        if not directions:
            return 0.5
        pos = sum(1 for d in directions if d > 0)
        neg = sum(1 for d in directions if d < 0)
        return max(pos, neg) / len(directions)
    
    def _detect_phase(
        self,
        indicators: PhaseIndicators,
        regime_state: str,
        crack_active: bool,
    ) -> Tuple[RegimePhase, float]:
        """Detect regime phase from indicators."""
        cfg = self.CONFIG
        
        # IGNITION score
        ignition_score = 0.0
        if abs(indicators.price_break_pct) >= cfg["ignition_price_break_min"]:
            ignition_score += 0.4
        if indicators.volume_ratio >= cfg["ignition_volume_min"]:
            ignition_score += 0.3
        if indicators.participation_rate <= cfg["ignition_participation_max"]:
            ignition_score += 0.2
        if crack_active:
            ignition_score += 0.1
        
        # EXPANSION score
        expansion_score = 0.0
        if indicators.adx >= cfg["expansion_adx_min"]:
            expansion_score += 0.3
        if indicators.adx_slope > 0:
            expansion_score += 0.2
        if indicators.volume_ratio >= cfg["expansion_volume_sustained_min"]:
            expansion_score += 0.2
        if indicators.participation_rate >= cfg["expansion_participation_min"]:
            expansion_score += 0.2
        if not indicators.rsi_divergence:
            expansion_score += 0.1
        
        # SATURATION score
        saturation_score = 0.0
        if abs(indicators.adx_slope) < cfg["saturation_adx_flat_threshold"]:
            saturation_score += 0.3
        if indicators.volume_trend < 0:
            saturation_score += 0.3
        if indicators.participation_rate >= cfg["saturation_participation_high"]:
            saturation_score += 0.3
        if indicators.rsi > 70 or indicators.rsi < 30:
            saturation_score += 0.1
        
        # EXHAUSTION score
        exhaustion_score = 0.0
        if indicators.volume_ratio < cfg["exhaustion_volume_collapse"]:
            exhaustion_score += 0.3
        if indicators.rsi_divergence:
            exhaustion_score += 0.4
        if indicators.reversal_candle:
            exhaustion_score += 0.2
        if indicators.adx_slope < -5:
            exhaustion_score += 0.1
        
        scores = {
            RegimePhase.IGNITION: ignition_score,
            RegimePhase.EXPANSION: expansion_score,
            RegimePhase.SATURATION: saturation_score,
            RegimePhase.EXHAUSTION: exhaustion_score,
        }
        
        best_phase = max(scores, key=scores.get)
        best_score = scores[best_phase]
        
        if best_score < 0.4:
            return RegimePhase.UNDEFINED, 0.0
        
        return best_phase, min(best_score, 1.0)
    
    def _detect_asset_phase(
        self,
        asset: str,
        prices: Dict[str, List[float]],
        volumes: Dict[str, List[float]],
    ) -> RegimePhase:
        """Detect phase for specific asset."""
        asset_prices = prices.get(asset, [])
        asset_volumes = volumes.get(asset, [])
        
        if len(asset_prices) < self.CONFIG["min_bars_required"]:
            return RegimePhase.UNDEFINED
        
        lookback = self.CONFIG.get(f"{asset}_lookback_bars", 20)
        asset_prices = asset_prices[-lookback:]
        asset_volumes = asset_volumes[-lookback:] if asset_volumes else []
        
        indicators = self._calculate_indicators(
            asset_prices, asset_volumes, {asset: asset_prices}
        )
        
        phase, _ = self._detect_phase(indicators, "UNKNOWN", False)
        return phase
    
    def _calculate_opportunity_density(
        self,
        phase: RegimePhase,
        crack_active: bool,
        agent_signals: Optional[Dict],
        indicators: PhaseIndicators,
    ) -> float:
        """Calculate opportunity quality score."""
        
        phase_score = PHASE_OPPORTUNITY_SCORES.get(phase, 0.5)
        crack_bonus = 0.2 if crack_active else 0.0
        
        alignment_score = 0.0
        if agent_signals:
            directions = []
            for key in ["quant_direction", "regime_direction"]:
                if key in agent_signals:
                    directions.append(agent_signals[key])
            if directions:
                same_sign = all(d >= 0 for d in directions) or all(d <= 0 for d in directions)
                alignment_score = 0.3 if same_sign else 0.0
        
        vol_factor = 1.0
        if 1.5 < indicators.volume_ratio < 3.0:
            vol_factor = 1.1
        elif indicators.volume_ratio > 4.0:
            vol_factor = 0.8
        
        density = (phase_score + crack_bonus + alignment_score) * vol_factor
        return max(0.0, min(1.0, density))
    
    def reset(self):
        self._last_phase = None
        self._phase_start_bar = 0
        self._current_bar = 0


# Singleton
_phase_detector: Optional[RegimePhaseDetector] = None

def get_phase_detector() -> RegimePhaseDetector:
    global _phase_detector
    if _phase_detector is None:
        _phase_detector = RegimePhaseDetector()
    return _phase_detector

def reset_phase_detector():
    global _phase_detector
    _phase_detector = None

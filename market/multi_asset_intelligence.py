"""
_DEPRECATED (T28): Not imported by main.py. Only used by strategic_coordinator.py
in a try/except fallback path. Contains conflicting max_drawdown values.

Original docstring:
Multi-Asset Two-Stage Intelligence System
==========================================
Trades BTC, ETH, SOL with coordinated strategy.

Architecture:
- Stage 1 (Heavy): Analyzes each asset + cross-asset correlations
- Stage 2 (Light): Monitors all assets, executes on triggers
- Portfolio: Manages allocation across assets

Risk Profile: AGGRESSIVE
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class RiskProfile(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    DEGEN = "degen"


@dataclass
class AssetConfig:
    """Configuration for a single asset."""
    symbol: str
    base: str  # BTC, ETH, SOL
    quote: str = "USDT"
    enabled: bool = True
    max_allocation_pct: float = 0.40  # Max 40% in single asset
    min_allocation_pct: float = 0.05  # Min 5% if trading
    timeframe: str = "1h"


@dataclass
class MultiAssetConfig:
    """Configuration for multi-asset system."""
    
    # Assets to trade - SOL focused portfolio
    assets: List[AssetConfig] = field(default_factory=lambda: [
        AssetConfig(symbol="SOL/USDT", base="SOL", max_allocation_pct=0.60),  # Primary: 50-60%
        AssetConfig(symbol="BTC/USDT", base="BTC", max_allocation_pct=0.30),  # Secondary: 20-30%
        AssetConfig(symbol="ETH/USDT", base="ETH", max_allocation_pct=0.30),  # Secondary: 20-30%
    ])
    
    # Exchange
    exchange: str = "kraken"
    
    # Risk profile
    risk_profile: RiskProfile = RiskProfile.AGGRESSIVE
    
    # Capital
    initial_capital: float = 100_000
    max_total_exposure: float = 1.00  # 100% - no cash buffer needed
    max_single_asset: float = 0.60    # Max 60% in SOL
    max_correlated_exposure: float = 0.50  # Max 50% in BTC+ETH together
    
    # Timing
    heavy_interval_hours: float = 1.0
    light_interval_seconds: float = 10.0
    
    # LLM
    heavy_model: str = "claude"  # gpt-4 or claude
    
    # Risk per trade based on profile
    @property
    def risk_per_trade(self) -> float:
        return {
            RiskProfile.CONSERVATIVE: 0.02,
            RiskProfile.MODERATE: 0.05,
            RiskProfile.AGGRESSIVE: 0.08,
            RiskProfile.DEGEN: 0.15,
        }[self.risk_profile]
    
    @property
    def max_drawdown(self) -> float:
        return {
            RiskProfile.CONSERVATIVE: 0.10,
            RiskProfile.MODERATE: 0.15,
            RiskProfile.AGGRESSIVE: 0.25,
            RiskProfile.DEGEN: 0.40,
        }[self.risk_profile]


# ============================================================================
# STRATEGIC STATE (Per Asset)
# ============================================================================

@dataclass
class AssetScenario:
    """Pre-computed scenario for an asset."""
    name: str
    asset: str
    condition: str
    trigger: Dict
    action: str  # LONG, SHORT, CLOSE, HOLD
    size_pct: float  # Percentage of portfolio
    stop_loss_pct: float
    take_profit_pct: float
    confidence: float
    reasoning: str


@dataclass
class AssetStrategicState:
    """Strategic state for a single asset."""
    asset: str
    timestamp: datetime
    
    # Market assessment
    regime: str
    regime_strength: float
    trend_direction: int  # 1, 0, -1
    
    # Levels
    support_levels: List[float]
    resistance_levels: List[float]
    current_price: float
    
    # Bias
    directional_bias: float  # -1 to 1
    conviction: float
    
    # Volatility
    volatility: float
    volatility_regime: str
    
    # Scenarios
    scenarios: List[AssetScenario]
    
    # Allocation
    recommended_allocation: float  # 0 to max_allocation
    
    # Valid until
    valid_until: datetime


@dataclass
class PortfolioStrategicState:
    """Combined strategic state for portfolio."""
    timestamp: datetime
    
    # Per-asset states
    asset_states: Dict[str, AssetStrategicState]
    
    # Cross-asset analysis
    market_regime: str  # risk_on, risk_off, mixed
    correlation_btc_eth: float
    correlation_btc_sol: float
    correlation_eth_sol: float
    
    # Portfolio recommendations
    total_recommended_exposure: float
    allocation_weights: Dict[str, float]  # BTC: 0.4, ETH: 0.3, SOL: 0.3
    
    # Risk
    portfolio_var_24h: float
    max_position_value: float
    
    # Valid until
    valid_until: datetime


# ============================================================================
# STAGE 1: HEAVY MODEL (MULTI-ASSET)
# ============================================================================

class MultiAssetHeavyModel:
    """
    Stage 1: Heavy strategic analysis for multiple assets.
    
    Analyzes:
    1. Each asset individually
    2. Cross-asset correlations
    3. Portfolio allocation
    """
    
    SYSTEM_PROMPT = """You are an expert crypto portfolio manager trading SOL, BTC, and ETH.
The trader is AGGRESSIVE with a SOL-focused strategy (50% SOL, 25% BTC, 25% ETH).
Deploy 100% of capital - no cash buffer needed. SOL is the primary position."""

    ANALYSIS_PROMPT = """
PORTFOLIO DATA:
{portfolio_data}

Current positions: {positions}

Provide analysis as JSON:
{{
    "market_regime": "risk_on|risk_off|mixed",
    
    "assets": {{
        "BTC": {{
            "regime": "bull_trend|bear_trend|sideways|volatile",
            "trend_direction": -1|0|1,
            "directional_bias": -1.0 to 1.0,
            "conviction": 0-1,
            "support_levels": [p1, p2],
            "resistance_levels": [p1, p2],
            "recommended_allocation": 0-0.5,
            "scenarios": [
                {{
                    "name": "scenario_name",
                    "trigger": {{"price_above": X}},
                    "action": "LONG|SHORT|CLOSE",
                    "size_pct": 0.1-0.5,
                    "stop_loss_pct": 0.02,
                    "take_profit_pct": 0.05,
                    "confidence": 0-1,
                    "reasoning": "why"
                }}
            ]
        }},
        "ETH": {{ ... same structure ... }},
        "SOL": {{ ... same structure ... }}
    }},
    
    "correlations": {{
        "btc_eth": 0-1,
        "btc_sol": 0-1,
        "eth_sol": 0-1
    }},
    
    "portfolio_allocation": {{
        "SOL": 0.50,
        "BTC": 0.25,
        "ETH": 0.25
    }},
    
    "total_exposure": 1.0,
    "key_insight": "most important thing"
}}

Be specific. Consider correlations - don't overweight correlated assets."""

    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.current_state: Optional[PortfolioStrategicState] = None
        self.llm_client = None
        self.llm_type = "disabled"
        self._llm_disable_until: Optional[datetime] = None
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM client with safe provider fallback."""
        if os.getenv("ENABLE_MULTI_ASSET_HEAVY_LLM", "false").lower() != "true":
            self.llm_client = None
            self.llm_type = "disabled"
            logger.info(
                "[MULTI_ASSET_HEAVY] External LLM disabled "
                "(set ENABLE_MULTI_ASSET_HEAVY_LLM=true to enable)"
            )
            return

        try:
            openai_key = os.getenv("OPENAI_API_KEY", "").strip()
            anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            provider = os.getenv("MULTI_ASSET_LLM_PROVIDER", "").strip().lower()
            heavy_model = (self.config.heavy_model or "").strip().lower()

            if not provider:
                if heavy_model in {"openai", "gpt-4", "gpt4", "gpt"}:
                    provider = "openai"
                elif heavy_model in {"anthropic", "claude"}:
                    provider = "anthropic"
                elif openai_key:
                    provider = "openai"
                elif anthropic_key:
                    provider = "anthropic"

            if provider == "anthropic" and not anthropic_key and openai_key:
                logger.warning(
                    "[MULTI_ASSET_HEAVY] ANTHROPIC_API_KEY missing, falling back to OpenAI"
                )
                provider = "openai"
            if provider == "openai" and not openai_key and anthropic_key:
                logger.warning(
                    "[MULTI_ASSET_HEAVY] OPENAI_API_KEY missing, falling back to Anthropic"
                )
                provider = "anthropic"

            if provider == "openai" and openai_key:
                import openai

                self.llm_client = openai.OpenAI(api_key=openai_key)
                self.llm_type = "openai"
                logger.info("[MULTI_ASSET_HEAVY] LLM provider initialized: openai")
            elif provider == "anthropic" and anthropic_key:
                import anthropic

                self.llm_client = anthropic.Anthropic(api_key=anthropic_key)
                self.llm_type = "anthropic"
                logger.info("[MULTI_ASSET_HEAVY] LLM provider initialized: anthropic")
            else:
                self.llm_client = None
                self.llm_type = "disabled"
                logger.warning(
                    "[MULTI_ASSET_HEAVY] No valid LLM provider configured; using fallback analysis"
                )
        except Exception as e:
            self.llm_client = None
            self.llm_type = "disabled"
            logger.warning(f"[MULTI_ASSET_HEAVY] LLM init failed: {e}")

    def _hard_disable_llm(self, reason: str):
        """Disable external LLM calls for this process after hard failures."""
        self.llm_client = None
        self.llm_type = "disabled"
        self._llm_disable_until = datetime.now(timezone.utc) + timedelta(hours=1)
        logger.warning(f"[MULTI_ASSET_HEAVY] LLM disabled for 1h: {reason}")

    def analyze(
        self,
        asset_data: Dict[str, pd.DataFrame],
        asset_features: Dict[str, Dict],
        positions: Dict = None
    ) -> PortfolioStrategicState:
        """Run heavy analysis for all assets."""
        
        logger.info("[MULTI_ASSET_HEAVY] Stage-1 analysis started")
        logger.debug("[MULTI_ASSET_HEAVY] Strategic analysis in progress")
        
        start = time.time()
        
        # Build portfolio summary
        portfolio_data = self._build_portfolio_summary(asset_data, asset_features)
        
        # Get LLM analysis
        if self.llm_client:
            analysis = self._get_llm_analysis(portfolio_data, positions or {})
        else:
            analysis = self._get_fallback_analysis(asset_features)
        
        # Build state
        state = self._build_state(analysis, asset_features)
        self.current_state = state
        
        elapsed = time.time() - start
        
        # Log results
        logger.info(f"[MULTI_ASSET_HEAVY] Analysis complete in {elapsed:.1f}s")
        logger.info(f"[MULTI_ASSET_HEAVY] Market regime: {state.market_regime}")
        for asset, alloc in state.allocation_weights.items():
            asset_state = state.asset_states.get(asset)
            if asset_state:
                logger.debug(f"[MULTI_ASSET_HEAVY] {asset}: {asset_state.regime} | bias {asset_state.directional_bias:+.2f} | alloc {alloc:.0%}")
        logger.info(f"[MULTI_ASSET_HEAVY] Total exposure target: {state.total_recommended_exposure:.0%}")
        
        return state
    
    def _build_portfolio_summary(
        self,
        asset_data: Dict[str, pd.DataFrame],
        asset_features: Dict[str, Dict]
    ) -> str:
        """Build portfolio summary for LLM."""
        
        lines = []
        
        for asset in ["BTC", "ETH", "SOL"]:
            features = asset_features.get(asset, {})
            if not features:
                continue
            
            lines.append(f"\n=== {asset} ===")
            lines.append(f"Price: ${features.get('close', 0):,.2f}")
            lines.append(f"24h Change: {features.get('roc_24', 0):.2f}%")
            lines.append(f"RSI: {features.get('rsi', 50):.1f}")
            lines.append(f"MACD Hist: {features.get('macd_histogram', 0):.4f}")
            lines.append(f"ADX: {features.get('adx', 20):.1f}")
            lines.append(f"ATR%: {features.get('atr_pct', 2):.2f}%")
            lines.append(f"EMA Align: {features.get('ema_alignment', 0)}")
            lines.append(f"Volume Ratio: {features.get('volume_ratio', 1):.2f}x")
        
        # Correlations (simplified)
        lines.append("\n=== CORRELATIONS (approx) ===")
        lines.append("BTC-ETH: High (0.85)")
        lines.append("BTC-SOL: Medium-High (0.75)")
        lines.append("ETH-SOL: Medium-High (0.70)")
        
        return "\n".join(lines)
    
    def _get_llm_analysis(self, portfolio_data: str, positions: Dict) -> Dict:
        """Get analysis from LLM."""
        if self.llm_client is None or self.llm_type == "disabled":
            return None
        if self._llm_disable_until and datetime.now(timezone.utc) < self._llm_disable_until:
            return None

        prompt = self.ANALYSIS_PROMPT.format(
            portfolio_data=portfolio_data,
            positions=json.dumps(positions)
        )
        
        try:
            if self.llm_type == "openai":
                response = self.llm_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=3000,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            else:
                response = self.llm_client.messages.create(
                    model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
                    max_tokens=3000,
                    system=self.SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}]
                )
                return json.loads(response.content[0].text)
        except Exception as e:
            err = str(e).lower()
            if self.llm_type == "anthropic" and (
                "404" in err or "not_found" in err or "model:" in err
            ):
                self._hard_disable_llm("anthropic model unavailable")
            else:
                self._llm_disable_until = datetime.now(timezone.utc) + timedelta(minutes=10)
                logger.warning(
                    f"[MULTI_ASSET_HEAVY] LLM call failed; fallback for 10m: {e}"
                )
            return None
    
    def _get_fallback_analysis(self, asset_features: Dict[str, Dict]) -> Dict:
        """Fallback analysis without LLM."""
        
        assets_analysis = {}
        
        for asset in ["BTC", "ETH", "SOL"]:
            f = asset_features.get(asset, {})
            
            # Determine regime
            adx = f.get('adx', 20)
            ema_align = f.get('ema_alignment', 0)
            
            if adx > 25 and ema_align == 1:
                regime = "bull_trend"
                bias = 0.6
            elif adx > 25 and ema_align == -1:
                regime = "bear_trend"
                bias = -0.6
            else:
                regime = "sideways"
                bias = 0.0
            
            close = f.get('close', 0)
            
            assets_analysis[asset] = {
                "regime": regime,
                "trend_direction": ema_align,
                "directional_bias": bias,
                "conviction": 0.5,
                "support_levels": [close * 0.95, close * 0.90],
                "resistance_levels": [close * 1.05, close * 1.10],
                "recommended_allocation": 0.35 if bias > 0 else 0.25,
                "scenarios": [
                    {
                        "name": f"{asset.lower()}_momentum",
                        "trigger": {"rsi_above": 55, "macd_positive": True},
                        "action": "LONG",
                        "size_pct": 0.15,
                        "stop_loss_pct": 0.03,
                        "take_profit_pct": 0.06,
                        "confidence": 0.5,
                        "reasoning": "Momentum continuation"
                    }
                ]
            }
        
        return {
            "market_regime": "mixed",
            "assets": assets_analysis,
            "correlations": {"btc_eth": 0.85, "btc_sol": 0.75, "eth_sol": 0.70},
            "portfolio_allocation": {"SOL": 0.50, "BTC": 0.25, "ETH": 0.25},
            "total_exposure": 1.0
        }
    
    def _build_state(
        self,
        analysis: Dict,
        asset_features: Dict[str, Dict]
    ) -> PortfolioStrategicState:
        """Build portfolio state from analysis."""
        
        if analysis is None:
            analysis = self._get_fallback_analysis(asset_features)
        
        asset_states = {}
        
        for asset in ["BTC", "ETH", "SOL"]:
            a = analysis.get("assets", {}).get(asset, {})
            f = asset_features.get(asset, {})
            
            scenarios = []
            for s in a.get("scenarios", []):
                scenarios.append(AssetScenario(
                    name=s.get("name", "unnamed"),
                    asset=asset,
                    condition=s.get("condition", ""),
                    trigger=s.get("trigger", {}),
                    action=s.get("action", "HOLD"),
                    size_pct=s.get("size_pct", 0.1),
                    stop_loss_pct=s.get("stop_loss_pct", 0.03),
                    take_profit_pct=s.get("take_profit_pct", 0.05),
                    confidence=s.get("confidence", 0.5),
                    reasoning=s.get("reasoning", "")
                ))
            
            asset_states[asset] = AssetStrategicState(
                asset=asset,
                timestamp=datetime.now(),
                regime=a.get("regime", "unknown"),
                regime_strength=0.5,
                trend_direction=a.get("trend_direction", 0),
                support_levels=a.get("support_levels", []),
                resistance_levels=a.get("resistance_levels", []),
                current_price=f.get("close", 0),
                directional_bias=a.get("directional_bias", 0),
                conviction=a.get("conviction", 0.5),
                volatility=f.get("atr_pct", 2) / 100,
                volatility_regime="normal",
                scenarios=scenarios,
                recommended_allocation=a.get("recommended_allocation", 0.25),
                valid_until=datetime.now() + timedelta(hours=self.config.heavy_interval_hours)
            )
        
        correlations = analysis.get("correlations", {})
        allocation = analysis.get("portfolio_allocation", {})
        
        return PortfolioStrategicState(
            timestamp=datetime.now(),
            asset_states=asset_states,
            market_regime=analysis.get("market_regime", "mixed"),
            correlation_btc_eth=correlations.get("btc_eth", 0.85),
            correlation_btc_sol=correlations.get("btc_sol", 0.75),
            correlation_eth_sol=correlations.get("eth_sol", 0.70),
            total_recommended_exposure=analysis.get("total_exposure", 0.5),
            allocation_weights={k: v for k, v in allocation.items() if k != "cash"},
            portfolio_var_24h=0.05,
            max_position_value=self.config.initial_capital * self.config.max_single_asset,
            valid_until=datetime.now() + timedelta(hours=self.config.heavy_interval_hours)
        )


# ============================================================================
# STAGE 2: LIGHT MODEL (MULTI-ASSET)
# ============================================================================

@dataclass
class MultiAssetSignal:
    """Signal for multi-asset execution."""
    timestamp: datetime
    asset: str
    action: str
    size_pct: float  # Of portfolio
    entry_price: float
    stop_loss: float
    take_profit: float
    triggered_scenario: str
    trigger_reason: str
    confidence: float
    urgency: str


class MultiAssetLightModel:
    """
    Stage 2: Light model monitoring multiple assets.
    
    Checks all asset triggers in parallel (GPU).
    """
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.portfolio_state: Optional[PortfolioStrategicState] = None
        self.signal_history: deque = deque(maxlen=1000)
    
    def update_strategy(self, state: PortfolioStrategicState):
        """Update with latest portfolio state."""
        self.portfolio_state = state
        total_scenarios = sum(len(s.scenarios) for s in state.asset_states.values())
        logger.debug(f"[MULTI_ASSET_LIGHT] Updated: {total_scenarios} scenarios across {len(state.asset_states)} assets")
    
    def evaluate(
        self,
        prices: Dict[str, float],
        features: Dict[str, Dict]
    ) -> List[MultiAssetSignal]:
        """
        Evaluate all assets against triggers.
        
        Returns list of signals (may be multiple if several trigger).
        """
        if self.portfolio_state is None:
            return []
        
        signals = []
        
        for asset, asset_state in self.portfolio_state.asset_states.items():
            price = prices.get(asset, 0)
            asset_features = features.get(asset, {})
            
            if not price:
                continue
            
            # Check each scenario
            for scenario in asset_state.scenarios:
                if self._check_trigger(scenario.trigger, price, asset_features):
                    signal = self._create_signal(scenario, price, asset_features)
                    signals.append(signal)
            
            # Check urgent conditions
            urgent = self._check_urgent(asset, price, asset_features, asset_state)
            if urgent:
                signals.append(urgent)
        
        # Store signals
        for s in signals:
            self.signal_history.append(s)
        
        return signals
    
    def _check_trigger(
        self,
        trigger: Dict,
        price: float,
        features: Dict
    ) -> bool:
        """Check if trigger conditions met."""
        
        if 'price_above' in trigger and price <= trigger['price_above']:
            return False
        if 'price_below' in trigger and price >= trigger['price_below']:
            return False
        
        rsi = features.get('rsi', 50)
        if 'rsi_above' in trigger and rsi <= trigger['rsi_above']:
            return False
        if 'rsi_below' in trigger and rsi >= trigger['rsi_below']:
            return False
        
        macd = features.get('macd_histogram', 0)
        if 'macd_positive' in trigger and trigger['macd_positive'] and macd <= 0:
            return False
        if 'macd_negative' in trigger and trigger['macd_negative'] and macd >= 0:
            return False
        
        vol_ratio = features.get('volume_ratio', 1)
        if 'volume_above' in trigger and vol_ratio <= trigger['volume_above']:
            return False
        
        return True
    
    def _check_urgent(
        self,
        asset: str,
        price: float,
        features: Dict,
        asset_state: AssetStrategicState
    ) -> Optional[MultiAssetSignal]:
        """Check for urgent conditions."""
        
        roc_1h = features.get('roc_1', 0)
        vol_ratio = features.get('volume_ratio', 1)
        
        # Strong move with volume in trend direction
        if (abs(roc_1h) > 2.0 and vol_ratio > 2.0):
            
            if roc_1h > 0 and asset_state.directional_bias > 0.3:
                return MultiAssetSignal(
                    timestamp=datetime.now(),
                    asset=asset,
                    action="LONG",
                    size_pct=asset_state.recommended_allocation * 1.2,
                    entry_price=price,
                    stop_loss=price * 0.97,
                    take_profit=price * 1.06,
                    triggered_scenario="momentum_surge",
                    trigger_reason=f"{asset} +{roc_1h:.1f}% with {vol_ratio:.1f}x volume",
                    confidence=0.7,
                    urgency="high"
                )
        
        return None
    
    def _create_signal(
        self,
        scenario: AssetScenario,
        price: float,
        features: Dict
    ) -> MultiAssetSignal:
        """Create signal from triggered scenario."""
        
        if scenario.action == "LONG":
            stop = price * (1 - scenario.stop_loss_pct)
            tp = price * (1 + scenario.take_profit_pct)
        elif scenario.action == "SHORT":
            stop = price * (1 + scenario.stop_loss_pct)
            tp = price * (1 - scenario.take_profit_pct)
        else:
            stop = tp = price
        
        return MultiAssetSignal(
            timestamp=datetime.now(),
            asset=scenario.asset,
            action=scenario.action,
            size_pct=scenario.size_pct,
            entry_price=price,
            stop_loss=stop,
            take_profit=tp,
            triggered_scenario=scenario.name,
            trigger_reason=scenario.reasoning,
            confidence=scenario.confidence,
            urgency="high" if scenario.confidence > 0.7 else "medium"
        )


# ============================================================================
# PORTFOLIO MANAGER
# ============================================================================

class PortfolioManager:
    """
    Manages portfolio-level risk and allocation.
    
    Ensures:
    - Total exposure within limits
    - Single asset exposure within limits
    - Correlated exposure within limits
    """
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        self.capital = config.initial_capital
        self.positions: Dict[str, Dict] = {}  # asset -> position info
        self.equity_history = [config.initial_capital]
    
    def validate_signal(
        self,
        signal: MultiAssetSignal,
        portfolio_state: PortfolioStrategicState
    ) -> Tuple[bool, str, Dict]:
        """
        Validate signal against portfolio constraints.
        
        Returns: (valid, reason, adjusted_sizing)
        """
        
        # Calculate current exposure
        current_exposure = self._get_current_exposure()
        asset_exposure = self._get_asset_exposure(signal.asset)
        
        # Proposed position value
        proposed_value = self.capital * signal.size_pct
        
        # Check total exposure
        new_total = current_exposure + signal.size_pct
        if new_total > self.config.max_total_exposure:
            available = self.config.max_total_exposure - current_exposure
            if available < 0.05:
                return False, "Max portfolio exposure reached", {}
            signal.size_pct = available
        
        # Check single asset exposure
        new_asset = asset_exposure + signal.size_pct
        if new_asset > self.config.max_single_asset:
            available = self.config.max_single_asset - asset_exposure
            if available < 0.05:
                return False, f"Max {signal.asset} exposure reached", {}
            signal.size_pct = min(signal.size_pct, available)
        
        # Check correlated exposure (BTC + ETH together)
        if signal.asset in ["BTC", "ETH"]:
            corr_exposure = (
                self._get_asset_exposure("BTC") +
                self._get_asset_exposure("ETH") +
                signal.size_pct
            )
            if corr_exposure > self.config.max_correlated_exposure:
                return False, "High BTC-ETH correlation, reduce exposure", {}
        
        # Calculate final sizing
        position_value = self.capital * signal.size_pct
        size_units = position_value / signal.entry_price
        risk_amount = size_units * abs(signal.entry_price - signal.stop_loss)
        risk_pct = risk_amount / self.capital
        
        sizing = {
            'size_pct': signal.size_pct,
            'position_value': position_value,
            'size_units': size_units,
            'risk_amount': risk_amount,
            'risk_pct': risk_pct,
            'entry': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit
        }
        
        return True, "OK", sizing
    
    def _get_current_exposure(self) -> float:
        """Get current total exposure as fraction of capital."""
        total = sum(p.get('value', 0) for p in self.positions.values())
        return total / self.capital
    
    def _get_asset_exposure(self, asset: str) -> float:
        """Get exposure to specific asset."""
        pos = self.positions.get(asset, {})
        return pos.get('value', 0) / self.capital
    
    def update_position(self, asset: str, position: Dict):
        """Update position after trade."""
        self.positions[asset] = position
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary."""
        return {
            'capital': self.capital,
            'total_exposure': self._get_current_exposure(),
            'positions': self.positions,
            'by_asset': {
                asset: self._get_asset_exposure(asset)
                for asset in ["BTC", "ETH", "SOL"]
            }
        }


# ============================================================================
# FACTORY
# ============================================================================

def create_multi_asset_config(
    risk_profile: str = "aggressive",
    capital: float = 100_000
) -> MultiAssetConfig:
    """Create multi-asset configuration."""
    
    profile_map = {
        "conservative": RiskProfile.CONSERVATIVE,
        "moderate": RiskProfile.MODERATE,
        "aggressive": RiskProfile.AGGRESSIVE,
        "degen": RiskProfile.DEGEN
    }
    
    return MultiAssetConfig(
        risk_profile=profile_map.get(risk_profile, RiskProfile.AGGRESSIVE),
        initial_capital=capital
    )

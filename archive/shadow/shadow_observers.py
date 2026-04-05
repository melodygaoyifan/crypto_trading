"""
================================================================================
HMATS v6.1.4 - Shadow Observer Layer
================================================================================
Purpose: 将中优先级模块以 Shadow Mode 集成，只允许观察和记录

集成的 Shadow 模块:
    1. LeadLagEngine - L0 观察级 (Observation-only)
    2. PassiveAggressive - L1 执行观察级 (Execution-quality observer)
    3. SolDexMonitor - L1 特征源 (Feature-only)

核心约束:
    OFF 不能触发 entry
    OFF 不能 veto 任何交易
    OFF 不能改变仓位、方向、杠杆
    OFF 不能进入 OPPORTUNITY 判定
    OFF 不能影响 TradeGate / RiskController
    ON 只允许：计算 -> 记录 -> 展示

接线位置:
    Market Data -> Decision/Execution 完成后 -> [Shadow Observers] -> ShadowLedger/RuntimeState

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ShadowEventType(Enum):
    """Shadow 模块事件类型 - 用于 ShadowLedger"""
    LEAD_LAG_OBSERVATION = "LEAD_LAG_OBSERVATION"
    EXECUTION_QUALITY_ANALYSIS = "EXECUTION_QUALITY_ANALYSIS"
    DEX_LIQUIDITY_OBSERVATION = "DEX_LIQUIDITY_OBSERVATION"


@dataclass
class ShadowObservation:
    """Shadow 观察结果 - 只读数据结构"""
    event_type: ShadowEventType
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    # 明确标记: 无决策权
    has_decision_authority: bool = field(default=False, init=False)
    can_trigger_entry: bool = field(default=False, init=False)
    can_veto_trade: bool = field(default=False, init=False)
    can_affect_position: bool = field(default=False, init=False)
    
    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data,
            "metadata": {
                "has_decision_authority": self.has_decision_authority,
                "can_trigger_entry": self.can_trigger_entry,
                "can_veto_trade": self.can_veto_trade,
                "can_affect_position": self.can_affect_position,
                "mode": "SHADOW_ONLY"
            }
        }


# =============================================================================
# LEAD-LAG SHADOW OBSERVER
# =============================================================================

class LeadLagShadowObserver:
    """
    LeadLag Engine Shadow Observer
    
    角色: L0 · 观察级 (Observation-only)
    
    允许的行为:
        - 连接 Binance -> Kraken (fallback if unavailable)
        - 计算 lead_lag_signal, confidence, latency_ms
        - 输出到 ShadowLedger 和 RuntimeState
    
    禁止的行为:
        OFF 不允许生成 trade signal
        OFF 不允许写入 strategy intent
        OFF 不允许影响 TwoStage / AuthorityMatrix
    """
    
    def __init__(self):
        self._engine = None
        self._enabled = False
        self._shadow_ledger = None
        self._runtime_state = None
        self._observation_count = 0
        
        self._init_engine()
        self._init_integrations()
        
        logger.info("[SHADOW] LeadLagShadowObserver initialized (OBSERVATION-ONLY)")
    
    def _init_engine(self):
        """Initialize the underlying engine in observation mode."""
        try:
            from market.lead_lag_engine import LeadLagAlphaEngine
            self._engine = LeadLagAlphaEngine()
            self._enabled = True
            logger.info("[SHADOW] LeadLagAlphaEngine loaded")
        except ImportError as e:
            logger.warning(f"[SHADOW] LeadLagAlphaEngine not available: {e}")
            self._enabled = False
        except Exception as e:
            logger.warning(f"[SHADOW] LeadLagAlphaEngine init failed: {e}")
            self._enabled = False
    
    def _init_integrations(self):
        """Initialize ShadowLedger and RuntimeState connections."""
        try:
            from data_mgmt.shadow_ledger_manager import get_shadow_ledger
            self._shadow_ledger = get_shadow_ledger()
        except (ImportError, Exception):
            pass
        
        try:
            from core.runtime_state import get_runtime_state
            self._runtime_state = get_runtime_state()
        except (ImportError, Exception):
            pass
    
    def observe(self, market_data: Dict = None) -> Optional[ShadowObservation]:
        """
        Observe and record lead-lag signals.
        
        [WARN]️ This method is OBSERVATION-ONLY.
        It does NOT affect any trading decisions.
        
        Args:
            market_data: Current market data snapshot
            
        Returns:
            ShadowObservation with lead-lag data (no decision authority)
        """
        if not self._enabled:
            return None
        
        try:
            # Get observation from engine
            observation_data = self._get_observation(market_data)
            
            # Create shadow observation
            observation = ShadowObservation(
                event_type=ShadowEventType.LEAD_LAG_OBSERVATION,
                source="LeadLagShadowObserver",
                data=observation_data,
            )
            
            self._observation_count += 1
            
            # Write to ShadowLedger
            self._write_to_shadow_ledger(observation)
            
            # Write to RuntimeState
            self._update_runtime_state(observation_data)
            
            return observation
            
        except Exception as e:
            logger.debug(f"[SHADOW] LeadLag observation failed: {e}")
            return None
    
    def _get_observation(self, market_data: Dict = None) -> Dict:
        """Get observation data from engine."""
        if self._engine is None:
            return self._get_mock_observation()
        
        try:
            # Try to get real data from engine
            if hasattr(self._engine, 'get_signal'):
                signal = self._engine.get_signal()
                return {
                    "lead_lag_signal": signal.get("direction", 0.0),
                    "confidence": signal.get("confidence", 0.0),
                    "latency_ms": signal.get("latency_ms", 0.0),
                    "binance_lead": signal.get("binance_lead", False),
                    "kraken_lag_ms": signal.get("kraken_lag_ms", 0.0),
                    "source": "LeadLagEngine",
                }
            elif hasattr(self._engine, 'get_state'):
                state = self._engine.get_state()
                return {
                    "lead_lag_signal": state.get("signal", 0.0),
                    "confidence": state.get("confidence", 0.0),
                    "latency_ms": state.get("latency", 0.0),
                    "source": "LeadLagEngine",
                }
            else:
                return self._get_mock_observation()
        except Exception:
            return self._get_mock_observation()
    
    def _get_mock_observation(self) -> Dict:
        """Get mock observation when engine unavailable."""
        return {
            "lead_lag_signal": 0.0,
            "confidence": 0.0,
            "latency_ms": 0.0,
            "binance_lead": False,
            "kraken_lag_ms": 0.0,
            "source": "MOCK",
            "note": "Engine unavailable, using mock data"
        }
    
    def _write_to_shadow_ledger(self, observation: ShadowObservation):
        """Write observation to ShadowLedger."""
        if self._shadow_ledger:
            try:
                self._shadow_ledger.log_event(
                    event_type=ShadowEventType.LEAD_LAG_OBSERVATION.value,
                    source="LeadLagShadowObserver",
                    data=observation.to_dict(),
                )
            except Exception as e:
                logger.debug(f"[SHADOW] ShadowLedger write failed: {e}")
    
    def _update_runtime_state(self, data: Dict):
        """Update RuntimeState for dashboard visibility."""
        if self._runtime_state:
            try:
                self._runtime_state.set("lead_lag_shadow", {
                    "enabled": True,
                    "mode": "SHADOW_ONLY",
                    "has_decision_authority": False,
                    "observations": self._observation_count,
                    "latest": data,
                    "updated_at": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.debug(f"[SHADOW] RuntimeState update failed: {e}")
    
    def get_status(self) -> Dict:
        """Get observer status."""
        return {
            "enabled": self._enabled,
            "mode": "SHADOW_ONLY",
            "has_decision_authority": False,
            "observations": self._observation_count,
        }


# =============================================================================
# PASSIVE-AGGRESSIVE SHADOW OBSERVER
# =============================================================================

class PassiveAggressiveShadowObserver:
    """
    Passive-Aggressive Execution Shadow Observer
    
    角色: L1 · 执行观察级 (Execution-quality observer)
    
    允许的行为:
        - 在执行完成后计算: maker可能性, taker必要性, 理论滑点改善, fee差异
        - 输出到 ShadowLedger 和 RuntimeState
    
    禁止的行为:
        OFF 不允许改变真实下单方式
        OFF 不允许 override execution engine
        OFF 不允许插入到下单路径
    """
    
    def __init__(self):
        self._analyzer = None
        self._enabled = False
        self._shadow_ledger = None
        self._runtime_state = None
        self._analysis_count = 0
        self._total_theoretical_savings_bps = 0.0
        
        self._init_analyzer()
        self._init_integrations()
        
        logger.info("[SHADOW] PassiveAggressiveShadowObserver initialized (POST-TRADE ANALYSIS ONLY)")
    
    def _init_analyzer(self):
        """Initialize the underlying analyzer."""
        try:
            from execution.passive_aggressive import PassiveAggressiveExecutor
            self._analyzer = PassiveAggressiveExecutor()
            self._enabled = True
            logger.info("[SHADOW] PassiveAggressiveExecutor loaded")
        except ImportError as e:
            logger.warning(f"[SHADOW] PassiveAggressiveExecutor not available: {e}")
            self._enabled = False
        except Exception as e:
            logger.warning(f"[SHADOW] PassiveAggressiveExecutor init failed: {e}")
            self._enabled = False
    
    def _init_integrations(self):
        """Initialize ShadowLedger and RuntimeState connections."""
        try:
            from data_mgmt.shadow_ledger_manager import get_shadow_ledger
            self._shadow_ledger = get_shadow_ledger()
        except (ImportError, Exception):
            pass
        
        try:
            from core.runtime_state import get_runtime_state
            self._runtime_state = get_runtime_state()
        except (ImportError, Exception):
            pass
    
    def analyze_execution(
        self,
        order_result: Dict,
        market_state: Dict = None,
    ) -> Optional[ShadowObservation]:
        """
        Analyze execution quality AFTER trade is complete.
        
        [WARN]️ This method is POST-TRADE ANALYSIS ONLY.
        It does NOT change how orders are executed.
        Real orders follow original execution logic.
        
        Args:
            order_result: Completed order result
            market_state: Market state at execution time
            
        Returns:
            ShadowObservation with execution quality analysis
        """
        if not self._enabled and self._analyzer is None:
            # Use mock analysis
            pass
        
        try:
            analysis_data = self._analyze(order_result, market_state)
            
            observation = ShadowObservation(
                event_type=ShadowEventType.EXECUTION_QUALITY_ANALYSIS,
                source="PassiveAggressiveShadowObserver",
                data=analysis_data,
            )
            
            self._analysis_count += 1
            self._total_theoretical_savings_bps += analysis_data.get("theoretical_savings_bps", 0.0)
            
            # Write to ShadowLedger
            self._write_to_shadow_ledger(observation)
            
            # Write to RuntimeState
            self._update_runtime_state(analysis_data)
            
            return observation
            
        except Exception as e:
            logger.debug(f"[SHADOW] Execution analysis failed: {e}")
            return None
    
    def _analyze(self, order_result: Dict, market_state: Dict = None) -> Dict:
        """Analyze execution quality."""
        # Extract order info
        side = order_result.get("side", "unknown")
        fill_price = order_result.get("avg_price", 0.0)
        size = order_result.get("filled_size", 0.0)
        was_taker = order_result.get("is_taker", True)
        
        # Market context
        best_bid = market_state.get("best_bid", fill_price) if market_state else fill_price
        best_ask = market_state.get("best_ask", fill_price) if market_state else fill_price
        spread_bps = ((best_ask - best_bid) / best_bid * 10000) if best_bid > 0 else 0
        
        # Fee analysis (Kraken fees)
        TAKER_FEE_BPS = 26.0
        MAKER_FEE_BPS = 16.0
        
        actual_fee_bps = TAKER_FEE_BPS if was_taker else MAKER_FEE_BPS
        
        # Could maker analysis
        could_maker = spread_bps > 5.0  # If spread > 5bps, could have tried maker
        
        # Theoretical savings
        if was_taker and could_maker:
            theoretical_savings_bps = TAKER_FEE_BPS - MAKER_FEE_BPS  # 10 bps
        else:
            theoretical_savings_bps = 0.0
        
        # Slippage analysis
        if side == "buy":
            expected_price = best_ask
            slippage_bps = ((fill_price - expected_price) / expected_price * 10000) if expected_price > 0 else 0
        else:
            expected_price = best_bid
            slippage_bps = ((expected_price - fill_price) / expected_price * 10000) if expected_price > 0 else 0
        
        return {
            "order_id": order_result.get("order_id", "unknown"),
            "side": side,
            "size": size,
            "fill_price": fill_price,
            "was_taker": was_taker,
            "could_maker": could_maker,
            "spread_bps": spread_bps,
            "actual_fee_bps": actual_fee_bps,
            "theoretical_fee_bps": MAKER_FEE_BPS if could_maker else actual_fee_bps,
            "theoretical_savings_bps": theoretical_savings_bps,
            "slippage_bps": slippage_bps,
            "recommendation": "Consider passive order" if could_maker and was_taker else "Execution OK",
            "analysis_mode": "SHADOW_POST_TRADE",
            "affects_execution": False,
        }
    
    def _write_to_shadow_ledger(self, observation: ShadowObservation):
        """Write observation to ShadowLedger."""
        if self._shadow_ledger:
            try:
                self._shadow_ledger.log_event(
                    event_type=ShadowEventType.EXECUTION_QUALITY_ANALYSIS.value,
                    source="PassiveAggressiveShadowObserver",
                    data=observation.to_dict(),
                )
            except Exception as e:
                logger.debug(f"[SHADOW] ShadowLedger write failed: {e}")
    
    def _update_runtime_state(self, data: Dict):
        """Update RuntimeState for dashboard visibility."""
        if self._runtime_state:
            try:
                self._runtime_state.set("execution_quality", {
                    "enabled": True,
                    "mode": "SHADOW_POST_TRADE_ANALYSIS",
                    "affects_execution": False,
                    "has_decision_authority": False,
                    "analyses": self._analysis_count,
                    "total_theoretical_savings_bps": self._total_theoretical_savings_bps,
                    "latest": data,
                    "updated_at": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.debug(f"[SHADOW] RuntimeState update failed: {e}")
    
    def get_status(self) -> Dict:
        """Get observer status."""
        return {
            "enabled": self._enabled or True,  # Mock always available
            "mode": "SHADOW_POST_TRADE_ANALYSIS",
            "affects_execution": False,
            "has_decision_authority": False,
            "analyses": self._analysis_count,
            "total_theoretical_savings_bps": self._total_theoretical_savings_bps,
        }


# =============================================================================
# SOL DEX MONITOR SHADOW OBSERVER
# =============================================================================

class SolDexShadowObserver:
    """
    Solana DEX Monitor Shadow Observer
    
    角色: L1 · 特征源 (Feature-only)
    
    允许的行为:
        - 采集 Jupiter / Raydium / Orca 流动性
        - 计算 dex_flow_score, liquidity_imbalance
        - 输出 feature 到 ShadowLedger 和 RuntimeState
        - 作为 OnchainSentimentFusion 的附加 feature (只读)
    
    禁止的行为:
        OFF 不允许生成交易信号
        OFF 不允许进入 OPPORTUNITY
        OFF 不允许直接影响策略权重
    """
    
    def __init__(self):
        self._monitor = None
        self._enabled = False
        self._shadow_ledger = None
        self._runtime_state = None
        self._observation_count = 0
        
        self._init_monitor()
        self._init_integrations()
        
        logger.info("[SHADOW] SolDexShadowObserver initialized (FEATURE-ONLY)")
    
    def _init_monitor(self):
        """Initialize the underlying monitor."""
        try:
            from liquidity.sol_dex_monitor import SolanaDEXMonitor
            self._monitor = SolanaDEXMonitor()
            self._enabled = True
            logger.info("[SHADOW] SolanaDEXMonitor loaded")
        except ImportError as e:
            logger.warning(f"[SHADOW] SolanaDEXMonitor not available: {e}")
            self._enabled = False
        except Exception as e:
            logger.warning(f"[SHADOW] SolanaDEXMonitor init failed: {e}")
            self._enabled = False
    
    def _init_integrations(self):
        """Initialize ShadowLedger and RuntimeState connections."""
        try:
            from data_mgmt.shadow_ledger_manager import get_shadow_ledger
            self._shadow_ledger = get_shadow_ledger()
        except (ImportError, Exception):
            pass
        
        try:
            from core.runtime_state import get_runtime_state
            self._runtime_state = get_runtime_state()
        except (ImportError, Exception):
            pass
    
    def observe(self) -> Optional[ShadowObservation]:
        """
        Observe DEX liquidity and record features.
        
        [WARN]️ This method outputs FEATURES ONLY.
        It does NOT generate trading signals.
        It does NOT trigger OPPORTUNITY mode.
        
        Returns:
            ShadowObservation with DEX liquidity features
        """
        try:
            feature_data = self._get_features()
            
            observation = ShadowObservation(
                event_type=ShadowEventType.DEX_LIQUIDITY_OBSERVATION,
                source="SolDexShadowObserver",
                data=feature_data,
            )
            
            self._observation_count += 1
            
            # Write to ShadowLedger
            self._write_to_shadow_ledger(observation)
            
            # Write to RuntimeState
            self._update_runtime_state(feature_data)
            
            return observation
            
        except Exception as e:
            logger.debug(f"[SHADOW] DEX observation failed: {e}")
            return None
    
    def _get_features(self) -> Dict:
        """Get DEX liquidity features."""
        if self._monitor is None:
            return self._get_mock_features()
        
        try:
            if hasattr(self._monitor, 'get_liquidity_snapshot'):
                snapshot = self._monitor.get_liquidity_snapshot()
                return {
                    "dex_flow_score": snapshot.get("flow_score", 0.0),
                    "liquidity_imbalance": snapshot.get("imbalance", 0.0),
                    "jupiter_depth": snapshot.get("jupiter_depth", 0.0),
                    "raydium_depth": snapshot.get("raydium_depth", 0.0),
                    "orca_depth": snapshot.get("orca_depth", 0.0),
                    "dex_cex_arb_bps": snapshot.get("arb_bps", 0.0),
                    "mev_pressure": snapshot.get("mev_pressure", 0.0),
                    "source": "SolanaDEXMonitor",
                    "is_feature_only": True,
                    "generates_signal": False,
                }
            elif hasattr(self._monitor, 'get_state'):
                state = self._monitor.get_state()
                return {
                    "dex_flow_score": state.get("flow_score", 0.0),
                    "liquidity_imbalance": state.get("imbalance", 0.0),
                    "source": "SolanaDEXMonitor",
                    "is_feature_only": True,
                    "generates_signal": False,
                }
            else:
                return self._get_mock_features()
        except Exception:
            return self._get_mock_features()
    
    def _get_mock_features(self) -> Dict:
        """Get mock features when monitor unavailable."""
        return {
            "dex_flow_score": 0.0,
            "liquidity_imbalance": 0.0,
            "jupiter_depth": 0.0,
            "raydium_depth": 0.0,
            "orca_depth": 0.0,
            "dex_cex_arb_bps": 0.0,
            "mev_pressure": 0.0,
            "source": "MOCK",
            "is_feature_only": True,
            "generates_signal": False,
            "note": "Monitor unavailable, using mock data"
        }
    
    def get_features_for_fusion(self) -> Dict:
        """
        Get features for OnchainSentimentFusion.
        
        [WARN]️ This returns READ-ONLY features.
        They can be used as additional context but
        DO NOT affect fusion weights or decisions.
        """
        features = self._get_features()
        features["read_only"] = True
        features["affects_fusion_weights"] = False
        return features
    
    def _write_to_shadow_ledger(self, observation: ShadowObservation):
        """Write observation to ShadowLedger."""
        if self._shadow_ledger:
            try:
                self._shadow_ledger.log_event(
                    event_type=ShadowEventType.DEX_LIQUIDITY_OBSERVATION.value,
                    source="SolDexShadowObserver",
                    data=observation.to_dict(),
                )
            except Exception as e:
                logger.debug(f"[SHADOW] ShadowLedger write failed: {e}")
    
    def _update_runtime_state(self, data: Dict):
        """Update RuntimeState for dashboard visibility."""
        if self._runtime_state:
            try:
                self._runtime_state.set("onchain_features", {
                    "enabled": True,
                    "mode": "SHADOW_FEATURE_ONLY",
                    "generates_signal": False,
                    "has_decision_authority": False,
                    "observations": self._observation_count,
                    "dex_liquidity": data,
                    "updated_at": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.debug(f"[SHADOW] RuntimeState update failed: {e}")
    
    def get_status(self) -> Dict:
        """Get observer status."""
        return {
            "enabled": self._enabled or True,  # Mock always available
            "mode": "SHADOW_FEATURE_ONLY",
            "generates_signal": False,
            "has_decision_authority": False,
            "observations": self._observation_count,
        }


# =============================================================================
# SHADOW OBSERVER MANAGER
# =============================================================================

class ShadowObserverManager:
    """
    Unified manager for all Shadow observers.
    
    Ensures all observers are called AFTER decision/execution,
    not before or during.
    """
    
    def __init__(self):
        self._lead_lag = None
        self._passive_aggressive = None
        self._sol_dex = None
        
        self._enabled = False
        self._init_observers()
        
        logger.info("[SHADOW] ShadowObserverManager initialized")
    
    def _init_observers(self):
        """Initialize all shadow observers based on flags."""
        try:
            from configs.sota_flags import get_sota_flags
            flags = get_sota_flags()
            
            if getattr(flags, 'ENABLE_LEAD_LAG_SHADOW', True):
                self._lead_lag = LeadLagShadowObserver()
            
            if getattr(flags, 'ENABLE_PASSIVE_AGGRESSIVE_SHADOW', True):
                self._passive_aggressive = PassiveAggressiveShadowObserver()
            
            if getattr(flags, 'ENABLE_SOLDEX_MONITOR_SHADOW', True):
                self._sol_dex = SolDexShadowObserver()
            
            self._enabled = True
        except ImportError:
            # No flags, enable all by default
            self._lead_lag = LeadLagShadowObserver()
            self._passive_aggressive = PassiveAggressiveShadowObserver()
            self._sol_dex = SolDexShadowObserver()
            self._enabled = True
    
    def observe_post_decision(self, market_data: Dict = None):
        """
        Run all shadow observations AFTER decision is made.
        
        [WARN]️ Must be called AFTER TradeGate, RiskController, and execution.
        Must NOT affect any trading decisions.
        """
        if not self._enabled:
            return
        
        # Lead-Lag observation
        if self._lead_lag:
            try:
                self._lead_lag.observe(market_data)
            except Exception as e:
                logger.debug(f"[SHADOW] LeadLag observation error: {e}")
        
        # DEX observation
        if self._sol_dex:
            try:
                self._sol_dex.observe()
            except Exception as e:
                logger.debug(f"[SHADOW] DEX observation error: {e}")
    
    def analyze_post_execution(self, order_result: Dict, market_state: Dict = None):
        """
        Analyze execution AFTER order is complete.
        
        [WARN]️ Must be called AFTER order is filled.
        Must NOT change how future orders are executed.
        """
        if not self._enabled:
            return
        
        if self._passive_aggressive:
            try:
                self._passive_aggressive.analyze_execution(order_result, market_state)
            except Exception as e:
                logger.debug(f"[SHADOW] Execution analysis error: {e}")
    
    def get_status(self) -> Dict:
        """Get status of all shadow observers."""
        return {
            "enabled": self._enabled,
            "mode": "SHADOW_ONLY",
            "affects_trading": False,
            "observers": {
                "lead_lag": self._lead_lag.get_status() if self._lead_lag else {"enabled": False},
                "passive_aggressive": self._passive_aggressive.get_status() if self._passive_aggressive else {"enabled": False},
                "sol_dex": self._sol_dex.get_status() if self._sol_dex else {"enabled": False},
            }
        }
    
    def get_dex_features(self) -> Dict:
        """Get DEX features for fusion (read-only)."""
        if self._sol_dex:
            return self._sol_dex.get_features_for_fusion()
        return {}


# =============================================================================
# SINGLETON
# =============================================================================

_shadow_observer_manager: Optional[ShadowObserverManager] = None


def get_shadow_observer_manager() -> ShadowObserverManager:
    """Get or create singleton instance."""
    global _shadow_observer_manager
    if _shadow_observer_manager is None:
        _shadow_observer_manager = ShadowObserverManager()
    return _shadow_observer_manager


def reset_shadow_observer_manager():
    """Reset singleton."""
    global _shadow_observer_manager
    _shadow_observer_manager = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def observe_shadows(market_data: Dict = None):
    """Convenience function to run shadow observations."""
    manager = get_shadow_observer_manager()
    manager.observe_post_decision(market_data)


def analyze_execution_shadow(order_result: Dict, market_state: Dict = None):
    """Convenience function to analyze execution."""
    manager = get_shadow_observer_manager()
    manager.analyze_post_execution(order_result, market_state)


__all__ = [
    "ShadowEventType",
    "ShadowObservation",
    "LeadLagShadowObserver",
    "PassiveAggressiveShadowObserver",
    "SolDexShadowObserver",
    "ShadowObserverManager",
    "get_shadow_observer_manager",
    "reset_shadow_observer_manager",
    "observe_shadows",
    "analyze_execution_shadow",
]

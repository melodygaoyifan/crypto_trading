"""
================================================================================
HMATS v5.2.1 - 链上图谱 Alpha 引擎
On-Chain Graph Alpha Engine
================================================================================
Purpose: 分析 Solana 链上地址网络、交易热度和鲸鱼活动

SOTA 对照:
- 分析 Solana 链上地址网络、交易热度和鲸鱼活动
- 将链上指标纳入信号融合

Features:
1. 地址网络分析 (中心性、聚类)
2. 鲸鱼钱包追踪
3. DEX 流动性监控
4. 交易热度指标
5. 链上-价格相关性分析
6. EventBus 集成

Data Sources (Mock/Real):
- Solana RPC endpoints
- Helius API (account activity)
- Birdeye API (DEX data)
- Jupiter aggregator

Author: HMATS v5.2.1 SOTA Upgrade
Version: 5.2.1
================================================================================
"""

import logging
import asyncio
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class WhaleAction(Enum):
    """鲸鱼行为类型"""
    ACCUMULATE = "accumulate"       # 累积
    DISTRIBUTE = "distribute"       # 分发
    TRANSFER_IN = "transfer_in"     # 转入交易所
    TRANSFER_OUT = "transfer_out"   # 转出交易所
    STAKE = "stake"                 # 质押
    UNSTAKE = "unstake"            # 解质押
    LP_ADD = "lp_add"              # 添加流动性
    LP_REMOVE = "lp_remove"        # 移除流动性


class OnChainSignalType(Enum):
    """链上信号类型"""
    WHALE_ACCUMULATION = "whale_accumulation"
    WHALE_DISTRIBUTION = "whale_distribution"
    EXCHANGE_INFLOW = "exchange_inflow"
    EXCHANGE_OUTFLOW = "exchange_outflow"
    DEX_VOLUME_SPIKE = "dex_volume_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    NETWORK_ACTIVITY_SURGE = "network_activity_surge"
    SMART_MONEY_MOVE = "smart_money_move"
    STAKING_FLOW = "staking_flow"


class AddressType(Enum):
    """地址类型"""
    UNKNOWN = "unknown"
    WHALE = "whale"
    EXCHANGE = "exchange"
    DEFI_PROTOCOL = "defi_protocol"
    SMART_MONEY = "smart_money"
    RETAIL = "retail"
    VALIDATOR = "validator"
    TEAM = "team"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WalletProfile:
    """钱包画像"""
    address: str
    address_type: AddressType
    
    # 余额
    sol_balance: float = 0.0
    token_balances: Dict[str, float] = field(default_factory=dict)
    total_value_usd: float = 0.0
    
    # 历史行为
    total_transactions: int = 0
    avg_transaction_size: float = 0.0
    first_seen: Optional[datetime] = None
    last_active: Optional[datetime] = None
    
    # 标签
    labels: List[str] = field(default_factory=list)
    is_whale: bool = False
    is_smart_money: bool = False
    
    # 网络指标
    in_degree: int = 0          # 入边数 (收款次数)
    out_degree: int = 0         # 出边数 (发款次数)
    centrality_score: float = 0.0


@dataclass
class Transaction:
    """链上交易记录"""
    tx_hash: str
    timestamp: datetime
    from_address: str
    to_address: str
    amount: float
    token: str  # "SOL" or token address
    value_usd: float
    tx_type: str  # "transfer", "swap", "stake", etc.
    
    # DEX 相关
    is_dex_swap: bool = False
    dex_name: Optional[str] = None
    swap_from_token: Optional[str] = None
    swap_to_token: Optional[str] = None


@dataclass
class DEXMetrics:
    """DEX 指标"""
    timestamp: datetime
    dex_name: str  # "Raydium", "Orca", "Jupiter"
    
    volume_24h_usd: float = 0.0
    volume_change_pct: float = 0.0
    
    # SOL 相关
    sol_buy_volume: float = 0.0
    sol_sell_volume: float = 0.0
    net_sol_flow: float = 0.0
    
    # 流动性
    tvl_usd: float = 0.0
    tvl_change_pct: float = 0.0
    
    # 交易计数
    swap_count: int = 0
    unique_traders: int = 0


@dataclass
class NetworkMetrics:
    """网络指标"""
    timestamp: datetime
    
    # 活跃度
    active_addresses_24h: int = 0
    new_addresses_24h: int = 0
    transactions_24h: int = 0
    
    # 价值流动
    total_volume_24h_usd: float = 0.0
    avg_transaction_value: float = 0.0
    
    # 鲸鱼活动
    whale_transactions: int = 0
    whale_volume_usd: float = 0.0
    
    # 质押
    staking_deposits: float = 0.0
    staking_withdrawals: float = 0.0
    net_staking_flow: float = 0.0


@dataclass
class OnChainAlphaSignal:
    """链上 Alpha 信号"""
    timestamp: datetime
    signal_type: OnChainSignalType
    symbol: str  # "SOL"
    
    direction: float      # -1 (bearish) to 1 (bullish)
    confidence: float     # 0 to 1
    strength: float       # 信号强度
    
    # 来源
    source_addresses: List[str] = field(default_factory=list)
    source_transactions: List[str] = field(default_factory=list)
    
    # 解释
    reasoning: str = ""
    metrics: Dict = field(default_factory=dict)
    
    # 时效
    ttl_hours: int = 24
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "signal_type": self.signal_type.value,
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": self.confidence,
            "strength": self.strength,
            "reasoning": self.reasoning,
            "ttl_hours": self.ttl_hours
        }


@dataclass
class OnChainGraphConfig:
    """链上图谱配置"""
    # 数据源
    solana_rpc_url: str = "https://api.mainnet-beta.solana.com"
    helius_api_key: str = ""
    birdeye_api_key: str = ""
    
    # 鲸鱼定义
    whale_threshold_sol: float = 10000.0       # 10K SOL
    whale_threshold_usd: float = 1000000.0     # $1M
    smart_money_min_pnl: float = 0.5           # 50% 累计收益
    
    # 交易所地址 (已知)
    exchange_addresses: Set[str] = field(default_factory=lambda: {
        "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",  # Binance
        "5tzFkiKscXHK5ZXCGbXZxdw7gTjjD1mBwuoFbhUvuAi9",  # Coinbase
        # ... 更多交易所地址
    })
    
    # 信号阈值
    volume_spike_threshold: float = 2.0        # 2x 正常成交量
    whale_movement_threshold_usd: float = 500000.0
    inflow_outflow_ratio_alert: float = 2.0
    
    # 更新频率
    update_interval_seconds: int = 60
    history_window_hours: int = 72


# =============================================================================
# ADDRESS NETWORK ANALYZER
# =============================================================================

class AddressNetworkAnalyzer:
    """地址网络分析器"""
    
    def __init__(self, config: OnChainGraphConfig = None):
        self.config = config or OnChainGraphConfig()
        
        # 地址图谱
        self.wallet_profiles: Dict[str, WalletProfile] = {}
        self.transaction_graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # 已知地址
        self.known_whales: Set[str] = set()
        self.known_smart_money: Set[str] = set()
    
    def add_transaction(self, tx: Transaction):
        """添加交易到图谱"""
        # 更新边权重
        self.transaction_graph[tx.from_address][tx.to_address] += tx.value_usd
        
        # 更新钱包画像
        self._update_wallet(tx.from_address, tx, is_sender=True)
        self._update_wallet(tx.to_address, tx, is_sender=False)
    
    def _update_wallet(self, address: str, tx: Transaction, is_sender: bool):
        """更新钱包画像"""
        if address not in self.wallet_profiles:
            self.wallet_profiles[address] = WalletProfile(
                address=address,
                address_type=self._classify_address(address),
                first_seen=tx.timestamp
            )
        
        profile = self.wallet_profiles[address]
        profile.total_transactions += 1
        profile.last_active = tx.timestamp
        
        if is_sender:
            profile.out_degree += 1
        else:
            profile.in_degree += 1
        
        # 更新平均交易额
        n = profile.total_transactions
        profile.avg_transaction_size = (
            (profile.avg_transaction_size * (n - 1) + tx.value_usd) / n
        )
        
        # 检查是否为鲸鱼
        if profile.total_value_usd >= self.config.whale_threshold_usd:
            profile.is_whale = True
            self.known_whales.add(address)
    
    def _classify_address(self, address: str) -> AddressType:
        """分类地址"""
        if address in self.config.exchange_addresses:
            return AddressType.EXCHANGE
        if address in self.known_whales:
            return AddressType.WHALE
        if address in self.known_smart_money:
            return AddressType.SMART_MONEY
        return AddressType.UNKNOWN
    
    def compute_centrality(self, top_n: int = 100) -> Dict[str, float]:
        """计算地址中心性"""
        # 简化版 PageRank
        addresses = list(self.wallet_profiles.keys())
        n = len(addresses)
        
        if n == 0:
            return {}
        
        # 初始化
        scores = {addr: 1.0 / n for addr in addresses}
        damping = 0.85
        
        # 迭代
        for _ in range(20):
            new_scores = {}
            for addr in addresses:
                # 入边贡献
                incoming = sum(
                    scores.get(src, 0) * self.transaction_graph[src].get(addr, 0)
                    for src in self.transaction_graph
                    if addr in self.transaction_graph[src]
                )
                
                total_out = sum(self.transaction_graph[addr].values())
                if total_out > 0:
                    incoming /= total_out
                
                new_scores[addr] = (1 - damping) / n + damping * incoming
            
            scores = new_scores
        
        # 更新画像
        for addr, score in scores.items():
            if addr in self.wallet_profiles:
                self.wallet_profiles[addr].centrality_score = score
        
        # 返回 Top N
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_scores[:top_n])
    
    def detect_whale_movements(self, min_value_usd: float = None) -> List[Dict]:
        """检测鲸鱼动向"""
        threshold = min_value_usd or self.config.whale_movement_threshold_usd
        movements = []
        
        for addr in self.known_whales:
            if addr not in self.wallet_profiles:
                continue
            
            profile = self.wallet_profiles[addr]
            
            # 检查最近活动
            if profile.last_active:
                time_since = datetime.now(timezone.utc) - profile.last_active
                if time_since < timedelta(hours=24):
                    # 获取该地址的出边
                    outflows = self.transaction_graph.get(addr, {})
                    total_out = sum(outflows.values())
                    
                    if total_out >= threshold:
                        movements.append({
                            "address": addr,
                            "type": profile.address_type.value,
                            "total_outflow_usd": total_out,
                            "destinations": len(outflows),
                            "last_active": profile.last_active.isoformat()
                        })
        
        return movements


# =============================================================================
# ON-CHAIN GRAPH ALPHA ENGINE
# =============================================================================

class OnChainGraphAlphaEngine:
    """
    链上图谱 Alpha 引擎
    
    核心功能:
    1. 监控鲸鱼钱包活动
    2. 分析 DEX 流动性和交易量
    3. 追踪交易所资金流
    4. 识别聪明钱动向
    5. 生成链上 Alpha 信号
    """
    
    def __init__(self, config: OnChainGraphConfig = None):
        self.config = config or OnChainGraphConfig()
        
        # 组件
        self.network_analyzer = AddressNetworkAnalyzer(config)
        
        # 数据存储
        self.transactions: deque = deque(maxlen=10000)
        self.dex_metrics_history: deque = deque(maxlen=500)
        self.network_metrics_history: deque = deque(maxlen=500)
        
        # 当前状态
        self.current_dex_metrics: Dict[str, DEXMetrics] = {}
        self.current_network_metrics: Optional[NetworkMetrics] = None
        
        # 信号
        self.active_signals: List[OnChainAlphaSignal] = []
        self.signal_history: deque = deque(maxlen=200)
        
        # 事件回调
        self.event_callbacks: List[Callable] = []
        
        logger.info("OnChainGraphAlphaEngine initialized")
    
    def register_event_callback(self, callback: Callable):
        """注册事件回调"""
        self.event_callbacks.append(callback)
    
    def _emit_event(self, event_type: str, data: Dict):
        """发布事件"""
        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "OnChainGraphAlphaEngine",
            "data": data
        }
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def ingest_transaction(self, tx: Transaction):
        """摄入交易数据"""
        self.transactions.append(tx)
        self.network_analyzer.add_transaction(tx)
        
        # 检测重要事件
        self._check_significant_transaction(tx)
    
    def _check_significant_transaction(self, tx: Transaction):
        """检查重要交易"""
        # 鲸鱼交易
        if tx.value_usd >= self.config.whale_movement_threshold_usd:
            from_profile = self.network_analyzer.wallet_profiles.get(tx.from_address)
            to_profile = self.network_analyzer.wallet_profiles.get(tx.to_address)
            
            # 交易所流入
            if to_profile and to_profile.address_type == AddressType.EXCHANGE:
                self._generate_signal(
                    OnChainSignalType.EXCHANGE_INFLOW,
                    direction=-0.5,  # 利空
                    confidence=0.7,
                    strength=min(1.0, tx.value_usd / 1000000),
                    reasoning=f"大额转入交易所: ${tx.value_usd:,.0f}",
                    addresses=[tx.from_address]
                )
            
            # 交易所流出
            elif from_profile and from_profile.address_type == AddressType.EXCHANGE:
                self._generate_signal(
                    OnChainSignalType.EXCHANGE_OUTFLOW,
                    direction=0.5,  # 利多
                    confidence=0.7,
                    strength=min(1.0, tx.value_usd / 1000000),
                    reasoning=f"大额流出交易所: ${tx.value_usd:,.0f}",
                    addresses=[tx.to_address]
                )
    
    def update_dex_metrics(self, metrics: DEXMetrics):
        """更新 DEX 指标"""
        self.current_dex_metrics[metrics.dex_name] = metrics
        self.dex_metrics_history.append(metrics)
        
        # 检测异常
        self._check_dex_anomalies(metrics)
    
    def _check_dex_anomalies(self, metrics: DEXMetrics):
        """检测 DEX 异常"""
        # 成交量飙升
        if metrics.volume_change_pct > self.config.volume_spike_threshold:
            self._generate_signal(
                OnChainSignalType.DEX_VOLUME_SPIKE,
                direction=0.3,  # 轻微利多 (活跃度增加)
                confidence=0.6,
                strength=min(1.0, metrics.volume_change_pct / 5),
                reasoning=f"{metrics.dex_name} 成交量飙升 {metrics.volume_change_pct:.0%}",
                metrics={"volume_change": metrics.volume_change_pct}
            )
        
        # TVL 大幅下降 (流动性危机)
        if metrics.tvl_change_pct < -0.2:  # -20%
            self._generate_signal(
                OnChainSignalType.LIQUIDITY_CRISIS,
                direction=-0.7,
                confidence=0.8,
                strength=abs(metrics.tvl_change_pct),
                reasoning=f"{metrics.dex_name} TVL 下降 {metrics.tvl_change_pct:.0%}",
                metrics={"tvl_change": metrics.tvl_change_pct}
            )
        
        # 净买入/卖出不平衡
        if metrics.sol_buy_volume + metrics.sol_sell_volume > 0:
            ratio = metrics.sol_buy_volume / max(1, metrics.sol_sell_volume)
            if ratio > self.config.inflow_outflow_ratio_alert:
                self._generate_signal(
                    OnChainSignalType.WHALE_ACCUMULATION,
                    direction=0.6,
                    confidence=0.65,
                    strength=min(1.0, ratio / 5),
                    reasoning=f"DEX 买入/卖出比 {ratio:.1f}x",
                    metrics={"buy_sell_ratio": ratio}
                )
            elif ratio < 1 / self.config.inflow_outflow_ratio_alert:
                self._generate_signal(
                    OnChainSignalType.WHALE_DISTRIBUTION,
                    direction=-0.6,
                    confidence=0.65,
                    strength=min(1.0, (1 / max(ratio, 0.01) - 1) / 8),  # [FIX-AG10] Was 1/ratio/5 (always 1.0 for ratio<0.2). Now: ratio=0.5→0.125, 0.25→0.375, 0.1→1.0
                    reasoning=f"DEX 卖出/买入比 {1/ratio:.1f}x",
                    metrics={"sell_buy_ratio": 1/ratio}
                )
    
    def update_network_metrics(self, metrics: NetworkMetrics):
        """更新网络指标"""
        self.current_network_metrics = metrics
        self.network_metrics_history.append(metrics)
        
        # 检测网络异常
        self._check_network_anomalies(metrics)
    
    def _check_network_anomalies(self, metrics: NetworkMetrics):
        """检测网络异常"""
        # 检查历史比较
        if len(self.network_metrics_history) < 10:
            return
        
        historical = list(self.network_metrics_history)[-24:]  # 24小时历史
        
        # 活跃地址飙升
        avg_active = np.mean([m.active_addresses_24h for m in historical])
        if metrics.active_addresses_24h > avg_active * 1.5:
            self._generate_signal(
                OnChainSignalType.NETWORK_ACTIVITY_SURGE,
                direction=0.4,
                confidence=0.55,
                strength=min(1.0, (metrics.active_addresses_24h / avg_active - 1)),
                reasoning=f"活跃地址飙升至 {metrics.active_addresses_24h:,}",
                metrics={"active_addresses": metrics.active_addresses_24h}
            )
        
        # 质押流向
        if metrics.net_staking_flow > self.config.whale_movement_threshold_usd:
            self._generate_signal(
                OnChainSignalType.STAKING_FLOW,
                direction=0.4,  # 质押增加 = 利多
                confidence=0.6,
                strength=min(1.0, metrics.net_staking_flow / 5000000),
                reasoning=f"净质押流入 ${metrics.net_staking_flow:,.0f}",
                metrics={"net_staking": metrics.net_staking_flow}
            )
        elif metrics.net_staking_flow < -self.config.whale_movement_threshold_usd:
            self._generate_signal(
                OnChainSignalType.STAKING_FLOW,
                direction=-0.4,  # 解质押 = 利空
                confidence=0.6,
                strength=min(1.0, abs(metrics.net_staking_flow) / 5000000),
                reasoning=f"净质押流出 ${abs(metrics.net_staking_flow):,.0f}",
                metrics={"net_staking": metrics.net_staking_flow}
            )
    
    def _generate_signal(self, signal_type: OnChainSignalType,
                         direction: float, confidence: float, strength: float,
                         reasoning: str, addresses: List[str] = None,
                         metrics: Dict = None):
        """生成信号"""
        signal = OnChainAlphaSignal(
            timestamp=datetime.now(timezone.utc),
            signal_type=signal_type,
            symbol="SOL",
            direction=direction,
            confidence=confidence,
            strength=strength,
            source_addresses=addresses or [],
            reasoning=reasoning,
            metrics=metrics or {},
            ttl_hours=24
        )
        
        self.active_signals.append(signal)
        self.signal_history.append(signal)
        
        # 清理过期信号
        self._cleanup_expired_signals()
        
        # 发布事件
        self._emit_event("ONCHAIN_SIGNAL", signal.to_dict())
        
        logger.info(f"OnChain signal: {signal_type.value}, dir={direction:.2f}, {reasoning}")
    
    def _cleanup_expired_signals(self):
        """清理过期信号"""
        now = datetime.now(timezone.utc)
        self.active_signals = [
            s for s in self.active_signals
            if (now - s.timestamp).total_seconds() < s.ttl_hours * 3600
        ]
    
    def get_aggregated_signal(self) -> Tuple[float, float]:
        """
        获取聚合链上信号
        
        Returns:
            (direction, confidence) 其中 direction 在 -1 到 1 之间
        """
        if not self.active_signals:
            return 0.0, 0.0
        
        # 加权聚合
        total_weight = 0.0
        weighted_direction = 0.0
        
        for signal in self.active_signals:
            weight = signal.confidence * signal.strength
            weighted_direction += signal.direction * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_direction = weighted_direction / total_weight
            avg_confidence = total_weight / len(self.active_signals)
            return avg_direction, avg_confidence
        
        return 0.0, 0.0
    
    def get_whale_summary(self) -> Dict:
        """获取鲸鱼活动摘要"""
        movements = self.network_analyzer.detect_whale_movements()
        
        total_inflow = sum(m["total_outflow_usd"] for m in movements 
                          if "exchange" in m.get("type", "").lower())
        
        return {
            "known_whales": len(self.network_analyzer.known_whales),
            "active_whales_24h": len(movements),
            "total_movements": movements,
            "exchange_inflow_usd": total_inflow,
            "top_movers": movements[:5]
        }
    
    def get_dex_summary(self) -> Dict:
        """获取 DEX 摘要"""
        if not self.current_dex_metrics:
            return {"available": False}
        
        total_volume = sum(m.volume_24h_usd for m in self.current_dex_metrics.values())
        total_tvl = sum(m.tvl_usd for m in self.current_dex_metrics.values())
        
        net_flow = sum(m.net_sol_flow for m in self.current_dex_metrics.values())
        
        return {
            "available": True,
            "total_volume_24h": total_volume,
            "total_tvl": total_tvl,
            "net_sol_flow": net_flow,
            "dexes": {name: m.volume_24h_usd for name, m in self.current_dex_metrics.items()}
        }
    
    def get_dashboard_data(self) -> Dict:
        """获取仪表盘数据"""
        agg_dir, agg_conf = self.get_aggregated_signal()
        
        return {
            "aggregated_signal": {
                "direction": round(agg_dir, 3),
                "confidence": round(agg_conf, 3),
                "interpretation": "bullish" if agg_dir > 0.2 else "bearish" if agg_dir < -0.2 else "neutral"
            },
            "active_signals": len(self.active_signals),
            "recent_signals": [s.to_dict() for s in self.active_signals[-5:]],
            "whale_summary": self.get_whale_summary(),
            "dex_summary": self.get_dex_summary(),
            "network_metrics": {
                "active_addresses": self.current_network_metrics.active_addresses_24h if self.current_network_metrics else 0,
                "transactions_24h": self.current_network_metrics.transactions_24h if self.current_network_metrics else 0,
            }
        }
    
    def get_status(self) -> Dict:
        """获取引擎状态"""
        return {
            "transactions_ingested": len(self.transactions),
            "known_wallets": len(self.network_analyzer.wallet_profiles),
            "known_whales": len(self.network_analyzer.known_whales),
            "active_signals": len(self.active_signals),
            "signal_history": len(self.signal_history),
            "dex_tracked": list(self.current_dex_metrics.keys()),
        }


# =============================================================================
# MOCK DATA GENERATOR (testing / __main__ ONLY - NOT for production)
#
# OnChainDataSimulator uses ``import random`` internally.  It is intentionally
# kept in this file so existing tests that import it still work, but it MUST
# NOT be instantiated or called from main.py or any production code path.
# =============================================================================

class OnChainDataSimulator:
    """链上数据模拟器 (testing / __main__ only - NOT for production)."""

    _PRODUCTION_GUARD = True  # set to False in tests if needed

    def __init__(self, engine: OnChainGraphAlphaEngine):
        self.engine = engine
        self.tx_counter = 0

    def generate_random_transaction(self) -> Transaction:
        """Generate a random transaction (test helper)."""
        import random

        self.tx_counter += 1

        addresses = (
            [f"Whale{i}xxx" for i in range(10)]
            + [f"Exchange{i}xxx" for i in range(3)]
            + [f"Retail{i}xxx" for i in range(50)]
        )

        from_addr = random.choice(addresses)
        to_addr = random.choice([a for a in addresses if a != from_addr])

        is_whale = "Whale" in from_addr
        amount = random.uniform(1, 100) if not is_whale else random.uniform(1000, 50000)
        sol_price = 150

        return Transaction(
            tx_hash=f"tx_{self.tx_counter}_{hashlib.md5(str(self.tx_counter).encode()).hexdigest()[:8]}",
            timestamp=datetime.now(timezone.utc),
            from_address=from_addr,
            to_address=to_addr,
            amount=amount,
            token="SOL",
            value_usd=amount * sol_price,
            tx_type="transfer",
            is_dex_swap="Exchange" in from_addr or "Exchange" in to_addr,
        )

    def generate_dex_metrics(self, dex_name: str) -> DEXMetrics:
        """Generate DEX metrics (test helper)."""
        import random

        base_volume = 10000000
        volume = base_volume * random.uniform(0.5, 2.0)
        buy_ratio = random.uniform(0.3, 0.7)

        return DEXMetrics(
            timestamp=datetime.now(timezone.utc),
            dex_name=dex_name,
            volume_24h_usd=volume,
            volume_change_pct=random.uniform(-0.5, 1.0),
            sol_buy_volume=volume * buy_ratio,
            sol_sell_volume=volume * (1 - buy_ratio),
            net_sol_flow=volume * (buy_ratio - 0.5) * 2,
            tvl_usd=volume * 5,
            tvl_change_pct=random.uniform(-0.1, 0.1),
            swap_count=int(volume / 1000),
            unique_traders=int(volume / 5000),
        )

    def simulate_market_scenario(self, scenario: str = "normal"):
        """Simulate a market scenario (test helper)."""
        import random

        n_txs = 100 if scenario == "normal" else 300
        for _ in range(n_txs):
            tx = self.generate_random_transaction()
            if scenario == "accumulation" and "Exchange" in tx.to_address:
                continue
            elif scenario == "distribution" and "Exchange" in tx.from_address:
                continue
            self.engine.ingest_transaction(tx)

        for dex in ["Raydium", "Orca", "Jupiter"]:
            metrics = self.generate_dex_metrics(dex)
            if scenario == "accumulation":
                metrics.sol_buy_volume *= 1.5
                metrics.net_sol_flow = metrics.sol_buy_volume - metrics.sol_sell_volume
            elif scenario == "distribution":
                metrics.sol_sell_volume *= 1.5
                metrics.net_sol_flow = metrics.sol_buy_volume - metrics.sol_sell_volume
            self.engine.update_dex_metrics(metrics)

        network = NetworkMetrics(
            timestamp=datetime.now(timezone.utc),
            active_addresses_24h=random.randint(50000, 150000),
            new_addresses_24h=random.randint(1000, 5000),
            transactions_24h=n_txs * 100,
            total_volume_24h_usd=random.uniform(100000000, 500000000),
            whale_transactions=random.randint(10, 50),
            whale_volume_usd=random.uniform(10000000, 50000000),
            net_staking_flow=random.uniform(-5000000, 5000000),
        )
        self.engine.update_network_metrics(network)


# =============================================================================
# V6 AGENT WRAPPER - OnChainGraphAlphaAgent
# =============================================================================

class OnChainGraphAlphaAgent:
    """
    V6-compatible agent wrapper around OnChainGraphAlphaEngine.

    Exposes the same generate_signal() / to_agent_signal_dict() contract
    used by the other overhauled HMATS agents.  No network calls inside
    generate_signal() - reads the latest engine snapshot only.

    Authority: ADVISE (modulates confidence, does not generate independent
    entry/exit decisions).
    """

    # Freshness gate - if engine snapshot is older than this, emit neutral
    MAX_SNAPSHOT_AGE_SECONDS: float = 14400.0  # 4 hours (1 tick)

    def __init__(
        self,
        engine: Optional[OnChainGraphAlphaEngine] = None,
        directional_alpha_enabled: bool = False,
    ):
        self._engine = engine
        self._last_signal: Optional[Dict[str, Any]] = None
        self._last_gating_reason: Optional[str] = None
        self.directional_alpha_enabled = directional_alpha_enabled

    @property
    def engine(self) -> Optional[OnChainGraphAlphaEngine]:
        return self._engine

    @engine.setter
    def engine(self, eng: OnChainGraphAlphaEngine) -> None:
        self._engine = eng

    def generate_signal(
        self,
        asset: str = "SOL",
        price: float = 0.0,
        regime: str = "",
    ) -> Dict[str, Any]:
        """
        Generate an on-chain graph alpha signal.

        Always returns a dict (never None, never raises).
        Returns neutral signal on missing engine / no data / stale snapshot.
        """
        self._last_gating_reason = None

        # Gate: no engine
        if self._engine is None:
            self._last_gating_reason = "no_engine"
            return self._neutral_signal(asset, reason="no_engine")

        # Gate: no active signals
        if not self._engine.active_signals:
            self._last_gating_reason = "no_data"
            return self._neutral_signal(asset, reason="no_onchain_data")

        # Gate: freshness
        import time as _time

        latest_ts = max(
            (s.timestamp for s in self._engine.active_signals),
            default=None,
        )
        if latest_ts is None:
            self._last_gating_reason = "no_timestamp"
            return self._neutral_signal(asset, reason="no_timestamp")

        age = _time.time() - latest_ts.timestamp()
        if age > self.MAX_SNAPSHOT_AGE_SECONDS:
            self._last_gating_reason = f"stale({age:.0f}s)"
            return self._neutral_signal(asset, reason="stale_snapshot")

        # Aggregate engine signals
        direction, confidence = self._engine.get_aggregated_signal()

        # v6.4: if directional alpha is disabled, zero out direction
        if not self.directional_alpha_enabled:
            direction = 0.0
            confidence = 0.0

        # Clamp
        direction = max(-1.0, min(1.0, direction))
        confidence = max(0.0, min(1.0, confidence))

        # Collect reasons from active signals
        reasons = []
        for sig in self._engine.active_signals[:5]:
            reasons.append(sig.signal_type.value)

        data_quality = min(1.0, len(self._engine.active_signals) / 3.0)

        signal = {
            "onchain_graph_direction": round(direction, 4),
            "onchain_graph_confidence": round(confidence, 4),
            "onchain_graph_data_quality": round(data_quality, 3),
            "onchain_graph_valid": True,
            "onchain_graph_reasons": reasons,
            "onchain_graph_n_signals": len(self._engine.active_signals),
            "asset": asset,
            "asof_timestamp": latest_ts.timestamp() if latest_ts else 0.0,
            "data_age_seconds": round(age, 1) if latest_ts else None,
        }

        self._last_signal = signal
        return signal

    def to_agent_signal_dict(self, asset: str = "SOL") -> Dict[str, Any]:
        """
        Return latest signal as a flat dict for Authority Fusion.

        Calls generate_signal() if no cached signal exists.
        """
        if self._last_signal is None or self._last_signal.get("asset") != asset:
            return self.generate_signal(asset=asset)
        return self._last_signal

    def to_market_data_patch(self, asset: str = "SOL") -> Dict[str, Any]:
        """
        Return canonical on-chain feature keys for market_data enrichment.

        These are FEATURE inputs (not directional alpha) written into the
        market_data dict consumed by constitution / trade-gate.

        Keys produced (all Optional[float], None = unknown):
          onchain_whale_flow, onchain_exchange_flow, onchain_dex_volume,
          onchain_staking_flow, onchain_data_quality
        """
        if self._engine is None:
            return {
                "onchain_whale_flow": None,
                "onchain_exchange_flow": None,
                "onchain_dex_volume": None,
                "onchain_staking_flow": None,
                "onchain_data_quality": 0.0,
            }

        # Pull engine-level summaries
        whale_summary = self._engine.get_whale_summary()
        dex_summary = self._engine.get_dex_summary()
        net_metrics = self._engine.current_network_metrics

        return {
            "onchain_whale_flow": whale_summary.get("exchange_inflow_usd"),
            "onchain_exchange_flow": (
                dex_summary.get("net_sol_flow") if dex_summary.get("available") else None
            ),
            "onchain_dex_volume": (
                dex_summary.get("total_volume_24h") if dex_summary.get("available") else None
            ),
            "onchain_staking_flow": (
                net_metrics.net_staking_flow if net_metrics else None
            ),
            "onchain_data_quality": min(
                1.0, len(self._engine.active_signals) / 3.0
            ) if self._engine.active_signals else 0.0,
        }

    def get_feature_snapshot(
        self, asset: str = "SOL", now_ts: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        V6.4 feature snapshot - metrics WITHOUT forced direction.

        Returns activity/quality scoring data for downstream consumers.
        No RPC/network calls - reads cached engine state only.
        """
        import time as _time

        now_iso = (now_ts or datetime.now(timezone.utc)).isoformat() + "Z"

        if self._engine is None:
            return {
                "available": False,
                "whale_activity_count": 0,
                "dex_volume_change_pct": None,
                "exchange_inflow_usd": None,
                "exchange_outflow_usd": None,
                "staking_net_flow": None,
                "active_signals": 0,
                "data_quality": "missing",
                "asof_ts": now_iso,
                "data_age_seconds": 0.0,
                "provenance": ["onchain_graph_alpha", asset, "no_engine"],
            }

        whale = self._engine.get_whale_summary()
        dex = self._engine.get_dex_summary()
        net = self._engine.current_network_metrics
        n_signals = len(self._engine.active_signals)

        # Compute data age
        latest_ts = max(
            (s.timestamp for s in self._engine.active_signals),
            default=None,
        )
        age = (_time.time() - latest_ts.timestamp()) if latest_ts else 0.0

        if n_signals == 0:
            dq = "missing"
        elif age > self.MAX_SNAPSHOT_AGE_SECONDS:
            dq = "stale"
        else:
            dq = "ok"

        return {
            "available": n_signals > 0,
            "whale_activity_count": whale.get("active_whales_24h", 0),
            "dex_volume_change_pct": (
                dex.get("net_sol_flow") if dex.get("available") else None
            ),
            "exchange_inflow_usd": whale.get("exchange_inflow_usd"),
            "exchange_outflow_usd": None,  # not tracked separately
            "staking_net_flow": net.net_staking_flow if net else None,
            "active_signals": n_signals,
            "data_quality": dq,
            "asof_ts": now_iso,
            "data_age_seconds": round(age, 1),
            "provenance": ["onchain_graph_alpha", asset],
        }

    @staticmethod
    def _neutral_signal(asset: str, reason: str = "gated") -> Dict[str, Any]:
        """Fail-closed neutral signal."""
        return {
            "onchain_graph_direction": 0.0,
            "onchain_graph_confidence": 0.0,
            "onchain_graph_data_quality": 0.0,
            "onchain_graph_valid": False,
            "onchain_graph_reasons": [reason],
            "onchain_graph_n_signals": 0,
            "asset": asset,
            "asof_timestamp": 0.0,
            "data_age_seconds": None,
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_onchain_engine: Optional[OnChainGraphAlphaEngine] = None
_onchain_agent: Optional[OnChainGraphAlphaAgent] = None


def get_onchain_alpha_engine(config: OnChainGraphConfig = None) -> OnChainGraphAlphaEngine:
    """获取链上 Alpha 引擎单例 (UNUSED - use get_onchain_graph_agent instead)"""
    global _onchain_engine
    if _onchain_engine is None:
        _onchain_engine = OnChainGraphAlphaEngine(config)
    elif config is not None and _onchain_engine.config != config:
        logger.info("[ONCHAIN_GRAPH] Config changed; refreshing engine singleton")
        _onchain_engine = OnChainGraphAlphaEngine(config)
    return _onchain_engine


def get_onchain_graph_agent(engine: Optional[OnChainGraphAlphaEngine] = None) -> OnChainGraphAlphaAgent:
    """Get singleton OnChainGraphAlphaAgent."""
    global _onchain_agent
    resolved_engine = engine or get_onchain_alpha_engine()
    if _onchain_agent is None:
        _onchain_agent = OnChainGraphAlphaAgent(engine=resolved_engine)
    elif _onchain_agent.engine is not resolved_engine:
        logger.info("[ONCHAIN_GRAPH] Engine changed; refreshing agent singleton")
        _onchain_agent = OnChainGraphAlphaAgent(engine=resolved_engine)
    return _onchain_agent


def reset_onchain_alpha_engine():
    """重置单例"""
    global _onchain_engine, _onchain_agent
    _onchain_engine = None
    _onchain_agent = None


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建引擎
    engine = OnChainGraphAlphaEngine()
    
    # 创建模拟器
    simulator = OnChainDataSimulator(engine)
    
    print("\n=== Simulating Accumulation Scenario ===")
    simulator.simulate_market_scenario("accumulation")
    
    # 获取信号
    direction, confidence = engine.get_aggregated_signal()
    print(f"\nAggregated Signal: direction={direction:.3f}, confidence={confidence:.3f}")
    
    # 仪表盘数据
    dashboard = engine.get_dashboard_data()
    print(f"\nDashboard Data:")
    print(json.dumps(dashboard, indent=2, default=str))
    
    # 状态
    print(f"\nEngine Status: {engine.get_status()}")

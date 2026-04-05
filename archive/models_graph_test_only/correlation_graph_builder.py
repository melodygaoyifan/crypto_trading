"""
================================================================================
HMATS v5.2.1 SOTA - 相关性图结构构建
Correlation Graph Builder
================================================================================

Purpose: 构建跨资产相关性图结构

节点: BTC / ETH / SOL / dominance / total mcap
边: rolling correlation + tail correlation

用途:
- 做空放权前提
- 熊市"系统性下行"提前识别
- 压制假反弹 long

================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CorrelationGraphConfig:
    """相关性图配置"""
    
    # 节点
    nodes: List[str] = field(default_factory=lambda: [
        "BTC", "ETH", "SOL", "dominance", "total_mcap"
    ])
    
    # 相关性计算
    rolling_window: int = 24          # 24 * 4H = 4天
    tail_window: int = 96             # 16天
    tail_percentile: float = 0.05     # 尾部相关性的百分位
    
    # 边权重阈值
    edge_threshold: float = 0.3       # 相关性低于此不连边
    
    # 系统性风险
    systemic_threshold: float = 0.7   # 系统性相关性阈值
    
    # 更新频率
    update_interval_bars: int = 4     # 每 4 根 4H bar 更新一次


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GraphNode:
    """图节点"""
    name: str
    returns: deque = field(default_factory=lambda: deque(maxlen=200))
    
    # 节点特征
    volatility: float = 0.0
    momentum: float = 0.0
    tail_risk: float = 0.0
    
    # 最近更新
    last_price: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    def add_return(self, ret: float, price: float):
        """添加收益率"""
        self.returns.append(ret)
        self.last_price = price
        self.last_update = datetime.now()
        
        if len(self.returns) >= 10:
            returns_arr = np.array(self.returns)
            self.volatility = np.std(returns_arr[-24:]) if len(returns_arr) >= 24 else np.std(returns_arr)
            self.momentum = np.sum(returns_arr[-12:]) if len(returns_arr) >= 12 else np.sum(returns_arr)
            
            # 尾部风险 (左尾)
            neg_returns = returns_arr[returns_arr < 0]
            self.tail_risk = np.percentile(neg_returns, 5) if len(neg_returns) > 5 else 0


@dataclass
class GraphEdge:
    """图边"""
    source: str
    target: str
    
    # 边权重
    rolling_correlation: float = 0.0      # 滚动相关性
    tail_correlation: float = 0.0         # 尾部相关性
    combined_weight: float = 0.0          # 组合权重
    
    # 元数据
    last_update: datetime = field(default_factory=datetime.now)
    n_samples: int = 0


@dataclass
class CorrelationRegimeEmbedding:
    """
    相关性 Regime Embedding
    
    输出给下游模块使用
    """
    # 核心向量
    vector: np.ndarray                    # Graph embedding vector
    
    # 系统性风险得分
    systemic_risk_score: float = 0.0      # 0-1
    
    # 相关性状态
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    min_correlation: float = 0.0
    
    # 尾部相关性
    avg_tail_correlation: float = 0.0
    tail_correlation_spike: bool = False
    
    # Regime 判断
    regime: str = "normal"                # normal, compressed, exploded
    is_systemic_risk: bool = False
    
    # 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    node_count: int = 0
    edge_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector": self.vector.tolist() if isinstance(self.vector, np.ndarray) else self.vector,
            "systemic_risk_score": self.systemic_risk_score,
            "avg_correlation": self.avg_correlation,
            "max_correlation": self.max_correlation,
            "avg_tail_correlation": self.avg_tail_correlation,
            "tail_correlation_spike": self.tail_correlation_spike,
            "regime": self.regime,
            "is_systemic_risk": self.is_systemic_risk,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# CORRELATION CALCULATOR
# =============================================================================

class CorrelationCalculator:
    """相关性计算器"""
    
    def __init__(self, config: CorrelationGraphConfig):
        self.config = config
    
    def compute_rolling_correlation(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        window: int = None,
    ) -> float:
        """计算滚动相关性"""
        window = window or self.config.rolling_window
        
        if len(returns1) < window or len(returns2) < window:
            window = min(len(returns1), len(returns2))
        
        if window < 5:
            return 0.0
        
        r1 = returns1[-window:]
        r2 = returns2[-window:]
        
        # Pearson correlation
        mean1, mean2 = np.mean(r1), np.mean(r2)
        std1, std2 = np.std(r1), np.std(r2)
        
        if std1 == 0 or std2 == 0:
            return 0.0
        
        correlation = np.mean((r1 - mean1) * (r2 - mean2)) / (std1 * std2)
        
        return float(np.clip(correlation, -1, 1))
    
    def compute_tail_correlation(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        percentile: float = None,
    ) -> float:
        """
        计算尾部相关性
        
        仅在两个资产都处于尾部 (下跌) 时计算相关性
        这对于识别熊市系统性下行非常重要
        """
        percentile = percentile or self.config.tail_percentile
        window = self.config.tail_window
        
        if len(returns1) < window or len(returns2) < window:
            window = min(len(returns1), len(returns2))
        
        if window < 10:
            return 0.0
        
        r1 = returns1[-window:]
        r2 = returns2[-window:]
        
        # 找到尾部样本
        threshold1 = np.percentile(r1, percentile * 100)
        threshold2 = np.percentile(r2, percentile * 100)
        
        # 两者都在尾部的时刻
        tail_mask = (r1 <= threshold1) & (r2 <= threshold2)
        
        if tail_mask.sum() < 3:
            return 0.0
        
        tail_r1 = r1[tail_mask]
        tail_r2 = r2[tail_mask]
        
        # 计算尾部相关性
        mean1, mean2 = np.mean(tail_r1), np.mean(tail_r2)
        std1, std2 = np.std(tail_r1), np.std(tail_r2)
        
        if std1 == 0 or std2 == 0:
            return 1.0  # 完全同步下跌
        
        correlation = np.mean((tail_r1 - mean1) * (tail_r2 - mean2)) / (std1 * std2)
        
        return float(np.clip(correlation, -1, 1))


# =============================================================================
# CORRELATION GRAPH BUILDER
# =============================================================================

class CorrelationGraphBuilder:
    """
    相关性图构建器
    
    维护跨资产相关性图结构
    """
    
    def __init__(self, config: CorrelationGraphConfig = None):
        self.config = config or CorrelationGraphConfig()
        
        # 节点
        self._nodes: Dict[str, GraphNode] = {}
        for node_name in self.config.nodes:
            self._nodes[node_name] = GraphNode(name=node_name)
        
        # 边
        self._edges: Dict[Tuple[str, str], GraphEdge] = {}
        
        # 相关性计算器
        self._calc = CorrelationCalculator(self.config)
        
        # 统计
        self._update_count = 0
        self._last_full_update = datetime.now()
        
        logger.info(f"CorrelationGraphBuilder initialized with {len(self._nodes)} nodes")
    
    def update_node(self, name: str, ret: float, price: float):
        """更新节点数据"""
        if name not in self._nodes:
            self._nodes[name] = GraphNode(name=name)
        
        self._nodes[name].add_return(ret, price)
        self._update_count += 1
        
        # 定期更新边
        if self._update_count % self.config.update_interval_bars == 0:
            self._update_all_edges()
    
    def _update_all_edges(self):
        """更新所有边"""
        node_names = list(self._nodes.keys())
        
        for i in range(len(node_names)):
            for j in range(i + 1, len(node_names)):
                self._update_edge(node_names[i], node_names[j])
        
        self._last_full_update = datetime.now()
    
    def _update_edge(self, source: str, target: str):
        """更新单条边"""
        edge_key = (source, target)
        
        if edge_key not in self._edges:
            self._edges[edge_key] = GraphEdge(source=source, target=target)
        
        edge = self._edges[edge_key]
        
        # 获取收益率序列
        returns1 = np.array(self._nodes[source].returns)
        returns2 = np.array(self._nodes[target].returns)
        
        if len(returns1) < 10 or len(returns2) < 10:
            return
        
        # 计算相关性
        edge.rolling_correlation = self._calc.compute_rolling_correlation(returns1, returns2)
        edge.tail_correlation = self._calc.compute_tail_correlation(returns1, returns2)
        
        # 组合权重 (尾部相关性权重更高，熊市偏向)
        edge.combined_weight = 0.4 * edge.rolling_correlation + 0.6 * edge.tail_correlation
        
        edge.n_samples = min(len(returns1), len(returns2))
        edge.last_update = datetime.now()
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """获取邻接矩阵"""
        n = len(self._nodes)
        node_names = list(self._nodes.keys())
        node_idx = {name: i for i, name in enumerate(node_names)}
        
        adj = np.zeros((n, n))
        
        for (source, target), edge in self._edges.items():
            i, j = node_idx.get(source), node_idx.get(target)
            if i is not None and j is not None:
                weight = edge.combined_weight
                if abs(weight) >= self.config.edge_threshold:
                    adj[i, j] = weight
                    adj[j, i] = weight
        
        return adj
    
    def get_node_features(self) -> np.ndarray:
        """获取节点特征矩阵"""
        features = []
        for node_name in self._nodes.keys():
            node = self._nodes[node_name]
            features.append([
                node.volatility,
                node.momentum,
                node.tail_risk,
            ])
        return np.array(features)
    
    def compute_systemic_risk_score(self) -> float:
        """
        计算系统性风险得分
        
        基于:
        1. 平均相关性
        2. 尾部相关性 spike
        3. 全部资产同向下跌
        """
        if not self._edges:
            return 0.0
        
        correlations = [e.rolling_correlation for e in self._edges.values()]
        tail_correlations = [e.tail_correlation for e in self._edges.values()]
        
        avg_corr = np.mean(correlations)
        max_corr = np.max(correlations) if correlations else 0
        avg_tail = np.mean(tail_correlations) if tail_correlations else 0
        
        # 系统性风险 = 高相关性 + 高尾部相关性
        risk_score = 0.3 * avg_corr + 0.4 * max_corr + 0.3 * avg_tail
        
        # 如果所有节点都在下跌，额外增加风险
        all_down = all(
            node.momentum < 0 
            for node in self._nodes.values()
            if len(node.returns) > 0
        )
        
        if all_down:
            risk_score = min(risk_score + 0.2, 1.0)
        
        return float(np.clip(risk_score, 0, 1))
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        correlations = [e.rolling_correlation for e in self._edges.values()]
        tail_correlations = [e.tail_correlation for e in self._edges.values()]
        
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "avg_correlation": float(np.mean(correlations)) if correlations else 0,
            "max_correlation": float(np.max(correlations)) if correlations else 0,
            "avg_tail_correlation": float(np.mean(tail_correlations)) if tail_correlations else 0,
            "systemic_risk": self.compute_systemic_risk_score(),
            "last_update": self._last_full_update.isoformat(),
        }


# =============================================================================
# GNN ENCODER (Simplified)
# =============================================================================

class GNNEncoder:
    """
    GNN 编码器 (简化实现)
    
    将图结构编码为向量表示
    
    实际部署可使用 PyTorch Geometric 替换
    """
    
    def __init__(self, node_dim: int = 3, hidden_dim: int = 32, output_dim: int = 64):
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 简化：随机初始化权重
        np.random.seed(42)
        self._W1 = np.random.randn(node_dim, hidden_dim) * 0.1
        self._W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    
    def encode(
        self,
        node_features: np.ndarray,
        adjacency: np.ndarray,
    ) -> np.ndarray:
        """
        编码图结构
        
        简化的 GCN: H' = σ(D^(-1/2) A D^(-1/2) H W)
        这里简化为: H' = σ(A_norm H W)
        
        Args:
            node_features: shape (n_nodes, node_dim)
            adjacency: shape (n_nodes, n_nodes)
        
        Returns:
            graph_embedding: shape (output_dim,)
        """
        n_nodes = node_features.shape[0]
        
        # 归一化邻接矩阵
        degree = np.sum(adjacency, axis=1) + 1e-8
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        A_norm = D_inv_sqrt @ adjacency @ D_inv_sqrt
        
        # 添加自环
        A_norm = A_norm + np.eye(n_nodes)
        
        # Layer 1
        H1 = np.tanh(A_norm @ node_features @ self._W1)
        
        # Layer 2
        H2 = A_norm @ H1 @ self._W2
        
        # Global pooling (mean)
        graph_embedding = np.mean(H2, axis=0)
        
        # Normalize
        graph_embedding = graph_embedding / (np.linalg.norm(graph_embedding) + 1e-8)
        
        return graph_embedding


# =============================================================================
# REGIME GRAPH EMBEDDING
# =============================================================================

class RegimeGraphEmbedding:
    """
    Regime 图 Embedding
    
    整合图结构和系统性风险，输出 CorrelationRegimeEmbedding
    """
    
    def __init__(self, config: CorrelationGraphConfig = None):
        self.config = config or CorrelationGraphConfig()
        
        self._graph_builder = CorrelationGraphBuilder(self.config)
        self._gnn_encoder = GNNEncoder()
        
        # 历史 embedding
        self._history: deque = deque(maxlen=50)
        
        # 尾部相关性历史 (用于检测 spike)
        self._tail_corr_history: deque = deque(maxlen=30)
    
    def update(self, asset_returns: Dict[str, Tuple[float, float]]):
        """
        更新图结构
        
        Args:
            asset_returns: {asset: (return, price)}
        """
        for asset, (ret, price) in asset_returns.items():
            self._graph_builder.update_node(asset, ret, price)
    
    def embed(self) -> CorrelationRegimeEmbedding:
        """
        生成图 embedding
        """
        # 获取图结构
        adj = self._graph_builder.get_adjacency_matrix()
        node_features = self._graph_builder.get_node_features()
        
        # GNN 编码
        graph_vector = self._gnn_encoder.encode(node_features, adj)
        
        # 计算统计
        status = self._graph_builder.get_status()
        
        avg_corr = status["avg_correlation"]
        max_corr = status["max_correlation"]
        avg_tail_corr = status["avg_tail_correlation"]
        systemic_risk = status["systemic_risk"]
        
        # 检测尾部相关性 spike
        self._tail_corr_history.append(avg_tail_corr)
        tail_spike = False
        if len(self._tail_corr_history) >= 10:
            recent_tail = list(self._tail_corr_history)[-5:]
            historical_tail = list(self._tail_corr_history)[:-5]
            if historical_tail:
                historical_mean = np.mean(historical_tail)
                if np.mean(recent_tail) > historical_mean + 0.2:
                    tail_spike = True
        
        # 判断 regime
        if avg_corr > 0.7 or systemic_risk > self.config.systemic_threshold:
            regime = "exploded"
        elif avg_corr < 0.3:
            regime = "compressed"
        else:
            regime = "normal"
        
        # 构建输出
        embedding = CorrelationRegimeEmbedding(
            vector=graph_vector,
            systemic_risk_score=systemic_risk,
            avg_correlation=avg_corr,
            max_correlation=max_corr,
            min_correlation=float(np.min([e.rolling_correlation for e in self._graph_builder._edges.values()])) if self._graph_builder._edges else 0,
            avg_tail_correlation=avg_tail_corr,
            tail_correlation_spike=tail_spike,
            regime=regime,
            is_systemic_risk=systemic_risk > self.config.systemic_threshold,
            node_count=status["node_count"],
            edge_count=status["edge_count"],
        )
        
        self._history.append(embedding)
        
        # 日志
        logger.info(
            f"CorrelationRegimeEmbedding: regime={regime} "
            f"systemic_risk={systemic_risk:.2f} avg_corr={avg_corr:.2f} "
            f"tail_spike={tail_spike}"
        )
        
        return embedding
    
    def get_graph_builder(self) -> CorrelationGraphBuilder:
        """获取图构建器"""
        return self._graph_builder


# =============================================================================
# SINGLETON
# =============================================================================

_regime_graph_embedding: Optional[RegimeGraphEmbedding] = None


def get_regime_graph_embedding() -> RegimeGraphEmbedding:
    """获取单例"""
    global _regime_graph_embedding
    if _regime_graph_embedding is None:
        _regime_graph_embedding = RegimeGraphEmbedding()
    return _regime_graph_embedding


def reset_regime_graph_embedding():
    """重置单例"""
    global _regime_graph_embedding
    _regime_graph_embedding = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CorrelationGraphConfig',
    'GraphNode',
    'GraphEdge',
    'CorrelationRegimeEmbedding',
    'CorrelationCalculator',
    'CorrelationGraphBuilder',
    'GNNEncoder',
    'RegimeGraphEmbedding',
    'get_regime_graph_embedding',
    'reset_regime_graph_embedding',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing Correlation Graph Builder")
    print("=" * 60)
    
    # 创建图
    config = CorrelationGraphConfig()
    graph = RegimeGraphEmbedding(config)
    
    # 模拟熊市数据 (高相关性下跌)
    print("\n1. Simulating bear market (high correlation downside)...")
    
    np.random.seed(42)
    
    for t in range(50):
        # 系统性下跌
        base_return = -0.01 * (1 + 0.3 * np.random.randn())
        
        returns = {
            "BTC": (base_return + 0.002 * np.random.randn(), 50000 * (1 + t * base_return)),
            "ETH": (base_return * 1.2 + 0.003 * np.random.randn(), 3000 * (1 + t * base_return)),
            "SOL": (base_return * 1.5 + 0.005 * np.random.randn(), 100 * (1 + t * base_return)),
            "dominance": (0.001 + 0.001 * np.random.randn(), 45),  # BTC dominance 上升
            "total_mcap": (base_return + 0.002 * np.random.randn(), 2e12),
        }
        
        graph.update(returns)
    
    # 生成 embedding
    print("\n2. Generating embedding...")
    embedding = graph.embed()
    
    print(f"\n3. Result:")
    print(f"   regime: {embedding.regime}")
    print(f"   systemic_risk_score: {embedding.systemic_risk_score:.2f}")
    print(f"   avg_correlation: {embedding.avg_correlation:.2f}")
    print(f"   max_correlation: {embedding.max_correlation:.2f}")
    print(f"   avg_tail_correlation: {embedding.avg_tail_correlation:.2f}")
    print(f"   tail_correlation_spike: {embedding.tail_correlation_spike}")
    print(f"   is_systemic_risk: {embedding.is_systemic_risk}")
    print(f"   vector shape: {embedding.vector.shape}")
    
    # 模拟牛市数据 (低相关性)
    print("\n4. Simulating bull market (lower correlation)...")
    
    graph2 = RegimeGraphEmbedding(config)
    
    for t in range(50):
        returns = {
            "BTC": (0.01 + 0.01 * np.random.randn(), 50000),
            "ETH": (0.015 + 0.02 * np.random.randn(), 3000),
            "SOL": (-0.005 + 0.03 * np.random.randn(), 100),  # 独立走势
            "dominance": (-0.002 + 0.001 * np.random.randn(), 42),
            "total_mcap": (0.012 + 0.01 * np.random.randn(), 2.5e12),
        }
        
        graph2.update(returns)
    
    embedding2 = graph2.embed()
    
    print(f"\n5. Bull market result:")
    print(f"   regime: {embedding2.regime}")
    print(f"   systemic_risk_score: {embedding2.systemic_risk_score:.2f}")
    print(f"   avg_correlation: {embedding2.avg_correlation:.2f}")
    
    print("\n✓ Correlation Graph Builder test complete")

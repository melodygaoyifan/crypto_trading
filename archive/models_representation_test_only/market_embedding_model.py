"""
================================================================================
HMATS v5.2.1 SOTA - 自监督市场表示模型
Market Embedding Model (Self-supervised)
================================================================================

Purpose: 学习统一的市场状态表示，替代手工拼 state

核心设计:
- 输入: price / LOB / flow / sentiment / vol
- 自监督目标: masking / contrastive / temporal consistency
- 输出: MarketEmbedding (128-d vector + confidence + regime_hint)

偏向性 (关键):
- 偏向"下行 + 崩塌 + 波动扩张"状态区分能力
- 不是预测上涨

接线:
- ModelAlphaAgent
- VolatilityAlphaAgent
- RiskAgent
- Execution (advisory only)

================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from collections import deque
import json

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MarketEmbeddingConfig:
    """市场表示配置"""
    
    # Embedding 维度
    embedding_dim: int = 128
    
    # 输入特征
    price_features: int = 16      # OHLCV + returns + momentum
    lob_features: int = 24        # spread, depth, imbalance, VPIN
    flow_features: int = 16       # exchange flow, funding, OI
    sentiment_features: int = 8   # fear index, social vol
    vol_features: int = 16        # realized vol multi-window
    
    # 编码器
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    
    # 时间窗口
    lookback_bars: int = 96       # 4H bars = 16 days
    
    # 自监督目标权重
    masking_weight: float = 0.4
    contrastive_weight: float = 0.3
    temporal_weight: float = 0.3
    
    # 熊市偏向
    downside_emphasis: float = 1.5    # 对下行状态的学习权重
    crash_sensitivity: float = 2.0    # 对崩塌模式的敏感度
    
    # Shadow mode
    shadow_mode: bool = True
    confidence_threshold: float = 0.6


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class RegimeHint(Enum):
    """Regime 提示 (embedding 模型输出的软标签)"""
    UNKNOWN = "unknown"
    SLOW_BEAR = "slow_bear"
    CRASH_CASCADE = "crash_cascade"
    BEAR_RALLY = "bear_rally"
    SIDEWAYS = "sideways"
    BULL_TREND = "bull_trend"


@dataclass
class MarketEmbedding:
    """
    市场状态表示
    
    这是整个系统的统一状态表示，替代手工 state 拼接
    """
    # 核心向量
    vector: np.ndarray              # 128-d embedding
    
    # 置信度
    confidence: float               # 0-1
    
    # Regime 提示 (可选)
    regime_hint: RegimeHint = RegimeHint.UNKNOWN
    regime_probs: Dict[str, float] = field(default_factory=dict)
    
    # 特定维度解释
    downside_score: float = 0.0     # 下行风险得分
    crash_score: float = 0.0        # 崩塌风险得分
    vol_expansion_score: float = 0.0  # 波动扩张得分
    
    # 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    asset: str = "BTC"
    model_version: str = "v1.0"
    
    # 来源特征摘要
    feature_summary: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector": self.vector.tolist() if isinstance(self.vector, np.ndarray) else self.vector,
            "confidence": self.confidence,
            "regime_hint": self.regime_hint.value,
            "regime_probs": self.regime_probs,
            "downside_score": self.downside_score,
            "crash_score": self.crash_score,
            "vol_expansion_score": self.vol_expansion_score,
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
        }
    
    @property
    def is_bearish(self) -> bool:
        """是否偏熊"""
        return self.regime_hint in [RegimeHint.SLOW_BEAR, RegimeHint.CRASH_CASCADE]
    
    @property
    def is_high_risk(self) -> bool:
        """是否高风险状态"""
        return self.crash_score > 0.6 or self.vol_expansion_score > 0.7


@dataclass
class RawMarketFeatures:
    """
    原始市场特征输入
    
    C2 FIX: 添加数据有效性标记，区分真实零值和数据缺失
    """
    timestamp: datetime
    asset: str
    
    # Price features
    prices: np.ndarray              # OHLCV sequence
    returns: np.ndarray             # return sequence
    
    # LOB features
    spread_bps: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    depth_imbalance: float = 0.0
    vpin: float = 0.0
    lob_sequence: Optional[np.ndarray] = None
    
    # Flow features (C2 FIX: 添加有效性标记)
    exchange_inflow: float = 0.0
    exchange_outflow: float = 0.0
    funding_rate: float = 0.0
    open_interest: float = 0.0
    flow_sequence: Optional[np.ndarray] = None
    
    # C2 FIX: 数据有效性标记
    inflow_data_valid: bool = True      # 链上 inflow 数据是否有效
    outflow_data_valid: bool = True     # 链上 outflow 数据是否有效
    flow_invalid_reason: str = ""       # e.g. "API_TIMEOUT", "RATE_LIMIT", "MISSING"
    flow_source_timestamp: Optional[datetime] = None  # 数据源时间戳
    
    # Sentiment features
    fear_index: float = 50.0
    social_volume: float = 0.0
    sentiment_vol: float = 0.0
    
    # Vol features
    realized_vol_1h: float = 0.0
    realized_vol_4h: float = 0.0
    realized_vol_24h: float = 0.0
    vol_of_vol: float = 0.0
    vol_percentile: float = 50.0
    
    def is_flow_data_valid(self) -> bool:
        """C2 FIX: 检查 flow 数据是否全部有效"""
        return self.inflow_data_valid and self.outflow_data_valid
    
    def get_flow_validity_log(self) -> str:
        """C2 FIX: 生成 flow 数据有效性日志"""
        if self.is_flow_data_valid():
            return f"INFLOW={self.exchange_inflow:.2f} (VALID) OUTFLOW={self.exchange_outflow:.2f} (VALID)"
        else:
            return (
                f"INFLOW={self.exchange_inflow:.2f} ({'VALID' if self.inflow_data_valid else 'INVALID'}) "
                f"OUTFLOW={self.exchange_outflow:.2f} ({'VALID' if self.outflow_data_valid else 'INVALID'}) "
                f"reason={self.flow_invalid_reason}"
            )


# =============================================================================
# FEATURE ENCODER
# =============================================================================

class FeatureEncoder:
    """
    特征编码器
    
    将原始市场数据编码为标准化特征向量
    """
    
    def __init__(self, config: MarketEmbeddingConfig):
        self.config = config
        
        # 特征统计 (用于标准化)
        self._feature_means: Dict[str, float] = {}
        self._feature_stds: Dict[str, float] = {}
        self._n_samples = 0
    
    def encode(self, raw: RawMarketFeatures) -> np.ndarray:
        """
        编码原始特征为标准化向量
        
        Returns:
            np.ndarray: shape (total_features,)
        """
        features = []
        
        # 1. Price features
        price_feats = self._encode_price(raw)
        features.extend(price_feats)
        
        # 2. LOB features
        lob_feats = self._encode_lob(raw)
        features.extend(lob_feats)
        
        # 3. Flow features
        flow_feats = self._encode_flow(raw)
        features.extend(flow_feats)
        
        # 4. Sentiment features
        sent_feats = self._encode_sentiment(raw)
        features.extend(sent_feats)
        
        # 5. Vol features
        vol_feats = self._encode_vol(raw)
        features.extend(vol_feats)
        
        return np.array(features, dtype=np.float32)
    
    def _encode_price(self, raw: RawMarketFeatures) -> List[float]:
        """编码价格特征"""
        feats = []
        
        if raw.returns is not None and len(raw.returns) > 0:
            # 多时间窗口收益
            feats.append(np.mean(raw.returns[-4:]))   # 1H
            feats.append(np.mean(raw.returns[-24:]))  # 6H
            feats.append(np.mean(raw.returns[-96:]) if len(raw.returns) >= 96 else np.mean(raw.returns))  # 24H
            
            # 下行收益 (熊市偏向)
            neg_returns = raw.returns[raw.returns < 0]
            feats.append(np.mean(neg_returns) if len(neg_returns) > 0 else 0)
            
            # 最大回撤
            cumret = np.cumprod(1 + raw.returns)
            running_max = np.maximum.accumulate(cumret)
            drawdown = (cumret - running_max) / running_max
            feats.append(np.min(drawdown))
            
            # 动量
            feats.append(np.sum(raw.returns[-12:]))
            feats.append(np.sum(raw.returns[-48:]))
        else:
            feats.extend([0.0] * 7)
        
        # Padding
        while len(feats) < self.config.price_features:
            feats.append(0.0)
        
        return feats[:self.config.price_features]
    
    def _encode_lob(self, raw: RawMarketFeatures) -> List[float]:
        """编码订单簿特征"""
        feats = [
            raw.spread_bps / 100,  # 归一化
            np.tanh(raw.depth_imbalance),
            np.log1p(raw.bid_depth) / 20,
            np.log1p(raw.ask_depth) / 20,
            raw.vpin,
        ]
        
        # LOB 序列特征
        if raw.lob_sequence is not None and len(raw.lob_sequence) > 0:
            feats.append(np.mean(raw.lob_sequence))
            feats.append(np.std(raw.lob_sequence))
        else:
            feats.extend([0.0, 0.0])
        
        while len(feats) < self.config.lob_features:
            feats.append(0.0)
        
        return feats[:self.config.lob_features]
    
    def _encode_flow(self, raw: RawMarketFeatures) -> List[float]:
        """
        编码资金流特征
        
        C2 FIX: 处理数据有效性
        - 若 data_valid=False，使用中性值而非原始值
        - 不允许把 value=0 当作真实信号
        """
        # C2 FIX: 检查数据有效性
        if not raw.is_flow_data_valid():
            logger.warning(
                f"[FLOW_DATA_INVALID] asset={raw.asset} reason={raw.flow_invalid_reason} "
                f"- using neutral encoding (fail-closed)"
            )
            # 返回中性值，不使用可能无效的数据
            return [0.0] * self.config.flow_features
        
        # C2 FIX: 日志记录有效数据
        logger.debug(f"[FLOW_DATA_VALID] {raw.get_flow_validity_log()}")
        
        net_flow = raw.exchange_inflow - raw.exchange_outflow
        
        feats = [
            np.tanh(net_flow / 1e8),  # 归一化到 -1, 1
            np.tanh(raw.exchange_inflow / 1e8),
            raw.funding_rate * 100,  # 放大
            np.log1p(raw.open_interest) / 25,
        ]
        
        while len(feats) < self.config.flow_features:
            feats.append(0.0)
        
        return feats[:self.config.flow_features]
    
    def _encode_sentiment(self, raw: RawMarketFeatures) -> List[float]:
        """编码情绪特征"""
        # Fear index: 0-100, 归一化到 -1, 1 (低=恐慌=-1)
        fear_norm = (raw.fear_index - 50) / 50
        
        feats = [
            fear_norm,
            np.tanh(raw.social_volume / 1000),
            raw.sentiment_vol,
        ]
        
        while len(feats) < self.config.sentiment_features:
            feats.append(0.0)
        
        return feats[:self.config.sentiment_features]
    
    def _encode_vol(self, raw: RawMarketFeatures) -> List[float]:
        """编码波动率特征"""
        feats = [
            raw.realized_vol_1h,
            raw.realized_vol_4h,
            raw.realized_vol_24h,
            raw.vol_of_vol,
            (raw.vol_percentile - 50) / 50,  # 归一化
        ]
        
        # Vol term structure
        if raw.realized_vol_1h > 0 and raw.realized_vol_24h > 0:
            feats.append(raw.realized_vol_1h / raw.realized_vol_24h - 1)  # 期限结构
        else:
            feats.append(0.0)
        
        while len(feats) < self.config.vol_features:
            feats.append(0.0)
        
        return feats[:self.config.vol_features]


# =============================================================================
# EMBEDDING NETWORK
# =============================================================================

class EmbeddingNetwork:
    """
    Embedding 网络 (简化实现，不依赖 PyTorch)
    
    实际部署可替换为 PyTorch 版本
    """
    
    def __init__(self, config: MarketEmbeddingConfig):
        self.config = config
        
        # 输入维度
        self.input_dim = (
            config.price_features +
            config.lob_features +
            config.flow_features +
            config.sentiment_features +
            config.vol_features
        )
        
        # 简化：使用随机初始化的权重矩阵
        # 实际应用中应使用预训练权重
        np.random.seed(42)
        
        self._W1 = np.random.randn(self.input_dim, config.hidden_dim) * 0.1
        self._b1 = np.zeros(config.hidden_dim)
        
        self._W2 = np.random.randn(config.hidden_dim, config.hidden_dim) * 0.1
        self._b2 = np.zeros(config.hidden_dim)
        
        self._W_out = np.random.randn(config.hidden_dim, config.embedding_dim) * 0.1
        self._b_out = np.zeros(config.embedding_dim)
        
        # Regime 分类头
        self._W_regime = np.random.randn(config.embedding_dim, 5) * 0.1  # 5 regimes
        self._b_regime = np.zeros(5)
        
        # 风险评分头 (熊市偏向)
        self._W_risk = np.random.randn(config.embedding_dim, 3) * 0.1  # downside, crash, vol_exp
        self._b_risk = np.zeros(3)
        
        logger.info(f"EmbeddingNetwork initialized: input={self.input_dim}, embedding={config.embedding_dim}")
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        前向传播
        
        Returns:
            (embedding, regime_logits, risk_scores)
        """
        # Layer 1
        h1 = np.tanh(x @ self._W1 + self._b1)
        
        # Dropout (inference时不用)
        
        # Layer 2
        h2 = np.tanh(h1 @ self._W2 + self._b2)
        
        # Embedding output
        embedding = h2 @ self._W_out + self._b_out
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)  # L2 normalize
        
        # Regime classification
        regime_logits = embedding @ self._W_regime + self._b_regime
        
        # Risk scores (sigmoid)
        risk_raw = embedding @ self._W_risk + self._b_risk
        risk_scores = 1 / (1 + np.exp(-risk_raw))  # sigmoid
        
        return embedding, regime_logits, risk_scores
    
    def load_weights(self, path: str) -> bool:
        """加载预训练权重"""
        try:
            weights = np.load(path, allow_pickle=True).item()
            self._W1 = weights['W1']
            self._b1 = weights['b1']
            self._W2 = weights['W2']
            self._b2 = weights['b2']
            self._W_out = weights['W_out']
            self._b_out = weights['b_out']
            self._W_regime = weights['W_regime']
            self._b_regime = weights['b_regime']
            self._W_risk = weights['W_risk']
            self._b_risk = weights['b_risk']
            logger.info(f"Loaded weights from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}")
            return False


# =============================================================================
# MARKET EMBEDDING MODEL
# =============================================================================

class MarketEmbeddingModel:
    """
    市场表示模型
    
    主入口，整合编码器和网络
    """
    
    def __init__(self, config: MarketEmbeddingConfig = None):
        self.config = config or MarketEmbeddingConfig()
        
        self._encoder = FeatureEncoder(self.config)
        self._network = EmbeddingNetwork(self.config)
        
        # 历史 embedding 缓存
        self._embedding_history: deque = deque(maxlen=100)
        
        # 统计
        self._stats = {
            "total_embeddings": 0,
            "bearish_count": 0,
            "high_risk_count": 0,
        }
        
        # Regime 映射
        self._regime_map = [
            RegimeHint.SLOW_BEAR,
            RegimeHint.CRASH_CASCADE,
            RegimeHint.BEAR_RALLY,
            RegimeHint.SIDEWAYS,
            RegimeHint.BULL_TREND,
        ]
        
        logger.info("MarketEmbeddingModel initialized")
    
    # 工程审计修复: 数据陈旧性阈值
    MAX_EMBEDDING_AGE_SEC = 300  # 5分钟
    
    def embed(self, raw: RawMarketFeatures) -> Optional[MarketEmbedding]:
        """
        生成市场 embedding
        
        这是主入口
        
        工程审计修复 C1: 添加数据陈旧性检测
        - 超过 MAX_EMBEDDING_AGE_SEC 的数据将返回 None
        - 调用方必须处理 None 情况
        """
        # C1 FIX: 检查数据陈旧性 (Fail-Closed)
        # [P165] Was `datetime.utcnow() - raw.timestamp`: a naive-vs-aware
        # subtraction that raised TypeError for every tz-aware caller — i.e.
        # the staleness guard could not run at all once the P40/P97 tz-aware
        # sweep made timestamps aware. Same class as P40/P97. Treat a naive
        # timestamp as UTC rather than rejecting it, so both shapes work.
        _ts = raw.timestamp
        if _ts.tzinfo is None:
            _ts = _ts.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - _ts).total_seconds()
        if age_seconds > self.MAX_EMBEDDING_AGE_SEC:
            # M1 FIX: 结构化拒绝日志
            logger.warning(
                f"[EMBEDDING_REJECTED] type=MarketEmbedding reason=STALE "
                f"age={age_seconds:.0f}s max={self.MAX_EMBEDDING_AGE_SEC}s "
                f"asset={raw.asset}"
            )
            self._stats["stale_rejections"] = self._stats.get("stale_rejections", 0) + 1
            return None
        
        # M1 FIX: 检查 flow 数据有效性并记录
        if not raw.is_flow_data_valid():
            logger.warning(
                f"[EMBEDDING_DEGRADED] type=MarketEmbedding reason=INVALID_FLOW "
                f"asset={raw.asset} flow_reason={raw.flow_invalid_reason}"
            )
        
        # 1. 编码特征
        features = self._encoder.encode(raw)
        
        # 2. 网络前向
        vector, regime_logits, risk_scores = self._network.forward(features)
        
        # 3. Regime 概率
        regime_probs = self._softmax(regime_logits)
        regime_idx = np.argmax(regime_probs)
        regime_hint = self._regime_map[regime_idx]
        
        # 4. 构建输出
        embedding = MarketEmbedding(
            vector=vector,
            confidence=float(np.max(regime_probs)),
            regime_hint=regime_hint,
            regime_probs={r.value: float(regime_probs[i]) for i, r in enumerate(self._regime_map)},
            downside_score=float(risk_scores[0]),
            crash_score=float(risk_scores[1]),
            vol_expansion_score=float(risk_scores[2]),
            timestamp=raw.timestamp,
            asset=raw.asset,
            feature_summary={
                "vol_percentile": raw.vol_percentile,
                "fear_index": raw.fear_index,
                "spread_bps": raw.spread_bps,
            },
        )
        
        # 5. 更新统计
        self._embedding_history.append(embedding)
        self._stats["total_embeddings"] += 1
        if embedding.is_bearish:
            self._stats["bearish_count"] += 1
        if embedding.is_high_risk:
            self._stats["high_risk_count"] += 1
        
        # 6. 日志
        self._log_embedding(embedding)
        
        return embedding
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _log_embedding(self, emb: MarketEmbedding):
        """记录日志"""
        logger.info(
            f"MarketEmbedding: asset={emb.asset} regime={emb.regime_hint.value} "
            f"conf={emb.confidence:.2f} downside={emb.downside_score:.2f} "
            f"crash={emb.crash_score:.2f} vol_exp={emb.vol_expansion_score:.2f}"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "stats": self._stats,
            "last_embedding": self._embedding_history[-1].to_dict() if self._embedding_history else None,
            "config": {
                "embedding_dim": self.config.embedding_dim,
                "shadow_mode": self.config.shadow_mode,
            },
        }


# =============================================================================
# EMBEDDING REGISTRY
# =============================================================================

class EmbeddingRegistry:
    """
    Embedding 注册表
    
    管理多资产的 embedding，提供统一访问接口
    """
    
    def __init__(self):
        self._models: Dict[str, MarketEmbeddingModel] = {}
        self._latest_embeddings: Dict[str, MarketEmbedding] = {}
        
        # 默认配置
        self._default_config = MarketEmbeddingConfig()
    
    def get_model(self, asset: str) -> MarketEmbeddingModel:
        """获取或创建资产的 embedding 模型"""
        if asset not in self._models:
            self._models[asset] = MarketEmbeddingModel(self._default_config)
        return self._models[asset]
    
    def update_embedding(self, asset: str, raw: RawMarketFeatures) -> MarketEmbedding:
        """更新资产的 embedding"""
        model = self.get_model(asset)
        embedding = model.embed(raw)
        self._latest_embeddings[asset] = embedding
        return embedding
    
    def get_latest(self, asset: str) -> Optional[MarketEmbedding]:
        """获取最新 embedding"""
        return self._latest_embeddings.get(asset)
    
    def get_all_latest(self) -> Dict[str, MarketEmbedding]:
        """获取所有资产的最新 embedding"""
        return self._latest_embeddings.copy()
    
    def get_systemic_risk(self) -> float:
        """
        计算系统性风险
        
        基于所有资产的 crash_score 和 downside_score
        """
        if not self._latest_embeddings:
            return 0.0
        
        crash_scores = [e.crash_score for e in self._latest_embeddings.values()]
        downside_scores = [e.downside_score for e in self._latest_embeddings.values()]
        
        # 系统性风险 = 平均 crash + 最大 downside
        systemic = np.mean(crash_scores) * 0.6 + np.max(downside_scores) * 0.4
        
        return float(systemic)


# =============================================================================
# SINGLETON
# =============================================================================

_embedding_registry: Optional[EmbeddingRegistry] = None


def get_embedding_registry() -> EmbeddingRegistry:
    """获取 embedding 注册表单例"""
    global _embedding_registry
    if _embedding_registry is None:
        _embedding_registry = EmbeddingRegistry()
    return _embedding_registry


def reset_embedding_registry():
    """重置单例"""
    global _embedding_registry
    _embedding_registry = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MarketEmbeddingConfig',
    'RegimeHint',
    'MarketEmbedding',
    'RawMarketFeatures',
    'FeatureEncoder',
    'EmbeddingNetwork',
    'MarketEmbeddingModel',
    'EmbeddingRegistry',
    'get_embedding_registry',
    'reset_embedding_registry',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing MarketEmbeddingModel")
    print("=" * 60)
    
    # 创建模型
    config = MarketEmbeddingConfig()
    model = MarketEmbeddingModel(config)
    
    # 模拟熊市数据
    print("\n1. Bear market scenario...")
    raw_bear = RawMarketFeatures(
        timestamp=datetime.now(),
        asset="BTC",
        prices=np.array([100, 99, 97, 95, 92, 90]),
        returns=np.array([-0.01, -0.02, -0.02, -0.03, -0.02]),
        spread_bps=25.0,
        bid_depth=200000,
        ask_depth=600000,
        depth_imbalance=-0.5,
        vpin=0.8,
        exchange_inflow=50000000,
        exchange_outflow=10000000,
        funding_rate=-0.001,
        fear_index=15.0,
        sentiment_vol=0.6,
        realized_vol_1h=0.8,
        realized_vol_4h=0.7,
        realized_vol_24h=0.5,
        vol_of_vol=0.6,
        vol_percentile=90.0,
    )
    
    emb_bear = model.embed(raw_bear)
    
    print(f"\nBear market embedding:")
    print(f"  regime_hint: {emb_bear.regime_hint.value}")
    print(f"  confidence: {emb_bear.confidence:.2f}")
    print(f"  downside_score: {emb_bear.downside_score:.2f}")
    print(f"  crash_score: {emb_bear.crash_score:.2f}")
    print(f"  vol_expansion_score: {emb_bear.vol_expansion_score:.2f}")
    print(f"  is_bearish: {emb_bear.is_bearish}")
    print(f"  is_high_risk: {emb_bear.is_high_risk}")
    print(f"  vector shape: {emb_bear.vector.shape}")
    
    # 模拟牛市数据
    print("\n2. Bull market scenario...")
    raw_bull = RawMarketFeatures(
        timestamp=datetime.now(),
        asset="BTC",
        prices=np.array([100, 101, 103, 105, 108, 110]),
        returns=np.array([0.01, 0.02, 0.02, 0.03, 0.02]),
        spread_bps=3.0,
        bid_depth=800000,
        ask_depth=400000,
        depth_imbalance=0.4,
        vpin=0.2,
        exchange_inflow=10000000,
        exchange_outflow=30000000,
        funding_rate=0.001,
        fear_index=75.0,
        sentiment_vol=0.1,
        realized_vol_1h=0.2,
        realized_vol_4h=0.2,
        realized_vol_24h=0.25,
        vol_of_vol=0.1,
        vol_percentile=25.0,
    )
    
    emb_bull = model.embed(raw_bull)
    
    print(f"\nBull market embedding:")
    print(f"  regime_hint: {emb_bull.regime_hint.value}")
    print(f"  confidence: {emb_bull.confidence:.2f}")
    print(f"  downside_score: {emb_bull.downside_score:.2f}")
    print(f"  crash_score: {emb_bull.crash_score:.2f}")
    
    # 测试 Registry
    print("\n3. Testing EmbeddingRegistry...")
    registry = get_embedding_registry()
    
    registry.update_embedding("BTC", raw_bear)
    registry.update_embedding("ETH", raw_bear)
    
    systemic_risk = registry.get_systemic_risk()
    print(f"  Systemic risk: {systemic_risk:.2f}")
    
    print("\n✓ MarketEmbeddingModel test complete")

"""
================================================================================
HMATS v5.2.1 SOTA - 自监督预训练目标
Self-supervised Pretraining Objectives
================================================================================

Purpose: 定义 MarketEmbeddingModel 的自监督学习目标

目标:
1. Masking Objective - 掩码预测
2. Contrastive Objective - 对比学习
3. Temporal Consistency - 时序一致性

熊市偏向:
- 对下行状态的学习权重更高
- 对崩塌模式更敏感
- penalize tail loss

================================================================================
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# MASKING OBJECTIVE
# =============================================================================

@dataclass
class MaskingConfig:
    """掩码目标配置"""
    mask_ratio: float = 0.15          # 掩码比例
    mask_strategies: List[str] = field(default_factory=lambda: ["random", "block", "feature"])
    reconstruction_loss_type: str = "mse"
    
    # 熊市偏向
    downside_mask_weight: float = 2.0  # 对下行特征的掩码重建权重更高


class MaskingObjective:
    """
    掩码预测目标
    
    随机掩码部分输入特征，要求模型预测被掩码的值
    学习市场状态的内在结构
    """
    
    def __init__(self, config: MaskingConfig = None):
        self.config = config or MaskingConfig()
    
    def create_mask(self, features: np.ndarray, strategy: str = "random") -> Tuple[np.ndarray, np.ndarray]:
        """
        创建掩码
        
        Args:
            features: shape (batch, feature_dim) or (feature_dim,)
            strategy: "random", "block", "feature"
        
        Returns:
            (masked_features, mask)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        batch_size, feature_dim = features.shape
        mask = np.zeros_like(features, dtype=bool)
        
        if strategy == "random":
            # 随机掩码
            n_mask = int(feature_dim * self.config.mask_ratio)
            for i in range(batch_size):
                mask_indices = np.random.choice(feature_dim, n_mask, replace=False)
                mask[i, mask_indices] = True
        
        elif strategy == "block":
            # 块状掩码
            block_size = int(feature_dim * self.config.mask_ratio)
            for i in range(batch_size):
                start = np.random.randint(0, feature_dim - block_size + 1)
                mask[i, start:start + block_size] = True
        
        elif strategy == "feature":
            # 按特征类型掩码
            # 假设特征按顺序排列: price, lob, flow, sentiment, vol
            feature_groups = [16, 24, 16, 8, 16]  # 每组的维度
            group_idx = np.random.randint(0, len(feature_groups))
            
            start = sum(feature_groups[:group_idx])
            end = start + feature_groups[group_idx]
            mask[:, start:end] = True
        
        masked_features = features.copy()
        masked_features[mask] = 0.0  # 用 0 替换掩码位置
        
        return masked_features.squeeze(), mask.squeeze()
    
    def compute_loss(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        mask: np.ndarray,
        is_downside: bool = False,
    ) -> float:
        """
        计算重建损失
        
        Args:
            original: 原始特征
            reconstructed: 重建特征
            mask: 掩码
            is_downside: 是否为下行状态 (熊市偏向)
        
        Returns:
            loss value
        """
        # 只计算被掩码位置的损失
        masked_original = original[mask]
        masked_reconstructed = reconstructed[mask]
        
        if self.config.reconstruction_loss_type == "mse":
            loss = np.mean((masked_original - masked_reconstructed) ** 2)
        elif self.config.reconstruction_loss_type == "mae":
            loss = np.mean(np.abs(masked_original - masked_reconstructed))
        else:
            loss = np.mean((masked_original - masked_reconstructed) ** 2)
        
        # 熊市偏向: 对下行状态的重建错误惩罚更重
        if is_downside:
            loss *= self.config.downside_mask_weight
        
        return float(loss)


# =============================================================================
# CONTRASTIVE OBJECTIVE
# =============================================================================

@dataclass
class ContrastiveConfig:
    """对比学习配置"""
    temperature: float = 0.1
    positive_margin: float = 0.2
    negative_margin: float = 0.8
    
    # 熊市偏向
    crash_contrast_weight: float = 2.0  # 对崩塌样本的对比权重更高


class ContrastiveObjective:
    """
    对比学习目标
    
    让相似市场状态的 embedding 靠近，不同状态的 embedding 远离
    
    正样本: 相邻时间窗口、相同 regime
    负样本: 不同 regime、对立市场状态
    """
    
    def __init__(self, config: ContrastiveConfig = None):
        self.config = config or ContrastiveConfig()
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def compute_infonce_loss(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negatives: List[np.ndarray],
        is_crash: bool = False,
    ) -> float:
        """
        计算 InfoNCE loss
        
        Args:
            anchor: 锚点 embedding
            positive: 正样本 embedding
            negatives: 负样本 embedding 列表
            is_crash: 是否为崩塌样本 (熊市偏向)
        
        Returns:
            loss value
        """
        pos_sim = self.compute_similarity(anchor, positive) / self.config.temperature
        
        neg_sims = [
            self.compute_similarity(anchor, neg) / self.config.temperature
            for neg in negatives
        ]
        
        # 数值稳定的 log-sum-exp
        all_sims = [pos_sim] + neg_sims
        max_sim = max(all_sims)
        
        denominator = sum(np.exp(s - max_sim) for s in all_sims)
        numerator = np.exp(pos_sim - max_sim)
        
        loss = -np.log(numerator / denominator + 1e-8)
        
        # 熊市偏向: 对崩塌样本的对比损失更重要
        if is_crash:
            loss *= self.config.crash_contrast_weight
        
        return float(loss)
    
    def compute_triplet_loss(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray,
    ) -> float:
        """
        计算 Triplet loss
        
        max(0, d(anchor, positive) - d(anchor, negative) + margin)
        """
        pos_dist = 1 - self.compute_similarity(anchor, positive)
        neg_dist = 1 - self.compute_similarity(anchor, negative)
        
        margin = self.config.positive_margin
        
        loss = max(0, pos_dist - neg_dist + margin)
        
        return float(loss)
    
    def create_pairs(
        self,
        embeddings: List[np.ndarray],
        regimes: List[str],
    ) -> List[Tuple[int, int, int]]:
        """
        创建对比学习配对 (anchor, positive, negative)
        
        正样本规则:
        - 相同 regime
        - 时间相近
        
        负样本规则:
        - 不同 regime
        - 对立状态 (bear vs bull)
        """
        pairs = []
        n = len(embeddings)
        
        for i in range(n):
            # 找正样本: 相同 regime
            positives = [j for j in range(n) if j != i and regimes[j] == regimes[i]]
            
            # 找负样本: 不同 regime
            negatives = [j for j in range(n) if regimes[j] != regimes[i]]
            
            if positives and negatives:
                pos = np.random.choice(positives)
                neg = np.random.choice(negatives)
                pairs.append((i, pos, neg))
        
        return pairs


# =============================================================================
# TEMPORAL CONSISTENCY OBJECTIVE
# =============================================================================

@dataclass
class TemporalConfig:
    """时序一致性配置"""
    consistency_window: int = 4       # 一致性窗口 (4H bars)
    smoothness_weight: float = 0.5
    transition_penalty: float = 1.0
    
    # 熊市偏向
    crash_transition_weight: float = 2.0  # 对崩塌转换的检测更敏感


class TemporalConsistencyObjective:
    """
    时序一致性目标
    
    确保:
    1. 相邻时间的 embedding 平滑变化
    2. Regime 转换有明确的边界
    3. 崩塌转换被更准确识别
    """
    
    def __init__(self, config: TemporalConfig = None):
        self.config = config or TemporalConfig()
    
    def compute_smoothness_loss(
        self,
        embeddings: List[np.ndarray],
    ) -> float:
        """
        计算平滑性损失
        
        相邻 embedding 不应该跳变太大
        """
        if len(embeddings) < 2:
            return 0.0
        
        losses = []
        for i in range(len(embeddings) - 1):
            diff = embeddings[i + 1] - embeddings[i]
            loss = np.mean(diff ** 2)
            losses.append(loss)
        
        return float(np.mean(losses) * self.config.smoothness_weight)
    
    def compute_transition_loss(
        self,
        embeddings: List[np.ndarray],
        regimes: List[str],
    ) -> float:
        """
        计算转换损失
        
        Regime 转换时，embedding 应该有明确变化
        """
        if len(embeddings) < 2:
            return 0.0
        
        losses = []
        for i in range(len(embeddings) - 1):
            regime_changed = regimes[i] != regimes[i + 1]
            
            diff_norm = np.linalg.norm(embeddings[i + 1] - embeddings[i])
            
            if regime_changed:
                # 转换时应该有变化，如果变化太小则惩罚
                target_diff = 0.5
                loss = max(0, target_diff - diff_norm) ** 2
                
                # 如果是向崩塌转换，权重更高
                if regimes[i + 1] == "crash_cascade":
                    loss *= self.config.crash_transition_weight
            else:
                # 非转换时应该平滑
                loss = max(0, diff_norm - 0.2) ** 2
            
            losses.append(loss)
        
        return float(np.mean(losses) * self.config.transition_penalty)
    
    def compute_total_loss(
        self,
        embeddings: List[np.ndarray],
        regimes: List[str],
    ) -> Tuple[float, Dict[str, float]]:
        """计算总时序损失"""
        smoothness = self.compute_smoothness_loss(embeddings)
        transition = self.compute_transition_loss(embeddings, regimes)
        
        total = smoothness + transition
        
        return total, {
            "smoothness": smoothness,
            "transition": transition,
        }


# =============================================================================
# COMBINED PRETRAINING OBJECTIVE
# =============================================================================

@dataclass
class PretrainingConfig:
    """预训练配置"""
    masking_weight: float = 0.4
    contrastive_weight: float = 0.3
    temporal_weight: float = 0.3
    
    # 熊市偏向
    downside_emphasis: float = 1.5
    crash_sensitivity: float = 2.0
    
    # 尾部损失惩罚
    tail_loss_percentile: float = 0.95
    tail_loss_weight: float = 2.0


class CombinedPretrainingObjective:
    """
    组合预训练目标
    
    整合所有自监督目标，加入熊市偏向
    """
    
    def __init__(self, config: PretrainingConfig = None):
        self.config = config or PretrainingConfig()
        
        self.masking = MaskingObjective()
        self.contrastive = ContrastiveObjective()
        self.temporal = TemporalConsistencyObjective()
        
        # Simple linear reconstructor (feature_dim -> hidden -> feature_dim)
        self._reconstructor_weights = None
        self._reconstructor_bias = None
    
    def _reconstruct_features(self, masked_features: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Reconstruct masked features using simple interpolation + learned weights.
        
        For pretraining, uses a combination of:
        1. Local interpolation from unmasked neighbors
        2. Global mean imputation
        3. Optional learned linear projection
        """
        if masked_features.ndim == 1:
            masked_features = masked_features.reshape(1, -1)
            mask = mask.reshape(1, -1)
        
        reconstructed = masked_features.copy()
        batch_size, feature_dim = masked_features.shape
        
        for b in range(batch_size):
            mask_indices = np.where(mask[b])[0]
            unmask_indices = np.where(~mask[b])[0]
            
            if len(unmask_indices) == 0:
                # All masked - use zeros
                continue
            
            # Get statistics from unmasked features
            unmasked_mean = np.mean(masked_features[b, unmask_indices])
            unmasked_std = np.std(masked_features[b, unmask_indices]) + 1e-8
            
            for idx in mask_indices:
                # Find nearest unmasked neighbors
                distances = np.abs(unmask_indices - idx)
                nearest_indices = unmask_indices[np.argsort(distances)[:3]]
                
                if len(nearest_indices) > 0:
                    # Weighted interpolation from neighbors
                    weights = 1.0 / (np.abs(nearest_indices - idx) + 1)
                    weights = weights / np.sum(weights)
                    interpolated = np.sum(masked_features[b, nearest_indices] * weights)
                    
                    # Blend with global mean
                    reconstructed[b, idx] = 0.7 * interpolated + 0.3 * unmasked_mean
                else:
                    reconstructed[b, idx] = unmasked_mean
        
        return reconstructed.squeeze()
    
    def compute_total_loss(
        self,
        batch_data: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算总损失
        
        Args:
            batch_data: 包含
                - features: 原始特征
                - embeddings: embedding 向量
                - regimes: regime 标签
                - is_downside: 是否下行状态
        
        Returns:
            (total_loss, loss_breakdown)
        """
        features = batch_data.get("features", [])
        embeddings = batch_data.get("embeddings", [])
        regimes = batch_data.get("regimes", [])
        is_downside = batch_data.get("is_downside", [False] * len(features))
        
        losses = {}
        
        # 1. Masking loss
        if len(features) > 0:
            masking_losses = []
            for i, feat in enumerate(features):
                masked, mask = self.masking.create_mask(feat)
                # Reconstruct using interpolation-based method
                reconstructed = self._reconstruct_features(masked, mask)
                loss = self.masking.compute_loss(feat, reconstructed, mask, is_downside[i])
                masking_losses.append(loss)
            losses["masking"] = np.mean(masking_losses) * self.config.masking_weight
        else:
            losses["masking"] = 0.0
        
        # 2. Contrastive loss
        if len(embeddings) >= 3:
            pairs = self.contrastive.create_pairs(embeddings, regimes)
            contrastive_losses = []
            for anchor_idx, pos_idx, neg_idx in pairs:
                loss = self.contrastive.compute_triplet_loss(
                    embeddings[anchor_idx],
                    embeddings[pos_idx],
                    embeddings[neg_idx],
                )
                contrastive_losses.append(loss)
            losses["contrastive"] = np.mean(contrastive_losses) * self.config.contrastive_weight if contrastive_losses else 0.0
        else:
            losses["contrastive"] = 0.0
        
        # 3. Temporal loss
        if len(embeddings) >= 2:
            temporal_loss, temporal_breakdown = self.temporal.compute_total_loss(embeddings, regimes)
            losses["temporal"] = temporal_loss * self.config.temporal_weight
            losses["temporal_smoothness"] = temporal_breakdown["smoothness"]
            losses["temporal_transition"] = temporal_breakdown["transition"]
        else:
            losses["temporal"] = 0.0
        
        # 4. Tail loss penalty (熊市偏向)
        all_losses = [losses["masking"], losses["contrastive"], losses["temporal"]]
        if all_losses:
            tail_threshold = np.percentile(all_losses, self.config.tail_loss_percentile * 100)
            tail_penalty = sum(max(0, l - tail_threshold) for l in all_losses)
            losses["tail_penalty"] = tail_penalty * self.config.tail_loss_weight
        else:
            losses["tail_penalty"] = 0.0
        
        # 总损失
        total = sum(losses.values())
        
        return total, losses


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MaskingConfig',
    'MaskingObjective',
    'ContrastiveConfig',
    'ContrastiveObjective',
    'TemporalConfig',
    'TemporalConsistencyObjective',
    'PretrainingConfig',
    'CombinedPretrainingObjective',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing Self-supervised Pretraining Objectives")
    print("=" * 60)
    
    # 测试 Masking
    print("\n1. Masking Objective...")
    masking = MaskingObjective()
    
    features = np.random.randn(80)
    masked, mask = masking.create_mask(features, "random")
    print(f"   Mask ratio: {mask.sum() / len(mask):.2f}")
    
    loss = masking.compute_loss(features, masked, mask, is_downside=True)
    print(f"   Reconstruction loss: {loss:.4f}")
    
    # 测试 Contrastive
    print("\n2. Contrastive Objective...")
    contrastive = ContrastiveObjective()
    
    anchor = np.random.randn(128)
    positive = anchor + np.random.randn(128) * 0.1
    negative = np.random.randn(128)
    
    triplet_loss = contrastive.compute_triplet_loss(anchor, positive, negative)
    print(f"   Triplet loss: {triplet_loss:.4f}")
    
    # 测试 Temporal
    print("\n3. Temporal Consistency...")
    temporal = TemporalConsistencyObjective()
    
    embeddings = [np.random.randn(128) for _ in range(10)]
    regimes = ["slow_bear"] * 5 + ["crash_cascade"] * 5
    
    total, breakdown = temporal.compute_total_loss(embeddings, regimes)
    print(f"   Total temporal loss: {total:.4f}")
    print(f"   Breakdown: {breakdown}")
    
    # 测试组合目标
    print("\n4. Combined Objective...")
    combined = CombinedPretrainingObjective()
    
    batch = {
        "features": [np.random.randn(80) for _ in range(10)],
        "embeddings": embeddings,
        "regimes": regimes,
        "is_downside": [True] * 5 + [False] * 5,
    }
    
    total_loss, loss_breakdown = combined.compute_total_loss(batch)
    print(f"   Total loss: {total_loss:.4f}")
    print(f"   Breakdown: {loss_breakdown}")
    
    print("\n✓ Self-supervised Pretraining test complete")

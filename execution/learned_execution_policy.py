"""
================================================================================
HMATS v5.2.1 SOTA - 学习型执行策略
Learned Execution Policy
================================================================================

Purpose: LOB-conditioned 执行策略，提供执行建议

核心约束 (不可违反):
- OFF 不得决定方向
- OFF 不得绕过 production_market_impact
- ON 必须可被熔断
- ON 仅 advisory 模式

输入:
- LOB snapshot
- MarketEmbedding

输出 (仅 advisory):
- slice timing
- aggressiveness
- limit offset

================================================================================
"""

import hashlib
import json
import logging
import numpy as np
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


DEFAULT_LEP_MANIFEST_CANDIDATES = (
    "models/learned_execution_policy/manifest.json",
    "models/learned_execution_policy/latest/manifest.json",
    "data/learned_execution_policy_manifest.json",
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LearnedExecutionConfig:
    """学习型执行策略配置"""
    
    # 模式
    mode: str = "shadow"              # shadow, advisory, active (永远不用 active)
    
    # 输入维度
    lob_feature_dim: int = 20
    embedding_dim: int = 128
    
    # 策略网络
    hidden_dim: int = 64
    output_dim: int = 4               # [wait_prob, market_prob, passive_prob, aggressive_prob]
    
    # 约束
    max_aggressiveness: float = 0.8   # 最大激进度
    min_slice_interval_sec: float = 5.0  # 最小拆单间隔
    
    # 熔断
    fuse_threshold_slippage_bps: float = 50.0  # 滑点超过此值触发熔断
    fuse_consecutive_failures: int = 3
    
    # 熊市调整
    bear_market_passiveness: float = 1.3  # 熊市更被动
    
    # 置信度
    min_confidence_for_advice: float = 0.6

    # 权重加载: 未经 manifest 验证的权重一律不算 learned
    require_validated_manifest: bool = True
    manifest_path: str = ""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ExecutionAction(Enum):
    """执行动作"""
    WAIT = "wait"
    MARKET = "market"
    LIMIT_PASSIVE = "limit_passive"
    LIMIT_AGGRESSIVE = "limit_aggressive"


@dataclass
class ExecutionAdvice:
    """
    执行建议
    
    [WARN]️ 仅供参考，不是直接指令
    """
    # 建议动作
    recommended_action: ExecutionAction
    action_probs: Dict[str, float]
    
    # 参数建议
    slice_size_pct: float              # 0-1
    wait_time_sec: float
    limit_offset_bps: float
    aggressiveness: float              # 0-1
    
    # 置信度
    confidence: float
    
    # 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 来源
    model_version: str = "v1.0"
    is_fallback: bool = False
    fallback_reason: str = ""
    
    # 熊市标记
    bear_market_adjusted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_action": self.recommended_action.value,
            "action_probs": self.action_probs,
            "slice_size_pct": self.slice_size_pct,
            "wait_time_sec": self.wait_time_sec,
            "limit_offset_bps": self.limit_offset_bps,
            "aggressiveness": self.aggressiveness,
            "confidence": self.confidence,
            "is_fallback": self.is_fallback,
            "bear_market_adjusted": self.bear_market_adjusted,
        }


@dataclass
class LOBFeatures:
    """LOB 特征输入"""
    spread_bps: float = 0.0
    imbalance: float = 0.0
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    vpin: float = 0.0
    
    # 近期变化
    spread_change_pct: float = 0.0
    depth_change_pct: float = 0.0
    
    # 队列估计
    estimated_queue_position: int = 0
    
    def to_vector(self) -> np.ndarray:
        """转换为向量"""
        return np.array([
            self.spread_bps / 100,
            self.imbalance,
            np.log1p(self.bid_depth_usd) / 15,
            np.log1p(self.ask_depth_usd) / 15,
            self.vpin,
            self.spread_change_pct,
            self.depth_change_pct,
            self.estimated_queue_position / 100,
        ], dtype=np.float32)


@dataclass
class ExecutionState:
    """执行状态"""
    # 目标
    target_quantity: float
    executed_quantity: float
    side: str                          # "buy" or "sell"
    
    # 价格
    arrival_price: float
    current_mid_price: float
    
    # 时间
    deadline_sec: float                # 剩余时间
    
    # 已实现
    avg_fill_price: float = 0.0
    slippage_bps: float = 0.0
    
    @property
    def progress(self) -> float:
        return self.executed_quantity / max(self.target_quantity, 1e-8)
    
    @property
    def remaining(self) -> float:
        return self.target_quantity - self.executed_quantity
    
    def to_vector(self) -> np.ndarray:
        """转换为向量"""
        return np.array([
            self.progress,
            1 - self.progress,
            self.deadline_sec / 3600,  # 归一化到小时
            self.slippage_bps / 100,
            1 if self.side == "buy" else -1,
        ], dtype=np.float32)


# =============================================================================
# POLICY NETWORK
# =============================================================================

class ExecutionPolicyNetwork:
    """
    执行策略网络 (简化实现)
    
    输入: LOB features + MarketEmbedding + ExecutionState
    输出: Action probabilities
    """
    
    def __init__(self, config: LearnedExecutionConfig):
        self.config = config
        self.weights_loaded = False
        self.weights_source = ""
        self.last_load_error = ""
        
        # 输入维度
        self.input_dim = (
            8 +                           # LOB features
            config.embedding_dim +        # Market embedding
            5                             # Execution state
        )
        
        # 简化：随机初始化权重
        np.random.seed(42)
        
        self._W1 = np.random.randn(self.input_dim, config.hidden_dim) * 0.1
        self._b1 = np.zeros(config.hidden_dim)
        
        self._W2 = np.random.randn(config.hidden_dim, config.output_dim) * 0.1
        self._b2 = np.zeros(config.output_dim)
        
        # 参数头
        self._W_params = np.random.randn(config.hidden_dim, 4) * 0.1  # [slice, wait, offset, aggr]
        self._b_params = np.zeros(4)
        
        logger.info(f"ExecutionPolicyNetwork initialized: input={self.input_dim}")
    
    def forward(
        self,
        lob_features: np.ndarray,
        market_embedding: np.ndarray,
        exec_state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播
        
        Returns:
            (action_probs, params)
        """
        # 拼接输入
        x = np.concatenate([lob_features, market_embedding, exec_state])
        
        # Layer 1
        h = np.tanh(x @ self._W1 + self._b1)
        
        # Action head (softmax)
        action_logits = h @ self._W2 + self._b2
        action_probs = self._softmax(action_logits)
        
        # Params head (sigmoid for bounded output)
        params_raw = h @ self._W_params + self._b_params
        params = 1 / (1 + np.exp(-params_raw))  # sigmoid
        
        return action_probs, params
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def _validate_weight_blob(self, weights: Dict[str, Any]) -> None:
        expected_shapes = {
            "W1": (self.input_dim, self.config.hidden_dim),
            "b1": (self.config.hidden_dim,),
            "W2": (self.config.hidden_dim, self.config.output_dim),
            "b2": (self.config.output_dim,),
            "W_params": (self.config.hidden_dim, 4),
            "b_params": (4,),
        }
        for key, shape in expected_shapes.items():
            if key not in weights:
                raise KeyError(f"missing key: {key}")
            arr = np.asarray(weights[key])
            if tuple(arr.shape) != tuple(shape):
                raise ValueError(
                    f"shape mismatch for {key}: expected {shape}, got {tuple(arr.shape)}"
                )

    def _apply_weight_blob(self, weights: Dict[str, Any], *, source: str) -> None:
        self._validate_weight_blob(weights)
        self._W1 = np.asarray(weights["W1"], dtype=np.float64)
        self._b1 = np.asarray(weights["b1"], dtype=np.float64)
        self._W2 = np.asarray(weights["W2"], dtype=np.float64)
        self._b2 = np.asarray(weights["b2"], dtype=np.float64)
        self._W_params = np.asarray(weights["W_params"], dtype=np.float64)
        self._b_params = np.asarray(weights["b_params"], dtype=np.float64)
        self.weights_loaded = True
        self.weights_source = str(source)
        self.last_load_error = ""

    def load_weights(self, path: str) -> bool:
        """加载权重"""
        try:
            weights = np.load(path, allow_pickle=True).item()
            self._apply_weight_blob(weights, source=str(path))
            logger.info(f"Loaded policy weights from {path}")
            return True
        except Exception as e:
            self.weights_loaded = False
            self.weights_source = ""
            self.last_load_error = str(e)
            logger.warning(f"Failed to load policy weights: {e}")
            return False


# =============================================================================
# FALLBACK POLICY
# =============================================================================

class HeuristicFallbackPolicy:
    """
    启发式回退策略
    
    当学习策略不可用或置信度低时使用
    """
    
    def __init__(self, config: LearnedExecutionConfig):
        self.config = config
    
    def get_advice(
        self,
        lob: LOBFeatures,
        exec_state: ExecutionState,
        is_bear_market: bool = False,
    ) -> ExecutionAdvice:
        """
        获取启发式建议
        """
        # 基于进度决定激进度
        progress = exec_state.progress
        time_pressure = max(0, 1 - exec_state.deadline_sec / 3600)
        
        # 基础激进度
        aggressiveness = 0.3 + 0.4 * time_pressure + 0.2 * progress
        
        # 熊市调整: 更被动
        if is_bear_market:
            aggressiveness /= self.config.bear_market_passiveness
        
        aggressiveness = min(aggressiveness, self.config.max_aggressiveness)
        
        # 决定动作
        if progress >= 0.9 or time_pressure > 0.9:
            # 接近完成或时间紧迫: 激进
            action = ExecutionAction.LIMIT_AGGRESSIVE
        elif lob.spread_bps > 20 or lob.imbalance * (-1 if exec_state.side == "buy" else 1) > 0.3:
            # Spread 大或方向不利: 等待
            action = ExecutionAction.WAIT
        elif aggressiveness > 0.6:
            action = ExecutionAction.LIMIT_AGGRESSIVE
        else:
            action = ExecutionAction.LIMIT_PASSIVE
        
        # 参数
        slice_size = 0.1 + 0.2 * aggressiveness
        wait_time = 10 - 8 * aggressiveness
        limit_offset = 2 + 5 * (1 - aggressiveness)
        
        return ExecutionAdvice(
            recommended_action=action,
            action_probs={
                "wait": 0.25,
                "market": 0.1,
                "limit_passive": 0.35,
                "limit_aggressive": 0.3,
            },
            slice_size_pct=slice_size,
            wait_time_sec=wait_time,
            limit_offset_bps=limit_offset,
            aggressiveness=aggressiveness,
            confidence=0.5,
            is_fallback=True,
            fallback_reason="heuristic_default",
            bear_market_adjusted=is_bear_market,
        )


# =============================================================================
# LEARNED EXECUTION POLICY
# =============================================================================

class LearnedExecutionPolicy:
    """
    学习型执行策略
    
    [WARN]️ 核心约束:
    - 永远不决定方向
    - 永远不绕过 production_market_impact
    - 必须可被熔断
    """
    
    def __init__(self, config: LearnedExecutionConfig = None):
        self.config = config or LearnedExecutionConfig()
        
        # 策略网络
        self._policy = ExecutionPolicyNetwork(self.config)
        
        # 回退策略
        self._fallback = HeuristicFallbackPolicy(self.config)
        
        # 熔断状态
        self._fused = False
        self._fuse_reason = ""
        self._consecutive_failures = 0
        
        # 统计
        self._stats = {
            "total_advices": 0,
            "network_advices": 0,
            "fallback_advices": 0,
            "fuse_count": 0,
        }
        
        # 市场状态
        self._is_bear_market = False
        self._last_advice_source = "uninitialized"
        self._last_advice_reason = ""
        self._validated_manifest_required = bool(
            getattr(self.config, "require_validated_manifest", True)
        )
        self._validated_manifest_loaded = False
        self._validated_manifest_path = ""
        self._validated_manifest_error = ""
        self._validated_weights_sha256 = ""

        self._autodiscover_validated_manifest()

        logger.info(f"LearnedExecutionPolicy initialized in {self.config.mode} mode")

    def _resolve_manifest_path(self) -> Optional[Path]:
        candidates: List[str] = []
        configured = str(getattr(self.config, "manifest_path", "") or "").strip()
        if configured:
            candidates.append(configured)
        env_path = str(os.getenv("HMATS_LEP_MANIFEST", "") or "").strip()
        if env_path:
            candidates.append(env_path)
        candidates.extend(DEFAULT_LEP_MANIFEST_CANDIDATES)
        for raw_path in candidates:
            path = Path(raw_path)
            if path.exists():
                return path
        return Path(candidates[0]) if candidates else None

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _autodiscover_validated_manifest(self) -> None:
        manifest_path = self._resolve_manifest_path()
        if manifest_path is None:
            self._validated_manifest_error = "manifest_not_configured"
            return
        if manifest_path.exists():
            self.load_validated_manifest(str(manifest_path))
        else:
            self._validated_manifest_path = str(manifest_path)
            self._validated_manifest_error = "manifest_not_found"

    def load_validated_manifest(self, manifest_path: str) -> bool:
        manifest_file = Path(manifest_path)
        self._validated_manifest_path = str(manifest_file)
        self._validated_manifest_loaded = False
        self._validated_manifest_error = ""
        self._validated_weights_sha256 = ""
        if not manifest_file.exists():
            self._validated_manifest_error = "manifest_not_found"
            return False
        try:
            with manifest_file.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle) or {}
            weights_rel = str(
                manifest.get("weights_path")
                or manifest.get("artifact_path")
                or manifest.get("model_path")
                or ""
            ).strip()
            if not weights_rel:
                raise ValueError("manifest missing weights_path")
            weights_path = Path(weights_rel)
            if not weights_path.is_absolute():
                weights_path = (manifest_file.parent / weights_path).resolve()
                # [P22 2026-04-24] Defense-in-depth: a manifest claiming
                # weights_path="../../../etc/passwd" would resolve outside
                # manifest dir. Reject any weights path that escapes the
                # manifest's directory.
                manifest_root = manifest_file.parent.resolve()
                try:
                    weights_path.relative_to(manifest_root)
                except ValueError:
                    raise ValueError(
                        f"weights_path escapes manifest dir: {weights_path}"
                    )
            if not weights_path.exists():
                raise FileNotFoundError(f"weights missing: {weights_path}")

            expected_input_dim = int(manifest.get("input_dim", self._policy.input_dim))
            expected_hidden_dim = int(
                manifest.get("hidden_dim", self.config.hidden_dim)
            )
            expected_output_dim = int(
                manifest.get("output_dim", self.config.output_dim)
            )
            if expected_input_dim != self._policy.input_dim:
                raise ValueError(
                    f"input_dim mismatch: expected {self._policy.input_dim}, got {expected_input_dim}"
                )
            if expected_hidden_dim != self.config.hidden_dim:
                raise ValueError(
                    f"hidden_dim mismatch: expected {self.config.hidden_dim}, got {expected_hidden_dim}"
                )
            if expected_output_dim != self.config.output_dim:
                raise ValueError(
                    f"output_dim mismatch: expected {self.config.output_dim}, got {expected_output_dim}"
                )

            expected_sha = str(manifest.get("weights_sha256", "") or "").strip().lower()
            actual_sha = self._sha256_file(weights_path).lower()
            if expected_sha and expected_sha != actual_sha:
                raise ValueError("weights sha256 mismatch")

            if not self._policy.load_weights(str(weights_path)):
                raise ValueError(self._policy.last_load_error or "weights_load_failed")

            self._validated_manifest_loaded = True
            self._validated_manifest_error = ""
            self._validated_manifest_path = str(manifest_file.resolve())
            self._validated_weights_sha256 = actual_sha
            return True
        except Exception as exc:
            self._validated_manifest_loaded = False
            self._validated_manifest_error = str(exc)
            return False

    def _network_usable(self) -> bool:
        if not bool(self._policy.weights_loaded):
            return False
        if not self._validated_manifest_required:
            return True
        return bool(self._validated_manifest_loaded)

    def set_bear_market(self, is_bear: bool):
        """设置熊市状态"""
        self._is_bear_market = is_bear
        if is_bear:
            logger.info("LearnedExecutionPolicy: Bear market mode activated")
    
    def get_advice(
        self,
        lob: LOBFeatures,
        market_embedding: Optional[np.ndarray],
        exec_state: ExecutionState,
    ) -> ExecutionAdvice:
        """
        获取执行建议
        
        [WARN]️ 这是 advisory only，不是直接指令
        
        工程审计修复 H3: crash_cascade 时强制使用 fallback
        """
        self._stats["total_advices"] += 1
        
        # H3 FIX: crash_cascade 时强制使用 fallback
        # 检查 LOB 极端情况 (spread > 100bps 或 depth 极低)
        if lob.spread_bps > 100:
            logger.warning(
                f"LearnedExecutionPolicy: EXTREME_SPREAD ({lob.spread_bps:.1f}bps) - forced fallback"
            )
            return self._get_fallback(lob, exec_state, reason="extreme_spread")
        
        # 检查熔断
        if self._fused:
            logger.warning(f"LearnedExecutionPolicy FUSED: {self._fuse_reason}")
            return self._get_fallback(lob, exec_state, reason=f"fused:{self._fuse_reason}")
        
        # 检查 shadow mode
        if self.config.mode == "shadow":
            # Shadow mode: 记录但不使用网络输出
            if market_embedding is not None and self._network_usable():
                self._policy.forward(
                    lob.to_vector(),
                    market_embedding,
                    exec_state.to_vector(),
                )
            return self._get_fallback(lob, exec_state, reason="shadow_mode")
        
        # Advisory mode
        if market_embedding is None:
            return self._get_fallback(lob, exec_state, reason="missing_embedding")
        if not self._network_usable():
            return self._get_fallback(lob, exec_state, reason="unvalidated_policy")

        try:
            # 网络前向
            action_probs, params = self._policy.forward(
                lob.to_vector(),
                market_embedding,
                exec_state.to_vector(),
            )
            
            # 解析动作
            actions = [
                ExecutionAction.WAIT,
                ExecutionAction.MARKET,
                ExecutionAction.LIMIT_PASSIVE,
                ExecutionAction.LIMIT_AGGRESSIVE,
            ]
            best_action = actions[np.argmax(action_probs)]
            
            # 解析参数
            slice_size = params[0] * 0.5  # 最大 50%
            wait_time = params[1] * 30    # 最大 30 秒
            limit_offset = params[2] * 20  # 最大 20 bps
            aggressiveness = params[3]
            
            # 熊市调整
            if self._is_bear_market:
                aggressiveness /= self.config.bear_market_passiveness
                wait_time *= 1.5
            
            # 约束
            aggressiveness = min(aggressiveness, self.config.max_aggressiveness)
            
            # 置信度
            confidence = float(np.max(action_probs))
            
            # 置信度检查
            if confidence < self.config.min_confidence_for_advice:
                return self._get_fallback(lob, exec_state, reason="low_confidence")
            
            self._stats["network_advices"] += 1
            self._last_advice_source = "network"
            self._last_advice_reason = "policy_output"
            
            return ExecutionAdvice(
                recommended_action=best_action,
                action_probs={
                    "wait": float(action_probs[0]),
                    "market": float(action_probs[1]),
                    "limit_passive": float(action_probs[2]),
                    "limit_aggressive": float(action_probs[3]),
                },
                slice_size_pct=float(slice_size),
                wait_time_sec=float(wait_time),
                limit_offset_bps=float(limit_offset),
                aggressiveness=float(aggressiveness),
                confidence=confidence,
                is_fallback=False,
                bear_market_adjusted=self._is_bear_market,
            )
            
        except Exception as e:
            logger.warning(f"Policy forward failed: {e}")
            return self._get_fallback(lob, exec_state, reason=f"policy_error:{e}")
    
    def _get_fallback(
        self,
        lob: LOBFeatures,
        exec_state: ExecutionState,
        reason: str = "fallback",
    ) -> ExecutionAdvice:
        """获取回退建议"""
        self._stats["fallback_advices"] += 1
        self._last_advice_source = "fallback"
        self._last_advice_reason = str(reason or "fallback")
        return self._fallback.get_advice(lob, exec_state, self._is_bear_market)
    
    def record_result(self, slippage_bps: float, fill_ratio: float):
        """
        记录执行结果
        
        用于熔断检测
        
        工程审计修复 H2: 同时检查 slippage 和 fill_ratio
        """
        failed = False
        failure_reasons = []
        
        # H2 FIX: 检查 slippage
        if abs(slippage_bps) > self.config.fuse_threshold_slippage_bps:
            failed = True
            failure_reasons.append(f"slippage={slippage_bps:.1f}bps")
        
        # H2 FIX: 检查 fill_ratio (新增)
        if fill_ratio < 0.3:
            failed = True
            failure_reasons.append(f"fill_ratio={fill_ratio:.2f}")
        
        if failed:
            self._consecutive_failures += 1
            logger.warning(
                f"LearnedExecutionPolicy execution failure #{self._consecutive_failures}: "
                f"{', '.join(failure_reasons)}"
            )
            
            if self._consecutive_failures >= self.config.fuse_consecutive_failures:
                self._trigger_fuse(f"consecutive_failures: {', '.join(failure_reasons)}")
        else:
            self._consecutive_failures = max(0, self._consecutive_failures - 1)
    
    def _trigger_fuse(self, reason: str):
        """触发熔断"""
        self._fused = True
        self._fuse_reason = reason
        self._stats["fuse_count"] += 1
        logger.error(f"LearnedExecutionPolicy FUSED: {reason}")
    
    def reset_fuse(self):
        """重置熔断"""
        self._fused = False
        self._fuse_reason = ""
        self._consecutive_failures = 0
        logger.info("LearnedExecutionPolicy fuse reset")
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        if self.config.mode == "shadow":
            effective_mode = "SHADOW_LEARNED" if self._network_usable() else "SHADOW_HEURISTIC"
        elif self._network_usable():
            effective_mode = "ADVISORY_LEARNED"
        else:
            effective_mode = "ADVISORY_HEURISTIC"
        return {
            "mode": self.config.mode,
            "effective_mode": effective_mode,
            "fused": self._fused,
            "fuse_reason": self._fuse_reason,
            "is_bear_market": self._is_bear_market,
            "weights_loaded": bool(self._policy.weights_loaded),
            "weights_source": str(self._policy.weights_source or ""),
            "heuristic_only": not self._network_usable(),
            "network_usable": bool(self._network_usable()),
            "validated_manifest_required": bool(self._validated_manifest_required),
            "validated_manifest_loaded": bool(self._validated_manifest_loaded),
            "validated_manifest_path": str(self._validated_manifest_path or ""),
            "validated_manifest_error": str(self._validated_manifest_error or ""),
            "validated_weights_sha256": str(self._validated_weights_sha256 or ""),
            "last_advice_source": str(self._last_advice_source or ""),
            "last_advice_reason": str(self._last_advice_reason or ""),
            "stats": self._stats,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_learned_policy: Optional[LearnedExecutionPolicy] = None


def get_learned_execution_policy(
    config: Optional[LearnedExecutionConfig] = None
) -> LearnedExecutionPolicy:
    """获取单例。

    If a materially different config is supplied later in the same process,
    refresh the singleton so callers do not silently keep running under the
    first config that happened to initialize it.
    """
    global _learned_policy
    if config is None:
        config = LearnedExecutionConfig(mode="advisory")
    if _learned_policy is None:
        _learned_policy = LearnedExecutionPolicy(config)
    elif _learned_policy.config != config:
        logger.info(
            "LearnedExecutionPolicy config changed; refreshing singleton "
            f"({getattr(_learned_policy.config, 'mode', 'unknown')} -> {config.mode})"
        )
        _learned_policy = LearnedExecutionPolicy(config)
    return _learned_policy


def reset_learned_execution_policy():
    """重置单例"""
    global _learned_policy
    _learned_policy = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'LearnedExecutionConfig',
    'ExecutionAction',
    'ExecutionAdvice',
    'LOBFeatures',
    'ExecutionState',
    'ExecutionPolicyNetwork',
    'HeuristicFallbackPolicy',
    'LearnedExecutionPolicy',
    'get_learned_execution_policy',
    'reset_learned_execution_policy',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing Learned Execution Policy")
    print("=" * 60)
    
    # 创建策略
    config = LearnedExecutionConfig(mode="advisory")
    policy = LearnedExecutionPolicy(config)
    
    # 模拟输入
    print("\n1. Testing with mock inputs...")
    
    lob = LOBFeatures(
        spread_bps=15.0,
        imbalance=-0.2,
        bid_depth_usd=300000,
        ask_depth_usd=500000,
        vpin=0.6,
    )
    
    market_embedding = np.random.randn(128)
    
    exec_state = ExecutionState(
        target_quantity=1.0,
        executed_quantity=0.3,
        side="sell",
        arrival_price=50000,
        current_mid_price=49500,
        deadline_sec=600,
    )
    
    advice = policy.get_advice(lob, market_embedding, exec_state)
    
    print(f"\n2. Execution Advice:")
    print(f"   recommended_action: {advice.recommended_action.value}")
    print(f"   action_probs: {advice.action_probs}")
    print(f"   slice_size_pct: {advice.slice_size_pct:.2f}")
    print(f"   wait_time_sec: {advice.wait_time_sec:.1f}")
    print(f"   limit_offset_bps: {advice.limit_offset_bps:.1f}")
    print(f"   aggressiveness: {advice.aggressiveness:.2f}")
    print(f"   confidence: {advice.confidence:.2f}")
    print(f"   is_fallback: {advice.is_fallback}")
    
    # 测试熊市模式
    print("\n3. Testing bear market mode...")
    policy.set_bear_market(True)
    
    advice_bear = policy.get_advice(lob, market_embedding, exec_state)
    print(f"   aggressiveness (bear): {advice_bear.aggressiveness:.2f}")
    print(f"   bear_market_adjusted: {advice_bear.bear_market_adjusted}")
    
    # 测试熔断
    print("\n4. Testing fuse mechanism...")
    
    for i in range(4):
        policy.record_result(slippage_bps=60.0, fill_ratio=0.5)  # 超过阈值
    
    status = policy.get_status()
    print(f"   fused: {status['fused']}")
    print(f"   fuse_reason: {status['fuse_reason']}")
    
    # 熔断后的建议
    advice_fused = policy.get_advice(lob, market_embedding, exec_state)
    print(f"   advice after fuse: is_fallback={advice_fused.is_fallback}")
    
    print("\n✓ Learned Execution Policy test complete")

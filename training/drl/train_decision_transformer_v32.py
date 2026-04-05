#!/usr/bin/env python3
"""
================================================================================
HMATS Decision Transformer v3.2 - 完整优化版
================================================================================

🎯 准确率提升技术:
  ON Regime 感知智能专家 (+3-5%)
  ON 智能 Oracle - 考虑路径/回撤/风险 (+2-3%)
  ON 动态损失权重 (Uncertainty Weighting) (+1-2%)
  ON 在线难例挖掘 OHEM (+1-2%)
  ON 修复 R-Drop - 回归用 MSE 一致性
  ON 修复 FGM - AMP 兼容
  ON 分层 RTG + 多任务学习
  ON Label Smoothing (方向分类)

🚀 GPU 优化:
  ON 混合精度 (AMP) - 速度 2x
  ON 优化 DataLoader (num_workers=8, pin_memory, prefetch)
  ON 大批量训练 (batch_size=256-512)
  ON 课程学习 Sampler (避免重建 DataLoader)
  ON 非阻塞数据传输
  ON CUDA 优化 (TF32, cuDNN benchmark)

📊 预期性能:
  - 方向准确率: 65% -> 78-82%
  - 训练速度: 3-10x 提升

Usage:
    python train_decision_transformer_v32.py                    # 标准训练
    python train_decision_transformer_v32.py --batch-size 512   # 最大配置
    python train_decision_transformer_v32.py --epochs 10        # 快速测试

================================================================================
"""

# 🆕 v4: 禁用 torch.compile/dynamo (避免 Triton 依赖)
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import warnings
import logging
import random
import time
from copy import deepcopy

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import autocast, GradScaler

try:
    from transformers import GPT2Model, GPT2Config
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("DT_v3.2")


# =============================================================================
# CUDA 优化
# =============================================================================

def setup_cuda_optimizations() -> float:
    """设置 CUDA 优化，返回 GPU 显存 (GB)"""
    if not torch.cuda.is_available():
        logger.warning("CUDA 不可用")
        return 0
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 🆕 P0: BF16 精度设置 (5090 优化)
    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        torch.set_float32_matmul_precision('medium')
        logger.info("⚡ BF16 精度已启用")
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    logger.info(f"🚀 CUDA 优化已启用: {gpu_name}, {gpu_mem:.1f}GB")
    return gpu_mem


# 🆕 P0: torch.compile 包装器
def compile_model_if_available(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
    """
    使用 torch.compile 优化模型 (PyTorch 2.0+)
    
    Args:
        model: PyTorch 模型
        mode: 编译模式
            - "default": 平衡模式
            - "reduce-overhead": 减少开销 (推荐)
            - "max-autotune": 最大优化 (慢启动)
    
    Returns:
        编译后的模型 (或原模型如果不支持)
    """
    if not hasattr(torch, 'compile'):
        logger.warning("torch.compile 不可用 (需要 PyTorch 2.0+)")
        return model
    
    try:
        compiled = torch.compile(model, mode=mode)
        logger.info(f"ON 模型已编译 (mode={mode})")
        return compiled
    except Exception as e:
        logger.warning(f"torch.compile 失败: {e}")
        return model


# =============================================================================
# Feature Manifest Loader (shared 122-feature pipeline with DRL v7)
# =============================================================================

def load_feature_manifest(path: str = "configs/feature_manifest.json") -> Dict[str, Any]:
    """Load feature manifest (same 122 features as DRL v7 TQC).

    Returns dict with keys: all_features, no_scale_features, total_feature_count.
    """
    manifest_path = Path(path)
    if not manifest_path.exists():
        # Try relative to script directory
        script_dir = Path(__file__).resolve().parent.parent.parent
        manifest_path = script_dir / path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Feature manifest not found: {path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert len(manifest["all_features"]) == manifest["total_feature_count"] == 122, \
        f"Feature manifest mismatch: {len(manifest['all_features'])} != 122"
    return manifest


# =============================================================================
# 配置
# =============================================================================

@dataclass
class DTConfigV32:
    """Decision Transformer v3.2 配置
    
    🆕 v3.1: 支持风险偏好配置
    """
    
    # ============================================================
    # 🆕 v3.1: 风险偏好配置
    # ============================================================
    risk_profile: str = "aggressive"      # aggressive, moderate, conservative
    direction_bias: str = "short"         # short, neutral, long
    
    # SOL 特定优化
    sol_vol_multiplier: float = 1.3       # SOL 波动率调整系数
    btc_lead_weight: float = 0.4          # BTC 领先指标权重 (原 0.3)
    bear_reward_multiplier: float = 1.2   # 熊市做空奖励乘数
    
    # Per-asset training
    asset: str = "BTC"
    feature_manifest_path: str = "configs/feature_manifest.json"

    # 模型架构 (v5: reduced capacity - was 8/384/8/1536/96, overfitting 23-26% gap)
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 1024
    dropout: float = 0.3  # v5: 0.1->0.3 (primary regularizer for small data)
    state_dim: int = 126  # 122 features + 4 env state (aligned with TQC obs_dim)
    action_dim: int = 1
    context_length: int = 48  # v5: 96->48 (doubles trajectory count ~300->~600)
    max_episode_length: int = 512
    
    # 分层 RTG
    use_hierarchical_rtg: bool = True
    rtg_scales: List[int] = field(default_factory=lambda: [1, 4, 24, 168])
    rtg_hidden_dim: int = 64
    
    # LoRA (v5: reduced rank for smaller model)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # 训练 (v5: longer training + stronger regularization)
    epochs: int = 1200  # v5b: 400->1200 (val still climbing at 400)
    batch_size: int = 128
    learning_rate: float = 1e-5  # v5b: 3e-5->1e-5 (slower learning for longer training)
    weight_decay: float = 0.05  # v5: 0.01->0.05 (stronger L2 regularization)
    warmup_ratio: float = 0.15  # 🆕 v4: 增加预热 (0.1 -> 0.15)
    gradient_accumulation: int = 1
    max_grad_norm: float = 0.5  # 🆕 v4: 更严格的梯度裁剪 (1.0 -> 0.5)
    
    # DataLoader 优化
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True
    
    # 混合精度
    use_amp: bool = True
    
    # 🆕 动态损失权重
    use_dynamic_loss_weights: bool = False  # 🆕 v4: 默认禁用，先确保基本训练正常
    action_loss_weight: float = 1.0
    direction_loss_weight: float = 1.0  # 🆕 v4: 提高方向损失权重
    reward_loss_weight: float = 0.3
    
    # 🆕 Label Smoothing
    use_label_smoothing: bool = True
    label_smoothing: float = 0.1
    
    # 课程学习
    use_curriculum: bool = True
    curriculum_start: float = 0.5  # 🆕 v3.1: 提高到 50%，避免早期样本不足
    curriculum_epochs: int = 50
    
    # 🆕 对抗训练 (FGM + AMP 兼容)
    use_fgm: bool = True  # v5b: keep ON - helps ETH +5.9%, SOL +7.1% (adversarial augmentation)
    fgm_epsilon: float = 0.1  # 🆕 v4: 降低 epsilon (0.3 -> 0.1)
    
    # 🆕 R-Drop (修复版)
    use_rdrop: bool = False  # 🆕 v4: 默认禁用，R-Drop 容易导致训练不稳定
    rdrop_alpha: float = 0.1  # 🆕 v4: 降低 alpha (0.5 -> 0.1)
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # 🆕 OHEM (sparse binary feature handling - helps learn from rare events)
    use_ohem: bool = True  # v7: enabled per training tech stack standard
    ohem_ratio: float = 0.7
    
    # 数据增强
    use_augmentation: bool = True
    noise_std: float = 0.02  # v5: 0.01->0.02 (more augmentation for regularization)
    
    # 🆕 专家策略
    use_regime_aware_expert: bool = True
    use_smart_oracle: bool = True
    oracle_windows: List[int] = field(default_factory=lambda: [4, 12, 24, 48])
    
    # 🆕 P0: torch.compile 优化
    use_torch_compile: bool = False  # 🆕 v4: 默认禁用，调试时更容易定位问题
    compile_mode: str = "reduce-overhead"  # default, reduce-overhead, max-autotune
    
    # 验证
    val_ratio: float = 0.15
    early_stopping: int = 200  # v5b: 80->200 (patience for 1200 epoch training)
    log_interval: int = 10
    
    # 🆕 P0: 数据泄露检测
    check_data_leakage: bool = True
    leakage_lookahead: int = 24


# =============================================================================
# Yang-Zhang 波动率
# =============================================================================

class YangZhangVolatility:
    @staticmethod
    def calculate(df: pd.DataFrame, window: int = 24) -> pd.Series:
        log_ho = np.log(df['high'] / df['open'])
        log_lo = np.log(df['low'] / df['open'])
        log_co = np.log(df['close'] / df['open'])
        log_oc = np.log(df['open'] / df['close'].shift(1))
        
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        overnight_var = log_oc.rolling(window).var()
        open_close_var = log_co.rolling(window).var()
        rs_var = rs.rolling(window).mean()
        
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz_var = overnight_var + k * open_close_var + (1 - k) * rs_var
        
        return (np.sqrt(yz_var.abs()) * np.sqrt(24 * 365)).fillna(0)


# =============================================================================
# 特征工程 (80维)
# =============================================================================

class FeatureEngineerV32:
    """80维特征工程"""
    
    def __init__(self, config: DTConfigV32):
        self.config = config
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 收益率 (10)
        for p in [1, 2, 4, 8, 12, 24, 48, 72, 96, 168]:
            df[f'ret_{p}h'] = df['close'].pct_change(p).fillna(0).clip(-0.5, 0.5)
        
        # YZ 波动率 (5)
        for w in [12, 24, 48, 96, 168]:
            df[f'yz_vol_{w}h'] = YangZhangVolatility.calculate(df, w)
        
        # 简单波动率 (5)
        hourly_ret = df['close'].pct_change(1).fillna(0)
        for w in [12, 24, 48, 96, 168]:
            df[f'simple_vol_{w}h'] = hourly_ret.rolling(w).std().fillna(0) * np.sqrt(24*365)
        
        # 波动率指标 (2)
        df['vol_ratio'] = (df['yz_vol_24h'] / (df['yz_vol_168h'] + 1e-10)).clip(0, 5)
        df['vol_regime'] = (df['yz_vol_24h'] > df['yz_vol_24h'].rolling(168).mean()).astype(float)
        
        # 动量 (8)
        for p in [4, 8, 12, 24, 48, 96, 168, 336]:
            df[f'mom_{p}h'] = df['close'].pct_change(p).fillna(0).clip(-1, 1)
        
        # Z-score (4)
        for w in [24, 48, 168, 336]:
            ma = df['close'].rolling(w).mean()
            std = df['close'].rolling(w).std()
            df[f'zscore_{w}h'] = ((df['close'] - ma) / (std + 1e-10)).fillna(0).clip(-5, 5)
        
        # 技术指标 (8)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = (100 - (100 / (1 + rs))).fillna(50) / 100 - 0.5
        
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ((ema12 - ema26) / df['close']).fillna(0).clip(-0.1, 0.1)
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        bb_ma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_position'] = ((df['close'] - bb_ma) / (2 * bb_std + 1e-10)).fillna(0).clip(-1, 1)
        df['bb_width'] = (4 * bb_std / bb_ma).fillna(0).clip(0, 1)
        df['trend_strength'] = (df['close'].rolling(24).mean() / df['close'].rolling(168).mean() - 1).fillna(0).clip(-0.3, 0.3)
        
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = (tr.rolling(14).mean() / df['close']).fillna(0).clip(0, 0.2)
        
        # 成交量 (4)
        if 'volume' in df.columns:
            vol_ma = df['volume'].rolling(24).mean()
            df['vol_zscore'] = ((df['volume'] - vol_ma) / (vol_ma.std() + 1e-10)).fillna(0).clip(-5, 5)
            df['vol_trend'] = (df['volume'].rolling(24).mean() / df['volume'].rolling(168).mean() - 1).fillna(0)
            df['vol_spike'] = (df['volume'] / (vol_ma + 1e-10)).fillna(1).clip(0, 10) - 1
            df['vol_momentum'] = df['volume'].pct_change(24).fillna(0).clip(-2, 2)
        else:
            df['vol_zscore'] = df['vol_trend'] = df['vol_spike'] = df['vol_momentum'] = 0
        
        # 做空信号 (5)
        df['fear_spike'] = (df['ret_4h'] < -0.02).astype(float)
        df['vol_expansion'] = ((df['yz_vol_24h'] > df['yz_vol_168h'] * 1.5) & (df['ret_4h'] < 0)).astype(float)
        low_20 = df['close'].rolling(20).min()
        df['breakdown'] = (df['close'] < low_20.shift(1)).astype(float)
        df['panic'] = (df['ret_4h'] < -0.04).astype(float)
        df['capitulation'] = ((df['ret_4h'] < -0.06) & (df['vol_spike'] > 2)).astype(float)
        
        # Regime 编码 (12)
        for regime in ['TRENDING_BULL', 'TRENDING_BEAR', 'VOLATILE_BULL', 'VOLATILE_BEAR', 
                       'RANGING_LOW_VOL', 'RANGING_HIGH_VOL']:
            df[f'regime_{regime}'] = 0.0
        
        return df.fillna(0)
    
    def get_feature_columns(self) -> List[str]:
        cols = []
        for p in [1, 2, 4, 8, 12, 24, 48, 72, 96, 168]:
            cols.append(f'ret_{p}h')
        for w in [12, 24, 48, 96, 168]:
            cols.append(f'yz_vol_{w}h')
        for w in [12, 24, 48, 96, 168]:
            cols.append(f'simple_vol_{w}h')
        cols.extend(['vol_ratio', 'vol_regime'])
        for p in [4, 8, 12, 24, 48, 96, 168, 336]:
            cols.append(f'mom_{p}h')
        for w in [24, 48, 168, 336]:
            cols.append(f'zscore_{w}h')
        cols.extend(['rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position', 'bb_width', 'trend_strength', 'atr'])
        cols.extend(['vol_zscore', 'vol_trend', 'vol_spike', 'vol_momentum'])
        cols.extend(['fear_spike', 'vol_expansion', 'breakdown', 'panic', 'capitulation'])
        return cols[:self.config.state_dim]
    
    # P5: GMM regime name -> DT regime name mapping (for expert config lookup)
    GMM_TO_DT_REGIME = {
        'MOMENTUM_RALLY':       'TRENDING_BULL',
        'QUIET_ACCUMULATION':   'RANGING_LOW_VOL',
        'WEAK_CONSOLIDATION':   'RANGING_LOW_VOL',
        'VOLATILE_CHOP':        'RANGING_HIGH_VOL',
        'EXTREME_VOLATILITY':   'VOLATILE_BEAR',
        'PANIC_SELLOFF':        'TRENDING_BEAR',
    }

    def get_regime(self, df: pd.DataFrame, idx: int) -> str:
        """P5: Use GMM regime labels from parquet when available, fall back to rules."""
        # Prefer GMM regime column (consistent with DRL v7 pipeline)
        if 'regime' in df.columns:
            regime_val = df.iloc[idx].get('regime')
            if isinstance(regime_val, str) and regime_val in self.GMM_TO_DT_REGIME:
                return self.GMM_TO_DT_REGIME[regime_val]
            elif isinstance(regime_val, (int, float)) and not (regime_val != regime_val):  # not NaN
                regime_val = int(regime_val)
                # Load GMM regime names from config if available
                gmm_names = self._get_gmm_regime_names()
                if 0 <= regime_val < len(gmm_names):
                    gmm_name = gmm_names[regime_val]
                    return self.GMM_TO_DT_REGIME.get(gmm_name, 'UNKNOWN')

        # Fallback: rule-based classification (original behavior)
        if idx < 168:
            return 'UNKNOWN'
        try:
            ret_24h = df.iloc[idx]['close'] / df.iloc[idx-24]['close'] - 1
            ret_168h = df.iloc[idx]['close'] / df.iloc[idx-168]['close'] - 1
            vol = df.iloc[idx].get('yz_vol_24h', 0.5)

            if ret_168h > 0.1 and ret_24h > 0:
                return 'VOLATILE_BULL' if vol > 0.6 else 'TRENDING_BULL'
            elif ret_168h < -0.1 and ret_24h < 0:
                return 'VOLATILE_BEAR' if vol > 0.6 else 'TRENDING_BEAR'
            elif vol < 0.3:
                return 'RANGING_LOW_VOL'
            return 'RANGING_HIGH_VOL'
        except:
            return 'UNKNOWN'

    _gmm_regime_names_cache = None

    def _get_gmm_regime_names(self) -> list:
        """Load GMM regime names from config (cached)."""
        if self._gmm_regime_names_cache is not None:
            return self._gmm_regime_names_cache
        import json
        from pathlib import Path
        for p in [Path("data/gmm_models/gmm_config.json"),
                  Path("models/regime_classifier/gmm_config.json")]:
            if p.exists():
                try:
                    with open(p) as f:
                        names = json.load(f).get("regime_names", [])
                    if names and len(names) == 6:
                        FeatureEngineerV32._gmm_regime_names_cache = names
                        return names
                except Exception:
                    pass
        default = ["VOLATILE_CHOP", "EXTREME_VOLATILITY", "MOMENTUM_RALLY",
                    "PANIC_SELLOFF", "WEAK_CONSOLIDATION", "QUIET_ACCUMULATION"]
        FeatureEngineerV32._gmm_regime_names_cache = default
        return default


# =============================================================================
# 🆕 Regime 感知智能专家
# =============================================================================

class RegimeAwareExpert:
    """Regime 感知的智能专家策略"""
    
    # ============================================================
    # Regime 名称映射 (DT v3.2 -> HMATS 6-Regime GMM names)
    # P5: Updated to map through GMM regime names
    # ============================================================
    HMATS_REGIME_MAP = {
        # DT regime -> GMM regime ID
        'TRENDING_BULL': 5,     # -> STRONG_BULL / MOMENTUM_RALLY
        'TRENDING_BEAR': 0,     # -> STRONG_BEAR / PANIC_SELLOFF
        'VOLATILE_BULL': 4,     # -> MODERATE_BULL / QUIET_ACCUMULATION
        'VOLATILE_BEAR': 1,     # -> MODERATE_BEAR / EXTREME_VOLATILITY
        'RANGING_LOW_VOL': 2,   # -> NEUTRAL / WEAK_CONSOLIDATION
        'RANGING_HIGH_VOL': 3,  # -> VOLATILE_CHOP
        'UNKNOWN': 2,           # -> NEUTRAL (默认)
    }

    HMATS_REGIME_NAMES = {
        0: 'PANIC_SELLOFF',
        1: 'EXTREME_VOLATILITY',
        2: 'WEAK_CONSOLIDATION',
        3: 'VOLATILE_CHOP',
        4: 'QUIET_ACCUMULATION',
        5: 'MOMENTUM_RALLY',
    }
    
    # ============================================================
    # 🆕 v3.1: 高风险做空优化配置
    # ============================================================
    # 风险偏好: aggressive (高风险), moderate, conservative
    # 方向偏好: short (主做空), neutral, long
    
    # 激进做空配置 (默认)
    REGIME_CONFIGS_AGGRESSIVE_SHORT = {
        # 熊市：激进做空
        'TRENDING_BEAR': {
            'aggression': 1.5,       # 原 1.3, +15%
            'lookforward': 48, 
            'entry_thresh': 0.006,   # 原 0.008, 更敏感
            'position_bias': -0.40   # 原 -0.25, +60%
        },
        'VOLATILE_BEAR': {
            'aggression': 0.8,       # 原 0.6, +33%
            'lookforward': 12, 
            'entry_thresh': 0.018,   # 原 0.025, 更敏感
            'position_bias': -0.20   # 原 -0.10, +100%
        },
        
        # 震荡：轻微做空偏好
        'RANGING_LOW_VOL': {
            'aggression': 0.4,       # 原 0.3
            'lookforward': 24, 
            'entry_thresh': 0.012,
            'position_bias': -0.03   # 原 0.0
        },
        'RANGING_HIGH_VOL': {
            'aggression': 0.6,       # 原 0.4, +50%
            'lookforward': 16, 
            'entry_thresh': 0.015,
            'position_bias': -0.05   # 原 0.0
        },
        
        # 牛市：减少做多，保留对冲
        'VOLATILE_BULL': {
            'aggression': 0.5,       # 原 0.6, -17%
            'lookforward': 12, 
            'entry_thresh': 0.030,   # 原 0.025, 更迟钝
            'position_bias': 0.00    # 原 0.05
        },
        'TRENDING_BULL': {
            'aggression': 1.0,       # 原 1.3, -23%
            'lookforward': 48, 
            'entry_thresh': 0.012,   # 原 0.008
            'position_bias': 0.08    # 原 0.15, -47%
        },
        
        'UNKNOWN': {
            'aggression': 0.5, 
            'lookforward': 24, 
            'entry_thresh': 0.015, 
            'position_bias': -0.02   # 原 0.0, 默认轻微做空
        },
    }
    
    # 原始保守配置 (备用)
    REGIME_CONFIGS_CONSERVATIVE = {
        'TRENDING_BULL': {'aggression': 1.3, 'lookforward': 48, 'entry_thresh': 0.008, 'position_bias': 0.15},
        'TRENDING_BEAR': {'aggression': 1.3, 'lookforward': 48, 'entry_thresh': 0.008, 'position_bias': -0.25},
        'VOLATILE_BULL': {'aggression': 0.6, 'lookforward': 12, 'entry_thresh': 0.025, 'position_bias': 0.05},
        'VOLATILE_BEAR': {'aggression': 0.6, 'lookforward': 12, 'entry_thresh': 0.025, 'position_bias': -0.10},
        'RANGING_LOW_VOL': {'aggression': 0.3, 'lookforward': 24, 'entry_thresh': 0.015, 'position_bias': 0.0},
        'RANGING_HIGH_VOL': {'aggression': 0.4, 'lookforward': 16, 'entry_thresh': 0.02, 'position_bias': 0.0},
        'UNKNOWN': {'aggression': 0.5, 'lookforward': 24, 'entry_thresh': 0.015, 'position_bias': 0.0},
    }
    
    # 🆕 默认使用激进做空配置
    REGIME_CONFIGS = REGIME_CONFIGS_AGGRESSIVE_SHORT
    
    @classmethod
    def to_hmats_regime(cls, dt_regime: str) -> int:
        """将 DT Regime 名称转换为 HMATS Regime ID"""
        return cls.HMATS_REGIME_MAP.get(dt_regime, 2)
    
    @classmethod
    def to_hmats_name(cls, dt_regime: str) -> str:
        """将 DT Regime 名称转换为 HMATS Regime 名称"""
        regime_id = cls.to_hmats_regime(dt_regime)
        return cls.HMATS_REGIME_NAMES.get(regime_id, 'NEUTRAL')
    
    @classmethod
    def get_configs_for_profile(cls, risk_profile: str, direction_bias: str) -> Dict:
        """根据风险偏好返回对应的配置"""
        if risk_profile == "conservative" or direction_bias == "long":
            return cls.REGIME_CONFIGS_CONSERVATIVE
        else:
            return cls.REGIME_CONFIGS_AGGRESSIVE_SHORT
    
    def __init__(self, config: DTConfigV32):
        self.config = config
        self.oracle_windows = config.oracle_windows
        
        # 🆕 v3.1: 根据配置选择偏好
        self.active_configs = self.get_configs_for_profile(
            config.risk_profile, 
            config.direction_bias
        )
        
        # SOL 特定调整
        self.sol_vol_multiplier = config.sol_vol_multiplier
        self.bear_reward_multiplier = config.bear_reward_multiplier
    
    def get_action(self, df: pd.DataFrame, idx: int, regime: str) -> Tuple[float, float]:
        """获取 Regime 感知动作和奖励"""
        params = self.active_configs.get(regime, self.active_configs['UNKNOWN'])
        
        if self.config.use_smart_oracle:
            return self._smart_oracle(df, idx, params, regime)
        return self._multi_oracle(df, idx, params)
    
    def _smart_oracle(self, df: pd.DataFrame, idx: int, params: Dict, regime: str = None) -> Tuple[float, float]:
        """智能 Oracle - 考虑路径、回撤、风险
        
        🆕 v3.1: 支持 SOL 特定优化和做空奖励乘数
        """
        lookforward = params['lookforward']
        aggression = params['aggression']
        entry_thresh = params['entry_thresh']
        position_bias = params['position_bias']
        
        if idx + lookforward >= len(df):
            return 0.0, 0.0
        
        prices = df['close'].iloc[idx:idx + lookforward + 1].values
        entry_price = prices[0]
        final_return = prices[-1] / entry_price - 1
        
        # 路径分析
        returns = prices / entry_price - 1
        max_profit = returns.max()
        
        # 最大回撤
        peak = np.maximum.accumulate(prices)
        max_drawdown = ((peak - prices) / peak).max()
        
        # 波动率 (🆕 v3.1: SOL 调整)
        if len(prices) > 2:
            daily_returns = np.diff(prices) / prices[:-1]
            volatility = daily_returns.std() * np.sqrt(24 * 365)
            # SOL 波动率调整
            volatility = volatility / self.sol_vol_multiplier
        else:
            volatility = 0
        
        # 计算动作
        if abs(final_return) < entry_thresh:
            base_action = position_bias
        else:
            base_action = final_return * 25
        
        # 回撤惩罚
        if max_drawdown > 0.04:
            dd_penalty = 0.4
        elif max_drawdown > 0.03:
            dd_penalty = 0.6
        elif max_drawdown > 0.02:
            dd_penalty = 0.8
        else:
            dd_penalty = 1.0
        
        # 波动率调整
        vol_adj = 0.5 if volatility > 0.8 else (0.7 if volatility > 0.6 else 1.0)
        
        # 浮盈回吐惩罚
        profit_penalty = 0.6 if (max_profit > 0.03 and final_return < max_profit * 0.3) else 1.0
        
        action = base_action * aggression * dd_penalty * vol_adj * profit_penalty
        action = float(np.clip(action, -1, 1))
        
        # 🆕 v3.1: 做空奖励乘数 (熊市做空获得更高奖励)
        reward_multiplier = 1.0
        if regime in ['TRENDING_BEAR', 'VOLATILE_BEAR'] and action < 0:
            reward_multiplier = self.bear_reward_multiplier
        
        # 风险调整奖励
        reward = action * final_return * 100 * dd_penalty * reward_multiplier
        
        return action, reward
    
    def _multi_oracle(self, df: pd.DataFrame, idx: int, params: Dict) -> Tuple[float, float]:
        """多时间尺度 Oracle"""
        aggression = params['aggression']
        position_bias = params['position_bias']
        weights = [0.1, 0.2, 0.35, 0.35]
        
        actions = []
        for w, wt in zip(self.oracle_windows, weights):
            if idx + w < len(df):
                ret = df.iloc[idx + w]['close'] / df.iloc[idx]['close'] - 1
                actions.append(np.clip(ret * 25, -1, 1) * wt)
        
        if actions:
            action = float(np.clip(sum(actions) * aggression + position_bias, -1, 1))
            reward = action * (df.iloc[min(idx + 24, len(df)-1)]['close'] / df.iloc[idx]['close'] - 1) * 100
        else:
            action, reward = position_bias, 0.0
        
        return action, reward


# =============================================================================
# 🆕 动态损失权重 (Uncertainty Weighting)
# =============================================================================

class DynamicLossWeights(nn.Module):
    """基于不确定性的动态损失权重"""
    
    def __init__(self, num_tasks: int = 3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, *losses) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0
        weights = {}
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss = total_loss + precision * loss + self.log_vars[i]
            weights[f'w{i}'] = precision.item()
        return total_loss, weights


# =============================================================================
# 🆕 FGM (AMP 兼容)
# =============================================================================

class FGMWithAMP:
    """AMP 兼容的 FGM 对抗训练"""
    
    def __init__(self, model: nn.Module, epsilon: float = 0.3):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}
    
    def attack(self, scaler: GradScaler = None):
        """执行对抗扰动 (在 unscale 梯度后调用)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and ('encoder' in name or 'state_encoder' in name):
                self.backup[name] = param.data.clone()
                if param.grad is not None:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = self.epsilon * param.grad / norm
                        param.data.add_(r_at)
    
    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# 🆕 OHEM (在线难例挖掘)
# =============================================================================

class OHEM:
    """在线难例挖掘"""
    
    def __init__(self, ratio: float = 0.7, min_kept: int = 16):
        self.ratio = ratio
        self.min_kept = min_kept
    
    def __call__(self, loss: torch.Tensor) -> torch.Tensor:
        batch_size = loss.size(0)
        num_kept = max(int(batch_size * self.ratio), min(self.min_kept, batch_size))
        _, hard_indices = torch.topk(loss, num_kept)
        return loss[hard_indices].mean()


# =============================================================================
# EMA
# =============================================================================

class EMA:
    """指数移动平均"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# 🆕 课程学习 Sampler (避免重建 DataLoader)
# =============================================================================

class CurriculumSampler(Sampler):
    """课程学习采样器
    
    🆕 v3.1 修复: 确保早期 epochs 也有足够样本
    """
    
    def __init__(self, difficulties: List[float], config: DTConfigV32):
        self.difficulties = difficulties
        self.config = config
        self.current_epoch = 0
        self.sorted_indices = sorted(range(len(difficulties)), key=lambda i: difficulties[i])
        
        # 🆕 确保最小样本数
        self.min_samples = max(config.batch_size * 2, 64)  # 至少 2 个批次
    
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
    
    def __iter__(self):
        if not self.config.use_curriculum:
            indices = list(range(len(self.difficulties)))
            random.shuffle(indices)
            return iter(indices)
        
        progress = min(1.0, (self.current_epoch + 1) / self.config.curriculum_epochs)
        ratio = self.config.curriculum_start + (1 - self.config.curriculum_start) * progress
        n_samples = int(len(self.sorted_indices) * ratio)
        
        # 🆕 确保至少有 min_samples 个样本
        n_samples = max(n_samples, min(self.min_samples, len(self.sorted_indices)))
        
        indices = self.sorted_indices[:n_samples]
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        if not self.config.use_curriculum:
            return len(self.difficulties)
        progress = min(1.0, (self.current_epoch + 1) / self.config.curriculum_epochs)
        ratio = self.config.curriculum_start + (1 - self.config.curriculum_start) * progress
        n_samples = int(len(self.sorted_indices) * ratio)
        
        # 🆕 确保至少有 min_samples 个样本
        n_samples = max(n_samples, min(self.min_samples, len(self.sorted_indices)))
        return n_samples


# =============================================================================
# 分层 RTG 编码器
# =============================================================================

class HierarchicalRTGEncoder(nn.Module):
    """分层 RTG 编码器"""
    
    def __init__(self, config: DTConfigV32):
        super().__init__()
        self.scales = config.rtg_scales
        
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, config.rtg_hidden_dim),
                nn.LayerNorm(config.rtg_hidden_dim),
                nn.GELU(),
                nn.Linear(config.rtg_hidden_dim, config.rtg_hidden_dim),
            )
            for _ in self.scales
        ])
        
        self.fusion = nn.Sequential(
            nn.Linear(config.rtg_hidden_dim * len(self.scales), config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
        )
    
    def forward(self, rtg_multi: torch.Tensor) -> torch.Tensor:
        encoded = [enc(rtg_multi[..., i:i+1]) for i, enc in enumerate(self.encoders)]
        return self.fusion(torch.cat(encoded, dim=-1))


# =============================================================================
# Decision Transformer v3.2 模型
# =============================================================================

class DecisionTransformerV32(nn.Module):
    """Decision Transformer v3.2"""
    
    def __init__(self, config: DTConfigV32):
        super().__init__()
        self.config = config
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
        )
        
        # RTG encoder
        if config.use_hierarchical_rtg:
            self.rtg_encoder = HierarchicalRTGEncoder(config)
        else:
            self.rtg_encoder = nn.Sequential(
                nn.Linear(1, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
            )
        
        # Transformer
        if TRANSFORMERS_AVAILABLE:
            gpt_config = GPT2Config(
                vocab_size=1,
                n_embd=config.hidden_size,
                n_layer=config.num_layers,
                n_head=config.num_heads,
                n_inner=config.intermediate_size,
                resid_pdrop=config.dropout,
                embd_pdrop=config.dropout,
                attn_pdrop=config.dropout,
            )
            self.transformer = GPT2Model(gpt_config)
            
            if config.use_lora and PEFT_AVAILABLE:
                lora_config = LoraConfig(
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=["c_attn", "c_proj"],
                )
                self.transformer = get_peft_model(self.transformer, lora_config)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # Position embedding
        self.pos_emb = nn.Embedding(config.max_episode_length * 3, config.hidden_size)
        
        # Gated residual
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid(),
        )
        
        # Prediction heads
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.action_dim),
            nn.Tanh(),
        )
        self.direction_head = nn.Linear(config.hidden_size, 3)
        self.reward_head = nn.Linear(config.hidden_size, 1)
    
    def forward(self, states, actions, rtg, timesteps):
        batch_size, seq_len = states.shape[:2]
        
        state_emb = self.state_encoder(states)
        action_emb = self.action_encoder(actions)
        rtg_emb = self.rtg_encoder(rtg)
        
        # Interleaved sequence
        stacked = torch.zeros(batch_size, seq_len * 3, self.config.hidden_size, device=states.device)
        stacked[:, 0::3] = rtg_emb
        stacked[:, 1::3] = state_emb
        stacked[:, 2::3] = action_emb
        
        # Position embedding
        positions = torch.arange(seq_len * 3, device=states.device).unsqueeze(0).expand(batch_size, -1)
        positions = positions.clamp(0, self.config.max_episode_length * 3 - 1)
        stacked = stacked + self.pos_emb(positions)
        
        # Transformer
        if TRANSFORMERS_AVAILABLE:
            out = self.transformer(inputs_embeds=stacked).last_hidden_state
        else:
            out = self.transformer(stacked)
        
        # Extract state outputs
        state_out = out[:, 1::3]
        
        # Gated residual
        gate_weight = self.gate(torch.cat([state_out, state_emb], dim=-1))
        fused = gate_weight * state_out + (1 - gate_weight) * state_emb
        
        return {
            'action_preds': self.action_head(fused),
            'direction_logits': self.direction_head(fused),
            'reward_preds': self.reward_head(fused),
            'hidden': fused,
        }


# =============================================================================
# 数据集
# =============================================================================

class TrajectoryDatasetV32(Dataset):
    """轨迹数据集 v3.2 - aligned with TQC 126-dim obs space.

    State = 122 pre-computed features (from DRL parquet) + 4 simulated env state
    (position_ratio, position_direction, pnl_ratio, drawdown).
    """

    def __init__(self, df: pd.DataFrame, config: DTConfigV32, feature_eng: FeatureEngineerV32,
                 expert: RegimeAwareExpert, is_train: bool = True,
                 scaler=None, feature_cols: List[str] = None,
                 no_scale_cols: List[str] = None):
        self.config = config
        self.is_train = is_train
        self.trajectories = []
        self.scaler = scaler  # Shared RobustScaler (fit on train only)

        # --- Load 122 feature columns from manifest ---
        if feature_cols is None:
            manifest = load_feature_manifest(config.feature_manifest_path)
            feature_cols = manifest["all_features"]
            no_scale_cols = manifest.get("no_scale_features", [])
        self.feature_cols = feature_cols
        self.no_scale_cols = no_scale_cols or []

        # Validate feature columns exist in DataFrame
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            logger.error(f"   Missing {len(missing)} feature columns in data: {missing[:10]}...")
            logger.error(f"   Available columns: {list(df.columns)[:20]}...")
            logger.error(f"   Ensure data is from DRL training parquet (122 pre-computed features)")
            return

        logger.info(f"   Feature columns: {len(feature_cols)} (aligned with TQC)")

        # --- Fill NaN ---
        nan_count = df[feature_cols].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"   {nan_count} NaN values in features, filling with 0")
            df = df.copy()
            df[feature_cols] = df[feature_cols].fillna(0)

        # --- Scaling (RobustScaler, matching DRL v7) ---
        from sklearn.preprocessing import RobustScaler

        scale_cols = [c for c in feature_cols if c not in self.no_scale_cols]
        no_scale_col_indices = [feature_cols.index(c) for c in self.no_scale_cols if c in feature_cols]
        scale_col_indices = [i for i in range(len(feature_cols)) if i not in no_scale_col_indices]

        features_raw = df[feature_cols].values.astype(np.float32)

        if self.scaler is None and is_train:
            # Fit scaler on training data only
            self.scaler = RobustScaler()
            self.scaler.fit(features_raw[:, scale_col_indices])
            logger.info(f"   RobustScaler fit on {len(scale_cols)} scalable features "
                        f"({len(self.no_scale_cols)} no_scale skipped)")

        if self.scaler is not None:
            features_scaled = features_raw.copy()
            features_scaled[:, scale_col_indices] = self.scaler.transform(
                features_raw[:, scale_col_indices]
            )
            # Post-scale clip to [-10, 10] (matching DRL v7)
            features_scaled[:, scale_col_indices] = np.clip(
                features_scaled[:, scale_col_indices], -10, 10
            )
        else:
            features_scaled = features_raw

        # --- Create trajectories ---
        ctx = config.context_length
        n_skipped_nan = 0
        n_skipped_short = 0

        logger.info(f"   Creating trajectories (context_length={ctx})...")

        for start in range(0, len(df) - ctx - 48, ctx // 2):
            try:
                feat_block = features_scaled[start:start + ctx]

                if np.isnan(feat_block).any():
                    n_skipped_nan += 1
                    continue

                # Expert actions and rewards
                actions = []
                rewards = []
                regimes = []

                for i in range(start, start + ctx):
                    regime = feature_eng.get_regime(df, i)
                    regimes.append(regime)
                    action, reward = expert.get_action(df, i, regime)
                    actions.append(action)
                    rewards.append(reward)

                actions_arr = np.array(actions, dtype=np.float32)
                rewards_arr = np.array(rewards, dtype=np.float32).reshape(-1, 1)

                # Simulate 4 env state variables (matching TQC _get_observation)
                env_states = self._simulate_env_state(df, start, ctx, actions_arr)

                # Concatenate: [122 features | 4 env state] = 126 dims
                states = np.hstack([feat_block, env_states])
                assert states.shape == (ctx, 126), \
                    f"State shape mismatch: {states.shape} != ({ctx}, 126)"

                actions_arr = actions_arr.reshape(-1, 1)

                # Hierarchical RTG
                rtg = self._compute_rtg(rewards_arr, df, start)
                timesteps = np.arange(ctx, dtype=np.int64)

                # Difficulty score
                difficulty = np.std(rewards_arr) + np.mean(np.abs(np.diff(actions_arr.flatten())))

                self.trajectories.append({
                    'states': states.astype(np.float32),
                    'actions': actions_arr,
                    'rewards': rewards_arr,
                    'rtg': rtg,
                    'timesteps': timesteps,
                    'difficulty': difficulty,
                    'regimes': regimes,
                })
            except Exception as e:
                n_skipped_short += 1
                continue

        if n_skipped_nan > 0:
            logger.warning(f"   Skipped {n_skipped_nan} trajectories with NaN")
        if n_skipped_short > 0:
            logger.warning(f"   Skipped {n_skipped_short} error trajectories")

        # Sort by difficulty (for curriculum learning)
        if self.trajectories:
            self.trajectories.sort(key=lambda x: x['difficulty'])
            self.difficulties = [t['difficulty'] for t in self.trajectories]
        else:
            self.difficulties = []

        logger.info(f"   Created {len(self.trajectories)} trajectories "
                    f"(state_dim={config.state_dim})")

    def _simulate_env_state(self, df: pd.DataFrame, start: int, ctx: int,
                            actions: np.ndarray) -> np.ndarray:
        """Simulate position_ratio, position_direction, pnl_ratio, drawdown
        from expert actions + price data, matching TQC env _get_observation()."""
        env_states = np.zeros((ctx, 4), dtype=np.float32)
        initial_balance = 10000.0
        total_pnl = 0.0
        peak_balance = initial_balance
        prev_action = 0.0  # No position at start of episode

        closes = df['close'].iloc[start:start + ctx + 1].values

        for t in range(ctx):
            # Env state BEFORE taking action_t (matches TQC obs order)
            cur_balance = initial_balance + total_pnl
            env_states[t, 0] = prev_action                          # position_ratio
            env_states[t, 1] = float(np.sign(prev_action))          # position_direction
            env_states[t, 2] = total_pnl / initial_balance          # pnl_ratio
            env_states[t, 3] = max(0.0, (peak_balance - cur_balance) / peak_balance) \
                if peak_balance > 0 else 0.0                         # drawdown

            # Simulate PnL from holding prev_action position over this bar
            if t + 1 < len(closes) and closes[t] > 0:
                price_return = (closes[t + 1] - closes[t]) / closes[t]
                step_pnl = prev_action * price_return * initial_balance
                total_pnl += step_pnl
                peak_balance = max(peak_balance, initial_balance + total_pnl)

            prev_action = actions[t]

        return env_states

    def _compute_rtg(self, rewards: np.ndarray, df: pd.DataFrame, start: int) -> np.ndarray:
        ctx = self.config.context_length
        scales = self.config.rtg_scales

        rtg = np.zeros((ctx, len(scales)), dtype=np.float32)

        for si, scale in enumerate(scales):
            for i in range(ctx):
                end_idx = min(i + scale, ctx)
                rtg[i, si] = rewards[i:end_idx].sum()

        return rtg

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        t = self.trajectories[idx]
        states = t['states'].copy()

        # Data augmentation (only on features, not env state)
        if self.is_train and self.config.use_augmentation:
            if random.random() < 0.3:
                noise = np.random.randn(*states.shape).astype(np.float32) * self.config.noise_std
                # Don't add noise to last 4 dims (env state) and no_scale features
                noise[:, -4:] = 0  # Preserve env state exactly
                states = states + noise

        return {
            'states': torch.from_numpy(states),
            'actions': torch.from_numpy(t['actions']),
            'rewards': torch.from_numpy(t['rewards']),
            'rtg': torch.from_numpy(t['rtg']),
            'timesteps': torch.from_numpy(t['timesteps']),
        }


# =============================================================================
# 🆕 修复的 R-Drop 损失
# =============================================================================

def compute_rdrop_loss(
    out1: Dict[str, torch.Tensor],
    out2: Dict[str, torch.Tensor],
    config: DTConfigV32,
) -> torch.Tensor:
    """
    修复的 R-Drop 损失
    - 回归任务: MSE 一致性
    - 分类任务: 对称 KL 散度
    """
    # 回归一致性 (动作预测)
    mse_consistency = F.mse_loss(out1['action_preds'], out2['action_preds'])
    
    # 分类一致性 (方向预测) - 对称 KL
    p = F.log_softmax(out1['direction_logits'], dim=-1)
    q = F.log_softmax(out2['direction_logits'], dim=-1)
    p_prob = F.softmax(out1['direction_logits'], dim=-1)
    q_prob = F.softmax(out2['direction_logits'], dim=-1)
    
    kl_pq = F.kl_div(p, q_prob, reduction='batchmean')
    kl_qp = F.kl_div(q, p_prob, reduction='batchmean')
    symmetric_kl = (kl_pq + kl_qp) / 2
    
    return mse_consistency + 0.5 * symmetric_kl


# =============================================================================
# GPU 优化训练器
# =============================================================================

class TrainerV32:
    """GPU 优化训练器 v3.2"""
    
    def __init__(self, model: nn.Module, train_ds: Dataset, val_ds: Dataset, 
                 config: DTConfigV32, device: str):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 🆕 P0: torch.compile 优化
        if config.use_torch_compile:
            self.model = compile_model_if_available(self.model, config.compile_mode)
        
        # 🆕 课程学习 Sampler
        self.curriculum_sampler = CurriculumSampler(train_ds.difficulties, config)
        
        # 🆕 v3.1 修复: 检查批次大小与数据量的兼容性
        min_samples_early = int(len(train_ds) * config.curriculum_start) if config.use_curriculum else len(train_ds)
        if config.batch_size > min_samples_early:
            adjusted_batch_size = max(32, min_samples_early // 2)
            logger.warning(f"[WARN]️ batch_size={config.batch_size} > 早期样本数={min_samples_early}")
            logger.warning(f"   自动调整 batch_size: {config.batch_size} -> {adjusted_batch_size}")
            config.batch_size = adjusted_batch_size
        
        # 优化的 DataLoader
        # 🆕 v3.1 修复: 小数据集时 drop_last=False
        use_drop_last = len(train_ds) >= config.batch_size * 2
        
        loader_kwargs = {
            'batch_size': config.batch_size,
            'num_workers': config.num_workers,
            'pin_memory': config.pin_memory and device == 'cuda',
            'prefetch_factor': config.prefetch_factor if config.num_workers > 0 else None,
            'persistent_workers': config.persistent_workers and config.num_workers > 0,
            'drop_last': use_drop_last,  # 🆕 动态设置
        }
        
        self.train_loader = DataLoader(train_ds, sampler=self.curriculum_sampler, **loader_kwargs)
        
        # 🆕 v3.1 修复: 验证集单独配置，不使用 drop_last
        val_loader_kwargs = {
            'batch_size': min(config.batch_size, len(val_ds)) if val_ds and len(val_ds) > 0 else config.batch_size,
            'num_workers': config.num_workers,
            'pin_memory': config.pin_memory and device == 'cuda',
            'prefetch_factor': config.prefetch_factor if config.num_workers > 0 else None,
            'persistent_workers': config.persistent_workers and config.num_workers > 0,
            'drop_last': False,  # 🆕 验证集永远不 drop
            'shuffle': False,
        }
        self.val_loader = DataLoader(val_ds, **val_loader_kwargs) if val_ds and len(val_ds) > 0 else None
        
        if self.val_loader:
            logger.info(f"📦 Val DataLoader: batch_size={val_loader_kwargs['batch_size']}, samples={len(val_ds)}")
        
        logger.info(f"📦 DataLoader: batch_size={config.batch_size}, workers={config.num_workers}, drop_last={use_drop_last}")
        
        # 🆕 动态损失权重
        if config.use_dynamic_loss_weights:
            self.loss_weights = DynamicLossWeights(num_tasks=3).to(device)
            params = list(model.parameters()) + list(self.loss_weights.parameters())
        else:
            self.loss_weights = None
            params = model.parameters()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        
        # 学习率调度器
        total_steps = len(self.train_loader) * config.epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        logger.info(f"📈 总训练步数: {total_steps}, 预热步数: {warmup_steps}")
        
        # 🆕 v4: 如果步数太少，使用简单的 StepLR 而不是 OneCycleLR
        if total_steps < 100:
            logger.warning(f"[WARN]️ 训练步数过少 ({total_steps})，使用 StepLR 调度器")
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, total_steps // 3), gamma=0.5
            )
            self.use_step_scheduler = True
        else:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.learning_rate,
                total_steps=total_steps, pct_start=config.warmup_ratio,
            )
            self.use_step_scheduler = False
        
        # 混合精度
        self.scaler = GradScaler() if config.use_amp else None
        if config.use_amp:
            logger.info("⚡ AMP: 已启用")
        
        # 🆕 FGM (AMP 兼容)
        self.fgm = FGMWithAMP(model, config.fgm_epsilon) if config.use_fgm else None
        
        # EMA
        self.ema = EMA(model, config.ema_decay) if config.use_ema else None
        
        # 🆕 OHEM
        self.ohem = OHEM(config.ohem_ratio) if config.use_ohem else None
        
        self.best_acc = 0
        self.patience_cnt = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.curriculum_sampler.set_epoch(epoch)
        
        total_loss, total_dir_acc, n = 0, 0, 0
        step_times = []
        grad_norms = []  # 🆕 v4: 记录梯度范数
        
        for step, batch in enumerate(self.train_loader):
            step_start = time.time()
            
            # 非阻塞传输
            states = batch['states'].to(self.device, non_blocking=True)
            actions = batch['actions'].to(self.device, non_blocking=True)
            rewards = batch['rewards'].to(self.device, non_blocking=True)
            rtg = batch['rtg'].to(self.device, non_blocking=True)
            timesteps = batch['timesteps'].to(self.device, non_blocking=True)
            
            # 🆕 v4: 检查输入数据
            if torch.isnan(states).any() or torch.isinf(states).any():
                logger.warning(f"[WARN]️ Step {step}: states 包含 nan/inf, 跳过")
                continue
            
            self.optimizer.zero_grad()
            
            # 混合精度前向
            with autocast(enabled=self.config.use_amp):
                out = self.model(states, actions, rtg, timesteps)
                
                # 🆕 v4: 检查输出
                if torch.isnan(out['action_preds']).any() or torch.isnan(out['direction_logits']).any():
                    logger.warning(f"[WARN]️ Step {step}: 模型输出包含 nan, 跳过")
                    continue
                
                # 计算损失
                action_loss = F.mse_loss(out['action_preds'], actions, reduction='none').mean(dim=(1, 2))
                
                # 方向损失 (Label Smoothing)
                true_dir = (actions.squeeze(-1) > 0.1).long() - (actions.squeeze(-1) < -0.1).long() + 1
                if self.config.use_label_smoothing:
                    dir_loss = F.cross_entropy(
                        out['direction_logits'].view(-1, 3), true_dir.view(-1),
                        label_smoothing=self.config.label_smoothing, reduction='none'
                    ).view(states.size(0), -1).mean(dim=1)
                else:
                    dir_loss = F.cross_entropy(
                        out['direction_logits'].view(-1, 3), true_dir.view(-1), reduction='none'
                    ).view(states.size(0), -1).mean(dim=1)
                
                reward_loss = F.mse_loss(out['reward_preds'], rewards, reduction='none').mean(dim=(1, 2))
                
                # 🆕 OHEM
                if self.ohem:
                    action_loss = self.ohem(action_loss)
                    dir_loss = self.ohem(dir_loss)
                    reward_loss = self.ohem(reward_loss)
                else:
                    action_loss = action_loss.mean()
                    dir_loss = dir_loss.mean()
                    reward_loss = reward_loss.mean()
                
                # 🆕 动态损失权重
                if self.loss_weights:
                    loss, _ = self.loss_weights(action_loss, dir_loss, reward_loss)
                else:
                    loss = (self.config.action_loss_weight * action_loss +
                            self.config.direction_loss_weight * dir_loss +
                            self.config.reward_loss_weight * reward_loss)
                
                # 🆕 v4: R-Drop 稳定性改进
                if self.config.use_rdrop and epoch >= 5:  # 前5个epoch不用R-Drop
                    out2 = self.model(states, actions, rtg, timesteps)
                    rdrop_loss = compute_rdrop_loss(out, out2, self.config)
                    # 🆕 v4: 限制 R-Drop 损失
                    rdrop_loss = torch.clamp(rdrop_loss, 0, 1.0)
                    loss = loss + self.config.rdrop_alpha * rdrop_loss
            
            # 🆕 v4: 检查损失是否有效
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                logger.warning(f"[WARN]️ Step {step}: loss 无效 ({loss.item():.2f}), 跳过")
                self.optimizer.zero_grad()
                continue
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
                
                # 🆕 FGM (AMP 兼容)
                if self.fgm:
                    self.scaler.unscale_(self.optimizer)
                    self.fgm.attack(self.scaler)
                    
                    with autocast(enabled=self.config.use_amp):
                        adv_out = self.model(states, actions, rtg, timesteps)
                        adv_loss = F.mse_loss(adv_out['action_preds'], actions)
                    
                    self.scaler.scale(adv_loss).backward()
                    self.fgm.restore()
                else:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.fgm:
                    self.fgm.attack()
                    adv_out = self.model(states, actions, rtg, timesteps)
                    adv_loss = F.mse_loss(adv_out['action_preds'], actions)
                    adv_loss.backward()
                    self.fgm.restore()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            # 🆕 v4: OneCycleLR 每步更新，StepLR 不在这里更新
            if not self.use_step_scheduler:
                self.scheduler.step()
            
            if self.ema:
                self.ema.update(self.model)
            
            # 统计 (减少 .item() 调用)
            if step % self.config.log_interval == 0:
                total_loss += loss.item()
                pred_dir = out['direction_logits'].argmax(dim=-1)
                total_dir_acc += (pred_dir == true_dir).float().mean().item()
                n += 1
            
            step_times.append(time.time() - step_start)
        
        samples_per_sec = self.config.batch_size / np.mean(step_times) if step_times else 0
        
        return {
            'loss': total_loss / max(n, 1),
            'dir_acc': total_dir_acc / max(n, 1),
            'samples_per_sec': samples_per_sec,
        }
    
    def validate(self) -> Dict[str, float]:
        if not self.val_loader:
            logger.warning("[WARN]️ val_loader 不存在!")
            return {'val_loss': 0, 'val_dir_acc': 0}
        
        # 🆕 v4: 诊断信息
        logger.debug(f"开始验证: {len(self.val_loader)} 批次")
        
        if self.ema:
            self.ema.apply(self.model)
        
        self.model.eval()
        total_loss, total_acc, n = 0, 0, 0
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                states = batch['states'].to(self.device, non_blocking=True)
                actions = batch['actions'].to(self.device, non_blocking=True)
                rtg = batch['rtg'].to(self.device, non_blocking=True)
                timesteps = batch['timesteps'].to(self.device, non_blocking=True)
                
                # 🆕 v4: 检查输入
                if torch.isnan(states).any() or torch.isinf(states).any():
                    logger.warning(f"[WARN]️ 验证批次 {batch_idx}: states 包含 nan/inf")
                    continue
                
                with autocast(enabled=self.config.use_amp):
                    out = self.model(states, actions, rtg, timesteps)
                
                # 🆕 v4: 检查输出
                if torch.isnan(out['direction_logits']).any():
                    logger.warning(f"[WARN]️ 验证批次 {batch_idx}: direction_logits 包含 nan")
                    continue
                
                true_dir = (actions.squeeze(-1) > 0.1).long() - (actions.squeeze(-1) < -0.1).long() + 1
                pred_dir = out['direction_logits'].argmax(dim=-1)
                
                # 🆕 v4: 收集所有预测用于诊断
                all_preds.extend(pred_dir.view(-1).cpu().tolist())
                all_true.extend(true_dir.view(-1).cpu().tolist())
                
                batch_acc = (pred_dir == true_dir).float().mean().item()
                total_acc += batch_acc
                total_loss += F.mse_loss(out['action_preds'], actions).item()
                n += 1
        
        if self.ema:
            self.ema.restore(self.model)
        
        # 🆕 v4: 诊断: 如果没有验证批次
        if n == 0:
            logger.warning("[WARN]️ 验证集为空或所有批次都无效!")
            return {'val_loss': 0, 'val_dir_acc': 0}
        
        # 🆕 v4: 输出预测分布诊断
        if all_preds:
            from collections import Counter
            pred_dist = Counter(all_preds)
            true_dist = Counter(all_true)
            logger.debug(f"预测分布: {dict(pred_dist)}")
            logger.debug(f"真实分布: {dict(true_dist)}")
        
        return {
            'val_loss': total_loss / n,
            'val_dir_acc': total_acc / n,
        }
    
    def train(self, save_dir: str = './models') -> Dict[str, Any]:
        os.makedirs(save_dir, exist_ok=True)
        
        history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
        best_model_state = None
        
        logger.info(f"🚀 开始训练 v3.2 | {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            # 🆕 v4: StepLR 在 epoch 结束后更新
            if self.use_step_scheduler:
                self.scheduler.step()
            
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['dir_acc'])
            history['val_acc'].append(val_metrics.get('val_dir_acc', 0))
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start
            
            if (epoch + 1) % 10 == 0 or epoch < 5:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['dir_acc']*100:.1f}% | "
                    f"Val Acc: {val_metrics.get('val_dir_acc', 0)*100:.1f}% | "
                    f"Speed: {train_metrics['samples_per_sec']:.0f} samples/s | "
                    f"Time: {epoch_time:.1f}s"
                )
            
            # Early stopping
            val_acc = val_metrics.get('val_dir_acc', 0)
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience_cnt = 0
                best_model_state = deepcopy(self.model.state_dict())
                
                # 保存最佳模型
                torch.save({
                    'model_state': best_model_state,
                    'config': self.config,
                    'epoch': epoch,
                    'val_acc': val_acc,
                }, os.path.join(save_dir, 'dt_v32_best.pt'))
            else:
                self.patience_cnt += 1
                if self.patience_cnt >= self.config.early_stopping:
                    logger.info(f"⏹️ Early stopping at epoch {epoch+1}")
                    break
        
        # 恢复最佳模型
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # 保存最终模型
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
            'history': history,
            'best_val_acc': self.best_acc,
        }, os.path.join(save_dir, 'dt_v32_final.pt'))
        
        logger.info(f"ON 训练完成 | 最佳验证准确率: {self.best_acc*100:.1f}%")
        
        return history


# =============================================================================
# 主函数
# =============================================================================

def generate_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """生成合成数据用于测试"""
    np.random.seed(42)
    
    # 模拟价格序列
    price = 50000
    prices = [price]
    
    for _ in range(n_samples - 1):
        # 带趋势和波动的随机游走
        trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3]) * 0.001
        noise = np.random.randn() * 0.02
        price = price * (1 + trend + noise)
        prices.append(price)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'open': prices,
        'high': [p * (1 + abs(np.random.randn()) * 0.01) for p in prices],
        'low': [p * (1 - abs(np.random.randn()) * 0.01) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples),
    })
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Decision Transformer v3.2')
    parser.add_argument('--asset', type=str, default='BTC', choices=['BTC', 'ETH', 'SOL'], help='Asset to train')
    parser.add_argument('--data', type=str, default=None, help='数据文件路径 (auto-resolved from --asset if not set)')
    parser.add_argument('--epochs', type=int, default=1200, help='训练轮数 (v5b: 1200)')
    parser.add_argument('--batch-size', type=int, default=None, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率 (v5b: 1e-5)')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--no-amp', action='store_true', help='禁用混合精度')
    parser.add_argument('--no-regime', action='store_true', help='禁用 Regime 感知')
    parser.add_argument('--no-smart-oracle', action='store_true', help='禁用智能 Oracle')
    parser.add_argument('--save-dir', type=str, default='./models', help='保存目录')
    parser.add_argument('--simple', action='store_true', help='🆕 简化模式: 禁用所有高级特性用于诊断')
    parser.add_argument('--legacy-arch', action='store_true', help='Use legacy 8-layer/384-hidden architecture (pre-v5 regularization)')
    args = parser.parse_args()

    # Auto-resolve data path from --asset if --data not specified
    if not args.data:
        # Look for DRL training parquet (122 pre-computed features, aligned with TQC)
        candidates = [
            f"training/training_data/drl_training/{args.asset}_4H_full.parquet",
            f"data/drl_training/{args.asset}_4H_full.parquet",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                args.data = cand
                break
        if not args.data:
            logger.error(f"No DRL training data found for {args.asset}. Searched: {candidates}")
            logger.error("Run the data pipeline first: python scripts/rebuild_pipeline.py")
            return

    # 设置 CUDA
    gpu_mem = setup_cuda_optimizations()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 🆕 简化模式
    if args.simple:
        logger.info("🔧 简化诊断模式: 禁用所有高级特性")
        config = DTConfigV32(
            asset=args.asset,
            epochs=args.epochs,
            batch_size=args.batch_size or 32,  # 更小的 batch
            learning_rate=args.lr or 1e-4,  # 更大的学习率
            num_workers=min(args.num_workers, 4),
            use_amp=False,  # 禁用 AMP
            use_regime_aware_expert=False,
            use_smart_oracle=False,
            # 禁用所有高级特性
            use_dynamic_loss_weights=False,
            use_label_smoothing=False,
            use_curriculum=False,
            use_fgm=False,
            use_rdrop=False,
            use_ema=False,
            use_ohem=False,
            use_augmentation=False,
            use_torch_compile=False,
            use_lora=False,
            use_hierarchical_rtg=False,
            # 更简单的模型
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            context_length=48,
        )
    else:
        # 配置
        config = DTConfigV32(
            asset=args.asset,
            epochs=args.epochs,
            batch_size=args.batch_size or (512 if gpu_mem >= 24 else 256 if gpu_mem >= 16 else 128),
            learning_rate=args.lr,
            num_workers=args.num_workers,
            use_amp=not args.no_amp,
            use_regime_aware_expert=not args.no_regime,
            use_smart_oracle=not args.no_smart_oracle,
        )

    # --legacy-arch: restore pre-v5 8-layer architecture for comparison
    if args.legacy_arch and not args.simple:
        logger.info("🔄 Legacy architecture: restoring 8-layer/384-hidden config")
        config.num_layers = 8
        config.hidden_size = 384
        config.num_heads = 8
        config.intermediate_size = 1536
        config.context_length = 96
        config.dropout = 0.1
        config.weight_decay = 0.01
        config.lora_r = 32
        config.lora_alpha = 64

    logger.info(f"📋 配置: asset={config.asset}, batch_size={config.batch_size}, epochs={config.epochs}, state_dim={config.state_dim}")
    logger.info(f"   Architecture: {config.num_layers}L/{config.hidden_size}H/{config.num_heads}heads, ctx={config.context_length}, dropout={config.dropout}")
    logger.info(f"   Regime感知: {config.use_regime_aware_expert}, 智能Oracle: {config.use_smart_oracle}")
    logger.info(f"   动态损失权重: {config.use_dynamic_loss_weights}, OHEM: {config.use_ohem}")
    logger.info(f"   FGM: {config.use_fgm}, R-Drop: {config.use_rdrop}")
    
    # Load DRL training data (122 pre-computed features, aligned with TQC)
    logger.info(f"Loading data: {args.data}")
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    elif args.data.endswith('.csv'):
        df = pd.read_csv(args.data, parse_dates=['timestamp'])
    else:
        try:
            df = pd.read_parquet(args.data)
        except Exception:
            df = pd.read_csv(args.data, parse_dates=['timestamp'])

    logger.info(f"   Rows: {len(df)}, Columns: {len(df.columns)}")

    # Load feature manifest
    manifest = load_feature_manifest(config.feature_manifest_path)
    feature_cols = manifest["all_features"]
    no_scale_cols = manifest.get("no_scale_features", [])
    logger.info(f"   Feature manifest: {len(feature_cols)} features, "
                f"{len(no_scale_cols)} no_scale")

    # Validate 122 features exist in parquet
    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        logger.error(f"Missing {len(missing_feats)} features in data: {missing_feats[:10]}...")
        logger.error(f"Data must be DRL training parquet with 122 pre-computed features")
        return

    # Validate OHLCV columns (needed for expert/regime/env_state simulation)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing OHLCV columns: {missing_cols}")
        return

    # Clean data
    before_len = len(df)
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    if len(df) < before_len:
        logger.warning(f"   Dropped {before_len - len(df)} rows with NaN OHLCV")

    min_required = config.context_length + 100
    if len(df) < min_required:
        logger.error(f"Insufficient data: need {min_required}+, got {len(df)}")
        return

    # Feature engineering (kept ONLY for get_regime() - features come from parquet)
    feature_eng = FeatureEngineerV32(config)
    expert = RegimeAwareExpert(config)

    # Train/val split
    split_idx = int(len(df) * (1 - config.val_ratio))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    # Create training dataset (fits scaler on train data)
    logger.info(f"Creating training set: {len(train_df)} rows")
    train_ds = TrajectoryDatasetV32(
        train_df, config, feature_eng, expert, is_train=True,
        feature_cols=feature_cols, no_scale_cols=no_scale_cols,
    )

    if len(train_ds) == 0:
        logger.error("Training dataset is empty!")
        logger.error(f"   Need at least {config.context_length + 48} rows, got {len(train_df)}")
        return

    # Create validation dataset (reuse train scaler)
    logger.info(f"Creating validation set: {len(val_df)} rows")
    val_ds = TrajectoryDatasetV32(
        val_df, config, feature_eng, expert, is_train=False,
        scaler=train_ds.scaler, feature_cols=feature_cols, no_scale_cols=no_scale_cols,
    )

    if len(val_ds) == 0:
        logger.warning("Validation dataset is empty, skipping validation")
        val_ds = None

    logger.info(f"Dataset ready: train={len(train_ds)} trajectories, "
                f"val={len(val_ds) if val_ds else 0} trajectories, state_dim=126")

    # Create model
    model = DecisionTransformerV32(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params/1e6:.2f}M (state_encoder: Linear(126, {config.hidden_size}))")

    # Save directory per-asset
    save_dir = os.path.join(args.save_dir, 'decision_transformer', config.asset)

    # Train
    trainer = TrainerV32(model, train_ds, val_ds, config, device)
    history = trainer.train(save_dir)

    # Save scaler alongside model for runtime inference
    try:
        import joblib
        scaler_path = os.path.join(save_dir, 'dt_scaler.pkl')
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(train_ds.scaler, scaler_path)
        logger.info(f"Scaler saved: {scaler_path}")
    except Exception as e:
        logger.warning(f"Failed to save scaler: {e}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

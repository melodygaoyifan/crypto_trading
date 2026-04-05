"""
DRL v5.5 "Pragmatic" 配置
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class RiskConfig:
    """风控配置"""
    max_leverage: float = 3.0
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.15
    max_drawdown_pct: float = 0.25
    max_daily_loss_pct: float = 0.15
    max_single_position_pct: float = 0.40
    cvar_alpha: float = 0.05
    cvar_weight: float = 0.15


@dataclass
class DRLConfig:
    """DRL v5.5 完整配置"""
    
    # === 基础 ===
    asset: str = "SOL"
    version: str = "5.5.0"
    codename: str = "Pragmatic"
    
    # === 风控 ===
    risk: RiskConfig = field(default_factory=RiskConfig)
    
    # === 双向策略 ===
    use_bidirectional: bool = True
    base_short_bias: float = 0.10
    adaptive_bias: bool = True
    
    # === 架构 ===
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.1
    
    # === 状态空间 ===
    state_dim: int = 128
    action_dim: int = 1
    
    # === Context ===
    context_length: int = 168
    max_episode_length: int = 1024
    
    # === RTG-Free ===
    use_rtg_free: bool = True
    use_learned_rtg: bool = True
    rtg_hidden_dim: int = 64
    rtg_scales: List[int] = field(default_factory=lambda: [1, 4, 12, 24, 48, 96, 168])
    
    # === 软 MoE (6 专家) ===
    use_moe: bool = True
    num_experts: int = 6  # 匹配 6-Regime
    num_experts_per_tok: int = 2
    soft_routing: bool = True
    
    # === Cross-Asset ===
    use_cross_asset: bool = True
    cross_asset_dim: int = 16
    
    # === GMM (6 Regime) ===
    use_gmm_labels: bool = True
    gmm_n_components: int = 6  # 6-Regime: STRONG_BEAR/MODERATE_BEAR/NEUTRAL/VOLATILE/MODERATE_BULL/STRONG_BULL
    gmm_model_path: str = "./models/gmm_regime.pkl"
    
    # === LoRA ===
    use_lora: bool = True
    lora_r: int = 48
    lora_alpha: int = 96
    lora_dropout: float = 0.05
    
    # === 训练 ===
    epochs: int = 300
    batch_size: int = 256
    gradient_accumulation: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 0.02
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # === 集成 ===
    use_ensemble: bool = True
    num_ensemble: int = 3
    ensemble_method: str = "weighted"
    
    # === GPU 优化 ===
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    use_gradient_checkpointing: bool = True
    use_torch_compile: bool = True
    compile_mode: str = "max-autotune"
    
    # === Offline RL ===
    use_iql: bool = True
    iql_expectile: float = 0.7
    use_cql: bool = True
    cql_alpha: float = 0.8
    
    # === 奖励权重 ===
    reward_weights: Dict = field(default_factory=lambda: {
        'return': 0.35,
        'sharpe': 0.25,
        'sortino': 0.15,
        'calmar': 0.10,
        'cvar': 0.15,
    })
    
    # === 准确率技术 ===
    use_contrastive: bool = True
    contrastive_weight: float = 0.25
    use_rdrop: bool = True
    rdrop_alpha: float = 0.4
    use_fgm: bool = True
    fgm_epsilon: float = 0.4
    use_ohem: bool = True
    ohem_ratio: float = 0.6
    use_ema: bool = True
    ema_decay: float = 0.9995
    
    # === 验证 ===
    val_ratio: float = 0.15
    early_stopping: int = 40
    log_interval: int = 5
    
    # === 路径 ===
    save_dir: str = "./models"
    log_dir: str = "./logs"
    data_dir: str = "./data"
    
    def get_device(self) -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_amp_dtype(self):
        if self.amp_dtype == "bfloat16":
            return torch.bfloat16
        elif self.amp_dtype == "float16":
            return torch.float16
        return torch.float32


# 预设配置
CONFIG_PRESETS = {
    'default': DRLConfig(),
    
    'fast': DRLConfig(
        epochs=50,
        batch_size=128,
        num_layers=8,
        use_ensemble=False,
        use_torch_compile=False,
    ),
    
    'full': DRLConfig(
        epochs=500,
        batch_size=384,
        num_layers=16,
        num_ensemble=5,
        use_torch_compile=True,
    ),
    
    'low_memory': DRLConfig(
        batch_size=64,
        gradient_accumulation=8,
        hidden_size=512,
        num_layers=8,
        use_ensemble=False,
        use_gradient_checkpointing=True,
    ),
}


def get_config(preset: str = 'default', **overrides) -> DRLConfig:
    """获取配置"""
    config = CONFIG_PRESETS.get(preset, CONFIG_PRESETS['default'])
    
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

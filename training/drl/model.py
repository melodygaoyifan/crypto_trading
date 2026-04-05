"""
DRL v5.5 "Pragmatic" 核心模型

特点:
- RTG-Free 模式
- 双向专家 MoE (6专家)
- 软路由门控
- 简化 Cross-Asset
- GPU 深度优化

🆕 v3.1: 高风险做空优化
- 激进做空偏好配置
- SOL 特定调整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

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


# =============================================================================
# 🆕 v3.1: 风险偏好配置
# =============================================================================

def get_biases_for_profile(risk_profile: str = "aggressive", direction_bias: str = "short") -> Dict[str, float]:
    """
    根据风险偏好返回对应的方向偏好
    
    Args:
        risk_profile: aggressive, moderate, conservative
        direction_bias: short, neutral, long
    
    Returns:
        BIASES 字典
    """
    if risk_profile == "conservative" or direction_bias == "long":
        # 保守/做多配置
        return {
            'STRONG_BEAR': -0.25,
            'MODERATE_BEAR': -0.10,
            'NEUTRAL': 0.00,
            'VOLATILE_CHOP': 0.00,
            'MODERATE_BULL': 0.05,
            'STRONG_BULL': 0.15,
        }
    elif risk_profile == "moderate" or direction_bias == "neutral":
        # 中性配置
        return {
            'STRONG_BEAR': -0.30,
            'MODERATE_BEAR': -0.15,
            'NEUTRAL': 0.00,
            'VOLATILE_CHOP': 0.00,
            'MODERATE_BULL': 0.03,
            'STRONG_BULL': 0.10,
        }
    else:
        # 激进做空配置 (默认)
        return {
            'STRONG_BEAR': -0.40,
            'MODERATE_BEAR': -0.20,
            'NEUTRAL': 0.00,
            'VOLATILE_CHOP': -0.05,
            'MODERATE_BULL': 0.00,
            'STRONG_BULL': 0.08,
        }


# =============================================================================
# RTG-Free 编码器
# =============================================================================

class RTGFreeEncoder(nn.Module):
    """可学习的 RTG 编码器 - 不需要真实 RTG 输入
    
    🆕 v3.1: 修复 regime 维度为 6 (HMATS 6-Regime)
    """
    
    def __init__(self, hidden_size: int, rtg_hidden_dim: int = 64, n_regimes: int = 6):
        super().__init__()
        
        # 可学习基础 RTG
        self.base_rtg = nn.Parameter(torch.zeros(1, 1, rtg_hidden_dim))
        
        # Regime 条件 RTG - 🆕 v3.1: 支持 6 个 regime
        self.regime_rtg_gen = nn.Sequential(
            nn.Linear(n_regimes, rtg_hidden_dim),  # 🆕 4 -> 6
            nn.GELU(),
            nn.Linear(rtg_hidden_dim, rtg_hidden_dim),
        )
        
        # 历史统计 RTG
        self.history_encoder = nn.Sequential(
            nn.Linear(8, rtg_hidden_dim),
            nn.GELU(),
            nn.Linear(rtg_hidden_dim, rtg_hidden_dim),
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(rtg_hidden_dim * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
    
    def forward(self, regime_probs: torch.Tensor = None, history_stats: torch.Tensor = None,
                batch_size: int = 1, seq_len: int = 1) -> torch.Tensor:
        device = self.base_rtg.device
        
        base = self.base_rtg.expand(batch_size, seq_len, -1)
        
        if regime_probs is not None:
            regime_rtg = self.regime_rtg_gen(regime_probs).unsqueeze(1).expand(-1, seq_len, -1)
        else:
            regime_rtg = torch.zeros(batch_size, seq_len, base.size(-1), device=device)
        
        if history_stats is not None:
            history_rtg = self.history_encoder(history_stats).unsqueeze(1).expand(-1, seq_len, -1)
        else:
            history_rtg = torch.zeros(batch_size, seq_len, base.size(-1), device=device)
        
        return self.fusion(torch.cat([base, regime_rtg, history_rtg], dim=-1))


# =============================================================================
# 双向专家
# =============================================================================

class BidirectionalExpert(nn.Module):
    """双向专家 - 各有方向偏好 (6 Regime)
    
    🆕 v3.1: 高风险做空优化
    """
    
    # 原始保守偏好 (备用)
    BIASES_CONSERVATIVE = {
        'STRONG_BEAR': -0.25,
        'MODERATE_BEAR': -0.10,
        'NEUTRAL': 0.00,
        'VOLATILE_CHOP': 0.00,
        'MODERATE_BULL': 0.05,
        'STRONG_BULL': 0.15,
    }
    
    # 🆕 激进做空偏好 (默认)
    BIASES_AGGRESSIVE_SHORT = {
        'STRONG_BEAR': -0.40,    # 原 -0.25, +60%
        'MODERATE_BEAR': -0.20,  # 原 -0.10, +100%
        'NEUTRAL': 0.00,
        'VOLATILE_CHOP': -0.05,  # 原 0.00, 震荡轻微做空
        'MODERATE_BULL': 0.00,   # 原 0.05, 减少做多
        'STRONG_BULL': 0.08,     # 原 0.15, -47%
    }
    
    # 默认使用激进做空配置
    BIASES = BIASES_AGGRESSIVE_SHORT
    
    def __init__(self, hidden_size: int, expert_type: str):
        super().__init__()
        self.expert_type = expert_type
        
        intermediate = hidden_size * 4
        self.fc1 = nn.Linear(hidden_size, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden_size)
        self.act = nn.GELU()
        self.direction_bias = nn.Parameter(torch.tensor([self.BIASES.get(expert_type, 0.0)]))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fc2(self.act(self.fc1(x))), self.direction_bias.expand(x.size(0), 1)


class SoftMoE(nn.Module):
    """软路由 MoE - 6专家 (匹配 6-Regime)
    
    STRONG_BEAR / MODERATE_BEAR / NEUTRAL / VOLATILE_CHOP / MODERATE_BULL / STRONG_BULL
    
    🆕 v3.1: 高风险做空优化
    """
    
    EXPERT_TYPES = ['STRONG_BEAR', 'MODERATE_BEAR', 'NEUTRAL', 
                    'VOLATILE_CHOP', 'MODERATE_BULL', 'STRONG_BULL']
    
    # 🆕 激进做空偏好 (默认)
    BIASES = {
        'STRONG_BEAR': -0.40,    # 原 -0.25
        'MODERATE_BEAR': -0.20,  # 原 -0.10
        'NEUTRAL': 0.00,
        'VOLATILE_CHOP': -0.05,  # 原 0.00
        'MODERATE_BULL': 0.00,   # 原 0.05
        'STRONG_BULL': 0.08,     # 原 0.15
    }
    
    def __init__(self, hidden_size: int, num_experts: int = 6, num_per_tok: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.num_per_tok = num_per_tok
        
        self.experts = nn.ModuleList([
            BidirectionalExpert(hidden_size, self.EXPERT_TYPES[i])
            for i in range(num_experts)
        ])
        
        # 更新专家的方向偏好
        for i, expert in enumerate(self.experts):
            expert_type = self.EXPERT_TYPES[i]
            expert.direction_bias = nn.Parameter(
                torch.tensor([self.BIASES.get(expert_type, 0.0)])
            )
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_experts),
        )
        
        self.load_balance_weight = 0.01
    
    def forward(self, x: torch.Tensor, regime_hint: torch.Tensor = None) -> Dict:
        batch, seq_len, hidden = x.shape
        
        gate_logits = self.gate(x)
        
        if regime_hint is not None:
            hint_bonus = F.one_hot(regime_hint.clamp(0, 5), self.num_experts).float()
            gate_logits = gate_logits + hint_bonus.unsqueeze(1) * 1.5
        
        if self.training:
            routing_probs = F.gumbel_softmax(gate_logits, tau=0.5, hard=False)
        else:
            routing_probs = F.softmax(gate_logits, dim=-1)
        
        top_probs, top_indices = torch.topk(routing_probs, self.num_per_tok, dim=-1)
        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        output = torch.zeros_like(x)
        direction_biases = []
        
        flat_x = x.view(-1, hidden)
        flat_indices = top_indices.view(-1, self.num_per_tok)
        flat_probs = top_probs.view(-1, self.num_per_tok)
        
        for i, expert in enumerate(self.experts):
            mask = (flat_indices == i).any(dim=-1)
            if not mask.any():
                continue
            
            expert_out, expert_bias = expert(flat_x[mask])
            weight_mask = (flat_indices[mask] == i).float()
            weights = (flat_probs[mask] * weight_mask).sum(dim=-1, keepdim=True)
            
            output.view(-1, hidden)[mask] += weights * expert_out
            direction_biases.append(expert_bias.mean() * weights.mean())

        direction_bias = torch.stack(direction_biases).mean() if direction_biases else torch.tensor(0.0, device=x.device)
        
        expert_usage = routing_probs.mean(dim=(0, 1))
        load_balance_loss = self.num_experts * (expert_usage ** 2).sum() * self.load_balance_weight
        
        return {'output': output, 'direction_bias': direction_bias, 
                'routing_probs': routing_probs, 'load_balance_loss': load_balance_loss}


# =============================================================================
# 简化 Cross-Asset
# =============================================================================

class SimplifiedCrossAsset(nn.Module):
    """简化的 Cross-Asset - BTC/ETH 关键统计"""
    
    def __init__(self, hidden_size: int, cross_dim: int = 16):
        super().__init__()
        
        self.btc_encoder = nn.Sequential(
            nn.Linear(cross_dim // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size // 4),
        )
        
        self.eth_encoder = nn.Sequential(
            nn.Linear(cross_dim // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size // 4),
        )
        
        self.lead_lag_encoder = nn.Sequential(
            nn.Linear(4, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size // 4),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 4 * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
    
    def forward(self, main_hidden: torch.Tensor, btc_feat: torch.Tensor = None,
                eth_feat: torch.Tensor = None, lead_lag: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len, hidden = main_hidden.shape
        device = main_hidden.device
        
        btc_enc = self.btc_encoder(btc_feat).unsqueeze(1).expand(-1, seq_len, -1) if btc_feat is not None \
                  else torch.zeros(batch, seq_len, hidden // 4, device=device)
        eth_enc = self.eth_encoder(eth_feat).unsqueeze(1).expand(-1, seq_len, -1) if eth_feat is not None \
                  else torch.zeros(batch, seq_len, hidden // 4, device=device)
        lead_lag_enc = self.lead_lag_encoder(lead_lag).unsqueeze(1).expand(-1, seq_len, -1) if lead_lag is not None \
                       else torch.zeros(batch, seq_len, hidden // 4, device=device)
        
        return self.fusion(torch.cat([main_hidden, btc_enc, eth_enc, lead_lag_enc], dim=-1))


# =============================================================================
# 主模型
# =============================================================================

class DRLModelV55(nn.Module):
    """DRL v5.5 主模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )
        
        # RTG-Free
        if config.use_rtg_free:
            self.rtg_encoder = RTGFreeEncoder(
                config.hidden_size, 
                config.rtg_hidden_dim,
                n_regimes=6  # 🆕 v4: 修复维度匹配 (HMATS 6-Regime)
            )
        
        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
        )
        
        # 位置嵌入
        self.pos_emb = nn.Embedding(config.max_episode_length * 3, config.hidden_size)
        
        # Transformer Backbone
        if TRANSFORMERS_AVAILABLE:
            gpt_config = GPT2Config(
                vocab_size=1, n_embd=config.hidden_size, n_layer=config.num_layers,
                n_head=config.num_heads, n_inner=config.intermediate_size,
                resid_pdrop=config.dropout, embd_pdrop=config.dropout, attn_pdrop=config.dropout,
            )
            self.transformer = GPT2Model(gpt_config)
            
            if config.use_gradient_checkpointing:
                self.transformer.gradient_checkpointing_enable()
            
            if config.use_lora and PEFT_AVAILABLE:
                lora_config = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha,
                                        lora_dropout=config.lora_dropout, target_modules=["c_attn", "c_proj"])
                self.transformer = get_peft_model(self.transformer, lora_config)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size, nhead=config.num_heads,
                dim_feedforward=config.intermediate_size, dropout=config.dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # MoE
        if config.use_moe:
            self.moe = SoftMoE(config.hidden_size, config.num_experts, config.num_experts_per_tok)
        
        # Cross-Asset
        if config.use_cross_asset:
            self.cross_asset = SimplifiedCrossAsset(config.hidden_size, config.cross_asset_dim)
        
        # Gate
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid(),
        )
        
        # 预测头
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.action_dim),
            nn.Tanh(),
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 3),
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Regime 预测 (6 类)
        self.regime_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 6),
        )
        
        # IQL
        if config.use_iql:
            self.q_head = nn.Sequential(
                nn.Linear(config.hidden_size + config.action_dim, config.hidden_size // 2),
                nn.GELU(),
                nn.Linear(config.hidden_size // 2, 1),
            )
            self.v_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.GELU(),
                nn.Linear(config.hidden_size // 2, 1),
            )
        
        # 对比学习
        if config.use_contrastive:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.GELU(),
                nn.Linear(config.hidden_size // 2, 128),
            )
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor, 
                timesteps: torch.Tensor = None, regime_probs: torch.Tensor = None,
                history_stats: torch.Tensor = None, btc_features: torch.Tensor = None,
                eth_features: torch.Tensor = None, lead_lag: torch.Tensor = None,
                return_q: bool = False) -> Dict[str, torch.Tensor]:
        
        batch, seq_len = states.shape[:2]
        device = states.device
        
        # 编码
        state_emb = self.state_encoder(states)
        action_emb = self.action_encoder(actions)
        
        # RTG-Free
        if self.config.use_rtg_free:
            rtg_emb = self.rtg_encoder(regime_probs, history_stats, batch, seq_len)
        else:
            rtg_emb = torch.zeros(batch, seq_len, self.config.hidden_size, device=device)
        
        # Interleave
        stacked = torch.zeros(batch, seq_len * 3, self.config.hidden_size, device=device)
        stacked[:, 0::3] = rtg_emb
        stacked[:, 1::3] = state_emb
        stacked[:, 2::3] = action_emb
        
        # Position
        positions = torch.arange(seq_len * 3, device=device).unsqueeze(0).expand(batch, -1)
        positions = positions.clamp(0, self.config.max_episode_length * 3 - 1)
        stacked = stacked + self.pos_emb(positions)
        
        # Transformer
        if TRANSFORMERS_AVAILABLE and hasattr(self.transformer, 'base_model'):
            out = self.transformer.base_model(inputs_embeds=stacked).last_hidden_state
        elif TRANSFORMERS_AVAILABLE:
            out = self.transformer(inputs_embeds=stacked).last_hidden_state
        else:
            out = self.transformer(stacked)
        
        state_out = out[:, 1::3]
        
        # Regime 预测
        regime_logits = self.regime_predictor(state_out[:, -1])
        pred_regime_probs = F.softmax(regime_logits, dim=-1)
        routing_regime = regime_probs if regime_probs is not None else pred_regime_probs
        regime_hint = routing_regime.argmax(dim=-1)
        
        # MoE
        moe_result = None
        if self.config.use_moe:
            moe_result = self.moe(state_out, regime_hint)
            state_out = moe_result['output']
        
        # Cross-Asset
        if self.config.use_cross_asset:
            state_out = self.cross_asset(state_out, btc_features, eth_features, lead_lag)
        
        # Gated residual
        gate_weight = self.gate(torch.cat([state_out, self.state_encoder(states)], dim=-1))
        fused = gate_weight * state_out + (1 - gate_weight) * self.state_encoder(states)
        
        # 预测
        action_preds = self.action_head(fused)
        direction_logits = self.direction_head(fused)
        confidence = self.confidence_head(fused)
        
        if moe_result is not None:
            action_preds = action_preds + moe_result['direction_bias'].view(1, 1, 1)
        
        result = {
            'action_preds': action_preds,
            'direction_logits': direction_logits,
            'confidence': confidence,
            'hidden': fused,
            'regime_logits': regime_logits,
            'predicted_regime_probs': pred_regime_probs,
        }
        
        if moe_result:
            result['moe_load_balance_loss'] = moe_result['load_balance_loss']
            result['routing_probs'] = moe_result['routing_probs']
        
        if return_q and self.config.use_iql:
            result['q_values'] = self.q_head(torch.cat([fused, actions], dim=-1))
            result['v_values'] = self.v_head(fused)
        
        if self.config.use_contrastive:
            result['contrastive_features'] = self.contrastive_proj(fused[:, -1])
        
        return result


# =============================================================================
# 集成模型
# =============================================================================

class EnsembleModel(nn.Module):
    """集成模型 - 3模型集成"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.models = nn.ModuleList([DRLModelV55(config) for _ in range(config.num_ensemble)])
        self.weights = nn.Parameter(torch.ones(config.num_ensemble) / config.num_ensemble)
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = [m(**kwargs) for m in self.models]
        weights = F.softmax(self.weights, dim=0)
        
        result = {}
        
        # 加权平均
        action_preds = torch.stack([o['action_preds'] for o in outputs], dim=0)
        result['action_preds'] = (action_preds * weights.view(-1, 1, 1, 1)).sum(dim=0)
        
        direction_logits = torch.stack([o['direction_logits'] for o in outputs], dim=0)
        result['direction_logits'] = (direction_logits * weights.view(-1, 1, 1, 1)).sum(dim=0)
        
        confidences = torch.stack([o['confidence'] for o in outputs], dim=0)
        mean_conf = (confidences * weights.view(-1, 1, 1, 1)).sum(dim=0)
        
        # 一致性
        directions = torch.stack([o['direction_logits'].argmax(dim=-1) for o in outputs], dim=0)
        agreement = (directions == directions[0]).float().mean(dim=0)
        result['confidence'] = mean_conf * (0.8 + 0.4 * agreement.unsqueeze(-1))
        
        result['hidden'] = outputs[0]['hidden']
        result['regime_logits'] = outputs[0]['regime_logits']
        result['predicted_regime_probs'] = outputs[0]['predicted_regime_probs']
        result['model_agreement'] = agreement
        
        if 'moe_load_balance_loss' in outputs[0]:
            result['moe_load_balance_loss'] = sum(o['moe_load_balance_loss'] for o in outputs) / len(outputs)
        
        return result

"""
Stage 6: FiLM conditional adapter - regime-conditioned feature extraction.

Based on MacroHFT (KDD'24): Feature-wise Linear Modulation (FiLM) uses
regime probabilities to condition hidden representations, rather than
treating them as flat input features.

Usage:
    from models.film_extractor import LSTMFiLMExtractor

    policy_kwargs = {
        "features_extractor_class": LSTMFiLMExtractor,
        "features_extractor_kwargs": {"single_obs_dim": 126, "regime_dim": 8},
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    import gymnasium as gym
except ImportError:
    import gym


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation conditioned on regime probabilities.

    h' = gamma(regime) * h + beta(regime)
    """

    def __init__(self, regime_dim: int, hidden_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(regime_dim, 64),
            nn.SiLU(),
            nn.Linear(64, hidden_dim * 2),  # gamma + beta
        )
        # Initialize close to identity: gamma~1, beta~0
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.constant_(self.adapter[-1].bias[:hidden_dim], 1.0)  # gamma=1
        nn.init.zeros_(self.adapter[-1].bias[hidden_dim:])           # beta=0

    def forward(self, h: torch.Tensor, regime_probs: torch.Tensor) -> torch.Tensor:
        params = self.adapter(regime_probs)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma * h + beta


class LSTMFiLMExtractor(BaseFeaturesExtractor):
    """
    LSTM + FiLM regime conditioning for frame-stacked observations.

    Separates regime_proba (8 dims) from obs, processes non-regime features
    through LSTM, then applies FiLM conditioning using regime probabilities.

    Obs layout (single timestep, 126 dims):
        [0:114]   = scaled features (102 base + 5 denoised + 7 external)
        [114:122] = regime_proba_0..7 (8 dims, no-scale)
        [122:126] = env state (position_ratio, position_direction, pnl_ratio, drawdown)

    With VecFrameStack(8): obs = (batch, 126*8) = (batch, 1008)
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 single_obs_dim: int = 126, regime_dim: int = 8,
                 features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        total_dim = int(np.prod(observation_space.shape))
        self.single_obs_dim = single_obs_dim
        self.n_stack = total_dim // single_obs_dim
        self.regime_dim = regime_dim
        self.non_regime_dim = single_obs_dim - regime_dim  # 126 - 8 = 118

        self.input_proj = nn.Linear(self.non_regime_dim, 256)
        self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True, dropout=0.1)
        self.film = FiLMLayer(regime_dim, 256)
        self.output = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, features_dim))

        self._debug_count = 0

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.n_stack, self.single_obs_dim)

        # regime_proba at [-12:-4] in each obs (indices 114:122)
        regime_probs = x[:, -1, -12:-4]  # (B, 8) -- latest timestep only

        # Verify regime probs sum ~ 1.0 for first 10 batches
        if self._debug_count < 10:
            with torch.no_grad():
                prob_sum = regime_probs.sum(dim=-1).mean().item()
                if self._debug_count == 0:
                    print(f"[FiLM DEBUG] regime_probs shape={regime_probs.shape}, "
                          f"mean_sum={prob_sum:.4f}, "
                          f"sample={regime_probs[0].cpu().numpy()}")
            self._debug_count += 1

        # Non-regime features: [0:114] + [122:126]
        features = torch.cat([x[:, :, :-12], x[:, :, -4:]], dim=-1)  # (B, n_stack, 118)

        h = F.silu(self.input_proj(features))
        h, _ = self.lstm(h)
        h = h[:, -1, :]                # last timestep
        h = self.film(h, regime_probs)  # FiLM conditioning
        return self.output(h)


class FiLMLayerLite(nn.Module):
    """Lite FiLM: direct linear mapping, no hidden layer.

    Original FiLMLayer: Linear(8->64) + SiLU + Linear(64->512) = ~41K params
    Lite:               Linear(8->512) = ~4K params
    """

    def __init__(self, regime_dim: int, hidden_dim: int):
        super().__init__()
        self.adapter = nn.Linear(regime_dim, hidden_dim * 2)
        # Identity init: gamma~1, beta~0
        nn.init.zeros_(self.adapter.weight)
        nn.init.constant_(self.adapter.bias[:hidden_dim], 1.0)
        nn.init.zeros_(self.adapter.bias[hidden_dim:])

    def forward(self, h: torch.Tensor, regime_probs: torch.Tensor) -> torch.Tensor:
        params = self.adapter(regime_probs)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma * h + beta


class LSTMFiLMLiteExtractor(BaseFeaturesExtractor):
    """
    LSTM + FiLM-Lite: same LSTM backbone size as Stage 4 Plain LSTM
    (hidden=128, 1 layer), with a single-Linear FiLM adapter on top.
    Total params ~150K (vs ~130K plain LSTM).

    Obs layout identical to LSTMFiLMExtractor (126 dims per timestep).
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 single_obs_dim: int = 126, regime_dim: int = 8,
                 features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        total_dim = int(np.prod(observation_space.shape))
        self.single_obs_dim = single_obs_dim
        self.n_stack = total_dim // single_obs_dim
        self.regime_dim = regime_dim
        self.non_regime_dim = single_obs_dim - regime_dim  # 118

        # Match Stage 4 LSTM backbone: hidden=128, 1 layer
        self.lstm = nn.LSTM(self.non_regime_dim, 128, num_layers=1, batch_first=True)
        self.film = FiLMLayerLite(regime_dim, 128)  # +2K params
        self.ln = nn.LayerNorm(128)
        self.output = nn.Linear(128, features_dim)

        self._debug_count = 0

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.n_stack, self.single_obs_dim)

        regime_probs = x[:, -1, -12:-4]  # (B, 8) latest timestep

        if self._debug_count < 10:
            with torch.no_grad():
                if self._debug_count == 0:
                    total_params = sum(p.numel() for p in self.parameters())
                    prob_sum = regime_probs.sum(dim=-1).mean().item()
                    print(f"[FiLM-Lite] params={total_params:,}, "
                          f"regime_sum={prob_sum:.4f}, "
                          f"sample={regime_probs[0].cpu().numpy()}")
                    assert total_params < 300_000, \
                        f"FiLM-Lite params {total_params:,} exceed 300K budget"
            self._debug_count += 1

        # Non-regime features: [0:114] + [122:126]
        features = torch.cat([x[:, :, :-12], x[:, :, -4:]], dim=-1)  # (B, n_stack, 118)

        _, (h_n, _) = self.lstm(features)
        h = h_n[-1]                       # last layer hidden: (B, 128)
        h = self.film(h, regime_probs)     # FiLM conditioning
        h = self.ln(h)
        return self.output(h)


# ── Stage 6 corrected: separate gamma/beta FiLM + Position A/B ──────────

class FiLMLayerV2(nn.Module):
    """FiLM v2: separate gamma and beta projections with proper identity init.

    h' = gamma(regime) * h + beta(regime)

    Unlike FiLMLayer (chunked from single adapter), this uses independent
    Linear projections so gamma and beta can learn independently.
    """

    def __init__(self, regime_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(regime_dim, hidden_dim)
        self.beta_proj = nn.Linear(regime_dim, hidden_dim)
        # Identity init: gamma≈1, beta≈0
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h: torch.Tensor, regime_probs: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(regime_probs)
        beta = self.beta_proj(regime_probs)
        return gamma * h + beta


class LSTMFiLMPosAExtractor(BaseFeaturesExtractor):
    """Position A: FiLM BEFORE LSTM - modulate projected input features per regime.

    Pipeline: non_regime -> input_proj(118->128) -> FiLM(regime) -> LSTM -> LN -> output

    Rationale: regime-conditions the feature representation before temporal
    processing, so the LSTM sees regime-adapted features at every timestep.

    Obs layout (126 dims per timestep):
        [0:114]   = scaled features
        [114:122] = regime_proba_0..7 (8 dims, no-scale)
        [122:126] = env state
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 single_obs_dim: int = 126, regime_dim: int = 8,
                 features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        total_dim = int(np.prod(observation_space.shape))
        self.single_obs_dim = single_obs_dim
        self.n_stack = total_dim // single_obs_dim
        self.regime_dim = regime_dim
        self.non_regime_dim = single_obs_dim - regime_dim  # 118

        self.input_proj = nn.Linear(self.non_regime_dim, 128)
        self.film = FiLMLayerV2(regime_dim, 128)
        self.lstm = nn.LSTM(128, 128, num_layers=1, batch_first=True)
        self.ln = nn.LayerNorm(128)
        self.output = nn.Linear(128, features_dim)

        self._debug_count = 0

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.n_stack, self.single_obs_dim)

        regime_probs = x[:, -1, -12:-4]  # (B, 8) latest timestep

        if self._debug_count < 10:
            with torch.no_grad():
                if self._debug_count == 0:
                    total_params = sum(p.numel() for p in self.parameters())
                    prob_sum = regime_probs.sum(dim=-1).mean().item()
                    print(f"[FiLM-PosA] params={total_params:,}, "
                          f"regime_sum={prob_sum:.4f}, "
                          f"sample={regime_probs[0].cpu().numpy()}")
                    assert total_params < 500_000, \
                        f"FiLM-PosA params {total_params:,} exceed 500K budget"
            self._debug_count += 1

        # Non-regime features: [0:114] + [122:126] = 118 dims
        features = torch.cat([x[:, :, :-12], x[:, :, -4:]], dim=-1)

        # Project -> FiLM -> LSTM
        h = F.silu(self.input_proj(features))        # (B, n_stack, 128)
        h = self.film(h, regime_probs.unsqueeze(1).expand(-1, self.n_stack, -1))  # broadcast regime to all timesteps
        h_out, _ = self.lstm(h)
        h = h_out[:, -1, :]                          # last timestep
        h = self.ln(h)
        return self.output(h)


class LSTMFiLMPosBExtractor(BaseFeaturesExtractor):
    """Position B: FiLM AFTER LSTM - modulate LSTM output per regime.

    Pipeline: non_regime -> LSTM -> FiLM(regime) -> LN -> output

    Rationale: LSTM extracts temporal patterns first, then FiLM adapts
    the representation based on the current regime.

    Obs layout identical to PosA (126 dims per timestep).
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 single_obs_dim: int = 126, regime_dim: int = 8,
                 features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        total_dim = int(np.prod(observation_space.shape))
        self.single_obs_dim = single_obs_dim
        self.n_stack = total_dim // single_obs_dim
        self.regime_dim = regime_dim
        self.non_regime_dim = single_obs_dim - regime_dim  # 118

        self.lstm = nn.LSTM(self.non_regime_dim, 128, num_layers=1, batch_first=True)
        self.film = FiLMLayerV2(regime_dim, 128)
        self.ln = nn.LayerNorm(128)
        self.output = nn.Linear(128, features_dim)

        self._debug_count = 0

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.n_stack, self.single_obs_dim)

        regime_probs = x[:, -1, -12:-4]  # (B, 8) latest timestep

        if self._debug_count < 10:
            with torch.no_grad():
                if self._debug_count == 0:
                    total_params = sum(p.numel() for p in self.parameters())
                    prob_sum = regime_probs.sum(dim=-1).mean().item()
                    print(f"[FiLM-PosB] params={total_params:,}, "
                          f"regime_sum={prob_sum:.4f}, "
                          f"sample={regime_probs[0].cpu().numpy()}")
                    assert total_params < 500_000, \
                        f"FiLM-PosB params {total_params:,} exceed 500K budget"
            self._debug_count += 1

        # Non-regime features: [0:114] + [122:126] = 118 dims
        features = torch.cat([x[:, :, :-12], x[:, :, -4:]], dim=-1)

        # LSTM -> FiLM
        _, (h_n, _) = self.lstm(features)
        h = h_n[-1]                       # last layer hidden: (B, 128)
        h = self.film(h, regime_probs)    # FiLM conditioning
        h = self.ln(h)
        return self.output(h)

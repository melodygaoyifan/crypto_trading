"""
Stage 4: Custom feature extractors for TQC architecture comparison.

All extractors implement BaseFeaturesExtractor from SB3, outputting a
fixed-size feature vector consumed by TQC's MLP critic/actor heads.

For sequence models (LSTM/GRU/CNN), observations are frame-stacked via
VecFrameStack(n_stack=N). The extractor reshapes (obs_dim*N,) -> (N, obs_dim)
then processes the sequence.

Usage:
    from models.feature_extractors import get_extractor_class, ARCH_CONFIGS

    policy_kwargs = {
        "features_extractor_class": get_extractor_class("lstm"),
        "features_extractor_kwargs": ARCH_CONFIGS["lstm"],
        "net_arch": [256, 256],  # actor/critic heads after extractor
    }
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    import gymnasium as gym
except ImportError:
    import gym


class MLPExtractor(BaseFeaturesExtractor):
    """Standard MLP feature extractor (baseline).

    Architecture: obs -> Linear(hidden) -> ReLU -> Linear(hidden) -> ReLU -> features
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 hidden_dim: int = 256, n_layers: int = 2):
        features_dim = hidden_dim
        super().__init__(observation_space, features_dim)
        obs_dim = int(np.prod(observation_space.shape))

        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class LSTMExtractor(BaseFeaturesExtractor):
    """LSTM feature extractor for frame-stacked observations.

    Expects obs shape (batch, obs_dim * n_stack) from VecFrameStack.
    Reshapes to (batch, n_stack, obs_dim), runs LSTM, returns last hidden.
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 single_obs_dim: int = 121, lstm_hidden: int = 128,
                 n_lstm_layers: int = 1, dropout: float = 0.0):
        features_dim = lstm_hidden
        super().__init__(observation_space, features_dim)

        total_dim = int(np.prod(observation_space.shape))
        self.single_obs_dim = single_obs_dim
        self.n_stack = total_dim // single_obs_dim

        self.lstm = nn.LSTM(
            input_size=single_obs_dim,
            hidden_size=lstm_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(lstm_hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        # Reshape: (batch, obs_dim*n_stack) -> (batch, n_stack, obs_dim)
        x = obs.view(batch, self.n_stack, self.single_obs_dim)
        _, (h_n, _) = self.lstm(x)
        # h_n: (n_layers, batch, hidden) -> take last layer
        return self.ln(h_n[-1])


class GRUExtractor(BaseFeaturesExtractor):
    """GRU feature extractor for frame-stacked observations.

    Same approach as LSTM but uses GRU (fewer parameters, faster).
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 single_obs_dim: int = 121, gru_hidden: int = 128,
                 n_gru_layers: int = 1, dropout: float = 0.0):
        features_dim = gru_hidden
        super().__init__(observation_space, features_dim)

        total_dim = int(np.prod(observation_space.shape))
        self.single_obs_dim = single_obs_dim
        self.n_stack = total_dim // single_obs_dim

        self.gru = nn.GRU(
            input_size=single_obs_dim,
            hidden_size=gru_hidden,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(gru_hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        x = obs.view(batch, self.n_stack, self.single_obs_dim)
        _, h_n = self.gru(x)
        return self.ln(h_n[-1])


class CNNExtractor(BaseFeaturesExtractor):
    """1D CNN feature extractor for frame-stacked observations.

    Treats stacked observations as a 1D sequence with obs_dim channels.
    Conv1d(in=obs_dim, out=filters) across the time dimension.
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 single_obs_dim: int = 121, n_filters: int = 64,
                 kernel_sizes: tuple = (3, 3), output_dim: int = 128):
        features_dim = output_dim
        super().__init__(observation_space, features_dim)

        total_dim = int(np.prod(observation_space.shape))
        self.single_obs_dim = single_obs_dim
        self.n_stack = total_dim // single_obs_dim

        layers = []
        in_ch = single_obs_dim
        for ks in kernel_sizes:
            padding = ks // 2
            layers.extend([
                nn.Conv1d(in_ch, n_filters, kernel_size=ks, padding=padding),
                nn.ReLU(),
            ])
            in_ch = n_filters

        self.conv = nn.Sequential(*layers)
        # Global average pooling -> linear projection
        self.fc = nn.Sequential(
            nn.Linear(n_filters, output_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        # Reshape: (batch, obs_dim*n_stack) -> (batch, n_stack, obs_dim) -> (batch, obs_dim, n_stack)
        x = obs.view(batch, self.n_stack, self.single_obs_dim)
        x = x.permute(0, 2, 1)  # Conv1d expects (batch, channels, length)
        x = self.conv(x)
        # Global average pooling over time
        x = x.mean(dim=2)
        return self.fc(x)


# Architecture configs for each extractor type
# single_obs_dim is set dynamically at runtime
ARCH_CONFIGS = {
    "mlp": {"hidden_dim": 256, "n_layers": 2},
    "lstm": {"lstm_hidden": 128, "n_lstm_layers": 1},
    "gru": {"gru_hidden": 128, "n_gru_layers": 1},
    "cnn": {"n_filters": 64, "kernel_sizes": (3, 3), "output_dim": 128},
}

_EXTRACTOR_MAP = {
    "mlp": MLPExtractor,
    "lstm": LSTMExtractor,
    "gru": GRUExtractor,
    "cnn": CNNExtractor,
}

# Stage 6: FiLM adapter (lazy import to avoid circular deps)
def _get_lstm_film():
    from models.film_extractor import LSTMFiLMExtractor
    return LSTMFiLMExtractor

def _get_lstm_film_lite():
    from models.film_extractor import LSTMFiLMLiteExtractor
    return LSTMFiLMLiteExtractor

def _get_lstm_film_a():
    from models.film_extractor import LSTMFiLMPosAExtractor
    return LSTMFiLMPosAExtractor

def _get_lstm_film_b():
    from models.film_extractor import LSTMFiLMPosBExtractor
    return LSTMFiLMPosBExtractor

_EXTRACTOR_MAP["lstm_film"] = _get_lstm_film
_EXTRACTOR_MAP["lstm_film_lite"] = _get_lstm_film_lite
_EXTRACTOR_MAP["lstm_film_a"] = _get_lstm_film_a
_EXTRACTOR_MAP["lstm_film_b"] = _get_lstm_film_b

ARCH_CONFIGS["lstm_film"] = {"regime_dim": 8}
ARCH_CONFIGS["lstm_film_lite"] = {"regime_dim": 8}
ARCH_CONFIGS["lstm_film_a"] = {"regime_dim": 8}
ARCH_CONFIGS["lstm_film_b"] = {"regime_dim": 8}

# Sequence models require frame stacking
SEQUENCE_ARCHS = frozenset(["lstm", "gru", "cnn", "lstm_film", "lstm_film_lite",
                            "lstm_film_a", "lstm_film_b"])
DEFAULT_N_STACK = 8  # 8 × 4H = 32 hours of context


def get_extractor_class(arch: str) -> type:
    """Get the feature extractor class for the given architecture name."""
    if arch not in _EXTRACTOR_MAP:
        raise ValueError(f"Unknown architecture: {arch}. Choose from {list(_EXTRACTOR_MAP)}")
    cls = _EXTRACTOR_MAP[arch]
    return cls() if callable(cls) and not isinstance(cls, type) else cls


def get_extractor_kwargs(arch: str, single_obs_dim: int) -> dict:
    """Get kwargs for the feature extractor, with single_obs_dim injected."""
    kwargs = dict(ARCH_CONFIGS[arch])
    if arch in SEQUENCE_ARCHS:
        kwargs["single_obs_dim"] = single_obs_dim
    return kwargs

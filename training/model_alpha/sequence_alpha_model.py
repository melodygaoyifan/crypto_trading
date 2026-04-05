from dataclasses import dataclass, asdict
from typing import Dict, Any

import torch
import torch.nn as nn


@dataclass
class SequenceAlphaConfig:
    input_dim: int = 122
    seq_len: int = 48
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.10
    arch: str = "gru"
    n_classes: int = 3
    target_scale_bps: float = 64.0
    horizon_bars: int = 1
    fee_bps: float = 7.0
    neutral_bps: float = 6.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SequenceAlphaNet(nn.Module):
    def __init__(self, config: SequenceAlphaConfig):
        super().__init__()
        self.config = config

        if config.arch.lower() == "lstm":
            self.encoder = nn.LSTM(
                input_size=config.input_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )
        elif config.arch.lower() == "gru":
            self.encoder = nn.GRU(
                input_size=config.input_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unsupported sequence alpha arch: {config.arch}")

        self.norm = nn.LayerNorm(config.hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.n_classes),
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if isinstance(self.encoder, nn.LSTM):
            _, (h_n, _) = self.encoder(x)
        else:
            _, h_n = self.encoder(x)

        hidden = self.norm(h_n[-1])
        logits = self.classifier(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return {"logits": logits, "value": value}

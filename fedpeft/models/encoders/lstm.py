"""
LSTM baseline - the model of the published WISDM-W results.

FedAIoT (Alam et al., arXiv 2310.00109, App. D.1) classifies 10-s windows of WISDM
(200 samples x 6 channels, every channel z-scored) with a one-layer LSTM whose hidden
size equals its input size (6); the 200 hidden states are flattened (1200 features)
and classified by Dropout -> Linear(1200 -> 128) -> ReLU -> Dropout -> Linear(128 -> 12).

    input   all data channels at the dataset rate, raw units
    network BatchNorm1d over the channels (the z-scoring of FedAIoT's loader, learned
            from the data instead of precomputed) -> LSTM(C -> hidden) -> flatten -> Dropout
    then    the usual shared head g and classifier h of the client model (1200 -> 64 -> classes)

No pretrained weights exist: it is always trained from scratch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseEncoder, InputSpec


class LstmEncoder(BaseEncoder):
    name = "lstm"
    display_name = "LSTM"
    architecture = "LSTM (hidden = input size) over the window, flattened, as in FedAIoT"
    pretrained_source = "None (trained from scratch only)"

    def __init__(self, channels: tuple[str, ...], rate_hz: float, n_samples: int, acc_unit: str = "m/s2",
                 hidden: int = 6, dropout: float = 0.2):
        super().__init__()
        self.spec = InputSpec(rate_hz=rate_hz, n_samples=n_samples, n_channels=len(channels), acc_unit=acc_unit)
        self.input_channels = tuple(channels)
        self.norm = nn.BatchNorm1d(len(channels))
        self.lstm = nn.LSTM(len(channels), hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.feat_dim = hidden * n_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # (B, C, L)
        h, _ = self.lstm(self.norm(x).transpose(1, 2))       # (B, L, hidden)
        return self.drop(h.flatten(1))

    def lora_targets(self) -> list[str]:
        raise RuntimeError("The LSTM baseline is trained from scratch only (no LoRA methods)")

    def load_pretrained(self, path: str) -> int:
        raise RuntimeError("The LSTM baseline has no pretrained weights (use it with from-scratch methods only)")

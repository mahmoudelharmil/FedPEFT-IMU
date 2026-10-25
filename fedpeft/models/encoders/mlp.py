"""
MLP baseline - the model of the published HARBox results.

ClusterFL (Ouyang et al., MobiSys 2021) and FedCHAR (Li et al., IMWUT 2023) classify
the flattened 2-s window (9 channels x 100 samples = 900 features) with one fully
connected layer of 300 units.  We include it so that our reproduction of their
protocol can be checked against their reported FedAvg numbers.

    input   all data channels at the dataset rate, raw units (as in their code)
    network flatten -> Linear(C*L -> 300) -> ReLU      (z in R^300)
    then    the usual shared head g and classifier h of the client model

No pretrained weights exist: it is always trained from scratch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseEncoder, InputSpec


class MlpEncoder(BaseEncoder):
    name = "mlp"
    display_name = "MLP"
    architecture = "Flatten + one fully connected layer (300 units), as in ClusterFL / FedCHAR"
    pretrained_source = "None (trained from scratch only)"

    def __init__(self, channels: tuple[str, ...], rate_hz: float, n_samples: int, acc_unit: str = "m/s2",
                 hidden: int = 300):
        super().__init__()
        self.spec = InputSpec(rate_hz=rate_hz, n_samples=n_samples, n_channels=len(channels), acc_unit=acc_unit)
        self.input_channels = tuple(channels)
        self.fc = nn.Linear(len(channels) * n_samples, hidden)
        self.act = nn.ReLU()
        self.feat_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # (B, C, L)
        return self.act(self.fc(x.flatten(1)))

    def lora_targets(self) -> list[str]:
        return ["fc"]

    def load_pretrained(self, path: str) -> int:
        raise RuntimeError("The MLP baseline has no pretrained weights (use it with from-scratch methods only)")

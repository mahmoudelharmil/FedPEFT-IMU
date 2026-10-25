"""
Conv-GRU baseline - a model in the style of the published UCI HAR results.

FedMultimodal (Feng et al., KDD 2023) treats the accelerometer and the gyroscope of UCI HAR
as two modalities: each is encoded by 1-D convolutions followed by a GRU (hidden size 128),
the two representations are fused (average pooling over time + concatenation) and two dense
layers classify.  We follow that structure; layer counts and widths of the convolutions are
not fully specified in the paper, so this is an approximation of their model, used for a
re-run of the published baseline next to the numbers copied from the paper.

    input   all data channels at the dataset rate, raw units
    network BatchNorm1d over the channels (standardisation), then per sensor
            (acc, gyro, mag): 3 x [Conv1d(k=5) -> ReLU -> Dropout] -> GRU(128) -> mean over time;
            the sensors are concatenated (z in R^{128 x sensors})
    then    the usual shared head g and classifier h of the client model

No pretrained weights exist: it is always trained from scratch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseEncoder, InputSpec


class _SensorBranch(nn.Module):
    def __init__(self, c_in: int, filters: int, hidden: int, dropout: float):
        super().__init__()
        layers, c = [], c_in
        for _ in range(3):
            layers += [nn.Conv1d(c, filters, kernel_size=5, padding=2), nn.ReLU(), nn.Dropout(dropout)]
            c = filters
        self.conv = nn.Sequential(*layers)
        self.gru = nn.GRU(filters, hidden, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, c_in, L)
        h, _ = self.gru(self.conv(x).transpose(1, 2))          # (B, L, hidden)
        return h.mean(dim=1)


class ConvGruEncoder(BaseEncoder):
    name = "convgru"
    display_name = "Conv-GRU"
    architecture = "per-sensor 1-D conv + GRU(128), mean-pooled and concatenated (FedMultimodal style)"
    pretrained_source = "None (trained from scratch only)"

    def __init__(self, channels: tuple[str, ...], rate_hz: float, n_samples: int, acc_unit: str = "g",
                 filters: int = 64, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.spec = InputSpec(rate_hz=rate_hz, n_samples=n_samples, n_channels=len(channels), acc_unit=acc_unit)
        self.input_channels = tuple(channels)
        sensors: dict[str, list[int]] = {}
        for i, ch in enumerate(channels):                      # "acc_x" -> sensor "acc"
            sensors.setdefault(ch.split("_")[0], []).append(i)
        self.sensor_index = list(sensors.values())
        self.norm = nn.BatchNorm1d(len(channels))
        self.branches = nn.ModuleList(_SensorBranch(len(idx), filters, hidden, dropout) for idx in self.sensor_index)
        self.feat_dim = hidden * len(self.sensor_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, C, L)
        x = self.norm(x)
        return torch.cat([branch(x[:, idx]) for branch, idx in zip(self.branches, self.sensor_index)], dim=1)

    def lora_targets(self) -> list[str]:
        raise RuntimeError("The Conv-GRU baseline is trained from scratch only (no LoRA methods)")

    def load_pretrained(self, path: str) -> int:
        raise RuntimeError("The Conv-GRU baseline has no pretrained weights (use it with from-scratch methods only)")

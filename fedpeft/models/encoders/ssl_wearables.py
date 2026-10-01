"""
ssl-wearables "harnet10" (Yuan et al., npj Digital Medicine 2024) - Oxford ResNet-V2.

Official code / weights: https://github.com/OxWearables/ssl-wearables  (weights:
model_check_point/mtl_best.mdl).  Pretrained with multi-task self-supervision
(arrow of time, permutation, time-warp) on 700,000 person-days of UK-Biobank
wrist accelerometry - the same Axivity AX3 device family as Capture-24.

Architecture (identical module names to the official `Resnet` class):
    input     3 channels (acc in g), 30 Hz, 300 samples (10 s)
    5 stages  x -> Conv -> [ResBlock]^m -> BN -> ReLU -> anti-aliased Downsample
              channels 64, 128, 256, 512, 1024 ; downsample 2, 2, 5, 5, 3  (300 -> 1)
    output    z in R^1024

This is the "large" encoder (~10 M parameters) that shows the communication story.
LoRA targets: every convolution kernel (reshaped to C_out x C_in*k, PDF Figure L6).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseEncoder, InputSpec


class Downsample(nn.Module):
    """Anti-aliased downsampling with a fixed (non-trainable) triangle/box filter."""

    def __init__(self, channels: int, factor: int = 2, order: int = 1):
        super().__init__()
        self.stride = factor
        self.padding = int(order * (factor - 1) / 2)
        box = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box)
        kernel = torch.tensor(kernel / kernel.sum(), dtype=torch.float32)
        self.register_buffer("kernel", kernel[None, None, :].repeat(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.kernel.to(x.dtype), stride=self.stride, padding=self.padding, groups=x.shape[1])


class ResBlock(nn.Module):
    """bn-relu-conv-bn-relu-conv with an identity shortcut."""

    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 1, pad, bias=False, padding_mode="circular")
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, 1, pad, bias=False, padding_mode="circular")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(self.relu(self.bn1(x)))
        x = self.conv2(self.relu(self.bn2(x)))
        return x + identity


def make_layer(c_in: int, c_out: int, conv_k: int, n_res: int, res_k: int, down: int, order: int) -> nn.Sequential:
    """One ResNet stage: Conv -> [ResBlock]^m -> BN -> ReLU -> Downsample."""
    modules = [nn.Conv1d(c_in, c_out, conv_k, 1, (conv_k - 1) // 2, bias=False, padding_mode="circular")]
    modules += [ResBlock(c_out, res_k) for _ in range(n_res)]
    modules += [nn.BatchNorm1d(c_out), nn.ReLU(True), Downsample(c_out, down, order)]
    return nn.Sequential(*modules)


# (out_channels, conv kernel, #resblocks, resblock kernel, downsample factor, filter order)
HARNET10_CFG = [
    (64, 5, 2, 5, 2, 2),
    (128, 5, 2, 5, 2, 2),
    (256, 5, 2, 5, 5, 1),
    (512, 5, 2, 5, 5, 1),
    (1024, 5, 0, 5, 3, 1),
]


class SslWearablesEncoder(BaseEncoder):
    name = "ssl_wearables"
    display_name = "ssl-wearables"
    architecture = "ResNet-V2 1-D (harnet10), 5 stages, 64-1024 channels"
    pretrained_source = "Official Oxford harnet10 weights, self-supervised on UK-Biobank wrist data"
    spec = InputSpec(rate_hz=30, n_samples=300, n_channels=3, acc_unit="g")

    def __init__(self):
        super().__init__()
        layers, c_in = [], 3
        for i, (c_out, k, m, rk, down, order) in enumerate(HARNET10_CFG):
            layers.append((f"layer{i + 1}", make_layer(c_in, c_out, k, m, rk, down, order)))
            c_in = c_out
        self.feature_extractor = nn.Sequential()
        for name, layer in layers:
            self.feature_extractor.add_module(name, layer)
        self.feat_dim = c_in
        self._kaiming_init()

    def _kaiming_init(self) -> None:
        """Same initialisation as the official code (used for training from scratch)."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # x: (B, 3, 300)
        return self.feature_extractor(x).flatten(1)          # (B, 1024)

    def lora_targets(self) -> list[str]:
        return [name for name, m in self.named_modules() if isinstance(m, nn.Conv1d)]

    def load_pretrained(self, path: str) -> int:
        # Saved from nn.DataParallel -> keys start with "module."; the MTL heads are ignored.
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}
        return self._load_matching(state)

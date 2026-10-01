"""
Component 4 - LoRA (Low-Rank Adaptation, Hu et al., ICLR 2022).

For a frozen pretrained weight W0 (d_out x d_in) LoRA learns two thin matrices

        B : d_out x r      (initialised to zeros)
        A : r x d_in       (random initialisation)

and the layer computes   h = W0 x + (alpha / r) * B (A x).

Because B = 0 at the start, training begins exactly from the pretrained model
(PDF p. 5-6).  Only A and B receive gradients, so the optimiser stores state only
for them - that is why LoRA also cuts client memory.

Convolutions (PDF p. 7, Figure L6): a Conv kernel of shape (C_out, C_in, k...) is
viewed as a 2-D matrix (C_out) x (C_in * k...).  The LoRA product B A has that
shape, and we apply it efficiently as two convolutions:
        A  -> a conv with r output channels and the original kernel size
        B  -> a 1x1 conv from r channels back to C_out.

All factors are stored in the same 2-D layout for every layer type, so the
server-side aggregation code (fedpeft/federated/aggregation.py) never needs to
know whether a layer was Linear, Conv1d or Conv2d.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """Wraps a frozen nn.Linear / nn.Conv1d / nn.Conv2d and adds a trainable low-rank update."""

    def __init__(self, base: nn.Module, rank: int, scaling: float, dropout: float = 0.0):
        super().__init__()
        if isinstance(base, nn.Linear):
            self.kind = "linear"
        elif isinstance(base, nn.Conv1d):
            self.kind = "conv1d"
        elif isinstance(base, nn.Conv2d):
            self.kind = "conv2d"
        else:
            raise TypeError(f"LoRA does not support {type(base).__name__}")
        if self.kind != "linear" and base.groups != 1:
            raise ValueError("LoRA on grouped convolutions is not supported")

        self.base = base                                   # frozen W0 (+ server residual, see aggregation)
        weight = base.weight
        self.fan_out = weight.shape[0]                     # d_out  (C_out)
        self.fan_in = weight[0].numel()                    # d_in   (C_in * kernel)
        self.kernel = tuple(weight.shape[2:])              # () for Linear
        self.scaling = scaling                             # alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Trainable factors, stored as 2-D matrices for every layer type.
        self.lora_A = nn.Parameter(torch.empty(rank, self.fan_in, device=weight.device))
        self.lora_B = nn.Parameter(torch.zeros(self.fan_out, rank, device=weight.device))
        self.reset_lora()

    # ------------------------------------------------------------------ utils
    @property
    def rank(self) -> int:
        return self.lora_A.shape[0]

    def reset_lora(self) -> None:
        """A ~ Kaiming-uniform (same as the LoRA reference code), B = 0  ->  B A = 0."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def set_factors(self, A: torch.Tensor, B: torch.Tensor) -> None:
        """Replace A and B (the rank may change: heterogeneous-rank clients, V4/V5)."""
        dev = self.base.weight.device
        self.lora_A = nn.Parameter(A.detach().to(dev, torch.float32).clone())
        self.lora_B = nn.Parameter(B.detach().to(dev, torch.float32).clone())

    def delta_weight(self) -> torch.Tensor:
        """The effective update (alpha/r) * B A as a (fan_out x fan_in) matrix."""
        return self.scaling * (self.lora_B @ self.lora_A)

    # ---------------------------------------------------------------- forward
    def _conv(self, x: torch.Tensor, weight: torch.Tensor, first: bool) -> torch.Tensor:
        """Run a convolution with the geometry of the base layer (first=True) or a 1x1 conv."""
        conv = F.conv1d if self.kind == "conv1d" else F.conv2d
        if not first:
            return conv(x, weight)
        b = self.base
        if b.padding_mode != "zeros":                      # e.g. circular padding in ssl-wearables
            x = F.pad(x, b._reversed_padding_repeated_twice, mode=b.padding_mode)
            return conv(x, weight, stride=b.stride, dilation=b.dilation)
        return conv(x, weight, stride=b.stride, padding=b.padding, dilation=b.dilation)

    def _merged_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: use W = W0 + (alpha/r) B A directly (PDF Fig. L4 "merge") - no extra cost."""
        weight = self.base.weight + self.delta_weight().view_as(self.base.weight).to(self.base.weight.dtype)
        if self.kind == "linear":
            return F.linear(x, weight, self.base.bias)
        return self.base._conv_forward(x, weight, self.base.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            return self.base(x)
        if not self.training or isinstance(self.dropout, nn.Identity):
            # Without LoRA dropout both paths are mathematically identical; the merged
            # kernel is faster (one convolution instead of three) and autograd still
            # sends gradients only to A and B.
            return self._merged_forward(x)
        out = self.base(x)                                 # W0 x   (frozen path)
        xd = self.dropout(x)
        r = self.rank
        if self.kind == "linear":
            low = F.linear(F.linear(xd, self.lora_A), self.lora_B)            # B (A x)
        else:
            c_in = self.base.weight.shape[1]
            ones = (1,) * len(self.kernel)
            a = self._conv(xd, self.lora_A.view(r, c_in, *self.kernel), first=True)   # down to r channels
            low = self._conv(a, self.lora_B.view(self.fan_out, r, *ones), first=False)  # back up to C_out
        return out + self.scaling * low


def inject_lora(model: nn.Module, target_names: list[str], rank: int, scaling: float,
                dropout: float) -> dict[str, LoRALayer]:
    """Replace each named sub-module by a LoRALayer wrapping it. Returns {name: layer}."""
    layers = {}
    for name in target_names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, child)
        layer = LoRALayer(base, rank, scaling, dropout)
        setattr(parent, child, layer)
        layers[name] = layer
    return layers

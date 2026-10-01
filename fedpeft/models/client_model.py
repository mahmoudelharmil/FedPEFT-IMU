"""
The complete client model (PDF Part 2, Figure 2) and the ROLE of every tensor.

    x (3 x 300 @ 30 Hz)
      -> 1 Preprocessor -> 2 InputAdapter -> 3 Encoder (+4 LoRA, 5 norms inside)
      -> z -> 6 SharedHead g -> e -> 7 PersonalHead h -> logits
                               e -> 8 prototypes

Roles (one per state-dict entry) - the federated methods decide what each role does:

    encoder      frozen pretrained weights W0 (the LoRA base weights are here too)
    lora         LoRA factors A and B
    norm         LayerNorm / BatchNorm scale + shift (gamma, beta)
    norm_stats   BatchNorm running mean / variance
    adapter      input adapter
    shared_head  g
    head         h
    const        fixed buffers (graph adjacency, anti-aliasing kernels) - never change
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .components import InputAdapter, PersonalHead, Preprocessor, SharedHead
from .encoders.base import BaseEncoder
from .encoders.limu_bert import LimuLayerNorm
from .lora import LoRALayer, inject_lora

NORM_TYPES = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, LimuLayerNorm)
ROLES = ("encoder", "lora", "norm", "norm_stats", "adapter", "shared_head", "head", "const")


class ClientModel(nn.Module):
    def __init__(self, encoder: BaseEncoder, n_classes: int, embed_dim: int = 64,
                 base_rate_hz: int = 30, window_s: int = 10):
        super().__init__()
        self.preprocess = Preprocessor(encoder.spec, base_rate_hz, window_s)   # 1
        self.adapter = InputAdapter(3, encoder.spec.n_channels)                # 2
        self.encoder = encoder                                                 # 3 (+4, 5 inside)
        self.shared_head = SharedHead(encoder.feat_dim, embed_dim)             # 6
        self.head = PersonalHead(embed_dim, n_classes)                         # 7
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.lora_layers: dict[str, LoRALayer] = {}   # plain dict: layers are registered inside the encoder

    # ---------------------------------------------------------------- LoRA
    def add_lora(self, rank: int, scaling: float, dropout: float) -> None:
        """Component 4: wrap the encoder's target layers with LoRA branches."""
        self.lora_layers = inject_lora(self.encoder, self.encoder.lora_targets(), rank, scaling, dropout)

    def lora_key(self, layer_name: str) -> str:
        """State-dict key of the (frozen) base weight of a LoRA layer."""
        return f"encoder.{layer_name}.base.weight"

    # ------------------------------------------------------------- forward
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Windows -> 64-d activity embedding e = g(E(adapter(pre(x))))."""
        return self.shared_head(self.encoder(self.adapter(self.preprocess(x))))

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        e = self.embed(x)
        logits = self.head(e)
        return (logits, e) if return_embedding else logits

    # --------------------------------------------------------------- roles
    def roles(self) -> dict[str, str]:
        """Map every state-dict key to its role (see module docstring)."""
        norm_params, norm_buffers = set(), set()
        for mname, module in self.encoder.named_modules():
            if isinstance(module, NORM_TYPES):
                for pname, _ in module.named_parameters(recurse=False):
                    norm_params.add(f"encoder.{mname}.{pname}")
                for bname, _ in module.named_buffers(recurse=False):
                    norm_buffers.add(f"encoder.{mname}.{bname}")
        param_keys = {k for k, _ in self.named_parameters()}

        roles = {}
        for key in self.state_dict().keys():
            if key.startswith("adapter."):
                roles[key] = "adapter"
            elif key.startswith("shared_head."):
                roles[key] = "shared_head"
            elif key.startswith("head."):
                roles[key] = "head"
            elif key.endswith("lora_A") or key.endswith("lora_B"):
                roles[key] = "lora"
            elif key in norm_params:
                roles[key] = "norm"
            elif key in norm_buffers:
                roles[key] = "norm_stats"
            elif key in param_keys:
                roles[key] = "encoder"
            else:
                roles[key] = "const"     # e.g. the ST-GCN adjacency, Downsample kernels
        return roles

    def keys_with_roles(self, wanted: set[str]) -> list[str]:
        return [k for k, r in self.roles().items() if r in wanted]

    def set_trainable(self, trainable_roles: set[str], roles: dict[str, str] | None = None) -> list[nn.Parameter]:
        """Freeze everything except parameters whose role is in `trainable_roles`.

        `roles` may be passed in (cached) because computing it walks the whole model.
        """
        roles = roles or self.roles()
        params = []
        for name, p in self.named_parameters():
            p.requires_grad = roles[name] in trainable_roles
            if p.requires_grad:
                params.append(p)
        self._norm_trainable = "norm" in trainable_roles
        return params

    def train(self, mode: bool = True) -> "ClientModel":
        """Like nn.Module.train, but frozen BatchNorms keep using their stored statistics.

        When the norm role is frozen (V1-V4), BatchNorm layers must stay in eval mode,
        otherwise their running statistics would silently drift during training.
        """
        super().train(mode)
        if mode and not getattr(self, "_norm_trainable", True):
            for m in self.encoder.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
        return self

    # -------------------------------------------------------------- counts
    def count_by_role(self) -> dict[str, int]:
        """Number of values (parameters + buffers) per role - used for the parameter budget figure."""
        counts = {r: 0 for r in ROLES}
        state = self.state_dict()
        for key, role in self.roles().items():
            if state[key].dtype.is_floating_point:
                counts[role] += state[key].numel()
        return counts

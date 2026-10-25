"""
UniMTS (Zhang et al., NeurIPS 2024) - unified pre-training for motion time series.

Official code / weights: https://github.com/xiyuanzh/UniMTS and
https://huggingface.co/xiyuanz/UniMTS  (Apache-2.0).  The ST-GCN below is a
commented copy of the official `ST_GCN_18` so the checkpoint loads unchanged.

Idea of UniMTS: IMU signals are synthesised for all 22 joints of a human skeleton
(from motion-capture data), and a spatio-temporal graph network is aligned with
LLM-enriched text descriptions (CLIP-style).  A real device is placed on the joint
where it was worn; every other joint receives zeros.

Our use:
    input    3 channels (acc in m/s^2), 20 Hz, 200 samples (10 s), placed on joint 21
             (right wrist in the HumanML3D skeleton) -> tensor (B, 3, 200, 22, 1)
    network  10 ST-GCN blocks (graph conv over joints + temporal conv, kernel 9)
             channels 64 -> 128 -> 256 -> 512, temporal stride 2 twice
    output   global average pool -> z in R^512

LoRA targets: the graph convolution (1x1 conv) and the temporal convolution of
every block (Conv2d kernels reshaped to C_out x C_in*k).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseEncoder, InputSpec


# ----------------------------------------------------------------------------
# Skeleton graph (22 HumanML3D joints, "spatial" partition strategy)
# ----------------------------------------------------------------------------
def _hop_distance(num_node: int, edges: list[tuple[int, int]], max_hop: int = 1) -> np.ndarray:
    A = np.zeros((num_node, num_node))
    for i, j in edges:
        A[j, i] = A[i, j] = 1
    hop = np.full((num_node, num_node), np.inf)
    reach = np.stack([np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]) > 0
    for d in range(max_hop, -1, -1):
        hop[reach[d]] = d
    return hop


def _normalize_digraph(A: np.ndarray) -> np.ndarray:
    Dl = A.sum(0)
    Dn = np.diag([1.0 / d if d > 0 else 0.0 for d in Dl])
    return A @ Dn


def skeleton_adjacency() -> np.ndarray:
    """Return the (3, 22, 22) adjacency of the spatial strategy: root / closer / further."""
    num_node, center = 22, 0
    neighbours = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7), (9, 0), (10, 9), (11, 10),
                  (12, 11), (13, 12), (14, 11), (15, 14), (16, 15), (17, 16), (18, 11), (19, 18), (20, 19), (21, 20)]
    edges = [(i, i) for i in range(num_node)] + neighbours
    hop = _hop_distance(num_node, edges)
    adjacency = np.zeros((num_node, num_node))
    adjacency[hop <= 1] = 1
    norm = _normalize_digraph(adjacency)
    A = []
    for h in (0, 1):
        root, close, further = (np.zeros((num_node, num_node)) for _ in range(3))
        for i in range(num_node):
            for j in range(num_node):
                if hop[j, i] == h:
                    if hop[j, center] == hop[i, center]:
                        root[j, i] = norm[j, i]
                    elif hop[j, center] > hop[i, center]:
                        close[j, i] = norm[j, i]
                    else:
                        further[j, i] = norm[j, i]
        if h == 0:
            A.append(root)
        else:
            A.extend([root + close, further])
    return np.stack(A)


# ----------------------------------------------------------------------------
# ST-GCN building blocks (official implementation, lightly commented)
# ----------------------------------------------------------------------------
class ConvTemporalGraphical(nn.Module):
    """Graph convolution: a 1x1 conv producing K groups, each mixed by one adjacency."""

    def __init__(self, c_in: int, c_out: int, k: int):
        super().__init__()
        self.kernel_size = k
        self.conv = nn.Conv2d(c_in, c_out * k, kernel_size=(1, 1))   # <- LoRA

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        return torch.einsum("nkctv,kvw->nctw", x, A.to(x.dtype)).contiguous()


class StGcnBlock(nn.Module):
    """Graph conv -> BN -> ReLU -> temporal conv (kernel 9) -> BN (+ residual) -> ReLU."""

    def __init__(self, c_in: int, c_out: int, kernel_size: tuple[int, int], stride: int = 1,
                 dropout: float = 0.0, residual: bool = True):
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = ConvTemporalGraphical(c_in, c_out, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, (kernel_size[0], 1), (stride, 1), padding),   # <- LoRA
            nn.BatchNorm2d(c_out),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = None
            self.res_mode = "zero"
        elif c_in == c_out and stride == 1:
            self.residual = None
            self.res_mode = "identity"
        else:
            self.residual = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1, stride=(stride, 1)),
                                          nn.BatchNorm2d(c_out))
            self.res_mode = "conv"
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        if self.res_mode == "zero":
            res = 0
        elif self.res_mode == "identity":
            res = x
        else:
            res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


class UniMTSEncoder(BaseEncoder):
    name = "unimts"
    display_name = "UniMTS"
    architecture = "ST-GCN, 10 graph blocks over a 22-joint skeleton, 64-512 channels"
    pretrained_source = "Official UniMTS weights (motion-capture-synthesised IMU aligned with text)"
    spec = InputSpec(rate_hz=20, n_samples=200, n_channels=3, acc_unit="m/s2")

    def __init__(self, wrist_joint: int = 21, n_samples: int = 200):
        super().__init__()
        # The graph network is length-independent (temporal convolutions + global pooling);
        # n_samples only tells the preprocessor how long a window to deliver (HARBox: 40).
        self.spec = InputSpec(rate_hz=20, n_samples=n_samples, n_channels=3, acc_unit="m/s2")
        self.wrist_joint = wrist_joint
        self.register_buffer("A", torch.tensor(skeleton_adjacency(), dtype=torch.float32))
        k = (9, self.A.size(0))                               # temporal kernel 9, spatial kernel 3
        self.data_bn = nn.BatchNorm1d(3 * self.A.size(1))    # per (joint, channel) input normalisation
        self.st_gcn_networks = nn.ModuleList([
            StGcnBlock(3, 64, k, 1, residual=False),
            StGcnBlock(64, 64, k, 1), StGcnBlock(64, 64, k, 1), StGcnBlock(64, 64, k, 1),
            StGcnBlock(64, 128, k, 2), StGcnBlock(128, 128, k, 1), StGcnBlock(128, 128, k, 1),
            StGcnBlock(128, 256, k, 2), StGcnBlock(256, 256, k, 1), StGcnBlock(256, 512, k, 1),
        ])
        # Learnable importance weight for every edge of every block.
        self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks])
        self.feat_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (B, 3, 200) in m/s^2
        B, C, T = x.shape
        V = self.A.size(1)
        # Place the wrist signal on its joint; all other joints are zero (as in UniMTS).
        full = x.new_zeros(B, C, T, V)
        full[..., self.wrist_joint] = x
        # data_bn normalises every (joint, channel) pair: (B, V*C, T)
        h = full.permute(0, 3, 1, 2).reshape(B, V * C, T)
        h = self.data_bn(h)
        h = h.view(B, V, C, T).permute(0, 2, 3, 1).contiguous()   # (B, C, T, V)
        for block, importance in zip(self.st_gcn_networks, self.edge_importance):
            h = block(h, self.A * importance)
        return F.avg_pool2d(h, h.size()[2:]).flatten(1)          # (B, 512)

    def lora_targets(self) -> list[str]:
        names = []
        for i in range(len(self.st_gcn_networks)):
            names += [f"st_gcn_networks.{i}.gcn.conv", f"st_gcn_networks.{i}.tcn.2"]
        return names

    def load_pretrained(self, path: str) -> int:
        # Our download script already extracted the "acc." (ST-GCN) part of the
        # official 274 MB checkpoint; accept both forms.
        state = torch.load(path, map_location="cpu", weights_only=True)
        if any(k.startswith("acc.") for k in state):
            state = {k[4:]: v for k, v in state.items() if k.startswith("acc.")}
        return self._load_matching(state)

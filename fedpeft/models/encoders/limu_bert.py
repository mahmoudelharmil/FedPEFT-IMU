"""
LIMU-BERT (Xu et al., SenSys 2021) - a small BERT-style Transformer for phone IMU data.

Official code / weights: https://github.com/dapowan/LIMU-BERT-Public  (MIT licence)
We re-implement the encoder with IDENTICAL parameter names so the official
checkpoint `saved/pretrain_base_hhar_20_120/hhar.pt` loads directly.

Architecture ("base_v1" config):
    input      6 channels (acc in g + gyro), 20 Hz, 120 samples (6 s)
    embedding  Linear(6 -> 72) + LayerNorm + learned position embedding (120 x 72)
    encoder    4 Transformer layers that SHARE one set of weights (ALBERT-style):
               multi-head self-attention (4 heads) + FFN (72 -> 144 -> 72)
    output     (B, 120, 72) -> mean over time -> z in R^72

LoRA targets (PDF default): the query and value projections.  Because the four
layers share weights, the LoRA factors are shared as well.  lora_scope="all" wraps every
linear layer of the block instead (q, k, v, output projection and both FFN layers).

Capture-24 only has an accelerometer, so the input adapter (component 2) maps the
3 acc channels into the 6-channel input (acc -> acc identity, gyro rows start at 0).

Window length: the official model sees 120 samples (6 s).  With `seq_len=200` it
sees the whole 10-s window; the 120 pretrained position embeddings are linearly
interpolated to 200 positions (the usual way to extend a Transformer's input).
With a shorter window (HARBox: 2 s = 40 samples) the first 40 position embeddings
are used unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ACC, GYRO, BaseEncoder, InputSpec


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU exactly as in the LIMU-BERT code (erf form)."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LimuLayerNorm(nn.Module):
    """TF-style LayerNorm with parameters named gamma/beta (as in the checkpoint)."""

    def __init__(self, hidden: int, eps: float = 1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden))
        self.beta = nn.Parameter(torch.zeros(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        return self.gamma * (x - u) / torch.sqrt(s + self.eps) + self.beta


class Embeddings(nn.Module):
    def __init__(self, feature_num: int, hidden: int, seq_len: int):
        super().__init__()
        self.lin = nn.Linear(feature_num, hidden)            # factorised input embedding
        self.pos_embed = nn.Embedding(seq_len, hidden)       # learned position embedding
        self.norm = LimuLayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # x: (B, S, F)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        e = self.norm(self.lin(x))                           # emb_norm = True in the official config
        return self.norm(e + self.pos_embed(pos))


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.proj_q = nn.Linear(hidden, hidden)   # <- LoRA
        self.proj_k = nn.Linear(hidden, hidden)
        self.proj_v = nn.Linear(hidden, hidden)   # <- LoRA
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # (B, S, D)
        B, S, D = x.shape
        q, k, v = (p(x).view(B, S, self.n_heads, -1).transpose(1, 2)
                   for p in (self.proj_q, self.proj_k, self.proj_v))   # (B, H, S, W)
        h = F.scaled_dot_product_attention(q, k, v)                     # softmax(q k^T / sqrt(W)) v
        return h.transpose(1, 2).reshape(B, S, D)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden: int, hidden_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden_ff)
        self.fc2 = nn.Linear(hidden_ff, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(gelu(self.fc1(x)))


class Transformer(nn.Module):
    """LIMU-BERT Transformer: one block applied n_layers times (shared weights)."""

    def __init__(self, feature_num=6, hidden=72, hidden_ff=144, n_layers=4, n_heads=4, seq_len=120):
        super().__init__()
        self.embed = Embeddings(feature_num, hidden, seq_len)
        self.n_layers = n_layers
        self.attn = MultiHeadedSelfAttention(hidden, n_heads)
        self.proj = nn.Linear(hidden, hidden)
        self.norm1 = LimuLayerNorm(hidden)
        self.pwff = PositionWiseFeedForward(hidden, hidden_ff)
        self.norm2 = LimuLayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for _ in range(self.n_layers):
            # Same residual arrangement as the official implementation.
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class LimuBertEncoder(BaseEncoder):
    name = "limu_bert"
    display_name = "LIMU-BERT"
    architecture = "Transformer, 4 shared layers, d=72, 4 heads"
    pretrained_source = "Official LIMU-BERT weights pretrained on HHAR (phones + smartwatches)"
    spec = InputSpec(rate_hz=20, n_samples=120, n_channels=6, acc_unit="g")   # pretraining format
    input_channels = ACC + GYRO

    def __init__(self, seq_len: int = 120, lora_scope: str = "qv"):
        super().__init__()
        self.lora_scope = lora_scope
        self.spec = InputSpec(rate_hz=20, n_samples=seq_len, n_channels=6, acc_unit="g")
        self.transformer = Transformer(seq_len=seq_len)
        self.feat_dim = 72

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # x: (B, 6, 120)
        h = self.transformer(x.transpose(1, 2))             # (B, 120, 72)
        return h.mean(dim=1)                                 # temporal average -> (B, 72)

    def lora_targets(self) -> list[str]:
        if self.lora_scope == "all":
            return ["transformer.attn.proj_q", "transformer.attn.proj_k", "transformer.attn.proj_v",
                    "transformer.proj", "transformer.pwff.fc1", "transformer.pwff.fc2"]
        return ["transformer.attn.proj_q", "transformer.attn.proj_v"]

    def load_pretrained(self, path: str) -> int:
        # The checkpoint is the state dict of LIMUBertModel4Pretrain. Its encoder keys
        # ("transformer.embed.lin.weight", ...) are identical to ours; the decoder
        # keys ("fc.", "linear.", "decoder.") are simply ignored.
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if k.startswith("transformer.")}
        key = "transformer.embed.pos_embed.weight"
        n_pos = self.transformer.embed.pos_embed.weight.shape[0]
        if key in state and state[key].shape[0] > n_pos:      # shorter window (HARBox: 2 s = 40 samples):
            state[key] = state[key][:n_pos].clone()           # the first positions are exactly those times
        elif key in state and state[key].shape[0] < n_pos:    # longer window, e.g. 120 -> 200 positions
            pe = state[key].T.unsqueeze(0)                      # (1, 72, 120)
            state[key] = F.interpolate(pe, size=n_pos, mode="linear", align_corners=True)[0].T.contiguous()
        return self._load_matching(state)

"""
HALO (arXiv 2608.27233) - Heterogeneity-Aware, Language-aligned IMU foundation model.

HALO's code and weights are "to be released upon publication", so this is a
compact RE-IMPLEMENTATION of its IMU encoder (stage 1) that follows the paper:

  1. Adaptive-pooling tokenisation: the window is cut into 1-second patches and
     EVERY (patch, channel) pair becomes one D-dim token through a hybrid
     feature extractor:
        temporal branch   multi-scale 1-D CNN (kernels 3, 5, 7) + adaptive average pooling
        spectral branch   |FFT| of the patch pooled to a fixed number of bins + MLP
     Each patch-channel is z-score normalised first (removes device/subject bias).
  2. Sensor conditioning: token += gamma * e_c, gamma initialised to 0.1.
  3. Dual-branch Transformer (8 layers) alternating
        temporal self-attention over patches (within a channel)   and
        cross-channel self-attention over channels (within a patch).
  4. Cross-channel fusion with a small set of learnable queries, a light
     Transformer over the fused patches, multi-query attention pooling and an
     MLP projection to the session embedding.

Documented deviations (because we have no text model / single-device data):
  * e_c is a learned per-channel embedding instead of a frozen Sentence-BERT
    embedding of a textual sensor description.
  * Each token also receives the patch mean and log-std (a "statistics branch"),
    because z-scoring alone removes the movement intensity that separates e.g.
    sitting from walking in single-sensor wrist data.
  * Width is 192 instead of 384 so it trains on a 6 GB GPU (~4 M parameters,
    close to the PDF's example encoder).
Pretraining (MAE + patch contrastive) lives in fedpeft/training/ssl_halo.py.

LoRA targets: query and value projections of every attention block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseEncoder, InputSpec


class AttentionBlock(nn.Module):
    """Pre-LayerNorm Transformer block with separate q/k/v projections (LoRA-friendly)."""

    def __init__(self, d: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)     # <- LoRA
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)     # <- LoRA
        self.o = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (N, S, D)
        N, S, D = x.shape
        h = self.norm1(x)
        q, k, v = (p(h).view(N, S, self.n_heads, -1).transpose(1, 2) for p in (self.q, self.k, self.v))
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(N, S, D)
        x = x + self.drop(self.o(a))
        return x + self.drop(self.ff(self.norm2(x)))


class QueryPooling(nn.Module):
    """Multi-query attention pooling: n learnable queries attend over a token set."""

    def __init__(self, d: int, n_queries: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        self.norm = nn.LayerNorm(d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (N, S, D) -> (N, n_queries, D)
        N, S, D = x.shape
        h = self.norm(x)
        q = self.queries.unsqueeze(0).expand(N, -1, -1)
        q = q.reshape(N, -1, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = self.k(h).view(N, S, self.n_heads, -1).transpose(1, 2)
        v = self.v(h).view(N, S, self.n_heads, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(N, -1, D)
        return self.o(a)


class HaloTokenizer(nn.Module):
    """Hybrid (temporal CNN + spectral + statistics) tokeniser for each patch-channel pair."""

    def __init__(self, d: int, n_bins: int = 16, conv_ch: int = 32, pooled: int = 4):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(1, conv_ch, k, padding=k // 2) for k in (3, 5, 7)])
        self.pool = nn.AdaptiveAvgPool1d(pooled)        # length-independent -> any sampling rate works
        self.temporal = nn.Linear(3 * conv_ch * pooled, d)
        self.n_bins = n_bins
        self.spectral = nn.Sequential(nn.Linear(n_bins, d), nn.GELU(), nn.Linear(d, d))
        self.stats = nn.Linear(2, d)

    def forward(self, patches: torch.Tensor):
        """patches: (B, C, P, L) raw signal -> tokens (B, C, P, D), normalised patches (B, C, P, L)."""
        B, C, P, L = patches.shape
        mean = patches.mean(-1, keepdim=True)
        std = patches.std(-1, keepdim=True)
        z = (patches - mean) / (std + 1e-3)                             # per patch-channel z-score
        flat = z.reshape(B * C * P, 1, L)
        t = torch.cat([F.gelu(conv(flat)) for conv in self.convs], dim=1)
        t = self.temporal(self.pool(t).flatten(1))                      # temporal branch
        spec = torch.fft.rfft(flat.float().squeeze(1), dim=-1).abs()    # FFT in fp32 (any length)
        spec = F.adaptive_avg_pool1d(spec.unsqueeze(1), self.n_bins).squeeze(1)
        s = self.spectral(torch.log1p(spec).to(t.dtype))                # spectral branch
        stats = torch.cat([mean, torch.log(std + 1e-3)], dim=-1).reshape(B * C * P, 2)
        tok = t + s + self.stats(stats.to(t.dtype))                     # statistics branch
        return tok.view(B, C, P, -1), z


def sinusoid_table(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    table = torch.zeros(n, d)
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div)
    return table


class HaloEncoder(BaseEncoder):
    name = "halo"
    display_name = "HALO"
    architecture = "Patch-channel Transformer, 8 alternating temporal/channel layers, d=192"
    pretrained_source = "Our re-implementation, self-supervised (MAE + patch contrastive) on the Capture-24 pool users"
    spec = InputSpec(rate_hz=30, n_samples=300, n_channels=3, acc_unit="g")

    def __init__(self, d: int = 192, n_layers: int = 8, n_heads: int = 6, patch_s: float = 1.0,
                 out_dim: int = 256, n_pool_queries: int = 4, dropout: float = 0.1):
        super().__init__()
        self.patch_len = int(round(self.spec.rate_hz * patch_s))       # 30 samples = 1 s
        self.d = d
        self.tokenizer = HaloTokenizer(d)
        self.channel_embed = nn.Parameter(torch.randn(self.spec.n_channels, d) * 0.02)   # sensor conditioning e_c
        self.channel_gamma = nn.Parameter(torch.tensor(0.1))                             # gamma (init 0.1)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))                                 # learnable PE scale
        self.register_buffer("pos_table", sinusoid_table(64, d))
        self.mask_token = nn.Parameter(torch.zeros(d))                                   # used by MAE pretraining
        self.blocks = nn.ModuleList([AttentionBlock(d, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.fusion = QueryPooling(d, 1, n_heads)                        # channels -> 1 token per patch
        self.patch_block = AttentionBlock(d, n_heads, dropout=dropout)   # light Transformer over fused patches
        self.pool = QueryPooling(d, n_pool_queries, n_heads)             # patches -> session tokens
        self.proj = nn.Sequential(nn.LayerNorm(n_pool_queries * d), nn.Linear(n_pool_queries * d, out_dim),
                                  nn.GELU(), nn.Linear(out_dim, out_dim))
        self.feat_dim = out_dim

    # ---------------------------------------------------------------- pieces
    def patchify(self, x: torch.Tensor) -> torch.Tensor:       # (B, C, T) -> (B, C, P, L)
        return x.unfold(-1, self.patch_len, self.patch_len)

    def encode_tokens(self, x: torch.Tensor, patch_mask: torch.Tensor | None = None):
        """Tokenise + dual-branch Transformer.

        patch_mask: optional (B, P) bool, True = replace that patch by the mask token (MAE).
        Returns channel tokens (B, C, P, D) and the normalised patches (B, C, P, L).
        """
        tok, z = self.tokenizer(self.patchify(x))
        B, C, P, D = tok.shape
        if patch_mask is not None:
            tok = torch.where(patch_mask[:, None, :, None], self.mask_token.to(tok.dtype), tok)
        tok = tok + self.channel_gamma * self.channel_embed[None, :C, None, :]
        tok = tok + self.pos_scale * self.pos_table[None, None, :P, :]
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:   # temporal attention: patches of one channel talk to each other
                tok = block(tok.reshape(B * C, P, D)).view(B, C, P, D)
            else:            # cross-channel attention: channels of one patch talk to each other
                tok = block(tok.transpose(1, 2).reshape(B * P, C, D)).view(B, P, C, D).transpose(1, 2)
        return tok, z

    def fuse_patches(self, tok: torch.Tensor) -> torch.Tensor:  # (B, C, P, D) -> (B, P, D)
        B, C, P, D = tok.shape
        fused = self.fusion(tok.transpose(1, 2).reshape(B * P, C, D)).view(B, P, D)
        return self.patch_block(fused)

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, 3, 300)
        tok, _ = self.encode_tokens(x)
        patches = self.fuse_patches(tok)
        session = self.pool(patches).flatten(1)                # (B, n_queries * D)
        return self.proj(session)                              # (B, 256)

    def lora_targets(self) -> list[str]:
        names = []
        for i in range(len(self.blocks)):
            names += [f"blocks.{i}.q", f"blocks.{i}.v"]
        return names + ["patch_block.q", "patch_block.v"]

    def load_pretrained(self, path: str) -> int:
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {(k[len("encoder."):] if k.startswith("encoder.") else k): v for k, v in state.items()}
        return self._load_matching(state)

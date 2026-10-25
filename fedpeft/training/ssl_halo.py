"""
HALO stage-1 self-supervised pretraining (HALO paper, "Stage 1 Pretraining Objectives").

HALO has no public weights yet, so we pretrain our re-implementation ourselves on
the UNLABELLED windows of the pool users (disjoint from FL clients and held-out
users, so nothing leaks into the evaluation).

Loss = lambda_mae * MAE + lambda_con * patch-contrastive      (both lambdas = 1)

    MAE              50 % of the 1-s patches are replaced by a learned mask token and
                     the temporal encoder must reconstruct the z-normalised signal of
                     the masked patches (MSE on masked positions only).
    patch contrast   two augmented views of the same window; the same patch in both
                     views is a positive pair, every other patch in the batch a negative
                     (InfoNCE, temperature 0.2).
    augmentations    jittering, scaling, time warping (the paper's ablation shows these
                     matter most for generalisation).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.store import WindowStore
from ..models.components import Preprocessor
from ..models.encoders import build_encoder, checkpoint_path
from ..models.encoders.halo import HaloEncoder
from ..utils import get_logger, save_json, seed_everything
from .engine import autocast, make_scaler


# ----------------------------------------------------------------------------
# Augmentations (batched, on the GPU)
# ----------------------------------------------------------------------------
def augment(x: torch.Tensor, jitter: float = 0.02, scale: float = 0.1, warp: float = 0.2) -> torch.Tensor:
    """Scaling + jittering + smooth random time warping of a batch (B, C, T)."""
    B, C, T = x.shape
    x = x * (1 + scale * torch.randn(B, C, 1, device=x.device))           # scaling
    x = x + jitter * torch.randn_like(x)                                     # jittering
    # Time warp: a smooth random speed profile -> monotone sampling positions.
    speed = 1 + warp * torch.randn(B, 1, 6, device=x.device)
    speed = F.interpolate(speed.clamp(min=0.3), size=T, mode="linear", align_corners=True).squeeze(1)
    pos = torch.cumsum(speed, dim=1)
    pos = (pos - pos[:, :1]) / (pos[:, -1:] - pos[:, :1]) * 2 - 1           # map to [-1, 1]
    grid = torch.stack([pos, torch.zeros_like(pos)], dim=-1).unsqueeze(1)  # (B, 1, T, 2)
    return F.grid_sample(x.unsqueeze(2), grid, align_corners=True).squeeze(2)


class HaloPretrainer(nn.Module):
    """Encoder + the two pretraining heads (MAE decoder, contrastive projector)."""

    def __init__(self, encoder: HaloEncoder, proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        d = encoder.d
        self.decoder = nn.Linear(d, encoder.patch_len)                     # token -> normalised patch
        self.projector = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, proj_dim))

    def mae_loss(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        B = x.shape[0]
        P = x.shape[-1] // self.encoder.patch_len
        mask = torch.rand(B, P, device=x.device) < mask_ratio
        mask[mask.sum(1) == 0, 0] = True                                    # at least one masked patch
        tok, z = self.encoder.encode_tokens(x, patch_mask=mask)
        err = ((self.decoder(tok).float() - z.float()) ** 2).mean(-1)      # (B, C, P)
        m = mask[:, None, :].expand_as(err)
        return err[m].mean()

    def contrastive_loss(self, v1: torch.Tensor, v2: torch.Tensor, temperature: float) -> torch.Tensor:
        p1 = self.projector(self.encoder.fuse_patches(self.encoder.encode_tokens(v1)[0]))   # (B, P, proj)
        p2 = self.projector(self.encoder.fuse_patches(self.encoder.encode_tokens(v2)[0]))
        p1 = F.normalize(p1.flatten(0, 1).float(), dim=-1)
        p2 = F.normalize(p2.flatten(0, 1).float(), dim=-1)
        logits = p1 @ p2.t() / temperature
        target = torch.arange(len(p1), device=p1.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def pretrain_halo(cfg, split: dict, seed: int, device: torch.device) -> dict:
    """Pretrain HALO on the pool users of `split` and save the encoder weights."""
    log = get_logger("halo_ssl")
    seed_everything(seed)
    h = cfg.halo_ssl
    store = WindowStore(cfg.paths.processed_dir, split["pool"], cfg.data.label_scheme, h.subsample_every, device)
    # The encoder is configured for the dataset (Capture-24: 3 acc axes at 30 Hz, 1-s
    # patches; HARBox: 9 axes at 50 Hz, 0.4-s patches) and sees exactly what it will
    # see later: the stored windows after the same unit conversion / resampling.
    encoder = build_encoder("halo", cfg, pretrained=False, seed=seed, data=store.spec).to(device)
    preprocess = Preprocessor(encoder.spec, store.spec, cfg.encoders.halo.get("fit", "crop")).to(device)
    model = HaloPretrainer(encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=h.lr, weight_decay=1e-5)
    scaler = make_scaler(device, cfg.runtime.amp)
    n = len(store.y)
    steps_per_epoch = max(1, math.ceil(n / h.batch_size))   # the last batch may be smaller
    total = steps_per_epoch * h.epochs
    log.info(f"HALO SSL: {n:,} unlabelled windows from {len(split['pool'])} pool users, "
             f"{sum(p.numel() for p in encoder.parameters()):,} encoder parameters, {total} steps")

    history, step, t0 = [], 0, time.perf_counter()
    for epoch in range(1, h.epochs + 1):
        order = torch.randperm(n, device=device)
        sums = {"mae": 0.0, "con": 0.0}
        model.train()
        for i in range(steps_per_epoch):
            lr = h.lr * 0.5 * (1 + math.cos(math.pi * step / total))       # cosine schedule
            for g in opt.param_groups:
                g["lr"] = lr
            x = preprocess(store.X[order[i * h.batch_size:(i + 1) * h.batch_size]])
            with autocast(device, cfg.runtime.amp):
                mae = model.mae_loss(x, h.mask_ratio)
                con = model.contrastive_loss(augment(x), augment(x), h.temperature)
                loss = mae + con
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sums["mae"] += mae.item()
            sums["con"] += con.item()
            step += 1
        rec = {"epoch": epoch, "mae": sums["mae"] / steps_per_epoch, "contrastive": sums["con"] / steps_per_epoch,
               "elapsed_s": time.perf_counter() - t0}
        history.append(rec)
        log.info(f"epoch {epoch}: MAE {rec['mae']:.4f}  contrastive {rec['contrastive']:.4f}  ({rec['elapsed_s']:.0f}s)")

    path = checkpoint_path(cfg, "halo", seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), path)
    summary = {"seed": seed, "pool_users": split["pool"], "n_windows": n, "history": history,
               "checkpoint": str(path), "time_s": time.perf_counter() - t0}
    save_json(summary, Path(cfg.paths.results_dir) / cfg.profile / "pretraining" / f"halo_ssl_seed{seed}.json")
    log.info(f"saved {path}")
    return summary

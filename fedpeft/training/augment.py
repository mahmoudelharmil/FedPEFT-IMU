"""
Training-time data augmentation for IMU windows (configs: `augment:`).

Applied to the stored windows of a training batch, before the preprocessor, so it
works the same for every encoder.  Off for Capture-24 (one wrist device per user),
on for HARBox, where 77 phone models are held in many different ways:

    rotation  one random 3-D rotation per window, applied to every sensor triplet
              (acc, gyro and mag rotate together, as for a phone held at another
              angle); the angle is uniform in [0, rotation_deg] around a random axis
    scale     one random gain per window, x * (1 + N(0, scale_std))   (sensor gain)
    jitter    Gaussian noise relative to each channel's std in the window

Each window is augmented with probability p (the rest stay untouched, so the
model also keeps seeing the real data).

Speed: rotations and gains of a whole batch are built on the CPU as ONE
per-window C x C matrix (block-diagonal rotation x gain) and applied with a single
batched matrix product on the GPU - tiny models are limited by the number of GPU
kernel launches, not by arithmetic.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def random_rotations(n: int, max_deg: float, rng: np.random.Generator) -> np.ndarray:
    """n rotation matrices (n, 3, 3): random axis, angle uniform in [0, max_deg] (Rodrigues' formula)."""
    axis = rng.standard_normal((n, 3))
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-8)
    angle = rng.random(n) * math.radians(max_deg)
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    s, c = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


class Augmenter:
    """Callable: (B, C, L) windows -> augmented float copy."""

    def __init__(self, triplets: list[tuple[int, int, int]], n_channels: int, rotation_deg: float = 0.0,
                 scale_std: float = 0.0, jitter_std: float = 0.0, p: float = 1.0, seed: int = 0):
        self.triplets = [list(t) for t in triplets]
        self.n_channels = n_channels
        self.rotation_deg = float(rotation_deg)
        self.scale_std = float(scale_std)
        self.jitter_std = float(jitter_std)
        self.p = float(p)
        self.rng = np.random.default_rng(seed)

    def matrices(self, B: int) -> tuple[np.ndarray, np.ndarray]:
        """(B, C, C) per-window transforms (block-diagonal rotations times a gain) and the
        mask of augmented windows."""
        M = np.tile(np.eye(self.n_channels, dtype=np.float32), (B, 1, 1))
        on = self.rng.random(B) < self.p
        if self.rotation_deg > 0 and self.triplets:
            R = random_rotations(B, self.rotation_deg, self.rng)
            R[~on] = np.eye(3)
            for idx in self.triplets:               # the same rotation for acc, gyro and mag
                M[:, idx[0]:idx[0] + 3, idx[0]:idx[0] + 3] = R
        if self.scale_std > 0:
            gain = 1 + self.scale_std * self.rng.standard_normal(B)
            M *= np.where(on, gain, 1.0)[:, None, None].astype(np.float32)
        return M, on

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        M, on = self.matrices(x.shape[0])
        out = torch.bmm(torch.from_numpy(M).to(x.device, non_blocking=True), x.float())
        if self.jitter_std > 0:
            mask = torch.from_numpy(on.astype(np.float32)).to(x.device)[:, None, None]
            out = out + mask * self.jitter_std * out.std(dim=-1, keepdim=True) * torch.randn_like(out)
        return out


def make_augmenter(cfg, spec, seed: int = 0) -> Augmenter | None:
    """The augmenter of the experiment, or None when augmentation is off.

    `spec` is the DataSpec of the stored windows (its channel groups give the triplets,
    which must be consecutive channels such as acc_x, acc_y, acc_z).
    """
    a = cfg.get("augment") or {}
    if not a.get("enabled"):
        return None
    triplets = [tuple(ix) for _, ix in spec.groups if len(ix) == 3 and list(ix) == list(range(ix[0], ix[0] + 3))]
    return Augmenter(triplets, spec.n_channels, a.get("rotation_deg", 0.0), a.get("scale_std", 0.0),
                     a.get("jitter_std", 0.0), a.get("p", 1.0), seed)

"""
Components 1, 2, 6, 7 and 8 of the FedPEFT-IMU client model (PDF p. 15-18).

    1 Preprocessor    fixed, no parameters: convert units / resample / crop (or tile) so
                      every window looks like what the encoder saw during pretraining
    2 InputAdapter    1x1 convolution, starts as the identity (tens of parameters)
    6 SharedHead g    feature z -> 64-d activity embedding e   (federated)
    7 PersonalHead h  embedding e -> class scores              (local in V5)
    8 prototypes      per-class mean embedding + the prototype loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.store import DataSpec
from .encoders.base import InputSpec

G_TO_MS2 = 9.80665


# ----------------------------------------------------------------------------
# 1. Preprocessing (encoder-specific part)
# ----------------------------------------------------------------------------
MAG_SCALE = 1.0 / 50.0      # magnetometer uT -> about unit size (the Earth's field is 25-65 uT)


def unit_scales(data: DataSpec, acc_unit: str) -> torch.Tensor:
    """Per-channel factor that converts the stored units into the encoder's units.

        acc   g <-> m/s^2 as the encoder expects (LIMU-BERT, ssl-wearables, HALO: g; UniMTS: m/s^2)
        gyro  rad/s, kept (LIMU-BERT was pretrained on rad/s)
        mag   uT / 50 (no encoder was pretrained on it; this only keeps it well scaled)
    """
    scale = torch.ones(data.n_channels)
    for group, idx in data.groups:
        unit = data.unit(group)
        if group == "acc":
            if unit == "g" and acc_unit == "m/s2":
                scale[list(idx)] = G_TO_MS2
            elif unit == "m/s2" and acc_unit == "g":
                scale[list(idx)] = 1.0 / G_TO_MS2
        elif group == "mag":
            scale[list(idx)] = MAG_SCALE
    return scale


class Preprocessor(nn.Module):
    """Map the canonical window of the dataset to the encoder's input format.

    Steps (all deterministic, identical on every client):
        units     per channel: stored units -> encoder units (unit_scales)
        resample  dataset rate -> encoder rate, with an anti-aliasing filter
        length    centre crop to the encoder's window length (e.g. 6 s of Capture-24's
                  10 s for LIMU-BERT); if the encoder needs MORE samples than the
                  window has, `fit` decides: "tile" repeats the window (ssl-wearables
                  needs 10 s, a HARBox window is 2 s), "pad" adds zeros
    """

    def __init__(self, spec: InputSpec, data: DataSpec, fit: str = "crop"):
        super().__init__()
        self.base_len = data.n_samples
        self.resampled_len = int(round(spec.rate_hz * data.window_s))
        self.crop_len = spec.n_samples
        self.fit = fit
        self.register_buffer("scale", unit_scales(data, spec.acc_unit).view(1, -1, 1), persistent=False)
        if self.crop_len > self.resampled_len and fit not in ("tile", "pad"):
            raise ValueError(f"Encoder window ({self.crop_len} samples) is longer than the stored window "
                             f"({self.resampled_len} at the encoder rate): set encoders.<name>.fit to tile or pad")

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, C, L), float16
        x = x.float() * self.scale
        if self.resampled_len != self.base_len:
            # Treat the signal as a 1-pixel-high image to get antialiased linear resampling.
            x = F.interpolate(x.unsqueeze(2), size=(1, self.resampled_len), mode="bilinear",
                              antialias=True, align_corners=False).squeeze(2)
        if self.crop_len < self.resampled_len:
            start = (self.resampled_len - self.crop_len) // 2
            x = x[..., start:start + self.crop_len]
        elif self.crop_len > self.resampled_len:
            if self.fit == "tile":
                reps = -(-self.crop_len // self.resampled_len)
                x = x.repeat(1, 1, reps)[..., :self.crop_len]
            else:
                x = F.pad(x, (0, self.crop_len - self.resampled_len))
        return x


# ----------------------------------------------------------------------------
# 2. Input adapter
# ----------------------------------------------------------------------------
class InputAdapter(nn.Module):
    """1x1 conv mapping device channels -> encoder channels, initialised by channel MEANING.

    Every encoder channel whose name also exists in the data (e.g. "acc_x") starts
    as a copy of that data channel; all other weights start at zero.
    Examples:
        Capture-24 (acc) -> LIMU-BERT (acc + gyro): acc copied, gyro inputs start at 0
        HARBox (acc + gyro + mag) -> LIMU-BERT: acc and gyro copied, mag unused at first
    The adapter may learn device-specific axis mixing from there.
    """

    def __init__(self, in_channels: list[str] | int, out_channels: list[str] | int):
        super().__init__()
        ins = list(in_channels) if not isinstance(in_channels, int) else [f"c{i}" for i in range(in_channels)]
        outs = list(out_channels) if not isinstance(out_channels, int) else [f"c{i}" for i in range(out_channels)]
        self.proj = nn.Conv1d(len(ins), len(outs), kernel_size=1)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            for j, name in enumerate(outs):
                if name in ins:
                    self.proj.weight[j, ins.index(name), 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ----------------------------------------------------------------------------
# 6 + 7. Heads
# ----------------------------------------------------------------------------
class SharedHead(nn.Module):
    """g: encoder feature z -> activity embedding e (PDF: "256 -> 64 with a non-linearity").

    A LayerNorm first puts the very different feature scales of the four encoders
    on a common footing (it belongs to g, so it is federated together with it).
    """

    def __init__(self, feat_dim: int, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(feat_dim), nn.Dropout(dropout),
                                 nn.Linear(feat_dim, embed_dim), nn.ReLU())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PersonalHead(nn.Module):
    """h: linear classifier on the embedding. Global in V0-V4, personal (local) in V5."""

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.fc(e)

    @torch.no_grad()
    def init_from_prototypes(self, protos: torch.Tensor, present: torch.Tensor) -> None:
        """Make h equal to the (cosine) nearest-prototype classifier.

        argmin_c ||e/|e| - P_c/|P_c| ||^2  ==  argmax_c (P_c/|P_c|) . e , so
        W_c = P_c / |P_c| and b_c = 0 (absent classes get a very low bias).
        Used for few-shot adaptation: training starts from the zero-shot solution.
        """
        self.fc.weight.copy_(F.normalize(protos, dim=1))
        bias = torch.zeros_like(self.fc.bias)
        bias[~present] = -1e4
        self.fc.bias.copy_(bias)


# ----------------------------------------------------------------------------
# 8. Class prototypes
# ----------------------------------------------------------------------------
@torch.no_grad()
def class_prototypes(emb: torch.Tensor, y: torch.Tensor, n_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """p_c = mean embedding of the windows labelled c.  Returns (C x D prototypes, C counts)."""
    emb = emb.float()
    sums = torch.zeros(n_classes, emb.shape[1], device=emb.device).index_add_(0, y, emb)
    counts = torch.bincount(y, minlength=n_classes).float()
    protos = sums / counts.clamp(min=1).unsqueeze(1)
    return protos, counts


def prototype_loss(emb: torch.Tensor, y: torch.Tensor, protos: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """L_proto = mean over the batch of || e_i/|e_i| - P_{y_i}/|P_{y_i}| ||^2.

    This is the PDF's squared-distance prototype loss computed on L2-normalised
    vectors (= 2 - 2 cos).  Normalising makes the term scale-free: with raw
    distances the loss could be lowered simply by shrinking every embedding, and
    its size would differ wildly between encoders.  Only classes that already have
    a global prototype contribute.
    """
    valid = present[y]
    if not valid.any():
        return emb.new_zeros(())
    diff = F.normalize(emb[valid].float(), dim=1) - F.normalize(protos[y[valid]].float(), dim=1)
    return (diff ** 2).sum(1).mean()


def nearest_prototype_logits(emb: torch.Tensor, protos: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """Zero-shot classifier for new users: score_c = cosine(e, P_c) (absent classes -> -inf)."""
    scores = F.normalize(emb.float(), dim=1) @ F.normalize(protos.float(), dim=1).T
    scores[:, ~present] = float("-inf")
    return scores

"""
Common interface of the four IMU encoders (component 3 of the PDF).

Every encoder declares:
    spec             what input it expects (rate, length, channels, units) -> used by
                     the Preprocessor (component 1) and the InputAdapter (component 2)
    feat_dim         size of the feature vector z it returns for one window
    lora_targets()   names of the layers that receive LoRA (component 4)
    load_pretrained  how to read the official checkpoint into this re-implementation
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class InputSpec:
    rate_hz: float      # sampling rate the encoder was pretrained at
    n_samples: int      # samples per window the encoder consumes
    n_channels: int     # channels after the input adapter (e.g. 6 = acc + gyro for LIMU-BERT)
    acc_unit: str       # "g" or "m/s2"

    @property
    def seconds(self) -> float:
        return self.n_samples / self.rate_hz


class BaseEncoder(nn.Module):
    """Base class: subclasses set `name`, `display_name`, `spec`, `feat_dim`."""

    name: str = "base"
    display_name: str = "Base"
    architecture: str = ""          # one-line description for the report
    pretrained_source: str = ""     # where the official weights come from
    spec: InputSpec
    feat_dim: int

    def lora_targets(self) -> list[str]:
        """Module names (relative to the encoder) that get a LoRA branch."""
        raise NotImplementedError

    def load_pretrained(self, path: str) -> int:
        """Load official weights. Returns the number of tensors that were loaded."""
        raise NotImplementedError

    # ------------------------------------------------------------------ helper
    def _load_matching(self, state: dict[str, torch.Tensor], strip_prefix: str = "",
                       min_fraction: float = 0.9) -> int:
        """Copy every tensor whose (prefix-stripped) name and shape match this module.

        Fails loudly if less than `min_fraction` of our own tensors were found, so a
        silently mismatched checkpoint can never produce "pretrained" results.
        """
        own = self.state_dict()
        loaded = {}
        for key, value in state.items():
            if strip_prefix and not key.startswith(strip_prefix):
                continue
            k = key[len(strip_prefix):]
            if k in own and own[k].shape == value.shape:
                loaded[k] = value
        own.update(loaded)
        self.load_state_dict(own)
        fraction = len(loaded) / max(1, len(own))
        if fraction < min_fraction:
            raise RuntimeError(f"{self.name}: only {len(loaded)}/{len(own)} tensors matched the checkpoint")
        return len(loaded)

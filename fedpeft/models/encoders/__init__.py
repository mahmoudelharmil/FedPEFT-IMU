"""
Encoder registry.

    build_encoder("ssl_wearables", cfg, pretrained=True, seed=0)

returns a ready encoder with the official weights loaded (or random weights when
pretrained=False, i.e. "learning from scratch").
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseEncoder, InputSpec
from .halo import HaloEncoder
from .limu_bert import LimuBertEncoder
from .ssl_wearables import SslWearablesEncoder
from .unimts import UniMTSEncoder

ENCODERS: dict[str, type[BaseEncoder]] = {
    "limu_bert": LimuBertEncoder,
    "ssl_wearables": SslWearablesEncoder,
    "unimts": UniMTSEncoder,
    "halo": HaloEncoder,
}


def encoder_display_name(name: str) -> str:
    return ENCODERS[name].display_name


def checkpoint_path(cfg, name: str, seed: int = 0) -> Path:
    """Absolute path of the pretrained checkpoint of encoder `name`."""
    rel = cfg.encoders[name].checkpoint.format(seed=seed)
    return Path(cfg.paths.weights_dir) / rel


def build_encoder(name: str, cfg, pretrained: bool, seed: int = 0) -> BaseEncoder:
    """Instantiate an encoder and optionally load its pretrained weights."""
    if name == "unimts":
        encoder = UniMTSEncoder(wrist_joint=cfg.encoders.unimts.wrist_joint)
    elif name == "limu_bert":
        encoder = LimuBertEncoder(seq_len=int(cfg.encoders.limu_bert.get("seq_len", 120)))
    else:
        encoder = ENCODERS[name]()
    if pretrained:
        path = checkpoint_path(cfg, name, seed)
        if not path.exists():
            raise FileNotFoundError(
                f"Pretrained weights for '{name}' not found at {path}. "
                f"Run scripts/02_download_weights.py (and 03_pretrain_halo.py for HALO) first.")
        encoder.load_pretrained(str(path))
    return encoder


__all__ = ["BaseEncoder", "InputSpec", "ENCODERS", "build_encoder", "encoder_display_name", "checkpoint_path"]

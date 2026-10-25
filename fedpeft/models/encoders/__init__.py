"""
Encoder registry.

    build_encoder("ssl_wearables", cfg, pretrained=True, seed=0, data=spec)

returns a ready encoder with the official weights loaded (or random weights when
pretrained=False, i.e. "learning from scratch").  `data` (a DataSpec) describes the
stored windows; encoders whose input format follows the data (HALO's channels, the
MLP baseline) are built from it.
"""

from __future__ import annotations

from pathlib import Path

from .base import ACC, BaseEncoder, InputSpec
from .halo import HaloEncoder
from .convgru import ConvGruEncoder
from .limu_bert import LimuBertEncoder
from .lstm import LstmEncoder
from .mlp import MlpEncoder
from .ssl_wearables import SslWearablesEncoder
from .unimts import UniMTSEncoder

ENCODERS: dict[str, type[BaseEncoder]] = {
    "limu_bert": LimuBertEncoder,
    "ssl_wearables": SslWearablesEncoder,
    "unimts": UniMTSEncoder,
    "halo": HaloEncoder,
    "mlp": MlpEncoder,
    "lstm": LstmEncoder,
    "convgru": ConvGruEncoder,
}


def encoder_display_name(name: str) -> str:
    return ENCODERS[name].display_name


def checkpoint_path(cfg, name: str, seed: int = 0) -> Path:
    """Absolute path of the pretrained checkpoint of encoder `name`."""
    rel = cfg.encoders[name].checkpoint.format(seed=seed)
    return Path(cfg.paths.weights_dir) / rel


def build_encoder(name: str, cfg, pretrained: bool, seed: int = 0, data=None) -> BaseEncoder:
    """Instantiate an encoder (configured for the dataset) and optionally load its pretrained weights."""
    ecfg = cfg.encoders.get(name, {})
    if name == "unimts":
        encoder = UniMTSEncoder(wrist_joint=ecfg.wrist_joint, n_samples=int(ecfg.get("n_samples", 200)))
    elif name == "limu_bert":
        encoder = LimuBertEncoder(seq_len=int(ecfg.get("seq_len", 120)), lora_scope=str(ecfg.get("lora_scope", "qv")))
    elif name == "halo":
        channels = tuple(data.channels) if (data is not None and ecfg.get("in_channels") == "all") else ACC
        encoder = HaloEncoder(rate_hz=float(ecfg.get("rate_hz", 30)), n_samples=int(ecfg.get("n_samples", 300)),
                              patch_s=float(ecfg.get("patch_s", 1.0)), channels=channels)
    elif name == "mlp":
        if data is None:
            raise ValueError("the MLP baseline is built from the data format: pass data=DataSpec")
        encoder = MlpEncoder(tuple(data.channels), data.rate_hz, data.n_samples,
                             acc_unit=data.unit("acc") or "g", hidden=int(ecfg.get("hidden", 300)))
    elif name == "convgru":
        if data is None:
            raise ValueError("the Conv-GRU baseline is built from the data format: pass data=DataSpec")
        encoder = ConvGruEncoder(tuple(data.channels), data.rate_hz, data.n_samples, acc_unit=data.unit("acc") or "g",
                                 hidden=int(ecfg.get("hidden", 128)))
    elif name == "lstm":
        if data is None:
            raise ValueError("the LSTM baseline is built from the data format: pass data=DataSpec")
        encoder = LstmEncoder(tuple(data.channels), data.rate_hz, data.n_samples,
                              acc_unit=data.unit("acc") or "g", hidden=int(ecfg.get("hidden", 6)))
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

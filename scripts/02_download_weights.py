"""
Step 2 - download the official pretrained encoder weights.

    python scripts/02_download_weights.py

    LIMU-BERT      github.com/dapowan/LIMU-BERT-Public   saved/pretrain_base_hhar_20_120/hhar.pt  (~0.3 MB)
    ssl-wearables  github.com/OxWearables/ssl-wearables  model_check_point/mtl_best.mdl          (~42 MB)
    UniMTS         huggingface.co/xiyuanz/UniMTS         checkpoint/UniMTS.pth                   (~275 MB)
                   -> only the ST-GCN part ("acc.*", ~12 MB) is kept; the CLIP text
                      encoder inside the checkpoint is not needed for classification.
    HALO           no public weights -> run scripts/03_pretrain_halo.py
"""

from pathlib import Path

import torch
from _common import setup

from fedpeft.utils import download_file

SOURCES = {
    "limu_bert": ("https://raw.githubusercontent.com/dapowan/LIMU-BERT-Public/master/"
                  "saved/pretrain_base_hhar_20_120/hhar.pt", "limu_bert/pretrain_base_hhar_20_120.pt"),
    "ssl_wearables": ("https://raw.githubusercontent.com/OxWearables/ssl-wearables/main/"
                      "model_check_point/mtl_best.mdl", "ssl_wearables/mtl_best.mdl"),
    "unimts": ("https://huggingface.co/xiyuanz/UniMTS/resolve/main/checkpoint/UniMTS.pth",
               "unimts/UniMTS_full.pth"),
}

if __name__ == "__main__":
    cfg, device, log, _ = setup("02_download_weights", __doc__)
    wdir = Path(cfg.paths.weights_dir)
    for name, (url, rel) in SOURCES.items():
        final = wdir / (cfg.encoders[name].checkpoint if name != "unimts" else "unimts/unimts_stgcn.pt")
        if final.exists():
            log.info(f"{name}: already present ({final})")
            continue
        path = download_file(url, wdir / rel, desc=name)
        log.info(f"{name}: downloaded {path.stat().st_size / 1e6:.1f} MB")
        if name == "unimts":
            # Keep only the IMU encoder (ST-GCN) weights of the CLIP-style model.
            full = torch.load(path, map_location="cpu", weights_only=True)
            stgcn = {k: v for k, v in full.items() if k.startswith("acc.")}
            torch.save(stgcn, final)
            path.unlink()
            log.info(f"unimts: kept {len(stgcn)} ST-GCN tensors -> {final}")

    # Sanity check: every checkpoint must load into our re-implementations.
    from fedpeft.models.encoders import build_encoder
    for name in ("limu_bert", "ssl_wearables", "unimts"):
        enc = build_encoder(name, cfg, pretrained=True)
        log.info(f"{name}: OK ({sum(p.numel() for p in enc.parameters()):,} parameters)")

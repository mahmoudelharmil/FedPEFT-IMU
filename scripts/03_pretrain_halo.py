"""
Step 3 - self-supervised pretraining of the HALO encoder (no public weights exist).

    python scripts/03_pretrain_halo.py [--profile standard]

Uses only the UNLABELLED windows of the pool users of every split seed, saves
weights/halo/halo_ssl_pool_seed<seed>.pt and a loss log in results/<profile>/pretraining/.
"""

from pathlib import Path

from _common import setup

from fedpeft.data.splits import get_user_split
from fedpeft.models.encoders import checkpoint_path
from fedpeft.training.ssl_halo import pretrain_halo
from fedpeft.utils import load_json


def extra(p):
    p.add_argument("--force", action="store_true", help="retrain even if the checkpoint exists")


if __name__ == "__main__":
    cfg, device, log, args = setup("03_pretrain_halo", __doc__, extra)
    meta = load_json(Path(cfg.paths.processed_dir) / "meta.json")
    for seed in cfg.split.seeds:
        if checkpoint_path(cfg, "halo", seed).exists() and not args.force:
            log.info(f"seed {seed}: checkpoint exists, skipping (use --force to retrain)")
            continue
        split = get_user_split(cfg, meta["participants"], seed)
        pretrain_halo(cfg, split, seed, device)

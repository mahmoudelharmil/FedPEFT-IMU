"""
Step 6 - ablation: LoRA rank sweep (PDF p. 11, "Rank and placement r in {2, 4, 8, 16}").

    python scripts/06_run_ablations.py            # encoder from cfg.ablation.encoder

Runs V3 (exact LoRA aggregation) with every rank in cfg.ablation.ranks except the
default rank, whose result is the main V3 run.  Other ablations of the PDF are
covered by the main grid: naive vs exact (V2 vs V3), pretrained vs random
(V0 vs FedAvg-scratch), with vs without personal head + prototypes (V4 vs V5).
"""

from pathlib import Path

import torch
from _common import setup

from fedpeft.data.store import load_experiment_data
from fedpeft.federated.simulation import FederatedRun


def extra(p):
    p.add_argument("--rerun", action="store_true")


if __name__ == "__main__":
    cfg, device, log, args = setup("06_run_ablations", __doc__, extra)
    enc = cfg.ablation.encoder
    for seed in cfg.split.seeds:
        data = load_experiment_data(cfg, seed, device)
        for rank in cfg.ablation.ranks:
            if rank == cfg.model.lora.rank:
                continue
            out = Path(cfg.paths.results_dir) / cfg.profile / "federated" / f"{enc}__v3_lora_exact__r{rank}__seed{seed}.json"
            if out.exists() and not args.rerun:
                log.info(f"skip {out.name} (exists)")
                continue
            FederatedRun(cfg, data, enc, "v3_lora_exact", seed, device, rank_override=rank).run()
            torch.cuda.empty_cache()

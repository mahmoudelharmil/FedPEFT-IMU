"""
Step 4 - PHASE 1: centralized training of the four encoders.

    python scripts/04_run_centralized.py                       # all encoders, all modes
    python scripts/04_run_centralized.py --encoders halo --modes scratch

Modes: scratch (random init = "learning from scratch", the POC) and pretrained_ft.
Results: results/<profile>/centralized/<encoder>__<mode>__seed<seed>.json
"""

from pathlib import Path

import torch
from _common import csv_list, setup

from fedpeft.data.store import load_experiment_data
from fedpeft.training.centralized import run_centralized


def extra(p):
    p.add_argument("--encoders", help="comma separated subset of encoders")
    p.add_argument("--modes", help="comma separated subset of: scratch,pretrained_ft")
    p.add_argument("--rerun", action="store_true", help="re-run even if a result file exists")


if __name__ == "__main__":
    cfg, device, log, args = setup("04_run_centralized", __doc__, extra)
    encoders = csv_list(args.encoders, cfg.encoders.names)
    for seed in cfg.split.seeds:
        data = load_experiment_data(cfg, seed, device)
        for enc in encoders:
            # An encoder may restrict its modes (the MLP baseline has no pretrained weights).
            modes = csv_list(args.modes, cfg.encoders.get(enc, {}).get("modes") or cfg.centralized.modes)
            for mode in modes:
                out = Path(cfg.paths.results_dir) / cfg.profile / "centralized" / f"{enc}__{mode}__seed{seed}.json"
                if out.exists() and not args.rerun:
                    log.info(f"skip {out.name} (exists)")
                    continue
                run_centralized(cfg, data, enc, mode, seed, device)
                torch.cuda.empty_cache()
        del data

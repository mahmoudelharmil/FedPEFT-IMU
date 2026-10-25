"""
Step 5 - PHASE 2: federated learning with every method on every encoder.

    python scripts/05_run_federated.py
    python scripts/05_run_federated.py --encoders ssl_wearables --methods v3_lora_exact,v5_proposed

The target F1 used for "rounds / MB to target" is
    cfg.fl.target_f1_fraction x (centralized from-scratch macro-F1 of the same encoder),
so run step 4 first (the report re-computes it anyway from the saved curves).
Results: results/<profile>/federated/<encoder>__<method>__seed<seed>.json
"""

from pathlib import Path

import torch
from _common import csv_list, setup

from fedpeft.data.store import load_experiment_data
from fedpeft.federated.simulation import FederatedRun
from fedpeft.utils import load_json


def extra(p):
    p.add_argument("--encoders", help="comma separated subset of encoders")
    p.add_argument("--methods", help="comma separated subset of FL methods")
    p.add_argument("--rerun", action="store_true", help="re-run even if a result file exists")
    p.add_argument("--tag", help="ablation tag appended to the result file name (use with config overrides)")


def target_for(cfg, encoder: str, seed: int):
    """90 % (configurable) of the centralized from-scratch macro-F1 of this encoder."""
    path = Path(cfg.paths.results_dir) / cfg.profile / "centralized" / f"{encoder}__scratch__seed{seed}.json"
    if not path.exists():
        return None
    return cfg.fl.target_f1_fraction * load_json(path)["test"]["macro_f1"]


if __name__ == "__main__":
    cfg, device, log, args = setup("05_run_federated", __doc__, extra)
    encoders = csv_list(args.encoders, cfg.encoders.names)
    for seed in cfg.split.seeds:
        data = load_experiment_data(cfg, seed, device)
        for enc in encoders:
            target = target_for(cfg, enc, seed)
            # An encoder may restrict its methods (the MLP baseline: from-scratch methods only).
            restricted = cfg.encoders.get(enc, {}).get("methods")
            methods = csv_list(args.methods, restricted or cfg.fl.methods)
            if restricted:
                methods = [m for m in methods if m in restricted]
            for method in methods:
                tag = f"__{args.tag}" if args.tag else ""
                out = Path(cfg.paths.results_dir) / cfg.profile / "federated" / f"{enc}__{method}{tag}__seed{seed}.json"
                if out.exists() and not args.rerun:
                    log.info(f"skip {out.name} (exists)")
                    continue
                FederatedRun(cfg, data, enc, method, seed, device, target_f1=target, tag=args.tag).run()
                torch.cuda.empty_cache()
        del data

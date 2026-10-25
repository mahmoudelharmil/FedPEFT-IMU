"""
Re-run the held-out (new-user) protocols of a finished federated run WITHOUT re-training it.

Every federated run stores the model a new user receives in
results/<profile>/models/fl__<encoder>__<method>[__tag]__seed<seed>.pt (fl.save_new_user_model).
This script rebuilds the run, loads that model and evaluates the zero-/few-shot protocols again,
for example with the transductive zero-shot rule switched on:

    python scripts/tools/reeval_heldout.py --profile harbox --encoder limu_bert --method v5_proposed \
        split.zero_shot_refine_iters=10

--write stores the new numbers in the run's result JSON under "heldout_<name>" (the original
"heldout" entry is kept); --name chooses <name> (default: "reeval"). --on-pool evaluates on the
pool (validation) users instead, to choose such settings without looking at the test users.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from _common import setup  # noqa: E402

from fedpeft.data.store import load_experiment_data  # noqa: E402
from fedpeft.federated.simulation import FederatedRun  # noqa: E402
from fedpeft.utils import load_json, save_json  # noqa: E402


def extra(p):
    p.add_argument("--encoder", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--tag", help="tag of the run (e.g. plora)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write", action="store_true", help="store the result in the run's JSON")
    p.add_argument("--name", default="reeval", help="the JSON key becomes heldout_<name>")
    p.add_argument("--on-pool", action="store_true",
                   help="evaluate on the POOL (validation) users instead of the held-out test users: "
                        "for choosing settings such as the refinement prior without touching the test users")
    p.add_argument("--zero-only", action="store_true", help="skip the few-shot budgets (faster)")


def summary(h: dict) -> str:
    keys = ["zero_shot"] + [b["key"] for b in h["protocol"]["fewshot"]]
    return "  ".join(f"{k.replace('few_shot_', '')} {h[k]['mean_user_f1']:.3f}" for k in keys)


if __name__ == "__main__":
    cfg, device, log, args = setup("reeval_heldout", __doc__, extra)
    if args.zero_only:
        cfg.split.fewshot_shots, cfg.split.fewshot_minutes = None, []
    data = load_experiment_data(cfg, args.seed, device)
    if args.on_pool:            # the pool users, split like held-out users (adaptation part / test part)
        pool = list(data.split["pool"])
        data.split = dict(data.split, heldout=pool)
        data.heldout_adapt, data.heldout_test = {}, {}
        for user in pool:
            if cfg.split.get("mode", "chronological") == "chronological":
                data.heldout_adapt[user], data.heldout_test[user] = data.store.user_time_split(
                    user, cfg.split.heldout_adapt_frac)
            else:
                data.heldout_adapt[user], data.heldout_test[user] = data.store.user_class_time_split(
                    user, cfg.split.heldout_adapt_frac, int(cfg.split.get("gap_windows", 0)))
        args.write = False      # validation numbers are never stored as results
    run = FederatedRun(cfg, data, args.encoder, args.method, args.seed, device, tag=args.tag)
    run.load_new_user_model()
    heldout = run.evaluate_new_users()
    if heldout is None:
        log.info("this method has no new-user protocol")
        sys.exit(0)
    log.info(f"{args.encoder} {args.method}{run.tag}{' [pool users]' if args.on_pool else ''}: {summary(heldout)}")
    if args.on_pool:
        sys.exit(0)
    out = Path(cfg.paths.results_dir) / cfg.profile / "federated" / \
        f"{args.encoder}__{args.method}{run.tag}__seed{args.seed}.json"
    if out.exists():
        old = load_json(out).get("heldout")
        if old:
            log.info(f"{'stored result':>{len(args.encoder) + len(args.method) + len(run.tag) + 1}}: {summary(old)}")
        if args.write:
            res = load_json(out)
            res[f"heldout_{args.name}"] = heldout
            save_json(res, out)
            log.info(f"written to {out.name} as heldout_{args.name}")

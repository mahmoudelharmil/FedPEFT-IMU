"""
Re-decode the HMM scores of personal-classifier runs (V5) from their saved logits.

Why: the first version of the personal-head HMM divided each client's emissions by the
client's own prior, in which a class the client never had is ~1e-4.  That boosts
those classes by about +9 nats and the HMM then predicts activities the person never
does (UniMTS V5: 0.599 per window, 0.374 after "smoothing").  The rule was changed to
shrink the client prior halfway towards the global prior (fedpeft/federated/simulation.py).
This tool applies the corrected rule to results that were produced before the fix,
using only the saved test logits - no retraining, and the rule was fixed in advance.

    python scripts/tools/redecode_personal_hmm.py --profile standard
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fedpeft.config import load_config  # noqa: E402
from fedpeft.data.store import load_experiment_data  # noqa: E402
from fedpeft.training.hmm import ActivityHMM  # noqa: E402
from fedpeft.training.imbalance import class_prior  # noqa: E402
from fedpeft.training.metrics import confusion, evaluate_predictions  # noqa: E402
from fedpeft.utils import load_json, save_json  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="standard")
    args = ap.parse_args()
    cfg = load_config(args.profile)
    data = load_experiment_data(cfg, 0, torch.device("cpu"))
    store = data.store
    base = Path(cfg.paths.results_dir) / cfg.profile
    users = sorted(data.client_test)
    gprior = class_prior(sum(store.class_counts(i) for i in data.client_train.values()))

    for res_path in sorted((base / "federated").glob("*__v5_proposed__seed*.json")):
        r = load_json(res_path)
        if r["config"].get("hmm_prior_fixed"):
            print(f"skip {res_path.name} (already decoded with the bounded prior)")
            continue
        d = np.load(base / "logits" / ("fl__" + res_path.stem + ".npz"))
        logits = torch.as_tensor(d["test_logits"].astype(np.float32))
        hmm = ActivityHMM(data.n_classes, smoothing=0.0)
        hmm.add_counts(d["hmm_trans_counts"], d["hmm_init_counts"])
        y_true, y_pred, pos = {}, {}, 0
        for u in users:
            idx = data.client_test[u]
            lg = logits[pos:pos + len(idx)]
            pos += len(idx)
            mix = torch.log(0.5 * class_prior(store.class_counts(data.client_train[u])) + 0.5 * gprior)
            y_true[u] = store.y[idx].numpy()
            y_pred[u] = hmm.smooth(lg, store.sequence_breaks(idx), mix, tau_train=0.0)
        assert pos == len(logits), "saved logits do not match the client test sets"
        old = r["final_hmm"]["macro_f1"] if r.get("final_hmm") else None
        r["final_hmm"] = evaluate_predictions(y_true, y_pred, data.n_classes)
        r["confusion_hmm"] = confusion(y_true, y_pred, data.n_classes)
        r["config"]["hmm_prior_fixed"] = True
        save_json(r, res_path)
        print(f"{res_path.name}: HMM macro-F1 {old} -> {r['final_hmm']['macro_f1']:.4f} "
              f"(per window {r['final']['macro_f1']:.4f})")

"""
User-level split (PDF page 10, step 1 + appendix "held-out users explained").

Every participant is assigned to exactly ONE of three groups:

    clients   (~70 %)  take part in federated training (and are pooled for the
                       centralized baseline); each is later split by time.
    heldout   ( 20 %)  never seen during training; used only for zero-/few-shot tests.
    pool      ( 10 %)  unlabelled self-supervised pretraining data for HALO and
                       validation users for early stopping / learning-rate checks.

The split is fixed by a seed and written to results/splits/ so that the exact
user IDs can be published (as the PDF recommends).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils import load_json, save_json


def make_user_split(users: list[str], seed: int, heldout_frac: float, pool_frac: float,
                    heldout_users: int | None = None) -> dict:
    """Randomly (but reproducibly) assign users to clients / heldout / pool.

    heldout_users, if given, is an absolute number of held-out users (e.g. FedAIoT's 6
    test subjects of WISDM) and replaces heldout_frac."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(sorted(users)).tolist()
    n_heldout = int(heldout_users) if heldout_users else int(round(heldout_frac * len(order)))
    n_pool = int(round(pool_frac * len(order)))
    return {
        "seed": seed,
        "heldout": sorted(order[:n_heldout]),
        "pool": sorted(order[n_heldout:n_heldout + n_pool]),
        "clients": sorted(order[n_heldout + n_pool:]),
    }


def split_path(cfg, seed: int) -> Path:
    """results/splits/<file>.json - one file per dataset and group fractions.

    Capture-24 keeps its original name (user_split_seed0.json), so published
    results still point to the right file.
    """
    s = cfg.split
    if cfg.data.get("dataset", "capture24") == "capture24":
        name = f"user_split_seed{seed}.json"
    else:
        held = f"{int(s.heldout_users)}" if s.get("heldout_users") else f"{s.heldout_frac:g}"
        name = f"{cfg.data.dataset}_heldout{held}_pool{s.pool_frac:g}_seed{seed}.json"
    return Path(cfg.paths.results_dir) / "splits" / name


def get_user_split(cfg, users: list[str], seed: int) -> dict:
    """Load the split for `seed` from disk, or create and save it."""
    path = split_path(cfg, seed)
    if path.exists():
        split = load_json(path)
        if set(split["clients"] + split["heldout"] + split["pool"]) == set(users):
            return split
    split = make_user_split(users, seed, cfg.split.heldout_frac, cfg.split.pool_frac,
                            cfg.split.get("heldout_users"))
    save_json(split, path)
    return split


def time_split(n: int, first_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Step 2: chronological split of one user's n windows (never random by window).

    Returns (first part indices, last part indices).  Windows are stored in time
    order, so the first `first_frac` of them are the earliest recordings.
    """
    cut = int(np.floor(first_frac * n))
    idx = np.arange(n)
    return idx[:cut], idx[cut:]

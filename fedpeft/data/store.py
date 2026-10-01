"""
In-memory window store + the experiment data bundle.

All windows of all participants of one experiment are loaded ONCE into a single
tensor (on the GPU when available, in float16):

    store.X  : (N, 3, 300)  canonical windows, 30 Hz, 10 s, acceleration in g
    store.y  : (N,)         integer class labels
    store.slices[user] = (start, end)   contiguous, chronologically ordered rows

Everything else (client train/test parts, held-out adaptation/test parts, the
validation pool) is just an index tensor into this store.  This makes the FL
simulation fast (no DataLoader, no copies) and easy to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..utils import get_logger, load_json
from .splits import get_user_split, time_split

log = get_logger("data")


class WindowStore:
    """Holds the (sub-sampled) windows of a list of users as one big tensor."""

    def __init__(self, processed_dir: str | Path, users: list[str], scheme: str,
                 subsample_every: int, device: torch.device):
        processed_dir = Path(processed_dir)
        meta = load_json(processed_dir / "meta.json")
        self.class_names: list[str] = meta["schemes"][scheme]
        self.window_s: int = meta["window_s"]
        self.base_rate_hz: int = meta["base_rate_hz"]
        self.slices: dict[str, tuple[int, int]] = {}

        xs, ys, ts, pos = [], [], [], 0
        for user in users:
            with np.load(processed_dir / f"{user}.npz") as d:
                # Keep 1 of every `subsample_every` windows (keeps the 24-h class mix, cuts compute).
                X = d["X"][::subsample_every]
                y = d[f"y_{scheme}"][::subsample_every].astype(np.int64)
                t = d["t"][::subsample_every].astype(np.int64)             # window start, ms since epoch
            keep = y >= 0
            X, y, t = X[keep], y[keep], t[keep]
            self.slices[user] = (pos, pos + len(y))
            pos += len(y)
            xs.append(X)
            ys.append(y)
            ts.append(t)

        self.X = torch.from_numpy(np.concatenate(xs)).to(device)             # float16
        self.y = torch.from_numpy(np.concatenate(ys)).to(device)             # int64
        self.t_ms = np.concatenate(ts)                                        # CPU: only used for HMM sequences
        self.spacing_ms = subsample_every * self.window_s * 1000              # nominal gap between kept windows
        self.device = device
        log.info(f"WindowStore: {len(users)} users, {len(self.y):,} windows, "
                 f"{self.X.numel() * 2 / 1e6:.0f} MB on {device}")

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def sequence_breaks(self, idx: torch.Tensor, tolerance: float = 3.0) -> np.ndarray:
        """Positions where a chronological index list has a time gap (for HMM smoothing).

        Returns a boolean array: True at position i means "window i starts a new
        sequence" (first window, or more than `tolerance` x the nominal spacing
        after the previous one - e.g. the device was off or the time was unlabelled).
        """
        t = self.t_ms[idx.cpu().numpy()]
        breaks = np.ones(len(t), dtype=bool)
        if len(t) > 1:
            breaks[1:] = np.diff(t) > tolerance * self.spacing_ms
        return breaks

    def class_counts(self, idx: torch.Tensor) -> np.ndarray:
        """Number of windows per class among `idx`."""
        return np.bincount(self.y[idx].cpu().numpy(), minlength=self.n_classes)

    def user_index(self, user: str) -> torch.Tensor:
        """All row indices of one user, in chronological order."""
        start, end = self.slices[user]
        return torch.arange(start, end, device=self.device)

    def user_time_split(self, user: str, first_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Chronological split of one user's rows: (first part, last part)."""
        start, end = self.slices[user]
        first, last = time_split(end - start, first_frac)
        return (torch.as_tensor(first + start, device=self.device),
                torch.as_tensor(last + start, device=self.device))


@dataclass
class ExperimentData:
    """Everything an experiment needs to know about the data, as index tensors."""

    store: WindowStore
    split: dict
    client_train: dict[str, torch.Tensor] = field(default_factory=dict)   # first 80 % of each client
    client_test: dict[str, torch.Tensor] = field(default_factory=dict)    # last 20 % of each client
    heldout_adapt: dict[str, torch.Tensor] = field(default_factory=dict)  # held-out: few-shot label source
    heldout_test: dict[str, torch.Tensor] = field(default_factory=dict)   # held-out: evaluation part
    pool_index: torch.Tensor | None = None                                # validation / SSL users

    @property
    def clients(self) -> list[str]:
        return self.split["clients"]

    @property
    def heldout(self) -> list[str]:
        return self.split["heldout"]

    @property
    def n_classes(self) -> int:
        return self.store.n_classes

    def cat(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate the index tensors of several users."""
        return torch.cat([parts[u] for u in sorted(parts)])

    def class_distribution(self, idx: torch.Tensor) -> np.ndarray:
        return np.bincount(self.store.y[idx].cpu().numpy(), minlength=self.n_classes)


def load_experiment_data(cfg, seed: int, device: torch.device) -> ExperimentData:
    """Build the user split for `seed` and the index tensors of every part."""
    meta = load_json(Path(cfg.paths.processed_dir) / "meta.json")
    split = get_user_split(cfg, meta["participants"], seed)
    users = split["clients"] + split["heldout"] + split["pool"]
    store = WindowStore(cfg.paths.processed_dir, users, cfg.data.label_scheme,
                        cfg.data.subsample_every, device)

    data = ExperimentData(store=store, split=split)
    for user in split["clients"]:
        data.client_train[user], data.client_test[user] = store.user_time_split(user, cfg.split.train_time_frac)
    for user in split["heldout"]:
        data.heldout_adapt[user], data.heldout_test[user] = store.user_time_split(user, cfg.split.heldout_adapt_frac)
    data.pool_index = torch.cat([store.user_index(u) for u in split["pool"]])
    log.info(f"Split seed {seed}: {len(split['clients'])} clients, {len(split['heldout'])} held-out, "
             f"{len(split['pool'])} pool users")
    return data


def sample_fewshot(y: np.ndarray, minutes: float, window_s: int, rng: np.random.Generator) -> np.ndarray:
    """Pick `minutes` of labelled data per activity (at most) from the positions in `y`.

    Example: 1 minute with 10-s windows = up to 6 windows of every class the user has.
    Returns positions into `y`.
    """
    per_class = max(1, int(round(minutes * 60 / window_s)))
    chosen = []
    for c in np.unique(y):
        pos = np.flatnonzero(y == c)
        chosen.append(rng.choice(pos, size=min(per_class, len(pos)), replace=False))
    return np.sort(np.concatenate(chosen))

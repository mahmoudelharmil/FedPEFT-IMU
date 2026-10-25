"""
In-memory window store + the experiment data bundle.

All windows of all participants of one experiment are loaded ONCE into a single
tensor (on the GPU when available, in float16):

    store.X  : (N, C, L)    canonical windows of the dataset
                            Capture-24: (N, 3, 300) = 10 s at 30 Hz, acceleration in g
                            HARBox:     (N, 9, 100) =  2 s at 50 Hz, acc m/s^2 + gyro rad/s + mag uT
    store.y  : (N,)         integer class labels
    store.slices[user] = (start, end)   contiguous, chronologically ordered rows

Everything else (client train/test parts, held-out adaptation/test parts, the
validation pool) is just an index tensor into this store.  This makes the FL
simulation fast (no DataLoader, no copies) and easy to reason about.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..utils import get_logger, load_json
from .splits import get_user_split, time_split

log = get_logger("data")


@dataclass(frozen=True)
class DataSpec:
    """What one stored window looks like: rate, length and the meaning/unit of every channel.

    Capture-24's meta.json predates these keys, so the defaults describe it:
    3 accelerometer channels in g.
    """
    rate_hz: float
    window_s: float
    channels: tuple[str, ...]                # e.g. ("acc_x", "acc_y", "acc_z", "gyro_x", ...)
    groups: tuple[tuple[str, tuple[int, ...]], ...]   # (("acc", (0, 1, 2)), ("gyro", (3, 4, 5)), ...)
    units: tuple[tuple[str, str], ...]       # (("acc", "m/s2"), ("gyro", "rad/s"), ...)
    hop_s: float

    @classmethod
    def from_meta(cls, meta: dict) -> "DataSpec":
        channels = tuple(meta.get("channels", ["acc_x", "acc_y", "acc_z"]))
        groups = meta.get("channel_groups", {"acc": [0, 1, 2]})
        units = meta.get("channel_units", {"acc": "g"})
        return cls(rate_hz=float(meta["base_rate_hz"]), window_s=float(meta["window_s"]), channels=channels,
                   groups=tuple((g, tuple(ix)) for g, ix in groups.items()),
                   units=tuple(units.items()), hop_s=float(meta.get("hop_s", meta["window_s"])))

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_samples(self) -> int:
        return int(round(self.rate_hz * self.window_s))

    def unit(self, group: str) -> str | None:
        return dict(self.units).get(group)


def load_data_spec(processed_dir: str | Path) -> DataSpec:
    return DataSpec.from_meta(load_json(Path(processed_dir) / "meta.json"))


class WindowStore:
    """Holds the (sub-sampled) windows of a list of users as one big tensor."""

    def __init__(self, processed_dir: str | Path, users: list[str], scheme: str,
                 subsample_every: int, device: torch.device):
        processed_dir = Path(processed_dir)
        meta = load_json(processed_dir / "meta.json")
        self.spec = DataSpec.from_meta(meta)
        self.dataset = meta.get("dataset", "capture24")
        self.class_names: list[str] = meta["schemes"][scheme]
        self.window_s: float = meta["window_s"]
        self.base_rate_hz: int = meta["base_rate_hz"]
        self.slices: dict[str, tuple[int, int]] = {}

        xs, ys, ts, segs, pos = [], [], [], [], 0
        for user in users:
            with np.load(processed_dir / f"{user}.npz") as d:
                # Keep 1 of every `subsample_every` windows (keeps the 24-h class mix, cuts compute).
                X = d["X"][::subsample_every]
                y = d[f"y_{scheme}"][::subsample_every].astype(np.int64)
                t = d["t"][::subsample_every].astype(np.int64)             # window start, ms since epoch
                seg = d["seg"][::subsample_every] if "seg" in d else np.zeros(len(y), dtype=np.int32)
            keep = y >= 0
            X, y, t, seg = X[keep], y[keep], t[keep], seg[keep]
            self.slices[user] = (pos, pos + len(y))
            pos += len(y)
            xs.append(X)
            ys.append(y)
            ts.append(t)
            segs.append(seg)

        self.X = torch.from_numpy(np.concatenate(xs)).to(device)             # float16
        self.y = torch.from_numpy(np.concatenate(ys)).to(device)             # int64
        self.t_ms = np.concatenate(ts)                                        # CPU: only used for HMM sequences
        self.seg = np.concatenate(segs)                                       # recording session (HARBox)
        self.spacing_ms = subsample_every * self.spec.hop_s * 1000            # nominal gap between kept windows
        self.device = device
        log.info(f"WindowStore: {len(users)} users, {len(self.y):,} windows of {tuple(self.X.shape[1:])}, "
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

    def _index(self, rows: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.sort(rows), dtype=torch.long, device=self.device)

    def user_time_split(self, user: str, first_frac: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Chronological split of one user's rows: (first part, last part)."""
        start, end = self.slices[user]
        first, last = time_split(end - start, first_frac)
        return (torch.as_tensor(first + start, device=self.device),
                torch.as_tensor(last + start, device=self.device))

    def user_class_time_split(self, user: str, first_frac: float, gap: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Chronological split INSIDE every activity of one user, with `gap` windows dropped in between.

        HARBox records every activity in its own session, so a split of the whole
        recording would put entire activities into the test part.  Splitting each
        activity in time keeps every activity on both sides, and the gap removes the
        window that overlaps both parts (HARBox windows overlap by 50 %).
        """
        start, end = self.slices[user]
        rows = np.arange(start, end)
        y = self.y[start:end].cpu().numpy()
        first, last = [], []
        for c in np.unique(y):
            rc = rows[y == c]                       # chronological (rows are time-sorted)
            cut = int(np.floor(first_frac * len(rc)))
            first.append(rc[:cut])
            last.append(rc[cut + gap:])
        return self._index(np.concatenate(first)), self._index(np.concatenate(last))

    def user_random_split(self, user: str, rng: np.random.Generator, n_range: list[int] | None,
                          fracs: list[float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random (train, validation, test) windows of one user - the published HARBox protocol.

        n_range = [min, max]: the user first keeps a random number of windows in that
        range (FedCHAR: 50-500), then splits them at random by `fracs`.
        """
        start, end = self.slices[user]
        rows = np.arange(start, end)
        if n_range:
            k = int(rng.integers(n_range[0], n_range[1] + 1))
            rows = rng.choice(rows, size=min(k, len(rows)), replace=False)
        rows = rng.permutation(rows)
        n_tr = int(round(fracs[0] * len(rows)))
        n_va = int(round(fracs[1] * len(rows)))
        return self._index(rows[:n_tr]), self._index(rows[n_tr:n_tr + n_va]), self._index(rows[n_tr + n_va:])


@dataclass
class ExperimentData:
    """Everything an experiment needs to know about the data, as index tensors."""

    store: WindowStore
    split: dict
    client_train: dict[str, torch.Tensor] = field(default_factory=dict)   # first 80 % of each client
    client_test: dict[str, torch.Tensor] = field(default_factory=dict)    # last 20 % of each client
    client_val: dict[str, torch.Tensor] = field(default_factory=dict)     # random protocol only
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


def _user_rng(seed: int, user: str) -> np.random.Generator:
    """A reproducible random stream per (seed, user) - independent of the user order."""
    return np.random.default_rng([seed, zlib.crc32(user.encode())])


def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float, rng: np.random.Generator,
                        min_size: int = 10, max_tries: int = 1000) -> list[np.ndarray]:
    """Split positions 0..len(y)-1 into `n_clients` parts with Dirichlet(alpha) label skew.

    The usual scheme of label-skewed FL benchmarks (Hsu et al., 2019; FedAIoT "scheme #1"):
    for every class, the share of each client is drawn from Dir(alpha * 1); small alpha gives
    clients dominated by few classes and very different sizes.  Draws again until every
    client holds at least `min_size` windows.
    """
    classes = np.unique(y)
    for _ in range(max_tries):
        parts = [[] for _ in range(n_clients)]
        for c in classes:
            pos = rng.permutation(np.flatnonzero(y == c))
            share = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(share)[:-1] * len(pos)).astype(int)
            for k, chunk in enumerate(np.split(pos, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.array(p, dtype=np.int64)) for p in parts]
    raise RuntimeError(f"no Dirichlet partition with >= {min_size} windows per client after {max_tries} draws")


def load_experiment_data(cfg, seed: int, device: torch.device) -> ExperimentData:
    """Build the user split for `seed` and the index tensors of every part."""
    meta = load_json(Path(cfg.paths.processed_dir) / "meta.json")
    if cfg.split.get("official_split"):
        # the dataset's own subject split (UCI HAR: 21 training / 9 test subjects), as in published results
        off = meta["official_split"]
        split = {"seed": seed, "clients": list(off["train"]), "heldout": list(off["test"]), "pool": []}
    else:
        split = get_user_split(cfg, meta["participants"], seed)
    users = split["clients"] + split["heldout"] + split["pool"]
    store = WindowStore(cfg.paths.processed_dir, users, cfg.data.label_scheme,
                        cfg.data.subsample_every, device)

    data = ExperimentData(store=store, split=split)
    s = cfg.split
    mode = s.get("mode", "chronological")
    gap = int(s.get("gap_windows", 0) or 0)
    if mode == "dirichlet_per_user":
        # Every training subject's windows are dealt to `dirichlet_clients_per_user` clients with
        # Dirichlet label skew (the FedMultimodal protocol of UCI HAR: 21 subjects -> 105 clients).
        subjects = split["clients"]
        rng = np.random.default_rng(seed + 7000)
        fracs = list(s.get("dirichlet_fracs", [0.8, 0.1, 0.1]))
        names = []
        for subject in subjects:
            rows = store.user_index(subject)
            parts = dirichlet_partition(store.y[rows].cpu().numpy(), int(s.dirichlet_clients_per_user),
                                        float(s.dirichlet_alpha), rng, min_size=int(s.get("dirichlet_min_size", 5)))
            for k, pos in enumerate(parts):
                name = f"{subject}_c{k}"
                idx = rows[torch.as_tensor(rng.permutation(pos), device=rows.device)]
                # small clients (strong skew) still keep one validation and one test window
                n_va = max(1, int(round(fracs[1] * len(idx))))
                n_tr = len(idx) - n_va - max(1, int(round(fracs[2] * len(idx))))
                data.client_train[name] = idx[:n_tr].sort().values
                data.client_val[name] = idx[n_tr:n_tr + n_va].sort().values
                data.client_test[name] = idx[n_tr + n_va:].sort().values
                names.append(name)
        data.split = split = dict(split, clients=names, subjects=subjects)
    if mode == "dirichlet":
        # Clients are NOT users: all windows of the training subjects are pooled and dealt to
        # `dirichlet_clients` clients with Dirichlet label skew (the FedAIoT protocol).
        subjects = split["clients"]
        rows = torch.cat([store.user_index(u) for u in subjects])
        rng = np.random.default_rng(seed + 7000)
        parts = dirichlet_partition(store.y[rows].cpu().numpy(), int(s.dirichlet_clients),
                                    float(s.dirichlet_alpha), rng)
        fracs = list(s.get("dirichlet_fracs", [0.8, 0.1, 0.1]))
        names = [f"c{k:02d}" for k in range(len(parts))]
        for name, pos in zip(names, parts):
            idx = rows[torch.as_tensor(rng.permutation(pos), device=rows.device)]
            n_tr, n_va = int(round(fracs[0] * len(idx))), int(round(fracs[1] * len(idx)))
            data.client_train[name] = idx[:n_tr].sort().values
            data.client_val[name] = idx[n_tr:n_tr + n_va].sort().values
            data.client_test[name] = idx[n_tr + n_va:].sort().values
        data.split = split = dict(split, clients=names, subjects=subjects)
    for user in (split["clients"] if mode not in ("dirichlet", "dirichlet_per_user") else []):
        if mode == "chronological":
            data.client_train[user], data.client_test[user] = store.user_time_split(user, s.train_time_frac)
        elif mode == "per_class_chronological":
            data.client_train[user], data.client_test[user] = store.user_class_time_split(user, s.train_time_frac, gap)
        elif mode == "random":
            (data.client_train[user], data.client_val[user],
             data.client_test[user]) = store.user_random_split(user, _user_rng(seed, user),
                                                               s.get("random_windows_per_user"), list(s.random_fracs))
        else:
            raise ValueError(f"unknown split.mode '{mode}'")
    for user in split["heldout"]:
        if mode == "chronological":
            data.heldout_adapt[user], data.heldout_test[user] = store.user_time_split(user, s.heldout_adapt_frac)
        else:   # every activity on both sides (the few-shot labels need every activity)
            data.heldout_adapt[user], data.heldout_test[user] = store.user_class_time_split(
                user, s.heldout_adapt_frac, gap)
    if split["pool"]:
        data.pool_index = torch.cat([store.user_index(u) for u in split["pool"]])
    elif data.client_val:
        # No pool users (random protocol): the clients' validation windows select models.
        data.pool_index = data.cat(data.client_val)
    else:
        raise ValueError("no validation data: set split.pool_frac > 0 or use split.mode=random")
    n_tr = sum(len(v) for v in data.client_train.values())
    n_te = sum(len(v) for v in data.client_test.values())
    log.info(f"Split seed {seed} ({mode}): {len(split['clients'])} clients ({n_tr:,} train / {n_te:,} test "
             f"windows), {len(split['heldout'])} held-out, {len(split['pool'])} pool users, "
             f"{len(data.pool_index):,} validation windows")
    return data


def sample_fewshot(y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Pick `per_class` labelled windows of every activity (at most) from the positions in `y`.

    Returns positions into `y`.
    """
    chosen = []
    for c in np.unique(y):
        pos = np.flatnonzero(y == c)
        chosen.append(rng.choice(pos, size=min(per_class, len(pos)), replace=False))
    return np.sort(np.concatenate(chosen))


def fewshot_budgets(cfg, window_s: float) -> list[tuple[str, int, str]]:
    """The few-shot budgets of the experiment as (result key, windows per class, label).

    Capture-24 expresses them in minutes of labels (1 min = 6 windows of 10 s),
    HARBox in labelled windows ("shots", 2 s each).
    """
    shots = cfg.split.get("fewshot_shots")
    if shots is not None:                   # [] = no few-shot protocol (e.g. FedAIoT's test subjects)
        return [(f"few_shot_{k}shot", int(k), f"{k} window{'s' if k > 1 else ''} ({k * window_s:g} s)")
                for k in shots]
    return [(f"few_shot_{m}min", max(1, int(round(m * 60 / window_s))), f"{m} min")
            for m in cfg.split.fewshot_minutes]

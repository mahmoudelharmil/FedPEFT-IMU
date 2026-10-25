"""
UCI HAR: download, unzip and convert to per-subject window files.

"Human Activity Recognition Using Smartphones" (Anguita et al., ESANN 2013; UCI #240):
30 subjects wore a Samsung Galaxy S II on the waist and performed six activities twice in a
fixed sequence (standing, sitting, lying, walking, walking downstairs, walking upstairs).
The phone's accelerometer and gyroscope were sampled at 50 Hz and released as 2.56-s
windows (128 samples) with 50 % overlap, already split by subject into a training set
(21 subjects) and a test set (9 subjects).

Raw format ("Inertial Signals", one row = one window, 128 values):
    total_acc_{x,y,z}  accelerometer incl. gravity, in g      -> our acc channels
    body_gyro_{x,y,z}  gyroscope, in rad/s                    -> our gyro channels
    (body_acc = total_acc minus a low-pass gravity estimate: not used, the encoders
     expect raw acceleration with gravity)
    subject_*.txt, y_*.txt: subject id (1-30) and activity (1-6) of every window

The windows of a subject are stored in recording order. Where the second half of a window
equals the first half of the next one, the two belong to the same continuous recording; a
mismatch (or a new activity) starts a new recording ("seg"), which the per-activity
chronological split and the gap between train and test parts rely on.

What this module writes (data/processed_ucihar/):
    s01.npz    X: (N, 6, 128) float16 (raw units)   y_UCIHAR: (N,) int8
               t: (N,) int64 window start (ms, synthetic: 1.28 s per hop, 60 s per break)
               seg: (N,) int32 continuous recording
    meta.json  classes, sampling rate, channel names/groups/units, participants and the
               official train/test subjects (used by the published protocol)
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from ..utils import download_file, get_logger, save_json

UCIHAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
CLASS_NAMES = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Lying"]   # activity ids 1-6
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
CHANNEL_GROUPS = {"acc": [0, 1, 2], "gyro": [3, 4, 5]}
CHANNEL_UNITS = {"acc": "g", "gyro": "rad/s"}
RATE_HZ, WINDOW_SAMPLES, HOP_SAMPLES = 50, 128, 64
FILES = ["total_acc_x", "total_acc_y", "total_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"]
log = get_logger("ucihar")


def download_and_unzip_ucihar(raw_dir: str | Path) -> Path:
    """Fetch the UCI archive (once) and extract the text files. Returns 'UCI HAR Dataset/'."""
    raw_dir = Path(raw_dir) / "ucihar"
    data_dir = raw_dir / "UCI HAR Dataset"
    if (data_dir / "train" / "Inertial Signals").exists():
        return data_dir
    outer = raw_dir / "uci-har-outer.zip"
    if not outer.exists():
        log.info("Downloading UCI HAR (~60 MB) from the UCI repository ...")
        download_file(UCIHAR_URL, outer, desc="UCI HAR")
    inner = raw_dir / "UCI HAR Dataset.zip"
    if not inner.exists():
        with zipfile.ZipFile(outer) as zf:
            zf.extract("UCI HAR Dataset.zip", raw_dir)
    with zipfile.ZipFile(inner) as zf:
        zf.extractall(raw_dir, members=[m for m in zf.namelist() if m.endswith(".txt") and not m.startswith("__MACOSX")])
    return data_dir


def load_part(data_dir: Path, part: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X (N, 6, 128) in raw units, activity id 0-5, subject id 1-30 of the 'train' or 'test' part."""
    sig = data_dir / part / "Inertial Signals"
    X = np.stack([np.loadtxt(sig / f"{name}_{part}.txt", dtype=np.float32) for name in FILES], axis=1)
    y = np.loadtxt(data_dir / part / f"y_{part}.txt", dtype=np.int64) - 1
    s = np.loadtxt(data_dir / part / f"subject_{part}.txt", dtype=np.int64)
    return X, y, s


def subject_arrays(X: np.ndarray, y: np.ndarray) -> dict:
    """Windows of one subject (in recording order) with synthetic times and recording ids."""
    half = WINDOW_SAMPLES - HOP_SAMPLES
    same_signal = np.abs(X[:-1, :, HOP_SAMPLES:] - X[1:, :, :half]).max(axis=(1, 2)) < 1e-4
    continuous = same_signal & (y[:-1] == y[1:])
    seg = np.concatenate([[0], np.cumsum(~continuous)]).astype(np.int32)
    step_ms = np.where(continuous, int(HOP_SAMPLES / RATE_HZ * 1000), 60_000)     # unknown pauses: 60 s
    t = np.concatenate([[0], np.cumsum(step_ms)]).astype(np.int64)
    return {"X": X.astype(np.float16), "y_UCIHAR": y.astype(np.int8), "t": t, "seg": seg}


def prepare_ucihar(cfg) -> dict:
    """Download (if needed) and convert every subject. Writes one npz per subject + meta.json."""
    data_dir = download_and_unzip_ucihar(cfg.paths.raw_dir)
    out_dir = Path(cfg.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    official = {}
    participants, info = [], []
    for part in ("train", "test"):
        X, y, s = load_part(data_dir, part)
        official[part] = [f"s{u:02d}" for u in sorted(np.unique(s))]
        for u in sorted(np.unique(s)):
            pid = f"s{u:02d}"
            rows = np.flatnonzero(s == u)                       # recording order is the file order
            arrays = subject_arrays(X[rows], y[rows])
            np.savez(out_dir / f"{pid}.npz", **arrays)
            participants.append(pid)
            info.append({"pid": pid, "n_windows": int(len(rows)), "official": part,
                         "class_counts": np.bincount(arrays["y_UCIHAR"], minlength=len(CLASS_NAMES)).tolist(),
                         "recordings": int(arrays["seg"].max()) + 1})
    meta = {
        "dataset": "ucihar",
        "display_name": "UCI HAR",
        "default_scheme": "UCIHAR",
        "schemes": {"UCIHAR": CLASS_NAMES},
        "base_rate_hz": RATE_HZ,
        "window_s": WINDOW_SAMPLES / RATE_HZ,
        "hop_s": HOP_SAMPLES / RATE_HZ,
        "channels": CHANNELS,
        "channel_groups": CHANNEL_GROUPS,
        "channel_units": CHANNEL_UNITS,
        "participants": sorted(participants),
        "participant_info": sorted(info, key=lambda i: i["pid"]),
        "official_split": {"train": official["train"], "test": official["test"]},
    }
    save_json(meta, out_dir / "meta.json")
    log.info(f"{len(participants)} subjects, {sum(i['n_windows'] for i in info):,} windows -> {out_dir}")
    return meta

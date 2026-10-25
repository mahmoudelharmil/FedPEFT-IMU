"""
HARBox: download, unzip and convert to per-user window files.

HARBox (Ouyang et al., MobiSys 2021, "ClusterFL") was collected with an Android
app in a crowdsourcing manner: 121 users, 77 smartphone models, five activities
(phone call, hopping, typing, walking, waving), 9-axis IMU at 50 Hz.

Raw format (one file per user and activity, e.g. large_scale_HARBox/25/Walk_train.txt):
    every row  = timestamp_ms, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z
    100 rows   = one 2-s window;  units: acc m/s^2, gyro rad/s, mag uT
    consecutive windows of one recording session start 1 s apart, i.e. they
    OVERLAP BY 50 % (window i+1 repeats the second half of window i).

That overlap matters for evaluation: the authors' loader (data_pre.py) splits the
windows of each user at random, so most test windows share half their samples with
a training window.  fedpeft/data/splits.py therefore offers a leakage-free
protocol as well (per-activity chronological split with a one-window gap).

What this module writes (data/processed_harbox/):
    u025.npz   X: (N, 9, 100) float16 (raw units)   y_HARBox: (N,) int8
               t: (N,) int64 window start (ms)      seg: (N,) int32 recording session
    meta.json  classes, sampling rate, channel names/groups/units, participants
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from ..utils import download_file, get_logger, save_json

# The authors' Dropbox folder (linked from github.com/xmouyang/FL-Datasets-for-HAR);
# dl=1 returns the whole folder as FL-Datasets-for-HAR.zip (~175 MB, four datasets).
HARBOX_URL = ("https://www.dropbox.com/scl/fo/l6qqv2nki5l01cx0idvod/APb1gLUoNGE5xLFFGqcRRPc"
              "?rlkey=d5787orlidao7sjvthk9cquv8&dl=1")
FILE_CLASSES = ["Call", "Hop", "typing", "Walk", "Wave"]      # file names (the authors' class order)
CLASS_NAMES = ["Call", "Hop", "Typing", "Walk", "Wave"]        # display names, same order / ids
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z"]
CHANNEL_GROUPS = {"acc": [0, 1, 2], "gyro": [3, 4, 5], "mag": [6, 7, 8]}
CHANNEL_UNITS = {"acc": "m/s2", "gyro": "rad/s", "mag": "uT"}
RATE_HZ, WINDOW_SAMPLES, HOP_S = 50, 100, 1.0
log = get_logger("harbox")


# ----------------------------------------------------------------------------
# 1. Download + unzip
# ----------------------------------------------------------------------------
def download_and_unzip_harbox(raw_dir: str | Path) -> Path:
    """Fetch the authors' archive (once) and extract large_scale_HARBox/. Returns that folder."""
    raw_dir = Path(raw_dir) / "harbox"
    data_dir = raw_dir / "large_scale_HARBox"
    if data_dir.exists() and any(data_dir.iterdir()):
        return data_dir
    outer = raw_dir / "FL-Datasets-for-HAR.zip"
    if not outer.exists():
        log.info("Downloading the HARBox archive (~175 MB) from the authors' Dropbox ...")
        download_file(HARBOX_URL, outer, desc="FL-Datasets-for-HAR.zip")
    inner = raw_dir / "large_scale_HARBox.zip"
    if not inner.exists():
        with zipfile.ZipFile(outer) as zf:
            zf.extract("large_scale_HARBox.zip", raw_dir)
    with zipfile.ZipFile(inner) as zf:                      # skip the macOS resource forks
        zf.extractall(raw_dir, members=[m for m in zf.namelist() if not m.startswith("__MACOSX")])
    return data_dir


# ----------------------------------------------------------------------------
# 2. One user: text files -> windows
# ----------------------------------------------------------------------------
def load_user(user_dir: Path) -> dict | None:
    """All windows of one user, sorted by start time. None if the user has no data."""
    xs, ys, ts, segs = [], [], [], []
    next_seg = 0
    for class_id, name in enumerate(FILE_CLASSES):
        path = user_dir / f"{name}_train.txt"
        if not path.exists():
            continue
        rows = np.loadtxt(path, ndmin=2)
        if rows.size == 0:
            continue
        if rows.shape[1] != 10 or rows.shape[0] % WINDOW_SAMPLES:
            raise ValueError(f"{path}: unexpected shape {rows.shape}")
        w = rows.reshape(-1, WINDOW_SAMPLES, 10)
        start = w[:, 0, 0].astype(np.int64)
        # A new recording session starts where two windows are more than 1.5 hops apart.
        new_seg = np.ones(len(start), dtype=bool)
        new_seg[1:] = np.diff(start) > 1.5 * HOP_S * 1000
        seg = next_seg + np.cumsum(new_seg) - 1
        next_seg = int(seg.max()) + 1
        xs.append(w[:, :, 1:].transpose(0, 2, 1))           # (n, 9, 100), channels first
        ys.append(np.full(len(w), class_id, dtype=np.int8))
        ts.append(start)
        segs.append(seg.astype(np.int32))
    if not xs:
        return None
    X, y, t, seg = (np.concatenate(a) for a in (xs, ys, ts, segs))
    order = np.argsort(t, kind="stable")                    # chronological
    return {"X": X[order].astype(np.float16), "y_HARBox": y[order], "t": t[order], "seg": seg[order]}


# ----------------------------------------------------------------------------
# 3. All users
# ----------------------------------------------------------------------------
def prepare_harbox(cfg) -> dict:
    """Download (if needed) and convert every user. Writes one npz per user + meta.json."""
    data_dir = download_and_unzip_harbox(cfg.paths.raw_dir)
    out_dir = Path(cfg.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    users = sorted((int(p.name) for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit()))
    participants, info = [], []
    for uid in users:
        pid = f"u{uid:03d}"
        out = out_dir / f"{pid}.npz"
        if out.exists():
            with np.load(out) as d:
                y = d["y_HARBox"]
        else:
            arrays = load_user(data_dir / str(uid))
            if arrays is None:
                log.info(f"{pid}: no data, skipped")
                continue
            np.savez(out, **arrays)
            y = arrays["y_HARBox"]
        participants.append(pid)
        info.append({"pid": pid, "n_windows": int(len(y)),
                     "class_counts": np.bincount(y, minlength=len(CLASS_NAMES)).tolist()})
    meta = {
        "dataset": "harbox",
        "display_name": "HARBox",
        "default_scheme": "HARBox",
        "schemes": {"HARBox": CLASS_NAMES},
        "base_rate_hz": RATE_HZ,
        "window_s": WINDOW_SAMPLES / RATE_HZ,
        "hop_s": HOP_S,
        "channels": CHANNELS,
        "channel_groups": CHANNEL_GROUPS,
        "channel_units": CHANNEL_UNITS,
        "participants": participants,
        "participant_info": info,
    }
    save_json(meta, out_dir / "meta.json")
    n = sum(i["n_windows"] for i in info)
    log.info(f"{len(participants)} users, {n:,} windows -> {out_dir}")
    return meta

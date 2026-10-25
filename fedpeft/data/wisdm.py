"""
WISDM-2019: download, unzip and convert to per-subject window files.

"WISDM Smartphone and Smartwatch Activity and Biometrics Dataset" (Weiss, Fordham
University; UCI repository #507, 2019): 51 subjects (ids 1600-1650) performed 18
activities for about 3 minutes each, with a smartphone in the trouser pocket and a
smartwatch on the dominant wrist.  Each device logged its accelerometer and
gyroscope at a nominal 20 Hz.

Raw format (one file per subject, device and sensor, e.g. raw/watch/accel/data_1600_accel_watch.txt):
    every line = subject, activity code (A-S), timestamp (ns), x, y, z;
    units: accelerometer m/s^2 (gravity included), gyroscope rad/s
The two devices were not recorded synchronously, so - as in the FedAIoT benchmark
(Alam et al., 2024), whose protocol we reproduce - the watch and the phone are two
separate datasets; we use the WATCH (data.wisdm_device = watch) unless configured.

Activities: the 12 classes of the FedAIoT loader (activity_key_filtered.txt), i.e. the
18 activities without the three ball games (kicking, catch, dribbling) and without three
of the four eating activities - FedAIoT's class "eating" keeps only the sandwich
(code L), so every class is ~3 minutes per subject.

Windows: 10 s (200 samples) with a 5-s hop (50 % overlap), cut inside every
continuous recording of one activity, as FedAIoT does.  Accelerometer and gyroscope
rows are paired by their (shared) timestamp.

What this module writes (data/processed_wisdm_<device>/):
    s1600.npz  X: (N, 6, 200) float16 (raw units)   y_WISDM: (N,) int8
               t: (N,) int64 window start (ms)      seg: (N,) int32 continuous recording
    meta.json  classes, sampling rate, channel names/groups/units, participants
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import download_file, get_logger, save_json

WISDM_URL = ("https://archive.ics.uci.edu/static/public/507/"
             "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip")
SUBJECTS = range(1600, 1651)
# FedAIoT's 12 classes (datasets/wisdm/activity_key_filtered.txt): code -> (class id, name)
ACTIVITIES = {"A": (0, "Walking"), "B": (1, "Jogging"), "C": (2, "Stairs"), "D": (3, "Sitting"),
              "E": (4, "Standing"), "F": (5, "Typing"), "G": (6, "Brushing teeth"), "L": (7, "Eating"),
              "K": (8, "Drinking"), "Q": (9, "Writing"), "R": (10, "Clapping"), "S": (11, "Folding clothes")}
CLASS_NAMES = [name for _, name in sorted(ACTIVITIES.values())]
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
CHANNEL_GROUPS = {"acc": [0, 1, 2], "gyro": [3, 4, 5]}
CHANNEL_UNITS = {"acc": "m/s2", "gyro": "rad/s"}
RATE_HZ, WINDOW_SAMPLES, HOP_SAMPLES = 20, 200, 100
MAX_GAP_S = 1.0          # a longer pause between two samples starts a new continuous recording
log = get_logger("wisdm")


# ----------------------------------------------------------------------------
# 1. Download + unzip (only the raw sensor text files)
# ----------------------------------------------------------------------------
def download_and_unzip_wisdm(raw_dir: str | Path) -> Path:
    """Fetch the UCI archive (once) and extract wisdm-dataset/raw/. Returns that folder."""
    raw_dir = Path(raw_dir) / "wisdm"
    data_dir = raw_dir / "wisdm-dataset" / "raw"
    if (data_dir / "watch" / "accel").exists():
        return data_dir
    outer = raw_dir / "wisdm-dataset-outer.zip"
    if not outer.exists():
        log.info("Downloading WISDM-2019 (~310 MB) from the UCI repository ...")
        download_file(WISDM_URL, outer, desc="WISDM-2019")
    inner = raw_dir / "wisdm-dataset.zip"
    if not inner.exists():
        with zipfile.ZipFile(outer) as zf:
            zf.extract("wisdm-dataset.zip", raw_dir)
    with zipfile.ZipFile(inner) as zf:
        # The archive also ships compiled tools (arffmagic) and ARFF features: only the raw text is needed.
        keep = [m for m in zf.namelist() if m.startswith("wisdm-dataset/raw/") and m.endswith(".txt")]
        zf.extractall(raw_dir, members=keep)
    return data_dir


# ----------------------------------------------------------------------------
# 2. One subject: two text files -> windows
# ----------------------------------------------------------------------------
def _read_sensor(path: Path) -> pd.DataFrame:
    """subject, code, t_ns, x, y, z  (the trailing ';' of every line is removed).

    Repeated rows are dropped: some files contain a recording twice (e.g. subject 1629's
    accelerometer repeats every activity), which would otherwise yield duplicate windows."""
    df = pd.read_csv(path, header=None, names=["subject", "code", "t_ns", "x", "y", "z"],
                     dtype={"code": str}, converters={"z": lambda v: float(v.rstrip(";"))})
    return df.dropna().drop_duplicates(subset=["code", "t_ns"]).sort_values("t_ns")


def _on_grid(t: np.ndarray, v: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of the samples (t, v) onto `grid`, and whether a real sample
    lies within MAX_GAP_S / 2 of every grid point (no interpolation across pauses)."""
    out = np.stack([np.interp(grid, t, v[:, k]) for k in range(v.shape[1])], axis=1)
    i = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    nearest = np.minimum(np.abs(t[i] - grid), np.abs(t[i - 1] - grid))
    return out.astype(np.float32), nearest <= MAX_GAP_S / 2 * 1e9


def load_subject(data_dir: Path, sid: int, device: str = "watch") -> dict | None:
    """All windows of one subject, sorted by start time. None if a file is missing.

    Both sensors of every activity are resampled onto one 20-Hz grid (linear interpolation).
    For most subjects the two sensors already share 20-Hz timestamps, so this returns the
    recorded samples; four subjects (1637-1640) logged ~50 Hz (accelerometer) and ~25 Hz
    (gyroscope) - FedAIoT's join on identical timestamps keeps only a fraction of them.
    """
    acc_path = data_dir / device / "accel" / f"data_{sid}_accel_{device}.txt"
    gyr_path = data_dir / device / "gyro" / f"data_{sid}_gyro_{device}.txt"
    if not acc_path.exists() or not gyr_path.exists():
        return None
    acc, gyr = _read_sensor(acc_path), _read_sensor(gyr_path)
    step = int(round(1e9 / RATE_HZ))
    xs, ys, ts, segs = [], [], [], []
    next_seg = 0
    for code in [c for c in ACTIVITIES if c in set(acc["code"]) and c in set(gyr["code"])]:
        a, g = acc[acc["code"] == code], gyr[gyr["code"] == code]
        ta, tg = a["t_ns"].to_numpy(np.int64), g["t_ns"].to_numpy(np.int64)
        if len(ta) < 2 or len(tg) < 2:
            continue
        start, end = max(ta[0], tg[0]), min(ta[-1], tg[-1])
        grid = np.arange(start, end + 1, step, dtype=np.int64)
        if len(grid) < WINDOW_SAMPLES:
            continue
        va, ok_a = _on_grid(ta, a[["x", "y", "z"]].to_numpy(np.float64), grid)
        vg, ok_g = _on_grid(tg, g[["x", "y", "z"]].to_numpy(np.float64), grid)
        t, values, ok = grid, np.concatenate([va, vg], axis=1), ok_a & ok_g
        # continuous recordings of this activity: stretches of grid points with real samples nearby
        edges = np.flatnonzero(np.diff(np.r_[False, ok, False].astype(np.int8)))
        for lo, hi in zip(edges[::2], edges[1::2]):
            run_t, run_v = t[lo:hi], values[lo:hi]
            n = (len(run_t) - WINDOW_SAMPLES) // HOP_SAMPLES + 1
            if n <= 0:
                continue
            starts = np.arange(n) * HOP_SAMPLES
            xs.append(np.stack([run_v[s:s + WINDOW_SAMPLES].T for s in starts]))      # (n, 6, 200)
            ys.append(np.full(n, ACTIVITIES[code][0], dtype=np.int8))
            ts.append(run_t[starts] // 1_000_000)                                     # ns -> ms
            segs.append(np.full(n, next_seg, dtype=np.int32))
            next_seg += 1
    if not xs:
        return None
    X, y, t, seg = (np.concatenate(a) for a in (xs, ys, ts, segs))
    order = np.argsort(t, kind="stable")                    # chronological
    return {"X": X[order].astype(np.float16), "y_WISDM": y[order], "t": t[order], "seg": seg[order]}


# ----------------------------------------------------------------------------
# 3. All subjects
# ----------------------------------------------------------------------------
def prepare_wisdm(cfg) -> dict:
    """Download (if needed) and convert every subject. Writes one npz per subject + meta.json."""
    device = str(cfg.data.get("wisdm_device", "watch"))
    data_dir = download_and_unzip_wisdm(cfg.paths.raw_dir)
    out_dir = Path(cfg.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    participants, info = [], []
    for sid in SUBJECTS:
        pid = f"s{sid}"
        out = out_dir / f"{pid}.npz"
        if out.exists():
            with np.load(out) as d:
                y = d["y_WISDM"]
        else:
            arrays = load_subject(data_dir, sid, device)
            if arrays is None:
                log.info(f"{pid}: no {device} data, skipped")
                continue
            np.savez(out, **arrays)
            y = arrays["y_WISDM"]
        participants.append(pid)
        info.append({"pid": pid, "n_windows": int(len(y)),
                     "class_counts": np.bincount(y, minlength=len(CLASS_NAMES)).tolist()})
    meta = {
        "dataset": "wisdm",
        "display_name": f"WISDM-{'W' if device == 'watch' else 'P'}",
        "device": device,
        "default_scheme": "WISDM",
        "schemes": {"WISDM": CLASS_NAMES},
        "base_rate_hz": RATE_HZ,
        "window_s": WINDOW_SAMPLES / RATE_HZ,
        "hop_s": HOP_SAMPLES / RATE_HZ,
        "channels": CHANNELS,
        "channel_groups": CHANNEL_GROUPS,
        "channel_units": CHANNEL_UNITS,
        "participants": participants,
        "participant_info": info,
    }
    save_json(meta, out_dir / "meta.json")
    n = sum(i["n_windows"] for i in info)
    log.info(f"{len(participants)} subjects, {n:,} windows ({device}) -> {out_dir}")
    return meta

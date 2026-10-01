"""
Capture-24: download, unzip and convert to fixed-size windows.

What this module does (PDF component 1, "Preprocessing", the dataset-level part):

    raw CSV (100 Hz, x/y/z in g, free-text annotation per sample)
        |  map every annotation to a class of the chosen label scheme
        |  split the stream into continuous segments (no time gaps)
        |  anti-aliased resampling 100 Hz -> 30 Hz  (scipy.signal.resample_poly)
        |  cut non-overlapping 10-second windows (300 samples x 3 axes)
        |  window label = majority label of its 1,000 raw samples
        v
    data/processed/P001.npz   X: (N, 3, 300) float16   y_<scheme>: (N,) int8   t: (N,) int64

The per-encoder part of preprocessing (resampling to 20 Hz, unit conversion, ...)
happens later, on the GPU, inside `fedpeft.models.components.Preprocessor`.

Dataset reference:
    Chan, S. et al. CAPTURE-24: A large dataset of wrist-worn activity tracker data
    collected in the wild for human activity recognition. Sci Data 11, 1135 (2024).
"""

from __future__ import annotations

import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import download_file, get_logger, save_json

# Official download link used by the OxWearables/capture24 repository.
CAPTURE24_URL = (
    "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001"
    "/download_file?file_format=&safe_filename=capture24.zip&type_of_work=Dataset"
)
N_PARTICIPANTS = 151
log = get_logger("capture24")


# ----------------------------------------------------------------------------
# 1. Download + unzip
# ----------------------------------------------------------------------------
def download_capture24(raw_dir: str | Path) -> Path:
    """Download capture24.zip (~6.5 GB) into `raw_dir` (resumes if interrupted)."""
    raw_dir = Path(raw_dir)
    zip_path = raw_dir / "capture24.zip"
    if zip_path.exists() and zip_path.stat().st_size > 6e9:
        log.info(f"Found {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB), skipping download.")
        return zip_path
    log.info("Downloading Capture-24 (~6.5 GB). This can take a while ...")
    return download_file(CAPTURE24_URL, zip_path, desc="capture24.zip")


def unzip_capture24(raw_dir: str | Path) -> Path:
    """Extract the zip (only if the CSV files are not there yet). Returns the data folder."""
    raw_dir = Path(raw_dir)
    data_dir = raw_dir / "capture24"
    if len(list(data_dir.glob("P[0-9][0-9][0-9].csv.gz"))) >= N_PARTICIPANTS:
        return data_dir
    log.info("Unzipping capture24.zip ...")
    with zipfile.ZipFile(raw_dir / "capture24.zip") as zf:
        zf.extractall(raw_dir)
    return data_dir


# ----------------------------------------------------------------------------
# 2. Label dictionary
# ----------------------------------------------------------------------------
def load_label_dictionary(data_dir: Path) -> pd.DataFrame:
    """annotation-label-dictionary.csv maps the ~200 free-text annotations to label schemes.

    Columns look like: annotation, label:Willetts2018, label:Walmsley2020, ...
    """
    df = pd.read_csv(data_dir / "annotation-label-dictionary.csv", dtype=str)
    return df.set_index("annotation")


def available_schemes(label_dict: pd.DataFrame) -> list[str]:
    """Names of all label schemes in the dictionary (without the 'label:' prefix)."""
    return [c.split(":", 1)[1] for c in label_dict.columns if c.startswith("label:")]


def scheme_classes(label_dict: pd.DataFrame, scheme: str) -> list[str]:
    """Sorted class names of one scheme (index in this list = integer class id)."""
    return sorted(label_dict[f"label:{scheme}"].dropna().unique().tolist())


# ----------------------------------------------------------------------------
# 3. One participant: CSV -> windows
# ----------------------------------------------------------------------------
def _read_participant_csv(csv_path: Path) -> pd.DataFrame:
    """Read one P###.csv.gz fast with pyarrow (8-9 million rows per participant)."""
    import pyarrow as pa
    import pyarrow.csv as pacsv

    table = pacsv.read_csv(
        csv_path,
        convert_options=pacsv.ConvertOptions(
            column_types={"x": pa.float32(), "y": pa.float32(), "z": pa.float32(),
                          "annotation": pa.string(), "time": pa.string()},
        ),
    )
    df = table.to_pandas()
    # "2016-11-13 02:18:00.000000" -> int64 milliseconds since epoch.
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S.%f").astype("int64") // 1_000_000
    return df


def _continuous_segments(t_ms: np.ndarray, raw_rate: int) -> list[tuple[int, int]]:
    """Split a timestamp array into [start, end) runs without gaps (> 1.5 sample periods)."""
    period = 1000.0 / raw_rate
    gaps = np.where(np.diff(t_ms) > 1.5 * period)[0]
    starts = np.concatenate([[0], gaps + 1])
    ends = np.concatenate([gaps + 1, [len(t_ms)]])
    return list(zip(starts.tolist(), ends.tolist()))


def process_participant(csv_path: str, out_path: str, label_dict_path: str, schemes: list[str],
                        raw_rate: int = 100, base_rate: int = 30, window_s: int = 10) -> dict:
    """Convert one participant CSV into windows and save them as .npz.

    Runs in a worker process, so all arguments are plain strings / numbers.
    Returns a small summary (window counts per class) for the dataset card.
    """
    from scipy.signal import resample_poly

    csv_path, out_path = Path(csv_path), Path(out_path)
    label_dict = pd.read_csv(label_dict_path, dtype=str).set_index("annotation")
    df = _read_participant_csv(csv_path)

    # --- integer class code per raw sample, for every scheme (-1 = unlabelled) ---
    codes: dict[str, np.ndarray] = {}
    for scheme in schemes:
        classes = scheme_classes(label_dict, scheme)
        to_id = {name: i for i, name in enumerate(classes)}
        ann_to_id = label_dict[f"label:{scheme}"].map(lambda v: to_id.get(v, -1))
        codes[scheme] = df["annotation"].map(ann_to_id).fillna(-1).astype(np.int16).to_numpy()

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    t_ms = df["time"].to_numpy()

    raw_win = raw_rate * window_s     # 1,000 raw samples per window
    base_win = base_rate * window_s   # 300 resampled samples per window
    X_out, t_out = [], []
    y_out: dict[str, list[np.ndarray]] = {s: [] for s in schemes}

    for start, end in _continuous_segments(t_ms, raw_rate):
        n_win = (end - start) // raw_win
        if n_win == 0:
            continue
        seg_end = start + n_win * raw_win
        # Anti-aliased polyphase resampling of the continuous segment: 100 Hz -> 30 Hz.
        seg = resample_poly(xyz[start:seg_end], up=base_rate, down=raw_rate, axis=0).astype(np.float32)
        seg = seg[: n_win * base_win].reshape(n_win, base_win, 3).transpose(0, 2, 1)  # (n, 3, 300)

        # Majority label of each window, per scheme.
        keep = np.ones(n_win, dtype=bool)
        win_labels = {}
        for scheme in schemes:
            c = codes[scheme][start:seg_end].reshape(n_win, raw_win)
            n_cls = int(c.max()) + 2
            counts = np.stack([(c == k).sum(1) for k in range(-1, n_cls - 1)], axis=1)  # col 0 = unlabelled
            majority = counts.argmax(1) - 1
            win_labels[scheme] = majority.astype(np.int8)
        # A window is kept if it is labelled under the first (= default) scheme.
        keep &= win_labels[schemes[0]] >= 0

        X_out.append(seg[keep].astype(np.float16))
        t_out.append(t_ms[start:seg_end:raw_win][keep])
        for scheme in schemes:
            y_out[scheme].append(win_labels[scheme][keep])

    X = np.concatenate(X_out) if X_out else np.zeros((0, 3, base_win), np.float16)
    arrays = {"X": X, "t": np.concatenate(t_out) if t_out else np.zeros(0, np.int64)}
    for scheme in schemes:
        arrays[f"y_{scheme}"] = np.concatenate(y_out[scheme]) if y_out[scheme] else np.zeros(0, np.int8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)

    return {
        "participant": csv_path.name.split(".")[0],
        "n_windows": int(len(X)),
        "hours": round(len(X) * window_s / 3600, 2),
        "class_counts": {s: np.bincount(arrays[f"y_{s}"][arrays[f"y_{s}"] >= 0],
                                        minlength=len(scheme_classes(label_dict, s))).tolist()
                         for s in schemes},
    }


# ----------------------------------------------------------------------------
# 4. All participants
# ----------------------------------------------------------------------------
def prepare_capture24(cfg) -> dict:
    """Download (if needed), unzip, and convert all 151 participants in parallel."""
    raw_dir, out_dir = Path(cfg.paths.raw_dir), Path(cfg.paths.processed_dir)
    download_capture24(raw_dir)
    data_dir = unzip_capture24(raw_dir)

    label_dict = load_label_dictionary(data_dir)
    schemes = available_schemes(label_dict)
    # Put the configured scheme first: it decides which windows are kept.
    schemes.remove(cfg.data.label_scheme)
    schemes = [cfg.data.label_scheme] + schemes
    log.info(f"Label schemes: {schemes}  (default = {cfg.data.label_scheme})")

    csv_files = sorted(data_dir.glob("P[0-9][0-9][0-9].csv.gz"))
    jobs = []
    for csv in csv_files:
        out = out_dir / (csv.name.split(".")[0] + ".npz")
        if not out.exists():
            jobs.append((str(csv), str(out)))
    log.info(f"{len(csv_files)} participants found, {len(jobs)} still to convert.")

    summaries = []
    with ProcessPoolExecutor(max_workers=cfg.data.prepare_workers) as pool:
        futures = [pool.submit(process_participant, csv, out, str(data_dir / "annotation-label-dictionary.csv"),
                               schemes, cfg.data.raw_rate_hz, cfg.data.base_rate_hz, cfg.data.window_s)
                   for csv, out in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            s = fut.result()
            summaries.append(s)
            log.info(f"[{i}/{len(jobs)}] {s['participant']}: {s['n_windows']} windows ({s['hours']} h)")

    # Dataset card: classes of every scheme + per-participant summary (rebuilt from files).
    meta = {
        "dataset": "capture24",
        "default_scheme": cfg.data.label_scheme,
        "schemes": {s: scheme_classes(label_dict, s) for s in schemes},
        "base_rate_hz": cfg.data.base_rate_hz,
        "window_s": cfg.data.window_s,
        "participants": sorted(p.stem for p in out_dir.glob("P*.npz")),
    }
    # Participant metadata (age group, sex) is useful context for the report.
    meta_csv = data_dir / "metadata.csv"
    if meta_csv.exists():
        meta["participant_info"] = pd.read_csv(meta_csv, dtype=str).to_dict(orient="records")
    save_json(meta, out_dir / "meta.json")
    log.info(f"Done. Metadata written to {out_dir / 'meta.json'}")
    return meta

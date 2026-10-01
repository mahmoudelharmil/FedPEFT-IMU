"""
Small, dependency-free helpers used across the project.

    seed_everything   - reproducible runs
    get_device        - pick CUDA if available
    Timer             - wall-clock timing that respects asynchronous CUDA kernels
    save_json/load_json - result files (numpy/torch values are converted automatically)
    download_file     - robust HTTP download with resume + progress (uses the OS
                        certificate store, which matters behind antivirus TLS scanners)
    get_logger        - consistent console + file logging
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Reproducibility and device
# ----------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    """Return the preferred device if it is available, otherwise the CPU."""
    if preferred == "cuda" and torch.cuda.is_available():
        # cuDNN autotuner picks the fastest convolution algorithm for fixed shapes.
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def cuda_sync(device: torch.device) -> None:
    """Wait for all queued GPU work (needed before reading a timer)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


class Timer:
    """Context manager measuring wall-clock seconds (GPU-synchronised)."""

    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.seconds = 0.0

    def __enter__(self) -> "Timer":
        if self.device is not None:
            cuda_sync(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self.device is not None:
            cuda_sync(self.device)
        self.seconds = time.perf_counter() - self._start


# ----------------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------------
def _to_jsonable(obj: Any) -> Any:
    """Convert numpy / torch objects into plain Python for json.dump."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    """Write `obj` as pretty JSON, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(obj), fh, indent=2)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written result


def save_arrays(path: str | Path, **arrays) -> None:
    """Store numpy/torch arrays in one compressed .npz (float tensors as float16)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, v in arrays.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu()
            v = v.half().numpy() if v.is_floating_point() else v.numpy()
        out[k] = v
    np.savez_compressed(path, **out)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Console logger (+ optional file) with a compact, timestamped format."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


# ----------------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------------
def _use_os_certificates() -> None:
    """Let Python trust the Windows/macOS certificate store.

    Antivirus products (e.g. Norton) and corporate proxies re-sign HTTPS traffic
    with their own root certificate, which Python's bundled CA list does not
    know.  The `truststore` package makes `ssl` use the operating-system store.
    """
    try:
        import truststore  # type: ignore

        truststore.inject_into_ssl()
    except ImportError:
        pass


def download_file(url: str, dest: str | Path, desc: str | None = None,
                  chunk_mb: int = 4) -> Path:
    """Download `url` to `dest` with resume support and a progress bar."""
    import requests
    from tqdm import tqdm

    _use_os_certificates()
    dest = Path(dest)
    if dest.exists():                     # finished earlier
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    # Resume a previous partial download if there is one.
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, headers=headers, timeout=60, allow_redirects=True) as r:
        if r.status_code == 416:  # range not satisfiable -> already complete
            os.replace(part, dest)
            return dest
        r.raise_for_status()
        if done and r.status_code != 206:  # server ignored the Range header: restart
            done = 0
        total = int(r.headers.get("Content-Length", 0)) + done
        mode = "ab" if done else "wb"
        with open(part, mode) as fh, tqdm(total=total or None, initial=done, unit="B",
                                          unit_scale=True, desc=desc or dest.name) as bar:
            for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                fh.write(chunk)
                bar.update(len(chunk))
    os.replace(part, dest)
    return dest


def human_mb(n_bytes: float) -> float:
    """Bytes -> megabytes (10^6, the unit used for network traffic)."""
    return float(n_bytes) / 1e6

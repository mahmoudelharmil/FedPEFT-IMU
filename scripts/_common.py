"""
Shared start-up code for every script in scripts/.

    from _common import setup
    cfg, device, log = setup("my_script", description="...")

It makes the `fedpeft` package importable, parses --profile and key=value
overrides, loads the configuration, picks the device and creates a log file in
logs/<profile>/<script>.log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from fedpeft.config import add_common_cli_args, load_config  # noqa: E402
from fedpeft.utils import get_device, get_logger  # noqa: E402


def setup(name: str, description: str, extra_args=None):
    """Parse CLI, load config, choose device. `extra_args(parser)` can add script flags."""
    parser = argparse.ArgumentParser(description=description)
    add_common_cli_args(parser)
    if extra_args:
        extra_args(parser)
    args = parser.parse_args()
    # Windows consoles default to cp1252; make "·", "×", "α" print correctly in logs.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    cfg = load_config(args.profile, args.overrides)
    torch.set_num_threads(cfg.runtime.num_threads)
    device = get_device(cfg.runtime.device)
    log = get_logger(name, ROOT / "logs" / cfg.profile / f"{name}.log")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    log.info(f"profile={cfg.profile}  device={device} ({gpu})  torch={torch.__version__}")
    return cfg, device, log, args


def csv_list(text: str | None, default: list[str]) -> list[str]:
    """'a,b,c' -> ['a','b','c'];  None -> default."""
    return [t.strip() for t in text.split(",") if t.strip()] if text else list(default)

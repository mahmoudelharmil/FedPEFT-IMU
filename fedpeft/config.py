"""
Configuration loading.

The configuration is built in three layers, each overriding the previous one:

    1. configs/default.yaml                 (all defaults, heavily commented)
    2. configs/profiles/<profile>.yaml      (quick / standard / full)
    3. command-line overrides               e.g.  fl.rounds=10  data.subsample_every=12

The result is a `Config` object: a normal dict that also allows attribute access,
so you can write `cfg.fl.rounds` instead of `cfg["fl"]["rounds"]`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

# Project root = the folder that contains `fedpeft/` and `configs/`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


class Config(dict):
    """A dict with attribute access (recursively converts nested dicts)."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = Config(value) if isinstance(value, dict) else value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # make missing keys easy to diagnose
            raise AttributeError(f"Config has no key '{name}'") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> dict:
        """Convert back to plain nested dicts (for JSON / YAML dumping)."""
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in self.items()}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_override(text: str) -> tuple[list[str], Any]:
    """Turn "fl.rounds=10" into (["fl", "rounds"], 10). Values are parsed as YAML."""
    if "=" not in text:
        raise ValueError(f"Override '{text}' must look like key.sub=value")
    key, raw_value = text.split("=", 1)
    return key.strip().split("."), yaml.safe_load(raw_value)


def load_config(profile: str | None = "standard", overrides: Iterable[str] = ()) -> Config:
    """Load default.yaml, merge a profile and CLI overrides, and resolve paths."""
    with open(CONFIG_DIR / "default.yaml", "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    # Layer 2: profile file (optional).
    if profile:
        profile_path = CONFIG_DIR / "profiles" / f"{profile}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"Unknown profile '{profile}' ({profile_path})")
        with open(profile_path, "r", encoding="utf-8") as fh:
            data = _deep_merge(data, yaml.safe_load(fh) or {})
    data["profile"] = profile or "default"

    # Layer 3: command-line overrides.
    for text in overrides:
        keys, value = _parse_override(text)
        node = data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    # Make every path absolute (relative to the project root).
    for key, rel in data["paths"].items():
        data["paths"][key] = str((PROJECT_ROOT / rel).resolve())

    return Config(data)


def encoder_config(cfg: Config, encoder_name: str) -> Config:
    """Return `cfg` with the encoder's optional `budget` overrides merged in.

    Example (configs/default.yaml): UniMTS is ~17x slower per window than the other
    encoders on a 6 GB laptop GPU, so `encoders.unimts.budget` lowers its rounds /
    local steps.  Every experiment of that encoder then uses the same reduced
    budget, so its methods remain directly comparable with each other.
    """
    budget = cfg.encoders.get(encoder_name, {}).get("budget")
    if not budget:
        return cfg
    merged = Config(_deep_merge(cfg.to_dict(), budget.to_dict() if isinstance(budget, Config) else budget))
    merged["budget_applied"] = encoder_name
    return merged


def add_common_cli_args(parser) -> None:
    """Every script accepts --profile and a list of key=value overrides."""
    parser.add_argument("--profile", default="standard",
                        help="quick | standard | full (see configs/profiles/)")
    parser.add_argument("overrides", nargs="*",
                        help="optional config overrides, e.g. fl.rounds=10")

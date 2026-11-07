"""
The training settings, as the UI shows and edits them.

Settings come in layers (fedpeft/config.py): configs/default.yaml < configs/datasets/<dataset>.yaml
< configs/profiles/<profile>.yaml < command-line overrides.  The UI edits the profile (or the
dataset file), never default.yaml.  Edits keep the files' comments (ruamel.yaml); every save is
checked by loading the configuration, a backup of the old file is kept in logs/manager/backups/,
and a file that would not load is restored.
"""

from __future__ import annotations

import io
import re
import time
from pathlib import Path

import yaml
from ruamel.yaml import YAML

from ..config import CONFIG_DIR, encoder_config, load_config
from .plan import MANAGER_FILE, load_manager_config
from .state import STATE_DIR

PROFILES = CONFIG_DIR / "profiles"
DATASETS = CONFIG_DIR / "datasets"
BACKUPS = STATE_DIR / "backups"
NAME = re.compile(r"^[a-z0-9_]+$")

# The settings shown as a form (everything else: edit the YAML text).
SETTINGS = [
    ("Data and users", [
        ("split.seeds", "User splits (seeds)", "list"),
        ("split.heldout_frac", "Held-out (new) users, fraction", "float"),
        ("split.pool_frac", "Pool users (validation / SSL), fraction", "float"),
        ("split.train_time_frac", "Train part of each client's data", "float"),
        ("data.subsample_every", "Keep 1 of every N windows", "int"),
    ]),
    ("Federated training", [
        ("fl.rounds", "Rounds", "int"),
        ("fl.clients_per_round_frac", "Clients per round, fraction", "float"),
        ("fl.local_epochs", "Local epochs", "int"),
        ("fl.max_local_steps", "Max local steps per round", "int"),
        ("fl.batch_size", "Batch size", "int"),
        ("fl.eval_every", "Evaluate every N rounds", "int"),
        ("fl.lr_peft", "Learning rate, LoRA / heads (V1–V5)", "float"),
        ("fl.lr_full", "Learning rate, full fine-tuning (V0)", "float"),
        ("fl.lr_scratch", "Learning rate, from scratch (FedAvg)", "float"),
        ("fl.local_ft_steps", "+FT steps (shared classifier)", "int"),
        ("fl.final_personalization_steps", "V5 final personal steps", "int"),
        ("fl.methods", "Methods of the grid", "list"),
    ]),
    ("Model", [
        ("encoders.names", "Encoders", "list"),
        ("model.lora.rank", "LoRA rank (V2, V3; V4/V5 highest tier)", "int"),
        ("model.proto_weight", "Prototype loss weight (V5)", "float"),
        ("augment.enabled", "Training augmentation", "bool"),
    ]),
    ("Centralized training", [
        ("centralized.epochs", "Epochs", "int"),
        ("centralized.patience", "Early-stopping patience", "int"),
        ("centralized.batch_size", "Batch size", "int"),
        ("centralized.lr_scratch", "Learning rate, from scratch", "float"),
        ("centralized.lr_finetune", "Learning rate, pretrained", "float"),
    ]),
    ("HALO pretraining", [
        ("halo_ssl.epochs", "Epochs", "int"),
    ]),
]


def _rt() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def _dig(d, dotted: str):
    node = d
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return None, False
        node = node[k]
    return node, True


def default_help() -> dict:
    """{dotted key: inline comment} of configs/default.yaml (shown as help in the form)."""
    out, stack = {}, []
    for line in (CONFIG_DIR / "default.yaml").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\s*)([A-Za-z_][\w]*):\s*(.*)$", line)
        if not m:
            continue
        depth = len(m.group(1)) // 2
        stack = stack[:depth] + [m.group(2)]
        rest = m.group(3)
        if "#" in rest:
            out[".".join(stack)] = rest.split("#", 1)[1].strip()
    return out


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dataset_of(profile: str) -> str:
    data = _load_yaml(PROFILES / f"{profile}.yaml")
    default = _load_yaml(CONFIG_DIR / "default.yaml")
    return (data.get("data") or {}).get("dataset") or default["data"]["dataset"]


def roles() -> dict:
    """{profile: (dataset key, role)} for the profiles the manager uses."""
    m = load_manager_config()
    out = {}
    for ds, d in (m.get("datasets") or {}).items():
        out[d["profile"]] = (ds, "main")
        for pub in d.get("published") or []:
            out[pub["profile"]] = (ds, "published")
    return out


def list_profiles() -> list[dict]:
    r = roles()
    out = []
    for path in sorted(PROFILES.glob("*.yaml")):
        name = path.stem
        text = path.read_text(encoding="utf-8")
        desc = []
        for line in text.splitlines():
            s = line.strip()
            if not s.startswith("#"):
                if desc:
                    break
                continue
            s = s.lstrip("#").strip()
            if s and not set(s) <= set("-="):
                desc.append(s)
        ds_key, role = r.get(name, (None, "other"))
        out.append({"name": name, "dataset": dataset_of(name), "manager_dataset": ds_key, "role": role,
                    "description": " ".join(desc)[:300]})
    return out


def detail(name: str) -> dict:
    if not NAME.match(name) or not (PROFILES / f"{name}.yaml").exists():
        raise FileNotFoundError(name)
    cfg = load_config(name)
    prof = _load_yaml(PROFILES / f"{name}.yaml")
    ds_file = DATASETS / f"{cfg.data.dataset}.yaml"
    ds = _load_yaml(ds_file)
    helps = default_help()
    groups = []
    for title, keys in SETTINGS:
        rows = []
        for key, label, typ in keys:
            value, _ = _dig(cfg, key)
            pv, in_prof = _dig(prof, key)
            _, in_ds = _dig(ds, key)
            rows.append({"key": key, "label": label, "type": typ, "value": value,
                         "source": "profile" if in_prof else "dataset" if in_ds else "default",
                         "help": helps.get(key, "")})
        groups.append({"title": title, "settings": rows})
    budgets = []
    for enc in cfg.encoders.names:
        ec = encoder_config(cfg, enc)
        budgets.append({"encoder": enc, "rounds": ec.fl.rounds, "max_local_steps": ec.fl.max_local_steps,
                        "batch_size": ec.fl.batch_size, "centralized_epochs": ec.centralized.epochs,
                        "methods": list(cfg.encoders.get(enc, {}).get("methods") or cfg.fl.methods),
                        "budget": bool(cfg.encoders.get(enc, {}).get("budget"))})
    return {"name": name, "dataset": cfg.data.dataset, "profile_text": (PROFILES / f"{name}.yaml").read_text(encoding="utf-8"),
            "dataset_file": ds_file.name if ds_file.exists() else None,
            "dataset_text": ds_file.read_text(encoding="utf-8") if ds_file.exists() else "",
            "groups": groups, "budgets": budgets, "effective": yaml.safe_dump(cfg.to_dict(), sort_keys=False, width=110)}


# ---------------------------------------------------------------------- editing
def _target(kind: str, name: str) -> Path:
    if kind == "manager":
        return MANAGER_FILE
    if not NAME.match(name or ""):
        raise ValueError("invalid name")
    return (PROFILES if kind == "profile" else DATASETS) / f"{name}.yaml"


def _profiles_to_check(kind: str, name: str) -> list[str]:
    if kind == "profile":
        return [name]
    if kind == "dataset":
        return [p.stem for p in PROFILES.glob("*.yaml") if dataset_of(p.stem) == name]
    return []


def _validate(kind: str, name: str) -> None:
    if kind == "manager":
        from .plan import build_plan
        build_plan()
        return
    default_dataset = _load_yaml(CONFIG_DIR / "default.yaml")["data"]["dataset"]
    for prof in _profiles_to_check(kind, name):
        cfg = load_config(prof)
        ds = cfg.data.dataset
        if ds != default_dataset and not (DATASETS / f"{ds}.yaml").exists():
            raise ValueError(f"unknown dataset '{ds}' (no configs/datasets/{ds}.yaml)")
        for enc in cfg.encoders.names:
            if enc not in cfg.encoders:
                raise ValueError(f"encoder '{enc}' has no settings (encoders.{enc})")
            encoder_config(cfg, enc)
        for m in cfg.fl.methods:
            from ..federated.methods import METHODS
            if m not in METHODS:
                raise ValueError(f"unknown FL method '{m}'")


def save_text(kind: str, name: str, text: str) -> dict:
    """Replace a settings file (profile / dataset / manager) after checking it loads."""
    path = _target(kind, name)
    yaml.safe_load(text)                                      # syntax first
    old = path.read_text(encoding="utf-8") if path.exists() else None
    backup = None
    if old is not None:
        BACKUPS.mkdir(parents=True, exist_ok=True)
        backup = BACKUPS / f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.yaml"
        backup.write_text(old, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    try:
        _validate(kind, name)
    except Exception as exc:
        if old is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(old, encoding="utf-8")
        raise ValueError(f"not saved: the configuration does not load ({exc})") from exc
    return {"saved": str(path.name), "backup": backup.name if backup else None}


def set_value(name: str, key: str, value, unset: bool = False) -> dict:
    """Set (or remove) one dotted key in a profile, keeping its comments."""
    path = _target("profile", name)
    y = _rt()
    data = y.load(path.read_text(encoding="utf-8")) or {}
    keys = key.split(".")
    node = data
    for k in keys[:-1]:
        if k not in node or node[k] is None:
            if unset:
                return {"saved": path.name, "unchanged": True}
            node[k] = {}
        node = node[k]
    if unset:
        node.pop(keys[-1], None)
    else:
        node[keys[-1]] = value
    buf = io.StringIO()
    y.dump(data, buf)
    return save_text("profile", name, buf.getvalue())


def duplicate(name: str, new: str) -> dict:
    if not NAME.match(new):
        raise ValueError("use lower-case letters, digits and _ only")
    dst = PROFILES / f"{new}.yaml"
    if dst.exists():
        raise ValueError(f"profile {new} exists")
    text = (PROFILES / f"{name}.yaml").read_text(encoding="utf-8")
    dst.write_text(f"# Copy of profile '{name}' ({time.strftime('%Y-%m-%d')}).\n" + text, encoding="utf-8")
    return {"saved": dst.name}

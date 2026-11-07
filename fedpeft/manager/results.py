"""
One-line summaries of the result files (results/<profile>/{centralized,federated}/*.json).

The manager shows scores and estimates run times from them.  Files are re-read
only when they change (size + mtime).
"""

from __future__ import annotations

import json
import statistics
import threading
from collections import defaultdict
from pathlib import Path

from ..config import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"
_cache: dict[str, tuple[tuple, dict]] = {}
_lock = threading.Lock()


def _num(x):
    return float(x) if isinstance(x, (int, float)) else None


def summarize(path: Path) -> dict | None:
    """The numbers of one result file the manager needs (None if it is not a run result)."""
    try:
        r = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(r, dict) or "encoder" not in r:
        return None
    kind = path.parent.name                                   # centralized | federated
    held = (r.get("heldout") or {}).get("zero_shot") or {}
    s = {"kind": kind, "profile": path.parent.parent.name, "encoder": r["encoder"],
         "encoder_display": r.get("encoder_display", r["encoder"]), "seed": int(r.get("seed", 0)),
         "file": path.relative_to(PROJECT_ROOT).as_posix(),
         "zero_shot_f1": _num(held.get("mean_user_f1")),          # new users, no labels (mean over users)
         "zero_shot_acc": _num(held.get("accuracy"))}             # pooled accuracy (the FedAIoT metric)
    if kind == "centralized":
        t = r.get("test") or {}
        s.update(method=r.get("mode", ""), mode=r.get("mode", ""), tag="", macro_f1=_num(t.get("macro_f1")),
                 accuracy=_num(t.get("accuracy")), client_acc=_num(t.get("client_acc_mean")),
                 worst10_f1=_num(t.get("worst10_f1")), minutes=(_num(r.get("train_time_s")) or 0) / 60 or None)
    else:
        f = r.get("final") or {}
        ft = r.get("final_ft") or {}
        comm = r.get("comm") or {}
        s.update(method=r.get("method", ""), short=r.get("method_short", ""), tag=(r.get("tag") or "").lstrip("_"),
                 macro_f1=_num(f.get("macro_f1")), accuracy=_num(f.get("accuracy")),
                 client_acc=_num(f.get("client_acc_mean")), worst10_f1=_num(f.get("worst10_f1")),
                 worst10_acc=_num(f.get("worst10_acc")), ft_macro_f1=_num(ft.get("macro_f1")),
                 ft_client_acc=_num(ft.get("client_acc_mean")),
                 uplink_mb=_num(comm.get("uplink_mb_per_client_round")),
                 r_max=(r.get("config") or {}).get("r_max"),
                 minutes=(_num((r.get("resources") or {}).get("total_time_s")) or 0) / 60 or None)
    return s


def all_summaries() -> dict[str, dict]:
    """{project-relative path: summary} of every result file (cached)."""
    out = {}
    with _lock:
        seen = set()
        for path in RESULTS.glob("*/*/*.json"):
            if path.parent.name not in ("centralized", "federated"):
                continue
            key = path.relative_to(PROJECT_ROOT).as_posix()
            seen.add(key)
            try:
                st = path.stat()
            except OSError:
                continue
            stamp = (st.st_size, st.st_mtime_ns)
            hit = _cache.get(key)
            if not hit or hit[0] != stamp:
                _cache[key] = (stamp, summarize(path))
            if _cache[key][1]:
                out[key] = _cache[key][1]
        for key in set(_cache) - seen:
            del _cache[key]
    return out


def family(method: str) -> str:
    """Methods that cost about the same per round share an estimate (V5 ablations ~ V5)."""
    if method.startswith("v5"):
        return "v5"
    return method


class Durations:
    """Median run time (minutes) of earlier runs, to estimate how long a queued job takes."""

    def __init__(self, summaries: dict[str, dict]):
        self.exact, self.enc, self.any_profile = defaultdict(list), defaultdict(list), defaultdict(list)
        for s in summaries.values():
            m = s.get("minutes")
            if not m or m > 600:          # > 10 h: a run that included a sleep of the machine
                continue
            fam = family(s["method"]) if s["kind"] == "federated" else "central"
            self.exact[(s["profile"], s["encoder"], fam)].append(m)
            self.enc[(s["profile"], s["encoder"], s["kind"])].append(m)
            self.any_profile[(s["encoder"], fam)].append(m)

    def estimate(self, profile: str, encoder: str, method: str, kind: str) -> float | None:
        fam = family(method) if kind == "federated" else "central"
        for table, key in ((self.exact, (profile, encoder, fam)), (self.enc, (profile, encoder, kind)),
                           (self.any_profile, (encoder, fam))):
            if table.get(key):
                return float(statistics.median(table[key]))
        return None

"""
Live progress of a job, read from its log: phase, round / epoch, learning curve, final scores.

Phases of a federated run (scripts/05, 06):
    data -> train (every eval_every rounds the model is scored on a subset of the clients'
    test windows: a learning curve only, nothing is selected on it) -> personalise (V5) and
    test on all test windows -> +FT (each client fine-tunes the shared classifier) and new
    users (held-out users: zero-shot, few-shot) -> saved
Phases of a centralized run (scripts/04):
    data -> train + validate (val-F1 every epoch, early stopping, best epoch kept) -> test
    -> new users -> saved
"""

from __future__ import annotations

import re
from pathlib import Path

LINE = re.compile(r"^(\d\d:\d\d:\d\d) \| ([^|]+?) \| (.*)$")
FL_HEAD = re.compile(r"(\d+) clients, (\d+)/round, (\d+) rounds")
FL_ROUND = re.compile(r"round\s+(\d+)\s+loss ([\d.]+)\s+F1 ([\d.]+)\s+acc ([\d.]+)\s+worst10 ([\d.]+)\s+up ([\d.]+) MB\s+\(([\d.]+)s/round\)")
FL_PERS = re.compile(r"after personalisation: F1 ([\d.]+)")
FL_FT = re.compile(r"\+local fine-tuning: F1 ([\d.]+) client-acc ([\d.]+)")
FL_FINAL = re.compile(r"FINAL F1 ([\d.]+).*? acc ([\d.]+) client-acc ([\d.]+) worst10 ([\d.]+) \| up ([\d.]+) MB/client/round"
                      r"(?: \| held-out zero-shot ([\d.]+))?")
C_HEAD = re.compile(r"([\d,]+) train windows, ([\d,]+) validation windows")
C_EPOCH = re.compile(r"epoch\s+(\d+)\s+loss ([\d.]+)\s+val-F1 ([\d.]+)")
C_TEST = re.compile(r"TEST macro-F1 ([\d.]+)\s+acc ([\d.]+)\s+client-acc ([\d.]+)\s+worst10 ([\d.]+)(?:\s+\| held-out zero-shot ([\d.]+))?")
H_EPOCH = re.compile(r"epoch (\d+): MAE ([\d.]+)\s+contrastive ([\d.]+)")

FL_PHASES = [("data", "Load data"), ("train", "Train (rounds)"), ("finish", "Personalise + test"),
             ("newusers", "+FT and new users"), ("saved", "Saved")]
C_PHASES = [("data", "Load data"), ("train", "Train + validate"), ("test", "Test"), ("newusers", "New users"),
            ("saved", "Saved")]
H_PHASES = [("data", "Load data"), ("train", "Self-supervised epochs"), ("saved", "Saved")]
OTHER_PHASES = [("run", "Running"), ("saved", "Finished")]


def _tail_bytes(path: Path, max_bytes: int = 400_000) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def tail(path: Path, lines: int = 200) -> str:
    return "\n".join(_tail_bytes(path, 200_000).splitlines()[-lines:])


def parse(path: Path, script: str, total_steps: int | None) -> dict:
    """Read the log of one job; returns phase, step / total, curve and final scores."""
    text = _tail_bytes(path)
    kind = {"05_run_federated.py": "fl", "06_run_ablations.py": "fl", "04_run_centralized.py": "central",
            "03_pretrain_halo.py": "halo"}.get(script, "other")
    phases = {"fl": FL_PHASES, "central": C_PHASES, "halo": H_PHASES}.get(kind, OTHER_PHASES)
    out = {"kind": kind, "phases": [{"key": k, "label": l} for k, l in phases], "phase": phases[0][0],
           "step": 0, "total": total_steps, "unit": "round" if kind == "fl" else "epoch", "curve": [],
           "final": None, "error": None, "info": {}, "last_time": None}
    if not text:
        return out
    traceback = None
    for raw in text.splitlines():
        if raw.startswith("Traceback"):
            traceback = [raw]
            continue
        if traceback is not None and not LINE.match(raw):
            traceback.append(raw)
            continue
        m = LINE.match(raw)
        if not m:
            continue
        out["last_time"] = m.group(1)
        msg = m.group(3)
        if kind == "fl":
            if (h := FL_HEAD.search(msg)):
                out.update(phase="train", step=0, curve=[], final=None)
                out["total"] = int(h.group(3))
                out["info"] = {"clients": int(h.group(1)), "per_round": int(h.group(2)), "header": msg}
            elif (r := FL_ROUND.search(msg)):
                rnd = int(r.group(1))
                out["step"] = rnd
                out["curve"].append({"x": rnd, "loss": float(r.group(2)), "f1": float(r.group(3)),
                                     "acc": float(r.group(4)), "worst10": float(r.group(5)),
                                     "up_mb": float(r.group(6)), "s_per_round": float(r.group(7))})
                if out["total"] and rnd >= out["total"]:
                    out["phase"] = "finish"
            elif FL_PERS.search(msg) or FL_FT.search(msg):
                out["phase"] = "newusers"
            elif (f := FL_FINAL.search(msg)):
                out["phase"] = "saved"
                out["final"] = {"macro_f1": float(f.group(1)), "accuracy": float(f.group(2)),
                                "client_acc": float(f.group(3)), "worst10": float(f.group(4)),
                                "uplink_mb": float(f.group(5)),
                                "zero_shot": float(f.group(6)) if f.group(6) else None}
        elif kind == "central":
            if C_HEAD.search(msg):
                out.update(phase="train", step=0, curve=[], final=None)
            elif (e := C_EPOCH.search(msg)):
                out["step"] = int(e.group(1))
                out["curve"].append({"x": int(e.group(1)), "loss": float(e.group(2)), "f1": float(e.group(3))})
            elif "early stop" in msg:
                out["info"]["early_stop"] = msg
                out["phase"] = "test"
            elif (t := C_TEST.search(msg)):
                out["phase"] = "newusers"
                out["final"] = {"macro_f1": float(t.group(1)), "accuracy": float(t.group(2)),
                                "client_acc": float(t.group(3)), "worst10": float(t.group(4)),
                                "zero_shot": float(t.group(5)) if t.group(5) else None}
            elif msg.startswith("saved "):
                out["phase"] = "saved"
        elif kind == "halo":
            if msg.startswith("HALO SSL:"):
                out.update(phase="train", step=0, curve=[])
                out["info"] = {"header": msg}
            elif (e := H_EPOCH.search(msg)):
                out["step"] = int(e.group(1))
                out["curve"].append({"x": int(e.group(1)), "loss": float(e.group(2)), "f1": None,
                                     "contrastive": float(e.group(3))})
        if kind != "other" and out["phase"] == "data" and ("Split seed" in msg or "WindowStore" in msg):
            out["phase"] = "data"
    if kind == "central" and out["phase"] == "train" and out["total"] and out["step"] >= out["total"]:
        out["phase"] = "test"
    if traceback:
        out["error"] = "\n".join(traceback[-25:])
    return out

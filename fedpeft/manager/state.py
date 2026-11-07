"""
The persistent queue (logs/manager/queue.json) shared by the UI server and the runner.

    items     the queued jobs in run order, each with its status:
              pending | running | done | failed | skipped
    control   what the runner should do:
              mode          run | pause | stop        (pause: start no new job; stop: the runner exits)
              after_current pause | stop | null       (applied when the current job ends)
              request       suspend | resume | kill | skip | null   (acts on the running job now)

The runner writes its own heartbeat to logs/manager/runner.json.  Every change of
queue.json happens under a lock file, so the two processes never overwrite each
other.  Files are replaced atomically, so a reader never sees half a file.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from ..config import PROJECT_ROOT

STATE_DIR = PROJECT_ROOT / "logs" / "manager"
QUEUE_FILE = STATE_DIR / "queue.json"
RUNNER_FILE = STATE_DIR / "runner.json"
LOCK_FILE = STATE_DIR / "queue.lock"
JOB_LOGS = STATE_DIR / "jobs"
RUNNER_LOG = STATE_DIR / "runner.log"

FINISHED = ("done", "failed", "skipped")


def empty_state() -> dict:
    return {"items": [], "control": {"mode": "run", "after_current": None, "request": None}, "seq": 0}


def _read_json(path: Path, default):
    for _ in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except (ValueError, PermissionError):           # being replaced right now
            time.sleep(0.05)
    return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    for _ in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:                         # Windows: a reader has the file open
            time.sleep(0.05)
    os.replace(tmp, path)


@contextmanager
def locked(timeout: float = 15.0, stale: float = 30.0):
    """A cross-process lock: an exclusively created file (removed when stale)."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        try:
            fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - LOCK_FILE.stat().st_mtime > stale:
                    LOCK_FILE.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() - t0 > timeout:
                raise TimeoutError("queue lock busy")
            time.sleep(0.03)
    try:
        yield
    finally:
        LOCK_FILE.unlink(missing_ok=True)


def load() -> dict:
    s = _read_json(QUEUE_FILE, None) or empty_state()
    s.setdefault("control", empty_state()["control"])
    s.setdefault("items", [])
    s.setdefault("seq", 0)
    return s


def update(fn):
    """Read-modify-write the queue under the lock; returns fn's result."""
    with locked():
        s = load()
        out = fn(s)
        _write_json(QUEUE_FILE, s)
        return out


def runner_status() -> dict:
    return _read_json(RUNNER_FILE, {}) or {}


def write_runner_status(data: dict) -> None:
    _write_json(RUNNER_FILE, data)


def job_log_path(item: dict) -> Path:
    return JOB_LOGS / f"{item['key']}.log"


# ---------------------------------------------------------------------- queue edits
def new_item(state: dict, job: dict, rerun: bool = False) -> dict:
    state["seq"] = int(state.get("seq", 0)) + 1
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in job["id"])
    return {"key": f"{state['seq']:05d}_{safe}"[:150], "id": job["id"], "job": job, "rerun": bool(rerun),
            "status": "pending", "added": time.time(), "started": None, "ended": None, "attempts": 0,
            "rc": None, "error": None, "note": None}


def enqueue(state: dict, jobs: list[dict], rerun: bool = False, front: bool = False) -> int:
    """Add jobs that are not already waiting or running; returns how many were added."""
    active = {it["id"] for it in state["items"] if it["status"] in ("pending", "running")}
    fresh = [new_item(state, j, rerun) for j in jobs if j["id"] not in active]
    if front:
        running = [it for it in state["items"] if it["status"] == "running"]
        others = [it for it in state["items"] if it["status"] != "running"]
        state["items"] = running + fresh + others
    else:
        state["items"].extend(fresh)
    return len(fresh)


def find(state: dict, key: str) -> dict | None:
    return next((it for it in state["items"] if it["key"] == key), None)


def move(state: dict, key: str, where: str) -> None:
    items = state["items"]
    it = find(state, key)
    if not it:
        return
    i = items.index(it)
    items.pop(i)
    if where == "up":
        items.insert(max(0, i - 1), it)
    elif where == "down":
        items.insert(min(len(items), i + 1), it)
    elif where == "top":
        first = next((k for k, x in enumerate(items) if x["status"] in ("pending",)), len(items))
        items.insert(first, it)
    else:
        items.append(it)


def start_from(state: dict, key: str) -> int:
    """Skip every pending job before `key`, so the runner continues there; returns how many were skipped."""
    n = 0
    for it in state["items"]:
        if it["key"] == key:
            if it["status"] in FINISHED:
                it.update(status="pending", error=None, note="restarted here")
            break
        if it["status"] == "pending":
            it.update(status="skipped", note="skipped by 'start from here'", ended=time.time())
            n += 1
    return n


def counts(state: dict) -> dict:
    c = {k: 0 for k in ("pending", "running", "done", "failed", "skipped")}
    for it in state["items"]:
        c[it["status"]] = c.get(it["status"], 0) + 1
    return c

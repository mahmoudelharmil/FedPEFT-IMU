"""
The background worker: runs the queued jobs one at a time (one GPU job, the rule of this machine).

    python -m fedpeft.manager.runner          # normally started by the UI (detached)

It reads logs/manager/queue.json every second and obeys its control flags:
    pause after the current job, pause now (the job's process is suspended: it keeps its
    GPU memory and continues exactly where it stopped), continue, stop (the job is killed
    and put back in the queue; it restarts from its beginning), skip.
While a job runs the computer is kept from idle-sleeping (runner.keep_awake).  A job
whose result already exists is marked done without running.  The runner exits when
the queue is empty or it is told to stop; finished work is never lost, because every
job writes its own result file.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time

import psutil

from ..config import PROJECT_ROOT
from . import state
from .plan import Job, load_manager_config

TRAINING_SCRIPTS = ("03_pretrain_halo", "04_run_centralized", "05_run_federated", "06_run_ablations")
log = logging.getLogger("manager.runner")


# ---------------------------------------------------------------------- helpers
def keep_awake(on: bool) -> None:
    """Ask Windows not to idle-sleep while a job runs (no effect on other systems)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:                                   # pragma: no cover
        pass


def is_runner(pid) -> bool:
    try:
        p = psutil.Process(int(pid))
        return any("fedpeft.manager.runner" in c for c in p.cmdline())
    except (psutil.Error, ValueError, TypeError):
        return False


def other_training_processes(own: set[int]) -> list[dict]:
    """Training scripts that run outside this runner (e.g. an old .cmd queue).

    `own` and all their descendants are excluded: on Windows the venv's python.exe is a
    launcher whose child process is the real interpreter.
    """
    own = set(own)
    for pid in list(own):
        try:
            own.update(c.pid for c in psutil.Process(pid).children(recursive=True))
        except (psutil.Error, ValueError):
            pass
    found = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] in own or not (p.info["name"] or "").lower().startswith("python"):
                continue
            cmd = " ".join(p.info["cmdline"] or [])
            if any(s in cmd for s in TRAINING_SCRIPTS):
                found.append({"pid": p.info["pid"], "cmd": cmd[-160:]})
        except psutil.Error:
            continue
    return found


def tree(pid: int) -> list[psutil.Process]:
    try:
        p = psutil.Process(pid)
        return [p, *p.children(recursive=True)]
    except psutil.Error:
        return []


def suspend(pid: int) -> None:
    for p in tree(pid):
        try:
            p.suspend()
        except psutil.Error:
            pass


def resume(pid: int) -> None:
    for p in reversed(tree(pid)):
        try:
            p.resume()
        except psutil.Error:
            pass


def kill(pid: int) -> None:
    procs = tree(pid)
    for p in reversed(procs):
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=10)


def tail_lines(path, n: int = 25) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 20000))
            return "\n".join(fh.read().decode("utf-8", "replace").splitlines()[-n:])
    except OSError:
        return ""


# ---------------------------------------------------------------------- runner
class Runner:
    def __init__(self):
        self.status = {"pid": os.getpid(), "started": time.time(), "state": "starting", "current": None,
                       "current_id": None, "child_pid": None, "suspended": False, "note": None,
                       "heartbeat": time.time(), "jobs_run": 0}
        self.rcfg = (load_manager_config().get("runner") or {})
        self.last_beat = 0.0

    def beat(self, force: bool = False, **kw) -> None:
        self.status.update(kw)
        now = time.time()
        if force or kw or now - self.last_beat > 2:
            self.status["heartbeat"] = now
            try:
                state.write_runner_status(self.status)
            except OSError as exc:                         # e.g. a full disk: the job itself goes on
                log.warning(f"cannot write the runner status: {exc}")
            self.last_beat = now

    def free_gb(self) -> float:
        return shutil.disk_usage(PROJECT_ROOT).free / 1e9

    def set_control(self, **kw) -> None:
        state.update(lambda s: s["control"].update(kw))

    def next_item(self):
        def claim(s):
            ctl = s["control"]
            if ctl.get("mode") != "run":
                return None
            it = next((x for x in s["items"] if x["status"] == "pending"), None)
            if it:
                it.update(status="running", started=time.time(), ended=None, rc=None, error=None,
                          attempts=it.get("attempts", 0) + 1, note=None)
            return it
        return state.update(claim)

    def finish(self, key: str, **kw) -> None:
        def fn(s):
            it = state.find(s, key)
            if it:
                it.update(ended=time.time(), **kw)
            ctl = s["control"]
            if ctl.get("after_current") in ("pause", "stop"):
                ctl["mode"], ctl["after_current"] = ctl["after_current"], None
            if kw.get("status") == "failed" and self.rcfg.get("stop_on_error"):
                ctl["mode"] = "pause"
        # The queue file must record the outcome; if the disk is full, wait for space instead of dying.
        for attempt in range(2880):                        # up to a day
            try:
                state.update(fn)
                return
            except OSError as exc:
                if attempt == 0:
                    log.warning(f"cannot write the queue ({exc}); retrying every 30 s")
                time.sleep(30)
        raise RuntimeError("the queue file could not be written for a day")

    def run_item(self, item: dict) -> None:
        job = Job(**item["job"])
        if not item.get("rerun") and job.done():
            self.finish(item["key"], status="done", rc=0, note="result already existed")
            return
        path = state.job_log_path(item)
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = job.command(sys.executable, rerun=item.get("rerun", False))
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        flags = 0x08000000 if os.name == "nt" else 0        # CREATE_NO_WINDOW
        with open(path, "ab") as fh:
            fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  attempt {item.get('attempts', 1)}\n"
                     f"=== {' '.join(cmd)}\n".encode())
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                    creationflags=flags)
        log.info(f"start {item['id']} (pid {proc.pid})")
        self.beat(force=True, state="running", current=item["key"], current_id=item["id"], child_pid=proc.pid,
                  suspended=False, note=None)
        keep_awake(bool(self.rcfg.get("keep_awake", True)))
        outcome = None
        while proc.poll() is None:
            time.sleep(1.0)
            req = state.load()["control"].get("request")
            if req == "suspend" and not self.status["suspended"]:
                suspend(proc.pid)
                keep_awake(False)
                self.set_control(request=None)
                self.beat(force=True, state="suspended", suspended=True)
                log.info(f"suspended {item['id']}")
            elif req == "resume":
                if self.status["suspended"]:
                    resume(proc.pid)
                    keep_awake(bool(self.rcfg.get("keep_awake", True)))
                    log.info(f"resumed {item['id']}")
                self.set_control(request=None)
                self.beat(force=True, state="running", suspended=False)
            elif req in ("kill", "skip"):
                outcome = req
                kill(proc.pid)
                self.set_control(request=None)
                log.info(f"{'stopped' if req == 'kill' else 'skipped'} {item['id']}")
                break
            else:
                self.beat()
        rc = proc.wait()
        keep_awake(False)
        self.status["jobs_run"] += 1
        self.beat(force=True, child_pid=None, suspended=False)
        if outcome == "kill":
            self.finish(item["key"], status="pending", rc=None, started=None,
                        note="stopped by you; it restarts from its beginning")
        elif outcome == "skip":
            self.finish(item["key"], status="skipped", rc=rc, note="skipped while running")
        elif rc == 0 and job.done():
            self.finish(item["key"], status="done", rc=rc)
        else:
            err = tail_lines(path)
            why = f"exit code {rc}" if rc else "the script ended without writing its result"
            self.finish(item["key"], status="failed", rc=rc, error=f"{why}\n{err}")
            log.warning(f"failed {item['id']}: {why}")

    def loop(self) -> None:
        other = self.status.get("pid")
        prev = state.runner_status()
        if prev.get("pid") and prev.get("pid") != other and is_runner(prev["pid"]):
            log.info(f"another runner is active (pid {prev['pid']}); exiting")
            return
        # Jobs left "running" by a runner that died (reboot, crash) go back to the queue.
        def reset(s):
            for it in s["items"]:
                if it["status"] == "running":
                    it.update(status="pending", note="interrupted (the runner stopped); restarts from its beginning")
            if s["control"].get("mode") == "stop":
                s["control"]["mode"] = "run"
        state.update(reset)
        log.info(f"runner started (pid {os.getpid()})")
        self.beat(force=True, state="idle")
        while True:
            ctl = state.load()["control"]
            mode = ctl.get("mode", "run")
            if mode == "stop":
                self.set_control(mode="run", after_current=None, request=None)
                self.beat(force=True, state="stopped", current=None, current_id=None)
                log.info("stopped")
                return
            if mode == "pause":
                self.beat(state="paused", current=None, current_id=None)
                time.sleep(1.0)
                continue
            if self.rcfg.get("wait_for_other_gpu_jobs", True):
                others = other_training_processes({os.getpid()})
                if others:
                    self.beat(state="waiting", note=f"another training process is running (pid {others[0]['pid']})")
                    time.sleep(5.0)
                    continue
            free = self.free_gb()
            if free < float(self.rcfg.get("min_free_gb", 5)):
                self.beat(force=True, state="waiting", current=None, current_id=None,
                          note=f"only {free:.1f} GB free on the results disk; free some space to continue")
                time.sleep(30.0)
                continue
            item = self.next_item()
            if item is None:
                if any(x["status"] == "pending" for x in state.load()["items"]):
                    continue                                   # paused in the meantime
                self.beat(force=True, state="finished", current=None, current_id=None, note="queue is empty")
                log.info("queue is empty; runner exits")
                return
            try:
                self.run_item(item)
            except Exception as exc:                          # never leave a job marked running
                log.exception(f"runner error on {item['id']}")
                self.finish(item["key"], status="failed", error=f"runner error: {exc!r}")


def main() -> None:
    state.STATE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.FileHandler(state.RUNNER_LOG, encoding="utf-8"), logging.StreamHandler()])
    try:
        Runner().loop()
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()

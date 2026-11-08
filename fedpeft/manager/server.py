"""
HTTP API + single-page UI of the pipeline manager (standard library only).

    python manage.py [--port 8765]      ->  http://localhost:8765

The server only reads results / logs and edits the queue and the settings; the
training itself runs in the detached runner (runner.py), so closing the browser or
this server never stops a job.
"""

from __future__ import annotations

import json
import mimetypes
import shutil
import statistics
import os
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..config import CONFIG_DIR, PROJECT_ROOT
from . import profiles, progress, results, state
from .plan import STAGE_LABELS, STAGES, Job, PlanBuilder, load_manager_config
from .runner import is_runner, other_training_processes

STATIC = Path(__file__).parent / "static"
MAIN_METHODS = ["fedavg_scratch", "v0_full_ft", "v1_head_only", "v2_lora_naive", "v3_lora_exact",
                "v4_hetero_rank", "v5_proposed", "ditto_ft", "local_scratch", "ditto_scratch"]
METHOD_SHORT = {"fedavg_scratch": "FedAvg", "v0_full_ft": "V0", "v1_head_only": "V1", "v2_lora_naive": "V2",
                "v3_lora_exact": "V3", "v4_hetero_rank": "V4", "v5_proposed": "V5", "ditto_ft": "Ditto-FT",
                "local_scratch": "Local", "ditto_scratch": "Ditto", "scratch": "Central (scratch)",
                "pretrained_ft": "Central (pretrained)", "v5_abl_no_local_norm": "V5 − norm/adapter",
                "v5_abl_shared_head": "V5 − personal head", "v5_abl_fixed_rank": "V5 fixed rank 8"}
ENCODER_NAMES = {"limu_bert": "LIMU-BERT", "ssl_wearables": "ssl-wearables", "unimts": "UniMTS", "halo": "HALO",
                 "mlp": "MLP", "lstm": "LSTM", "convgru": "Conv-GRU"}


# ---------------------------------------------------------------------- plan cache
class PlanCache:
    """The plan is rebuilt only when a settings file changes (or every 5 minutes)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.stamp = None
        self.plan = None
        self.builder = None
        self.built = 0.0

    def _stamp(self):
        files = sorted(CONFIG_DIR.rglob("*.yaml"))
        return tuple((str(f), f.stat().st_mtime_ns) for f in files)

    def get(self):
        with self.lock:
            stamp = self._stamp()
            if self.plan is None or stamp != self.stamp or time.time() - self.built > 300:
                self.builder = PlanBuilder(results.all_summaries())
                self.plan = self.builder.build()
                self.stamp, self.built = stamp, time.time()
            return self.plan


PLANS = PlanCache()


# ---------------------------------------------------------------------- runner
def runner_info() -> dict:
    st = state.runner_status()
    alive = bool(st.get("pid")) and is_runner(st["pid"])
    st["alive"] = alive
    st["heartbeat_age"] = time.time() - st["heartbeat"] if st.get("heartbeat") else None
    return st


def launch_runner() -> dict:
    """Start the runner as a detached process (it survives this server and the terminal)."""
    info = runner_info()
    if info["alive"]:
        return {"started": False, "pid": info["pid"]}
    py = sys.executable
    if os.name == "nt":
        # A WMI-created process is outside this process's job object, so it outlives it.
        cmdline = f'"{py}" -m fedpeft.manager.runner'
        ps = ("$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
              f"@{{CommandLine='{cmdline}'; CurrentDirectory='{PROJECT_ROOT}'}}; $r.ProcessId")
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=60, creationflags=0x08000000)
        pid = (out.stdout or "").strip().splitlines()[-1] if out.stdout.strip() else None
    else:
        p = subprocess.Popen([py, "-m", "fedpeft.manager.runner"], cwd=PROJECT_ROOT, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pid = p.pid
    return {"started": True, "pid": pid}


# ---------------------------------------------------------------------- views
def queue_index(s: dict) -> dict:
    """{job id: latest queue item} (the newest item of a job wins)."""
    out = {}
    for it in s["items"]:
        out[it["id"]] = it
    return out


def job_status(job: Job, qi: dict | None) -> str:
    if qi and qi["status"] == "running":
        return "running"
    if job.done():
        return "done"
    if qi and qi["status"] in ("pending", "failed"):
        return "queued" if qi["status"] == "pending" else "failed"
    return "missing"


def job_view(job: Job, qi: dict | None, summaries: dict) -> dict:
    d = {k: getattr(job, k) for k in ("id", "label", "stage", "dataset", "profile", "encoder", "method", "tag", "seed",
                                       "group", "est_min", "total_steps", "step_unit")}
    d["status"] = job_status(job, qi)
    d["queue_key"] = qi["key"] if qi else None
    d["result"] = summaries.get(job.outputs[0]) if job.outputs else None
    d["command"] = "python " + " ".join(["scripts/" + job.script, *job.args])
    return d


def calibration(s: dict) -> tuple[float, int]:
    """Wall time / estimate of the jobs this queue finished.

    The estimates come from the run times stored in the result files, which leave out
    starting Python, loading the data and the new-user tests; the factor corrects for that.
    """
    ratios = [(it["ended"] - it["started"]) / 60 / it["job"]["est_min"] for it in s["items"]
              if it["status"] == "done" and it.get("started") and it.get("ended") and it["job"].get("est_min")
              and it.get("note") != "result already existed" and it["job"]["stage"] != "report"]
    if len(ratios) < 3:
        return 1.0, len(ratios)
    return min(3.0, max(1.0, statistics.median(ratios))), len(ratios)


def overview() -> dict:
    plan = PLANS.get()
    s = state.load()
    qi = queue_index(s)
    factor, n_cal = calibration(s)
    mcfg = plan.mcfg
    datasets = []
    missing_min = 0.0
    for ds in plan.datasets():
        dcfg = mcfg["datasets"][ds]
        stages = []
        for st in STAGES:
            jobs = [j for j in plan.jobs.values() if j.dataset == ds and j.stage == st]
            if not jobs:
                continue
            stat = [job_status(j, qi.get(j.id)) for j in jobs]
            miss = factor * sum((j.est_min or 10.0) for j, x in zip(jobs, stat) if x != "done")
            missing_min += miss
            stages.append({"stage": st, "label": STAGE_LABELS[st], "total": len(jobs),
                           "done": stat.count("done"), "queued": stat.count("queued"), "running": stat.count("running"),
                           "failed": stat.count("failed"), "missing_min": miss})
        datasets.append({"key": ds, "name": dcfg["name"], "profile": dcfg["profile"], "stages": stages})
    info = runner_info()
    current = None
    if info["alive"] and info.get("current"):
        it = state.find(s, info["current"])
        if it:
            current = current_view(it, factor)
    pending = [it for it in s["items"] if it["status"] == "pending"]
    queued_min = factor * sum((it["job"].get("est_min") or 10.0) for it in pending)
    others = [] if info.get("state") == "running" else _other_gpu(info)
    return {"time": time.time(), "runner": info, "control": s["control"], "counts": state.counts(s),
            "current": current, "datasets": datasets, "missing_min": missing_min, "queued_min": queued_min,
            "estimate_factor": factor, "estimate_jobs": n_cal, "other_gpu": others, "order": plan.datasets(),
            "disk_free_gb": shutil.disk_usage(PROJECT_ROOT).free / 1e9}


_OTHER = {"t": 0.0, "v": []}


def _other_gpu(info: dict) -> list:
    """Training processes outside the runner (scanned at most every 10 s)."""
    if time.time() - _OTHER["t"] > 10:
        own = {info.get("pid") or -1, info.get("child_pid") or -1}
        _OTHER.update(t=time.time(), v=other_training_processes(own))
    return _OTHER["v"]


def current_view(it: dict, factor: float = 1.0) -> dict:
    job = it["job"]
    prog = progress.parse(state.job_log_path(it), job["script"], job.get("total_steps"))
    elapsed = time.time() - it["started"] if it.get("started") else None
    eta = None
    if job.get("est_min") and elapsed:
        eta = max(0.0, factor * job["est_min"] * 60 - elapsed)
    if prog["total"] and prog["step"] and elapsed and prog["phase"] == "train":
        # time per round so far, plus what comes after training (personalisation, new-user tests)
        after = max(0.0, (factor - 1.0) * job["est_min"] * 60) if job.get("est_min") else 0.0
        eta = elapsed / prog["step"] * (prog["total"] - prog["step"]) + after
    return {"item": it, "progress": prog, "elapsed_s": elapsed, "eta_s": eta}


def plan_view(dataset: str) -> dict:
    plan = PLANS.get()
    s = state.load()
    qi = queue_index(s)
    summ = results.all_summaries()
    stages = []
    for st in STAGES:
        jobs = [j for j in plan.jobs.values() if j.dataset == dataset and j.stage == st]
        if jobs:
            stages.append({"stage": st, "label": STAGE_LABELS[st],
                           "jobs": [job_view(j, qi.get(j.id), summ) for j in jobs]})
    return {"dataset": dataset, "stages": stages}


def queue_view() -> dict:
    s = state.load()
    info = runner_info()
    items = []
    for it in s["items"]:
        d = {k: it.get(k) for k in ("key", "id", "status", "added", "started", "ended", "attempts", "rc", "note", "rerun")}
        j = it["job"]
        d.update(label=j["label"], dataset=j["dataset"], profile=j["profile"], stage=j["stage"],
                 est_min=j.get("est_min"), error=(it.get("error") or "")[-3000:] or None,
                 command="python " + " ".join(["scripts/" + j["script"], *j["args"]]))
        items.append(d)
    return {"items": items, "control": s["control"], "runner": info, "counts": state.counts(s)}


def _seed_mean(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def matrix_view() -> dict:
    """Encoders x datasets: every method's scores in the main profile of each dataset."""
    plan = PLANS.get()
    mcfg = plan.mcfg
    summ = results.all_summaries()
    s = state.load()
    qi = queue_index(s)
    by = {}
    for r in summ.values():
        if not r.get("tag") or r["kind"] == "centralized":
            by.setdefault((r["profile"], r["kind"], r["encoder"], r["method"]), []).append(r)
    datasets, encoders = [], []
    for ds in plan.datasets():
        prof = mcfg["datasets"][ds]["profile"]
        cfg = PLANS.builder.cfgs.get(prof)
        datasets.append({"key": ds, "name": mcfg["datasets"][ds]["name"], "profile": prof,
                         "encoders": list(cfg.encoders.names)})
        for e in cfg.encoders.names:
            if e not in encoders:
                encoders.append(e)
    cells = {}
    for d in datasets:
        for enc in d["encoders"]:
            for kind, methods in (("centralized", ["scratch", "pretrained_ft"]), ("federated", MAIN_METHODS)):
                for m in methods:
                    rows = by.get((d["profile"], kind, enc, m), [])
                    jid = (f"central/{d['profile']}/{enc}/{m}/s0" if kind == "centralized"
                           else f"fl/{d['profile']}/{enc}/{m}/s0")
                    job = plan.jobs.get(jid)
                    status = job_status(job, qi.get(jid)) if job else ("done" if rows else "n/a")
                    if not rows and status == "n/a":
                        continue
                    cells[f"{d['key']}|{enc}|{m}"] = {
                        "status": status, "seeds": sorted(r["seed"] for r in rows),
                        **{k: _seed_mean(rows, k) for k in ("macro_f1", "ft_macro_f1", "client_acc", "ft_client_acc",
                                                             "worst10_f1", "zero_shot_f1", "uplink_mb", "minutes")},
                        "seed0": next((r for r in rows if r["seed"] == 0), None)}
    return {"datasets": datasets, "encoders": [{"key": e, "name": ENCODER_NAMES.get(e, e)} for e in encoders],
            "methods": [{"key": m, "short": METHOD_SHORT.get(m, m), "kind": "centralized"} for m in ("scratch", "pretrained_ft")]
                       + [{"key": m, "short": METHOD_SHORT.get(m, m), "kind": "federated"} for m in MAIN_METHODS],
            "cells": cells}


def ablation_view(dataset: str) -> dict:
    plan = PLANS.get()
    mcfg = plan.mcfg
    dcfg = mcfg["datasets"][dataset]
    prof = dcfg["profile"]
    cfg = PLANS.builder.cfgs.get(prof)
    summ = results.all_summaries()
    s = state.load()
    qi = queue_index(s)
    variants = PLANS.builder.ablation_variants(prof, cfg)
    base = f"results/{prof}/federated"
    out = []
    encs = [e for e in (mcfg.get("ablation") or {}).get("encoders") or [] if e in cfg.encoders.names]
    for enc in encs:
        def res(method, tag="", seed=0):
            suffix = f"__{tag}" if tag else ""
            return summ.get(f"{base}/{enc}__{method}{suffix}__seed{seed}.json")

        def row(label, group, method, tag="", seed=0, jid=None, note=""):
            r = res(method, tag, seed)
            job = plan.jobs.get(jid) if jid else None
            status = job_status(job, qi.get(jid)) if job else ("done" if r else "missing")
            return {"label": label, "group": group, "method": method, "tag": tag, "seed": seed, "result": r,
                    "status": status, "job_id": jid if job else None, "note": note}
        v5 = res("v5_proposed")
        rows = [row("V5 (complete)", "reference", "v5_proposed", jid=f"fl/{prof}/{enc}/v5_proposed/s0")]
        default_rank = int(cfg.model.lora.rank)
        ranks = sorted({int(v["rank"]) for v in variants if v["encoder"] == enc and v["group"] == "rank"} | {default_rank})
        for rk in ranks:
            if rk == default_rank:
                rows.append(row(f"V3, LoRA rank {rk} (main V3)", "rank", "v3_lora_exact",
                                jid=f"fl/{prof}/{enc}/v3_lora_exact/s0"))
            else:
                rows.append(row(f"V3, LoRA rank {rk}", "rank", "v3_lora_exact", tag=f"r{rk}",
                                jid=f"rank/{prof}/{enc}/r{rk}/s0"))
        for v in variants:
            if v["encoder"] != enc or v["group"] == "rank" or v["seed"] != 0:
                continue
            label = METHOD_SHORT.get(v["label"], v["label"])
            jid = f"fl/{prof}/{enc}/{v['method']}{'+' + v['tag'] if v['tag'] else ''}/s0"
            rows.append(row(label, v["group"], v["method"], v["tag"], jid=jid))
        grid = [("V4 (no local parts, no prototypes)", "v4_hetero_rank"), ("V2 (naive LoRA averaging)", "v2_lora_naive"),
                ("V1 (head only)", "v1_head_only"), ("V0 (full fine-tuning)", "v0_full_ft"),
                ("FedAvg from scratch (no pretraining)", "fedavg_scratch")]
        for label, m in grid:
            rows.append(row(label, "grid", m, jid=f"fl/{prof}/{enc}/{m}/s0"))
        # tagged runs of this encoder that the study above does not define (earlier experiments)
        known = {(x["method"], x["tag"]) for x in rows}
        for r in summ.values():
            if (r["profile"] == prof and r["kind"] == "federated" and r["encoder"] == enc and r["seed"] == 0
                    and r.get("tag") and (r["method"], r["tag"]) not in known):
                rows.append({"label": f"{METHOD_SHORT.get(r['method'], r['method'])} [{r['tag']}]", "group": "other",
                             "method": r["method"], "tag": r["tag"], "seed": 0, "result": r, "status": "done",
                             "job_id": None, "note": "earlier experiment"})
        for x in rows:
            ref = v5 if x["group"] != "rank" else res("v3_lora_exact")
            f1 = (x["result"] or {}).get("macro_f1")
            x["delta_f1"] = f1 - ref["macro_f1"] if f1 is not None and ref and ref.get("macro_f1") is not None else None
        out.append({"encoder": enc, "name": ENCODER_NAMES.get(enc, enc), "rows": rows})
    return {"dataset": dataset, "name": dcfg["name"], "profile": prof, "encoders": out,
            "definition": mcfg.get("ablation")}


# ---------------------------------------------------------------------- actions
def enqueue(body: dict) -> dict:
    plan = PLANS.get()
    rerun = bool(body.get("rerun"))
    if body.get("ids"):
        jobs = [plan.jobs[i] for i in body["ids"] if i in plan.jobs]
        if not rerun:
            jobs = [j for j in jobs if not j.done()]
    else:
        stages = body.get("stages")
        jobs = plan.select(datasets=body.get("datasets"), stages=stages, from_stage=body.get("from_stage"),
                           only_missing=not rerun)
    added = state.update(lambda s: state.enqueue(s, [j.to_dict() for j in jobs], rerun=rerun,
                                                 front=bool(body.get("front"))))
    started = None
    if body.get("start"):
        state.update(lambda s: s["control"].update(mode="run", after_current=None))
        started = launch_runner()
    return {"added": added, "requested": len(jobs), "runner": started}


def control(body: dict) -> dict:
    action = body.get("action")
    info = runner_info()

    def fn(s):
        c = s["control"]
        if action == "start":
            c.update(mode="run", after_current=None)
            if info.get("suspended"):
                c["request"] = "resume"
        elif action == "pause_after":
            c["after_current"] = "pause"
        elif action == "pause_now":
            c["mode"] = "pause"
            if info.get("alive") and info.get("child_pid"):
                c["request"] = "suspend"
        elif action == "resume":
            c.update(mode="run", after_current=None, request="resume" if info.get("suspended") else None)
        elif action == "stop_after":
            c["after_current"] = "stop"
        elif action == "stop_now":
            c["mode"] = "stop"
            if info.get("alive") and info.get("child_pid"):
                c["request"] = "kill"
        elif action == "skip_current":
            if info.get("alive") and info.get("child_pid"):
                c["request"] = "skip"
        elif action == "cancel_after":
            c["after_current"] = None
        else:
            raise ValueError(f"unknown action {action}")
    state.update(fn)
    started = launch_runner() if action in ("start", "resume") else None
    return {"ok": True, "runner": started}


def item_action(body: dict) -> dict:
    key, action = body.get("key"), body.get("action")

    def fn(s):
        it = state.find(s, key)
        if not it:
            raise ValueError("no such queue item")
        if action == "skip" and it["status"] == "pending":
            it.update(status="skipped", ended=time.time(), note="skipped by you")
        elif action == "retry" and it["status"] in state.FINISHED:
            it.update(status="pending", error=None, rc=None, note="queued again")
            state.move(s, key, "end")
        elif action == "rerun":
            it.update(status="pending", error=None, rc=None, rerun=True, note="re-run (overwrites the result)")
        elif action == "remove" and it["status"] != "running":
            s["items"].remove(it)
        elif action in ("up", "down", "top", "end") and it["status"] == "pending":
            state.move(s, key, action)
        elif action == "start_from":
            return state.start_from(s, key)
        return None
    n = state.update(fn)
    if action == "start_from":
        state.update(lambda s: s["control"].update(mode="run", after_current=None))
        launch_runner()
    return {"ok": True, "skipped": n}


def clear(body: dict) -> dict:
    what = body.get("what", "finished")

    def fn(s):
        keep = []
        for it in s["items"]:
            if it["status"] == "running":
                keep.append(it)
            elif what == "finished" and it["status"] not in state.FINISHED:
                keep.append(it)
            elif what == "pending" and it["status"] != "pending":
                keep.append(it)
        removed = len(s["items"]) - len(keep)
        s["items"] = keep
        return removed
    return {"removed": state.update(fn)}


# ---------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "FedPEFT-Manager/1.0"

    def log_message(self, fmt, *args):            # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code: int = 200) -> None:
        self._send(code, json.dumps(data, default=str).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if url.path.startswith("/static/"):
                f = (STATIC / url.path[len("/static/"):]).resolve()
                if STATIC.resolve() not in f.parents or not f.exists():
                    return self._send(404, b"not found", "text/plain")
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                if ctype.startswith("text/") or ctype.endswith("javascript"):
                    ctype += "; charset=utf-8"
                return self._send(200, f.read_bytes(), ctype)
            routes = {
                "/api/overview": lambda: overview(),
                "/api/plan": lambda: plan_view(q["dataset"]),
                "/api/queue": lambda: queue_view(),
                "/api/matrix": lambda: matrix_view(),
                "/api/ablations": lambda: ablation_view(q["dataset"]),
                "/api/profiles": lambda: {"profiles": profiles.list_profiles(), "roles": {}},
                "/api/profile": lambda: profiles.detail(q["name"]),
                "/api/manager_config": lambda: {"text": (CONFIG_DIR / "manager.yaml").read_text(encoding="utf-8")},
                "/api/progress": lambda: self._progress(q["key"]),
                "/api/log": lambda: self._log(q["key"], int(q.get("lines", 400))),
                "/api/runner_log": lambda: {"text": progress.tail(state.RUNNER_LOG, int(q.get("lines", 200)))},
            }
            if url.path in routes:
                return self._json(routes[url.path]())
            return self._send(404, b"not found", "text/plain")
        except KeyError as exc:
            return self._json({"error": f"missing or unknown {exc}"}, 400)
        except Exception as exc:
            return self._json({"error": str(exc), "trace": traceback.format_exc()[-2000:]}, 500)

    def _progress(self, key: str) -> dict:
        s = state.load()
        it = state.find(s, key)
        if not it:
            raise KeyError(key)
        return current_view(it)

    def _log(self, key: str, lines: int) -> dict:
        s = state.load()
        it = state.find(s, key)
        if not it:
            raise KeyError(key)
        return {"text": progress.tail(state.job_log_path(it), lines)}

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = self._body()
            routes = {
                "/api/enqueue": enqueue,
                "/api/control": control,
                "/api/item": item_action,
                "/api/clear": clear,
                "/api/profile/save": lambda b: profiles.save_text(b["kind"], b.get("name", ""), b["text"]),
                "/api/profile/set": lambda b: profiles.set_value(b["name"], b["key"], b.get("value"), bool(b.get("unset"))),
                "/api/profile/duplicate": lambda b: profiles.duplicate(b["name"], b["new"]),
            }
            if url.path in routes:
                return self._json(routes[url.path](body))
            return self._send(404, b"not found", "text/plain")
        except (ValueError, KeyError, FileNotFoundError) as exc:
            return self._json({"error": str(exc)}, 400)
        except Exception as exc:
            return self._json({"error": str(exc), "trace": traceback.format_exc()[-2000:]}, 500)


def serve(port: int = 8765, host: str = "127.0.0.1") -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=PLANS.get, daemon=True).start()      # build the plan before the first request
    print(f"FedPEFT-IMU pipeline manager: http://localhost:{port}  (Ctrl+C stops the UI; jobs keep running)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

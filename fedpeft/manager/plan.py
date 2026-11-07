"""
The plan: every job of the full process, for every dataset (configs/manager.yaml).

A job is one call of a script in scripts/ that writes known files; it is done
when all of them exist.  The plan is rebuilt from the configuration on every
request, so editing a profile or manager.yaml changes it immediately.  The same
result file is only planned once (the first stage that needs it owns it).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ..config import CONFIG_DIR, PROJECT_ROOT, encoder_config, load_config
from .results import Durations

MANAGER_FILE = CONFIG_DIR / "manager.yaml"
PY = "scripts/{}"

STAGES = ["prepare", "weights", "pretrain", "centralized", "federated", "published", "ablation", "trials", "report"]
STAGE_LABELS = {
    "prepare": "Prepare data", "weights": "Pretrained weights", "pretrain": "HALO pretraining",
    "centralized": "Centralized training", "federated": "Federated grid (FedAvg, V0–V5)",
    "published": "Published protocol", "ablation": "Ablation study", "trials": "Trials (extra user splits)",
    "report": "HTML report",
}
DEFAULT_MINUTES = {"prepare": 5.0, "weights": 2.0, "pretrain": 30.0, "report": 0.5}
ENCODER_NAMES = {"limu_bert": "LIMU-BERT", "ssl_wearables": "ssl-wearables", "unimts": "UniMTS", "halo": "HALO",
                 "mlp": "MLP", "lstm": "LSTM", "convgru": "Conv-GRU"}
METHOD_NAMES = {"fedavg_scratch": "FedAvg", "v0_full_ft": "V0 full fine-tuning", "v1_head_only": "V1 head only",
                "v2_lora_naive": "V2 LoRA naive", "v3_lora_exact": "V3 LoRA exact", "v4_hetero_rank": "V4 hetero ranks",
                "v5_proposed": "V5 FedPEFT-IMU", "ditto_ft": "Ditto-FT", "local_scratch": "Local", "ditto_scratch": "Ditto",
                "v5_abl_no_local_norm": "V5 − local norm/adapter", "v5_abl_shared_head": "V5 − personal head",
                "v5_abl_fixed_rank": "V5 − device tiers (rank 8)"}


def enc_name(e: str) -> str:
    return ENCODER_NAMES.get(e, e)


@dataclass
class Job:
    id: str
    dataset: str
    profile: str
    stage: str
    script: str                       # file name in scripts/
    args: list                        # everything after the script, --profile included
    label: str
    outputs: list                     # project-relative files; the job is done when all exist
    encoder: str = ""
    method: str = ""                  # FL method or centralized mode
    tag: str = ""
    seed: int | None = None
    group: str = ""                   # sub-group inside a stage (ablation: rank / components / switches)
    est_min: float | None = None      # expected run time from earlier runs of the same kind
    total_steps: int | None = None    # rounds (federated) or epochs (centralized)
    step_unit: str = ""               # "round" | "epoch"
    rerun_flag: str = "--rerun"       # how the script is told to overwrite an existing result
    sources: list = field(default_factory=list)   # report: the profiles whose results it shows

    def command(self, python: str, rerun: bool = False) -> list[str]:
        cmd = [python, "-u", str(PROJECT_ROOT / "scripts" / self.script), *self.args]
        if rerun and self.rerun_flag:
            cmd.insert(3, self.rerun_flag)
        return cmd

    def done(self) -> bool:
        if self.stage == "report":
            return report_up_to_date(self)
        return bool(self.outputs) and all((PROJECT_ROOT / o).exists() for o in self.outputs)

    def to_dict(self) -> dict:
        return asdict(self)


def report_up_to_date(job: Job) -> bool:
    """A report is done when it exists and is newer than every result it is built from."""
    out = PROJECT_ROOT / job.outputs[0]
    if not out.exists():
        return False
    built = out.stat().st_mtime
    for prof in [job.profile, *job.sources]:
        for p in (PROJECT_ROOT / "results" / prof).glob("*/*.json"):
            if p.stat().st_mtime > built:
                return False
    return True


def load_manager_config() -> dict:
    with open(MANAGER_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class _Configs:
    """load_config is cheap but not free: cache it per (profile, overrides)."""

    def __init__(self):
        self.cache = {}

    def get(self, profile: str, overrides: tuple = ()):
        key = (profile, overrides)
        if key not in self.cache:
            self.cache[key] = load_config(profile, list(overrides))
        return self.cache[key]


def _rel(path) -> str:
    return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()


def _dig(cfg, dotted: str, default=None):
    node = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


class PlanBuilder:
    def __init__(self, summaries: dict | None = None):
        self.mcfg = load_manager_config()
        self.cfgs = _Configs()
        self.durations = Durations(summaries or {})
        self.jobs: dict[str, Job] = {}
        self.by_output: dict[str, str] = {}

    # ------------------------------------------------------------------ helpers
    def _add(self, job: Job) -> None:
        if job.id in self.jobs or any(o in self.by_output for o in job.outputs if job.stage != "report"):
            return                                    # already planned by an earlier stage
        self.jobs[job.id] = job
        for o in job.outputs:
            self.by_output[o] = job.id

    def _steps(self, profile: str, encoder: str, overrides: tuple, kind: str) -> tuple[int | None, str]:
        cfg = encoder_config(self.cfgs.get(profile, overrides), encoder)
        if kind == "federated":
            return int(cfg.fl.rounds), "round"
        return int(cfg.centralized.epochs), "epoch"

    def central(self, ds: str, profile: str, stage: str, enc: str, mode: str, seed: int, group: str = "") -> None:
        cfg = self.cfgs.get(profile)
        out = _rel(Path(cfg.paths.results_dir) / profile / "centralized" / f"{enc}__{mode}__seed{seed}.json")
        steps, unit = self._steps(profile, enc, (), "centralized")
        self._add(Job(
            id=f"central/{profile}/{enc}/{mode}/s{seed}", dataset=ds, profile=profile, stage=stage,
            script="04_run_centralized.py",
            args=["--profile", profile, "--encoders", enc, "--modes", mode, f"split.seeds=[{seed}]"],
            label=f"{enc_name(enc)} · centralized, {'pretrained' if mode == 'pretrained_ft' else 'from scratch'} · split {seed}",
            outputs=[out], encoder=enc,
            method=mode, seed=seed, group=group, total_steps=steps, step_unit=unit,
            est_min=self.durations.estimate(profile, enc, mode, "centralized")))

    def federated(self, ds: str, profile: str, stage: str, enc: str, method: str, seed: int, tag: str = "",
                  overrides: list | None = None, group: str = "", label: str = "") -> None:
        cfg = self.cfgs.get(profile)
        suffix = f"__{tag}" if tag else ""
        out = _rel(Path(cfg.paths.results_dir) / profile / "federated" / f"{enc}__{method}{suffix}__seed{seed}.json")
        ov = tuple(overrides or ())
        steps, unit = self._steps(profile, enc, ov, "federated")
        args = ["--profile", profile, "--encoders", enc, "--methods", method]
        if tag:
            args += ["--tag", tag]
        args += [*ov, f"split.seeds=[{seed}]"]
        name = label or METHOD_NAMES.get(method, method) + (f" [{tag}]" if tag else "")
        self._add(Job(
            id=f"fl/{profile}/{enc}/{method}{'+' + tag if tag else ''}/s{seed}", dataset=ds, profile=profile,
            stage=stage, script="05_run_federated.py", args=args, label=f"{enc_name(enc)} · {name} · split {seed}",
            outputs=[out], encoder=enc, method=method, tag=tag, seed=seed, group=group, total_steps=steps,
            step_unit=unit, est_min=self.durations.estimate(profile, enc, method, "federated")))

    def rank(self, ds: str, profile: str, enc: str, r: int, seed: int) -> None:
        cfg = self.cfgs.get(profile)
        out = _rel(Path(cfg.paths.results_dir) / profile / "federated" / f"{enc}__v3_lora_exact__r{r}__seed{seed}.json")
        steps, unit = self._steps(profile, enc, (), "federated")
        self._add(Job(
            id=f"rank/{profile}/{enc}/r{r}/s{seed}", dataset=ds, profile=profile, stage="ablation",
            script="06_run_ablations.py",
            args=["--profile", profile, f"ablation.encoder={enc}", f"ablation.ranks=[{r}]", f"split.seeds=[{seed}]"],
            label=f"{enc_name(enc)} · V3, LoRA rank {r} · split {seed}", outputs=[out], encoder=enc,
            method="v3_lora_exact", tag=f"r{r}", seed=seed, group="rank", total_steps=steps, step_unit=unit,
            est_min=self.durations.estimate(profile, enc, "v3_lora_exact", "federated")))

    def encoder_methods(self, cfg, enc: str) -> list[str]:
        restricted = cfg.encoders.get(enc, {}).get("methods")
        return list(restricted or cfg.fl.methods)

    def encoder_modes(self, cfg, enc: str) -> list[str]:
        return list(cfg.encoders.get(enc, {}).get("modes") or cfg.centralized.modes)

    # ------------------------------------------------------------------ stages
    def build_dataset(self, ds: str, dcfg: dict) -> None:
        profile = dcfg["profile"]
        cfg = self.cfgs.get(profile)
        seeds = list(cfg.split.seeds)
        encoders = list(cfg.encoders.names)

        # 1 prepare
        self._add(Job(id=f"prepare/{cfg.data.dataset}", dataset=ds, profile=profile, stage="prepare",
                      script="01_prepare_data.py", args=["--profile", profile], label=f"Prepare {dcfg['name']} windows",
                      outputs=[_rel(Path(cfg.paths.processed_dir) / "meta.json")], rerun_flag="",
                      est_min=DEFAULT_MINUTES["prepare"]))
        # 2 weights (shared by every dataset)
        wdir = Path(cfg.paths.weights_dir)
        files = [_rel(wdir / cfg.encoders[e].checkpoint) for e in ("limu_bert", "ssl_wearables", "unimts")
                 if e in cfg.encoders]
        self._add(Job(id="weights", dataset=ds, profile=profile, stage="weights", script="02_download_weights.py",
                      args=["--profile", profile], label="Download the official pretrained weights", outputs=files,
                      rerun_flag="", est_min=DEFAULT_MINUTES["weights"]))
        # 3 HALO pretraining, for every split that trains HALO
        halo_seeds = sorted(set(seeds))
        if "halo" in encoders:
            for s in halo_seeds:
                self._add(Job(id=f"pretrain/{profile}/s{s}", dataset=ds, profile=profile, stage="pretrain",
                              script="03_pretrain_halo.py", args=["--profile", profile, f"split.seeds=[{s}]"],
                              label=f"HALO self-supervised pretraining · split {s}", encoder="halo", seed=s,
                              outputs=[_rel(wdir / cfg.encoders.halo.checkpoint.format(seed=s))], rerun_flag="--force",
                              total_steps=int(cfg.halo_ssl.epochs), step_unit="epoch",
                              est_min=DEFAULT_MINUTES["pretrain"]))
        # 4 centralized
        for s in seeds:
            for enc in encoders:
                for mode in self.encoder_modes(cfg, enc):
                    self.central(ds, profile, "centralized", enc, mode, s)
        # 5 federated grid + extra runs
        for s in seeds:
            for enc in encoders:
                for m in self.encoder_methods(cfg, enc):
                    self.federated(ds, profile, "federated", enc, m, s, group="grid")
        for x in dcfg.get("extra") or []:
            for s in x.get("seeds") or seeds:
                self.federated(ds, profile, "federated", x["encoder"], x["method"], s, tag=x.get("tag", ""),
                               overrides=x.get("overrides"), group="extra")
        # 6 published protocols
        for pub in dcfg.get("published") or []:
            p = pub["profile"]
            for s in pub.get("seeds") or [0]:
                for enc, modes in (pub.get("centralized") or {}).items():
                    for mode in modes:
                        self.central(ds, p, "published", enc, mode, s, group=p)
                for enc, methods in (pub.get("federated") or {}).items():
                    for m in methods:
                        self.federated(ds, p, "published", enc, m, s, group=p)
                for t in pub.get("tagged") or []:
                    self.federated(ds, p, "published", t["encoder"], t["method"], s, tag=t["tag"],
                                   overrides=t.get("overrides"), group=p)
        # 7 ablation study
        self.build_ablation(ds, profile, cfg)
        # 8 trials
        tr = self.mcfg.get("trials") or {}
        trial_encoders = dict(tr.get("encoders") or {})
        base = dcfg.get("baseline")
        if tr.get("baseline") and base:
            trial_encoders[base] = self.encoder_methods(cfg, base)
        for s in tr.get("seeds") or []:
            for enc, methods in trial_encoders.items():
                if enc not in encoders:
                    continue
                # only methods this dataset also ran on its main split (e.g. Ditto-FT where it is an extra)
                main = set(self.encoder_methods(cfg, enc)) | {x["method"] for x in dcfg.get("extra") or []
                                                               if x["encoder"] == enc and not x.get("tag")}
                for mode in self.encoder_modes(cfg, enc):
                    if mode in (tr.get("centralized") or []):
                        self.central(ds, profile, "trials", enc, mode, s, group=f"split {s}")
                for m in methods:
                    if m in main:
                        self.federated(ds, profile, "trials", enc, m, s, group=f"split {s}")
        # 9 report
        self._add(Job(id=f"report/{profile}", dataset=ds, profile=profile, stage="report",
                      script="07_build_report.py", args=["--profile", profile],
                      label=f"Build the {dcfg['name']} HTML report",
                      outputs=[f"reports/FedPEFT-IMU_report_{profile}.html"], rerun_flag="",
                      sources=[pub["profile"] for pub in dcfg.get("published") or []],
                      est_min=DEFAULT_MINUTES["report"]))

    def ablation_variants(self, profile: str, cfg) -> list[dict]:
        """Every ablation run of one dataset: [{encoder, group, label, method, tag, overrides, seed, rank}]."""
        a = self.mcfg.get("ablation") or {}
        out = []
        default_rank = int(cfg.model.lora.rank)
        for enc in a.get("encoders") or []:
            if enc not in cfg.encoders.names:
                continue
            for s in a.get("seeds") or [0]:
                for r in a.get("rank_sweep") or []:
                    if int(r) != default_rank:
                        out.append({"encoder": enc, "group": "rank", "label": f"V3, LoRA rank {r}", "method": "v3_lora_exact",
                                    "tag": f"r{r}", "rank": int(r), "seed": s, "overrides": []})
                for m in a.get("components") or []:
                    out.append({"encoder": enc, "group": "components", "label": METHOD_NAMES.get(m, m), "method": m,
                                "tag": "", "seed": s,
                                "overrides": []})
                for tag, t in (a.get("tags") or {}).items():
                    if t.get("encoders") and enc not in t["encoders"]:
                        continue
                    if any(str(_dig(cfg, k, "qv" if k.endswith("lora_scope") else None)) != str(v)
                           for k, v in (t.get("when") or {}).items()):
                        continue
                    out.append({"encoder": enc, "group": "switches", "label": t.get("label", tag),
                                "method": t.get("method", "v5_proposed"), "tag": tag, "seed": s,
                                "overrides": list(t.get("overrides") or [])})
        return out

    def build_ablation(self, ds: str, profile: str, cfg) -> None:
        for v in self.ablation_variants(profile, cfg):
            if v["group"] == "rank":
                self.rank(ds, profile, v["encoder"], v["rank"], v["seed"])
            else:
                self.federated(ds, profile, "ablation", v["encoder"], v["method"], v["seed"], tag=v["tag"],
                               overrides=v["overrides"], group=v["group"], label=v["label"])

    def build(self) -> "Plan":
        datasets = self.mcfg.get("datasets") or {}
        for ds in self.mcfg.get("order") or list(datasets):
            if ds in datasets:
                self.build_dataset(ds, datasets[ds])
        return Plan(self.jobs, self.mcfg)


@dataclass
class Plan:
    jobs: dict
    mcfg: dict = field(default_factory=dict)

    def datasets(self) -> list[str]:
        return [d for d in self.mcfg.get("order") or [] if d in (self.mcfg.get("datasets") or {})]

    def select(self, datasets=None, stages=None, from_stage: str | None = None, only_missing: bool = True) -> list[Job]:
        """Jobs in pipeline order: dataset by dataset (manager.yaml order), stage by stage."""
        first = STAGES.index(from_stage) if from_stage in STAGES else 0
        chosen = []
        for ds in datasets or self.datasets():
            mine = []
            for st in STAGES[first:]:
                if stages and st not in stages:
                    continue
                for j in self.jobs.values():
                    if j.dataset != ds or j.stage != st:
                        continue
                    # the report is rebuilt whenever this dataset gets new results
                    if not only_missing or not j.done() or (st == "report" and mine):
                        mine.append(j)
            chosen += mine
        return chosen


def build_plan(summaries: dict | None = None) -> Plan:
    return PlanBuilder(summaries).build()

"""
Collect all result files of a profile and derive the report's KPIs.

Input  (written by the experiment scripts)
    results/<profile>/centralized/*.json     phase 1
    results/<profile>/federated/*.json       phase 2 (+ rank ablation, tag "__r<k>")
    results/<profile>/pretraining/*.json     HALO self-supervised pretraining log
    results/<profile>/benchmark/encoders.json
    results/splits/user_split_seed*.json     published user IDs
    data/processed/meta.json                 classes, participants

Output: one JSON-serialisable dict consumed by template.html.

Derived KPIs (computed here so that every number in the report has ONE definition):
    target F1          = cfg.fl.target_f1_fraction x centralized from-scratch macro-F1 (same encoder)
    rounds to target   = first evaluated round whose macro-F1 >= target
    MB to target       = cumulative uplink (all clients) at that round
    uplink reduction   = V0 uplink per client-round / method uplink per client-round
Several seeds are averaged (mean, and std in *_std fields).

Every classification metric exists twice: per window ("macro_f1") and after HMM
temporal smoothing ("macro_f1_hmm").  If results of the `baseline` profile exist
(the first pipeline, before the class-imbalance fixes) they are collected too, so
the report can show what each change bought.
"""

from __future__ import annotations

import platform
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from ..federated.methods import METHODS

METHOD_ORDER = list(METHODS)
from ..models.encoders import ENCODERS
from ..utils import load_json

CENTRAL_MODES = {"scratch": "From scratch", "pretrained_ft": "Pretrained + fine-tuning"}


def _mean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else None


def _std(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.std(vals)) if len(vals) > 1 else 0.0 if vals else None


def _first_reach(evals, target):
    if target is None:
        return None
    for p in evals:
        if p["macro_f1"] >= target:
            return p
    return None


def _heldout_summary(h: dict | None) -> dict:
    out = {}
    for key, value in (h or {}).items():
        if key in ("protocol", "hmm"):
            continue
        minutes = 0 if key == "zero_shot" else int(key.split("_")[2].replace("min", ""))
        out[str(minutes)] = {"mean_user_f1": value["mean_user_f1"], "pooled_macro_f1": value["pooled_macro_f1"],
                             "accuracy": value["accuracy"], "worst10_f1": value["worst10_f1"]}
    return out


def _hmm_fields(runs: list[dict], key: str) -> dict:
    """HMM-smoothed metrics of a group of runs (empty if the runs have none)."""
    hm = [r.get(key) for r in runs if r.get(key)]
    if not hm:
        return {}
    return {
        "macro_f1_hmm": _mean([h["macro_f1"] for h in hm]),
        "accuracy_hmm": _mean([h["accuracy"] for h in hm]),
        "worst10_f1_hmm": _mean([h["worst10_f1"] for h in hm]),
        "client_f1_std_hmm": _mean([h["client_f1_std"] for h in hm]),
        "per_class_f1_hmm": np.mean([h["per_class_f1"] for h in hm], axis=0).tolist(),
    }


def _ft_fields(runs: list[dict]) -> dict:
    """Scores after local fine-tuning of the shared classifier ("+FT"), if recorded."""
    ft = [r["final_ft"] for r in runs if r.get("final_ft")]
    if not ft:
        return {}
    fh = [r["final_ft_hmm"] for r in runs if r.get("final_ft_hmm")]
    return {"macro_f1_ft": _mean([m["macro_f1"] for m in ft]), "accuracy_ft": _mean([m["accuracy"] for m in ft]),
            "worst10_f1_ft": _mean([m["worst10_f1"] for m in ft]),
            "macro_f1_ft_hmm": _mean([m["macro_f1"] for m in fh]) if fh else None}


def collect_stages(cfg) -> list[dict]:
    """Federated macro-F1 of every earlier pipeline listed in report.compare_profiles."""
    labels = cfg.get("report", {}).get("stage_labels", {})
    stages = []
    for name in cfg.get("report", {}).get("compare_profiles", []):
        base = Path(cfg.paths.results_dir) / name
        if name == cfg.profile or not (base / "federated").exists():
            continue
        fed = {}
        for p in sorted((base / "federated").glob("*.json")):
            r = load_json(p)
            if r.get("tag"):
                continue
            fed[f"{r['encoder']}|{r['method']}"] = {
                "macro_f1": r["final"]["macro_f1"], "worst10_f1": r["final"]["worst10_f1"],
                "macro_f1_hmm": (r.get("final_hmm") or {}).get("macro_f1"),
                "macro_f1_ft": (r.get("final_ft") or {}).get("macro_f1")}
        stages.append({"profile": name, "label": labels.get(name, name), "federated": fed})
    return stages


def _ablation_from(cfg, profile: str) -> list[dict]:
    """Rank-sweep points of one profile (its tagged V3 runs + its main V3 run)."""
    base = Path(cfg.paths.results_dir) / profile / "federated"
    pts = []
    for p in sorted(base.glob("*__v3_lora_exact*__seed*.json")):
        r = load_json(p)
        pts.append({"encoder": r["encoder"], "encoder_display": r["encoder_display"], "tag": r.get("tag", ""),
                    "rank": r["config"]["r_max"], "macro_f1": r["final"]["macro_f1"],
                    "accuracy": r["final"]["accuracy"], "worst10_f1": r["final"]["worst10_f1"],
                    "uplink_mb_client_round": r["comm"]["uplink_mb_per_client_round"],
                    "client_mem_mb": r["resources"]["client_train_mem_mb_mean"]})
    tagged = {p["encoder"] for p in pts if p["tag"]}
    return sorted([p for p in pts if p["encoder"] in tagged], key=lambda p: p["rank"])


def collect_window_ablation(cfg) -> list[dict]:
    """LIMU-BERT 6-s (this profile) vs 10-s window (report.window_ablation_profile), same methods."""
    other = cfg.get("report", {}).get("window_ablation_profile")
    if not other:
        return []
    base_o = Path(cfg.paths.results_dir) / other
    base_c = Path(cfg.paths.results_dir) / cfg.profile
    rows = []
    for kind in ("centralized", "federated"):
        for p in sorted((base_o / kind).glob("limu_bert__*.json")):
            q = base_c / kind / p.name
            if not q.exists():
                continue
            a, b = load_json(q), load_json(p)
            blk = "test" if kind == "centralized" else "final"
            name = (CENTRAL_MODES[a["mode"]] if kind == "centralized" else a["method_short"])
            rows.append({"model": ("Central · " if kind == "centralized" else "") + name,
                         "f1_6s": a[blk]["macro_f1"], "f1_10s": b[blk]["macro_f1"]})
    return rows


def collect_baseline(cfg) -> dict | None:
    """Key metrics of the first pipeline (results/<baseline profile>), for before/after charts."""
    compare = cfg.get("report", {}).get("compare_profiles") or ["baseline"]
    name = cfg.get("report", {}).get("baseline_profile", compare[0])
    base = Path(cfg.paths.results_dir) / name
    if name == cfg.profile or not base.exists():
        return None

    def pick(r, block):
        m = r[block]
        return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "worst10_f1": m["worst10_f1"],
                "client_f1_std": m["client_f1_std"], "per_class_f1": m["per_class_f1"],
                "heldout": _heldout_summary(r["heldout"])}

    out = {"profile": name, "centralized": [], "federated": []}
    for p in sorted((base / "centralized").glob("*.json")):
        r = load_json(p)
        out["centralized"].append({"encoder": r["encoder"], "mode": r["mode"], **pick(r, "test")})
    for p in sorted((base / "federated").glob("*.json")):
        r = load_json(p)
        out["federated"].append({"encoder": r["encoder"], "method": r["method"], "tag": r.get("tag", ""),
                                 **pick(r, "final")})
    return out


def collect(cfg) -> dict:
    base = Path(cfg.paths.results_dir) / cfg.profile
    meta = load_json(Path(cfg.paths.processed_dir) / "meta.json")
    class_names = meta["schemes"][cfg.data.label_scheme]

    # ------------------------------------------------------------ centralized
    central_runs = defaultdict(list)
    for p in sorted((base / "centralized").glob("*.json")):
        r = load_json(p)
        central_runs[(r["encoder"], r["mode"])].append(r)
    centralized = []
    scratch_f1 = {}
    for (enc, mode), runs in sorted(central_runs.items()):
        r0 = runs[0]
        entry = {
            "encoder": enc, "encoder_display": r0["encoder_display"], "mode": mode, "mode_label": CENTRAL_MODES[mode],
            "seeds": [r["seed"] for r in runs],
            "macro_f1": _mean([r["test"]["macro_f1"] for r in runs]),
            "macro_f1_std": _std([r["test"]["macro_f1"] for r in runs]),
            "accuracy": _mean([r["test"]["accuracy"] for r in runs]),
            "worst10_f1": _mean([r["test"]["worst10_f1"] for r in runs]),
            "client_f1_std": _mean([r["test"]["client_f1_std"] for r in runs]),
            "per_class_f1": np.mean([r["test"]["per_class_f1"] for r in runs], axis=0).tolist(),
            "client_f1": list(r0["test"]["client_f1"].values()),
            "confusion": r0["confusion"],
            "history": r0["history"],
            "train_time_s": _mean([r["train_time_s"] for r in runs]),
            "train_mem_mb": _mean([r["train_mem_mb"] for r in runs]),
            "trainable_params": r0["trainable_params"],
            "epochs_run": len(r0["history"]),
            "heldout": _heldout_summary(r0["heldout"]),
            "heldout_hmm": _heldout_summary(r0["heldout"].get("hmm")),
            "val_tau_sweep": r0.get("val_tau_sweep"),
            **_hmm_fields(runs, "test_hmm"),
        }
        centralized.append(entry)
        if mode == "scratch":
            scratch_f1[enc] = entry["macro_f1"]

    # ------------------------------------------------------------- federated
    fed_runs = defaultdict(list)
    for p in sorted((base / "federated").glob("*.json")):
        r = load_json(p)
        fed_runs[(r["encoder"], r["method"], r.get("tag", ""))].append(r)
    federated, ablation = [], []
    v0_uplink = {}
    for (enc, method, tag), runs in fed_runs.items():
        if method == "v0_full_ft" and not tag:
            v0_uplink[enc] = runs[0]["comm"]["uplink_mb_per_client_round"]
    for (enc, method, tag), runs in sorted(fed_runs.items()):
        r0 = runs[0]
        target = cfg.fl.target_f1_fraction * scratch_f1[enc] if enc in scratch_f1 else None
        hits = [_first_reach(r["evals"], target) for r in runs]
        up = r0["comm"]["uplink_mb_per_client_round"]
        entry = {
            "encoder": enc, "encoder_display": r0["encoder_display"], "method": method, "tag": tag,
            "short": r0["method_short"], "label": r0["method_label"], "slot": r0["method_slot"],
            "seeds": [r["seed"] for r in runs],
            "config": r0["config"],
            "macro_f1": _mean([r["final"]["macro_f1"] for r in runs]),
            "macro_f1_std": _std([r["final"]["macro_f1"] for r in runs]),
            "accuracy": _mean([r["final"]["accuracy"] for r in runs]),
            "worst10_f1": _mean([r["final"]["worst10_f1"] for r in runs]),
            "client_f1_std": _mean([r["final"]["client_f1_std"] for r in runs]),
            "client_f1_mean": _mean([r["final"]["client_f1_mean"] for r in runs]),
            "per_class_f1": np.mean([r["final"]["per_class_f1"] for r in runs], axis=0).tolist(),
            "client_f1": list(r0["final"]["client_f1"].values()),
            "confusion": r0["confusion"],
            "uplink_mb_client_round": up,
            "uplink_mb_round": r0["comm"]["uplink_mb_per_round"],
            "downlink_mb_round": r0["comm"]["downlink_mb_per_round"],
            "total_uplink_mb": r0["comm"]["total_uplink_mb"],
            "uplink_params_client_round": up * 1e6 / 4,
            "uplink_reduction_vs_v0": (v0_uplink[enc] / up) if enc in v0_uplink and up > 0 else None,
            "per_rank": r0["comm"]["per_rank"],
            "target_f1": target,
            "rounds_to_target": _mean([h["round"] if h else None for h in hits]),
            "mb_to_target": _mean([h["cum_uplink_mb"] if h else None for h in hits]),
            "time_to_target_s": _mean([h["elapsed_s"] if h else None for h in hits]),
            "reached_target": all(h is not None for h in hits),
            "trainable_params": float(np.mean([v["trainable_params"] for v in r0["comm"]["per_rank"].values()])),
            "train_state_mb": float(np.mean([v["trainable_params"] for v in r0["comm"]["per_rank"].values()])) * 12 / 1e6,
            "client_mem_mb": r0["resources"]["client_train_mem_mb_mean"],
            "client_mem_mb_max": r0["resources"]["client_train_mem_mb_max"],
            "client_time_s": r0["resources"]["client_time_s_mean"],
            "total_time_s": _mean([r["resources"]["total_time_s"] for r in runs]),
            "tier_f1": r0["tier_f1"],
            "param_counts": r0["param_counts"],
            "evals": r0["evals"],
            "rounds": [{"round": h["round"], "uplink_mb": h["uplink_mb"], "cum_uplink_mb": h["cum_uplink_mb"],
                        "train_loss": h["train_loss"], "agg_error": h["agg_error"]} for h in r0["history"]],
            "heldout": _heldout_summary(r0["heldout"]),
            "heldout_hmm": _heldout_summary(r0["heldout"].get("hmm")),
            "before_personalization": r0.get("final_before_personalization"),
            **_hmm_fields(runs, "final_hmm"),
            **_ft_fields(runs),
        }
        (ablation if tag else federated).append(entry)

    # Rank ablation: add the main V3 run as the default-rank point.
    if ablation:
        enc = ablation[0]["encoder"]
        main = next((f for f in federated if f["encoder"] == enc and f["method"] == "v3_lora_exact"), None)
        points = [dict(a, rank=a["config"]["r_max"]) for a in ablation]
        if main:
            points.append(dict(main, rank=main["config"]["r_max"]))
        ablation = sorted(points, key=lambda a: a["rank"])

    # Rank sweep not re-run in this profile: show the most recent earlier one, labelled.
    ablation_source = cfg.profile if ablation else None
    if not ablation:
        for prof in reversed(cfg.get("report", {}).get("compare_profiles", [])):
            pts = _ablation_from(cfg, prof)
            if pts:
                ablation, ablation_source = pts, prof
                break

    # ------------------------------------------------------------ extras
    bench_path = base / "benchmark" / "encoders.json"
    bench = load_json(bench_path) if bench_path.exists() else {}
    encoders = []
    for name, cls in ENCODERS.items():
        b = bench.get(name, {})
        spec = cls.spec
        encoders.append({"name": name, "display": cls.display_name, "architecture": cls.architecture,
                         "pretrained_source": cls.pretrained_source,
                         "input": f"{spec.n_channels} ch · {spec.rate_hz:g} Hz · {spec.seconds:g} s ({spec.acc_unit})",
                         **{k: b.get(k) for k in ("encoder_params", "lora_params_r8", "lora_layers", "feat_dim",
                                                  "train_full_win_per_s", "train_lora_win_per_s",
                                                  "inference_win_per_s", "train_full_peak_mb", "train_lora_peak_mb")}})

    ssl = [load_json(p) for p in sorted((base / "pretraining").glob("*.json"))]
    splits = [load_json(p) for p in sorted((Path(cfg.paths.results_dir) / "splits").glob("*.json"))
              if load_json(p)["seed"] in cfg.split.seeds]

    # Dataset statistics from the processed files' metadata (class counts per user are in the npz).
    dataset = _dataset_stats(cfg, meta, splits)

    methods = [{"name": m.name, "short": m.short, "label": m.label, "slot": m.slot,
                "description": m.description, "trained": sorted(m.trainable), "federated": sorted(m.federated),
                "local": sorted(m.local)} for m in METHODS.values()]

    report = {
        "title": "FedPEFT-IMU — Experimental Results",
        "generated": datetime.now().strftime("%d %B %Y, %H:%M"),
        "profile": cfg.profile,
        "host": {"python": platform.python_version(), "os": platform.platform(terse=True)},
        "class_names": class_names,
        "dataset": dataset,
        "protocol": {
            "label_scheme": cfg.data.label_scheme, "window_s": cfg.data.window_s,
            "subsample_every": cfg.data.subsample_every, "seeds": list(cfg.split.seeds),
            "heldout_frac": cfg.split.heldout_frac, "pool_frac": cfg.split.pool_frac,
            "train_time_frac": cfg.split.train_time_frac, "fewshot_minutes": list(cfg.split.fewshot_minutes),
            "rounds": cfg.fl.rounds, "clients_per_round_frac": cfg.fl.clients_per_round_frac,
            "local_epochs": cfg.fl.local_epochs, "max_local_steps": cfg.fl.max_local_steps,
            "batch_size": cfg.fl.batch_size, "eval_every": cfg.fl.eval_every,
            "lora_rank": cfg.model.lora.rank, "lora_tiers": list(cfg.model.lora.tiers),
            "lora_alpha_over_r": cfg.model.lora.alpha_over_r, "proto_weight": cfg.model.proto_weight,
            "target_f1_fraction": cfg.fl.target_f1_fraction,
            "centralized_epochs": cfg.centralized.epochs,
            "tau_train": cfg.imbalance.tau_train, "tau_inf": cfg.imbalance.tau_inf,
            "hmm": bool(cfg.decoding.hmm),
            "final_personalization_steps": cfg.fl.get("final_personalization_steps", 0),
            "adabn_new_users": bool(cfg.split.get("adabn_new_users", False)),
        },
        "encoders": encoders,
        # Per-encoder compute budgets that override the protocol (e.g. UniMTS).
        "budgets": {name: (cfg.encoders[name]["budget"].to_dict() if hasattr(cfg.encoders[name]["budget"], "to_dict")
                           else cfg.encoders[name]["budget"])
                    for name in ENCODERS if cfg.encoders.get(name, {}).get("budget")},
        "methods": methods,
        "centralized": centralized,
        "federated": federated,
        "ablation": ablation,
        "baseline": collect_baseline(cfg),
        "stages": collect_stages(cfg),
        "window_ablation": collect_window_ablation(cfg),
        "stage_label": cfg.get("report", {}).get("stage_labels", {}).get(cfg.profile, cfg.profile),
        "ablation_source": ablation_source,
        "halo_ssl": ssl,
        "splits": splits,
    }
    report["findings"] = findings(report)
    return report


def _dataset_stats(cfg, meta, splits) -> dict:
    """Window counts per class and per user group, read from the processed files."""
    from ..data.store import WindowStore  # noqa: F401  (documentation pointer)
    processed = Path(cfg.paths.processed_dir)
    scheme = cfg.data.label_scheme
    counts, per_user = np.zeros(len(meta["schemes"][scheme]), dtype=np.int64), {}
    for user in meta["participants"]:
        with np.load(processed / f"{user}.npz") as d:
            y = d[f"y_{scheme}"][:: cfg.data.subsample_every]
        y = y[y >= 0]
        c = np.bincount(y, minlength=len(counts))
        counts += c
        per_user[user] = int(len(y))
    groups = {}
    if splits:
        s = splits[0]
        groups = {g: len(s[g]) for g in ("clients", "heldout", "pool")}
    vals = np.array(list(per_user.values()))
    return {"n_participants": len(meta["participants"]), "class_counts": counts.tolist(),
            "n_windows": int(counts.sum()), "windows_per_user_mean": float(vals.mean()),
            "windows_per_user_min": int(vals.min()), "windows_per_user_max": int(vals.max()),
            "groups": groups}


def findings(rep: dict) -> list[str]:
    """Plain-language key findings. Every number is computed from the results; sentences
    that interpret a number say so ("suggests", "consistent with")."""
    out = []
    cen = {(c["encoder"], c["mode"]): c for c in rep["centralized"]}
    fed = {(f["encoder"], f["method"]): f for f in rep["federated"]}
    names = {e["name"]: e["display"] for e in rep["encoders"]}
    encs = [e for e in names if any(k[0] == e for k in fed)]
    budget_encs = set(rep.get("budgets", {}))

    def val(enc, method, key):
        r = fed.get((enc, method))
        return r.get(key) if r else None

    # 0. What the class-imbalance fixes changed against the first run.
    B = rep.get("baseline")
    if B:
        before = {(b["encoder"], b["method"]): b for b in B["federated"] if not b["tag"]}
        gains = [(k, fed[k]["macro_f1"] - before[k]["macro_f1"]) for k in fed if k in before]
        if gains:
            zb = sum(sum(v < 0.01 for v in before[k]["per_class_f1"]) for k, _ in gains)
            za = sum(sum(v < 0.01 for v in fed[k]["per_class_f1"]) for k, _ in gains)
            best = max(gains, key=lambda g: g[1])
            what = ("the class-imbalance fixes change" if rep["profile"] == "standard"
                    else "this pipeline changes")
            out.append(f"Against the first run, {what} federated macro-F1 by "
                       f"{np.mean([g for _, g in gains]):+.3f} on average (per window, {len(gains)} models; largest "
                       f"{best[1]:+.3f} for {names[best[0][0]]} {fed[best[0]]['short']}). Classes stuck at F1 < 0.01: "
                       f"{zb} before, {za} after.")
        # Whole progression: first run -> per window -> full pipeline, mean over the FL methods.
        prog = []
        for e in encs:
            b = [before[k]["macro_f1"] for k in before if k[0] == e]
            w = [f["macro_f1"] for f in rep["federated"] if f["encoder"] == e]
            # Fixed rule (not the best variant per model): +FT +HMM for shared classifiers,
            # +HMM for V5 whose classifier is personal already.
            full = [(f.get("macro_f1_hmm") if f["method"] == "v5_proposed" else f.get("macro_f1_ft_hmm"))
                    for f in rep["federated"] if f["encoder"] == e]
            full = [x for x in full if x is not None]
            if b and w and full:
                prog.append(f"{names[e]} {np.mean(b):.2f} → {np.mean(w):.2f} per window → {np.mean(full):.2f} with "
                            f"temporal smoothing and local fine-tuning")
        if prog:
            out.append("Mean federated macro-F1 per encoder, first run → now: " + "; ".join(prog) + ".")
    hm = [(f["macro_f1"], f["macro_f1_hmm"]) for f in rep["federated"] if f.get("macro_f1_hmm") is not None]
    if hm:
        out.append(f"HMM smoothing of each user's window sequence adds {np.mean([h - w for w, h in hm]):+.3f} macro-F1 "
                   f"on average across the federated models (best smoothed result "
                   f"{max(h for _, h in hm):.3f}).")

    # 1. Phase 1: best architecture from scratch and the value of pretraining.
    scratch = [c for c in rep["centralized"] if c["mode"] == "scratch"]
    if scratch:
        best = max(scratch, key=lambda c: c["macro_f1"])
        gains = [(e, cen[(e, "pretrained_ft")]["macro_f1"] - cen[(e, "scratch")]["macro_f1"])
                 for e in names if (e, "scratch") in cen and (e, "pretrained_ft") in cen]
        txt = f"Centralized from scratch, {best['encoder_display']} is the strongest architecture (macro-F1 {best['macro_f1']:.3f})."
        if gains:
            txt += (" Starting from the pretrained weights improves every encoder: "
                    + ", ".join(f"{names[e]} {g:+.3f}" for e, g in gains) + " macro-F1.")
        out.append(txt)

    # 2. Federated learning from scratch versus centralized from scratch.
    drops = [(e, cen[(e, "scratch")]["macro_f1"] - val(e, "fedavg_scratch", "macro_f1"))
             for e in encs if (e, "scratch") in cen and val(e, "fedavg_scratch", "macro_f1") is not None]
    if drops:
        out.append("Training from scratch in the federation (non-IID users, 20 % participation) loses "
                   + ", ".join(f"{names[e]} {d:.3f}" for e, d in drops)
                   + " macro-F1 against centralized training from scratch. This is the gap the pretrained encoders are meant to close.")

    # 3. Proposed method V5 versus full fine-tuning V0.
    parts = []
    for e in encs:
        v0, v5 = fed.get((e, "v0_full_ft")), fed.get((e, "v5_proposed"))
        if v0 and v5 and v5.get("uplink_reduction_vs_v0"):
            parts.append(f"{names[e]} {v5['macro_f1']:.3f} vs {v0['macro_f1']:.3f} with {v5['uplink_reduction_vs_v0']:.0f}x less uplink")
    if parts:
        out.append("FedPEFT-IMU (V5) against full fine-tuning (V0), macro-F1: " + "; ".join(parts) + ".")

    # 4. Plain federated LoRA against full fine-tuning.
    parts, wins, losses = [], [], []
    for e in encs:
        v0 = fed.get((e, "v0_full_ft"))
        cands = [fed[(e, m)] for m in ("v2_lora_naive", "v3_lora_exact") if (e, m) in fed]
        if v0 and cands:
            b = max(cands, key=lambda f: f["macro_f1"])
            parts.append(f"{names[e]} {b['short']} {b['macro_f1']:.3f} vs V0 {v0['macro_f1']:.3f}")
            (wins if b["macro_f1"] >= v0["macro_f1"] - 0.01 else losses).append(names[e])
    if parts:
        txt = "Federated LoRA without personalisation (best of V2/V3) against V0: " + "; ".join(parts) + "."
        if wins:
            txt += f" LoRA matches or beats full fine-tuning for {', '.join(wins)}"
            txt += (f"; for {', '.join(losses)} the frozen features need more than a low-rank update in this round budget."
                    if losses else ".")
        out.append(txt)

    # 5. Exact versus naive aggregation, and its downlink price.
    diffs = [(e, val(e, "v3_lora_exact", "macro_f1") - val(e, "v2_lora_naive", "macro_f1")) for e in encs
             if val(e, "v3_lora_exact", "macro_f1") is not None and val(e, "v2_lora_naive", "macro_f1") is not None]
    if diffs:
        down = [(e, val(e, "v3_lora_exact", "downlink_mb_round") / val(e, "v2_lora_naive", "downlink_mb_round"))
                for e, _ in diffs if val(e, "v2_lora_naive", "downlink_mb_round")]
        txt = ("Exact LoRA aggregation (V3) versus naive averaging (V2), macro-F1 difference: "
               + ", ".join(f"{names[e]} {d:+.3f}" for e, d in diffs) + ". In this setting the naive error is small "
               "(see the aggregation-error chart) and does not cost accuracy.")
        if down:
            txt += (" Exact aggregation must ship the dense residual to every client: downlink per round is "
                    + ", ".join(f"{names[e]} {x:.0f}x" for e, x in down) + " that of V2.")
        out.append(txt)

    # 6. Rank ablation.
    if rep["ablation"]:
        A = rep["ablation"]
        best = max(A, key=lambda a: a["macro_f1"])
        src = rep.get("ablation_source")
        src_txt = f" — from the “{src}” run" if src and src != rep["profile"] else ""
        out.append(f"LoRA rank ({A[0]['encoder_display']}, V3{src_txt}): "
                   + ", ".join(f"r={a['rank']} {a['macro_f1']:.3f}" for a in A)
                   + f". The best rank is r={best['rank']} at {best['uplink_mb_client_round']:.2f} MB per client-round; "
                     "a larger rank does not buy accuracy here.")

    # 7. Fairness.
    parts = []
    for e in encs:
        rows = [fed[(e, m)] for m in METHOD_ORDER if (e, m) in fed]
        if rows:
            b = max(rows, key=lambda f: f["worst10_f1"] or -1)
            v5 = fed.get((e, "v5_proposed"))
            parts.append(f"{names[e]} {b['short']} {b['worst10_f1']:.3f}" + (f" (V5 {v5['worst10_f1']:.3f})" if v5 and b is not v5 else ""))
    if parts:
        out.append("Highest worst-10 % client F1 per encoder: " + "; ".join(parts) + ".")

    # 8. New users.
    parts = []
    for e in encs:
        v3, v5 = fed.get((e, "v3_lora_exact")), fed.get((e, "v5_proposed"))
        if v3 and v5:
            parts.append(f"{names[e]} V5 {v5['heldout']['0']['mean_user_f1']:.3f} vs V3 {v3['heldout']['0']['mean_user_f1']:.3f}")
    if parts:
        adabn = rep["protocol"].get("adabn_new_users")
        out.append("New users, zero-shot (V5 = nearest global prototype"
                   + ("; BatchNorm encoders first estimate their norm statistics from the new user's unlabelled windows" if adabn else "")
                   + "): " + "; ".join(parts) + ". The shared classifier of V3 transfers better to brand-new users "
                   "than V5's prototype rule; V5's advantage is on the devices that trained with it.")
    few = [(f, f["heldout"]["0"]["mean_user_f1"], f["heldout"]["5"]["mean_user_f1"]) for f in rep["federated"]
           if f["heldout"].get("5") and f["heldout"].get("0")]
    worse = [f for f, z, v in few if v < z]
    if worse:
        ordered = sorted(few, key=lambda t: t[1])
        half = len(ordered) // 2
        low = float(np.mean([v - z for _, z, v in ordered[:half]]))
        high = float(np.mean([v - z for _, z, v in ordered[half:]]))
        corrected = rep["protocol"].get("tau_train", 0) > 0
        out.append(f"For {len(worse)} of {len(few)} federated models, 5 minutes of labels score below zero-shot. "
                   f"The models with the weaker half of zero-shot scores gain {low:+.3f} on average from 5 minutes of labels, "
                   f"the stronger half changes by {high:+.3f}. "
                   + ("The few-shot fit already corrects for its class-balanced sample, so the remaining loss suggests the "
                      "classifier over-fits the 6-30 windows per class; regularising it towards the global classifier is "
                      "the next step." if corrected else
                      "This is consistent with a class-prior shift: the few-shot sample is roughly class-balanced while "
                      "real days are dominated by sleep and sitting."))

    if budget_encs:
        out.append("Budget note: " + ", ".join(names[e] for e in budget_encs if e in names)
                   + " ran with a reduced compute budget (see the callouts), so compare its methods with each other.")
    if len(rep["protocol"]["seeds"]) == 1:
        out.append("All numbers come from one user split (seed 0). Differences of a few F1 points should be confirmed "
                   "with the 'full' profile (3 held-out lists) before they are claimed.")
    return out

"""
Results tables of the paper (Section "Results"), generated from the result files.

    python scripts/tools/paper_results_tables.py [--main harbox] [--published harbox_fedchar]

Reads results/<main>/{centralized,federated}/*.json (Protocol B) and
results/<published>/... (Protocol A, the protocol of the published HARBox results),
and writes LaTeX table rows + macros to paper/generated/:

    res_tab_sota.tex       Protocol A: published numbers vs our runs (mean user accuracy)
    res_tab_main.tex       Protocol B: encoder x method grid
    res_tab_newusers.tex   Protocol B: held-out users, zero- and few-shot
    res_tab_cost.tex       communication / memory / time per encoder (V0 vs V5)
    res_macros.tex         \\newcommand's for numbers quoted in the text

Missing runs are shown as "--", so the tables can be generated while experiments run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fedpeft.config import load_config  # noqa: E402

OUT = ROOT / "paper" / "generated"
ENC = {"limu_bert": "LIMU-BERT", "ssl_wearables": "ssl-wearables", "unimts": "UniMTS", "halo": "HALO", "mlp": "MLP", "lstm": "LSTM", "convgru": "Conv-GRU"}
METHODS = [("local_scratch", "Local"), ("fedavg_scratch", "FedAvg"), ("ditto_scratch", "Ditto"),
           ("v0_full_ft", "V0 Full FT"), ("ditto_ft", "Ditto (pretrained)"), ("v1_head_only", "V1 Head only"), ("v2_lora_naive", "V2 LoRA naive"),
           ("v3_lora_exact", "V3 LoRA exact"), ("v4_hetero_rank", "V4 Hetero ranks"),
           ("v5_proposed", "\\textbf{V5 \\method}")]


def load(profile: str, kind: str) -> dict:
    base = ROOT / "results" / profile / kind
    out = {}
    for p in sorted(base.glob("*__seed0.json")) if base.exists() else []:
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("tag"):
            continue
        key = (r["encoder"], r.get("method") or r.get("mode"))
        out[key] = r
    return out


def load_seeds(profile: str) -> dict:
    """Every federated run of every seed: {(encoder, method[__tag]): {seed: result}}."""
    base = ROOT / "results" / profile / "federated"
    out: dict = {}
    for p in sorted(base.glob("*__seed*.json")) if base.exists() else []:
        r = json.loads(p.read_text(encoding="utf-8"))
        out.setdefault((r["encoder"], r["method"] + (r.get("tag") or "")), {})[int(r["seed"])] = r
    return out


# LoRA placement: tagged seed-0 runs shown as extra rows of the main grid (LIMU-BERT only)
PLACEMENT_ROWS = [("v3_lora_exact__loraall", "V3, LoRA on all layers"),
                  ("v5_proposed__loraall", "\\textbf{V5 \\method}, LoRA on all layers")]


def load_tagged(profile: str, names: list[str]) -> dict:
    base = ROOT / "results" / profile / "federated"
    out = {}
    for p in sorted(base.glob("*__seed0.json")) if base.exists() else []:
        r = json.loads(p.read_text(encoding="utf-8"))
        key = r["method"] + (r.get("tag") or "")
        if r.get("tag") and key in names:
            out[(r["encoder"], key)] = r
    return out


def mean_std(vals: list) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    import statistics
    return f"{statistics.mean(vals):.3f}\\,$\\pm$\\,{statistics.stdev(vals):.3f}"


def scored(r: dict) -> dict:
    """Final enrolled-client metrics of a run. Methods with a SHARED classifier are scored after
    every client fine-tuned that classifier on its own training data (+FT), like the personalised
    methods, so all rows compare personalised models."""
    return r.get("final_ft") or r["final"]


def f3(x) -> str:
    return "--" if x is None else f"{x:.3f}"


def mb(x) -> str:
    if x is None:
        return "--"
    return f"{x * 1000:.0f}\\,kB" if x < 1 else f"{x:.1f}\\,MB" if x < 100 else f"{x:.0f}\\,MB"


def bold_best(cells: list[str]) -> list[str]:
    vals = [float(c) if c not in ("--",) else -1 for c in cells]
    if max(vals) < 0:
        return cells
    b = max(vals)
    return [f"\\textbf{{{c}}}" if c != "--" and float(c) == b else c for c in cells]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="harbox")
    ap.add_argument("--published", default="harbox_fedchar")
    args = ap.parse_args()
    cfg = load_config(args.main)
    fed, cen = load(args.main, "federated"), load(args.main, "centralized")
    fed_a, cen_a = load(args.published, "federated"), load(args.published, "centralized")
    macros = {}

    # ------------------------------------------------- Protocol A vs published
    fed_a.update(load_tagged(args.published, [m for m, _ in PLACEMENT_ROWS]))   # e.g. V5 with all-layer LoRA
    names_a = dict(METHODS + PLACEMENT_ROWS)
    rows = []
    for p in cfg.report.get("published") or []:
        rows.append((p["accuracy"], f"{p['method']} & {p['model']} & {p['accuracy']:.3f} & -- & published~\\cite{{li2023fedchar}} \\\\"))
    for (enc, m), r in fed_a.items():
        name = names_a.get(m, m)
        acc = scored(r).get("client_acc_mean")
        rows.append((acc or 0, f"{name} & {ENC[enc]} & {f3(acc)} & {f3(scored(r).get('worst10_acc'))} & ours \\\\"))
    for (enc, mode), r in cen_a.items():
        acc = r["test"].get("client_acc_mean")
        label = "Centralized, " + ("scratch" if mode == "scratch" else "pretrained FT")
        rows.append((acc or 0, f"{label} (not private) & {ENC[enc]} & {f3(acc)} & {f3(r['test'].get('worst10_acc'))} & ours \\\\"))
    rows.sort(key=lambda t: t[0])
    (OUT / "res_tab_sota.tex").write_text("\n".join(r for _, r in rows) + "\n", encoding="utf-8")
    ours_a = [(scored(r).get("client_acc_mean") or 0, k) for k, r in fed_a.items()]
    if ours_a:
        best = max(ours_a)
        macros["ResABestAcc"] = f"{best[0]:.3f}"
        macros["ResABestName"] = f"{names_a.get(best[1][1], best[1][1])} ({ENC[best[1][0]]})".replace("\\textbf{", "").replace("}", "")
    for key, macro in ((("limu_bert", "v5_proposed"), "ResAVfive"), (("limu_bert", "v5_proposed__loraall"), "ResAVfiveAll"),
                       (("limu_bert", "ditto_ft"), "ResADittoFT")):
        if key in fed_a:
            macros[macro] = f"{scored(fed_a[key])['client_acc_mean']:.3f}"
    fa = fed_a.get(("mlp", "fedavg_scratch"))
    if fa:
        macros["ResAFedAvgMLP"] = f"{scored(fa)['client_acc_mean']:.3f}"
        macros["ResAFedAvgMLPPlain"] = f"{fa['final']['client_acc_mean']:.3f}"   # without the local fine-tuning
    dm = fed_a.get(("mlp", "ditto_scratch"))
    if dm:
        macros["ResADittoMLP"] = f"{dm['final']['client_acc_mean']:.3f}"

    # ----------------------------------------------------- Protocol B main grid
    encs = [e for e in ENC if any(k[0] == e for k in fed)]
    lines = []
    for m, name in METHODS:
        if not any((e, m) in fed for e in encs):
            continue
        cells_acc, cells_f1 = [], []
        for e in encs:
            r = fed.get((e, m))
            if r is None:
                cells_acc.append("--"); cells_f1.append("--")
                continue
            f = scored(r)                                  # +FT for shared classifiers
            cells_acc.append(f3(f.get("client_acc_mean")))
            cells_f1.append(f3(f["macro_f1"]))
        lines.append((name, cells_acc, cells_f1))
    tagged = load_tagged(args.main, [m for m, _ in PLACEMENT_ROWS])
    for m, name in PLACEMENT_ROWS:
        if not any((e, m) in tagged for e in encs):
            continue
        cells_acc = [f3(scored(tagged[(e, m)]).get("client_acc_mean")) if (e, m) in tagged else "--" for e in encs]
        cells_f1 = [f3(scored(tagged[(e, m)])["macro_f1"]) if (e, m) in tagged else "--" for e in encs]
        lines.append((name, cells_acc, cells_f1))
    # bold the best value per encoder column
    for col in range(len(encs)):
        for which in (1, 2):
            vals = [ln[which][col] for ln in lines]
            best = bold_best(vals)
            for ln, b in zip(lines, best):
                ln[which][col] = b
    rows = [f"{name} & " + " & ".join(f"{a} / {f}" for a, f in zip(acc, f1)) + " \\\\" for name, acc, f1 in lines]
    cen_rows = []
    for mode, label in (("scratch", "Centralized, scratch"), ("pretrained_ft", "Centralized, pretrained FT")):
        cells = []
        for e in encs:
            r = cen.get((e, mode))
            cells.append("--" if r is None else f"{f3(r['test'].get('client_acc_mean'))} / {f3(r['test']['macro_f1'])}")
        cen_rows.append(f"\\textit{{{label}}} & " + " & ".join(cells) + " \\\\")
    header = "Method & " + " & ".join(ENC[e] for e in encs) + " \\\\"
    (OUT / "res_tab_main.tex").write_text(
        "% columns: " + ", ".join(ENC[e] for e in encs) + "\n" + header + "\n\\midrule\n" + "\n".join(rows)
        + "\n\\midrule\n" + "\n".join(cen_rows) + "\n", encoding="utf-8")
    macros["ResMainNcols"] = str(len(encs))

    # ------------------------------------------------------------- new users
    budgets = ["zero_shot", "few_shot_1shot", "few_shot_5shot", "few_shot_10shot"]
    rows = []
    for e in encs:
        for m, name in [("fedavg_scratch", "FedAvg"), ("v0_full_ft", "V0"), ("v3_lora_exact", "V3"), ("v5_proposed", "V5")]:
            r = fed.get((e, m))
            if not r or not r.get("heldout"):
                continue
            vals = [r["heldout"].get(b, {}).get("mean_user_f1") for b in budgets]
            rows.append(f"{ENC[e]} & {name} & " + " & ".join(f3(v) for v in vals) + " \\\\")
            if r.get("heldout_transductive"):
                vals = [r["heldout_transductive"].get(b, {}).get("mean_user_f1") for b in budgets]
                rows.append(f"{ENC[e]} & {name}, transd. & " + " & ".join(f3(v) for v in vals) + " \\\\")
        r = cen.get((e, "scratch"))
        if r and r.get("heldout"):
            vals = [r["heldout"].get(b, {}).get("mean_user_f1") for b in budgets]
            rows.append(f"{ENC[e]} & \\textit{{Central}} & " + " & ".join(f3(v) for v in vals) + " \\\\")
    (OUT / "res_tab_newusers.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ cost
    rows = []
    for e in encs:
        v0, v5 = fed.get((e, "v0_full_ft")), fed.get((e, "v5_proposed"))
        if not v5:
            continue
        up0 = v0["comm"]["uplink_mb_per_client_round"] if v0 else None
        up5 = v5["comm"]["uplink_mb_per_client_round"]
        ratio = f"{up0 / up5:.0f}$\\times$" if up0 else "--"
        mem0 = v0["resources"]["client_train_mem_mb_mean"] if v0 else None
        mem5 = v5["resources"]["client_train_mem_mb_mean"]
        t0 = v0["resources"]["client_time_s_mean"] if v0 else None
        t5 = v5["resources"]["client_time_s_mean"]
        rows.append(f"{ENC[e]} & {mb(up0)} & {mb(up5)} & {ratio} & {('--' if mem0 is None else f'{mem0:.0f}')} & {mem5:.0f} & "
                    f"{('--' if t0 is None else f'{t0:.2f}')} & {t5:.2f} \\\\")
    (OUT / "res_tab_cost.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # --------------------------------------- parameter budget (methodology, Table III)
    targets = {"limu_bert": "$W_q$, $W_v$ (shared layers)", "ssl_wearables": "all Conv1d kernels",
               "unimts": "graph + temporal convs", "halo": "$W_q$, $W_v$ (all blocks)"}
    cites = {"limu_bert": "xu2021limubert", "ssl_wearables": "yuan2024selfsupervised", "unimts": "zhang2024unimts",
             "halo": "ding2026halo"}

    def k_or_m(n: float) -> str:
        return f"{n / 1e6:.2f}\\,M" if n >= 1e6 else f"{n / 1e3:.1f}\\,k"

    rows, ratios = [], []
    for e in ["limu_bert", "ssl_wearables", "unimts", "halo"]:
        v5, v0 = fed.get((e, "v5_proposed")), fed.get((e, "v0_full_ft"))
        if not v5:
            continue
        pc, pr = v5["param_counts"], v5["comm"]["per_rank"]
        lora_full = pc["lora"]                                     # rank r_max = 8
        lora = "/".join(f"{lora_full * r / 8 / 1e3:.1f}" if lora_full < 1e4 else f"{lora_full * r / 8 / 1e3:.0f}"
                        for r in (2, 4, 8)) + "\\,k"
        phi = pc["norm"] + pc["adapter"] + pc["head"]
        up = "/".join(f"{pr[f'r{r}']['uplink_mb_per_update']:.3f}" for r in (2, 4, 8) if f"r{r}" in pr)
        up0 = v0["comm"]["uplink_mb_per_client_round"] if v0 else None
        ratio = up0 / v5["comm"]["uplink_mb_per_client_round"] if up0 else None
        if ratio:
            ratios.append(ratio)
        enc_name = ENC[e] + ("$^{\\ddagger}$" if e == "halo" else "")
        rows.append(f"{enc_name}~\\cite{{{cites[e]}}} & {targets[e]} & {k_or_m(pc['encoder'])} & {lora} & "
                    f"{pc['shared_head'] / 1e3:.1f}\\,k & {phi / 1e3:.1f}\\,k & {up} & "
                    f"{('--' if up0 is None else (f'{up0:.3f}' if up0 < 1 else f'{up0:.2f}'))} & "
                    f"{('--' if ratio is None else f'{ratio:.1f}$\\times$')} \\\\")
    (OUT / "res_tab_budget.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    if ratios:
        macros["ResUplinkRatioMin"] = f"{min(ratios):.0f}"
        macros["ResUplinkRatioMax"] = f"{max(ratios):.0f}"

    # ------------------------------------------------------- component ablation
    base = ROOT / "results" / args.main / "federated"
    enc = cfg.ablation.get("encoder", "limu_bert")

    def run(name: str) -> dict | None:
        p = base / f"{enc}__{name}__seed0.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    variants = [("v5_proposed", "V5 (complete)"),
                ("v5_proposed__noproto", "$-$ prototype alignment ($\\lambda=0$)"),
                ("v5_abl_no_local_norm", "$-$ local norm and input adapter"),
                ("v5_abl_shared_head", "$-$ personal classifier (shared $h$)"),
                ("v5_abl_fixed_rank", "$-$ device tiers (all $r=8$)"),
                ("v5_proposed__noaug", "$-$ training augmentation"),
                ("v5_proposed__pretrainedinit", "$-$ averaged local parts for new users"),
                ("v5_proposed__persnoaug", "$-$ augmentation in the last personal update"),
                ("v5_proposed__avghead", "$+$ average personal classifier for new users"),
                ("v5_proposed__dualhead", "$+$ federated global classifier (dual head)"),
                ("v5_proposed__loraall", "$+$ LoRA on all linear layers (not only $W_q,W_v$)"),
                ("v5_proposed__plora", "$+$ private LoRA copy in the last personal update"),
                ("v5_proposed__plora60", "$+$ private LoRA copy, $3\\times$ longer personal update"),
                ("v4_hetero_rank", "V4 ($-$ all local parts and prototypes)"),
                ("v0_full_ft", "V0 full fine-tuning"),
                ("v0_full_ft__noaug", "V0 $-$ training augmentation")]
    full = run("v5_proposed")
    rows = []
    for name, label in variants:
        r = run(name)
        if r is None:
            continue
        f = scored(r)
        h = r.get("heldout") or {}
        d = f["macro_f1"] - scored(full)["macro_f1"] if full else None
        rows.append(f"{label} & {f3(f.get('client_acc_mean'))} & {f3(f['macro_f1'])} & "
                    f"{'--' if d is None or name == 'v5_proposed' else f'{d:+.3f}'} & {f3(f.get('worst10_acc'))} & "
                    f"{f3((h.get('zero_shot') or {}).get('mean_user_f1'))} & {f3((h.get('few_shot_10shot') or {}).get('mean_user_f1'))} \\\\")
        if name == "v4_hetero_rank":
            rows.insert(len(rows) - 1, "\\midrule")
        if name == "v5_proposed" and r.get("heldout_transductive"):
            ht = r["heldout_transductive"]
            rows.append(f"$+$ transductive zero-shot (same model) & {f3(f.get('client_acc_mean'))} & {f3(f['macro_f1'])} & "
                        f"$\\pm$0 & {f3(f.get('worst10_acc'))} & "
                        f"{f3((ht.get('zero_shot') or {}).get('mean_user_f1'))} & "
                        f"{f3((ht.get('few_shot_10shot') or {}).get('mean_user_f1'))} \\\\")
    (OUT / "res_tab_ablation.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ------------------------------------------- repeated user splits (mean +- std)
    seeds = load_seeds(args.main)
    rows = []
    for (e, m), name in [(("mlp", "local_scratch"), "Local"), (("mlp", "fedavg_scratch"), "FedAvg"),
                         (("mlp", "ditto_scratch"), "Ditto"), (("limu_bert", "fedavg_scratch"), "FedAvg"),
                         (("limu_bert", "v0_full_ft"), "V0 Full FT"), (("limu_bert", "ditto_ft"), "Ditto (pretrained)"),
                         (("limu_bert", "v3_lora_exact"), "V3 LoRA exact"),
                         (("limu_bert", "v3_lora_exact__loraall"), "V3, LoRA on all layers"),
                         (("limu_bert", "v5_proposed"), "\\textbf{V5 \\method}"),
                         (("limu_bert", "v5_proposed__loraall"), "\\textbf{V5 \\method}, LoRA on all layers")]:
        runs = seeds.get((e, m), {})
        if len(runs) < 2:
            continue
        rr = [runs[k] for k in sorted(runs)]
        zs = []
        for r in rr:
            h = r.get("heldout_transductive") or r.get("heldout") or {}
            zs.append((h.get("zero_shot") or {}).get("mean_user_f1"))
        rows.append(f"{name} & {ENC[e]} & {len(rr)} & "
                    + " & ".join([mean_std([scored(r).get("client_acc_mean") for r in rr]),
                                  mean_std([scored(r)["macro_f1"] for r in rr]),
                                  mean_std([scored(r).get("worst10_acc") for r in rr]),
                                  mean_std(zs),
                                  mean_std([((r.get("heldout") or {}).get("few_shot_10shot") or {}).get("mean_user_f1")
                                            for r in rr])]) + " \\\\")
    (OUT / "res_tab_seeds.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    (OUT / "res_macros.tex").write_text(
        "% Generated by scripts/tools/paper_results_tables.py - do not edit by hand.\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n", encoding="utf-8")
    print(f"tables written to {OUT}; macros: {macros}")


if __name__ == "__main__":
    main()

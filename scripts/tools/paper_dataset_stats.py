"""
Dataset statistics for the paper (Section "Dataset", paper/sections/dataset.tex).

Everything the dataset section reports is computed here from the processed
windows (data/processed/*.npz) and the published user split, so every number
in the paper can be regenerated with one command:

    python scripts/tools/paper_dataset_stats.py

Outputs (paper/generated/):
    dataset_stats.json      all numbers, machine-readable
    tab_classes.tex         class distribution table
    tab_partition.tex       user partition + demographics table
    tab_heterogeneity.tex   statistical-heterogeneity table
    macros.tex              \newcommand's for numbers quoted in the running text
Figures (paper/figures/):
    label_skew.pdf/.png     per-client label distributions, class presence, train/test shift

Two views of the data are used:
    full        every labelled 10-s window of all 151 participants (dataset description)
    experiment  the windows the experiments use: 1 window per minute
                (data.subsample_every = 6), user split of seed 0, 80/20 time split
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fedpeft.config import load_config  # noqa: E402

OUT = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"

# Display order and names of the Willetts2018 classes (index = class id in the npz files).
CLASS_IDS = ["sleep", "sit-stand", "mixed", "walking", "vehicle", "bicycling"]
DISPLAY = {"sleep": "Sleep", "sit-stand": "Sit/stand", "mixed": "Mixed activity",
           "walking": "Walking", "vehicle": "Vehicle", "bicycling": "Bicycling"}
DESCRIPTION = {
    "sleep": "Sleep periods from the sleep diary",
    "sit-stand": "Sedentary sitting or standing (desk work, TV, eating)",
    "mixed": "Light activity: household chores, manual work, sports",
    "walking": "Walking, incl.\\ stairs and walking while carrying loads",
    "vehicle": "Riding in or driving a car, bus or train",
    "bicycling": "Cycling",
}
# Colour per class (Okabe-Ito, colour-blind safe), used by the figure.
COLOURS = {"sleep": "#0072B2", "sit-stand": "#56B4E9", "mixed": "#009E73",
           "walking": "#E69F00", "vehicle": "#CC79A7", "bicycling": "#D55E00"}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint supports)."""
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def gini(x: np.ndarray) -> float:
    """Gini coefficient of non-negative quantities (0 = all equal)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def bouts(y: np.ndarray, t_ms: np.ndarray, window_ms: int) -> list[tuple[int, int]]:
    """(class, length in windows) of every uninterrupted run of one label.

    A run also ends at a time gap (consecutive windows more than one window apart).
    """
    out = []
    if len(y) == 0:
        return out
    start = 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[i - 1] or t_ms[i] - t_ms[i - 1] > 1.5 * window_ms:
            out.append((int(y[start]), i - start))
            start = i
    return out


def equivalent_dirichlet_alpha(target_jsd: float, sizes: np.ndarray, prior: np.ndarray,
                               reps: int = 30, seed: int = 0) -> float:
    """Concentration alpha of the synthetic label-skew model q_k ~ Dir(alpha * C * prior)
    (client sizes as in the data, labels drawn multinomially) whose mean client-to-global
    JSD equals the observed one.  Lets readers compare Capture-24's natural skew with the
    Dirichlet partitions common in the FL literature (smaller alpha = stronger skew)."""
    rng = np.random.default_rng(seed)
    C = len(prior)
    grid = np.logspace(-2, 2, 33)
    means = []
    for a in grid:
        vals = []
        for _ in range(reps):
            for n in sizes:
                q = rng.dirichlet(np.maximum(a * C * prior, 1e-3))
                counts = rng.multinomial(int(n), q)
                vals.append(jsd(counts.astype(float) + 0.0, prior))
        means.append(np.mean(vals))
    means = np.array(means)                       # decreasing in alpha
    # Interpolate log(alpha) as a function of mean JSD.
    order = np.argsort(means)
    return float(np.exp(np.interp(target_jsd, means[order], np.log(grid)[order])))


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
def load_user(processed: Path, user: str, scheme: str, ids: list[int]):
    """All labelled windows of one user: labels re-mapped to CLASS_IDS order, times (ms)."""
    with np.load(processed / f"{user}.npz") as d:
        y = d[f"y_{scheme}"].astype(np.int64)
        t = d["t"].astype(np.int64)
    keep = y >= 0
    remap = np.full(max(ids) + 1, -1)
    remap[ids] = np.arange(len(ids))
    return remap[y[keep]], t[keep]


def main() -> None:
    cfg = load_config("enhanced")
    processed = ROOT / cfg.paths.processed_dir
    meta = json.loads((processed / "meta.json").read_text(encoding="utf-8"))
    scheme = cfg.data.label_scheme
    file_classes = meta["schemes"][scheme]                  # alphabetical order of the npz ids
    ids = [file_classes.index(c) for c in CLASS_IDS]        # npz id of each display class
    C = len(CLASS_IDS)
    window_s = int(meta["window_s"])
    window_ms = window_s * 1000
    sub = int(cfg.data.subsample_every)
    split = json.loads((ROOT / cfg.paths.results_dir / "splits" / "user_split_seed0.json").read_text())
    group_of = {u: g for g in ("clients", "heldout", "pool") for u in split[g]}
    demo = pd.DataFrame(meta["participant_info"]).set_index("pid")

    stats: dict = {"classes": [DISPLAY[c] for c in CLASS_IDS], "window_s": window_s,
                   "subsample_every": sub, "n_participants": len(meta["participants"])}

    # ------------------------------------------------------------------ full
    full_counts = np.zeros(C, dtype=np.int64)
    hours_user, users_with = [], np.zeros(C, dtype=int)
    bout_len = {c: [] for c in range(C)}
    exp = {}                                                     # experiment view per user
    for user in meta["participants"]:
        y, t = load_user(processed, user, scheme, ids)
        cnt = np.bincount(y, minlength=C)
        full_counts += cnt
        users_with += cnt > 0
        hours_user.append(len(y) * window_s / 3600)
        for c, n in bouts(y, t, window_ms):
            bout_len[c].append(n * window_s / 60)               # minutes
        # Experiment view: 1 window in `sub`, exactly as fedpeft/data/store.py does it
        # (sub-sampling is applied BEFORE dropping unlabelled windows there).
        with np.load(processed / f"{user}.npz") as d:
            ys = d[f"y_{scheme}"][::sub].astype(np.int64)
            ts = d["t"][::sub].astype(np.int64)
        keep = ys >= 0
        remap = np.full(max(ids) + 1, -1)
        remap[ids] = np.arange(C)
        exp[user] = (remap[ys[keep]], ts[keep])

    total = int(full_counts.sum())
    stats["full"] = {
        "windows": total,
        "hours": total * window_s / 3600,
        "class_windows": full_counts.tolist(),
        "class_pct": (100 * full_counts / total).tolist(),
        "users_with_class": users_with.tolist(),
        "hours_per_user": {"min": float(np.min(hours_user)), "median": float(np.median(hours_user)),
                           "max": float(np.max(hours_user)), "mean": float(np.mean(hours_user))},
        "bout_minutes_median": [float(np.median(bout_len[c])) for c in range(C)],
        "bout_minutes_mean": [float(np.mean(bout_len[c])) for c in range(C)],
        "imbalance_ratio": float(full_counts.max() / full_counts.min()),
    }

    # ------------------------------------------------------------ experiment
    frac = float(cfg.split.train_time_frac)
    adapt_frac = float(cfg.split.heldout_adapt_frac)
    part_rows = {}
    for g in ("clients", "heldout", "pool"):
        users = split[g]
        n = [len(exp[u][0]) for u in users]
        sexes = demo.loc[users, "sex"].value_counts()
        ages = demo.loc[users, "age"].value_counts()
        part_rows[g] = {"users": len(users), "windows": int(np.sum(n)),
                        "hours_equiv": float(np.sum(n) * window_s * sub / 3600),
                        "female": int(sexes.get("F", 0)), "male": int(sexes.get("M", 0)),
                        "age": {a: int(ages.get(a, 0)) for a in ("18-29", "30-37", "38-52", "53+")}}
        if g == "clients":
            tr = sum(int(np.floor(frac * k)) for k in n)
            part_rows[g].update(train_windows=tr, test_windows=int(np.sum(n)) - tr)
        if g == "heldout":
            ad = sum(int(np.floor(adapt_frac * k)) for k in n)
            part_rows[g].update(adapt_windows=ad, test_windows=int(np.sum(n)) - ad)
    stats["partition"] = part_rows

    # Heterogeneity across the FL clients (training parts = what each client trains on).
    clients = split["clients"]
    tr_counts, te_counts = [], []
    for u in clients:
        y, _ = exp[u]
        cut = int(np.floor(frac * len(y)))
        tr_counts.append(np.bincount(y[:cut], minlength=C))
        te_counts.append(np.bincount(y[cut:], minlength=C))
    tr_counts, te_counts = np.array(tr_counts), np.array(te_counts)
    glob = tr_counts.sum(0) / tr_counts.sum()
    n_k = tr_counts.sum(1)
    jsd_global = np.array([jsd(tr_counts[k].astype(float), glob) for k in range(len(clients))])
    jsd_shift = np.array([jsd(tr_counts[k].astype(float) + 1e-12, te_counts[k].astype(float) + 1e-12)
                          for k in range(len(clients)) if te_counts[k].sum() > 0])
    classes_per_client = (tr_counts > 0).sum(1)
    unseen = (te_counts > 0) & (tr_counts == 0)           # class in a client's test part, absent from its train part
    unseen_windows = int((te_counts * (tr_counts == 0)).sum())
    alpha = equivalent_dirichlet_alpha(float(jsd_global.mean()), n_k, glob)
    stats["heterogeneity"] = {
        "global_prior_train": glob.tolist(),
        "clients_with_class_train_pct": (100 * (tr_counts > 0).mean(0)).tolist(),
        "classes_per_client_hist": {int(k): int(v) for k, v in
                                    zip(*np.unique(classes_per_client, return_counts=True))},
        "classes_per_client_mean": float(classes_per_client.mean()),
        "all_six_classes_pct": float(100 * (classes_per_client == C).mean()),
        "jsd_to_global": {"mean": float(jsd_global.mean()), "std": float(jsd_global.std()),
                          "median": float(np.median(jsd_global)), "max": float(jsd_global.max())},
        "jsd_train_test": {"mean": float(jsd_shift.mean()), "median": float(np.median(jsd_shift)),
                           "max": float(jsd_shift.max())},
        "clients_with_unseen_test_class_pct": float(100 * unseen.any(1).mean()),
        "unseen_test_windows_pct": float(100 * unseen_windows / te_counts.sum()),
        "windows_per_client": {"min": int(n_k.min()), "median": float(np.median(n_k)),
                               "max": int(n_k.max()), "cv": float(n_k.std() / n_k.mean()),
                               "gini": gini(n_k)},
        "equivalent_dirichlet_alpha": alpha,
        "dominant_class_share_mean": float((tr_counts.max(1) / n_k).mean()),
    }

    # Transition statistics at the experiment resolution (1 window per minute), as the
    # federated HMM sees them: self-transition probability per class.
    trans = np.ones((C, C))
    spacing = sub * window_ms
    for u in clients:
        y, t = exp[u]
        cut = int(np.floor(frac * len(y)))
        y, t = y[:cut], t[:cut]
        cont = np.diff(t) <= 3 * spacing
        np.add.at(trans, (y[:-1][cont], y[1:][cont]), 1)
    P = trans / trans.sum(1, keepdims=True)
    stats["hmm"] = {"self_transition": np.diag(P).tolist(), "transition_matrix": P.tolist()}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_tables(stats)
    make_figure(tr_counts, te_counts, clients, stats)
    print(json.dumps({k: stats[k] for k in ("full", "heterogeneity", "hmm")}, indent=1)[:4000])


# ----------------------------------------------------------------------------
# LaTeX output
# ----------------------------------------------------------------------------
def fmt_int(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", "{,}")


def write_tables(s: dict) -> None:
    f, h = s["full"], s["heterogeneity"]
    C = len(CLASS_IDS)

    # --- class distribution ---
    rows = []
    for i, c in enumerate(CLASS_IDS):
        rows.append(f"{DISPLAY[c]} & {DESCRIPTION[c]} & {fmt_int(f['class_windows'][i])} & "
                    f"{f['class_pct'][i]:.1f} & {f['users_with_class'][i]} & "
                    f"{h['clients_with_class_train_pct'][i]:.0f} & {f['bout_minutes_median'][i]:.1f} \\\\")
    rows.append("\\midrule")
    rows.append(f"Total & & {fmt_int(f['windows'])} & 100.0 & {s['n_participants']} & -- & -- \\\\")
    (OUT / "tab_classes.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # --- partition ---
    p = s["partition"]
    lab = {"clients": "FL clients", "heldout": "Held-out users", "pool": "Pool (SSL + validation)"}
    use = {"clients": "first 80\\,\\% train / last 20\\,\\% test",
           "heldout": "first 50\\,\\% adaptation / last 50\\,\\% test",
           "pool": "unlabelled SSL (HALO), model selection"}
    rows = []
    for g in ("clients", "heldout", "pool"):
        r = p[g]
        a = r["age"]
        rows.append(f"{lab[g]} & {r['users']} & {fmt_int(r['windows'])} & {r['female']}/{r['male']} & "
                    f"{a['18-29']}/{a['30-37']}/{a['38-52']}/{a['53+']} & {use[g]} \\\\")
    (OUT / "tab_partition.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # --- heterogeneity ---
    w = h["windows_per_client"]
    hist = h["classes_per_client_hist"]
    hist_txt = ", ".join(f"{k}:\\,{v}" for k, v in sorted(hist.items()))
    rows = [
        f"Windows per client (min / median / max) & {w['min']} / {w['median']:.0f} / {w['max']} \\\\",
        f"Quantity skew: coefficient of variation; Gini & {w['cv']:.2f}; {w['gini']:.2f} \\\\",
        f"Classes per client, mean (histogram) & {h['classes_per_client_mean']:.2f} ({hist_txt}) \\\\",
        f"Clients observing all {C} classes & {h['all_six_classes_pct']:.1f}\\,\\% \\\\",
        f"Share of the dominant class per client, mean & {100 * h['dominant_class_share_mean']:.1f}\\,\\% \\\\",
        f"JSD(client $\\|$ global), mean $\\pm$ sd (max) & {h['jsd_to_global']['mean']:.3f} $\\pm$ "
        f"{h['jsd_to_global']['std']:.3f} ({h['jsd_to_global']['max']:.3f}) \\\\",
        f"Equivalent Dirichlet concentration $\\alpha$ & {h['equivalent_dirichlet_alpha']:.2f} \\\\",
        f"JSD(train $\\|$ test) within a client, mean (max) & {h['jsd_train_test']['mean']:.3f} "
        f"({h['jsd_train_test']['max']:.3f}) \\\\",
        f"Clients whose test part has a class absent from their train part & "
        f"{h['clients_with_unseen_test_class_pct']:.1f}\\,\\% \\\\",
        f"Test windows of such locally unseen classes & {h['unseen_test_windows_pct']:.1f}\\,\\% \\\\",
    ]
    (OUT / "tab_heterogeneity.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # --- macros for the running text ---
    m = {
        "NWindowsFull": fmt_int(f["windows"]),
        "NHoursFull": fmt_int(f["hours"]),
        "HoursPerUserMedian": f"{f['hours_per_user']['median']:.1f}",
        "HoursPerUserMin": f"{f['hours_per_user']['min']:.1f}",
        "HoursPerUserMax": f"{f['hours_per_user']['max']:.1f}",
        "ImbalanceRatio": f"{f['imbalance_ratio']:.0f}",
        "PctSleep": f"{f['class_pct'][0]:.1f}",
        "PctSitStand": f"{f['class_pct'][1]:.1f}",
        "PctBicycling": f"{f['class_pct'][5]:.1f}",
        "PctVehicle": f"{f['class_pct'][4]:.1f}",
        "UsersBicycling": str(f["users_with_class"][5]),
        "NWindowsExp": fmt_int(sum(p[g]["windows"] for g in p)),
        "NTrainWindows": fmt_int(p["clients"]["train_windows"]),
        "NTestWindows": fmt_int(p["clients"]["test_windows"]),
        "NHeldoutTest": fmt_int(p["heldout"]["test_windows"]),
        "NPoolWindows": fmt_int(p["pool"]["windows"]),
        "ClientsBicyclingPct": f"{h['clients_with_class_train_pct'][5]:.0f}",
        "ClientsVehiclePct": f"{h['clients_with_class_train_pct'][4]:.0f}",
        "ClassesPerClient": f"{h['classes_per_client_mean']:.2f}",
        "AllSixPct": f"{h['all_six_classes_pct']:.0f}",
        "JSDMean": f"{h['jsd_to_global']['mean']:.3f}",
        "JSDMax": f"{h['jsd_to_global']['max']:.3f}",
        "DirAlpha": f"{h['equivalent_dirichlet_alpha']:.2f}",
        "UnseenClientsPct": f"{h['clients_with_unseen_test_class_pct']:.0f}",
        "UnseenWindowsPct": f"{h['unseen_test_windows_pct']:.1f}",
        "WinPerClientMedian": f"{h['windows_per_client']['median']:.0f}",
        "WinPerClientMin": str(h["windows_per_client"]["min"]),
        "WinPerClientMax": str(h["windows_per_client"]["max"]),
        "GiniClients": f"{h['windows_per_client']['gini']:.2f}",
        "SelfTransMin": f"{min(s['hmm']['self_transition']):.2f}",
        "SelfTransMax": f"{max(s['hmm']['self_transition']):.2f}",
    }
    (OUT / "macros.tex").write_text(
        "% Generated by scripts/tools/paper_dataset_stats.py - do not edit by hand.\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def make_figure(tr: np.ndarray, te: np.ndarray, clients: list[str], s: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42,
    })
    C = len(CLASS_IDS)
    cols = [COLOURS[c] for c in CLASS_IDS]
    fig = plt.figure(figsize=(7.16, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 1.05, 1.15], wspace=0.45,
                          left=0.055, right=0.99, bottom=0.19, top=0.80)

    # (a) per-client label distribution, clients sorted by their sleep + sit/stand share
    ax = fig.add_subplot(gs[0])
    q = tr / tr.sum(1, keepdims=True)
    order = np.lexsort((-q[:, 1], q[:, 0] + q[:, 1]))
    bottom = np.zeros(len(order))
    x = np.arange(len(order))
    for c in range(C):
        ax.bar(x, q[order, c], bottom=bottom, width=1.0, color=cols[c], linewidth=0,
               label=DISPLAY[CLASS_IDS[c]])
        bottom += q[order, c]
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"FL client (sorted), $K={len(order)}$")
    ax.set_ylabel("Share of training windows")
    ax.set_title("(a) Label distribution per client", loc="left")
    fig.legend(*ax.get_legend_handles_labels(), ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               frameon=False, handlelength=0.9, columnspacing=1.2, handletextpad=0.4)

    # (b) share of clients that have each class (train part) and global share
    ax = fig.add_subplot(gs[1])
    pres = 100 * (tr > 0).mean(0)
    glob = 100 * tr.sum(0) / tr.sum()
    y = np.arange(C)[::-1]
    ax.barh(y, pres, color=cols, height=0.62)
    for yi, p, g in zip(y, pres, glob):
        ax.text(p + 2, yi, f"{p:.0f}% ({g:.1f}%)", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[c] for c in CLASS_IDS])
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Clients with the class (%)")
    ax.set_title("(b) Class presence (global share)", loc="left")

    # (c) client-to-global JSD and within-client train/test JSD
    ax = fig.add_subplot(gs[2])
    glob_p = tr.sum(0) / tr.sum()
    j_g = [jsd(tr[k].astype(float), glob_p) for k in range(len(tr))]
    j_t = [jsd(tr[k] + 1e-12, te[k] + 1e-12) for k in range(len(tr)) if te[k].sum() > 0]
    bins = np.linspace(0, max(max(j_g), max(j_t)) * 1.02, 16)
    ax.hist(j_g, bins=bins, color="#0072B2", alpha=0.8, label="client vs. global")
    ax.hist(j_t, bins=bins, histtype="step", color="#D55E00", linewidth=1.2, label="own train vs. test")
    ax.set_xlabel("JSD (bits)")
    ax.set_ylabel("Clients")
    ax.set_title("(c) Label shift", loc="left")
    ax.legend(frameon=False, loc="upper right", handlelength=1.0)

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "label_skew.pdf")
    fig.savefig(FIG / "label_skew.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

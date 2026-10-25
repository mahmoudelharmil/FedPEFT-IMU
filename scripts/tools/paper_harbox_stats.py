"""
HARBox statistics for the paper (Section "Dataset", paper/sections/dataset.tex).

Every number the dataset section reports about HARBox is computed here from the
processed windows (data/processed_harbox/*.npz) and the published user split:

    python scripts/tools/paper_harbox_stats.py

Outputs (paper/generated/):
    harbox_stats.json         all numbers, machine-readable
    harbox_tab_classes.tex    class table rows
    harbox_tab_partition.tex  user partition rows (both protocols)
    harbox_tab_hetero.tex     heterogeneity rows
    harbox_macros.tex         \\newcommand's for the running text
Figure (paper/figures/):
    harbox_stats.pdf/.png     (a) windows per user and activity, (b) windows per user,
                              (c) leakage of the random protocol

It also measures the LEAKAGE of the random-split protocol of the published results:
consecutive HARBox windows overlap by 50 %, so a test window whose neighbour is in
the training set shares half of its samples with training data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fedpeft.config import load_config  # noqa: E402
from fedpeft.data.store import load_experiment_data  # noqa: E402

OUT = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"
DESCRIPTION = {"Call": "phone held at the ear during a call", "Hop": "hopping on the spot",
               "Typing": "typing on the phone", "Walk": "walking", "Wave": "waving the phone-holding hand"}
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits."""
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def neighbours_in(store, test_idx: np.ndarray, train_set: set) -> np.ndarray:
    """For every test window: does an OVERLAPPING window (same session, start within one
    window length) belong to the training set?"""
    t, seg = store.t_ms, store.seg
    win_ms = store.window_s * 1000
    out = np.zeros(len(test_idx), dtype=bool)
    for i, w in enumerate(test_idx):
        for nb in (w - 1, w + 1):
            if 0 <= nb < len(t) and seg[nb] == seg[w] and abs(t[nb] - t[w]) < win_ms and nb in train_set:
                out[i] = True
                break
    return out


def main() -> None:
    dev = torch.device("cpu")
    cfg_b = load_config("harbox")                 # leakage-free protocol (ours)
    cfg_a = load_config("harbox_fedchar")         # published protocol
    data_b = load_experiment_data(cfg_b, 0, dev)
    data_a = load_experiment_data(cfg_a, 0, dev)
    store = data_b.store
    meta = json.loads((Path(cfg_b.paths.processed_dir) / "meta.json").read_text())
    classes = meta["schemes"]["HARBox"]
    C = len(classes)
    y = store.y.numpy()

    # ------------------------------------------------------------ dataset
    per_user = {u: np.bincount(y[slice(*store.slices[u])], minlength=C) for u in store.slices}
    counts = np.array(list(per_user.values()))          # users x classes
    n_user = counts.sum(1)
    sessions = {u: len(np.unique(store.seg[slice(*store.slices[u])])) for u in store.slices}
    seconds = len(y) * meta["hop_s"] + len(store.slices) * 0          # distinct signal ~ hop per window
    stats = {
        "users": len(per_user), "windows": int(len(y)), "classes": classes,
        "class_windows": counts.sum(0).tolist(), "class_pct": (100 * counts.sum(0) / counts.sum()).tolist(),
        "users_with_class": (counts > 0).sum(0).tolist(),
        "windows_per_user": {"min": int(n_user.min()), "median": float(np.median(n_user)), "max": int(n_user.max()),
                             "mean": float(n_user.mean()), "cv": float(n_user.std() / n_user.mean())},
        "windows_per_user_class_median": float(np.median(counts[counts > 0])),
        "minutes_per_user_class_median": float(np.median(counts[counts > 0]) * meta["hop_s"] / 60),
        "classes_per_user_hist": {int(k): int(v) for k, v in zip(*np.unique((counts > 0).sum(1), return_counts=True))},
        "sessions_per_user_median": float(np.median(list(sessions.values()))),
        "signal_hours": float(seconds / 3600),
        "jsd_user_global": float(np.mean([jsd(c.astype(float), counts.sum(0).astype(float)) for c in counts])),
    }

    # ---------------------------------------------------------- protocols
    def part_stats(data):
        tr = sum(len(v) for v in data.client_train.values())
        te = sum(len(v) for v in data.client_test.values())
        va = sum(len(v) for v in data.client_val.values())
        return {"clients": len(data.clients), "heldout": len(data.heldout), "pool": len(data.split["pool"]),
                "train": tr, "val": va, "test": te,
                "heldout_windows": sum(len(v) for v in data.heldout_test.values()) + sum(len(v) for v in data.heldout_adapt.values()),
                "pool_windows": int(len(data.pool_index)) if not data.client_val else 0}
    stats["protocol_b"] = part_stats(data_b)
    stats["protocol_a"] = part_stats(data_a)

    # Leakage: fraction of test windows with an overlapping neighbour in the training set.
    leak = {}
    for name, data in (("protocol_a", data_a), ("protocol_b", data_b)):
        flags = []
        for u in data.clients:
            tr = set(data.client_train[u].numpy().tolist())
            flags.append(neighbours_in(store if data is data_b else data.store, data.client_test[u].numpy(), tr))
        f = np.concatenate(flags)
        leak[name] = float(100 * f.mean())
    stats["leakage_pct"] = leak

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "harbox_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_tex(stats, meta)
    make_figure(counts, classes, stats)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("classes",)}, indent=1))


def fmt_int(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", "{,}")


def write_tex(s: dict, meta: dict) -> None:
    rows = []
    for i, c in enumerate(s["classes"]):
        rows.append(f"{c} & {DESCRIPTION.get(c, '')} & {fmt_int(s['class_windows'][i])} & {s['class_pct'][i]:.1f} & "
                    f"{s['users_with_class'][i]} \\\\")
    rows.append("\\midrule")
    rows.append(f"Total & & {fmt_int(s['windows'])} & 100.0 & {s['users']} \\\\")
    (OUT / "harbox_tab_classes.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    a, b = s["protocol_a"], s["protocol_b"]
    rows = [
        f"Users: clients / held-out / pool & {b['clients']} / {b['heldout']} / {b['pool']} & {a['clients']} / 0 / 0 \\\\",
        f"Windows per client & all & 50--500 at random \\\\",
        f"Client split & per activity in time, 80/20, 1-window gap & random 72/8/20 \\\\",
        f"Train / validation / test windows & {fmt_int(b['train'])} / {fmt_int(b['pool_windows'])}$^{{\\dagger}}$ / {fmt_int(b['test'])} & "
        f"{fmt_int(a['train'])} / {fmt_int(a['val'])} / {fmt_int(a['test'])} \\\\",
        f"Test windows overlapping a training window & {s['leakage_pct']['protocol_b']:.1f}\\,\\% & {s['leakage_pct']['protocol_a']:.1f}\\,\\% \\\\",
        f"New-user evaluation & {b['heldout']} users (zero-/few-shot) & -- \\\\",
    ]
    (OUT / "harbox_tab_partition.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    w = s["windows_per_user"]
    hist = ", ".join(f"{k}:\\,{v}" for k, v in sorted(s["classes_per_user_hist"].items()))
    rows = [
        f"Windows per user (min / median / max) & {w['min']} / {w['median']:.0f} / {w['max']} \\\\",
        f"Quantity skew: coefficient of variation & {w['cv']:.2f} \\\\",
        f"Activities per user (histogram) & {hist} \\\\",
        f"Windows per user and activity, median & {s['windows_per_user_class_median']:.0f} "
        f"($\\approx${s['minutes_per_user_class_median']:.1f} min) \\\\",
        f"JSD(user label distribution $\\|$ global), mean & {s['jsd_user_global']:.3f} \\\\",
    ]
    (OUT / "harbox_tab_hetero.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    m = {
        "HBUsers": str(s["users"]), "HBWindows": fmt_int(s["windows"]),
        "HBWinUserMin": str(w["min"]), "HBWinUserMedian": f"{w['median']:.0f}", "HBWinUserMax": str(w["max"]),
        "HBWinUserClassMedian": f"{s['windows_per_user_class_median']:.0f}",
        "HBLeakA": f"{s['leakage_pct']['protocol_a']:.1f}", "HBLeakB": f"{s['leakage_pct']['protocol_b']:.1f}",
        "HBClientsB": str(b["clients"]), "HBHeldoutB": str(b["heldout"]), "HBPoolB": str(b["pool"]),
        "HBTrainB": fmt_int(b["train"]), "HBTestB": fmt_int(b["test"]),
        "HBTrainA": fmt_int(a["train"]), "HBValA": fmt_int(a["val"]), "HBTestA": fmt_int(a["test"]),
        "HBJSD": f"{s['jsd_user_global']:.3f}", "HBHours": f"{s['signal_hours']:.1f}",
    }
    (OUT / "harbox_macros.tex").write_text(
        "% Generated by scripts/tools/paper_harbox_stats.py - do not edit by hand.\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n", encoding="utf-8")


def make_figure(counts: np.ndarray, classes: list[str], s: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.16, 2.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 1.1, 1.1], wspace=0.48, left=0.075, right=0.99, bottom=0.2, top=0.8)

    ax = fig.add_subplot(gs[0])
    order = np.argsort(counts.sum(1))
    bottom = np.zeros(len(order))
    for c in range(len(classes)):
        ax.bar(np.arange(len(order)), counts[order, c], bottom=bottom, width=1.0, color=COLOURS[c], linewidth=0,
               label=classes[c])
        bottom += counts[order, c]
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xlabel(f"user (sorted), $n={len(order)}$")
    ax.set_ylabel("2-s windows")
    ax.set_title("(a) Windows per user and activity", loc="left")
    fig.legend(*ax.get_legend_handles_labels(), ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               frameon=False, handlelength=0.9, columnspacing=1.2)

    ax = fig.add_subplot(gs[1])
    per_class = counts[counts > 0]
    ax.hist(np.clip(per_class, 0, 150), bins=np.arange(0, 155, 5), color="#0072B2", alpha=0.85)
    ax.set_xlim(0, 152)
    ax.set_xlabel("windows per user and activity (150 = 150+)")
    ax.set_ylabel("user-activity pairs")
    ax.set_title("(b) Session length", loc="left")

    ax = fig.add_subplot(gs[2])
    leak = s["leakage_pct"]
    bars = ax.bar(["published\n(random)", "ours\n(time + gap)"], [leak["protocol_a"], leak["protocol_b"]],
                  color=["#D55E00", "#009E73"], width=0.55)
    for b, v in zip(bars, [leak["protocol_a"], leak["protocol_b"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=6.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("test windows sharing\nsamples with training (%)")
    ax.set_title("(c) Train/test leakage", loc="left")

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "harbox_stats.pdf")
    fig.savefig(FIG / "harbox_stats.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

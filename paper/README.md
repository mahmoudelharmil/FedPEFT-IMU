# FedPEFT-IMU manuscript (IEEE journal format)

Draft of the journal paper. Written so far: **Section III (proposed method)** and
**Section IV (dataset and experimental protocol)**. The other sections are placeholders.

| File | Content |
|---|---|
| `main.tex` | IEEEtran journal preamble, notation macros, section order |
| `sections/methodology.tex` | III. Proposed method: problem setting, parameter roles, client model, local objective, exact rank-heterogeneous LoRA aggregation (Lemma 1, Propositions 1–2, Corollary 1), label skew, inference (calibration, federated HMM, new users), Algorithm 1, cost analysis, privacy |
| `sections/dataset.tex` | IV. Capture-24 description, preprocessing, federated partition, heterogeneity analysis, encoders, compared methods, metrics, implementation details |
| `figures/fig_architecture.tex` | Fig. 1, system architecture (TikZ, vector) |
| `figures/fig_aggregation.tex` | Fig. 2, delta-form exact aggregation (TikZ) |
| `figures/label_skew.pdf` | Fig. 3, natural heterogeneity of the clients (generated) |
| `generated/` | dataset numbers, table rows and `\newcommand` macros (generated) |
| `references.bib` | bibliography (verified against publisher pages; re-check page numbers before submission) |

## Build

```bash
python scripts/tools/paper_dataset_stats.py   # regenerate generated/*.tex and figures/label_skew.*
python paper/build.py                          # pdflatex + bibtex + pdflatex x2 -> paper/main.pdf
python paper/build.py --preview                # also PNG page previews (needs PyMuPDF)
```

The project uses no packages beyond a standard TeX Live / MiKTeX / Overleaf installation. On a
minimal MiKTeX install the preamble skips `hyperref` and `booktabs` if they are missing
(links and rule thickness only).

## Every number is traceable

- Dataset statistics: `scripts/tools/paper_dataset_stats.py` → `generated/dataset_stats.json`.
- Parameter budgets and measured uplink (Table III): `results/enhanced/federated/*.json`
  (`param_counts`, `comm`).
- Hyperparameters (Table IX): `configs/default.yaml` + `configs/profiles/enhanced.yaml`.

## Before submission

- Repeat all experiments for split seeds 1 and 2 and report the mean ± std (`TODO` in `dataset.tex`).
- Write the abstract, introduction, related work, results and conclusion.

# FedPEFT-IMU

**Parameter-efficient federated adaptation of pretrained inertial encoders for human activity recognition.**

This project implements, runs and measures the complete *Idea 1* research plan
(`FedPEFT-IMU_Idea1_Complete.pdf`), first on the **Capture-24** dataset and then on
**HARBox** (the paper's benchmark) and **WISDM-2019** (an HTML results report):

1. **Phase 1: centralized proof of concept.** The four IMU encoders (LIMU-BERT,
   ssl-wearables, UniMTS, HALO) are trained **from scratch**, and also fine-tuned from
   their pretrained weights as a reference.
2. **Phase 2: federated learning.** The same encoders are trained with seven FL methods,
   from FedAvg-from-scratch to the proposed **FedPEFT-IMU (V5)**.
3. **An interactive HTML report** with every KPI the plan asks for:

| KPI | Where it is measured |
|---|---|
| Macro-F1, accuracy | pooled test windows of all clients (`training/metrics.py`) |
| Worst-10 % client F1, client-F1 std | per-client test on each client's last 20 % of recording |
| Uplink MB / round, MB to target F1 | bytes of every upload, measured (`federated/comm.py`) |
| Rounds to target, client memory, time | round loop (`federated/simulation.py`), peak GPU memory, wall-clock |
| Zero-/few-shot F1 on held-out users | 0, 1 and 5 minutes of labels (`training/heldout.py`) |

## Current status (25 October 2026)

| Dataset | Role | Experiments | Results |
|---|---|---|---|
| Capture-24 | first dataset (PDF plan) | complete | `reports/FedPEFT-IMU_report_enhanced.html` ([online](https://claude.ai/artifact/5sYgeLBh2ZNHXquDguZDbG)) |
| HARBox | main benchmark of the paper | complete (3 user splits for LIMU-BERT and the MLP) | `reports/FedPEFT-IMU_report_harbox.html` ([online](https://claude.ai/artifact/3y3fpSBYLe8QZE9XLyjRLm)), `paper/main.pdf` |
| WISDM-2019 (smartwatch) | third dataset, **HTML report only** (not in the paper) | protocol B complete except UniMTS; protocol A (published FedAIoT protocol) partly done | `reports/FedPEFT-IMU_report_wisdm.html` ([online](https://claude.ai/artifact/D3jfVwYYDEgfd2oMJSb4tB)) |

The online reports are private links; share them from the page's Share menu.

**Running now** (`scripts/tools/queues/wisdm_final.cmd`, one job at a time; progress in
`logs/wisdm/final.out`, about 15 h in total):
1. the last component ablations of V5 on WISDM (LoRA on W_q/W_v only, no augmentation, no prototypes);
2. the published WISDM protocol (FedAIoT, 400 rounds) with LIMU-BERT, ssl-wearables and the LSTM,
   on three random draws of the 6 test subjects (α = 0.1 and 0.5);
3. UniMTS on WISDM (slowest encoder, last).

When it finishes: `python scripts/07_build_report.py --profile wisdm` rebuilds the WISDM report
(the online version is then republished).

**Paper** (`paper/`, IEEE format, HARBox only): methodology, dataset and results sections are
written; the abstract, introduction, related work and conclusion are still placeholders.

**Lessons for running on this machine** (6 GB GTX 1660 Ti, 16 GB RAM): run at most one heavy
job at a time. Two concurrent jobs plus desktop applications filled the RAM, Windows paged the
jobs out and they made almost no progress for 19 hours. Queues are `.cmd` files started
detached (`Invoke-CimMethod Win32_Process Create`), so they survive closing the terminal.

## HARBox (second dataset, the paper's main benchmark)

[HARBox](https://github.com/xmouyang/FL-Datasets-for-HAR) (Ouyang et al., MobiSys 2021) is the
crowd-sourced smartphone dataset of the federated-HAR literature: 120 users with data, 77 phone
models, 5 activities (call, hop, typing, walk, wave), 9-axis IMU at 50 Hz, 2-s windows.

```powershell
.venv\Scripts\python scripts/01_prepare_data.py --profile harbox      # download (~175 MB) + convert
.venv\Scripts\python run_all.py --profile harbox --from 3             # leakage-free protocol (B)
.venv\Scripts\python run_all.py --profile harbox_fedchar --from 4     # the published protocol (A)
.venv\Scripts\python scripts/07_build_report.py --profile harbox      # report incl. comparison with published results
```

**Two evaluation protocols.** Consecutive HARBox windows overlap by 50 %. The published results
(ClusterFL, FedCHAR) split every user's windows *at random*. We measured that **79.7 % of the test
windows then share half of their samples with a training window**
(`scripts/tools/paper_harbox_stats.py`). We therefore evaluate twice:

| | Protocol A (`harbox_fedchar`) | Protocol B (`harbox`, ours) |
|---|---|---|
| Purpose | comparison with published numbers | all new claims |
| Users | all 120 are clients | 84 clients / 24 new users / 12 pool users |
| Client split | 50–500 random windows, random 72/8/20 % | every activity split in time 80/20, 1-window gap |
| Test windows overlapping training | 79.7 % | 0 % |
| Metric | mean test accuracy over users | the same + macro-F1, fairness, new users, bytes |

**What is different for HARBox** (all in `configs/datasets/harbox.yaml`):
- 9 input channels in physical units; each encoder gets its own format through the preprocessor
  (units, rate, crop or tile) and the input adapter (channels copied by name, e.g. LIMU-BERT takes
  acc + gyro, HALO all 9 axes).
- Training-time augmentation for every method: one random rotation (≤ 30°) applied jointly to the
  acc/gyro/mag triplets + a random gain (`fedpeft/training/augment.py`).
- No HMM: every activity is recorded in its own session.
- Baselines of the published results: the MLP model (900 → 300) with FedAvg, Local and Ditto.
- Few-shot budgets in labelled windows (1 / 5 / 10 per activity).
- Reduced, documented budgets for the slow encoders (ssl-wearables, HALO: 60 rounds; UniMTS: 40).

<!-- HARBOX_RESULTS -->
### HARBox results

Shared-classifier methods (FedAvg, V0–V4) are always scored after every client fine-tuned that
classifier on its own data ("+FT"), so every row compares personalised models. Full tables:
`paper/generated/res_tab_*.tex` (`scripts/tools/paper_results_tables.py`) and
`reports/FedPEFT-IMU_report_harbox.html`.

**Protocol A, the published protocol** (mean test accuracy over the 120 users):

| Method | Model | Accuracy | Source |
|---|---|---|---|
| FedAvg | MLP (300) | 0.694 | FedCHAR, IMWUT 2023, Tab. 4 |
| FedCHAR | MLP (300) | 0.917 | FedCHAR, IMWUT 2023, Tab. 4 |
| FedAvg (our re-run) | MLP | 0.797 (0.815 +FT) | ours |
| Local training, no federation | MLP | 0.878 | ours |
| Ditto | MLP | 0.935 | ours |
| V5 FedPEFT-IMU | LIMU-BERT | 0.940 | ours |
| V5 FedPEFT-IMU, LoRA on all layers | LIMU-BERT | 0.941 | ours |
| Ditto on the pretrained encoder | LIMU-BERT | **0.950** | ours |

Every personalised model we trained beats the best published result (0.917). The lead of Ditto
(pretrained) over V5 (0.009) exists only under this leaky protocol: under protocol B the two are
level (0.875 vs 0.877), and V5 uploads 5.2× less.

Purely local training already reaches 0.878 here, because 79.7 % of the test windows overlap a
training window. All claims below use protocol B.

**Protocol B, leakage-free** (mean user accuracy / macro-F1 of the 84 clients, user split 0):

| Method | LIMU-BERT | ssl-wearables | UniMTS | HALO |
|---|---|---|---|---|
| FedAvg (scratch) | 0.779 / 0.803 | 0.783 / 0.805 | 0.725 / 0.759 | 0.573 / 0.577 |
| V0 full fine-tuning | 0.793 / 0.813 | 0.810 / 0.823 | 0.798 / 0.817 | 0.714 / 0.738 |
| V3 LoRA, exact aggregation | 0.767 / 0.786 | 0.816 / 0.831 | 0.806 / 0.823 | 0.705 / 0.727 |
| **V5 FedPEFT-IMU** | 0.861 / 0.863 | **0.875 / 0.879** | **0.807 / 0.827** | **0.812 / 0.813** |
| **V5, LoRA on all layers** | **0.870 / 0.873** | – | – | – |
| Ditto (pretrained LIMU-BERT) | 0.871 / 0.877 | – | – | – |
| Centralized, pretrained (not private) | 0.814 / 0.829 | 0.847 / 0.857 | 0.824 / 0.836 | 0.813 / 0.824 |
| V5 uplink saving vs V0 | 8× (5.2× all layers) | 52× | 38× | 95× |

On the MLP: Local 0.824, FedAvg 0.766, Ditto 0.868.

**Three user splits** (LIMU-BERT, mean ± std): V5 with all-layer LoRA reaches 0.877 ± 0.015 user
accuracy (worst-10 % 0.581), level with Ditto on the pretrained encoder (0.875 ± 0.008, worst-10 %
0.569) at 5.2× less uplink, against 0.784 ± 0.013 for V0 and 0.769 ± 0.012 for FedAvg.

**New users** (24 held-out users, mean per-user macro-F1). With 10 labelled windows per activity, V5
is the best federated model on LIMU-BERT and HALO. Zero-shot, a shared classifier transfers better.
The transductive rule (below) and all-layer LoRA bring V5 to 0.727 ± 0.054, against 0.743 for V0.

**Added in this round** (each one is an option, so earlier results stay reproducible):

| Change | Where | Effect |
|---|---|---|
| Ditto on the pretrained encoder | `fl.methods: ditto_ft` | strongest baseline |
| LoRA on every linear layer of LIMU-BERT | `encoders.limu_bert.lora_scope=all` | V5 +0.010 F1, V3 +0.031 F1 |
| Transductive zero-shot: V5's prototypes refined on the new user's unlabelled windows (κ chosen on the pool users) | `split.zero_shot_refine_iters=10` | zero-shot +0.017 to +0.035 (none on ssl-wearables) |
| BatchNorm statistics frozen during few-shot adaptation | `fedpeft/training/heldout.py` | ssl-wearables 1-shot 0.751 → 0.796 |
| New-user model saved after every run, re-evaluation without re-training | `scripts/tools/reeval_heldout.py` | |
| Private LoRA copy in the final personal update (negative result) | `fl.personalize_lora=true` | no gain |

```powershell
# re-evaluate the new users of a finished run, e.g. with the transductive zero-shot rule
.venv\Scripts\python scripts/tools/reeval_heldout.py --profile harbox --encoder limu_bert --method v5_proposed split.zero_shot_refine_iters=10 --write --name transductive
# choose such a setting on the pool (validation) users, never on the test users
.venv\Scripts\python scripts/tools/reeval_heldout.py --profile harbox --encoder limu_bert --method v5_proposed --on-pool --zero-only split.zero_shot_refine_iters=10
```

## WISDM-2019 (third dataset, HTML report only)

[WISDM-2019](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
(Weiss, UCI #507): 51 subjects did 18 activities for about 3 minutes each, with a smartwatch on the
dominant wrist (accelerometer + gyroscope, 20 Hz). As in the FedAIoT benchmark we use the
**watch** and FedAIoT's **12 classes** (walking, jogging, stairs, sitting, standing, typing,
brushing teeth, eating, drinking, writing, clapping, folding clothes; "eating" is the sandwich
activity, as in FedAIoT's loader). Windows are 10 s with a 5-s hop.

```powershell
.venv\Scripts\python scripts/01_prepare_data.py --profile wisdm            # download (~310 MB) + convert
.venv\Scripts\python scripts/03_pretrain_halo.py --profile wisdm           # HALO SSL on the 5 pool users
.venv\Scripts\python scripts/04_run_centralized.py --profile wisdm
.venv\Scripts\python scripts/05_run_federated.py --profile wisdm           # leakage-free protocol (B)
.venv\Scripts\python scripts/05_run_federated.py --profile wisdm_fedaiot   # published protocol (A), alpha 0.1
.venv\Scripts\python scripts/05_run_federated.py --profile wisdm_fedaiot05 # published protocol (A), alpha 0.5
.venv\Scripts\python scripts/07_build_report.py --profile wisdm            # the HTML report
```

**Data cleaning** (`fedpeft/data/wisdm.py`). Two defects of the raw files are fixed:
subject 1629's files contain every recording twice (duplicated windows could land in both
training and test), and subjects 1637–1640 logged the accelerometer at ~50 Hz and the gyroscope
at ~25 Hz, so pairing the sensors by identical timestamps (as FedAIoT does) drops most of their
data. Repeated rows are removed and both sensors are interpolated onto one 20-Hz grid:
20,577 windows, 340–412 per subject, classes balanced (8–9 % each).

**Two protocols:**

| | Protocol A (`wisdm_fedaiot`, `wisdm_fedaiot05`) | Protocol B (`wisdm`, ours) |
|---|---|---|
| Purpose | comparison with FedAIoT's published numbers | all other results |
| Clients | 80 clients made by a Dirichlet label partition (α = 0.1 or 0.5) of 45 training subjects | 36 subjects = 36 clients |
| New users | 6 unseen test subjects (3 random draws) | 10 held-out + 5 pool subjects |
| Client split | random 80/10/10 per client | every activity split in time, 80/20, 1-window gap |
| Rounds, clients per round | 400 rounds, 10 % (8) per round, batch 32 (FedAIoT) | 100 rounds, 20 % (7) per round |
| Metric | accuracy of the global model on all windows of the test subjects | per-user accuracy, macro-F1, fairness, new users, bytes |

**Results so far** (protocol B, user split 0, mean user accuracy; shared classifiers after +FT):

| Method | LIMU-BERT | ssl-wearables | HALO | LSTM |
|---|---|---|---|---|
| FedAvg (scratch) | 0.763 | 0.735 | 0.465 | 0.708 |
| Local training | – | – | – | 0.745 |
| Ditto | – | – | – | 0.839 |
| V0 full fine-tuning | 0.808 | 0.872 | 0.712 | – |
| Ditto (pretrained) | 0.914 | – | – | – |
| V3 LoRA exact | 0.805 | 0.871 | 0.743 | – |
| **V5 FedPEFT-IMU** | 0.908 | **0.957** | **0.864** | – |
| Centralized, pretrained (not private) | 0.875 | 0.941 | 0.886 | – |
| V5 uplink saving vs V0 | 6× | 51× | 93× | |

- **Three user splits (LIMU-BERT):** V5 0.911 ± 0.002 (worst-10 % 0.787) vs Ditto (pretrained)
  0.908 ± 0.006 (0.759) vs V0 0.820 ± 0.018 (0.471), at 6× less uplink than Ditto.
- **New users:** zero-shot, a shared classifier transfers better (V0 0.705 vs V5 0.653 macro-F1 on
  LIMU-BERT); with one labelled window per activity V5 is ahead (0.852 vs 0.829), with ten it is the
  best model (0.910 LIMU-BERT, 0.922 ssl-wearables).
- **Ablation (LIMU-BERT):** LoRA on all linear layers 0.908 vs W_q/W_v only 0.885; without local
  norm and adapter 0.894; a shared classifier fine-tuned by each client 0.916 (as on HARBox, the
  local norms carry most of the personalisation).
- **Published protocol (in progress):** centralized, our best model reaches 0.851 accuracy on the
  unseen test subjects (ssl-wearables) against FedAIoT's 0.741; federated with α = 0.5, LIMU-BERT V0
  reaches 0.761 against FedAIoT's best 0.715. The α = 0.1 runs and the extra test-subject draws are
  running. Our re-run of FedAIoT's LSTM (0.592 at α = 0.1 on one draw) is below their 0.685.

## Client participation and device tiers

This section explains which users become clients, how many clients train in each round and how
they are picked, and how each client gets its LoRA rank tier (V4, V5).

### Which users are clients

Users are assigned to roles once per user split, at random: `make_user_split`
(`fedpeft/data/splits.py`) shuffles the user IDs with the split seed and takes 20 % as held-out
(new) users, 10 % as pool users and the remaining ~70 % as FL clients. The assignment is random
so that no property of a user (amount of data, activities, device) decides its role, and it is
saved to `results/splits/` so every run of a split uses the same users. Repeating the experiments
on seeds 1 and 2 (three different assignments) shows how much a result depends on who happens to
be a client. In WISDM protocol A the 6 test subjects are drawn the same way, and the windows of
the other 45 are dealt to 80 clients by a Dirichlet label partition (the FedAIoT protocol), so a
client there is a mix of subjects, not a person.

### How many clients train per round, and how they are chosen

| Dataset / protocol | Clients | Per round | Rounds | Participations per client (min / median / max) |
|---|---|---|---|---|
| Capture-24 (`enhanced`) | 106 | 21 (20 %) | 80 | 8 / 15 / 26 |
| HARBox B (`harbox`) | 84 | 17 (20 %) | 100 | 14 / 20 / 29 |
| HARBox A (`harbox_fedchar`) | 120 | 60 (50 %) | 100 | 38 / 50 / 62 |
| WISDM B (`wisdm`) | 36 | 7 (20 %) | 100 | 9 / 19 / 30 |
| WISDM A (`wisdm_fedaiot*`) | 80 | 8 (10 %) | 400 | 28 / 40 / 54 |

- **How many.** In cross-device FL only part of the devices is online and charging at any time,
  so a round uses a fraction of the clients (`fl.clients_per_round_frac`). Our own protocols use
  **20 %**, the usual setting since FedAvg (McMahan et al., 2017, used 10 %): enough clients per
  round for a stable average, few enough that the simulation stays affordable on one GPU. When we
  reproduce a published protocol we use **its** fraction: 50 % for FedCHAR (HARBox A) and 10 % for
  FedAIoT (WISDM A), so the comparison with their numbers is like for like.
- **How.** Every round draws `round(fraction × clients)` clients **uniformly at random without
  replacement** (`FederatedRun.run`, `self.rng.choice(..., replace=False)`).
- **Why uniform random.** It is the unbiased standard of FedAvg: no client is favoured for having
  more data, a lower loss or a faster device, so the result is not tied to a selection heuristic
  (power-of-choice, loss-based or resource-aware selection would change who trains and would mix
  that effect into the method comparison). Over the rounds every client takes part regularly (no
  client in any table above was never selected).
- **Same clients for every method.** The generator is seeded with `seed + 2000` and is used for
  nothing else, so for a given split every method trains with exactly the same clients in exactly
  the same rounds. Differences between methods are therefore not due to luckier client draws.
- **Evaluation** always uses **all** clients (each on its own test part), not only the ones that
  trained in the last round.

### How clients are assigned to rank tiers (V4, V5)

V4 and V5 simulate a mixed fleet of devices: every client trains LoRA adapters of rank **2, 4 or
8** (`model.lora.tiers`), standing for low-, mid- and high-end devices. All other methods use one
rank (8) for everybody.

- **How.** When a run starts, each client gets its tier **once, uniformly at random** from
  {2, 4, 8}, with a generator seeded with `seed + 1000` (`FederatedRun.__init__`). The tier is
  fixed for the whole training (a phone does not change hardware between rounds), and V4 and V5
  get the same assignment, so they are compared on the same fleet.
- **Why random.** None of the datasets records how capable a device is: Capture-24 uses one
  Axivity sensor model, WISDM one smartwatch model, and HARBox names 77 phone models but without
  memory or compute figures. A random assignment simulates a mixed fleet without inventing a
  capability proxy. It also keeps the tier independent of the user's data: if the weakest devices
  also had the hardest users, a lower score on rank-2 devices could not be told apart from the
  users being hard. With random tiers, the per-tier scores measure the cost of a lower rank alone.
  Studies of heterogeneous-rank LoRA in FL (e.g. HetLoRA, FlexLoRA) likewise give clients
  simulated ranks or resource budgets.
- **Resulting fleets** (split 0): HARBox B 31 / 23 / 30 clients at rank 2 / 4 / 8; WISDM B
  15 / 10 / 11; HARBox A 44 / 33 / 43; WISDM A 30 / 21 / 29; Capture-24 37 / 29 / 40.
- **What a tier changes.** A rank-r client trains and uploads only rank-r factors and receives the
  r strongest directions of the global update (the server re-factorises the aggregate with an SVD).
  Its update is aggregated in delta form, so a rank-2 client cannot erase the directions it never
  received. The saving is mainly **communication**: per update, rank 2 / 4 / 8 upload 23 / 25 / 30 kB
  (LIMU-BERT, HARBox) and 0.51 / 0.75 / 1.23 MB (ssl-wearables). Measured training memory is almost
  the same for all tiers (e.g. 63 MB on LIMU-BERT, 353–355 MB on ssl-wearables), because the
  activations of the frozen encoder dominate; on real devices a lower rank mainly saves
  bandwidth, optimizer state and adapter storage.
- **Low tiers are not penalised** (V5, mean client F1 per tier):

  | Run | rank 2 | rank 4 | rank 8 |
  |---|---|---|---|
  | HARBox B, LIMU-BERT | 0.879 | 0.850 | 0.835 |
  | WISDM B, LIMU-BERT | 0.895 | 0.912 | 0.888 |
  | WISDM B, ssl-wearables | 0.950 | 0.952 | 0.965 |

  A client keeps its tier for every encoder and method, so these columns compare **groups of
  users** as much as ranks. On HARBox the rank-2 group happens to hold easier users: it is ahead for
  V4 as well (0.756 vs 0.667 at rank 8) and for every encoder. The clean test of the rank itself is to
  give every client rank 8 (`v5_abl_fixed_rank`, HARBox), which changes macro-F1 only from 0.863 to
  0.866. The report shows this table for every encoder (Setup → "Who trains each round, and on which
  device tier").
  On Capture-24 the rank-8 clients were ahead (0.632 / 0.667 / 0.698 with ssl-wearables), so a
  small capacity cost of low ranks can appear on harder, more imbalanced data.
- **Limitation.** The tiers are simulated. Assigning them from real device capabilities (for
  example the memory of HARBox's phone models) would be a more realistic test and is left open.

## Capture-24 headline results (`enhanced` profile, seed 0)

Macro-F1 on the last 20 % of every client's recording (6 classes, Capture-24, 106 clients).
The complete set of KPIs is in `reports/FedPEFT-IMU_report_enhanced.html`; the earlier
pipelines are in `..._baseline.html` (first run) and `..._standard.html` (class-imbalance fixes).

**From the first run to the final pipeline**: mean macro-F1 of the seven federated methods per encoder.

| Encoder | First run | + class-imbalance fixes | + longer training (per window) | + HMM | + local fine-tuning + HMM |
|---|---|---|---|---|---|
| LIMU-BERT | 0.282 | 0.364 | 0.361 | 0.363 | **0.383** |
| ssl-wearables | 0.612 | 0.639 | 0.651 | 0.714 | **0.737** |
| UniMTS ¹ | 0.425 | 0.538 | 0.528 | 0.623 | **0.651** |
| HALO | 0.418 | 0.516 | 0.526 | 0.601 | **0.639** |

**Per method** (per window unless stated):

| Encoder | Centralized pretrained (+HMM) | FedAvg scratch | V0 full FT | V3 LoRA exact | **V5 proposed** | Best federated, full pipeline | V5 uplink vs V0 |
|---|---|---|---|---|---|---|---|
| LIMU-BERT | 0.535 (0.664) | 0.360 | 0.372 | 0.360 | **0.449** | 0.441 (FedAvg) | 9× less |
| ssl-wearables | 0.709 (0.802) | 0.432 | 0.679 | 0.688 | **0.697** | 0.781 (V1) | 50× less |
| UniMTS ¹ | 0.580 (0.712) | 0.208 | 0.593 | 0.562 | **0.646** | 0.737 (V2) | 36× less |
| HALO | 0.658 (0.740) | 0.345 | 0.571 | 0.580 | **0.629** | 0.715 (V2) | 91× less |

¹ UniMTS ran with a reduced compute budget (40 rounds, smaller updates); compare its methods with each other.

Main observations (all from one user split; confirm with the `full` profile):
- The first run lost most of its macro-F1 on three rare activities that federated training erased
  (bicycling, vehicle, walking at F1 = 0). The class-imbalance fixes recover them: +0.08 macro-F1
  on average, up to +0.20, and 35 → 13 classes stuck at zero.
- HMM temporal smoothing and local fine-tuning add another +0.06 to +0.13 on top; the best federated
  model reaches **0.78** macro-F1.
- **V5 (FedPEFT-IMU) has the best per-window macro-F1 on all four encoders**
  while uploading 9–91× less than full fine-tuning.
- Exact LoRA aggregation (V3) is no better than naive averaging (V2) here, and it must ship a dense residual,
  so its downlink is about as large as full fine-tuning.
- Small LoRA ranks suffice: r = 2 is as good as r = 16 on ssl-wearables.
- A 10-s window for LIMU-BERT (stretched position embeddings) is worse than its native 6 s; the
  rejected runs are kept in `results/enhanced_limu10s`.

---

## Why the first results were low, and what was changed

**Diagnosis.** In the first run (`results/baseline`) the frequent activities were already good (sleep
F1 ≈ 0.90, sit-stand ≈ 0.75). The macro-F1 was lost on the rare ones: bicycling (1 % of the
windows), vehicle (3.8 %) and walking (6.5 %) scored **F1 = 0.000** in most federated models.
Macro-F1 averages the six classes, so each zero costs about 0.17.

The cause is specific to federated learning with label skew. Most participants never cycle or
drive. With plain cross-entropy, every client without bicycling windows pushes the bicycling
score *down* on every step. FedAvg then averages the few clients who do cycle with about 20
who do not, so the class is erased from the shared model.

**Fixes** (identical rules for every method; turn them off with `--profile baseline`):

| # | Change | Where | Setting |
|---|---|---|---|
| 1 | **Balanced-softmax loss**: CE(f(x) + log π, y). Centrally π is the global label prior; in FL each client uses its *own* prior, so a class it lacks (log π ≈ −9) is no longer pushed down | `training/imbalance.py`, `engine.py` | `imbalance.tau_train: 1.0` |
| 2 | **Class-wise classifier aggregation**: row *c* of the shared classifier is averaged only over the clients that have class *c* (weighted by their count). Before, a rare class's update was divided by about 20 every round | `federated/aggregation.py` (`ClassRowAccumulator`) | `fl.classwise_head_aggregation: true` |
| 2b | **Head learning rate** for the full-model methods (FedAvg, V0): heads train at the PEFT rate, the body at the fine-tuning rate (discriminative learning rates). With one small rate, rare-class rows barely moved in 40 rounds | `federated/simulation.py` (`param_groups`) | `fl.lr_head: 2.0e-3` |
| 3 | **Calibrated decisions**: predict argmax f(x) + τ·log π<sub>global</sub>. τ = 0.5 was chosen on the validation users (τ = 0 → 0.42, τ = 0.5 → 0.46 validation F1 for LIMU-BERT; 0.53 → 0.63 for ssl-wearables); the test set was never used. V5's nearest-prototype rule validated best with τ = 0 | `training/imbalance.py`, `decoding.py` | `imbalance.tau_inf: 0.5`, `tau_inf_prototype: 0.0` |
| 4 | **HMM temporal smoothing**, the standard Capture-24 post-processing. Transitions come from the training label sequences, federated as 36 counts per client. Reported *in addition to* the per-window scores | `training/hmm.py` | `decoding.hmm: true` |
| 5 | **V5 for real devices**: V5's classifier is *personal*, so it keeps plain cross-entropy, argmax decisions and its own client's prior. Fixes 1 and 3 correct *shared* classifiers; applied to personal heads they cost 0.01–0.04 F1. One last update of each device's personal parts before testing. New users estimate BatchNorm statistics from their own *unlabelled* windows (AdaBN). Few-shot fitting corrects for its balanced label sample | `federated/simulation.py`, `training/heldout.py` | `fl.final_personalization_steps: 8`, `split.adabn_new_users: true` |

**Enhanced profile.** The `standard` learning curves were still rising at round 40 for most
models, and FL had done only half the training passes of the centralized reference. So
`--profile enhanced` keeps all the fixes above and adds four more changes:
- 80 rounds (UniMTS 40), the same number of passes over the data as centralized training.
- A **+FT** score: each client fine-tunes the shared classifier on its own data (the standard
  "FedAvg + fine-tuning" protocol), reported next to the shared-model score.
- HALO is pretrained on all the pool users' windows.

We also tried feeding LIMU-BERT the whole 10-s window, with its 120 pretrained position embeddings
stretched to 200. That helped only the from-scratch model and made the pretrained model worse in
FL, so the official 6-s window is kept. The rejected runs are in `results/enhanced_limu10s`.

Everything else (data, splits, rounds, budgets, encoders) is unchanged between `baseline` and `standard`, so the before/after
comparison in the report isolates these fixes. Raw test and validation logits of every run are
saved in `results/<profile>/logits/`, so other decoding rules can be studied without retraining.

---

## 1. Quick start

```powershell
# one-time: virtual environment with CUDA PyTorch (GTX 1660 Ti -> CUDA 12.8 wheels)
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt

# everything: data -> weights -> HALO pretraining -> phase 1 -> phase 2 -> ablation -> report
.venv\Scripts\python run_all.py --profile standard

# a 10-minute smoke test of the whole pipeline (numbers are not meaningful)
.venv\Scripts\python run_all.py --profile quick
```

Open `reports/FedPEFT-IMU_report_standard.html` in any browser. It works offline.

Every step can also be run on its own and **skips finished work**:

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/01_prepare_data.py` | downloads Capture-24 (6.5 GB), converts it to 10-s windows at 30 Hz |
| 2 | `scripts/02_download_weights.py` | official LIMU-BERT, ssl-wearables and UniMTS weights + load check |
| 3 | `scripts/03_pretrain_halo.py` | self-supervised HALO pretraining (no public weights exist) |
| 0 | `scripts/00_benchmark_encoders.py` | throughput and memory of the encoders on your GPU |
| 4 | `scripts/04_run_centralized.py` | **phase 1**: 4 encoders × {scratch, pretrained-FT} |
| 5 | `scripts/05_run_federated.py` | **phase 2**: 4 encoders × 7 FL methods |
| 6 | `scripts/06_run_ablations.py` | LoRA rank sweep r ∈ {2, 4, 8, 16} |
| 7 | `scripts/07_build_report.py` | the interactive HTML report |

Options: `--encoders halo,unimts`, `--methods v3_lora_exact,v5_proposed`, `--rerun`, and any
config override as `key.sub=value` (for example `fl.rounds=100`).

---

## 2. Project layout

```
idea1/
├── configs/
│   ├── default.yaml            every setting, commented (the single source of truth)
│   └── profiles/               quick | standard | full  (deep-merged over default.yaml)
├── fedpeft/                    the Python package
│   ├── config.py               YAML + profile + CLI overrides, per-encoder budgets
│   ├── utils.py                seeding, timing, JSON, robust downloads
│   ├── data/
│   │   ├── capture24.py        raw CSV -> 10-s windows (anti-aliased 100 -> 30 Hz)
│   │   ├── splits.py           user split (clients / held-out / pool) + time split
│   │   └── store.py            all windows on the GPU, index tensors per user/part
│   ├── models/
│   │   ├── components.py       1 preprocessor · 2 input adapter · 6 shared head · 7 personal head · 8 prototypes
│   │   ├── lora.py             4 LoRA for Linear / Conv1d / Conv2d (+ merged inference)
│   │   ├── client_model.py     the full 8-component model and the ROLE of every tensor
│   │   └── encoders/           3 limu_bert · ssl_wearables · unimts · halo
│   ├── training/
│   │   ├── engine.py           local training loop (CE + λ·L_proto), prediction, prototypes
│   │   ├── metrics.py          macro-F1, accuracy, worst-10 %, std, confusion
│   │   ├── heldout.py          zero-/few-shot evaluation of new users
│   │   ├── centralized.py      phase 1 runner
│   │   └── ssl_halo.py         HALO MAE + patch-contrastive pretraining
│   ├── federated/
│   │   ├── methods.py          FedAvg, V0-V5 as data: what is trained / sent / kept local
│   │   ├── aggregation.py      FedAvg, LoRA naive / exact (FedEx) / exact-SVD, prototypes
│   │   ├── server.py           global state + streaming aggregation
│   │   ├── comm.py             byte accounting of every message
│   │   └── simulation.py       the FL round loop and all KPIs
│   └── report/                 collect.py (KPIs) -> build.py -> template.html (Plotly)
├── scripts/                    numbered entry points (see table above)
│   └── tools/paper_dataset_stats.py   every dataset number / table / figure used in the paper
├── paper/                      journal manuscript (IEEEtran): methodology + dataset sections,
│                               TikZ architecture figures; build with python paper/build.py
├── tests/test_core.py          LoRA and aggregation maths (python -m pytest tests)
├── data/  weights/  results/  reports/  logs/     created by the pipeline
└── run_all.py
```

### How the pieces fit (one FL round)

```
server (global LoRA A,B · W0+residual · heads · prototypes)
   │ 1 download: first r_k LoRA rows + federated heads (+ prototypes)
   ▼
client k:  window ─► Preprocessor ─► InputAdapter ─► Encoder(W0 frozen + LoRA) ─► g ─► h
           2-3 local training: CE + λ·L_proto    (norm / adapter / h stay on the device in V5)
           4 local class prototypes
   │ 5 upload ONLY the federated roles  (bytes measured)
   ▼
server 6 aggregation: FedAvg · naive avg(B)avg(A) · exact avg(BA) + residual · SVD re-factorisation
```

The central design idea is that **methods are data, not code**. Every tensor in the client
model has a *role* (`encoder`, `lora`, `norm`, `norm_stats`, `adapter`, `shared_head`,
`head`), and a method (`federated/methods.py`) says which roles are trained, which are
sent, and which stay on the device. Adding a new variant is a few lines.

---

## 3. What exactly is run

**Dataset.** Capture-24 has 151 participants, a wrist Axivity AX3 at 100 Hz, and about 24 h of
free-living data each. Labels use the `Willetts2018` scheme with 6 classes: sleep, sit-stand,
mixed, walking, vehicle, bicycling. The data is cut into 10-s windows (majority label) and
resampled to 30 Hz with an anti-aliasing filter. The `standard` profile keeps one window per
minute, which preserves the 24-h class mix at a sixth of the compute.

**Protocol** (PDF p. 10 and the appendix):
- Users are split by ID into **70 % FL clients**, **20 % held-out users** and a **10 % pool**.
  The pool provides unlabelled HALO pretraining data and validation users.
- Inside each client, the **first 80 % of the recording trains and the last 20 % tests**.
  The split is never random by window.
- Held-out users: the first 50 % supplies few-shot labels (1 or 5 minutes per activity) and the
  last 50 % is tested.
- The user IDs of every split are written to `results/splits/` so they can be published.
- How many clients train per round, how they are drawn and how V4/V5 assign LoRA rank tiers is
  explained in [Client participation and device tiers](#client-participation-and-device-tiers).

**Encoders** (component 3). Each is a faithful re-implementation, so its official weights load
tensor for tensor:

| Encoder | Input | Params | Pretrained weights |
|---|---|---|---|
| LIMU-BERT | 6 ch, 20 Hz, 6 s | 0.05 M | official, HHAR (phones + watches) |
| ssl-wearables (harnet10) | 3 ch, 30 Hz, 10 s | 10.5 M | official, UK-Biobank wrist (700k person-days) |
| UniMTS (ST-GCN) | 3 ch on the wrist joint of a 22-joint skeleton, 20 Hz, 10 s | 5.2 M | official, motion-capture-synthesised IMU aligned with text |
| HALO | 3 ch, 30 Hz, 1-s patches | 4.6 M | **our** SSL pretraining (MAE + patch contrastive) on pool users. HALO's weights are not public yet |

**FL methods** (PDF p. 4):

| | Trained | Sent | Kept on device | Aggregation |
|---|---|---|---|---|
| FedAvg (scratch) | all, random init | all | — | FedAvg |
| V0 full FT | all, pretrained | all | — | FedAvg |
| V1 head only | g, h | g, h | — | FedAvg |
| V2 LoRA naive | LoRA r=8, g, h | same | — | avg(B)·avg(A) |
| V3 LoRA exact | LoRA r=8, g, h | same | — | FedEx-LoRA: avg(B·A), residual into W0 |
| V4 hetero ranks | LoRA r=2/4/8, g, h | same | — | zero-pad, exact, SVD re-factorisation |
| V5 **proposed** | LoRA r=2/4/8, g, h, norm, adapter | LoRA, g, prototypes | norm, adapter, h | V4 + prototype averaging |

---

## 4. Design decisions worth knowing

- **Measured, not estimated, communication.** Every upload and download is an explicit dict of
  tensors. Its fp32 size is what Flower would transmit. Exact aggregation (V3–V5) also has to
  send the dense residual to clients, and the report shows that downlink cost.
- **Exact aggregation with heterogeneous ranks.** Each client's *change* relative to the
  factors it received is aggregated. Rank-2 devices therefore cannot erase directions they never
  saw. With equal ranks this equals the PDF formula. The server re-factorises the result with an
  SVD, and whatever the rank-r factors miss goes into the residual, so the aggregate stays exact.
  This is verified in `tests/test_core.py`.
- **Prototype loss on normalised embeddings.** The PDF's squared distance is computed between
  L2-normalised vectors (2 − 2·cos). This makes it scale-free, so it cannot be minimised by
  shrinking the embedding, and it behaves the same for all four encoders. Zero-shot uses the
  cosine nearest prototype.
- **Per-encoder compute budget.** UniMTS runs a graph network over 22 joints, which costs about
  7 GMAC per window. That is about 70 windows/s on a GTX 1660 Ti, versus about 1,200 for
  ssl-wearables. `encoders.unimts.budget` in `default.yaml` therefore gives UniMTS fewer
  rounds and local steps. All UniMTS methods share this budget, so they stay comparable with
  each other. The report marks it.
- **fp32 on GTX 16xx.** These GPUs have no tensor cores, and mixed precision measured 2.5×
  *slower*, so `runtime.amp` is off.

## 5. Profiles

| Profile | Purpose | Data | FL rounds | Seeds | Runtime (GTX 1660 Ti) |
|---|---|---|---|---|---|
| `quick` | smoke test | 1 window / 5 min | 4 | 1 | ~15 min |
| `baseline` | the first pipeline, without the class-imbalance fixes (for comparison) | 1 window / min | 40 (UniMTS 25) | 1 | ~8 h |
| `enhanced` | **best results**: `standard` + FL as long as centralized, +FT scores, HALO pretrained on 2× data | 1 window / min | 80 (UniMTS 40) | 1 | ~10 h (two queues) |
| `standard` | the delivered report | 1 window / min | 40 (UniMTS 25) | 1 | ~8 h (two parallel queues) |
| `full` | paper-grade (PDF p. 12) | 1 window / 30 s | 200 | 3 | days |
| `harbox` | HARBox, leakage-free protocol B | all 2-s windows | 100 (ssl-wearables, HALO 60; UniMTS 40) | 1 (+2 for LIMU-BERT, MLP) | ~1 day |
| `harbox_fedchar` | HARBox, published protocol A (FedCHAR) | 50–500 random windows / user | 100 | 1 | a few hours |
| `wisdm` | WISDM-2019 watch, leakage-free protocol B | all 10-s windows | 100 (ssl-wearables, HALO 60; UniMTS 30) | 1 (+2 for LIMU-BERT, LSTM) | ~1 day |
| `wisdm_fedaiot`, `wisdm_fedaiot05` | WISDM-2019 watch, published protocol A (FedAIoT), α = 0.1 / 0.5 | all 10-s windows | 400 | 3 test-subject draws | ~10 h |

### Running faster on one GPU

The GPU work splits naturally into two independent queues. Running them in two terminals
roughly halves the wall-clock time, because the small models leave the GPU partly idle:

```powershell
# terminal 1
.venv\Scripts\python scripts/03_pretrain_halo.py
.venv\Scripts\python scripts/04_run_centralized.py --encoders ssl_wearables,halo
.venv\Scripts\python scripts/05_run_federated.py --encoders ssl_wearables,halo
.venv\Scripts\python scripts/06_run_ablations.py
# terminal 2
.venv\Scripts\python scripts/04_run_centralized.py --encoders limu_bert,unimts
.venv\Scripts\python scripts/05_run_federated.py --encoders limu_bert,unimts
# afterwards
.venv\Scripts\python scripts/07_build_report.py
```

**Watch the GPU memory on Windows.** When two processes need more than the card's 6 GB,
Windows does not raise an out-of-memory error. It silently moves GPU memory into system RAM,
and training becomes more than 10× slower. UniMTS therefore trains and evaluates with small
batches (`encoders.unimts.budget`), which keeps it near 2 GB.

## 6. References

- Capture-24: Chan et al., *Sci Data* 11, 1135 (2024)
- LIMU-BERT: Xu et al., SenSys 2021 · ssl-wearables: Yuan et al., *npj Digit. Med.* 2024
- UniMTS: Zhang et al., NeurIPS 2024 · HALO: arXiv 2608.27233
- LoRA: Hu et al., ICLR 2022 · FedEx-LoRA, FFA-LoRA, HetLoRA, FLoRA (see PDF p. 8)
- Logit adjustment / balanced softmax: Menon et al., ICLR 2021 · label skew in FL: FedLC (Zhang et al., ICML 2022),
  FedRS (Li & Zhan, KDD 2021), CReFF (Shang et al., IJCAI 2022) · AdaBN: Li et al., 2016
- HMM smoothing for Capture-24: Walmsley et al., Int J Obesity 2021; Chan et al., Sci Data 2024

"""
Unit tests for the mathematical core of FedPEFT-IMU.

    .venv\\Scripts\\python -m pytest tests -q        (or simply: python tests/test_core.py)

What is checked
    1. A LoRA layer computes exactly  W0 x + (alpha/r) B A x  for Linear, Conv1d
       (incl. circular padding) and Conv2d, in training (two-branch) and eval (merged) mode.
    2. The worked example of PDF Figure L7: naive averaging gives the wrong product.
    3. FedEx-style exact aggregation (V3) reproduces W0 + s * avg(B_k A_k) exactly.
    4. Exact aggregation with heterogeneous ranks + SVD re-factorisation (V4/V5)
       reproduces the exact averaged update, and truncation keeps the top directions.
    5. Upload sizes are counted as fp32 bytes.
    6. Class-imbalance helpers, HMM smoothing and class-wise classifier aggregation.
    7. Multi-dataset input pipeline: augmentation rotates acc/gyro/mag together, the input
       adapter copies channels by name, the preprocessor converts units and tiles short
       windows (HARBox), and the HARBox splits are disjoint (per-activity time split with
       a one-window gap; random split of 50-500 windows).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedpeft.federated.aggregation import LoRAAccumulator, svd_refactor  # noqa: E402
from fedpeft.federated.comm import payload_bytes  # noqa: E402
from fedpeft.models.lora import LoRALayer  # noqa: E402

torch.manual_seed(0)


def _check_layer(base: nn.Module, x: torch.Tensor) -> None:
    layer = LoRALayer(base, rank=4, scaling=2.0, dropout=0.0)
    with torch.no_grad():
        layer.lora_B.normal_()                      # make B A non-zero
    delta = layer.delta_weight().view_as(base.weight)
    expected_weight = base.weight + delta
    if isinstance(base, nn.Linear):
        expected = nn.functional.linear(x, expected_weight, base.bias)
    else:
        expected = base._conv_forward(x, expected_weight, base.bias)
    layer.train()
    assert torch.allclose(layer(x), expected, atol=1e-4), "merged LoRA forward (training) is wrong"
    layer.eval()
    assert torch.allclose(layer(x), expected, atol=1e-4), "merged LoRA forward (inference) is wrong"
    # Two-branch path (used when LoRA dropout > 0): a zero-probability Dropout keeps it deterministic.
    layer.dropout = nn.Dropout(0.0)
    layer.train()
    assert torch.allclose(layer(x), expected, atol=1e-4), "two-branch LoRA forward is wrong"


def test_lora_layers_match_merged_weight():
    _check_layer(nn.Linear(12, 7), torch.randn(5, 12))
    _check_layer(nn.Conv1d(6, 9, 5, padding=2, padding_mode="circular", bias=False), torch.randn(3, 6, 40))
    _check_layer(nn.Conv2d(4, 8, (9, 1), stride=(2, 1), padding=(4, 0)), torch.randn(2, 4, 30, 22))


def test_naive_average_is_wrong_pdf_example():
    # PDF Figure L7: two clients, rank 1, 2x2 layer.
    B1, A1 = torch.tensor([[1.0], [0.0]]), torch.tensor([[1.0, 0.0]])
    B2, A2 = torch.tensor([[0.0], [1.0]]), torch.tensor([[0.0, 1.0]])
    acc = LoRAAccumulator("lora_naive", torch.zeros(1, 2), torch.zeros(2, 1), scaling=1.0)
    acc.add(A1, B1, 1.0)
    acc.add(A2, B2, 1.0)
    A, B, residual, err = acc.result()
    assert residual is None
    assert torch.allclose(B @ A, torch.full((2, 2), 0.25)), "naive product should be all 0.25"
    assert err > 0.5, "the naive aggregation error must be large in this example"


def _random_clients(ranks, d_out=16, d_in=24, r_max=8, s=2.0):
    A0 = torch.randn(r_max, d_in)
    B0 = torch.randn(d_out, r_max) * 0.1
    clients = []
    for r in ranks:
        clients.append((A0[:r] + 0.1 * torch.randn(r, d_in), B0[:, :r] + 0.1 * torch.randn(d_out, r), float(torch.randint(1, 10, ()))))
    return A0, B0, clients, s


def test_fedex_exact_aggregation():
    A0, B0, clients, s = _random_clients([8, 8, 8])
    W0 = torch.randn(16, 24)
    acc = LoRAAccumulator("lora_exact", A0, B0, scaling=s)
    for A, B, w in clients:
        acc.add(A, B, w)
    A, B, residual, _ = acc.result()
    W_total = sum(w for *_, w in clients)
    exact = W0 + s * sum(w * (Bk @ Ak) for Ak, Bk, w in clients) / W_total
    assert torch.allclose(W0 + residual + s * (B @ A), exact, atol=1e-5)


def test_svd_exact_aggregation_with_heterogeneous_ranks():
    A0, B0, clients, s = _random_clients([2, 4, 8, 4])
    W0 = torch.randn(16, 24)
    acc = LoRAAccumulator("lora_exact_svd", A0, B0, scaling=s)
    for A, B, w in clients:
        acc.add(A, B, w)
    A, B, residual, _ = acc.result()
    W_total = sum(w for *_, w in clients)
    # Exact target: old global update + average CHANGE of every client.
    M = B0 @ A0 + sum(w * (Bk @ Ak - B0[:, :Ak.shape[0]] @ A0[:Ak.shape[0]]) for Ak, Bk, w in clients) / W_total
    assert torch.allclose(W0 + residual + s * (B @ A), W0 + s * M, atol=1e-4)
    # The first directions are the dominant ones (sorted singular values).
    norms = [(B[:, i:i + 1] @ A[i:i + 1]).norm() for i in range(A.shape[0])]
    assert all(norms[i] >= norms[i + 1] - 1e-5 for i in range(len(norms) - 1))


def test_svd_refactor_revives_dead_directions():
    M = torch.randn(10, 3) @ torch.randn(3, 20)          # rank 3 < requested rank 8
    A, B = svd_refactor(M, 8)
    assert torch.allclose(B @ A, M, atol=1e-4)
    assert (A[3:].abs().sum(1) > 0).all(), "zero-singular directions must get fresh A rows"
    assert torch.all(B[:, 3:] == 0)


def test_payload_bytes_fp32():
    payload = {"dense": {"w": torch.zeros(10, 10)}, "lora": {"l": (torch.zeros(8, 5), torch.zeros(3, 8))}, "n_k": 7}
    assert payload_bytes(payload) == (100 + 40 + 24) * 4 + 8


def test_class_prior_floor_and_adjustment():
    from fedpeft.training.imbalance import class_prior, logit_adjustment
    prior = class_prior([90, 10, 0])                   # third class absent on this client
    assert abs(float(prior.sum()) - 1.0) < 1e-6 and float(prior[2]) > 0
    adj = logit_adjustment(prior, 1.0)
    assert float(adj[2]) < -8, "absent classes must get a very negative adjustment"


def test_hmm_smooths_isolated_flip_and_respects_breaks():
    import numpy as np
    from fedpeft.training.hmm import ActivityHMM
    hmm = ActivityHMM(2, smoothing=0.0)
    y = np.array([0] * 50 + [1] * 50)
    breaks = np.zeros(100, dtype=bool); breaks[0] = True
    trans, init = ActivityHMM.count_transitions(y, breaks, 2)
    assert trans.sum() == 99 and trans[0, 1] == 1 and init[0] == 1
    hmm.add_counts(trans + 0.5, init + 0.5)
    # Emissions favour class 0 everywhere except one noisy window in the middle.
    scores = torch.full((20, 2), 0.0); scores[:, 0] = 2.0; scores[10] = torch.tensor([0.0, 2.5])
    no_break = np.zeros(20, dtype=bool); no_break[0] = True
    path = hmm.smooth(scores, no_break, torch.log(torch.tensor([0.5, 0.5])), tau_train=1.0)
    assert (path == 0).all(), "a single implausible window should be smoothed away"
    # With a sequence break right at that window it can no longer be smoothed from the left.
    br = no_break.copy(); br[10] = True; br[11] = True
    path2 = hmm.smooth(scores, br, torch.log(torch.tensor([0.5, 0.5])), tau_train=1.0)
    assert path2[10] == 1


def test_classwise_head_rows_ignore_clients_without_the_class():
    from fedpeft.federated.aggregation import ClassRowAccumulator
    ref = {"w": torch.zeros(3, 2), "b": torch.zeros(3)}
    acc = ClassRowAccumulator(ref)
    acc.add({"w": torch.ones(3, 2), "b": torch.ones(3)}, torch.tensor([10.0, 0.0, 0.0]))       # client A: class 0 only
    acc.add({"w": 3 * torch.ones(3, 2), "b": 3 * torch.ones(3)}, torch.tensor([10.0, 5.0, 0.0]))  # client B: classes 0, 1
    out = acc.result()
    assert torch.allclose(out["w"][0], torch.full((2,), 2.0))      # both clients, equal counts
    assert torch.allclose(out["w"][1], torch.full((2,), 3.0))      # only client B has class 1
    assert torch.allclose(out["b"][2], torch.tensor(0.0))           # nobody has class 2: row unchanged


def test_augmenter_rotates_sensor_triplets_together():
    from fedpeft.training.augment import Augmenter
    x = torch.randn(16, 9, 50)
    x[:, 3:6] = x[:, 0:3]                          # gyro = acc, so a shared rotation keeps them equal
    aug = Augmenter([(0, 1, 2), (3, 4, 5), (6, 7, 8)], 9, rotation_deg=45, scale_std=0.0, p=1.0, seed=1)
    y = aug(x)
    assert torch.allclose(y[:, 0:3].norm(dim=1), x[:, 0:3].norm(dim=1), atol=1e-4)   # rotations keep |a|
    assert torch.allclose(y[:, 0:3], y[:, 3:6], atol=1e-5)                            # same rotation for both
    assert not torch.allclose(y[:, 0:3], x[:, 0:3], atol=1e-3)                        # and it did rotate


def test_input_adapter_copies_channels_by_name():
    from fedpeft.models.components import InputAdapter
    data = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z"]
    enc = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]               # LIMU-BERT
    w = InputAdapter(data, enc).proj.weight[:, :, 0]
    assert torch.equal(w, torch.eye(9)[:6])                                          # mag unused at first
    w3 = InputAdapter(["acc_x", "acc_y", "acc_z"], enc).proj.weight[:, :, 0]         # Capture-24
    assert torch.equal(w3[:3], torch.eye(3)) and torch.equal(w3[3:], torch.zeros(3, 3))


def test_preprocessor_units_and_tiling():
    from fedpeft.data.store import DataSpec
    from fedpeft.models.components import G_TO_MS2, Preprocessor
    from fedpeft.models.encoders.base import InputSpec
    harbox = DataSpec(rate_hz=50, window_s=2, channels=("acc_x", "acc_y", "acc_z"), groups=(("acc", (0, 1, 2)),),
                      units=(("acc", "m/s2"),), hop_s=1)
    pre = Preprocessor(InputSpec(rate_hz=50, n_samples=300, n_channels=3, acc_unit="g"), harbox, fit="tile")
    x = torch.randn(2, 3, 100)
    out = pre(x)
    assert out.shape == (2, 3, 300)
    assert torch.allclose(out[..., :100], x / G_TO_MS2) and torch.allclose(out[..., 100:200], out[..., :100])


def _fake_store(y_per_user: dict) -> "object":
    """A WindowStore with labels only (enough for the split functions)."""
    from fedpeft.data.store import WindowStore
    store = object.__new__(WindowStore)
    ys, slices, pos = [], {}, 0
    for u, y in y_per_user.items():
        slices[u] = (pos, pos + len(y))
        pos += len(y)
        ys.append(torch.as_tensor(y))
    store.y, store.slices, store.device = torch.cat(ys), slices, torch.device("cpu")
    return store


def test_per_class_time_split_keeps_every_activity_and_a_gap():
    y = [0] * 10 + [1] * 10 + [2] * 5            # three activities recorded one after the other
    store = _fake_store({"u": y})
    train, test = store.user_class_time_split("u", 0.8, gap=1)
    tr, te = set(train.tolist()), set(test.tolist())
    assert not tr & te
    for c, (lo, hi) in {0: (0, 10), 1: (10, 20), 2: (20, 25)}.items():
        rows = list(range(lo, hi))
        cut = int(0.8 * len(rows))
        assert set(rows[:cut]) <= tr and set(rows[cut + 1:]) <= te     # earliest part trains, latest tests
        assert rows[cut] not in tr and rows[cut] not in te              # the overlapping window is dropped


def test_random_split_sizes_and_disjointness():
    import numpy as np
    store = _fake_store({"u": [0, 1, 2, 3, 4] * 100})
    tr, va, te = store.user_random_split("u", np.random.default_rng(0), [50, 500], [0.72, 0.08, 0.20])
    n = len(tr) + len(va) + len(te)
    assert 50 <= n <= 500
    assert abs(len(tr) - 0.72 * n) <= 1 and abs(len(va) - 0.08 * n) <= 1
    assert not (set(tr.tolist()) & set(te.tolist())) and not (set(va.tolist()) & set(te.tolist()))


def test_dirichlet_partition_covers_every_window_once():
    import numpy as np
    from fedpeft.data.store import dirichlet_partition
    y = np.repeat(np.arange(6), 200)
    for alpha in (0.1, 100.0):
        parts = dirichlet_partition(y, 10, alpha, np.random.default_rng(0), min_size=5)
        allpos = np.concatenate(parts)
        assert len(parts) == 10 and min(len(p) for p in parts) >= 5
        assert np.array_equal(np.sort(allpos), np.arange(len(y)))     # a partition: no loss, no duplicates
    # small alpha -> clients dominated by few classes; large alpha -> close to the global mix
    skew = lambda parts: np.mean([np.bincount(y[p], minlength=6).max() / len(p) for p in parts])
    assert skew(dirichlet_partition(y, 10, 0.1, np.random.default_rng(1), 5)) > \
        skew(dirichlet_partition(y, 10, 100.0, np.random.default_rng(1), 5)) + 0.2


def test_wisdm_windows_stay_inside_one_activity_recording(tmp_path=None):
    import tempfile
    import numpy as np
    from fedpeft.data.wisdm import load_subject
    root = Path(tempfile.mkdtemp()) / "raw"
    for sensor in ("accel", "gyro"):
        d = root / "watch" / sensor
        d.mkdir(parents=True)
        lines = []
        t = 0
        for code, n in (("A", 450), ("M", 300), ("D", 250)):     # walking, kicking (excluded), sitting
            for i in range(n):
                lines.append(f"1600,{code},{t},{i % 7}.5,1.0,9.8;")
                t += 50_000_000                                     # 20 Hz in ns
            t += 5_000_000_000                                      # 5-s pause between activities
        (d / f"data_1600_{sensor}_watch.txt").write_text("\n".join(lines) + "\n")
    out = load_subject(root, 1600, "watch")
    # walking: 450 samples -> windows at 0, 100, 200 (3); sitting: 250 -> 1; kicking dropped
    assert out["X"].shape == (4, 6, 200)
    assert np.bincount(out["y_WISDM"]).tolist()[0] == 3 and int(out["y_WISDM"][-1]) == 3
    assert len(np.unique(out["seg"])) == 2 and np.all(np.diff(out["t"]) > 0)


def test_wisdm_mixed_rates_and_repeated_rows_are_resampled_to_20hz():
    import tempfile
    import numpy as np
    from fedpeft.data.wisdm import load_subject
    root = Path(tempfile.mkdtemp()) / "raw"
    for sensor, rate in (("accel", 50), ("gyro", 25)):           # subjects 1637-1640 log like this
        d = root / "watch" / sensor
        d.mkdir(parents=True)
        step = int(1e9 / rate)
        lines = [f"1637,A,{i * step},{np.sin(i * step / 1e9):.6f},0.0,9.8;" for i in range(int(20.0 * rate))]
        (d / f"data_1637_{sensor}_watch.txt").write_text("\n".join(lines + lines) + "\n")   # the file repeats itself
    out = load_subject(root, 1637, "watch")
    # 20 s of walking on a 20-Hz grid = 400 samples -> windows at 0, 100, 200 (no duplicates)
    assert out["X"].shape == (3, 6, 200)
    t = np.arange(200) / 20.0
    assert np.allclose(out["X"][0, 0].astype(np.float32), np.sin(t), atol=2e-2)   # interpolated, not stretched


def test_transductive_refinement_moves_prototypes_to_the_user():
    from fedpeft.training.heldout import refine_prototypes
    torch.manual_seed(0)
    protos = torch.eye(3, 4)                                   # 3 classes along the first 3 axes
    present = torch.tensor([True, True, False])               # class 2 never seen in training
    # the new user's class-0 windows are rotated towards axis 3; class 1 sits on its prototype
    user0 = torch.tensor([0.6, 0.0, 0.0, 0.8]) + 0.01 * torch.randn(40, 4)
    user1 = torch.tensor([0.0, 1.0, 0.0, 0.0]) + 0.01 * torch.randn(40, 4)
    P = refine_prototypes(torch.cat([user0, user1]), protos, present, iters=10, prior=10.0)
    assert torch.allclose(P.norm(dim=1)[:2], torch.ones(2), atol=1e-5)
    assert P[0, 3] > 0.5                                       # moved to where the user's windows are
    assert P[1, 1] > 0.99                                      # unchanged where it already fitted
    assert torch.allclose(P[2], protos[2])                     # absent class: no window assigned, unchanged


def test_frozen_batchnorm_statistics_survive_training_mode():
    from fedpeft.models.client_model import ClientModel
    model = ClientModel.__new__(ClientModel)                   # a minimal ClientModel: only an encoder
    nn.Module.__init__(model)
    model.encoder = nn.Sequential(nn.Conv1d(2, 2, 1), nn.BatchNorm1d(2))
    model.freeze_bn_stats = True
    model._norm_trainable = True                               # scale/shift trainable, statistics frozen
    model.train()
    bn = model.encoder[1]
    before = bn.running_mean.clone()
    model.encoder(torch.randn(5, 2, 8) + 3.0)
    assert not bn.training and torch.equal(bn.running_mean, before)
    model.freeze_bn_stats = False
    model.train()
    assert bn.training


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")

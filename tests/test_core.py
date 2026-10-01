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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")

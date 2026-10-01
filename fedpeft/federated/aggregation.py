"""
Server-side aggregation rules (PDF p. 8 "LoRA in federated learning" and p. 19).

Notation: client k trained factors (A_k, B_k) with rank r_k and has weight w_k
(its number of training windows n_k).  s = alpha / r is the LoRA scaling.

  FedAvg (dense tensors)   theta = sum_k w_k theta_k / sum_k w_k

  LoRA naive  (V2)         A = avg(A_k),  B = avg(B_k)
                           -> the model uses B A = avg(B) avg(A)  !=  avg(B_k A_k)
                           For mixed ranks every row is averaged over the clients
                           that trained it (HetLoRA-style zero padding).

  LoRA exact  (V3)         FedEx-LoRA:  A = avg(A_k),  B = avg(B_k),
                           residual R = s * (avg(B_k A_k) - B A) is ADDED to the frozen
                           weight, so W0 + R + s B A = W0 + s avg(B_k A_k)  exactly.

  LoRA exact + SVD (V4/5)  zero-pad to r_max, aggregate the exact product, re-factor
                           it to rank r_max with an SVD, put the leftover into W0:
                               M  = B A + sum_k w_k (B_k A_k - B0_k A0_k) / sum_k w_k
                               M ~ U S V^T,  B = U sqrt(S),  A = sqrt(S) V^T  (top r_max)
                               R  = s * (M - B A)
                           (B0_k, A0_k) are the truncated factors client k received.
                           Aggregating the CHANGE of each client means a rank-2 device
                           cannot erase directions 3..8 that it never saw; with equal
                           ranks this equals the PDF formula avg(B_k A_k).
                           Each device then receives the first r_k SVD directions
                           (the most important ones).

  Prototypes               P_c = sum_k n_kc p_kc / sum_k n_kc  (classes a client lacks are ignored)

  Class-wise classifier    row c of the shared classifier h (weight row + bias) is averaged
                           only over the clients that HAVE class c, weighted by their count
                           n_kc (as in FedRS / CReFF).  With plain FedAvg a class that only a
                           few people have (bicycling) gets its update divided by ~20 every
                           round and is never learned.
"""

from __future__ import annotations

import math

import torch


def svd_refactor(M: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Best rank-`rank` factorisation M ~ B A with balanced factors (B = U sqrt(S), A = sqrt(S) V^T).

    Directions with (numerically) zero singular value would leave both factors at
    zero, which blocks all future gradients (dL/dA needs B, dL/dB needs A).  Those
    rows of A are therefore re-initialised randomly while their B columns stay zero,
    so the product - and hence the model - is unchanged.

    For large matrices a randomised SVD (Halko et al.) finds the top directions in a
    fraction of the time.  This does NOT make the aggregation approximate: whatever
    the rank-r factors miss is kept in the residual M - B A that goes into W0.
    """
    M = M.float()
    if min(M.shape) > 4 * rank:
        U, S, V = torch.svd_lowrank(M, q=rank + 8, niter=4)
        Vh = V.T
    else:
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    r = min(rank, S.numel())
    root = S[:r].sqrt()
    B = torch.zeros(M.shape[0], rank, device=M.device)
    A = torch.zeros(rank, M.shape[1], device=M.device)
    B[:, :r] = U[:, :r] * root
    A[:r] = root[:, None] * Vh[:r]
    dead = torch.ones(rank, dtype=torch.bool, device=M.device)
    if S.numel() > 0 and S[0] > 0:
        dead[:r] = S[:r] < 1e-6 * S[0]
    if dead.any():
        fresh = torch.empty(int(dead.sum()), M.shape[1], device=M.device)
        torch.nn.init.kaiming_uniform_(fresh, a=math.sqrt(5))
        A[dead] = fresh
        B[:, dead] = 0.0
    return A, B


def pad_rows(A: torch.Tensor, r_max: int) -> torch.Tensor:
    """Zero-pad A (r x d_in) to (r_max x d_in)."""
    if A.shape[0] == r_max:
        return A
    out = A.new_zeros(r_max, A.shape[1])
    out[: A.shape[0]] = A
    return out


def pad_cols(B: torch.Tensor, r_max: int) -> torch.Tensor:
    """Zero-pad B (d_out x r) to (d_out x r_max)."""
    if B.shape[1] == r_max:
        return B
    out = B.new_zeros(B.shape[0], r_max)
    out[:, : B.shape[1]] = B
    return out


class LoRAAccumulator:
    """Streams client LoRA uploads of ONE layer and produces the aggregated factors.

    Streaming (running weighted sums) means the server never has to hold all
    client uploads at once - important for the full-model baselines on a 6 GB GPU.
    """

    def __init__(self, rule: str, A_global: torch.Tensor, B_global: torch.Tensor, scaling: float):
        self.rule = rule
        self.A_old, self.B_old = A_global, B_global          # current global factors (rank r_max)
        self.r_max = A_global.shape[0]
        self.s = scaling
        self.W = 0.0
        self.sum_A = torch.zeros_like(A_global)
        self.sum_B = torch.zeros_like(B_global)
        self.row_w = torch.zeros(self.r_max, device=A_global.device)   # per-rank-row weight (hetero naive)
        self.sum_BA = torch.zeros(B_global.shape[0], A_global.shape[1], device=A_global.device)
        self.sum_delta = torch.zeros_like(self.sum_BA)

    def add(self, A_k: torch.Tensor, B_k: torch.Tensor, w: float) -> None:
        r = A_k.shape[0]
        self.W += w
        self.sum_A += w * pad_rows(A_k, self.r_max)
        self.sum_B += w * pad_cols(B_k, self.r_max)
        self.row_w[:r] += w
        BA_k = B_k @ A_k
        self.sum_BA += w * BA_k
        # Change relative to what this client received (first r rows of the global factors).
        self.sum_delta += w * (BA_k - self.B_old[:, :r] @ self.A_old[:r])

    def result(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
        """Return (A, B, residual for W0 or None, relative aggregation error of the naive rule)."""
        mean_BA = self.sum_BA / self.W
        # Naive per-row average (rows nobody trained keep their old value).
        rw = self.row_w.clamp(min=1e-12)
        A_naive = torch.where(self.row_w[:, None] > 0, self.sum_A / rw[:, None], self.A_old)
        B_naive = torch.where(self.row_w[None, :] > 0, self.sum_B / rw[None, :], self.B_old)

        # Diagnostic: how wrong would the naive product avg(B) avg(A) be compared with the
        # exact aggregate, relative to the size of this round's update?  The exact target
        # is B_old A_old + average client change (= avg(B_k A_k) when all clients have the
        # same rank and start, i.e. V2/V3).
        exact_target = self.B_old @ self.A_old + self.sum_delta / self.W
        update = (exact_target - self.B_old @ self.A_old).norm()
        naive_err = float((B_naive @ A_naive - exact_target).norm() / update.clamp(min=1e-12))

        if self.rule == "lora_naive":
            return A_naive, B_naive, None, naive_err
        if self.rule == "lora_exact":
            A, B = self.sum_A / self.W, self.sum_B / self.W
            residual = self.s * (mean_BA - B @ A)
            return A, B, residual, naive_err
        if self.rule == "lora_exact_svd":
            M = self.B_old @ self.A_old + self.sum_delta / self.W
            A, B = svd_refactor(M, self.r_max)
            residual = self.s * (M - B @ A)
            return A, B, residual, naive_err
        raise ValueError(f"unknown LoRA aggregation '{self.rule}'")


class DenseAccumulator:
    """Weighted FedAvg of ordinary tensors (heads, full-model weights, BN statistics)."""

    def __init__(self, reference: dict[str, torch.Tensor]):
        self.sums = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in reference.items()
                     if v.dtype.is_floating_point}
        self.latest: dict[str, torch.Tensor] = {}
        self.W = 0.0

    def add(self, tensors: dict[str, torch.Tensor], w: float) -> None:
        self.W += w
        for k, v in tensors.items():
            if k in self.sums:
                self.sums[k] += w * v.float()
            else:                      # integer buffers (e.g. BatchNorm step counters): keep the latest
                self.latest[k] = v.clone()

    def result(self) -> dict[str, torch.Tensor]:
        out = {k: s / self.W for k, s in self.sums.items()}
        out.update(self.latest)
        return out


class ClassRowAccumulator:
    """Class-wise weighted averaging of the classifier rows (see module docstring)."""

    def __init__(self, reference: dict[str, torch.Tensor]):
        self.ref = reference                         # {key: (C, ...) tensor}, rows = classes
        self.sums = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in reference.items()}
        self.n = None

    def add(self, tensors: dict[str, torch.Tensor], class_counts: torch.Tensor) -> None:
        counts = class_counts.float()
        self.n = counts.clone() if self.n is None else self.n + counts
        for k, v in tensors.items():
            shape = (-1,) + (1,) * (v.dim() - 1)
            self.sums[k] += counts.view(shape) * v.float()

    def result(self) -> dict[str, torch.Tensor]:
        """Rows of classes nobody had this round keep their previous value."""
        out = {}
        for k, ref in self.ref.items():
            shape = (-1,) + (1,) * (ref.dim() - 1)
            has = (self.n > 0).view(shape)
            out[k] = torch.where(has, self.sums[k] / self.n.clamp(min=1).view(shape), ref.float())
        return out


class PrototypeAccumulator:
    """P_c = sum_k n_kc p_kc / sum_k n_kc."""

    def __init__(self, n_classes: int, dim: int, device: torch.device):
        self.sum_p = torch.zeros(n_classes, dim, device=device)
        self.sum_n = torch.zeros(n_classes, device=device)

    def add(self, protos: torch.Tensor, counts: torch.Tensor) -> None:
        self.sum_p += counts[:, None] * protos
        self.sum_n += counts

    def result(self, old: torch.Tensor | None, old_present: torch.Tensor | None):
        present = self.sum_n > 0
        new = self.sum_p / self.sum_n.clamp(min=1)[:, None]
        if old is not None:   # a class nobody had this round keeps its previous prototype
            new = torch.where(present[:, None], new, old)
            present = present | old_present
        return new, present

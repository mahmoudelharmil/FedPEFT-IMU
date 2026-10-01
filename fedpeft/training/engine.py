"""
The training / inference engine shared by centralized training, FL clients and
few-shot adaptation.

    train_local(...)   optimise a model on a set of window indices
                       loss = CE(h(g(E(x))), y) + lambda * L_proto + mu/2 * ||w - w_global||^2
                       and measure time + peak memory while doing it
    predict(...)       batched inference -> logits and embeddings
    embeddings_prototypes(...)   local class prototypes after training (component 8)

Data never goes through a DataLoader: the whole WindowStore is already on the GPU,
so a "batch" is just an index tensor.
"""

from __future__ import annotations

import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.components import class_prototypes, prototype_loss


def autocast(device: torch.device, enabled: bool):
    """Mixed precision context (fp16 on CUDA). A no-op on CPU."""
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def make_scaler(device: torch.device, amp: bool) -> torch.amp.GradScaler:
    """fp16 loss scaler (disabled on CPU / without AMP). Starts at 2^12 to skip fewer steps."""
    return torch.amp.GradScaler("cuda", init_scale=2.0 ** 12, enabled=amp and device.type == "cuda")


def model_state_bytes(model: nn.Module) -> int:
    """Bytes of all parameters and buffers (what the device must hold to run the model)."""
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


def train_local(model: nn.Module, X: torch.Tensor, y: torch.Tensor, idx: torch.Tensor,
                params: list[nn.Parameter], *, lr: float, weight_decay: float, epochs: int,
                batch_size: int, max_steps: int | None = None, amp: bool = True,
                protos: torch.Tensor | None = None, proto_present: torch.Tensor | None = None,
                proto_weight: float = 0.0, prox_mu: float = 0.0,
                prox_ref: dict[str, torch.Tensor] | None = None,
                generator: torch.Generator | None = None,
                optimizer: torch.optim.Optimizer | None = None,
                scaler: torch.amp.GradScaler | None = None,
                logit_adj: torch.Tensor | None = None) -> dict:
    """Train `params` of `model` on windows X[idx], y[idx].

    A fresh AdamW optimiser is created unless one is passed in (the centralized
    runner keeps one optimiser across epochs; FL clients start fresh every round).
    The fp16 loss scaler should be passed in and re-used: a fresh scaler wastes the
    first few steps finding its scale, which matters when a round has ~20 steps.

    `logit_adj` (a vector tau * log(prior), see imbalance.py) is added to the logits
    inside the loss only - the "balanced softmax" that stops rare or locally absent
    classes from being pushed down.

    Returns a dict with the mean loss, number of steps / samples, wall-clock time
    and the peak GPU memory used by training (activations + gradients + optimiser).
    """
    device = X.device
    if optimizer is None:
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if scaler is None:
        scaler = make_scaler(device, amp)
    named = dict(model.named_parameters()) if prox_mu > 0 else {}

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
    t0 = time.perf_counter()

    model.train()
    steps, seen, loss_sum = 0, 0, 0.0
    done = False
    for _ in range(epochs):
        order = idx[torch.randperm(len(idx), generator=generator, device="cpu").to(device)]
        for start in range(0, len(order), batch_size):
            b = order[start:start + batch_size]
            if len(b) < 2:          # BatchNorm needs at least 2 samples
                continue
            xb, yb = X[b], y[b]
            with autocast(device, amp):
                logits, emb = model(xb, return_embedding=True)
                logits = logits.float()
                if logit_adj is not None:
                    logits = logits + logit_adj
                loss = F.cross_entropy(logits, yb)
                if proto_weight > 0 and protos is not None:
                    loss = loss + proto_weight * prototype_loss(emb, yb, protos, proto_present)
            if prox_mu > 0 and prox_ref:   # FedProx-style proximal term (optional ablation)
                prox = sum(((named[k] - v) ** 2).sum() for k, v in prox_ref.items() if k in named)
                loss = loss + 0.5 * prox_mu * prox
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            seen += len(b)
            loss_sum += float(loss.detach()) * len(b)
            if max_steps and steps >= max_steps:
                done = True
                break
        if done:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_extra = torch.cuda.max_memory_allocated() - mem_before
    else:
        peak_extra = 0
    elapsed = time.perf_counter() - t0
    flat = [p for g in params for p in g["params"]] if params and isinstance(params[0], dict) else params
    trainable = sum(p.numel() for p in flat)
    return {
        "loss": loss_sum / max(1, seen),
        "steps": steps,
        "samples": seen,
        "time_s": elapsed,
        # Device memory = model tensors + what training added on top (activations, grads, Adam state).
        "train_mem_mb": (model_state_bytes(model) + max(0, peak_extra)) / 1e6,
        "trainable_params": trainable,
        # Analytical optimiser footprint: weights' grads + Adam m and v (fp32).
        "optimizer_state_mb": trainable * 4 * 3 / 1e6,
    }


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, idx: torch.Tensor, batch_size: int = 512,
            amp: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, embeddings) for X[idx] in eval mode.

    A model may carry its own `eval_batch_size` (set from cfg.runtime.eval_batch_size):
    heavy encoders such as UniMTS need small inference batches on a 6 GB GPU.
    """
    batch_size = getattr(model, "eval_batch_size", None) or batch_size
    model.eval()
    logits, embs = [], []
    for start in range(0, len(idx), batch_size):
        b = idx[start:start + batch_size]
        with autocast(X.device, amp):
            lg, e = model(X[b], return_embedding=True)
        logits.append(lg.float())
        embs.append(e.float())
    if not logits:
        return torch.zeros(0, device=X.device), torch.zeros(0, device=X.device)
    return torch.cat(logits), torch.cat(embs)


@torch.no_grad()
def embeddings_prototypes(model: nn.Module, X: torch.Tensor, y: torch.Tensor, idx: torch.Tensor,
                          n_classes: int, max_windows: int = 512, amp: bool = True,
                          generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Local class prototypes p_{k,c} and counts n_{k,c} of one client (component 8).

    Uses at most `max_windows` of the client's training windows (random subset) to
    keep the extra forward pass cheap; the counts reported to the server are the
    client's TRUE class counts, so the server-side weighting stays correct.
    """
    sub = idx
    if len(idx) > max_windows:
        sub = idx[torch.randperm(len(idx), generator=generator, device="cpu")[:max_windows].to(idx.device)]
    _, emb = predict(model, X, sub, amp=amp)
    protos, _ = class_prototypes(emb, y[sub], n_classes)
    present = torch.bincount(y[sub], minlength=n_classes) > 0
    counts = torch.bincount(y[idx], minlength=n_classes).float() * present.float()
    return protos, counts

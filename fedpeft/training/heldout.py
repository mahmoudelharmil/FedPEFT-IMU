"""
Held-out users: zero-shot and few-shot evaluation (PDF p. 10 RQ3, p. 17 Figure 6, appendix).

Held-out users never took part in training.  For each of them:

    adaptation part  = first 50 % of their recording (few-shot labels are drawn from here)
    test part        = last 50 % of their recording (every protocol is scored on this)

Protocols
    zero-shot   0 minutes of labels
                * global-classifier models: the trained model as-is
                * prototype models (V5):   nearest global prototype (Fig. 6 B).  For
                  BatchNorm encoders the new user's normalisation statistics are first
                  estimated from their own UNLABELLED windows (AdaBN) - V5 trains with
                  per-device norms, so the "default" pretrained statistics do not match.
    few-shot    1 and 5 minutes of labels per activity
                * global-classifier models: re-fit the classifier h on the frozen
                  embedding (cheap: embeddings are computed once)
                * prototype models (V5):   train a NEW personal head h (initialised from
                  the global prototypes) + adapt the local norm and input adapter (Fig. 6 C)
                In both cases the few-shot sample is roughly class-balanced, unlike a real
                day, so the fit uses the balanced-softmax loss with the SAMPLE's prior and
                the prediction is calibrated with the global prior (see imbalance.py).

Every protocol is scored per window and after HMM smoothing (same Decoder as the
rest of the project).  Reported: mean per-user macro-F1 (+ pooled macro-F1 / accuracy).
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.store import ExperimentData, sample_fewshot
from ..models.components import nearest_prototype_logits
from .decoding import Decoder
from .engine import predict, train_local
from .imbalance import class_prior, logit_adjustment
from .metrics import evaluate_predictions

PROTO_TEMPERATURE = 0.1   # cosine similarity / T  ->  logit-like scores for the decoder


def adapt_bn_statistics(model: nn.Module, X: torch.Tensor, idx: torch.Tensor, batch_size: int = 64) -> bool:
    """AdaBN: re-estimate every BatchNorm's running mean/variance on UNLABELLED windows.

    Uses no labels, so it is still a zero-shot protocol. Returns False when the
    model has no BatchNorm (LayerNorm encoders need nothing).
    """
    bns = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    if not bns or len(idx) < 2:
        return False
    model.eval()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                  # cumulative average over all batches
        m.train()
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            b = idx[start:start + batch_size]
            if len(b) >= 2:
                model(X[b])
    for m in bns:
        m.momentum = 0.1
        m.eval()
    return True


def _fit_linear_head(head: nn.Module, emb: torch.Tensor, y: torch.Tensor, steps: int,
                     logit_adj: torch.Tensor | None, lr: float = 5e-3) -> None:
    """Few-shot re-fitting of a linear head on cached embeddings (full batch, balanced softmax)."""
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    head.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = head(emb)
        if logit_adj is not None:
            logits = logits + logit_adj
        F.cross_entropy(logits, y).backward()
        opt.step()
    head.eval()


def _summarise(y_true: dict, y_pred: dict, n_classes: int) -> dict:
    m = evaluate_predictions(y_true, y_pred, n_classes)
    return {
        "mean_user_f1": m["client_f1_mean"],
        "pooled_macro_f1": m["macro_f1"],
        "accuracy": m["accuracy"],
        "worst10_f1": m["worst10_f1"],
        "user_f1": m["client_f1"],
    }


def evaluate_heldout(model, data: ExperimentData, cfg, decoder: Decoder, *, mode: str = "global",
                     protos: torch.Tensor | None = None, proto_present: torch.Tensor | None = None,
                     fewshot_trainable_roles: set[str] | None = None, amp: bool = True,
                     seed: int = 0) -> dict:
    """Zero-shot + few-shot results on all held-out users.

    mode = "global"     the model's own classifier is used (centralized, FedAvg, V0-V4)
    mode = "prototype"  zero-shot by nearest global prototype, few-shot adapts norm +
                        adapter + a new personal head (V5)
    The passed `model` is never modified (a copy is adapted per user).
    """
    X, y = data.store.X, data.store.y
    n_classes = data.n_classes
    device = X.device
    rng = np.random.default_rng(seed)
    minutes_list = list(cfg.split.fewshot_minutes)
    steps = cfg.split.fewshot_steps
    tau_train = float(cfg.imbalance.tau_train)
    use_adabn = mode == "prototype" and cfg.split.get("adabn_new_users", True)
    # Prototype scores are class-balanced by construction (like tau_train = 1). Their cosine
    # values span a narrow range, so any prior term swamps them: on the validation users
    # plain nearest-prototype (tau = 0) is best, which is also the PDF's zero-shot rule.
    proto_decoder = dataclasses.replace(decoder, tau_train=1.0,
                                        tau_inf=float(cfg.imbalance.get("tau_inf_prototype", 0.0)))

    y_true, logits_by = {}, {0: {}}
    logits_by.update({m: {} for m in minutes_list})
    test_by_user = {}

    cap = cfg.split.get("max_heldout_test_windows")   # optional, for slow encoders
    cap_gen = torch.Generator().manual_seed(seed + 4000)
    adabn_used = False
    for user in data.heldout:
        adapt_idx, test_idx = data.heldout_adapt[user], data.heldout_test[user]
        if len(test_idx) == 0:
            continue
        if cap and len(test_idx) > cap:   # fixed random subset, same for every method of this encoder
            test_idx = test_idx[torch.randperm(len(test_idx), generator=cap_gen)[:cap].sort().values.to(test_idx.device)]
        test_by_user[user] = test_idx
        y_true[user] = y[test_idx].cpu().numpy()

        # ---------------- zero-shot ----------------
        user_base = model
        if use_adabn:
            user_base = copy.deepcopy(model)
            adabn_used = adapt_bn_statistics(user_base, X, adapt_idx, batch_size=min(64, model.eval_batch_size)) or adabn_used
        test_logits, test_emb = predict(user_base, X, test_idx, amp=amp)
        if mode == "prototype":
            logits_by[0][user] = nearest_prototype_logits(test_emb, protos, proto_present) / PROTO_TEMPERATURE
        else:
            logits_by[0][user] = test_logits

        # ---------------- few-shot -----------------
        adapt_y_np = y[adapt_idx].cpu().numpy()
        for minutes in minutes_list:
            pos = sample_fewshot(adapt_y_np, minutes, data.store.window_s, rng)
            shot_idx = adapt_idx[torch.as_tensor(pos, device=adapt_idx.device)]
            shot_adj = logit_adjustment(class_prior(np.bincount(y[shot_idx].cpu().numpy(), minlength=n_classes)),
                                        tau_train, device)
            if mode == "global":
                # Only the classifier h is adapted, on frozen embeddings (computed once).
                _, shot_emb = predict(model, X, shot_idx, amp=amp)
                head = copy.deepcopy(model.head).float()
                _fit_linear_head(head, shot_emb, y[shot_idx], steps, shot_adj)
                with torch.no_grad():
                    logits_by[minutes][user] = head(test_emb)
            else:
                # V5: new personal head from the global prototypes + local norm/adapter adaptation.
                user_model = copy.deepcopy(user_base)
                user_model.head.init_from_prototypes(protos, proto_present)
                params = user_model.set_trainable(fewshot_trainable_roles or {"head", "norm", "adapter"})
                train_local(user_model, X, y, shot_idx, params, lr=cfg.fl.lr_peft, weight_decay=0.0,
                            epochs=10_000, max_steps=steps,
                            batch_size=min(cfg.split.get("fewshot_batch", 64), len(shot_idx)), amp=amp,
                            logit_adj=shot_adj)
                logits_by[minutes][user], _ = predict(user_model, X, test_idx, amp=amp)
                del user_model
        if user_base is not model:
            del user_base

    def decode(dec: Decoder, logits: dict, smooth: bool) -> dict:
        if smooth:
            pred = {u: dec.smoothed(lg, data.store.sequence_breaks(test_by_user[u])) for u, lg in logits.items()}
        else:
            pred = {u: dec.window(lg) for u, lg in logits.items()}
        return _summarise(y_true, pred, n_classes)

    zero_dec = proto_decoder if mode == "prototype" else decoder
    out = {"protocol": {"adapt_frac": cfg.split.heldout_adapt_frac, "fewshot_steps": steps, "mode": mode,
                        "max_test_windows_per_user": cap, "adabn": adabn_used},
           "zero_shot": decode(zero_dec, logits_by[0], False)}
    for minutes in minutes_list:
        out[f"few_shot_{minutes}min"] = decode(decoder, logits_by[minutes], False)
    if decoder.hmm is not None:
        out["hmm"] = {"zero_shot": decode(zero_dec, logits_by[0], True)}
        for minutes in minutes_list:
            out["hmm"][f"few_shot_{minutes}min"] = decode(decoder, logits_by[minutes], True)
    return out

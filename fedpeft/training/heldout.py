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
    few-shot    a few labels per activity: 1 and 5 minutes (Capture-24) or 1, 5 and 10
                labelled 2-s windows (HARBox), see store.fewshot_budgets
                * global-classifier models: re-fit the classifier h on the frozen
                  embedding (cheap: embeddings are computed once)
                * prototype models (V5):   train a NEW personal head h (initialised from
                  the global prototypes) + adapt the local norm and input adapter (Fig. 6 C)
                In both cases the few-shot sample is roughly class-balanced, unlike a real
                day, so the fit uses the balanced-softmax loss with the SAMPLE's prior and
                the prediction is calibrated with the global prior (see imbalance.py).

Optional transductive zero-shot (split.zero_shot_refine_iters > 0, T3A-style): the class
prototypes - V5's global prototypes, or the normalised classifier rows of a shared head - are
refined by a few soft k-means steps on the new user's UNLABELLED adaptation windows (the same
windows AdaBN uses), then the user is served by the nearest refined prototype.  Still no labels.

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

from ..data.store import ExperimentData, fewshot_budgets, sample_fewshot
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


@torch.no_grad()
def refine_prototypes(emb: torch.Tensor, protos: torch.Tensor, present: torch.Tensor, iters: int,
                      prior: float, temperature: float = PROTO_TEMPERATURE) -> torch.Tensor:
    """Transductive prototype refinement on a new user's UNLABELLED windows (soft k-means).

        w_ic = softmax_c( cos(z_i, P_c) / T )                     (soft assignment)
        P_c <- normalise( sum_i w_ic z_i / |z_i|  +  prior * P0_c )  (P0 = the starting prototypes)

    `prior` is the weight of the starting prototype, in windows: a class with few windows
    stays close to it, a class the user performs often moves to where the user's data lie.
    Returns unit-norm prototypes (absent classes stay masked by `present`).
    """
    P0 = F.normalize(protos.float(), dim=1)
    Z = F.normalize(emb.float(), dim=1)
    P = P0
    for _ in range(iters):
        s = Z @ P.T / temperature
        s[:, ~present] = float("-inf")
        w = torch.softmax(s, dim=1)                    # (N, C)
        P = F.normalize(w.T @ Z + prior * P0, dim=1)
    return P


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
        "mean_user_acc": m["client_acc_mean"],
        "pooled_macro_f1": m["macro_f1"],
        "accuracy": m["accuracy"],
        "worst10_f1": m["worst10_f1"],
        "user_f1": m["client_f1"],
    }


def evaluate_heldout(model, data: ExperimentData, cfg, decoder: Decoder, *, mode: str = "global",
                     protos: torch.Tensor | None = None, proto_present: torch.Tensor | None = None,
                     fewshot_trainable_roles: set[str] | None = None, amp: bool = True,
                     seed: int = 0, zero_shot_rule: str = "default") -> dict:
    """Zero-shot + few-shot results on all held-out users.

    mode = "global"     the model's own classifier is used (centralized, FedAvg, V0-V4)
    mode = "prototype"  zero-shot by nearest global prototype, few-shot adapts norm +
                        adapter + a new personal head (V5)
    The passed `model` is never modified (a copy is adapted per user).
    """
    if not data.heldout:          # protocols without new users (e.g. the random HARBox protocol)
        return None
    X, y = data.store.X, data.store.y
    n_classes = data.n_classes
    device = X.device
    rng = np.random.default_rng(seed)
    budgets = fewshot_budgets(cfg, data.store.window_s)        # [(result key, windows per class, label)]
    budget_keys = [key for key, _, _ in budgets]
    steps = cfg.split.fewshot_steps
    tau_train = float(cfg.imbalance.tau_train)
    use_adabn = mode == "prototype" and cfg.split.get("adabn_new_users", True)
    # Transductive zero-shot (prototype rule of V5, or the shared classifier of the other methods).
    refine_iters = int(cfg.split.get("zero_shot_refine_iters", 0) or 0)
    refine_prior = float(cfg.split.get("zero_shot_refine_prior", 10.0))
    # Shared classifiers only with split.zero_shot_refine_shared (it lowered their validation scores).
    refine = (refine_iters > 0 and zero_shot_rule == "default"
              and (mode == "prototype" or bool(cfg.split.get("zero_shot_refine_shared", False))))
    # Prototype scores are class-balanced by construction (like tau_train = 1). Their cosine
    # values span a narrow range, so any prior term swamps them: on the validation users
    # plain nearest-prototype (tau = 0) is best, which is also the PDF's zero-shot rule.
    proto_decoder = dataclasses.replace(decoder, tau_train=1.0,
                                        tau_inf=float(cfg.imbalance.get("tau_inf_prototype", 0.0)))

    y_true, logits_by = {}, {0: {}}
    logits_by.update({key: {} for key in budget_keys})
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
        user_protos = protos
        if refine and len(adapt_idx) > 1:
            _, adapt_emb = predict(user_base, X, adapt_idx, amp=amp)      # unlabelled windows
            if mode == "prototype":
                start, start_present = protos, proto_present
            else:                    # a shared classifier: its (normalised) class rows are the prototypes
                start = user_base.head.fc.weight.detach()
                start_present = torch.ones(n_classes, dtype=torch.bool, device=start.device)
            user_protos = refine_prototypes(adapt_emb, start, start_present, refine_iters, refine_prior)
            logits_by[0][user] = nearest_prototype_logits(test_emb, user_protos, start_present) / PROTO_TEMPERATURE
        elif mode == "prototype" and zero_shot_rule == "global_head":   # dual-head V5: the federated global head
            with torch.no_grad():
                logits_by[0][user] = user_base.global_logits(test_emb).float()
        elif mode == "prototype" and zero_shot_rule != "head":
            logits_by[0][user] = nearest_prototype_logits(test_emb, protos, proto_present) / PROTO_TEMPERATURE
        elif mode == "prototype":       # the clients' average personal classifier (fl.new_user_head = average)
            logits_by[0][user] = test_logits
        else:
            logits_by[0][user] = test_logits

        # ---------------- few-shot -----------------
        adapt_y_np = y[adapt_idx].cpu().numpy()
        for key, per_class, _ in (budgets if len(adapt_idx) else []):
            pos = sample_fewshot(adapt_y_np, per_class, rng)
            shot_idx = adapt_idx[torch.as_tensor(pos, device=adapt_idx.device)]
            shot_adj = logit_adjustment(class_prior(np.bincount(y[shot_idx].cpu().numpy(), minlength=n_classes)),
                                        tau_train, device)
            if mode == "global":
                # Only the classifier h is adapted, on frozen embeddings (computed once).
                _, shot_emb = predict(model, X, shot_idx, amp=amp)
                head = copy.deepcopy(model.head).float()
                _fit_linear_head(head, shot_emb, y[shot_idx], steps, shot_adj)
                with torch.no_grad():
                    logits_by[key][user] = head(test_emb)
            else:
                # V5: new personal head from the global prototypes + local norm/adapter adaptation.
                user_model = copy.deepcopy(user_base)
                if zero_shot_rule == "global_head":     # start from the federated global classifier
                    user_model.head.load_state_dict(user_model.ghead.state_dict())
                else:
                    user_model.head.init_from_prototypes(user_protos, proto_present)
                params = user_model.set_trainable(fewshot_trainable_roles or {"head", "norm", "adapter"})
                user_model.freeze_bn_stats = True     # keep the AdaBN statistics of ALL the user's windows
                train_local(user_model, X, y, shot_idx, params, lr=cfg.fl.lr_peft, weight_decay=0.0,
                            epochs=10_000, max_steps=steps,
                            batch_size=min(cfg.split.get("fewshot_batch", 64), len(shot_idx)), amp=amp,
                            logit_adj=shot_adj)
                logits_by[key][user], _ = predict(user_model, X, test_idx, amp=amp)
                del user_model
        if user_base is not model:
            del user_base

    def decode(dec: Decoder, logits: dict, smooth: bool) -> dict:
        if smooth:
            pred = {u: dec.smoothed(lg, data.store.sequence_breaks(test_by_user[u])) for u, lg in logits.items()}
        else:
            pred = {u: dec.window(lg) for u, lg in logits.items()}
        return _summarise(y_true, pred, n_classes)

    # The global head of dual-head V5 is a shared, balanced-softmax classifier: decoded like one.
    zero_dec = proto_decoder if ((mode == "prototype" and zero_shot_rule != "global_head") or refine) else decoder
    out = {"protocol": {"adapt_frac": cfg.split.heldout_adapt_frac, "fewshot_steps": steps, "mode": mode,
                        "zero_shot_rule": zero_shot_rule,
                        "zero_shot_refine": {"iters": refine_iters, "prior": refine_prior} if refine else None,
                        "max_test_windows_per_user": cap, "adabn": adabn_used,
                        "fewshot": [{"key": k, "windows_per_class": n, "label": lab} for k, n, lab in budgets]},
           "zero_shot": decode(zero_dec, logits_by[0], False)}
    for key in budget_keys:
        out[key] = decode(decoder, logits_by[key], False)
    if decoder.hmm is not None:
        out["hmm"] = {"zero_shot": decode(zero_dec, logits_by[0], True)}
        for key in budget_keys:
            out["hmm"][key] = decode(decoder, logits_by[key], True)
    return out

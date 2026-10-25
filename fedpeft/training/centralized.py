"""
Phase 1 - centralized training (the proof of concept for the four encoders).

All FL clients' TRAINING parts (first 80 % of each recording) are pooled on one
machine and a single model is trained, exactly as if privacy did not matter.
This gives the reference numbers the federated methods are compared with.

Modes
    scratch        random initialisation, every weight trained  ("learning from scratch")
    pretrained_ft  official pretrained weights, every weight fine-tuned (upper reference)

Class imbalance: the loss uses logit adjustment with the global class prior and
predictions are prior-calibrated (see imbalance.py); results are reported per
window and after HMM temporal smoothing (hmm.py).

Model selection: early stopping on the macro-F1 of the validation (pool) users.
Test: each client's own last-20 % (per-client F1, fairness) + held-out users (zero/few-shot).
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..config import encoder_config
from ..data.store import ExperimentData, load_data_spec
from ..models.client_model import ClientModel
from ..models.encoders import build_encoder, encoder_display_name
from ..utils import get_logger, save_arrays, save_json, seed_everything
from .decoding import Decoder, make_decoder
from .augment import make_augmenter
from .engine import make_scaler, predict, train_local
from .heldout import evaluate_heldout
from .imbalance import logit_adjustment
from .metrics import macro_f1

TAU_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)   # prior strengths checked on the validation users

ALL_ROLES = {"encoder", "norm", "adapter", "shared_head", "head"}


def build_model(cfg, encoder_name: str, pretrained: bool, n_classes: int, seed: int,
                device: torch.device) -> ClientModel:
    """Encoder (+ weights) wrapped in the full 8-component client model."""
    data = load_data_spec(cfg.paths.processed_dir)
    encoder = build_encoder(encoder_name, cfg, pretrained=pretrained, seed=seed, data=data)
    fit = cfg.encoders.get(encoder_name, {}).get("fit", "crop")
    model = ClientModel(encoder, n_classes, cfg.model.embed_dim, data, fit).to(device)
    model.eval_batch_size = cfg.runtime.get("eval_batch_size", 512)   # inference batch (memory-bound encoders use less)
    return model


def evaluate_clients(model, data: ExperimentData, decoder: Decoder, amp: bool = True) -> dict:
    """Per-client test (last 20 % of each client) with ONE shared model.

    Returns {"window": (metrics, confusion), "hmm": (metrics, confusion)}.
    """
    X = data.store.X
    idx = data.cat(data.client_test)
    logits, _ = predict(model, X, idx, amp=amp)
    by_user, pos = {}, 0
    for user in sorted(data.client_test):
        n = len(data.client_test[user])
        by_user[user] = logits[pos:pos + n]
        pos += n
    return decoder.evaluate(data.store, by_user, data.client_test, data.n_classes)


def run_centralized(cfg, data: ExperimentData, encoder_name: str, mode: str, seed: int,
                    device: torch.device) -> dict:
    """Train + evaluate one (encoder, mode) combination and save its JSON result."""
    log = get_logger(f"central.{encoder_name}.{mode}")
    cfg = encoder_config(cfg, encoder_name)       # per-encoder compute budget (e.g. UniMTS)
    seed_everything(seed)
    c = cfg.centralized
    amp = cfg.runtime.amp
    pretrained = mode == "pretrained_ft"
    model = build_model(cfg, encoder_name, pretrained, data.n_classes, seed, device)
    params = model.set_trainable(ALL_ROLES)
    lr = c.lr_finetune if pretrained else c.lr_scratch
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=c.weight_decay)
    scaler = make_scaler(device, amp)

    X, y = data.store.X, data.store.y
    train_idx = data.cat(data.client_train)
    gen = torch.Generator().manual_seed(seed)
    # Class imbalance: balanced-softmax loss with the global prior + calibrated decisions.
    decoder, prior, _ = make_decoder(cfg, data)
    logit_adj = logit_adjustment(prior, cfg.imbalance.tau_train, device)
    augment = make_augmenter(cfg, data.store.spec)
    # Validation = pool users (a fixed random subset keeps validation fast).
    val_idx = data.pool_index
    max_val = c.get("max_val_windows") or 6000
    if len(val_idx) > max_val:
        keep = torch.randperm(len(val_idx), generator=gen)[:max_val]
        val_idx = val_idx[keep.to(val_idx.device)]
    log.info(f"{encoder_name}/{mode}: {len(train_idx):,} train windows, {len(val_idx):,} validation windows, "
             f"{sum(p.numel() for p in params):,} trainable parameters")

    history, best_f1, best_state, bad_epochs = [], -1.0, None, 0
    t_start = time.perf_counter()
    peak_mem = 0.0
    for epoch in range(1, c.epochs + 1):
        # Cosine learning-rate schedule over the planned epochs.
        for group in optimizer.param_groups:
            group["lr"] = lr * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / c.epochs))
        # Optional cap on windows per epoch (a fresh random subset each epoch) for slow encoders.
        epoch_idx = train_idx
        if c.get("max_train_windows") and len(train_idx) > c.max_train_windows:
            pick = torch.randperm(len(train_idx), generator=gen)[: c.max_train_windows]
            epoch_idx = train_idx[pick.to(train_idx.device)]
        stats = train_local(model, X, y, epoch_idx, params, lr=lr, weight_decay=c.weight_decay, epochs=1,
                            batch_size=c.batch_size, amp=amp, generator=gen, optimizer=optimizer,
                            scaler=scaler, logit_adj=logit_adj, augment=augment)
        peak_mem = max(peak_mem, stats["train_mem_mb"])
        val_logits, _ = predict(model, X, val_idx, amp=amp)
        val_f1 = macro_f1(y[val_idx].cpu().numpy(), decoder.window(val_logits), labels=np.arange(data.n_classes))
        history.append({"epoch": epoch, "train_loss": stats["loss"], "val_macro_f1": val_f1,
                        "epoch_time_s": stats["time_s"], "samples_per_s": stats["samples"] / stats["time_s"]})
        log.info(f"epoch {epoch:2d}  loss {stats['loss']:.4f}  val-F1 {val_f1:.4f}  "
                 f"({stats['time_s']:.0f}s, {stats['samples'] / stats['time_s']:.0f} win/s)")
        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad_epochs += 1
            if bad_epochs >= c.patience:
                log.info(f"early stop at epoch {epoch}")
                break
    train_time = time.perf_counter() - t_start
    model.load_state_dict(best_state)

    # Sensitivity of the validation macro-F1 to the prediction-time prior strength
    # (validation users only - used to choose cfg.imbalance.tau_inf, never the test set).
    val_logits, _ = predict(model, X, val_idx, amp=amp)
    tau_sweep = {str(t): macro_f1(y[val_idx].cpu().numpy(), decoder.window(val_logits, t),
                                  labels=np.arange(data.n_classes)) for t in TAU_GRID}

    # ---------------- test ----------------
    test = evaluate_clients(model, data, decoder, amp)
    # Raw network outputs, so decoding rules can be studied later without retraining
    # (choose them on the pool/validation logits, never on the test logits).
    test_idx_all = data.cat(data.client_test)
    save_arrays(Path(cfg.paths.results_dir) / cfg.profile / "logits" / f"central__{encoder_name}__{mode}__seed{seed}.npz",
                test_idx=test_idx_all, test_logits=predict(model, X, test_idx_all, amp=amp)[0],
                pool_idx=data.pool_index, pool_logits=predict(model, X, data.pool_index, amp=amp)[0],
                log_prior=decoder.log_prior,
                hmm_trans_counts=decoder.hmm.trans_counts if decoder.hmm else np.zeros(0),
                hmm_init_counts=decoder.hmm.init_counts if decoder.hmm else np.zeros(0))
    client_metrics, conf = test["window"]
    heldout = evaluate_heldout(model, data, cfg, decoder, mode="global", amp=amp, seed=seed)
    hmm_txt = f"  | +HMM F1 {test['hmm'][0]['macro_f1']:.4f}" if "hmm" in test else ""
    zs = f"  | held-out zero-shot {heldout['zero_shot']['mean_user_f1']:.4f}" if heldout else ""
    log.info(f"TEST macro-F1 {client_metrics['macro_f1']:.4f}  acc {client_metrics['accuracy']:.4f}  "
             f"client-acc {client_metrics['client_acc_mean']:.4f}  worst10 {client_metrics['worst10_f1']:.4f}"
             f"{hmm_txt}{zs}")

    result = {
        "kind": "centralized",
        "encoder": encoder_name,
        "encoder_display": encoder_display_name(encoder_name),
        "mode": mode,
        "seed": seed,
        "profile": cfg.profile,
        "n_classes": data.n_classes,
        "class_names": data.store.class_names,
        "param_counts": model.count_by_role(),
        "trainable_params": sum(p.numel() for p in params),
        "history": history,
        "best_val_macro_f1": best_f1,
        "train_time_s": train_time,
        "train_mem_mb": peak_mem,
        "n_train_windows": int(len(train_idx)),
        "test": client_metrics,
        "confusion": conf,
        "test_hmm": test["hmm"][0] if "hmm" in test else None,
        "confusion_hmm": test["hmm"][1] if "hmm" in test else None,
        "val_tau_sweep": tau_sweep,
        "heldout": heldout,
        "config": {"epochs": c.epochs, "lr": lr, "batch_size": c.batch_size,
                   "tau_train": cfg.imbalance.tau_train, "tau_inf": cfg.imbalance.tau_inf,
                   "hmm": bool(cfg.decoding.hmm),
                   "subsample_every": cfg.data.subsample_every,
                   "augment": cfg.get("augment", {}).get("enabled", False),
                   "max_train_windows": c.get("max_train_windows"),
                   "budget_applied": cfg.get("budget_applied")},
    }
    out = Path(cfg.paths.results_dir) / cfg.profile / "centralized" / f"{encoder_name}__{mode}__seed{seed}.json"
    save_json(result, out)
    log.info(f"saved {out}")
    return result

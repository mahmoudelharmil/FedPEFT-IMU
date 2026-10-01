"""
Step 0 (optional but recommended) - benchmark the four encoders on this machine.

    python scripts/00_benchmark_encoders.py

For every encoder it measures, on random windows:
    * parameter counts (total, LoRA r=8 on the PDF's target layers)
    * training throughput + peak memory when training EVERYTHING (FedAvg / V0),
      at the batch size the encoder uses in FL (cfg.fl.batch_size incl. its budget)
    * training throughput + peak memory with a frozen encoder + LoRA (V2-V5)
    * inference throughput
The numbers are used to size the experiment profiles and appear in the report.
Results: results/<profile>/benchmark/encoders.json
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from _common import setup

from fedpeft.config import encoder_config
from fedpeft.models.client_model import ClientModel
from fedpeft.models.encoders import ENCODERS, build_encoder
from fedpeft.training.engine import autocast, make_scaler
from fedpeft.utils import save_json


def measure(model, params, device, batch, steps, amp, train=True):
    """Windows per second and peak MB for `steps` iterations."""
    x = torch.randn(batch, 3, 300, device=device).half()
    y = torch.randint(0, 6, (batch,), device=device)
    opt = torch.optim.AdamW(params, lr=1e-4) if train else None
    scaler = make_scaler(device, amp)
    torch.cuda.synchronize()
    for i in range(steps + 3):
        if i == 3:                      # 3 warm-up iterations (cuDNN autotuning) are not measured
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
        if train:
            model.train()
            with autocast(device, amp):
                loss = F.cross_entropy(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            model.eval()
            with torch.no_grad(), autocast(device, amp):
                model(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return batch * steps / dt, torch.cuda.max_memory_allocated() / 1e6


if __name__ == "__main__":
    cfg, device, log, _ = setup("00_benchmark_encoders", __doc__)
    out = {}
    for name in ENCODERS:
        ecfg = encoder_config(cfg, name)
        batch = ecfg.fl.batch_size          # the batch size this encoder really trains with
        enc = build_encoder(name, cfg, pretrained=False)
        model = ClientModel(enc, 6, cfg.model.embed_dim).to(device)
        total = sum(p.numel() for p in enc.parameters())
        full_params = model.set_trainable({"encoder", "norm", "adapter", "shared_head", "head"})
        full_tp, full_mem = measure(model, full_params, device, batch, 10, cfg.runtime.amp)
        inf_tp, _ = measure(model, full_params, device, 4 * batch, 10, cfg.runtime.amp, train=False)
        model.add_lora(8, 2.0, 0.0)
        lora_params = model.set_trainable({"lora", "shared_head", "head"})
        lora_tp, lora_mem = measure(model, lora_params, device, batch, 10, cfg.runtime.amp)
        counts = model.count_by_role()
        out[name] = {
            "display_name": enc.display_name, "architecture": enc.architecture,
            "pretrained_source": enc.pretrained_source,
            "input": {"rate_hz": enc.spec.rate_hz, "n_samples": enc.spec.n_samples,
                      "n_channels": enc.spec.n_channels, "seconds": enc.spec.seconds, "unit": enc.spec.acc_unit},
            "feat_dim": enc.feat_dim, "encoder_params": total, "lora_params_r8": counts["lora"],
            "lora_layers": len(model.lora_layers), "param_counts_r8": counts,
            "train_full_win_per_s": full_tp, "train_full_peak_mb": full_mem,
            "train_lora_win_per_s": lora_tp, "train_lora_peak_mb": lora_mem,
            "inference_win_per_s": inf_tp, "batch_size": batch,
        }
        log.info(f"{name:14s} params {total/1e6:6.2f}M  LoRA(r=8) {counts['lora']:>8,}  "
                 f"train-full {full_tp:7.0f} win/s ({full_mem:5.0f} MB)  train-LoRA {lora_tp:7.0f} win/s "
                 f"({lora_mem:5.0f} MB)  infer {inf_tp:7.0f} win/s")
        del model, enc
        torch.cuda.empty_cache()
    save_json(out, Path(cfg.paths.results_dir) / cfg.profile / "benchmark" / "encoders.json")

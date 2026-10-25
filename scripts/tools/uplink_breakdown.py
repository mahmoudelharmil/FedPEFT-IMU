"""
Where the uplink saving of FedPEFT-IMU (V5) over full fine-tuning (V0) comes from.

    python scripts/tools/uplink_breakdown.py [--profile harbox]

For every encoder with a V0 and a V5 run, the upload of one client update is rebuilt from
the parameter counts stored in the result files and compared with the bytes measured
during training (fedpeft/federated/comm.py counts every tensor that leaves the device):

    V0 (full fine-tuning) sends every trainable value:
        encoder Omega + norm (incl. BatchNorm statistics) + input adapter + shared head g
        + classifier h  (+ a few bytes of class counts and the sample count n_k)
    V5 sends only:
        LoRA factors of rank r + shared head g + class prototypes (C x 64) + class counts
    every value is a 32-bit float = 4 bytes.

LoRA on a layer with input size d_in and output size d_out (a Conv kernel counts as
d_in = C_in x kernel size) has r x (d_in + d_out) values instead of d_in x d_out, so the
LoRA size grows linearly with the rank: |LoRA_r| = (r / 8) |LoRA_8|.  V5 clients have rank
2, 4 or 8 (device tiers), so its average upload per client update depends on how often each
tier took part.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="harbox")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    base = ROOT / "results" / args.profile / "federated"
    for v5_path in sorted(base.glob(f"*__v5_proposed__seed{args.seed}.json")):
        enc = v5_path.name.split("__")[0]
        v0_path = base / f"{enc}__v0_full_ft__seed{args.seed}.json"
        if not v0_path.exists():
            continue
        v5, v0 = (json.loads(p.read_text(encoding="utf-8")) for p in (v5_path, v0_path))
        pc = v5["param_counts"]
        n_classes = len(v5["final"]["per_class_f1"])
        embed = 64
        # ---- V0: every trainable value ----
        v0_values = pc["encoder"] + pc["norm"] + pc["norm_stats"] + pc["adapter"] + pc["shared_head"] + pc["head"]
        v0_formula = 4 * v0_values
        v0_measured = v0["comm"]["uplink_mb_per_client_round"] * 1e6
        # ---- V5: LoRA at rank r + g + prototypes (+ class counts) ----
        protos = n_classes * embed + n_classes
        per_rank = {}
        for key, rec in v5["comm"]["per_rank"].items():
            r = int(key[1:])
            values = round(pc["lora"] * r / 8) + pc["shared_head"] + protos
            per_rank[r] = (values, 4 * values, rec["uplink_mb_per_update"] * 1e6)
        # tier mix: how often each rank actually uploaded during training
        tiers = {int(k[1:]): v["n_clients"] for k, v in (v5.get("tier_f1") or {}).items()}
        v5_measured = v5["comm"]["uplink_mb_per_client_round"] * 1e6
        print(f"\n=== {enc} ({args.profile}) ===")
        print(f"V0 values: encoder {pc['encoder']:,} + norm {pc['norm']:,} + norm stats {pc['norm_stats']:,} + adapter "
              f"{pc['adapter']:,} + g {pc['shared_head']:,} + h {pc['head']:,} = {v0_values:,}")
        print(f"   x 4 bytes = {v0_formula / 1e6:.3f} MB   (measured {v0_measured / 1e6:.3f} MB)")
        for r, (values, formula, measured) in sorted(per_rank.items()):
            print(f"V5 rank {r}: LoRA {round(pc['lora'] * r / 8):,} + g {pc['shared_head']:,} + prototypes {protos} = "
                  f"{values:,} values x 4 = {formula / 1e6:.4f} MB (measured {measured / 1e6:.4f} MB)  "
                  f"-> {v0_formula / formula:.1f}x less than V0")
        if tiers:
            n = sum(tiers.values())
            expected = sum(tiers[r] * per_rank[r][1] for r in per_rank) / n
            print(f"tier mix {tiers} -> expected V5 upload {expected / 1e6:.4f} MB; measured average over all "
                  f"uploads {v5_measured / 1e6:.4f} MB")
        print(f"RATIO (measured): {v0_measured:,.0f} / {v5_measured:,.0f} bytes = {v0_measured / v5_measured:.1f}x;  "
              f"share of the encoder that LoRA r=8 adds: {pc['lora'] / pc['encoder'] * 100:.1f} %")


if __name__ == "__main__":
    main()

"""
The federated methods compared in this project (PDF p. 4, "Variants compared").

Each method is described ONLY by data: which roles of the client model are
trained, which are sent to the server (federated), and which stay on the device
(local).  Roles are defined in fedpeft/models/client_model.py.

    method            init        trained                       sent to server         kept on device
    ----------------  ----------  ----------------------------  ---------------------  -----------------
    FedAvg (scratch)  random      everything                    everything             -
    V0 full FT        pretrained  everything                    everything             -
    V1 head only      pretrained  g + h                         g + h                  -
    V2 LoRA naive     pretrained  LoRA + g + h                  LoRA + g + h           -
    V3 LoRA exact     pretrained  LoRA + g + h                  LoRA + g + h           -
    V4 hetero ranks   pretrained  LoRA(r=2/4/8) + g + h         LoRA + g + h           -
    V5 proposed       pretrained  LoRA(r=2/4/8) + g + h +       LoRA + g + prototypes  norm, adapter, h
                                  norm + adapter
"""

from __future__ import annotations

from dataclasses import dataclass, field

EVERYTHING = frozenset({"encoder", "norm", "adapter", "shared_head", "head"})


@dataclass(frozen=True)
class MethodSpec:
    name: str                          # identifier used in file names
    short: str                         # "V0", "V1", ... for tables
    label: str                         # human-readable name for the report
    description: str
    pretrained: bool                   # start from the pretrained encoder?
    trainable: frozenset               # roles optimised on the client
    federated: frozenset               # roles uploaded and aggregated by the server
    local: frozenset = frozenset()     # roles kept per device between rounds
    use_lora: bool = False
    hetero_rank: bool = False          # device tiers with LoRA rank 2 / 4 / 8
    aggregation: str = "fedavg"        # fedavg | lora_naive | lora_exact | lora_exact_svd
    use_prototypes: bool = False       # compute/send prototypes and add the prototype loss
    lr_key: str = "lr_peft"            # which learning rate of cfg.fl to use
    heldout_mode: str = "global"       # global classifier or nearest-prototype for new users
    slot: int = 1                      # categorical colour slot (1-8) used by every chart of the report
    extra: dict = field(default_factory=dict)


METHODS: dict[str, MethodSpec] = {
    "fedavg_scratch": MethodSpec(
        name="fedavg_scratch", short="FedAvg", label="FedAvg from scratch",
        description="Random initialisation; the whole model is trained and uploaded every round.",
        pretrained=False, trainable=EVERYTHING, federated=EVERYTHING | {"norm_stats"},
        lr_key="lr_scratch", slot=7),
    "v0_full_ft": MethodSpec(
        name="v0_full_ft", short="V0", label="V0 · Full fine-tuning",
        description="Pretrained encoder; the whole model is fine-tuned and uploaded (upper-cost baseline).",
        pretrained=True, trainable=EVERYTHING, federated=EVERYTHING | {"norm_stats"},
        lr_key="lr_full", slot=1),
    "v1_head_only": MethodSpec(
        name="v1_head_only", short="V1", label="V1 · Head only",
        description="Frozen pretrained encoder; only the heads g and h are trained and uploaded (lower-cost baseline).",
        pretrained=True, trainable=frozenset({"shared_head", "head"}),
        federated=frozenset({"shared_head", "head"}), slot=4),
    "v2_lora_naive": MethodSpec(
        name="v2_lora_naive", short="V2", label="V2 · LoRA, naive avg",
        description="LoRA (r=8) + heads; the server averages A and B separately: avg(B)·avg(A).",
        pretrained=True, trainable=frozenset({"lora", "shared_head", "head"}),
        federated=frozenset({"lora", "shared_head", "head"}), use_lora=True,
        aggregation="lora_naive", slot=5),
    "v3_lora_exact": MethodSpec(
        name="v3_lora_exact", short="V3", label="V3 · LoRA, exact avg",
        description="LoRA (r=8) + heads; exact FedEx-LoRA aggregation avg(B·A), residual folded into W0.",
        pretrained=True, trainable=frozenset({"lora", "shared_head", "head"}),
        federated=frozenset({"lora", "shared_head", "head"}), use_lora=True,
        aggregation="lora_exact", slot=3),
    "v4_hetero_rank": MethodSpec(
        name="v4_hetero_rank", short="V4", label="V4 · Heterogeneous ranks",
        description="Device tiers train LoRA ranks 2/4/8; zero-pad, exact aggregation, SVD re-factorisation.",
        pretrained=True, trainable=frozenset({"lora", "shared_head", "head"}),
        federated=frozenset({"lora", "shared_head", "head"}), use_lora=True, hetero_rank=True,
        aggregation="lora_exact_svd", slot=6),
    "v5_proposed": MethodSpec(
        name="v5_proposed", short="V5", label="V5 · FedPEFT-IMU (proposed)",
        description="V4 + local norm + local input adapter + personal head + class-prototype alignment.",
        pretrained=True, trainable=frozenset({"lora", "shared_head", "head", "norm", "adapter"}),
        federated=frozenset({"lora", "shared_head"}),
        local=frozenset({"norm", "norm_stats", "adapter", "head"}), use_lora=True, hetero_rank=True,
        aggregation="lora_exact_svd", use_prototypes=True, heldout_mode="prototype", slot=2),
}


def get_method(name: str) -> MethodSpec:
    if name not in METHODS:
        raise KeyError(f"Unknown FL method '{name}'. Choose from {list(METHODS)}")
    return METHODS[name]

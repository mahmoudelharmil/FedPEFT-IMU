"""
The FL server: holds the global state and aggregates client uploads.

Global state
    dense[k]        every federated non-LoRA tensor (heads; the whole model for FedAvg / V0)
    lora_A/B[name]  global LoRA factors of every adapted layer, at rank r_max
    base[name]      the frozen weight W0 of every adapted layer PLUS the accumulated
                    exact-aggregation residual (FedEx-LoRA / SVD).  Unchanged for V2.
    protos          global class prototypes (V5)

A round is:   begin_round() -> add_upload(...) for every selected client -> end_round()
"""

from __future__ import annotations

import torch

from ..models.client_model import ClientModel
from .aggregation import ClassRowAccumulator, DenseAccumulator, LoRAAccumulator, PrototypeAccumulator
from .methods import MethodSpec


class Server:
    def __init__(self, model: ClientModel, method: MethodSpec, roles: dict[str, str], scaling: float,
                 classwise_head: bool = True):
        self.method = method
        self.scaling = scaling
        state = model.state_dict()
        # Dense federated tensors (LoRA factors are handled separately).
        self.dense_keys = [k for k, r in roles.items() if r in method.federated and r != "lora"]
        # The shared classifier's class rows are averaged class-wise (see aggregation.py).
        self.class_row_keys = ([k for k in self.dense_keys
                                if (k.startswith("head.fc.") and "head" in method.federated)
                                or (k.startswith("ghead.fc.") and "global_head" in method.federated)]
                               if classwise_head else [])
        self.dense = {k: state[k].detach().clone() for k in self.dense_keys}
        # LoRA: global factors and the (residual-updated) frozen base weights.
        self.lora_names = list(model.lora_layers)
        self.lora_A = {n: model.lora_layers[n].lora_A.detach().clone() for n in self.lora_names}
        self.lora_B = {n: model.lora_layers[n].lora_B.detach().clone() for n in self.lora_names}
        self.base = {n: state[model.lora_key(n)].detach().clone() for n in self.lora_names}
        self.has_residual = False           # becomes True after the first exact aggregation
        self.protos: torch.Tensor | None = None
        self.proto_present: torch.Tensor | None = None
        self.last_agg_error: float | None = None

    # ------------------------------------------------------------ download
    def lora_for_rank(self, rank: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """The first `rank` rows of A and columns of B of every layer (what a tier-r device gets)."""
        return {n: (self.lora_A[n][:rank], self.lora_B[n][:, :rank]) for n in self.lora_names}

    # ---------------------------------------------------------------- round
    def begin_round(self) -> None:
        self._dense_acc = DenseAccumulator(self.dense) if self.dense_keys else None
        self._row_acc = (ClassRowAccumulator({k: self.dense[k] for k in self.class_row_keys})
                         if self.class_row_keys else None)
        self._lora_acc = {n: LoRAAccumulator(self.method.aggregation, self.lora_A[n], self.lora_B[n], self.scaling)
                          for n in self.lora_names}
        self._proto_acc = None

    def add_upload(self, upload: dict, weight: float) -> None:
        """Stream one client's upload into the running sums."""
        if self._dense_acc is not None:
            self._dense_acc.add(upload["dense"], weight)
        if self._row_acc is not None:
            self._row_acc.add({k: upload["dense"][k] for k in self.class_row_keys}, upload["class_counts"])
        for n, (A_k, B_k) in upload.get("lora", {}).items():
            self._lora_acc[n].add(A_k, B_k, weight)
        if "protos" in upload:
            protos, counts = upload["protos"]
            if self._proto_acc is None:
                self._proto_acc = PrototypeAccumulator(protos.shape[0], protos.shape[1], protos.device)
            self._proto_acc.add(protos, counts)

    def end_round(self) -> None:
        """Apply the aggregation rule of the method and update the global state."""
        if self._dense_acc is not None:
            new = self._dense_acc.result()
            if self._row_acc is not None and self._row_acc.n is not None:
                new.update(self._row_acc.result())      # class-wise rows replace the plain average
            for k in self.dense_keys:
                if k in new:
                    self.dense[k] = new[k].to(self.dense[k].dtype)
        errors = []
        for n, acc in self._lora_acc.items():
            if acc.W == 0:
                continue
            A, B, residual, err = acc.result()
            self.lora_A[n], self.lora_B[n] = A, B
            errors.append(err)
            if residual is not None:   # exact rules: fold the leftover into the frozen weight
                self.base[n] = self.base[n] + residual.view_as(self.base[n]).to(self.base[n].dtype)
                self.has_residual = True
        self.last_agg_error = float(sum(errors) / len(errors)) if errors else None
        if self._proto_acc is not None:
            self.protos, self.proto_present = self._proto_acc.result(self.protos, self.proto_present)

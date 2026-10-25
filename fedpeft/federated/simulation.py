"""
The federated round loop (PDF p. 19 "One training round" and p. 20 algorithm).

    for round t = 1..T:
        S <- random 10-20 % of the clients
        for every client k in S:
            1 download   global state (first r_k LoRA rows, heads, prototypes) + own local parts
            2-3 train    E local epochs (capped at max_local_steps): CE + lambda * L_proto
            4 prototypes local class means of the embedding (V5)
            5 upload     only the federated roles of the method  (bytes are measured)
        6 server aggregation (FedAvg / naive / exact / exact-SVD, prototype averaging)
        every `eval_every` rounds: evaluate ALL clients on their own test part
    methods with personal parts (V5): one final personalisation update per client
    new users: zero-/few-shot evaluation on the held-out users

Class imbalance: every client trains with a balanced-softmax loss using ITS OWN
label prior, so classes it has never seen are not pushed down (the reason rare
classes vanished in plain FedAvg); predictions are calibrated with the global
prior and additionally reported after HMM smoothing (training/imbalance.py, hmm.py).

KPIs recorded (PDF "METRICS"):
    macro-F1, accuracy, worst-10 % client F1, client-F1 std           (per evaluation)
    uplink MB per client / per round, cumulative MB, downlink MB     (per round)
    rounds to target F1, MB to target F1                             (from the curve)
    client training memory (peak), client training time               (per client update)
    zero-/few-shot F1 on held-out users                               (end of training)
"""

from __future__ import annotations

import dataclasses
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import encoder_config
from ..data.store import ExperimentData
from ..models.encoders import encoder_display_name
from ..training.centralized import build_model
from ..training.decoding import make_decoder
from ..training.augment import make_augmenter
from ..training.engine import embeddings_prototypes, make_scaler, predict, train_local
from ..training.heldout import evaluate_heldout
from ..training.imbalance import class_prior, logit_adjustment
from ..utils import get_logger, human_mb, save_arrays, save_json, seed_everything
from .comm import payload_bytes, tensor_bytes
from .methods import get_method
from .server import Server


@dataclass
class FLClient:
    """One device = one participant (Capture-24 participant / HARBox user)."""
    user: str
    train_idx: torch.Tensor          # first 80 % of the recording
    test_idx: torch.Tensor           # last 20 % of the recording
    rank: int                        # LoRA rank of its device tier
    local_state: dict | None = None  # tensors of the method's local roles (V5); None = never trained
    personal_state: dict | None = None   # Ditto: the client's personal copy of the whole model (on the CPU)
    personal_lora: dict | None = None    # V5 + personal LoRA: the client's private copy of the adapters
    rounds_participated: int = 0
    logit_adj: torch.Tensor | None = None   # tau * log(own label prior) for the balanced-softmax loss
    global_adj: torch.Tensor | None = None  # the same for the global head of dual-head V5

    @property
    def n_train(self) -> int:
        return len(self.train_idx)


def first_reach(evals: list[dict], target: float | None, key: str = "macro_f1") -> dict | None:
    """First evaluation point whose `key` reaches `target` (None if never reached)."""
    if target is None:
        return None
    for point in evals:
        if point[key] >= target:
            return point
    return None


class FederatedRun:
    """One (encoder, method, seed) federated experiment."""

    def __init__(self, cfg, data: ExperimentData, encoder_name: str, method_name: str, seed: int,
                 device: torch.device, rank_override: int | None = None, target_f1: float | None = None,
                 tag: str | None = None):
        cfg = encoder_config(cfg, encoder_name)            # per-encoder compute budget (e.g. UniMTS)
        self.cfg, self.data, self.device, self.seed = cfg, data, device, seed
        self.encoder_name = encoder_name
        self.method = get_method(method_name)
        self.target_f1 = target_f1
        self.amp = cfg.runtime.amp
        self.fl = cfg.fl
        # Tagged runs are ablations: "__r16" (LoRA rank sweep) or e.g. "__noproto" (config changed by
        # command-line overrides); the tag keeps their result file next to the main run.
        self.tag = f"__{tag}" if tag else (f"__r{rank_override}" if rank_override else "")
        self.log = get_logger(f"fl.{encoder_name}.{method_name}{self.tag}")
        seed_everything(seed)

        # ---- model: encoder (+ LoRA) + heads ----
        self.model = build_model(cfg, encoder_name, self.method.pretrained, data.n_classes, seed, device)
        lcfg = cfg.model.lora
        tiers = list(lcfg.tiers)
        self.r_max = max(tiers) if self.method.hetero_rank else (rank_override or lcfg.rank)
        self.scaling = float(lcfg.alpha_over_r)                  # alpha = 2r  ->  alpha / r = 2
        if self.method.use_lora:
            self.model.add_lora(self.r_max, self.scaling, lcfg.dropout)
        # Dual-head V5 (model.global_head): a federated global classifier next to the personal one.
        self.global_head = bool(self.method.extra.get("global_head") and cfg.model.get("global_head", False))
        if self.global_head:
            self.model.add_global_head()
        self.global_head_weight = float(cfg.model.get("global_head_weight", 1.0)) if self.global_head else 0.0
        self.roles = self.model.roles()
        self.initial_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.local_keys = [k for k, r in self.roles.items() if r in self.method.local]
        self.default_local = {k: self.initial_state[k] for k in self.local_keys}
        # What a client without local parts of its own (a brand-new user) starts from: the
        # initial (pretrained) values, or - fl.new_user_local_init = average - the clients'
        # average, set once at the end of training (see average_local_parts).
        self.new_user_local = self.default_local
        self.new_user_bytes = 0

        # ---- clients (device tiers assigned once, reproducibly) ----
        rng = np.random.default_rng(seed + 1000)
        self.clients = []
        for user in data.clients:
            rank = int(rng.choice(tiers)) if self.method.hetero_rank else self.r_max
            self.clients.append(FLClient(user, data.client_train[user], data.client_test[user], rank))
        self.rng = np.random.default_rng(seed + 2000)
        self.gen = torch.Generator().manual_seed(seed)
        # Intermediate evaluations may use a fixed random subset of every client's test
        # windows (slow encoders); the FINAL evaluation always uses all of them.
        cap = self.fl.get("max_eval_windows_per_client")
        eval_rng = torch.Generator().manual_seed(seed + 3000)
        self.eval_subset = {}
        for c in self.clients:
            idx = c.test_idx
            if cap and len(idx) > cap:
                idx = idx[torch.randperm(len(idx), generator=eval_rng)[:cap].sort().values.to(idx.device)]
            self.eval_subset[c.user] = idx
        # Class-imbalance handling (priors are statistics each client computes locally;
        # the 6 class counts + 6x6 HMM transition counts are uploaded ONCE).
        self.decoder, self.global_prior, self.stats_bytes = make_decoder(cfg, data)
        tau_train = float(cfg.imbalance.tau_train)
        # A PERSONAL classifier (V5) should model its own client's label distribution: a
        # non-cyclist's head is right to rarely predict cycling.  The prior-aware loss is a
        # fix for SHARED classifiers trained on skewed clients, so V5 trains with plain
        # cross-entropy, decides by arg-max and smooths with its client's own prior.
        self.ditto = bool(self.method.extra.get("ditto"))
        self.personal_head = "head" in self.method.local
        self.client_log_prior = {}
        for c in self.clients:
            prior = class_prior(data.store.class_counts(c.train_idx))
            # For the HMM of personal heads: the client's prior shrunk halfway towards the
            # global prior, so a class the client never had (prior ~1e-4) is not boosted by
            # dividing by a near-zero number.
            self.client_log_prior[c.user] = torch.log(0.5 * prior + 0.5 * self.global_prior)
            c.logit_adj = (logit_adjustment(prior, tau_train, device)
                           if tau_train > 0 and not self.personal_head else None)
            # the global head is SHARED, so it gets the usual balanced-softmax adjustment
            c.global_adj = logit_adjustment(prior, tau_train, device) if (self.global_head and tau_train > 0) else None
        if self.personal_head or self.ditto:      # personal classifiers decide by arg-max of plain CE
            self.client_decoder = dataclasses.replace(self.decoder, tau_train=0.0, tau_inf=0.0)
        else:
            self.client_decoder = self.decoder
        self.scaler = make_scaler(device, self.amp)
        self.augment = make_augmenter(cfg, data.store.spec)          # training-time augmentation (or None)
        self.server = Server(self.model, self.method, self.roles, self.scaling,
                             classwise_head=bool(self.fl.get("classwise_head_aggregation", True)))
        self.residual_bytes = sum(tensor_bytes(w) for w in self.server.base.values())

    # ------------------------------------------------------------------ state
    def load_client(self, client: FLClient | None, rank: int | None = None) -> None:
        """Put the model into the state client `client` has after the download.

        client=None means "a brand-new user": global state + default local parts.
        """
        updates = dict(self.server.dense)
        if self.method.local:
            local = client.local_state if (client is not None and client.local_state is not None) else self.new_user_local
            updates.update(local)
        if updates:
            self.model.load_state_dict(updates, strict=False)
        if self.method.use_lora:
            r = rank or (client.rank if client is not None else self.r_max)
            own = client.personal_lora if client is not None else None
            for name, layer in self.model.lora_layers.items():
                layer.base.weight.data.copy_(self.server.base[name])
                if own is not None:
                    layer.set_factors(*own[name])
                else:
                    A, B = self.server.lora_A[name][:r], self.server.lora_B[name][:, :r]
                    layer.set_factors(A, B)

    def download_bytes(self, rank: int) -> int:
        """Bytes the server sends to a device of LoRA rank `rank` this round."""
        n = payload_bytes(self.server.dense)
        if self.method.use_lora:
            n += payload_bytes(self.server.lora_for_rank(rank))
            if self.server.has_residual:
                n += self.residual_bytes      # exact aggregation must ship the dense residual too
        if self.server.protos is not None:
            n += tensor_bytes(self.server.protos)
        return n

    def make_upload(self, client: FLClient) -> dict:
        """Only the federated roles (+ prototypes) leave the device."""
        state = self.model.state_dict()
        upload = {"dense": {k: state[k] for k in self.server.dense_keys}, "n_k": client.n_train}
        if self.server.class_row_keys:          # 6 class counts for the class-wise classifier average
            upload["class_counts"] = torch.as_tensor(self.data.store.class_counts(client.train_idx),
                                                     dtype=torch.float32, device=self.device)
        if self.method.use_lora:
            upload["lora"] = {n: (l.lora_A.detach(), l.lora_B.detach()) for n, l in self.model.lora_layers.items()}
        if self.method.use_prototypes:
            upload["protos"] = embeddings_prototypes(self.model, self.data.store.X, self.data.store.y,
                                                     client.train_idx, self.data.n_classes,
                                                     max_windows=self.cfg.model.get("proto_max_windows", 512), amp=self.amp,
                                                     generator=self.gen)
        return upload

    # ------------------------------------------------------------- evaluate
    @torch.no_grad()
    def evaluate(self, full: bool = True) -> dict:
        """Every client, on its own test part, with the model it would actually run.

        full=False uses the (optional) fixed test subsets for cheap intermediate evaluations.
        Returns {"window": (metrics, confusion), "hmm": (metrics, confusion)}.
        """
        groups: OrderedDict[str, list[FLClient]] = OrderedDict()
        for c in self.clients:
            key = (c.user if (self.method.local or self.ditto)
                   else (f"r{c.rank}" if self.method.hetero_rank else "global"))
            groups.setdefault(key, []).append(c)
        X = self.data.store.X
        test_of = (lambda c: c.test_idx) if full else (lambda c: self.eval_subset[c.user])
        logits_by, idx_by = {}, {}
        for members in groups.values():
            self.load_client(members[0])
            if self.ditto and members[0].personal_state is not None:     # Ditto: the personal model
                self.model.load_state_dict(members[0].personal_state, strict=False)
            idx = torch.cat([test_of(m) for m in members])
            logits, _ = predict(self.model, X, idx, amp=self.amp)
            pos = 0
            for m in members:
                n = len(test_of(m))
                logits_by[m.user] = logits[pos:pos + n]
                idx_by[m.user] = test_of(m)
                pos += n
        self.last_logits = (logits_by, idx_by)          # kept for save_logits()
        own = self.client_log_prior if (self.personal_head or self.ditto) else None
        return self.client_decoder.evaluate(self.data.store, logits_by, idx_by, self.data.n_classes, own)

    def save_logits(self) -> None:
        """Store the final test logits of every client and the new-user model's logits on
        the pool (validation) users, for decoding studies without retraining."""
        logits_by, idx_by = self.last_logits
        users = sorted(logits_by)
        arrays = {"test_idx": torch.cat([idx_by[u] for u in users]),
                  "test_logits": torch.cat([logits_by[u] for u in users])}
        self.load_client(None, rank=self.r_max)
        pool_logits, pool_emb = predict(self.model, self.data.store.X, self.data.pool_index, amp=self.amp)
        arrays.update(pool_idx=self.data.pool_index, pool_logits=pool_logits, log_prior=self.decoder.log_prior)
        if self.server.protos is not None:
            from ..models.components import nearest_prototype_logits
            arrays["pool_proto_scores"] = nearest_prototype_logits(pool_emb, self.server.protos, self.server.proto_present)
        if self.decoder.hmm is not None:
            arrays.update(hmm_trans_counts=self.decoder.hmm.trans_counts, hmm_init_counts=self.decoder.hmm.init_counts)
        save_arrays(Path(self.cfg.paths.results_dir) / self.cfg.profile / "logits" /
                    f"fl__{self.encoder_name}__{self.method.name}{self.tag}__seed{self.seed}.npz", **arrays)

    def ditto_personal_step(self, client: FLClient, received: dict, lr: float) -> None:
        """Ditto (Li et al., ICML 2021): the client also trains its PERSONAL model v_k on

            min_v  L_k(v) + lambda/2 * || v - w ||^2       (w = the global model it received)

        with plain cross-entropy (a personal model may follow its own label distribution).
        The personal model never leaves the device.  It starts as a copy of w."""
        self.model.load_state_dict(client.personal_state if client.personal_state is not None else received)
        params = self.model.set_trainable(set(self.method.trainable), self.roles)
        names = {id(p): n for n, p in self.model.named_parameters()}
        prox_ref = {names[id(p)]: received[names[id(p)]] for p in params}
        train_local(self.model, self.data.store.X, self.data.store.y, client.train_idx, params, lr=lr,
                    weight_decay=self.fl.weight_decay, epochs=self.fl.local_epochs, batch_size=self.fl.batch_size,
                    max_steps=self.fl.max_local_steps, amp=self.amp, generator=self.gen, scaler=self.scaler,
                    prox_mu=float(self.fl.get("ditto_lambda", 1.0)), prox_ref=prox_ref, augment=self.augment)
        client.personal_state = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}

    def model_path(self) -> Path:
        return (Path(self.cfg.paths.results_dir) / self.cfg.profile / "models" /
                f"fl__{self.encoder_name}__{self.method.name}{self.tag}__seed{self.seed}.pt")

    def save_new_user_model(self) -> None:
        """Store the model a NEW user receives (global weights + default local parts) and the
        global prototypes, so the held-out protocols can be re-run without re-training
        (scripts/tools/reeval_heldout.py)."""
        path = self.model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                    "protos": None if self.server.protos is None else self.server.protos.detach().cpu(),
                    "proto_present": None if self.server.proto_present is None else self.server.proto_present.cpu()},
                   path)

    def load_new_user_model(self) -> None:
        saved = torch.load(self.model_path(), map_location=self.device)
        self.model.load_state_dict(saved["state"])
        if saved["protos"] is not None:
            self.server.protos = saved["protos"].to(self.device)
            self.server.proto_present = saved["proto_present"].to(self.device)

    def evaluate_new_users(self) -> dict | None:
        """Zero- and few-shot results on the held-out users, with the model currently loaded."""
        method = self.method
        if method.heldout_mode == "none":
            return None
        rule = ("global_head" if self.global_head else
                "head" if (method.local and self.fl.get("new_user_head", "prototype") == "average") else "default")
        return evaluate_heldout(self.model, self.data, self.cfg, self.decoder, mode=method.heldout_mode,
                                protos=self.server.protos, proto_present=self.server.proto_present,
                                fewshot_trainable_roles={"head", "norm", "adapter"}, amp=self.amp, seed=self.seed,
                                zero_shot_rule=rule)

    def average_local_parts(self) -> None:
        """Initialisation of NEW users for methods with local parts (V5).

        The local normalisation and input adapter of every client live in the space that the
        federated LoRA and head were trained with, while the initial (pretrained) values do not.
        Once, at the end of training, every client uploads these few parameters (not its
        personal classifier) and the server keeps their n_k-weighted average; a new user starts
        from it instead of the pretrained values.  The one-time upload is counted in bytes."""
        roles = {"norm", "norm_stats", "adapter"}
        if self.fl.get("new_user_head", "prototype") == "average":
            roles.add("head")          # also the clients' average classifier, used as the zero-shot rule
        keys = [k for k in self.local_keys if self.roles[k] in roles]
        owners = [c for c in self.clients if c.local_state is not None]
        if not keys or not owners:
            return
        total = float(sum(c.n_train for c in owners))
        avg = {}
        for k in keys:
            ref = owners[0].local_state[k]
            if ref.dtype.is_floating_point:
                avg[k] = sum(c.local_state[k].float() * (c.n_train / total) for c in owners).to(ref.dtype)
            else:                         # e.g. BatchNorm step counters
                avg[k] = ref.clone()
        self.new_user_local = dict(self.default_local, **avg)
        self.new_user_bytes = len(owners) * payload_bytes({k: owners[0].local_state[k] for k in keys})

    def param_groups(self, params: list, lr: float) -> list:
        """Full-model methods (FedAvg, V0) train the heads with `fl.lr_head` and the body with
        their own (smaller) rate - the usual discriminative learning rates for fine-tuning.
        With one small rate for everything, the rows of rare classes barely move in 40 rounds."""
        lr_head = self.fl.get("lr_head")
        if not lr_head or lr_head == lr or "encoder" not in self.method.trainable:
            return params
        head_ids = {id(p) for n, p in self.model.named_parameters()
                    if self.roles.get(n) in ("head", "shared_head")}
        head = [p for p in params if id(p) in head_ids]
        body = [p for p in params if id(p) not in head_ids]
        return [{"params": body, "lr": lr}, {"params": head, "lr": lr_head}]

    def local_finetune_eval(self) -> dict | None:
        """"+FT" evaluation for methods with a SHARED classifier (FedAvg, V0-V4): every client
        fine-tunes only the classifier h on its own training windows and is scored with that
        personal classifier.  This is the standard "FedAvg + local fine-tuning" protocol of
        personalised FL; the global-model scores are reported unchanged next to it.

        The fine-tuning uses the SAME balanced-softmax loss the client used in federated
        training, so the classifier keeps producing class-balanced scores and is decoded
        exactly like the shared model (mixing a plain-CE fine-tune into a balanced head and
        then dividing by the prior again double-counts the prior).  Nothing is uploaded."""
        steps = int(self.fl.get("local_ft_steps") or 0)
        if self.personal_head or self.ditto or steps <= 0 or "head" not in self.method.federated:
            return None
        X, y = self.data.store.X, self.data.store.y
        logits_by, idx_by = {}, {}
        for client in self.clients:
            self.load_client(client)                      # restores the global classifier first
            params = self.model.set_trainable({"head"}, self.roles)
            train_local(self.model, X, y, client.train_idx, params, lr=self.fl.lr_peft,
                        weight_decay=self.fl.weight_decay, epochs=self.fl.local_epochs,
                        batch_size=self.fl.batch_size, max_steps=steps, amp=self.amp,
                        generator=self.gen, scaler=self.scaler, logit_adj=client.logit_adj)
            logits_by[client.user], _ = predict(self.model, X, client.test_idx, amp=self.amp)
            idx_by[client.user] = client.test_idx
        return self.decoder.evaluate(self.data.store, logits_by, idx_by, self.data.n_classes)

    def personalize(self) -> float:
        """Final personalisation (methods with local parts, i.e. V5): every client updates
        ONLY its local parts (personal head, norm, adapter) once against the final global
        model, so no personal head is stale when the client is evaluated.  Nothing is sent.
        Returns the wall-clock seconds spent."""
        steps = int(self.fl.get("final_personalization_steps") or 0)
        roles = set(self.method.local) & set(self.method.trainable)
        if not self.method.local or steps <= 0 or not roles:
            return 0.0
        # fl.personalize_lora: the client also fine-tunes a PRIVATE copy of the LoRA adapters (never
        # uploaded) - the PEFT counterpart of personalising every weight as Ditto does.
        personal_lora = bool(self.fl.get("personalize_lora", False)) and self.method.use_lora
        if personal_lora:
            roles = roles | {"lora"}
        prox_mu = float(self.fl.get("personalize_prox", 0.0)) if personal_lora else 0.0
        X, y = self.data.store.X, self.data.store.y
        t0 = time.perf_counter()
        for client in self.clients:
            self.load_client(client)
            params = self.model.set_trainable(roles, self.roles)
            prox_ref = ({n: p.detach().clone() for n, p in self.model.named_parameters() if self.roles.get(n) == "lora"}
                        if prox_mu > 0 else None)
            train_local(self.model, X, y, client.train_idx, params, lr=self.fl[self.method.lr_key],
                        weight_decay=self.fl.weight_decay, epochs=self.fl.local_epochs, batch_size=self.fl.batch_size,
                        max_steps=steps, amp=self.amp, generator=self.gen, scaler=self.scaler,
                        protos=self.server.protos if self.method.use_prototypes else None,
                        proto_present=self.server.proto_present,
                        proto_weight=self.cfg.model.proto_weight if self.method.use_prototypes else 0.0,
                        logit_adj=client.logit_adj,
                        # fl.personalize_augment = false: the personal parts are fitted to the user's
                        # real (un-augmented) windows - a user's own grip/orientation is informative
                        augment=self.augment if self.fl.get("personalize_augment", True) else None,
                        global_head_weight=self.global_head_weight, global_adj=client.global_adj,
                        prox_mu=prox_mu, prox_ref=prox_ref)
            state = self.model.state_dict()
            client.local_state = {k: state[k].detach().clone() for k in self.local_keys}
            if personal_lora:
                client.personal_lora = {n: (l.lora_A.detach().clone(), l.lora_B.detach().clone())
                                        for n, l in self.model.lora_layers.items()}
        return time.perf_counter() - t0

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        fl, method = self.fl, self.method
        X, y = self.data.store.X, self.data.store.y
        n_sel = max(1, int(round(fl.clients_per_round_frac * len(self.clients))))
        lr = fl[method.lr_key]
        self.log.info(f"{encoder_display_name(self.encoder_name)} | {method.label}: {len(self.clients)} clients, "
                      f"{n_sel}/round, {fl.rounds} rounds, lr={lr}, r_max={self.r_max if method.use_lora else '-'}")

        history, evals = [], []
        cum_up = cum_down = 0.0
        mem_by_rank, time_by_rank, trainable_by_rank, up_by_rank = (defaultdict(list) for _ in range(4))
        t_run = time.perf_counter()

        for rnd in range(1, fl.rounds + 1):
            t_round = time.perf_counter()
            selected = self.rng.choice(len(self.clients), n_sel, replace=False)
            self.server.begin_round()
            up_bytes = down_bytes = 0
            losses, times = [], []
            for ci in selected:
                client = self.clients[ci]
                down_bytes += self.download_bytes(client.rank)                  # 1 download
                self.load_client(client)
                received = ({k: v.detach().clone() for k, v in self.model.state_dict().items()}
                            if self.ditto else None)
                params = self.param_groups(self.model.set_trainable(set(method.trainable), self.roles), lr)
                stats = train_local(                                            # 2-3 local training
                    self.model, X, y, client.train_idx, params, lr=lr, weight_decay=fl.weight_decay,
                    epochs=fl.local_epochs, batch_size=fl.batch_size, max_steps=fl.max_local_steps, amp=self.amp,
                    protos=self.server.protos if method.use_prototypes else None,
                    proto_present=self.server.proto_present,
                    proto_weight=self.cfg.model.proto_weight if method.use_prototypes else 0.0,
                    generator=self.gen, scaler=self.scaler, logit_adj=client.logit_adj, augment=self.augment,
                    global_head_weight=self.global_head_weight, global_adj=client.global_adj)
                upload = self.make_upload(client)                               # 4 prototypes, 5 upload
                nbytes = payload_bytes(upload)
                up_bytes += nbytes
                self.server.add_upload(upload, weight=client.n_train)
                if method.local:                                                # local parts stay on the device
                    state = self.model.state_dict()
                    client.local_state = {k: state[k].detach().clone() for k in self.local_keys}
                if self.ditto:                                                  # Ditto: personal model update
                    self.ditto_personal_step(client, received, lr)
                client.rounds_participated += 1
                losses.append(stats["loss"])
                times.append(stats["time_s"])
                mem_by_rank[client.rank].append(stats["train_mem_mb"])
                time_by_rank[client.rank].append(stats["time_s"])
                trainable_by_rank[client.rank].append(stats["trainable_params"])
                up_by_rank[client.rank].append(nbytes)
                del upload
            self.server.end_round()                                             # 6 aggregation
            if rnd == 1:
                up_bytes += self.stats_bytes     # class + transition counts, sent once by every client

            cum_up += up_bytes
            cum_down += down_bytes
            record = {
                "round": rnd,
                "uplink_mb": human_mb(up_bytes),
                "uplink_mb_per_client": human_mb(up_bytes / n_sel),
                "downlink_mb": human_mb(down_bytes),
                "cum_uplink_mb": human_mb(cum_up),
                "cum_downlink_mb": human_mb(cum_down),
                "train_loss": float(np.mean(losses)),
                "client_time_s": float(np.mean(times)),
                "round_time_s": time.perf_counter() - t_round,
                "agg_error": self.server.last_agg_error,
            }
            if rnd % fl.eval_every == 0 or rnd == fl.rounds:
                # The last round is always scored on ALL test windows (it is the final result).
                res = self.evaluate(full=(rnd == fl.rounds))
                metrics, conf = res["window"]
                point = {"round": rnd, "cum_uplink_mb": record["cum_uplink_mb"],
                         "elapsed_s": time.perf_counter() - t_run,
                         **{k: metrics[k] for k in ("macro_f1", "accuracy", "worst10_f1", "client_f1_std",
                                                    "client_f1_mean", "client_acc_mean")}}
                if "hmm" in res:
                    point["macro_f1_hmm"] = res["hmm"][0]["macro_f1"]
                evals.append(point)
                record.update({k: point[k] for k in ("macro_f1", "accuracy", "worst10_f1", "client_f1_std")})
                self.log.info(f"round {rnd:3d}  loss {record['train_loss']:.3f}  F1 {metrics['macro_f1']:.4f}  "
                              f"acc {metrics['accuracy']:.4f}  worst10 {metrics['worst10_f1']:.3f}  "
                              f"up {record['uplink_mb']:.2f} MB  ({record['round_time_s']:.1f}s/round)")
            history.append(record)

        total_time = time.perf_counter() - t_run

        # ---------------- final evaluation ----------------
        # (= the full evaluation of the last round, after the personalisation pass for V5)
        before_personalization = None
        personalization_s = self.personalize()
        if personalization_s > 0:
            before_personalization = {k: metrics[k] for k in ("macro_f1", "accuracy", "worst10_f1", "client_f1_std",
                                                              "client_acc_mean")}
            res = self.evaluate(full=True)
            metrics, conf = res["window"]
            self.log.info(f"after personalisation: F1 {metrics['macro_f1']:.4f} (was "
                          f"{before_personalization['macro_f1']:.4f}), {personalization_s:.0f}s")
        final = metrics
        final_hmm, conf_hmm = res["hmm"] if "hmm" in res else (None, None)
        if self.method.local and self.fl.get("new_user_local_init", "pretrained") == "average":
            self.average_local_parts()
        self.save_logits()
        t_ft = time.perf_counter()
        ft = self.local_finetune_eval()
        final_ft = ft["window"][0] if ft else None
        final_ft_hmm = ft["hmm"][0] if ft and "hmm" in ft else None
        if ft:
            self.log.info(f"+local fine-tuning: F1 {final_ft['macro_f1']:.4f} client-acc {final_ft['client_acc_mean']:.4f}"
                          + (f" (+HMM {final_ft_hmm['macro_f1']:.4f})" if final_ft_hmm else "")
                          + f", {time.perf_counter() - t_ft:.0f}s")
        tier_f1 = {}
        if method.hetero_rank:
            for r in sorted({c.rank for c in self.clients}):
                vals = [final["client_f1"][c.user] for c in self.clients if c.rank == r]
                tier_f1[f"r{r}"] = {"mean_f1": float(np.nanmean(vals)), "n_clients": len(vals)}

        # New users: global model (+ default local parts), highest rank.
        self.load_client(None, rank=self.r_max)
        if self.fl.get("save_new_user_model", True):
            self.save_new_user_model()
        heldout = self.evaluate_new_users()
        # split.transductive_report: V5 also reports the transductive zero-shot rule (same model)
        heldout_transductive = None
        if (heldout is not None and method.heldout_mode == "prototype"
                and self.cfg.split.get("transductive_report", False)):
            before = self.cfg.split.get("zero_shot_refine_iters", 0)
            self.cfg.split.zero_shot_refine_iters = int(before or 0) or 10
            try:
                heldout_transductive = self.evaluate_new_users()
            finally:                               # the config may be shared with the next run
                self.cfg.split.zero_shot_refine_iters = before

        hit = first_reach(evals, self.target_f1)
        per_rank = {f"r{r}": {"train_mem_mb": float(np.mean(mem_by_rank[r])),
                              "client_time_s": float(np.mean(time_by_rank[r])),
                              "trainable_params": int(np.mean(trainable_by_rank[r])),
                              "uplink_mb_per_update": human_mb(np.mean(up_by_rank[r]))}
                    for r in sorted(mem_by_rank)}
        all_mem = [m for v in mem_by_rank.values() for m in v]
        all_time = [t for v in time_by_rank.values() for t in v]

        result = {
            "kind": "federated",
            "encoder": self.encoder_name,
            "encoder_display": encoder_display_name(self.encoder_name),
            "method": method.name,
            "method_short": method.short,
            "method_label": method.label + ((f" (r={self.r_max})" if self.tag.startswith("__r") and self.tag[3:].isdigit()
                                             else f" [{self.tag[2:]}]") if self.tag else ""),
            "method_slot": method.slot,
            "tag": self.tag,
            "seed": self.seed,
            "profile": self.cfg.profile,
            "n_classes": self.data.n_classes,
            "class_names": self.data.store.class_names,
            "config": {"rounds": fl.rounds, "clients": len(self.clients), "clients_per_round": n_sel,
                       "local_epochs": fl.local_epochs, "max_local_steps": fl.max_local_steps,
                       "batch_size": fl.batch_size, "lr": lr,
                       "lr_head": self.fl.get("lr_head") if "encoder" in method.trainable else lr,
                       "r_max": self.r_max if method.use_lora else None,
                       "tiers": list(self.cfg.model.lora.tiers) if method.hetero_rank else None,
                       "aggregation": method.aggregation, "proto_weight": self.cfg.model.proto_weight
                       if method.use_prototypes else 0.0,
                       "tau_train": float(self.client_decoder.tau_train), "tau_inf": float(self.client_decoder.tau_inf),
                       "personal_head_plain_ce": bool(self.personal_head or self.ditto),
                       "ditto_lambda": float(self.fl.get("ditto_lambda", 1.0)) if self.ditto else None,
                       "hmm": bool(self.cfg.decoding.hmm),
                       "final_personalization_steps": int(self.fl.get("final_personalization_steps") or 0)
                       if method.local else 0,
                       "classwise_head": bool(self.server.class_row_keys),
                       "new_user_local_init": self.fl.get("new_user_local_init", "pretrained") if method.local else None,
                       "new_user_head": ("global_head" if self.global_head else
                                         self.fl.get("new_user_head", "prototype")) if method.local else None,
                       "global_head_weight": self.global_head_weight,
                       "personalize_lora": bool(self.fl.get("personalize_lora", False)) if method.local else None,
                       "new_user_init_upload_mb": human_mb(self.new_user_bytes),
                       "augment": self.augment is not None},
            "param_counts": self.model.count_by_role(),
            "history": history,
            "evals": evals,
            "final": final,
            "confusion": conf,
            "final_hmm": final_hmm,
            "confusion_hmm": conf_hmm,
            "final_before_personalization": before_personalization,
            "final_ft": final_ft,
            "final_ft_hmm": final_ft_hmm,
            "personalization_s": personalization_s,
            "tier_f1": tier_f1,
            "comm": {
                "uplink_mb_per_client_round": float(np.mean([h["uplink_mb_per_client"] for h in history])),
                "uplink_mb_per_round": float(np.mean([h["uplink_mb"] for h in history])),
                "downlink_mb_per_round": float(np.mean([h["downlink_mb"] for h in history])),
                "total_uplink_mb": human_mb(cum_up),
                "total_downlink_mb": human_mb(cum_down),
                "per_rank": per_rank,
            },
            "target": {"target_f1": self.target_f1,
                       "rounds_to_target": hit["round"] if hit else None,
                       "mb_to_target": hit["cum_uplink_mb"] if hit else None},
            "resources": {"client_train_mem_mb_mean": float(np.mean(all_mem)),
                          "client_train_mem_mb_max": float(np.max(all_mem)),
                          "client_time_s_mean": float(np.mean(all_time)),
                          "total_time_s": total_time},
            "participation": {c.user: c.rounds_participated for c in self.clients},
            "heldout": heldout,
        }
        if heldout_transductive is not None:
            result["heldout_transductive"] = heldout_transductive
        out = (Path(self.cfg.paths.results_dir) / self.cfg.profile / "federated" /
               f"{self.encoder_name}__{method.name}{self.tag}__seed{self.seed}.json")
        save_json(result, out)
        hmm_txt = f" (+HMM {final_hmm['macro_f1']:.4f})" if final_hmm else ""
        zs = f"held-out zero-shot {heldout['zero_shot']['mean_user_f1']:.3f} | " if heldout else ""
        self.log.info(f"FINAL F1 {final['macro_f1']:.4f}{hmm_txt} acc {final['accuracy']:.4f} "
                      f"client-acc {final['client_acc_mean']:.4f} worst10 {final['worst10_f1']:.3f} "
                      f"| up {result['comm']['uplink_mb_per_client_round']:.3f} MB/client/round | "
                      f"{zs}{total_time / 60:.1f} min")
        return result

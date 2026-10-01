"""
From network outputs to activity decisions - the same rules for every method.

    window level   argmax f(x) + tau_inf * log pi_global          (imbalance.py)
    + HMM          Viterbi over each user's chronological windows   (hmm.py)

Every evaluation (centralized test, FL clients, held-out users) goes through one
`Decoder`, so all methods are scored with identical decision rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .hmm import ActivityHMM, build_hmm
from .imbalance import calibrated_predictions, class_prior
from .metrics import confusion, evaluate_predictions


@dataclass
class Decoder:
    log_prior: torch.Tensor          # log of the global class prior (C,)
    tau_train: float                 # logit-adjustment strength used in the loss
    tau_inf: float                   # prior strength at prediction time
    hmm: ActivityHMM | None = None   # temporal smoothing (None = off)

    def window(self, logits: torch.Tensor, tau_inf: float | None = None) -> np.ndarray:
        return calibrated_predictions(logits, self.log_prior, self.tau_inf if tau_inf is None else tau_inf)

    def smoothed(self, logits: torch.Tensor, breaks: np.ndarray, log_prior: torch.Tensor | None = None) -> np.ndarray:
        return self.hmm.smooth(logits, breaks, self.log_prior if log_prior is None else log_prior, self.tau_train)

    def evaluate(self, store, logits_by_user: dict[str, torch.Tensor], idx_by_user: dict[str, torch.Tensor],
                 n_classes: int, log_prior_by_user: dict[str, torch.Tensor] | None = None) -> dict:
        """Metrics + confusion matrices, window-level and (if enabled) HMM-smoothed.

        `log_prior_by_user` gives each user's OWN label prior for the HMM emissions -
        used for personal classifiers, which model their own client's distribution.
        """
        y_true = {u: store.y[idx_by_user[u]].cpu().numpy() for u in logits_by_user}
        pred = {u: self.window(lg) for u, lg in logits_by_user.items()}
        out = {"window": (evaluate_predictions(y_true, pred, n_classes), confusion(y_true, pred, n_classes))}
        if self.hmm is not None:
            lp = log_prior_by_user or {}
            pred_h = {u: self.smoothed(lg, store.sequence_breaks(idx_by_user[u]), lp.get(u))
                      for u, lg in logits_by_user.items()}
            out["hmm"] = (evaluate_predictions(y_true, pred_h, n_classes), confusion(y_true, pred_h, n_classes))
        return out


def make_decoder(cfg, data) -> tuple[Decoder, torch.Tensor, int]:
    """Decoder for one experiment, built only from the FL clients' TRAINING parts.

    Returns (decoder, global class prior, bytes the clients upload once to share
    their class counts + transition counts).
    """
    counts = sum(data.store.class_counts(idx) for idx in data.client_train.values())
    prior = class_prior(counts)
    hmm, n_bytes = (build_hmm(data.store, data.client_train, data.n_classes)
                    if cfg.decoding.hmm else (None, 0))
    n_bytes += len(data.client_train) * data.n_classes * 8           # class counts
    dec = Decoder(torch.log(prior), float(cfg.imbalance.tau_train), float(cfg.imbalance.tau_inf), hmm)
    return dec, prior, n_bytes

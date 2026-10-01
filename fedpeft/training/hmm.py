"""
Temporal smoothing with a hidden Markov model (HMM).

Activities last minutes, while every prediction here is made for one 10-s window
on its own.  The Capture-24 authors therefore smooth window predictions with an HMM
(Walmsley et al. 2021; Chan et al., Sci Data 2024), which is the standard
post-processing for this dataset.  We do the same:

    hidden state   the true activity of each window, in time order
    transitions    P(activity_t | activity_t-1), counted from the TRAINING label
                   sequences (in FL: every client counts its own 6 x 6 transitions
                   once and the server adds them up - a tiny statistic, no raw data)
    emissions      the classifier's class-balanced scores f(x) (see imbalance.py),
                   i.e. log p(x | activity) up to a constant
    decoding       Viterbi: the single most likely activity sequence

Sequences are cut wherever two consecutive kept windows are far apart in time
(device off / unlabelled period), so the HMM never smooths across a gap.
"""

from __future__ import annotations

import numpy as np
import torch


class ActivityHMM:
    """Transition statistics + Viterbi decoding."""

    def __init__(self, n_classes: int, smoothing: float = 1.0):
        self.n = n_classes
        self.trans_counts = np.full((n_classes, n_classes), smoothing, dtype=np.float64)
        self.init_counts = np.full(n_classes, smoothing, dtype=np.float64)

    # ------------------------------------------------------------ statistics
    @staticmethod
    def count_transitions(y: np.ndarray, breaks: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
        """Transition and start counts of ONE user's chronological label sequence."""
        trans = np.zeros((n_classes, n_classes))
        init = np.zeros(n_classes)
        np.add.at(init, y[breaks], 1)
        cont = ~breaks[1:]                                  # pairs inside one sequence
        np.add.at(trans, (y[:-1][cont], y[1:][cont]), 1)
        return trans, init

    def add_counts(self, trans: np.ndarray, init: np.ndarray) -> None:
        """Server side: add one client's counts."""
        self.trans_counts += trans
        self.init_counts += init

    @property
    def log_trans(self) -> np.ndarray:
        return np.log(self.trans_counts / self.trans_counts.sum(1, keepdims=True))

    @property
    def log_init(self) -> np.ndarray:
        return np.log(self.init_counts / self.init_counts.sum())

    # ------------------------------------------------------------- decoding
    def viterbi(self, log_emit: np.ndarray) -> np.ndarray:
        """Most likely state path for one sequence of per-window log-emission scores (T x C)."""
        T = len(log_emit)
        if T == 0:
            return np.zeros(0, dtype=np.int64)
        lt = self.log_trans
        delta = self.log_init + log_emit[0]
        back = np.zeros((T, self.n), dtype=np.int64)
        for t in range(1, T):
            scores = delta[:, None] + lt                     # (from, to)
            back[t] = scores.argmax(0)
            delta = scores.max(0) + log_emit[t]
        path = np.empty(T, dtype=np.int64)
        path[-1] = int(delta.argmax())
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    def smooth(self, logits: torch.Tensor, breaks: np.ndarray, log_prior: torch.Tensor,
               tau_train: float) -> np.ndarray:
        """HMM-smoothed decisions for one user's chronological windows.

        The model output f(x) is turned into a log-likelihood score:
            log p(x | c)  ~  f_c(x) - (1 - tau_train) * log pi_c
        (with tau_train = 1 the network already outputs class-balanced scores).
        The emissions are log-softmax normalised per window, which only removes a
        constant and keeps the numbers well scaled.
        """
        f = logits.float().cpu()
        f = f - (1.0 - tau_train) * log_prior.cpu()
        log_emit = torch.log_softmax(f, dim=1).numpy()
        out = np.empty(len(log_emit), dtype=np.int64)
        starts = np.flatnonzero(breaks)
        ends = np.append(starts[1:], len(log_emit))
        for s, e in zip(starts, ends):
            out[s:e] = self.viterbi(log_emit[s:e])
        return out


def build_hmm(store, index_by_user: dict[str, torch.Tensor], n_classes: int) -> tuple[ActivityHMM, int]:
    """Collect transition counts from every user's (training) sequence.

    Returns the HMM and the number of bytes the clients would upload to share their
    counts once (int64 values), for honest communication accounting.
    """
    hmm = ActivityHMM(n_classes)
    for idx in index_by_user.values():
        y = store.y[idx].cpu().numpy()
        trans, init = ActivityHMM.count_transitions(y, store.sequence_breaks(idx), n_classes)
        hmm.add_counts(trans, init)
    n_bytes = len(index_by_user) * (n_classes * n_classes + n_classes) * 8
    return hmm, n_bytes

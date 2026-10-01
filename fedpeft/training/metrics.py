"""
Metrics (PDF p. 11, "METRICS"):

    Macro-F1 (primary), accuracy                  -> pooled over all test windows
    Worst-10 % client F1, client-F1 std           -> fairness across users
    (uplink MB, rounds/MB to target, memory, time -> fedpeft/federated/simulation.py)
    Zero-/few-shot F1 on held-out users           -> fedpeft/training/heldout.py

Definitions used everywhere in this project:
    per-client macro-F1   F1 averaged over the classes that occur in THAT client's
                          test data (a client who never cycles is not scored on cycling)
    worst-10 % client F1  mean per-client F1 of the lowest-scoring 10 % of clients
    client-F1 std         standard deviation of the per-client F1 values
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[float]:
    return f1_score(y_true, y_pred, average=None, labels=np.arange(n_classes), zero_division=0).tolist()


def fairness_summary(client_f1: dict[str, float]) -> dict:
    """Worst-10 %, spread and distribution of per-client F1."""
    values = np.array(sorted(v for v in client_f1.values() if not math.isnan(v)))
    if len(values) == 0:
        return {}
    k = max(1, int(math.ceil(0.10 * len(values))))
    return {
        "client_f1_mean": float(values.mean()),
        "client_f1_median": float(np.median(values)),
        "client_f1_std": float(values.std()),
        "worst10_f1": float(values[:k].mean()),       # mean of the worst 10 % of clients
        "p10_f1": float(np.percentile(values, 10)),   # 10th percentile (alternative definition)
        "client_f1_min": float(values[0]),
    }


def evaluate_predictions(y_true_by_user: dict[str, np.ndarray], y_pred_by_user: dict[str, np.ndarray],
                         n_classes: int) -> dict:
    """All classification metrics from per-user predictions."""
    users = sorted(y_true_by_user)
    y_true = np.concatenate([y_true_by_user[u] for u in users])
    y_pred = np.concatenate([y_pred_by_user[u] for u in users])
    client_f1 = {u: macro_f1(y_true_by_user[u], y_pred_by_user[u], labels=np.unique(y_true_by_user[u]))
                 for u in users}
    out = {
        "macro_f1": macro_f1(y_true, y_pred, labels=np.arange(n_classes)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": per_class_f1(y_true, y_pred, n_classes),
        "client_f1": client_f1,
        "n_test_windows": int(len(y_true)),
    }
    out.update(fairness_summary(client_f1))
    return out


def confusion(y_true_by_user: dict[str, np.ndarray], y_pred_by_user: dict[str, np.ndarray],
              n_classes: int) -> list[list[int]]:
    users = sorted(y_true_by_user)
    y_true = np.concatenate([y_true_by_user[u] for u in users])
    y_pred = np.concatenate([y_pred_by_user[u] for u in users])
    return confusion_matrix(y_true, y_pred, labels=np.arange(n_classes)).tolist()

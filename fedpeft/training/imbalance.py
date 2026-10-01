"""
Class imbalance: prior-aware training and calibrated prediction.

Why this module exists
----------------------
Capture-24 is dominated by sleep (37 %) and sitting (39 %); bicycling is 1 % of the
windows and most participants never cycle at all.  With plain cross-entropy:

  * centrally, rare classes are learned but under-predicted;
  * in federated learning it is much worse: a client WITHOUT bicycling windows
    actively pushes the bicycling logit down on every step, and averaging those
    updates erases the class from the global model (per-class F1 = 0.000 in the
    first standard run for bicycling / vehicle / walking in many FL models).

The fix is logit adjustment ("balanced softmax", Menon et al. ICLR 2021; used for
label skew in FL by FedLC, Zhang et al. ICML 2022):

  training   loss = CE( f(x) + tau_train * log(pi) , y )
             pi = the label prior of the data being trained on - the GLOBAL prior
             centrally, the client's OWN prior in FL.  A class the client does not
             have gets log(pi) ~ -9, so its logit is not pushed down any more.
             The network output f(x) then estimates class-balanced scores.
  prediction argmax_c  f_c(x) + tau_inf * log(pi_global_c)
             tau_inf = 0 predicts as if all classes were equally likely (best for
             balanced accuracy); tau_inf = tau_train gives the usual accuracy-optimal
             prediction under the true prior.  Macro-F1 is best in between; the value
             is fixed once in the config (chosen on the validation users).

The global prior is a 6-number statistic (per-class window counts summed over the
clients), which V5 already sends with its prototypes.
"""

from __future__ import annotations

import numpy as np
import torch

PRIOR_FLOOR = 1e-4   # prior given to a class that is absent from a client's data


def class_prior(counts, floor: float = PRIOR_FLOOR) -> torch.Tensor:
    """Smoothed class frequencies (absent classes get `floor`), as a float tensor."""
    c = torch.as_tensor(np.asarray(counts, dtype=np.float64))
    p = c / c.sum().clamp(min=1)
    p = p.clamp(min=floor)
    return (p / p.sum()).float()


def logit_adjustment(prior: torch.Tensor, tau: float, device=None) -> torch.Tensor:
    """tau * log(prior): the vector added to the logits (in the loss or at prediction)."""
    adj = tau * torch.log(prior)
    return adj.to(device) if device is not None else adj


def calibrated_predictions(logits: torch.Tensor, log_prior: torch.Tensor, tau_inf: float) -> np.ndarray:
    """Window-level class decisions: argmax(f(x) + tau_inf * log pi_global)."""
    return (logits.float() + tau_inf * log_prior.to(logits.device)).argmax(1).cpu().numpy()

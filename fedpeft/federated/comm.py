"""
Communication accounting.

Every message between a client and the server is an explicit (nested) dict of
tensors.  Its size on the wire is measured as the size of the tensors serialised
as uncompressed NumPy arrays - float32 for real values, int64 for integers - which
is exactly what Flower's `ndarrays_to_parameters` sends.  No compression and no
quantisation are applied (they would reduce every method equally).
"""

from __future__ import annotations

import torch


def tensor_bytes(t: torch.Tensor) -> int:
    """Bytes of one tensor on the wire (fp32 for floats, int64 for integers)."""
    return t.numel() * (4 if t.dtype.is_floating_point else 8)


def payload_bytes(obj) -> int:
    """Recursively add up the wire size of every tensor in a (nested) payload."""
    if isinstance(obj, torch.Tensor):
        return tensor_bytes(obj)
    if isinstance(obj, dict):
        return sum(payload_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(payload_bytes(v) for v in obj)
    return 8 if isinstance(obj, (int, float)) else 0   # scalars such as n_k

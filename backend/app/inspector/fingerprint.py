from __future__ import annotations

import hashlib

import torch


def fingerprint_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """Canonical weights fingerprint, independent of serialization format.

    Hashes name + shape + dtype + raw bytes of every tensor in name order, so the
    same weights match whether they were saved as .pt, .safetensors, or a
    checkpoint bundle.
    """
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            continue
        data = tensor.detach().cpu().contiguous()
        raw = data.reshape(1).view(torch.uint8) if data.dim() == 0 else data.view(torch.uint8)
        digest.update(name.encode("utf-8"))
        digest.update(str(list(data.shape)).encode("utf-8"))
        digest.update(str(data.dtype).encode("utf-8"))
        # view as raw bytes so dtypes without numpy support (e.g. bfloat16) still hash
        digest.update(raw.numpy().tobytes() if data.numel() else b"")
    return digest.hexdigest()

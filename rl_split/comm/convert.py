"""numpy/torch <-> message field conversion."""

from __future__ import annotations

import json

import numpy as np
import torch


def to_np(field, dtype, shape=None) -> np.ndarray:
    """Received sequence (list or bytes) -> numpy array."""
    if isinstance(field, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(field, dtype=dtype)
    else:
        arr = np.asarray(field, dtype=dtype)
    return arr.reshape(shape) if shape is not None else arr


def to_torch(field, dtype, shape, device) -> torch.Tensor:
    return torch.from_numpy(to_np(field, dtype, shape)).to(device)


def flat(t: torch.Tensor, dtype=np.float32) -> np.ndarray:
    """Tensor -> 1-D C-contiguous numpy array, the form int2dds copies as-is."""
    return np.ascontiguousarray(t.detach().to("cpu").numpy().astype(dtype, copy=False).reshape(-1))


# IDL kind code in _dds_type_info_fields -> element size (octet, long, unsigned long, float)
_ELEM_BYTES = {1: 1, 5: 4, 6: 4, 11: 4}


def payload_bytes(sample) -> int:
    """Size of a message's field contents (no CDR header or padding). Reads lengths only, so it is cheap."""
    total = 0
    for kind, name, code, *_ in sample._dds_type_info_fields:
        v = getattr(sample, name)
        if isinstance(v, np.ndarray):
            total += v.nbytes
        elif isinstance(v, (bytes, bytearray, memoryview, str)):
            total += len(v)
        elif kind == "seq" and code == 13:  # sequence<string>
            total += sum(len(s) for s in v)
        elif kind == "seq":
            total += len(v) * _ELEM_BYTES.get(code, 4)
        else:
            total += _ELEM_BYTES.get(code, 4)
    return total


def fmt_bytes(n: int) -> str:
    if n < 1e3:
        return f"{n}B"
    if n < 1e6:
        return f"{n / 1e3:.1f}KB"
    return f"{n / 1e6:.2f}MB"


def log_to_json(extras: dict) -> str:
    """extras["log"] (scalars or tensors) -> JSON {key: float}."""
    log = extras.get("log") or {}
    return json.dumps({k: float(v.float().mean()) if torch.is_tensor(v) else float(v) for k, v in log.items()})


def log_from_json(text: str) -> dict:
    return json.loads(text) if text else {}


def state_to_msg(models: dict[str, torch.nn.Module]) -> tuple[list[str], np.ndarray]:
    """{"actor": m, "critic": m} -> (names, flat float32 data) in sorted name order."""
    tensors = {}
    for prefix, model in models.items():
        for k, v in model.state_dict().items():
            tensors[f"{prefix}.{k}"] = v
    names = sorted(tensors)
    data = np.concatenate([flat(tensors[n]) for n in names]) if names else np.zeros(0, np.float32)
    return names, data


def load_state_from_msg(models: dict[str, torch.nn.Module], names: list[str], data) -> None:
    """Inverse of state_to_msg. Shapes and dtypes come from the local models."""
    data = to_np(data, np.float32)
    targets = {}
    for prefix, model in models.items():
        for k, v in model.state_dict().items():
            targets[f"{prefix}.{k}"] = v
    if sorted(targets) != list(names):
        raise ValueError("weights message does not match the local model")
    offset = 0
    states: dict[str, dict] = {p: {} for p in models}
    for n in names:
        ref = targets[n]
        chunk = data[offset : offset + ref.numel()]
        offset += ref.numel()
        prefix, key = n.split(".", 1)
        states[prefix][key] = torch.from_numpy(chunk.reshape(ref.shape).copy()).to(ref.device, ref.dtype)
    for prefix, model in models.items():
        model.load_state_dict(states[prefix])

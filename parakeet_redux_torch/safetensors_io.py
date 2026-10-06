"""Minimal safetensors reader (no external dependency).

The format: 8 bytes little-endian header length, then a JSON header mapping
tensor names to {dtype, shape, data_offsets}, then the raw data buffer.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": None,  # not used by this model
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "BOOL": np.bool_,
}


def load_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Load every tensor of a .safetensors file into a dict of numpy arrays."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        tensors: dict[str, np.ndarray] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = _DTYPES.get(info["dtype"])
            if dtype is None:
                raise ValueError(f"unsupported dtype {info['dtype']} for {name}")
            start, end = info["data_offsets"]
            f.seek(data_start + start)
            buf = f.read(end - start)
            arr = np.frombuffer(buf, dtype=dtype).reshape(info["shape"])
            tensors[name] = arr
    return tensors

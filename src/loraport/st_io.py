"""Minimal safetensors reader that also understands BF16 (numpy has no bfloat16).

The format is an 8-byte little-endian header length, a JSON header, then raw
bytes.  Reading it directly lets us widen BF16 to float32 without torch.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file as _save_file

from .errors import AdapterFormatError

_MAX_HEADER = 100_000_000  # safetensors spec limit
_ITEMSIZE = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8}


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


class SafetensorsFile:
    """Lazy reader: parses the header once, reads tensors on demand."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        try:
            size = self.path.stat().st_size
            with open(self.path, "rb") as f:
                head = f.read(8)
                if len(head) < 8:
                    raise AdapterFormatError(f"{self.path}: file is too small to be a safetensors file")
                (n,) = struct.unpack("<Q", head)
                if n > _MAX_HEADER or 8 + n > size:
                    raise AdapterFormatError(f"{self.path}: invalid safetensors header length {n}")
                header = json.loads(f.read(n))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            raise AdapterFormatError(f"{self.path}: cannot read safetensors header ({e})") from e
        self._offset = 8 + n
        self.metadata: dict[str, str] = header.pop("__metadata__", {}) or {}
        self.tensors: dict[str, TensorInfo] = {}
        data_size = size - self._offset
        for name, spec in header.items():
            try:
                start, end = spec["data_offsets"]
                info = TensorInfo(spec["dtype"], tuple(int(d) for d in spec["shape"]), int(start), int(end))
            except (KeyError, TypeError, ValueError) as e:
                raise AdapterFormatError(f"{self.path}: malformed header entry for {name!r}") from e
            if info.end > data_size or info.start > info.end:
                raise AdapterFormatError(f"{self.path}: tensor {name!r} points past the end of the file (truncated?)")
            self.tensors[name] = info
        self.nbytes = size

    def read(self, name: str) -> np.ndarray:
        """Read one tensor as float32 (F32/F16/BF16/F64 inputs)."""
        info = self.tensors[name]
        if info.dtype not in _ITEMSIZE:
            raise AdapterFormatError(
                f"{self.path}: tensor {name!r} has dtype {info.dtype}; LoRA adapters must be F32, F16 or BF16 "
                "(dequantize or re-save the adapter first)"
            )
        count = int(np.prod(info.shape, dtype=np.int64)) if info.shape else 1
        if info.end - info.start != count * _ITEMSIZE[info.dtype]:
            raise AdapterFormatError(f"{self.path}: tensor {name!r} byte size does not match its shape {info.shape}")
        with open(self.path, "rb") as f:
            f.seek(self._offset + info.start)
            raw = f.read(info.end - info.start)
        if info.dtype == "BF16":
            bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16  # lossless widening
            arr = bits.view(np.float32)
        else:
            np_dtype = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}[info.dtype]
            arr = np.frombuffer(raw, dtype=np_dtype).astype(np.float32)
        return arr.reshape(info.shape)


def save_float32(path: Path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> None:
    """Write float32 tensors (contiguous copies) with optional metadata."""
    clean = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in tensors.items()}
    _save_file(clean, str(path), metadata=metadata)


def save_bf16(path: Path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> None:
    """Write float32 arrays as BF16 (round to nearest even). Used for synthetic fixtures."""
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = metadata
    blobs: list[bytes] = []
    pos = 0
    for name, arr in tensors.items():
        bits = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
        rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
        blob = rounded.tobytes()
        header[name] = {"dtype": "BF16", "shape": list(arr.shape), "data_offsets": [pos, pos + len(blob)]}
        blobs.append(blob)
        pos += len(blob)
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)  # keep the data section 8-byte aligned
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for blob in blobs:
            f.write(blob)

"""In-memory representation shared by both formats.

Canonical layout (PEFT-style): ``A`` is ``(r, in)``, ``B`` is ``(out, r)`` and
``dW = scale * B @ A``.  Module names are the checkpoint key without the
format-specific prefix/suffix, e.g. ``model.layers.3.self_attn.q_proj``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")


def layer_of(name: str) -> int | None:
    """Return the transformer block index in a module name, or None."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def block_relative(name: str) -> str | None:
    """Return the part after ``layers.{i}.`` (e.g. ``self_attn.q_proj``), or None."""
    m = _LAYER_RE.search(name)
    return m.group(2) if m else None


def terminal(name: str) -> str:
    """Return the last dotted component (e.g. ``q_proj``)."""
    return name.rsplit(".", 1)[-1]


def number(x: float) -> int | float:
    """Render integral floats as ints (``80.0`` -> ``80``) for tidy JSON."""
    return int(x) if float(x).is_integer() else float(x)


@dataclass
class LoraModule:
    """One adapted linear layer."""

    A: np.ndarray  # (r, in)
    B: np.ndarray  # (out, r)
    scale: float

    @property
    def rank(self) -> int:
        return int(self.A.shape[0])

    @property
    def consistent(self) -> bool:
        """True when A and B are 2-D and share the rank dimension."""
        return self.A.ndim == 2 and self.B.ndim == 2 and self.A.shape[0] == self.B.shape[1]


@dataclass
class Adapter:
    """A LoRA adapter read from disk (always float32 in memory)."""

    format: str  # "mlx" or "peft"
    rank: int
    scale: float
    modules: dict[str, LoraModule]
    key_issues: list[str] = field(default_factory=list)  # tensors a runtime would ignore or that are unpaired
    unsupported: list[str] = field(default_factory=list)  # reasons loraport refuses to convert
    warnings: list[str] = field(default_factory=list)
    source_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def lora_alpha(self) -> int | float:
        """PEFT ``lora_alpha`` equivalent of this adapter (non-rslora convention)."""
        stored = self.source_meta.get("lora_alpha")
        return stored if stored is not None else number(round(self.scale * self.rank, 10))

    @property
    def layers(self) -> list[int]:
        return sorted({i for i in map(layer_of, self.modules) if i is not None})

"""Cross-check adapter tensor shapes against a base model ``config.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import BaseConfigError
from .model import Adapter, layer_of, terminal


@dataclass(frozen=True)
class BaseDims:
    num_layers: int
    hidden: int
    intermediate: int | None
    heads: int
    kv_heads: int
    head_dim: int

    def expected(self, term: str) -> tuple[int, int] | None:
        """Expected ``(in, out)`` of a well-known Llama-style projection, else None."""
        q_out = self.heads * self.head_dim
        kv_out = self.kv_heads * self.head_dim
        table = {
            "q_proj": (self.hidden, q_out),
            "k_proj": (self.hidden, kv_out),
            "v_proj": (self.hidden, kv_out),
            "o_proj": (q_out, self.hidden),
        }
        if self.intermediate is not None:
            table |= {
                "gate_proj": (self.hidden, self.intermediate),
                "up_proj": (self.hidden, self.intermediate),
                "down_proj": (self.intermediate, self.hidden),
            }
        return table.get(term)


def load_base_config(path: str | Path) -> BaseDims:
    """Read ``config.json`` (a file, or a folder that contains it)."""
    p = Path(path)
    if p.is_dir():
        p = p / "config.json"
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise BaseConfigError(f"{p}: cannot read base model config ({e})") from e
    if isinstance(cfg.get("text_config"), dict):  # multimodal wrappers
        cfg = cfg["text_config"]
    try:
        hidden = int(cfg["hidden_size"])
        heads = int(cfg["num_attention_heads"])
        layers = int(cfg["num_hidden_layers"])
    except (KeyError, TypeError, ValueError) as e:
        raise BaseConfigError(
            f"{p}: needs hidden_size, num_attention_heads and num_hidden_layers (missing or invalid: {e})"
        ) from e
    inter = cfg.get("intermediate_size")
    return BaseDims(
        layers,
        hidden,
        int(inter) if inter is not None else None,
        heads,
        int(cfg.get("num_key_value_heads") or heads),
        int(cfg.get("head_dim") or hidden // heads),
    )


def check_adapter(adapter: Adapter, dims: BaseDims) -> list[str]:
    """Return human-readable problems (empty list means consistent)."""
    problems: list[str] = []
    for name in sorted(adapter.modules):
        mod = adapter.modules[name]
        layer = layer_of(name)
        if layer is not None and not 0 <= layer < dims.num_layers:
            problems.append(
                f"{name}: layer {layer} is outside the base model's range 0..{dims.num_layers - 1} "
                f"(num_hidden_layers={dims.num_layers})"
            )
        exp = dims.expected(terminal(name))
        if exp is None or not mod.consistent:
            continue
        r = mod.rank
        want_a, want_b = (r, exp[0]), (exp[1], r)
        if tuple(mod.A.shape) != want_a or tuple(mod.B.shape) != want_b:
            problems.append(
                f"{name}: expected A{want_a} B{want_b} for the base model, "
                f"got A{tuple(mod.A.shape)} B{tuple(mod.B.shape)}"
            )
    return problems

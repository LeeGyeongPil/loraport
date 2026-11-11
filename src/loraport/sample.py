"""Synthetic practice adapters (mock) and deliberately broken variants.

Nothing here is a trained model: weights are random numbers with the exact
tensor layout and config fields of real mlx-lm / PEFT adapters, sized like a
tiny Llama (64 hidden, 4 layers).  The converter and verifier that run on them
are the real ones.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from .detect import load_adapter
from .formats.common import CONFIG_NAME
from .formats.mlx import write_mlx
from .formats.peft import PREFIX, write_peft
from .formats.peft import WEIGHTS_NAME as PEFT_WEIGHTS
from .model import Adapter, LoraModule
from .st_io import save_bf16

TINY_BASE = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 128,
}
_HEAD = TINY_BASE["hidden_size"] // TINY_BASE["num_attention_heads"]
_KV = TINY_BASE["num_key_value_heads"] * _HEAD
DIMS: dict[str, tuple[int, int]] = {  # module -> (in, out)
    "self_attn.q_proj": (64, 64),
    "self_attn.k_proj": (64, _KV),
    "self_attn.v_proj": (64, _KV),
    "self_attn.o_proj": (64, 64),
    "mlp.gate_proj": (64, 128),
    "mlp.up_proj": (64, 128),
    "mlp.down_proj": (128, 64),
}
TRAP_NAMES = ("no-prefix", "alpha-not-times-r", "ab-swapped", "mlp-missing")
EXTRA_TRAP = "untransposed"  # caught by the shape check; kept out of the demo's 4/4 count


def random_modules(layers: list[int], rank: int, scale: float, seed: int) -> dict[str, LoraModule]:
    """Random LoRA factors for every module of the given blocks."""
    rng = np.random.default_rng(seed)
    mods: dict[str, LoraModule] = {}
    for i in layers:
        for rel, (din, dout) in DIMS.items():
            bound = 1 / math.sqrt(din)
            A = rng.uniform(-bound, bound, (rank, din)).astype(np.float32)
            B = (rng.normal(size=(dout, rank)) * 0.01).astype(np.float32)
            mods[f"model.layers.{i}.{rel}"] = LoraModule(A, B, scale)
    return mods


def write_base_config(dst: Path) -> Path:
    """Write the tiny base model ``config.json`` that matches the practice adapters."""
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / "config.json"
    path.write_text(json.dumps(TINY_BASE, indent=2) + "\n", encoding="utf-8")
    return path


def make_mlx_adapter(dst: Path, *, rank: int = 4, scale: float = 20.0, num_layers: int = 2, seed: int = 0) -> Path:
    """MLX adapter on the last ``num_layers`` blocks (mlx-lm counts from the end)."""
    total = TINY_BASE["num_hidden_layers"]
    layers = list(range(total - num_layers, total))
    adapter = Adapter("mlx", rank, scale, random_modules(layers, rank, scale, seed))
    write_mlx(adapter, dst, total_layers=total)
    return dst


def make_peft_adapter(
    dst: Path,
    *,
    rank: int = 4,
    alpha: float | None = None,
    layers: tuple[int, ...] = (1, 3),
    rslora: bool = False,
    dtype: str = "float32",
    adapter_name: bool = False,
    seed: int = 1,
) -> Path:
    """PEFT adapter as PEFT itself would save it (optionally BF16/F16, rslora, `.default.` keys)."""
    alpha = float(alpha if alpha is not None else 2 * rank)
    scale = alpha / (math.sqrt(rank) if rslora else rank)
    mods = random_modules(list(layers), rank, scale, seed)
    mid = ".default" if adapter_name else ""
    tensors: dict[str, np.ndarray] = {}
    for name, m in mods.items():
        tensors[f"{PREFIX}{name}.lora_A{mid}.weight"] = m.A
        tensors[f"{PREFIX}{name}.lora_B{mid}.weight"] = m.B
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / PEFT_WEIGHTS
    if dtype == "bfloat16":
        save_bf16(path, tensors, {"format": "pt"})
    elif dtype == "float16":
        save_file({k: v.astype(np.float16) for k, v in tensors.items()}, str(path), metadata={"format": "pt"})
    else:
        save_file(tensors, str(path), metadata={"format": "pt"})
    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "tiny-llama",
        "r": rank,
        "lora_alpha": int(alpha) if alpha.is_integer() else alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": rslora,
        "use_dora": False,
        "target_modules": sorted({k.split(".")[-1] for k in DIMS}),
        "layers_to_transform": list(layers),
        "layers_pattern": "layers",
        "modules_to_save": None,
    }
    (dst / CONFIG_NAME).write_text(json.dumps(cfg, indent=2) + "\n")
    return dst


def make_trap(kind: str, mlx_dir: Path, dst: Path) -> Path:
    """Convert ``mlx_dir`` to PEFT, then break it in one well-known way."""
    write_peft(load_adapter(mlx_dir), dst)
    weights = dst / PEFT_WEIGHTS
    tensors = load_file(str(weights))
    cfg_path = dst / CONFIG_NAME
    cfg = json.loads(cfg_path.read_text())
    if kind == "no-prefix":
        tensors = {k[len(PREFIX):]: v for k, v in tensors.items()}
    elif kind == "alpha-not-times-r":
        cfg["lora_alpha"] = cfg["lora_alpha"] // cfg["r"]  # wrote scale instead of scale x rank
    elif kind == "ab-swapped":
        # square modules keep legal shapes after "swap A and B, transpose both"
        for key in [k for k in tensors if (".q_proj." in k or ".o_proj." in k) and k.endswith("lora_A.weight")]:
            other = key.replace("lora_A", "lora_B")
            a, b = tensors[key], tensors[other]
            tensors[key], tensors[other] = np.ascontiguousarray(b.T), np.ascontiguousarray(a.T)
    elif kind == "mlp-missing":
        tensors = {k: v for k, v in tensors.items() if ".mlp." not in k}
    elif kind == EXTRA_TRAP:
        tensors = {k: (np.ascontiguousarray(v.T) if "lora_A" in k else v) for k, v in tensors.items()}
    else:
        raise ValueError(f"unknown trap {kind!r}")
    save_file(tensors, str(weights), metadata={"format": "pt"})
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return dst

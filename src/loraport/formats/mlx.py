"""MLX (mlx-lm) adapter reader and writer.

Layout (mlx_lm/tuner/lora.py): ``{name}.lora_a (in, r)``, ``{name}.lora_b (r, out)``,
``dW = scale * lora_b.T @ lora_a.T``; ``adapter_config.json`` holds
``lora_parameters {rank, scale, dropout, keys}`` and ``num_layers`` (counted
from the last block).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..errors import AdapterFormatError
from ..model import Adapter, block_relative, layer_of, number
from ..st_io import SafetensorsFile, save_float32
from .common import CONFIG_NAME, pair_modules, read_json

WEIGHTS_NAME = "adapters.safetensors"
DEFAULT_SCALE = 20.0  # mlx-lm default
_KEY = re.compile(r"^(?P<name>.+)\.lora_(?P<leaf>a|b)$")


def read_mlx(path: Path) -> Adapter:
    cfg = read_json(path / CONFIG_NAME)
    st = SafetensorsFile(path / WEIGHTS_NAME)
    warnings: list[str] = []
    unsupported: list[str] = []
    issues: list[str] = []

    fine_tune = cfg.get("fine_tune_type", "lora")
    if fine_tune != "lora":
        unsupported.append(f"fine_tune_type={fine_tune!r} (only 'lora' adapters can be converted)")
    lp = cfg.get("lora_parameters")
    if not isinstance(lp, dict):
        raise AdapterFormatError(f"{path / CONFIG_NAME}: missing 'lora_parameters' object (is this an MLX adapter?)")
    if "scale" not in lp:
        if "alpha" in lp:
            raise AdapterFormatError(
                f"{path / CONFIG_NAME}: lora_parameters has 'alpha' but mlx-lm expects 'scale' "
                "(scale = alpha / rank); fix the config before converting"
            )
        warnings.append(f"lora_parameters.scale missing; assuming the mlx-lm default {DEFAULT_SCALE}")
    scale = float(lp.get("scale", DEFAULT_SCALE))
    rank = int(lp.get("rank", 8))

    a_keys: dict[str, str] = {}
    b_keys: dict[str, str] = {}
    for key, info in st.tensors.items():
        m = _KEY.match(key)
        if m:
            if len(info.shape) != 2:
                unsupported.append(
                    f"{key}: {len(info.shape)}-D tensor {info.shape} (MoE/SwitchLinear LoRA is not supported yet)"
                )
                continue
            (a_keys if m["leaf"] == "a" else b_keys)[m["name"]] = key
        elif key.endswith(".m"):
            unsupported.append(f"{key}: DoRA magnitude tensor (DoRA is not supported)")
        else:
            issues.append(f"{key}: not a lora_a/lora_b tensor, it would be dropped by conversion")
    modules = pair_modules(st, a_keys, b_keys, scale, True, issues, "lora_a", "lora_b")

    meta = {
        "tensor_count": len(st.tensors),
        "dtypes": sorted({i.dtype for i in st.tensors.values()}),
        "nbytes": st.nbytes,
        "num_layers": cfg.get("num_layers"),
        "keys": lp.get("keys"),
    }
    return Adapter("mlx", rank, scale, modules, issues, unsupported, warnings, meta)


def write_mlx(adapter: Adapter, dst: Path, *, total_layers: int | None = None, base_model_id: str | None = None) -> Path:
    """Write ``adapter`` in MLX layout; returns the destination folder."""
    names = sorted(adapter.modules)
    layers = []
    keys = set()
    for name in names:
        rel, layer = block_relative(name), layer_of(name)
        if rel is None or layer is None:
            raise AdapterFormatError(
                f"module {name!r} has no 'layers.<n>.' part; mlx-lm adapters can only address transformer blocks"
            )
        layers.append(layer)
        keys.add(rel)
    if total_layers is not None and max(layers) >= total_layers:
        raise AdapterFormatError(
            f"adapter touches layer {max(layers)} but the base model has only {total_layers} layers"
        )
    # mlx-lm counts blocks from the end; blocks without tensors stay at dW=0 (load_adapters uses strict=False)
    num_layers = total_layers - min(layers) if total_layers is not None else -1

    tensors: dict[str, np.ndarray] = {}
    for name in names:
        mod = adapter.modules[name]
        tensors[f"{name}.lora_a"] = mod.A.T
        tensors[f"{name}.lora_b"] = mod.B.T
    dst.mkdir(parents=True, exist_ok=True)
    save_float32(dst / WEIGHTS_NAME, tensors)
    cfg: dict[str, object] = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": adapter.rank,
            "scale": number(adapter.scale),
            "dropout": 0.0,
            "keys": sorted(keys),
        },
    }
    if base_model_id:
        cfg["model"] = base_model_id
    (dst / CONFIG_NAME).write_text(json.dumps(cfg, indent=4, sort_keys=True) + "\n", encoding="utf-8")
    return dst

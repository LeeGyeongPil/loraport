"""PEFT adapter reader and writer.

Layout: ``base_model.model.{name}.lora_A.weight (r, in)`` and
``...lora_B.weight (out, r)``; ``dW = (lora_alpha / r) * B @ A`` (``/ sqrt(r)``
with ``use_rslora``).  Keys written by in-memory state dicts may carry the
adapter name (``lora_A.default.weight``); both spellings are accepted.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from ..errors import AdapterFormatError
from ..model import Adapter, layer_of, number, terminal
from ..st_io import SafetensorsFile, save_float32
from .common import CONFIG_NAME, pair_modules, read_json

WEIGHTS_NAME = "adapter_model.safetensors"
PREFIX = "base_model.model."
_KEY = re.compile(r"^base_model\.model\.(?P<name>.+?)\.lora_(?P<leaf>[AB])(?:\.[^.]+)?\.weight$")
_BARE = re.compile(r"^(?P<name>.+?)\.lora_(?P<leaf>[AB])(?:\.[^.]+)?\.weight$")


def _unsupported_config(cfg: dict) -> list[str]:
    reasons = []
    if cfg.get("use_dora"):
        reasons.append("use_dora=true (DoRA is not supported)")
    if cfg.get("bias", "none") != "none":
        reasons.append(f"bias={cfg['bias']!r} (bias training is not supported)")
    if cfg.get("lora_bias"):
        reasons.append("lora_bias=true (not supported)")
    if cfg.get("modules_to_save"):
        reasons.append(f"modules_to_save={cfg['modules_to_save']!r} (fully trained modules are not supported)")
    if cfg.get("rank_pattern"):
        reasons.append("rank_pattern is set (per-module ranks are not supported yet)")
    if cfg.get("alpha_pattern"):
        reasons.append("alpha_pattern is set (per-module alpha is not supported yet)")
    return reasons


def read_peft(path: Path) -> Adapter:
    cfg = read_json(path / CONFIG_NAME)
    if str(cfg.get("peft_type", "")).upper() != "LORA":
        raise AdapterFormatError(f"{path / CONFIG_NAME}: peft_type is {cfg.get('peft_type')!r}, expected 'LORA'")
    st = SafetensorsFile(path / WEIGHTS_NAME)
    warnings: list[str] = []
    unsupported = _unsupported_config(cfg)
    issues: list[str] = []

    if "r" not in cfg or "lora_alpha" not in cfg:
        warnings.append("r/lora_alpha missing in adapter_config.json; using PEFT defaults (8)")
    rank = int(cfg.get("r", 8))
    alpha = float(cfg.get("lora_alpha", 8))
    if rank <= 0:
        raise AdapterFormatError(f"{path / CONFIG_NAME}: r must be positive, got {rank}")
    rslora = bool(cfg.get("use_rslora"))
    scale = alpha / (math.sqrt(rank) if rslora else rank)

    a_keys: dict[str, str] = {}
    b_keys: dict[str, str] = {}
    for key in st.tensors:
        if "lora_embedding_" in key:
            unsupported.append(f"{key}: embedding LoRA is not supported")
        elif "lora_magnitude_vector" in key:
            unsupported.append(f"{key}: DoRA magnitude vector (DoRA is not supported)")
        elif (m := _KEY.match(key)) is not None:
            target = a_keys if m["leaf"] == "A" else b_keys
            if m["name"] in target:
                issues.append(f"{key}: duplicate of {target[m['name']]} (same module and leaf)")
            else:
                target[m["name"]] = key
        elif _BARE.match(key):
            issues.append(
                f"{key}: PEFT ignores it because the key lacks the 'base_model.model.' prefix "
                "(the adapter would load with dW=0)"
            )
        else:
            issues.append(f"{key}: not a lora_A/lora_B weight, it would be dropped by conversion")
    modules = pair_modules(st, a_keys, b_keys, scale, False, issues, "lora_A", "lora_B")

    meta = {
        "tensor_count": len(st.tensors),
        "dtypes": sorted({i.dtype for i in st.tensors.values()}),
        "nbytes": st.nbytes,
        "lora_alpha": number(alpha),
        "use_rslora": rslora,
        "layers_to_transform": cfg.get("layers_to_transform"),
        "target_modules": cfg.get("target_modules"),
    }
    return Adapter("peft", rank, scale, modules, issues, unsupported, warnings, meta)


def write_peft(adapter: Adapter, dst: Path, *, base_model_id: str | None = None) -> Path:
    """Write ``adapter`` in PEFT layout (float32, ``use_rslora=false``)."""
    tensors = {}
    for name in sorted(adapter.modules):
        mod = adapter.modules[name]
        tensors[f"{PREFIX}{name}.lora_A.weight"] = mod.A
        tensors[f"{PREFIX}{name}.lora_B.weight"] = mod.B
    dst.mkdir(parents=True, exist_ok=True)
    save_float32(dst / WEIGHTS_NAME, tensors, metadata={"format": "pt"})
    layers = adapter.layers
    cfg: dict[str, object] = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model_id,
        "r": adapter.rank,
        "lora_alpha": number(round(adapter.scale * adapter.rank, 10)),
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "use_rslora": False,
        "use_dora": False,
        "inference_mode": True,
        "target_modules": sorted({terminal(n) for n in adapter.modules}),
        "modules_to_save": None,
    }
    if layers and all(layer_of(n) is not None for n in adapter.modules):
        cfg["layers_to_transform"] = layers
        cfg["layers_pattern"] = "layers"
    (dst / CONFIG_NAME).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return dst

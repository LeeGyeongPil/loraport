"""Helpers shared by the MLX and PEFT readers/writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import AdapterFormatError
from ..model import LoraModule
from ..st_io import SafetensorsFile

CONFIG_NAME = "adapter_config.json"


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise AdapterFormatError(f"{path}: file not found") from e
    except (OSError, json.JSONDecodeError) as e:
        raise AdapterFormatError(f"{path}: cannot parse JSON ({e})") from e
    if not isinstance(data, dict):
        raise AdapterFormatError(f"{path}: expected a JSON object")
    return data


def pair_modules(
    st: SafetensorsFile,
    a_keys: dict[str, str],
    b_keys: dict[str, str],
    scale: float,
    transpose: bool,
    issues: list[str],
    a_leaf: str,
    b_leaf: str,
) -> dict[str, LoraModule]:
    """Read A/B tensors that have a partner; report the ones that do not.

    ``a_keys``/``b_keys`` map module name -> tensor key.  When ``transpose`` is
    true the tensors are in MLX layout and are transposed to ``(r,in)``/``(out,r)``.
    """
    modules: dict[str, LoraModule] = {}
    for name in sorted(a_keys.keys() - b_keys.keys()):
        issues.append(f"{a_keys[name]}: no matching {b_leaf} tensor, module {name!r} is incomplete and was skipped")
    for name in sorted(b_keys.keys() - a_keys.keys()):
        issues.append(f"{b_keys[name]}: no matching {a_leaf} tensor, module {name!r} is incomplete and was skipped")
    for name in sorted(a_keys.keys() & b_keys.keys()):
        a = st.read(a_keys[name])
        b = st.read(b_keys[name])
        if transpose:
            a, b = a.T, b.T
        modules[name] = LoraModule(np.ascontiguousarray(a), np.ascontiguousarray(b), scale)
    return modules

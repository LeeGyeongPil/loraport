"""Fixtures: every adapter is generated at test time (no network, no model files)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from loraport.sample import make_mlx_adapter, write_base_config


@pytest.fixture
def mlx_dir(tmp_path: Path) -> Path:
    return make_mlx_adapter(tmp_path / "mlx", rank=4, scale=20.0, num_layers=2, seed=0)


@pytest.fixture
def base_cfg(tmp_path: Path) -> Path:
    return write_base_config(tmp_path / "base")


def edit_adapter(path: Path, weights_name: str, *, cfg: dict | None = None, add: dict | None = None,
                 drop: tuple[str, ...] = ()) -> None:
    """Patch an adapter folder in place: update config fields, add or drop tensors."""
    if cfg:
        p = path / "adapter_config.json"
        data = json.loads(p.read_text())
        data.update(cfg)
        p.write_text(json.dumps(data))
    if add or drop:
        w = path / weights_name
        tensors = load_file(str(w))
        tensors = {k: v for k, v in tensors.items() if not any(d in k for d in drop)}
        tensors.update({k: np.asarray(v, dtype=np.float32) for k, v in (add or {}).items()})
        save_file(tensors, str(w), metadata={"format": "pt"})

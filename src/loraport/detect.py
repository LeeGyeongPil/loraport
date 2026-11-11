"""Decide whether a folder holds an MLX or a PEFT adapter and load it."""

from __future__ import annotations

from pathlib import Path

from .errors import AdapterFormatError
from .formats.mlx import WEIGHTS_NAME as MLX_WEIGHTS
from .formats.mlx import read_mlx
from .formats.peft import WEIGHTS_NAME as PEFT_WEIGHTS
from .formats.peft import read_peft
from .model import Adapter


def detect_format(path: Path) -> str:
    """Return ``"mlx"`` or ``"peft"`` based on the weight file name in ``path``."""
    if not path.is_dir():
        raise AdapterFormatError(f"{path}: not a directory (pass the adapter folder, not a file)")
    has_mlx = (path / MLX_WEIGHTS).is_file()
    has_peft = (path / PEFT_WEIGHTS).is_file()
    if has_mlx and has_peft:
        raise AdapterFormatError(
            f"{path}: contains both {MLX_WEIGHTS} and {PEFT_WEIGHTS}; keep one adapter per folder"
        )
    if has_mlx:
        return "mlx"
    if has_peft:
        return "peft"
    raise AdapterFormatError(
        f"{path}: no adapter found; expected {MLX_WEIGHTS} (mlx-lm) or {PEFT_WEIGHTS} (PEFT) "
        "together with adapter_config.json"
    )


def load_adapter(path: str | Path) -> Adapter:
    """Load an adapter folder in either format."""
    p = Path(path)
    return read_mlx(p) if detect_format(p) == "mlx" else read_peft(p)

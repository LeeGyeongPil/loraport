"""loraport: convert LoRA adapters between MLX (mlx-lm) and PEFT and verify dW."""

from __future__ import annotations

__version__ = "0.1.1"

from .convert import ConvertResult, convert  # noqa: E402
from .delta import Report, compare  # noqa: E402
from .detect import load_adapter  # noqa: E402
from .errors import AdapterFormatError, BaseConfigError, LoraportError, UnsupportedAdapterError  # noqa: E402
from .formats.mlx import write_mlx as save_mlx  # noqa: E402
from .formats.peft import write_peft as save_peft  # noqa: E402
from .model import Adapter, LoraModule  # noqa: E402

__all__ = [
    "Adapter",
    "AdapterFormatError",
    "BaseConfigError",
    "ConvertResult",
    "LoraModule",
    "LoraportError",
    "Report",
    "UnsupportedAdapterError",
    "__version__",
    "compare",
    "convert",
    "load_adapter",
    "save_mlx",
    "save_peft",
]

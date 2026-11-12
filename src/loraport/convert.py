"""Convert between MLX and PEFT adapter folders, then verify the result."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .basecfg import check_adapter, load_base_config
from .delta import Report, compare
from .detect import load_adapter
from .errors import AdapterFormatError, BaseConfigError, LoraportError, UnsupportedAdapterError
from .formats.common import CONFIG_NAME
from .formats.mlx import WEIGHTS_NAME as MLX_WEIGHTS
from .formats.mlx import write_mlx
from .formats.peft import WEIGHTS_NAME as PEFT_WEIGHTS
from .formats.peft import write_peft
from .model import Adapter

Target = Literal["peft", "mlx"]


@dataclass
class ConvertResult:
    dst: Path
    adapter: Adapter  # the source adapter as read
    report: Report | None  # None when verification was skipped
    next_steps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # source tensors that did not make it into the output

    @property
    def ok(self) -> bool:
        """True only if nothing was dropped and (when verified) the report passed."""
        return not self.dropped and (self.report is None or self.report.ok)


def ensure_supported(adapter: Adapter) -> None:
    """Raise ``UnsupportedAdapterError`` if the adapter uses a feature we refuse to convert."""
    if adapter.unsupported:
        raise UnsupportedAdapterError(adapter.unsupported)


def _validate(adapter: Adapter) -> None:
    ensure_supported(adapter)
    if not adapter.modules:
        raise AdapterFormatError("no LoRA modules found in the adapter; nothing to convert")
    bad = []
    for name, mod in sorted(adapter.modules.items()):
        if not mod.consistent:
            bad.append(f"{name}: A{tuple(mod.A.shape)} and B{tuple(mod.B.shape)} do not share the rank dimension")
        elif mod.rank != adapter.rank:
            bad.append(f"{name}: tensor rank {mod.rank} differs from the configured rank {adapter.rank}")
    if bad:
        raise AdapterFormatError(
            "inconsistent adapter, refusing to convert (was lora_A/lora_B transposed or the config edited?): "
            + "; ".join(bad[:5])
            + (f" (+{len(bad) - 5} more)" if len(bad) > 5 else "")
        )


def _next_steps(to: Target, dst: Path, base: str | None, num_layers_unknown: bool) -> list[str]:
    base_ref = base or "<BASE_MODEL>"
    if to == "peft":
        return [
            f"vLLM:      vllm serve {base_ref} --enable-lora --lora-modules my-adapter={dst}",
            f"llama.cpp: python convert_lora_to_gguf.py --base <BASE_MODEL_DIR> --outfile adapter.gguf {dst}",
        ]
    steps = [f'mlx-lm:    mlx_lm.generate --model {base_ref} --adapter-path {dst} --prompt "Hello"']
    if num_layers_unknown:
        steps.append("num_layers=-1 was written (no --base-config); blocks without tensors keep dW=0.")
    return steps


def convert(
    src: str | Path,
    dst: str | Path,
    to: Target,
    *,
    base_config: str | Path | None = None,
    base_model_id: str | None = None,
    verify: bool = True,
    rtol: float = 1e-5,
    force: bool = False,
) -> ConvertResult:
    """Convert the adapter in ``src`` to ``to`` ("peft" or "mlx") inside ``dst``.

    Raises ``UnsupportedAdapterError`` / ``AdapterFormatError`` / ``BaseConfigError``
    before writing anything.  Tensors the converter could not carry over are
    never silent: they show up in ``ConvertResult.dropped`` and in the report.
    """
    if to not in ("peft", "mlx"):
        raise LoraportError(f"unknown target format {to!r}; use 'peft' or 'mlx'")
    adapter = load_adapter(src)
    _validate(adapter)
    if adapter.format == to:
        raise AdapterFormatError(
            f"{src} is already a {to} adapter; convert to the other format "
            f"(--to {'mlx' if to == 'peft' else 'peft'}) or use `loraport verify`"
        )
    total_layers = None
    if base_config is not None:
        dims = load_base_config(base_config)
        problems = check_adapter(adapter, dims)
        if problems:
            raise BaseConfigError(
                "adapter does not fit the base model config:\n  " + "\n  ".join(problems), problems
            )
        total_layers = dims.num_layers

    out = Path(dst)
    if not force and any((out / n).exists() for n in (PEFT_WEIGHTS, MLX_WEIGHTS, CONFIG_NAME)):
        raise LoraportError(f"{out} already contains adapter files; pass --force to overwrite them")

    if to == "peft":
        write_peft(adapter, out, base_model_id=base_model_id)
    else:
        write_mlx(adapter, out, total_layers=total_layers, base_model_id=base_model_id)

    result_adapter = load_adapter(out)
    dropped = list(adapter.key_issues)
    report = compare(adapter, result_adapter, rtol=rtol) if verify else None
    notes = list(adapter.warnings)
    return ConvertResult(
        out, adapter, report, _next_steps(to, out, base_model_id, to == "mlx" and total_layers is None), notes, dropped
    )

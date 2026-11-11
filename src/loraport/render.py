"""Human-readable and JSON rendering of adapters and reports."""

from __future__ import annotations

from typing import Any

from . import __version__
from .delta import Report
from .model import Adapter, block_relative, number

_LIMIT = 12


def _lines(title: str, items: list[str]) -> list[str]:
    if not items:
        return []
    out = [f"  {title} ({len(items)}):"]
    out += [f"    - {s}" for s in items[:_LIMIT]]
    if len(items) > _LIMIT:
        out.append(f"    ... and {len(items) - _LIMIT} more (use --json for the full list)")
    return out


def render_report(report: Report, label_a: str = "a", label_b: str = "b") -> str:
    """Multi-line verification report with a fix hint per problem class."""
    out = [f"{report.headline()}   [{label_a} vs {label_b}, rtol {report.rtol:g}]"]
    if report.n_reference == 0:
        out.append(f"  {label_a} contains no LoRA modules; nothing to compare")
    out += _lines(f"modules in {label_a} but not in {label_b}", report.missing)
    if report.missing:
        out.append(f"    hint: {label_b} would run these layers without LoRA (dW = 0); re-run the conversion")
    out += _lines(f"modules only in {label_b}", report.extra)
    out += _lines("shape mismatch", report.shape_mismatch)
    if report.shape_mismatch:
        out.append("    hint: A must be (r, in) and B (out, r); check that nothing was transposed twice or not at all")
    out += _lines("ignored or dropped tensors", report.key_issues)
    bad = [f"{n}: rel err {report.per_module[n]:.3g}" for n in report.bad_modules]
    out += _lines("dW differs", bad)
    if bad:
        out.append("    hint: check lora_alpha = scale x rank, lora_A/lora_B order, and rslora")
    return "\n".join(out)


def adapter_summary(adapter: Adapter, path: str) -> dict[str, Any]:
    """Facts shown by ``inspect`` (also the ``--json`` payload)."""
    kinds: dict[str, int] = {}
    for name in adapter.modules:
        rel = block_relative(name) or name
        kinds[rel] = kinds.get(rel, 0) + 1
    meta = adapter.source_meta
    return {
        "path": path,
        "loraport_version": __version__,
        "format": adapter.format,
        "rank": adapter.rank,
        "scale": number(adapter.scale),
        "lora_alpha": adapter.lora_alpha,
        "use_rslora": bool(meta.get("use_rslora", False)),
        "layers": adapter.layers,
        "modules": sorted(adapter.modules),
        "module_kinds": dict(sorted(kinds.items())),
        "tensor_count": meta.get("tensor_count", 0),
        "dtypes": meta.get("dtypes", []),
        "bytes": meta.get("nbytes", 0),
        "key_issues": adapter.key_issues,
        "unsupported": adapter.unsupported,
        "warnings": adapter.warnings,
    }


def render_inspect(adapter: Adapter, path: str) -> str:
    s = adapter_summary(adapter, path)
    layers = s["layers"]
    span = f"{layers[0]}..{layers[-1]} ({len(layers)} blocks)" if layers else "none"
    if adapter.format == "mlx":
        scale_line = f"scale {s['scale']} x rank {s['rank']} = PEFT lora_alpha {s['lora_alpha']}"
    else:
        rs = ", use_rslora" if s["use_rslora"] else ""
        scale_line = f"lora_alpha {s['lora_alpha']} / rank {s['rank']}{rs} = MLX scale {s['scale']:g}"
    out = [
        f"{path}: {adapter.format.upper()} adapter",
        f"  {scale_line}",
        f"  layers: {span}",
        f"  modules: {len(s['modules'])} ({', '.join(f'{k} x{v}' for k, v in s['module_kinds'].items())})",
        f"  tensors: {s['tensor_count']} ({', '.join(s['dtypes'])}), {s['bytes']} bytes",
    ]
    out += _lines("unsupported by loraport", s["unsupported"])
    out += _lines("ignored or dropped tensors", s["key_issues"])
    out += _lines("warnings", s["warnings"])
    return "\n".join(out)

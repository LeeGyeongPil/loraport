"""Command line interface: inspect, convert, verify, demo.

Exit codes: 0 success / verification passed, 1 verification failed or tensors
were dropped, 2 invalid input or unsupported adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .convert import convert, ensure_supported
from .delta import compare
from .detect import load_adapter
from .errors import LoraportError, UnsupportedAdapterError
from .render import adapter_summary, render_inspect, render_report
from .sample import TRAP_NAMES, make_mlx_adapter, make_trap, write_base_config


def _cmd_inspect(args: argparse.Namespace) -> int:
    adapter = load_adapter(args.path)
    if args.json:
        print(json.dumps(adapter_summary(adapter, str(args.path)), indent=2))
    else:
        print(render_inspect(adapter, str(args.path)))
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    res = convert(
        args.src,
        args.dst,
        args.to,
        base_config=args.base_config,
        base_model_id=args.base_model_id,
        verify=not args.no_verify,
        rtol=args.rtol,
        force=args.force,
    )
    print(f"wrote {args.to} adapter to {res.dst}")
    for note in res.notes:
        print(f"note: {note}")
    if res.report is not None:
        print(render_report(res.report, "source", "converted"))
    else:
        print("verification skipped (--no-verify)")
        if res.dropped:
            print(f"  source tensors that were NOT carried over ({len(res.dropped)}):")
            for line in res.dropped[:12]:
                print(f"    - {line}")
    if res.ok:
        print("next steps:")
        for line in res.next_steps:
            print(f"  {line}")
        return 0
    print("conversion is NOT safe to use: see the problems above", file=sys.stderr)
    return 1


def _cmd_verify(args: argparse.Namespace) -> int:
    a, b = load_adapter(args.a), load_adapter(args.b)
    ensure_supported(a)
    ensure_supported(b)
    report = compare(a, b, rtol=args.rtol)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_report(report, str(args.a), str(args.b)))
    return 0 if report.ok else 1


def _headline(res_report) -> str:  # type: ignore[no-untyped-def]
    return res_report.headline() if res_report is not None else "not verified"


def _cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.out)
    print("loraport demo: synthetic practice adapters (mock), random numbers, not a trained model")
    mlx_dir = make_mlx_adapter(out / "mlx")
    base_cfg = write_base_config(out / "base")
    print(f"1. made a practice MLX adapter in {mlx_dir} (rank 4, scale 20, last 2 of 4 blocks)")

    # real conversion + real verification, both directions
    to_peft = convert(mlx_dir, out / "peft", "peft", force=True, base_config=base_cfg)
    print(f"2. MLX -> PEFT: {_headline(to_peft.report)}")
    back = convert(out / "peft", out / "mlx_roundtrip", "mlx", force=True, base_config=base_cfg)
    print(f"3. PEFT -> MLX: {_headline(back.report)}")

    print("4. common mistakes, each compared with the original MLX adapter:")
    reference = load_adapter(mlx_dir)
    caught = 0
    for kind in TRAP_NAMES:
        rep = compare(reference, load_adapter(make_trap(kind, mlx_dir, out / "traps" / kind)))
        caught += 0 if rep.ok else 1
        print(f"   {kind:<18} {'caught' if not rep.ok else 'MISSED'}: {rep.headline()}")
    print(f"{caught}/{len(TRAP_NAMES)} traps caught")
    return 0 if to_peft.ok and back.ok and caught == len(TRAP_NAMES) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loraport",
        description="Convert LoRA adapters between MLX (mlx-lm) and PEFT and verify the weight delta.",
    )
    p.add_argument("--version", action="version", version=f"loraport {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("inspect", help="show format, rank, scale and layers of an adapter folder")
    sp.add_argument("path", type=Path)
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.set_defaults(func=_cmd_inspect)

    sp = sub.add_parser("convert", help="convert an adapter to the other format and verify it")
    sp.add_argument("src", type=Path)
    sp.add_argument("dst", type=Path)
    sp.add_argument("--to", choices=("peft", "mlx"), required=True)
    sp.add_argument("--base-config", type=Path, help="base model config.json (or its folder) for shape/layer checks")
    sp.add_argument("--base-model-id", help="hub id written into the output config (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    sp.add_argument("--rtol", type=float, default=1e-5, help="max relative dW error (default 1e-5)")
    sp.add_argument("--no-verify", action="store_true", help="skip the automatic verification")
    sp.add_argument("--force", action="store_true", help="overwrite adapter files in dst")
    sp.set_defaults(func=_cmd_convert)

    sp = sub.add_parser("verify", help="compare the dW of two adapters (any formats)")
    sp.add_argument("a", type=Path, help="reference adapter")
    sp.add_argument("b", type=Path, help="adapter to check")
    sp.add_argument("--rtol", type=float, default=1e-5)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_verify)

    sp = sub.add_parser("demo", help="run the whole flow on synthetic practice adapters (mock)")
    sp.add_argument("--out", type=Path, default=Path("loraport-demo"))
    sp.set_defaults(func=_cmd_demo)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except UnsupportedAdapterError as e:
        print("error: refusing to convert an unsupported adapter:", file=sys.stderr)
        for reason in e.reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 2
    except LoraportError as e:  # includes BaseConfigError, AdapterFormatError
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

"""CLI behaviour and the demo (AC-1, AC-9)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from loraport.cli import main
from loraport.sample import make_peft_adapter


def test_demo_runs_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """AC-1."""
    assert main(["demo", "--out", str(tmp_path / "demo")]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("loraport demo: synthetic practice adapters (mock)")
    assert out.count("PASS") == 2
    assert "4/4 traps caught" in out and "MISSED" not in out


def test_demo_creates_dora_folder_that_convert_refuses(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    demo = tmp_path / "demo"
    assert main(["demo", "--out", str(demo)]) == 0
    assert (demo / "dora").is_dir()
    capsys.readouterr()
    assert main(["convert", str(demo / "dora"), str(demo / "x"), "--to", "peft"]) == 2
    assert "fine_tune_type='dora'" in capsys.readouterr().err
    assert not (demo / "x").exists()


def test_demo_via_subprocess(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "loraport.cli", "demo", "--out", str(tmp_path / "d")],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "4/4 traps caught" in proc.stdout


def test_inspect_json(mlx_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", str(mlx_dir), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["format"] == "mlx" and data["rank"] == 4 and data["scale"] == 20
    assert data["lora_alpha"] == 80 and data["layers"] == [2, 3]
    assert len(data["modules"]) == 14 and data["tensor_count"] == 28
    assert data["key_issues"] == [] and data["unsupported"] == []


def test_inspect_text_peft_rslora(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = make_peft_adapter(tmp_path / "p", rank=16, alpha=32, rslora=True)
    assert main(["inspect", str(d)]) == 0
    out = capsys.readouterr().out
    assert "PEFT adapter" in out and "use_rslora" in out and "MLX scale 8" in out


def test_convert_autoverify_and_next_steps(mlx_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dst = tmp_path / "peft"
    assert main(["convert", str(mlx_dir), str(dst), "--to", "peft", "--base-model-id", "org/model"]) == 0
    out = capsys.readouterr().out
    assert "PASS 14/14 modules" in out
    assert "--lora-modules" in out and "vllm serve org/model --enable-lora" in out
    assert "convert_lora_to_gguf.py" in out and "lora-preflight" in out


def test_convert_failure_exit_code(mlx_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from conftest import edit_adapter
    import numpy as np

    edit_adapter(mlx_dir, "adapters.safetensors", add={"model.extra.weight": np.zeros(3)})
    assert main(["convert", str(mlx_dir), str(tmp_path / "o"), "--to", "peft"]) == 1
    captured = capsys.readouterr()
    assert "FAIL" in captured.out and "next steps" not in captured.out
    assert "NOT safe" in captured.err


def test_input_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", str(tmp_path / "missing")]) == 2
    assert "error:" in capsys.readouterr().err
    assert main(["verify", str(tmp_path), str(tmp_path)]) == 2


def test_argparse_errors_exit_2() -> None:
    with pytest.raises(SystemExit) as e:
        main(["convert", "a", "b"])  # --to is required
    assert e.value.code == 2


def test_missing_path_exit_2_no_such_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", str(tmp_path / "missing")]) == 2
    assert "no such directory" in capsys.readouterr().err

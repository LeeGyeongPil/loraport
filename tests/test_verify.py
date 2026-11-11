"""verify catches the classic mistakes (AC-4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from loraport import compare, convert, load_adapter
from loraport.cli import main
from loraport.sample import EXTRA_TRAP, TRAP_NAMES, make_trap


def test_good_conversion_passes(mlx_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    convert(mlx_dir, tmp_path / "peft", "peft")
    assert main(["verify", str(mlx_dir), str(tmp_path / "peft")]) == 0
    assert capsys.readouterr().out.startswith("PASS 14/14")


@pytest.mark.parametrize("kind", TRAP_NAMES)
def test_traps_detected(kind: str, mlx_dir: Path, tmp_path: Path) -> None:
    trap = make_trap(kind, mlx_dir, tmp_path / kind)
    rep = compare(load_adapter(mlx_dir), load_adapter(trap))
    assert not rep.ok
    assert main(["verify", str(mlx_dir), str(trap)]) == 1  # non-zero exit for CI
    if kind == "no-prefix":
        assert len(rep.missing) == 14 and rep.key_issues  # PEFT would load this with dW = 0
    elif kind == "alpha-not-times-r":
        assert rep.worst_rel_err == pytest.approx(0.75, abs=1e-6) and rep.worst_rel_err >= 0.5  # 1 - 1/r
    elif kind == "ab-swapped":
        assert rep.worst_rel_err >= 0.1
        assert all(("q_proj" in n or "o_proj" in n) for n in rep.bad_modules) and len(rep.bad_modules) == 4
        assert not rep.shape_mismatch  # shapes are legal on square modules; only dW reveals it
    elif kind == "mlp-missing":
        assert len(rep.missing) == 6 and all(".mlp." in n for n in rep.missing)


def test_untransposed_shape_mismatch(mlx_dir: Path, tmp_path: Path) -> None:
    trap = make_trap(EXTRA_TRAP, mlx_dir, tmp_path / "t")
    rep = compare(load_adapter(mlx_dir), load_adapter(trap))
    assert not rep.ok and len(rep.shape_mismatch) == 14
    assert "(4, 64)" in rep.shape_mismatch[0] or "(4, 128)" in rep.shape_mismatch[0]
    assert main(["verify", str(mlx_dir), str(trap)]) == 1


def test_json_report_is_valid(mlx_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    trap = make_trap("mlp-missing", mlx_dir, tmp_path / "t")
    assert main(["verify", str(mlx_dir), str(trap), "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is False and len(data["missing"]) == 6 and data["modules_in_reference"] == 14

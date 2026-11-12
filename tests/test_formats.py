"""Format detection, reading, dtypes and malformed inputs (AC-7 and friends)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from loraport import AdapterFormatError, load_adapter
from loraport.detect import detect_format
from loraport.sample import make_peft_adapter
from loraport.st_io import SafetensorsFile, save_bf16


def test_detect_and_read_mlx(mlx_dir: Path) -> None:
    assert detect_format(mlx_dir) == "mlx"
    ad = load_adapter(mlx_dir)
    assert (ad.format, ad.rank, ad.scale) == ("mlx", 4, 20.0)
    assert len(ad.modules) == 14 and ad.layers == [2, 3]
    mod = ad.modules["model.layers.2.mlp.down_proj"]
    assert mod.A.shape == (4, 128) and mod.B.shape == (64, 4)  # canonical (r,in)/(out,r)
    assert not ad.key_issues and ad.lora_alpha == 80


def test_detect_peft_and_rslora_scale(tmp_path: Path) -> None:
    d = make_peft_adapter(tmp_path / "p", rank=16, alpha=32, rslora=True)
    ad = load_adapter(d)
    assert detect_format(d) == "peft" and ad.format == "peft"
    assert ad.scale == pytest.approx(32 / 4)  # alpha / sqrt(r)
    assert ad.layers == [1, 3]


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_bf16_peft_input(tmp_path: Path, dtype: str) -> None:
    """AC-7: half-precision PEFT adapters load without torch and convert."""
    from loraport import convert

    d = make_peft_adapter(tmp_path / "p", dtype=dtype)
    ad = load_adapter(d)
    assert ad.source_meta["dtypes"] == [{"bfloat16": "BF16", "float16": "F16"}[dtype]]
    ref = make_peft_adapter(tmp_path / "ref", dtype="float32")
    a32, ahalf = load_adapter(ref), ad
    name = "model.layers.1.self_attn.q_proj"
    tol = 2 ** -7 if dtype == "bfloat16" else 2 ** -10  # half-precision rounding of the same numbers
    assert np.allclose(a32.modules[name].A, ahalf.modules[name].A, rtol=tol, atol=1e-6)
    res = convert(d, tmp_path / "mlx", "mlx")
    assert res.ok and res.report is not None and res.report.worst_rel_err < 1e-6


def test_bf16_widening_is_lossless(tmp_path: Path) -> None:
    x = np.random.default_rng(3).normal(size=(5, 7)).astype(np.float32)
    path = tmp_path / "t.safetensors"
    save_bf16(path, {"w": x}, {"format": "pt"})
    got = SafetensorsFile(path).read("w")
    bits = x.view(np.uint32)
    expect = (((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)
    assert got.dtype == np.float32 and np.array_equal(got, expect)  # exact widening of the bf16 bits
    assert np.allclose(got, x, rtol=2 ** -7)


def test_reader_matches_safetensors_library(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    t = {"a": rng.normal(size=(3, 4)).astype(np.float32), "b": rng.normal(size=(2,)).astype(np.float16)}
    path = tmp_path / "x.safetensors"
    save_file(t, str(path))
    ours = SafetensorsFile(path)
    lib = load_file(str(path))
    assert np.array_equal(ours.read("a"), lib["a"])
    assert np.array_equal(ours.read("b"), lib["b"].astype(np.float32))


def test_default_adapter_name_in_keys(tmp_path: Path) -> None:
    """AC-7: keys like ...lora_A.default.weight are ordinary modules."""
    plain = load_adapter(make_peft_adapter(tmp_path / "a", adapter_name=False))
    named = load_adapter(make_peft_adapter(tmp_path / "b", adapter_name=True))
    assert sorted(named.modules) == sorted(plain.modules) and not named.key_issues
    assert "model.layers.1.mlp.up_proj" in named.modules


def test_missing_prefix_is_reported_not_loaded(tmp_path: Path) -> None:
    from loraport.sample import make_mlx_adapter, make_trap

    m = make_mlx_adapter(tmp_path / "m")
    ad = load_adapter(make_trap("no-prefix", m, tmp_path / "t"))
    assert not ad.modules and len(ad.key_issues) == 28
    assert "base_model.model." in ad.key_issues[0]


def test_unpaired_tensor_is_reported(mlx_dir: Path) -> None:
    from conftest import edit_adapter

    edit_adapter(mlx_dir, "adapters.safetensors", drop=("layers.3.mlp.up_proj.lora_b",))
    ad = load_adapter(mlx_dir)
    assert len(ad.modules) == 13
    assert any("lora_b" in s and "up_proj" in s for s in ad.key_issues)


def test_empty_folder_is_not_an_adapter(tmp_path: Path) -> None:
    target = tmp_path / "nothing"
    target.mkdir()
    with pytest.raises(AdapterFormatError, match="no adapter found"):
        load_adapter(target)


def test_missing_path_says_no_such_directory(tmp_path: Path) -> None:
    with pytest.raises(AdapterFormatError, match="no such directory") as exc:
        load_adapter(tmp_path / "nothing")
    assert "not a directory" not in str(exc.value)
    assert "not a file" not in str(exc.value)


def test_file_path_keeps_not_a_directory_message(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x")
    with pytest.raises(AdapterFormatError, match="not a directory"):
        load_adapter(target)


def test_both_formats_in_one_folder_is_ambiguous(tmp_path: Path, mlx_dir: Path) -> None:
    peft = make_peft_adapter(tmp_path / "p")
    (mlx_dir / "adapter_model.safetensors").write_bytes((peft / "adapter_model.safetensors").read_bytes())
    with pytest.raises(AdapterFormatError, match="both"):
        load_adapter(mlx_dir)


def test_truncated_safetensors_is_an_error(mlx_dir: Path) -> None:
    w = mlx_dir / "adapters.safetensors"
    w.write_bytes(w.read_bytes()[:-100])
    with pytest.raises(AdapterFormatError, match="truncated|past the end"):
        load_adapter(mlx_dir)


def test_garbage_header_length(mlx_dir: Path) -> None:
    (mlx_dir / "adapters.safetensors").write_bytes(struct.pack("<Q", 10**12) + b"{}")
    with pytest.raises(AdapterFormatError, match="header"):
        load_adapter(mlx_dir)


def test_mlx_config_with_alpha_gets_a_fix_hint(mlx_dir: Path) -> None:
    p = mlx_dir / "adapter_config.json"
    cfg = json.loads(p.read_text())
    cfg["lora_parameters"] = {"rank": 4, "alpha": 16}
    p.write_text(json.dumps(cfg))
    with pytest.raises(AdapterFormatError, match="scale = alpha / rank"):
        load_adapter(mlx_dir)


def test_peft_type_must_be_lora(tmp_path: Path) -> None:
    from conftest import edit_adapter

    d = make_peft_adapter(tmp_path / "p")
    edit_adapter(d, "adapter_model.safetensors", cfg={"peft_type": "PREFIX_TUNING"})
    with pytest.raises(AdapterFormatError, match="peft_type"):
        load_adapter(d)

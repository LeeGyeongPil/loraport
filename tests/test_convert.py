"""Conversion correctness: keys, shapes, config, dW preservation (AC-2, AC-3, AC-8)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open

from loraport import AdapterFormatError, BaseConfigError, LoraportError, convert, load_adapter
from loraport.delta import dense_delta
from loraport.sample import make_peft_adapter


def test_mlx_to_peft_keys_shapes_config(mlx_dir: Path, tmp_path: Path) -> None:
    res = convert(mlx_dir, tmp_path / "peft", "peft")
    assert res.ok
    with safe_open(str(tmp_path / "peft" / "adapter_model.safetensors"), framework="np") as f:
        keys = set(f.keys())
        meta = f.metadata()
        a = f.get_tensor("base_model.model.model.layers.2.mlp.down_proj.lora_A.weight")
        b = f.get_tensor("base_model.model.model.layers.2.mlp.down_proj.lora_B.weight")
    assert meta == {"format": "pt"}
    assert len(keys) == 28
    for i in (2, 3):
        for mod in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            assert f"base_model.model.model.layers.{i}.{mod}.lora_A.weight" in keys
            assert f"base_model.model.model.layers.{i}.{mod}.lora_B.weight" in keys
    assert a.shape == (4, 128) and b.shape == (64, 4)  # (r,in) and (out,r)
    cfg = json.loads((tmp_path / "peft" / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA" and cfg["r"] == 4
    assert cfg["lora_alpha"] == 80  # scale 20 x rank 4
    assert cfg["use_rslora"] is False and cfg["use_dora"] is False
    assert cfg["layers_to_transform"] == [2, 3]
    assert cfg["target_modules"] == sorted(
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


def _max_dense_rel_err(a, b) -> float:
    worst = 0.0
    for name, ma in a.modules.items():
        d1, d2 = dense_delta(ma), dense_delta(b.modules[name])
        worst = max(worst, float(np.linalg.norm(d1 - d2) / np.linalg.norm(d1)))
    return worst


@pytest.mark.parametrize("rank", [1, 4, 16])
@pytest.mark.parametrize("scale", [0.5, 20.0])
def test_roundtrip_delta_preserved(rank: int, scale: float, tmp_path: Path, base_cfg: Path) -> None:
    from loraport.sample import make_mlx_adapter

    src = make_mlx_adapter(tmp_path / "mlx", rank=rank, scale=scale)
    to_peft = convert(src, tmp_path / "peft", "peft", base_config=base_cfg)
    back = convert(tmp_path / "peft", tmp_path / "mlx2", "mlx", base_config=base_cfg)
    assert to_peft.ok and back.ok
    orig = load_adapter(src)
    for other in (load_adapter(tmp_path / "peft"), load_adapter(tmp_path / "mlx2")):
        assert _max_dense_rel_err(orig, other) <= 1e-6  # independent dense check, not the low-rank code
    assert load_adapter(tmp_path / "mlx2").scale == pytest.approx(scale)


@pytest.mark.parametrize("rslora", [False, True])
@pytest.mark.parametrize("layers", [(1, 3), (0, 1, 2, 3), (2,)])
def test_peft_to_mlx_delta_preserved(rslora: bool, layers: tuple[int, ...], tmp_path: Path, base_cfg: Path) -> None:
    src = make_peft_adapter(tmp_path / "peft", rank=8, alpha=16, rslora=rslora, layers=layers)
    res = convert(src, tmp_path / "mlx", "mlx", base_config=base_cfg)
    assert res.ok and res.report is not None
    assert _max_dense_rel_err(load_adapter(src), load_adapter(tmp_path / "mlx")) <= 1e-6
    expected_scale = 16 / np.sqrt(8) if rslora else 16 / 8
    assert load_adapter(tmp_path / "mlx").scale == pytest.approx(expected_scale)


def test_num_layers_rule(tmp_path: Path, base_cfg: Path) -> None:
    """num_layers = L - min(layer) with a base config, -1 without."""
    src = make_peft_adapter(tmp_path / "peft", layers=(1, 3))
    convert(src, tmp_path / "with", "mlx", base_config=base_cfg)
    convert(src, tmp_path / "without", "mlx")
    with_cfg = json.loads((tmp_path / "with" / "adapter_config.json").read_text())
    without_cfg = json.loads((tmp_path / "without" / "adapter_config.json").read_text())
    assert with_cfg["num_layers"] == 4 - 1
    assert without_cfg["num_layers"] == -1
    assert with_cfg["lora_parameters"]["keys"] == sorted(
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
    res = convert(src, tmp_path / "again", "mlx")
    assert any("num_layers=-1" in s for s in res.next_steps)  # tells the user why


def test_config_fields_whitelisted(mlx_dir: Path, tmp_path: Path) -> None:
    """AC-8: training arguments (paths, wandb, ...) never reach the PEFT config."""
    p = mlx_dir / "adapter_config.json"
    cfg = json.loads(p.read_text())
    cfg.update({"model": "/srv/fixture/models/private-llama", "data": "/srv/fixture/data", "wandb": "my-project",
                "adapter_path": "/srv/fixture/adapters", "iters": 1000, "seed": 7})
    p.write_text(json.dumps(cfg))
    convert(mlx_dir, tmp_path / "plain", "peft")
    convert(mlx_dir, tmp_path / "named", "peft", base_model_id="Qwen/Qwen2.5-1.5B-Instruct")
    for d in ("plain", "named"):
        text = (tmp_path / d / "adapter_config.json").read_text()
        for leaked in ("/srv/fixture/", "wandb", "my-project", "private-llama", "iters", "adapter_path"):
            assert leaked not in text
    plain = json.loads((tmp_path / "plain" / "adapter_config.json").read_text())
    named = json.loads((tmp_path / "named" / "adapter_config.json").read_text())
    assert plain["base_model_name_or_path"] is None
    assert named["base_model_name_or_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
    allowed = {"peft_type", "task_type", "base_model_name_or_path", "r", "lora_alpha", "lora_dropout", "bias",
               "fan_in_fan_out", "use_rslora", "use_dora", "inference_mode", "target_modules", "modules_to_save",
               "layers_to_transform", "layers_pattern"}
    assert set(named) <= allowed
    # the MLX direction is clean too
    convert(tmp_path / "plain", tmp_path / "mlx_back", "mlx")
    assert "/srv/fixture/" not in (tmp_path / "mlx_back" / "adapter_config.json").read_text()


def test_dropped_tensor_is_never_silent(mlx_dir: Path, tmp_path: Path) -> None:
    from conftest import edit_adapter

    edit_adapter(mlx_dir, "adapters.safetensors", add={"model.embed_tokens.weight": np.zeros((4, 4))})
    res = convert(mlx_dir, tmp_path / "peft", "peft")
    assert not res.ok and res.dropped and res.report is not None and not res.report.ok
    assert any("embed_tokens" in s for s in res.report.key_issues)
    res2 = convert(mlx_dir, tmp_path / "peft2", "peft", verify=False)
    assert not res2.ok and res2.dropped  # even without verification


def test_incomplete_module_is_reported(mlx_dir: Path, tmp_path: Path) -> None:
    from conftest import edit_adapter

    edit_adapter(mlx_dir, "adapters.safetensors", drop=("layers.2.mlp.gate_proj.lora_a",))
    res = convert(mlx_dir, tmp_path / "peft", "peft")
    assert not res.ok and any("gate_proj" in s for s in res.dropped)


def test_refuses_to_overwrite_without_force(mlx_dir: Path, tmp_path: Path) -> None:
    convert(mlx_dir, tmp_path / "peft", "peft")
    with pytest.raises(LoraportError, match="--force"):
        convert(mlx_dir, tmp_path / "peft", "peft")
    assert convert(mlx_dir, tmp_path / "peft", "peft", force=True).ok


def test_same_format_target_rejected(mlx_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(AdapterFormatError, match="already a mlx adapter"):
        convert(mlx_dir, tmp_path / "out", "mlx")


def test_inconsistent_ranks_rejected(tmp_path: Path) -> None:
    from conftest import edit_adapter

    d = make_peft_adapter(tmp_path / "p", rank=4)
    edit_adapter(d, "adapter_model.safetensors", cfg={"r": 8, "lora_alpha": 16})
    with pytest.raises(AdapterFormatError, match="rank"):
        convert(d, tmp_path / "out", "mlx")


def test_base_config_missing_file(mlx_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(BaseConfigError, match="cannot read"):
        convert(mlx_dir, tmp_path / "o", "peft", base_config=tmp_path / "nope.json")

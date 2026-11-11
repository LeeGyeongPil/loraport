"""Refusals and base-config checks (AC-6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import edit_adapter

from loraport import BaseConfigError, UnsupportedAdapterError, convert
from loraport.cli import main
from loraport.sample import make_mlx_adapter, make_peft_adapter

Z = np.zeros


def _mlx_dora(d: Path) -> None:
    edit_adapter(d, "adapters.safetensors", cfg={"fine_tune_type": "dora"},
                 add={"model.layers.2.self_attn.q_proj.m": Z(64)})


def _mlx_full(d: Path) -> None:
    edit_adapter(d, "adapters.safetensors", cfg={"fine_tune_type": "full"})


def _mlx_moe(d: Path) -> None:
    edit_adapter(d, "adapters.safetensors", add={"model.layers.3.mlp.switch_mlp.up_proj.lora_a": Z((8, 4, 64)),
                                                   "model.layers.3.mlp.switch_mlp.up_proj.lora_b": Z((8, 64, 4))})


def _peft_cfg(**cfg: object):
    return lambda d: edit_adapter(d, "adapter_model.safetensors", cfg=dict(cfg))


def _peft_embedding(d: Path) -> None:
    edit_adapter(d, "adapter_model.safetensors",
                 add={"base_model.model.model.embed_tokens.lora_embedding_A": Z((4, 128))})


CASES = [
    ("mlx", _mlx_dora, "DoRA|dora"),
    ("mlx", _mlx_full, "fine_tune_type"),
    ("mlx", _mlx_moe, "3-D"),
    ("peft", _peft_cfg(use_dora=True), "DoRA"),
    ("peft", _peft_cfg(modules_to_save=["lm_head"]), "modules_to_save"),
    ("peft", _peft_cfg(rank_pattern={"q_proj": 8}), "rank_pattern"),
    ("peft", _peft_cfg(alpha_pattern={"q_proj": 8}), "alpha_pattern"),
    ("peft", _peft_cfg(bias="all"), "bias"),
    ("peft", _peft_cfg(lora_bias=True), "lora_bias"),
    ("peft", _peft_embedding, "embedding"),
]


@pytest.mark.parametrize(("fmt", "mutate", "pattern"), CASES)
def test_unsupported_rejected(fmt: str, mutate, pattern: str, tmp_path: Path,
                              capsys: pytest.CaptureFixture[str]) -> None:
    src = make_mlx_adapter(tmp_path / "src") if fmt == "mlx" else make_peft_adapter(tmp_path / "src")
    mutate(src)
    target = "peft" if fmt == "mlx" else "mlx"
    with pytest.raises(UnsupportedAdapterError, match=pattern) as exc:
        convert(src, tmp_path / "out", target)
    assert exc.value.reasons
    assert not (tmp_path / "out").exists()  # nothing half-written
    assert main(["convert", str(src), str(tmp_path / "out"), "--to", target]) == 2
    assert "refusing to convert" in capsys.readouterr().err
    assert main(["verify", str(src), str(src)]) == 2  # verify refuses as well instead of comparing nonsense


def test_base_config_shape_mismatch(mlx_dir: Path, base_cfg: Path, tmp_path: Path,
                                    capsys: pytest.CaptureFixture[str]) -> None:
    import json

    cfg = json.loads(base_cfg.read_text())
    cfg["intermediate_size"] = 256  # adapter was made for 128
    cfg["num_hidden_layers"] = 3  # adapter touches layer 3
    base_cfg.write_text(json.dumps(cfg))
    with pytest.raises(BaseConfigError) as exc:
        convert(mlx_dir, tmp_path / "out", "peft", base_config=base_cfg)
    text = "\n".join(exc.value.problems)
    assert "model.layers.2.mlp.down_proj: expected A(4, 256) B(64, 4)" in text and "got A(4, 128) B(64, 4)" in text
    assert "layer 3 is outside" in text
    assert not (tmp_path / "out").exists()
    assert main(["convert", str(mlx_dir), str(tmp_path / "out"), "--to", "peft", "--base-config", str(base_cfg)]) == 2
    assert "model.layers.2.mlp.down_proj" in capsys.readouterr().err


def test_base_config_missing_fields(mlx_dir: Path, tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"hidden_size": 64}')
    with pytest.raises(BaseConfigError, match="num_hidden_layers"):
        convert(mlx_dir, tmp_path / "out", "peft", base_config=tmp_path)  # a folder works too

"""Optional (mlx marker): load a converted adapter with mlx-lm's own ``load_adapters``.

Runs only on a Mac with mlx-lm installed and a usable Metal device (not inside
sandboxes); otherwise skipped.  Written against mlx-lm 0.32.x and not run in the
build sandbox, which has no Metal device.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from loraport import convert, load_adapter
from loraport.delta import dense_delta
from loraport.sample import TINY_BASE, make_peft_adapter, write_base_config

pytestmark = pytest.mark.mlx


def test_load_adapters_roundtrip(tmp_path: Path) -> None:
    try:
        pytest.importorskip("mlx.core")
        llama = pytest.importorskip("mlx_lm.models.llama")
        utils = pytest.importorskip("mlx_lm.tuner.utils")
    except Exception as e:  # e.g. "No Metal device available"
        pytest.skip(f"mlx unusable here: {e}")

    peft_dir = make_peft_adapter(tmp_path / "peft", layers=(1, 3))
    base_cfg = write_base_config(tmp_path / "base")
    convert(peft_dir, tmp_path / "mlx", "mlx", base_config=base_cfg)

    args = llama.ModelArgs(
        model_type="llama", hidden_size=TINY_BASE["hidden_size"], num_hidden_layers=TINY_BASE["num_hidden_layers"],
        intermediate_size=TINY_BASE["intermediate_size"], num_attention_heads=TINY_BASE["num_attention_heads"],
        num_key_value_heads=TINY_BASE["num_key_value_heads"], rms_norm_eps=1e-5, vocab_size=TINY_BASE["vocab_size"],
    )
    model = llama.Model(args)
    model = utils.load_adapters(model, str(tmp_path / "mlx"))

    reference = load_adapter(peft_dir)
    for name, mod in reference.modules.items():
        parts = name.split(".")  # model.layers.{i}.{block}.{proj}
        layer = model.layers[int(parts[2])]
        proj = getattr(getattr(layer, parts[3]), parts[4])
        got = proj.scale * (np.array(proj.lora_b).T.astype(np.float64) @ np.array(proj.lora_a).T.astype(np.float64))
        expected = dense_delta(mod)
        assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < 1e-5
    # block 2 is inside num_layers (3) but absent from the PEFT adapter: it must stay at dW = 0
    assert np.allclose(np.array(model.layers[2].self_attn.q_proj.lora_b), 0)

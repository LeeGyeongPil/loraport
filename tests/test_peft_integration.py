"""Optional (AC-10): load loraport's output with the real PEFT library.

Needs the ``peft`` extra (``uv run --extra peft pytest -m peft``); otherwise skipped.
The base model is a random tiny Llama built from a config, nothing is downloaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from loraport import convert, load_adapter
from loraport.delta import dense_delta
from loraport.sample import TINY_BASE, make_mlx_adapter

pytestmark = pytest.mark.peft


def test_peft_merge_matches_mlx_delta(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from peft import PeftModel

    mlx = make_mlx_adapter(tmp_path / "mlx")
    res = convert(mlx, tmp_path / "peft", "peft")
    assert res.ok

    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        hidden_size=TINY_BASE["hidden_size"], num_hidden_layers=TINY_BASE["num_hidden_layers"],
        intermediate_size=TINY_BASE["intermediate_size"], num_attention_heads=TINY_BASE["num_attention_heads"],
        num_key_value_heads=TINY_BASE["num_key_value_heads"], vocab_size=TINY_BASE["vocab_size"],
    )
    base = transformers.LlamaForCausalLM(cfg).eval()
    before = {k: v.detach().clone() for k, v in base.state_dict().items()}
    merged = PeftModel.from_pretrained(base, str(tmp_path / "peft")).merge_and_unload()
    after = merged.state_dict()

    reference = load_adapter(mlx)  # dW computed from the MLX tensors, independent of PEFT
    changed, worst = 0, 0.0
    for name, mod in reference.modules.items():
        key = f"{name}.weight"
        actual = (after[key] - before[key]).double().numpy()
        if np.linalg.norm(actual) > 0:
            changed += 1
        expected = dense_delta(mod)
        worst = max(worst, float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)))
    unchanged = [k for k in before if not any(k == f"{n}.weight" for n in reference.modules)
                 and not np.array_equal(before[k].numpy(), after[k].numpy())]
    assert changed == len(reference.modules) == 14
    assert not unchanged
    assert worst <= 1e-5

# MLX vs PEFT LoRA format notes

Facts checked on 2026-11-11 against mlx-lm 0.32.2 (installed source), PEFT 0.21.2 (a saved adapter) and the sources linked below.

| | mlx-lm | PEFT |
|---|---|---|
| weights file | `adapters.safetensors` | `adapter_model.safetensors` (metadata `format=pt`) |
| key | `model.layers.{i}.{mod}.lora_a` / `.lora_b` | `base_model.model.model.layers.{i}.{mod}.lora_A.weight` / `.lora_B.weight` |
| A shape | `lora_a (in, r)` | `lora_A (r, in)` |
| B shape | `lora_b (r, out)` | `lora_B (out, r)` |
| scale | `lora_parameters.scale` (default 20) | `lora_alpha / r`, or `lora_alpha / sqrt(r)` with `use_rslora` |
| dW | `scale * lora_b.T @ lora_a.T` | `(alpha / r) * B @ A` |
| layer range | `num_layers` counted from the last block (`-1` = all) | `layers_to_transform` list |

Conversion rules used by loraport: `lora_alpha = scale x r`; `scale = lora_alpha / r` (`/ sqrt(r)` for rslora); transpose A and B; `num_layers = L - min(layer)` when the base model's `num_hidden_layers` is known, otherwise `-1` (blocks without tensors keep `lora_b = 0`, mlx-lm's `load_adapters` loads with `strict=False`).

## Pitfalls loraport detects
1. Missing `base_model.model.` prefix: PEFT only warns and loads dW = 0.
2. `lora_alpha = scale` instead of `scale x r`: effect shrinks by r.
3. A and B swapped and both transposed on square modules (q/o): shapes stay legal, dW becomes its transpose.
4. Dropped tensors (for example a missing MLP block): the file still loads.
5. lora_A not transposed: wrong shape.

## Notes
- llama.cpp `convert_lora_to_gguf.py` reads only `lora_alpha` (not `r`, not `use_rslora`). loraport always writes `use_rslora=false` with `lora_alpha = scale x r`, so the scale is right for it.
- vLLM rejects DoRA, `lora_bias`, bias training and `modules_to_save` other than classifier heads.
- The tensors are the same for a 4-bit base model, but outputs differ slightly if the adapter is attached to a different base precision. loraport proves dW equality, not output equality.
- The mlx-lm side was verified from source and from a real adapter header on the Hugging Face hub; `load_adapters` itself could not be run in the build sandbox (no Metal device). `tests/test_mlx_integration.py` covers it on a normal Mac terminal.

## Sources (2026-11-11)
- https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/tuner
- https://github.com/ml-explore/mlx-lm/issues/1119
- https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint.md
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_lora_to_gguf.py
- https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/lora/peft_helper.py

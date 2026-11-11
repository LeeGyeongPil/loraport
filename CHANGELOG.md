# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-11

### Added
- `loraport inspect`, `convert`, `verify` and `demo` commands.
- Two-way conversion between mlx-lm and PEFT LoRA adapters (key names, transposes, `lora_alpha = scale x rank`, rslora, `num_layers`, `layers_to_transform`).
- Per-module dW verification with a low-rank Gram computation (no out x in matrix), missing/extra/shape-mismatch/ignored-key reporting and exit codes 0/1/2.
- BF16/F16/F32 safetensors reading without torch.
- Explicit refusal of DoRA, MoE, embedding LoRA, `modules_to_save`, `rank_pattern`/`alpha_pattern` and bias adapters.
- Optional `--base-config` shape and layer-range check.
- Python API: `load_adapter`, `convert`, `compare`, `save_peft`, `save_mlx`.

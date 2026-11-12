# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.2] - 2026-11-13

### Fixed
- A path that does not exist now reports `no such directory (check the path)` instead of the misleading `not a directory (pass the adapter folder, not a file)`. Exit code 2 and the exception type are unchanged.

## [0.1.1] - 2026-11-13

### Fixed
- `demo` now creates `loraport-demo/dora` so README section 5 reproduces the DoRA refusal; the section 5 command uses `uv run`.

## [0.1.0] - 2026-11-11

### Added
- `loraport inspect`, `convert`, `verify` and `demo` commands.
- Two-way conversion between mlx-lm and PEFT LoRA adapters (key names, transposes, `lora_alpha = scale x rank`, rslora, `num_layers`, `layers_to_transform`).
- Per-module dW verification with a low-rank Gram computation (no out x in matrix), missing/extra/shape-mismatch/ignored-key reporting and exit codes 0/1/2.
- BF16/F16/F32 safetensors reading without torch.
- Explicit refusal of DoRA, MoE, embedding LoRA, `modules_to_save`, `rank_pattern`/`alpha_pattern` and bias adapters.
- Optional `--base-config` shape and layer-range check.
- Python API: `load_adapter`, `convert`, `compare`, `save_peft`, `save_mlx`.

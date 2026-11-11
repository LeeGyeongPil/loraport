"""Exceptions raised by loraport. The CLI maps every ``LoraportError`` to exit code 2."""

from __future__ import annotations

from collections.abc import Iterable


class LoraportError(Exception):
    """Base class for all expected, user-fixable errors."""


class AdapterFormatError(LoraportError):
    """The folder is not a readable MLX or PEFT LoRA adapter (or cannot be written)."""


class UnsupportedAdapterError(LoraportError):
    """The adapter uses a feature loraport refuses to convert instead of silently breaking it."""

    def __init__(self, reasons: Iterable[str]) -> None:
        self.reasons: list[str] = list(reasons)
        super().__init__("unsupported adapter: " + "; ".join(self.reasons))


class BaseConfigError(LoraportError):
    """The base model ``config.json`` is unreadable or disagrees with the adapter tensors."""

    def __init__(self, message: str, problems: Iterable[str] = ()) -> None:
        self.problems: list[str] = list(problems)
        super().__init__(message)

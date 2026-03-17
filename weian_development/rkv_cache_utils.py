"""Utilities for controlling R-KV generation/cache state."""
from __future__ import annotations

from typing import Iterable


_CACHE_ATTRS = (
    "past_key_values",
    "_past_key_values",
    "_past",
    "_cache",
    "_key_value_cache",
    "_model_cache",
    "cache",
)


def _clear_attrs(obj: object, attrs: Iterable[str]) -> None:
    for attr in attrs:
        if hasattr(obj, attr):
            setattr(obj, attr, None)


def reset_model_cache(model: object) -> None:
    """Best-effort cleanup so every prompt starts from a fresh cache."""
    _clear_attrs(model, _CACHE_ATTRS)
    inner = getattr(model, "model", None)
    if inner is not None:
        _clear_attrs(inner, _CACHE_ATTRS)

    for module in getattr(model, "modules", lambda: [])():
        reset_fn = getattr(module, "reset_compression_state", None)
        if callable(reset_fn):
            reset_fn()

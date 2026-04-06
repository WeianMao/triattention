"""Compatibility helpers for scheduler/request key shapes.

vLLM may expose `num_scheduled_tokens` with keys that are:
- request-id strings, or
- request objects / wrappers carrying `request_id` / `req_id`.

TriAttention runtime uses these helpers to normalize keys consistently across
scheduler/runtime/hook codepaths.
"""

from __future__ import annotations

from typing import Any, Iterator


def req_id_from_scheduled_key(key: Any) -> Any:
    """Normalize a scheduler `num_scheduled_tokens` key to `req_id` string."""
    if isinstance(key, (str, int)):
        return key
    req_id = getattr(key, "request_id", None)
    if isinstance(req_id, (str, int)):
        return req_id
    req_id = getattr(key, "req_id", None)
    if isinstance(req_id, (str, int)):
        return req_id
    return None


def iter_scheduled_token_items(
    scheduler_output: Any,
) -> Iterator[tuple[Any, Any, int]]:
    """Yield `(raw_key, req_id, scheduled_tokens)` preserving mapping order."""
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
    if not isinstance(scheduled, dict):
        return
    for raw_key, raw_value in scheduled.items():
        req_id = req_id_from_scheduled_key(raw_key)
        if req_id is None:
            continue
        try:
            scheduled_tokens = int(raw_value)
        except (TypeError, ValueError):
            continue
        yield raw_key, req_id, scheduled_tokens


def get_scheduled_token_items(
    scheduler_output: Any,
) -> list[tuple[Any, Any, int]]:
    """Return normalized scheduled items with per-output best-effort caching.

    This avoids re-normalizing `num_scheduled_tokens` keys multiple times in the
    same decode step (runner gating + effective override build).
    """
    if scheduler_output is None:
        return []
    try:
        cached = getattr(scheduler_output, "_triattention_cached_scheduled_token_items", None)
    except Exception:
        cached = None
    if isinstance(cached, list):
        return cached
    items = list(iter_scheduled_token_items(scheduler_output))
    try:
        setattr(scheduler_output, "_triattention_cached_scheduled_token_items", items)
    except Exception:
        # Some scheduler output objects may not allow dynamic attributes.
        pass
    return items

"""Track effective KV cache length per request."""

from __future__ import annotations


class EffectiveCacheLenTracker:
    """Maintain request-level effective cache length.

    `num_computed_tokens` is monotonic request progress, while effective cache
    length may shrink after compression. This tracker bridges the two.
    """

    def __init__(self) -> None:
        self._effective_len: dict[str, int] = {}
        self._last_num_computed: dict[str, int] = {}

    def reset_request(self, req_id: str, num_computed_tokens: int) -> None:
        self._effective_len.pop(req_id, None)
        self._last_num_computed[req_id] = max(0, int(num_computed_tokens))

    def remove_request(self, req_id: str) -> None:
        self._effective_len.pop(req_id, None)
        self._last_num_computed.pop(req_id, None)

    def observe_num_computed(self, req_id: str, num_computed_tokens: int) -> int:
        """Update tracker with latest computed tokens and return effective base len."""
        current = max(0, int(num_computed_tokens))
        if req_id not in self._effective_len:
            self._last_num_computed[req_id] = current
            return current

        effective = self._effective_len[req_id]
        last = self._last_num_computed.get(req_id, current)

        if current >= last:
            effective += current - last
        else:
            # Defensive path for unusual rollback cases.
            effective = min(effective, current)

        self._effective_len[req_id] = max(0, effective)
        self._last_num_computed[req_id] = current
        return self._effective_len[req_id]

    def apply_compression(
        self,
        req_id: str,
        cache_len_after: int,
        num_computed_tokens: int,
    ) -> None:
        self._effective_len[req_id] = max(0, int(cache_len_after))
        self._last_num_computed[req_id] = max(0, int(num_computed_tokens))

    def snapshot(self) -> dict[str, tuple[int, int | None]]:
        keys = set(self._effective_len.keys()) | set(self._last_num_computed.keys())
        return {
            req_id: (
                self._effective_len.get(req_id, -1),
                self._last_num_computed.get(req_id),
            )
            for req_id in keys
        }


"""Debug logging for compression events: decoded text before/after each trigger.

When enabled via TRIATTN_RUNTIME_DEBUG_COMPRESSION_LOG=true, this module
captures the last N token IDs before compression and the next N token IDs
after compression, decodes them via the model tokenizer, and writes a
formatted log entry to a file.

This is strictly a debug/diagnosis tool. When disabled (the default), no
tokenizer is loaded and no file I/O occurs.
"""

from __future__ import annotations

import datetime
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import TriAttentionRuntimeConfig


@dataclass
class _PendingPostCapture:
    """Tracks post-compression token collection for one compression event."""

    req_id: str
    event_id: int
    step: int
    before_tokens: int
    after_tokens: int
    remaining: int
    collected_ids: list[int] = field(default_factory=list)


class CompressionDebugLog:
    """Singleton-style debug logger for compression context capture."""

    def __init__(self, config: TriAttentionRuntimeConfig) -> None:
        self._enabled = config.debug_compression_log
        self._context_tokens = max(1, config.debug_compression_log_context_tokens)
        self._log_path: Path | None = None
        self._tokenizer: Any = None
        self._tokenizer_loaded = False
        self._model_path = config.model_path
        self._event_counter = 0
        self._pending: dict[str, _PendingPostCapture] = {}
        self._lock = threading.Lock()

        if self._enabled:
            log_path_raw = config.debug_compression_log_path.strip()
            if not log_path_raw:
                pid = os.getpid()
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path_raw = f"/tmp/triattn_compression_debug_{pid}_{ts}.log"
            self._log_path = Path(log_path_raw)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer_loaded:
            return self._tokenizer
        self._tokenizer_loaded = True
        if self._model_path is None:
            return None
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._model_path), trust_remote_code=True
            )
        except Exception:
            self._tokenizer = None
        return self._tokenizer

    def _decode(self, token_ids: list[int]) -> str:
        tokenizer = self._ensure_tokenizer()
        if tokenizer is None or not token_ids:
            return f"<raw_ids:{token_ids}>"
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=False)
        except Exception:
            return f"<decode_error:ids={token_ids[:10]}...>"

    def _write(self, text: str) -> None:
        if self._log_path is None:
            return
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(text)

    def record_compression_event(
        self,
        *,
        req_id: str,
        step: int,
        before_tokens: int,
        after_tokens: int,
        base_runner: Any,
    ) -> None:
        """Called immediately after compression is applied.

        Captures the pre-compression context (last N tokens) and sets up
        collection for post-compression tokens.
        """
        if not self._enabled:
            return

        with self._lock:
            self._event_counter += 1
            event_id = self._event_counter

        # --- Capture pre-compression token IDs ---
        pre_token_ids = self._extract_recent_token_ids(
            base_runner=base_runner,
            req_id=req_id,
            count=self._context_tokens,
        )

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        pre_text = self._decode(pre_token_ids)

        header = (
            f"\n{'=' * 60}\n"
            f"Compression Event #{event_id}\n"
            f"{'=' * 60}\n"
            f"Timestamp : {ts}\n"
            f"Request   : {req_id}\n"
            f"Step      : {step}\n"
            f"KV Before : {before_tokens} tokens\n"
            f"KV After  : {after_tokens} tokens\n"
            f"Evicted   : {before_tokens - after_tokens} tokens\n"
            f"\n"
            f"--- PRE-COMPRESSION (last {len(pre_token_ids)} tokens) ---\n"
            f"{pre_text}\n"
            f"\n"
            f"<<<COMPRESSION #{event_id} TRIGGERED>>>\n"
            f"\n"
        )
        self._write(header)

        # --- Set up post-compression collection ---
        pending = _PendingPostCapture(
            req_id=req_id,
            event_id=event_id,
            step=step,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            remaining=self._context_tokens,
        )
        with self._lock:
            # If there was a previous pending capture for this request
            # (interrupted by a new compression), flush it as incomplete.
            old = self._pending.pop(req_id, None)
            if old is not None:
                self._flush_incomplete(old)
            self._pending[req_id] = pending

    def collect_post_token(self, req_id: str, token_id: int) -> None:
        """Called each decode step to collect one new token for pending captures."""
        if not self._enabled:
            return

        with self._lock:
            pending = self._pending.get(req_id)
            if pending is None:
                return
            pending.collected_ids.append(token_id)
            pending.remaining -= 1
            if pending.remaining > 0:
                return
            # Collection complete — remove from pending.
            del self._pending[req_id]

        # Flush outside the lock.
        post_text = self._decode(pending.collected_ids)
        self._write(
            f"--- POST-COMPRESSION (next {len(pending.collected_ids)} tokens) ---\n"
            f"{post_text}\n"
            f"\n"
            f"{'=' * 60}\n\n"
        )

    def flush_all_pending(self) -> None:
        """Flush any incomplete post-compression captures (e.g. on request finish)."""
        if not self._enabled:
            return
        with self._lock:
            to_flush = list(self._pending.values())
            self._pending.clear()
        for p in to_flush:
            self._flush_incomplete(p)

    def flush_request(self, req_id: str) -> None:
        """Flush pending capture for a specific request (e.g. on request finish)."""
        if not self._enabled:
            return
        with self._lock:
            pending = self._pending.pop(req_id, None)
        if pending is not None:
            self._flush_incomplete(pending)

    def _flush_incomplete(self, pending: _PendingPostCapture) -> None:
        collected = len(pending.collected_ids)
        expected = collected + pending.remaining
        if collected == 0:
            self._write(
                f"--- POST-COMPRESSION (0/{expected} tokens collected, incomplete) ---\n"
                f"<no tokens collected before next event or request end>\n"
                f"\n"
                f"{'=' * 60}\n\n"
            )
            return
        post_text = self._decode(pending.collected_ids)
        self._write(
            f"--- POST-COMPRESSION ({collected}/{expected} tokens, incomplete) ---\n"
            f"{post_text}\n"
            f"\n"
            f"{'=' * 60}\n\n"
        )

    @staticmethod
    def _extract_recent_token_ids(
        *,
        base_runner: Any,
        req_id: str,
        count: int,
    ) -> list[int]:
        """Extract the most recent `count` token IDs for a request from input_batch."""
        input_batch = getattr(base_runner, "input_batch", None)
        if input_batch is None:
            return []
        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        if not isinstance(req_id_to_index, dict):
            return []
        req_index = req_id_to_index.get(req_id)
        if not isinstance(req_index, int):
            return []
        try:
            active_len = int(getattr(input_batch, "num_tokens_no_spec")[req_index])
        except Exception:
            active_len = 0
        if active_len <= 0:
            return []
        try:
            token_row = getattr(input_batch, "token_ids_cpu")[req_index]
            start = max(0, active_len - count)
            return [int(x) for x in token_row[start:active_len].tolist()]
        except Exception:
            return []

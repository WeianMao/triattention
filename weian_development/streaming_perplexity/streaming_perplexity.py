"""Streaming attention backed perplexity evaluator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from weian_development.compute_reasoning_perplexity import (
    PerplexityConfig,
    PerplexityEvaluator,
)


@dataclass
class StreamingPerplexityConfig(PerplexityConfig):
    """Configuration for streaming perplexity evaluation."""

    stream_window: int = 4096
    attention_sink_size: int = 4


class StreamingPerplexityEvaluator(PerplexityEvaluator):
    """Evaluator that bounds KV cache size via streaming attention."""

    def __init__(self, config: StreamingPerplexityConfig) -> None:
        self.stream_window = config.stream_window
        self.attention_sink_size = max(int(config.attention_sink_size), 0)
        super().__init__(config)

    def _compute_log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        seq_len = input_ids.size(1)
        if seq_len <= 1:
            return torch.empty(0, dtype=torch.float32)

        log_probs = torch.empty(seq_len - 1, dtype=torch.float32, device=self.device)
        stream_window = max(int(self.stream_window), 0)
        sink_size = min(self.attention_sink_size, seq_len)
        sink_tokens = input_ids[:, :sink_size] if sink_size > 0 else None
        position_ids_full = torch.arange(seq_len, device=self.device).unsqueeze(0)
        sink_positions = position_ids_full[:, :sink_size] if sink_size > 0 else None

        with torch.no_grad():
            for idx in range(seq_len - 1):
                if stream_window > 0:
                    window_start = max(0, idx + 1 - stream_window)
                else:
                    window_start = 0

                if sink_tokens is not None and window_start > sink_size:
                    context_tokens = torch.cat(
                        [sink_tokens, input_ids[:, window_start : idx + 1]], dim=1
                    )
                    context_positions = torch.cat(
                        [sink_positions, position_ids_full[:, window_start : idx + 1]],
                        dim=1,
                    )
                else:
                    context_tokens = input_ids[:, : idx + 1]
                    context_positions = position_ids_full[:, : idx + 1]

                outputs = self.model(  # type: ignore[attr-defined]
                    input_ids=context_tokens,
                    position_ids=context_positions,
                    use_cache=False,
                )
                logits = outputs.logits[:, -1, :]
                next_token_id = input_ids[0, idx + 1]
                log_probs[idx] = torch.log_softmax(logits, dim=-1)[0, next_token_id]

        return log_probs.cpu()

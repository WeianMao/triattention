"""Compression state management for TriAttention.

This module manages the runtime state of KV cache compression, including
budget usage and compression scheduling.

Design Alignment:
- Aligns with R-KV state management patterns
- Implements dual reset mechanism (slot reuse + scheduler hook)

Note on Position Tracking:
Position tracking (position_indices) has been DEPRECATED and removed.
Since phase calculation uses t*omega + phi_rot (where phi_rot is computed from K_rot),
per-token position tracking is NOT needed for scoring.
This saves significant GPU memory.
"""
from typing import Dict, List, Optional

import torch

from .config import TriAttentionConfig


class CompressionState:
    """Manages compression state for a single request.

    This class tracks:
    - Absolute position in the sequence
    - Position indices for each cached KV token
    - Prefill length (for protection)
    - Compression scheduling state
    """

    def __init__(self, config: TriAttentionConfig):
        """Initialize compression state.

        Args:
            config: TriAttention configuration
        """
        self.config = config

        # Global position tracking
        self.absolute_position: int = 0
        """Current absolute position in the sequence (monotonically increasing)."""

        self.compression_count: int = 0
        """Number of compressions performed so far."""

        self.prefill_length: int = 0
        """Length of the initial prefill (protected if protect_prefill=True)."""

        self.tokens_in_round: int = 0
        """Number of tokens added since last compression."""

        # NOTE: position_indices storage has been REMOVED to save GPU memory.
        # Since phase calculation uses t*omega + phi_rot (phi_rot computed from K_rot),
        # per-token position tracking is NOT needed for scoring.
        # Only scalar position tracking (absolute_position, current_cache_len) is kept.

        # Current cache length tracking
        self.current_cache_len: int = 0
        """Current number of tokens in the KV cache."""

        # Last compression step
        self.last_prune_step: int = 0
        """Absolute position at which last compression occurred."""

    def reset(self) -> None:
        """Reset all state for a new sequence.

        This should be called:
        1. When starting a new request
        2. When a request is cancelled or finished (slot reuse)
        3. Via scheduler hook for cleanup
        """
        self.absolute_position = 0
        self.compression_count = 0
        self.prefill_length = 0
        self.tokens_in_round = 0
        self.current_cache_len = 0
        self.last_prune_step = 0

    def should_compress(self, current_len: int) -> bool:
        """Determine if compression should be triggered.

        Compression follows R-KV slack mode logic:
        - Trigger when cache reaches (budget + divide_length)
        - Compress down to budget
        - This creates cache fluctuation in [budget, budget + divide_length]

        Implementation Note:
        - During prefill: Use current_len from vLLM (state not initialized yet)
        - After first compression: Use internally tracked current_cache_len
          This is critical because vLLM's seq_len doesn't update after compression
        - Track vLLM's seq_len to detect new tokens added during decode

        Args:
            current_len: Current sequence length reported by vLLM (monotonically increasing)

        Returns:
            True if compression should be triggered
        """
        # Update internal tracking with new tokens from vLLM
        if self.absolute_position == 0:
            # First call - initialize from prefill
            self.initialize(current_len)
            effective_cache_len = current_len
        else:
            # Calculate new tokens added since last call
            new_tokens = current_len - self.absolute_position
            if new_tokens > 0:
                # Update state with new tokens
                self.append_tokens(new_tokens)
            # Use internally tracked cache length
            effective_cache_len = self.current_cache_len

        # Calculate effective size (excluding protected prefill if enabled)
        if self.config.protect_prefill:
            effective_size = max(0, effective_cache_len - self.prefill_length)
        else:
            effective_size = effective_cache_len

        # R-KV slack mode: trigger at budget + divide_length
        trigger_threshold = self.config.kv_budget + self.config.divide_length

        return effective_size >= trigger_threshold

    def initialize(
        self,
        seq_len: int,
    ) -> None:
        """Initialize state for the initial cache.

        Args:
            seq_len: Initial sequence length (prefill length)
        """
        self.current_cache_len = seq_len
        self.prefill_length = seq_len
        self.absolute_position = seq_len

    def append_tokens(self, num_new_tokens: int) -> None:
        """Update state for newly added tokens.

        Args:
            num_new_tokens: Number of new tokens added
        """
        self.current_cache_len += num_new_tokens
        self.absolute_position += num_new_tokens
        self.tokens_in_round += num_new_tokens

    def update_after_compression(
        self,
        new_cache_len: int,
    ) -> None:
        """Update state after compression.

        Args:
            new_cache_len: New cache length after compression
        """
        self.current_cache_len = new_cache_len
        self.last_prune_step = self.absolute_position
        self.tokens_in_round = 0
        self.compression_count += 1

    def get_effective_budget(self) -> int:
        """Get the effective budget for compression.

        Returns:
            Effective budget considering prefill protection
        """
        if self.config.protect_prefill:
            # Reserve space for prefill tokens
            return max(0, self.config.kv_budget - self.prefill_length)
        return self.config.kv_budget

    def get_round_start(self) -> int:
        """Get current round start position for scoring.

        Returns:
            The absolute position to use as round_start in scoring.
        """
        return self.absolute_position

    def to_dict(self) -> dict:
        """Convert state to dictionary for debugging/logging.

        Returns:
            Dictionary representation of state
        """
        return {
            "absolute_position": self.absolute_position,
            "compression_count": self.compression_count,
            "prefill_length": self.prefill_length,
            "tokens_in_round": self.tokens_in_round,
            "current_cache_len": self.current_cache_len,
            "last_prune_step": self.last_prune_step,
        }

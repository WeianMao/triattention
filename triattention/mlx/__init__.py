"""
TriAttention MLX — Apple Silicon port
======================================
Native MLX implementation of TriAttention KV compression.

Original: https://github.com/WeianMao/triattention (MIT)
MLX port:  https://github.com/DeadByDawn101/triattention

Usage:
    from triattention.mlx import apply_triattention_mlx

    model, tokenizer = mlx_lm.load("deadbydawn101/gemma-4-E4B-Agentic-Opus-Reasoning-GeminiCLI-mlx-4bit")
    apply_triattention_mlx(model, stats_path="triattention/calibration/gemma4_e4b_stats.npz", kv_budget=2048)
"""

from .triattention_mlx import apply_triattention_mlx, TriAttentionMLXConfig

__all__ = ["apply_triattention_mlx", "TriAttentionMLXConfig"]

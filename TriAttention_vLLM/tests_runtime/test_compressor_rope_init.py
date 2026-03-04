from __future__ import annotations

from pathlib import Path

import torch

from triattention.compressor import TriAttentionCompressor
from triattention.config import TriAttentionConfig


def test_compressor_prefers_metadata_inv_freq(tmp_path: Path):
    stats_path = tmp_path / "stats_with_inv_freq.pt"
    torch.save(
        {
            "metadata": {
                "num_attention_heads": 1,
                "num_kv_heads": 1,
                "head_dim": 4,
                "num_layers": 1,
                "inv_freq": torch.tensor([0.5, 0.25], dtype=torch.float32),
                "rope_theta": 10000.0,
            },
            "layer_stats": {
                0: {
                    "freq_scale_sq": torch.ones(1, 2, dtype=torch.float32),
                }
            },
        },
        stats_path,
    )

    cfg = TriAttentionConfig(
        stats_path=stats_path,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        topk_dtype=torch.float32,
        use_triton_scoring=False,
    )
    compressor = TriAttentionCompressor(cfg)
    compressor._lazy_init()

    assert compressor.inv_freq is not None
    assert torch.allclose(
        compressor.inv_freq,
        torch.tensor([0.5, 0.25], dtype=torch.float32),
    )
    assert compressor.omega is not None
    assert torch.allclose(compressor.omega, 2.0 * torch.pi * compressor.inv_freq)

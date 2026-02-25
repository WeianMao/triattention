from __future__ import annotations

import torch

from triattention.utils import _convert_rkv_stats


def _minimal_stats(model_path: str | None = None) -> dict:
    metadata: dict[str, object] = {"head_dim": 4}
    if model_path is not None:
        metadata["model_path"] = model_path
    return {
        "metadata": metadata,
        "stats": {
            "layer00_head00": {
                "q_abs_mean": torch.tensor([2.0, 3.0]),
                "q_mean_real": torch.tensor([0.1, 0.2]),
                "q_mean_imag": torch.tensor([0.3, 0.4]),
            },
            "layer00_head01": {
                "q_abs_mean": torch.tensor([6.0, 8.0]),
                "q_mean_real": torch.tensor([0.5, 0.6]),
                "q_mean_imag": torch.tensor([0.7, 0.8]),
            },
        },
    }


def test_convert_rkv_stats_freq_scale_sq_not_derived_from_q_abs_mean_sq():
    stats = _minimal_stats()
    metadata, head_stats = _convert_rkv_stats(
        stats=stats,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_kv_heads=1,
    )
    assert metadata["num_kv_heads"] == 1
    layer_stats = head_stats[0]
    q_abs_mean = layer_stats["q_abs_mean"]
    freq_scale_sq = layer_stats["freq_scale_sq"]
    assert q_abs_mean.shape == (1, 2)
    assert freq_scale_sq.shape == (1, 2)
    assert torch.allclose(freq_scale_sq, torch.ones_like(freq_scale_sq))
    assert not torch.allclose(freq_scale_sq, q_abs_mean.pow(2))


def test_convert_rkv_stats_invalid_model_path_falls_back_to_ones_freq_scale_sq():
    stats = _minimal_stats(model_path="/tmp/triattention_runtime_nonexistent_model_path")
    _, head_stats = _convert_rkv_stats(
        stats=stats,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_kv_heads=1,
    )
    freq_scale_sq = head_stats[0]["freq_scale_sq"]
    assert torch.allclose(freq_scale_sq, torch.ones_like(freq_scale_sq))

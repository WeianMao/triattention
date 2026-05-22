from pathlib import Path

import torch

from triattention.sglang.stats_loader import (
    load_stats,
    validate_stats_against_model,
)
from triattention.vllm.core.utils import load_frequency_stats


def _gemma4_like_payload() -> dict:
    stats = {}
    for layer_idx, freq_count in [(0, 128), (1, 256)]:
        for head_idx in range(4):
            value = float(layer_idx + head_idx + 1)
            stats[f"layer{layer_idx:02d}_head{head_idx:02d}"] = {
                "q_mean_real": torch.full((freq_count,), value),
                "q_mean_imag": torch.full((freq_count,), value + 0.5),
                "q_abs_mean": torch.full((freq_count,), value + 1.0),
            }
    return {
        "metadata": {
            "head_dim": 512,
            "layer_head_dims": [256, 512],
            "num_key_value_heads": 2,
            "rope_style": "half",
            "rope_theta": 10000.0,
        },
        "stats": stats,
    }


def test_vllm_rkv_loader_handles_gemma4_mixed_head_dims(tmp_path: Path) -> None:
    path = tmp_path / "gemma4_stats.pt"
    torch.save(_gemma4_like_payload(), path)

    metadata, head_stats = load_frequency_stats(
        path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert metadata["head_dim"] == 512
    assert metadata["num_kv_heads"] == 2
    assert metadata["layer_head_dims"] == [256, 512]
    assert metadata["layer_freq_counts"] == [128, 256]

    assert head_stats[0]["q_mean_complex"].shape == (2, 256, 2)
    assert head_stats[1]["q_mean_complex"].shape == (2, 256, 2)
    assert torch.all(head_stats[0]["q_mean_complex"][:, 128:, :] == 0)
    assert torch.all(head_stats[0]["q_abs_mean"][:, 128:] == 1)


def test_sglang_rkv_loader_handles_gemma4_mixed_head_dims(tmp_path: Path) -> None:
    path = tmp_path / "gemma4_stats.pt"
    torch.save(_gemma4_like_payload(), path)

    bundle = load_stats(
        str(path),
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_kv_heads=2,
    )

    assert bundle.head_dim == 512
    assert bundle.num_kv_heads == 2
    assert bundle.num_attention_heads == 4
    assert bundle.gqa_group_size == 2
    assert bundle.layer_freq_counts == [128, 256]

    assert bundle.head_stats[0]["q_mean_complex"].shape == (4, 256, 2)
    assert bundle.head_stats[1]["q_mean_complex"].shape == (4, 256, 2)
    assert torch.all(bundle.head_stats[0]["q_mean_complex"][:, 128:, :] == 0)
    assert torch.all(bundle.head_stats[0]["q_abs_mean"][:, 128:] == 1)

    validate_stats_against_model(
        bundle,
        model_num_layers=2,
        model_num_kv_heads=2,
        model_head_dim=256,
    )
    validate_stats_against_model(
        bundle,
        model_num_layers=2,
        model_num_kv_heads=2,
        model_head_dim=512,
    )

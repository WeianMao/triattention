from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


# Ensure we import the R-KV-local weian_development package rather than the top-level one.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weian_development.speckv.round_pruning_utils import determine_rope_style, invert_rope


MODEL_PATH = Path(
    os.environ.get(
        "SPECKV_LLAMA_MODEL_PATH",
        "/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B",
    )
)


def _apply_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Mirror the even/odd rotary application to validate the inverse branch."""
    even = x[..., ::2]
    odd = x[..., 1::2]
    rotated_even = even * cos[..., ::2] - odd * sin[..., ::2]
    rotated_odd = odd * cos[..., 1::2] + even * sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="DeepSeek llama config not found at MODEL_PATH")
def test_llama_half_style_round_trip_matches_model_apply() -> None:
    config = AutoConfig.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    rotary = LlamaRotaryEmbedding(config=config, device="cpu")
    style = determine_rope_style(config)
    assert style == "half"

    seq_len = 8
    head_dim = rotary.inv_freq.numel() * 2
    base = torch.randn(1, seq_len, head_dim, dtype=torch.float32)
    position_ids = torch.arange(seq_len, device=base.device).unsqueeze(0)
    cos, sin = rotary(base, position_ids)

    rotated, _ = apply_rotary_pos_emb(base, base, cos, sin)
    restored = invert_rope(rotated[0], cos[0], sin[0], rotary.attention_scaling, style=style)
    max_err = (restored - base[0]).abs().max().item()
    assert max_err < 1e-6


def test_interleaved_branch_inverts_asymmetric_pairs() -> None:
    torch.manual_seed(0)
    seq_len, head_dim = 3, 8
    base = torch.randn(seq_len, head_dim, dtype=torch.float32)
    # Keep even/odd angles distinct but away from singular det ~ 0 to avoid numerical blow-up.
    angles_even = torch.randn(seq_len, head_dim // 2)
    angles_odd = angles_even + 0.3
    cos_even = torch.cos(angles_even)
    sin_even = torch.sin(angles_even)
    cos_odd = torch.cos(angles_odd)
    sin_odd = torch.sin(angles_odd)
    scale = 1.7

    cos = torch.empty(seq_len, head_dim, dtype=torch.float32)
    sin = torch.empty(seq_len, head_dim, dtype=torch.float32)
    cos[..., ::2] = cos_even * scale
    cos[..., 1::2] = cos_odd * scale
    sin[..., ::2] = sin_even * scale
    sin[..., 1::2] = sin_odd * scale

    rotated = _apply_interleaved(base, cos, sin)
    restored = invert_rope(rotated, cos, sin, scale, style="interleaved")
    assert torch.allclose(restored, base, atol=1e-5)

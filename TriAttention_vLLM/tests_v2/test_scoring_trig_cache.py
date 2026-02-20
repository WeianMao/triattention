from __future__ import annotations

import torch

from triattention.config import TriAttentionConfig
from triattention.scoring import compute_scores_triton


class _FakeTrigCache:
    def __init__(self, divide_length: int, num_positions: int, num_offsets: int, freq_count: int):
        self.divide_length = divide_length
        self.num_positions = num_positions
        self.num_offsets = num_offsets
        self.freq_count = freq_count

    def get_trig_values(self, round_start: int):
        if round_start % self.divide_length != 0:
            raise ValueError("round_start must align with divide_length")
        cos = torch.ones((self.num_offsets, self.freq_count), dtype=torch.float32)
        sin = torch.zeros((self.num_offsets, self.freq_count), dtype=torch.float32)
        return cos, sin


def _build_inputs():
    key_states = torch.randn((1, 2, 5, 4), dtype=torch.float32)
    head_stats = {
        "q_mean_complex": torch.randn((2, 2, 2), dtype=torch.float32),
        "q_abs_mean": torch.rand((2, 2), dtype=torch.float32),
    }
    omega = torch.rand((2,), dtype=torch.float32)
    offsets = torch.tensor([0.0, -1.0], dtype=torch.float32)
    freq_scale_sq = torch.ones((2, 2), dtype=torch.float32)
    config = TriAttentionConfig(
        kv_budget=16,
        divide_length=4,
        window_size=1,
        pruning_mode="per_head",
        score_aggregation="mean",
        use_triton_scoring=True,
        disable_mlr=False,
    )
    return key_states, head_stats, omega, offsets, freq_scale_sq, config


def test_compute_scores_triton_uses_trig_cache_when_aligned():
    import triattention.kernels.triton_scoring as kernel_module

    captured = {}
    old_kernel = kernel_module.speckv_scoring

    def _fake_kernel(**kwargs):
        captured["trig_cache"] = kwargs.get("trig_cache")
        captured["trig_values"] = kwargs.get("trig_values")
        k = kwargs["K_rot"]
        return torch.zeros((k.shape[0], k.shape[1], k.shape[2]), dtype=torch.float32)

    kernel_module.speckv_scoring = _fake_kernel
    try:
        key_states, head_stats, omega, offsets, freq_scale_sq, config = _build_inputs()
        trig_cache = _FakeTrigCache(
            divide_length=4,
            num_positions=32,
            num_offsets=int(offsets.shape[0]),
            freq_count=int(omega.shape[0]),
        )
        compute_scores_triton(
            key_states=key_states,
            cache_positions=None,
            head_stats=head_stats,
            omega=omega,
            offsets=offsets,
            freq_scale_sq=freq_scale_sq,
            config=config,
            round_start=8,
            trig_cache=trig_cache,
        )
    finally:
        kernel_module.speckv_scoring = old_kernel

    assert captured["trig_cache"] is trig_cache
    assert captured["trig_values"] is None


def test_compute_scores_triton_uses_shifted_trig_values_when_unaligned():
    import triattention.kernels.triton_scoring as kernel_module

    captured = {}
    old_kernel = kernel_module.speckv_scoring

    def _fake_kernel(**kwargs):
        captured["trig_cache"] = kwargs.get("trig_cache")
        captured["trig_values"] = kwargs.get("trig_values")
        k = kwargs["K_rot"]
        return torch.zeros((k.shape[0], k.shape[1], k.shape[2]), dtype=torch.float32)

    kernel_module.speckv_scoring = _fake_kernel
    try:
        key_states, head_stats, omega, offsets, freq_scale_sq, config = _build_inputs()
        trig_cache = _FakeTrigCache(
            divide_length=4,
            num_positions=32,
            num_offsets=int(offsets.shape[0]),
            freq_count=int(omega.shape[0]),
        )
        compute_scores_triton(
            key_states=key_states,
            cache_positions=None,
            head_stats=head_stats,
            omega=omega,
            offsets=offsets,
            freq_scale_sq=freq_scale_sq,
            config=config,
            round_start=10,
            trig_cache=trig_cache,
        )
    finally:
        kernel_module.speckv_scoring = old_kernel

    assert captured["trig_cache"] is None
    trig_values = captured["trig_values"]
    assert isinstance(trig_values, tuple)
    assert len(trig_values) == 2
    assert tuple(trig_values[0].shape) == (2, 2)
    assert tuple(trig_values[1].shape) == (2, 2)


def test_compute_scores_triton_falls_back_when_round_start_out_of_cache_range():
    import triattention.kernels.triton_scoring as kernel_module

    captured = {}
    old_kernel = kernel_module.speckv_scoring

    def _fake_kernel(**kwargs):
        captured["trig_cache"] = kwargs.get("trig_cache")
        captured["trig_values"] = kwargs.get("trig_values")
        k = kwargs["K_rot"]
        return torch.zeros((k.shape[0], k.shape[1], k.shape[2]), dtype=torch.float32)

    kernel_module.speckv_scoring = _fake_kernel
    try:
        key_states, head_stats, omega, offsets, freq_scale_sq, config = _build_inputs()
        trig_cache = _FakeTrigCache(
            divide_length=4,
            num_positions=2,
            num_offsets=int(offsets.shape[0]),
            freq_count=int(omega.shape[0]),
        )
        # max_round_start = 8, use 33 to force fallback path.
        compute_scores_triton(
            key_states=key_states,
            cache_positions=None,
            head_stats=head_stats,
            omega=omega,
            offsets=offsets,
            freq_scale_sq=freq_scale_sq,
            config=config,
            round_start=33,
            trig_cache=trig_cache,
        )
    finally:
        kernel_module.speckv_scoring = old_kernel

    assert captured["trig_cache"] is None
    assert captured["trig_values"] is None

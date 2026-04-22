"""Frequency statistics loader for TriAttention sglang integration.

Loads precomputed per-layer, per-head query frequency statistics from
a ``.pt`` file and validates them against the target model's
configuration.  The loaded stats are consumed by the scoring pipeline
to evaluate KV token importance without running full attention.

The heavy lifting (parsing two file formats — TriAttention native and
R-KV legacy — and GQA head aggregation) is delegated to
``triattention.vllm.core.utils.load_frequency_stats``.  This module
adds:

* Environment-variable-driven path resolution.
* Model-dimension validation (num_layers, num_kv_heads, head_dim).
* A simple ``StatsBundle`` container that scheduler / worker hooks
  can hold onto after init.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable for stats file path
# ---------------------------------------------------------------------------

ENV_STATS_PATH: str = "TRIATTN_RUNTIME_SPARSE_STATS_PATH"


class StatsBundle:
    """Immutable container for loaded and validated frequency statistics.

    Note: Stats are stored at **attention-head** granularity (e.g. 64
    heads for Llama-3-8B) rather than KV-head granularity (8 heads).
    This matches the HF reference which scores each attention head
    independently, then aggregates per-KV-head via max.

    Attributes:
        metadata: Model/calibration metadata extracted from the stats
            file (head_dim, num_kv_heads, num_layers, rope_style, ...).
        head_stats: Per-layer dict of per-attention-head tensors.  Key is
            ``layer_idx`` (int), value is a dict with at least:

            * ``q_mean_complex`` -- ``[num_attention_heads, freq_count, 2]``
            * ``freq_scale_sq``  -- ``[freq_count]`` (shared across heads)
            * ``q_abs_mean``     -- ``[num_attention_heads, freq_count]``

        omega: Rotary base frequencies ``[freq_count]`` (== ``inv_freq``
            from model or stats metadata).  ``None`` if derivation
            failed -- the caller must supply omega from another source.
        num_layers_with_stats: How many layers actually have stats.
        num_attention_heads: Total Q/attention heads (e.g. 64).
        gqa_group_size: num_attention_heads // num_kv_heads (e.g. 8).
    """

    def __init__(
        self,
        metadata: Dict[str, Any],
        head_stats: Dict[int, Dict[str, torch.Tensor]],
        omega: Optional[torch.Tensor],
        num_attention_heads: int = 0,
        gqa_group_size: int = 1,
    ) -> None:
        self.metadata = metadata
        self.head_stats = head_stats
        self.omega = omega
        self.num_layers_with_stats = len(head_stats)
        # expose attention-head granularity info
        self.num_attention_heads = num_attention_heads
        self.gqa_group_size = gqa_group_size

    @property
    def head_dim(self) -> int:
        return int(self.metadata.get("head_dim", 0))

    @property
    def num_kv_heads(self) -> int:
        return int(self.metadata.get("num_kv_heads", 0))

    @property
    def num_layers(self) -> int:
        return int(self.metadata.get("num_layers", 0))

    @property
    def freq_count(self) -> int:
        return self.head_dim // 2

    @property
    def rope_style(self) -> str:
        return str(self.metadata.get("rope_style", "half"))


def load_stats(
    stats_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
    num_kv_heads: Optional[int] = None,
) -> StatsBundle:
    """Load frequency statistics from a ``.pt`` file.

    Note: This function loads stats at **attention-head** granularity
    (e.g. 64 heads) without GQA mean-pooling.  The previous implementation
    delegated to ``load_frequency_stats`` which mean-pooled 64 attention
    heads down to 8 KV heads at load time, destroying per-head signal.

    The HF reference (``pruning_utils.py:load_head_frequency_stats``)
    keeps all 64 attention heads and only aggregates (via max) after
    per-head scoring.  This function now mirrors that behavior.

    Resolution order for the file path:

    1. Explicit ``stats_path`` argument.
    2. ``TRIATTN_RUNTIME_SPARSE_STATS_PATH`` environment variable.

    Args:
        stats_path: Override path.  When ``None``, falls back to the
            environment variable.
        device: Target device for loaded tensors.
        dtype: Target dtype for loaded tensors (omega is always
            float32 regardless of this setting).
        num_kv_heads: Number of KV heads in the target model.  Used to
            compute ``gqa_group_size`` but NOT for aggregation.

    Returns:
        A validated :class:`StatsBundle`.

    Raises:
        FileNotFoundError: If no stats file is found.
        ValueError: If the stats file is structurally invalid.
    """
    # --- resolve path ---
    if stats_path is None:
        stats_path = os.environ.get(ENV_STATS_PATH)
    if stats_path is None:
        raise FileNotFoundError(
            f"No stats file specified.  Set {ENV_STATS_PATH} or "
            f"pass stats_path explicitly."
        )

    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")

    logger.info("Loading frequency stats from %s (HF-aligned, no GQA pool)", path)

    # Load raw stats directly, preserving all attention heads.
    # This replaces the previous delegation to load_frequency_stats()
    # which applied mean-pool from num_attention_heads to num_kv_heads.
    payload = torch.load(path, map_location=device)

    # --- detect format (R-KV per-head or TriAttention native) ---
    rkv_metadata = payload.get("metadata", {})
    rkv_stats_raw: Dict[str, Dict[str, torch.Tensor]] = payload.get("stats", {})

    # Infer num_layers and num_attention_heads from per-head keys
    layer_nums: set = set()
    head_nums: set = set()
    for key in rkv_stats_raw.keys():
        if key.startswith("layer") and "_head" in key:
            parts = key.split("_")
            if len(parts) == 2:
                layer_nums.add(int(parts[0].replace("layer", "")))
                head_nums.add(int(parts[1].replace("head", "")))

    num_layers = len(layer_nums)
    num_attention_heads = len(head_nums)
    head_dim = rkv_metadata.get("head_dim", 128)
    freq_count = head_dim // 2

    # Compute GQA group size (do NOT aggregate)
    effective_num_kv_heads = num_kv_heads if num_kv_heads else num_attention_heads
    gqa_group_size = max(1, num_attention_heads // effective_num_kv_heads)

    # --- derive inv_freq / omega ---
    omega: Optional[torch.Tensor] = None
    inv_freq_raw = rkv_metadata.get("inv_freq")
    if isinstance(inv_freq_raw, torch.Tensor):
        omega = inv_freq_raw.to(device=device, dtype=torch.float32)
        omega = omega[:freq_count].contiguous()
    elif isinstance(inv_freq_raw, (list, tuple)):
        omega = torch.tensor(inv_freq_raw, device=device, dtype=torch.float32)
        omega = omega[:freq_count].contiguous()
    else:
        # Fallback: derive from model config
        model_id = rkv_metadata.get("model_name", rkv_metadata.get("model_path"))
        if model_id:
            try:
                from transformers import AutoConfig
                from triattention.common.rope_utils import build_rotary
                model_config = AutoConfig.from_pretrained(
                    str(model_id), trust_remote_code=True,
                )
                rotary = build_rotary(
                    cache_device=device,
                    model_path=Path(str(model_id)),
                    dtype=dtype,
                    config=model_config,
                )
                inv_freq = getattr(rotary, "inv_freq", None)
                if isinstance(inv_freq, torch.Tensor):
                    omega = inv_freq.to(device=device, dtype=torch.float32)[
                        :freq_count
                    ].contiguous()
            except Exception:
                pass

    # --- derive freq_scale_sq (shared across all heads, like HF) ---
    freq_scale_sq: Optional[torch.Tensor] = None
    model_id = rkv_metadata.get("model_name", rkv_metadata.get("model_path"))
    if model_id:
        try:
            from transformers import AutoConfig
            from triattention.common.rope_utils import (
                build_rotary,
                compute_frequency_scaling,
            )
            model_config = AutoConfig.from_pretrained(
                str(model_id), trust_remote_code=True,
            )
            rotary = build_rotary(
                cache_device=device,
                model_path=Path(str(model_id)),
                dtype=dtype,
                config=model_config,
            )
            freq_scale = compute_frequency_scaling(
                rotary=rotary,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            ).to(device=device, dtype=torch.float32)
            freq_scale_sq = freq_scale.pow(2)  # [freq_count]
        except Exception:
            pass
    if freq_scale_sq is None:
        freq_scale_sq = torch.ones(freq_count, device=device, dtype=torch.float32)

    # --- build per-layer stats at attention-head granularity ---
    # Each layer stores tensors with dim-0 = num_attention_heads
    # (e.g. 64), NOT num_kv_heads (8).  No mean-pooling.
    head_stats: Dict[int, Dict[str, torch.Tensor]] = {}

    for layer_idx in sorted(layer_nums):
        all_q_mean_real = []
        all_q_mean_imag = []
        all_q_abs_mean = []

        for head_idx in sorted(head_nums):
            key = f"layer{layer_idx:02d}_head{head_idx:02d}"
            if key in rkv_stats_raw:
                head_data = rkv_stats_raw[key]
                if "q_abs_mean" in head_data:
                    all_q_abs_mean.append(
                        head_data["q_abs_mean"].to(device=device, dtype=torch.float32)
                    )
                else:
                    all_q_abs_mean.append(
                        torch.ones(freq_count, device=device, dtype=torch.float32)
                    )
                if "q_mean_real" in head_data and "q_mean_imag" in head_data:
                    all_q_mean_real.append(
                        head_data["q_mean_real"].to(device=device, dtype=torch.float32)
                    )
                    all_q_mean_imag.append(
                        head_data["q_mean_imag"].to(device=device, dtype=torch.float32)
                    )

        # Stack all attention heads WITHOUT GQA aggregation.
        # Shape: [num_attention_heads, freq_count]
        q_abs_mean_stacked = torch.stack(all_q_abs_mean, dim=0)

        layer_entry: Dict[str, torch.Tensor] = {
            "freq_scale_sq": freq_scale_sq.clone(),  # [freq_count] (shared)
            "q_abs_mean": q_abs_mean_stacked,  # [num_attention_heads, freq_count]
        }

        if all_q_mean_real and all_q_mean_imag:
            q_mean_real_stacked = torch.stack(all_q_mean_real, dim=0)
            q_mean_imag_stacked = torch.stack(all_q_mean_imag, dim=0)
            # [num_attention_heads, freq_count, 2]
            layer_entry["q_mean_complex"] = torch.stack(
                [q_mean_real_stacked, q_mean_imag_stacked], dim=-1
            )

        head_stats[layer_idx] = layer_entry

    # --- build metadata ---
    metadata: Dict[str, Any] = {
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": effective_num_kv_heads,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "rope_style": rkv_metadata.get("rope_style", "half"),
        "rope_type": rkv_metadata.get("rope_type"),
        "rope_theta": rkv_metadata.get("rope_theta", 10000.0),
        "gqa_group_size": gqa_group_size,
        "rkv_metadata": rkv_metadata,
    }
    if omega is not None:
        metadata["inv_freq"] = omega

    bundle = StatsBundle(
        metadata=metadata,
        head_stats=head_stats,
        omega=omega,
        num_attention_heads=num_attention_heads,
        gqa_group_size=gqa_group_size,
    )

    logger.info(
        "Stats loaded (HF-aligned): %d layers, head_dim=%d, "
        "num_attention_heads=%d, num_kv_heads=%d, gqa_group_size=%d, "
        "rope_style=%s, omega=%s",
        bundle.num_layers_with_stats,
        bundle.head_dim,
        num_attention_heads,
        effective_num_kv_heads,
        gqa_group_size,
        bundle.rope_style,
        "present" if omega is not None else "MISSING",
    )

    return bundle


def validate_stats_against_model(
    bundle: StatsBundle,
    model_num_layers: int,
    model_num_kv_heads: int,
    model_head_dim: int,
) -> None:
    """Verify that loaded stats are compatible with the target model.

    Mismatched stats produce silently wrong scores (GIGO), so this
    check should be called at init time before any compression runs.

    Args:
        bundle: Loaded stats bundle.
        model_num_layers: Number of transformer layers in the model.
        model_num_kv_heads: Number of KV heads per layer.
        model_head_dim: Dimension of each attention head.

    Raises:
        ValueError: If any dimension mismatches.
    """
    errors = []

    if bundle.head_dim != model_head_dim:
        errors.append(
            f"head_dim mismatch: stats={bundle.head_dim}, "
            f"model={model_head_dim}"
        )

    # GQA head count validation.
    # Stats num_kv_heads may equal num_attention_heads (Q heads) due to
    # naming in stats generation, or it may already be GQA-aggregated to
    # match model KV heads.  Accept if stats heads == model KV heads
    # (aggregated) OR stats heads is an integer multiple of model KV heads
    # (un-aggregated Q heads, valid for GQA).
    stats_heads = bundle.num_kv_heads
    if stats_heads != model_num_kv_heads:
        if stats_heads == 0 or model_num_kv_heads == 0 or stats_heads % model_num_kv_heads != 0:
            errors.append(
                f"num_kv_heads mismatch: stats={stats_heads}, "
                f"model={model_num_kv_heads} "
                f"(stats heads must equal or be a multiple of model KV heads)"
            )
        else:
            gqa_ratio = stats_heads // model_num_kv_heads
            logger.info(
                "Stats have %d heads vs model %d KV heads "
                "(GQA ratio %d) — accepted.",
                stats_heads, model_num_kv_heads, gqa_ratio,
            )

    # Stats may cover only a subset of layers (sparse layer handling).
    # But if stats claim more layers than the model has, something is
    # definitely wrong.
    if bundle.num_layers > model_num_layers:
        errors.append(
            f"num_layers in stats ({bundle.num_layers}) exceeds "
            f"model layers ({model_num_layers})"
        )

    # Validate per-layer tensor shapes.
    # Stats are now at attention-head granularity (e.g. 64 heads).
    # Accept if tensor dim-0 == model_num_kv_heads (aggregated) or is
    # a multiple of model_num_kv_heads (attention-head granularity).
    for layer_idx, layer_data in bundle.head_stats.items():
        q_mean = layer_data.get("q_mean_complex")
        if q_mean is not None:
            actual_heads = q_mean.shape[0]
            expected_freq = model_head_dim // 2
            if actual_heads != model_num_kv_heads and (
                actual_heads == 0 or model_num_kv_heads == 0
                or actual_heads % model_num_kv_heads != 0
            ):
                errors.append(
                    f"layer {layer_idx} q_mean_complex head dim "
                    f"{actual_heads} incompatible with model KV heads "
                    f"{model_num_kv_heads}"
                )
            if q_mean.shape[-2] != expected_freq or q_mean.shape[-1] != 2:
                errors.append(
                    f"layer {layer_idx} q_mean_complex shape "
                    f"{tuple(q_mean.shape)} has wrong freq/complex dims "
                    f"(expected [*, {expected_freq}, 2])"
                )

        freq_sq = layer_data.get("freq_scale_sq")
        if freq_sq is not None:
            # freq_scale_sq can be 1D [freq_count] (shared)
            # or 2D [num_heads, freq_count].
            expected_freq = model_head_dim // 2
            if freq_sq.dim() == 1:
                if freq_sq.shape[0] != expected_freq:
                    errors.append(
                        f"layer {layer_idx} freq_scale_sq dim "
                        f"{freq_sq.shape[0]} != expected {expected_freq}"
                    )
            else:
                actual_heads = freq_sq.shape[0]
                if actual_heads != model_num_kv_heads and (
                    actual_heads == 0 or model_num_kv_heads == 0
                    or actual_heads % model_num_kv_heads != 0
                ):
                    errors.append(
                        f"layer {layer_idx} freq_scale_sq head dim "
                        f"{actual_heads} incompatible with model KV heads "
                        f"{model_num_kv_heads}"
                    )
                if freq_sq.shape[-1] != expected_freq:
                    errors.append(
                        f"layer {layer_idx} freq_scale_sq freq dim "
                        f"{freq_sq.shape[-1]} != expected {expected_freq}"
                    )
        # Only check the first layer with stats to avoid verbose output.
        break

    if errors:
        raise ValueError(
            "Stats / model dimension mismatch:\n  "
            + "\n  ".join(errors)
        )

    if bundle.num_layers_with_stats < model_num_layers:
        logger.warning(
            "Stats cover %d of %d model layers.  Layers without "
            "stats will receive zero importance scores (no eviction "
            "guidance for those layers).",
            bundle.num_layers_with_stats,
            model_num_layers,
        )

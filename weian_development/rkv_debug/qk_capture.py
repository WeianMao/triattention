"""
Lightweight side-channel Q/K capture for debugging KV compression runs.

Design goals:
- No behavior change to generation logic; capture is best-effort and guarded by an explicit
  activation call. If anything fails during capture, the exception is swallowed and capture
  is disabled for the current context.
- Storage layout is per sample: <root>/shardXX/runYYY_sampleZZZZZ/{qk_layerNN.pt, metadata.json}.
- Only prefill-stage tensors are saved (i.e., when past_key_value is None), one file per layer.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Set

import torch


@dataclass
class CaptureState:
    root: Path
    shard_id: int
    run_id: int
    sample_id: int
    prefill_length: int
    model_info: Dict[str, object] = field(default_factory=dict)
    captured_layers: Set[int] = field(default_factory=set)
    disabled: bool = False

    def sample_dir(self) -> Path:
        return self.root / f"shard{self.shard_id:02d}" / f"run{self.run_id:03d}_sample{self.sample_id:05d}"


_STATE: Optional[CaptureState] = None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def activate_capture(
    root: Path,
    *,
    shard_id: int,
    run_id: int,
    sample_id: int,
    prefill_length: int,
    model_info: Optional[Dict[str, object]] = None,
) -> None:
    """
    Enable capture for the current sample. Subsequent calls replace the previous state.
    """
    global _STATE
    model_info = model_info or {}
    root = root.resolve()
    _STATE = CaptureState(
        root=root,
        shard_id=shard_id,
        run_id=run_id,
        sample_id=sample_id,
        prefill_length=prefill_length,
        model_info=model_info,
    )
    target = _STATE.sample_dir()
    _ensure_dir(target)
    meta = {
        "shard_id": shard_id,
        "run_id": run_id,
        "sample_id": sample_id,
        "prefill_length": prefill_length,
        **model_info,
    }
    try:
        (target / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    except Exception as exc:  # pragma: no cover - best effort
        sys.stderr.write(f"[qk_capture] failed to write metadata: {exc}\n")
        _STATE.disabled = True


def deactivate_capture() -> None:
    """Clear any active capture context."""
    global _STATE
    _STATE = None


def _should_skip(layer_idx: int, past_key_value) -> bool:
    if _STATE is None or _STATE.disabled:
        return True
    # Only capture during prefill (no past cache yet).
    if past_key_value is not None:
        return True
    if layer_idx in _STATE.captured_layers:
        return True
    return False


def maybe_capture_qk(layer_idx: int, query_states: torch.Tensor, key_states: torch.Tensor, past_key_value) -> None:
    """
    Store pre-RKV-compression Q/K for the current layer (prefill only).
    """
    global _STATE
    if _should_skip(layer_idx, past_key_value):
        return
    assert _STATE is not None  # guarded by _should_skip

    target = _STATE.sample_dir() / f"qk_layer{layer_idx:02d}.pt"
    try:
        payload = {
            "q": query_states.detach().to("cpu"),
            "k": key_states.detach().to("cpu"),
        }
        torch.save(payload, target)
        _STATE.captured_layers.add(layer_idx)
    except Exception as exc:  # pragma: no cover - best effort
        sys.stderr.write(f"[qk_capture] failed to capture layer {layer_idx}: {exc}\n")
        _STATE.disabled = True


def parse_sample_filter(value: Optional[str]) -> Optional[Set[int]]:
    if not value:
        return None
    samples: Set[int] = set()
    for token in value.replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        try:
            samples.add(int(token))
        except ValueError:
            continue
    return samples


def capture_requested_for_sample(sample_id: int, env_var: str = "RKV_QK_CAPTURE_SAMPLES") -> bool:
    """
    Returns True if no filter is provided, or if sample_id is listed in env_var.
    """
    filt = parse_sample_filter(os.environ.get(env_var))
    if filt is None:
        return True
    return sample_id in filt


def patch_llama_attention_for_capture() -> bool:
    """
    Patch HF LlamaAttention.forward to mirror the stock implementation while calling maybe_capture_qk.

    This is intended for SpeckV/fullkv runs that skip the R-KV monkeypatch but still need Q/K dumps.
    Returns True if the patch is applied, False otherwise.
    """
    try:
        from transformers.models.llama import modeling_llama
    except Exception as exc:  # pragma: no cover - best effort
        sys.stderr.write(f"[qk_capture] unable to import modeling_llama for capture: {exc}\n")
        return False

    cls = modeling_llama.LlamaAttention
    if getattr(cls, "_rkv_qk_capture_patched", False):
        return True

    apply_rotary_pos_emb = modeling_llama.apply_rotary_pos_emb
    eager_attention_forward: Callable = modeling_llama.eager_attention_forward
    ALL_ATTENTION_FUNCTIONS = modeling_llama.ALL_ATTENTION_FUNCTIONS

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        maybe_capture_qk(getattr(self, "layer_idx", -1), query_states, key_states, past_key_value)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    cls.forward = forward
    cls._rkv_qk_capture_patched = True
    return True

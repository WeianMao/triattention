"""Debug-only first-layer attention capture utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from vllm.forward_context import get_forward_context


def _append_debug_event(dump_dir: Path, payload: dict[str, Any]) -> None:
    debug_path = dump_dir / "debug_events.jsonl"
    with debug_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _tensor_to_list(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().to(device="cpu", dtype=torch.float32).tolist()


def _tensor_summary(tensor: torch.Tensor, *, max_items: int = 64) -> dict[str, Any]:
    cpu = tensor.detach().to(device="cpu")
    flat = cpu.reshape(-1)
    if not cpu.dtype.is_floating_point and int(flat.numel()) <= 4096:
        return {
            "shape": list(cpu.shape),
            "values": flat.to(dtype=torch.int64).tolist(),
        }
    if cpu.dtype.is_floating_point:
        sample = flat[:max_items].to(dtype=torch.float32).tolist()
    else:
        sample = flat[:max_items].to(dtype=torch.int64).tolist()
    return {
        "shape": list(cpu.shape),
        "sample": sample,
    }


def _serialize_positions(positions: torch.Tensor) -> dict[str, Any]:
    cpu = positions.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    return {
        "shape": list(positions.shape),
        "values": cpu.tolist(),
    }


def _capture_forward_context(layer_name: str) -> dict[str, Any]:
    try:
        forward_context = get_forward_context()
    except Exception as exc:
        return {"error": f"forward_context_unavailable:{type(exc).__name__}"}

    payload: dict[str, Any] = {}
    attn_metadata = getattr(forward_context, "attn_metadata", None)
    if isinstance(attn_metadata, dict):
        layer_meta = attn_metadata.get(layer_name)
    else:
        layer_meta = None
    if layer_meta is not None:
        for attr in (
            "num_actual_tokens",
            "max_query_len",
            "max_seq_len",
        ):
            value = getattr(layer_meta, attr, None)
            if value is not None:
                payload[attr] = int(value)
        for attr in (
            "seq_lens",
            "seq_lens_cpu",
            "query_start_loc",
            "query_start_loc_cpu",
            "block_table",
            "block_table_tensor",
            "slot_mapping",
        ):
            value = getattr(layer_meta, attr, None)
            if torch.is_tensor(value):
                payload[attr] = _tensor_summary(value)

    slot_mapping = getattr(forward_context, "slot_mapping", None)
    if isinstance(slot_mapping, dict):
        layer_slot_mapping = slot_mapping.get(layer_name)
        if torch.is_tensor(layer_slot_mapping):
            payload["forward_context_slot_mapping"] = _tensor_summary(layer_slot_mapping)
    elif slot_mapping is None:
        payload["forward_context_slot_mapping"] = "missing"
    return payload


def _parse_target_steps() -> list[int]:
    raw = os.environ.get("TRIATTN_DEBUG_CAPTURE_LAYER0_ATTN_STEPS", "").strip()
    if not raw:
        return []
    steps: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            steps.append(int(item))
        except ValueError:
            continue
    return sorted(set(step for step in steps if step >= 0))


def _parse_relative_small_batch_steps() -> list[int]:
    raw = os.environ.get("TRIATTN_DEBUG_CAPTURE_LAYER0_ATTN_REL_SMALL_BATCH_STEPS", "").strip()
    if not raw:
        return []
    steps: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            steps.append(int(item))
        except ValueError:
            continue
    return sorted(set(step for step in steps if step >= 0))


def _resolve_layers(model: Any):
    candidates = [
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "model", None),
        model,
    ]
    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return layers
    return None


def _resolve_model_from_worker(worker: Any) -> Any | None:
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None:
        return None
    getter = getattr(model_runner, "get_model", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    return getattr(model_runner, "model", None)


def install_layer0_attention_capture(worker: Any) -> bool:
    dump_dir_raw = os.environ.get("TRIATTN_DEBUG_CAPTURE_LAYER0_ATTN_DIR", "").strip()
    target_steps = _parse_target_steps()
    relative_small_batch_steps = _parse_relative_small_batch_steps()
    if not dump_dir_raw or (not target_steps and not relative_small_batch_steps):
        return False
    if getattr(worker, "_triattn_layer0_attn_capture_installed", False):
        return True

    dump_dir = Path(dump_dir_raw)
    dump_dir.mkdir(parents=True, exist_ok=True)
    _append_debug_event(
        dump_dir,
        {
            "event": "install_attempt",
            "worker_type": type(worker).__name__,
        },
    )
    model = _resolve_model_from_worker(worker)
    if model is None:
        _append_debug_event(dump_dir, {"event": "install_failed", "reason": "model_none"})
        return False
    layers = _resolve_layers(model)
    if not layers:
        _append_debug_event(dump_dir, {"event": "install_failed", "reason": "layers_missing"})
        return False
    self_attn = getattr(layers[0], "self_attn", None)
    if self_attn is None:
        _append_debug_event(dump_dir, {"event": "install_failed", "reason": "self_attn_missing"})
        return False
    state = {
        "decode_step": -1,
        "target_steps": set(target_steps),
        "relative_small_batch_steps": set(relative_small_batch_steps),
        "captured_steps": set(),
        "seen_calls": 0,
        "active_step": None,
        "small_batch_index": -1,
        "active_relative_small_batch_step": None,
    }

    orig_self_attn_forward = getattr(self_attn, "forward")
    if not getattr(self_attn, "_triattn_debug_forward_wrapped", False):
        def _wrapped_self_attn_forward(positions: torch.Tensor, hidden_states: torch.Tensor):
            state["seen_calls"] += 1
            step = int(state["seen_calls"])
            targeted = step in state["target_steps"]
            state["active_step"] = step if targeted else None
            relative_small_batch_target = None
            if hidden_states.ndim >= 2 and int(hidden_states.shape[0]) <= 16:
                state["small_batch_index"] += 1
                relative_index = int(state["small_batch_index"])
                if relative_index in state["relative_small_batch_steps"]:
                    relative_small_batch_target = relative_index
                    state["active_relative_small_batch_step"] = relative_index
            if targeted:
                qkv, _ = self_attn.qkv_proj(hidden_states)
                q, k, v = qkv.split([self_attn.q_size, self_attn.kv_size, self_attn.kv_size], dim=-1)
                total_tokens = q.shape[0]
                if hasattr(self_attn, "q_norm") and hasattr(self_attn, "k_norm"):
                    q = self_attn.q_norm(
                        q.view(total_tokens, self_attn.num_heads, self_attn.head_dim)
                    ).view(total_tokens, self_attn.q_size)
                    k = self_attn.k_norm(
                        k.view(total_tokens, self_attn.num_kv_heads, self_attn.head_dim)
                    ).view(total_tokens, self_attn.kv_size)
                q_rot, k_rot = self_attn.rotary_emb(positions, q, k)
                payload = {
                    "seen_call": step,
                    "positions": _serialize_positions(positions),
                    "hidden_states": _tensor_to_list(hidden_states[:1]),
                    "q_pre": _tensor_to_list(q[:1]),
                    "k_pre": _tensor_to_list(k[:1]),
                    "v_pre": _tensor_to_list(v[:1]),
                    "q_post": _tensor_to_list(q_rot[:1]),
                    "k_post": _tensor_to_list(k_rot[:1]),
                }
                (dump_dir / f"layer0_self_attn_state_{step:04d}.json").write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
            elif relative_small_batch_target is not None:
                qkv, _ = self_attn.qkv_proj(hidden_states)
                q, k, v = qkv.split([self_attn.q_size, self_attn.kv_size, self_attn.kv_size], dim=-1)
                total_tokens = q.shape[0]
                if hasattr(self_attn, "q_norm") and hasattr(self_attn, "k_norm"):
                    q = self_attn.q_norm(
                        q.view(total_tokens, self_attn.num_heads, self_attn.head_dim)
                    ).view(total_tokens, self_attn.q_size)
                    k = self_attn.k_norm(
                        k.view(total_tokens, self_attn.num_kv_heads, self_attn.head_dim)
                    ).view(total_tokens, self_attn.kv_size)
                q_rot, k_rot = self_attn.rotary_emb(positions, q, k)
                payload = {
                    "seen_call": step,
                    "relative_small_batch_index": int(relative_small_batch_target),
                    "positions": _serialize_positions(positions),
                    "hidden_states": _tensor_to_list(hidden_states[:1]),
                    "q_pre": _tensor_to_list(q[:1]),
                    "k_pre": _tensor_to_list(k[:1]),
                    "v_pre": _tensor_to_list(v[:1]),
                    "q_post": _tensor_to_list(q_rot[:1]),
                    "k_post": _tensor_to_list(k_rot[:1]),
                }
                (dump_dir / f"layer0_self_attn_state_rel_small_batch_{int(relative_small_batch_target):04d}.json").write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
            try:
                output = orig_self_attn_forward(positions, hidden_states)
                current_step = state.get("active_step")
                current_relative_step = state.get("active_relative_small_batch_step")
                capture_id = current_step if current_step is not None else current_relative_step
                if capture_id is not None and torch.is_tensor(output):
                    payload = {
                        "seen_call": int(capture_id),
                        "self_attn_out": _tensor_to_list(output[:1]),
                    }
                    suffix = f"{int(capture_id):04d}"
                    if current_relative_step is not None and current_step is None:
                        suffix = f"rel_small_batch_{int(current_relative_step):04d}"
                        payload["relative_small_batch_index"] = int(current_relative_step)
                    (dump_dir / f"layer0_self_attn_output_{suffix}.json").write_text(
                        json.dumps(payload, ensure_ascii=False),
                        encoding="utf-8",
                    )
                return output
            finally:
                if state["active_step"] == step:
                    state["active_step"] = None
                if relative_small_batch_target is not None and state["active_relative_small_batch_step"] == relative_small_batch_target:
                    state["active_relative_small_batch_step"] = None

        self_attn.forward = _wrapped_self_attn_forward
        self_attn._triattn_debug_forward_wrapped = True

    attn_module = getattr(self_attn, "attn", None)
    orig_attn_forward = getattr(attn_module, "forward", None)
    if attn_module is not None and callable(orig_attn_forward) and not getattr(attn_module, "_triattn_debug_forward_wrapped", False):
        layer_name = getattr(attn_module, "layer_name", "layer0_attn")

        def _wrapped_attn_forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, output_shape: torch.Size | None = None):
            step = state.get("active_step")
            relative_step = state.get("active_relative_small_batch_step")
            current_step = step if step is not None else relative_step
            if current_step in state["target_steps"] or current_step is not None:
                payload = {
                    "seen_call": int(current_step),
                    "query_in": _tensor_to_list(query[:1]),
                    "key_in": _tensor_to_list(key[:1]),
                    "value_in": _tensor_to_list(value[:1]),
                    "forward_context": _capture_forward_context(layer_name),
                }
                suffix = f"{int(current_step):04d}"
                if relative_step is not None and step is None:
                    suffix = f"rel_small_batch_{int(relative_step):04d}"
                    payload["relative_small_batch_index"] = int(relative_step)
                (dump_dir / f"layer0_attn_inputs_{suffix}.json").write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
            output = orig_attn_forward(query, key, value, output_shape)
            step = state.get("active_step")
            relative_step = state.get("active_relative_small_batch_step")
            current_step = step if step is not None else relative_step
            if current_step is not None and torch.is_tensor(output):
                payload = {
                    "seen_call": int(current_step),
                    "attn_output": _tensor_to_list(output[:1]),
                }
                suffix = f"{int(current_step):04d}"
                if relative_step is not None and step is None:
                    suffix = f"rel_small_batch_{int(relative_step):04d}"
                    payload["relative_small_batch_index"] = int(relative_step)
                (dump_dir / f"layer0_attn_output_{suffix}.json").write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
            return output

        attn_module.forward = _wrapped_attn_forward
        attn_module._triattn_debug_forward_wrapped = True

    def _hook(_module: Any, _args: Any, output: Any) -> None:
        step = int(state["seen_calls"])
        tensor = output[0] if isinstance(output, (tuple, list)) and output else output
        if not torch.is_tensor(tensor):
            if step <= 8:
                _append_debug_event(
                    dump_dir,
                    {
                        "event": "hook_non_tensor",
                        "seen_calls": step,
                        "output_type": type(output).__name__,
                    },
                )
            return
        if step <= 8:
            _append_debug_event(
                dump_dir,
                {
                    "event": "hook_tensor",
                    "seen_calls": step,
                    "shape": list(tensor.shape),
                    "ndim": int(tensor.ndim),
                },
            )
        elif step % 100 == 0:
            _append_debug_event(
                dump_dir,
                {
                    "event": "hook_tensor_periodic",
                    "seen_calls": step,
                    "shape": list(tensor.shape),
                    "ndim": int(tensor.ndim),
                },
            )
        if tensor.ndim == 3:
            flat = tensor.reshape(-1, tensor.shape[-1])
        elif tensor.ndim == 2:
            flat = tensor
        else:
            return
        relative_index = None
        if int(flat.shape[0]) <= 16:
            relative_index = int(state["small_batch_index"])
            _append_debug_event(
                dump_dir,
                {
                    "event": "hook_small_batch",
                    "seen_calls": step,
                    "relative_small_batch_index": relative_index,
                    "shape": list(flat.shape),
                },
            )
        should_capture_absolute = step in state["target_steps"]
        should_capture_relative = int(flat.shape[0]) <= 16 and relative_index in state["relative_small_batch_steps"]
        if not should_capture_absolute and not should_capture_relative:
            return
        capture_key = f"abs:{step}" if should_capture_absolute else f"rel:{relative_index}"
        if capture_key in state["captured_steps"]:
            return
        state["captured_steps"].add(capture_key)
        payload = {
            "decode_step": step,
            "shape": list(flat.shape),
            "values": flat.detach().to(device="cpu", dtype=torch.float32).tolist(),
        }
        suffix = f"{step:04d}"
        if should_capture_relative and not should_capture_absolute:
            suffix = f"rel_small_batch_{relative_index:04d}"
            payload["relative_small_batch_index"] = relative_index
        (dump_dir / f"decode_step_{suffix}.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    worker._triattn_layer0_attn_capture_handle = self_attn.register_forward_hook(_hook)
    worker._triattn_layer0_attn_capture_installed = True
    _append_debug_event(
        dump_dir,
        {
            "event": "install_success",
            "target_steps": sorted(state["target_steps"]),
            "relative_small_batch_steps": sorted(state["relative_small_batch_steps"]),
        },
    )
    return True

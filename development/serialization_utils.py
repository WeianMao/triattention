"""Utility helpers for high-fidelity DeepConf serialization."""
from __future__ import annotations

import gzip
from typing import Any, Dict, List, Tuple

import msgpack
import numpy as np

try:  # Optional acceleration
    import zstandard as zstd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    zstd = None  # type: ignore

try:
    from vllm.logprobs import Logprob as VllmLogprob
except ImportError:  # pragma: no cover - docs or tooling
    VllmLogprob = None  # type: ignore

TYPE_KEY = "__dc_type__"
MAGIC = b"DCMP"
VERSION = 1
METHOD_GZIP = b"G"
METHOD_ZSTD = b"Z"
METHOD_NONE = b"N"


class SerializationError(RuntimeError):
    """Raised when serialization fails due to unsupported structure."""


def _normalize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, np.generic):
        return _normalize(obj.item())

    if isinstance(obj, list):
        return [_normalize(item) for item in obj]

    if isinstance(obj, tuple):
        return {TYPE_KEY: "tuple", "items": [_normalize(item) for item in obj]}

    if isinstance(obj, set):
        return {TYPE_KEY: "set", "items": [_normalize(item) for item in obj]}

    if isinstance(obj, dict):
        return _normalize_dict(obj)

    if VllmLogprob is not None and isinstance(obj, VllmLogprob):
        return {
            TYPE_KEY: "vllm_logprob",
            "logprob": _normalize(obj.logprob),
            "rank": _normalize(obj.rank),
            "decoded_token": _normalize(obj.decoded_token),
        }

    if isinstance(obj, np.ndarray):
        return {
            TYPE_KEY: "np.ndarray",
            "dtype": str(obj.dtype),
            "shape": obj.shape,
            "data": obj.tolist(),
        }

    raise SerializationError(f"Unsupported type for serialization: {type(obj)!r}")


def _normalize_dict(mapping: Dict[Any, Any]) -> Any:
    simple: Dict[Any, Any] = {}
    complex_items: List[Tuple[Any, Any]] = []
    simple_mode = True

    for key, value in mapping.items():
        normalized_value = _normalize(value)

        if key is None or isinstance(key, (bool, int, float, str)):
            if simple_mode:
                simple[key] = normalized_value
            else:
                complex_items.append((_normalize(key), normalized_value))
            continue

        if isinstance(key, np.generic):
            scalar_key = _normalize(key.item())
            if simple_mode and (scalar_key is None or isinstance(scalar_key, (bool, int, float, str))):
                simple[scalar_key] = normalized_value
            else:
                if simple_mode:
                    simple_mode = False
                    complex_items.extend((_normalize(k), v) for k, v in simple.items())
                    simple.clear()
                complex_items.append((scalar_key, normalized_value))
            continue

        if simple_mode:
            simple_mode = False
            complex_items.extend((_normalize(k), v) for k, v in simple.items())
            simple.clear()

        complex_items.append((_normalize(key), normalized_value))

    if simple_mode:
        return simple

    return {
        TYPE_KEY: "dict",
        "items": [[key, value] for key, value in complex_items],
    }


def _denormalize(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_denormalize(item) for item in obj]

    if not isinstance(obj, dict):
        return obj

    type_tag = obj.get(TYPE_KEY)
    if not type_tag:
        return {key: _denormalize(value) for key, value in obj.items()}

    if type_tag == "tuple":
        return tuple(_denormalize(item) for item in obj["items"])

    if type_tag == "set":
        return set(_denormalize(item) for item in obj["items"])

    if type_tag == "dict":
        return {
            _denormalize(key): _denormalize(value)
            for key, value in obj["items"]
        }

    if type_tag == "vllm_logprob":
        if VllmLogprob is None:
            raise RuntimeError("vllm must be available to restore logprobs")
        return VllmLogprob(
            logprob=_denormalize(obj["logprob"]),
            rank=_denormalize(obj["rank"]),
            decoded_token=_denormalize(obj["decoded_token"]),
        )

    if type_tag == "np.ndarray":
        arr = np.array(obj["data"], dtype=obj["dtype"])
        return arr.reshape(obj["shape"])

    raise SerializationError(f"Unknown serialized type tag: {type_tag}")


def dump_msgpack(data: Any, path: str, *, compression: str = "gzip", compression_level: int = 3) -> None:
    normalized = _normalize(data)
    packed = msgpack.packb(normalized, use_bin_type=True)

    if compression == "zstd":
        if zstd is None:
            raise SerializationError("Compression 'zstd' requested but python 'zstandard' package is not installed")
        compressed = zstd.ZstdCompressor(level=compression_level).compress(packed)
        method = METHOD_ZSTD
    elif compression == "gzip":
        compressed = gzip.compress(packed, compresslevel=compression_level)
        method = METHOD_GZIP
    elif compression == "none":
        compressed = packed
        method = METHOD_NONE
    else:
        raise SerializationError(f"Unknown compression mode: {compression}")

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(VERSION.to_bytes(1, "big"))
        f.write(method)
        f.write(compressed)


def load_msgpack(path: str) -> Any:
    with open(path, "rb") as f:
        header = f.read(len(MAGIC))
        if header != MAGIC:
            raise SerializationError("Invalid DeepConf message pack header")
        version_byte = f.read(1)
        if not version_byte:
            raise SerializationError("Missing serializer version byte")
        version = int.from_bytes(version_byte, "big")
        if version != VERSION:
            raise SerializationError(f"Unsupported serializer version: {version}")
        method = f.read(1)
        payload = f.read()

    if method == METHOD_ZSTD:
        if zstd is None:
            raise SerializationError("zstd payload encountered but 'zstandard' is unavailable")
        packed = zstd.ZstdDecompressor().decompress(payload)
    elif method == METHOD_GZIP:
        packed = gzip.decompress(payload)
    elif method == METHOD_NONE:
        packed = payload
    else:
        raise SerializationError(f"Unknown compression method byte: {method}")

    unpacked = msgpack.unpackb(packed, raw=False, strict_map_key=False)
    return _denormalize(unpacked)


__all__ = [
    "dump_msgpack",
    "load_msgpack",
    "SerializationError",
]

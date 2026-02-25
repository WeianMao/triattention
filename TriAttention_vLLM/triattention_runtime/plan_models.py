"""Structured plan/result models for V2 selector/layout/reclaim pipeline.

These models are intentionally lightweight and keep hook return compatibility
by exposing `to_dict()` / `to_hook_result_dict()` helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KeepPlan:
    """Logical keep-selection result produced by the HF-aligned selector layer."""

    mode: str
    indices: Any
    semantic: str | None = None

    @classmethod
    def from_selector_result(cls, result: dict[str, Any]) -> "KeepPlan":
        return cls(
            mode=str(result.get("mode", "shared")),
            indices=result.get("indices"),
            semantic=(
                str(result.get("semantic"))
                if result.get("semantic") is not None
                else None
            ),
        )

    @property
    def selection_mode_label(self) -> str:
        if self.semantic:
            return f"{self.mode}:{self.semantic}"
        return self.mode

    def keep_count(self) -> int:
        if self.mode == "per_head":
            if hasattr(self.indices, "ndim") and hasattr(self.indices, "shape"):
                if int(getattr(self.indices, "ndim")) != 2:
                    raise ValueError(
                        "per_head indices tensor must be 2D, "
                        f"got ndim={getattr(self.indices, 'ndim')}"
                    )
                return int(self.indices.shape[1])
            if isinstance(self.indices, list):
                if not self.indices:
                    return 0
                first_row = self.indices[0]
                return len(first_row) if isinstance(first_row, list) else 0
            raise ValueError(
                f"unsupported per_head indices type: {type(self.indices).__name__}"
            )
        if hasattr(self.indices, "numel"):
            return int(self.indices.numel())
        if isinstance(self.indices, list):
            return len(self.indices)
        raise ValueError(f"unsupported shared indices type: {type(self.indices).__name__}")

    def to_selector_result(self) -> dict[str, Any]:
        out = {
            "mode": self.mode,
            "indices": self.indices,
        }
        if self.semantic is not None:
            out["semantic"] = self.semantic
        return out


@dataclass(frozen=True)
class ReclaimGroup:
    gid: int
    block_ids_before: list[int]
    block_ids_after: list[int]
    block_ids_removed: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gid": self.gid,
            "block_ids_before": list(self.block_ids_before),
            "block_ids_after": list(self.block_ids_after),
            "block_ids_removed": list(self.block_ids_removed),
        }


@dataclass(frozen=True)
class ReclaimEvent:
    mode: str
    groups: list[ReclaimGroup]

    def reclaimed_block_count(self) -> int:
        return sum(len(group.block_ids_removed) for group in self.groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "groups": [group.to_dict() for group in self.groups],
        }


@dataclass(frozen=True)
class PlacementPlan:
    """Physical layout/reclaim result produced by the layout engine path."""

    cache_len_after: int
    selector_status: str
    selection_mode: str
    effective_tokens_before: int
    budget_total: int
    recent_unabsorbed_tokens: int | None
    block_reclaim: ReclaimEvent | None = None

    @property
    def reclaimed_block_count(self) -> int:
        if self.block_reclaim is None:
            return 0
        return self.block_reclaim.reclaimed_block_count()

    def to_hook_result_dict(self) -> dict[str, Any]:
        return {
            "applied": True,
            "reason": f"kv_compacted:{self.selection_mode}",
            "cache_len_after": self.cache_len_after,
            "selector_status": self.selector_status,
            "block_reclaim": (
                self.block_reclaim.to_dict() if self.block_reclaim is not None else None
            ),
            "effective_tokens_before": self.effective_tokens_before,
            "budget_total": self.budget_total,
            "reclaimed_block_count": self.reclaimed_block_count,
            "recent_unabsorbed_tokens": self.recent_unabsorbed_tokens,
        }

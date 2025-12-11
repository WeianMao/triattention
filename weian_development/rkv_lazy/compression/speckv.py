"""SpeckV hook wrapper to align with R-KV compression registration."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from weian_development.speckv.rkv_speckv_generate import apply_speckv_generate_patch

__all__ = ["apply_speckv_generate_patch"]

# Re-export SpeckV patch so callers can import from rkv.compression.* like other methods.


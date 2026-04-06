"""Optional JSONL trace sink for runtime debugging (env-gated)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TRACE_PATH = os.environ.get("TRIATTN_RUNTIME_TRACE_PATH", "").strip()
_TRACE_ENABLED = bool(_TRACE_PATH)


def trace_event(event: str, **payload: Any) -> None:
    """Append one JSONL event when trace sink is enabled."""
    if not _TRACE_ENABLED:
        return
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
    }
    row.update(payload)
    try:
        path = Path(_TRACE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")
    except Exception:
        # Debug tracing must never break inference.
        return

"""Process-level helpers (e.g. renaming command line for monitoring tools)."""
from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path

from setproctitle import setproctitle  # type: ignore

PR_SET_NAME = 15
_MAX_COMM_LENGTH = 15  # Linux comm limit is 16 including null terminator


def mask_process_command(label: str = "PD-L1_binder") -> None:
    truncated = label[:_MAX_COMM_LENGTH]

    # Update /proc/self/comm (visible in htop/top)
    try:
        with Path("/proc/self/comm").open("wb") as f:
            f.write(truncated.encode("utf-8", errors="ignore"))
    except Exception:
        pass

    # Use prctl to set process name
    try:
        libc_path = ctypes.util.find_library("c")
        if libc_path:
            libc = ctypes.CDLL(libc_path, use_errno=True)
            libc.prctl(PR_SET_NAME, truncated.encode("utf-8"), 0, 0, 0)
    except Exception:
        pass

    # Update full command/title via setproctitle
    try:
        setproctitle(truncated)
    except Exception:
        pass

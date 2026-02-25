"""Stable lite regression gate for TriAttention V2.

This script avoids environment-dependent pytest plugin hangs by:
1) forcing PYTEST_DISABLE_PLUGIN_AUTOLOAD=1;
2) running targeted modules with per-command timeout;
3) executing smoke test as the final aggregate check.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, timeout_sec: int, env: dict[str, str], cwd: Path) -> None:
    print(f"[gate] {' '.join(cmd)} (timeout={timeout_sec}s)")
    subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=True,
        timeout=timeout_sec,
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    py = sys.executable
    quick_pytest_modules = [
        "tests_runtime/test_hook_impl.py",
        "tests_runtime/test_kv_compaction.py",
        "tests_runtime/test_runner.py",
        "tests_runtime/test_scheduler.py",
        "tests_runtime/test_scoring_trig_cache.py",
    ]
    for mod in quick_pytest_modules:
        _run(
            [py, "-m", "pytest", "-q", mod],
            timeout_sec=300,
            env=env,
            cwd=root,
        )

    _run(
        [py, "tests_runtime/run_smoke.py"],
        timeout_sec=900,
        env=env,
        cwd=root,
    )
    print("[gate] lite regression passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

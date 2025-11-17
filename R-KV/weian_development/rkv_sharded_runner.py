"""Helper to launch shard-aware R-KV HuggingFace evaluation (prefers PD-L1_binder naming)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "R-KV" / "weian_development" / "rkv_sharded_eval.py"

for path in {REPO_ROOT, SCRIPT_PATH.parent}:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from weian_development.process_utils import mask_process_command


def main() -> None:
    mask_process_command("PD-L1_binder")
    sys.argv = [str(SCRIPT_PATH)] + sys.argv[1:]
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")


if __name__ == "__main__":
    main()

"""Helper to launch LazyEviction evaluation with masked process title."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PATH = ROOT
LAZY_DIR = PROJECT_PATH / "LazyEviction"
SHARDED_EVAL_PATH = PROJECT_PATH / "weian_development" / "lazy_eviction_evaluation_sharded.py"

for path in {PROJECT_PATH, LAZY_DIR, SHARDED_EVAL_PATH.parent}:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from weian_development.process_utils import mask_process_command

def main() -> None:
    mask_process_command("PD-L1_binder")
    sys.argv = [str(SHARDED_EVAL_PATH)] + sys.argv[1:]
    runpy.run_path(str(SHARDED_EVAL_PATH), run_name="__main__")


if __name__ == "__main__":
    main()

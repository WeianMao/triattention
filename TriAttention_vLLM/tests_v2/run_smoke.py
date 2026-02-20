"""Run TriAttention v2 Phase 1 smoke tests without pytest dependency."""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path


TEST_MODULES = [
    "tests_v2.test_compare_results",
    "tests_v2.test_config",
    "tests_v2.test_dispatch_sharding",
    "tests_v2.test_effective_len_tracker",
    "tests_v2.test_executor",
    "tests_v2.test_hook_impl",
    "tests_v2.test_kv_compaction",
    "tests_v2.test_planner",
    "tests_v2.test_runner",
    "tests_v2.test_scoring_trig_cache",
    "tests_v2.test_scheduler",
    "tests_v2.test_state",
    "tests_v2.test_utils_rkv_stats",
    "tests_v2.test_v2_eval_runner",
]


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    total = 0
    for module_name in TEST_MODULES:
        module = importlib.import_module(module_name)
        tests = [
            (name, func)
            for name, func in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("test_")
        ]
        for test_name, test_func in sorted(tests):
            test_func()
            total += 1
            print(f"[ok] {module_name}.{test_name}")

    print(f"smoke passed: {total} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

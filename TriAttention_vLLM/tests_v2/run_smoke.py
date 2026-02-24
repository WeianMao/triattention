"""Run TriAttention v2 Phase 1 smoke tests without pytest dependency."""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

OPTIONAL_IMPORT_ROOTS = {
    "vllm",
}


def discover_test_modules(tests_root: Path) -> list[str]:
    modules: list[str] = []
    for path in sorted(tests_root.glob("test_*.py")):
        if path.name == "__init__.py":
            continue
        modules.append(f"tests_v2.{path.stem}")
    return modules


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    tests_root = Path(__file__).resolve().parent

    total = 0
    for module_name in discover_test_modules(tests_root):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing_root = (exc.name or "").split(".", 1)[0]
            if missing_root in OPTIONAL_IMPORT_ROOTS:
                print(f"[skip] {module_name} (missing optional dependency: {exc.name})")
                continue
            raise
        tests = [
            (name, func)
            for name, func in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("test_")
        ]
        for test_name, test_func in sorted(tests):
            sig = inspect.signature(test_func)
            # This smoke runner intentionally executes only zero-arg tests so it
            # does not need pytest fixtures (e.g. tmp_path, monkeypatch).
            if sig.parameters:
                print(f"[skip] {module_name}.{test_name} (requires fixtures/args)")
                continue
            test_func()
            total += 1
            print(f"[ok] {module_name}.{test_name}")

    print(f"smoke passed: {total} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

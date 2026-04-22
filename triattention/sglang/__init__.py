"""TriAttention sglang Integration — KV cache compression plugin for sglang.

Usage:
    # Python mode (embedded in your launch script)
    from triattention.sglang import install_sglang_integration
    install_sglang_integration()
    # Then start sglang server normally

    # Command line mode
    python -m triattention.sglang --model <path> [sglang args...]

    # TP multi-process: use install_tp_hooks() before Engine construction
    from triattention.sglang import install_tp_hooks
    install_tp_hooks()
    # Engine(...) will now inject TriAttention into each TP subprocess

Environment variables:
    ENABLE_TRIATTENTION=1          Master switch (default: enabled)
    TRIATTENTION_QUIET=1           Suppress startup banner
    TRIATTN_RUNTIME_SPARSE_STATS_PATH=/path/to/stats.pt
    TRIATTN_RUNTIME_KV_BUDGET=512
    TRIATTN_RUNTIME_DIVIDE_LENGTH=128
    TRIATTN_RUNTIME_WINDOW_SIZE=32
    TRIATTN_SGLANG_DISABLE_RADIX_CACHE_CHECK=0

    Refer to triattention.sglang.config for the full list.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_installed: bool = False

_BANNER = """\
╔══════════════════════════════════════════════════════════╗
║  TriAttention sglang Integration — KV Cache Compression ║
╚══════════════════════════════════════════════════════════╝"""


def install_sglang_integration(
    server_args: Optional[object] = None,
    *,
    quiet: Optional[bool] = None,
) -> bool:
    """Install all TriAttention monkey-patches into the sglang runtime.

    This is the single entry point for activating TriAttention in an sglang
    server process.  It reads configuration from environment variables,
    validates compatibility, and registers all scheduler / input-prep hooks
    via :mod:`triattention.sglang.integration`.

    This function is **idempotent**: calling it multiple times is safe and
    will only install hooks once.

    Args:
        server_args: Optional sglang ``ServerArgs`` instance.  When provided,
            the installer can validate launch-time flags (e.g. radix cache
            disabled).  When ``None``, validation is skipped and the caller
            is responsible for ensuring a compatible configuration.
        quiet: Override the ``TRIATTENTION_QUIET`` env var.  When ``True``,
            suppress the startup banner.  When ``None``, read from env.

    Returns:
        ``True`` if hooks were installed (or were already installed),
        ``False`` if the integration is disabled via env var.
    """
    global _installed

    if _installed:
        logger.debug("TriAttention: install_sglang_integration() "
                      "called again — already installed, skipping.")
        return True

    # --- Check master switch ---
    from triattention.sglang.config import (
        ENV_ENABLE_TRIATTENTION,
        ENV_QUIET,
        _env_bool,
    )

    enabled = _env_bool(ENV_ENABLE_TRIATTENTION, default=True)
    if not enabled:
        logger.info("TriAttention: disabled via %s=0, skipping.",
                     ENV_ENABLE_TRIATTENTION)
        return False

    if quiet is None:
        quiet = _env_bool(ENV_QUIET, default=False)

    try:
        # --- Install all monkey-patches ---
        from triattention.sglang.integration import install_all_hooks

        install_all_hooks()
        _installed = True

        if not quiet:
            print(_BANNER)
            print("[TriAttention] All sglang hooks installed successfully.")

        logger.info("TriAttention: sglang integration installed.")
        return True

    except Exception:
        logger.exception(
            "TriAttention: failed to install sglang integration. "
            "The server will continue WITHOUT KV compression."
        )
        return False


def _patch_func_defaults(
    module_name: str, func_name: str, old_value: object, new_value: object
) -> None:
    """Replace *old_value* with *new_value* in a function's __defaults__.

    This is necessary because ``def f(x=run_scheduler_process)`` captures the
    value at definition time.  Even after patching the module attribute, the
    function's default tuple still holds the original reference.
    """
    import sys

    mod = sys.modules.get(module_name)
    if mod is None:
        return
    func = getattr(mod, func_name, None)
    if func is None:
        return
    defaults = func.__defaults__
    if defaults is None:
        return
    new_defaults = tuple(
        new_value if v is old_value else v for v in defaults
    )
    if new_defaults != defaults:
        func.__defaults__ = new_defaults
        logger.info(
            "TriAttention: patched default args of %s.%s",
            module_name, func_name,
        )


def _triattention_scheduler_process(*args, **kwargs):
    """Wrapper for ``run_scheduler_process`` that installs TriAttention hooks.

    Defined at module level so that ``multiprocessing`` with ``spawn``
    context can pickle it (local closures are not picklable).

    In ``spawn`` mode the child process is a fresh Python interpreter, so
    ``sglang.srt.managers.scheduler.run_scheduler_process`` has NOT been
    patched yet.  We simply:
    1. Import the *original* ``run_scheduler_process`` (still unpatched
       in this fresh process).
    2. Call :func:`install_sglang_integration` to install TriAttention
       hooks (scheduler monkey-patches, etc.) in this child.
    3. Delegate to the original ``run_scheduler_process``.
    """
    # Step 1: grab the original before we patch anything.
    from sglang.srt.managers.scheduler import run_scheduler_process as _original

    # Step 2: install TriAttention hooks in this subprocess.
    install_sglang_integration(quiet=True)

    # Step 3: run the original scheduler loop (now with hooks active).
    return _original(*args, **kwargs)


def install_tp_hooks() -> None:
    """Patch ``run_scheduler_process`` for TP multi-process.

    sglang launches each TP rank as an independent subprocess via
    ``mp.Process(target=run_scheduler_process, ...)``.
    In ``spawn`` mode, the child process does NOT inherit monkey-patches
    from the parent.

    This function wraps the original ``run_scheduler_process`` so that
    each child process calls :func:`install_sglang_integration` before
    running the scheduler loop.

    **Must be called before** ``Engine(...)`` or ``launch_server(...)``
    is invoked.

    Example::

        from triattention.sglang import install_tp_hooks
        install_tp_hooks()
        # Now create Engine or call launch_server — each TP subprocess
        # will have TriAttention hooks installed automatically.
    """
    # Patch module-level bindings directly for TP hook injection.
    #
    # sglang's launch_server() and Engine._launch_scheduler_processes() pass
    # run_scheduler_process as a default parameter value captured at import
    # time.  Patching Engine.run_scheduler_process_func alone is NOT enough
    # because launch_server() never reads that class attribute — it uses its
    # own default argument.  We must replace the binding in every module that
    # imports run_scheduler_process so that the wrapped version is used
    # regardless of the call path.

    import sglang.srt.managers.scheduler as _sched_mod

    original_func = _sched_mod.run_scheduler_process

    # 1. Patch the canonical definition in the scheduler module
    _sched_mod.run_scheduler_process = _triattention_scheduler_process
    logger.info("TriAttention: patched sglang.srt.managers.scheduler"
                ".run_scheduler_process")

    # 2. Patch re-exported bindings in modules that do
    #    ``from sglang.srt.managers.scheduler import run_scheduler_process``
    #    so their default parameter values pick up the wrapper.
    _reexport_modules = [
        "sglang.srt.entrypoints.engine",
        "sglang.srt.entrypoints.http_server",
        "sglang.srt.managers.data_parallel_controller",
    ]
    # Optional: ray module may not be installed
    try:
        import sglang.srt.ray.http_server  # noqa: F401
        _reexport_modules.append("sglang.srt.ray.http_server")
    except (ImportError, ModuleNotFoundError):
        pass

    import sys
    for mod_name in _reexport_modules:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "run_scheduler_process"):
            setattr(mod, "run_scheduler_process", _triattention_scheduler_process)
            logger.info("TriAttention: patched %s.run_scheduler_process",
                        mod_name)

    # 3. Also patch Engine.run_scheduler_process_func for the Engine(...)
    #    code path (Engine.__init__ reads self.run_scheduler_process_func).
    from sglang.srt.entrypoints.engine import Engine
    Engine.run_scheduler_process_func = staticmethod(
        _triattention_scheduler_process
    )
    logger.info(
        "TriAttention: patched Engine.run_scheduler_process_func "
        "for TP subprocess injection."
    )

    # 4. Patch default parameter values of launch_server / run_data_parallel_controller_process.
    #    Python captures default arg values at function definition time, so
    #    replacing the module attribute alone won't update functions already
    #    defined with ``def f(x=run_scheduler_process)``.  We must also
    #    rewrite the function defaults.
    _patch_func_defaults(
        "sglang.srt.entrypoints.http_server", "launch_server",
        original_func, _triattention_scheduler_process,
    )
    _patch_func_defaults(
        "sglang.srt.managers.data_parallel_controller",
        "run_data_parallel_controller_process",
        original_func, _triattention_scheduler_process,
    )
    try:
        _patch_func_defaults(
            "sglang.srt.ray.http_server", "launch_server",
            original_func, _triattention_scheduler_process,
        )
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass


__all__ = [
    "install_sglang_integration",
    "install_tp_hooks",
]

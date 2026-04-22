"""Launcher wrapper — start an sglang server with TriAttention enabled.

Usage::

    python -m triattention.sglang --model <path> [sglang args...]
    python -m triattention.sglang.launcher --model <path> [sglang args...]

The launcher:

1. Sets ``ENABLE_TRIATTENTION=1`` if not already set.
2. Calls :func:`~triattention.sglang.install_tp_hooks` to ensure every
   TP subprocess gets TriAttention monkey-patches.
3. Appends ``--disable-radix-cache`` (Phase 1 requirement).
4. Forwards all CLI arguments to sglang's server entry point.

Alternatively, users can call ``install_sglang_integration()`` and
``install_tp_hooks()`` directly in their own launch scripts.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    """Entry point for ``python -m triattention.sglang.launcher``."""
    # --- Ensure master switch is on ---
    os.environ.setdefault("ENABLE_TRIATTENTION", "1")

    # --- Install TP hooks (before Engine is created) ---
    from triattention.sglang import install_sglang_integration, install_tp_hooks

    # Install hooks in *this* process (the main/parent process).
    install_sglang_integration()

    # Patch Engine.run_scheduler_process_func so that each TP child
    # process also gets TriAttention hooks.
    install_tp_hooks()

    # --- Prepare sglang CLI args ---
    argv = sys.argv[1:]

    # Phase 1 hard constraint: radix cache must be disabled.
    # Append --disable-radix-cache if the user did not already pass it.
    if "--disable-radix-cache" not in argv:
        argv.append("--disable-radix-cache")

    # --- Parse server args and launch ---
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(argv)

    try:
        from sglang.launch_server import run_server

        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()

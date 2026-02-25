#!/usr/bin/env python3
"""Default TriAttention vLLM sharded runner (stable entrypoint wrapper)."""

from __future__ import annotations

try:  # package import path (e.g. tests)
    from .vllm_triattention_runtime_runner import *  # type: ignore # noqa: F401,F403
    from .vllm_triattention_runtime_runner import (  # type: ignore
        _apply_runtime_env,
        main,
        parse_arguments,
        setup_vllm_engine,
    )
except ImportError:  # script execution by path
    from vllm_triattention_runtime_runner import *  # type: ignore # noqa: F401,F403
    from vllm_triattention_runtime_runner import (  # type: ignore
        _apply_runtime_env,
        main,
        parse_arguments,
        setup_vllm_engine,
    )


if __name__ == "__main__":
    main(parse_arguments())

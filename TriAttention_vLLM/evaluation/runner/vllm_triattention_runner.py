#!/usr/bin/env python3
"""Default TriAttention vLLM sharded runner (compat wrapper).

This file is the stable, user-facing runner entrypoint.
The current implementation is provided by the existing `vllm_triattention_v2_runner`
module for compatibility and risk control.
"""

from __future__ import annotations

try:  # package import path (e.g. tests)
    from .vllm_triattention_v2_runner import *  # type: ignore # noqa: F401,F403
    from .vllm_triattention_v2_runner import main, parse_arguments  # type: ignore
except ImportError:  # script execution by path
    from vllm_triattention_v2_runner import *  # type: ignore # noqa: F401,F403
    from vllm_triattention_v2_runner import main, parse_arguments  # type: ignore


if __name__ == "__main__":
    main(parse_arguments())

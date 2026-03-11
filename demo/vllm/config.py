"""Configuration for the OpenAI-compatible vLLM gateway demo."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DemoConfig:
    host: str = os.getenv("DEMO_HOST", "127.0.0.1")
    port: int = int(os.getenv("DEMO_PORT", "8010"))

    backend_base_url: str = os.getenv(
        "VLLM_BACKEND_URL", os.getenv("BASELINE_VLLM_URL", "http://127.0.0.1:8001")
    )
    triattention_backend_base_url: str = os.getenv(
        "TRIATTENTION_VLLM_URL", "http://127.0.0.1:8002"
    )

    forward_backend: str = os.getenv("DEMO_FORWARD_BACKEND", "triattention")

    request_timeout_s: float = float(os.getenv("DEMO_REQUEST_TIMEOUT_S", "600"))
    connect_timeout_s: float = float(os.getenv("DEMO_CONNECT_TIMEOUT_S", "15"))
    secondary_stream_idle_timeout_s: float = float(
        os.getenv("DEMO_SECONDARY_STREAM_IDLE_TIMEOUT_S", "12")
    )

    default_max_tokens: int = int(os.getenv("DEMO_DEFAULT_MAX_TOKENS", "512"))
    max_tokens_cap: int = int(os.getenv("DEMO_MAX_TOKENS_CAP", "4000"))
    default_temperature: float = float(os.getenv("DEMO_DEFAULT_TEMPERATURE", "0.6"))
    default_top_p: float = float(os.getenv("DEMO_DEFAULT_TOP_P", "0.95"))
    default_seed: int = int(os.getenv("DEMO_DEFAULT_SEED", "888"))


CONFIG = DemoConfig()

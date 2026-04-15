"""TriAttention v2 worker integration."""

from __future__ import annotations

import os
from pathlib import Path

from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker as VLLMGPUWorker

from .config import TriAttentionRuntimeConfig
from .hook_impl import install_runner_compression_hook
from .runner import TriAttentionModelRunner

logger = init_logger(__name__)


def _debug_early_install_proxy_enabled() -> bool:
    return os.environ.get("TRIATTN_DEBUG_EARLY_INSTALL_PROXY", "0") == "1"


def _maybe_backfill_model_path(worker: VLLMGPUWorker, config: TriAttentionRuntimeConfig) -> None:
    if config.model_path is not None:
        return
    model_config = getattr(worker, "model_config", None)
    model_path = getattr(model_config, "model", None)
    if isinstance(model_path, str) and model_path.strip():
        config.model_path = Path(model_path.strip())


class TriAttentionWorker(VLLMGPUWorker):
    """GPU worker that injects TriAttention model-runner proxy."""

    def init_device(self):
        super().init_device()
        if isinstance(self.model_runner, TriAttentionModelRunner):
            return

        # Keep native vLLM GPUModelRunner untouched during warmup/graph-capture and
        # pre-trigger decode. We lazily wrap on the first step that carries a
        # TriAttention signal (trigger/compressed-request update), which minimizes
        # impact on the common no-compression path.
        self._triattention_runtime_config = TriAttentionRuntimeConfig.from_env()
        _maybe_backfill_model_path(self, self._triattention_runtime_config)
        self._triattention_runner_proxy_installed = False
        if _debug_early_install_proxy_enabled():
            self._ensure_triattention_runner_proxy()
            logger.debug("TriAttentionWorker: eagerly installed runner proxy during init_device")

    def _ensure_triattention_runner_proxy(self) -> None:
        if getattr(self, "_triattention_runner_proxy_installed", False):
            return
        if isinstance(self.model_runner, TriAttentionModelRunner):
            self._triattention_runner_proxy_installed = True
            return
        config = getattr(self, "_triattention_runtime_config", None) or TriAttentionRuntimeConfig.from_env()
        _maybe_backfill_model_path(self, config)
        base_runner = self.model_runner
        install_runner_compression_hook(base_runner=base_runner, config=config)
        self.model_runner = TriAttentionModelRunner(
            base_runner=base_runner,
            config=config,
        )
        self._triattention_runner_proxy_installed = True
        logger.info(
            "TriAttentionWorker lazily injected runner proxy: budget=%d divide_length=%d "
            "seq_len_override_patch=%s stats_path=%s model_path=%s protect_prefill=%s "
            "window_size=%s",
            config.kv_budget,
            config.divide_length,
            "deferred",
            str(config.sparse_stats_path) if config.sparse_stats_path is not None else None,
            str(config.model_path) if config.model_path is not None else None,
            config.protect_prefill,
            config.window_size,
        )

    def execute_model(self, scheduler_output):  # type: ignore[override]
        # Sparse scheduler signals are empty in the common pre-trigger path.
        # Install the proxy only when TriAttention behavior is actually needed.
        signals = getattr(scheduler_output, "triattention_signals", None)
        if signals:
            self._ensure_triattention_runner_proxy()
        return super().execute_model(scheduler_output)

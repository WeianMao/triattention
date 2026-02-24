"""TriAttention v2 worker integration."""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker as VLLMGPUWorker

from .config import TriAttentionV2Config
from .hook_impl import install_runner_compression_hook
from .runner import TriAttentionModelRunner

logger = init_logger(__name__)


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
        self._triattention_v2_config = TriAttentionV2Config.from_env()
        self._triattention_runner_proxy_installed = False

    def _ensure_triattention_runner_proxy(self) -> None:
        if getattr(self, "_triattention_runner_proxy_installed", False):
            return
        if isinstance(self.model_runner, TriAttentionModelRunner):
            self._triattention_runner_proxy_installed = True
            return
        config = getattr(self, "_triattention_v2_config", None) or TriAttentionV2Config.from_env()
        base_runner = self.model_runner
        install_runner_compression_hook(base_runner=base_runner, config=config)
        self.model_runner = TriAttentionModelRunner(
            base_runner=base_runner,
            config=config,
        )
        self._triattention_runner_proxy_installed = True
        logger.info(
            "TriAttentionWorker lazily injected runner proxy: budget=%d divide_length=%d "
            "seq_len_override_patch=%s",
            config.kv_budget,
            config.divide_length,
            "deferred",
        )

    def execute_model(self, scheduler_output):  # type: ignore[override]
        # Sparse scheduler signals are empty in the common pre-trigger path.
        # Install the proxy only when TriAttention behavior is actually needed.
        signals = getattr(scheduler_output, "triattention_signals", None)
        if signals:
            self._ensure_triattention_runner_proxy()
        return super().execute_model(scheduler_output)

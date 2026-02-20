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

        config = TriAttentionV2Config.from_env()
        base_runner = self.model_runner
        install_runner_compression_hook(base_runner=base_runner, config=config)
        self.model_runner = TriAttentionModelRunner(
            base_runner=base_runner,
            config=config,
        )
        logger.info(
            "TriAttentionWorker injected runner proxy: budget=%d divide_length=%d",
            config.kv_budget,
            config.divide_length,
        )

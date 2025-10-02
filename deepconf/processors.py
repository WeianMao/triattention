import torch
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)
from vllm import SamplingParams
from vllm.config import VllmConfig
from collections import deque
from typing import Optional, List, Callable, Any, Dict
from abc import ABC, abstractmethod
import time
import functools
import multiprocessing

class ConfPerReqLogitsProcessor:
    """The request-level logits processor masks out all logits except the
    token id identified by `target_token`"""

    def __init__(self, threshold: float, eos_token_id: int, conf_group_size: int, conf_topk: int) -> None:
        """Specify `confidence`"""
        self.threshold = threshold
        self.conf_list = []
        self.eos_token_id = eos_token_id
        self.conf_topk = conf_topk
        self.conf_group_list = deque(maxlen=conf_group_size)
        self.conf_grouped = 0.0
        self.conf_group_size = conf_group_size

    def compute_conf(self, logits: torch.Tensor) -> float:
        # Compute the confidence score based on the logits
        probabilities = torch.softmax(logits, dim=-1)
        top_probs, _ = torch.topk(probabilities, self.conf_topk, dim=-1)
        log_probs = torch.log(top_probs)
        return -log_probs.sum().item() / self.conf_topk

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        new_conf = self.compute_conf(logits)

        if len(self.conf_group_list) < self.conf_group_size:
            self.conf_group_list.append(new_conf)
            self.conf_grouped += new_conf
        else:
            self.conf_grouped -= self.conf_group_list.popleft()
            self.conf_group_list.append(new_conf)
            self.conf_grouped += new_conf

        if len(self.conf_group_list) >= self.conf_group_size and self.conf_grouped / len(self.conf_group_list) < self.threshold:
            val_to_keep = logits[self.eos_token_id].item()
            logits[:] = float("-inf")
            logits[self.eos_token_id] = val_to_keep
        return logits
    

class WrappedPerReqLogitsProcessor(AdapterLogitsProcessor):
    """Example of overriding the wrapper class `__init__()` in order to utilize
    info about the device type"""

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        self.is_cuda = device.type == "cuda"

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        """This method returns a new request-level logits processor, customized
        to the `target_token` value associated with a particular request.

        Returns None if the logits processor should not be applied to the
        particular request. To use the logits processor the request must have
        a "target_token" custom argument with an integer value, and the device
        must be "cuda"-type

        Args:
          params: per-request sampling params

        Returns:
          `Callable` request logits processor, or None
        """
        if (
            not self.is_cuda
            or (
                conf_threshold := params.extra_args
                and params.extra_args.get("conf_threshold")
            )
            is None
            or (eos_token_id := params.extra_args
                and params.extra_args.get("eos_token_id")
            ) is None
            or (
                conf_group_size := params.extra_args
                and params.extra_args.get("conf_group_size")
            ) is None
            or (
                conf_topk := params.extra_args
                and params.extra_args.get("conf_topk")
            ) is None
        ):
            # print("Not using ConfPerReqLogitsProcessor", params.extra_args)
            return None
        # print(f"Using ConfPerReqLogitsProcessor with threshold {conf_threshold}, eos_token_id {eos_token_id}, group_size {conf_group_size}, topk {conf_topk}")
        return ConfPerReqLogitsProcessor(conf_threshold, eos_token_id, conf_group_size, conf_topk)

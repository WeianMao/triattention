from types import SimpleNamespace

from triattention_runtime.executor import RunnerHookCompressionExecutor
from triattention_runtime.signals import CompressionSignal


def _signal() -> CompressionSignal:
    return CompressionSignal(
        req_id="r1",
        should_compress=True,
        reason="length_threshold",
        estimated_cache_len=128,
        step=5,
        kv_usage=None,
        protect_prefill=True,
        prefill_len=64,
    )


def test_missing_hook_returns_skip():
    base_runner = SimpleNamespace()
    executor = RunnerHookCompressionExecutor(base_runner)

    result = executor.execute("r1", _signal(), scheduler_output=SimpleNamespace())
    assert result.applied is False
    assert result.reason == "runner_hook_missing"


def test_bool_hook_result():
    class BaseRunner:
        def triattention_apply_compression(self, req_id, signal, scheduler_output):
            assert req_id == "r1"
            assert signal.should_compress is True
            return True

    executor = RunnerHookCompressionExecutor(BaseRunner())
    result = executor.execute("r1", _signal(), scheduler_output=SimpleNamespace())
    assert result.applied is True
    assert result.reason == "applied"


def test_dict_hook_result():
    class BaseRunner:
        def triattention_apply_compression(self, req_id, signal, scheduler_output):
            return {
                "applied": True,
                "reason": "hook_applied",
                "cache_len_after": 64,
                "block_reclaim": {"groups": [{"gid": 0, "block_ids_after": [1]}]},
            }

    executor = RunnerHookCompressionExecutor(BaseRunner())
    result = executor.execute("r1", _signal(), scheduler_output=SimpleNamespace())
    assert result.applied is True
    assert result.reason == "hook_applied"
    assert result.cache_len_after == 64
    assert isinstance(result.details, dict)
    assert result.details["block_reclaim"]["groups"][0]["gid"] == 0

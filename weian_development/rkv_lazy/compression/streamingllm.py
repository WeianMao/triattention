import torch


class StreamingLLM:
    def __init__(
        self,
        budget=128,
        first_tokens=4,
        **kwargs,
    ):
        assert budget - first_tokens > 0, "budget must be greater than first_tokens"
        self.budget = budget
        self.first_tokens = first_tokens

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        prefix_len: int = 0,
    ):
        prefix_len = max(0, int(prefix_len))
        prefix_k = key_states[:, :, :prefix_len, :] if prefix_len > 0 else None
        prefix_v = value_states[:, :, :prefix_len, :] if prefix_len > 0 else None
        key_states = key_states[:, :, prefix_len:, :] if prefix_len > 0 else key_states
        value_states = value_states[:, :, prefix_len:, :] if prefix_len > 0 else value_states

        kv_cache_len = key_states.shape[-2]

        if kv_cache_len < self.budget:
            if prefix_k is None:
                return key_states, value_states
            return torch.cat([prefix_k, key_states], dim=2), torch.cat([prefix_v, value_states], dim=2)

        local_window_size = self.budget - self.first_tokens
        key_states = torch.cat(
            [
                key_states[:, :, : self.first_tokens],
                key_states[:, :, -local_window_size:],
            ],
            dim=2,
        )
        value_states = torch.cat(
            [
                value_states[:, :, : self.first_tokens],
                value_states[:, :, -local_window_size:],
            ],
            dim=2,
        )
        if prefix_k is not None:
            key_states = torch.cat([prefix_k, key_states], dim=2)
            value_states = torch.cat([prefix_v, value_states], dim=2)
        return key_states, value_states

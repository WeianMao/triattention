import torch


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    # ===== helpers for legacy Cache interop =====
    @staticmethod
    def _convert_to_legacy(past_key_values):
        if past_key_values is None:
            return None, None
        if hasattr(past_key_values, "to_legacy_cache") and hasattr(
            past_key_values.__class__, "from_legacy_cache"
        ):
            return past_key_values.to_legacy_cache(), past_key_values.__class__
        return past_key_values, None

    @staticmethod
    def _convert_from_legacy(legacy_cache, cache_cls):
        if legacy_cache is None:
            return None
        if cache_cls is not None:
            return cache_cls.from_legacy_cache(legacy_cache)
        return legacy_cache

    def _slice_keep_start_recent(self, legacy_cache):
        seq_len = self._extract_seq_len(legacy_cache)
        if seq_len <= 0 or seq_len <= self.cache_size:
            return legacy_cache
        trimmed = []
        for entry in legacy_cache:
            if not entry or entry[0] is None or entry[1] is None:
                trimmed.append(entry)
                continue
            k, v = entry
            trimmed.append(
                (
                    torch.cat(
                        [
                            self.k_slice(k, 0, self.start_size),
                            self.k_slice(k, seq_len - self.recent_size, seq_len),
                        ],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [
                            self.v_slice(v, 0, self.start_size),
                            self.v_slice(v, seq_len - self.recent_size, seq_len),
                        ],
                        dim=self.v_seq_dim,
                    ),
                )
            )
        return tuple(trimmed)

    def __call__(self, past_key_values):
        legacy_cache, cache_cls = self._convert_to_legacy(past_key_values)
        if legacy_cache is None or len(legacy_cache) == 0:
            return past_key_values
        trimmed = self._slice_keep_start_recent(legacy_cache)
        if trimmed is legacy_cache:
            return past_key_values
        return self._convert_from_legacy(trimmed, cache_cls)

    def evict_for_space(self, past_key_values, num_coming):
        legacy_cache, cache_cls = self._convert_to_legacy(past_key_values)
        if legacy_cache is None or len(legacy_cache) == 0:
            return past_key_values
        seq_len = self._extract_seq_len(legacy_cache)
        if seq_len <= 0 or seq_len + num_coming <= self.cache_size:
            return past_key_values
        trimmed = []
        for entry in legacy_cache:
            if not entry or entry[0] is None or entry[1] is None:
                trimmed.append(entry)
                continue
            k, v = entry
            trimmed.append(
                (
                    torch.cat(
                        [
                            self.k_slice(k, 0, self.start_size),
                            self.k_slice(
                                k, seq_len - self.recent_size + num_coming, seq_len
                            ),
                        ],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [
                            self.v_slice(v, 0, self.start_size),
                            self.v_slice(
                                v, seq_len - self.recent_size + num_coming, seq_len
                            ),
                        ],
                        dim=self.v_seq_dim,
                    ),
                )
            )
        trimmed = tuple(trimmed)
        return self._convert_from_legacy(trimmed, cache_cls)

    def evict_range(self, past_key_values, start, end):
        legacy_cache, cache_cls = self._convert_to_legacy(past_key_values)
        if legacy_cache is None or len(legacy_cache) == 0:
            return past_key_values
        seq_len = self._extract_seq_len(legacy_cache)
        if seq_len <= 0:
            return past_key_values
        assert start <= end and end <= seq_len
        trimmed = []
        for entry in legacy_cache:
            if not entry or entry[0] is None or entry[1] is None:
                trimmed.append(entry)
                continue
            k, v = entry
            trimmed.append(
                (
                    torch.cat(
                        [
                            self.k_slice(k, 0, start),
                            self.k_slice(k, end, seq_len),
                        ],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [
                            self.v_slice(v, 0, start),
                            self.v_slice(v, end, seq_len),
                        ],
                        dim=self.v_seq_dim,
                    ),
                )
            )
        trimmed = tuple(trimmed)
        return self._convert_from_legacy(trimmed, cache_cls)

    def _extract_seq_len(self, legacy_cache):
        for entry in legacy_cache:
            if entry and entry[0] is not None:
                return entry[0].size(self.k_seq_dim)
        return 0

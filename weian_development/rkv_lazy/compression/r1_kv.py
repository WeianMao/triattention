import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cal_similarity, compute_attention_scores


class R1KV:
    def __init__(
        self,
        budget=128,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.1,
        retain_ratio=0.1,
        retain_direction="last",
        record_kept_token_indices=False,
        fp32_topk: bool = False,
        # Round-based compression params (for fair comparison with sparse_prefill_keep)
        round_window: int = 0,           # 0 = disabled; >0 = round size (e.g., 363)
        round_base_budget: int = None,   # target budget at round start (e.g., 1129)
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction
        self.use_fp32_topk = fp32_topk

        # Round-based compression state
        self.round_window = round_window
        self.use_round_compression = round_window > 0
        if self.use_round_compression:
            self.round_base_budget = round_base_budget if round_base_budget else (budget - round_window)
            assert self.round_base_budget > window_size, "round_base_budget must be > window_size"
        else:
            self.round_base_budget = budget
        self.tokens_in_round = 0

        # for recording kept token indices
        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []
            self.kept_similarity_scores = []
            self.kept_final_scores = []

    def _split_prefix(self, key_states, value_states, prefix_len: int):
        if prefix_len is None or prefix_len <= 0:
            return None, None, key_states, value_states
        prefix_len = min(prefix_len, key_states.shape[-2])
        if prefix_len <= 0:
            return None, None, key_states, value_states
        prefix_k = key_states[:, :, :prefix_len, :]
        prefix_v = value_states[:, :, :prefix_len, :]
        return prefix_k, prefix_v, key_states[:, :, prefix_len:, :], value_states[:, :, prefix_len:, :]

    def _do_compression(self, key_states, query_states, value_states, target_budget: int):
        """Compress KV cache to target_budget size."""
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len <= target_budget:
            return key_states, value_states

        attn_weights = compute_attention_scores(query_states, key_states)

        attn_weights_sum = nn.functional.softmax(
            attn_weights[:, :, -self.window_size :, : -self.window_size],
            dim=-1,
            dtype=torch.float32,
        ).mean(dim=-2)
        if not self.use_fp32_topk:
            attn_weights_sum = attn_weights_sum.to(query_states.dtype)

        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )

        similarity_cos = cal_similarity(
            key_states,
            retain_ratio=self.retain_ratio,
            retain_direction=self.retain_direction,
        )[:, : -self.window_size]
        if self.use_fp32_topk:
            similarity_cos = similarity_cos.to(torch.float32)
        else:
            similarity_cos = similarity_cos.to(query_states.dtype)

        final_score = attn_cache * self.mix_lambda - similarity_cos * (1 - self.mix_lambda)
        score_for_topk = final_score if self.use_fp32_topk else final_score.to(query_states.dtype)

        keep_count = target_budget - self.window_size
        indices = score_for_topk.topk(keep_count, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past_compress = key_states[:, :, : -self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = value_states[:, :, : -self.window_size, :].gather(dim=2, index=indices)
        k_cur = key_states[:, :, -self.window_size :, :]
        v_cur = value_states[:, :, -self.window_size :, :]
        return torch.cat([k_past_compress, k_cur], dim=2), torch.cat([v_past_compress, v_cur], dim=2)

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        prefix_len: int = 0,
    ):
        prefix_k, prefix_v, key_states, value_states = self._split_prefix(key_states, value_states, prefix_len)
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        # Round-based compression (for fair comparison with sparse_prefill_keep)
        if self.use_round_compression:
            # Auto-detect new sample: prefix_len=0 means this is prefill stage (new sample)
            # Reset tokens_in_round to align with sparse_prefill_keep which creates new pruner per sample
            if prefix_len == 0:
                self.tokens_in_round = 0

            self.tokens_in_round += 1
            should_start_new_round = self.tokens_in_round >= self.round_window

            if should_start_new_round and kv_cache_len > self.round_base_budget:
                # End of round: compress to round_base_budget
                key_states, value_states = self._do_compression(
                    key_states, query_states, value_states, self.round_base_budget
                )
                self.tokens_in_round = 0
            elif kv_cache_len > self.budget:
                # Safety: enforce max budget
                key_states, value_states = self._do_compression(
                    key_states, query_states, value_states, self.budget
                )
            # Concat prefix and return
            if prefix_k is not None:
                key_states = torch.cat([prefix_k, key_states], dim=2)
                value_states = torch.cat([prefix_v, value_states], dim=2)
            return key_states, value_states

        # Original logic (non-round-based)
        if kv_cache_len < self.budget:
            if prefix_k is None:
                return key_states, value_states
            return torch.cat([prefix_k, key_states], dim=2), torch.cat([prefix_v, value_states], dim=2)
        else:
            attn_weights = compute_attention_scores(query_states, key_states)

            attn_weights_sum = nn.functional.softmax(
                attn_weights[:, :, -self.window_size :, : -self.window_size],
                dim=-1,
                dtype=torch.float32,
            ).mean(dim=-2)
            if not self.use_fp32_topk:
                attn_weights_sum = attn_weights_sum.to(query_states.dtype)
            # TODO: Softmax then reduce head

            attn_cache = F.max_pool1d(
                attn_weights_sum,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )

            similarity_cos = cal_similarity(
                key_states,
                retain_ratio=self.retain_ratio,
                retain_direction=self.retain_direction,
            )[:, : -self.window_size]
            if self.use_fp32_topk:
                similarity_cos = similarity_cos.to(torch.float32)
            else:
                similarity_cos = similarity_cos.to(query_states.dtype)

            final_score = attn_cache * self.mix_lambda - similarity_cos * (
                1 - self.mix_lambda
            )

            score_for_topk = final_score if self.use_fp32_topk else final_score.to(query_states.dtype)
            # shape: (bsz, num_kv_heads, budget - window_size)
            indices = score_for_topk.topk(self.budget - self.window_size, dim=-1).indices

            #####################################################
            ###### Store evicted token indices start ############
            #####################################################
            # shape: (num_kv_heads, budget - window_size)
            if self.record_kept_token_indices:
                indices_cl = indices.clone().squeeze(0).to("cpu")

                similarity_cos_analysis = cal_similarity(
                    key_states,
                    retain_ratio=self.retain_ratio,
                    retain_direction=self.retain_direction,
                )

                attn_weights_sum_analysis = (
                    nn.functional.softmax(
                        attn_weights,
                        dim=-1,
                        dtype=torch.float32,
                    )
                    .mean(dim=-2)
                    .to(query_states.dtype)
                )

                attn_cache_analysis = F.max_pool1d(
                    attn_weights_sum_analysis,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1,
                )

                final_score_analysis = attn_cache_analysis * self.mix_lambda - similarity_cos_analysis * (
                    1 - self.mix_lambda
                )

                recent_window_indices = torch.arange(
                    kv_cache_len - self.window_size, kv_cache_len, device="cpu"
                ).expand(indices_cl.shape[0], -1)
                cur_indices = torch.cat([indices_cl, recent_window_indices], dim=-1)

                #####################################################
                ### Store final scores, attention and similarity ####
                #####################################################

                # Gather the scores for the kept tokens
                attn_scores = attn_cache_analysis.clone().squeeze(0).to("cpu")
                sim_scores = similarity_cos_analysis.clone().squeeze(0).to("cpu")
                fin_scores = final_score_analysis.clone().squeeze(0).to("cpu")

                # print(f"cur_indices {cur_indices} attn_cache_analysis {attn_cache_analysis.shape} similarity_cos_analysis {similarity_cos_analysis.shape} final_score_analysis {final_score_analysis.shape}")

                # Gather the scores based on index
                kept_attn = torch.gather(attn_scores, dim=1, index=cur_indices)
                kept_sim = torch.gather(sim_scores, dim=1, index=cur_indices)
                kept_final = torch.gather(fin_scores, dim=1, index=cur_indices)

                #####################################################

                if self.evicted_token_num > 0:
                    prev_indices = self.kept_token_indices[-1]
                    mask = cur_indices < self.budget

                    for i in range(cur_indices.shape[0]):
                        positions = torch.where(mask[i])[0]

                        # For each position, get the value and use it as an index into prev_indices
                        for pos in positions:
                            val = cur_indices[i, pos].item()
                            cur_indices[i, pos] = prev_indices[i, val]

                    # For values >= self.budget, add the evicted token count
                    cur_indices[~mask] += self.evicted_token_num

                #####################################################
                ### Store final scores, attention and similarity ####
                #####################################################
                self.kept_attention_scores.append(kept_attn)
                self.kept_similarity_scores.append(kept_sim)
                self.kept_final_scores.append(kept_final)
                #####################################################

                self.kept_token_indices.append(cur_indices)
                self.evicted_token_num += kv_cache_len - self.budget
            ######################################################

            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            k_past_compress = key_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            v_past_compress = value_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            k_cur = key_states[:, :, -self.window_size :, :]
            v_cur = value_states[:, :, -self.window_size :, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            if prefix_k is not None:
                key_states = torch.cat([prefix_k, key_states], dim=2)
                value_states = torch.cat([prefix_v, value_states], dim=2)
            return key_states, value_states

    def reset_compression_state(self) -> None:
        self.tokens_in_round = 0  # Reset round counter for new sample
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []
            self.kept_similarity_scores = []
            self.kept_final_scores = []

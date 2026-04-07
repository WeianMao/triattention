import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import cal_similarity, compute_attention_scores


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
        protect_prefill: bool = False,
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
        # Prefill protection: when True, prefill tokens are always preserved during compression
        # (like TriAttention's default behavior); when False, all tokens compete for budget (original R-KV behavior)
        self.protect_prefill = protect_prefill
        self.prefill_length = 0  # Set by attach_prefill_length() before first decode

        # for recording kept token indices
        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []
            self.kept_similarity_scores = []
            self.kept_final_scores = []

    def attach_prefill_length(self, prefill_length: int) -> None:
        """Set the prefill length (number of prompt tokens) for prefill protection."""
        self.prefill_length = prefill_length

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
    ):
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len < self.budget:
            return key_states, value_states
        else:
            # Prefill protection mode: if enabled, only compress decode tokens while preserving prefill
            if self.protect_prefill and self.prefill_length > 0:
                return self._update_kv_protect_prefill(key_states, query_states, value_states)

            # Original R-KV behavior: all tokens compete for budget
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
            return key_states, value_states

    def reset_compression_state(self) -> None:
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []
            self.kept_similarity_scores = []
            self.kept_final_scores = []

    def _update_kv_protect_prefill(
        self,
        key_states,
        query_states,
        value_states,
    ):
        """
        R-KV compression with prefill protection: prefill tokens are always preserved,
        only decode tokens compete for the remaining budget.

        This provides an ablation to compare with TriAttention's default prefill protection behavior.
        """
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]
        prefill_len = self.prefill_length

        # Calculate effective budget for decode tokens after reserving space for prefill
        decode_budget = self.budget - prefill_len
        if decode_budget <= self.window_size:
            # Not enough budget for decode compression, just keep prefill + recent window
            k_prefill = key_states[:, :, :prefill_len, :]
            v_prefill = value_states[:, :, :prefill_len, :]
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            return torch.cat([k_prefill, k_cur], dim=2), torch.cat([v_prefill, v_cur], dim=2)

        # Number of decode tokens (excluding window)
        decode_len = kv_cache_len - prefill_len - self.window_size
        if decode_len <= 0:
            # All tokens are prefill + window, no compression needed
            return key_states, value_states

        # Check if compression is needed
        decode_keep = decode_budget - self.window_size
        if decode_len <= decode_keep:
            # Not enough decode tokens to exceed budget, no compression needed
            return key_states, value_states

        # Extract segments: [prefill | decode | window]
        k_prefill = key_states[:, :, :prefill_len, :]
        v_prefill = value_states[:, :, :prefill_len, :]
        k_decode = key_states[:, :, prefill_len:-self.window_size, :]
        v_decode = value_states[:, :, prefill_len:-self.window_size, :]
        k_window = key_states[:, :, -self.window_size:, :]
        v_window = value_states[:, :, -self.window_size:, :]

        # Compute attention scores on decode tokens only (excluding prefill)
        # This ensures decode tokens compete fairly among themselves, matching TriAttention behavior
        attn_weights = compute_attention_scores(query_states, k_decode)

        # Softmax over decode positions only, then take mean over query window
        attn_weights_sum = nn.functional.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32,
        ).mean(dim=-2)
        if not self.use_fp32_topk:
            attn_weights_sum = attn_weights_sum.to(query_states.dtype)

        # Max pooling for attention scores (already only decode tokens)
        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )

        # Compute similarity on decode tokens only
        similarity_cos = cal_similarity(
            k_decode,
            retain_ratio=self.retain_ratio,
            retain_direction=self.retain_direction,
        )
        if self.use_fp32_topk:
            similarity_cos = similarity_cos.to(torch.float32)
        else:
            similarity_cos = similarity_cos.to(query_states.dtype)

        # Final score: attention * lambda - similarity * (1 - lambda)
        final_score = attn_cache * self.mix_lambda - similarity_cos * (1 - self.mix_lambda)

        score_for_topk = final_score if self.use_fp32_topk else final_score.to(query_states.dtype)

        # Select top decode tokens
        indices = score_for_topk.topk(decode_keep, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        # Gather selected decode tokens
        k_decode_compress = k_decode.gather(dim=2, index=indices)
        v_decode_compress = v_decode.gather(dim=2, index=indices)

        # Concatenate: prefill + compressed_decode + window
        key_states = torch.cat([k_prefill, k_decode_compress, k_window], dim=2)
        value_states = torch.cat([v_prefill, v_decode_compress, v_window], dim=2)

        return key_states, value_states

import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
from model.temp_cacheobs import TempCache
import numpy as np
import os

# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Window_LAZYKVCluster():

    current_decoding_step = 0
    jump_step = 0
    jump_layer = 0

    def __init__(self, num_hidden_layers = 32, decoding_recent_size = 256, max_kv_capacity = 256, obs_size = 3):
        ##### Add decoding window #####
        self.decoding_recent_size = decoding_recent_size
        self.max_kv_capacity = max_kv_capacity
        self.num_hidden_layers = num_hidden_layers
        self.obs_size = obs_size

    
    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]  
        bsz, num_heads, q_len, head_dim = query_states.shape   

        return key_states, value_states


    def update_kv_in_decoding(self, key_states, query_states, value_states, attention_mask, 
                    num_key_value_groups, obs_size, layer_index, obs_count=1):
        layer_idx = layer_index
        assert query_states.shape[-2] == 1
        
        bsz, num_heads, k_len, head_dim = key_states.shape
        q_len = query_states.shape[-2]

        window_size = self.decoding_recent_size
        kv_capacity = self.max_kv_capacity
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # print("obs_size:", self.obs_size)
        TempCache.record_recurrence(attn_weights, layer_idx, obs_count)

        if k_len <= kv_capacity:
            return key_states, value_states, attn_weights

        if obs_count % self.obs_size != 0:
            return key_states, value_states, attn_weights
        

        s = torch.where(TempCache.layer_max_period[layer_idx] == 0, torch.tensor(0.0), 2/(1+torch.exp((TempCache.layer_max_period[layer_idx]-1))))
        importance_scores = 2/(1+torch.exp(1+(obs_count - TempCache.layer_last_recurrent[layer_idx]) / (TempCache.layer_max_period[layer_idx] + 1e-5))) + s

        non_recent_len = kv_capacity - obs_size

        imp_scores_non_recent = importance_scores[:, :, :-obs_size]
        
        # Get top-k indices for non-recent positions
        _, top_indices = imp_scores_non_recent.topk(non_recent_len, dim=-1)

        max_period_cur = TempCache.layer_max_period[layer_idx][:, :, -obs_size:]
        last_recurrent_cur = TempCache.layer_last_recurrent[layer_idx][:, :, -obs_size:]
        max_period_compress = TempCache.layer_max_period[layer_idx].gather(dim=2, index=top_indices)
        last_recurrent_compress = TempCache.layer_last_recurrent[layer_idx].gather(dim=2, index=top_indices)
        TempCache.layer_max_period[layer_idx] = torch.cat([max_period_compress, max_period_cur], dim = 2)
        TempCache.layer_last_recurrent[layer_idx] = torch.cat([last_recurrent_compress, last_recurrent_cur], dim = 2)

        indices = top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        
        k_selected = key_states.gather(dim=2, index=indices)
        v_selected = value_states.gather(dim=2, index=indices)
     
        k_cur = key_states[:, :, -obs_size:, :]
        v_cur = value_states[:, :, -obs_size:, :]
        
        key_states = torch.cat([k_selected, k_cur], dim=2)
        value_states = torch.cat([v_selected, v_cur], dim=2)

        return key_states, value_states, attn_weights

def init_Window_LAZY(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'decoding_recent_size'):
            self.config.decoding_recent_size = 256
        if not hasattr(self.config, 'max_kv_capacity'):
            self.config.max_kv_capacity = 256
        if not hasattr(self.config, 'obs_size'):
            self.config.obs_size = 3
    
    
    self.kv_cluster = Window_LAZYKVCluster(
        num_hidden_layers = num_hidden_layers,
        decoding_recent_size=self.config.decoding_recent_size,
        max_kv_capacity=self.config.max_kv_capacity,
        obs_size=self.config.obs_size
        )

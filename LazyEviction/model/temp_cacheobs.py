import torch
import numpy as np
import os

class TempCache:
    """全局临时缓存类，用于按层存储和管理注意力权重"""
    
    # 静态变量存储注意力权重
    layer_max_period = {}    # 每层最大权重值的周期
    layer_last_recurrent = {}    # 每层上次最大权重值的step
    
    @classmethod
    def record_recurrence(cls, weights, layer_idx,current_decoding_step):
        """
        添加新的注意力权重到指定层的缓存
        
        Args:
            weights: torch.Tensor, 注意力权重张量
            layer_idx: int, 模型层索引
        """

        # 获取 attn_weights 的形状
        bsz, num_heads, _, k_len = weights.shape
        
        # 为该层初始化缓存列表（如果不存在）
        if layer_idx not in cls.layer_last_recurrent:
            cls.layer_last_recurrent[layer_idx] = torch.zeros_like(weights).squeeze(2)  # 移除q_len维度
            cls.layer_max_period[layer_idx] = torch.zeros_like(weights).squeeze(2)  # 初始化最大周期为0
        else:
            # decoding阶段，k_len+1，扩展缓存列表, 扩展时填充的初始值是 current_decoding_step - 1
            padding = (current_decoding_step - 1)*torch.ones((bsz, num_heads, 1), dtype=torch.int32, device=weights.device)
            padding_period = torch.zeros((bsz, num_heads, 1), dtype=torch.int32, device=weights.device) 
            # print(f"padding shape: {padding.shape}, cls.layer_last_recurrent[layer_idx] shape: {cls.layer_last_recurrent[layer_idx].shape}")
            cls.layer_last_recurrent[layer_idx] = torch.cat([cls.layer_last_recurrent[layer_idx], padding], dim=2)
            cls.layer_max_period[layer_idx] = torch.cat([cls.layer_max_period[layer_idx], padding_period], dim=2)
        
        old_layer_last_recurrent = cls.layer_last_recurrent[layer_idx].clone() # 记录更新前的last_recurrent，算周期时会用到
        
        # 找到当前 step 中重要的 token（attn_weights > alpha）
        important_mask = weights.squeeze(2) >= cls.alpha  # [bsz, num_heads, k_len]
        
        # 更新 last_important_step：重要的 token 设置为 current_step
        cls.layer_last_recurrent[layer_idx][important_mask] = current_decoding_step
        
        # 计算周期：当前 step - 上次重要 step
        period = cls.layer_last_recurrent[layer_idx] - old_layer_last_recurrent
        cls.layer_max_period[layer_idx] = torch.maximum(cls.layer_max_period[layer_idx], period)
        

    @classmethod
    def reset(cls, layer_idx=None):
        if layer_idx is None:
            cls.layer_max_period = {}    # 每层最大权重值的周期
            cls.layer_last_recurrent = {}    # 每层上次最大权重值的step
            
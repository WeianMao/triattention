from transformers.models.llama import modeling_llama
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen3 import modeling_qwen3
from .modeling import (
    LlamaAttention_init,
    LlamaAttention_forward,
    Qwen2Attention_init,
    Qwen2Attention_forward,
    Qwen3Attention_init,
    Qwen3Attention_forward,
    CausalLM_forward,
)


def replace_llama(compression_config):
    def init_wrapper(self, config, layer_idx):
        LlamaAttention_init(self, config, layer_idx, compression_config)

    modeling_llama.LlamaAttention.__init__ = init_wrapper
    modeling_llama.LlamaAttention.forward = LlamaAttention_forward
    modeling_llama.LlamaForCausalLM.forward = CausalLM_forward


def replace_qwen2(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen2Attention_init(self, config, layer_idx, compression_config)

    modeling_qwen2.Qwen2Attention.__init__ = init_wrapper
    modeling_qwen2.Qwen2Attention.forward = Qwen2Attention_forward
    modeling_qwen2.Qwen2ForCausalLM.forward = CausalLM_forward

def replace_qwen3(compression_config):
    def init_wrapper(self, config, layer_idx):
        Qwen3Attention_init(self, config, layer_idx, compression_config)

    modeling_qwen3.Qwen3Attention.__init__ = init_wrapper
    modeling_qwen3.Qwen3Attention.forward = Qwen3Attention_forward
    modeling_qwen3.Qwen3ForCausalLM.forward = CausalLM_forward
from importlib.metadata import version
import transformers

from importlib.metadata import version
import transformers


def replace_llama(method):

    from model.llama_model import llama_flash_attn2_forward_Window_LAZY,llama_attn_forward_Window_LAZY,llama_sdpa_attn_forward_Window_LAZY
   
    if method == "window_lazy":
        print("Using Window_LAZY!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_Window_LAZY
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_Window_LAZY
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_Window_LAZY
        

def replace_qwen(method):

    from model.qwen_model import qwen2_sdpa_attn_forward_Window_LAZY, qwen2_flash_attn2_forward_Window_LAZY, qwen2_attn_forward_Window_LAZY

    if method == "window_lazy":
        print("Using Window_LAZY!")
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attn_forward_Window_LAZY
        transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention.forward = qwen2_sdpa_attn_forward_Window_LAZY
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_flash_attn2_forward_Window_LAZY

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Union

import torch
import torch.nn as nn
import transformers
from accelerate.big_modeling import (init_empty_weights,
                                     load_checkpoint_and_dispatch)
from module import get_named_linears, set_op_by_name
from tqdm import tqdm
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from typing_extensions import Annotated, Doc

# from turbomind import Linear
import turbomind


class TurbomindForCausalLM(nn.Module):

    def __init__(
        self,
        model,
        is_quantized,
        config,
        quant_config,
    ):
        """The base model for all AutoAWQ models.

        Args:
            model: The pretrained or quantized model.
            is_quantized: Indicates if the current model is quantized
            config: The config of the model.
            quant_config: The quantization config of the model.
        """
        super().__init__()
        self.model: PreTrainedModel = model
        self.is_quantized: bool = is_quantized
        self.search_result = None
        self.config: PretrainedConfig = config
        self.quant_config = quant_config

    def to(self, device: Annotated[str,
                                   Doc('The device to move your model to.')]):
        """A utility function for moving the model to a device."""
        return self.model.to(device)

    def forward(self, *args, **kwargs):
        """A forward function that mimics the torch forward."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """A generate function that mimics the HF generate function."""
        with torch.inference_mode():
            return self.model.generate(*args, **kwargs)

    @classmethod
    def from_quantized(self,
                       model_path: str,
                       torch_dtype: torch.dtype = torch.float16,
                       device_map: Union[str, Dict] = 'balanced',
                       **config_kwargs: Dict):
        """A method for initialization of a quantized model, usually in INT4.

        Args:
            model_path (str): The model path
            max_seq_len (int): The maximum sequence cached sequence length of
                the model. Larger values may increase loading time and
                memory usage.
            torch_dtype: The dtype to load the model as. May not work with
                other values than float16.
            device_map: A device map that will be passed onto the model
                loading method from transformers.
        **config_kwargs: Additional kwargs that are passed to the config
            during initialization
        """
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        quant_config = config.quantization_config

        target_cls = getattr(transformers, config.architectures[0])

        # Load model
        with init_empty_weights():
            model = target_cls._from_config(config=config,
                                            torch_dtype=torch_dtype)
        # Prepare quantized linear layers, replace nn.Linear
        self._load_quantized_modules(
            self,
            model,
            quant_config,
        )

        model.tie_weights()

        # loads the weights into modules and distributes
        # across available devices automatically
        load_checkpoint_and_dispatch(
            model,
            checkpoint=model_path,
            device_map=device_map,
            no_split_module_classes=[model.model.layers[0].__class__.__name__],
            dtype=torch_dtype,
        )

        # model = turbomind_post_init(model)
        for _, submodule in model.named_modules():
            if isinstance(submodule, turbomind.Linear):
                submodule.post_init()

        model.eval()

        return self(
            model,
            is_quantized=True,
            config=config,
            quant_config=quant_config,
        )

    def _load_quantized_modules(self, model, quant_config):
        assert quant_config['quant_method'] in ['awq', 'gptq']
        if quant_config['quant_method'] == 'awq':
            assert quant_config['version'] == 'gemm'

        # Get blocks of model
        layers = model.model.layers

        for i in tqdm(range(len(layers)), desc='Replacing layers...'):
            layer = layers[i]

            # Get every linear layer in a block
            named_linears = get_named_linears(layer)

            # # Filter out the linear layers we don't want to include
            # named_linears = exclude_layers_to_not_quantize(
            #     named_linears, quant_config.modules_to_not_convert)

            # Replace nn.Linear with turbomind Linear
            for name, module in named_linears.items():
                q_linear_module = turbomind.Linear
                q_linear = q_linear_module.from_linear(
                    module, quant_config['bits'], quant_config['group_size'],
                    quant_config['quant_method'], True)
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)

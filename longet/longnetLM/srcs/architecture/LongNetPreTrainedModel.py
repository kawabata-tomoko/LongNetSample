import math

import torch.nn.init as init
from longnetLM.config.config import LongnetConfig
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel


class LongNetPreTrainedModel(PreTrainedModel):
    is_encoder_decoder = True
    config_class = LongnetConfig
    base_model_prefix = "ln"
    supports_gradient_checkpointing = False
    _skip_keys_device_placement = "past_key_values"
    def __init__(self, config, *inputs, **kwargs):
        self.config = config
        super().__init__(config, *inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            try:
                init.xavier_uniform_(
                    module.weight,
                    gain=init.calculate_gain(self.config.activation_fn),
                )
            except ValueError:
                init.xavier_uniform_(
                    module.weight,
                    gain=1 / math.sqrt(2),
                )
            if module.bias is not None:
                init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def post_init(self):
        super().post_init()
        if self.config.deepnorm:
            if self.is_encoder_decoder:
                init_scale = (
                    math.pow(
                        math.pow(self.config.encoder_layers, 4)
                        * self.config.decoder_layers,
                        0.0625,
                    )
                    / 1.15
                )
            else:
                init_scale = math.pow(8.0 * self.config.encoder_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.div_(init_scale)

        if self.config.subln:
            if self.is_encoder_decoder:
                init_scale = math.sqrt(
                    math.log(3 * self.config.decoder_layers)
                    * math.log(2 * self.config.encoder_layers)
                    / 3
                )
            else:
                init_scale = math.sqrt(math.log(self.config.encoder_layers * 2))
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.mul_(init_scale)
    
    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func = checkpoint):
        is_gradient_checkpointing_set = False

        # Apply it on the top-level module in case the top-level modules supports it
        # for example, LongT5Stack inherits from PreTrainedModel.
        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " gradient_checkpointing to modules of the model that uses checkpointing."
            )
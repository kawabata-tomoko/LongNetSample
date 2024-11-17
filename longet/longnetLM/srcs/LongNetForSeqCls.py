import torch
import torch.nn as nn
import torch.nn.functional as F
from longnetLM.srcs.architecture import (
    LongNetDecoderLM,
    LongNetEncoderDecoderLM,
    LongNetEncoderLM,
    LongNetPreTrainedModel,
)
from longnetLM.srcs.architecture.utils import ACTIVATE_FN
from transformers.modeling_outputs import (
    CausalLMOutput,
    Seq2SeqLMOutput,
    SequenceClassifierOutput,
)


class SequenceClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.hidden = torch.nn.Linear(
            config.embed_dim, config.embed_dim * 2, dtype=torch.float32
        )  # .to(torch.bfloat16)
        self.classifier = torch.nn.Linear(
            config.embed_dim * 2, self.num_labels, dtype=torch.float32
        )  # .to(torch.bfloat16)#load as bf16
        self.ln_hidden = torch.nn.LayerNorm(config.embed_dim * 2, dtype=torch.float32)

    def forward(self, features):
        x = self.hidden(features.to(self.hidden.weight.dtype))
        x = ACTIVATE_FN[getattr(self.config,"activation_fn", "gelu")](x)
        x = self.ln_hidden(x)
        return self.classifier(x)


class LongNetEncoderForSequenceClassification(LongNetPreTrainedModel):
    supports_gradient_checkpointing = True
    def __init__(self, config,**kwargs):
        self.config = config
        super().__init__(config=config,**kwargs)
        self.model = LongNetEncoderLM(config)
        self.lm_head = SequenceClassificationHead(config)
        self.post_init()
            
    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func):
        self.model.gradient_checkpointing = enable
        super()._set_gradient_checkpointing(enable, gradient_checkpointing_func)
    
    def get_input_embeddings(self):
        return self.model.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.model.embedding.word_embeddings = value
    def forward(
        self,
        input_ids,
        labels,
        attention_mask=None,
        embedding_mask=None,
        position_ids=None,
        inputs_embeds=None,
        output_hidden_states=False,
        past_key_values=None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            embedding_mask=embedding_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            labels = labels.to(prediction_scores.device)
            loss = loss_fct(
                prediction_scores.view(-1, self.lm_head.num_labels), labels.view(-1)
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states if output_hidden_states else None
        )


class LongNetDecoderForSequenceClassification(LongNetPreTrainedModel):
    supports_gradient_checkpointing = True
    def __init__(self, config,**kwargs):
        super().__init__(config,**kwargs)
        self.config = config
        self.model = LongNetDecoderLM(config)
        self.lm_head = SequenceClassificationHead(config)
        self.post_init()
            
    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func):
        self.model.gradient_checkpointing = enable
        super()._set_gradient_checkpointing(enable, gradient_checkpointing_func)
    
    def get_input_embeddings(self):
        return self.model.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.model.embedding.word_embeddings = value
        
    def forward(
        self,
        input_ids,
        labels,
        eos_index=None,
        attention_mask=None,
        embedding_mask=None,
        position_ids=None,
        inputs_embeds=None,
        output_hidden_states=False,
        past_key_values=None,
        encoder_outputs=None,
        incremental_state=None,
        **kwargs,
    ):
        # eos_index = (
        #     eos_index
        #     if eos_index is not None
        #     else torch.ones(input_ids.shape[0], 1, dtype=int) * input_ids.shape[1] - 1
        # )

        # Language Model Head
        decoder_out = self.model(
            input_ids,
            attention_mask=attention_mask,
            embedding_mask=embedding_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            encoder_outputs=encoder_outputs,
            incremental_state=incremental_state,
        )

        hidden_state = decoder_out[0]
        # hidden_state = hidden_state.to(dtype=self.lm_head.hidden.weight.dtype).gather(
        #     1, eos_index.unsqueeze(-1).expand(-1, -1, hidden_state.size(-1)).to(device=hidden_state.device)
        # )
        # prediction_scores = self.lm_head(hidden_state)
        prediction_scores = self.lm_head(hidden_state[:, -1, :])
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            labels = labels.to(prediction_scores.device)
            loss = loss_fct(
                prediction_scores.view(-1, self.lm_head.num_labels), labels.view(-1)
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=decoder_out.hidden_states,
        )


# Not Implemented!˝

# class LongNetForSeq2SeqLM(LongNetPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config=config)
#         self.config = config
#         self.model = LongNetEncoderDecoderLM(config)

#     def forward(
#         self,
#         input_ids,
#         prev_output_tokens,
#         labels=None,
#         attention_mask=None,
#         embedding_mask=None,
#         position_ids=None,
#         inputs_embeds=None,
#         output_hidden_states=False,
#         past_key_values=None,
#         encoder_outputs=None,
#         incremental_state=None,
#         **kwargs,
#     ) -> Seq2SeqLMOutput:
#         # Encoder-Decoder model
#         outputs = self.model(
#             input_ids=input_ids,
#             prev_output_tokens=prev_output_tokens,
#             attention_mask=attention_mask,
#             embedding_mask=embedding_mask,
#             position_ids=position_ids,
#             inputs_embeds=inputs_embeds,
#             output_hidden_states=output_hidden_states,
#             past_key_values=past_key_values,
#             encoder_outputs=encoder_outputs,
#             incremental_state=incremental_state,
#             **kwargs,
#         )

#         # Language Model Head
#         lm_logits = torch.matmul(
#             outputs.last_hidden_state,
#             self.model.embedding.word_embeddings.weight.T,
#         )
#         loss = None
#         if labels is not None:
#             # Shift labels so that tokens < n predict n
#             shift_logits = lm_logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(
#                 shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
#             )
#         return Seq2SeqLMOutput(
#             loss=loss,
#             logits=lm_logits,
#             past_key_values=outputs.past_key_values,
#             decoder_hidden_states=(
#                 outputs.decoder_hidden_states if output_hidden_states else None
#             ),
#             encoder_last_hidden_state=outputs.encoder_last_hidden_state,
#             encoder_hidden_states=(
#                 outputs.encoder_hidden_states if output_hidden_states else None
#             ),
#         )

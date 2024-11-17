import torch
import torch.nn as nn
import torch.nn.functional as F
from longnetLM.srcs.architecture import (
    LongNetDecoderLM,
    LongNetEncoderDecoderLM,
    LongNetEncoderLM,
    LongNetPreTrainedModel,
)
from transformers.modeling_outputs import (
    CausalLMOutput,
    MaskedLMOutput,
    Seq2SeqLMOutput,
)


class MaskedLMHead(nn.Module):
    """ESM Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias
        return x


class LongNetEncoderForMaskedLM(LongNetPreTrainedModel):
    def __init__(self, config):
        self.config = config
        super().__init__(config=config)
        self.model = LongNetEncoderLM(config)
        self.lm_head = MaskedLMHead(config)
        self.post_init()

    def forward(
        self,
        input_ids,
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
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            labels = labels.to(prediction_scores.device)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LongNetDecoderForCasualLM(LongNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = LongNetDecoderLM(config)

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
        encoder_outputs=None,
        incremental_state=None,
        **kwargs,
    ):
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
        lm_logits = torch.matmul(
            decoder_out.last_hidden_state,
            self.model.embedding.word_embeddings.weight.T,
        )
        loss = None
        if labels is not None:
            # Shift labels so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=past_key_values,
            hidden_states=decoder_out.hidden_states if output_hidden_states else None,
            attentions=None,
        )


class LongNetForSeq2SeqLM(LongNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config=config)
        self.config = config
        self.model = LongNetEncoderDecoderLM(config)

    def forward(
        self,
        input_ids,
        prev_output_tokens,
        labels=None,
        attention_mask=None,
        embedding_mask=None,
        position_ids=None,
        inputs_embeds=None,
        output_hidden_states=False,
        past_key_values=None,
        encoder_outputs=None,
        incremental_state=None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        # Encoder-Decoder model
        outputs = self.model(
            input_ids=input_ids,
            prev_output_tokens=prev_output_tokens,
            attention_mask=attention_mask,
            embedding_mask=embedding_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            encoder_outputs=encoder_outputs,
            incremental_state=incremental_state,
            **kwargs,
        )

        # Language Model Head
        lm_logits = torch.matmul(
            outputs.last_hidden_state,
            self.model.embedding.word_embeddings.weight.T,
        )
        loss = None
        if labels is not None:
            # Shift labels so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=(
                outputs.decoder_hidden_states if output_hidden_states else None
            ),
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=(
                outputs.encoder_hidden_states if output_hidden_states else None
            ),
        )

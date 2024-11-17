import torch
import torch.nn as nn
import torch.nn.functional as F
from longnetLM.srcs.architecture import LongNetPreTrainedModel
from longnetLM.srcs.architecture.layers import Decoder, Encoder, LongNetEmbeddings
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput


class LongNetEncoderLM(LongNetPreTrainedModel):
    supports_gradient_checkpointing = True
    is_encoder_decoder=False
    
    def __init__(self, config):
        super().__init__(config=config)
        self.config = config
        self.embedding = LongNetEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_init()
            
    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func):
        self.gradient_checkpointing = enable
        super()._set_gradient_checkpointing(enable, gradient_checkpointing_func)
    
    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value

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
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        # Check embedding_mask
        if embedding_mask is None:
            embedding_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length), device=device
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length), device=device
            )

        embeddings = self.embedding(
            input_ids,
            position_ids=position_ids,
            attention_mask=embedding_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            incremental_state=None,
        )
        outputs = self.encoder(
            embeddings=embeddings,
            padding_mask=None,
            attn_mask=attention_mask,
            incremental_state=None,
            output_hidden_states=output_hidden_states,
        )

        return BaseModelOutput(
            last_hidden_state=outputs["hidden_state"],
            hidden_states=(
                outputs["all_hidden_states"] if output_hidden_states else None
            ),
            attentions=None,
        )


class LongNetDecoderLM(LongNetPreTrainedModel):
    supports_gradient_checkpointing = True
    is_encoder_decoder=False
    
    def __init__(self, config):
        super().__init__(config=config)
        self.config = config
        self.embedding = LongNetEmbeddings(config)
        self.decoder = Decoder(config)
        self.post_init()
            
    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func):
        self.gradient_checkpointing = enable
        super()._set_gradient_checkpointing(enable, gradient_checkpointing_func)
    
    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value
    def forward(
        self,
        input_ids,
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
        # Decoder
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        # Check embedding_mask
        if embedding_mask is None:
            embedding_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length), device=device
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length), device=device
            )

        embeddings = self.embedding(
            input_ids,
            position_ids=position_ids,
            attention_mask=embedding_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            incremental_state=None,
        )
        decoder_out = self.decoder(
            embeddings,
            padding_mask=attention_mask,
            encoder_out=encoder_outputs,
            output_hidden_states=output_hidden_states,
            incremental_state=incremental_state,
            **kwargs,
        )
        return BaseModelOutput(
            last_hidden_state=decoder_out["hidden_state"],
            hidden_states=(
                decoder_out["all_hidden_states"] if output_hidden_states else None
            ),
            attentions=None,
        )


class LongNetEncoderDecoderLM(LongNetPreTrainedModel):
    supports_gradient_checkpointing = True
    def __init__(self, config):
        super().__init__(config=config)
        self.config = config
        self.embedding = LongNetEmbeddings(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)  # Assuming Decoder is defined elsewhere
        self.post_init()
            
    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func):
        self.gradient_checkpointing = enable
        super()._set_gradient_checkpointing(enable, gradient_checkpointing_func)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value
        
    def forward(
        self,
        input_ids,
        prev_output_tokens,
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
        # Embedding
        embeddings = self.embedding(input_ids)
        prev_output_embeddings = self.embedding(prev_output_tokens)

        # Encoder
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        # Check embedding_mask
        if embedding_mask is None:
            embedding_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length), device=device
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length), device=device
            )

        embeddings = self.embedding(
            input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            incremental_state=None,
        )
        encoder_out = self.encoder(
            embeddings=embeddings,
            padding_mask=None,
            attn_mask=attention_mask,
            incremental_state=None,
            output_hidden_states=output_hidden_states,
        )

        # Decoder
        decoder_out = self.decoder(
            prev_output_embeddings,
            padding_mask=attention_mask,
            encoder_out=encoder_outputs,
            output_hidden_states=output_hidden_states,
            incremental_state=incremental_state,
            **kwargs,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_out["hidden_state"],
            decoder_hidden_states=(
                decoder_out["all_hidden_states"] if output_hidden_states else None
            ),
            encoder_last_hidden_state=encoder_out["hidden_state"],
            encoder_hidden_states=(
                encoder_out["all_hidden_states"] if output_hidden_states else None
            ),
        )

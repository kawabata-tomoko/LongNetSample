# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
from transformers import PretrainedConfig
from .LongNetTokenizer import LongNetTokenizer

class LongnetConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        ##initialize
        print(kwargs)
        self.__dict__.update(kwargs)
        self.tokenizer_class=LongNetTokenizer
        self.bos_token_id=kwargs.pop("bos_token_id", 1)
        self.pad_token_id=kwargs.pop("pad_token_id", 1)
        self.eos_token_id=kwargs.pop("eos_token_id", 2)
        self.mask_token_id=kwargs.pop("mask_token_id", 4)
        super().__init__(
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            **kwargs
            )
        self.initializer_range=kwargs.pop("initializer_range", 0.02)
        self.vocab_size = kwargs.pop("vocab_size", -1)
        self.num_labels = kwargs.pop("num_labels", 2)
        
     
        self.dropout = kwargs.pop("dropout", 0.0)
        self.drop_path_rate = kwargs.pop("drop_path_rate", 0.0)
 
        self.deepnorm = kwargs.pop("deepnorm", False)
        self.subln = kwargs.pop("subln", True)
        self.layernorm_eps = kwargs.pop(
            "layernorm_eps", 1e-6
        )
        
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)
        self.activation_dropout = kwargs.pop("activation_dropout", 0.0)
        self.activation_fn = kwargs.pop("activation_fn", "gelu")
        self.attention_type=kwargs.pop("attention_type","DilatedAttention")
        
        self.embed_dim = kwargs.pop("embed_dim", 768)
        self.no_scale_embedding = kwargs.pop("no_scale_embedding", True)
        self.layernorm_embedding = kwargs.pop("layernorm_embedding", False)
        self.rel_pos_buckets = kwargs.pop("rel_pos_buckets", 0)
        self.max_rel_pos = kwargs.pop("max_rel_pos", 0)
        self.max_positions = kwargs.pop("max_positions", 65536)

        self.num_attention_heads = kwargs.pop("num_attention_heads", 12)
        self.encoder_ffn_embed_dim = kwargs.pop("encoder_ffn_embed_dim", 3072)
        self.encoder_layers = kwargs.pop("encoder_layers", 12)
        self.encoder_normalize_before = kwargs.pop("encoder_normalize_before", True)
        self.normalize_output = kwargs.pop("normalize_output", True)
    
        self.decoder_attention_heads = kwargs.pop("decoder_attention_heads", 12)
        self.decoder_ffn_embed_dim = kwargs.pop(
            "decoder_ffn_embed_dim", 1280
        )
        self.decoder_layers = kwargs.pop("decoder_layers", 12)
        self.decoder_normalize_before = kwargs.pop("decoder_normalize_before", True)
        

        # Dilated Attention
        self.flash_attention = kwargs.pop("flash_attention", False)
        self.segment_length = kwargs.pop("segment_length", None)
        self.dilated_ratio = kwargs.pop("dilated_ratio", None)
        self.seq_parallel = kwargs.pop("seq_parallel", False)


        self.postprocessing()
     

    def to_dict(self):
        return super().to_dict()

    def override(self, args):
        for hp in self.__dict__.keys():
            if getattr(args, hp, None) is not None:
                self.__dict__[hp] = getattr(args, hp, None)
        self.postprocessing()

    def postprocessing(self):
        if self.segment_length is not None and self.segment_length != "" and type(self.segment_length)==str:
            self.segment_length = eval(self.segment_length)
        if self.dilated_ratio is not None and self.dilated_ratio != "" and type(self.dilated_ratio)==str:
            self.dilated_ratio = eval(self.dilated_ratio)

        if self.deepnorm:
            self.encoder_normalize_before = False
            self.subln = False
        if self.subln:
            self.encoder_normalize_before = True
            self.deepnorm = False


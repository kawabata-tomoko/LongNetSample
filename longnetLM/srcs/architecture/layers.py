# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Licensed under The MIT License [see LICENSE for details]

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from longnetLM.srcs.architecture.LongNetPreTrainedModel import LongNetPreTrainedModel
from longnetLM.srcs.architecture.utils import (
    all_gather_func,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    padding_to_multiple_of,
    ACTIVATE_FN
)
from timm.models.layers import drop_path
from torch import nn
from torch.nn import LayerNorm

try:
    from flash_attn.flash_attn_interface import flash_attn_func
except ModuleNotFoundError:
    flash_attn_func=None




class RelativePositionBias(nn.Module):
    def __init__(
        self, bidirectional=True, num_buckets=32, max_distance=128, n_heads=12
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen, step=None):
        step = 0 if step is None else step
        context_position = torch.arange(
            step,
            step + qlen,
            dtype=torch.long,
            device=self.relative_attention_bias.weight.device,
        )[:, None]
        memory_position = torch.arange(
            klen, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)

        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(
            rp_bucket
        )  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, qlen, klen)
        return values

    def forward(self, batch_size, qlen, klen, step=None):
        # shape (batch * num_heads, qlen, klen)
        return (
            self.compute_bias(qlen, klen, step)
            .repeat(batch_size, 1, 1, 1)
            .view(-1, qlen, klen)
        )


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
        layernorm_eps,
        subln=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = ACTIVATE_FN.get(str(activation_fn), None)
        if self.activation_fn is None:
            raise NotImplementedError
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x)
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return "p={}".format(self.drop_prob)


def create_position_ids_from_embeddings(
    embeddings, padding_idx, past_key_values_length=0
):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    config:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    mask = embeddings.ne(padding_idx).int()
    incremental_indices = (
        torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
    ) * mask
    return incremental_indices.long() + padding_idx


class LongNetEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.mask_token_id = config.mask_token_id
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id
        )

        if config.layernorm_embedding:
            self.layer_norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        else:
            self.layer_norm = None

        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_positions).expand((1, -1)),
            persistent=False,
        )
        self.embed_scale = (
            1.0 if config.no_scale_embedding else math.sqrt(config.embed_dim)
        )

        self.position_embeddings = nn.Embedding(
            config.max_positions,
            config.embed_dim,
            padding_idx=self.padding_idx,
        )

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        past_key_values_length=0,
        incremental_state=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_embeddings(
                    input_ids, self.padding_idx, past_key_values_length
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds
                )

        if incremental_state is not None and not self.is_first_step(incremental_state):
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[:, -1:]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        x = self.embed_scale * inputs_embeds

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            x += position_embeddings

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if attention_mask is not None:
            x = (x * attention_mask.unsqueeze(-1)).to(
                x.dtype
            )
        x = self.dropout(x)

        return x

    def is_first_step(self, incremental_state):
        if incremental_state is None:
            return False
        return incremental_state.get("is_first_step", False)

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        config:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        config,
        self_attention=False,
        encoder_decoder_attention=False,
        subln=False,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.inner_attn_ln = (
            LayerNorm(self.embed_dim, eps=config.layernorm_eps)
            if subln and self.self_attention
            else None
        )
        self.dropout_module = nn.Dropout(self.dropout)

    def attention_ops(
        self,
        q,
        k,
        v,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
        is_causal=False,
    ):
        if not self.config.flash_attention:
            q *= self.scaling
            attn_weights = torch.bmm(q, k.transpose(1, 2))

            if attn_mask is not None:
                attn_weights = torch.nan_to_num(attn_weights)
                attn_mask = attn_mask.unsqueeze(0)
                attn_weights += attn_mask

            if key_padding_mask is not None:
                attn_weights = rearrange(
                    attn_weights, "(b h) t s -> b h t s", h=self.num_heads
                )
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
                attn_weights = rearrange(attn_weights, "b h t s -> (b h) t s")

            if rel_pos is not None:
                rel_pos = rel_pos.view(attn_weights.size())
                attn_weights = attn_weights + rel_pos

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
                attn_weights
            )
            attn_probs = self.dropout_module(attn_weights)

            attn = torch.bmm(attn_probs, v)
            attn = rearrange(attn, "(b h) l d -> b l (h d)", h=self.num_heads)
        else:
            assert rel_pos is None
            assert flash_attn_func is not None
            q = rearrange(q, "(b h) l d -> b l h d", h=self.num_heads)
            k = rearrange(k, "(b h) l d -> b l h d", h=self.num_heads)
            v = rearrange(v, "(b h) l d -> b l h d", h=self.num_heads)
            attn, lse, _ = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout,
                softmax_scale=None,
                causal=is_causal,
                return_attn_probs=True,
            )
            attn = rearrange(attn, "b l h d -> b l (h d)")
            attn_weights = lse[:, :, : attn.size(1)]

        return attn, attn_weights

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
        is_first_step=False,
        is_causal=False,
    ):
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = rearrange(q, "b l (h d) -> (b h) l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> (b h) l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> (b h) l d", h=self.num_heads)

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                prev_value = incremental_state["prev_value"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            incremental_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            src_len = k.size(1)

        attn, attn_weights = self.attention_ops(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rel_pos=rel_pos,
            is_causal=is_causal,
        )

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)

        return attn, attn_weights


class DilatedAttention(MultiheadAttention):

    def dense_to_sparse(self, x, ratio):
        length = x.size(1)
        padding = padding_to_multiple_of(length, ratio)
        head_padding = padding_to_multiple_of(self.num_heads, ratio)

        if padding > 0 or head_padding > 0:
            x = F.pad(x, (0, 0, 0, head_padding, 0, padding), value=0.0)

        x = rearrange(x, "b (l r1) (r2 h) d -> b l h d r1 r2", r1=ratio, r2=ratio)
        x = torch.diagonal(x, offset=0, dim1=4, dim2=5)
        x = rearrange(x, "b l h d r -> b l (r h) d")

        if head_padding > 0:
            x = x[:, :, : self.num_heads]

        return x

    def sparse_to_dense(self, out, lse, ratio):
        head_padding = padding_to_multiple_of(self.num_heads, ratio)

        if head_padding > 0:
            out = F.pad(out, (0, 0, 0, head_padding), value=0.0)
            lse = F.pad(lse, (0, 0, 0, head_padding), value=-1e8)

        out = rearrange(out, "b l (r h) d -> b l h d r", r=ratio)
        out = torch.diag_embed(out, offset=0, dim1=4, dim2=5)
        out = rearrange(out, "b l h d r1 r2 -> b (r2 h) (l r1) d", r1=ratio, r2=ratio)

        lse = rearrange(lse, "b (r h) l -> b l h r", r=ratio)
        lse = torch.diag_embed(lse, offset=0, dim1=3, dim2=4)
        lse = lse.masked_fill_(lse == 0, -1e8)
        lse = rearrange(lse, "b l h r1 r2 -> b (r2 h) (l r1) 1", r1=ratio, r2=ratio)

        if head_padding > 0:
            out = out[:, : self.num_heads]
            lse = lse[:, : self.num_heads]

        return out, lse

    def gather_kv(self, x, sl, seq_len, is_causal=True):
        bsz = x.size(0)
        assert sl % seq_len == 0
        num_rank_per_segment = sl // seq_len

        x = all_gather_func(x)
        current_rank = get_data_parallel_rank()
        x = rearrange(x, "(w b) l h d -> w b l h d", b=bsz)

        if is_causal:
            if current_rank > 0:
                x = x[:current_rank]
            else:
                x = x[:1] * 0

        current_segment = current_rank // num_rank_per_segment * num_rank_per_segment
        x = x[current_segment : current_segment + num_rank_per_segment]

        x = rearrange(x, "w b l h d -> b (w l) h d")
        return x

    def gathering(
        self, x, dr, sl, is_causal=True, offset=0, is_kv=False, seq_parall=True
    ):

        curr_x = x
        if offset > 0:
            curr_x = F.pad(curr_x, (0, 0, 0, 0, offset % sl, 0), value=0.0)
        seq_len = curr_x.size(1)
        should_gather_kv = (
            is_kv
            and (get_data_parallel_world_size() > 1)
            and (sl > seq_len)
            and seq_parall
        )
        _sl = sl
        sl = min(sl, seq_len)
        padding = padding_to_multiple_of(seq_len, sl)

        if padding > 0:
            curr_x = F.pad(curr_x, (0, 0, 0, 0, 0, padding), value=0.0)

        curr_x = rearrange(curr_x, "b (n g) h d -> (b n) g h d", g=sl)
        curr_x = self.dense_to_sparse(curr_x, dr)

        if should_gather_kv:
            curr_x = self.gather_kv(curr_x, _sl, seq_len, is_causal)

        curr_x = rearrange(curr_x, "b l h d -> (b h) l d")

        return curr_x

    def scattering(self, outs, lses, seq_len, bsz, offset=0):
        assert len(outs) == len(lses)
        assert len(outs) % len(self.config.dilated_ratio) == 0
        all_outs, all_lses = [], []
        drs = self.config.dilated_ratio
        if len(outs) > len(drs):
            drs = drs * (len(outs) // len(drs))

        for dr, o, lse in zip(drs, outs, lses):
            o = rearrange(o, "b l (h d) -> b l h d", h=self.num_heads)
            o, lse = self.sparse_to_dense(o, lse, dr)
            o = rearrange(o, "(b n) h g d -> (b h) (n g) d", b=bsz)
            lse = rearrange(lse, "(b n) h g 1 -> (b h) (n g) 1", b=bsz)
            o = o[:, offset : offset + seq_len]
            lse = lse[:, offset : offset + seq_len]

            all_outs.append(o)
            all_lses.append(lse)

        with torch.no_grad():
            max_lse = torch.stack(all_lses, dim=0)
            max_lse = max_lse.max(0)[0]
            all_lses = [torch.exp(lse - max_lse) for lse in all_lses]
            lse_sum = torch.stack(all_lses, dim=0).sum(0)
            all_lses = [lse / lse_sum for lse in all_lses]

        out = 0
        for o, lse in zip(all_outs, all_lses):
            out += o * lse.type_as(o)
        out = rearrange(out, "(b h) l d -> b l (h d)", h=self.num_heads)

        return out

    def forward(
        self,
        query,
        key,
        value,
        incremental_state=None,
        key_padding_mask=None,
        attn_mask=None,
        rel_pos=None,
        is_first_step=False,
        is_causal=False,
    ):
        assert self.config.flash_attention
        assert rel_pos is None
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = rearrange(q, "b l (h d) -> (b h) l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> (b h) l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> (b h) l d", h=self.num_heads)

        if incremental_state is not None and not is_first_step:
            offset = src_len - 1
        else:
            offset = 0

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                prev_value = incremental_state["prev_value"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            incremental_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            src_len = k.size(1)

        q = rearrange(q, "(b h) l d -> b l h d", h=self.num_heads)
        k = rearrange(k, "(b h) l d -> b l h d", h=self.num_heads)
        v = rearrange(v, "(b h) l d -> b l h d", h=self.num_heads)

        outs, lses = [], []
        for sl, dr in zip(self.config.segment_length, self.config.dilated_ratio):
            ki = self.gathering(
                k,
                dr,
                sl,
                is_causal=is_causal,
                offset=0,
                is_kv=True,
                seq_parall=self.config.seq_parallel,
            )
            vi = self.gathering(
                v,
                dr,
                sl,
                is_causal=is_causal,
                offset=0,
                is_kv=True,
                seq_parall=self.config.seq_parallel,
            )
            qi = self.gathering(
                q,
                dr,
                sl,
                is_causal=is_causal,
                offset=offset,
                is_kv=False,
                seq_parall=self.config.seq_parallel,
            )

            out, lse = self.attention_ops(
                qi,
                ki,
                vi,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rel_pos=rel_pos,
                is_causal=is_causal,
            )

            outs.append(out)
            lses.append(lse)

        attn = self.scattering(outs, lses, tgt_len, bsz, offset=offset)

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)

        return attn, None


class EncoderLayer(nn.Module):
    def __init__(self, config, depth, is_encoder_decoder=False):
        super().__init__()
        self.config = config

        self.embed_dim = config.embed_dim
        self.normalize_before = config.encoder_normalize_before
        self.ffn_dim = config.encoder_ffn_embed_dim
        self.attention_type = config.attention_type
        if self.attention_type == "DilatedAttention":
            mha = DilatedAttention
        else:
            mha = MultiheadAttention

        self.self_attn = mha(
            config,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=config.subln,
        )

        self.dropout_module = torch.nn.Dropout(config.dropout)
        if config.drop_path_rate > 0:
            drop_path_prob = np.linspace(
                0, config.drop_path_rate, config.encoder_layers
            )[depth]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

        self.ffn = FeedForwardNetwork(
            self.embed_dim,
            self.ffn_dim,
            config.activation_fn,
            config.dropout,
            config.activation_dropout,
            config.layernorm_eps,
            config.subln,
        )

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, eps=config.layernorm_eps)
        self.final_layer_norm = LayerNorm(self.embed_dim, eps=config.layernorm_eps)
        if config.deepnorm:
            if is_encoder_decoder:
                self.alpha = (
                    math.pow(
                        math.pow(config.encoder_layers, 4) * config.decoder_layers,
                        0.0625,
                    )
                    * 0.81
                )
            else:
                self.alpha = math.pow(2.0 * config.encoder_layers, 0.25)
        else:
            self.alpha = 1.0

    def forward(
        self,
        x,
        padding_mask,
        attn_mask=None,
        rel_pos=None,
        incremental_state=None,
    ):

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
            rel_pos=rel_pos,
            incremental_state=incremental_state,
        )
        x = self.dropout_module(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = residual * self.alpha + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.ffn(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = residual * self.alpha + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class Encoder(LongNetPreTrainedModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        self.config = config
        self.is_encoder_decoder = config.is_encoder_decoder
        super().__init__(config=config, **kwargs)

        self.dropout_module = torch.nn.Dropout(config.dropout)

        embed_dim = config.embed_dim

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    config,
                    depth,
                    is_encoder_decoder=self.is_encoder_decoder,
                )
                for depth in range(config.encoder_layers)
            ]
        )

        self.num_layers = len(self.layers)

        if config.encoder_normalize_before and config.normalize_output:
            self.layer_norm = LayerNorm(embed_dim, eps=config.layernorm_eps)
        else:
            self.layer_norm = None

        if config.rel_pos_buckets > 0 and config.max_rel_pos > 0:
            self.relative_position = RelativePositionBias(
                num_buckets=config.rel_pos_buckets,
                max_distance=config.max_rel_pos,
                n_heads=config.num_attention_heads,
            )
        else:
            self.relative_position = None

    def forward(
        self,
        embeddings,
        padding_mask=None,
        attn_mask=None,
        incremental_state=None,
        output_hidden_states=False,
        **kwargs,
    ):
        assert embeddings is not None

        if padding_mask is None:
            padding_mask = torch.zeros_like(embeddings, device=embeddings.device).bool()

        hidden_state = embeddings * (1 - padding_mask.type_as(embeddings))

        rel_pos_bias = None
        if self.relative_position is not None:
            rel_pos_bias = self.relative_position(
                batch_size=hidden_state.size(0),
                qlen=hidden_state.size(1),
                klen=hidden_state.size(1),
            )

        # incremental_state is not None during inference if we use the bidirectional encoder as a generator as in s2s-ft (https://arxiv.org/abs/2110.13640)
        all_hidden_states = [] if not output_hidden_states is None else None
        for idx, layer in enumerate(self.layers):
            hidden_state = layer(
                hidden_state,
                padding_mask=(padding_mask if incremental_state is None else None),
                attn_mask=attn_mask,
                rel_pos=rel_pos_bias,
                incremental_state=(
                    incremental_state[idx] if incremental_state is not None else None
                ),
            )
            if not output_hidden_states is None:
                all_hidden_states.append(hidden_state)

        if self.layer_norm is not None:
            hidden_state = self.layer_norm(hidden_state)

        return {
            "hidden_state": hidden_state,
            "padding_mask": padding_mask,
            "all_hidden_states": all_hidden_states,
        }


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        depth,
        is_encoder_decoder=False,
    ):
        super().__init__()
        self.config = config

        self.embed_dim = config.embed_dim
        self.dropout_module = torch.nn.Dropout(config.dropout)

        if config.drop_path_rate > 0:
            drop_path_prob = np.linspace(
                0, config.drop_path_rate, config.decoder_layers
            )[depth]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None
        self.attention_type = config.attention_type
        if self.attention_type == "DilatedAttention":
            mha = DilatedAttention
        else:
            mha = MultiheadAttention
        self.self_attn = mha(
            config,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=config.subln,
        )

        self.normalize_before = config.decoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, eps=config.layernorm_eps)

        if not is_encoder_decoder:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = mha(
                config,
                self_attention=False,
                encoder_decoder_attention=True,
                subln=config.subln,
            )
            self.encoder_attn_layer_norm = LayerNorm(
                self.embed_dim, eps=config.layernorm_eps
            )

        self.ffn_dim = config.decoder_ffn_embed_dim
        self.ffn = FeedForwardNetwork(
            self.embed_dim,
            self.ffn_dim,
            config.activation_fn,
            config.dropout,
            config.activation_dropout,
            config.layernorm_eps,
            config.subln,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, eps=config.layernorm_eps)

        if config.deepnorm:
            if is_encoder_decoder:
                self.alpha = math.pow(3.0 * config.decoder_layers, 0.25)
            else:
                self.alpha = math.pow(2.0 * config.decoder_layers, 0.25)
        else:
            self.alpha = 1.0

    def forward(
        self,
        embddings,
        encoder_out=None,
        padding_mask=None,
        incremental_state=None,
        attention_mask=None,
        self_attn_padding_mask=None,
        self_attn_rel_pos=None,
        cross_attn_rel_pos=None,
        is_first_step=False,
    ):
        residual = embddings
        if self.normalize_before:
            embddings = self.self_attn_layer_norm(embddings)

        embddings, attn = self.self_attn(
            query=embddings,
            key=embddings,
            value=embddings,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            attn_mask=attention_mask,
            rel_pos=self_attn_rel_pos,
            is_first_step=is_first_step,
            is_causal=True,
        )
        embddings = self.dropout_module(embddings)

        if self.drop_path is not None:
            embddings = self.drop_path(embddings)

        embddings = residual * self.alpha + embddings
        if not self.normalize_before:
            embddings = self.self_attn_layer_norm(embddings)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = embddings
            if self.normalize_before:
                embddings = self.encoder_attn_layer_norm(embddings)

            embddings, attn = self.encoder_attn(
                query=embddings,
                key=encoder_out,  # TODO: Check this
                value=encoder_out,
                key_padding_mask=padding_mask,
                incremental_state=None,
                rel_pos=cross_attn_rel_pos,
            )
            embddings = self.dropout_module(embddings)

            if self.drop_path is not None:
                embddings = self.drop_path(embddings)

            embddings = residual * self.alpha + embddings
            if not self.normalize_before:
                embddings = self.encoder_attn_layer_norm(embddings)

        residual = embddings
        if self.normalize_before:
            embddings = self.final_layer_norm(embddings)
        embddings = self.ffn(embddings)

        if self.drop_path is not None:
            embddings = self.drop_path(embddings)

        embddings = residual * self.alpha + embddings
        if not self.normalize_before:
            embddings = self.final_layer_norm(embddings)

        return embddings, attn


class Decoder(LongNetPreTrainedModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.config = config
        self.is_encoder_decoder = config.is_encoder_decoder

        self.dropout_module = torch.nn.Dropout(config.dropout)

        embed_dim = config.embed_dim
        self.embed_dim = embed_dim
        self.embed_scale = 1.0 if config.no_scale_embedding else math.sqrt(embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    config,
                    depth,
                    is_encoder_decoder=self.is_encoder_decoder,
                )
                for depth in range(config.decoder_layers)
            ]
        )

        self.num_layers = len(self.layers)

        if config.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim, eps=config.layernorm_eps)
        else:
            self.layer_norm = None

        self.self_attn_relative_position = None
        self.cross_attn_relative_position = None

        if config.rel_pos_buckets > 0 and config.max_rel_pos > 0:
            self.self_attn_relative_position = RelativePositionBias(
                num_buckets=config.rel_pos_buckets,
                max_distance=config.max_rel_pos,
                n_heads=config.num_attention_heads,
            )
            if self.is_encoder_decoder:
                self.cross_attn_relative_position = RelativePositionBias(
                    num_buckets=config.rel_pos_buckets,
                    max_distance=config.max_rel_pos,
                    n_heads=config.num_attention_heads,
                )
        self.post_init()

    def forward(
        self,
        embeddings,
        padding_mask=None,
        encoder_out=None,
        output_hidden_states=False,
        incremental_state=None,
        **kwargs,
    ):

        is_first_step = (
            False
            if incremental_state is None
            else incremental_state.get("is_first_step", False)
        )

        # relative position
        self_attn_rel_pos_bias = None
        slen = embeddings.size(1)
        if self.self_attn_relative_position is not None:
            self_attn_rel_pos_bias = self.self_attn_relative_position(
                batch_size=embeddings.size(0), qlen=slen, klen=slen
            )
            if incremental_state is not None and not is_first_step:
                self_attn_rel_pos_bias = self_attn_rel_pos_bias[-1:, :, :]
        cross_attn_rel_pos_bias = None
        if self.cross_attn_relative_position is not None:
            cross_attn_rel_pos_bias = self.cross_attn_relative_position(
                batch_size=embeddings.size(0),
                qlen=slen,
                klen=encoder_out["hidden_state"].size(1),
            )
            if incremental_state is not None and not is_first_step:
                cross_attn_rel_pos_bias = cross_attn_rel_pos_bias[-1:, :, :]
                
        hidden_state=embeddings
        # decoder layers
        all_hidden_states = [hidden_state] if not output_hidden_states is None else None

        for idx, layer in enumerate(self.layers):
            if incremental_state is None or is_first_step:
                if not self.config.flash_attention:
                    attention_mask = torch.triu(
                        torch.zeros([hidden_state.size(1), hidden_state.size(1)])
                        .float()
                        .fill_(float("-inf"))
                        .type_as(hidden_state),
                        1,
                    )
                else:
                    attention_mask = None
                if is_first_step and incremental_state is not None:
                    if idx not in incremental_state:
                        incremental_state[idx] = {}
            else:
                attention_mask = None
                if idx not in incremental_state:
                    incremental_state[idx] = {}

            hidden_state, _ = layer(
                hidden_state,
                encoder_out["hidden_state"] if encoder_out is not None else None,
                encoder_out["padding_mask"] if encoder_out is not None else None,
                incremental_state[idx] if incremental_state is not None else None,
                attention_mask=attention_mask,
                self_attn_padding_mask=padding_mask,
                self_attn_rel_pos=self_attn_rel_pos_bias,
                cross_attn_rel_pos=cross_attn_rel_pos_bias,
                is_first_step=is_first_step,
            )
            all_hidden_states.append(hidden_state)

        if self.layer_norm is not None:
            hidden_state = self.layer_norm(hidden_state)

        return {"hidden_state": hidden_state, "all_hidden_states": all_hidden_states}


class EncoderDecoder(LongNetPreTrainedModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(
            config=config,
        )
        self.config = config
        self.embedding = LongNetEmbeddings(config=config)
        self.encoder = Encoder(
            config,
            **kwargs,
        )

        self.decoder = Decoder(
            config,
            **kwargs,
        )

    def forward(
        self,
        input_ids,
        prev_output_tokens,
        **kwargs,
    ):
        embeddings = self.embedding(input_ids)
        prev_output_embeddings = self.embedding(prev_output_tokens)
        encoder_out = self.encoder(embeddings)
        decoder_out = self.decoder(
            prev_output_embeddings,
            encoder_out=encoder_out,
        )
        return decoder_out

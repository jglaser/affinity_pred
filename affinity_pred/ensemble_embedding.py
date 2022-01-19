from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput
from transformers.modeling_utils import apply_chunking_to_forward

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from torch.utils import checkpoint

import math

# wrapper class for HAttention1D
class BertHAttention1D(nn.Module):
    def __init__(
        self,
        config,
        local_block_size=16, # for tokens within this distance we always use the full attention
        mask_mode='add',
    ):
        super().__init__()

        attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.num_attention_heads * attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        dtype = self.query.weight.dtype
        eps = 1e-4 if dtype == torch.float16 else 1e-9

        from h_transformer_1d.h_transformer_1d import HAttention1D

        self.attn = HAttention1D(
            dim=config.hidden_size,
            heads=config.num_attention_heads,
            dim_head=attention_head_size,
            block_size=local_block_size,
            eps=eps,
        )

        self.attn.to_qkv = torch.nn.Identity() # passthrough
        self.attn.to_out = torch.nn.Identity()

        if mask_mode != 'add' and mask_mode != 'mul':
            raise ValueError

        self.mask_mode = mask_mode

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.key(encoder_hidden_states)
            value_layer = self.value(encoder_hidden_states)
        else:
            key_layer = self.key(hidden_states)
            value_layer = self.value(hidden_states)

        query_layer = self.query(hidden_states)

        if is_cross_attention:
            attention_mask = encoder_attention_mask

        if attention_mask is not None:
            if self.mask_mode == 'add':
                # make boolean (multiplicative)
                attention_mask = (attention_mask >= 0)
            for dim in range(attention_mask.dim()-1,0,-1):
                attention_mask = attention_mask.squeeze(dim)
            attention_mask = attention_mask.type(torch.bool)

        pad_len_1 = 0
        pad_len_2 = 0

        if is_cross_attention:
            pad_len_1 = max(0, encoder_hidden_states.size()[1]-hidden_states.size()[1])
            pad_len_2 = max(0, hidden_states.size()[1]-encoder_hidden_states.size()[1])

            if pad_len_1 > 0:
                # pad query sequence with zeros, the value doesn't matter because we'll be
                # truncating the output again
                query_layer = F.pad(query_layer, pad=[0,0,0,pad_len_1], value=0.0)
            elif pad_len_2 > 0:
                # pad keys and values
                key_layer = F.pad(key_layer, pad=[0,0,0,pad_len_2], value=0.0)
                value_layer = F.pad(value_layer, pad=[0,0,0,pad_len_2], value=0.0)

                # we must be careful to also pad the mask to make the extra attention
                # matrix values vanish under the softmax
                attention_mask = F.pad(attention_mask, pad=[0,pad_len_2], value=False)

        qkv = torch.stack([query_layer, key_layer, value_layer], dim=2)
        qkv = torch.flatten(qkv, start_dim=2)

        context_layer = self.attn(qkv,
            mask=attention_mask
        )

        if pad_len_1 > 0:
            # truncate queries
            context_layer = context_layer[:,:hidden_states.size()[1]]

        outputs = (context_layer, )

        return outputs

# attention with O(N) memory footprint
class BertLinearMemAttention(nn.Module):
    def __init__(
        self,
        config,
        query_chunk_size=1024,
        key_chunk_size=4096,
    ):
        super().__init__()

        from linear_mem_attention_pytorch.fast_attn import Attention

        attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.num_attention_heads * attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attn = Attention(
            dim=config.hidden_size,
            heads=config.num_attention_heads,
            dim_head=attention_head_size,
            bias=False,
        )

        self.attn.to_q = torch.nn.Identity()
        self.attn.to_kv = torch.nn.Identity()
        self.attn.to_out = torch.nn.Identity()

        self.query_chunk_size = query_chunk_size
        self.key_chunk_size = key_chunk_size

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.key(encoder_hidden_states)
            value_layer = self.value(encoder_hidden_states)
        else:
            key_layer = self.key(hidden_states)
            value_layer = self.value(hidden_states)

        query_layer = self.query(hidden_states)

        if is_cross_attention:
            attention_mask = encoder_attention_mask

        if attention_mask is not None:
            if attention_mask.dtype == torch.float16 or attention_mask.dtype == torch.float32:
                attention_mask = (attention_mask >= 0)
                for dim in range(attention_mask.dim()-1,0,-1):
                    attention_mask = attention_mask.squeeze(dim)
            attention_mask = attention_mask.type(torch.bool)

        kv = torch.stack([key_layer, value_layer], dim=2)
        kv = torch.flatten(kv, start_dim=2)

        context_layer = self.attn(
            x=query_layer,
            context=kv,
            mask=attention_mask,
            query_chunk_size=self.query_chunk_size,
            key_chunk_size=self.key_chunk_size,
        )

        outputs = (context_layer, )

        return outputs

# a linear version of the self attention (without softmax) for residual connections
class BertLinearSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_probs = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # convert to 0/1
            attention_mask = (attention_mask >= 0).int()
            attention_probs = attention_scores * attention_mask

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

# linear residual connection (without LayerNorm)
class BertLinearSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor

class CrossAttentionLayer(nn.Module):
    def __init__(
            self,
            config,
            other_config,
            attn_mode='bert',
            local_block_size=16,
            query_chunk_size=1024,
            key_chunk_size=4096,
            mask_mode='mul',
            inv_fn_encoder=None,
        ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.crossattention = BertAttention(config)

        # linear residual connection
        self.linear_crossattention = BertAttention(config)
        self.linear_crossattention.self = BertLinearSelfAttention(config)
        self.linear_crossattention.output =  BertLinearSelfOutput(config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if attn_mode not in ('bert','hierarchical','linear_mem'):
            raise ValueError

        if attn_mode == 'hierarchical':
            self.crossattention.self = BertHAttention1D(
                config=config,
                mask_mode=mask_mode,
                local_block_size=local_block_size,
            )
        elif attn_mode == 'linear_mem':
            self.crossattention.self = BertLinearMemAttention(
                config=config,
                query_chunk_size=query_chunk_size,
                key_chunk_size=key_chunk_size,
            )

        self.inv_fn_encoder = inv_fn_encoder
        self.mask_mode = mask_mode
        self.attn_mode = attn_mode

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.crossattention.self.key = nn.Linear(other_config.hidden_size, self.crossattention.self.all_head_size)
        self.crossattention.self.value = nn.Linear(other_config.hidden_size, self.crossattention.self.all_head_size)

        self.linear_crossattention.self.key = nn.Linear(other_config.hidden_size, self.crossattention.self.all_head_size)
        self.linear_crossattention.self.value = nn.Linear(other_config.hidden_size, self.crossattention.self.all_head_size)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask,
        output_attentions=False,
    ):

        inv_encoder_attention_mask = self.inv_fn_encoder(encoder_attention_mask)
        if self.mask_mode == 'mul' and self.attn_mode=='bert':
            if encoder_attention_mask is not None:
                if self.inv_fn_encoder is None:
                    raise ValueError("Need encoder inversion function multiplicative -> additive for attention mask")

                # invert attention mask
                encoder_attention_mask = inv_encoder_attention_mask

        cross_attention_outputs = self.crossattention(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        residual = self.linear_crossattention(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=inv_encoder_attention_mask,
            output_attentions=output_attentions,
        )
        residual_output = residual[0]
        outputs += residual[1:]

        attention_output = self.dense(attention_output)
        attention_output = self.layer_norm(attention_output + residual_output)

        # add cross-attn cache to positions 3,4 of present_key_value tuple
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class EnsembleEmbedding(torch.nn.Module):

    supports_gradient_checkpointing = True

    def __init__(
            self,
            seq_model_name,
            smiles_model_name,
            n_cross_attention_layers=3,
            attn_mode='bert',
            local_block_size=512,
            query_chunk_size_seq=2048,
            key_chunk_size_seq=2048,
            query_chunk_size_smiles=512,
            key_chunk_size_smiles=512,
        ):
        super().__init__()

        self.gradient_checkpointing = False

        self.seq_model = BertModel.from_pretrained(seq_model_name)

        self.smiles_model = BertModel.from_pretrained(smiles_model_name)

        smiles_config = self.smiles_model.config

        self.aggregate_hidden_size =self.seq_model.config.hidden_size+smiles_config.hidden_size

        if attn_mode != 'bert':

            # swap the self-attention layers
            seq_layers = self.seq_model.encoder.layer

            for layer in seq_layers:
                if attn_mode == 'hierarchical':
                    attention = BertHAttention1D(
                        config=self.seq_model.config,
                        mask_mode='add',
                        local_block_size=local_block_size,
                    )
                elif attn_mode == 'linear_mem':
                    attention = BertLinearMemAttention(
                        config=self.seq_model.config,
                        query_chunk_size=query_chunk_size_seq,
                        key_chunk_size=key_chunk_size_seq,
                    )

                attention.query = layer.attention.self.query
                attention.key = layer.attention.self.key
                attention.value = layer.attention.self.value

                layer.attention.self = attention

            smiles_layers = self.smiles_model.encoder.layer
            for layer in smiles_layers:
                if attn_mode == 'hierarchical':
                    attention = BertHAttention1D(
                        config=smiles_config,
                        mask_mode='add',
                        local_block_size=local_block_size,
                    )
                elif attn_mode == 'linear_mem':
                    attention = BertLinearMemAttention(
                        config=smiles_config,
                        query_chunk_size=query_chunk_size_smiles,
                        key_chunk_size=key_chunk_size_smiles,
                    )

                attention.query = layer.attention.self.query
                attention.key = layer.attention.self.key
                attention.value = layer.attention.self.value

                layer.attention.self = attention

        # Cross-attention layers
        self.n_cross_attention_layers = n_cross_attention_layers

        self.cross_attention_seq = nn.ModuleList([CrossAttentionLayer(
                config=self.seq_model.config,
                other_config=smiles_config,
                attn_mode=attn_mode,
                local_block_size=local_block_size,
                query_chunk_size=query_chunk_size_seq,
                key_chunk_size=key_chunk_size_smiles,
                inv_fn_encoder=self.smiles_model.invert_attention_mask,
            ) for _ in range(n_cross_attention_layers)])
        self.cross_attention_smiles = nn.ModuleList([CrossAttentionLayer(
                config=smiles_config,
                other_config=self.seq_model.config,
                attn_mode=attn_mode,
                local_block_size=local_block_size,
                query_chunk_size=query_chunk_size_smiles,
                key_chunk_size=key_chunk_size_seq,
                inv_fn_encoder=self.seq_model.invert_attention_mask,
            ) for _ in range(n_cross_attention_layers)])

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        self.seq_model.gradient_checkpointing_enable()
        self.smiles_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.seq_model.gradient_checkpointing_disable()
        self.smiles_model.gradient_checkpointing_disable()

    def forward(
            self,
            input_ids_1=None,
            attention_mask_1=None,
            input_ids_2=None,
            attention_mask_2=None,
            output_attentions=False,
    ):
        outputs = []

        # embed amino acids, sharing the same model
        encoder_outputs = self.seq_model(
            input_ids=input_ids_1,
            attention_mask=attention_mask_1,
        )
        hidden_seq = encoder_outputs.last_hidden_state

        # encode SMILES
        encoder_outputs = self.smiles_model(
            input_ids=input_ids_2,
            attention_mask=attention_mask_2,
        )
        hidden_smiles = encoder_outputs.last_hidden_state

        def cross(attn_1, attn_2, hidden_1, hidden_2, attention_mask_1, attention_mask_2):
            attention_output_1 = attn_1(
                hidden_1,
                hidden_2,
                attention_mask_2,
            )
            attention_output_2 = attn_2(
                hidden_2,
                hidden_1,
                attention_mask_1,
            )
            return attention_output_1[0], attention_output_2[0]

        for i in range(self.n_cross_attention_layers):
            if self.gradient_checkpointing:
                hidden_seq, hidden_smiles = checkpoint.checkpoint(
                    cross,
                    self.cross_attention_seq[i],
                    self.cross_attention_smiles[i],
                    hidden_seq,
                    hidden_smiles,
                    attention_mask_1,
                    attention_mask_2,
                )
            else:
                hidden_seq, hidden_smiles = cross(
                    self.cross_attention_seq[i],
                    self.cross_attention_smiles[i],
                    hidden_seq,
                    hidden_smiles,
                    attention_mask_1,
                    attention_mask_2,
                )

        # mean pooling over sequence length
        cls_seq = self.seq_model.pooler(hidden_seq)
        cls_smiles = self.smiles_model.pooler(hidden_smiles)
        last_hidden_states = torch.cat([cls_seq, cls_smiles], dim=1)

        return last_hidden_states

class ProteinLigandAffinity(EnsembleEmbedding):
    def __init__(self,
            seq_model_name,
            smiles_model_name,
            **kwargs
            ):
        super().__init__(
            seq_model_name,
            smiles_model_name,
            **kwargs)

        self.linear = torch.nn.Linear(self.aggregate_hidden_size, 1)

    def forward(
            self,
            input_ids_1=None,
            attention_mask_1=None,
            input_ids_2=None,
            attention_mask_2=None,
            labels=None,
            output_attentions=False,
    ):
        embedding = super().forward(
            input_ids_1=input_ids_1,
            attention_mask_1=attention_mask_1,
            input_ids_2=input_ids_2,
            attention_mask_2=attention_mask_2,
            output_attentions=output_attentions
        )

        logits = self.linear(embedding)

        # convert to float32 at the end to work around bug with MPI backend
        logits = logits.type(torch.float32)

        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits.view(-1, 1), labels.view(-1,1).type(logits.dtype))
            return (loss, logits)
        else:
            return logits

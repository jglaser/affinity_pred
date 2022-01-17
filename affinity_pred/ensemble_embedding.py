from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput
from transformers.modeling_utils import apply_chunking_to_forward

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from torch.utils import checkpoint

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
class BertLinearAttention(nn.Module):
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
            inv_fn=None,
            inv_fn_encoder=None,
        ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.crossattention = BertAttention(config)

        if attn_mode not in ('bert','hierarchical','linear'):
            raise ValueError

        if attn_mode == 'hierarchical':
            self.crossattention.self = BertHAttention1D(
                config=config,
                mask_mode=mask_mode,
                local_block_size=local_block_size,
            )
        elif attn_mode == 'linear':
            self.crossattention.self = BertLinearAttention(
                config=config,
                query_chunk_size=query_chunk_size,
                key_chunk_size=key_chunk_size,
            )

        self.inv_fn = inv_fn
        self.inv_fn_encoder = inv_fn_encoder
        self.mask_mode = mask_mode
        self.attn_mode = attn_mode

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.crossattention.self.key = nn.Linear(other_config.hidden_size, self.crossattention.self.all_head_size)
        self.crossattention.self.value = nn.Linear(other_config.hidden_size, self.crossattention.self.all_head_size)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask,
        output_attentions=False,
    ):

        if self.mask_mode == 'mul' and self.attn_mode=='bert':
            if encoder_attention_mask is not None:
                if self.inv_fn_encoder is None:
                    raise ValueError("Need encoder inversion function multiplicative -> additive for attention mask")

                # invert attention mask
                encoder_attention_mask = self.inv_fn_encoder(encoder_attention_mask)

        cross_attention_outputs = self.crossattention(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = cross_attention_outputs[1:]  # add cross attentions if we output attention weights

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
            query_chunk_size_seq=256,
            query_chunk_size_smiles=512,
            key_chunk_size_seq=256,
            key_chunk_size_smiles=512,
        ):
        super().__init__()

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
                elif attn_mode == 'linear':
                    attention = BertLinearAttention(
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
                elif attn_mode == 'linear':
                    attention = BertLinearAttention(
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
                inv_fn=self.seq_model.invert_attention_mask,
                inv_fn_encoder=self.smiles_model.invert_attention_mask,
            ) for _ in range(n_cross_attention_layers)])
        self.cross_attention_smiles = nn.ModuleList([CrossAttentionLayer(
                config=smiles_config,
                other_config=self.seq_model.config,
                attn_mode=attn_mode,
                local_block_size=local_block_size,
                query_chunk_size=query_chunk_size_smiles,
                key_chunk_size=key_chunk_size_seq,
                inv_fn=self.smiles_model.invert_attention_mask,
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

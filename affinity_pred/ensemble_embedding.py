from transformers import BertModel, T5EncoderModel
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput
from transformers.modeling_utils import apply_chunking_to_forward

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from torch.utils import checkpoint

import copy

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
            seq_model_type='bert',
            n_layers=3,
            attn_mode='bert',
            local_block_size=512,
            query_chunk_size=2048,
            key_chunk_size=512,
            pooler_dropout_prob=0,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        ):
        super().__init__()

        self.gradient_checkpointing = False

        if not seq_model_type in ('bert', 't5'):
            raise ValueError("Unsupported sequence model type")

        self.seq_model_type = seq_model_type
        if seq_model_type == 't5':
            self.seq_model = T5EncoderModel.from_pretrained(
                seq_model_name,
                is_encoder_decoder=False,
            )

            # translate parameters to BERT style cross attention
            self.seq_model.config.attention_probs_dropout_prob = 0
            self.seq_model.config.hidden_dropout_prob = 0
            self.seq_model.config.layer_norm_eps = 1e-4
            self.seq_model.config.intermediate_size = self.seq_model.config.d_ff
            self.seq_model.config.hidden_act = self.seq_model.config.feed_forward_proj
            self.seq_model.config.num_attention_heads = self.seq_model.config.d_kv
            self.seq_model.config.hidden_size = self.seq_model.config.d_model
            self.seq_model.config.max_position_embeddings = 10000
            self.seq_model.config.type_vocab_size = 1
            self.seq_model.config.initializer_range = 0.02
        else:
            self.seq_model = BertModel.from_pretrained(
                seq_model_name,
                add_pooling_layer=False,
                )

            self.seq_model.config.hidden_dropout_prob = hidden_dropout_prob
            self.seq_model.config.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.smiles_model = BertModel.from_pretrained(
            smiles_model_name,
            add_pooling_layer=False
        )
        self.smiles_model.config.hidden_dropout_prob = hidden_dropout_prob
        self.smiles_model.config.attention_probs_dropout_prob = attention_probs_dropout_prob

        smiles_config = self.smiles_model.config

        if attn_mode != 'bert':
            if seq_model_type == 'bert':
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
                            query_chunk_size=query_chunk_size,
                            key_chunk_size=key_chunk_size,
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
                        query_chunk_size=query_chunk_size,
                        key_chunk_size=key_chunk_size,
                    )

                attention.query = layer.attention.self.query
                attention.key = layer.attention.self.key
                attention.value = layer.attention.self.value

                layer.attention.self = attention

        # use the configuration of the model with the larger hidden dimensions
        self.hidden_size = self.seq_model.config.hidden_size
        config = copy.deepcopy(self.seq_model.config)
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        config.num_hidden_layers = n_layers

        self.bert = BertModel(config, add_pooling_layer = True)

        if attn_mode != 'bert':
            # swap the self-attention layers
            layers = self.bert.encoder.layer

            for layer in layers:
                if attn_mode == 'hierarchical':
                    attention = BertHAttention1D(
                        config=config,
                        mask_mode='add',
                        local_block_size=local_block_size,
                    )
                elif attn_mode == 'linear':
                    attention = BertLinearAttention(
                        config=config,
                        query_chunk_size=query_chunk_size,
                        key_chunk_size=key_chunk_size,
                    )

                attention.query = layer.attention.self.query
                attention.key = layer.attention.self.key
                attention.value = layer.attention.self.value

                layer.attention.self = attention

        # don't need those since we're passing embededings directly into the model
        self.bert.embeddings.word_embeddings = None

        # linear transformations to embed the individual hidden vector spaces into the common one
        self.linear_smiles = torch.nn.Linear(
            self.smiles_model.config.hidden_size,
            self.hidden_size,
        )

        self.linear_seq = torch.nn.Linear(
            self.seq_model.config.hidden_size,
            self.hidden_size,
        )

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        self.seq_model.gradient_checkpointing_enable()
        self.smiles_model.gradient_checkpointing_enable()
        self.bert.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.seq_model.gradient_checkpointing_disable()
        self.smiles_model.gradient_checkpointing_disable()
        self.bert.gradient_checkpointing_disable()

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

        # embed individual outputs
        hidden_seq = self.linear_seq(hidden_seq)
        hidden_smiles = self.linear_smiles(hidden_smiles)

        # concatenate the inputs

        # put the ligand (BERT) model first, because it has a [CLS] token
        hidden_states = torch.cat([hidden_smiles, hidden_seq], dim=1)
        attention_mask = torch.cat([attention_mask_2, attention_mask_1], dim=1)

        bert_output = self.bert(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask
        )

        return bert_output.pooler_output

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

        self.linear = torch.nn.Linear(self.hidden_size, 1)

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

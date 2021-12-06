from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput
from transformers.modeling_utils import apply_chunking_to_forward

from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled

from h_transformer_1d import HAttention1D

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

# wrapper class for HAttention1D
class BertHAttention1D(nn.Module):
    def __init__(
        self,
        config,
        local_block_size=16, # for tokens within this distance we always use the full attention
    ):
        super().__init__()

        attention_head_size = config.hidden_size // config.num_attention_heads
        self.attn = HAttention1D(
            dim=config.hidden_size,
            heads=config.num_attention_heads,
            dim_head=attention_head_size,
            block_size=local_block_size,
        )

        self.all_head_size = config.num_attention_heads * attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attn.to_qkv = torch.nn.Identity() # passthrough
        self.attn.to_out = torch.nn.Identity()

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

        qkv = torch.stack([query_layer, key_layer, value_layer], dim=2)
        qkv = torch.flatten(qkv, start_dim=2)

        if attention_mask is not None:
            attention_mask = attention_mask.type(torch.bool)

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.type(torch.bool)

        context_layer = self.attn(qkv,
            mask=attention_mask,
            key_value_mask = encoder_attention_mask
        )

        outputs = (context_layer, )

        return outputs

class CrossAttentionLayer(nn.Module):
    def __init__(self,
            config,
            other_config,
            use_hierarchical_attention=False):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.crossattention = BertAttention(config)

        if use_hierarchical_attention:
            self.crossattention.self = BertHAttention1D(config)

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.crossattention.self.key = nn.Linear(other_config.hidden_size, self.crossattention.self.all_head_size)
        self.crossattention.self.value = nn.Linear(other_config.hidden_size, self.crossattention.self.all_head_size)

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
        cross_attention_outputs = self.crossattention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            past_key_value=None
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
    def __init__(self, seq_model_name, smiles_model_name, max_seq_length,
                 n_cross_attention_layers=3):
        super().__init__()

        # enable gradient checkpointing
        seq_config = BertConfig.from_pretrained(seq_model_name)
        seq_config.gradient_checkpointing=True
        self.seq_model = BertModel.from_pretrained(seq_model_name,config=seq_config)

        smiles_config = BertConfig.from_pretrained(smiles_model_name)
        smiles_config.gradient_checkpointing=True
        self.smiles_model = BertModel.from_pretrained(smiles_model_name,config=smiles_config)

        self.aggregate_hidden_size = seq_config.hidden_size+smiles_config.hidden_size

        self.max_seq_length = max_seq_length

        # for deepspeed stage 3 (to estimate buffer sizes)
        self.config = BertConfig(hidden_size = self.seq_model.config.hidden_size + self.smiles_model.config.hidden_size)

        # upgrade the self-attention layers to hierarchical ones
        layers = self.seq_model.encoder.layer

        for layer in layers:
            h_attention = BertHAttention1D(config=seq_config)
            h_attention.query = layer.attention.self.query
            h_attention.key = layer.attention.self.key
            h_attention.value = layer.attention.self.value

            layer.attention.self = h_attention

        self.pad_token_id_seq = seq_config.pad_token_id if hasattr(
            seq_config, 'pad_token_id') and seq_config.pad_token_id is not None else 0

        self.pad_token_id_smiles = smiles_config.pad_token_id if hasattr(
            smiles_config, 'pad_token_id') and smiles_config.pad_token_id is not None else 0

        # Cross-attention layers
        self.n_cross_attention_layers = n_cross_attention_layers

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.Init(config=deepspeed_config()):
                self.cross_attention_seq = nn.ModuleList([CrossAttentionLayer(config=seq_config,
                    other_config=smiles_config,
                    use_hierarchical_attention=True) for _ in range(n_cross_attention_layers)])
                self.cross_attention_smiles = nn.ModuleList([CrossAttentionLayer(config=smiles_config,
                    other_config=seq_config,
                    use_hierarchical_attention=False) for _ in range(n_cross_attention_layers)])
        else:
            self.cross_attention_seq = nn.ModuleList([CrossAttentionLayer(config=seq_config,
                other_config=smiles_config,
                use_hierarchical_attention=True) for _ in range(n_cross_attention_layers)])
            self.cross_attention_smiles = nn.ModuleList([CrossAttentionLayer(config=smiles_config,
                other_config=seq_config,
                use_hierarchical_attention=False) for _ in range(n_cross_attention_layers)])

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=False,
    ):
        outputs = []
        input_ids_1 = input_ids[:,:self.max_seq_length]
        attention_mask_1 = attention_mask[:,:self.max_seq_length]

        input_shape = input_ids_1.size()
        device = input_ids_1.device

        embedding_output = self.seq_model.embeddings(
                    input_ids=input_ids_1
                )

        encoder_outputs = self.seq_model.encoder(
            embedding_output,
            attention_mask=attention_mask_1, # 0/1 with HAttention
            head_mask=head_mask
            )
        sequence_output = encoder_outputs[0]

        # smiles model with full attention
        input_ids_2 = input_ids[:,self.max_seq_length:]
        input_shape = input_ids_2.size()
        attention_mask_2 = attention_mask[:,self.max_seq_length:]

        encoder_outputs = self.smiles_model(input_ids=input_ids_2,
                                         attention_mask=attention_mask_2,
                                         return_dict=False)
        smiles_output = encoder_outputs[0]

        hidden_seq = sequence_output
        hidden_smiles = smiles_output

        padded_attention_mask_2 = attention_mask_2
        pad_len = attention_mask_1.size()[1]-attention_mask_2.size()[1]
        assert pad_len >= 0
        padded_attention_mask_2 = F.pad(padded_attention_mask_2, [0, pad_len], value=0)

        # pad the hidden layers with the pad token to make the cross-attention matrix square
        pad_len = hidden_seq.size()[1]-hidden_smiles.size()[1]
        batch_size = hidden_smiles.shape[0]
        pad_input_ids = hidden_smiles.new_full(
            (batch_size, pad_len),
            self.pad_token_id_smiles,
            dtype=torch.long)
        pad_token_type_ids = hidden_smiles.new_full(
            (batch_size, pad_len),
            0, # token type 0
            dtype=torch.long)
        position_ids = self.smiles_model.embeddings.position_ids
        assert position_ids.shape[1] <= pad_len
        pad_len_position_ids = pad_len - position_ids.shape[1]
        position_ids = F.pad(position_ids, (0, pad_len_position_ids), value=self.pad_token_id_smiles)
        pad_inputs_embeds = self.smiles_model.embeddings(
            input_ids=pad_input_ids,
            token_type_ids=pad_token_type_ids,
            position_ids=position_ids,
        )

        for i in range(self.n_cross_attention_layers):
            padded_smiles = torch.cat([hidden_smiles, pad_inputs_embeds], dim=-2)

            attention_output_1 = self.cross_attention_seq[i](
                hidden_states=hidden_seq,
                attention_mask=attention_mask_1,
                encoder_hidden_states=padded_smiles,
                encoder_attention_mask=padded_attention_mask_2,
                output_attentions=output_attentions)

            attention_output_2 = self.cross_attention_smiles[i](
                hidden_states=hidden_smiles,
                attention_mask=attention_mask_2,
                encoder_hidden_states=hidden_seq,
                encoder_attention_mask=attention_mask_1,
                output_attentions=output_attentions)

            hidden_seq = attention_output_1[0]
            hidden_smiles = attention_output_2[0]

        mean_seq = torch.mean(hidden_seq,axis=1)
        mean_smiles = torch.mean(hidden_smiles,axis=1)
        last_hidden_states = torch.cat([mean_seq, mean_smiles], dim=1)

        if output_attentions:
            attentions_seq = attention_output_1[1]
            attentions_smiles = attention_output_2[1]

        return last_hidden_states

class ProteinLigandAffinity(torch.nn.Module):
    def __init__(self, seq_model_name, smiles_model_name, max_seq_length,
                 n_cross_attention_layers):
        super().__init__()

        self.embedding = EnsembleEmbedding(seq_model_name,
            smiles_model_name,
            max_seq_length,
            n_cross_attention_layers
        )

        if is_deepspeed_zero3_enabled():
            import deepspeed
            with deepspeed.zero.Init(config=deepspeed_config()):
                self.linear = torch.nn.Linear(self.embedding.aggregate_hidden_size, 1)
        else:
            self.linear = torch.nn.Linear(self.embedding.aggregate_hidden_size, 1)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=False,
    ):
        embedding = self.embedding(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions
        )

        logits = self.linear(embedding)

        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits.view(-1, 1), labels.view(-1,1).half())
            return (loss, logits)
        else:
            return logits

from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput
from transformers.modeling_utils import apply_chunking_to_forward

from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

class CrossAttentionLayer(nn.Module):
    def __init__(self, config, other_config,
                 ds_sparsity_config=None,
                 max_seq_length=None):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.crossattention = BertAttention(config)

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.crossattention.self.key = nn.Linear(other_config.hidden_size, self.crossattention.self.all_head_size)
        self.crossattention.self.value = nn.Linear(other_config.hidden_size, self.crossattention.self.all_head_size)

        self.sparsity_config = ds_sparsity_config

        if self.sparsity_config is not None:
            # replace the self attention layer
            from sparse_self_attention import BertSparseSelfAttention

            assert max_seq_length > 0

            deepspeed_sparse_self_attn = BertSparseSelfAttention(
                config=config,
                sparsity_config=self.sparsity_config,
                max_seq_length=max_seq_length)
            deepspeed_sparse_self_attn.query = self.crossattention.self.query
            deepspeed_sparse_self_attn.key = self.crossattention.self.key
            deepspeed_sparse_self_attn.value = self.crossattention.self.value

            self.crossattention.self = deepspeed_sparse_self_attn

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

class EnsembleSequenceRegressor(torch.nn.Module):
    def __init__(self, seq_model_name, smiles_model_name, max_seq_length,
                 sparse_attention=False,
                 n_cross_attention_layers=3):
        super().__init__()

        # enable gradient checkpointing
        seq_config = BertConfig.from_pretrained(seq_model_name)
        seq_config.gradient_checkpointing=True
        self.seq_model = BertModel.from_pretrained(seq_model_name,config=seq_config)

        if sparse_attention:
            # replicate the position embeddings of the pre-trained model to the
            # new desired maximum sequence length
            from deepspeed.ops.sparse_attention import SparseAttentionUtils
            self.sparse_attention_utils = SparseAttentionUtils
            class WrapModel(object):
                def __init__(self, bert_model):
                    self.bert = bert_model

            try:
                self.sparse_attention_utils.extend_position_embedding(
                    WrapModel(self.seq_model),
                    max_seq_length
                )
            except:
                # already enough embeddings
                pass

        smiles_config = BertConfig.from_pretrained(smiles_model_name)
        smiles_config.gradient_checkpointing=True
        self.smiles_model = BertModel.from_pretrained(smiles_model_name,config=smiles_config)

        self.max_seq_length = max_seq_length

        # for deepspeed stage 3 (to estimate buffer sizes)
        self.config = BertConfig(hidden_size = self.seq_model.config.hidden_size + self.smiles_model.config.hidden_size)

        self.sparsity_config = None
        if sparse_attention:
            try:
                from deepspeed.ops.sparse_attention import FixedSparsityConfig as STConfig
                self.sparsity_config = STConfig(num_heads=self.seq_model.config.num_attention_heads)
            except:
                pass

        if self.sparsity_config is not None:
            config = seq_config
            sparsity_config = self.sparsity_config
            layers = self.seq_model.encoder.layer

            from sparse_self_attention import BertSparseSelfAttention

            for layer in layers:
                deepspeed_sparse_self_attn = BertSparseSelfAttention(
                    config=config,
                    sparsity_config=sparsity_config,
                    max_seq_length=self.max_seq_length)
                deepspeed_sparse_self_attn.query = layer.attention.self.query
                deepspeed_sparse_self_attn.key = layer.attention.self.key
                deepspeed_sparse_self_attn.value = layer.attention.self.value

                layer.attention.self = deepspeed_sparse_self_attn

            self.pad_token_id_seq = seq_config.pad_token_id if hasattr(
                seq_config, 'pad_token_id') and seq_config.pad_token_id is not None else 0

            self.pad_token_id_smiles = smiles_config.pad_token_id if hasattr(
                smiles_config, 'pad_token_id') and smiles_config.pad_token_id is not None else 0

        # Cross-attention layers
        self.n_cross_attention_layers = n_cross_attention_layers

        # use sparse attention only for the sequence part
        if is_deepspeed_zero3_enabled():
            with deepspeed.zero.Init(config=deepspeed_config()):
                self.cross_attention_seq = nn.ModuleList([CrossAttentionLayer(config=seq_config,
                    other_config=smiles_config,
                    ds_sparsity_config=self.sparsity_config,
                    max_seq_length=self.max_seq_length) for _ in range(n_cross_attention_layers)])
        else:
            self.cross_attention_seq = nn.ModuleList([CrossAttentionLayer(config=seq_config,
                other_config=smiles_config,
                ds_sparsity_config=self.sparsity_config,
                max_seq_length=self.max_seq_length) for _ in range(n_cross_attention_layers)])

        if is_deepspeed_zero3_enabled():
            with deepspeed.zero.Init(config=deepspeed_config()):
                self.cross_attention_smiles = nn.ModuleList([CrossAttentionLayer(config=smiles_config,
                    other_config=seq_config,
                    ds_sparsity_config=None) for _ in range(n_cross_attention_layers)])
        else:
            self.cross_attention_smiles = nn.ModuleList([CrossAttentionLayer(config=smiles_config,
                other_config=seq_config,
                ds_sparsity_config=None) for _ in range(n_cross_attention_layers)])

        if is_deepspeed_zero3_enabled():
            with deepspeed.zero.Init(config=deepspeed_config()):
                self.linear = torch.nn.Linear(seq_config.hidden_size+smiles_config.hidden_size, 1)
        else:
            self.linear = torch.nn.Linear(seq_config.hidden_size+smiles_config.hidden_size, 1)

    def pad_to_block_size(self,
                          block_size,
                          input_ids,
                          attention_mask,
                          pad_token_id):
        batch_size, seq_len = input_ids.shape

        pad_len = (block_size - seq_len % block_size) % block_size
        if pad_len > 0:
            if input_ids is not None:
                input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
            # pad attention mask without attention on the padding tokens
            attention_mask = F.pad(attention_mask, (0, pad_len), value=False)

        return pad_len, input_ids, attention_mask

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
        outputs = []
        input_ids_1 = input_ids[:,:self.max_seq_length]
        attention_mask_1 = attention_mask[:,:self.max_seq_length]

        if self.sparsity_config is not None:
            # sequence model with sparse attention
            pad_len_1, input_ids_1, attention_mask_1 = self.pad_to_block_size(
                block_size=self.sparsity_config.block,
                input_ids=input_ids_1,
                attention_mask=attention_mask_1,
                pad_token_id=self.pad_token_id_seq)

        input_shape = input_ids_1.size()
        device = input_ids_1.device
        extended_attention_mask_1: torch.Tensor = self.seq_model.get_extended_attention_mask(attention_mask_1, input_shape, device)

        embedding_output = self.seq_model.embeddings(
                    input_ids=input_ids_1
                )

        encoder_outputs = self.seq_model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask_1,
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

        # 2D cross-attention masks
        padded_attention_mask_2 = attention_mask_2
        if self.sparsity_config is not None:
            # pad smiles to seq len to make the cross attention mask square
            assert attention_mask_2.size()[1] < attention_mask_1.size()[1]
            pad_len = attention_mask_1.size()[1]-attention_mask_2.size()[1]
            padded_attention_mask_2 = F.pad(padded_attention_mask_2, [0, pad_len], value=0)

        # this goes to the dense cross attention layer in any case, so doesn't require padding
        cross_attention_mask_1 = attention_mask_1[:,None,:]*attention_mask_2[:,:,None]

        # convert values to -inf..0
        inv_cross_attention_mask_1 = self.smiles_model.invert_attention_mask(
            cross_attention_mask_1
        )

        cross_attention_mask_2 = padded_attention_mask_2[:,None,:]*attention_mask_1[:,:,None]
        inv_cross_attention_mask_2 = self.seq_model.invert_attention_mask(
            cross_attention_mask_2
        )

        hidden_seq = sequence_output
        hidden_smiles = smiles_output

        if self.sparsity_config is not None:
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
            padded_smiles = hidden_smiles
            if self.sparsity_config is not None:
                padded_smiles = torch.cat([padded_smiles, pad_inputs_embeds], dim=-2)

            attention_output_1 = self.cross_attention_seq[i](
                hidden_states=hidden_seq,
                attention_mask=attention_mask_1,
                encoder_hidden_states=padded_smiles,
                encoder_attention_mask=inv_cross_attention_mask_2,
                output_attentions=output_attentions)

            attention_output_2 = self.cross_attention_smiles[i](
                hidden_states=hidden_smiles,
                attention_mask=attention_mask_2,
                encoder_hidden_states=hidden_seq,
                encoder_attention_mask=inv_cross_attention_mask_1,
                output_attentions=output_attentions)

            hidden_seq = attention_output_1[0]
            hidden_smiles = attention_output_2[0]

        if self.sparsity_config is not None and pad_len_1 > 0:
            hidden_seq =  self.sparse_attention_utils.unpad_sequence_output(
                pad_len_1, hidden_seq)

        mean_seq = torch.mean(hidden_seq,axis=1)
        mean_smiles = torch.mean(hidden_smiles,axis=1)
        last_hidden_states = torch.cat([mean_seq, mean_smiles], dim=1)

        if output_attentions:
            attentions_seq = attention_output_1[1]
            attentions_smiles = attention_output_2[1]

        logits = self.linear(last_hidden_states).squeeze(-1)

        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits.view(-1, 1), labels.view(-1,1).half())
            return (loss, logits)
        else:
            if output_attentions:
                return logits, (attentions_seq, attentions_smiles)
            else:
                return logits

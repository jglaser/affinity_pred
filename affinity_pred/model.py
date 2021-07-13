from transformers import BertModel, BertConfig
from transformers.integrations import deepspeed_config, is_deepspeed_zero3_enabled

import torch
from torch.nn import functional as F

class MLP(torch.nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self,ninput):
        super().__init__()
        nhidden = 1000
        self.layers = torch.nn.Sequential(
               torch.nn.Linear(ninput, nhidden),
               torch.nn.ReLU(),
               torch.nn.Linear(nhidden, nhidden),
               torch.nn.ReLU(),
               torch.nn.Linear(nhidden, nhidden),
               torch.nn.ReLU(),
               torch.nn.Linear(nhidden, 1)
#        torch.nn.Linear(ninput, 1)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

class EnsembleSequenceRegressor(torch.nn.Module):
    def __init__(self, seq_model_name, smiles_model_name, max_seq_length, sparse_attention=True,
                 *args, **kwargs):
        super().__init__()

        # enable gradient checkpointing
        seq_config = BertConfig.from_pretrained(seq_model_name)
        seq_config.gradient_checkpointing=True
        self.seq_model = BertModel.from_pretrained(seq_model_name,config=seq_config)

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
            # replace the self attention layer of the sequence model
            from deepspeed.ops.sparse_attention import SparseAttentionUtils
            self.sparse_attention_utils = SparseAttentionUtils

            config = seq_config
            sparsity_config = self.sparsity_config
            layers = self.seq_model.encoder.layer

            from affinity_pred.sparse_self_attention import BertSparseSelfAttention

            for layer in layers:
                deepspeed_sparse_self_attn = BertSparseSelfAttention(
                    config=config,
                    sparsity_config=sparsity_config,
                    max_seq_length=self.max_seq_length)
                deepspeed_sparse_self_attn.query = layer.attention.self.query
                deepspeed_sparse_self_attn.key = layer.attention.self.key
                deepspeed_sparse_self_attn.value = layer.attention.self.value

                layer.attention.self = deepspeed_sparse_self_attn

            self.pad_token_id = seq_config.pad_token_id if hasattr(
                seq_config, 'pad_token_id') and seq_config.pad_token_id is not None else 0

        if is_deepspeed_zero3_enabled():
            import deepspeed
            with deepspeed.zero.Init(config=deepspeed_config()):
                self.cls = MLP(seq_config.hidden_size+smiles_config.hidden_size)
        else:
            self.cls = MLP(seq_config.hidden_size+smiles_config.hidden_size)

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
    ):
        outputs = []
        input_ids_1 = input_ids[:,:self.max_seq_length]
        attention_mask_1 = attention_mask[:,:self.max_seq_length]

        # sequence model with sparse attention
        input_shape = input_ids_1.size()
        device = input_ids_1.device
        extended_attention_mask: torch.Tensor = self.seq_model.get_extended_attention_mask(attention_mask_1, input_shape, device)
        if self.seq_model.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.seq_model.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.sparsity_config is not None:
            pad_len_1, input_ids_1, attention_mask_1 = self.pad_to_block_size(
                block_size=self.sparsity_config.block,
                input_ids=input_ids_1,
                attention_mask=attention_mask_1,
                pad_token_id=self.pad_token_id)

        embedding_output = self.seq_model.embeddings(
                    input_ids=input_ids_1
                )
        encoder_outputs = self.seq_model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask
            )
        sequence_output = encoder_outputs[0]

        if self.sparsity_config is not None and pad_len_1 > 0:
            sequence_output = self.sparse_attention_utils.unpad_sequence_output(
                pad_len_1, sequence_output)

        pooled_output = self.seq_model.pooler(sequence_output) if self.seq_model.pooler is not None else None
        outputs.append((sequence_output, pooled_output))

        # smiles model with full attention
        input_ids_2 = input_ids[:,self.max_seq_length:]
        attention_mask_2 = attention_mask[:,self.max_seq_length:]
        outputs.append(self.smiles_model(input_ids=input_ids_2,
                                         attention_mask=attention_mask_2,
                                         return_dict=False))

        # output is a tuple (hidden_state, pooled_output)
        last_hidden_states = torch.cat([output[1] for output in outputs], dim=1)

        logits = self.cls(last_hidden_states).squeeze(-1)

        if labels is not None:
            # crossentropyloss: https://pytorch.org/docs/stable/nn.html#crossentropyloss
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits.view(-1, 1), labels.view(-1,1).half())
            return (loss, logits)
        else:
            return logits

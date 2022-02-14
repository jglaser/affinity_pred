from transformers import BertModel, BertConfig
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled

import torch
from torch.nn import functional as F
from torch import Tensor

# from torch 1.9.0
def gaussian_nll_loss(
    input: torch.Tensor,
    target: Tensor,
    var: Tensor,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> Tensor:
    r"""Gaussian negative log likelihood loss.

    See :class:`~torch.nn.GaussianNLLLoss` for details.

    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        var: tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full (bool, optional): include the constant term in the loss calculation. Default: ``False``.
        eps (float, optional): value added to var, for stability. Default: 1e-6.
        reduction (string, optional): specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    # Check var size
    # If var.size == input.size, the case is heteroscedastic and no further checks are needed.
    # Otherwise:
    if var.size() != input.size():

        # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2)
        # -> unsqueeze var so that var.shape = (10, 2, 1)
        # this is done so that broadcasting can happen in the loss calculation
        if input.size()[:-1] == var.size():
            var = torch.unsqueeze(var, -1)

        # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
        # This is also a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
        elif input.size()[:-1] == var.size()[:-1] and var.size(-1) == 1:  # Heteroscedastic case
            pass

        # If none of the above pass, then the size of var is incorrect.
        else:
            raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate the loss
    loss = 0.5 * (torch.log(var) + (input - target)**2 / var)
    if full:
        loss += 0.5 * math.log(2 * math.pi)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

class MLP(torch.nn.Module):
    '''
    Multilayer Perceptron with two outputs
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
#        torch.nn.Linear(ninput, 2)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

class EnsembleSequenceRegressor(torch.nn.Module):
    def __init__(self, seq_model_name, smiles_model_name, max_seq_length, sparse_attention=True, *args, **kwargs):
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
            with deepspeed.zero.Init(config=deepspeed_config()):
                self.mu = MLP(seq_config.hidden_size+smiles_config.hidden_size)
                self.var = MLP(seq_config.hidden_size+smiles_config.hidden_size)
        else:
            self.mu = MLP(seq_config.hidden_size+smiles_config.hidden_size)
            self.var = MLP(seq_config.hidden_size+smiles_config.hidden_size)

    def pad_to_block_size(self,
                          block_size,
                          input_ids,
                          inputs_embeds,
                          attention_mask,
                          pad_token_id):
        batch_size, seq_len = input_ids.shape if input_ids is not None else inputs_embeds.shape[:-1]

        pad_len = (block_size - seq_len % block_size) % block_size
        if pad_len > 0:
            if inputs_embeds is not None:
                pad_input_ids = inputs_embeds.new_full((batch_size,
                                                        pad_len),
                                                       pad_token_id,
                                                       dtype=torch.long)
                pad_inputs_embeds = self.seq_model.embeddings(pad_input_ids)
                inputs_embeds = torch.cat([inputs_embeds, pad_inputs_embeds], dim=-2)
            if input_ids is not None:
                input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
            # pad attention mask without attention on the padding tokens
            attention_mask = F.pad(attention_mask, (0, pad_len), value=False)

        return pad_len, input_ids, inputs_embeds, attention_mask

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds_1=None,
            inputs_embeds_2=None,
            labels=None,
            seq_len=None
    ):
        outputs = []

        if seq_len is None:
            seq_len = self.max_seq_length

        input_ids_1 = None
        if input_ids is not None:
            input_ids_1 = input_ids[:,:seq_len]

        attention_mask_1 = attention_mask[:,:seq_len]

        if self.sparsity_config is not None:
            pad_len_1, input_ids_1, inputs_embeds_1, attention_mask_1 = self.pad_to_block_size(
                block_size=self.sparsity_config.block,
                input_ids=input_ids_1,
                inputs_embeds=inputs_embeds_1,
                attention_mask=attention_mask_1,
                pad_token_id=self.pad_token_id)

        # sequence model with sparse attention
        input_shape = attention_mask_1.size()
        device = attention_mask_1.device
        extended_attention_mask: torch.Tensor = self.seq_model.get_extended_attention_mask(attention_mask_1, input_shape, device)

        if input_ids is not None:
            embedding_output = self.seq_model.embeddings(
                        input_ids=input_ids_1
                    )
        else:
            embedding_output = inputs_embeds_1

        encoder_outputs = self.seq_model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            )
        sequence_output = encoder_outputs[0]

        if self.sparsity_config is not None and pad_len_1 > 0:
            sequence_output = self.sparse_attention_utils.unpad_sequence_output(
                pad_len_1, sequence_output)

        pooled_output = self.seq_model.pooler(sequence_output) if self.seq_model.pooler is not None else None
        outputs.append((sequence_output, pooled_output))

        # smiles model with full attention
        if input_ids is not None:
            input_ids_2 = input_ids[:,seq_len:]
        attention_mask_2 = attention_mask[:,seq_len:]

        if input_ids is not None:
            inputs_embeds_2 = self.smiles_model.embeddings(
                        input_ids=input_ids_2
            )

        outputs.append(self.smiles_model(inputs_embeds=inputs_embeds_2,
                                         attention_mask=attention_mask_2,
                                         return_dict=False))

        # output is a tuple (hidden_state, pooled_output)
        last_hidden_states = torch.cat([output[1] for output in outputs], dim=1)

        mu = self.mu(last_hidden_states)
        var = self.var(last_hidden_states)
        var = torch.nn.Softplus()(var) # make positive

        logits = torch.cat([mu,var],dim=1)

        if labels is not None:
            loss = gaussian_nll_loss(mu, labels.view(-1,1).half(), var)
            return (loss, logits.unsqueee(1))
        else:
            return logits.unsqueeze(1)

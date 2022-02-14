import warnings
from typing import Dict, List, Optional, Tuple, Union
import torch
from path_explain.explainers.embedding_explainer_torch import EmbeddingExplainerTorch

from model import EnsembleSequenceRegressor

import numpy as np

class EnsembleExplainer(object):
    def __init__(
        self,
        model: EnsembleSequenceRegressor,
        seq_tokenizer,
        smiles_tokenizer,
        internal_batch_size=None
    ):
        """
        Args:
            model (EnsembleSequenceRegressor): Pretrained ensemble regressor
        Raises:
            AttributionTypeNotSupportedError:
        """
        self.seq_attributions = None
        self.smiles_attributions = None

        self.model = model

        self.seq_tokenizer = seq_tokenizer
        self.smiles_tokenizer = smiles_tokenizer
        self.internal_batch_size = internal_batch_size

    def _calculate_attributions(  # type: ignore
        self,
        input_ids,
        attention_mask,
        seq_len,
        interaction_index=None,
        use_expectation=False,
        return_attributions=True,
        return_interactions=True,
        **kwargs
    ):

        if seq_len is None:
            seq_len = self.model.max_seq_length

        ref_input_ids = np.array(input_ids.cpu().numpy())

        ref_input_ids[:,0:seq_len] = [
            [self.seq_tokenizer.pad_token_id
                if token != self.seq_tokenizer.cls_token_id and token != self.seq_tokenizer.sep_token_id
                else token for token in seq[:seq_len] 
            ] for seq in input_ids.cpu().numpy()]

        ref_input_ids[:,seq_len:] = [
            [self.smiles_tokenizer.pad_token_id
                if token != self.smiles_tokenizer.cls_token_id and token != self.smiles_tokenizer.sep_token_id
                else token for token in seq[seq_len:] 
            ] for seq in input_ids.cpu().numpy()]

        ref_input_ids = torch.tensor(ref_input_ids)
        ref_input_ids = ref_input_ids.to(input_ids.device)

        def embedding_model(input_ids, **kwargs):
            embeddings_1 = self.model.seq_model.embeddings(input_ids[:,:seq_len])
            embeddings_2 = self.model.smiles_model.embeddings(input_ids[:,seq_len:])

            # pad the smaller embedding
            pad_len = embeddings_1.shape[2] - embeddings_2.shape[2]
            embeddings_2 = torch.nn.functional.pad(embeddings_2, (0, pad_len))

            embeddings = torch.cat([embeddings_1, embeddings_2], 1)
            embeddings = embeddings.detach().clone()
            return embeddings

        def prediction_model(inputs_embeds):
            inputs_embeds_1 = inputs_embeds[:,:seq_len]
            inputs_embeds_2 = inputs_embeds[:,seq_len:,
                :self.model.smiles_model.config.hidden_size]
            return self.model(
                inputs_embeds_1=inputs_embeds_1,
                inputs_embeds_2=inputs_embeds_2,
                attention_mask=attention_mask,
                seq_len=seq_len,
            )

        batch_embedding = embedding_model(input_ids)
        batch_embedding.requires_grad = True
        baseline_embedding = embedding_model(ref_input_ids)

        explainer = EmbeddingExplainerTorch(prediction_model)

        result = ()
        if return_attributions:
            attributions = explainer.attributions(
                        inputs=batch_embedding,
                        baseline=baseline_embedding,
                        use_expectation=use_expectation,
                        batch_size=self.internal_batch_size,
                        **kwargs
            )
            result = (attributions, )

        if return_interactions:
            interactions = explainer.interactions(
                        inputs=batch_embedding,
                        baseline=baseline_embedding,
                        use_expectation=use_expectation,
                        interaction_index=interaction_index,
                        batch_size=self.internal_batch_size,
                        **kwargs
            )
            result += (interactions, )

        if len(result) == 1:
            result = result[0]

        return result

    def __call__(
        self,
        input_ids,
        attention_mask,
        **kwargs
    ):
        """
        Calculates attribution for `input_ids` using the model.
        This explainer also allows for attributions with respect to a particlar embedding type.
        Returns:
            tuple: (List of associated attribution scores for protein sequence, and for SMILES)
        """

        return self._calculate_attributions(
            input_ids,
            attention_mask,
            **kwargs
        )

import warnings
from typing import Dict, List, Optional, Tuple, Union
import torch
from path_explain.explainers.embedding_explainer_torch import EmbeddingExplainerTorch

from ensemble_embedding import ProteinLigandAffinity

import numpy as np

class EnsembleExplainer(object):
    def __init__(
        self,
        model: ProteinLigandAffinity,
    ):
        """
        Args:
            model (ProteinLigandAffinity): Pretrained ensemble regressor
        Raises:
            AttributionTypeNotSupportedError:
        """

        self.model = model

    def __call__(  # type: ignore
        self,
        input_ids_1,
        attention_mask_1,
        input_ids_2,
        attention_mask_2,
        batch_size=1,
        interaction_index=None,
        use_expectation=False,
        return_attributions=True,
        return_interactions=True,
        **kwargs
    ):

        seq_len = input_ids_1.shape[1]

        ref_input_ids_1 = torch.zeros_like(input_ids_1)
        ref_input_ids_2 = torch.zeros_like(input_ids_2)

        def embedding_model(input_ids_1, input_ids_2):
            embeddings_1 = self.model.seq_model.embeddings(input_ids_1)
            embeddings_2 = self.model.smiles_model.embeddings(input_ids_2)

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
                attention_mask_1=attention_mask_1,
                attention_mask_2=attention_mask_2,
            )

        batch_embedding = embedding_model(input_ids_1, input_ids_2)
        batch_embedding.requires_grad = True
        baseline_embedding = embedding_model(ref_input_ids_1, ref_input_ids_2)

        explainer = EmbeddingExplainerTorch(prediction_model)

        result = ()
        if return_attributions:
            attributions = explainer.attributions(
                        inputs=batch_embedding,
                        baseline=baseline_embedding,
                        use_expectation=use_expectation,
                        batch_size=batch_size,
                        **kwargs
            )
            result = (attributions, )

        if return_interactions:
            interactions = explainer.interactions(
                        inputs=batch_embedding,
                        baseline=baseline_embedding,
                        use_expectation=use_expectation,
                        interaction_index=interaction_index,
                        batch_size=batch_size,
                        **kwargs
            )
            result += (interactions, )

        if len(result) == 1:
            result = result[0]

        return result

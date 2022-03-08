from transformers import DataCollatorWithPadding, default_data_collator, PreTrainedTokenizerBase
from transformers.data.data_collator import default_data_collator, DataCollatorMixin, DataCollatorForLanguageModeling

from transformers import Pipeline

from torch.utils.data import DataLoader

from typing import List, Dict, Any

import os
import collections


class EnsembleDataCollatorWithPadding:
    def __init__(self,
                 smiles_tokenizer,
                 seq_tokenizer,
                 smiles_padding=True,
                 smiles_max_length=None,
                 seq_padding=True,
                 seq_max_length=None):

        self.smiles_collator = DataCollatorWithPadding(smiles_tokenizer, smiles_padding, smiles_max_length)
        self.seq_collator = DataCollatorWithPadding(seq_tokenizer, seq_padding, seq_max_length)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # individually collate protein and ligand sequences into batches
        batch_1 = self.seq_collator([{'input_ids': b['input_ids_1'], 'attention_mask': b['attention_mask_1']} for b in features])
        batch_2 = self.smiles_collator([{'input_ids': b['input_ids_2'], 'attention_mask': b['attention_mask_2']} for b in features])

        batch_merged = default_data_collator([{k: v for k,v in f.items()
                                              if k not in ('input_ids_1','attention_mask_1','input_ids_2','attention_mask_2')}
                                            for f in features])
        batch_merged['input_ids_1'] = batch_1['input_ids']
        batch_merged['attention_mask_1'] = batch_1['attention_mask']
        batch_merged['input_ids_2'] = batch_2['input_ids']
        batch_merged['attention_mask_2'] = batch_2['attention_mask']
        return batch_merged

class EnsembleTokenizer:
    def __init__(self,
                 smiles_tokenizer,
                 seq_tokenizer,
    ):
        self.smiles_tokenizer = smiles_tokenizer
        self.seq_tokenizer = seq_tokenizer

    def __call__(self, features, **kwargs):
        item = {}

        is_batched = isinstance(features, (list, tuple))

        seq_args = {}
        smiles_args = {}
        if 'seq_max_length' in kwargs:
            seq_args['max_length'] = kwargs['seq_max_length']
        if 'smiles_max_length' in kwargs:
            smiles_args['max_length'] = kwargs['smiles_max_length']
        if 'seq_truncation' in kwargs:
            seq_args['truncation'] = kwargs['seq_truncation']
        if 'smiles_truncation' in kwargs:
            smiles_args['truncation'] = kwargs['smiles_truncation']

        if is_batched:
            seq_encodings = self.seq_tokenizer([f['seq'] for f in features], **seq_args)
        else:
            seq_encodings = self.seq_tokenizer(features['seq'], **seq_args)

        item['input_ids_1'] = seq_encodings['input_ids']
        item['attention_mask_1'] = seq_encodings['attention_mask']

        if is_batched:
            smiles_encodings = self.smiles_tokenizer([f['smiles_canonical'] for f in features], **smiles_args)
        else:
            smiles_encodings = self.smiles_tokenizer(features['smiles_canonical'], **smiles_args)

        item['input_ids_2'] = smiles_encodings['input_ids']
        item['attention_mask_2'] = smiles_encodings['attention_mask']

        return item

class ProteinLigandScoring(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if 'seq_truncation' in kwargs:
            preprocess_kwargs['seq_truncation'] = kwargs['seq_truncation']

        if 'seq_max_length' in kwargs:
            preprocess_kwargs['seq_max_length'] = kwargs['seq_max_length']

        if 'smiles_truncation' in kwargs:
            preprocess_kwargs['smiles_truncation'] = kwargs['smiles_truncation']

        if 'smiles_max_length' in kwargs:
            preprocess_kwargs['smiles_max_length'] = kwargs['smiles_max_length']

        return preprocess_kwargs, {}, {}

    def __init__(self,
        model,
        seq_tokenizer,
        smiles_tokenizer,
        **kwargs
        ):
        self.seq_tokenizer = seq_tokenizer
        self.smiles_tokenizer = smiles_tokenizer
        self.data_collator = EnsembleDataCollatorWithPadding(self.smiles_tokenizer,
                                                             self.seq_tokenizer)
        super().__init__(model=model,
                         tokenizer=EnsembleTokenizer(self.smiles_tokenizer,
                                                    self.seq_tokenizer),
                         **kwargs)
        
    def preprocess(self, inputs, **kwargs):
        tokenized_input = self.tokenizer(inputs, **kwargs)
        return tokenized_input

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return outputs

    def postprocess(self, model_outputs):
        return model_outputs.numpy()
    

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        from transformers.pipelines.pt_utils import PipelineDataset, PipelineIterator
        if isinstance(inputs, collections.abc.Sized):
            dataset = PipelineDataset(inputs, self.preprocess, preprocess_params)
        else:
            if num_workers > 1:
                logger.warning(
                    "For iterable dataset using num_workers>1 is likely to result"
                    " in errors since everything is iterable, setting `num_workers=1`"
                    " to guarantee correctness."
                )
                num_workers = 1
            dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            logger.info("Disabling tokenizer parallelism, we're using DataLoader multithreading already")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        collate_fn = self.data_collator
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

class ProteinLigandDataCollatorForLanguageModeling:
    def __init__(self,
        tokenizer_seq: PreTrainedTokenizerBase,
        tokenizer_smiles: PreTrainedTokenizerBase,
        mlm: bool = True,
        mlm_probability_seq: float = 0.15,
        mlm_probability_smiles: float = 0.15,
    ):
        self.collator_seq = DataCollatorForLanguageModeling(
            tokenizer=tokenizer_seq,
            mlm=mlm,
            mlm_probability=mlm_probability_seq,
            )

        self.collator_smiles = DataCollatorForLanguageModeling(
            tokenizer=tokenizer_smiles,
            mlm=mlm,
            mlm_probability=mlm_probability_smiles,
            )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # individually collate protein and ligand sequences into batches
        batch_1 = self.collator_seq([{'input_ids': b['input_ids_1'],
            'attention_mask': b['attention_mask_1'],
            'special_tokens_mask': b['special_tokens_mask_1']} for b in features])
        batch_2 = self.collator_smiles([{'input_ids': b['input_ids_2'],
            'attention_mask': b['attention_mask_2'],
            'special_tokens_mask': b['special_tokens_mask_2']} for b in features])

        batch_merged = default_data_collator([{k: v for k,v in f.items()
                                              if k not in ('input_ids_1',
                                                           'attention_mask_1',
                                                           'special_tokens_mask_1',
                                                           'input_ids_2',
                                                           'attention_mask_2',
                                                           'special_tokens_mask_2',
                                              )} for f in features])
        batch_merged['input_ids_1'] = batch_1['input_ids']
        batch_merged['attention_mask_1'] = batch_1['attention_mask']
        batch_merged['labels_1'] = batch_1['labels']
        batch_merged['input_ids_2'] = batch_2['input_ids']
        batch_merged['attention_mask_2'] = batch_2['attention_mask']
        batch_merged['labels_2'] = batch_2['labels']
        return batch_merged



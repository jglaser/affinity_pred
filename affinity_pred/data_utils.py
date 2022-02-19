from transformers import DataCollatorWithPadding, default_data_collator, PreTrainedTokenizerBase

from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineDataset, PipelineIterator

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

        if is_batched:
            seq_encodings = self.seq_tokenizer([f['seq'] for f in features])
        else:
            seq_encodings = self.seq_tokenizer(features['seq'], **kwargs)

        item['input_ids_1'] = seq_encodings['input_ids']
        item['attention_mask_1'] = seq_encodings['attention_mask']

        if is_batched:
            smiles_encodings = self.smiles_tokenizer([f['smiles_canonical'] for f in features], **kwargs)
        else:
            smiles_encodings = self.smiles_tokenizer(features['smiles_canonical'], **kwargs)

        item['input_ids_2'] = smiles_encodings['input_ids']
        item['attention_mask_2'] = smiles_encodings['attention_mask']

        return item

class ProteinLigandScoring(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
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
        
    def preprocess(self, inputs):
        tokenized_input = self.tokenizer(inputs)#, return_tensors=self.framework) 
        return tokenized_input

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return outputs

    def postprocess(self, model_outputs):
        return model_outputs.numpy()
    

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
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

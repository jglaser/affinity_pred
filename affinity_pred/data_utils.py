from transformers import DataCollatorWithPadding, default_data_collator, PreTrainedTokenizerBase
from typing import List, Dict, Any

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

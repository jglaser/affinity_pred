import torch

import transformers
from transformers import AutoModelForSequenceClassification, BertModel, RobertaModel, BertTokenizerFast, RobertaTokenizer
from transformers import PreTrainedModel, BertConfig, RobertaConfig
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator
from transformers.tokenization_utils_base import BatchEncoding
from transformers import EvalPrediction

from transformers import AutoModelForMaskedLM
from transformers import AdamW

from transformers import HfArgumentParser
from dataclasses import dataclass, field

from transformers.integrations import deepspeed_config, is_deepspeed_zero3_enabled, deepspeed_init
import deepspeed

from transformers.trainer_pt_utils import DistributedTensorGatherer, nested_concat

from torch.nn import functional as F
from torch.utils.data import Dataset

import toolz
import time
from functools import partial

from rdkit import Chem

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd

import re
import gc
import os
import json
import pandas as pd
import numpy as np
import requests
from tqdm.auto import tqdm

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DD

seq_model_name = "Rostlab/prot_bert_bfd" # for fine-tuning

# this logic is necessary because online-downloading and caching doesn't seem to work
if os.path.exists('seq_tokenizer'):
    seq_tokenizer = BertTokenizerFast.from_pretrained('seq_tokenizer/', do_lower_case=False)
else:
    seq_tokenizer = BertTokenizerFast.from_pretrained(seq_model_name, do_lower_case=False)
    seq_tokenizer.save_pretrained('seq_tokenizer/')

model_directory = '/home/xvg/maskedevolution/models/bert_large_1B/model'
tokenizer_directory =  '/home/xvg/maskedevolution/models/bert_large_1B/tokenizer'
tokenizer_config = json.load(open(tokenizer_directory+'/config.json','r'))

smiles_tokenizer =  BertTokenizerFast.from_pretrained(tokenizer_directory, **tokenizer_config)
max_smiles_length = min(200,BertConfig.from_pretrained(model_directory).max_position_embeddings)

# Mpro has 306 residues
max_seq_length = min(4096,BertConfig.from_pretrained(seq_model_name).max_position_embeddings)

def expand_seqs(seqs):
    input_fixed = ["".join(seq.split()) for seq in seqs]
    input_fixed = [re.sub(r"[UZOB]", "X", seq) for seq in input_fixed]
    return [list(seq) for seq in input_fixed]

@dataclass
class InferenceArguments:
    checkpoint: str = field(
        default=None
    )

    batch_size: int = field(
        default=1
    )

    in_file: str = field(
        default=None
    )

    out_file: str = field(
        default=None
    )

    seq: str = field(
        default=None
    )

    smiles_column: str = field(
        default='smiles'
    )

    seq_column: str = field(
        default='seq'
    )

#
# parser - used to handle deepspeed case as well
parser = HfArgumentParser([TrainingArguments,InferenceArguments])
training_args, inference_args = parser.parse_args_into_dataclasses()

def encode_canonical(item):
    seq_encodings = seq_tokenizer(expand_seqs(item['seq'])[0],
								 is_split_into_words=True,
								 return_offsets_mapping=False,
								 truncation=True,
								 padding='max_length',
								 add_special_tokens=True,
								 max_length=max_seq_length)

    try:
        item['smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(item['smiles'][0]))]
    except Exception as e:
        print(repr(e))
        pass

    smiles_encodings = smiles_tokenizer(item['smiles'][0],
                                        padding='max_length',
                                        max_length=max_smiles_length,
                                        add_special_tokens=True,
                                        truncation=True)
    item['input_ids'] = torch.cat([torch.tensor(seq_encodings['input_ids']),
                                    torch.tensor(smiles_encodings['input_ids'])])
    item['token_type_ids'] = torch.cat([torch.tensor(seq_encodings['token_type_ids']),
                                    torch.tensor(smiles_encodings['token_type_ids'])])
    item['attention_mask'] = torch.cat([torch.tensor(seq_encodings['attention_mask']),
                                        torch.tensor(smiles_encodings['attention_mask'])])
    item.pop('smiles')
    item.pop('seq')
    return item


class TestDataset(Dataset):
    def __init__(self, df, seq_column='seq', smiles_column='smiles'):
        self.df = df
        self.seq_column = seq_column
        self.smiles_column = smiles_column
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        item = {'smiles': [row[self.smiles_column]], 'seq': [row[self.seq_column]]}
        item = encode_canonical(item)
        
        # get first (single) item
        #item['input_ids'] = item['input_ids'][0]
        #item['token_type_ids'] = item['token_type_ids'][0]
        #item['attention_mask'] = item['attention_mask'][0]
             
        #item['labels'] = float(row.affinity)
        
        return item

    def __len__(self):
        return len(self.df)

def main():
    torch.manual_seed(training_args.seed)

    def transform(seq, smiles_canonical):
        item = {'seq': [seq], 'smiles': [smiles_canonical]}
        return encode_canonical(item)

    # load the model and predict a batch
    def predict(df, return_dict=False):
        from affinity_pred.model import EnsembleSequenceRegressor

        def model_init():
            return EnsembleSequenceRegressor(seq_model_name, model_directory,  max_seq_length=max_seq_length, sparse_attention=True)

        trainer = Trainer(
            model_init=model_init,                # the instantiated <F0><9F><A4><97> Transformers model to be trained
            args=training_args,                   # training arguments, defined above
        )

        if inference_args.seq is not None:
            df['seq'] = inference_args.seq

        dataset = TestDataset(df, smiles_column=inference_args.smiles_column,
								  seq_column=inference_args.seq_column)

        loader = trainer.get_test_dataloader(dataset)

		# activate dropout (training mode)
        model = trainer._wrap_model(trainer.model,training=False)
        model = model.half().to(trainer.args.device)

        checkpoint = torch.load(inference_args.checkpoint,
            map_location=trainer.args.device)

        model.load_state_dict(checkpoint,strict=False)

        model.train()

        def apply_batchnorm(m):
            if re.search('Norm',m.__class__.__name__):
                m.eval()

        # necessary for correctness of minibatch predictions 
        model.apply(apply_batchnorm)

        world_size = max(1, trainer.args.world_size)
        num_examples = trainer.num_examples(loader)
        gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=loader.sampler.batch_size)
        for inputs in tqdm(loader):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(trainer.args.device)
            logits = model(**inputs)
            logits = logits.detach()
            gatherer.add_arrays(trainer._gather_and_numpify(logits, "eval_preds"))

        preds = gatherer.finalize()
        df['affinity'] = pd.Series(data=preds, index=df.index).astype('float32')
        return df

    df = pd.read_csv(inference_args.in_file).dropna().reset_index(drop=True)

    df_pred = predict(df)

    if int(os.environ['RANK']) == 0:
        df_pred.to_parquet(inference_args.out_file)

if __name__ == "__main__":
    main()

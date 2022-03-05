import torch
import logging

import transformers
from transformers import BertModel, BertTokenizer, T5Tokenizer, AutoTokenizer
from transformers import PreTrainedModel, BertConfig
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator
from transformers.tokenization_utils_base import BatchEncoding
from transformers import EvalPrediction

from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.pre_tokenizers import Digits
from tokenizers.pre_tokenizers import Sequence
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.pre_tokenizers import Split

from tokenizers import Regex
from tokenizers import pre_tokenizers
from tokenizers import normalizers
from tokenizers.normalizers import Replace

from dataclasses import dataclass, field
from enum import Enum

from transformers import HfArgumentParser
from transformers.trainer_utils import is_main_process
from transformers.trainer_utils import get_last_checkpoint

from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
import deepspeed

import datasets
from torch.utils.data import Dataset
from torch.nn import functional as F

from ensemble_embedding import ProteinLigandAffinityCosine, ProteinLigandConfigCosine
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error


import os
import json
from tqdm.auto import tqdm

import torch.distributed as dist

from ensemble_embedding import ProteinLigandAffinityMLP
from ensemble_embedding import ProteinLigandConfig

logger = logging.getLogger(__name__)

def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)

def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics
    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))

def expand_seqs(seqs):
    input_fixed = ["".join(seq.split()) for seq in seqs]
    input_fixed = [re.sub(r"[UZOB]", "X", seq) for seq in input_fixed]
    return [list(seq) for seq in input_fixed]

class AffinityDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        #affinity = item['neg_log10_affinity_M']
        item['labels'] = float(item['neg_log10_affinity_M'])

        # drop the non-encoded input
        item.pop('smiles_can')
        item.pop('seq')
        item.pop('neg_log10_affinity_M')
        item.pop('affinity')
        item.pop('affinity_uM')
        item.pop('smiles')
        return item

    def __len__(self):
        return len(self.dataset)

def compute_metrics(p: EvalPrediction):
    preds_list, out_label_list = p.predictions, p.label_ids

    return {
        "mse": mean_squared_error(out_label_list, preds_list),
        "mae": mean_absolute_error(out_label_list, preds_list),
    }

@dataclass
class ModelArguments:
    model_type: str = field(
        default='bert',
        metadata = {'choices': ['bert','regex']},
    )

    seq_model_name: str = field(
        default=None
    )

    seq_model_type: str = field(
        default='bert'
    )

    smiles_model_dir: str = field(
        default=None
    )

    smiles_tokenizer_dir: str = field(
        default=None
    )

    max_seq_length: int = field(
        default=512
    )

    max_smiles_length: int = field(
        default=512
    )

    attn_mode: str = field(
        default='bert'
    )

    local_block_size: int = field(
        default=512,
    )

    attn_query_chunk_size: int = field(
        default=512,
    )

    attn_key_chunk_size: int = field(
        default=512,
    )


@dataclass
class DataArguments:
    dataset: str = field(
        default=None
    )

    split: str = field(
        default='train'
    )


def main():
    # on-the-fly tokenization
    def encode(item):
        seq_encodings = seq_tokenizer(item['seq'][0],
                                 return_offsets_mapping=False,
                                 truncation=True,
                                 padding='max_length',
                                 add_special_tokens=True,
                                 max_length=max_seq_length)

        item['input_ids_1'] = [torch.tensor(seq_encodings['input_ids'])]
        item['attention_mask_1'] = [torch.tensor(seq_encodings['attention_mask'])]

        smiles_encodings = smiles_tokenizer(item['smiles_can'][0],
                                            padding='max_length',
                                            max_length=max_smiles_length,
                                            add_special_tokens=True,
                                            truncation=True)

        item['input_ids_2'] = [torch.tensor(smiles_encodings['input_ids'])]
        item['attention_mask_2'] = [torch.tensor(smiles_encodings['attention_mask'])]

        return item

    use_mpi = False
    if torch.distributed.is_mpi_available() and 'OMPI_COMM_WORLD_RANK' in os.environ:
        torch.distributed.init_process_group(backend='mpi')
        if torch.distributed.get_rank() == 0:
            print('Using MPI backend with torch.distributed')
        use_mpi = True

    parser = HfArgumentParser([TrainingArguments,ModelArguments, DataArguments])

    (training_args, model_args, data_args) = parser.parse_args_into_dataclasses()

    if 'LOCAL_RANK' in os.environ:
        training_args.local_rank = int(os.environ["LOCAL_RANK"])
    elif use_mpi:
        # now set the local task id to 0 to enable DDP
        training_args.local_rank = 0

    # error out when there are unused parameters
    training_args.ddp_find_unused_parameters=False

    smiles_tokenizer_directory = model_args.smiles_tokenizer_dir
    smiles_model_directory = model_args.smiles_model_dir
    tokenizer_config = json.load(open(smiles_tokenizer_directory+'/config.json','r'))

    smiles_tokenizer =  AutoTokenizer.from_pretrained(smiles_tokenizer_directory, **tokenizer_config)

    if model_args.model_type == 'regex':
        smiles_tokenizer.backend_tokenizer.pre_tokenizer = Sequence([WhitespaceSplit(),Split(Regex(r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""), behavior='isolated')])

    normalizer = normalizers.Sequence([Replace(Regex('[UZOB]'),'X'),Replace(Regex('\s'),'')])
    pre_tokenizer = pre_tokenizers.Split(Regex(''),behavior='isolated')
    seq_tokenizer = AutoTokenizer.from_pretrained(model_args.seq_model_name, do_lower_case=False)
    seq_tokenizer.backend_tokenizer.normalizer = normalizer
    seq_tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizer

    max_seq_length = model_args.max_seq_length
    max_smiles_length = min(smiles_tokenizer.model_max_length, model_args.max_smiles_length)

    # seed the weight initialization
    torch.manual_seed(training_args.seed)

    if os.path.exists(data_args.dataset):
        # manually initialize dataset without downloading
        builder = datasets.load_dataset_builder(path=data_args.dataset)
        # Download and prepare data
        builder.download_and_prepare(
            try_from_hf_gcs=False,
        )
        # Build dataset for splits
        keep_in_memory = datasets.info_utils.is_small_dataset(builder.info.dataset_size)
        ds = builder.as_dataset(split=data_args.split, in_memory=keep_in_memory)
    else:
        ds = datasets.load_dataset(data_args.dataset,
            split=data_args.split)

    # keep a small holdout data set
    split_test = ds.train_test_split(train_size=0.99, seed=0)

    # further split the train set
    f = 0.9
    split = split_test['train'].train_test_split(train_size=f, seed=training_args.seed)
    train = split['train']
    validation = split['test']
    train.set_transform(encode)
    validation.set_transform(encode)

    train_dataset = AffinityDataset(train)
    val_dataset = AffinityDataset(validation)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    seq_config = BertConfig.from_pretrained(model_args.seq_model_name)

    smiles_config = BertConfig.from_pretrained(smiles_model_directory)
    smiles_config.hidden_dropout_prob=0
    smiles_config.attention_probs_dropout_prob=0
    seq_config.hidden_dropout_prob = 0
    seq_config.attention_probs_dropout_prob = 0
    config = ProteinLigandConfigCosine(
        seq_config=seq_config,
        smiles_config=smiles_config,
        seq_model_type=model_args.seq_model_type,
        attn_mode=model_args.attn_mode,
        local_block_size=model_args.local_block_size,
        query_chunk_size=model_args.attn_query_chunk_size,
        key_chunk_size=model_args.attn_key_chunk_size,
    )

    model = ProteinLigandAffinityCosine(config)

    model.embedding.load_pretrained(model_args.seq_model_name,
        model_args.smiles_model_dir)

    trainer = Trainer(
        model=model,
        args=training_args,                   # training arguments, defined above
        train_dataset=train_dataset,          # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
    )

    all_metrics = {}
    logger.info("*** Train ***")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(training_args.output_dir)
    metrics = train_result.metrics

    if trainer.is_world_process_zero():
        handle_metrics("train", metrics, training_args.output_dir)
        all_metrics.update(metrics)

        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))

#        # create the repo
#        trainer.push_to_hub()
#        seq_tokenizer.save_pretrained(training_args.hub_model_id+'/seq_tokenizer')
#        smiles_tokenizer.save_pretrained(training_args.hub_model_id+'/smiles_tokenizer')
#
#        model_card_args = {
#            'language': 'protein, SMILES',
#            'tags': 'protein ligand affinity prediction',
#            'finetuned_from': model_args.seq_model_name +',' + model_args.smiles_model_dir,
#            'dataset': data_args.dataset,
#            'dataset_args': data_args.split,
#        }
#        trainer.push_to_hub(**model_card_args)

if __name__ == "__main__":
    main()

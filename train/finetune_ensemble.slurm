#!/bin/bash
#SBATCH -A STF006
#SBATCH -J finetune_regex
#SBATCH -o %x-%j.out
#SBATCH -t 12:00:00
#SBATCH -p batch
#SBATCH -N 16
#SBATCH --array=1
#SBATCH --ntasks-per-node=4

source /gpfs/alpine/world-shared/med106/eju1/spock_build/miniconda/etc/profile.d/conda.sh
conda activate /gpfs/alpine/world-shared/bip214/spock-env

module load rocm/4.3.0

export HF_HOME=/gpfs/alpine/world-shared/bip214/affinity_pred/train
export HF_DATASETS_CACHE=/gpfs/alpine/world-shared/bip214/affinity_pred/train/dataset-cache
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/world-shared/bip214/affinity_pred/train/build_spock

export OMP_NUM_THREADS=32
export PYTHONUNBUFFERED=1

export ENSEMBLE_ID=$SLURM_ARRAY_TASK_ID

NLAYERS=10
LR=1e-4
DATASET=no_kras
MAX_SEQ_LENGTH=4096
BATCH_SIZE=1

srun python ../affinity_pred/finetune.py \
    --deepspeed='ds_config_stage2.json'\
    --model_type='regex' \
    --dataset='/gpfs/alpine/world-shared/bip214/binding_affinity'\
    --split=${DATASET}\
    --n_cross_attention=${NLAYERS}\
    --max_seq_length=${MAX_SEQ_LENGTH}\
    --output_dir=./results_spock_bs${BATCH_SIZE}_regex_${NLAYERS}cross_lr${LR}_${DATASET}_seqlen${MAX_SEQ_LENGTH}_${ENSEMBLE_ID}\
    --num_train_epochs=45\
    --per_device_train_batch_size=${BATCH_SIZE}\
    --per_device_eval_batch_size=${BATCH_SIZE}\
    --learning_rate=${LR}\
    --weight_decay=0.01\
    --warmup_steps=5\
    --logging_dir=./logs_spock_bs${BATCH_SIZE}_regex_${NLAYERS}cross_lr${LR}_${DATASET}_seqlen${MAX_SEQ_LENGTH}_${ENSEMBLE_ID}\
    --logging_steps=1\
    --evaluation_strategy="epoch"\
    --gradient_accumulation_steps=1\
    --fp16=True\
    --run_name="seq_smiles_affinity"\
    --save_strategy="epoch"\
    --seed=$((42+${ENSEMBLE_ID}))

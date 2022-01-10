#!/bin/bash
#SBATCH -A STF006
#SBATCH -J finetune
#SBATCH -o slurm-%j.out
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 24
#SBATCH --array=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
##SBATCH -d afterany:54274

echo $SLURM_JOB_NODELIST

source /gpfs/alpine/world-shared/med106/eju1/spock_build/miniconda/etc/profile.d/conda.sh
conda activate /gpfs/alpine/bip214/world-shared/crusher-env

module load rocm/4.5.0
module load gcc/11.2.0

module load craype-accel-amd-gfx90a
export MPICH_GPU_SUPPORT_ENABLED=1

export HF_HOME=/gpfs/alpine/world-shared/bip214/affinity_pred/train
export HF_DATASETS_CACHE=/gpfs/alpine/world-shared/bip214/affinity_pred/train/dataset-cache
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/world-shared/bip214/affinity_pred/train/build_spock

export OMP_NUM_THREADS=32
export PYTHONUNBUFFERED=1

export ENSEMBLE_ID=$SLURM_ARRAY_TASK_ID

NLAYERS=1
#NLAYERS=5
LR=3e-5
DATASET=no_kras
MAX_SEQ_LENGTH=1024
#MAX_SEQ_LENGTH=2048
ATTN_MODE=linear
#ATTN_MODE=bert
HATTENTION_BLOCKSIZE=512
#HATTENTION_BLOCKSIZE=2048
PER_DEVICE_BATCH_SIZE=16
CLUSTER=crusher
BATCH_SIZE=$((${PER_DEVICE_BATCH_SIZE}*${SLURM_NTASKS}))

if [ "$ATTN_MODE" = "hierarchical" ]; then
    ID_STR=${CLUSTER}_attn_${ATTN_MODE}_${HATTENTION_BLOCKSIZE}_bs${BATCH_SIZE}_regex_${NLAYERS}cross_lr${LR}_${DATASET}_seqlen${MAX_SEQ_LENGTH}_${ENSEMBLE_ID}
else
    ID_STR=${CLUSTER}_attn_${ATTN_MODE}_bs${BATCH_SIZE}_regex_${NLAYERS}cross_lr${LR}_${DATASET}_seqlen${MAX_SEQ_LENGTH}_${ENSEMBLE_ID}
fi

srun -l /bin/hostname | sort -n

export LD_PRELOAD="/opt/cray/pe/gcc/11.2.0/snos/lib64/libstdc++.so.6 /gpfs/alpine/world-shared/bip214/rocm_smi_lib/build/rocm_smi/librocm_smi64.so"
export LD_PRELOAD="${LD_PRELOAD} ${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa.so"

srun python ../affinity_pred/finetune.py \
    --smiles_tokenizer_dir='/gpfs/alpine/world-shared/med106/blnchrd/models/bert_large_plus_clean_regex/tokenizer'\
    --smiles_model_dir='/gpfs/alpine/world-shared/med106/blnchrd/automatedmutations/pretraining/run/job_86neeM/output'\
    --model_type='regex' \
    --seq_model_name='Rostlab/prot_bert_bfd'\
    --dataset='/gpfs/alpine/world-shared/bip214/binding_affinity'\
    --split=${DATASET}\
    --n_cross_attention=${NLAYERS}\
    --attn_mode=${ATTN_MODE}\
    --local_block_size=${HATTENTION_BLOCKSIZE}\
    --max_seq_length=${MAX_SEQ_LENGTH}\
    --seq_chunk_size=512\
    --output_dir=./results_${ID_STR}\
    --num_train_epochs=45\
    --per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE}\
    --per_device_eval_batch_size=${PER_DEVICE_BATCH_SIZE}\
    --learning_rate=${LR}\
    --weight_decay=0.01\
    --warmup_steps=5\
    --logging_dir=./logs_${ID_STR}\
    --logging_steps=1\
    --evaluation_strategy="steps"\
    --eval_steps=500\
    --gradient_accumulation_steps=1\
    --fp16=True\
    --run_name="seq_smiles_affinity"\
    --save_strategy="steps"\
    --save_steps=500\
    --seed=$((42+${ENSEMBLE_ID}))

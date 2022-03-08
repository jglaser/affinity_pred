#!/bin/bash
#SBATCH -A STF006
#SBATCH -J finetune
#SBATCH -o slurm-%j.out
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 120
#SBATCH --array=1
#SBATCH --ntasks-per-node=1

echo $SLURM_JOB_NODELIST

source /gpfs/alpine/world-shared/med106/eju1/spock_build/miniconda/etc/profile.d/conda.sh
conda activate /gpfs/alpine/bip214/world-shared/crusher-env

module load rocm/4.5.0
module load gcc/11.2.0

#module load craype-accel-amd-gfx90a
#export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE

export HF_HOME=/gpfs/alpine/world-shared/bip214/affinity_pred/train
export HF_DATASETS_CACHE=/gpfs/alpine/world-shared/bip214/affinity_pred/train/dataset-cache
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/world-shared/bip214/affinity_pred/train/build_spock
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export OMP_NUM_THREADS=32
export PYTHONUNBUFFERED=1

export ENSEMBLE_ID=$SLURM_ARRAY_TASK_ID

#NLAYERS=5
LR=3e-5
DATASET=no_kras
MAX_SEQ_LENGTH=2048
ATTN_MODE=bert
PER_DEVICE_BATCH_SIZE=32
PER_DEVICE_EVAL_BATCH_SIZE=32 # due to large size of logits
CLUSTER=crusher
BATCH_SIZE=$((${PER_DEVICE_BATCH_SIZE}*${SLURM_NTASKS}*8))

ID_STR=${CLUSTER}_attn_${ATTN_MODE}_bs${BATCH_SIZE}_regex_lr${LR}_${DATASET}_seqlen${MAX_SEQ_LENGTH}_${ENSEMBLE_ID}_mlm

srun -l /bin/hostname | sort -n

export LD_PRELOAD="/opt/cray/pe/gcc/11.2.0/snos/lib64/libstdc++.so.6 /gpfs/alpine/world-shared/bip214/rocm_smi_lib/build/rocm_smi/librocm_smi64.so"
#export LD_PRELOAD="${LD_PRELOAD} ${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa.so"

export NCCL_ASYNC_ERROR_HANDLING=1
MASTER_IP=`ip -f inet addr show hsn0 | sed -En -e 's/.*inet ([0-9.]+).*/\1/p' | head -1`:29400
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
#export NCCL_DEBUG=INFO

#export TORCH_DISTRIBUTED_DEBUG=INFO

srun -l -c 8 torchrun --nproc_per_node 8 --nnodes ${SLURM_NNODES} --rdzv_id=${SLURM_JOB_ID} --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP\
    ../affinity_pred/finetune.py \
    --smiles_tokenizer_dir='/gpfs/alpine/world-shared/med106/blnchrd/models/bert_large_plus_clean_regex/tokenizer'\
    --smiles_model_dir='/gpfs/alpine/world-shared/med106/blnchrd/automatedmutations/pretraining/run/job_86neeM/output'\
    --model_type='regex' \
    --seq_model_name='Rostlab/prot_bert_bfd'\
    --dataset='/gpfs/alpine/world-shared/bip214/binding_affinity'\
    --split=${DATASET}\
    --attn_mode=${ATTN_MODE}\
    --max_seq_length=${MAX_SEQ_LENGTH}\
    --output_dir=./results_${ID_STR}\
    --num_train_epochs=45\
    --per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE}\
    --per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE}\
    --learning_rate=${LR}\
    --weight_decay=0.01\
    --warmup_steps=10\
    --logging_dir=./logs_${ID_STR}\
    --logging_steps=1\
    --evaluation_strategy="steps"\
    --eval_steps=50\
    --gradient_accumulation_steps=1\
    --fp16=False\
    --run_name="seq_smiles_affinity"\
    --save_strategy="steps"\
    --save_steps=50\
    --seed=$((42+${ENSEMBLE_ID})) \
    --gradient_checkpointing \
    --lr_scheduler_type=constant_with_warmup \
    --eval_accumulation_steps=1

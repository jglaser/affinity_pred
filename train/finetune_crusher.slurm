#!/bin/bash
#SBATCH -A STF006
#SBATCH -J finetune_regex
#SBATCH -o slurm-%j.out
#SBATCH -t 2:00:00
#SBATCH -p batch
##SBATCH -N 150
#SBATCH -N 1
#SBATCH --array=1
#SBATCH --ntasks-per-node=8
#SBATCH -x crusher183

echo $SLURM_JOB_NODELIST

source /gpfs/alpine/world-shared/med106/eju1/spock_build/miniconda/etc/profile.d/conda.sh
conda activate /gpfs/alpine/bip214/world-shared/crusher-env

module load rocm/4.5.0
module load gcc/11.2.0

export HF_HOME=/gpfs/alpine/world-shared/bip214/affinity_pred/train
export HF_DATASETS_CACHE=/gpfs/alpine/world-shared/bip214/affinity_pred/train/dataset-cache
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/world-shared/bip214/affinity_pred/train/build_spock

export OMP_NUM_THREADS=32
export PYTHONUNBUFFERED=1

export ENSEMBLE_ID=$SLURM_ARRAY_TASK_ID

NLAYERS=5
LR=5e-4
DATASET=no_kras
MAX_SEQ_LENGTH=2048
HATTENTION=False
PER_DEVICE_BATCH_SIZE=8
CLUSTER=crusher
BATCH_SIZE=$((${PER_DEVICE_BATCH_SIZE}*${SLURM_NTASKS}))

MACHINEFILE="nodes.$SLURM_JOB_ID"
#srun -l /bin/hostname | sort -n | awk '{print $2}' > $MACHINEFILE
srun -l /bin/hostname | sort -n > $MACHINEFILE
cat $MACHINEFILE

export LD_PRELOAD="/opt/cray/pe/gcc/11.2.0/snos/lib64/libstdc++.so.6 /gpfs/alpine/world-shared/bip214/rocm_smi_lib/build/rocm_smi/librocm_smi64.so"

srun -l python ../affinity_pred/finetune.py \
    --deepspeed='ds_config_rocm.json'\
    --smiles_tokenizer_dir='/gpfs/alpine/world-shared/med106/blnchrd/models/bert_large_plus_clean_regex/tokenizer'\
    --smiles_model_dir='/gpfs/alpine/world-shared/med106/blnchrd/automatedmutations/pretraining/run/job_86neeM/output'\
    --model_type='regex' \
    --seq_model_name='Rostlab/prot_bert_bfd'\
    --dataset='/gpfs/alpine/world-shared/bip214/binding_affinity'\
    --split=${DATASET}\
    --n_cross_attention=${NLAYERS}\
    --hierarchical_attention=${HATTENTION}\
    --max_seq_length=${MAX_SEQ_LENGTH}\
    --output_dir=./results_${CLUSTER}_hatt_${HATTENTION}_bs${BATCH_SIZE}_regex_${NLAYERS}cross_lr${LR}_${DATASET}_seqlen${MAX_SEQ_LENGTH}_${ENSEMBLE_ID}\
    --num_train_epochs=45\
    --per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE}\
    --per_device_eval_batch_size=${PER_DEVICE_BATCH_SIZE}\
    --learning_rate=${LR}\
    --weight_decay=0.01\
    --warmup_steps=5\
    --logging_dir=./logs_${CLUSTER}_hatt_${HATTENTION}_bs${BATCH_SIZE}_regex_${NLAYERS}cross_lr${LR}_${DATASET}_seqlen${MAX_SEQ_LENGTH}_${ENSEMBLE_ID}\
    --logging_steps=1\
    --evaluation_strategy="epoch"\
    --gradient_accumulation_steps=1\
    --fp16=True\
    --run_name="seq_smiles_affinity"\
    --save_strategy="epoch"\
    --seed=$((42+${ENSEMBLE_ID}))

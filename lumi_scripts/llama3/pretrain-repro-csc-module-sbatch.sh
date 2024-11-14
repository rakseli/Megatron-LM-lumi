#!/bin/bash

#SBATCH --job-name=test-pretraining-csc-module
#SBATCH --time=01:00:00
#SBATCH --partition=dev-g
#SBATCH --nodes=16
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=480G
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000353
#SBATCH --output=repro-logs/%x_%j.output
#SBATCH --error=repro-logs/%x_%j.error

set -euo pipefail

module purge
module load LUMI/24.03 partition/G
ml use /appl/local/csc/modulefiles/
ml load pytorch/2.4
module load PrgEnv-amd
module load rocm/6.0.3


mkdir -p workdir_csc_module
wd=$(realpath workdir_csc_module)

PP_SIZE=2
TP_SIZE=2

echo "Option -d devel, start from scratch"
SAVE_CHECKPOINT_DIR="$wd/model_dir"
LOAD_CHECKPOINT_DIR=$SAVE_CHECKPOINT_DIR

TENSORBOARD_PATH="$wd/tensorboard/$SLURM_JOB_NAME"
mkdir -p "$SAVE_CHECKPOINT_DIR" "$wd/tensorboard"

LOG_INTERVAL=1
SAVE_INTERVAL=100
EVAL_INTERVAL=10
EVAL_STEPS=10
LOAD_OPTIMIZER_STATES=false


# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
#debug variables
export TORCH_DISTRIBUTED_DEBUG=INFO
#export NCCL_DEBUG=INFO
export RCCL_KERNEL_COLL_TRACE_ENABLE=1
export NCCL_DEBUG_SUBSYS=INIT,COLL
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH

DATA_ROOT="./error-repro"
CACHE_PATH="${DATA_ROOT}/processed/index-cache"
TRAIN_DATA="1.0 ${DATA_ROOT}/processed/enwiki"
TOKENIZER="${DATA_ROOT}/tokenizer"

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=64
NLAYERS=32
NHIDDEN=4096
NHEADS=32
FFN_HIDDEN_SIZE=14336
NUM_QUERY_GROUPS=8
SEQ_LEN=8192

INIT_METHOD_STD=0.00747017

LEARNING_RATE=2e-5
MIN_LEARNING_RATE=1e-8


TOTAL_TOKENS=3_400_000_000
TOTAL_TOKENS=${TOTAL_TOKENS//_}    # drop "_" for bash math
TRAIN_SAMPLES=$((TOTAL_TOKENS/SEQ_LEN))
LR_DECAY_SAMPLES=$TRAIN_SAMPLES
LR_WARMUP_SAMPLES=$((GLOBAL_BATCH_SIZE*1072))

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --lr $LEARNING_RATE \
    --min-lr $MIN_LEARNING_RATE \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --use-distributed-optimizer \
    "

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LEN \
    --data-cache-path $CACHE_PATH \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type HFPretrainedTokenizer \
    --hf_tokenizer_path $TOKENIZER \
    --bf16 \
    --disable-bias-linear \
    --init-method-std $INIT_METHOD_STD \
    --normalization RMSNorm \
    --norm-epsilon 1e-05 \
    --seed 42 \
    --no-bias-dropout-fusion \
    --untie-embeddings-and-output-weights \
    --no-bias-dropout-fusion \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups $NUM_QUERY_GROUPS \
    --no-query-key-layer-scaling \
    --use-flash-attn \
    --swiglu \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --position-embedding-type rope \
    --use-rope-scaling \
    --rope-theta 500000.0 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --recompute-activations \
    --make-vocab-size-divisible-by 1 \
    --distributed-timeout-minutes 180 \
    $OPTIMIZER_ARGS \
    "


OUTPUT_ARGS=" \
    --log-interval $LOG_INTERVAL \
    --save $SAVE_CHECKPOINT_DIR \
    --load $LOAD_CHECKPOINT_DIR \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters $EVAL_STEPS \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

PARALLEL_ARGS=" \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --sequence-parallel
"

CMD=" \
    ../../pretrain_gpt.py \
    $PARALLEL_ARGS \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --data-path $TRAIN_DATA \
    --dataloader-type single \
    --num-workers 0 \
    "

c="fe"

# Bind mask for one thread per core
BIND_MASK_1="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

BIND_MASK="$BIND_MASK_1"

echo $CMD

echo "START $SLURM_JOBID: $(date)"

if [ "$SLURM_JOB_PARTITION" = "dev-g" ]; then
    echo "Lumi dev-g partition is used, CPU binding is not used"
    srun --label ./launch.sh $CMD
else
  echo "Using --cpu-bind=mask_cpu:$BIND_MASK"
  srun \
    --cpu-bind=mask_cpu:$BIND_MASK \
    --label launch.sh $CMD
fi

echo "END $SLURM_JOBID: $(date)"
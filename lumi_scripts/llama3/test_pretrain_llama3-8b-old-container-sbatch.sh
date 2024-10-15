#!/bin/bash

#SBATCH --job-name=test-llama31-8B-continued-pretraining-8-nodes-old-container
#SBATCH --time=01:30:00
##SBATCH --time=02-00:00:00 production
#SBATCH --partition=dev-g
##SBATCH --partition=standard-g
#SBATCH --nodes=8
##SBATCH --nodes=64 production
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=480G
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000353
#SBATCH --output=llama-logs/%x_%j.output
#SBATCH --error=llama-logs/%x_%j.error

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS

# compilers in the container
export CC=gcc-10
export CXX=g++-10

export MEMORY_OPT_ALLREDUCE_SIZE=150000000
echo "MEMORY_OPT_ALLREDUCE_SIZE $MEMORY_OPT_ALLREDUCE_SIZE"

mkdir -p workdir
wd=$(realpath workdir)
set -euo pipefail

# singularity setup
CONTAINER="/scratch/project_462000353/containers/vaino_flashattention_v2_new"
SING_BIND="/scratch/project_462000353"

PP_SIZE=1
TP_SIZE=2

OPTSTRING=":drp"


while getopts ${OPTSTRING} opt; do
  case ${opt} in

    d)
      echo "Option -d devel, start from scratch"
      SAVE_CHECKPOINT_DIR="/scratch/project_462000353/akselir/llama31-8b-megatron-format-devel-8-nodes-old-container-checkpoints-tp$TP_SIZE-pp$PP_SIZE"
      LOAD_CHECKPOINT_DIR="/scratch/project_462000353/models/llama31-8b-tp$TP_SIZE-pp$PP_SIZE-megatron-format"
      TENSORBOARD_PATH="$wd/tensorboard/$SLURM_JOB_NAME"
      rm -rf "$SAVE_CHECKPOINT_DIR" "$TENSORBOARD_PATH"
      mkdir "$SAVE_CHECKPOINT_DIR"
      LOG_INTERVAL=1
      SAVE_INTERVAL=100
      EVAL_INTERVAL=10
      EVAL_STEPS=10
      LOAD_OPTIMIZER_STATES=false
      ;;
    r)
      echo "Option -r is used, training is started from scratch,so optimizer is not loaded, but results are saved"
      LOAD_CHECKPOINT_DIR="/scratch/project_462000353/models/llama31-8b-tp$TP_SIZE-pp$PP_SIZE-megatron-format"
      SAVE_CHECKPOINT_DIR="/scratch/project_462000353/models/llama31-8b-tp$TP_SIZE-pp$PP_SIZE-megatron-format"
      TENSORBOARD_PATH="$wd/tensorboard/$SLURM_JOB_NAME"
      LOG_INTERVAL=1
      SAVE_INTERVAL=1000
      EVAL_INTERVAL=4000
      EVAL_STEPS=100
      LOAD_OPTIMIZER_STATES=false
      ;;
      
    p)
      echo "Option -p is used, training is continued so optimizer should be loaded"
      LOAD_CHECKPOINT_DIR="/scratch/project_462000353/models/llama31-8b-megatron-format-tp$TP_SIZE-pp$PP_SIZE"
      SAVE_CHECKPOINT_DIR="/scratch/project_462000353/models/llama31-8b-megatron-format-tp$TP_SIZE-pp$PP_SIZE"
      TENSORBOARD_PATH="$wd/tensorboard/$SLURM_JOB_NAME"
      LOAD_OPTIMIZER_STATES=true
      LOG_INTERVAL=1
      SAVE_INTERVAL=1000
      EVAL_INTERVAL=4000
      EVAL_STEPS=100
      ;;
    ?)
      echo "Invalid option: -${OPTARG}."
      exit 1
      ;;
  esac
done



DATA_ROOT="/scratch/project_462000353/data/processed-llama31/merged"
CACHE_PATH="${DATA_ROOT}/index-cache"
TRAIN_DATA="0.7 ${DATA_ROOT}/finnish 0.15 ${DATA_ROOT}/fineweb-edu 0.01 ${DATA_ROOT}/xling 0.04 ${DATA_ROOT}/starcoder"

TOKENIZER="/scratch/project_462000353/models/llama31-8b"

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


TOTAL_TOKENS=75_000_000_000
TOTAL_TOKENS=${TOTAL_TOKENS//_}    # drop "_" for bash math
TRAIN_SAMPLES=$((TOTAL_TOKENS/SEQ_LEN))
LR_DECAY_SAMPLES=$TRAIN_SAMPLES
LR_WARMUP_SAMPLES=$((GLOBAL_BATCH_SIZE*1072))



if $LOAD_OPTIMIZER_STATES; then
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
else
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
    --no-load-optim \
    --no-load-rng \
    --finetune \
    "
fi

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
    --overlap-p2p-communication \
    --recompute-activations \
    --make-vocab-size-divisible-by 1 \
    --distributed-timeout-minutes 60
    $OPTIMIZER_ARGS \
    "


OUTPUT_ARGS=" \
    --log-interval $LOG_INTERVAL \
    --save $SAVE_CHECKPOINT_DIR \
    --save-interval $SAVE_INTERVAL \
    --load $LOAD_CHECKPOINT_DIR \
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
    --num-workers 7 \
    "

c="fe"

# Bind mask for one thread per core
BIND_MASK_1="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

# Bind mask for two threads per core
#BIND_MASK_2="0x${c}00000000000000${c}000000000000,0x${c}00000000000000${c}00000000000000,0x${c}00000000000000${c}0000,0x${c}00000000000000${c}000000,0x${c}00000000000000${c},0x${c}00000000000000${c}00,0x${c}00000000000000${c}00000000,0x${c}00000000000000${c}0000000000"

BIND_MASK="$BIND_MASK_1"
echo "Using --cpu-bind=mask_cpu:$BIND_MASK"

# add a pythonuserbase to an empty dir to avoid problems with user's local
# python install being imported into the singularity container.
mkdir -p pythonuserbase
export PYTHONUSERBASE=pythonuserbase

echo $CMD

echo "START $SLURM_JOBID: $(date)"

if [ ! -d "$wd"/cray-deps ] ; then
  rm -rf "$wd"/cray-deps
  mkdir "$wd"/cray-deps
  cp /usr/lib64/libcxi* $wd/cray-deps
fi

if [ "$SLURM_JOB_PARTITION" = "dev-g" ]; then
  echo "Lumi dev-g partition is used, CPU binding is not used"
  srun \
      --label \
      --cpu-bind=mask_cpu:$BIND_MASK \
      singularity exec \
      -B $PWD \
      -B /opt/cray:/opt/cray \
      -B "$wd"/cray-deps:/opt/cray-deps \
      -B "$wd":/workdir \
      -B "$SING_BIND" \
      "$CONTAINER" \
      ./old_launch.sh \
      $CMD
else
  echo "Using --cpu-bind=mask_cpu:$BIND_MASK"
  srun \
      --label \
      --cpu-bind=mask_cpu:$BIND_MASK \
      singularity exec \
      -B $PWD \
      -B /opt/cray:/opt/cray \
      -B "$wd"/cray-deps:/opt/cray-deps \
      -B "$wd":/workdir \
      -B "$SING_BIND" \
      "$CONTAINER" \
      ./old_launch.sh \
      $CMD
fi

echo "END $SLURM_JOBID: $(date)"

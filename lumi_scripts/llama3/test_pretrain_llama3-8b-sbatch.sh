#!/bin/bash

#SBATCH --job-name=test-llama31-8B-continued-pretraining-2-nodes-explicit-vocab-size-standard-g

# testing 
#SBATCH --time=00:30:00
##SBATCH --time=02-00:00:00 production
##SBATCH --partition=dev-g
#SBATCH --partition=standard-g
#SBATCH --nodes=2
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

set -euo pipefail

module purge
ml use /appl/local/csc/modulefiles/
ml pytorch/2.4

mkdir -p workdir
wd=$(realpath workdir)

PP_SIZE=2
TP_SIZE=2

OPTSTRING=":drp"

while getopts ${OPTSTRING} opt; do
  case ${opt} in

    d)
      echo "Option -d devel, start from scratch"
      SAVE_CHECKPOINT_DIR="/scratch/project_462000353/akselir/llama31-8b-megatron-format-devel-checkpoints-tp$TP_SIZE-pp$PP_SIZE"
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

# distributed setup
export NCCL_IFNAME=hsn
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=INFO 
export NCCL_DEBUG_SUBSYS=INIT,COLL
export NCCL_DEBUG_FILE=/tmp/$(whoami)-rccl-rank$SLURM_PROCID.txt
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999



DATA_ROOT="/flash/project_462000353/continued_pretraining/data/mistral_tokenizer_extended_fin/merged_datasets_standard"
TRAIN_DATA="0.7 ${DATA_ROOT}/finnish 0.15 ${DATA_ROOT}/fineweb-ebu 0.01 ${DATA_ROOT}/Tatoeba 0.04 ${DATA_ROOT}/starcoderdata"

TOKENIZER="/scratch/project_462000353/models/llama31-8b"

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=512
NLAYERS=32
NHIDDEN=4096
NHEADS=32
FFN_HIDDEN_SIZE=14336
NUM_QUERY_GROUPS=8
SEQ_LEN=8192

LEARNING_RATE=2e-5
MIN_LEARNING_RATE=1e-8
export MEMORY_OPT_ALLREDUCE_SIZE=150000000
echo "MEMORY_OPT_ALLREDUCE_SIZE $MEMORY_OPT_ALLREDUCE_SIZE"

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
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tokenizer-type HFPretrainedTokenizer \
    --hf_tokenizer_path $TOKENIZER \
    --bf16 \
    --disable-bias-linear \
    --init-method-std 0.0048 \
    --no-gradient-accumulation-fusion \
    --normalization RMSNorm \
    --norm-epsilon 1e-05 \
    --seed 1234 \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --use-flash-attn \
    --position-embedding-type rope \
    --use-rope-scaling \
    --rope-theta 500000.0 \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --no-query-key-layer-scaling \
    --no-masked-softmax-fusion \
    --sequence-parallel \
    --num-query-groups $NUM_QUERY_GROUPS \
    --group-query-attention \
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

CMD=" \
    ../../pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
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
echo "Using --cpu-bind=mask_cpu:$BIND_MASK"


echo $CMD

echo "START $SLURM_JOBID: $(date)"

if [ "$SLURM_JOB_PARTITION" = "dev-g" ]; then
  echo "Lumi dev-g partition is used, CPU binding is not used"
  srun \
    --label launch.sh \
     $CMD
else
  echo "Using --cpu-bind=mask_cpu:$BIND_MASK"
  srun \
    --cpu-bind=mask_cpu:$BIND_MASK \
    --label launch.sh \
     $CMD
fi

echo "END $SLURM_JOBID: $(date)"

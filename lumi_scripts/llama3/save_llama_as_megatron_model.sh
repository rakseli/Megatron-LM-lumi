#!/bin/bash

#SBATCH --job-name=convert_llama3
#SBATCH --time=00:30:00
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000353
#SBATCH --output=llama-logs/%x_%j.output
#SBATCH --error=llama-logs/%x_%j.error

module purge
ml use /appl/local/csc/modulefiles/
ml load pytorch/2.4

#targets are needed for right model loading in training phase
TARGET_TP=4
TARGET_PP=4
#HF format model
HF_FORMAT_DIR=/scratch/project_462000353/models/llama31-8b
#output dir model
MEGATRON_FORMAT_DIR="/scratch/project_462000353/models/llama31-8b-tp$TARGET_TP-pp$TARGET_PP-megatron-format"
mkdir -p $MEGATRON_FORMAT_DIR
TOKENIZER_MODEL=$HF_FORMAT_DIR

python3 ../../tools/checkpoint/util.py \
  --model-type GPT \
  --loader loader_llama_mistral \
  --saver megatron \
  --target-tensor-parallel-size ${TARGET_TP} \
  --target-pipeline-parallel-size ${TARGET_PP} \
  --load-dir ${HF_FORMAT_DIR} \
  --save-dir ${MEGATRON_FORMAT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --model-size llama3-8B \
  --checkpoint-type hf \
  --bf16 \
  --megatron-path /scratch/project_462000353/akselir/poro-length-extrapolation/Megatron-LM-lumi/megatron \

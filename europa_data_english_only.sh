#!/bin/bash
TRAIN_DATA="0.04000383024140028 /scratch/project_462000353/europa-data/europa-final-merge/en/redpajama-v2/merged-train.jsonl_text_document"
TRAIN_DATA=${TRAIN_DATA}" 0.0007339151520762105 /scratch/project_462000353/europa-data/europa-final-merge/en/british-library/merged-train.jsonl_text_document"
TRAIN_DATA=${TRAIN_DATA}" 0.007526743082733043 /scratch/project_462000353/europa-data/europa-final-merge/en/cosmopedia/merged-train.jsonl_text_document"
TRAIN_DATA=${TRAIN_DATA}" 0.00025024056041623945 /scratch/project_462000353/europa-data/europa-final-merge/en/natural_instructions/merged-train.jsonl_text_document"
TRAIN_DATA=${TRAIN_DATA}" 0.020729286363884245 /scratch/project_462000353/europa-data/europa-final-merge/en/peS2o/merged-train.jsonl_text_document"
TRAIN_DATA=${TRAIN_DATA}" 0.024193489681497402 /scratch/project_462000353/europa-data/europa-final-merge/en/pile-of-law/merged-train.jsonl_text_document"
TRAIN_DATA=${TRAIN_DATA}" 0.0015767108381223897 /scratch/project_462000353/europa-data/europa-final-merge/en/pubmed/merged-train.jsonl_text_document"
TRAIN_DATA=${TRAIN_DATA}" 0.008266545203468057 /scratch/project_462000353/europa-data/europa-final-merge/en/pubmed-central/merged-train.jsonl_text_document"
TRAIN_DATA=${TRAIN_DATA}" 0.026084744293533906 /scratch/project_462000353/europa-data/europa-final-merge/en/reddit/merged-train.jsonl_text_document"
TRAIN_DATA=${TRAIN_DATA}" 0.0009542737277193715 /scratch/project_462000353/europa-data/europa-final-merge/multiling/europarl/merged-train.jsonl_text_document"
TRAIN_DATA=${TRAIN_DATA}" 0.017272785060048108 /scratch/project_462000353/europa-data/europa-final-merge/multiling/wikipedia/merged-train.jsonl_text_document"
TRAIN_DATA=${TRAIN_DATA}" 0.044149123838940325 /scratch/project_462000353/europa-data/europa-final-merge/xling/tatoeba/merged-train.jsonl_text_document"
TRAIN_DATA=${TRAIN_DATA}" 0.06933997241842715 /scratch/project_462000353/europa-data/europa-final-merge/code/starcoder/merged-train.jsonl_text_document"

VALIDATION_DATA="0.04000383024140028 /scratch/project_462000353/europa-data/europa-final-merge/en/redpajama-v2/merged-validate.jsonl_text_document"
VALIDATION_DATA=${VALIDATION_DATA}" 0.0007339151520762105 /scratch/project_462000353/europa-data/europa-final-merge/en/british-library/merged-validate.jsonl_text_document"
VALIDATION_DATA=${VALIDATION_DATA}" 0.007526743082733043 /scratch/project_462000353/europa-data/europa-final-merge/en/cosmopedia/merged-validate.jsonl_text_document"
VALIDATION_DATA=${VALIDATION_DATA}" 0.00025024056041623945 /scratch/project_462000353/europa-data/europa-final-merge/en/natural_instructions/merged-validate.jsonl_text_document"
VALIDATION_DATA=${VALIDATION_DATA}" 0.020729286363884245 /scratch/project_462000353/europa-data/europa-final-merge/en/peS2o/merged-validate.jsonl_text_document"
VALIDATION_DATA=${VALIDATION_DATA}" 0.024193489681497402 /scratch/project_462000353/europa-data/europa-final-merge/en/pile-of-law/merged-validate.jsonl_text_document"
VALIDATION_DATA=${VALIDATION_DATA}" 0.0015767108381223897 /scratch/project_462000353/europa-data/europa-final-merge/en/pubmed/merged-validate.jsonl_text_document"
VALIDATION_DATA=${VALIDATION_DATA}" 0.008266545203468057 /scratch/project_462000353/europa-data/europa-final-merge/en/pubmed-central/merged-validate.jsonl_text_document"
VALIDATION_DATA=${VALIDATION_DATA}" 0.026084744293533906 /scratch/project_462000353/europa-data/europa-final-merge/en/reddit/merged-validate.jsonl_text_document"
VALIDATION_DATA=${VALIDATION_DATA}" 0.0009542737277193715 /scratch/project_462000353/europa-data/europa-final-merge/multiling/europarl/merged-validate.jsonl_text_document"
VALIDATION_DATA=${VALIDATION_DATA}" 0.017272785060048108 /scratch/project_462000353/europa-data/europa-final-merge/multiling/wikipedia/merged-validate.jsonl_text_document"
VALIDATION_DATA=${VALIDATION_DATA}" 0.044149123838940325 /scratch/project_462000353/europa-data/europa-final-merge/xling/tatoeba/merged-validate.jsonl_text_document"
VALIDATION_DATA=${VALIDATION_DATA}" 0.06933997241842715 /scratch/project_462000353/europa-data/europa-final-merge/code/starcoder/merged-validate.jsonl_text_document"

export TRAIN_DATA
export VALIDATION_DATA
echo "set TRAIN_DATA and VALIDATION_DATA environment variables"
#!/bin/bash

# Common args
PYTHON_SCRIPT="/scratch/sundong/doyoon/playground/forgetting/evaluate/main.py"
DATASET="CIFAR10"
IS_INSTRUCT=1
GPUS="0,1,2,3"
BATCH_SIZE=512
OUT_DIR="./results/qwen2.5vl3b-instruct-cifar10"
WANDB_PROJECT="vlm-forgetting"
WANDB_GROUP="rec-effect-on-cifar10"

# Structured run config
declare -a MODEL_NAMES=(
  "Qwen/Qwen2.5-VL-3B-Instruct"
  "cosmos1030/Qwen2.5_VL-3B-rec-SFT-500steps"
  "omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps"
)

declare -a RUN_NAMES=(
  "qwen2_5_vl"
  "rec_sft_500_steps"
  "rec_grpo_500_steps"
)

# Run all
for i in "${!MODEL_NAMES[@]}"; do
  MODEL_NAME="${MODEL_NAMES[$i]}"
  RUN_NAME="${RUN_NAMES[$i]}"

  echo "▶ Running: $RUN_NAME"
  python "$PYTHON_SCRIPT" \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --is_instruct "$IS_INSTRUCT" \
    --gpus "$GPUS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUT_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_group_name "$WANDB_GROUP" \
    --wandb_run_name "$RUN_NAME"
done

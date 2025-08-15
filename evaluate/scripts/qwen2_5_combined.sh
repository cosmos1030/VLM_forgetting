#!/bin/bash

# Common args
PYTHON_SCRIPT="/public/home/group_ucb/dyk6208/VLM_forgetting/evaluate/main.py"
IS_INSTRUCT=1
GPUS="0,1,2,3"
BATCH_SIZE=512
OUT_DIR="./results/dummy"
WANDB_PROJECT="vlm-forgetting"

MODEL_DIRS=(
  "/public/home/group_ucb/dyk6208/LLaMA-Factory/saves/qwen2_5_vl-3b/full/cifar10_sft"
  "/public/home/group_ucb/dyk6208/LLaMA-Factory/saves/qwen2_5_vl-3b/full/cifar100_sft"
)

DATASETS=("CIFAR10" "CIFAR100" "MNIST")

# Checkpoints: 400 ~ 3600 step by 400
for MODEL_DIR in "${MODEL_DIRS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for step in {400..3600..400}; do
      MODEL_NAME="${MODEL_DIR}/checkpoint-${step}"

      # Extract short model tag for wandb group naming
      MODEL_TAG=$(basename "$MODEL_DIR")  # e.g., cifar10_sft

      RUN_NAME="sft_train_${MODEL_TAG}_test_${DATASET}_qwen2_5_step_${step}"
      WANDB_GROUP="sft_train_${MODEL_TAG}_test_${DATASET}_qwen2_5"

      echo "▶ Running: $RUN_NAME"
      TRANSFORMERS_VERBOSITY=debug python "$PYTHON_SCRIPT" \
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
  done
done


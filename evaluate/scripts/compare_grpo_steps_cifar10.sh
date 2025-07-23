#!/bin/bash

# Common args
PYTHON_SCRIPT="/scratch/sundong/doyoon/playground/forgetting/evaluate/main.py"
DATASET="CIFAR10"
IS_INSTRUCT=1
GPUS="0,1,2,3"
BATCH_SIZE=512
OUT_DIR="./results/dummy"
WANDB_PROJECT="vlm-forgetting"
WANDB_GROUP="cifar100_trained_grpo_cifar10_testing"

# Zero-shot run
python "$PYTHON_SCRIPT" \
  --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
  --dataset "$DATASET" \
  --is_instruct "$IS_INSTRUCT" \
  --gpus "$GPUS" \
  --batch_size "$BATCH_SIZE" \
  --output_dir "$OUT_DIR" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_group_name "$WANDB_GROUP" \
  --wandb_run_name "zero_shot"

# Finetuned checkpoints: 1000 ~ 7000
for step in {200..600..100}; do
  MODEL_NAME="/scratch/sundong/doyoon/playground/forgetting/train/fully_finetuning_grpo/checkpoints/GRPO-Qwen2.5-VL-3B-Instruct-cifar10/checkpoint-${step}"
  RUN_NAME="${step}_steps"

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

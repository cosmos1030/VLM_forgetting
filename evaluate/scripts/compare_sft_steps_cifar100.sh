#!/bin/bash

# # Common args
# PYTHON_SCRIPT="/scratch/sundong/doyoon/playground/forgetting/evaluate/main.py"
# DATASET="CIFAR100"
# IS_INSTRUCT=1
# GPUS="0,1,2,3"
# BATCH_SIZE=256
# OUT_DIR="./results/dummy"
# WANDB_PROJECT="vlm-forgetting"
# WANDB_GROUP="cifar100_trained_sft_cifar100_test"

# # Zero-shot run
# python "$PYTHON_SCRIPT" \
#   --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
#   --dataset "$DATASET" \
#   --is_instruct "$IS_INSTRUCT" \
#   --gpus "$GPUS" \
#   --batch_size "$BATCH_SIZE" \
#   --output_dir "$OUT_DIR" \
#   --wandb_project "$WANDB_PROJECT" \
#   --wandb_group_name "$WANDB_GROUP" \
#   --wandb_run_name "zero_shot_cifar100"

# # Finetuned checkpoints: 1000 ~ 7000
# for step in {1000..6000..1000}; do
#   MODEL_NAME="/scratch/sundong/doyoon/playground/forgetting/train/fully_finetuning_sft/checkpoints/cifar100-sft-final-run/checkpoint-${step}"
#   RUN_NAME="sft_${step}_steps_cifar100"

#   echo "▶ Running: $RUN_NAME"
#   python "$PYTHON_SCRIPT" \
#     --model_name "$MODEL_NAME" \
#     --dataset "$DATASET" \
#     --is_instruct "$IS_INSTRUCT" \
#     --gpus "$GPUS" \
#     --batch_size "$BATCH_SIZE" \
#     --output_dir "$OUT_DIR" \
#     --wandb_project "$WANDB_PROJECT" \
#     --wandb_group_name "$WANDB_GROUP" \
#     --wandb_run_name "$RUN_NAME"
# done




# Common args
PYTHON_SCRIPT="/scratch/sundong/doyoon/playground/forgetting/evaluate/main.py"
DATASET="CIFAR100"
IS_INSTRUCT=1
GPUS="0,1,2,3"
BATCH_SIZE=256
OUT_DIR="./results/dummy"
WANDB_PROJECT="vlm-forgetting"
WANDB_GROUP="cifar10_trained_sft_cifar100_test"

# # Zero-shot run
# python "$PYTHON_SCRIPT" \
#   --model_name "Qwen/Qwen2.5-VL-3B-Instruct" \
#   --dataset "$DATASET" \
#   --is_instruct "$IS_INSTRUCT" \
#   --gpus "$GPUS" \
#   --batch_size "$BATCH_SIZE" \
#   --output_dir "$OUT_DIR" \
#   --wandb_project "$WANDB_PROJECT" \
#   --wandb_group_name "$WANDB_GROUP" \
#   --wandb_run_name "zero_shot_cifar10"

# Finetuned checkpoints: 1000 ~ 7000
for step in {1000..18000..1000}; do
  MODEL_NAME="/scratch/sundong/doyoon/playground/forgetting/LLaMA-Factory/saves/qwen2_5_vl-3b/full/cifar10_sft/checkpoint-${step}"
  RUN_NAME="sft_${step}_steps_train_cifar10_test_cifar100"

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

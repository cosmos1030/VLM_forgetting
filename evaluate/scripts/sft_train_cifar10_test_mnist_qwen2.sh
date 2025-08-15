#!/bin/bash

# Common args
PYTHON_SCRIPT="/public/home/group_ucb/dyk6208/VLM_forgetting/evaluate/main.py"
DATASET="MNIST"
IS_INSTRUCT=0
GPUS="0,1,2,3"
BATCH_SIZE=512
OUT_DIR="./results/dummy"
WANDB_PROJECT="vlm-forgetting"
WANDB_GROUP="sft_train_cifar10_test_mnist_qwen2"


# Finetuned checkpoints: 1000 ~ 7000
for step in {400..3600..400}; do
  MODEL_NAME="/public/home/group_ucb/dyk6208/LLaMA-Factory/saves/qwen2_vl-2b/full/cifar10_sft/checkpoint-${step}"
  RUN_NAME="sft_train_cifar10_test_mnist_qwen2_step_${step}"

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

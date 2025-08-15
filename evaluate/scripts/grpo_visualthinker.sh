#!/bin/bash

# Common args
PYTHON_SCRIPT="/public/home/group_ucb/dyk6208/VLM_forgetting/evaluate/main.py"
IS_INSTRUCT=0
GPUS="0,1,2,3"
BATCH_SIZE=512
OUT_DIR="./results/dummy"
WANDB_PROJECT="vlm-forgetting"

# List of datasets
DATASETS=("MNIST" "CIFAR10" "CIFAR100")

# Finetuned checkpoints: 500 ~ 3500 step 간격 500
for DATASET in "${DATASETS[@]}"; do
  WANDB_GROUP="grpo_train_sat_test_${DATASET,,}_qwen2"  # 소문자로
  for STEP in {500..3500..500}; do
     MODEL_NAME="/public/home/group_ucb/dyk6208/VisualThinker-R1-Zero/src/open-r1-multimodal/outputs/Qwen2-VL-2B-GRPO-Base-SAT/checkpoint-${STEP}"
    RUN_NAME="grpo_train_sat_test_${DATASET,,}_qwen2_step_${STEP}"

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


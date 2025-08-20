#!/bin/bash
# ------------------------------
# [1] 데이터 및 모델 경로 설정
# ------------------------------
# ⭐️ 변수 이름은 그대로 유지했습니다.
data_paths="/public/home/group_ucb/dyk6208/VLM_forgetting/train/fully_finetuning_grpo/data/cifar10_train.jsonl"
image_folders="/public/home/group_ucb/dyk6208/VLM_forgetting/train/fully_finetuning_grpo/images/cifar10"
model_path="/public/home/group_ucb/dyk6208/models/Qwen2.5-VL-3B-Instruct"
# ------------------------------
# [2] 실험 이름 설정
# ------------------------------
export EXP_NAME="GRPO-Qwen2.5-VL-3B-Instruct-cifar10_v3"

# ------------------------------
# [3] 로깅 / 디버그 설정 (기존과 동일)
# ------------------------------
export DEBUG_MODE="true"
mkdir -p runs/${EXP_NAME}/log
export LOG_PATH="runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

# ------------------------------
# [4] WandB 설정 (기존과 동일)
# ------------------------------
export WANDB_DISABLED=false

# ------------------------------
# [5] GPU 설정 (기존과 동일)
# ------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ------------------------------
# [6] torchrun을 통한 학습 실행 (수정됨)
# ------------------------------
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12351 \
  /public/home/group_ucb/dyk6208/VLM_forgetting/train/fully_finetuning_grpo/grpo_jsonl.py \
    --output_dir checkpoints/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 400 \
    --num_generations 8 \
    --max_completion_length 2048 \
    --reward_funcs accuracy format \
    --beta 0.04 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --report_to wandb \
    --deepspeed /public/home/group_ucb/dyk6208/VLM_forgetting/train/fully_finetuning_grpo/local_scripts/zero3.json \
    --dataset_name "cifar10"

echo "✅ Training completed for ${EXP_NAME}"

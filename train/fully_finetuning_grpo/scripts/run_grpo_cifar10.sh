#!/bin/bash

# ------------------------------
# [0] 프로젝트 경로 설정
# ------------------------------
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
cd /scratch/sundong/doyoon/playground/forgetting/VLM-R1/src/open-r1-multimodal

# ------------------------------
# [1] 데이터 및 모델 경로 설정
# ------------------------------
data_paths="/scratch/sundong/doyoon/playground/forgetting/VLM-R1/data/cifar10_train.jsonl"  # CIFAR-10 JSONL 학습 데이터
image_folders="/scratch/sundong/doyoon/playground/forgetting/VLM-R1/images/cifar10"         # 이미지 루트 폴더
model_path="Qwen/Qwen2.5-VL-3B-Instruct"
is_reward_customized_from_vlm_module=False

# ------------------------------
# [2] 실험 이름 및 태스크 타입
# ------------------------------
export EXP_NAME="Qwen2.5-VL-3B-Instruct-cifar10"
TASK_TYPE="default"  # <-- 'cifar10'이 아님. 반드시 'default'로 유지

# ------------------------------
# [3] 로깅 / 디버그 설정
# ------------------------------
export DEBUG_MODE="true"
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

# ------------------------------
# [4] WandB 설정
# ------------------------------
export WANDB_DISABLED=false

# ------------------------------
# [5] GPU 설정
# ------------------------------
export CUDA_VISIBLE_DEVICES=1,2,3,4

# ------------------------------
# [6] torchrun을 통한 학습 실행
# ------------------------------
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12349 \
  /scratch/sundong/doyoon/playground/forgetting/VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 50 \
    --num_generations 8 \
    --max_completion_length 2048 \
    --reward_funcs accuracy format \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name this_is_not_used \
    --deepspeed /scratch/sundong/doyoon/playground/forgetting/VLM-R1/src/open-r1-multimodal/local_scripts/zero3.json

echo "✅ Training completed for ${EXP_NAME}"

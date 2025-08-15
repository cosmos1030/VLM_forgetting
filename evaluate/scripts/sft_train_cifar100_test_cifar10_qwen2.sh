# Common args
PYTHON_SCRIPT="/public/home/group_ucb/dyk6208/VLM_forgetting/evaluate/main.py"
DATASET="CIFAR10"
IS_INSTRUCT=0
GPUS="0,1,2,3"
BATCH_SIZE=512
BASE_RUN_NAME="sft_train_cifar100_test_cifar10_qwen2"
WANDB_PROJECT="vlm-forgetting"
WANDB_GROUP=$BASE_RUN_NAME

# Finetuned checkpoints: 200 ~ 1800
for step in {5000..30000..5000}; do
  RUN_NAME="${step}_steps"
  MODEL_NAME="/public/home/group_ucb/dyk6208/LLaMA-Factory/saves/qwen2_vl-2b/full/cifar100_sft/checkpoint-${step}"
  OUT_DIR="./results/${BASE_RUN_NAME}/${RUN_NAME}"

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


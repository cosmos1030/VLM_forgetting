#!/bin/bash

# ------------------------------
# [0] 실행할 학습 스크립트 경로 설정
# ------------------------------
TRAIN_SCRIPT="scripts/run_grpo_cifar10_revised.sh"
LOG_FILE="log/train.out"

# ------------------------------
# [1] 백그라운드 실행
# ------------------------------
echo "🚀 Launching training script: ${TRAIN_SCRIPT}"
echo "📂 Logging to: ${LOG_FILE}"

nohup bash ${TRAIN_SCRIPT} > ${LOG_FILE} 2>&1 &

echo "✅ Training started in background. To monitor progress:"
echo "    tail -f ${LOG_FILE}"
echo "    ps aux | grep ${TRAIN_SCRIPT}"

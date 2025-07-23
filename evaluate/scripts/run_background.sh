#!/bin/bash

TRAIN_SCRIPT="scripts/compare_grpo_steps_cifar100.sh"
mkdir -p 'log'
LOG_FILE="log/train.out"

echo "🚀 Launching training script: ${TRAIN_SCRIPT}"
echo "📂 Logging to: ${LOG_FILE}"

nohup bash ${TRAIN_SCRIPT} > ${LOG_FILE} 2>&1 &

echo "✅ Training started in background. To monitor progress:"
echo "    tail -f ${LOG_FILE}"
echo "    ps aux | grep ${TRAIN_SCRIPT}"
tail -f ${LOG_FILE}
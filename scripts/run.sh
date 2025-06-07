#!/usr/bin/env bash
# Give privilage: chmod +x run.sh

# qwen2vl2b SAT grpo trained
# python /clifford-data/home/doyoonkim/projects/zero_shot/main.py \
#   --model_name turningpoint-ai/VisualThinker-R1-Zero \
#   --dataset CIFAR10 \
#   --is_instruct 0 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 512 \
#   --output_dir ./results/visualthinker-grpo-cifar10

python /clifford-data/home/doyoonkim/projects/zero_shot/main.py \
  --model_name turningpoint-ai/VisualThinker-R1-Zero \
  --dataset CIFAR100 \
  --is_instruct 0 \
  --gpus 0,1,2,3,4,5,6,7,8,9 \
  --batch_size 128 \
  --output_dir ./results/visualthinker-grpo-cifar100

python /clifford-data/home/doyoonkim/projects/zero_shot/main.py \
  --model_name turningpoint-ai/VisualThinker-R1-Zero \
  --dataset MNIST \
  --is_instruct 0 \
  --gpus 0,1,2,3,4,5,6,7,8,9 \
  --batch_size 128 \
  --output_dir ./results/visualthinker-grpo-mnist

# qwen2vl2b SAT sft trained
python /clifford-data/home/doyoonkim/projects/zero_shot/main.py \
  --model_name cosmos1030/Qwen2_VL-2B-SFT_revised2 \
  --dataset CIFAR10 \
  --is_instruct 0 \
  --gpus 0,1,2,3,4,5,6,7,8,9 \
  --batch_size 128 \
  --output_dir ./results/visualthinker-sft-cifar10

python /clifford-data/home/doyoonkim/projects/zero_shot/main.py \
  --model_name cosmos1030/Qwen2_VL-2B-SFT_revised2 \
  --dataset CIFAR100 \
  --is_instruct 0 \
  --gpus 0,1,2,3,4,5,6,7,8,9 \
  --batch_size 128 \
  --output_dir ./results/visualthinker-sft-cifar100

python /clifford-data/home/doyoonkim/projects/zero_shot/main.py \
  --model_name cosmos1030/Qwen2_VL-2B-SFT_revised2 \
  --dataset MNIST \
  --is_instruct 0 \
  --gpus 0,1,2,3,4,5,6,7,8,9 \
  --batch_size 128 \
  --output_dir ./results/visualthinker-sft-mnist

# pure qwen2vl-2b
python /clifford-data/home/doyoonkim/projects/zero_shot/main.py \
  --model_name Qwen/Qwen2-VL-2B \
  --dataset CIFAR10 \
  --is_instruct 0 \
  --gpus 0,1,2,3,4,5,6,7,8,9 \
  --batch_size 128 \
  --output_dir ./results/qwen2vl2b-cifar10

python /clifford-data/home/doyoonkim/projects/zero_shot/main.py \
  --model_name Qwen/Qwen2-VL-2B \
  --dataset CIFAR100 \
  --is_instruct 0 \
  --gpus 0,1,2,3,4,5,6,7,8,9 \
  --batch_size 128 \
  --output_dir ./results/qwen2vl2b-cifar100

python /clifford-data/home/doyoonkim/projects/zero_shot/main.py \
  --model_name Qwen/Qwen2-VL-2B \
  --dataset MNIST \
  --is_instruct 0 \
  --gpus 0,1,2,3,4,5,6,7,8,9 \
  --batch_size 128 \
  --output_dir ./results/qwen2vl2b-mnist


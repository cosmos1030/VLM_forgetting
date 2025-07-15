#!/usr/bin/env bash
# Give privilage: chmod +x run.sh

#rec sft-500
#cosmos1030/Qwen2.5_VL-3B-rec-SFT
python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
  --model_name cosmos1030/Qwen2.5_VL-3B-rec-SFT-500steps \
  --dataset CIFAR10 \
  --is_instruct 1 \
  --gpus 0,1,2,3,4,5,6,7,8,9 \
  --batch_size 256 \
  --output_dir ./results/qwen2.5vl3b-instruct-rec-cifar10-sft-500steps

python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
  --model_name cosmos1030/Qwen2.5_VL-3B-rec-SFT-500steps \
  --dataset CIFAR100 \
  --is_instruct 1 \
  --gpus 0,1,2,3,4,5,6,7,8,9 \
  --batch_size 128 \
  --output_dir ./results/qwen2.5vl3b-instruct-rec-cifar100-sft-500steps

python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
  --model_name cosmos1030/Qwen2.5_VL-3B-rec-SFT-500steps \
  --dataset MNIST \
  --is_instruct 1 \
  --gpus 0,1,2,3,4,5,6,7,8,9 \
  --batch_size 256 \
  --output_dir ./results/qwen2.5vl3b-instruct-rec-mnist-sft-500steps


# #gui sft
# #cosmos1030/Qwen2.5_VL-3B-rec-SFT
# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name cosmos1030/Qwen2.5_VL-3B-Instruct-GUI-SFT \
#   --dataset CIFAR10 \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 256 \
#   --output_dir ./results/qwen2.5vl3b-instruct-gui-cifar10-sft

python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
  --model_name cosmos1030/Qwen2.5_VL-3B-Instruct-GUI-SFT \
  --dataset CIFAR100 \
  --is_instruct 1 \
  --gpus 0,1,2,3,4,5,6,7,8,9 \
  --batch_size 128 \
  --output_dir ./results/qwen2.5vl3b-instruct-gui-cifar100-sft

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name cosmos1030/Qwen2.5_VL-3B-Instruct-GUI-SFT \
#   --dataset MNIST \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 256 \
#   --output_dir ./results/qwen2.5vl3b-instruct-gui-mnist-sft

# rec sft
# cosmos1030/Qwen2.5_VL-3B-rec-SFT
# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name cosmos1030/Qwen2.5_VL-3B-rec-SFT \
#   --dataset CIFAR10 \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 256 \
#   --output_dir ./results/qwen2.5vl3b-instruct-rec-cifar10-sft

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name cosmos1030/Qwen2.5_VL-3B-rec-SFT \
#   --dataset CIFAR100 \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 128 \
#   --output_dir ./results/qwen2.5vl3b-instruct-rec-cifar100-sft

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name cosmos1030/Qwen2.5_VL-3B-rec-SFT \
#   --dataset MNIST \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 256 \
#   --output_dir ./results/qwen2.5vl3b-instruct-rec-mnist-sft


# gui

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name konkazzz/GT-r1 \
#   --dataset CIFAR10 \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 256 \
#   --output_dir ./results/qwen2.5vl3b-instruct-gui-cifar10

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name konkazzz/GT-r1 \
#   --dataset CIFAR100 \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 128 \
#   --output_dir ./results/qwen2.5vl3b-instruct-gui-cifar100

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name konkazzz/GT-r1 \
#   --dataset MNIST \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 256 \
#   --output_dir ./results/qwen2.5vl3b-instruct-gui-mnist

# rec
# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps \
#   --dataset CIFAR10 \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 256 \
#   --output_dir ./results/qwen2.5vl3b-instruct-rec-cifar10

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps \
#   --dataset CIFAR100 \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 128 \
#   --output_dir ./results/qwen2.5vl3b-instruct-rec-cifar100

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps \
#   --dataset MNIST \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 256 \
#   --output_dir ./results/qwen2.5vl3b-instruct-rec-mnist

# math
# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name omlab/VLM-R1-Qwen2.5VL-3B-Math-0305 \
#   --dataset CIFAR10 \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 256 \
#   --output_dir ./results/qwen2.5vl3b-instruct-math-cifar10

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name omlab/VLM-R1-Qwen2.5VL-3B-Math-0305 \
#   --dataset CIFAR100 \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 128 \
#   --output_dir ./results/qwen2.5vl3b-instruct-math-cifar100

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name omlab/VLM-R1-Qwen2.5VL-3B-Math-0305 \
#   --dataset MNIST \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 256 \
#   --output_dir ./results/qwen2.5vl3b-instruct-math-mnist

#ovd
# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321 \
#   --dataset CIFAR10 \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 256 \
#   --output_dir ./results/qwen2.5vl3b-instruct-ovd-cifar10

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321 \
#   --dataset CIFAR100 \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 128 \
#   --output_dir ./results/qwen2.5vl3b-instruct-ovd-cifar100

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321 \
#   --dataset MNIST \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 256 \
#   --output_dir ./results/qwen2.5vl3b-instruct-ovd-mnist

# qwen2.5vl3b-instruct
# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name Qwen/Qwen2.5-VL-3B-Instruct \
#   --dataset CIFAR10 \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 512 \
#   --output_dir ./results/qwen2.5vl3b-instruct-cifar10

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name Qwen/Qwen2.5-VL-3B-Instruct \
#   --dataset CIFAR100 \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 128 \
#   --output_dir ./results/qwen2.5vl3b-instruct-cifar100

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name Qwen/Qwen2.5-VL-3B-Instruct \
#   --dataset MNIST \
#   --is_instruct 1 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 512 \
#   --output_dir ./results/qwen2.5vl3b-instruct-mnist

# qwen2vl2b SAT grpo trained
# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name turningpoint-ai/VisualThinker-R1-Zero \
#   --dataset CIFAR10 \
#   --is_instruct 0 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 512 \
#   --output_dir ./results/visualthinker-grpo-cifar10

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name turningpoint-ai/VisualThinker-R1-Zero \
#   --dataset CIFAR100 \
#   --is_instruct 0 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 128 \
#   --output_dir ./results/visualthinker-grpo-cifar100

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name turningpoint-ai/VisualThinker-R1-Zero \
#   --dataset MNIST \
#   --is_instruct 0 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 128 \
#   --output_dir ./results/visualthinker-grpo-mnist

# # qwen2vl2b SAT sft trained
# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name cosmos1030/Qwen2_VL-2B-SFT_revised2 \
#   --dataset CIFAR10 \
#   --is_instruct 0 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 128 \
#   --output_dir ./results/visualthinker-sft-cifar10

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name cosmos1030/Qwen2_VL-2B-SFT_revised2 \
#   --dataset CIFAR100 \
#   --is_instruct 0 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 128 \
#   --output_dir ./results/visualthinker-sft-cifar100

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name cosmos1030/Qwen2_VL-2B-SFT_revised2 \
#   --dataset MNIST \
#   --is_instruct 0 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 128 \
#   --output_dir ./results/visualthinker-sft-mnist

# # pure qwen2vl-2b
# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name Qwen/Qwen2-VL-2B \
#   --dataset CIFAR10 \
#   --is_instruct 0 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 128 \
#   --output_dir ./results/qwen2vl2b-cifar10

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name Qwen/Qwen2-VL-2B \
#   --dataset CIFAR100 \
#   --is_instruct 0 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 128 \
#   --output_dir ./results/qwen2vl2b-cifar100

# python /clifford-data/home/doyoonkim/projects/vlm_forgetting/main.py \
#   --model_name Qwen/Qwen2-VL-2B \
#   --dataset MNIST \
#   --is_instruct 0 \
#   --gpus 0,1,2,3,4,5,6,7,8,9 \
#   --batch_size 128 \
#   --output_dir ./results/qwen2vl2b-mnist


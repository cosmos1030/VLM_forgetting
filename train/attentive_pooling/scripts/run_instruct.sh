python attentive_pooling_instruct.py \
    --dataset CIFAR100 \
    --model_id /public/home/group_ucb/dyk6208/Qwen2.5-VL-3B-Instruct \
    --cuda_device 0

python attentive_pooling_instruct.py \
    --dataset CIFAR10 \
    --model_id /public/home/group_ucb/dyk6208/Qwen2.5-VL-3B-Instruct \
    --cuda_device 0

python attentive_pooling_instruct.py \
    --dataset CIFAR100 \
    --model_id /public/home/group_ucb/dyk6208/LLaMA-Factory/saves/qwen2_5_vl-3b/full/cifar10_sft/checkpoint-31250 \
    --cuda_device 0

python attentive_pooling_instruct.py \
    --dataset CIFAR10 \
    --model_id /public/home/group_ucb/dyk6208/LLaMA-Factory/saves/qwen2_5_vl-3b/full/cifar100_sft/checkpoint-31250 \
    --cuda_device 0

python attentive_pooling_instruct.py \
    --dataset CIFAR10 \
    --model_id /public/home/group_ucb/dyk6208/VLM_forgetting/train/fully_finetuning_grpo/checkpoints/GRPO-Qwen2.5-VL-3B-Instruct-cifar100/checkpoint-7200 \
    --cuda_device 0

python attentive_pooling_instruct.py \
    --dataset CIFAR100 \
    --model_id /public/home/group_ucb/dyk6208/VLM_forgetting/train/fully_finetuning_grpo/checkpoints/GRPO-Qwen2.5-VL-3B-Instruct-cifar10/checkpoint-7200 \
    --cuda_device 0
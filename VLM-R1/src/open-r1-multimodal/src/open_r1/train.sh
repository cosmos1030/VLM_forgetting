# train.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9

# --config_file 인자를 사용하여 DeepSpeed 설정을 불러옵니다.
accelerate launch --config_file=/clifford-data/home/doyoonkim/projects/vlm_forgetting/VLM-R1/src/open-r1-multimodal/configs/zero3.yaml gui_sft.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name /clifford-data/home/doyoonkim/projects/vlm_forgetting/VLM-R1/src/open-r1-multimodal/data_jsonl/gui_multi-image.jsonl \
    --image_root /clifford-data/home/doyoonkim/projects/vlm_forgetting/VLM-R1/gui_multi-image \
    --output_dir ./qwen2.5-vl-3b-sft-gui-defect \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --logging_steps 5 \
    --fp16 True \
    --gradient_checkpointing \
    --max_seq_length 2048 \
    --save_steps 1000 \
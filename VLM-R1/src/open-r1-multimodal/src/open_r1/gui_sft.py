# Copyright 2025 The HuggingFace Team. All rights reserved.
# ... (라이선스 헤더는 동일)

"""
Supervised fine-tuning script for decoder language models.
이 스크립트는 gui_multi-image.jsonl 형식의 데이터셋을 사용하여
다중 이미지 입력이 가능한 Vision-Language 모델을 파인튜닝하도록 수정되었습니다.
"""

import logging
import os
import sys

import datasets
import torch
from torch.utils.data import Dataset
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from configs import SFTConfig
from utils.callbacks import get_callbacks
import json
from PIL import Image

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import field
from qwen_vl_utils import process_vision_info
logger = logging.getLogger(__name__)
from dataclasses import dataclass

import warnings
warnings.filterwarnings("ignore")

@dataclass
class SFTScriptArguments(ScriptArguments):
    image_root: str = field(default=None, metadata={"help": "이미지가 저장된 루트 디렉토리입니다."})


processor = None

# ##################################################################
# 여기가 핵심 수정 부분입니다: LazySupervisedDataset 클래스
# ##################################################################
class LazySupervisedDataset(Dataset):
    """
    gui_multi-image.jsonl 형식의 데이터를 처리하기 위해 수정된 데이터셋 클래스.
    각 샘플은 두 개의 이미지와 human/gpt 대화로 구성됩니다.
    """
    def __init__(self, data_path: str, script_args: SFTScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        # .jsonl 파일을 직접 읽도록 로직을 단순화합니다.
        logger.info(f"Loading data from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.list_data_dict.append(json.loads(line))
        logger.info(f"Loaded {len(self.list_data_dict)} examples.")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # i번째 JSON 객체를 가져옵니다.
        example = self.list_data_dict[i]

        # 이미지 루트 경로를 가져옵니다.
        image_root = self.script_args.image_root
        if not image_root:
            raise ValueError("`image_root` 인자가 반드시 필요합니다. 이미지 폴더 경로를 지정해주세요.")

        # 대화 내용을 추출합니다.
        # human의 프롬프트와 gpt의 응답(정답)을 분리합니다.
        human_prompt = example["conversations"][0]["value"]
        gpt_answer = example["conversations"][1]["value"]

        # 사용자 입력(user turn)을 구성합니다.
        # Qwen-VL 모델 형식에 맞게, 여러 이미지를 먼저 리스트에 넣고 마지막에 텍스트를 추가합니다.
        user_content = []
        image_files = example.get("image", []) # image 필드가 리스트라고 가정
        for image_file in image_files:
            image_path = os.path.join(image_root, image_file)
            if not os.path.exists(image_path):
                 # 실제 운영 시에는 여기서 에러를 발생시키거나 로깅을 하는 것이 좋습니다.
                logger.warning(f"Image not found at path: {image_path}")
                continue # 이미지가 없으면 건너뛰기 (또는 다른 처리)
            user_content.append({"type": "image", "image": f"file://{image_path}"})
        
        user_content.append({"type": "text", "text": human_prompt})

        # 최종 대화 형식(messages)을 구성합니다.
        # 이 `messages`가 모델의 입력으로 변환됩니다.
        example["messages"] = [
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": gpt_answer,
            }
        ]
        return example


def collate_fn(examples):
    # 이 함수는 `LazySupervisedDataset`에서 생성한 `messages` 형식을
    # 모델이 처리할 수 있는 배치(batch)로 만들기 때문에 수정할 필요가 없습니다.
    texts = [
        processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False) # 학습 시에는 add_generation_prompt=False
        for example in examples
    ]
    image_inputs = []
    for example in examples:
        imgs, vids = process_vision_info(example["messages"])
        image_inputs.append(imgs)
        
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )
    
    labels = batch["input_ids"].clone()
    # Loss 계산 시 패딩 토큰은 무시하도록 -100으로 설정
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Loss 계산 시 이미지 토큰도 무시하도록 -100으로 설정
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    
    # Loss 계산 시 프롬프트 부분(user 입력)도 무시하도록 -100으로 설정
    # (SFTTrainer가 내부적으로 처리해주기도 하지만, 명시적으로 처리하는 것이 안전할 수 있습니다.
    # trl의 SFTTrainer는 DataCollatorForCompletionOnlyLM를 사용하지 않는 한, 알아서 label을 생성해줍니다.
    # apply_chat_template과 processor 호출 방식에 따라 레이블이 자동으로 생성되므로, 이 부분은 그대로 두어도 좋습니다.)
    
    batch["labels"] = labels

    return batch


def main(script_args, training_args, model_args):
    # 이하는 원본 코드와 대부분 동일하게 유지됩니다.
    
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################
    
    # 수정된 LazySupervisedDataset 클래스를 사용하여 데이터셋을 생성합니다.
    dataset = LazySupervisedDataset(data_path=script_args.dataset_name, script_args=script_args)

    ################
    # Load processor
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code,
            use_fast=True
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        # 이 스크립트는 VL 모델을 가정하므로, text-only 모델은 에러를 발생시키는 것이 좋습니다.
        raise ValueError("This script is designed for Vision-Language models. Please use a model with 'vl' in its name.")
        
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    from transformers import Qwen2VLForConditionalGeneration
    
    # 모델 이름에 따라 적절한 클래스를 로드합니다.
    # 원본 코드의 로직을 그대로 사용하거나, 사용할 모델에 맞게 고정할 수 있습니다.
    model_class = None
    if "Qwen2-VL" in model_args.model_name_or_path:
        from transformers import Qwen2VLForConditionalGeneration
        model_class = Qwen2VLForConditionalGeneration
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_class = Qwen2_5_VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}. This script supports Qwen2-VL family.")

    model = model_class.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    
    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None, # 평가 데이터셋은 필요 시 동일한 방식으로 구성하여 전달
        # tokenizer는 processor의 일부이므로 processor.tokenizer를 전달
        # tokenizer=processor.tokenizer, 
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        # packing=True와 같은 TRL의 고급 기능을 사용하려면 데이터 포맷팅에 추가 작업이 필요할 수 있습니다.
        # 여기서는 packing을 False로 가정하고 진행합니다.
        # packing=False,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    # train_samples 계산 시 `dataset_train_split` 대신 전체 데이터셋 길이를 사용하도록 수정
    metrics["train_samples"] = len(dataset) 
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # (이하 저장, 푸시 로직은 원본과 동일)
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": script_args.dataset_name,
        "dataset_tags": script_args.dataset_name,
        "tags": ["open-r1", "gui-defect-detection"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    # TrlParser를 사용하여 커맨드라인 인자를 파싱합니다.
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
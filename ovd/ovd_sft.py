```python
# Copyright 2025 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

import datasets
import torch
from torch.utils.data import Dataset
import transformers
from datasets import load_dataset
from transformers import AutoProcessor, set_seed, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
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
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# SFTConfig: extends Transformers TrainingArguments for SFT-specific options
# ----------------------------------------------------------------------------
@dataclass
class SFTConfig(TrainingArguments):
    """
    Training arguments for supervised fine-tuning of Vision-Language models.
    Inherits all TrainingArguments parameters.
    """
    # Add any additional SFT-specific arguments if needed
    pass

# ----------------------------------------------------------------------------
# get_callbacks: returns list of Trainer callbacks
# ----------------------------------------------------------------------------
def get_callbacks(training_args: SFTConfig, model_args: ModelConfig) -> List[Any]:
    callbacks = []
    if getattr(training_args, 'push_to_hub', False):
        try:
            from trl import PushToHubCallback
            callbacks.append(PushToHubCallback)
        except ImportError:
            logger.warning("trl.PushToHubCallback not available.")
    return callbacks

# ----------------------------------------------------------------------------
# Dataset for OVDEval -> OVD detection instructions
# ----------------------------------------------------------------------------
class LazyOVDEvalDataset(Dataset):
    """
    Convert OVDEval JSON annotations into OVD-style question-answer pairs,
    including bounding box outputs and hard negative yes/no questions.
    """
    def __init__(self, data_dir: str, script_args: ScriptArguments):
        super().__init__()
        self.script_args = script_args
        self.examples: List[Dict[str, Any]] = []

        # load all JSON splits
        for fname in os.listdir(data_dir):
            if not fname.endswith('.json'):
                continue
            path = os.path.join(data_dir, fname)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # build category mapping
            cat_map = {c['id']: c['name'] for c in data['categories']}
            # image metadata map
            img_meta = {img['id']: img for img in data['images']}
            # group annotations by image
            ann_by_img: Dict[int, List[Dict[str, Any]]] = {}
            for ann in data['annotations']:
                ann_by_img.setdefault(ann['image_id'], []).append(ann)

            # for each image, create QA pairs
            for img_id, anns in ann_by_img.items():
                meta = img_meta[img_id]
                file_name = meta['file_name']
                # positive object detection examples
                for ann in anns:
                    label = cat_map[ann['category_id']]
                    bbox = ann['bbox']  # [x, y, w, h]
                    # 1) Where is X? -> bounding box
                    prompt = (
                        f"<|image|>\nWhere is the '{label}' in the image?"
                    )
                    answer = (
                        f"The '{label}' is at [x={int(bbox[0])}, y={int(bbox[1])},"
                        f" width={int(bbox[2])}, height={int(bbox[3])}]."
                    )
                    self.examples.append({
                        'image': file_name,
                        'prompt': prompt,
                        'answer': answer
                    })
                # 2) yes/no questions for hard negatives
                pos_labels = {cat_map[ann['category_id']] for ann in anns}
                neg_labels = [neg for neg in data.get('neg_text', []) if neg not in pos_labels]
                # limit to one negative per image
                if neg_labels:
                    neg_label = neg_labels[0]
                    prompt = (
                        f"<|image|>\nIs there a '{neg_label}' present in the image?"
                    )
                    answer = (
                        "Yes, it is present." if neg_label in pos_labels else "No, it is not present."
                    )
                    self.examples.append({
                        'image': file_name,
                        'prompt': prompt,
                        'answer': answer
                    })

        logger.info(f"Loaded {len(self.examples)} OVD examples from {data_dir}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        image_path = os.path.join(self.script_args.image_root, ex['image'])
        if not os.path.exists(image_path):
            logger.warning(f"Missing image: {image_path}")
        # build messages sequence
        msgs = []
        msgs.append({'type': 'image', 'image': f'file://{image_path}'})
        msgs.append({'type': 'text', 'text': ex['prompt']})
        return {
            'messages': [
                {'role': 'user', 'content': msgs},
                {'role': 'assistant', 'content': ex['answer']}
            ]
        }

# ----------------------------------------------------------------------------
# Data collator
# ----------------------------------------------------------------------------
def collate_fn(examples: List[Dict[str, Any]]):
    texts = [
        processor.apply_chat_template(ex['messages'], tokenize=False, add_generation_prompt=False)
        for ex in examples
    ]
    image_inputs = [process_vision_info(ex['messages'])[0] for ex in examples]
    batch = processor(text=texts, images=image_inputs, return_tensors='pt', padding=True)
    labels = batch['input_ids'].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch['labels'] = labels
    return batch

# ----------------------------------------------------------------------------
# Main training
# ----------------------------------------------------------------------------
def main(script_args: ScriptArguments, training_args: SFTConfig, model_args: ModelConfig):
    set_seed(training_args.seed)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger.setLevel(training_args.get_process_log_level())

    last_ckpt = None
    if os.path.isdir(training_args.output_dir):
        last_ckpt = get_last_checkpoint(training_args.output_dir)

    dataset = LazyOVDEvalDataset(data_dir=script_args.dataset_name, script_args=script_args)

    global processor
    if 'vl' not in model_args.model_name_or_path.lower():
        raise ValueError('Require a vision-language model')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True
    )
    processor.pad_token = processor.eos_token
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    quant_conf = get_quantization_config(model_args)
    model_kwargs = {
        'revision': model_args.model_revision,
        'trust_remote_code': model_args.trust_remote_code,
        'torch_dtype': getattr(torch, model_args.torch_dtype) if model_args.torch_dtype else None,
        'use_cache': not training_args.gradient_checkpointing,
        'device_map': get_kbit_device_map() if quant_conf else None,
        'quantization_config': quant_conf,
    }
    if 'qwen2.5-vl' in model_args.model_name_or_path.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    else:
        from transformers import Qwen2VLForConditionalGeneration as ModelClass
    model = ModelClass.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {'skip_prepare_dataset': True}
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args)
    )

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(
            finetuned_from=model_args.model_name_or_path,
            dataset=script_args.dataset_name
        )
    if training_args.push_to_hub:
        trainer.push_to_hub(
            finetuned_from=model_args.model_name_or_path,
            dataset=script_args.dataset_name
        )

if __name__ == '__main__':
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
